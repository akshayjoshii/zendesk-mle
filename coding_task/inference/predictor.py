import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, PreTrainedTokenizerBase
from peft import PeftModel, PeftConfig
from typing import List, Union, Dict, Any

from coding_task.data.utils import basic_text_cleanup
from coding_task.logging_utils import get_logger
from coding_task.inference.config import InferenceConfig

class TextClassifierPredictor:
    """Loads a trained PEFT text classification model and performs inference."""

    def __init__(self, config: InferenceConfig):
        """
        Initializes the predictor by loading the model, tokenizer, and config.

        Args:
            config (InferenceConfig): Configuration object for inference.
        """
        self.config = config
        self.logger = get_logger(
            logger_name="InferencePredictor",
            log_file_path=config.log_file,
            stream=True
        )
        self.device = torch.device(self.config.device)
        self.logger.info(f"Using device: {self.device}")

        self.tokenizer: PreTrainedTokenizerBase
        self.model: PeftModel
        self.id2label: Dict[int, str]
        self.label2id: Dict[str, int]
        self.task_type: str # 'multiclass' or 'multilabel'
        self.num_labels: int

        self._load_resources()

    def _load_resources(self):
        """Loads the tokenizer, model configuration, base model, and PEFT adapter."""
        model_path = self.config.model_path
        self.logger.info(f"Loading resources from: {model_path}")

        if not os.path.isdir(model_path):
            self.logger.error(f"Model path not found or is not a directory: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.logger.info("Tokenizer loaded successfully.")
            # use specified max_seq_length if provided, otherwise keep tokenizer's default
            if self.config.max_seq_length:
                self.logger.info(f"Overriding max_seq_length to: {self.config.max_seq_length}")
                self.tokenizer.model_max_length = self.config.max_seq_length
            else:
                # Infer from tokenizer config if possible, default if not set
                self.config.max_seq_length = self.tokenizer.model_max_length or 512
                self.logger.info(f"Using tokenizer's max_seq_length: {self.config.max_seq_length}")

        except Exception as e:
            self.logger.error(f"Failed to load tokenizer from {model_path}: {e}")
            raise

        # Load Model Config - to get base model name, labels, problem_type
        try:
            # load PEFT config first to find base model path if needed
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name_or_path = peft_config.base_model_name_or_path
            self.logger.info(f"Inferred base model from PEFT config: {base_model_name_or_path}")

            # load base model's config to get label mappings and problem type
            config = AutoConfig.from_pretrained(model_path)
            self.id2label = config.id2label
            self.label2id = config.label2id
            self.num_labels = config.num_labels

            if config.problem_type == "single_label_classification":
                self.task_type = "multiclass"
            elif config.problem_type == "multi_label_classification":
                self.task_type = "multilabel"
            else:
                self.logger.warning(f"Unknown problem_type '{config.problem_type}' in model config. Assuming 'multiclass'.")
                self.task_type = "multiclass"

            self.logger.info(f"Loaded model config. Task type: {self.task_type}, Num Labels: {self.num_labels}")
            self.logger.debug(f"id2label mapping: {self.id2label}")

        except Exception as e:
            self.logger.error(f"Failed to load model config from {model_path}: {e}")
            # Attempt to load label maps directly if config loading fails
            try:
                with open(os.path.join(model_path, "id2label.json"), 'r') as f:
                    self.id2label = {int(k): v for k, v in json.load(f).items()}
                with open(os.path.join(model_path, "label2id.json"), 'r') as f:
                    self.label2id = json.load(f)
                self.num_labels = len(self.id2label)
                # Cannot infer task_type reliably without config, user might need to specify
                self.logger.warning("Could not load AutoConfig. Loaded label maps directly. Task type inference might be inaccurate.")
                self.task_type = "multiclass" # fallback
                self.logger.warning(f"Assuming task type: {self.task_type}. Verify this is correct.")

            except Exception as e_map:
                 self.logger.error(f"Failed to load config AND label maps from {model_path}: {e_map}")
                 raise ValueError("Could not load necessary model configuration or label maps.") from e_map


        # Load Base Model
        try:
            self.logger.info(f"Loading base model: {base_model_name_or_path}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name_or_path,
                config=config # pass the loaded config
            )
        except Exception as e:
            self.logger.error(f"Failed to load base model '{base_model_name_or_path}': {e}")
            raise

        # Load PEFT Adapter
        try:
            self.logger.info(f"Loading PEFT adapter weights from: {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model = self.model.merge_and_unload() # merge adapter for faster inference
            self.logger.info("PEFT adapter loaded and merged successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load PEFT adapter from {model_path}: {e}")
            raise

        # set model to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device} and set to evaluation mode.")


    def _preprocess(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Cleans and tokenizes input text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        # basic cleanup consistent with training
        cleaned_texts = [basic_text_cleanup(text) for text in texts]

        inputs = self.tokenizer(
            cleaned_texts,
            padding=True, # Pad to max length in batch
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt" # return Torch tensors
        )
        return inputs

    def _postprocess(self, logits: torch.Tensor) -> List[Dict[str, Any]]:
        """Processes model logits into predictions based on task type."""
        predictions = []
        if self.task_type == "multiclass":
            probs = torch.softmax(logits, dim=-1)
            scores, predicted_indices = torch.max(probs, dim=-1)
            for i in range(logits.shape[0]):
                idx = predicted_indices[i].item()
                label = self.id2label.get(idx, f"UNKNOWN_LABEL_{idx}")
                score = scores[i].item()
                result = {"label": label}
                if self.config.include_probabilities:
                    result["score"] = score
                    # Optionally add all probabilities
                    # result["all_scores"] = {self.id2label.get(j, f"UNKNOWN_{j}"): p.item() for j, p in enumerate(probs[i])}
                predictions.append(result)

        elif self.task_type == "multilabel":
            probs = torch.sigmoid(logits)
            predicted_indices = (probs > self.config.multilabel_threshold).int()
            for i in range(logits.shape[0]):
                indices = torch.where(predicted_indices[i] == 1)[0]
                labels = [self.id2label.get(idx.item(), f"UNKNOWN_LABEL_{idx.item()}") for idx in indices]
                scores = {self.id2label.get(idx.item(), f"UNKNOWN_{idx.item()}"): probs[i, idx].item() for idx in indices}
                result = {"labels": labels} # return list of labels
                if self.config.include_probabilities:
                    result["scores"] = scores
                    # Optionally add all probabilities
                    # result["all_scores"] = {self.id2label.get(j, f"UNKNOWN_{j}"): p.item() for j, p in enumerate(probs[i])}
                predictions.append(result)
        else:
            self.logger.error(f"Unsupported task type '{self.task_type}' during postprocessing.")
            # return raw logits or raise error
            predictions = [{"error": f"Unsupported task type: {self.task_type}"} for _ in range(logits.shape[0])]

        return predictions


    @torch.no_grad() # disable grad calculations for inference
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predicts labels for a single text or a list of texts.

        Args:
            texts (Union[str, List[str]]): Input text or list of texts.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Prediction result(s).
                For multiclass: {'label': str, 'score'?: float}
                For multilabel: {'labels': List[str], 'scores'?: Dict[str, float]}
        """
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts_list = [texts]
        else:
            texts_list = texts

        results = []
        num_texts = len(texts_list)
        self.logger.info(f"Starting prediction for {num_texts} text(s)...")

        for i in range(0, num_texts, self.config.batch_size):
            batch_texts = texts_list[i : i + self.config.batch_size]
            self.logger.debug(f"Processing batch {i // self.config.batch_size + 1} ({len(batch_texts)} texts)")

            # preprocess
            inputs = self._preprocess(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # move batch to device

            # predict
            outputs = self.model(**inputs)
            logits = outputs.logits.detach().cpu() # move logits to CPU

            batch_predictions = self._postprocess(logits)
            results.extend(batch_predictions)

        self.logger.info("Prediction finished.")
        return results[0] if is_single_input else results

    def predict_from_file(self, input_file: str, text_column: str) -> pd.DataFrame:
        """
        Reads text from a file, predicts labels, and returns results in a DataFrame.

        Args:
            input_file (str): Path to the input file (CSV or TSV).
            text_column (str): Name of the column containing text.

        Returns:
            pd.DataFrame: DataFrame with original data and added prediction columns.
        """
        self.logger.info(f"Loading data from file: {input_file}")
        try:
            delimiter = '\t' if input_file.lower().endswith('.tsv') else ','

            # keep_default_na=False to treat empty strings as empty & not NaN
            df = pd.read_csv(input_file, delimiter=delimiter, keep_default_na=False)
            self.logger.info(f"Loaded {len(df)} rows from file.")
        except Exception as e:
            self.logger.error(f"Failed to read input file {input_file}: {e}")
            raise

        if text_column not in df.columns:
            self.logger.error(f"Text column '{text_column}' not found in file {input_file}. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Text column '{text_column}' not found in {input_file}")

        texts_to_predict = df[text_column].astype(str).tolist()

        # batch prediction
        all_predictions = self.predict(texts_to_predict)

        # Handle different output structures from _postprocess
        if self.task_type == "multiclass":
            df['predicted_label'] = [p.get('label', 'ERROR') for p in all_predictions]
            if self.config.include_probabilities:
                df['predicted_score'] = [p.get('score', 0.0) for p in all_predictions]
        elif self.task_type == "multilabel":
            df['predicted_labels'] = [p.get('labels', []) for p in all_predictions]
            if self.config.include_probabilities:
                 # store scores dict as string or handle differently if needed
                df['predicted_scores'] = [str(p.get('scores', {})) for p in all_predictions]
        else:
             df['prediction_error'] = [p.get('error', 'Unknown error') for p in all_predictions]

        self.logger.info("File processing and prediction complete.")
        return df


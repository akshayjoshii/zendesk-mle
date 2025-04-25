import os
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features, Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, Tuple, List, Any, Optional, Union

from coding_task.data.utils import CustomTextDataset, basic_text_cleanup
from coding_task.logging_utils import get_logger
from coding_task.constants import LOG_DIR

from coding_task.train.config import DataConfig

logger = get_logger(
    logger_name="DataProcessor",
    log_file_path=os.path.join(LOG_DIR, "data_processor.log"),
    stream=True
)

class DataProcessor:
    """
    Loads data using CustomTextDataset, processes labels for multiclass/multilabel,
    tokenizes text, and creates Hugging Face Dataset objects ready for training.
    """

    def __init__(self, data_config: DataConfig, tokenizer: Any):
        """
        Initializes the DataProcessor.

        Args:
            data_config (DataConfig): Configuration for data loading & processing.
            tokenizer (PreTrainedTokenizerBase): Initialized tokenizer instance.
        """
        self.config = data_config
        self.tokenizer = tokenizer
        self.label_map: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.num_labels: int = 0
        self.mlb: Optional[MultiLabelBinarizer] = None # for multilabel binarization

        self._validate_config()

    def _validate_config(self):
        """Validates the DataConfig based on the task type."""
        if self.config.task_type == "multilabel" and not self.config.unpack_multi_labels:
            logger.error("Task type is 'multilabel' but 'unpack_multi_labels' is False.")
            raise ValueError("For multilabel classification using delimiter-separated labels, "
                             "'unpack_multi_labels' must be True in DataConfig to use the "
                             "CustomTextDataset unpacking feature.")
        elif self.config.task_type == "multiclass" and self.config.unpack_multi_labels:
            logger.warning("Task type is 'multiclass' but 'unpack_multi_labels' is True. "
                           "Labels will be unpacked by CustomTextDataset, which might lead "
                           "to unexpected behavior or errors during label mapping for multiclass. "
                           "Set unpack_multi_labels=False for standard multiclass.")
        elif self.config.task_type not in ["multiclass", "multilabel"]:
             raise ValueError(f"Unsupported task_type: {self.config.task_type}. Choose 'multiclass' or 'multilabel'.")

    def _load_and_preprocess_df(self) -> pd.DataFrame:
        """Loads data using CustomTextDataset and applies basic preprocessing."""
        logger.info(f"Loading data from: {self.config.dataset_path} using {'Dask' if self.config.use_dask else 'Pandas'}")
        loader = CustomTextDataset(
            file_path=self.config.dataset_path,
            column_names=[self.config.text_column, self.config.label_column], # expecting these two columns
            use_dask=self.config.use_dask,
            unpack_multi_labels=self.config.unpack_multi_labels,
            label_column_name=self.config.label_column if self.config.unpack_multi_labels else None,
            label_delimiter=self.config.label_delimiter if self.config.unpack_multi_labels else None
        )

        # load the DF (potentially unpacked by CustomTextDataset if set)
        df_loaded = loader.load_to_dataframe()

        logger.info("Applying text cleanup...")
        # make sure cleanup_fn is defined or use basic_text_cleanup
        cleanup_function = basic_text_cleanup # or allow passing a custom function via config
        df_clean = loader.preprocess_dataframe(
            df_loaded,
            preprocess_col_names=[self.config.text_column],
            cleanup_fn=cleanup_function
        )

        # make super sure that label column is string for consistent processing before mapping/binarizing
        if self.config.label_column in df_clean.columns:
             if self.config.use_dask:
                 # dask requires explicit meta for type conversion if not already string
                 df_clean[self.config.label_column] = df_clean[self.config.label_column].astype(str, 
                                                    meta=(self.config.label_column, 'object'))
             else:
                 df_clean[self.config.label_column] = df_clean[self.config.label_column].astype(str)
        else:
             # ths should'not happen if column_names were correct and unpacking worked correctly
             raise ValueError(f"Label column '{self.config.label_column}' not found after loading/preprocessing.")

        # compute Dask DF now if used, as label processing and splitting require concrete data
        if self.config.use_dask:
            logger.info("Computing Dask DataFrame...")
            df_pandas = df_clean.compute()
            logger.info(f"Dask DataFrame computed to Pandas DataFrame. Shape: {df_pandas.shape}")
            return df_pandas
        else:
            logger.info(f"Loaded and preprocessed Pandas DataFrame shape: {df_clean.shape}")
            logger.debug(f"DataFrame head:\n{df_clean.head()}")
            return df_clean

    def _prepare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates label mappings and transforms label column based on task type."""
        logger.info(f"Preparing labels for task type: {self.config.task_type}")

        if self.config.task_type == "multiclass":
            # treat each unique string in the label column as a distinct class
            # This handles both single labels and un-unpacked combined labels like "A+B" as one class
            unique_labels = sorted(df[self.config.label_column].unique())
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.id2label = {i: label for label, i in self.label_map.items()}
            self.num_labels = len(unique_labels)
            logger.info(f"Found {self.num_labels} unique labels for multiclass task.")
            logger.debug(f"Label map: {self.label_map}")

            # appy mapping to create integer labels
            df['labels'] = df[self.config.label_column].map(self.label_map)

        elif self.config.task_type == "multilabel":
            # asumes unpack_multi_labels=True was used in CustomTextDataset.
            # The dataframe `df` now has multiple rows for original multi-label entries.
            # we need to group by the text to reconstruct the list of labels per sample,
            # then binarize.
            logger.info("Grouping unpacked labels by text to prepare for multi-hot encoding...")
            label_groups = df.groupby(self.config.text_column)[self.config.label_column].apply(list).reset_index()
            # label_groups now has columns: [text_column, label_column] where label_column contains lists of strings

            # get all unique individual labels across all samples
            all_individual_labels = set(label for sublist in label_groups[self.config.label_column] for label in sublist)
            unique_labels = sorted(list(all_individual_labels))

            # create label mapping for multi-hot encoding
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.id2label = {i: label for label, i in self.label_map.items()}
            self.num_labels = len(unique_labels)
            logger.info(f"Found {self.num_labels} unique individual labels for multilabel task.")
            logger.debug(f"Label map: {self.label_map}")

            # Use MultiLabelBinarizer to create multi-hot vectors
            self.mlb = MultiLabelBinarizer(classes=unique_labels)
            multi_hot_labels = self.mlb.fit_transform(label_groups[self.config.label_column])

            # add the multi-hot labels back to the grouped dataframe
            # HF trainer that we use expects float labels for BCEWithLogitsLoss
            label_groups['labels'] = [list(map(float, row)) for row in multi_hot_labels]

            # keep only the text and the new 'labels' columns
            df = label_groups[[self.config.text_column, 'labels']]
            logger.info(f"Reconstructed DataFrame shape for multilabel: {df.shape}")
            logger.debug(f"Multilabel DataFrame head:\n{df.head()}")

        else:
            raise ValueError(f"Unsupported task_type: {self.config.task_type}")

        # Drop original label column if it still exists and isn't the target 'labels'
        if self.config.label_column in df.columns and self.config.label_column != 'labels':
             df = df.drop(columns=[self.config.label_column])

        # Check for NaNs in the final 'labels' column
        if df['labels'].isnull().any():
            logger.warning("NaN values found in the 'labels' column after processing. Check data and label mapping.")
            df = df.dropna(subset=['labels'])

        return df

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenizes text data using the instance's tokenizer."""
        tokenized_batch = self.tokenizer(
            examples[self.config.text_column],
            padding=False, # pad dynamically per batch using DataCollator (better than static padding)
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors=None, # return lists for dataset mapping
        )
        # The label column should already be named 'labels' from _prepare_labels
        if 'labels' in examples:
             tokenized_batch["labels"] = examples['labels']
        return tokenized_batch


    def load_and_prepare_datasets(self) -> Tuple[DatasetDict, Dict[int, str], Dict[str, int], int]:
        """
        Main method to load, process, tokenize data, and return a DatasetDict
        along with label mappings.
        """
        df_full = self._load_and_preprocess_df()
        df_processed = self._prepare_labels(df_full)

        # Split data - using the processed pandas DF
        if self.config.validation_split_ratio > 0 and self.config.validation_split_ratio < 1.0:
            logger.info(f"Splitting data into train/validation ({1.0 - self.config.validation_split_ratio:.1f}/{self.config.validation_split_ratio:.1f})")
            train_df, val_df = train_test_split(
                df_processed,
                test_size=self.config.validation_split_ratio,
                random_state=42,
                # TODO: Stratification might not be needed if split is large enough
                # stratify=df_processed['labels'] if self.config.task_type == 'multiclass' else None
            )
            logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        elif self.config.validation_split_ratio == 0:
            logger.info("Using full dataset for training (validation_split_ratio=0).")
            train_df = df_processed
            val_df = None
        else:
             logger.warning(f"Invalid validation_split_ratio ({self.config.validation_split_ratio}). Using full dataset for training.")
             train_df = df_processed
             val_df = None


        # Define features for the HF Dataset for clarity and type safety
        hf_features_dict = {self.config.text_column: Value("string")}
        if self.config.task_type == "multiclass":
            # use ClassLabel for multiclass
             hf_features_dict['labels'] = ClassLabel(num_classes=self.num_labels, names=list(self.label_map.keys()))
        elif self.config.task_type == "multilabel":
             # use Sequence of floats for multi-hot encoded labels (BCEWithLogitsLoss expects floats)
             hf_features_dict['labels'] = Sequence(feature=Value(dtype='float32'), length=self.num_labels)

        hf_features = Features(hf_features_dict)
        logger.debug(f"Hugging Face Dataset features: {hf_features}")

        # Convert to HF DatasetDict
        logger.info("Converting DataFrame(s) to Hugging Face DatasetDict...")
        dataset_dict = DatasetDict()
        dataset_dict['train'] = Dataset.from_pandas(train_df, features=hf_features, preserve_index=False)
        if val_df is not None:
            dataset_dict['validation'] = Dataset.from_pandas(val_df, features=hf_features, preserve_index=False)

        # toknize datasets using .map() for efficiency
        logger.info("Tokenizing datasets...")
        # Specify columns to remove after tokenization
        remove_cols = [self.config.text_column]
        tokenized_datasets = dataset_dict.map(
            self._tokenize_function,
            batched=True,
            remove_columns=remove_cols,
            desc="Running tokenizer on dataset",
        )

        # Set format for PyTorch compatibility with Trainer
        tokenized_datasets.set_format("torch")
        logger.info("Dataset processing complete.")
        logger.debug(f"Sample tokenized train example: {tokenized_datasets['train'][0]}")
        logger.debug(f"Tokenized dataset features: {tokenized_datasets['train'].features}")

        return tokenized_datasets, self.id2label, self.label_map, self.num_labels


import os
import json
import sys

import torch
from transformers import set_seed, HfArgumentParser, AutoTokenizer

from coding_task.train.config import DataConfig, ModelConfig, PeftConfig, TrainingConfig
from coding_task.train.data_processor import DataProcessor
from coding_task.train.model_loader import load_model_for_training
from coding_task.train.metrics import compute_metrics_fn
from coding_task.train.trainer import create_trainer

from coding_task.logging_utils import get_logger
from coding_task.constants import LOG_DIR

logger = get_logger(
    logger_name="TrainMain",
    log_file_path=os.path.join(LOG_DIR, "train_main.log"),
    stream=True
)

def main():
    # HfArgumentParser allows parsing dataclasses directly from CLI args
    parser = HfArgumentParser((DataConfig, ModelConfig, PeftConfig, TrainingConfig))

    # Check if args are provided via CLI, otherwise use defaults
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If a single JSON file path is provided, load args from it
        logger.info(f"Loading configuration from JSON file: {sys.argv[1]}")
        data_args, model_args, peft_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # parse args from command line
        logger.info("Parsing configuration from command line arguments.")
        data_args, model_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        training_args.fp16 = False  # Disable FP16 if running on CPU

    logger.info("--- Data Configuration ---")
    logger.info(data_args)
    logger.info("--- Model Configuration ---")
    logger.info(model_args)
    logger.info("--- PEFT Configuration ---")
    logger.info(peft_args)
    logger.info("--- Training Configuration ---")
    logger.info(training_args)

    # for exp reproducibility
    set_seed(training_args.seed)
    logger.info(f"Set random seed to: {training_args.seed}")
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {training_args.output_dir}")

    # Load tokenizer for data processing
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True
    )
    # add pad token if missing - common for models like Llama, GPT
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Starting data processing for Train/Validation...")
    data_processor = DataProcessor(data_config=data_args, tokenizer=tokenizer)
    tokenized_datasets, id2label, label2id, num_labels = data_processor.load_and_prepare_datasets()
    logger.info(f"Data processing complete. Number of labels: {num_labels}")
    logger.info(f"Label mapping (id2label): {id2label}")

    # save label mappings for later inference
    try:
        with open(os.path.join(training_args.output_dir, "label2id.json"), "w") as f:
            json.dump(label2id, f, indent=2)
        with open(os.path.join(training_args.output_dir, "id2label.json"), "w") as f:
            # Convert int keys to str for JSON compatibility if necessary
            id2label_save = {str(k): v for k, v in id2label.items()}
            json.dump(id2label_save, f, indent=2)
        logger.info(f"Label mappings saved to {training_args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to save label mappings: {e}")

    logger.info("Loading model for training...")
    model = load_model_for_training(
        model_config=model_args,
        peft_config=peft_args,
        data_config=data_args,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    # If pad token was added to tokenizer, ensure model's config reflects this
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
         logger.info(f"Setting model config pad_token_id to: {tokenizer.pad_token_id}")
         model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Model loading complete.")
    logger.info("Setting up Trainer...")

    # Get the compute_metrics function tailored to the task type
    compute_metrics = compute_metrics_fn(data_args.task_type, id2label)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_config=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        compute_metrics=compute_metrics,
    )
    logger.info("Trainer setup complete.")
    logger.info("*** Starting Training ***")
    train_result = trainer.train()
    logger.info("*** Training Finished ***")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # saves optimizer, scheduler, RNG state etc.
    logger.info(f"Training metrics: {metrics}")

    # Evaluation is automatically done during training if evaluation_strategy is 'steps' or 'epoch'
    # we can run a final evaluation explicitly if needed, especially if load_best_model_at_end=True
    if training_args.evaluation_strategy != "no" and tokenized_datasets.get("validation"):
        logger.info("*** Starting Final Evaluation on Validation Set ***")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        # Ensure metrics are prefixed correctly for saving/logging
        eval_metrics_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
        trainer.log_metrics("eval", eval_metrics_log)
        trainer.save_metrics("eval", eval_metrics_log)
        logger.info(f"Final validation metrics: {eval_metrics_log}")


    logger.info("*** Starting Test Set Evaluation ***")
    if data_args.test_dataset_path:
        logger.info(f"Loading test dataset from: {data_args.test_dataset_path}")

        if not os.path.exists(data_args.test_dataset_path):
             logger.error(f"Test dataset path not found: {data_args.test_dataset_path}")
             raise FileNotFoundError(f"Test dataset path not found: {data_args.test_dataset_path}")

        test_data_args = DataConfig(
            dataset_path=data_args.test_dataset_path,
            text_column=data_args.text_column,
            label_column=data_args.label_column,
            task_type=data_args.task_type,
            unpack_multi_labels=data_args.unpack_multi_labels,
            label_delimiter=data_args.label_delimiter,
            use_dask=data_args.use_dask,
            max_seq_length=data_args.max_seq_length,
            validation_split_ratio=0 # ensure test set is not split
        )

        # Initialize test processor PASSING the maps from training data
        # This ensures the label dimension is consistent with the model output layer
        test_processor = DataProcessor(
            data_config=test_data_args,
            tokenizer=tokenizer,
            label_map=label2id, # map from training
            id2label=id2label,   # map from training
            num_labels=num_labels  # num_labels from training
        )

        # Process the test dataset using the consistent label mapping
        # The result is a DatasetDict, test data is under the 'train' key because validation_split_ratio=0
        tokenized_test_dataset_dict, _, _, _ = test_processor.load_and_prepare_datasets()
        test_dataset = tokenized_test_dataset_dict.get("train") # Use .get() for safety

        # Test set is loaded as "train" in DatasetDict
        # because the DataProcessor class is designed to handle datasets for training and validation
        # TODO: Refactor this to avoid confusion, with more time I would fix this at highest priority!!
        if test_dataset:
                logger.info("*** Starting Test Set Evaluation ***")
                test_metrics = trainer.evaluate(eval_dataset=test_dataset) # Evaluate the test dataset
                # Ensure metrics are prefixed correctly for saving/logging
                test_metrics_log = {f"test_{k}": v for k, v in test_metrics.items()}
                trainer.log_metrics("test", test_metrics_log)
                trainer.save_metrics("test", test_metrics_log)
                logger.info(f"Test set evaluation metrics: {test_metrics_log}")
        else:
            logger.warning("Test dataset processing resulted in an empty dataset. Skipping test evaluation.")
    else:
        logger.warning("No test dataset provided. Skipping test evaluation.")
    logger.info("*** Evaluation Finished ***")

    # if load_best_model_at_end=True, the best model checkpoint is loaded.
    # We save the final model state explicitly.
    logger.info(f"Saving final model/adapter to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir) # Saves adapter & config (if PEFT) or full model
    model.config.save_pretrained(training_args.output_dir)

    # also Save the tokenizer alongside the model
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save training arguments for reference
    try:
        training_args_dict = training_args.to_dict()
        with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
            json.dump(training_args_dict, f, indent=2)
        logger.info(f"Training arguments saved to {training_args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to save training arguments: {e}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()

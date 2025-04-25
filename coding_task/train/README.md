# Intent Classification Model PEFT Training & Evaluation

This directory contains the scripts and modules necessary for training intent classification models using HF Transformers and Parameter-Efficient Fine-Tuning (PEFT) techniques. This also implements advanced acceleration using DeepSpeed, FSDP, AutoWarp for massive training speedup.

## Features

*   **Hugging Face Integration**: Leverages the `transformers` library for models, tokenizers, and the `Trainer` API.
*   **PEFT Methods**: Supports efficient fine-tuning using methods like LoRA and IA3 via the `peft` library.
*   **Task Types**: Handles both `multiclass` and `multilabel` intent classification tasks.
*   **Flexible Configuration**: Training parameters, model choices, PEFT settings, and data options can be configured via:
    *   Command-line arguments.
    *   A single JSON configuration file.
*   **Data Processing**: Includes robust data loading and preprocessing:
    *   Handles TSV files.
    *   Automatic train/validation splitting.
    *   Tokenization with padding and truncation.
    *   Support for multi-label unpacking (e.g., labels separated by `+`).
    *   Optional Dask integration for large datasets (`--use_dask True`).
*   **Evaluation**: Computes standard and advanced classification metrics during training and evaluation:
    *   **For `multiclass` tasks:**
        *   Accuracy
        *   Precision (Micro, Macro, Weighted)
        *   Recall (Micro, Macro, Weighted)
        *   F1 Score (Micro, Macro, Weighted)
        *   Balanced Accuracy
        *   Cohen's Kappa
        *   Matthews Correlation Coefficient (MCC)
        *   ROC AUC (One-vs-Rest, Weighted)
    *   **For `multilabel` tasks:**
        *   Precision (Micro, Macro, Weighted, Samples)
        *   Recall (Micro, Macro, Weighted, Samples)
        *   F1 Score (Micro, Macro, Weighted, Samples)
        *   Subset Accuracy (Exact Match Ratio)
        *   Hamming Loss
        *   Jaccard Score (Micro, Macro, Weighted, Samples)
        *   ROC AUC (Micro, Macro, Weighted)
        *   Average Precision (Micro, Macro, Weighted)
*   **Training Utilities**:
    *   Checkpoint saving (`--save_strategy`, `--save_steps`, `--save_total_limit`).
    *   Loading the best model based on a chosen metric (`--load_best_model_at_end`, `--metric_for_best_model`).
    *   Early stopping based on evaluation performance (`EarlyStoppingCallback` is automatically added if `load_best_model_at_end` is True).
    *   Learning rate scheduling (`--lr_scheduler_type`, `--warmup_ratio`).
    *   Gradient accumulation (`--gradient_accumulation_steps`).
    *   Mixed-precision training (`--fp16`).
    *   Gradient checkpointing (`--gradient_checkpointing`).
*   **Logging**: Comprehensive logging:
    *   Console output (`PrinterCallback`).
    *   File logging for different modules (e.g., `train_main.log`, `data_processor.log`).
    *   TensorBoard integration (`--report_to tensorboard`, `--logging_dir`).
*   **Reproducibility**: Set a random seed for consistent results (`--seed`).
*   **Model Customization**:
    *   Choose any sequence classification model from the Hugging Face Hub (`--model_name_or_path`).
    *   Option to freeze the base model's weights and train only the adapter/head (`--freeze_base_model True`).

## Configuration

The training pipeline is configured using four dataclasses defined in `config.py`:

1.  `DataConfig`: Parameters related to data loading and processing.
2.  `ModelConfig`: Parameters for the base Hugging Face model.
3.  `PeftConfig`: Parameters for the chosen PEFT method (e.g., LoRA, IA3).
4.  `TrainingConfig`: Parameters controlling the `transformers.Trainer` behavior.

You can set these parameters in two ways:

**1. Command-Line Arguments:**

Pass arguments directly when running `main.py`. Arguments correspond to the fields in the dataclasses. Use `--help` to see all available options.

```bash
python -m coding_task.train.main --help
```

Example:

```bash
python -m coding_task.train.main \
    --dataset_path ./coding_task/data/atis/train.tsv \
    --output_dir ./results/my_custom_run \
    --model_name_or_path bert-base-uncased \
    --task_type multiclass \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --method lora \
    --lora_r 8 \
    --freeze_base_model False \
    # ... other arguments
```

**2. JSON Configuration File:**

Create a JSON file containing the desired configuration values, mapping directly to the dataclass fields.

Example (`config.json`):

```json
{
  "dataset_path": "./coding_task/data/atis/train.tsv",
  "output_dir": "./results/my_json_run",
  "model_name_or_path": "xlm-roberta-base",
  "task_type": "multilabel",
  "unpack_multi_labels": true,
  "label_delimiter": "+",
  "num_train_epochs": 5,
  "per_device_train_batch_size": 8,
  "learning_rate": 5e-5,
  "freeze_base_model": true,
  "method": "lora",
  "lora_r": 16,
  "lora_alpha": 32,
  "report_to": ["tensorboard"],
  "logging_steps": 50,
  "evaluation_strategy": "epoch",
  "save_strategy": "epoch",
  "load_best_model_at_end": true,
  "metric_for_best_model": "eval_f1_micro",
  "greater_is_better": true,
  "fp16": true
}
```

Then run the script, providing the path to the JSON file:

```bash
python -m coding_task.train.main config.json
```

## Running Training

Ensure you are in the project root directory (`zendesk-mle`).

**Using Provided Scripts:**

Convenience scripts are provided for common scenarios:

*   **Multiclass Training (LoRA):**
    ```bash
    bash ./coding_task/train/train_multiclass.sh
    ```
*   **Multilabel Training (LoRA):**
    ```bash
    bash ./coding_task/train/train_multilabel.sh
    ```

Feel free to modify these scripts or use them as templates for your own configurations.

**Running Directly:**

As shown in the configuration section, you can run `main.py` directly with command-line arguments or a JSON file.

```bash
# From project root (e.g., /workspaces/zendesk-mle)
python -m coding_task.train.main [arguments or path/to/config.json]
```

## Output

After training completes, the specified `--output_dir` will contain:

*   **Checkpoints**: Subdirectories like `checkpoint-XXX` containing model weights (adapter weights if using PEFT) and trainer state (if `save_strategy` is not `no`).
*   **Final Model**: The best or last model saved (adapter config/weights if PEFT, full model otherwise).
    *   `adapter_model.safetensors` (or `.bin`)
    *   `adapter_config.json`
    *   `README.md`
    *   Potentially `pytorch_model.bin` if not using PEFT or saving full model.
*   **Tokenizer Files**:
    *   `tokenizer.json`
    *   `tokenizer_config.json`
    *   `special_tokens_map.json`
    *   Vocabulary files (e.g., `sentencepiece.bpe.model`)
*   **Label Mappings**:
    *   `label2id.json`: Mapping from label names to integer IDs.
    *   `id2label.json`: Mapping from integer IDs to label names.
*   **Training Arguments**:
    *   `training_args.json`: The configuration used for the training run.
*   **Evaluation Results**:
    *   `all_results.json`: Contains metrics from training and evaluation steps.
    *   `train_results.json`: Summary of training metrics.
    *   `eval_results.json`: Metrics from the final evaluation run (if performed).
*   **Trainer State**:
    *   `trainer_state.json`: Information about the training progress (epochs, steps, logs).

Logs (including TensorBoard logs if enabled) will be saved in the directory specified by `--logging_dir` (defaults to `coding_task/logs/training_logs`).

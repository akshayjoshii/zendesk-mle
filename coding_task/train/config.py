import os
from dataclasses import dataclass, field
from typing import Optional, List, Any

from coding_task.constants import LOG_DIR

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_path: str = field(metadata={"help": "Path to the training dataset file (e.g., train.tsv)."})
    text_column: str = field(default="atis_text", metadata={"help": "Name of the column containing the text data."})
    label_column: str = field(default="atis_labels", metadata={"help": "Name of the column containing the label(s)."})
    task_type: str = field(default="multiclass", metadata={"help": "Type of classification task: 'multiclass' or 'multilabel'."})
    unpack_multi_labels: bool = field(default=False, metadata={"help": "Use CustomTextDataset's unpack feature for multilabel tasks. MUST be True if task_type='multilabel' and labels are delimiter-separated."})
    label_delimiter: str = field(default="+", metadata={"help": "Delimiter used in the label column if unpack_multi_labels is True."})
    use_dask: bool = field(default=False, metadata={"help": "Use Dask via CustomTextDataset for loading large datasets."})
    validation_split_ratio: float = field(default=0.1, metadata={"help": "Fraction of data to use for validation. Set to 0 for no validation split."})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum sequence length for tokenizer."})
    # test_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to the test dataset file."})

@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name_or_path: str = field(default="xlm-roberta-base", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store downloaded pretrained models."})
    freeze_base_model: bool = field(default=True, metadata={"help": "Whether to freeze the parameters of the base model."})

@dataclass
class PeftConfig:
    """Configuration for Parameter-Efficient Fine-Tuning (PEFT)."""
    method: str = field(default="lora", metadata={"help": "PEFT method to use (e.g., 'lora', 'ia3', 'adapter', etc.)."})
    # LoRA specific parameters (add parameters for other methods as needed)
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling parameter."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout probability."})
    lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "List of module names or regex patterns to apply LoRA to. If None, PEFT library attempts auto-detection."})
    # if IA3 adapter: ia3_target_modules: Optional[List[str]] = field(default=["key", "value", "output.dense"])

@dataclass
class TrainingConfig:
    """Configuration for the training process using Hugging Face Trainer."""
    output_dir: str = field(metadata={"help": "Directory where model predictions and checkpoints will be written."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=32, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Initial learning rate (AdamW optimizer) for the adapter."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use."})
    warmup_ratio: float = field(default=0.06, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}) # default from HF Trainer
    logging_dir: str = field(default=os.path.join(LOG_DIR, "training_logs"), metadata={"help": "Directory for TensorBoard logs."})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy to adopt during training (`no`, `steps`, `epoch`)."})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run an evaluation every X steps (if evaluation_strategy='steps')."})
    save_strategy: str = field(default="epoch", metadata={"help": "Checkpoint save strategy to adopt during training (`no`, `steps`, `epoch`)."})
    save_steps: Optional[int] = field(default=None, metadata={"help": "Save checkpoint every X steps (if save_strategy='steps')."})
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limit the total number of checkpoints. Deletes the older checkpoints."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether to load the best model found during training at the end of training."})
    metric_for_best_model: Optional[str] = field(default="eval_f1", metadata={"help": "The metric to use to compare two different models."})
    greater_is_better: Optional[bool] = field(default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision training instead of 32-bit."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"], metadata={"help": "The list of integrations to report results to."})


"""     
Accelerator/Distributed Training
    These are often handled by the launchers - for e.g., accelerate config, torchrun args

    # But can be specified here as welll - e.g., for DeepSpeed:
    1. Microsoft DeepSpeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config file."})
    2. Pytorch Fully-sharded Data Parallel: Optional[str] = field(default=None, metadata={"help": "FSDP strategy."})
    3. fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(default=None, metadata={"help": "Class name(s) for FSDP auto wrap policy."})
"""


import os
from dataclasses import dataclass, field
from typing import Optional

from coding_task.constants import LOG_DIR

@dataclass
class InferenceConfig:
    """Configuration for the inference process."""
    model_path: str = field(metadata={"help": "Path to the trained PEFT model directory (containing adapter_config.json, tokenizer, label maps etc.)."})
    input_text: Optional[str] = field(default=None, metadata={"help": "A single text string to classify."})
    input_file: Optional[str] = field(default=None, metadata={"help": "Path to a file containing text to classify (e.g., CSV/TSV). One text per line/row."})
    text_column: str = field(default="text", metadata={"help": "Name of the column containing text data in the input_file."})
    output_file: Optional[str] = field(default=None, metadata={"help": "Path to save the predictions (CSV format). If None, prints to console."})
    device: str = field(default="cpu", metadata={"help": "Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)."})
    batch_size: int = field(default=8, metadata={"help": "Batch size for inference if processing a file."})
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "Override max sequence length for tokenizer. If None, uses tokenizer's default or training value."})
    multilabel_threshold: float = field(default=0.5, metadata={"help": "Threshold for converting probabilities to binary predictions in multilabel classification."})
    include_probabilities: bool = field(default=False, metadata={"help": "Whether to include predicted probabilities/scores in the output."})
    # add cache_dir if needed for model loading: cache_dir: Optional[str] = None
    log_file: str = field(default=os.path.join(LOG_DIR, "inference_logs", "inference.log"), metadata={"help": "Path to the inference log file."})


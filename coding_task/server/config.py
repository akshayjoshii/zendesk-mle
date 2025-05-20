import os
import torch
from pydantic_settings import BaseSettings
from pydantic import Field, DirectoryPath, PositiveInt, validator


# Helper function to determine default device
def get_default_device() -> str:
    """Returns 'cuda' if available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to construct default log path relative to this config file
def get_default_log_path() -> str:
    """Returns a default log path in project_root/logs/"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, "logs", "api_server.log")

class ServerSettings(BaseSettings):
    """Manages API Server configuration via environment variables, .env file, or defaults."""

    # Model Loading
    model_path: DirectoryPath = Field(
        default="./results/BEST_atis_multilabel_xlmr_lora",
        description="Path to the trained PEFT model directory.",
        alias="MODEL_PATH" # reads from MODEL_PATH env var
    )
    inference_device: str = Field(
        default_factory=get_default_device,
        description="Device for inference ('cuda', 'cpu', etc.). Auto-detects CUDA by default.",
        alias="INFERENCE_DEVICE"
    )

    # Server Config
    api_host: str = Field(
        default="0.0.0.0",
        description="Host address for the API server.",
        alias="API_HOST"
    )
    api_port: PositiveInt = Field(
        default=8000,
        description="Port for the API server.",
        alias="API_PORT"
    )

    log_file: str = Field(
        default_factory=get_default_log_path,
        description="Path to the API server log file.",
        alias="API_LOG_FILE"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).",
        alias="LOG_LEVEL"
    )

    # Redis Configuration
    redis_host: str = Field(
        default="127.0.0.1",
        description="Hostname or IP address of the Redis server.",
        alias="REDIS_HOST"
    )
    redis_port: int = Field(
        default=6379,
        description="Port number of the Redis server.",
        alias="REDIS_PORT"
    )
    redis_queue_name: str = Field(
        default="intent_requests",
        description="Name of the Redis list to be used as the request queue.",
        alias="REDIS_QUEUE_NAME"
    )
    redis_result_channel_prefix: str = Field(
        default="results:",
        description="Prefix for Redis Pub/Sub channels used for publishing results. Full channel will be prefix + request_id.",
        alias="REDIS_RESULT_CHANNEL_PREFIX"
    )
    redis_request_timeout: int = Field(
        default=30,
        description="Timeout in seconds for waiting for a result from Redis Pub/Sub for a given request.",
        alias="REDIS_REQUEST_TIMEOUT"
    )

    # TODO: Inference specific settings if needed at server level
    # e.g., override predictor's batch size via server config
    # inference_batch_size: Optional[PositiveInt] = Field(default=None, alias="INFERENCE_BATCH_SIZE")

    # Pydantic settings config
    class Config:
        env_file = '.env' # load .env file if present in the same directory as this config.py
        env_file_encoding = 'utf-8'
        extra = 'ignore' # ignore extra env variables not defined in the model

    # optional validator
    @validator('log_level')
    def log_level_must_be_valid(cls, v):
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

# This instance will automatically load settings when the module is imported
settings = ServerSettings()

try:
    log_dir = os.path.dirname(settings.log_file)
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create log directory '{log_dir}'. Logging may fail. Error: {e}")


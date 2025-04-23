import logging
import os
from typing import Optional

from .constants import LOG_DIR

def get_logger(
    logger_name: str = "intent_recognition",
    log_file_path: str = os.path.join(LOG_DIR, "intent_recognition.log"),
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    stream: bool = True,
    file_mode: str = "a"
) -> logging.Logger:
    """
    Creates and configures a logger with both file and optional stream handlers.

    Args:
        logger_name (str): Name of the logger.
        log_file_path (str): Path to the log file.
        log_level (int): Logging level (e.g., logging.INFO).
        log_format (Optional[str]): Log message format.
        stream (bool): If True, also logs to stdout.
        file_mode (str): File mode for the log file ('a' for append, 'w' for overwrite).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # prevent adding multiple handler in interactive envs
    if not logger.handlers:
        log_format = log_format or "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # ensure log dir exists
        os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode=file_mode)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # optional stream handler (console)
        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # avoid log message duplication
        logger.propagate = False

    return logger

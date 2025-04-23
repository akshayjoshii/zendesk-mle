""" File to track constants used in the project. """
import os

# constants for file paths
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs")
ROOT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'coding_task')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
""" File to track constants used in the project. """
import os

# constants for file paths
ROOT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'coding_task')
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs")
EVAL_DIR: str = os.path.join(os.path.dirname(__file__), "evaluation")
PLOT_DIR: str = os.path.join(os.path.dirname(__file__), "evaluation", "plots")
MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "results")
BEST_MODEL: str = os.path.join(MODEL_DIR, "BEST_atis_multilabel_xlmr_lora")
# constants for model and API
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
import torch

from src.neumann.utils import SAVE_LOAD_TYPE
def get_config():
    config = {
        "model_name": "Net",
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "num_of_train_epochs": 2,
        "add_run_id": False,
        "learning_rate": 0.001,
        "save_model": SAVE_LOAD_TYPE.MODEL,
        "reload_model":SAVE_LOAD_TYPE.MODEL
    }
    return config

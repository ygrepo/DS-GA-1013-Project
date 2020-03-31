import torch

from src.neumann.utils import SAVE_LOAD_TYPE, MODEL


def get_config(model: MODEL):
    config = {
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "add_run_id": False,
        "save_model": SAVE_LOAD_TYPE.MODEL,
        "reload_model": SAVE_LOAD_TYPE.NO_ACTION
    }

    if model == MODEL.net:
        config.update({
            "model": MODEL.net,
            "num_of_train_epochs": 200,
            "learning_rate": 0.001,
            "training_batch_size": 128,
            "test_batch_size": 100,
        })
        return config

    if model == MODEL.neumann:
        config.update({
            "model": MODEL.neumann,
            "num_of_train_epochs": 100,
            "n_block": 6,  # B in the Neumann networks paper
            "image_dimension": 32,
            "batch_size": 32,
            "n_samples": 30000,  # Size of training set
            "color_channels": 3,  # Number of spectral channels.
            "learning_rate": 0.1,
            "training_batch_size": 128,
            "test_batch_size": 100,
        })
        return config

    if model == MODEL.resnet:
        config.update({
            "model": MODEL.resnet,
            "num_of_train_epochs": 200,
            "learning_rate": 0.1,
            "training_batch_size": 128,
            "test_batch_size": 100,
        })
        return config

    raise ValueError("Not handled model!")

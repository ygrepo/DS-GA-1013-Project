import torch

from src.neumann.utils import SAVE_LOAD_TYPE, MODEL, TRAINER


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
            "trainer": TRAINER.classifier,
            "num_of_train_epochs": 200,
            "learning_rate": 0.001,
            "training_batch_size": 128,
            "test_batch_size": 100,
        })
        return config

    if model == MODEL.neumann:
        config.update({
            "model": MODEL.neumann,
            "trainer": TRAINER.on_loss,
            #"lr_anneal_rate": 0.97,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "learning_rate": 0.1,
            "num_of_train_epochs": 15,
            "n_blocks": 6,  # B in the Neumann networks paper
            "image_dimension": 32,
            "training_batch_size": 32,
            "test_batch_size": 32,
            "preconditioned": True,
            "n_cg_iterations": 10,
            "reconstruct_test": True,
            "max_samples": 10
        })
        return config

    if model == MODEL.rednet:
        config.update({
            "model": MODEL.rednet,
            "trainer": TRAINER.on_loss,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "learning_rate": 0.1,
            "num_of_train_epochs": 15,
            "image_dimension": 32,
            "training_batch_size": 32,
            "test_batch_size": 32,
            "reconstruct_test": True,
            "max_samples": 10
        })
        return config

    if model == MODEL.resnet:
        config.update({
            "model": MODEL.resnet,
            "trainer": TRAINER.classifier,
            "num_of_train_epochs": 200,
            "learning_rate": 0.1,
            "training_batch_size": 128,
            "test_batch_size": 10,
        })
        return config

    raise ValueError("Not handled model!")

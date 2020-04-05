import os
import time
from typing import Dict, Any
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import numpy as np

import imageio

from src.neumann.RedNet import REDNet10, REDNet20, REDNet30
from src.neumann.config import get_config
from src.neumann.data_utils import load_cifar,load_test_dataset, get_train_valid_loader, get_test_loader
from src.neumann.model import Net, NeumannNetwork
from src.neumann.learned_component_resnet_nblock import ResNet
from src.neumann.operators_blur_cifar import BlurModel, GramianModel,CorruptionModel
from src.neumann.trainer import ClassificationTrainer, OnLossTrainer
from src.neumann.utils import set_seed, MODEL, TRAINER, load_model


def make_model(config: Dict[str, Any]):
    model_type = config["model"]
    if model_type == MODEL.net:
        model = Net()
        model = model.to(config["device"])
        if config["device"] == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        return model, criterion, optimizer

    if model_type == MODEL.neumann:
        forward_adjoint = BlurModel(config["device"])
        forward_gramian = GramianModel(forward_adjoint)
        corruption_model = CorruptionModel(config["device"], forward_adjoint)
        reg_model = REDNet10(num_features=config["image_dimension"])
        reg_model = reg_model.to(config["device"])
        model = NeumannNetwork(forward_gramian=forward_gramian, corruption_model=corruption_model,
                               forward_adjoint=forward_adjoint, reg_network=reg_model, config=config)
        model = model.to(config["device"])
        if config["device"] == "cuda":
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.MSELoss()
        return model, criterion, optimizer

    if model_type == MODEL.resnet:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
        model = model.to(config["device"])
        if config["device"] == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        return model, criterion, optimizer

    if model_type == MODEL.rednet:
        corruption_model = CorruptionModel(config["device"], BlurModel(config["device"]))
        model = REDNet10(corruption_model,num_features=config["image_dimension"])
        model = model.to(config["device"])
        if config["device"] == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        return model, criterion, optimizer
    raise ValueError("Unknown model!")


def make_trainer(model, optimizer, criterion, train_loader, test_loader, run_id, config: Dict[str, Any]):
    trainer_type = config["trainer"]
    if trainer_type == TRAINER.classifier:
        return ClassificationTrainer(model, optimizer, criterion, train_loader, test_loader, run_id, config)

    if trainer_type == TRAINER.on_loss:
        return OnLossTrainer(model, optimizer, criterion, train_loader, test_loader, run_id, config)

    raise ValueError("Unknown trainer!")




def train(config: Dict[str, Any], run_id: str):
    print("Creating model:{}".format(config["model"]))
    model, criterion, optimizer = make_model(config)
    print("loading training data")
    train_loader, val_loader = get_train_valid_loader(Path("data"), config)
    trainer = make_trainer(model, optimizer, criterion, train_loader, val_loader, run_id, config)

    trainer.train_epochs()

def test(config: Dict[str, Any], run_id: str):
    print("Creating model:{}".format(config["model"]))
    model, criterion, optimizer = make_model(config)
    print("loading testing data")
    test_loader = get_test_loader(Path("data"), config)
    trainer = make_trainer(model, optimizer, criterion, None, test_loader, run_id, config)

    trainer.test_epochs()

def test_reconstruction(config: Dict[str, Any], data_path: Path=Path("data"), path: Path=Path("data/testing/cifar-results/")):
    print("Creating model:{}".format(config["model"]))
    model, criterion, optimizer = make_model(config)
    _, _, _, start_epoch = load_model(config["model"], model, optimizer)
    print("loading testing data")
    loader = get_test_loader(data_path, config)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            print(data.shape)
            for i in range(data.shape[0]):
                input = data[i,:,:,:].unsqueeze(0)
                corruption_model = model.corruption_model
                corrupted_image = corruption_model(input)
                output = model(input)
                input = input.detach().cpu().numpy().squeeze()
                input = np.transpose(input, (1,2,0))
                imageio.imwrite(path / (str(i) + "_true.png"), input)
                output = output.detach().cpu().numpy().squeeze()
                output = np.transpose(output, (1,2,0))
                imageio.imwrite(path / (str(i) + "_reconst.png"), output)
                corrupted_image = corrupted_image.detach().cpu().numpy().squeeze()
                corrupted_image = np.transpose(corrupted_image, (1,2,0))
                imageio.imwrite(path / (str(i) + "_corr.png"), corrupted_image)







def main():
    set_seed()

    run_id = str(int(time.time()))
    print("loading config")
    config = get_config(MODEL.neumann)
    model_name = config["model"]
    print("Starting run={} for model:{}".format(run_id, model_name))

    try:
        user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
        print(user_paths)
    except KeyError:
        pass

    train(config, run_id)
    #test(config, run_id)
    #test_reconstruction(config)

if __name__ == "__main__":
    main()
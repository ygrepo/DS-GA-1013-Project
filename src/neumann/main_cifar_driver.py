import os
import time
from typing import Dict, Any

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.neumann.RedNet import REDNet10
from src.neumann.config import get_config
from src.neumann.data_utils import load_cifar
from src.neumann.model import Net, NeumannNetwork
from src.neumann.learned_component_resnet_nblock import ResNet
from src.neumann.operators_blur_cifar import BlurModel, GramianModel
from src.neumann.trainer import ClassificationTrainer, InverseProblemTrainer
from src.neumann.utils import set_seed, MODEL, TRAINER


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
        reg_model = ResNet(config["device"])
        #reg_model = REDNet10(num_features=32)

        reg_model = reg_model.to(config["device"])
        if config["device"] == "cuda":
            reg_model = nn.DataParallel(reg_model)

        forward_adjoint = BlurModel(config["device"])
        forward_gramian = GramianModel(forward_adjoint)
        corruption_model = BlurModel(config["device"], add_noise=True)
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
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
        model = model.to(config["device"])
        #if config["device"] == 'cuda':
        #    model = nn.DataParallel(model)
        #    cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        return model, criterion, optimizer

    raise ValueError("Unknown model!")


def make_trainer(model, optimizer, criterion, train_loader, test_loader, run_id, config):
    trainer_type = config["trainer"]
    if trainer_type == TRAINER.classifier:
        return ClassificationTrainer(model, optimizer, criterion, train_loader, test_loader, run_id, config)

    if trainer_type == TRAINER.inverse_problem:
        return InverseProblemTrainer(model, optimizer, criterion, train_loader, test_loader, run_id, config)

    raise ValueError("Unknown trainer!")


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
    user_paths = []

print("loading data")
train_loader, test_loader = load_cifar("Data", config)
model, criterion, optimizer = make_model(config)
trainer = make_trainer(model, optimizer, criterion, train_loader, test_loader, run_id, config)

trainer.train_epochs()

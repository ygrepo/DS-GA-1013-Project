import os
import time
from typing import Dict, Any

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.neumann.config import get_config
from src.neumann.data_utils import load_cifar
from src.neumann.model import Net, NeumannNetwork
from src.neumann.operators_blur_cifar import blur_model, blur_noise, blur_gramian
from src.neumann.trainer import Trainer
from src.neumann.utils import set_seed, MODEL


def make_model(model_type: MODEL, config: Dict[str, Any]):
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
        return NeumannNetwork(forward_gramian=blur_gramian, corruption_model=blur_noise,
                              forward_adjoint=blur_model, reg_network=None, config=config)

    if model_type == MODEL.resnet:
        #model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
        model = model.to(config["device"])
        if config["device"] == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        return model, criterion, optimizer


    raise ValueError("Unknown model!")


set_seed()

run_id = str(int(time.time()))
print("loading config")
config = get_config(MODEL.resnet)
model_name = config["model"]
print("Starting run={} for model:{}".format(run_id, model_name))

try:
    user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []

print("loading data")
train_loader, test_loader = load_cifar("Data", config)
model, criterion, optimizer = make_model(MODEL.net, config)
add_run_id = config["add_run_id"]
trainer = Trainer(model_name, model, optimizer, criterion, train_loader, test_loader, run_id, config)

trainer.train_epochs()
# trainer.test()

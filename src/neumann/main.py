import os
import time

import torch.nn as nn
import torch.optim as optim

from src.neumann.config import get_config
from src.neumann.data_utils import load_cifar
from src.neumann.model import Net
from src.neumann.trainer import Trainer
from src.neumann.utils import set_seed

set_seed()

run_id = str(int(time.time()))
print("loading config")
config = get_config()
model_name = config["model_name"]
print("Starting run={} for model:{}".format(run_id, model_name))

try:
    user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []

print("loading data")
train_loader, testloader = load_cifar("Data")
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
add_run_id = config["add_run_id"]
trainer = Trainer(model_name, model, optimizer, criterion, train_loader, run_id, add_run_id, config)

#trainer.train_epochs()
trainer.test()

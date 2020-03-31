import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

class NeumannNetwork(nn.Module):

    def __init__(self, forward_gramian, corruption_model, forward_adjoint, reg_network, config: Dict[str, Any]):
        super(NeumannNetwork, self).__init__()
        self.forward_gramian = forward_gramian
        self.corruption_model = corruption_model
        self.forward_adjoint = forward_adjoint
        self.reg_network = reg_network
        self.iterations = config["n_block"]
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

    def forward(self, true_beta):
        network_input = self.forward_adjoint(self.corruption_model(true_beta))
        print(network_input.shape)
        network_input = self.eta * network_input
        print(network_input.shape)
        runner = network_input
        neumann_sum = runner

        # unrolled gradient iterations
        for i in range(self.iterations):
            print(self.forward_gramian(runner).shape)
            linear_component = runner - self.eta * self.forward_gramian(runner)
            regularizer_output = self.reg_network(runner)
            print(regularizer_output.shape)
            runner = linear_component - regularizer_output
            neumann_sum = neumann_sum + runner

        return neumann_sum


class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

class NeumannNetwork(nn.Module):

    def __init__(self, forward_gramian, corruption_model, forward_adjoint, reg_network, config: Dict[str, Any],
                 preconditioned: bool=False, n_iterations: int=10):
        super(NeumannNetwork, self).__init__()
        self.forward_gramian = forward_gramian
        self.corruption_model = corruption_model
        self.forward_adjoint = forward_adjoint
        self.reg_network = reg_network
        self.n_blocks = config["n_blocks"]
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.lambda_param = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.preconditioned = preconditioned
        self.n_iterations = n_iterations

    def forward(self, true_beta):
        network_input = self.forward_adjoint(self.corruption_model(true_beta))
        if self.preconditioned:
            network_input = self.cg_pseudoinverse(network_input)
        else:
            network_input = self.eta * network_input
        runner = network_input
        neumann_sum = runner

        # unrolled gradient iterations
        for i in range(self.n_blocks):
            if self.preconditioned:
                linear_component = self.eta * self.cg_pseudoinverse(runner)
                learned_component = - self.lambda_param * self.reg_network(runner)

            else:
                linear_component = runner - self.eta * self.forward_gramian(runner)
                learned_component = - self.eta * self.reg_network(runner)

            runner = linear_component + learned_component
            neumann_sum = neumann_sum + runner

        return neumann_sum

    def cg_pseudoinverse(self, input):

        rtr = input.sum()
        p = input.clone()
        i = 0
        x = torch.zeros_like(input)
        while (i < self.n_iterations) \
            and rtr > 1e-10:
            Ap = self.forward_gramian(p) + self.eta * p
            alpha = torch.conj(p) * Ap
            alpha = rtr / alpha.sum()
            x = x + alpha * p
            r = r - alpha * Ap
            r2 = r * r
            rtr_new = r2.sum()
            beta = rtr_new / rtr
            p = p + beta * p

            i += 1
            rtr = rtr_new

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

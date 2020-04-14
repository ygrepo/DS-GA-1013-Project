from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import masks
import corruption_model


class NeumannNetwork(nn.Module):

    def __init__(self, mask_func, reg_network=None, config: Dict[str, Any] = None):
        super(NeumannNetwork, self).__init__()

        self.corr = corruption_model.corruption_model()
        self.mask_func = mask_func

        self.reg_network = reg_network

        self.n_blocks = config["n_blocks"]
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.lambda_param = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.preconditioned = config["preconditioned"]
        self.n_iterations = config["n_cg_iterations"]

    def forward(self, kspace):
        mask = self.mask_func(kspace.shape)
        self.corr.set_mask(mask)

        network_input = self.corr.XT(self.corr.kspace_to_image(kspace))
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
                learned_component = - self.lambda_param * self.real_to_complex( self.reg_network( self.complex_abs(runner) ) )

            else:
                linear_component = runner - self.eta * self.corr.XT_X(runner)
                learned_component = - self.eta * self.real_to_complex( self.reg_network( self.complex_abs(runner) ) )

            runner = linear_component + learned_component
            neumann_sum = neumann_sum + runner

        return self.complex_abs( neumann_sum )

    def parameters(self):
        return [self.eta, self.lambda_param] + self.reg_network.parameters()

        # Mask should be of dim C*H*W

    # Used to convert a complex tensor to a real tensor
    def complex_abs(self, x):
        return torch.sqrt(torch.sum(x * x, dim=-1))

    def real_to_complex(self, x):
        y = torch.unsqueeze(x, -1)
        return torch.cat([y, torch.zeros(y.shape)], dim=-1)

    def cg_pseudoinverse(self, input):
        Ap = self.corr.XT_X(input) + self.eta * input
        return self.real_to_complex( torch.inverse( self.complex_abs(Ap) ) )

        # rtr = input.sum()
        # p = input.clone()
        # r = p
        # i = 0
        # x = torch.zeros_like(input)
        # while (i < self.n_iterations) \
        #         and rtr > 1e-10:
        #     Ap = self.forward_gramian(p) + self.eta * p
        #     alpha = p.conj() * Ap
        #     alpha = rtr / alpha.sum()
        #     x = x + alpha * p
        #     r = r - alpha * Ap
        #     r2 = r * r
        #     rtr_new = r2.sum()
        #     beta = rtr_new / rtr
        #     p = p + beta * p
        #
        #     i += 1
        #     rtr = rtr_new
        #
        # return x


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

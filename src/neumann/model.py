import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

import utils

def apply_mask(data, mask):
    return torch.where(mask == 0, torch.Tensor([0]), data)


        self.corruption_model = lambda data : # Apply X*data
        self.forward_adjoint = lambda data : utils.fft2(apply_mask( utils.ifft2(data), mask)) # Apply X^T*data


def complex_abs(x):
    return torch.sqrt(torch.sum(x*x, dim=-1))

def real_to_complex(x):
    y = torch.unsqueeze(x, -1)
    return torch.cat([y, torch.zeros(y.shape)], dim=-1)

def corruption_model_helper(data, mask):
    x = real_to_complex(data)
    x = utils.fft2(x)
    x = apply_mask(x, mask)
    x = utils.ifft2(x)
    x = complex_abs(x)
    return x 

def forward_adjoint_helper(data, mask):
    x = real_to_complex(data)
    x = utils.ifft2(x)
    x = apply_mask(x, mask)
    x = utils.fft2(x)
    x = complex_abs(x)
    return x 


# Nuemann network for fastMRI
class NeumannNetwork(nn.Module):

    def __init__(self, reg_network, config: Dict[str, Any]):
        super(NeumannNetwork, self).__init__()
        self.kspace_to_img = None
        self.forward_gramian = None
        self.corruption_model = None
        self.forward_adjoint = None
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

    def parameters(self):
        return [self.eta,] + self.reg_network.parameters() 

    
    # Mask should be of dim C*H*W
    # All function's input and output are complex values.
    def set_transforms(self, mask):
        self.kspace_to_img = lambda data : utils.ifft2(apply_mask(data, mask))
        self.corruption_model = lambda data : corruption_model_helper(data, mask) # Apply X*data
        self.forward_adjoint = lambda data : forward_adjoint(data, mask) # Apply X^T*data
        self.forward_gramian = lambda data : self.forward_adjoint( self.corruption_model(data) ) # Apply X^T*X*data
        # Note that matrix from of Fourier & mask is symmetric

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

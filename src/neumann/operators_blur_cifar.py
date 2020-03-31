import numpy as np
import torch
from torch import nn


class BlurModel:
    def __init__(self, device, add_noise: bool = False, kernel_size: int = 5, padding: int = 2,
                 channels: int = 3, filter_sigma: float = 0.001, mean_noise: float = 0.0, sigma_noise: float = 1.0):
        self.device = device
        self.add_noise = add_noise
        filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                           padding=(padding, padding), kernel_size=(kernel_size, kernel_size), groups=channels, bias=False)
        blur_kernel = self.special_gauss(size=kernel_size, sigma=filter_sigma)
        blur_kernel_repeat = blur_kernel.reshape((kernel_size, kernel_size, 1, 1))
        blur_kernel_repeat = np.repeat(blur_kernel_repeat, channels, axis=2)
        blur_kernel_repeat = np.transpose(blur_kernel_repeat, (2, 3, 0, 1))
        blur_kernel = torch.from_numpy(blur_kernel_repeat).float()
        filter.weight.data = blur_kernel
        filter.weight.requires_grad = False
        self.filter = filter
        self.normal_dist = torch.distributions.Normal(loc=torch.tensor([mean_noise]), scale=torch.tensor([sigma_noise]))

    def special_gauss(self, size, sigma):
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / (g.sum())

    def __call__(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            output = self.filter(input)
            if not self.add_noise:
                return output
            sample = self.normal_dist.sample((output.view(-1).size())).reshape(output.size()).to(self.device)
            return output.add(sample)

class GramianModel:

    def __init__(self, blur_model: BlurModel):
        self.blur_model = blur_model

    def __call__(self, input):
        return self.blur_model(self.blur_model(input))

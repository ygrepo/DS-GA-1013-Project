import numpy as np
import torch
from torch import nn


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / (g.sum())


batch_size = 32
dimension1 = 32
dimension2 = 32
color_dimension = 3
sigma = 0.001

blur_kernel = fspecial_gauss(size=5, sigma=sigma)
blur_kernel_repeat = blur_kernel.reshape((5, 5, 1, 1))
blur_kernel_repeat = np.repeat(blur_kernel_repeat, color_dimension, axis=2)
blur_kernel_repeat = np.transpose(blur_kernel_repeat, (2, 3, 0, 1))
blur_kernel = torch.from_numpy(blur_kernel_repeat).double()


def blur_model(channels=3, weight=blur_kernel):
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=5, groups=channels, bias=False)
    gaussian_filter.weight.data = weight
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def add_noise(img, sigma):
    noisy_img = img + np.random.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img


def blur_noise(input, sigma, channels=3, weight=blur_kernel):
    filter = blur_model(channels, weight)
    blured_img = filter(input)
    return add_noise(blured_img, sigma)


def blur_gramian(img, channels=3, weights=blur_kernel):
    filter = blur_model(channels, weights)
    gramian_output = filter(img)
    gramian_output = filter(gramian_output)
    return gramian_output

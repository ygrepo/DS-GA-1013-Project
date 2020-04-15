import utils
import torch

def apply_mask(data, mask):
    return torch.where(mask == 0, torch.Tensor([0]), data)

'''
# Functions for multicoil
def replicate_new_axis(data, reps):
    return torch.cat(reps*[data.unsqueeze(0),])

def root_sum_square(data):
    return torch.sqrt((data ** 2).sum(axis=0))
'''

# Corruption functions used by Nuemann networks
# All inputs must be complex values (last dim must have length 2)
class corruption_model():

    def __init__(self):
        self.mask = None

    # Mask must have the same dimensions as the image batch to be corrupted
    def set_mask(self, mask):
        self.mask = mask

    def kspace_to_image(self, data):
        x = apply_mask(data, self.mask)
        x = utils.ifft2(x)
        return x 

    def XT(self, data):
        x = utils.ifft2(data)
        x = apply_mask(x, self.mask)
        x = utils.fft2(x)
        return x 

    def XT_X(self, data):
        x = utils.ifft2(data)
        x = apply_mask(x, self.mask)
        x = utils.fft2(x)
        x = utils.fft2(x)
        x = apply_mask(x, self.mask)
        x = utils.ifft2(x)
        return x

import model
import torch
import torch.nn as nn


def trivial_mask_func(shape):
	return torch.randint(0, 2, shape)

if __name__ == '__main__':
	trivial_reg_network = nn.Identity()

	for preconditioned in [False, True]:
		config = {"n_blocks": 3, "preconditioned": preconditioned, "n_cg_iterations": 3}

		net = model.NeumannNetwork(trivial_mask_func, trivial_reg_network, config)

		test_shape = (32, 16, 16, 2)
		test_kspace = torch.randn(test_shape)
		output = net(test_kspace)

		print("preconditioned = " + str(preconditioned))
		if output.shape == test_shape[:-1]:
			print("Dimensions are correct!")
		else:
			print("Dimensions are incorrect :/")
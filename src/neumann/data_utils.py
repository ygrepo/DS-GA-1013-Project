from pathlib import Path
from typing import Dict, Any

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import matplotlib as plt

from src.neumann.config import get_config
from src.neumann.operators_blur_cifar import BlurModel, GramianModel
from src.neumann.utils import imshow, MODEL

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar(path: Path, config: Dict[str, Any]):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    # trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["training_batch_size"], shuffle=True,
                                              num_workers=2)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["test_batch_size"], shuffle=False,
                                             num_workers=2)

    return trainloader, testloader


def get_train_valid_loader(data_dir: Path,
                           config: Dict[str, Any],
                           augment=False,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training_batch_size"], sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["training_batch_size"], sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    config: Dict[str, Any],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["test_batch_size"], shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def load_test_dataset(path: Path = Path("data/testing")):
    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=transform_test
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )
    return loader


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = CIFAR10_CLASSES[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = CIFAR10_CLASSES[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def main():
    print("loading config")
    config = get_config(MODEL.resnet)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_loader = load_test_dataset()
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(images.shape, labels)
    # imshow(images[0, :,:,:])

    trainloader, testloader = load_cifar(Path("data"), config)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    forward_adjoint = BlurModel(device)
    imshow(torchvision.utils.make_grid(forward_adjoint(images)))
    print(" ".join("%5s" % CIFAR10_CLASSES[labels[j]] for j in range(4)))

    forward_gramian = GramianModel(device)
    imshow(torchvision.utils.make_grid(forward_gramian(images)))
    print(" ".join("%5s" % CIFAR10_CLASSES[labels[j]] for j in range(4)))

    corruption_model = BlurModel(device, add_noise=True)
    imshow(torchvision.utils.make_grid(corruption_model(images)))
    print(" ".join("%5s" % CIFAR10_CLASSES[labels[j]] for j in range(4)))


if __name__ == "__main__":
    main()

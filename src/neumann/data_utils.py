from pathlib import Path
from typing import Dict, Any

import torch
import torchvision
import torchvision.transforms as transforms

from src.neumann.config import get_config
from src.neumann.operators_blur_cifar import BlurModel, GramianModel
from src.neumann.utils import imshow, MODEL

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar(path: Path, config: Dict[str, Any]):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["training_batch_size"], shuffle=True,
                                              num_workers=2)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["test_batch_size"], shuffle=False,
                                             num_workers=2)

    return trainloader, testloader

def load_test_dataset(path: Path=Path("data/testing")):
    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform= transform_test
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )
    return loader

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

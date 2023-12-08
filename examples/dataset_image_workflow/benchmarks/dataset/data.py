import os

import numpy as np
import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import TensorDataset

from .utils import cached
from examples.dataset_cifar_workflow.benchmarks.dataset.utils import split_dataset, build_transforms

cache_root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'cache'))

cifar_train = datasets.CIFAR10(root=cache_root, download=True, train=True, transform=transforms.ToTensor())
cifar_train_X = torch.stack([u[0] for u in cifar_train])
augment_transform, regular_transform, whiten_transform = build_transforms(cifar_train_X)

cifar_train_set_augment = datasets.CIFAR10(root=cache_root, download=True, train=True, transform=whiten_transform)
cifar_test_set = datasets.CIFAR10(root=cache_root, download=True, train=False, transform=whiten_transform)
cifar_spec_train_set = datasets.CIFAR10(root=cache_root, download=True, train=True, transform=whiten_transform)
cifar_spec_test_set = datasets.CIFAR10(root=cache_root, download=True, train=False, transform=whiten_transform)
train_targets = cifar_train_set_augment.targets
test_targets = cifar_test_set.targets

def faster_train(device):
    global cifar_train_set_augment
    global cifar_test_set
    global cifar_spec_train_set
    global cifar_spec_test_set
    cifar_train_set_augment = cached(cifar_train_set_augment, device=device)
    cifar_test_set = cached(cifar_test_set, device=device)
    cifar_spec_train_set = cached(cifar_spec_train_set, device=device)
    cifar_spec_test_set = cached(cifar_spec_test_set, device=device)

def uploader_data(order=None):
    train_indices, order = split_dataset(torch.asarray(train_targets), 12500, split="uploader", order=order)
    valid_indices, _ = split_dataset(torch.asarray(test_targets), 2000, split="uploader", order=order)

    return (Subset(cifar_train_set_augment, train_indices),
            Subset(cifar_test_set, valid_indices),
            Subset(cifar_spec_train_set, train_indices),
            order)

def user_data(indices=None, order=None):
    if indices is None:
        indices, order = split_dataset(torch.asarray(test_targets), 3000, split="user", order=order)

    return Subset(cifar_test_set, indices), Subset(cifar_spec_test_set, indices), indices, order
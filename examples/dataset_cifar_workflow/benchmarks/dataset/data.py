import os

import numpy as np
import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from examples.dataset_cifar_workflow.benchmarks.dataset.utils import split_dataset, build_transforms

cache_root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'cache'))

cifar_data = torch.stack([u[0] for u in datasets.CIFAR10(root="cache", download=True,
                                                         train=True, transform=transforms.ToTensor())])
augment_transform, regular_transform, whiten_transform = build_transforms(cifar_data)

cifar_train_set_augment = datasets.CIFAR10(root="cache", download=True,
                                           train=True, transform=whiten_transform)
cifar_test_set = datasets.CIFAR10(root="cache", download=True,
                                  train=False, transform=whiten_transform)
cifar_spec_train_set = datasets.CIFAR10(root="cache", download=True,
                                        train=True, transform=whiten_transform)
cifar_spec_test_set = datasets.CIFAR10(root="cache", download=True,
                                       train=False, transform=whiten_transform)

def uploader_data(order=None):
    train_indices, order = split_dataset(torch.asarray(cifar_train_set_augment.targets), 12500, split="uploader", order=order)
    valid_indices, _ = split_dataset(torch.asarray(cifar_test_set.targets), 2000, split="uploader", order=order)

    return (Subset(cifar_train_set_augment, train_indices),
            Subset(cifar_test_set, valid_indices),
            Subset(cifar_spec_train_set, train_indices),
            order)

def user_data(indices=None, order=None):
    if indices is None:
        indices, order = split_dataset(torch.asarray(cifar_spec_test_set.targets), 3000, split="user", order=order)

    return Subset(cifar_test_set, indices), Subset(cifar_spec_test_set, indices), indices, order
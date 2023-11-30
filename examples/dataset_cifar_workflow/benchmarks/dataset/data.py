import os

import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets

from examples.dataset_cifar_workflow.benchmarks.dataset.utils import build_transform, sample_by_labels, split_dataset


cache_root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'cache'))

augment_transform, regular_transform = build_transform((32, 32))
cifar_train_set_augment = datasets.CIFAR10(root="cache", download=True,
                                     train=True, transform=augment_transform)
cifar_train_set = datasets.CIFAR10(root="cache", download=True,
                             train=True, transform=regular_transform)
cifar_test_set = datasets.CIFAR10(root="cache", download=True,
                            train=False, transform=regular_transform)

def uploader_data():
    train_indices, order = split_dataset(torch.asarray(cifar_train_set_augment.targets), 12500, split="uploader")
    valid_indices, _ = split_dataset(torch.asarray(cifar_test_set.targets), 2000, split="uploader", order=order)

    return (Subset(cifar_train_set_augment, train_indices),
            Subset(cifar_test_set, valid_indices),
            Subset(cifar_train_set, train_indices))

def user_data():
    test_indices, order = split_dataset(torch.asarray(cifar_test_set.targets), 3000, split="user")

    return Subset(cifar_test_set, test_indices)
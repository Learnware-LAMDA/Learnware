import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import transforms

import learnware
from examples.dataset_cifar_workflow.benchmarks.dataset import user_data, split_dataset
from examples.dataset_image_workflow.get_data import get_zca_matrix, transform_data
from learnware import setup_seed
from learnware.specification import generate_rkme_image_spec, RKMEImageSpecification


def f(d):
    return np.exp(-d / 0.00005)

def get_spec(path, order=None):
    if path is not None and os.path.exists(path):
        spec = RKMEImageSpecification()
        spec.load(path)
        return spec, spec.msg

    test_user, spec_user, _, order = user_data(order=order)
    loader = DataLoader(spec_user, batch_size=3000, shuffle=True)
    sampled_X, _ = next(iter(loader))
    spec = generate_rkme_image_spec(sampled_X, whitening=False)
    spec.msg = order

    if path is not None:
        spec.save(path)

    return spec, order

DATA_ROOT = "cache"
def get_cifar10(output_channels=3, image_size=32, z_score=True, order=None):
    ds_train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    X_train = ds_train.data
    y_train = ds_train.targets
    ds_test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([image_size, image_size])]))

    X_test = ds_test.data
    y_test = ds_test.targets

    X_train = torch.Tensor(np.moveaxis(X_train, 3, 1))
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(np.moveaxis(X_test, 3, 1))
    y_test = torch.Tensor(y_test).long()

    if output_channels == 1:
        X_train = torch.mean(X_train, 1, keepdim=True)
        X_test = torch.mean(X_test, 1, keepdim=True)

    if z_score:
        X_test = (X_test - torch.mean(X_train, [0, 2, 3], keepdim=True)) / (torch.std(X_train, [0, 2, 3], keepdim=True))
        X_train = (X_train - torch.mean(X_train, [0, 2, 3], keepdim=True)) / (
            torch.std(X_train, [0, 2, 3], keepdim=True))

    whitening_mat = get_zca_matrix(X_train, reg_coef=0.1)
    train_X = transform_data(X_train, whitening_mat)
    test_X = transform_data(X_train, whitening_mat)

    selected_data_indexes, order = split_dataset(y_test, 10000, split="user", order=order)

    return TensorDataset(test_X[selected_data_indexes], y_test[selected_data_indexes]), order




if __name__ == "__main__":
    # old1, order1 = get_spec("spec_1_V100.json", order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # old2, order2 = get_spec("spec_2_A100.json", order=[2, 3, 4, 5, 6, 7, 0, 1, 8, 9])

    old3, order3 = get_spec("spec_3_A100.json", order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    old4, order4 = get_spec("spec_6_A100.json", order=[2, 3, 4, 5, 6, 7, 0, 1, 8, 9])

    print(order3, order4)
    print(f(old3.dist(old4)))


import random
from functools import reduce

import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import transforms, v2


def sample_by_labels(labels: torch.Tensor, weights, total_num):
    weights = np.asarray(weights)

    norm_factor = np.sum(weights)
    last_non_zero = np.argwhere(weights > 0)[-1].item()
    category_nums = [int(w * total_num / norm_factor) for w in weights[:last_non_zero]]
    category_nums += [total_num - sum(category_nums)]
    category_nums += [0] * (weights.shape[0] - last_non_zero - 1)

    selected_cls_indexes = [
        random.sample(list(torch.where(labels == c)[0]), k=n)
            for c, n in enumerate(category_nums)
    ]

    return selected_cls_indexes


USER_WEIGHTS = [3, 3, 1, 1, 1, 1, 0, 0, 0, 0]
UPLOADER_WEIGHTS = [4, 4, 1, 1, 0, 0, 0, 0, 0, 0]
def split_dataset(labels, size, split="uploader", order=None):
    if split == "uploader":
        weights = np.asarray(UPLOADER_WEIGHTS)
    elif split == "user":
        weights = np.asarray(USER_WEIGHTS)
    else:
        raise Exception(split)

    if order is None:
        order = list(range(len(weights)))
        random.shuffle(order)

    selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(labels, weights[order], size))
    selected_data_indexes = torch.stack(selected_data_indexes)

    return selected_data_indexes, order

def build_zca_matrix(X, reg_coef=0.1):
    X = (X - torch.mean(X, [0, 2, 3], keepdim=True)) / (torch.std(X, [0, 2, 3], keepdim=True))

    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat) / X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov.cuda() + reg_amount * torch.eye(cov.shape[0]).cuda())
    inv_sqrt_zca_eigs = s ** (-0.5)
    whitening_transform = torch.einsum(
        'ij,j,kj->ik', u, inv_sqrt_zca_eigs, u)

    return whitening_transform.cpu()

def build_transforms(train_X):
    size = train_X.shape[2], train_X.shape[3]
    whitening_matrix = build_zca_matrix(train_X)

    mean_vector = torch.mean(train_X, [0, 2, 3], keepdim=True).squeeze(0)
    std_vector = torch.std(train_X, [0, 2, 3], keepdim=True).squeeze(0)

    augment_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vector, std=std_vector),
    ])

    regular_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vector, std=std_vector),
    ])

    whiten_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vector, std=std_vector),
        # transform_data
        transforms.LinearTransformation(whitening_matrix, torch.zeros_like(train_X[0].reshape(-1)))
    ])

    return augment_transform, regular_transform, whiten_transform
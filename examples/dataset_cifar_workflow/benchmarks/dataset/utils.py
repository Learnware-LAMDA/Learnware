import random
from functools import reduce

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset

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

def build_transform(size):
    augment_transform = transforms.Compose([
        transforms.Resize(size),
        v2.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    regular_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return augment_transform, regular_transform
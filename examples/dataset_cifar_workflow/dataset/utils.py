import random
from functools import reduce

import numpy as np
import torch
from torch.utils.data import TensorDataset


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

def split_dataset(data_x, data_y, size, split="uploader"):
    if split == "uploader":
        weights = np.asarray(UPLOADER_WEIGHTS)
    elif split == "user":
        weights = np.asarray(USER_WEIGHTS)
    else:
        raise Exception(split)

    order = list(range(len(weights)))
    random.shuffle(order)

    selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(data_y, weights[order], size))
    selected_data_indexes = torch.stack(selected_data_indexes)
    selected_X = data_x[selected_data_indexes].numpy()
    selected_y = data_y[selected_data_indexes].numpy()

    return TensorDataset(selected_X, selected_y), weights[order]
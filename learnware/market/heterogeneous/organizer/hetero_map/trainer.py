import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import trange

from .feature_extractor import FeatureTokenizer
from .....logger import get_module_logger

logger = get_module_logger("hetero_mapping_trainer")


class Trainer:
    def __init__(
        self,
        model,
        train_set_list,
        collate_fn=None,
        output_dir="./ckpt",
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        eval_batch_size=256,
        **kwargs,
    ):
        """args:
        train_set_list: a list of training sets [(x_1,y_1),(x_2,y_2),...]
        patience: the max number of early stop patience
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        """
        self.model = model
        if isinstance(train_set_list, tuple):
            train_set_list = [train_set_list]

        self.train_set_list = train_set_list
        self.collate_fn = collate_fn
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn) for trainset in train_set_list
        ]
        self.output_dir = output_dir
        # os.makedirs(self.output_dir, exist_ok=True)
        self.args = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epoch": num_epoch,
            "eval_batch_size": eval_batch_size,
            "num_training_steps": self._get_num_train_steps(train_set_list, num_epoch, batch_size),
        }
        self.args["steps_per_epoch"] = int(self.args["num_training_steps"] / (num_epoch * len(self.train_set_list)))
        self.optimizer = None

    def train(self, verbose: bool = True):
        self._create_optimizer()
        start_time = time.time()
        final_train_loss = 0
        for epoch in trange(self.args["num_epoch"], desc="Epoch"):
            ite = 0
            train_loss_all = 0
            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()
                    loss = self.model(data)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1

            if verbose:
                logger.info(
                    "epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs".format(
                        epoch,
                        train_loss_all,
                        self.optimizer.param_groups[0]["lr"],
                        time.time() - start_time,
                    )
                )
            final_train_loss = train_loss_all

        logger.info("training complete, cost {:.1f} secs.".format(time.time() - start_time))
        return final_train_loss

    def save_model(self, output_dir=None):
        if output_dir is None:
            logger.info("no path assigned for save mode, default saved to ./ckpt/model.pt !")
            output_dir = self.output_dir

        logger.info(f"saving model checkpoint to {output_dir}")
        self.model.save(output_dir)
        # self.collate_fn.save(output_dir)

        if self.args is not None:
            train_args = {}
            for k, v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(
                os.path.join(output_dir, "training_args.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(train_args))

    def _create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self._get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            decay_params_dict = {n: p for n, p in self.model.named_parameters() if n in decay_parameters}
            no_decay_params_dict = {n: p for n, p in self.model.named_parameters() if n not in decay_parameters}

            optimizer_grouped_parameters = [
                {
                    "params": list(decay_params_dict.values()),
                    "weight_decay": self.args["weight_decay"],
                },
                {"params": list(no_decay_params_dict.values()), "weight_decay": 0.0},
            ]

            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args["lr"])

    def _get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def _build_dataloader(self, trainset, batch_size, collator, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
        )
        return trainloader

    def _get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self._get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result


class TrainDataset(Dataset):
    def __init__(self, trainset):
        self.x = trainset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x.iloc[index - 1 : index]
        return x


class TransTabCollatorForCL:
    """support positive pair sampling for contrastive learning."""

    def __init__(
        self,
        feature_tokenizer=None,
        overlap_ratio=0.5,
        num_partition=3,
        **kwargs,
    ):
        self.feature_tokenizer = feature_tokenizer or FeatureTokenizer(disable_tokenizer_parallel=True)
        assert num_partition > 0, f"number of contrastive subsets must be greater than 0, got {num_partition}"
        assert isinstance(num_partition, int), f"number of constrative subsets must be int, got {type(num_partition)}"
        assert overlap_ratio >= 0 and overlap_ratio < 1, f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        self.overlap_ratio = overlap_ratio
        self.num_partition = num_partition

    def __call__(self, data):
        """
        Take a list of subsets (views) from the original tests.
        """
        # 1. build positive pairs
        # 2. encode each pair using feature extractor
        df_x = pd.concat([row for row in data])
        if self.num_partition > 1:
            sub_x_list = self._build_positive_pairs(df_x, self.num_partition)
        else:
            sub_x_list = self._build_positive_pairs_single_view(df_x)
        input_x_list = []
        for sub_x in sub_x_list:
            inputs = self.feature_tokenizer(sub_x)
            input_x_list.append(inputs)
        res = {"input_sub_x": input_x_list}
        return res

    def _build_positive_pairs(self, x, n):
        """
        Builds positive pairs of sub-dataframes from the input dataframe x.

        Args:
            x (pandas.DataFrame): Input dataframe.
            n (int): Number of sub-dataframes to split x into.

        Returns:
            list: List of sub-dataframes, each containing a positive pair of columns from x.
        """
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(math.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i + 1][:overlap]])
            elif overlap > 0 and i == n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i - 1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def _build_positive_pairs_single_view(self, x):
        """
        Builds positive pairs for a single view of data by corrupting half of the columns and shuffling the corrupted columns.

        Args:
            x (pandas.DataFrame): The input data.

        Returns:
            list: A list of two pandas DataFrames, where each DataFrame contains the original data with half of the columns corrupted and shuffled.
        """
        x_cols = x.columns.tolist()
        sub_x_list = [x]
        n_corrupt = int(len(x_cols) * 0.5)
        corrupt_cols = x_cols[:n_corrupt]
        x_corrupt = x.copy()[corrupt_cols]
        np.random.shuffle(x_corrupt.values)
        sub_x_list.append(pd.concat([x.copy().drop(corrupt_cols, axis=1), x_corrupt], axis=1))
        return sub_x_list

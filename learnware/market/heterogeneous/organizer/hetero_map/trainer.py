import math
import os
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import trange

from .....logger import get_module_logger
from .feature_extractor import FeatureTokenizer

logger = get_module_logger("hetero_mapping_trainer")


class Trainer:
    def __init__(
        self,
        model: Any,
        train_set_list: List[Any],
        collate_fn: Callable = None,
        output_dir: str = "./ckpt",
        num_epoch: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0,
        eval_batch_size: int = 256,
        **kwargs,
    ):
        """
        The initialization method for trainer.

        Parameters
        ----------
        model : Any
            The model to be trained.
        train_set_list : List[Any]
            A list of training datasets.
        collate_fn : Callable, optional
            The collate function to be used, by default None.
        output_dir : str, optional
            The directory where the trained model checkpoints will be saved, by default "./ckpt".
        num_epoch : int, optional
            Number of epochs for training, by default 10.
        batch_size : int, optional
            Batch size for training, by default 64.
        lr : float, optional
            Learning rate, by default 1e-4.
        weight_decay : float, optional
            Weight decay, by default 0.
        eval_batch_size : int, optional
            Batch size for evaluation, by default 256.
        kwargs : dict
            Additional keyword arguments.
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

    def train(self, verbose: bool = True) -> float:
        """
        Trains the model using the provided training data.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display verbose output, by default True.

        Returns
        -------
        float
            The final training loss.
        """

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

    def save_model(self, output_dir: str = None):
        """
        Saves the trained model to the specified directory.

        Parameters
        ----------
        output_dir : str, optional
            The directory where the model will be saved, by default None.
        """

        logger.info(f"saving model checkpoint to {output_dir}")
        self.model.save(output_dir)

    def _create_optimizer(self):
        """Creates an optimizer for training the model."""

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

    def _get_num_train_steps(self, train_set_list: List[Any], num_epoch: int, batch_size: int) -> int:
        """
        Calculates the total number of training steps.

        Parameters
        ----------
        train_set_list : List[Any]
            A list of training datasets.
        num_epoch : int
            Number of training epochs.
        batch_size : int
            Batch size for training.

        Returns
        -------
        int
            The total number of training steps.
        """

        total_step = 0
        for trainset in train_set_list:
            x_train = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def _build_dataloader(
        self, trainset: Any, batch_size: int, collator: Callable, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Builds a DataLoader for training.

        Parameters
        ----------
        trainset : Any
            The training dataset.
        batch_size : int
            Batch size for the DataLoader.
        collator : Callable
            Collate function for the DataLoader.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.

        Returns
        -------
        torch.utils.data.DataLoader
            The DataLoader for the training data.
        """

        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
        )
        return trainloader

    def _get_parameter_names(self, model: Any, forbidden_layer_types: List[torch.dtype]) -> List[str]:
        """
        Retrieves the names of parameters not inside forbidden layers.

        Parameters
        ----------
        model : Any
            The model from which to retrieve parameters.
        forbidden_layer_types : List[torch.dtype]
            A list of layer types to exclude.

        Returns
        -------
        List[str]
            A list of parameter names not inside the forbidden layers.
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
    """Collator class supporting positive pair sampling for contrastive learning."""

    def __init__(
        self,
        feature_tokenizer: Callable = None,
        overlap_ratio: float = 0.5,
        num_partition: int = 3,
        **kwargs,
    ):
        """
        The initialization method for TransTabCollatorForCL.

        Parameters
        ----------
        feature_tokenizer : Callable, optional
            The tokenizer used to process data, by default None.
        overlap_ratio : float, optional
            The ratio of overlap between partitions, must be between 0 and 1 (exclusive), by default 0.5.
        num_partition : int, optional
            The number of partitions to create from the data for contrastive learning, by default 3.
        """
        self.feature_tokenizer = feature_tokenizer or FeatureTokenizer(disable_tokenizer_parallel=True)
        assert num_partition > 0, f"number of contrastive subsets must be greater than 0, got {num_partition}"
        assert isinstance(num_partition, int), f"number of constrative subsets must be int, got {type(num_partition)}"
        assert overlap_ratio >= 0 and overlap_ratio < 1, f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        self.overlap_ratio = overlap_ratio
        self.num_partition = num_partition

    def __call__(self, data: List[Any]) -> Dict[str, Any]:
        """
        Processes the data into subsets for contrastive learning.

        Parameters
        ----------
        data : List[Any]
            The input data to be processed.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the processed data subsets.
        """
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

    def _build_positive_pairs(self, x: pd.DataFrame, n: int) -> List[pd.DataFrame]:
        """
        Builds positive pairs of sub-dataframes from the input dataframe.

        Parameters
        ----------
        x : pd.DataFrame
            The input dataframe.
        n : int
            The number of sub-dataframes to create.

        Returns
        -------
        List[pd.DataFrame]
            A list of sub-dataframes, each containing a positive pair of columns.
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

    def _build_positive_pairs_single_view(self, x: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Builds positive pairs for a single view of data by corrupting half of the columns and shuffling the corrupted columns..

        Parameters
        ----------
        x : pd.DataFrame
            The input dataframe.

        Returns
        -------
        List[pd.DataFrame]
            A list containing two dataframes, one with original data and one with shuffled columns.
        """
        x_cols = x.columns.tolist()
        sub_x_list = [x]
        n_corrupt = int(len(x_cols) * 0.5)
        corrupt_cols = x_cols[:n_corrupt]
        x_corrupt = x.copy()[corrupt_cols]
        np.random.shuffle(x_corrupt.values)
        sub_x_list.append(pd.concat([x.copy().drop(corrupt_cols, axis=1), x_corrupt], axis=1))
        return sub_x_list

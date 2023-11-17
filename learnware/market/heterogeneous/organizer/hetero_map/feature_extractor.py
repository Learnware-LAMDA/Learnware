import math
import os
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.init as nn_init
from torch import Tensor, nn
from transformers import BertTokenizerFast

from .....config import C as conf


class WordEmbedding(nn.Module):
    """Encodes tokens drawn from column names into word embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_idx: int = 0,
        hidden_dropout_prob: float = 0,
        layer_norm_eps: float = 1e-5,
    ):
        """
        The initialization method for word embedding.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        hidden_dim : int
            The dimension of the hidden layer.
        padding_idx : int, optional
            The index of the padding token, by default 0.
        hidden_dropout_prob : float, optional
            The dropout probability for the hidden layer, by default 0.
        layer_norm_eps : float, optional
            The epsilon value for layer normalization, by default 1e-5.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the WordEmbedding module.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input token IDs.

        Returns
        -------
        torch.Tensor
            The word embeddings corresponding to the input token IDs.
        """
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NumEmbedding(nn.Module):
    """Encode tokens drawn from column names and the corresponding numerical features."""

    def __init__(self, hidden_dim: int):
        """
        The initialization method for num embedding.

        Parameters
        ----------
        hidden_dim : int
            The dimension of the hidden layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))  # add bias
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

    def forward(self, col_emb: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the NumEmbedding module.

        Parameters
        ----------
        col_emb : torch.Tensor
            The numerical column embeddings with shape (# numerical columns, emb_dim).
        x_ts : torch.Tensor
            The numerical features with shape (bs, emb_dim).

        Returns
        -------
        torch.Tensor
            The combined feature embeddings.
        """
        col_emb = col_emb.unsqueeze(0).expand((x_ts.shape[0], -1, -1))
        feat_emb = col_emb * x_ts.unsqueeze(-1).float() + self.num_bias
        return feat_emb


class FeatureTokenizer:
    """Process input dataframe to input indices towards encoder, usually used to build dataloader for paralleling loading."""

    def __init__(
        self,
        disable_tokenizer_parallel: bool = True,
        **kwargs,
    ):
        """
        The initialization method for feature tokenizer.
        .
        Parameters
        ----------
        disable_tokenizer_parallel : bool, optional
            Whether to disable tokenizer parallelism, by default True.
        """
        cache_dir = conf.cache_path
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
        self.tokenizer.__dict__["model_max_length"] = 512
        if disable_tokenizer_parallel:  # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, x: pd.DataFrame, shuffle: bool = False, keep_input_grad: bool = False) -> Dict:
        """
        Tokenizes the input DataFrame.

        Parameters
        ----------
        x : pd.DataFrame
            The input DataFrame with column names and features.
        shuffle : bool, optional
            Whether to shuffle column order during training, by default False.
        keep_input_grad : bool, optional
            Whether to keep input gradients, by default False.

        Returns
        -------
        Dict
            A dictionary with tokenized inputs.
        """
        encoded_inputs = {"x_num": None, "num_col_input_ids": None}

        index_cols = (
            [i for i in range(len(x.columns))] if not shuffle else np.random.shuffle([i for i in range(len(x.columns))])
        )
        num_cols = [x.columns[i] for i in index_cols]
        x_num = x.iloc(axis=1)[index_cols].fillna(0)
        if keep_input_grad:
            x_num_ts = torch.tensor(x_num.values, dtype=float, requires_grad=True)  # keep the grad
        else:
            x_num_ts = torch.tensor(x_num.values, dtype=float)

        num_col_ts = self.tokenizer(
            num_cols,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        encoded_inputs["x_num"] = x_num_ts
        encoded_inputs["num_col_input_ids"] = num_col_ts["input_ids"]
        encoded_inputs["num_att_mask"] = num_col_ts["attention_mask"]  # mask out attention

        return encoded_inputs

    def forward(self, cols: List[str], x: torch.Tensor) -> Dict:
        """
        Processes the input data and generates encoded inputs suitable for model encoding.

        Parameters
        ----------
        cols: List[str]
            A list containing all column names in order.

        x: torch.Tensor
            The tensor containing numerical features.

        Returns
        -------
        Dict
            - 'x_num': Tensor containing numerical features.
            - 'num_col_input_ids': Tensor containing tokenized IDs of numerical columns.
            - 'num_att_mask': Attention mask for the numerical column tokens.
        """
        encoded_inputs = {
            "x_num": None,
            "num_col_input_ids": None,
        }
        num_cols = cols
        num_col_ts = self.tokenizer(
            num_cols,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded_inputs["x_num"] = x
        encoded_inputs["num_col_input_ids"] = num_col_ts["input_ids"]
        encoded_inputs["num_att_mask"] = num_col_ts["attention_mask"]  # mask out attention

        return encoded_inputs


class FeatureProcessor(nn.Module):
    """Process inputs from feature extractor to map them to embeddings."""

    def __init__(
        self,
        vocab_size: int = None,
        hidden_dim: int = 128,
        hidden_dropout_prob: float = 0,
        pad_token_id: int = 0,
        device: Union[str, torch.device] = "cuda:0",
    ):
        """
        The initialization method for feature processor.

        Parameters
        ----------
        vocab_size : int, optional
            The size of the vocabulary.
        hidden_dim : int, optional
            The dimension of the hidden layer, by default 128.
        hidden_dropout_prob : float, optional
            The dropout probability for the hidden layer, by default 0.
        pad_token_id : int, optional
            The index of the padding token, by default 0.
        device : Union[str, torch.device], optional
            The device to run the module on, by default "cuda:0".
        """
        super().__init__()
        self.word_embedding = WordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
        )
        self.num_embedding = NumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs: torch.Tensor, att_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Averages the embeddings based on the attention mask.

        Parameters
        ----------
        embs : torch.Tensor
            The embeddings tensor.
        att_mask : torch.Tensor, optional
            The attention mask to apply on the embeddings. If None, the mean of the embeddings is returned, by default None.

        Returns
        -------
        torch.Tensor
            The resulting averaged embeddings.
        """
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(1) / att_mask.sum(1, keepdim=True).to(embs.device)
            return embs

    def forward(
        self,
        x_num: torch.Tensor = None,
        num_col_input_ids: torch.Tensor = None,
        num_att_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the FeatureProcessor module.

        Parameters
        ----------
        x_num : torch.Tensor, optional
            The numerical features.
        num_col_input_ids : torch.Tensor, optional
            The input IDs for numerical columns.
        num_att_mask : torch.Tensor, optional
            The attention mask.

        Returns
        -------
        torch.Tensor
            The processed feature embeddings.
        """
        x_num = x_num.to(self.device)
        num_col_emb = self.word_embedding(num_col_input_ids.to(self.device))
        num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)

        num_feat_embedding = self.num_embedding(num_col_emb, x_num)
        num_feat_embedding = self.align_layer(num_feat_embedding).float()

        attention_mask = torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(
            num_feat_embedding.device
        )
        return {"embedding": num_feat_embedding, "attention_mask": attention_mask}


class CLSToken(nn.Module):
    """Add a learnable cls token embedding at the end of each sequence."""

    def __init__(self, hidden_dim: int):
        """
        The initialization method for CLSToken.

        Parameters
        ----------
        hidden_dim : int
            The dimension of the hidden layer.
        """
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions) -> torch.Tensor:
        """
        Expands the CLS token embedding to match the leading dimensions of the input.

        Parameters
        ----------
        leading_dimensions : tuple
            A variable number of integer arguments representing the leading dimensions to which the CLS token embedding will be expanded.

        Returns
        -------
        torch.Tensor
            Expanded CLS token embedding.
        """
        new_dims = (1,) * (len(leading_dimensions) - 1)
        # cls token (128,) -> view(*new_dims, -1) -> (1, 128)
        # (1, 128) -> expand(*leading_dimensions, -1) -> (64, 1, 128)
        # here expand means "shared", the cls token embedding remains the same for each sample
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Performs a forward pass by adding a learnable CLS token to the embedding.

        Parameters
        ----------
        embedding : torch.Tensor
            The input embedding tensor.
        attention_mask : torch.Tensor, optional
            The attention mask for the input tensor, by default None.

        Returns
        -------
        torch.Tensor
            Output embedding with the CLS token added.
        """
        # embedding shape: (64, 11, 128), where 11 is the largest sequence length after tokenizing
        # after concat, learnable cls token [self.weight] is added to each semantic embedding
        # embedding shape: (64, d+1, 128)
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {"embedding": embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.shape[0], 1).to(attention_mask.device),
                    attention_mask,
                ],
                1,
            )
        outputs["attention_mask"] = attention_mask
        return outputs

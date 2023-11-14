import os
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.init as nn_init
from torch import Tensor, nn
from transformers import BertTokenizerFast

from .....config import C as conf


class WordEmbedding(nn.Module):
    """
    Encode tokens drawn from column names
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NumEmbedding(nn.Module):
    """
    Encode tokens drawn from column names and the corresponding numerical features.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))  # add bias
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

    def forward(self, col_emb, x_ts) -> Tensor:
        """args:
        col_emb: numerical column embedding, (# numerical columns, emb_dim)
        x_ts: numerical features, (bs, emb_dim)
        """
        col_emb = col_emb.unsqueeze(0).expand((x_ts.shape[0], -1, -1))
        feat_emb = col_emb * x_ts.unsqueeze(-1).float() + self.num_bias
        return feat_emb


class FeatureTokenizer:
    """
    Process input dataframe to input indices towards encoder,
    usually used to build dataloader for paralleling loading.
    """

    def __init__(
        self,
        disable_tokenizer_parallel=True,
        **kwargs,
    ):
        """args:
        disable_tokenizer_parallel: true if use extractor for collator function in torch.DataLoader
        """
        cache_dir = conf["cache_path"]
        os.makedirs(cache_dir, exist_ok=True)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
        self.tokenizer.__dict__["model_max_length"] = 512
        if disable_tokenizer_parallel:  # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, x, shuffle=False, keep_input_grad=False) -> Dict:
        """
        Parameters
        ----------
        x: pd.DataFrame
            with column names and features.

        shuffle: bool
            if shuffle column order during the training.

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
            }
        """
        encoded_inputs = {"x_num": None, "num_col_input_ids": None}
        num_cols = x.columns.tolist() if not shuffle else np.random.shuffle(x.columns.tolist())
        x_num = x[num_cols].fillna(0)

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

    def forward(self, cols, x) -> Dict:
        """
        Parameters
        ----------
        cols: List[str]
            Contain all column names in order.

        x: torch.Tensor

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
            }
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
    """
    Process inputs from feature extractor to map them to embeddings.
    """

    def __init__(
        self,
        vocab_size=None,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        device="cuda:0",
    ):
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

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(1) / att_mask.sum(1, keepdim=True).to(embs.device)
            return embs

    def forward(
        self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        **kwargs,
    ) -> Tensor:
        """args:
        x: pd.DataFrame with column names and features.
        shuffle: if shuffle column order during the training.
        num_mask: indicate the NaN place of numerical features, 0: NaN 1: normal.
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
    """add a learnable cls token embedding at the end of each sequence."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)
        # cls token (128,) -> view(*new_dims, -1) -> (1, 128)
        # (1, 128) -> expand(*leading_dimensions, -1) -> (64, 1, 128)
        # here expand means "shared", the cls token embedding remains the same for each sample
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
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

import os
import numpy as np
import pandas as pd
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .....specification import HeteroMapTableSpecification, RKMETableSpecification
from .feature_extractor import *
from .trainer import Trainer, TransTabCollatorForCL


class HeteroMap(nn.Module):
    """
    This class is based on 'TransTab' project as described in the paper
    "TransTab: A flexible transferable tabular learning framework". The original project is available at
    https://github.com/RyanWangZf/transtab and is licensed under the BSD 2-Clause License.

    Modifications:
    - Simplified the original code to focus primarily on methods related to numerical features.
    - Retained only the unsupervised training method.
    - While the original paper and the TransTab framework utilized the module for final predictions, this version
      is modified for feature extraction purposes only.

    The class implements a neural network module for processing tabular data, specifically tuned for numerical features.

    Args:
        feature_tokenizer (FeatureTokenizer, optional): Tokenizer for feature representation.
        hidden_dim (int, optional): Dimension of hidden layer.
        num_layer (int, optional): Number of layers in the transformer encoder.
        num_attention_head (int, optional): Number of attention heads in the transformer.
        hidden_dropout_prob (float, optional): Dropout probability for hidden layers.
        ffn_dim (int, optional): Dimension of feedforward network.
        projection_dim (int, optional): Dimension for projection head.
        overlap_ratio (float, optional): Overlap ratio for tokenization.
        num_partition (int, optional): Number of partitions for collation.
        temperature (float, optional): Temperature parameter for contrastive learning.
        base_temperature (float, optional): Base temperature parameter.
        activation (str, optional): Activation function for transformer layers.
        device (str, optional): Device to run the model on.
        checkpoint (str, optional): Path to a pre-trained model checkpoint.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        feature_tokenizer=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.5,
        num_partition=3,
        temperature=10,
        base_temperature=10,
        activation="relu",
        device="cuda:0",
        checkpoint=None,
        **kwargs,
    ):
        super(HeteroMap, self).__init__()

        self.model_args = {
            "num_partition": num_partition,
            "overlap_ratio": overlap_ratio,
            "hidden_dim": hidden_dim,
            "num_layer": num_layer,
            "num_attention_head": num_attention_head,
            "hidden_dropout_prob": hidden_dropout_prob,
            "ffn_dim": ffn_dim,
            "projection_dim": projection_dim,
            "activation": activation,
        }
        self.model_args.update(kwargs)

        if feature_tokenizer is None:
            feature_tokenizer = FeatureTokenizer(**kwargs)

        self.feature_tokenizer = feature_tokenizer

        self.feature_processor = FeatureProcessor(
            vocab_size=feature_tokenizer.vocab_size,
            pad_token_id=feature_tokenizer.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
        )

        self.encoder = TransformerMultiLayer(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )
        self.cls_token = CLSToken(hidden_dim=hidden_dim)
        self.collate_fn = TransTabCollatorForCL(
            feature_tokenizer=feature_tokenizer, overlap_ratio=overlap_ratio, num_partition=num_partition
        )

        self.projection_head = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.device = device
        self.to(device)

    @staticmethod
    def load(checkpoint=None):
        """Load the model state_dict and feature_tokenizer configuration
        from the ``checkpoint``.

        Parameters
        ----------
        checkpoint: str
            the directory path to load.
        """
        # load model weight state dict
        model_info = torch.load(checkpoint, map_location="cpu")
        model = HeteroMap(**model_info["model_args"])
        model.load_state_dict(model_info["model_state_dict"], strict=False)
        return model

    def save(self, checkpoint):
        """Save the model state_dict and feature_tokenizer configuration
        to the ``checkpoint``.

        Parameters
        ----------
        checkpoint: str
            the directory path to save.
        """
        # save model weight state dict
        model_info = {
            "model_state_dict": self.state_dict(),
            "model_args": self.model_args
        }
        torch.save(model_info, checkpoint)

    def forward(self, x, y=None):
        # do positive sampling
        feat_x_list = []
        if isinstance(x, dict):
            # pretokenized inputs
            for input_x in x["input_sub_x"]:
                feat_x = self.feature_processor(**input_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                feat_x_proj = feat_x[:, 0, :]
                feat_x_proj = self.projection_head(feat_x_proj)
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f"expect input x to be dict(pretokenized), get {type(x)} instead")

        # compute cl loss (multi-view InfoNCE loss)
        feat_x_multiview = torch.stack(feat_x_list, axis=1)  # bs, n_view, emb_dim
        loss = self._self_supervised_contrastive_loss(feat_x_multiview)
        return loss

    # def hetero_mapping(self, rkme_spec: RKMETableSpecification, features: dict) -> HeteroMapTableSpecification:
    def hetero_mapping(self, rkme_spec: RKMETableSpecification, features: dict) -> HeteroMapTableSpecification:
        hetero_spec = HeteroMapTableSpecification()
        data = rkme_spec.get_z()
        cols = [features.get(str(i), "") for i in range(data.shape[1])]
        hetero_input_df = pd.DataFrame(data=data, columns=cols)
        hetero_embedding = self._extract_batch_features(hetero_input_df)
        hetero_spec.generate_stat_spec_from_system(hetero_embedding, rkme_spec)
        return hetero_spec

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i + 1][:overlap]])
            elif overlap > 0 and i == n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i - 1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def _extract_features(self, x, cols=None):
        """Make forward pass given the input feature ``x``.

        Parameters
        ----------
        x: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of feature_tokenizer

        Returns
        -------
        output_features: numpy.ndarray
            the [CLS] embedding at the end of transformer encoder.
        """
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.feature_tokenizer(x)
        elif isinstance(x, torch.Tensor):
            inputs = self.feature_tokenizer.forward(cols, x)
        else:
            raise ValueError(f"feature_tokenizer takes inputs with dict or pd.DataFrame, find {type(x)}.")

        outputs = self.feature_processor(**inputs)  # outputs is dict, "embedding" and "mask"
        outputs = self.cls_token(**outputs)  # add the cls embedding

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs)  # bs, seqlen+1, hidden_dim
        output_features = encoder_output[:, 0, :]

        return output_features

    def _extract_batch_features(self, x_test, eval_batch_size=256) -> np.ndarray:
        self.eval()
        output_feas_list = []
        for i in range(0, len(x_test), eval_batch_size):
            bs_x_test = x_test.iloc[i : i + eval_batch_size]
            with torch.no_grad():
                output_features = self._extract_features(bs_x_test).detach().cpu().numpy()
                output_feas_list.append(output_features)

        all_output_features = np.concatenate(output_feas_list, 0)
        return all_output_features

    def _self_supervised_contrastive_loss(self, features):
        """Compute the self-supervised VPCL loss.

        Parameters
        ----------
        features: torch.Tensor
            the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.

        Returns
        -------
        loss: torch.Tensor
            the computed self-supervised VPCL loss.
        """
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        # [[0,1],[2,3]] -> [0,2,1,3]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "selu":
        return F.selu
    elif activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TransformerLayer(nn.Module):
    __config__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        device=None,
        dtype=None,
        use_layer_norm=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g  # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None, **kwargs) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:  # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        return x


class TransformerMultiLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation="relu",
    ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=hidden_dim,
                    nhead=num_attention_head,
                    dropout=hidden_dropout_prob,
                    dim_feedforward=ffn_dim,
                    batch_first=True,
                    layer_norm_eps=1e-5,
                    norm_first=False,
                    use_layer_norm=True,
                    activation=activation,
                )
            ]
        )
        if num_layer > 1:
            encoder_layer = TransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
            )
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer - 1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        """args:
        embedding: bs, num_token, hidden_dim
        """
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .....utils import allocate_cuda_idx, choose_device
from .....specification import HeteroMapTableSpecification, RKMETableSpecification
from .feature_extractor import CLSToken, FeatureProcessor, FeatureTokenizer
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
    """

    def __init__(
        self,
        feature_tokenizer: FeatureTokenizer = None,
        hidden_dim: int = 128,
        num_layer: int = 2,
        num_attention_head: int = 8,
        hidden_dropout_prob: float = 0,
        ffn_dim: int = 256,
        projection_dim: int = 128,
        overlap_ratio: float = 0.5,
        num_partition: int = 3,
        temperature: int = 10,
        base_temperature: int = 10,
        activation: Union[str, Callable] = "relu",
        cuda_idx: int = None,
        **kwargs,
    ):
        """
        The initialization method for hetero map.

        Parameters
        ----------
        feature_tokenizer : FeatureTokenizer, optional
            Tokenizer for feature representation, by default None
        hidden_dim : int, optional
            Dimension of hidden layer, by default 128
        num_layer : int, optional
            Number of layers in the transformer encoder, by default 2
        num_attention_head : int, optional
            Number of attention heads in the transformer, by default 8
        hidden_dropout_prob : int, optional
            Dropout probability for hidden layers, by default 0
        ffn_dim : int, optional
            Dimension of feedforward network, by default 256
        projection_dim : int, optional
            Dimension for projection head, by default 128
        overlap_ratio : float, optional
            Overlap ratio for tokenizatio, by default 0.5
        num_partition : int, optional
            Number of partitions for collatio, by default 3
        temperature : int, optional
            Temperature parameter for contrastive learnin, by default 10
        base_temperature : int, optional
            Base temperature paramete, by default 10
        activation : Union[str, Callable], optional
            Activation function for transformer layer, by default "relu"
        cuda_idx : int, optional
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device, by default None
        kwargs:
            Additional arguments to be passed to the feature tokenizer
        """
        super(HeteroMap, self).__init__()

        cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        device = choose_device(cuda_idx)
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
        self.to(device)

    def to(self, device: Union[str, torch.device]):
        """Moves the model and all its submodules to the specified device

        Parameters
        ----------
        device : Union[str, torch.device]
            The target device to which the model and its components should be moved.

        Returns
        -------
        HeteroMap
            The instance of HeteroMap after moving to the specified device.
        """
        super(HeteroMap, self).to(device)
        if hasattr(self, "feature_processor"):
            self.feature_processor.device = device
        self.device = device
        return self

    @staticmethod
    def load(checkpoint: str = None):
        """Load the model state_dict and architecture configuration from the specified checkpoint.

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

    def save(self, checkpoint: str):
        """Save the model state_dict and architecture configuration to the specified checkpoint.

        Parameters
        ----------
        checkpoint: str
            the directory path to save.
        """
        # save model weight state dict
        model_info = {"model_state_dict": self.state_dict(), "model_args": self.model_args}
        torch.save(model_info, checkpoint)

    def forward(self, x: dict):
        """Processes the input data 'x', performs positive sampling, and computes contrastive loss.

        Parameters
        ----------
        x : dict
            Pre-tokenized input tabular data in the form of a dictionary

        Returns
        -------
        torch.Tensor
            The self-supervised VPCL loss
        """
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

    def hetero_mapping(self, rkme_spec: RKMETableSpecification, features: dict) -> HeteroMapTableSpecification:
        """Generate HeteroMapTableSpecification from given tabular data's statistical specification and descriptions of features.

        Parameters
        ----------
        rkme_spec : RKMETableSpecification
            The RKME specification from the tabular data
        features : dict
            A dictionary mapping each feature's numerical identifier to its semantic description.

        Returns
        -------
        HeteroMapTableSpecification
            The resulting HeteroMapTableSpecification
        """
        hetero_spec = HeteroMapTableSpecification()
        data = rkme_spec.get_z()
        cols = [features.get(str(i), "Unknown Feature") for i in range(data.shape[1])]
        hetero_input_df = pd.DataFrame(data=data, columns=cols)
        hetero_embedding = self._extract_batch_features(hetero_input_df)
        hetero_spec.generate_stat_spec_from_system(hetero_embedding, rkme_spec)
        return hetero_spec

    def _build_positive_pairs(self, x: pd.DataFrame, n: int):
        """
        Builds positive pairs by splitting the input DataFrame into 'n' parts with some overlap.

        Parameters
        ----------
        x : pd.DataFrame
            The input DataFrame to be split.
        n : int
            The number of partitions to divide the DataFrame into.

        Returns
        -------
        List[pd.DataFrame]
            A list of DataFrames, each representing a partition of the input DataFrame with some overlap.
        """
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

    def _extract_features(self, x: Union[dict, pd.DataFrame], cols=None):
        """Performs a forward pass with the given input feature `x`, and extracts features.

        Parameters
        ----------
        x: Union[dict, pd.DataFrame]
            pd.DataFrame: A batch of raw tabular samples
            dict: The output of feature_tokenizer

        Returns
        -------
        output_features: numpy.ndarray
            The [CLS] embedding at the end of transformer encoder
        """
        if isinstance(x, pd.DataFrame):
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

    def _extract_batch_features(self, x_test: pd.DataFrame, eval_batch_size=256) -> np.ndarray:
        """Performs forward passes on a batch of input features `x_test`, extracting and returning features as an array.

        Parameters
        ----------
        x_test : pd.DataFrame
            A batch of raw tabular samples
        eval_batch_size : int, optional
            The size of each batch for processing, by default 256

        Returns
        -------
        np.ndarray
            An array containing the extracted features from all batches
        """
        self.eval()
        output_feas_list = []
        for i in range(0, len(x_test), eval_batch_size):
            bs_x_test = x_test.iloc[i : i + eval_batch_size]
            with torch.no_grad():
                output_features = self._extract_features(bs_x_test).detach().cpu().numpy()
                output_feas_list.append(output_features)

        all_output_features = np.concatenate(output_feas_list, 0)
        return all_output_features

    def _self_supervised_contrastive_loss(self, features: torch.Tensor):
        """
        Compute the self-supervised VPCL loss.

        Parameters
        ----------
        features : torch.Tensor
            The encoded features of multiple partitions of input tables, with shape (bs, n_partition, proj_dim).

        Returns
        -------
        torch.Tensor
            The computed self-supervised VPCL loss.
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


class TransformerLayer(nn.Module):
    """A custom Transformer layer implemented as a PyTorch module."""

    __config__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        device: Union[str, torch.device] = None,
        dtype: torch.dtype = None,
        use_layer_norm: bool = True,
    ):
        """
        The initialization method for transformer layer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input
        nhead : int
            The number of heads in the multiheadattention models
        dim_feedforward : int, optional
            The dimension of the feedforward network model, by default 2048
        dropout : float, optional
            The dropout value, by default 0.1
        activation : Union[str, Callable], optional
            The activation function to use, by default F.relu
        layer_norm_eps : float, optional
            The epsilon used for layer normalization, by default 1e-5
        batch_first : bool, optional
            Whether to use (batch, seq, feature) format for input and output tensors, by default True
        norm_first : bool, optional
            Whether to perform layer normalization before attention and feedforward operations, by default False
        device : Union[str, torch.device], optional
            The device on which the layer is to be run, by default None
        dtype : torch.dtype, optional
            The data type of the layer's parameters, by default None
        use_layer_norm : bool, optional
            Whether to use layer normalization, by default True
        """
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
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies a self-attention block to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the self-attention block.
        attn_mask : torch.Tensor
            The attention mask for the self-attention operation.
        key_padding_mask : torch.Tensor
            The key padding mask for the self-attention operation.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the self-attention block.
        """
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
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a feed-forward block to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the feed-forward block.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the feed-forward block.
        """
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g  # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    @staticmethod
    def _get_activation_fn(activation: str) -> Callable:
        """
        Retrieves the activation function based on the provided activation name.

        Parameters
        ----------
        activation : str
            Name of the activation function. Supported values are "relu", "gelu", "selu", and "leakyrelu".

        Returns
        -------
        Callable
            The corresponding activation function from torch.nn.functional.
        """
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "selu":
            return F.selu
        elif activation == "leakyrelu":
            return F.leaky_relu
        raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pass the input through the encoder layer.

        Parameters
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence, by default None
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch, by default None
        is_causal : torch.Tensor, optional
            A flag indicating whether the layer should be causal, by default None

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the encoder layer.
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
    """A custom multi-layer Transformer module."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layer: int = 2,
        num_attention_head: int = 2,
        hidden_dropout_prob: float = 0,
        ffn_dim: int = 256,
        activation: Union[str, Callable] = "relu",
    ):
        """
        The initialization method for align transformer multilayer.

        Parameters
        ----------
        hidden_dim : int, optional
            Dimension of the hidden layer in the Transformer, by default 128.
        num_layer : int, optional
            Number of Transformer layers, by default 2.
        num_attention_head : int, optional
            Number of attention heads in each Transformer layer, by default 2.
        hidden_dropout_prob : float, optional
            Dropout probability for the hidden layers, by default 0.
        ffn_dim : int, optional
            Dimension of the feedforward network model, by default 256.
        activation : Union[str, Callable], optional
            The activation function to be used, by default "relu".
        """
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

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Passes the input embedding through the Transformer encoder layers.

        Parameters
        ----------
        embedding : torch.Tensor
            The input embedding tensor with shape (batch size, number of tokens, hidden dimension).
        attention_mask : torch.Tensor, optional
            The attention mask for the input tensor, by default None.

        Returns
        -------
        Tensor
            The output tensor after processing through Transformer encoder layers.
        """
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

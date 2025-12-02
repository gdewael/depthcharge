"""Custom Transformer layers for depthcharge."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attn import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input.
    nhead : int
        The number of heads in the multiheadattention models.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
    dropout : float, optional
        The dropout value.
    activation : str or callable, optional
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: "relu"
    layer_norm_eps : float, optional
        The eps value in layer normalization components.
    norm_first : bool, optional
        If True, layer norm is done prior to attention and feedforward
        operations (pre-norm). Otherwise it's done after (post-norm).
    attention_backend : str, optional
        Attention implementation: "native" (default), "flex", or "sdpa".

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | callable = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True,
        attention_backend: str = "sdpa",
    ) -> None:
        """Initialize a TransformerEncoderLayer."""
        super().__init__()

        assert (
            batch_first==True
        ), "TransformerEncoderLayer requires batch_first=True"

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.attention_backend = attention_backend

        if attention_backend == "native":
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.self_attn = MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
                attention_backend=attention_backend,
            )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                f"activation should be 'relu', 'gelu', or a callable, "
                f"not {activation}"
            )

        

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Parameters
        ----------
        src : Tensor of shape (batch_size, seq_len, d_model)
            The sequence to the encoder layer.
        src_mask : Tensor, optional
            The mask for the src sequence with shape (seq_len, seq_len)
            or (batch_size * num_heads, seq_len, seq_len).
        src_key_padding_mask : Tensor, optional
            The mask for the src keys per batch with shape
            (batch_size, seq_len). True values indicate positions that
            should be masked (not attended to).
        is_causal : bool, optional
            If True, applies a causal mask as src_mask. Should not be
            provided together with src_mask.

        Returns
        -------
        Tensor of shape (batch_size, seq_len, d_model)
            The output of the encoder layer.

        """
        if self.norm_first:
            # Pre-norm architecture
            src = src + self._sa_block(
                self.norm1(src),
                src_mask,
                src_key_padding_mask,
                is_causal,
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-norm architecture (default for PyTorch)
            src = self.norm1(
                src
                + self._sa_block(
                    src,
                    src_mask,
                    src_key_padding_mask,
                    is_causal,
                )
            )
            src = self.norm2(src + self._ff_block(src))

        return src

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    """A custom Transformer decoder layer.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input.
    nhead : int
        The number of heads in the multiheadattention models.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
    dropout : float, optional
        The dropout value.
    activation : str or callable, optional
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: "relu"
    layer_norm_eps : float, optional
        The eps value in layer normalization components.
    norm_first : bool, optional
        If True, layer norm is done prior to attention and feedforward
        operations (pre-norm). Otherwise it's done after (post-norm).
    attention_backend : str, optional
        Attention implementation: "native" (default), "flex", or "sdpa".

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | callable = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True,
        attention_backend: str = "sdpa",
    ) -> None:
        """Initialize a TransformerDecoderLayer."""
        super().__init__()

        assert (
            batch_first==True
        ), "TransformerDecoderLayer requires batch_first=True"

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.attention_backend = attention_backend

        attn_args = {
            "embed_dim": d_model,
            "num_heads": nhead,
            "dropout": dropout,
            "batch_first": True,
        }
        
        if attention_backend == "native":
            self.self_attn = nn.MultiheadAttention(
                **attn_args,
            )
        else:
            self.self_attn = MultiheadAttention(
                **attn_args,
                attention_backend=attention_backend,
            )
            
        if attention_backend == "native":
            self.multihead_attn = nn.MultiheadAttention(
                **attn_args,
            )
        else:
            self.multihead_attn = MultiheadAttention(
                **attn_args,
                attention_backend=attention_backend,
            )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                f"activation should be 'relu', 'gelu', or a callable, "
                f"not {activation}"
            )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt : Tensor of shape (batch_size, tgt_seq_len, d_model)
            The sequence to the decoder layer.
        memory : Tensor of shape (batch_size, src_seq_len, d_model)
            The sequence from the last layer of the encoder.
        tgt_mask : Tensor, optional
            The mask for the tgt sequence.
        memory_mask : Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask : Tensor, optional
            The mask for the tgt keys per batch with shape
            (batch_size, tgt_seq_len).
        memory_key_padding_mask : Tensor, optional
            The mask for the memory keys per batch with shape
            (batch_size, src_seq_len).
        tgt_is_causal : bool, optional
            If True, applies a causal mask as tgt_mask.
        memory_is_causal : bool, optional
            If True, applies a causal mask as memory_mask.

        Returns
        -------
        Tensor of shape (batch_size, tgt_seq_len, d_model)
            The output of the decoder layer.

        """
        if self.norm_first:
            # Pre-norm architecture
            tgt = tgt + self._sa_block(
                self.norm1(tgt),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
            )
            tgt = tgt + self._mha_block(
                self.norm2(tgt),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # Post-norm architecture (default for PyTorch)
            tgt = self.norm1(
                tgt
                + self._sa_block(
                    tgt,
                    tgt_mask,
                    tgt_key_padding_mask,
                    tgt_is_causal,
                )
            )
            tgt = self.norm2(
                tgt
                + self._mha_block(
                    tgt,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt
    
    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:

        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # multihead (cross) attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module, from PyTorch source code
    import copy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    """Custom Transformer encoder.

    A stack of N encoder layers, replicates `torch.nn.TransformerEncoder`.

    Parameters
    ----------
    encoder_layer : TransformerEncoderLayer
        An instance of the TransformerEncoderLayer class.
    num_layers : int
        The number of sub-encoder-layers in the encoder.
    norm : nn.Module, optional
        The layer normalization component (optional).

    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize a TransformerEncoder."""
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Parameters
        ----------
        src : Tensor of shape (batch_size, seq_len, d_model)
            The sequence to the encoder.
        mask : Tensor, optional
            The mask for the src sequence with shape (seq_len, seq_len)
            or (batch_size * num_heads, seq_len, seq_len).
        src_key_padding_mask : Tensor, optional
            The mask for the src keys per batch with shape
            (batch_size, seq_len). True values indicate positions that
            should be masked (not attended to).
        is_causal : bool, optional
            If True, applies a causal mask as src_mask. Should not be
            provided together with mask.

        Returns
        -------
        Tensor of shape (batch_size, seq_len, d_model)
            The output of the encoder.

        """
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Custom Transformer decoder.

    A stack of N decoder layers. This implementation exactly replicates the
    behavior of `torch.nn.TransformerDecoder` and serves as a foundation for
    future extensions.

    Parameters
    ----------
    decoder_layer : TransformerDecoderLayer
        An instance of the TransformerDecoderLayer class.
    num_layers : int
        The number of sub-decoder-layers in the decoder.
    norm : nn.Module, optional
        The layer normalization component (optional).

    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize a TransformerDecoder."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs (and masks) through the decoder layers in turn.

        Parameters
        ----------
        tgt : Tensor of shape (batch_size, tgt_seq_len, d_model)
            The sequence to the decoder.
        memory : Tensor of shape (batch_size, src_seq_len, d_model)
            The sequence from the last layer of the encoder.
        tgt_mask : Tensor, optional
            The mask for the tgt sequence.
        memory_mask : Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask : Tensor, optional
            The mask for the tgt keys per batch with shape
            (batch_size, tgt_seq_len).
        memory_key_padding_mask : Tensor, optional
            The mask for the memory keys per batch with shape
            (batch_size, src_seq_len).
        tgt_is_causal : bool, optional
            If True, applies a causal mask as tgt_mask.
        memory_is_causal : bool, optional
            If True, applies a causal mask as memory_mask.

        Returns
        -------
        Tensor of shape (batch_size, tgt_seq_len, d_model)
            The output of the decoder.

        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

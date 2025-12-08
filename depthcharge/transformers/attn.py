"""Custom attention mechanisms for depthcharge transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend


class MultiheadAttention(nn.Module):
    """Multi-head attention with nested tensor support via scaled_dot_product_attention.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model.
    num_heads : int
        Number of parallel attention heads. embed_dim must be divisible
        by num_heads.
    dropout : float, optional
        Dropout probability on attention weights. Default: 0.0
    bias : bool, optional
        If True, add bias to input/output projections. Default: True
    batch_first : bool, optional
        If True, input and output tensors are (batch, seq, feature).
        Only batch_first=True is supported. Default: True
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K.
        If None, no rotary embeddings are used. Default: None
    enable_sdpa_math : bool, optional
        If True, enable SDPA math kernel. Default: True
    enable_sdpa_mem_efficient : bool, optional
        If True, enable SDPA memory efficient kernel. Default: True
    enable_sdpa_flash_attention : bool, optional
        If True, enable SDPA flash attention kernel. Default: True

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rotary_embedding: nn.Module | None = None,
        enable_sdpa_math: bool = True,
        enable_sdpa_mem_efficient: bool = True,
        enable_sdpa_flash_attention: bool = True,
    ) -> None:
        """Initialize MultiheadAttention."""
        super().__init__()

        assert (
            batch_first is True
        ), "MultiheadAttention requires batch_first=True"

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"


        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.rotary_embedding = rotary_embedding

        self.context_manager = []
        if enable_sdpa_math:
            self.context_manager.append(SDPBackend.MATH)
        if enable_sdpa_flash_attention:
            self.context_manager.append(SDPBackend.FLASH_ATTENTION)
        if enable_sdpa_mem_efficient:
            self.context_manager.append(SDPBackend.EFFICIENT_ATTENTION)

        self.use_context_manager = not all(
            [enable_sdpa_math, enable_sdpa_flash_attention, enable_sdpa_mem_efficient]
        )

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass of multi-head attention.

        Supports both dense and nested tensors. For variable-length sequences,
        pass nested tensors to avoid padding overhead.

        Parameters
        ----------
        query : Tensor or NestedTensor
            Query embeddings. For dense: (batch, tgt_len, embed_dim).
            For nested: variable-length sequences without padding.
        key : Tensor or NestedTensor
            Key embeddings. For dense: (batch, src_len, embed_dim).
            For nested: variable-length sequences without padding.
        value : Tensor or NestedTensor
            Value embeddings. For dense: (batch, src_len, embed_dim).
            For nested: variable-length sequences without padding.
        key_padding_mask : Tensor, optional
            Placeholder for compatibility with torch.nn.MultiheadAttention.
            Should always be None.
        attn_mask : Tensor, optional
            Placeholder for compatibility with torch.nn.MultiheadAttention.
            Should always be None.
        need_weights : bool, optional
            Placeholder for compatibility with torch.nn.MultiheadAttention.
            Always returns None for attention weights. Default: False
        is_causal : bool, optional
            If True, apply causal masking (lower triangular). Works with
            both dense and nested tensors. Default: False
        positions : Tensor, optional
            Position values for rotary embeddings. Only used if rotary_embedding
            is configured. If None, uses integer positions. Can be:
            - (batch, seq_len) for per-sample positions (e.g., m/z values)
            - (seq_len,) for shared positions
            - Nested tensor (batch, jagged_seq_len) for jagged sequences
            Default: None

        Returns
        -------
        attn_output : Tensor or NestedTensor
            Attention output. Same layout as inputs (dense or nested).
        attn_weights : None
            Placeholder for compatibility with `torch.nn.MultiHeadAttention`, always None.

        """
        assert (
            key_padding_mask is None
        ), """
        key_padding_mask should not be used with scaled_dot_product_attention.
        Pass nested query, key and value tensors instead when irregular sequence length.
        """
        assert (
            attn_mask is None
        ), """
        attn_mask should not be used with scaled_dot_product_attention.
        """

        # Apply input projections
        if query is key and key is value: # self-attention case
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            Q, K, V = qkv.chunk(3, dim=-1)
        else: # cross-attention case
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
            else:
                b_q, b_k, b_v = None, None, None

            Q = F.linear(query, w_q, b_q)
            K = F.linear(key, w_k, b_k)
            V = F.linear(value, w_v, b_v)
        
        # [bsz, seqlen, embed_dim] -> [bsz, nhead, seqlen, headdim]
        Q = Q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        K = K.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        V = V.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # Apply rotary embeddings
        if self.rotary_embedding is not None:
            Q, K = self.rotary_embedding(Q, K, positions=positions)

        attn_output = self._sdpa_attention(Q, K, V, is_causal=is_causal)
        
        # [bsz, nhead, seqlen, headdim] -> [bsz, seqlen, embed_dim] 
        attn_output = attn_output.transpose(1, 2).flatten(-2, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output, None

    def _sdpa_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        if self.use_context_manager:
            with sdpa_kernel(self.context_manager):
                attn_output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        return attn_output
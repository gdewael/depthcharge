"""Custom attention mechanisms for depthcharge transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend


class MultiheadAttention(nn.Module):
    """Custom multi-head attention.

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
        Currently only batch_first=True is supported. Default: True
    attention_backend : str, optional
        Attention implementation to use. Options:
        - "sdpa": Uses F.scaled_dot_product_attention (default)
        - "flex": Uses torch.nn.attention.flex_attention (requires PyTorch 2.5+)
        Default: "sdpa"
    enable_sdpa_math : bool, optional
        If True, enable SDPA math kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_mem_efficient : bool, optional
        If True, enable SDPA memory efficient kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_flash_attention : bool, optional
        If True, enable SDPA flash attention kernel when using "sdpa" backend.
        Default: True

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        attention_backend: str = "sdpa",
        enable_sdpa_math: bool = True,
        enable_sdpa_mem_efficient: bool = True,
        enable_sdpa_flash_attention: bool = True,
    ) -> None:
        """Initialize CustomMultiheadAttention."""
        super().__init__()

        assert (
            batch_first is True
        ), "CustomMultiheadAttention requires batch_first=True"

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        assert attention_backend in [
            "sdpa",
            "flex",
        ], f"attention_backend must be 'sdpa' or 'flex', got '{attention_backend}'"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.attention_backend = attention_backend
        
        self.context_manager = []
        if enable_sdpa_math:
            self.context_manager.append(SDPBackend.MATH)
        if enable_sdpa_flash_attention:
            self.context_manager.append(SDPBackend.FLASH_ATTENTION)
        if enable_sdpa_mem_efficient:
            self.context_manager.append(SDPBackend.EFFICIENT_ATTENTION)

        self.use_context_manager = False
        if self.attention_backend == "sdpa":
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
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass of multi-head attention.

        Parameters
        ----------
        query : Tensor of shape (batch, tgt_len, embed_dim)
            Query embeddings.
        key : Tensor of shape (batch, src_len, embed_dim)
            Key embeddings.
        value : Tensor of shape (batch, src_len, embed_dim)
            Value embeddings.
        key_padding_mask : Tensor, optional
            Mask for padding keys. Shape (batch, src_len).
            True indicates positions that should be masked (not attended to).
        need_weights : bool, optional
            Placeholder for compatibility with `torch.nn.MultiHeadAttention`.
            Non-functional. In reality, always False: never return attention weights. Default: False
        attn_mask : Tensor, optional
            Attention mask. Shape (tgt_len, src_len) or
            (batch * num_heads, tgt_len, src_len).
        is_causal : bool, optional
            If True, apply causal mask. Should not be used together
            with attn_mask. Default: False

        Returns
        -------
        attn_output : Tensor of shape (batch, tgt_len, embed_dim)
            Attention output.
        attn_weights : None
            Placeholder for compatibility with `torch.nn.MultiHeadAttention`, always None.

        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        
        if torch.equal(query, key) and torch.equal(key, value): # self-attention case
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

        # [bsz, seqlen, embed_dim] -> [bsz, nhead, seqlen, seqlen, headdim]
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attention_backend == "sdpa":
            attn_output = self._sdpa_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
        elif self.attention_backend == "flex":
            attn_output = self._flex_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
        else:
            raise ValueError(
                f"Unknown attention backend: {self.attention_backend}"
            )

        # [bsz, nhead, seqlen, seqlen, headdim] -> [bsz, seqlen, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None

    def _is_simple_causal_mask(self, attn_mask: Tensor) -> bool:
        # Only handle 2D masks (seq_len, seq_len)
        if attn_mask.dim() != 2:
            return False

        # Must be square
        if attn_mask.shape[0] != attn_mask.shape[1]:
            return False

        seq_len = attn_mask.shape[0]

        # Check for boolean mask (True = masked, False = not masked)
        if attn_mask.dtype == torch.bool:
            # Upper triangle should be True (masked), lower should be False
            expected = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_mask.device, dtype=torch.bool),
                diagonal=1
            )
            return torch.equal(attn_mask, expected)

        # Check for float mask (-inf = masked, 0.0 = not masked)
        # Allow for floating point precision issues
        lower_triangle = torch.tril(attn_mask, diagonal=0)
        upper_triangle = torch.triu(attn_mask, diagonal=1)

        # Lower triangle + diagonal should be all zeros (or close to it)
        lower_is_zero = torch.allclose(lower_triangle, torch.zeros_like(lower_triangle), atol=1e-6)

        # Upper triangle should be all -inf
        upper_is_neginf = torch.all(torch.isinf(upper_triangle) & (upper_triangle < 0))

        return lower_is_zero and upper_is_neginf

    def _sdpa_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        # Auto-detect simple causal masks and optimize by using is_causal=True
        if not is_causal and attn_mask is not None and self._is_simple_causal_mask(attn_mask):
            attn_mask = None
            is_causal = True

        if is_causal and attn_mask is not None:
            # Create explicit causal mask and combine with provided attn_mask
            tgt_len = Q.shape[2]
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            causal_mask_float = torch.zeros(
                tgt_len, tgt_len, device=Q.device, dtype=Q.dtype
            )
            causal_mask_float.masked_fill_(causal_mask, float("-inf"))

            attn_mask = attn_mask + causal_mask_float
            is_causal = False

        
        if key_padding_mask is not None and not (is_causal and attn_mask is None):
            # Convert key_padding_mask to attention mask format for SDPA
            key_padding_mask_expanded = key_padding_mask.view(
                key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1]
            )
            # Convert True -> -inf, False -> 0.0
            attn_mask_float = torch.zeros_like(
                key_padding_mask_expanded, dtype=Q.dtype
            )
            attn_mask_float.masked_fill_(key_padding_mask_expanded, float("-inf"))

            # Combine with existing attn_mask if present
            if attn_mask is not None:
                attn_mask = attn_mask + attn_mask_float
            else:
                attn_mask = attn_mask_float
                
        # NOTE: Important case implicitly handled: when is_causal is True and attn_mask is None.
        # In this case, we ignore key_padding_mask and apply simple causal maksing
        
        
        if self.use_context_manager:
            with sdpa_kernel(self.context_manager):
                 attn_output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        return attn_output

    def _flex_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        try:
            from torch.nn.attention import flex_attention
        except ImportError:
            raise ImportError(
                "flex_attention requires PyTorch 2.5+. "
                "Please upgrade PyTorch or use attention_backend='sdpa' or 'native'."
            )
        
        raise NotImplementedError(
            "flex_attention is not yet implemented in this MultiheadAttention class."
        )
        # Placeholder for future implementation

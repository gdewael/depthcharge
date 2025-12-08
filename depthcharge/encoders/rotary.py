"""Rotary Position Embeddings (RoPE) for Transformers."""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class RotaryEmbedding(nn.Module):
    """Rotary embedding as in RoFormer / RoPE.

    Applies rotary positional embeddings to query and key tensors in attention.
    Unlike additive positional encodings, RoPE rotates the Q and K vectors based
    on their position, preserving relative position information in the attention
    computation.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.

    """

    def __init__(
        self,
        head_dim: int,
        min_wavelength: float = 2 * np.pi,
        max_wavelength: float = 20_000 * np.pi,
    ) -> None:
        """Initialize RotaryEmbedding."""
        super().__init__()
        self.head_dim = head_dim
        
        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        thetas = 1.0 / base / (scale ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("thetas", thetas)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch, num_heads, seq_len, head_dim)
            or nested tensor (batch, num_heads, jagged_seq_len, head_dim)
        k : torch.Tensor
            Key tensor of shape (batch, num_heads, seq_len, head_dim)
            or nested tensor (batch, num_heads, jagged_seq_len, head_dim)
        positions : torch.Tensor, optional
            Position values for each element in the sequence.
            - If None: Use integer positions torch.arange(seq_len)
            - Shape: (batch, seq_len) for per-sample positions
            - Can be nested tensor for jagged sequences. Shape: (batch, jagged_seq_len)
            Examples: integer positions [0,1,2,..], or m/z values [100.5, 150.2, ...]
            Default: None

        Returns
        -------
        q_rotated : torch.Tensor
            Query with rotary embeddings applied, same shape as input
        k_rotated : torch.Tensor
            Key with rotary embeddings applied, same shape as input

        """
        route_to_collapsed = hasattr(q, 'is_nested') and q.is_nested

        if route_to_collapsed:
            q_to_rot = q._values  # (num_heads, total_tokens, head_dim)
            k_to_rot = k._values
        else:
            q_to_rot, k_to_rot = q, k    

        # position handling:
        if route_to_collapsed:
            if positions is not None and hasattr(positions, 'is_nested') and positions.is_nested:
                pos_collapsed = positions._values
            elif positions is None:
                offsets = q.offsets()
                offset_per_token = torch.repeat_interleave(offsets[:-1], offsets.diff())
                pos_collapsed = torch.arange(q_to_rot.size(1), device=q_to_rot.device) - offset_per_token
            else:
                pos_collapsed = positions.flatten()
            pos_to_use = pos_collapsed.unsqueeze(0).expand(q.size(1), -1)
            
        else:
            pos_to_use = self.default_pos(q_to_rot) if positions is None else positions
            if pos_to_use.ndim == 2 and q_to_rot.ndim == 4:
                pos_to_use = pos_to_use.unsqueeze(1).expand(-1, q_to_rot.size(1), -1)

        # Actual rotary embedding application
        sin, cos = self.get_rotations(pos_to_use, self.thetas)
        q_rot = q_to_rot * cos.to(q_to_rot) + self.rotate_every_two(q_to_rot) * sin.to(q_to_rot)
        k_rot = k_to_rot * cos.to(k_to_rot) + self.rotate_every_two(k_to_rot) * sin.to(k_to_rot)

        if route_to_collapsed:
            q_out = q.clone()
            k_out = k.clone()
            q_out._values.copy_(q_rot)
            k_out._values.copy_(k_rot)

            return q_out, k_out
        else:
            return q_rot, k_rot
    
    @staticmethod
    def default_pos(x):
        return torch.arange(x.size(-2), device=x.device).expand(*x.shape[:-1])

    @staticmethod
    def get_rotations(pos, thetas):
        mthetas = pos[..., None] * thetas # (..., seq_len, head_dim/2)
        sin, cos = map(
            lambda t: repeat(t, "b ... h  -> b ... (h j)", j=2),
            (mthetas.sin(), mthetas.cos()),
        )
        sin, cos = map(lambda t: t.to(thetas), (sin, cos))
        return sin, cos

    @staticmethod
    def rotate_every_two(x):
        x = x.clone()
        x = rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d j -> ... (d j)")

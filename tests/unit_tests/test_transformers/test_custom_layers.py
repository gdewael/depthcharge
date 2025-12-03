"""Test custom transformer layers for numerical equivalence with PyTorch implementation."""

import pytest
import torch
import torch.nn as nn
from depthcharge.utils import generate_tgt_mask
from depthcharge.transformers import MultiheadAttention
from depthcharge.transformers.layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
        

@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0), # dropout always 0 because PyTorch's SDPA is non-deterministic with dropout
            (256, 8, 2048, 0.0),
            (512, 8, 512, 0.0),
            (64, 4, 256, 0.0), 
        ],
    )
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
def test_encoder_equivalence(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_len, norm_first, is_causal
):
    """Test numerical equivalence with PyTorch TransformerEncoderLayer.
    By default, tests the equivalence of the sdpa attention backend."""
    
    torch.manual_seed(42)
    pytorch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
    )
    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
    )

    custom_layer.load_state_dict(pytorch_layer.state_dict())

    # Init input for both pytorch and custom separately to allow gradient comparison
    torch.manual_seed(43)
    src_pytorch = torch.randn(batch_size, seq_len, d_model)
    src_custom = src_pytorch.clone()
    src_pytorch.requires_grad = True
    src_custom.requires_grad = True
    
    key_padding_mask = torch.zeros(batch_size, seq_len).bool()
    key_padding_mask[:, -3:] = True
    
    torch.manual_seed(44)
    pytorch_output = pytorch_layer(
        src_pytorch,
        is_causal=False,
        src_mask = (generate_tgt_mask(seq_len) if is_causal else None), 
        src_key_padding_mask=key_padding_mask
    )
    torch.manual_seed(44)
    custom_output = custom_layer(
        src_custom,
        is_causal=is_causal,
        src_mask=None,
        src_key_padding_mask=key_padding_mask
    )

    if is_causal:
        custom_output = custom_output[:, :-3]  # Compare only unmasked positions
        pytorch_output = pytorch_output[:, :-3]
        
    torch.testing.assert_close(
        custom_output, # Compare only unmasked positions
        pytorch_output,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )
    
    # Backward pass
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        src_custom.grad, 
        src_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )


@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0), # dropout always 0 because PyTorch's SDPA is non-deterministic with dropout
            (256, 8, 2048, 0.0),
            (512, 8, 512, 0.0),
            (64, 4, 256, 0.0),
        ],
    )
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
def test_decoder_equivalence(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_len, norm_first, is_causal
):
    """Test numerical equivalence with PyTorch TransformerDecoderLayer."""

    torch.manual_seed(42)
    pytorch_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
    )
    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
    )

    custom_layer.load_state_dict(pytorch_layer.state_dict())

    # Init input for both pytorch and custom separately to allow gradient comparison
    torch.manual_seed(43)
    tgt_pytorch = torch.randn(batch_size, seq_len, d_model)
    tgt_custom = tgt_pytorch.clone()
    tgt_pytorch.requires_grad = True
    tgt_custom.requires_grad = True

    # Memory tensor for cross-attention with seq_len + 5
    memory = torch.randn(batch_size, seq_len + 5, d_model)

    tgt_key_padding_mask = torch.zeros(batch_size, seq_len).bool()
    tgt_key_padding_mask[:, -3:] = True

    memory_key_padding_mask = torch.zeros(batch_size, seq_len + 5).bool()
    memory_key_padding_mask[:, -3:] = True


    torch.manual_seed(44)
    pytorch_output = pytorch_layer(
        tgt_pytorch,
        memory,
        tgt_mask=(generate_tgt_mask(seq_len) if is_causal else None),
        tgt_is_causal=False,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    torch.manual_seed(44)
    custom_output = custom_layer(
        tgt_custom,
        memory,
        tgt_mask=None,
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    
    if is_causal:
        custom_output = custom_output[:, :-3]  # Compare only unmasked positions
        pytorch_output = pytorch_output[:, :-3]
        
    torch.testing.assert_close(
        custom_output,
        pytorch_output,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )

    # Backward pass
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        tgt_custom.grad,
        tgt_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )


@pytest.mark.parametrize(
    "embed_dim,num_heads,dropout",
    [
        (128, 8, 0.0),
        (256, 8, 0.1),
        (512, 8, 0.2),
        (64, 4, 0.3),
    ],
)
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_selfattention_equivalence(
    embed_dim, num_heads, dropout, batch_size, seq_len, precision, is_causal
):
    """Test MultiheadAttention self-attention equivalence with PyTorch."""

    torch.manual_seed(42)
    pytorch_attn = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    ).to(precision)
    torch.manual_seed(42)
    custom_attn = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
        attention_backend="sdpa",
    ).to(precision)

    custom_attn.load_state_dict(pytorch_attn.state_dict())

    torch.manual_seed(43)
    x_pytorch = torch.randn(batch_size, seq_len, embed_dim).to(precision)
    x_custom = x_pytorch.clone()
    x_pytorch.requires_grad = True
    x_custom.requires_grad = True

    key_padding_mask = torch.zeros(batch_size, seq_len).bool()
    key_padding_mask[:, -3:] = True  # Mask last 3 positions

    if is_causal:
        attn_mask = generate_tgt_mask(seq_len)
    else:
        attn_mask = None

    torch.manual_seed(44)
    pytorch_out, _ = pytorch_attn(
        x_pytorch, x_pytorch, x_pytorch, need_weights=False, key_padding_mask=(None if is_causal else key_padding_mask), attn_mask=attn_mask, is_causal=is_causal,
    )
    torch.manual_seed(44)
    custom_out, _ = custom_attn(
        x_custom, x_custom, x_custom, need_weights=False, key_padding_mask=key_padding_mask, is_causal=is_causal,
    )

    torch.testing.assert_close(
        custom_out,
        pytorch_out,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom attention output does not match PyTorch",
    )

    pytorch_loss = pytorch_out.sum()
    custom_loss = custom_out.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        x_custom.grad,
        x_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )


@pytest.mark.parametrize(
    "embed_dim,num_heads,dropout",
    [
        (128, 8, 0.0),
        (256, 8, 0.0),
        (64, 4, 0.0),
    ],
)
@pytest.mark.parametrize("batch_size,tgt_len,src_len", [(2, 5, 10), (4, 8, 15)])
@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_crossattention_equivalence(
    embed_dim, num_heads, dropout, batch_size, tgt_len, src_len, precision
):
    """Test MultiheadAttention cross-attention equivalence with PyTorch."""

    torch.manual_seed(42)
    pytorch_attn = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    ).to(precision)
    torch.manual_seed(42)
    custom_attn = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
        attention_backend="sdpa",
    ).to(precision)

    custom_attn.load_state_dict(pytorch_attn.state_dict())

    torch.manual_seed(43)
    query_pytorch = torch.randn(batch_size, tgt_len, embed_dim).to(precision)
    query_custom = query_pytorch.clone()
    kv = torch.randn(batch_size, src_len, embed_dim).to(precision)
    query_pytorch.requires_grad = True
    query_custom.requires_grad = True

    key_padding_mask = torch.zeros(batch_size, src_len).bool()
    key_padding_mask[:, -3:] = True  # Mask last 3 positions

    # For cross-attention, causal mask would typically apply if tgt_len == src_len
    # But we'll include it anyway for testing purposes

    torch.manual_seed(44)
    pytorch_out, _ = pytorch_attn(query_pytorch, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
    torch.manual_seed(44)
    custom_out, _ = custom_attn(query_custom, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)

    torch.testing.assert_close(
        custom_out,
        pytorch_out,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom cross-attention output does not match PyTorch",
    )

    pytorch_loss = pytorch_out.sum()
    custom_loss = custom_out.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    torch.testing.assert_close(
        query_custom.grad,
        query_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )
"""Test custom transformer layers for numerical equivalence with PyTorch implementation."""

import pytest
import torch
import torch.nn as nn
from depthcharge.utils import generate_tgt_mask
from depthcharge.transformers.layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
        

@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0),
            (256, 8, 2048, 0.1),
            (512, 8, 512, 0.2),
            (64, 4, 256, 0.3),
        ],
    )
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
def test_encoder_equivalence(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_len, norm_first, is_causal
):
    """Test numerical equivalence with PyTorch TransformerEncoderLayer."""
    
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
    
    if is_causal:
        src_mask = generate_tgt_mask(seq_len)
    else:
        src_mask = None
    
    torch.manual_seed(44)
    pytorch_output = pytorch_layer(src_pytorch, src_mask = src_mask, src_key_padding_mask=key_padding_mask)
    torch.manual_seed(44)
    custom_output = custom_layer(src_custom, src_mask = src_mask, src_key_padding_mask=key_padding_mask)

    torch.testing.assert_close(
        custom_output,
        pytorch_output,
        rtol=1e-5,
        atol=1e-7,
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
        rtol=1e-5,
        atol=1e-7,
        msg="Gradients do not match",
    )


@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0),
            (256, 8, 2048, 0.1),
            (512, 8, 512, 0.2),
            (64, 4, 256, 0.3),
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
    
    if is_causal:
        tgt_mask = generate_tgt_mask(seq_len)
    else:
        tgt_mask = None

    torch.manual_seed(44)
    pytorch_output = pytorch_layer(
        tgt_pytorch,
        memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    torch.manual_seed(44)
    custom_output = custom_layer(
        tgt_custom,
        memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    torch.testing.assert_close(
        custom_output,
        pytorch_output,
        rtol=1e-5,
        atol=1e-7,
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
        rtol=1e-5,
        atol=1e-7,
        msg="Gradients do not match",
    )

"""Test RotaryEmbedding implementation."""

import pytest
import torch
import numpy as np
from depthcharge.encoders.rotary import RotaryEmbedding
from depthcharge.transformers import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    SpectrumTransformerEncoder,
)


def attn(q, k):
    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    return scores


@pytest.mark.parametrize("shift", [0, 5, 10, 100])
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_shift_invariance(shift, batch_size, seq_len):
    """Test that RoPE produces shift-invariant attention patterns.
    """
    head_dim = 16
    rope = RotaryEmbedding(head_dim=head_dim)

    torch.manual_seed(42)
    q = torch.randn(batch_size, 8, seq_len, head_dim)
    torch.manual_seed(43)
    k = torch.randn(batch_size, 8, seq_len, head_dim)

    pos_0 = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    q_rot_0, k_rot_0 = rope(q, k, positions=pos_0)


    q_rot_shifted, k_rot_shifted = rope(q, k, positions=pos_0 + shift)

    torch.testing.assert_close(
        attn(q_rot_0, k_rot_0),
        attn(q_rot_shifted, k_rot_shifted),
        rtol=1e-3,
        atol=1e-5,
        msg=f"Attention patterns differ with shift={shift}, seq_len={seq_len}",
    )


@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
@pytest.mark.parametrize("custom_positions", [True, False])
def test_dense_vs_nested(batch_size, seq_len, custom_positions):
    """Test that dense and nested tensors produce identical outputs with default positions.
    """
    head_dim = 16
    rope = RotaryEmbedding(head_dim=head_dim)

    torch.manual_seed(42)
    q_dense = torch.randn(batch_size, 8, seq_len, head_dim)
    torch.manual_seed(43)
    k_dense = torch.randn(batch_size, 8, seq_len, head_dim)
    if custom_positions:
        torch.manual_seed(44)
        positions = torch.rand(batch_size, seq_len) * 100
        positions_nested = torch.nested.nested_tensor(list(positions), layout=torch.jagged)
    else:
        positions = None
        positions_nested = None
    
    q_rot_dense, k_rot_dense = rope(q_dense, k_dense, positions=positions)

    q_nested = torch.nested.nested_tensor(list(q_dense.transpose(1,2)), layout=torch.jagged).transpose(1,2)
    k_nested = torch.nested.nested_tensor(list(k_dense.transpose(1,2)), layout=torch.jagged).transpose(1,2)

    # Apply RoPE to nested tensors
    q_rot_nested, k_rot_nested = rope(q_nested, k_nested,positions=positions_nested)

    torch.testing.assert_close(
        q_rot_nested.to_padded_tensor(0.0),
        q_rot_dense,
        rtol=1e-3,
        atol=1e-5,
        msg=f"Rope rotates differently for dense vs nested.",
    )
    torch.testing.assert_close(
        k_rot_nested.to_padded_tensor(0.0),
        k_rot_dense,
        rtol=1e-3,
        atol=1e-5,
        msg=f"Rope rotates differently for dense vs nested.",
    )


@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_encoder(batch_size, seq_len):
    """Test rotary within TransformerEncoder.
    """
    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True,
        dropout=0.0,
        norm_first=True,
        attention_backend="sdpa",
        activation="relu",
        rotary_embedding=RotaryEmbedding(head_dim=256//8)
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    )

    torch.manual_seed(43)
    x = torch.randn(batch_size, seq_len, 256)
    
    result_spec_pos = custom_transformer(
        src = x,
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    )
    
    result_infer_pos = custom_transformer(
        src = x,
        mask = None,
    )
    torch.testing.assert_close(
        result_infer_pos,
        result_spec_pos,
        rtol=1e-3,
        atol=1e-5,
    )

@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_decoder(batch_size, seq_len):
    """Test rotary within TransformerDecoder.
    """
    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True,
        dropout=0.0,
        norm_first=True,
        attention_backend="sdpa",
        activation="relu",
        rotary_embedding=RotaryEmbedding(head_dim=256//8)
    )
    custom_transformer = TransformerDecoder(
        custom_layer,
        num_layers=2,
    )

    torch.manual_seed(43)
    x = torch.randn(batch_size, seq_len, 256)
    torch.manual_seed(44)
    mem = torch.randn(batch_size, seq_len+5, 256)
    result_spec_pos = custom_transformer(
        tgt = x,
        memory=mem,
        tgt_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    )
    
    result_infer_pos = custom_transformer(
        tgt = x,
        memory=mem,
    )
    torch.testing.assert_close(
        result_infer_pos,
        result_spec_pos,
        rtol=1e-3,
        atol=1e-5,
    )
    
def test_spectrumencoder_dense():
    torch.manual_seed(42)
    model = SpectrumTransformerEncoder(
        d_model=128,
        nhead=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        rotary_embedding=RotaryEmbedding(head_dim=16),
    )
    torch.manual_seed(43)
    mz_array = torch.rand(8, 15)
    torch.manual_seed(44)
    int_array = torch.rand(8, 15) * 1_000 + 50

    out = model(mz_array, int_array)
    
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Nested tensors with SDPA require CUDA")
def test_spectrumencoder_jagged():
    torch.manual_seed(42)
    model = SpectrumTransformerEncoder(
        d_model=128,
        nhead=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.0,
        rotary_embedding=RotaryEmbedding(head_dim=16),
    )
    torch.manual_seed(43)
    mz_array = torch.rand(8, 15)
    torch.manual_seed(44)
    int_array = torch.rand(8, 15) * 1_000 + 50

    result = model(mz_array, int_array)
    
    mz_array_nest = torch.nested.nested_tensor(list(mz_array), layout=torch.jagged)
    int_array_nest = torch.nested.nested_tensor(list(int_array), layout=torch.jagged)
    result_nest = model(mz_array_nest, int_array_nest)
    
    torch.testing.assert_close(
        result,
        result_nest,
        rtol=1e-3,
        atol=1e-5,
    )
"""Test custom transformer layers for numerical equivalence with PyTorch implementation."""

import pytest
import torch
import torch.nn as nn
from depthcharge.utils import generate_tgt_mask
from depthcharge.transformers import MultiheadAttention
from depthcharge.transformers.layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
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
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_encoder_equivalence_dense(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_len, norm_first, is_causal, backend, activation
):
    """Test numerical equivalence with PyTorch TransformerEncoderLayer using dense tensors.

    Tests with fixed-length sequences (no padding mask).
    """

    torch.manual_seed(42)
    pytorch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerEncoder(
        pytorch_layer,
        num_layers=2,
    )
    
    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    )

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    # Create fixed-length dense tensors
    torch.manual_seed(43)
    src_pytorch = torch.randn(batch_size, seq_len, d_model)
    src_custom = src_pytorch.clone()
    src_pytorch.requires_grad = True
    src_custom.requires_grad = True

    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        src_pytorch,
        is_causal=is_causal,
        mask=(generate_tgt_mask(seq_len) if is_causal else None),
        src_key_padding_mask=None
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        src_custom,
        is_causal=is_causal,
        mask=None,
        src_key_padding_mask=None
    )

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
        src_custom.grad,
        src_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Nested tensors with SDPA require CUDA")
@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0), # dropout always 0 because PyTorch's SDPA is non-deterministic with dropout
            (256, 8, 2048, 0.0),
            (512, 8, 512, 0.0),
            (64, 4, 256, 0.0),
        ],
    )
@pytest.mark.parametrize("batch_size,seq_lens", [(2, (3,5)), (4, (8,20,10,4)), (3, (5,21,3))])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_encoder_equivalence_jagged(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_lens, norm_first, is_causal, backend, activation
):
    """Test numerical equivalence with PyTorch TransformerEncoderLayer using jagged tensors.

    Tests with variable-length sequences using nested tensors (CUDA only).
    """
    device = torch.device("cuda")

    torch.manual_seed(42)
    pytorch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerEncoder(
        pytorch_layer,
        num_layers=2,
    ).to(device)
    
    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    ).to(device)

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    # Create nested tensors with variable lengths
    src = []
    for l in seq_lens:
        torch.manual_seed(43)
        src.append(torch.randn(l, d_model, device=device))
    src = torch.nested.nested_tensor(src, layout=torch.jagged)

    src_padded = src.to_padded_tensor(0.0)
    src.requires_grad = True
    src_padded.requires_grad = True

    key_padding_mask = src_padded.sum(dim=2) == 0.0

    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        src_padded,
        is_causal=is_causal,
        mask=(generate_tgt_mask(max(seq_lens)).to(device) if is_causal else None),
        src_key_padding_mask=key_padding_mask
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        src,
        is_causal=is_causal,
        src_mask=None,
        src_key_padding_mask=None
    )

    torch.testing.assert_close(
        custom_output.to_padded_tensor(0.0),
        torch.nested.nested_tensor(
            [p[:l] for p, l  in zip(pytorch_output, seq_lens)],
            layout=torch.jagged,
        ).to_padded_tensor(0.0),
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )

    # Backward pass
    pytorch_output[key_padding_mask] = 0.0
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        src.grad.to_padded_tensor(0.0), 
        torch.nested.nested_tensor(
            [p[:l] for p, l  in zip(src_padded.grad, seq_lens)],
            layout=torch.jagged
        ).to_padded_tensor(0.0),
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
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_decoder_equivalence_dense(
    d_model, nhead, dim_feedforward, dropout, batch_size, seq_len, norm_first, is_causal, backend, activation
):
    """Test numerical equivalence with PyTorch TransformerDecoderLayer using dense tensors.

    Tests with fixed-length sequences (no padding mask).
    """

    torch.manual_seed(42)
    pytorch_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerDecoder(
        pytorch_layer,
        num_layers=2,
    )
        
    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerDecoder(
        custom_layer,
        num_layers=2,
    )

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    # Init input for both pytorch and custom separately to allow gradient comparison
    torch.manual_seed(43)
    tgt_pytorch = torch.randn(batch_size, seq_len, d_model)
    tgt_custom = tgt_pytorch.clone()
    tgt_pytorch.requires_grad = True
    tgt_custom.requires_grad = True

    # Memory tensor for cross-attention with seq_len + 5
    memory = torch.randn(batch_size, seq_len + 5, d_model)


    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        tgt_pytorch,
        memory,
        tgt_mask=(generate_tgt_mask(seq_len) if is_causal else None),
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        tgt_custom,
        memory,
        tgt_mask=None,
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Nested tensors with SDPA require CUDA")
@pytest.mark.parametrize(
        "d_model,nhead,dim_feedforward,dropout",
        [
            (128, 8, 1024, 0.0), # dropout always 0 because PyTorch's SDPA is non-deterministic with dropout
            (256, 8, 2048, 0.0),
            (512, 8, 512, 0.0),
            (64, 4, 256, 0.0),
        ],
    )
@pytest.mark.parametrize("batch_size,tgt_lens,mem_lens", [(2, (3,5), (8,10)), (4, (8,20,10,4), (15,25,18,10)), (3, (5,21,3), (12,28,8))])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_decoder_equivalence_jagged(
    d_model, nhead, dim_feedforward, dropout, batch_size, tgt_lens, mem_lens, norm_first, is_causal, backend, activation
):
    """Test numerical equivalence with PyTorch TransformerDecoderLayer using jagged tensors.

    Tests with variable-length sequences using nested tensors (CUDA only).
    """
    device = torch.device("cuda")

    torch.manual_seed(42)
    pytorch_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerDecoder(
        pytorch_layer,
        num_layers=2,
    ).to(device)
    
    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    ).to(device)

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    # Create nested tensors with variable lengths for target
    tgt = []
    for l in tgt_lens:
        torch.manual_seed(43)
        tgt.append(torch.randn(l, d_model, device=device))
    tgt = torch.nested.nested_tensor(tgt, layout=torch.jagged)

    tgt_padded = tgt.to_padded_tensor(0.0)
    tgt.requires_grad = True
    tgt_padded.requires_grad = True

    # Create nested tensors with variable lengths for memory
    memory = []
    for l in mem_lens:
        torch.manual_seed(43)
        memory.append(torch.randn(l, d_model, device=device))
    memory = torch.nested.nested_tensor(memory, layout=torch.jagged)

    memory_padded = memory.to_padded_tensor(0.0)
    tgt_key_padding_mask = tgt_padded.sum(dim=2) == 0.0
    memory_key_padding_mask = memory_padded.sum(dim=2) == 0.0

    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        tgt_padded,
        memory_padded,
        tgt_mask=(generate_tgt_mask(max(tgt_lens)).to(device) if is_causal else None),
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        tgt,
        memory,
        tgt_mask=None,
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )

    torch.testing.assert_close(
        custom_output.to_padded_tensor(0.0),
        torch.nested.nested_tensor(
            [p[:l] for p, l  in zip(pytorch_output, tgt_lens)],
            layout=torch.jagged,
        ).to_padded_tensor(0.0),
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )

    # Backward pass
    pytorch_output[tgt_key_padding_mask] = 0.0
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        tgt.grad.to_padded_tensor(0.0),
        torch.nested.nested_tensor(
            [p[:l] for p, l  in zip(tgt_padded.grad, tgt_lens)],
            layout=torch.jagged
        ).to_padded_tensor(0.0),
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )
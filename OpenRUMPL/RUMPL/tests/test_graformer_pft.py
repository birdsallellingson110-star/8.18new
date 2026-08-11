import pathlib
import sys

import torch


LIB = pathlib.Path(__file__).resolve().parents[1] / 'lib'
sys.path.insert(0, str(LIB))

from models.graformer_pft import (  # noqa: E402
    GraFormerPFTEncoder,
    build_h36m17_normalized_adjacency,
)


def test_adjacency_is_h36m17_row_normalized_with_self_loops():
    adjacency = build_h36m17_normalized_adjacency()
    assert adjacency.shape == (17, 17)
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(17))
    assert (adjacency.diagonal() > 0).all()
    # Pelvis connects to both hips and the spine; right knee only to hip/ankle.
    assert adjacency[0, 1] > 0 and adjacency[0, 4] > 0 and adjacency[0, 7] > 0
    assert adjacency[2, 0] == 0


def test_attention_and_full_preserve_shape_and_backpropagate():
    generator = torch.Generator().manual_seed(29)
    inputs = torch.randn(2, 17, 32, generator=generator)
    for mode in ('attention', 'full'):
        model = GraFormerPFTEncoder(
            dim=32, depth=2, num_heads=4, dropout=0.0, mode=mode
        )
        train_inputs = inputs.clone().requires_grad_(True)
        output = model(train_inputs)
        assert output.shape == inputs.shape
        output.square().mean().backward()
        assert train_inputs.grad is not None
        assert torch.isfinite(train_inputs.grad).all()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_paper_ablation_shares_common_initialization():
    torch.manual_seed(41)
    attention = GraFormerPFTEncoder(
        dim=32, depth=2, num_heads=4, mode='attention'
    )
    torch.manual_seed(41)
    full = GraFormerPFTEncoder(
        dim=32, depth=2, num_heads=4, mode='full'
    )
    attention_state = attention.state_dict()
    full_state = full.state_dict()
    common_names = sorted(set(attention_state) & set(full_state))
    assert common_names
    for name in common_names:
        torch.testing.assert_close(
            attention_state[name], full_state[name], rtol=0, atol=0
        )


def test_full_mode_exercises_repeated_chebyshev_blocks():
    torch.manual_seed(53)
    attention = GraFormerPFTEncoder(
        dim=32, depth=2, num_heads=4, dropout=0.0, mode='attention'
    ).eval()
    torch.manual_seed(53)
    full = GraFormerPFTEncoder(
        dim=32, depth=2, num_heads=4, dropout=0.0, mode='full'
    ).eval()
    inputs = torch.randn(2, 17, 32)
    with torch.no_grad():
        attention_output = attention(inputs)
        full_output = full(inputs)
    assert not torch.allclose(attention_output, full_output)


if __name__ == '__main__':
    test_adjacency_is_h36m17_row_normalized_with_self_loops()
    test_attention_and_full_preserve_shape_and_backpropagate()
    test_paper_ablation_shares_common_initialization()
    test_full_mode_exercises_repeated_chebyshev_blocks()
    print('GraFormer PFT tests passed')

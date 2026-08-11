import pathlib
import sys

import torch


LIB = pathlib.Path(__file__).resolve().parents[1] / 'lib'
sys.path.insert(0, str(LIB))

from models.semantic_graph_encoder import (  # noqa: E402
    SemanticGraphPreVFTEncoder,
    build_exact_hop_adjacency,
)


def test_h36m_hops_are_exact_and_symmetric():
    hops = build_exact_hop_adjacency()
    assert hops.shape == (4, 17, 17)
    assert torch.equal(hops, hops.transpose(-1, -2))
    assert not hops.diagonal(dim1=-2, dim2=-1).any()
    # Pelvis--right hip is 1 hop; pelvis--right knee is 2 hops.
    assert hops[0, 0, 1] == 1
    assert hops[1, 0, 2] == 1
    assert hops[:, 0, 2].sum() == 1


def _make_inputs(batch=2, views=3, dim=32):
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(batch, 17, views, dim, generator=generator)
    rays = torch.randn(batch, 17, views, 3, generator=generator)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    return tokens, rays


def test_position_and_full_modes_preserve_shape_and_backpropagate():
    tokens, rays = _make_inputs()
    for mode in ('position', 'full'):
        model = SemanticGraphPreVFTEncoder(
            dim=32,
            depth=2,
            num_heads=4,
            drop_path=0.0,
            mode=mode,
        )
        train_tokens = tokens.clone().requires_grad_(True)
        output = model(train_tokens, rays)
        assert output.shape == tokens.shape
        output.square().mean().backward()
        assert train_tokens.grad is not None
        assert torch.isfinite(train_tokens.grad).all()


def test_encoder_is_equivariant_to_view_permutation():
    tokens, rays = _make_inputs(views=4)
    permutation = torch.tensor([2, 0, 3, 1])
    for mode in ('position', 'full'):
        model = SemanticGraphPreVFTEncoder(
            dim=32,
            depth=2,
            num_heads=4,
            drop_path=0.0,
            mode=mode,
        ).eval()
        with torch.no_grad():
            reference = model(tokens, rays)
            permuted = model(
                tokens[:, :, permutation], rays[:, :, permutation]
            )
        torch.testing.assert_close(
            permuted,
            reference[:, :, permutation],
            rtol=1e-5,
            atol=1e-6,
        )


def test_full_mode_uses_geometry_while_position_control_does_not():
    tokens, rays = _make_inputs()
    altered_rays = rays.roll(shifts=1, dims=1)
    position_model = SemanticGraphPreVFTEncoder(
        dim=32, depth=2, num_heads=4, drop_path=0.0, mode='position'
    ).eval()
    full_model = SemanticGraphPreVFTEncoder(
        dim=32, depth=2, num_heads=4, drop_path=0.0, mode='full'
    ).eval()
    with torch.no_grad():
        position_a = position_model(tokens, rays)
        position_b = position_model(tokens, altered_rays)
        full_a = full_model(tokens, rays)
        full_b = full_model(tokens, altered_rays)
    torch.testing.assert_close(position_a, position_b, rtol=0, atol=0)
    assert not torch.allclose(full_a, full_b)


if __name__ == '__main__':
    test_h36m_hops_are_exact_and_symmetric()
    test_position_and_full_modes_preserve_shape_and_backpropagate()
    test_encoder_is_equivariant_to_view_permutation()
    test_full_mode_uses_geometry_while_position_control_does_not()
    print('semantic graph encoder tests passed')

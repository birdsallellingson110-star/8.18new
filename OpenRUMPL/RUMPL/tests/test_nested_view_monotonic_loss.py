import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from core.loss import nested_view_monotonic_loss


def test_zero_penalty_when_added_view_improves():
    target = torch.zeros(2, 3, 3)
    subset = torch.full_like(target, 2.0, requires_grad=True)
    superset = torch.full_like(target, 1.0, requires_grad=True)
    _, mono, _, _, harmful_rate = nested_view_monotonic_loss(subset, superset, target)
    assert mono.item() == 0.0
    assert harmful_rate.item() == 0.0


def test_monotonic_term_does_not_worsen_subset_to_reduce_loss():
    target = torch.zeros(1, 2, 3)
    subset = torch.full_like(target, 1.0, requires_grad=True)
    superset = torch.full_like(target, 2.0, requires_grad=True)
    _, mono, _, _, harmful_rate = nested_view_monotonic_loss(subset, superset, target)
    mono.backward()
    assert harmful_rate.item() == 1.0
    assert subset.grad is None
    assert superset.grad is not None
    assert superset.grad.abs().sum().item() > 0


def test_nested_gt_supervises_both_predictions():
    target = torch.zeros(1, 2, 3)
    subset = torch.full_like(target, 1.0, requires_grad=True)
    superset = torch.full_like(target, 2.0, requires_grad=True)
    gt_loss, _, _, _, _ = nested_view_monotonic_loss(subset, superset, target)
    gt_loss.backward()
    assert subset.grad.abs().sum().item() > 0
    assert superset.grad.abs().sum().item() > 0


if __name__ == '__main__':
    test_zero_penalty_when_added_view_improves()
    test_monotonic_term_does_not_worsen_subset_to_reduce_loss()
    test_nested_gt_supervises_both_predictions()
    print('Nested-view monotonic loss tests passed')

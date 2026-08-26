import argparse
import importlib
import sys
from pathlib import Path

import torch


AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT))
e5 = importlib.import_module("train_e5_ray_conditioned_dual_stream_20260821")


def make_model(mode="ray-cross"):
    e5._ARGS = argparse.Namespace(
        heads=4, observation_mode=mode, cross_gate_init=-2.0
    )
    return e5.RayConditionedDualStream(
        window_length=3, hidden_dim=32, layers=1,
        relative_scale_m=0.1, root_scale_m=0.05,
        root_mode="learned",
    )


def test_zero_initialized_model_is_exact_identity():
    torch.manual_seed(1)
    model = make_model()
    pose = torch.randn(2, 3, 11, 17, 3)
    rays = torch.randn(2, 3, 17, 4, 7)
    rays[..., :3] = torch.nn.functional.normalize(rays[..., :3], dim=-1)
    rays[..., 6] = torch.sigmoid(rays[..., 6])
    tasks = torch.arange(11)[None].expand(2, -1)
    output = model(pose, tasks, rays)
    torch.testing.assert_close(output, pose, rtol=0.0, atol=0.0)


def test_unselected_view_cannot_change_task_observation():
    torch.manual_seed(2)
    model = make_model()
    pose = torch.randn(1, 3, 11, 17, 3)
    rays = torch.randn(1, 3, 17, 4, 7)
    rays[..., :3] = torch.nn.functional.normalize(rays[..., :3], dim=-1)
    rays[..., 6] = torch.sigmoid(rays[..., 6])
    original = model.observation(pose, rays)[:, :, 0]
    changed = rays.clone()
    # TASKS[0] is (0, 1), so cameras 2 and 3 must be completely masked.
    changed[..., 2, :] += 1000.0
    masked = model.observation(pose, changed)[:, :, 0]
    torch.testing.assert_close(masked, original, rtol=0.0, atol=1e-6)

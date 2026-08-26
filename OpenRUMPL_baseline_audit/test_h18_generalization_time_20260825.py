#!/usr/bin/env python3
"""Regression tests for camera/dataset-independent H18 temporal features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_e2_clean_temporal_residual_20260818 import (
    TASK_COUNT,
    TemporalPoseModel,
    load_initial_checkpoint,
    physical_time_offsets,
)


def make_model(*, continuous_time: bool = True) -> TemporalPoseModel:
    return TemporalPoseModel(
        window_length=9,
        hidden_dim=16,
        layers=1,
        residual_scale_m=0.10,
        camera_independent=True,
        continuous_time=continuous_time,
        reference_dt_s=0.1,
        max_time_period_s=2.0,
    )


def test_physical_offsets_do_not_depend_on_source_frame_numbering() -> None:
    rows = np.arange(9, dtype=np.int64)[None]
    h36m_ids = np.arange(9, dtype=np.int64) * 5
    cmu_ids = np.arange(9, dtype=np.int64) * 3
    h36m = physical_time_offsets(h36m_ids, rows, 50.0, torch.device("cpu"))
    cmu = physical_time_offsets(cmu_ids, rows, 30.0, torch.device("cpu"))
    torch.testing.assert_close(h36m, cmu, atol=1e-7, rtol=0.0)


def test_motion_features_have_a_physical_time_scale() -> None:
    model = make_model()
    shape = (1, 9, 1, 17, 3)
    velocity_m_s = torch.randn(1, 1, 1, 17, 3)

    t_h36m = torch.arange(9, dtype=torch.float32)[None] * 0.1
    t_cmu = torch.arange(9, dtype=torch.float32)[None] / 30.0
    linear_h36m = torch.zeros(shape) + t_h36m[:, :, None, None, None] * velocity_m_s
    linear_cmu = torch.zeros(shape) + t_cmu[:, :, None, None, None] * velocity_m_s
    v_h36m, _ = model.motion_features(linear_h36m, t_h36m)
    v_cmu, _ = model.motion_features(linear_cmu, t_cmu)
    torch.testing.assert_close(v_h36m, v_cmu, atol=2e-6, rtol=2e-5)

    acceleration_m_s2 = torch.randn(1, 1, 1, 17, 3)
    quadratic_h36m = (
        0.5 * t_h36m[:, :, None, None, None].square() * acceleration_m_s2
    )
    quadratic_cmu = (
        0.5 * t_cmu[:, :, None, None, None].square() * acceleration_m_s2
    )
    _, a_h36m = model.motion_features(quadratic_h36m, t_h36m)
    _, a_cmu = model.motion_features(quadratic_cmu, t_cmu)
    torch.testing.assert_close(
        a_h36m[:, 2:], a_cmu[:, 2:], atol=2e-6, rtol=2e-5
    )


def test_continuous_time_path_remains_se3_equivariant() -> None:
    torch.manual_seed(25)
    model = make_model().eval()
    torch.nn.init.normal_(model.output[-1].weight, std=0.02)
    torch.nn.init.normal_(model.output[-1].bias, std=0.01)
    pose = torch.randn(2, 9, TASK_COUNT, 17, 3)
    task_ids = torch.arange(TASK_COUNT)[None].expand(2, -1)
    delta_t_s = (
        torch.arange(-4, 5, dtype=torch.float32)[None].expand(2, -1) * 0.1
    )

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    translation = torch.randn(3)
    transformed = torch.einsum("...i,oi->...o", pose, q) + translation

    with torch.inference_mode():
        prediction = model(pose, task_ids, delta_t_s)
        transformed_prediction = model(transformed, task_ids, delta_t_s)
    expected = torch.einsum("...i,oi->...o", prediction, q) + translation
    error_mm = (transformed_prediction - expected).norm(dim=-1).max() * 1000.0
    assert float(error_mm) < 0.01, float(error_mm)


def test_current_legacy_checkpoint_still_loads_strictly() -> None:
    checkpoint = Path(
        "/mnt/data/cjyoutput/camera_generalization_20260824/"
        "hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/"
        "hrnet/canonical_h18/model_batch8_accum8_eval32/model_best.pth.tar"
    )
    if not checkpoint.exists():
        return
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state["args"]
    model = TemporalPoseModel(
        int(args["window_length"]),
        int(args["hidden_dim"]),
        int(args["layers"]),
        float(args["residual_scale_m"]),
        camera_independent=bool(args.get("camera_independent", False)),
    )
    model.load_state_dict(state["state_dict"], strict=True)
    assert model.camera_independent
    assert not model.continuous_time

    converted = TemporalPoseModel(
        int(args["window_length"]),
        int(args["hidden_dim"]),
        int(args["layers"]),
        float(args["residual_scale_m"]),
        camera_independent=bool(args.get("camera_independent", False)),
        continuous_time=True,
        reference_dt_s=0.1,
        max_time_period_s=2.0,
    )
    report = load_initial_checkpoint(
        converted, str(checkpoint), int(args["frame_stride"]), 50.0,
        torch.device("cpu"),
    )
    assert report["legacy_time_conversion_max_abs"] < 1e-6

    torch.manual_seed(26)
    pose = torch.randn(1, 9, TASK_COUNT, 17, 3)
    task_ids = torch.arange(TASK_COUNT)[None]
    delta_t_s = torch.arange(-4, 5, dtype=torch.float32)[None] * 0.1
    model.eval()
    converted.eval()
    with torch.inference_mode():
        legacy_prediction = model(pose, task_ids)
        converted_prediction = converted(pose, task_ids, delta_t_s)
    torch.testing.assert_close(
        converted_prediction, legacy_prediction, atol=2e-6, rtol=2e-6
    )


def main() -> None:
    test_physical_offsets_do_not_depend_on_source_frame_numbering()
    test_motion_features_have_a_physical_time_scale()
    test_continuous_time_path_remains_se3_equivariant()
    test_current_legacy_checkpoint_still_loads_strictly()
    print("H18 generalization/time regression tests passed")


if __name__ == "__main__":
    main()

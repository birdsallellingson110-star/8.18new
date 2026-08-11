#!/usr/bin/env python3
"""Collect reproducible live status for the controlled H36M experiments."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
OUTPUT_JSON = ROOT / "LIVE_EXPERIMENT_STATUS_20260801.json"
OUTPUT_MD = ROOT / "LIVE_EXPERIMENT_STATUS_20260801.md"
HISTORY_JSONL = ROOT / "EXPERIMENT_STATUS_HISTORY_20260801.jsonl"

EXPERIMENTS = [
    {
        "id": "H0",
        "tag": "H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731",
        "control": "original real-H36M RUMPL",
        "isolated_variable": "A1D-refined 2D + confidence-weighted triangulation residual anchor",
        "goal": "controlled strengthened baseline",
        "base": ROOT / "H0_a1d_refined_rumpl_tri_anchor",
    },
    {
        "id": "H32",
        "tag": "H32_a1dRefined2D_confGeom_triAnchor_curriculum_seed0_20260731",
        "control": "H0",
        "isolated_variable": "per-joint VFT confidence+geometry bias",
        "goal": "diagnose bias at the original RUMPL fusion location",
        "base": ROOT / "H32_tri_anchor_gbt_bias",
    },
    {
        "id": "H33",
        "tag": "H33_a1d_nobias_triAnchor_fixedK2First15_thenW6to1to1_seed0_20260731",
        "control": "H0",
        "isolated_variable": "more aggressive V2 curriculum",
        "goal": "test whether optimization allocation alone closes V2 gap",
        "base": ROOT / "H33_ultra_v2_a1d_nobias",
    },
    {
        "id": "H34",
        "tag": "H34_a1d_nobias_triAnchor_bone01_reproj01_ray01_curriculum_seed0_20260731",
        "control": "H0",
        "isolated_variable": "bone+reprojection+ray auxiliary losses, each lambda=0.1",
        "goal": "test geometry supervision without attention bias",
        "base": ROOT / "H34_a1d_nobias_geom_losses",
    },
    {
        "id": "H37-J0",
        "tag": "H37_J0_globalJV1_plain_A1D_triAnchor_retainedRUMPL_seed0_20260801",
        "control": "H0",
        "isolated_variable": "one ReZero-gated global joint-view attention block, no bias",
        "goal": "measure global observation fusion before RUMPL view collapse",
        "base": ROOT / "H37_global_joint_view_A1D_tri_anchor",
    },
    {
        "id": "H37-J1",
        "tag": "H37_J1_globalJV1_confgeom_A1D_triAnchor_retainedRUMPL_seed0_20260801",
        "control": "H37-J0",
        "isolated_variable": "learnable confidence+pairwise-ray-distance attention bias",
        "goal": "measure bias at the paper-aligned global fusion location",
        "base": ROOT / "H37_global_joint_view_A1D_tri_anchor",
    },
    {
        "id": "H39-U0",
        "tag": "H39_U0_undistortPoints_A1D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260801",
        "control": "H0",
        "isolated_variable": "H36M cv2.undistortPoints before pinhole ray construction",
        "goal": "remove camera-model mismatch from the correct baseline",
        "base": ROOT / "H39_h36m_undistort_A1D_tri_anchor",
    },
    {
        "id": "H35",
        "tag": "H35_a1dH21_nobias_triAnchor_curriculum_seed0_20260731",
        "control": "H0",
        "isolated_variable": "A1D followed by H21 pose-query 2D refinement",
        "goal": "test a stronger 2D input while retaining H0 architecture",
        "base": ROOT / "H35_a1d_h21_tri_anchor",
        "queue_pattern": "queue_H35_after_H34_20260731.sh",
        "preparation_pattern": "export_h21_refined_mmpose_pkl.py",
        "preparation_log": Path(
            "/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/"
            "datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_h21_legswap/"
            "export_train.log"
        ),
    },
    {
        "id": "H36-P0",
        "tag": "H36_P0_plucker_A1D_triAnchor_retainedRUMPL_curriculum_seed0_20260801",
        "control": "H0",
        "isolated_variable": "Plucker ray input",
        "goal": "isolate GBT line representation inside retained RUMPL",
        "base": ROOT / "H36_paper_input_retained_rumpl",
        "queue_pattern": "queue_H36_paper_inputs_20260801.sh",
    },
    {
        "id": "H36-P1",
        "tag": "H36_P1_harmonic15_A1D_triAnchor_retainedRUMPL_curriculum_seed0_20260801",
        "control": "H0",
        "isolated_variable": "harmonic-15 ray encoding",
        "goal": "isolate GBT high-frequency input encoding inside retained RUMPL",
        "base": ROOT / "H36_paper_input_retained_rumpl",
        "queue_pattern": "queue_H36_paper_inputs_20260801.sh",
    },
    {
        "id": "H36-P2",
        "tag": "H36_P2_plucker_harmonic15_A1D_triAnchor_retainedRUMPL_curriculum_seed0_20260801",
        "control": "H36-P0 and H36-P1",
        "isolated_variable": "Plucker + harmonic-15 combination",
        "goal": "conditional combination only if both isolated inputs pass",
        "base": ROOT / "H36_paper_input_retained_rumpl",
        "queue_pattern": "queue_H36_paper_inputs_20260801.sh",
    },
    {
        "id": "H38-R0",
        "tag": "H38_R0_relative_A1D_triAnchor_retainedRUMPL_seed0_20260801",
        "control": "H0",
        "isolated_variable": "gated relative cross-view residual",
        "goal": "improve view fusion without replacing RUMPL VFT/PFT",
        "base": ROOT / "H38_relative_centered_retained_rumpl",
        "queue_pattern": "queue_H38_relative_centered_20260801.sh",
    },
    {
        "id": "H38-C0",
        "tag": "H38_C0_center_A1D_triAnchor_retainedRUMPL_seed0_20260801",
        "control": "H0",
        "isolated_variable": "anchor-centered ray coordinates",
        "goal": "separate absolute position from RUMPL residual learning",
        "base": ROOT / "H38_relative_centered_retained_rumpl",
        "queue_pattern": "queue_H38_relative_centered_20260801.sh",
    },
    {
        "id": "H38-RC",
        "tag": "H38_RC_relative_center_A1D_triAnchor_retainedRUMPL_seed0_20260801",
        "control": "H38-R0 and H38-C0",
        "isolated_variable": "relative fusion + anchor-centered rays",
        "goal": "conditional combination only if both isolated modules improve V2/V4",
        "base": ROOT / "H38_relative_centered_retained_rumpl",
        "queue_pattern": "queue_H38_relative_centered_20260801.sh",
    },
    {
        "id": "H41",
        "tag": "H41_mtf_point_H35_A1D_H21_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "zero-gated MTF-point inspired relative-view residual",
        "goal": "test relative source-target view messages without replacing RUMPL VFT/PFT",
        "base": ROOT / "H41_H45_paper_mask_fusion",
    },
    {
        "id": "H42",
        "tag": "H42_mtf_mask04_H35_A1D_H21_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "MTF-style non-diagonal attention-edge mask rate 0.4",
        "goal": "test the public MTF random-mask training mechanism in RUMPL VFT",
        "base": ROOT / "H41_H45_paper_mask_fusion",
    },
    {
        "id": "H43",
        "tag": "H43_gif_mask02_H35_A1D_H21_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "Masked Gifformer fully-random attention-edge mask M=0.2",
        "goal": "paper-rate ablation for variable-view robustness",
        "base": ROOT / "H41_H45_paper_mask_fusion",
    },
    {
        "id": "H44",
        "tag": "H44_gif_mask05_H35_A1D_H21_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "Masked Gifformer fully-random attention-edge mask M=0.5",
        "goal": "test the paper-best robustness mask rate",
        "base": ROOT / "H41_H45_paper_mask_fusion",
    },
    {
        "id": "H45",
        "tag": "H45_gif_mask08_H35_A1D_H21_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "Masked Gifformer fully-random attention-edge mask M=0.8",
        "goal": "measure over-masking degradation",
        "base": ROOT / "H41_H45_paper_mask_fusion",
    },
    {
        "id": "H46",
        "tag": "H46_H35_exact_replay_seed0_20260801",
        "control": "H35",
        "isolated_variable": "none; exact seed-0 replay",
        "goal": "measure baseline reproducibility before attributing module effects",
        "base": ROOT / "H46_H48_root_cause",
    },
    {
        "id": "H47",
        "tag": "H47_H35_PFT_single_pass_seed0_20260801",
        "control": "H46 exact replay",
        "isolated_variable": "execute the final public RUMPL PFT block once instead of twice",
        "goal": "isolate the released PFT repeat-last control-flow quirk",
        "base": ROOT / "H46_H48_root_cause",
    },
    {
        "id": "H48",
        "tag": "H48_A1DmatchedH21_RUMPL_triAnchor_seed0_20260801",
        "control": "H35",
        "isolated_variable": "retrain H21 on A1D seed coordinates before exporting A1D-to-H21 inputs",
        "goal": "remove the raw-HRNet-train/A1D-inference distribution mismatch",
        "base": ROOT / "H46_H48_root_cause",
        "queue_pattern": "launch_H48_a1d_matched_h21_chain_20260801.sh",
        "preparation_pattern": "H48_H21_a1d_matched_v2focus_reg005",
        "preparation_log": ROOT / "H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/train.log",
    },
    {
        "id": "H49",
        "tag": "H49_H35_oldH21_workers12_seed0_20260802",
        "control": "H35",
        "isolated_variable": "none; strict workers=12 replay with old H21 input",
        "goal": "remove the workers=6 mismatch discovered in H46",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H50",
        "tag": "H50_A1DmatchedH21_workers12_seed0_20260802",
        "control": "H49",
        "isolated_variable": "A1D-matched H21 input; all RUMPL settings including workers=12 fixed",
        "goal": "strict controlled confirmation of the H48 input-distribution gain",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H51",
        "tag": "H51_H35_oldH21_workers12_seed1_20260802",
        "control": "H49 old-H21 baseline family",
        "isolated_variable": "paired replication seed=1",
        "goal": "estimate old-H21 baseline variance for the H50 contribution",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H52",
        "tag": "H52_A1DmatchedH21_workers12_seed1_20260802",
        "control": "H51",
        "isolated_variable": "A1D-matched H21 input at paired seed=1",
        "goal": "confirm the H50 gain across seeds",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H53",
        "tag": "H53_H35_oldH21_workers12_seed2_20260802",
        "control": "H49 old-H21 baseline family",
        "isolated_variable": "paired replication seed=2",
        "goal": "estimate old-H21 baseline variance for the H50 contribution",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H54",
        "tag": "H54_A1DmatchedH21_workers12_seed2_20260802",
        "control": "H53",
        "isolated_variable": "A1D-matched H21 input at paired seed=2",
        "goal": "confirm the H50 gain across seeds",
        "base": ROOT / "H49_H50_worker12_control",
    },
    {
        "id": "H59",
        "tag": "H59_H58balancedH21_RUMPL_workers12_seed0_20260802",
        "control": "H50",
        "isolated_variable": "H21 view sampling 3:1:1 to 1:1:1",
        "goal": "transfer the H58 2D/anchor screening gain into full RUMPL",
        "base": ROOT / "H59_h58_balanced_full",
        "preparation_pattern": "mmpose_hrnet_coco_a1d_h21_h58_balanced_views_legswap/h36m_train.pkl",
        "preparation_log": Path(
            "/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/"
            "datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_h21_h58_balanced_views_legswap/"
            "export_train.log"
        ),
    },
    {
        "id": "H63",
        "tag": "H63_H62balanced10kH21_RUMPL_workers12_seed0_20260802",
        "control": "H59",
        "isolated_variable": "continue balanced-view H21 from 5000 to 10000 steps",
        "goal": "transfer the H62 V3/V4 anchor gain into full RUMPL",
        "base": ROOT / "H63_h62_balanced10k_full",
        "preparation_pattern": "mmpose_hrnet_coco_a1d_h21_h62_balanced10k_legswap/h36m_train.pkl",
        "preparation_log": Path(
            "/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/"
            "datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_h21_h62_balanced10k_legswap/"
            "export_train.log"
        ),
    },
    {
        "id": "H64",
        "tag": "H64_H50_A1DmatchedH21_workers12_seed0_20260802_finalState_20260803",
        "control": "H50 model_best",
        "isolated_variable": "evaluate final_state instead of model_best",
        "goal": "audit whether two-view checkpoint selection hides a better all-view model",
        "base": ROOT / "H64_H66_final_state_audit",
    },
    {
        "id": "H65",
        "tag": "H65_H54_A1DmatchedH21_workers12_seed2_20260802_finalState_20260803",
        "control": "H54 model_best",
        "isolated_variable": "evaluate final_state instead of model_best",
        "goal": "repeat checkpoint-selection audit on the strongest seed",
        "base": ROOT / "H64_H66_final_state_audit",
    },
    {
        "id": "H66",
        "tag": "H66_H59_H58balancedH21_RUMPL_workers12_seed0_20260802_finalState_20260803",
        "control": "H59 model_best",
        "isolated_variable": "evaluate final_state instead of model_best",
        "goal": "repeat checkpoint-selection audit on balanced-view training",
        "base": ROOT / "H64_H66_final_state_audit",
    },
    {
        "id": "H67",
        "tag": "H67_H50_plucker_workers12_seed0_20260803",
        "control": "H50",
        "isolated_variable": "Plucker line representation",
        "goal": "test whether geometry-complete ray tokens improve retained RUMPL fusion",
        "base": ROOT / "H67_H69_h50_representation",
    },
    {
        "id": "H68",
        "tag": "H68_H50_harmonic15_workers12_seed0_20260803",
        "control": "H50",
        "isolated_variable": "harmonic ray encoding L=15",
        "goal": "test higher-frequency spatial encoding without replacing RUMPL",
        "base": ROOT / "H67_H69_h50_representation",
    },
    {
        "id": "H69",
        "tag": "H69_H50_anchorCenteredRays_workers12_seed0_20260803",
        "control": "H50",
        "isolated_variable": "triangulation-anchor-centered ray coordinates",
        "goal": "reduce absolute-coordinate burden while retaining RUMPL fusion",
        "base": ROOT / "H67_H69_h50_representation",
    },
    {
        "id": "H70",
        "tag": "H70_H50_geometryUncertaintyToken_workers12_seed0_20260803",
        "control": "H50",
        "isolated_variable": "zero-init analytic ray-normal-matrix uncertainty embedding on [FUS]",
        "goal": "let retained RUMPL identify near-parallel camera pairs and rely more on pose prior",
        "base": ROOT / "H70_h50_geometry_uncertainty",
    },
    {
        "id": "H71",
        "tag": "H71_H63input_anchorCenteredRays_workers12_seed0_20260803",
        "control": "H63 and H69",
        "isolated_variable": "combine H63 balanced10k input with H69 anchor-centered rays",
        "goal": "test additive gains from two independently successful non-conflicting modules",
        "base": ROOT / "H71_H75_confirm_fuse",
    },
    {
        "id": "H72",
        "tag": "H72_H63balanced10kInput_workers12_seed1_20260803",
        "control": "H63 seed0 family",
        "isolated_variable": "RUMPL seed=1",
        "goal": "confirm H63 input-training gain across seeds",
        "base": ROOT / "H71_H75_confirm_fuse",
    },
    {
        "id": "H73",
        "tag": "H73_H63balanced10kInput_workers12_seed2_20260803",
        "control": "H63 seed0 family",
        "isolated_variable": "RUMPL seed=2",
        "goal": "confirm H63 input-training gain across seeds",
        "base": ROOT / "H71_H75_confirm_fuse",
    },
    {
        "id": "H74",
        "tag": "H74_H69anchorCenteredRays_workers12_seed1_20260803",
        "control": "H69 seed0 family",
        "isolated_variable": "RUMPL seed=1",
        "goal": "confirm anchor-centered ray gain across seeds",
        "base": ROOT / "H71_H75_confirm_fuse",
    },
    {
        "id": "H75",
        "tag": "H75_H69anchorCenteredRays_workers12_seed2_20260803",
        "control": "H69 seed0 family",
        "isolated_variable": "RUMPL seed=2",
        "goal": "confirm anchor-centered ray gain across seeds",
        "base": ROOT / "H71_H75_confirm_fuse",
    },
    {
        "id": "H76",
        "tag": "H76_H50_anchorCenteredPlucker_workers12_seed0_20260803",
        "control": "H67 and H69",
        "isolated_variable": "express Plucker moments in triangulation-anchor-centered coordinates",
        "goal": "combine H67 weak-pair gains with H69 multi-view gains while retaining RUMPL",
        "base": ROOT / "H76_h50_centered_plucker",
    },
    {
        "id": "H77",
        "tag": "H77_H76anchorCenteredPlucker_workers12_seed1_20260803",
        "control": "H76 seed0 family",
        "isolated_variable": "RUMPL seed=1",
        "goal": "confirm centered-Plucker gain across seeds",
        "base": ROOT / "H77_H79_h76_confirm",
    },
    {
        "id": "H78",
        "tag": "H78_H76anchorCenteredPlucker_workers12_seed2_20260803",
        "control": "H76 seed0 family",
        "isolated_variable": "RUMPL seed=2",
        "goal": "confirm centered-Plucker gain across seeds",
        "base": ROOT / "H77_H79_h76_confirm",
    },
    {
        "id": "H79",
        "tag": "H79_H76anchorCenteredPlucker_exactReplay_workers12_seed0_20260803",
        "control": "H76",
        "isolated_variable": "none; exact seed-0 replay",
        "goal": "verify deterministic reproducibility of the large H76 gain",
        "base": ROOT / "H77_H79_h76_confirm",
    },
    {
        "id": "H80",
        "tag": "H80_H76_jointCenteredPlucker_workers12_seed0_20260803",
        "control": "H76",
        "isolated_variable": "center each Plucker ray on its matching per-joint triangulation anchor",
        "goal": "align ray-token coordinates with the per-joint residual target and reduce remaining limb-direction error",
        "base": ROOT / "H80_joint_centered_plucker",
    },
    {
        "id": "H81",
        "tag": "H81_H76_perJointResidualGate_workers12_seed0_20260803",
        "control": "H76",
        "isolated_variable": "17 learnable per-joint PFT residual scales initialized to one",
        "goal": "allow strong head corrections while suppressing harmful root/hip/left-knee residuals",
        "base": ROOT / "H81_H83_targeted_pft",
    },
    {
        "id": "H82",
        "tag": "H82_H76_postPFTGraphResidual_workers12_seed0_20260803",
        "control": "H76",
        "isolated_variable": "zero-initialized skeleton graph residual after global PFT",
        "goal": "add local kinematic message only after retained RUMPL global joint fusion",
        "base": ROOT / "H81_H83_targeted_pft",
    },
    {
        "id": "H83",
        "tag": "H83_H76_jointSpecificHead_workers12_seed0_20260803",
        "control": "H76",
        "isolated_variable": "zero-initialized joint-specific correction beside the shared 3D head",
        "goal": "resolve measured conflicts where one shared residual head helps head joints but hurts selected lower joints",
        "base": ROOT / "H81_H83_targeted_pft",
    },
]

HASH_FILES = [
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/dataset/joints_dataset_rumpl.py"),
    ROOT / "H0_a1d_refined_rumpl_tri_anchor.yaml",
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H37_global_joint_view_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H39_h36m_undistort_baseline_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H34_a1d_nobias_geom_losses_20260731.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H35_a1d_h21_tri_anchor_20260731.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H36_paper_input_rumpl_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H38_relative_centered_rumpl_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/queue_H35_after_H34_20260731.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/queue_H36_paper_inputs_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/queue_H37_global_joint_view_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/queue_H38_relative_centered_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H41_H45_paper_mask_fusion_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H46_H47_root_cause_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H48_a1d_matched_h21_chain_20260801.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H49_H50_worker12_control_20260802.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H55_H58_h21_screen_20260802.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H63_h62_balanced10k_full_20260802.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H64_H66_final_state_audit_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H67_H69_h50_representation_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H70_h50_geometry_uncertainty_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H71_H75_confirm_fuse_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H76_h50_centered_plucker_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H77_H79_h76_confirm_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H80_joint_centered_plucker_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H81_H83_targeted_pft_20260803.sh"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_anchor_centered_ray_points.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_targeted_pft_residuals.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_geometry_uncertainty_token.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/diagnose_h36m_rumpl_error_structure_20260803.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/compare_h36m_2d_inputs.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/screen_h36m_2d_and_anchor.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit/diagnose_h35_anchor_confidence_20260801.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_view_attention_mask.py"),
    Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_pft_repeat_last.py"),
]

EPOCH_PATTERN = re.compile(r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]")
ENV_PREFIXES = (
    "RUMPL_",
    "GBT_",
    "TRAIN_",
    "CAA_",
    "DEPRO_",
    "REPROJ_",
    "RAY_",
    "BONE_",
    "VFT_",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_snapshot(tag: str) -> list[dict]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw_cmd = (proc / "cmdline").read_bytes()
            command = raw_cmd.replace(b"\0", b" ").decode(errors="replace").strip()
            if "run/train_rumpl.py" not in command or tag not in command:
                continue
            environ = {}
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                key_text = key.decode(errors="replace")
                if key_text.startswith(ENV_PREFIXES) or key_text == "CUDA_VISIBLE_DEVICES":
                    environ[key_text] = value.decode(errors="replace")
            matches.append(
                {
                    "pid": int(proc.name),
                    "command": command,
                    "environment": dict(sorted(environ.items())),
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    matches.sort(key=lambda item: item["pid"])
    return matches


def any_process_matches(pattern: str | None) -> bool:
    if not pattern:
        return False
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
            if pattern in command:
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def find_log(base: Path, tag: str) -> Path | None:
    candidates = sorted((base / "logs").glob(f"{tag}*train.log"))
    return candidates[-1] if candidates else None


def last_epoch(log_path: Path | None) -> dict | None:
    if log_path is None or not log_path.is_file():
        return None
    last_match = None
    with log_path.open("r", errors="replace") as stream:
        for line in stream:
            match = EPOCH_PATTERN.search(line)
            if match:
                last_match = match
    if last_match is None:
        return None
    return {
        "epoch_zero_based": int(last_match.group(1)),
        "iteration": int(last_match.group(2)),
        "iterations_per_epoch": int(last_match.group(3)),
    }


def last_preparation_progress(log_path: Path | None) -> dict | None:
    if log_path is None or not log_path.is_file():
        return None
    last_payload = None
    with log_path.open("r", errors="replace") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "groups_done" in payload and "groups_total" in payload:
                last_payload = payload
    if last_payload is None:
        return None
    done = int(last_payload["groups_done"])
    total = int(last_payload["groups_total"])
    return {
        "groups_done": done,
        "groups_total": total,
        "percent": 100.0 * done / total,
        "elapsed_seconds": float(last_payload.get("elapsed_seconds", 0.0)),
        "mean_joint_delta_px": last_payload.get("mean_joint_delta_px"),
    }


def eval_metrics(base: Path, tag: str) -> tuple[dict, dict]:
    metrics = {}
    records = {}
    for views in (2, 3, 4):
        table_path = base / "eval" / tag / f"V{views}" / "table2.json"
        if not table_path.is_file():
            continue
        payload = json.loads(table_path.read_text())
        metrics[f"V{views}"] = payload["table2_action_equal"]["all17_mm"]
        records[f"V{views}"] = payload["records"]
    return metrics, records


def status_for(experiment: dict) -> dict:
    processes = process_snapshot(experiment["tag"])
    log_path = find_log(experiment["base"], experiment["tag"])
    metrics, records = eval_metrics(experiment["base"], experiment["tag"])
    done_files = sorted((experiment["base"] / "completed").glob(f"{experiment['tag']}*.done"))
    queued = any_process_matches(experiment.get("queue_pattern"))
    preparing = any_process_matches(experiment.get("preparation_pattern"))
    preparation_log = experiment.get("preparation_log")
    preparation_progress = last_preparation_progress(preparation_log)
    if len(metrics) == 3 and done_files:
        status = "completed"
    elif processes:
        status = "running"
    elif preparing:
        status = "preparing_data"
    elif queued:
        status = "queued"
    elif log_path:
        status = "stopped_or_waiting"
    else:
        status = "planned"
    runtime_environment = processes[0]["environment"] if processes else {}
    return {
        **{key: str(value) if isinstance(value, Path) else value for key, value in experiment.items()},
        "status": status,
        "log": str(log_path) if log_path else None,
        "progress": last_epoch(log_path),
        "preparation_progress": preparation_progress,
        "metrics_action_equal_all17_mm": metrics,
        "evaluation_records": records,
        "main_process": processes[0] if processes else None,
        "runtime_environment": runtime_environment,
        "process_count_including_dataloader_workers": len(processes),
        "done_file": str(done_files[-1]) if done_files else None,
    }


def metric_text(metrics: dict) -> str:
    if not metrics:
        return "—"
    return "/".join(
        f"{metrics.get(f'V{views}', float('nan')):.3f}" for views in (2, 3, 4)
    )


def progress_text(progress: dict | None) -> str:
    if progress is None:
        return "—"
    return (
        f"epoch {progress['epoch_zero_based'] + 1}/20, "
        f"iter {progress['iteration']}/{progress['iterations_per_epoch']}"
    )


def item_progress_text(item: dict) -> str:
    if item["progress"] is not None:
        return progress_text(item["progress"])
    preparation = item.get("preparation_progress")
    if preparation is not None and item["status"] == "preparing_data":
        return (
            f"data {preparation['groups_done']}/{preparation['groups_total']} "
            f"({preparation['percent']:.1f}%)"
        )
    return "—"


def render_markdown(payload: dict) -> str:
    lines = [
        "# H36M 实验实时台账",
        "",
        f"更新时间：{payload['captured_at']}",
        "",
        "固定评价口径：正式 S9/S11、action-equal All-17 MPJPE；结果顺序为 V2/V3/V4。",
        "",
        "| ID | 对照 | 唯一变量 | 目标 | 状态 | 训练进度 | V2/V3/V4 (mm) |",
        "|---|---|---|---|---|---|---:|",
    ]
    for item in payload["experiments"]:
        lines.append(
            f"| {item['id']} | {item['control']} | "
            f"{item['isolated_variable']} | {item['goal']} | "
            f"{item['status']} | {item_progress_text(item)} | "
            f"{metric_text(item['metrics_action_equal_all17_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## 固定目标与门控",
            "",
            "- 总目标：在严格同协议下持续降低 V2/V3/V4 All-17 MPJPE，越低越好；不再把 40/30 mm 当作停止条件。",
            "- 第一里程碑：V2 < 40 mm、V4 < 30 mm，且 V3 不明显退化。",
            "- 论文目标：建立输入2D、额外数据、相机标定、关节集合和评价聚合均一致的对照表，逐篇超过可比论文；不同协议数字不得直接混比。",
            "- 统计要求：正式贡献报告至少三 seed 均值与标准差，并优先采用三个视角均改善的方案。",
            "- H39-U0 对 H0：只判断去畸变的基线净作用，不作为Transformer贡献。",
            "- H37-J0 对 H0：判断全局 joint-view 融合净作用。",
            "- H37-J1 对 H37-J0：判断 confidence/geometry bias 净作用。",
            "- 值得继续的最低标准：V2或V4改善至少1.0 mm，另外两个视角退化不超过0.5 mm。",
            "- 达标后才拆 confidence-only/geometry-only，并补 seed 1/2；不达标不跑融合组合。",
            "",
            "## 运行环境快照",
            "",
            "完整命令、PID和环境变量保存在同目录的 `LIVE_EXPERIMENT_STATUS_20260801.json`。",
            "",
            "## 文件 SHA-256",
            "",
        ]
    )
    for path, digest in payload["sha256"].items():
        lines.append(f"- `{digest}`  `{path}`")
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> None:
    payload = {
        "captured_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "metric_protocol": "H36M S9/S11 action-equal All-17 MPJPE",
        "target_mm": {"V2": 40.0, "V4": 30.0},
        "expected_records": {"V2": 12126, "V3": 8084, "V4": 2021},
        "experiments": [status_for(experiment) for experiment in EXPERIMENTS],
        "sha256": {str(path): sha256(path) for path in HASH_FILES},
    }
    atomic_write(OUTPUT_JSON, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    atomic_write(OUTPUT_MD, render_markdown(payload))
    history_item = {
        "captured_at": payload["captured_at"],
        "sha256": payload["sha256"],
        "experiments": [
            {
                "id": item["id"],
                "tag": item["tag"],
                "status": item["status"],
                "progress": item["progress"],
                "preparation_progress": item["preparation_progress"],
                "metrics_action_equal_all17_mm": item[
                    "metrics_action_equal_all17_mm"
                ],
                "main_pid": item["main_process"]["pid"]
                if item["main_process"]
                else None,
                "runtime_environment": item["runtime_environment"],
            }
            for item in payload["experiments"]
        ],
    }
    with HISTORY_JSONL.open("a") as stream:
        stream.write(json.dumps(history_item, ensure_ascii=False) + "\n")
    print(OUTPUT_JSON)
    print(OUTPUT_MD)
    print(HISTORY_JSONL)


if __name__ == "__main__":
    main()

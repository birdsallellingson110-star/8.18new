# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------
#
# Portions of this file are:
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import get_model_name
from core.loss import nested_view_monotonic_loss
from core.evaluate import accuracy, calc_distance_per_dim, calc_mpjpe, print_per_kp
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.utils_plot import plot_3d_points, plot_2d_points, plot_3d_points_plotly
import random 
import pickle
import json
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torchinfo import summary

logger = logging.getLogger(__name__)


def get_keep_ratio(final_ratio, ite, total):
    # Start from 1.0, gradually decrease to the given ratio
    ratio = 1.0 - (1 - final_ratio) * ite / total
    return ratio


def set_rumpl_train_mode(model):
    """Keep a frozen RUMPL backbone deterministic during adapter fine-tuning.

    ``model.train()`` recursively enables stochastic-depth/dropout in the whole
    H76 backbone, even when ``requires_grad`` is false for those parameters.
    That makes a targeted adapter fit a different function from the one used at
    evaluation.  For scoped experiments, start from eval mode and re-enable
    training only on modules that own trainable parameters.  The ``all`` path
    remains exactly the historical behavior.
    """
    scope = os.environ.get('RUMPL_TRAIN_SCOPE', 'all').strip() or 'all'
    if scope == 'all':
        model.train()
        return

    model.eval()
    core = getattr(model, 'module', model)

    # Set ``training`` flags bottom-up instead of calling ``train()`` on a
    # parent: the latter would recursively re-enable frozen DropPath/Dropout
    # children.  A container is marked train only when its subtree contains a
    # trainable parameter, while every frozen stochastic leaf stays eval.
    def _configure(module):
        child_trainable = False
        for child in module.children():
            # Do not use ``any(generator)`` here: its short-circuit would leave
            # siblings after the first trainable child unconfigured.
            child_trainable = _configure(child) or child_trainable
        own_trainable = any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
        module.training = own_trainable or child_trainable
        return module.training

    _configure(core)
    enabled = [
        name or '<root>'
        for name, module in core.named_modules()
        if module.training and any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
    ]
    if not enabled:
        raise RuntimeError(
            f'RUMPL_TRAIN_SCOPE={scope} has no trainable module to put in train mode'
        )
    if not getattr(core, '_rumpl_targeted_mode_logged', False):
        print(
            f'[RUMPL_TARGETED_MODE] scope={scope} backbone=eval '
            f'train_modules={enabled}',
            flush=True,
        )
        core._rumpl_targeted_mode_logged = True


def confidence_weighted_reprojection_loss(pred_3d, packed_2d, view_indices=None, beta=0.01):
    """Robust reprojection loss for packed [xy,K5,Rt12,confidence] observations."""
    if packed_2d.shape[-1] < 20:
        raise ValueError('Reprojection loss requires packed joints_2ds with 20 channels')
    if view_indices is not None:
        packed_2d = packed_2d[:, :, view_indices, :]
    camera = packed_2d[..., 2:19]
    intrinsics = camera[..., :5]
    rt = camera[..., 5:].reshape(*camera.shape[:-1], 3, 4)
    points_h = torch.cat([pred_3d, torch.ones_like(pred_3d[..., :1])], dim=-1)
    points_cam = torch.einsum('bjvrc,bjc->bjvr', rt, points_h)
    depth = points_cam[..., 2]
    depth_safe = torch.where(depth.abs() > 1e-6, depth, torch.ones_like(depth))
    xz = points_cam[..., 0] / depth_safe
    yz = points_cam[..., 1] / depth_safe
    fx, fy, skew, cx, cy = intrinsics.unbind(dim=-1)
    projected = torch.stack([fx * xz + skew * yz + cx, fy * yz + cy], dim=-1)
    observed = packed_2d[..., :2]
    confidence = packed_2d[..., 19].clamp(0, 1)
    valid = (
        (depth > 1e-6)
        & torch.isfinite(projected).all(dim=-1)
        & torch.isfinite(observed).all(dim=-1)
    )
    weights = confidence * valid.to(dtype=confidence.dtype)
    error = F.smooth_l1_loss(projected, observed, reduction='none', beta=beta).sum(dim=-1)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def confidence_weighted_ray_loss(pred_3d, rays, view_indices=None, beta=0.01):
    """Robust point-to-ray distance in the model's world-coordinate frame."""
    if rays.shape[-1] < 7:
        raise ValueError('Ray loss requires packed [direction, point, confidence] inputs')
    if view_indices is not None:
        rays = rays[:, :, view_indices, :]
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-8)
    point = rays[..., 3:6]
    offset = pred_3d[:, :, None, :] - point
    perpendicular = offset - (offset * direction).sum(dim=-1, keepdim=True) * direction
    distance = torch.linalg.vector_norm(perpendicular, dim=-1)
    confidence = rays[..., 6].clamp(0, 1)
    valid = torch.isfinite(distance) & torch.isfinite(confidence)
    weights = confidence * valid.to(dtype=confidence.dtype)
    error = F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction='none', beta=beta)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)




NUM_VIEW = {'multiview_h36m': 4, 'multiview_skipose': 6}

# pick the neighboring camera, same as Epipolar Transformers
cam_rank = {
    'multiview_h36m':
        {
            0: 2,  # cam1 -> cam3
            1: 3,  # cam2 -> cam4
            2: 0,  # cam3 -> cam1
            3: 1   # cam4 -> cam2
        },
    'multiview_skipose':
        {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4
        }
}


cam_pair = {
    'multiview_h36m': [[0, 2], [1, 3]],
    'multiview_skipose':[[0, 1], [2, 3], [4, 5]],
}


def get_epipolar_field(points1, center1, points2, center2, power=10, eps=1e-10):
    # Points1 / Points2: (B, N, 3)
    # Center1 / Center2: (B, 1, 3)
    # power: a higher value will generate a sharpen map along the epipolar line
    # Return: ()
    num_p1 = points1.shape[1]  # N1 = H * W

    # norm vector of  space C1C2P1 (Eq 3 in paper)
    vec_c1_c2 = center2 - center1 + eps         # (B, 1, 3)
    vec_c1_p1 = points1 - center1               # (B, N1, 3)
    space_norm_vec = torch.cross(vec_c1_p1, vec_c1_c2.repeat(1, num_p1, 1), dim=2) # (B, N, 3) x (B, N, 3) -> (B, N, 3)
    space_norm_vec_norm = F.normalize(space_norm_vec, dim=2, p=2)  # (B, N1, 3)

    vec_c2_p2 = points2 - center2  # (B, N2, 3)
    vec_c2_p2_norm = F.normalize(vec_c2_p2, dim=2, p=2)  # (B, N2, 3)

    # Eq 4 in paper
    cos = torch.bmm(space_norm_vec_norm, vec_c2_p2_norm.transpose(2, 1))    # (B, N1, 3) * (B, 3, N2) -> (B, N1, N2)

    field = 1 - cos.abs()
    field = field ** power
    field[field < 1e-5] = 1e-5      # avoid 0
    return field


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict):

    # total Epoch 
    total_epoch = config.TRAIN.END_EPOCH
    total_iter = len(data) * total_epoch
    cur_iter = epoch * len(data)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_x = AverageMeter()
    loss_y = AverageMeter()
    loss_z = AverageMeter()
    avg_acc = AverageMeter()

    # Keep frozen-backbone targeted runs deterministic; preserve the historical
    # full-network behavior for RUMPL_TRAIN_SCOPE=all.
    set_rumpl_train_mode(model)

    end = time.time()
    for i, (middle_points, closest_points_all, target, rays, meta, joints_2ds) in enumerate(data):
        # one subject, one action, in different views
        # middle_points: (B, 17, 1, 3)
        # closest_points_all: (B, 17, n_views, 3)
        # target: (B, 17, 3)
        data_time.update(time.time() - end)

        
        # ============= sample two views =============
        # centers = [meta[idx]['cam_center'].float() for idx in range(len(input))]
        # rays = [meta[idx]['rays'].float() for idx in range(len(input))]
        
        # direction_vectors = [meta[idx]['direction_vectors'].float() for idx in range(len(input))]
        # intersection_points = [meta[idx]['intersection_points'].float() for idx in range(len(input))]
        # direction_angles = [meta[idx]['direction_angles'].float() for idx in range(len(input))]
        # persons_height = [meta[idx]['persons_height'].float() for idx in range(len(input))]

        # ================== model forward ==================
        ratio = get_keep_ratio(0.7, cur_iter, total=total_iter)
        ratio = 0.7 
        # if config.NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD:
        #     output, x_intermediate = model(input, centers=centers, rays=rays)
        # elif config.MODEL == 'multiview_transformer_enc_dec':
        #     target_pose = target[0].cuda(non_blocking=True)
        #     output = model(input, centers=centers, rays=rays, direction_vectors=direction_vectors, intersection_points=intersection_points,
        #                    direction_angles=direction_angles, target_pose=target_pose,
        #                    )  # list, (B, num_joints, H=64, W=64)
        # else:
        if config.NETWORK.FEED_CAMERA_CALIBRATION or config.NETWORK.FEED_ONLY_2D:
            output = model(joints_2ds, is_training=False)
        elif config.NETWORK.APPLY_VIEW_FUSION:
            output = model(rays, is_training=True, epoch=epoch)
        elif config.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS:
            output = model(middle_points)
        else:
            output = model(closest_points_all)
        if i % 1000 == 0 and os.environ.get('GBT_LEARNABLE_BIAS', '0') == '1':
            backbone = getattr(model, 'module', model)
            backbone = getattr(backbone, 'features', backbone)
            conf_scales = []
            geom_scales = []
            for block in backbone.blocks_view_fusion:
                if hasattr(block.attn, 'gbt_conf_scale'):
                    conf_scales.append(block.attn.gbt_conf_scale.detach().square().item())
                if hasattr(block.attn, 'gbt_geom_scale'):
                    geom_scales.append(block.attn.gbt_geom_scale.detach().square().item())
            print(
                '[GBT_LEARNED] '
                f'conf={[round(value, 5) for value in conf_scales]} '
                f'geom={[round(value, 5) for value in geom_scales]}',
                flush=True,
            )
        
        

        # ================== Loss on the final heatmap (64 * 64) ==================
        # # if loss is weighted
        # if config.LOSS.WEIGHT_ON_VISIBLIITY:
        #     joints_2d_vis = [meta[idx]['joints_vis'].float() for idx in range(len(input))]
        #     joints_2d_vis = torch.stack(joints_2d_vis, dim=1)  # (B, num_views, num_joints)
        #     w = joints_2d_vis.sum(dim=1)  # (B, num_joints)
        #     w = torch.where(w > 1, 1.0, 0.0)
        # else: # if loss is not weighted
        #     w = weight[0]
            
            
        # t = target[0]
        # w = torch.ones_like(target)
        target = target.cuda(non_blocking=True)
        # w = w.cuda(non_blocking=True)
        # direction_vectors = [d.cuda(non_blocking=True) for d in direction_vectors]
        # if config.LOSS.TYPE == 'PlaneProjectionLoss':
        #     loss, loss_axis = criterion(output, t, w, rays_direction=direction_vectors)
        # elif config.NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD:
        #     loss, loss_axis = criterion(x_intermediate + [output], t, w)
        # else:
        loss, loss_axis = criterion(output, target)
        _mono_w = float(os.environ.get('MONO_W', 0.0))
        _mono_gt_w = float(os.environ.get('MONO_GT_W', 1.0))
        if _mono_w > 0 and rays.shape[2] >= 3:
            _mono_vt = min(5, rays.shape[2])
            _mono_k = int(torch.randint(2, _mono_vt, (1,), device=rays.device).item())
            _mono_order = torch.randperm(_mono_vt, device=rays.device)
            _mono_subset_idx = _mono_order[:_mono_k].tolist()
            _mono_superset_idx = _mono_order[:_mono_k + 1].tolist()
            _mono_subset = model(
                rays[:, :, _mono_subset_idx, :], is_training=False, apply_vft_mask=False
            )
            _mono_superset = model(
                rays[:, :, _mono_superset_idx, :], is_training=False, apply_vft_mask=False
            )
            _mono_gt, _mono, _mono_es, _mono_et, _mono_har = nested_view_monotonic_loss(
                _mono_subset,
                _mono_superset,
                target,
                margin=float(os.environ.get('MONO_MARGIN', 0.0)),
            )
            loss = loss + _mono_gt_w * _mono_gt + _mono_w * _mono
            if i % 1000 == 0:
                print(
                    f'[MONO] iter {i}: k={_mono_k}->{_mono_k + 1} '
                    f'views={_mono_subset_idx}->{_mono_superset_idx} '
                    f'e_s={_mono_es.item():.4f} e_t={_mono_et.item():.4f} '
                    f'har={_mono_har.item():.4f} gt={_mono_gt.item():.4f} '
                    f'mono={_mono.item():.4f} gt_w={_mono_gt_w} mono_w={_mono_w}',
                    flush=True,
                )
        _kd_loss = None  # RCG: 蒸馏项单独存(与任务loss分开求梯度)
        _reproj_main_w = float(os.environ.get('REPROJ_LAMBDA', 0.0))
        _reproj_student_w = float(os.environ.get('STUDENT_REPROJ_W', 0.0))
        _reproj_beta = float(os.environ.get('REPROJ_BETA', 0.01))
        _ray_main_w = float(os.environ.get('RAY_LAMBDA', 0.0))
        _ray_student_w = float(os.environ.get('STUDENT_RAY_W', 0.0))
        _ray_beta = float(os.environ.get('RAY_BETA', 0.01))
        _packed_2d = None
        _packed_rays = None
        _rp_main = None
        _ray_main = None
        if _reproj_main_w > 0 or _reproj_student_w > 0:
            _packed_2d = joints_2ds.cuda(non_blocking=True)
        if _reproj_main_w > 0:
            _rp_main = confidence_weighted_reprojection_loss(
                output, _packed_2d, beta=_reproj_beta
            )
            loss = loss + _reproj_main_w * _rp_main
        if _ray_main_w > 0 or _ray_student_w > 0:
            _packed_rays = rays.cuda(non_blocking=True)
        if _ray_main_w > 0:
            _ray_main = confidence_weighted_ray_loss(output, _packed_rays, beta=_ray_beta)
            loss = loss + _ray_main_w * _ray_main
        # ===== 层次1: 骨长一致性约束(env BONE_LAMBDA>0 开启, =0 时 baseline 完全不变) =====
        _bone_lam = float(os.environ.get('BONE_LAMBDA', 0.0))
        if _bone_lam > 0:
            _BONES = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(5,11),(6,12)]
            _SYM = [(0,2),(1,3),(4,6),(5,7),(10,11)]  # 左右对称骨对(BONES内index)
            _blp = torch.stack([(output[:,a]-output[:,b]).norm(dim=-1) for a,b in _BONES], 1)
            _blg = torch.stack([(target[:,a]-target[:,b]).norm(dim=-1) for a,b in _BONES], 1)
            _l_len = torch.nn.functional.l1_loss(_blp, _blg)
            _l_sym = sum(torch.nn.functional.l1_loss(_blp[:,p], _blp[:,q]) for p,q in _SYM) / len(_SYM)
            _mode = os.environ.get('BONE_MODE', 'full')  # full=match-GT+0.3对称(AMASS特定); sym=纯对称(普适,可迁移)
            if _mode == 'sym':
                _bone_term = _bone_lam * _l_sym
            else:
                _bone_term = _bone_lam * (_l_len + 0.3 * _l_sym)
            if i % 1000 == 0:
                print(f"[BONE] mode={_mode} iter {i}: main_loss={loss.item():.4f} bone_term={_bone_term.item():.4f} (len={_l_len.item():.4f} sym={_l_sym.item():.4f}) lam={_bone_lam}", flush=True)
            loss = loss + _bone_term
        # ===== 创新: V=5→V=2 自蒸馏(env DISTILL_LAMBDA>0 开启) =====
        # 同场景: teacher=5视角(no grad, 准), student=2视角 → 强迫V=2融合/先验逼近V=5
        # 消融用两个独立权重: _dw=蒸馏项权重, _gw=student-GT项权重。
        #   C(both,=之前)=DISTILL_W 1 + STUDENT_GT_W 1; A(纯GT)=DISTILL_W 0 + STUDENT_GT_W 1; B(纯蒸馏)=DISTILL_W 1 + STUDENT_GT_W 0
        _dw = float(os.environ.get('DISTILL_W', os.environ.get('DISTILL_LAMBDA', 0.0)))
        _fw = float(os.environ.get('FEAT_DISTILL_W', 0.0))   # 特征级蒸馏权重(蒸head前特征, 不与输出-GT直接竞争→冲突少)
        _gw = float(os.environ.get('STUDENT_GT_W', 1.0)) if (_dw > 0 or _fw > 0 or os.environ.get('STUDENT_GT_W')) else 0.0
        if (_dw > 0 or _fw > 0 or _gw > 0) and rays.shape[2] >= 3:
            _vt = min(5, rays.shape[2])
            # STUDENT_VIEWS: 'rand'=随机k∈[2,_vt-1](通用版,去2视角特殊性); 或固定整数(如'2')
            _sv = os.environ.get('STUDENT_VIEWS', '2')
            _ks = int(torch.randint(2, _vt, (1,)).item()) if _sv == 'rand' else int(_sv)
            _bb = getattr(model, 'module', model)   # DataParallel包装时取.module
            _bb = getattr(_bb, 'features', _bb)      # MultiView_RUMPL_G.features = MultiView_RUMPL(含head)
            _teacher_eval = os.environ.get('DISTILL_TEACHER_EVAL', '0') == '1'

            # Snapshot once per batch rather than once per teacher/candidate
            # forward.  Six candidate calls otherwise repeat the mode walk and
            # can dominate the already expensive hard-mining loop.
            _teacher_modes = None
            if _teacher_eval:
                _teacher_modes = [(module, module.training) for module in model.modules()]
                model.eval()

            def _distill_no_grad_forward(_input, **_kwargs):
                """Run a teacher/candidate forward without student gradients."""
                with torch.no_grad():
                    return model(_input, **_kwargs)

            def _restore_student_mode():
                nonlocal _teacher_modes
                if _teacher_modes is not None:
                    for module, was_training in _teacher_modes:
                        module.training = was_training
                    _teacher_modes = None

            if _fw > 0 and not hasattr(_bb.head, '_feat_hooked'):  # 注册hook抓head输入特征(仅一次)
                def _grab(mod, inp, out):
                    mod._feat_in = inp[0]
                _bb.head.register_forward_hook(_grab); _bb.head._feat_hooked = True
            _teach = _distill_no_grad_forward(rays[:, :, :_vt, :], is_training=False)
            _feat_t = _bb.head._feat_in.detach().clone() if _fw > 0 else None
            _hard_pair = None
            _hard_views = None
            _student_idx = None
            _aux_w = float(os.environ.get('AUX_MULTIK_W', 0.0))
            _aux_d = None
            _aux_g = None
            _aux_ks = None
            _aux_views = None
            if os.environ.get('HARD_VIEW_MINING', '0') == '1' and _fw == 0:
                _pairs = [(a, b) for a in range(_vt) for b in range(a + 1, _vt)]
                _cand_n = min(int(os.environ.get('HARD_VIEW_CAND', len(_pairs))), len(_pairs))
                _order = torch.randperm(len(_pairs), device=rays.device)[:_cand_n].tolist()
                _best_score, _best_pair = None, None
                # Candidate selection is used only to choose the hard pair;
                # none of its forward graphs contributes gradients.  Keeping
                # this block explicitly no-grad preserves the scores/RNG
                # stream while avoiding six unnecessary autograd graphs per
                # training batch.
                with torch.no_grad():
                    for _pi in _order:
                        _pair = _pairs[_pi]
                        _cand = _distill_no_grad_forward(
                            rays[:, :, list(_pair), :], is_training=False
                        )
                        _score = (((_cand - _teach.detach()) ** 2).sum(-1) + 1e-8).sqrt().mean()
                        if _best_score is None or _score > _best_score:
                            _best_score, _best_pair = _score, _pair
                _hard_pair = _best_pair
                if _ks == 2:
                    _idx = list(_best_pair)
                else:
                    _remaining = [v for v in range(_vt) if v not in _best_pair]
                    _extra_n = min(_ks - 2, len(_remaining))
                    _extra_order = torch.randperm(len(_remaining), device=rays.device)[:_extra_n].tolist()
                    _idx = list(_best_pair) + [_remaining[j] for j in _extra_order]
                _hard_views = tuple(_idx)
                _student_idx = _idx
                _restore_student_mode()
                _stud = model(rays[:, :, _idx, :], is_training=False, apply_vft_mask=True)
                if _aux_w > 0 and _vt >= 4:
                    _aux_min = max(3, int(os.environ.get('AUX_MULTIK_MIN', 3)))
                    _aux_max = min(_vt - 1, int(os.environ.get('AUX_MULTIK_MAX', 4)))
                    if _aux_min <= _aux_max:
                        _aux_ks = int(torch.randint(_aux_min, _aux_max + 1, (1,)).item())
                        _aux_remaining = [v for v in range(_vt) if v not in _best_pair]
                        _aux_extra_n = min(_aux_ks - 2, len(_aux_remaining))
                        _aux_order = torch.randperm(
                            len(_aux_remaining), device=rays.device
                        )[:_aux_extra_n].tolist()
                        _aux_idx = list(_best_pair) + [_aux_remaining[j] for j in _aux_order]
                        _aux_views = tuple(_aux_idx)
                        _aux_stud = model(
                            rays[:, :, _aux_idx, :], is_training=False, apply_vft_mask=True
                        )
            else:
                _student_idx = list(range(_ks))
                _restore_student_mode()
                _stud = model(rays[:, :, :_ks, :], is_training=False, apply_vft_mask=True)
            _feat_s = _bb.head._feat_in if _fw > 0 else None
            _dist_pj = (((_stud - _teach.detach()) ** 2).sum(-1) + 1e-8).sqrt()          # (B,17) 逐关节
            _legw = float(os.environ.get('LEG_DISTILL_W', 1.0))                          # 腿蒸馏降权(治下肢被先验带偏)
            if _legw != 1.0:
                _jw = torch.ones(17, device=_dist_pj.device); _jw[[11, 12, 13, 14, 15, 16]] = _legw
                _d = (_dist_pj * _jw).mean()
            else:
                _d = _dist_pj.mean()                                                     # 输出级蒸馏
            _g = (((_stud - target) ** 2).sum(-1) + 1e-8).sqrt().mean()                 # student对GT监督
            if _aux_views is not None:
                _aux_dist_pj = (((_aux_stud - _teach.detach()) ** 2).sum(-1) + 1e-8).sqrt()
                _aux_d = (_aux_dist_pj * _jw).mean() if _legw != 1.0 else _aux_dist_pj.mean()
                _aux_g = (((_aux_stud - target) ** 2).sum(-1) + 1e-8).sqrt().mean()
            _dfeat = ((_feat_s - _feat_t) ** 2).mean() if _fw > 0 else 0.0              # 特征级蒸馏
            _rp_stud = None
            _ray_stud = None
            if _reproj_student_w > 0:
                _rp_stud = confidence_weighted_reprojection_loss(
                    _stud, _packed_2d, view_indices=_student_idx, beta=_reproj_beta
                )
            if _ray_student_w > 0:
                _ray_stud = confidence_weighted_ray_loss(
                    _stud, _packed_rays, view_indices=_student_idx, beta=_ray_beta
                )
            if i % 1000 == 0:
                _fv = _dfeat.item() if _fw > 0 else 0.0
                print(f"[DISTILL] iter {i}: main={loss.item():.4f} d(s-t)={_d.item():.4f} g={_g.item():.4f} feat={_fv:.4f} dw={_dw} fw={_fw} gw={_gw} legw={_legw} teacher_eval={int(_teacher_eval)} k={_ks} vt={_vt} hard_pair={_hard_pair} hard_views={_hard_views} rcg={os.environ.get('RCG','0')}", flush=True)
                _rpm = _rp_main.item() if _rp_main is not None else 0.0
                _rps = _rp_stud.item() if _rp_stud is not None else 0.0
                print(f"[REPROJ] iter {i}: main={_rpm:.6f} student={_rps:.6f} main_w={_reproj_main_w} student_w={_reproj_student_w} beta={_reproj_beta}", flush=True)
                _raym = _ray_main.item() if _ray_main is not None else 0.0
                _rays = _ray_stud.item() if _ray_stud is not None else 0.0
                print(f"[RAY_LOSS] iter {i}: main={_raym:.6f} student={_rays:.6f} main_w={_ray_main_w} student_w={_ray_student_w} beta={_ray_beta}", flush=True)
                _aux_dv = _aux_d.item() if _aux_d is not None else 0.0
                _aux_gv = _aux_g.item() if _aux_g is not None else 0.0
                print(f"[AUX_MULTIK] iter {i}: k={_aux_ks} views={_aux_views} d={_aux_dv:.4f} g={_aux_gv:.4f} w={_aux_w}", flush=True)
            loss = loss + _gw * _g                      # GT监督算作"姿态任务"
            if _rp_stud is not None:
                loss = loss + _reproj_student_w * _rp_stud
            if _ray_stud is not None:
                loss = loss + _ray_student_w * _ray_stud
            if _aux_d is not None:
                loss = loss + _aux_w * (_dw * _aux_d + _gw * _aux_g)
            _kd_total = _dw * _d + _fw * _dfeat
            if os.environ.get('RCG', '0') == '1':
                _kd_loss = _kd_total                    # 蒸馏项(输出+特征)单独, backward时PCGrad调和
            else:
                loss = loss + _kd_total
        # target = t
        
        # ================== log input and output in wandb ==================
        # if config.WANDB and config.WANDB_LOG_IMG:
        #     if i % config.LOG_IMAGE_FREQ == 0:
        #         # 3d target of the last sample in the batch
        #         # fig_target = plt.figure()
        #         # plot_3d_points(fig_target, t[-1].cpu().numpy())
                
        #         fig_target = plot_3d_points_plotly(t[-1].cpu().numpy())
                
        #         # fig_pred = plt.figure()
        #         # plot_3d_points(fig_pred, output[-1].detach().cpu().numpy())
        #         fig_pred = plot_3d_points_plotly(output[-1].detach().cpu().numpy())
        #         joints_2d_org = [meta[idx]['joints_2d_org'].cpu().numpy() for idx in range(len(input))]
                
        #         figs_2d = []
        #         for view in range(len(target)):
        #             fig = plt.figure()
        #             plot_2d_points(fig, input[view][-1, :, :2].cpu().numpy(), 
        #                            input[view][-1, :, 2].cpu().numpy(), 
        #                            to_plot_org=joints_2d_org[view][-1])
        #             figs_2d.append(fig)
        #         wandb.log({"train/3d_target": fig_target,
        #                    "train/3d_pred": fig_pred,
        #                    "epoch": epoch,
        #                    })
        #         for i, fig in enumerate(figs_2d):
        #             wandb.log({"train/2d_input_view{}".format(i): fig,
        #                        "epoch": epoch,})
        #         cols = ["3d_target", "3d_pred"]
        #         for view in range(len(target)):
        #             cols.append("2d_input_view{}".format(view))
        #         table = wandb.Table(columns=cols)
        #         path_to_plotly_html_fig_target = "./plotly_figure_fig_target.html"
        #         fig_target.write_html(path_to_plotly_html_fig_target, auto_play=False)
        #         path_to_plotly_html_fig_pred = "./plotly_figure_fig_pred.html"
        #         fig_pred.write_html(path_to_plotly_html_fig_pred, auto_play=False)
        #         row = [wandb.Html(path_to_plotly_html_fig_target), wandb.Html(path_to_plotly_html_fig_pred)]
        #         for fig in figs_2d:
        #             row.append(wandb.Image(fig))
        #         table.add_data(*row)
        #         wandb.log({"train/3d_2d_table": table, "epoch": epoch})
        #         print('wandb training image logged!')
                
                
        
        # _, mpjep__ = evaluate(output.detach().cpu().numpy(), target.detach().cpu().numpy(), data.dataset.actual_joints, config)
        # print('mpjep__:', mpjep__)


        if _kd_loss is not None:
            # ===== RCG: 冲突梯度调和 (ssrn-4789240, PCGrad) =====
            # g_pe=任务(loss)梯度, g_kd=蒸馏梯度; 若冲突(点积<0)则各自投影到对方法平面, 再相加
            _params = [p for p in model.parameters() if p.requires_grad]
            optim.zero_grad(); loss.backward(retain_graph=True)
            g_pe = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in _params]
            optim.zero_grad(); _kd_loss.backward()
            g_kd = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in _params]
            dot = sum((a * b).sum() for a, b in zip(g_kd, g_pe))
            if dot < 0:
                pe_sq = sum((b * b).sum() for b in g_pe) + 1e-12
                kd_sq = sum((a * a).sum() for a in g_kd) + 1e-12
                g_final = [(gk - (dot / pe_sq) * gp) + (gp - (dot / kd_sq) * gk) for gk, gp in zip(g_kd, g_pe)]
                if i % 1000 == 0:
                    print(f"[RCG] iter {i}: 梯度冲突(dot={dot.item():.4f}), 已调和", flush=True)
            else:
                g_final = [gk + gp for gk, gp in zip(g_kd, g_pe)]
            optim.zero_grad()
            for p, g in zip(_params, g_final):
                p.grad = g
            optim.step()
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()
        losses.update(loss.item(), len(middle_points) * middle_points[0].size(0))
        loss_x.update(loss_axis[0].item(), len(middle_points) * middle_points[0].size(0))
        loss_y.update(loss_axis[1].item(), len(middle_points) * middle_points[0].size(0))
        loss_z.update(loss_axis[2].item(), len(middle_points) * middle_points[0].size(0))

        # # ================== accuracy based on heatmap (64 * 64) ==================
        # nviews = len(output)
        # acc = [None] * nviews
        # cnt = [None] * nviews
        # pre = [None] * nviews
        # for j in range(nviews):
        #     _, acc[j], cnt[j], pre[j] = accuracy(
        #         output[j].detach().cpu().numpy(),
        #         target[j].detach().cpu().numpy(), target_coords=config.TARGET_COORDS)
        # acc = np.mean(acc)
        # cnt = np.mean(cnt)
        # avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss_x {loss_x.val:.5f} ({loss_x.avg:.5f})\t' \
                  'Loss_y {loss_y.val:.5f} ({loss_y.avg:.5f})\t' \
                  'Loss_z {loss_z.val:.5f} ({loss_z.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\tRatio {ratio:.3f}' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(data), ratio, batch_time=batch_time,
                      speed=len(middle_points) * middle_points[0].size(0) / batch_time.val,
                      data_time=data_time, loss=losses, loss_x=loss_x, loss_y=loss_y, loss_z=loss_z,
                      acc=avg_acc, memory=gpu_memory_usage, ratio=ratio)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            
            if config.WANDB:
                wandb.log({
                    'train/train_loss_iter': losses.val,
                    'train/train_loss_avg': losses.avg,
                    'train/train_loss_x_iter': loss_x.val,
                    'train/train_loss_x_avg': loss_x.avg,
                    'train/train_loss_y_iter': loss_y.val,
                    'train/train_loss_y_avg': loss_y.avg,
                    'train/train_loss_z_iter': loss_z.val,
                    'train/train_loss_z_avg': loss_z.avg,
                    # 'train/train_accuracy_iter': avg_acc.val,
                    # 'train/train_accuracy_avg': avg_acc.avg,
                    
                })

            for k in range(len(middle_points)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'train'), view_name, i)
                # save_debug_images(config, input[k], meta[k], target[k],
                #                   pre[k] * 4, output[k], prefix)

    # epoch loss summary
    msg = 'Summary Epoch: [{0}]\tLoss ({loss.avg:.5f})\tLoss_x ({loss_x.avg:.5f})\tLoss_y ({loss_y.avg:.5f})\tLoss_z ({loss_z.avg:.5f})\tAccuracy {acc.avg:.3f}'.format(epoch,
                                                                                                                                                                        loss=losses,
                                                                                                                                                                        loss_x=loss_x,
                                                                                                                                                                        loss_y=loss_y,
                                                                                                                                                                        loss_z=loss_z,
                                                                                                                                                                        acc=avg_acc)
    logger.info(msg)
    
    if config.WANDB:
        wandb.log({
            'train/train_loss_epoch': losses.avg,
            'train/train_loss_x_epoch': loss_x.avg,
            'train/train_loss_y_epoch': loss_y.avg,
            'train/train_loss_z_epoch': loss_z.avg,
            # 'train/train_accuracy_epoch': avg_acc.avg,
            'epoch': epoch,
        })


def validate(config,
             loader,
             dataset,
             model,
             criterion,
             output_dir,
             writer_dict=None,
             epoch=None,
             is_mmpose=False,
             print_macs_summary=False):

    model.eval()
    batch_time = AverageMeter()
    inference_time = AverageMeter()
    losses = AverageMeter()
    loss_x = AverageMeter()
    loss_y = AverageMeter()
    loss_z = AverageMeter()
    avg_acc = AverageMeter()

    n_view = 6 if config.DATASET.TEST_DATASET == 'multiview_skipose' else 4
    n_view = 5 if config.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else n_view
    n_view = len(config.DATASET.TEST_VIEWS) if config.DATASET.TEST_VIEWS is not None else n_view
    nsamples = len(dataset)

    njoints = config.NETWORK.NUM_JOINTS                 # 17
    height = int(config.NETWORK.HEATMAP_SIZE[0])        # 64
    width = int(config.NETWORK.HEATMAP_SIZE[1])         # 64
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_gts = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_confs = np.zeros((nsamples, njoints), dtype=np.float32)      # (#sample, 17)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)           # (#sample, 17,64, 64)
    
    all_3d_confs = np.ones((nsamples, njoints), dtype=np.float32)      # (#sample, 17) 3D confidence of CMU annotations
    fnames = []
    valid_camera_ids = []
    
    validation_name = 'val' if not is_mmpose else 'val_mmpose'

    idx = 0
    with torch.no_grad():
        end = time.time()
        
        for i, (middle_points, closest_points_all, target, rays, meta, joints_2ds) in enumerate(loader):
            # one subject, one action, in different views
            # middle_points: (B, 17, 1, 3)
            # closest_points_all: (B, 17, n_views, 3)
            # target: (B, 17, 3)
            # ======================== combinations of input ========================
            batch_size = middle_points.shape[0]
            if 'image' in meta:
                fnames += meta['image']
                if 'fname_camera_ids' in meta:
                    valid_camera_ids += meta['fname_camera_ids']
                    
            # if config.DATASET.TEST_MMPOSE_CONFS_TH is not None:
            confs = rays[:, :, :, -1].clone().detach().cpu().numpy()
            # print('confs:', confs.shape)
            all_confs[idx:idx + batch_size] = confs.min(axis=-1)
                

            # rays = [meta[j]['rays'].float()  for j in range(len(input))  ] 
            # centers = [meta[j]['cam_center'].float() for j in range(len(input))]
            # direction_vectors = [meta[idx]['direction_vectors'].float() for idx in range(len(input))]
            # intersection_points = [meta[idx]['intersection_points'].float() for idx in range(len(input))]
            # direction_angles = [meta[idx]['direction_angles'].float() for idx in range(len(input))]
            # persons_height = [meta[idx]['persons_height'].float() for idx in range(len(input))]
        
            if print_macs_summary:
                summary(model, input_data=[rays])
                raise
            start_inference = time.time()
            # if config.NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD:
            #     output, x_intermediate = model(input, centers=centers, rays=rays)
            # else:
            #     output = model(input, centers=centers, rays=rays, direction_vectors=direction_vectors, intersection_points=intersection_points, direction_angles=direction_angles,
            #                    persons_height=persons_height,)  
            if config.NETWORK.FEED_CAMERA_CALIBRATION or config.NETWORK.FEED_ONLY_2D:
                output = model(joints_2ds, is_training=False)
            elif config.NETWORK.APPLY_VIEW_FUSION:
                output = model(rays, is_training=False)
            elif config.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS:
                output = model(middle_points)
            else:
                output = model(closest_points_all)
            inference_time.update(time.time() - start_inference)
            """
                debug code for plotting pred and target in 3D
                
                import matplotlib.pyplot as plt
                from mpl_toolkits import mplot3d
                body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
                fig = plt.figure() 
                ax = fig.add_subplot(111, projection='3d')
                to_plot = target[0][0].cpu().numpy()
                ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
                for edge in body_edges:
                    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')

                to_plot = output[0].cpu().numpy()
                ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='r')
                for edge in body_edges:
                    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='orange')

                max_range = np.array([to_plot[:, 0].max() - to_plot[:, 0].min(), to_plot[:, 1].max() - to_plot[:, 1].min(), to_plot[:, 2].max() - to_plot[:, 2].min()]).max() / 2.0
                x_mean = to_plot[:, 0].mean()
                y_mean = to_plot[:, 1].mean()
                z_mean = to_plot[:, 2].mean()
                ax.set_xlim(x_mean - max_range, x_mean + max_range)
                ax.set_ylim(y_mean - max_range, y_mean + max_range)
                ax.set_zlim(z_mean - max_range, z_mean + max_range)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(0, 90)
                plt.savefig('test25.png')
            """

            
            # ======================== Loss calculation ========================
            # if loss is weighted
            # if config.LOSS.WEIGHT_ON_VISIBLIITY:
            #     joints_2d_vis = [meta[idx]['joints_vis'].float() for idx in range(len(input))]
            #     joints_2d_vis = torch.stack(joints_2d_vis, dim=1)  # (B, num_views, num_joints)
            #     w = joints_2d_vis.sum(dim=1)  # (B, num_joints)
            #     w = torch.where(w > 1, 1.0, 0.0)
            # else: # if loss is not weighted
            #     w = weight[0]
                
            # t = target[0]
            # w = weight[0]
            # t = t.cuda(non_blocking=True)
            # w = w.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            loss, loss_axis = criterion(output, target)
            
            # direction_vectors = [d.cuda(non_blocking=True) for d in direction_vectors]
            # if config.LOSS.TYPE == 'PlaneProjectionLoss':
            #     loss, loss_axis = criterion(output, t, w, rays_direction=direction_vectors)
            # elif config.NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD:
            #     loss, loss_axis = criterion(x_intermediate + [output], t, w)
            # else:
            #     loss, loss_axis = criterion(output, t, w)
            # target = t
            losses.update(loss.item(), len(middle_points) * middle_points[0].size(0))
            loss_x.update(loss_axis[0].item(), len(middle_points) * middle_points[0].size(0))
            loss_y.update(loss_axis[1].item(), len(middle_points) * middle_points[0].size(0))
            loss_z.update(loss_axis[2].item(), len(middle_points) * middle_points[0].size(0))
            
            # if config.WANDB and config.WANDB_LOG_IMG:
            #     if i % config.LOG_IMAGE_FREQ == 0:
            #         # 3d target of the last sample in the batch
            #         # fig_target = plt.figure()
            #         # plot_3d_points(fig_target, t[-1].cpu().numpy())
                    
            #         fig_target = plot_3d_points_plotly(t[-1].cpu().numpy())
                    
            #         # fig_pred = plt.figure()
            #         # plot_3d_points(fig_pred, output[-1].detach().cpu().numpy())
            #         fig_pred = plot_3d_points_plotly(output[-1].detach().cpu().numpy())
                    
            #         figs_2d = []
            #         for view in range(len(target)):
            #             fig = plt.figure()
            #             plot_2d_points(fig, input[view][-1, :, :2].cpu().numpy(), input[view][-1, :, 2].cpu().numpy())
            #             figs_2d.append(fig)
            #         wandb.log({"{}/3d_target".format(validation_name): fig_target,
            #                 "{}/3d_pred".format(validation_name): fig_pred,
            #                 "epoch": epoch,
            #                 })
            #         for i, fig in enumerate(figs_2d):
            #             wandb.log({"{}/2d_input_view{}".format(validation_name, i): fig,
            #                     "epoch": epoch,})
            #         cols = ["3d_target", "3d_pred"]
            #         for view in range(len(target)):
            #             cols.append("2d_input_view{}".format(view))
            #         table = wandb.Table(columns=cols)
            #         path_to_plotly_html_fig_target = "./plotly_figure_fig_target.html"
            #         fig_target.write_html(path_to_plotly_html_fig_target, auto_play=False)
            #         path_to_plotly_html_fig_pred = "./plotly_figure_fig_pred.html"
            #         fig_pred.write_html(path_to_plotly_html_fig_pred, auto_play=False)
            #         row = [wandb.Html(path_to_plotly_html_fig_target), wandb.Html(path_to_plotly_html_fig_pred)]
            #         for fig in figs_2d:
            #             row.append(wandb.Image(fig))
            #         table.add_data(*row)
            #         wandb.log({"{}/3d_2d_table".format(validation_name): table, "epoch": epoch})
                    
            #         print('wandb {} image logged!'.format(validation_name))

            # ================== accuracy based on heatmap (64 * 64) ==================
            # nviews = len(output)
            # acc = [None] * nviews
            # cnt = [None] * nviews
            # pre = [None] * nviews
            # for j in range(nviews):
            #     _, acc[j], cnt[j], pre[j] = accuracy(
            #         output[j].detach().cpu().numpy(),
            #         target[j].detach().cpu().numpy())       # threshold: 64 / 10 * 0.5
            # acc = np.mean(acc)
            # cnt = np.mean(cnt)
            # avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            # ======================== Save prediction (heatmap + coords.) ========================
            preds = np.zeros((batch_size, njoints, 3), dtype=np.float32)     # (bs * #view, 17, 3)
            # heatmaps = np.zeros(
            #     (nimgs, njoints, height, width), dtype=np.float32)      # (bs * #view, 17, 64, 64)
            # for k, o, m in zip(range(nviews), output, meta):
            #     # o: (bs, 17, 64, 64)
            #     pred, maxval = get_final_preds(config,
            #                                    o.clone().cpu().numpy(),
            #                                    m['center'].numpy(),
            #                                    m['scale'].numpy())
            #     # pred:   (bs, num_joints=17, 2)    coordinate in original image (1000, 1000)
            #     # maxval: (bs, num_joints=17, 1)    peak value on heatmap
            #     pred = pred[:, :, 0:2]          # (bs, 17, 2)
            #     pred = np.concatenate((pred, maxval), axis=2)       # (bs, 17, 3)
            #     preds[k::nviews] = pred
            #     heatmaps[k::nviews] = o.clone().cpu().numpy()       # (bs, 17, 64, 64)

            preds = output.clone().cpu().numpy()
            gts = target.clone().cpu().numpy()
            if 'room_scaled' in meta:
                if 'room_scaled_equal' in meta:
                    room_scale = meta['room_x_scale'][0].item()
                    room_center = meta['room_center'][0].clone().cpu().numpy()
                    preds = preds * room_scale + room_center
                    gts = gts * room_scale + room_center
                else:
                    room_x_scale = meta['room_x_scale'][0].item()
                    room_y_scale = meta['room_y_scale'][0].item()
                    preds[:, :, 0] = preds[:, :, 0] * room_x_scale
                    preds[:, :, 1] = preds[:, :, 1] * room_y_scale
                    gts[:, :, 0] = gts[:, :, 0] * room_x_scale
                    gts[:, :, 1] = gts[:, :, 1] * room_y_scale
                    
            if 'shift_room_tri' in meta:
                preds = preds - meta['shift_room_tri'].clone().cpu().numpy()[:, np.newaxis, :]
                gts = gts - meta['shift_room_tri'].clone().cpu().numpy()[:, np.newaxis, :]
                
            all_preds[idx:idx + batch_size] = preds                      # (bs * #view, 17, 3) in original image
            # all_heatmaps[idx:idx + nimgs] = heatmaps
            all_gts[idx:idx + batch_size] = gts    # (bs * #view, 17, 3) in original image
            if 'joints_3d_conf' in meta:
                all_3d_confs[idx:idx + batch_size] = meta['joints_3d_conf'].clone().cpu().numpy().squeeze()
            idx += batch_size

            # # ======================== Log ========================
            if i % config.PRINT_FREQ == 0:
            # if True:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Loss_x {loss_x.val:.5f} ({loss_x.avg:.5f})\t' \
                       'Loss_y {loss_y.val:.5f} ({loss_y.avg:.5f})\t' \
                       'Loss_z {loss_z.val:.5f} ({loss_z.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, loss_x=loss_x, loss_y=loss_y, loss_z=loss_z, acc=avg_acc)
                logger.info(msg)

                for k in range(len(output)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    # save_debug_images(config, input[k], meta[k], target[k],
                    #                   pre[k] * 4, output[k], prefix)
                    
            if config.WANDB:
                wandb.log({
                    '{}/val_loss_iter'.format(validation_name): losses.val,
                    '{}/val_loss_avg'.format(validation_name): losses.avg,
                    '{}/val_loss_x_iter'.format(validation_name): loss_x.val,
                    '{}/val_loss_x_avg'.format(validation_name): loss_x.avg,
                    '{}/val_loss_y_iter'.format(validation_name): loss_y.val,
                    '{}/val_loss_y_avg'.format(validation_name): loss_y.avg,
                    '{}/val_loss_z_iter'.format(validation_name): loss_z.val,
                    '{}/val_loss_z_avg'.format(validation_name): loss_z.avg,
                    # 'val/val_accuracy_iter': avg_acc.val,
                    # 'val/val_accuracy_avg': avg_acc.avg,
                })

        #
        if is_mmpose:
            logger.info('\n****************************** ! Validating on MMPOSE ! ************************* ')
        msg = '----Test----: [{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t' \
                'Loss_y {loss_y.val:.4f} ({loss_y.avg:.4f})\t' \
                'Loss_z {loss_z.val:.4f} ({loss_z.avg:.4f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    loss=losses,
                    loss_x=loss_x,
                    loss_y=loss_y,
                    loss_z=loss_z,
                    acc=avg_acc)
        logger.info(msg)
        
        msg = 'Inference Time: sum (avg) {inference_time.sum:.3f}s ({inference_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s'.format(
                    inference_time=inference_time,
                    speed=nsamples / inference_time.sum,
                    )
        logger.info(msg)
        
        # write speed to a file
        with open(os.path.join(output_dir, 'speed.txt'), 'w') as f:
            f.write(str(nsamples / inference_time.sum))
            
        
        if config.WANDB:
            wandb.log({
                '{}/val_loss_epoch'.format(validation_name): losses.avg,
                '{}/val_loss_x_epoch'.format(validation_name): loss_x.avg,
                '{}/val_loss_y_epoch'.format(validation_name): loss_y.avg,
                '{}/val_loss_z_epoch'.format(validation_name): loss_z.avg,
                'epoch': epoch,
                # 'val/val_accuracy_epoch': avg_acc.avg,
            })

        # ======================= save all heatmaps and joint locations =======================
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        if os.environ.get('UNIQUE_EVAL_ARTIFACTS', '0') == '1' and config.TEST.EVAL_COMMENTS:
            stem, extension = os.path.splitext(save_file)
            safe_comments = ''.join(
                char if char.isalnum() or char in ('-', '_') else '_'
                for char in config.TEST.EVAL_COMMENTS
            )
            save_file = f'{stem}_{safe_comments}{extension}'
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        # file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['gts'] = all_gts[:, u, :]
        file['joint_names_order'] = a
        file.close()
        
        save_file = config.TEST.PRED_GT_LOCATION_FILE
        test_name = config.DATASET.TEST_DATASET
        state = config.TEST.STATE
        eval_comments = config.TEST.EVAL_COMMENTS
        if is_mmpose:
            use_mmpose = 'mmpose'
        else:
            use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
        if eval_comments != '':
            use_mmpose += '_{}'.format(eval_comments)
        file_name = os.path.join(output_dir, '{}_{}_{}_{}.pkl'.format(save_file.replace('.pkl', ''), test_name, use_mmpose, state))
        with open(file_name, 'wb') as f:
            pickle.dump({'pred': all_preds[:, u, :], 'gt': all_gts[:, u, :]}, f)
        
        # save gts and preds to a separate file with frame names
        if len(fnames) > 0:
            if config.DATASET.TEST_DATASET.startswith('multiview_rich'):
                fnames = [fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[0] for fname in fnames]
                fnames = [fname + rich_camera_id for fname, rich_camera_id in zip(fnames, valid_camera_ids)]
                # for idx, rich_camera_id in enumerate(rich_camera_ids):
                #     for cam_id in rich_camera_id:
                #         fnames[idx] += '_{:02d}'.format(cam_id.item())
            elif config.DATASET.TEST_DATASET.startswith('multiview_amass_rumpl') and valid_camera_ids != []:
                fnames = [fname + valid_camera_id for fname, valid_camera_id in zip(fnames, valid_camera_ids)]
            elif config.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic'):
                fnames = [fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[-1].replace('.jpg','') for fname in fnames]
                fnames = [fname + valid_camera_id for fname, valid_camera_id in zip(fnames, valid_camera_ids)]
            elif config.DATASET.TEST_DATASET.startswith('multiview_openmplposer'):
                fnames = [fname + rich_camera_id for fname, rich_camera_id in zip(fnames, valid_camera_ids)]
            else:
                fnames = [fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[-1].replace('.jpg','') for fname in fnames]
            # to_dump = {}
            # for fname, pred_dump, gt_dump, conf_3d in zip(fnames, all_preds[:, u, :], all_gts[:, u, :], all_3d_confs[:, u]):
            #     to_dump[fname] = {'GT': gt_dump, 'pred': pred_dump, 'conf_3d': conf_3d}
            to_dump = {
                'fnames': fnames,
                'pred': all_preds,
                'gt': all_gts,
                'confs_2d': all_confs,
            }
            file_name = os.path.join(output_dir, '{}_{}_{}_{}_dict.pkl'.format(save_file.replace('.pkl', ''), test_name, use_mmpose, state))
            with open(file_name, 'wb') as f:
                pickle.dump(to_dump, f)
                
        if config.DATASET.TEST_MMPOSE_CONFS_TH is not None:
            all_confs = all_confs.mean(axis=1)
            all_preds = all_preds[all_confs > config.DATASET.TEST_MMPOSE_CONFS_TH]
            all_gts = all_gts[all_confs > config.DATASET.TEST_MMPOSE_CONFS_TH]
            all_3d_confs = all_3d_confs[all_confs > config.DATASET.TEST_MMPOSE_CONFS_TH]
            logger.info('MMPOSE confs threshold applied: {}'.format(config.DATASET.TEST_MMPOSE_CONFS_TH))
        
        logger.info('Num samples: {}'.format(len(all_preds)))

        # ======================== evaluate MPJPE  ========================
        logger.info('Start 3D eval ....')
        per_action = True if config.DATASET.TEST_DATASET.startswith('multiview_h36m') else False
        logger.info(' -------------------- Relative --------------------')
        # >>>>>>>>>>>>>>>>>>>> relative evaluation
        name_value, perf_indicator = evaluate(all_preds[:, u, :], all_gts[:, u, :], dataset.actual_joints, config, output_dir, conf_3d=all_3d_confs, relative_evaluation=True, per_action=per_action, fnames=fnames, epoch=epoch, use_mmpose=use_mmpose, validation_name=validation_name)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')
        logger.info('Evaluate {}'.format(str(perf_indicator)))
        # <<<<<<<<<<<<<<<<<<<< relative evaluation
        
        logger.info(' -------------------- Absolute --------------------')
        
        # name_value, perf_indicator = dataset.evaluate(all_preds)
        name_value, perf_indicator = evaluate(all_preds[:, u, :], all_gts[:, u, :], dataset.actual_joints, config, output_dir, conf_3d=all_3d_confs, epoch=epoch, per_action=per_action, fnames=fnames, use_mmpose=use_mmpose, validation_name=validation_name)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')
        logger.info('Evaluate {}'.format(str(perf_indicator)))
        perf_indicator = 1 / perf_indicator
        logger.info('perf_indicator: {}'.format(perf_indicator))

    return perf_indicator

def index_to_action_names_h36m():
    return {
        2: 'Direction',
        3: 'Discuss',
        4: 'Eating',
        5: 'Greet',
        6: 'Phone',
        7: 'Photo',
        8: 'Pose',
        9: 'Purchase',
        10: 'Sitting',
        11: 'SittingDown',
        12: 'Smoke',
        13: 'Wait',
        14: 'WalkDog',
        15: 'Walk',
        16: 'WalkTwo'
    }

def evaluate(pred, gt, actual_joints, config, output_dir, conf_3d=None, relative_evaluation=False, epoch=None, per_action=False, fnames=None, use_mmpose=None, validation_name='val'):
        pred = pred.copy()
        gt = gt.copy()
        
        if config.DATASET.OUTPUT_IN_METER:
            pred = pred * 100
            gt = gt * 100
        
        if relative_evaluation:
            # Paper KP* metric: pelvis-anchored (mid-hip). For COCO 17, pelvis = (J[11] + J[12]) / 2.
            # For H36M, joint 0 already is pelvis.
            kp_std = getattr(config.DATASET, 'CMU_KEYPOINT_STANDARD', 'coco').lower()
            if kp_std == 'coco':
                pelvis_gt   = (gt[:,   11:12, :] + gt[:,   12:13, :]) / 2.0
                pelvis_pred = (pred[:, 11:12, :] + pred[:, 12:13, :]) / 2.0
            else:
                pelvis_gt   = gt[:,   0:1, :]
                pelvis_pred = pred[:, 0:1, :]
            gt   = gt   - pelvis_gt
            pred = pred - pelvis_pred

        if conf_3d is not None:
            gt[conf_3d <= 0] = np.nan
            pred[conf_3d <= 0] = np.nan

        # Already pre-centered above; pass 'absolute' so calc_mpjpe doesn't double-subtract nose.
        mode = 'absolute'
        pjpe, mpjpe = calc_mpjpe(gt, pred, mode=mode)
        
        name_values = collections.OrderedDict()
        joint_names = actual_joints
        for i in range(len(joint_names)):
            name_values[joint_names[i]] = pjpe[i]
            
        logger.info('3D MPJPE (cm): {}'.format(mpjpe))
        to_log = print_per_kp(pjpe, actual_joints.values())
        logger.info('3D MPJPE per keypoint: {}'.format(to_log))
        
        distance_per_dim_per_kp, distance_per_dim = calc_distance_per_dim(pred, gt) 
        logger.info('Distance per dimension: {}'.format(distance_per_dim))
        to_log = print_per_kp(distance_per_dim_per_kp, actual_joints.values())
        logger.info('Distance per dimension per keypoint: {}'.format(to_log))
        
        if config.WANDB:
            if relative_evaluation:
                wandb.log({
                    '{}/val_rel_3d_mpjpe_epoch'.format(validation_name): mpjpe,
                    '{}/val_rel_distance_x_epoch'.format(validation_name): distance_per_dim[0],
                    '{}/val_rel_distance_y_epoch'.format(validation_name): distance_per_dim[1],
                    '{}/val_rel_distance_z_epoch'.format(validation_name): distance_per_dim[2],
                    'epoch': epoch,
                })
            else:
                wandb.log({
                    '{}/val_3d_mpjpe_epoch'.format(validation_name): mpjpe,
                    '{}/val_distance_x_epoch'.format(validation_name): distance_per_dim[0],
                    '{}/val_distance_y_epoch'.format(validation_name): distance_per_dim[1],
                    '{}/val_distance_z_epoch'.format(validation_name): distance_per_dim[2],
                    'epoch': epoch,
                })
            
                log_wandb_per_kp = {'kp_{}/{}_{}_3d_mpjpe_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), pjpe)}
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_x_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 0])})
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_y_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 1])})
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_z_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 2])})
                log_wandb_per_kp.update({'epoch': epoch})
                wandb.log(log_wandb_per_kp)
            
        test_name = config.DATASET.TEST_DATASET
        if use_mmpose is None:
            use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
        absolute_or_relative = 'relative' if relative_evaluation else 'absolute'
        state = config.TEST.STATE
        pkl_file = os.path.join(output_dir, 'mpjpe_{}_{}_{}_{}.pkl'.format(test_name, use_mmpose, state, absolute_or_relative))
        with open(pkl_file, 'wb') as f:
            pickle.dump(name_values, f)
        
        
        if per_action:
            action_names = index_to_action_names_h36m()
            actions = np.array([int(fname.split('_')[3]) for fname in fnames])
            mpjpe_per_action = {}
            pjpe_per_action = {}
            distance_per_dim_per_action = {}
            for action in action_names:
                idx = actions == action
                if np.sum(idx) > 0:
                    gt_action = gt[idx]
                    pred_action = pred[idx]
                    pjpe_action, mpjpe_action = calc_mpjpe(gt_action, pred_action, mode=mode)
                    distance_per_dim_per_kp_action, distance_per_dim_action = calc_distance_per_dim(pred_action, gt_action) 
                    mpjpe_per_action[action] = mpjpe_action
                    pjpe_per_action[action] = pjpe_action
                    distance_per_dim_per_action[action] = distance_per_dim_action
            to_log = print_per_kp(mpjpe_per_action.values(), action_names.values())
            logger.info('3D MPJPE per action: {}'.format(to_log))
            pkl_file = os.path.join(output_dir, 'mpjpe_perAction_{}_{}_{}_{}.pkl'.format(test_name, use_mmpose, state, absolute_or_relative))
            to_dump = {k: v for k, v in zip(action_names.values(), mpjpe_per_action.values())}
            with open(pkl_file, 'wb') as f:
                pickle.dump(to_dump, f)
            # for action in action_names:
            #     to_log = print_per_kp(pjpe_per_action[action], actual_joints.values())
            #     logger.info('3D MPJPE per keypoint for action {}: {}'.format(action, to_log))
            if config.WANDB:
                if relative_evaluation:
                    log_wandb_per_action = {'action_{}/{}_rel_3d_mpjpe_epoch'.format(action_names[action], validation_name): v for action, v in mpjpe_per_action.items()}
                    log_wandb_per_action.update({'epoch': epoch})
                    wandb.log(log_wandb_per_action)
                else:
                    log_wandb_per_action = {'action_{}/{}_3d_mpjpe_epoch'.format(action_names[action], validation_name): v for action, v in mpjpe_per_action.items()}
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_x_epoch'.format(action_names[action], validation_name, action_names[action]): v[0] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_y_epoch'.format(action_names[action], validation_name, action_names[action]): v[1] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_z_epoch'.format(action_names[action], validation_name, action_names[action]): v[2] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'epoch': epoch})
                    wandb.log(log_wandb_per_action)
                    
                    
                
            #         log_wandb_per_kp = {'kp_{}/val_{}_3d_mpjpe_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), pjpe)}
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_x_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 0])})
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_y_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 1])})
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_z_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 2])})
            #         log_wandb_per_kp.update({'epoch': epoch})
            #         wandb.log(log_wandb_per_kp)
                
        return name_values, mpjpe 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def inference(config,
             loader,
             dataset,
             model,
             output_dir,):

    model.eval()
    batch_time = AverageMeter()
    inference_time = AverageMeter()

    n_view = 6 if config.DATASET.TEST_DATASET == 'multiview_skipose' else 4
    n_view = 5 if config.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else n_view
    n_view = len(config.DATASET.TEST_VIEWS) if config.DATASET.TEST_VIEWS is not None else n_view
    nsamples = len(dataset)

    njoints = config.NETWORK.NUM_JOINTS                 # 17
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_confs = np.zeros((nsamples, njoints), dtype=np.float32)      # (#sample, 17)


    idx = 0
    with torch.no_grad():
        end = time.time()
        
        for i, (middle_points, closest_points_all, target, rays, meta, joints_2ds) in enumerate(loader):
            # middle_points: (B, 17, 1, 3)
            # closest_points_all: (B, 17, n_views, 3)
            # target: (B, 17, 3)

            # ======================== combinations of input ========================
            batch_size = middle_points.shape[0]

            confs = rays[:, :, :, -1].clone().detach().cpu().numpy()
            # print('confs:', confs.shape)
            all_confs[idx:idx + batch_size] = confs.min(axis=-1)
            
            
            start_inference = time.time()
            if config.NETWORK.FEED_CAMERA_CALIBRATION or config.NETWORK.FEED_ONLY_2D:
                output = model(joints_2ds, is_training=False)
            elif config.NETWORK.APPLY_VIEW_FUSION:
                output = model(rays, is_training=False)
            elif config.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS:
                output = model(middle_points)
            else:
                output = model(closest_points_all)
            inference_time.update(time.time() - start_inference)
            
            batch_time.update(time.time() - end)
            end = time.time()

            # ======================== Save prediction (heatmap + coords.) ========================
            preds = np.zeros((batch_size, njoints, 3), dtype=np.float32)     # (bs * #view, 17, 3)

            preds = output.clone().cpu().numpy()
            if 'room_scaled' in meta:
                if 'room_scaled_equal' in meta:
                    room_scale = meta['room_x_scale'][0].item()
                    room_center = meta['room_center'][0].clone().cpu().numpy()
                    preds = preds * room_scale + room_center
                    gts = gts * room_scale + room_center
                else:
                    room_x_scale = meta['room_x_scale'][0].item()
                    room_y_scale = meta['room_y_scale'][0].item()
                    preds[:, :, 0] = preds[:, :, 0] * room_x_scale
                    preds[:, :, 1] = preds[:, :, 1] * room_y_scale
                    gts[:, :, 0] = gts[:, :, 0] * room_x_scale
                    gts[:, :, 1] = gts[:, :, 1] * room_y_scale
            all_preds[idx:idx + batch_size] = preds                      # (bs * #view, 17, 3) in original image
            idx += batch_size

            # # ======================== Log ========================
            if i % config.PRINT_FREQ == 0:
            # if True:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time)
                logger.info(msg)

    np.savez(output_dir, points3d=all_preds)
    logger.info('Saved 3D points to {}'.format(output_dir))

    return

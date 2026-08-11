"""ST-VFT Phase 1 — clip 级 dataloader (设计文档 v1 §3)。

读 clip 级 stage_V_room (米制),输出时序窗口样本喂给 STVFT。

ray 几何 **严格复刻 RUMPL** (baseline clip_full_conf 运行时 config),保证 ST-VFT 和
baseline 输入表示逐字节一致 (控制变量):
  - INPUTS_NORMALIZED=true   : 2D 归一化 normalize_screen
  - NORMALIZE_CAMERAS=true   : K 归一化
  - USE_T=false              : cam_center = camera['T'] (世界系光心)
  - INTERSECTION_RAY_WITH=Closest : intersection = 射线与 x/y/z 平面交点中离原点最近者
  - DOWNSAMPLE=1, USE_GRID=false, NO_AUGMENTATION=true
  - ray = [direction(3), intersection(3)] + conf(1)
    (ST-VFT 把 conf 拆出单独传, 因 ray_embed 分开编码; 与 baseline 模型内部拆分等价)

复刻自 RUMPL joints_dataset_rumpl.py: normalize_screen_coordinates / create_3d_ray_coords /
generate_direction_vectors_and_intersection_points / line_properties_batch,
以及 multiview_amass_rumpl.py get_rays 的 NORMALIZE_CAMERAS 段。
"""
import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_W, IMAGE_H = 1920, 1080  # 渲染分辨率 (02_clip --image-width/height)


def normalize_screen(X, w, h):
    """RUMPL normalize_screen_coordinates: [0,w]→[-1,1] 保持长宽比。X: (...,2)"""
    return (X / w) * 2 - np.array([1.0, h / w], dtype=X.dtype)


def _closest_plane_intersection(p1, p2):
    """复刻 RUMPL line_properties_batch 的 Closest 分支。
    line 由 p1→p2 定义; 求与 x=0/y=0/z=0 平面交点, 取离原点 (0,0,0) 最近者。
    p1, p2: (J,3)。返回 (J,3)。
    """
    J = p1.shape[0]
    out = np.zeros((J, 3), dtype=np.float32)
    BIG = 100_000.0
    for i in range(J):
        x1, y1, z1 = p1[i]
        x2, y2, z2 = p2[i]
        cands = []
        # x=0 平面
        if x2 != x1:
            t = -x1 / (x2 - x1)
            cands.append((0.0, y1 + t * (y2 - y1), z1 + t * (z2 - z1)))
        else:
            cands.append((0.0, BIG, BIG))
        # y=0 平面
        if y2 != y1:
            t = -y1 / (y2 - y1)
            cands.append((x1 + t * (x2 - x1), 0.0, z1 + t * (z2 - z1)))
        else:
            cands.append((BIG, 0.0, BIG))
        # z=0 平面
        if z2 != z1:
            t = -z1 / (z2 - z1)
            cands.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1), 0.0))
        else:
            cands.append((BIG, BIG, 0.0))
        cands = np.array(cands, dtype=np.float32)
        norms = np.linalg.norm(cands, axis=1)
        out[i] = cands[int(np.argmin(norms))]
    return out


def build_rays_one_view(joints_2d, cam):
    """对单帧单视角的 17 个 2D 关节, 复刻 RUMPL 构造 [direction(3), intersection(3)]。
    joints_2d: (17,2) 像素; cam: dict{fx,fy,cx,cy,R(3,3),T(3,1)} 米制, T=光心。
    返回 directions(17,3), intersections(17,3)。
    """
    fx, fy = float(cam['fx']), float(cam['fy'])
    cx, cy = float(cam['cx']), float(cam['cy'])
    R = np.asarray(cam['R'], dtype=np.float32)            # (3,3)
    T = np.asarray(cam['T'], dtype=np.float32).reshape(3)  # (3,) 世界系光心

    # RUMPL get_rays: if cx<cy swap image_size
    W, H = (IMAGE_W, IMAGE_H) if cx >= cy else (IMAGE_H, IMAGE_W)

    # NORMALIZE_CAMERAS=true: 归一化 K
    cc = normalize_screen(np.array([cx, cy], dtype=np.float32), W, H)  # (2,)
    fx_n, fy_n = fx / W * 2, fy / W * 2
    cx_n, cy_n = cc[0], cc[1]

    # INPUTS_NORMALIZED=true: 归一化 2D
    j_n = normalize_screen(joints_2d.astype(np.float32), W, H)  # (17,2)

    # create_3d_ray_coords (downsample=1, use_grid=false): 反投影到相机系
    x = (j_n[:, 0] - cx_n) / fx_n
    y = (j_n[:, 1] - cy_n) / fy_n
    coords_cam = np.stack([x, y, np.ones_like(x)], axis=1)  # (17,3)
    # use_t=false: coords_world = R^T @ coords_cam^T + T
    coords_world = (R.T @ coords_cam.T + T.reshape(3, 1)).T   # (17,3)

    cam_center = np.broadcast_to(T.reshape(1, 3), coords_world.shape)  # (17,3)

    # generate_direction_vectors_and_intersection_points:
    # line_properties_batch(points1=coords_world, points2=cam_center)
    #   direction = points2 - points1 = cam_center - coords_world
    directions = (cam_center - coords_world).astype(np.float32)        # (17,3)
    intersections = _closest_plane_intersection(coords_world, cam_center)  # (17,3)
    return directions, intersections


class STVFTClipDataset(Dataset):
    """读 clip 级 stage_V_room, 输出时序窗口样本。

    __getitem__ 返回 dict:
      rays:      (J, V, L, 6)   [direction, intersection], 米制
      confs:     (J, V, L, 1)
      delta_ts:  (V, L)         秒, (arange(L)-t_target)/fps (逐clip读fps)
      gt_3d:     (L, J, 3)      米制 (全窗口 GT, 供 per-t supervision)
    """

    # COCO-17 OKS sigmas (标准, 与 baseline COCO_PERSON_SIGMAS 一致)
    _SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072,
                        .062, .062, .107, .107, .087, .087, .089, .089])

    def __init__(self, data_glob, L_window=5, t_target=None, max_views=20, min_oks=0.0,
                 perturb=0.0, perturb_offset_px=120.0, eval_fixed_window=False):
        self.L = L_window
        self.t_target = (L_window // 2) if t_target is None else t_target
        self.max_views = max_views
        # eval 用: True=每clip取居中固定窗口(确定性), 否则随机连续L帧。
        # 修 bug: 随机窗口会让 B(算一次) vs C(每次重算) 比在不同数据上 → ΔMPJPE 混入采样噪声。
        self.eval_fixed_window = eval_fixed_window
        self.min_oks = min_oks   # >0: 滤掉中心帧中位OKS<阈值的clip(HRNet检测崩的垃圾)
        # 帧质量扰动(路B 决策4, 制造"落差"教模型选择性借): perturb=总扰动概率(0.6=30%中心+30%邻帧+40%不扰)
        self.perturb = perturb
        self.perturb_offset_px = perturb_offset_px
        from collections import Counter
        self.perturb_stats = Counter()   # 调试: 验三种情景比例
        self.clips = []   # list of dict (每个 clip 一条)
        files = sorted(glob.glob(data_glob))
        assert len(files) > 0, f"no pkl matched: {data_glob}"
        n_drop = 0
        for f in files:
            d = pickle.load(open(f, 'rb'))
            N = d['joints_3d'].shape[0]
            ct = d['joints_3d'].shape[1] // 2
            for i in range(N):
                if min_oks > 0:
                    hr, gt = d['joints_2d_mmpose'][i, ct], d['joints_2d_amass'][i, ct]  # (V,17,2)
                    oks_v = []
                    for v in range(hr.shape[0]):
                        g = gt[v].astype(np.float64)
                        area = (g[:, 0].max() - g[:, 0].min()) * (g[:, 1].max() - g[:, 1].min())
                        d2 = ((hr[v].astype(np.float64) - g) ** 2).sum(1)
                        oks_v.append(np.mean(np.exp(-d2 / ((self._SIGMAS * 2) ** 2) / (area + 1e-9) / 2)))
                    if np.median(oks_v) < min_oks:
                        n_drop += 1
                        continue
                self.clips.append({
                    'joints_3d': d['joints_3d'][i],              # (27,17,3)
                    'joints_2d': d['joints_2d_mmpose'][i],       # (27,20,17,2)
                    'confs': d['confs_2d_mmpose'][i],            # (27,20,17,1)
                    'cams': d['camera_parameters_all'][i],       # list[20] dict
                    'fps': float(d['frame_rate'][i]),
                })
        if min_oks > 0:
            print(f"[STVFTClipDataset] min_oks={min_oks}: 丢弃 {n_drop} 个检测崩的clip, 留 {len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def _perturb_plan(self, L, tt):
        """返回 (bad_li, lower_conf)。bad_li=None=不扰动。
        总 perturb 概率内: 50%中心崩/50%邻帧崩; 崩帧内 50% mode1(降conf)/50% mode2(高conf错,conf不变)。"""
        if self.perturb <= 0 or np.random.rand() >= self.perturb:
            self.perturb_stats['none'] += 1
            return None, False
        if np.random.rand() < 0.5:
            bad_li = tt                              # 情景A 中心帧崩
            self.perturb_stats['center'] += 1
        else:
            others = [i for i in range(L) if i != tt]
            bad_li = int(np.random.choice(others))   # 情景B 邻帧崩
            self.perturb_stats['neighbor'] += 1
        lower_conf = np.random.rand() < 0.5          # mode1 降conf / mode2 高conf错
        self.perturb_stats['m1' if lower_conf else 'm2'] += 1
        return bad_li, lower_conf

    def __getitem__(self, idx):
        c = self.clips[idx]
        T_total = c['joints_3d'].shape[0]   # 27
        V = min(self.max_views, c['joints_2d'].shape[1])
        L, tt = self.L, self.t_target

        # 窗口: 训练随机连续L帧; eval用居中固定窗口(确定性, 让B/C同窗口可比)
        if self.eval_fixed_window:
            start = (T_total - L) // 2
        else:
            start = np.random.randint(0, T_total - L + 1)
        frames = np.arange(start, start + L)

        bad_li, lower_conf = self._perturb_plan(L, tt)   # 帧质量扰动计划

        J = c['joints_3d'].shape[1]  # 17
        rays = np.zeros((J, V, L, 6), dtype=np.float32)
        confs = np.zeros((J, V, L, 1), dtype=np.float32)
        for li, fr in enumerate(frames):
            for v in range(V):
                j2d = c['joints_2d'][fr, v].astype(np.float32)   # (17,2)
                cf = c['confs'][fr, v, :, 0].astype(np.float32)  # (17,)
                if li == bad_li:   # 这帧崩(所有视角)
                    off = self.perturb_offset_px
                    if lower_conf:   # mode1 低conf崩: 散开(不确定在哪) + 降conf
                        j2d = j2d + np.random.uniform(-off, off, j2d.shape).astype(np.float32)
                        cf = cf * 0.1
                    else:            # mode2 高conf错: 整体平移(自信检测到错人/错位置), conf不变
                        j2d = j2d + np.random.uniform(-off, off, (1, 2)).astype(np.float32)
                cam = c['cams'][v]
                dirs, inters = build_rays_one_view(j2d, cam)
                rays[:, v, li, 0:3] = dirs
                rays[:, v, li, 3:6] = inters
                confs[:, v, li, 0] = cf

        fps = c['fps']
        dt = (np.arange(L, dtype=np.float32) - tt) / fps     # (L,)
        delta_ts = np.broadcast_to(dt[None, :], (V, L)).copy()  # (V,L)

        gt_3d = c['joints_3d'][frames].astype(np.float32)    # (L,17,3)

        return {
            'rays': torch.from_numpy(rays),
            'confs': torch.from_numpy(confs),
            'delta_ts': torch.from_numpy(delta_ts),
            'gt_3d': torch.from_numpy(gt_3d),
        }


def make_collate_random_views(min_views=2, max_views=5, seed=None):
    """batch 级 random-views collate: 每个 batch 随机 k∈[min,max] 个视角 (batch 内统一)。"""
    rng = np.random.default_rng(seed)

    def collate(batch):
        V_total = batch[0]['rays'].shape[1]
        k = int(rng.integers(min_views, max_views + 1))
        k = min(k, V_total)
        view_idx = rng.choice(V_total, size=k, replace=False)
        view_idx = torch.from_numpy(np.sort(view_idx))

        rays = torch.stack([b['rays'][:, view_idx] for b in batch])       # (B,J,k,L,6)
        confs = torch.stack([b['confs'][:, view_idx] for b in batch])     # (B,J,k,L,1)
        delta_ts = torch.stack([b['delta_ts'][view_idx] for b in batch])  # (B,k,L)
        gt_3d = torch.stack([b['gt_3d'] for b in batch])                  # (B,L,J,3)
        return {'rays': rays, 'confs': confs, 'delta_ts': delta_ts, 'gt_3d': gt_3d}

    return collate


def make_collate_fixed_views(n_views, seed=0):
    """eval 用: 每 batch 固定取 n_views 个视角 (确定性)。与 12w 评估同口径(固定视角数,
    不像训练的随机 k∈[2,5])。每 batch 随机抽 n_views 个但种子固定 → 可复现。"""
    rng = np.random.default_rng(seed)

    def collate(batch):
        V_total = batch[0]['rays'].shape[1]
        k = min(n_views, V_total)
        view_idx = torch.from_numpy(np.sort(rng.choice(V_total, size=k, replace=False)))
        rays = torch.stack([b['rays'][:, view_idx] for b in batch])
        confs = torch.stack([b['confs'][:, view_idx] for b in batch])
        delta_ts = torch.stack([b['delta_ts'][view_idx] for b in batch])
        gt_3d = torch.stack([b['gt_3d'] for b in batch])
        return {'rays': rays, 'confs': confs, 'delta_ts': delta_ts, 'gt_3d': gt_3d}

    return collate


def test_stvft_dataset():
    glob_pat = "/mnt/data/cjydata/mhp_workspace/sanity/stage_V_room/train/*.pkl"
    ds = STVFTClipDataset(glob_pat, L_window=5)
    print(f"dataset size: {len(ds)} clips")
    s = ds[0]
    print("sample shapes:", {k: tuple(v.shape) for k, v in s.items()})
    assert s['rays'].shape[0] == 17 and s['rays'].shape[-1] == 6
    assert s['confs'].shape[-1] == 1
    # ray 数值合理性 (非全0/NaN)
    assert torch.isfinite(s['rays']).all(), "rays 含 NaN/Inf"
    assert s['rays'].abs().sum() > 0, "rays 全0"
    print("ray dir 范围:", s['rays'][..., :3].min().item(), s['rays'][..., :3].max().item())
    print("ray inter 范围:", s['rays'][..., 3:6].min().item(), s['rays'][..., 3:6].max().item())
    print("delta_ts[0]:", s['delta_ts'][0].tolist(), "(应 [-2/fps..2/fps], 居中=0)")

    from torch.utils.data import DataLoader
    collate = make_collate_random_views(2, 5, seed=0)
    dl = DataLoader(ds, batch_size=4, collate_fn=collate)
    b = next(iter(dl))
    print("batch shapes:", {k: tuple(v.shape) for k, v in b.items()})
    print("STVFTClipDataset PASS")


if __name__ == "__main__":
    test_stvft_dataset()

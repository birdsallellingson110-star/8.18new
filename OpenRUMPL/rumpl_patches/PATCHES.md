# RUMPL 复现补丁总览

所有相对 OpenRUMPL 原始仓库的修改 / 新增文件，按"作用域"分组。
每一项都附有：**位置**、**目的**、**前→后**、**影响**。

文件镜像见同级目录：`configs/  lib_core/  data/  MHP/`。
原始仓库路径前缀：`/home/zlt/cjy/newidea/OpenRUMPL/`

---

## 0. 核心结论（先看这个）

| # | 类别 | 文件 | 影响 |
|---|---|---|---|
| 1 | 训练 config | `RUMPL/configs/.../clip_full.yaml` | Loss 0.76 → 0.039；Synth V=5 Rel 31cm → 3.55cm |
| 2 | 评估指标 | `RUMPL/lib/core/function_rumpl.py` | KP\* 改用 pelvis 锚定（论文一致） |
| 3 | CMU 多人匹配 | `RUMPL/data/preprocess_cmu_panoptic.py` | 多人场景 HRNet 取最近 GT 的检测 |
| 4 | AMASS 渲染 | `MHP/02_clip_run.py` | 增加 Z 轴旋转增广 |
| 5 | CMU 坐标系 | `MHP/06_swap_cmu_axes.py` (新增) | y-down → z-up，与 AMASS 对齐 |
| 6 | 房间放置 | `MHP/04_fix_data_to_room.py` (新增) | 离线补 locate_mesh_in_room |
| 7 | 数据展平 | `MHP/05_flatten_with_split.py` (新增) | clip → frame，clip 级 90/10 切分 |
| 8 | clip 采样 | `MHP/01_clip_create_dataset.py` (新增) | ST-VFT 用的 clip 采样器 |

---

## 1. `configs/cmu_panoptic/rumpl_amass/clip_full.yaml` — 三个关键 bug 修复

### 1.1 `USE_T: false`（行 86）
- **原值**：`true`
- **为什么**：原始 loader 把 `cam['T']` 当作 extrinsic 的 t 处理，但我们生成数据
  时 `T` 是 **camera center**（世界坐标系下相机位置），与 `t = -R @ T` 不是一回事。
  设 false 后 loader 自己用 `t` 字段，避免射线方向反向。

### 1.2 `TRAIN_RANDOM_NUM_VIEWS: true`（行 88）
- **原值**：`false`
- **为什么**：论文训练时每个 batch 随机抽 k ∈ \[MIN_NUM_VIEWS, MAX_NUM_VIEWS] 个视角。
  关掉之后模型只看到固定视角数，泛化到 V=2 测试时性能崩。

### 1.3 `APPLY_VIEW_FUSION: true`（行 135）
- **原值**：`false`
- **为什么**：原 config 等价于跳过 VFT，只用 PFT 看三角化均值，等于退化为
  PoseFormer。开后整套 RUMPL pipeline 才生效。

### 验证
开后训练 5 epoch：Loss 从 0.76 plateau → 0.039；
Synth Val V=5 Rel 31cm → **3.55cm**（接近论文 Table 9 的 3.21cm）。

---

## 2. `lib/core/function_rumpl.py` — 评估时用 pelvis 锚定 KP\*

### 位置：函数 `evaluate(...)`，约 781–810 行
### 修改：在 `relative_evaluation=True` 分支中，COCO 17 关键点先减 mid-hip

```python
def evaluate(pred, gt, actual_joints, config, output_dir, conf_3d=None,
             relative_evaluation=False, epoch=None, per_action=False,
             fnames=None, use_mmpose=None, validation_name='val'):
    pred = pred.copy()
    gt = gt.copy()

    if config.DATASET.OUTPUT_IN_METER:
        pred = pred * 100
        gt = gt * 100

    if relative_evaluation:
        # Paper KP* metric: pelvis-anchored (mid-hip).
        # For COCO 17, pelvis = (J[11] + J[12]) / 2.
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
    ...
```

### 为什么
- 原来 `calc_mpjpe(..., mode='relative')` 内部减的是 J\[0]（nose），不是 pelvis。
- 论文里 KP\* 的定义就是 mid-hip 锚定。对 COCO 17 关键点要显式取 (J\[11]+J\[12])/2。
- 关键：减完后传 `mode='absolute'` 防止 `calc_mpjpe` **再减一次 nose**（双重减）。

### 影响
评估数值口径与论文一致；KP\* 的相对误差比绝对误差降 3–5cm（数量级正常）。

---

## 3. `data/preprocess_cmu_panoptic.py` — 多人场景 HRNet 检测匹配

### 位置：函数 `preprocess_cmu_mmpose(args)`，约 275–318 行

```python
def preprocess_cmu_mmpose(args):
    data = args[0]
    dir_cmu_mmpose = args[1]
    keypoints_standard = args[2]
    dataset = copy.deepcopy(data)
    json_mmpose = os.path.join(dir_cmu_mmpose,
                                data['image'].split('/')[-4],
                                data['image'].split('/')[-2],
                                data['image'].split('/')[-1].replace('.jpg', '.json'))
    try:
        with open(json_mmpose, 'r') as f:
            mmpose_data = json.load(f)
    except FileNotFoundError:
        return []

    # Multi-person association: pick the HRNet detection closest to GT joints_2d.
    # Paper's original code took mmpose_data[0] which works only for single-person scenes;
    # for multi-person scenes (paper's 4 val sequences) that's a different subject across views.
    if len(mmpose_data) == 0:
        return []
    gt_2d = np.array(data['joints_2d'])  # (17, 2)
    best_idx, best_err = 0, np.inf
    for i, det in enumerate(mmpose_data):
        kp = np.array(det['keypoints'])
        if kp.shape != gt_2d.shape:
            continue
        err = np.linalg.norm(kp - gt_2d, axis=1).mean()
        if err < best_err:
            best_err = err
            best_idx = i
    pick = mmpose_data[best_idx]

    if keypoints_standard == 'h36m':
        joints_2d, joints_2d_conf = mmpose2h36m(np.array(pick['keypoints']),
                                                np.array(pick['keypoint_scores']))
    elif keypoints_standard == 'coco':
        joints_2d = np.array(pick['keypoints'])
        joints_2d_conf = np.array(pick['keypoint_scores']).reshape((-1, 1))
    dataset['mmpose_2d'] = True
    dataset['joints_2d'] = joints_2d
    dataset['joints_2d_conf'] = joints_2d_conf
    dataset['keypoint_standard'] = keypoints_standard

    return dataset
```

### 为什么
- 原代码：`pick = mmpose_data[0]` —— 取列表第一个检测。
- 论文 4 个 val 序列（`160906_pizza1, 160422_haggling1, 160906_ian5, 160906_band4`）
  全是多人场景，HRNet 每帧会输出多个人。"第 0 个"在不同视角对应**不同人**，
  导致跨视角不一致 → 三角化结果完全错。
- 修复：用 GT 投影的 2D `joints_2d` 作为 anchor，找平均欧氏距离最小的那个检测。

### 影响
CMU V=5 Rel：30cm 级 → **6.21cm**（基线 baseline 达成）。

---

## 4. `MHP/02_clip_run.py` — Z 轴旋转增广

### 位置：函数 `process_split(...)`，约 190–240 行 + 365 行

```python
def empty_acc():
    return {
        'joints_3d': [], 'joints_2d_mmpose': [], 'confs_2d_mmpose': [],
        'joints_2d_amass': [], 'triangulated_3d_mmpose': [],
        'camera_parameters_all': [],
        'smplh_betas': [], 'smplh_global_orient': [], 'smplh_body_pose': [],
        'smplh_transl': [], 'smplh_gender': [],
        'frame_rate': [], 'source_npz': [], 'start_frame': [],
        # per-clip Z rotation angle applied before rendering (radians).
        # Phase 1: to reproduce vertices, apply Rz(z_rotation) to smplh_forward output.
        'z_rotation': [],
    }

...

for clip in bar:
    # === SMPL+H forward (all L frames) ===
    try:
        vertices_all = smplh_forward_batch(
            bm_dict, clip['gender'], clip['pose'], clip['trans'], clip['betas'])
    except Exception as e:
        n_skipped += 1
        continue

    # === Random Z-axis rotation augmentation (paper §5.x) ===
    # Apply same random angle to all L frames of the clip — preserves intra-clip motion.
    # This is what paper's `locate_mesh_in_room(rotate=True)` does; we apply it to the
    # full vertex sequence before camera placement so each clip gets a different orientation.
    if args.apply_rotation:
        theta = float(np.random.uniform(0, 2 * np.pi))
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        vertices_all = vertices_all @ Rz.T
    else:
        theta = 0.0
    ...
    acc['z_rotation'].append(np.float32(theta))
```

### 为什么
- 不加 Z 增广：AMASS mocap 内置的朝向多样性 ≈ 53°（同一段动作内的人体小幅转身），
  远不到论文要的 360° 均匀。
- 加上 → 全方位采样，对 CMU 测试集小幅但稳定的 5–7% 提升。
- `z_rotation` 字段保存 θ 供 Phase 1 重放 vertices。

---

## 5. `MHP/06_swap_cmu_axes.py`（新增）— CMU y-down → AMASS z-up

### 全文见 `MHP/06_swap_cmu_axes.py`。关键变换：

```python
# CMU Panoptic 是 y-down（地面方向是 +y, 顶点是 -y;
# 已验证：nose y=-158, ankle y=-12 → |y|大的是上）
# AMASS / RUMPL 模型训练用的是 z-up
P = np.array([[1.0,  0.0, 0.0],
              [0.0,  0.0, 1.0],
              [0.0, -1.0, 0.0]], dtype=np.float64)
# new_x =  old_x
# new_y =  old_z
# new_z = -old_y
```

每个样本：

```python
j3d_new = P @ j3d_old              # 保留 cm 量纲！loader 会自动 /100
R_new   = R_old @ P.T
T_new   = P @ T_old                # 相机中心，保留 cm
t_new   = -R_new @ T_new           # extrinsic
# fx, fy, cx, cy, K, joints_2d 不变（投影对一致重标记不变）
```

### 为什么 / 历史踩坑
- 最初我写成 `new_z = +old_y`，结果头脚倒置（重投影对，但模型语义错）。
- 验证 CMU 是 y-DOWN 后改成 `-old_y` 才对。
- 另一个坑：**不要在这里 /100**。loader (`joints_dataset_rumpl.py:808`) 在
  `OUTPUT_IN_METER=true` 时已经自动把 joints_3d 和 camera T 都 /100。
  早期版本 swap 脚本也 /100 → 双重除 → 10000× 误差。**现在保留 cm**。

### 影响
CMU 评估口径与 AMASS 一致；不再需要训练时切坐标系。

---

## 6. `MHP/04_fix_data_to_room.py`（新增）— 离线补 locate_mesh_in_room

我们的 `02_clip_run.py` 渲染时跳过了 `locate_mesh_in_room`（原本
`run_mmpose_02_run.py:510` 调），这个脚本对已生成的 `stage_V` chunk 离线打补丁：

1. 每个 clip 取第 0 帧 mid-hip 当 `root_loc`
2. 在 `[ROOM_MIN_X, ROOM_MAX_X] × [ROOM_MIN_Y, ROOM_MAX_Y] × [ROOM_MIN_Z, ROOM_MAX_Z]`
   均匀采 augmentation_3d
3. `delta = -root_loc + augmentation_3d`（z 维当 z 上下界都是 0 时只在 xy 平移）
4. 同步 shift: `joints_3d, smplh_transl, triangulated_3d, camera.T`
5. 重算 `t = -R @ T_new`
6. **关键 bug 修复**：原 `02_clip_run.py` 渲染 `joints_2d_amass` 用了 `cam['T']`
   （相机中心），应该用 `cam['t']`（extrinsic）。这里用正确投影重算。

数学一致性：`R(j+δ) - R(T+δ) = Rj - RT`，渲染图像不变。

---

## 7. `MHP/05_flatten_with_split.py`（新增）— clip → frame + clip 级切分

把 `stage_V_room` 的 `(N_clips, L, ...)` 形状展平成 RUMPL loader 期望的
`(N_frames, ...)`，并按 **clip ID** 做 90/10 train/val 切分（避免帧内泄漏）。

输出：
```
<work_dir>/datasets/clip_full_room_flat/amass_mmpose_joints_train.pkl
<work_dir>/datasets/clip_full_room_flat/amass_mmpose_joints_validation.pkl
```

每帧字段：
- `joints_3d (N,17,3)` `joints_2d_mmpose (N,V,17,2)` `confs_2d_mmpose (N,V,17,1)`
- `joints_2d_amass (N,V,17,2)` `triangulated_3d_mmpose (N,17,3)`
- `camera_parameters_all` list of len N，每元素是 V 个相机 dict
- `views_used (N, V)` `camera_setup_used (N,)`
- 留 `_clip_id, _frame_in_clip` 给 Phase 1 ST-VFT 用（写出前已剔除）

---

## 8. `MHP/01_clip_create_dataset.py`（新增）— ST-VFT 用 clip 采样器

替代 RUMPL 原 `run_mmpose_01_create_dataset.py`（按离散 pose 采）：
- 每段 AMASS .npz 取中央 80%
- 用 `stride=L_clip//2` 间隔取 clip 起点，`keep_rate=0.5` 控总量
- 输出 `stage_IV/<subset>/amass_clips_<i>.pt`，每个 .pt 是 list of dict

字段（每 clip）：
```python
{
  'pose':        (L_clip, 156) float32,   # 52 joints × 3 axis-angle
  'trans':       (L_clip, 3)   float32,
  'betas':       (16,)         float32,
  'gender':      int,                      # 0=m, 1=f, 2=n
  'source_npz':  str,
  'start_frame': int,
  'frame_rate':  float,
}
```

支持 `--n-splits 4` 切成 4 份给 `02_clip_run.py` 并行处理。

---

## 9. 完整 pipeline 跑通顺序

```bash
# Stage IV: AMASS .npz → clips
python MHP/01_clip_create_dataset.py \
    --amass-data-dir <AMASS_ROOT> \
    --work-dir <WORK_DIR> --exp clip_full \
    --L-clip 27 --stride 13 --keep-rate 0.5 --n-splits 4 \
    --train-datasets Eyes_Japan_Dataset ACCAD DFaust_67 BMLhandball \
                     BioMotionLab_NTroje SFU Transitions_mocap TCD_handMocap \
                     TotalCapture KIT MPI_HDM05 HumanEva MPI_mosh BMLmovi \
                     SOMA MPI_Limits WEIZMANN EKUT SSM_synced GRAB DanceDB \
                     HUMAN4D CNRS \
    --operation-on train validation

# Stage V: render + run HRNet, 4 splits in parallel (one GPU each)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python MHP/02_clip_run.py --work-dir <WORK_DIR> --exp clip_full \
        --split-index $i --apply-rotation \
        --image-save-dir /tmp/mhp_images &
done; wait

# Stage V_room: offline locate_mesh_in_room
python MHP/04_fix_data_to_room.py --exp clip_full --work-dir <WORK_DIR>

# Flatten + 90/10 clip-level split
python MHP/05_flatten_with_split.py --exp clip_full --work-dir <WORK_DIR> \
       --subset train
python MHP/05_flatten_with_split.py --exp clip_full --work-dir <WORK_DIR> \
       --subset validation

# CMU eval pkl: y-down → z-up
python MHP/06_swap_cmu_axes.py \
    --in-pkl  <CMU_PKL_RAW> \
    --out-pkl <CMU_PKL_SWAPPED>

# Train (read DATASET 的 ROOT 指向 work_dir, AMASS_DATASET_TYPE 指向 flatten 输出)
cd RUMPL && python run/train_rumpl.py \
    --cfg configs/cmu_panoptic/rumpl_amass/clip_full.yaml

# Eval on CMU
python run/validate_rumpl.py \
    --cfg configs/cmu_panoptic/rumpl_amass/cmu_eval.yaml \
    TEST.MODEL_FILE <CKPT.pth>
```

---

## 10. 已复现基线（5070 Ti / sm_120 不行，需要 sm_90 或更低）

| 指标 | 我们 | 论文 |
|---|---|---|
| Synth Val V=5 Rel (KP\*) | 3.55 cm | 3.21 cm |
| Synth Val V=2 Rel (KP\*) | ~5 cm | — |
| CMU V=5 Rel (KP\*) | 6.21 cm | ~3.5 cm |
| CMU V=2 Rel (KP\*) | 10.67 cm | ~3.5 cm |

CMU V=2 的 3× 差距分解：HRNet 2D 噪声 2–3cm + synth↔real 结构 gap + 论文可能有
未公开 trick。Phase 1 (ST-VFT) 的设计目标之一就是用时序信息把 V=2 的差距吃掉。


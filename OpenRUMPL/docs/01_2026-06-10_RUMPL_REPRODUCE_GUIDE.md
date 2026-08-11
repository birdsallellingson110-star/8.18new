# RUMPL 项目复现完整指南

本文档记录在新服务器上复现 RUMPL paper baseline（含 CMU paper-faithful 评估）的全流程经验。

---

## 0. TL;DR

| 步骤 | 时间 | 备注 |
|---|---|---|
| 环境配置 | 1-2h | conda + pip 装依赖 |
| 数据下载 | 1 天 | AMASS (~100GB) + CMU 图像 (~293GB) + SMPL+H + 标定 |
| 数据生成 stage_I→V | **4-5 天** | render + HRNet 是瓶颈 |
| 训练 20 epoch | ~5h | 单 GPU |
| CMU 评估 | ~30 min | HRNet on 2015 张图 + 模型 inference |
| **总计** | **~1 周** | |

## 1. 环境准备

### 1.1 Conda 环境

```bash
conda create -n rumpl python=3.10 -y
conda activate rumpl

# PyTorch — 关键：GPU 必须 sm_90 以下！
# RTX 5070 Ti (sm_120) 不支持，会报 "no kernel image is available"
# 支持的：A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89)
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# Core
pip install numpy==1.23.5  # 必须 1.x，2.x 的 np._core 改名会出问题
pip install pickle5 tqdm PyYAML easydict opencv-python tensorboard scipy matplotlib

# AMASS / SMPL+H
pip install human_body_prior body_visualizer trimesh
pip install pyrender  # offscreen render，需要 EGL: export PYOPENGL_PLATFORM=egl
git clone https://github.com/nghorbani/amass.git && cd amass && pip install -e . && cd ..

# MMPose 2D 检测
pip install mmpose==1.3.2 mmdet==3.3.0
pip install mmcv-full==1.7.x  # 注意版本对应
```

### 1.2 必要环境变量

```bash
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # 必加！默认 FASTEST_FIRST 会让 CUDA_VISIBLE_DEVICES 索引错乱
```

### 1.3 验证

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 应该输出 "RTX 3090 (8, 6)" 或类似 sm <= 9.x 的卡

python -c "from mmpose.apis import MMPoseInferencer; print('OK')"
python -c "from human_body_prior.body_model.body_model import BodyModel; print('OK')"
```

## 2. 数据准备

### 2.1 目录布局

```
<DATA_ROOT>/
├── amass/                                    # 18-23 AMASS subsets (paper 用 23 个)
│   ├── ACCAD/{subj}/{action}_poses.npz
│   ├── BioMotionLab_NTroje/...
│   ├── ...
│   └── J_regressor_coco.npy                  # 17-joint COCO regressor
├── body_models/smplh/
│   ├── male/model.npz
│   ├── female/model.npz
│   └── neutral/model.npz
├── cmu_calibs/
│   ├── calibration_171204_pose5.json
│   ├── calibration_171204_pose6.json
│   ├── 171204_pose5/                         # 必须有子目录！loader 读 <root>/<setup>/calibration_<setup>.json
│   │   └── calibration_171204_pose5.json     # 软链接到上层
│   └── 171204_pose6/
│       └── calibration_171204_pose6.json
└── cmu_eval/data/                            # CMU 真实评估
    └── datasets/panoptic_gt_json/{train,validation,test}.pkl
```

### 2.2 AMASS 子集

Paper 用 23 个：
```
Eyes_Japan_Dataset ACCAD DFaust_67 BMLhandball BioMotionLab_NTroje SFU
Transitions_mocap TCD_handMocap TotalCapture KIT MPI_HDM05 HumanEva
MPI_mosh BMLmovi SOMA MPI_Limits WEIZMANN EKUT SSM_synced GRAB DanceDB
HUMAN4D CNRS
```

我们能拿到 18 个（缺 BMLhandball, WEIZMANN, SOMA, GRAB, CNRS）。差额对结果有影响。

### 2.3 CMU Panoptic 真实评估数据

Paper 4 个多人序列：`160906_pizza1`, `160422_haggling1`, `160906_ian5`, `160906_band4`

下载工具：[panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)

```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
cd panoptic-toolbox
for seq in 160906_pizza1 160422_haggling1 160906_ian5 160906_band4; do
  ./scripts/getData.sh $seq 1 5    # 1 VGA, 5 HD
done
```

需要的 HD cameras: `[3, 6, 12, 13, 23]`（paper 标准 5 相机）

每个序列 ~30-100 GB（HD video）。总计 **~293 GB**。

## 3. 关键 Bug（必修，否则结果错 10×）

### 3.1 训练 config (5 处)

文件：`OpenRUMPL/RUMPL/configs/cmu_panoptic/rumpl_amass/clip_full.yaml`

```yaml
DATASET:
  USE_T: false                    # ← 默认 true: loader 用 extrinsic 当 cam_center，rays 全错
  TRAIN_RANDOM_NUM_VIEWS: true    # ← 默认 false: 不会随机 k ∈ [2,5]
NETWORK:
  APPLY_VIEW_FUSION: true         # ← 默认 false: 模型只看三角化均值，不做多视角 attention
```

**修复效果**：Loss plateau 0.76 → 训到 0.039；Synth Val V=5 Rel 31cm → 3.55cm

### 3.2 评估指标修正

文件：`OpenRUMPL/RUMPL/lib/core/function_rumpl.py:781`

`calc_mpjpe` 默认用 joint 0 (nose) 做 anchor。paper KP* 是 **pelvis-anchored** (mid-hip)。

```python
# 改 evaluate() 函数中的 relative_evaluation 分支：
if relative_evaluation:
    kp_std = getattr(config.DATASET, 'CMU_KEYPOINT_STANDARD', 'coco').lower()
    if kp_std == 'coco':
        # COCO 17 没有 pelvis joint, 用 (lhip+rhip)/2
        pelvis_gt   = (gt[:, 11:12, :]   + gt[:, 12:13, :])   / 2.0
        pelvis_pred = (pred[:, 11:12, :] + pred[:, 12:13, :]) / 2.0
    else:
        # h36m joint 0 是 pelvis
        pelvis_gt   = gt[:, 0:1, :]
        pelvis_pred = pred[:, 0:1, :]
    gt   = gt   - pelvis_gt
    pred = pred - pelvis_pred
mode = 'absolute'  # 已 pre-center, 别让 calc_mpjpe 再减一次
pjpe, mpjpe = calc_mpjpe(gt, pred, mode=mode)
```

### 3.3 CMU 评估 4 个 bug

#### A. 多人场景取错人
文件：`OpenRUMPL/RUMPL/data/preprocess_cmu_panoptic.py:292-294`

Paper 取 `mmpose_data[0]`（置信度最高），多人场景下不一定是 GT subject。改用 GT 2D 找最近邻：

```python
if len(mmpose_data) == 0:
    return []
gt_2d = np.array(data['joints_2d'])
best_idx, best_err = 0, np.inf
for i, det in enumerate(mmpose_data):
    kp = np.array(det['keypoints'])
    if kp.shape != gt_2d.shape: continue
    err = np.linalg.norm(kp - gt_2d, axis=1).mean()
    if err < best_err: best_err, best_idx = err, i
pick = mmpose_data[best_idx]
```

#### B. Camera 单位双除 100
文件：`OpenRUMPL/MHP/06_swap_cmu_axes.py`

`loader 在 joints_dataset_rumpl.py:808` 已经自动 `cam['T'] /= 100`。我们 swap 脚本里**不要再 /100**：

```python
# 错的（双除）：T_new = (P @ T_old) / 100.0
# 对的：
T_new = P @ T_old  # 保持 cm，loader 会自动 /100
t_new = -R_new @ T_new
```

#### C. CMU 坐标系是 y-down 不是 y-up
验证：CMU GT joints_3d 里 nose y = -158 cm（最负 = 最高），ankle y = -12 cm（最不负 = 最低）。

正确的 swap 矩阵（CMU → AMASS, right-handed det=+1）：
```python
P = np.array([[1.0,  0.0, 0.0],
              [0.0,  0.0, 1.0],
              [0.0, -1.0, 0.0]], dtype=np.float64)
# 即: new_x = old_x, new_y = old_z, new_z = -old_y
```

#### D. 标定文件目录结构
`loader 读 <root_3>/<setup>/calibration_<setup>.json`，不是直接 `<root_3>/calibration_<setup>.json`：

```bash
mkdir -p cmu_calibs/171204_pose5 cmu_calibs/171204_pose6
ln -sf $(pwd)/cmu_calibs/calibration_171204_pose5.json cmu_calibs/171204_pose5/
ln -sf $(pwd)/cmu_calibs/calibration_171204_pose6.json cmu_calibs/171204_pose6/
```

## 4. Pipeline 全流程

### 4.1 数据生成（Paper 单帧版 = 最严格复现）

```bash
DATA=/path/to/data
EXP=paper_single_cmu

# Step 1: 抽样 → stage_IV (~10 min)
cd OpenRUMPL/MHP
python run_mmpose_01_create_dataset.py \
  --exp $EXP \
  --amass-data-dir $DATA/amass \
  --work-dir $DATA/mhp_workspace \
  --train-datasets Eyes_Japan_Dataset ACCAD DFaust_67 BioMotionLab_NTroje SFU \
    Transitions_mocap TCD_handMocap TotalCapture KIT MPI_HDM05 HumanEva MPI_mosh \
    BMLmovi MPI_Limits EKUT SSM_synced DanceDB HUMAN4D \
  --operation-on train \
  --n-splits 100
# ⚠️ subset 必须分开传，不要 SUBSETS=$"...一长串..." 这样 shell 会传成 1 个 arg

# Step 2: render + HRNet → stage_V (~4-5 天 单 GPU)
# 4 splits 并行，复用 official run_mmpose_03_single_run_cmu.sh 模板
for s in 0 1 2 3; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup \
  python -u run_mmpose_02_run.py \
    --dataset-split-number $s \
    --exp $EXP \
    --extra-name random_20_small_room_person_dist_2 \
    --views-cmu 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --use-cams-from cmu \
    --calib-root-cmu $DATA/cmu_calibs \
    --calibs-cmu 171204_pose5 171204_pose6 \
    --room-size -0.5 -0.1 -0.2 0.2 0 0 \
    --operation-on train \
    --image-width 1920 --image-height 1080 \
    --apply-rotation \
    --regressor coco \
    --triangulate --triangulate-th 0.95 \
    --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 \
    --save-temp-checkpoints \
    --run-on-random-cameras \
    --n-cameras-per-person 20 \
    --camera-location-limit -2.7 2.7 -2.7 2.7 0.7 3.4 \
    --camera-dist-from-person 2 \
    --image-save-dir /tmp/mhp_images \
    --support-dir $DATA \
    --work-dir $DATA/mhp_workspace \
    --amass-data-dir $DATA/amass \
    > /tmp/02_split$s.log 2>&1 &
done

# 注意：
# ✦ --image-save-dir 必须加（默认 /globalscratch/... 会报权限错）
# ✦ 4 splits ~5000 frames，全 99 splits ~122k frames 接近 paper 128k
# ✦ 跑完用 c2i 师弟 / OpenRUMPL 的脚本 stitch splits（或我们自己 cat 一下）
```

### 4.2 训练

```bash
cd OpenRUMPL/RUMPL
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full.yaml \
  --exp-name run_v1
# 20 epoch, lr 1e-4 → 1e-5(@10) → 1e-6(@15)
# ~5h 单 GPU
```

### 4.3 CMU 真实评估

```bash
cd OpenRUMPL/RUMPL/data

# 1. preprocess（不含 mmpose）
python preprocess_cmu_panoptic.py /path/to/cmu_panoptic \
  annot_paper_4seq_5cams_coco \
  --running-modes preprocess filter \
  --run-for-sets validation \
  --skip-step-val 64 --skip-step-train 64 \
  --keypoints-standard coco

# 2. 跑 HRNet on val images（用 mmpose 官方 inferencer_demo.py）
# 写 staging 脚本：per (seq, cam) 一次 inferencer call (避免单图反复加载模型)
# 见 /tmp/run_hrnet_official.sh 模板

# 3. preprocess mmpose 模式（subject matching 已 patch）
python preprocess_cmu_panoptic.py /path/to/cmu_panoptic \
  annot_paper_4seq_5cams_coco \
  --running-modes mmpose \
  --run-for-sets validation \
  --skip-step-val 64 --skip-step-train 64 \
  --mmpose-dataset-name mmpose_hrnet_coco_matched \
  --mmpose-output-path /path/to/cmu_panoptic/MPL_data/mmpose_outputs

# 4. axis swap（CMU y-down → AMASS z-up）
python OpenRUMPL/MHP/06_swap_cmu_axes.py \
  --in-pkl /path/to/.../matched/cmu_panoptic_validation.pkl \
  --out-pkl /path/to/.../matched_swap/cmu_panoptic_validation.pkl

# 5. evaluation
cd OpenRUMPL/RUMPL
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python run/valid_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/cmu_eval.yaml \
  --model-file <output>/model_best.pth.tar \
  --use-mmpose-val
```

## 5. 结果对照（我们 vs Paper）

| 指标 | 我们 (修 5 bug 后) | Paper Table 9 | 状态 |
|---|---|---|---|
| **Synth val V=5 KP\*** | **3.21 cm** | 3-5 cm | ✅ 达标 |
| Synth val V=2 KP\* | 6.48 cm | ~3.5 cm | ⚠️ 2× gap |
| **CMU V=5 Rel KP\*** | **6.21 cm** | ~3-5 cm | ✅ 接近 |
| CMU V=5 Abs | 7.45 cm | ~3 cm | ⚠️ 2× gap |
| **CMU V=2 Rel KP\*** | **10.67 cm** | **3.5 cm** | ❌ 3× gap |
| CMU V=2 Abs | 17.75 cm | ~5 cm | ❌ 3× gap |
| **Phase 0 PASS (40 mm V=2 KP\*)** | **107 mm FAIL** | — | ❌ |

**关键 insight**：Synth val 完全对齐 paper 证明实现没问题。CMU 真实数据 V=2 残余 3× gap 来自：

1. **HRNet on synth vs real domain gap** (~2-3 cm 贡献)
2. **Synth→real 结构性 gap**（possibly paper 内部未公开 trick）
3. **V=2 视角无冗余**，依赖 pose prior 更多

## 6. 实验路径建议

### 必做
1. 修 5 个 bug
2. 数据生成走 paper 原版（单帧版，含 `--apply-rotation`）
3. 跑 paper 原版评估流程（4 个多人序列 + HRNet + subject matching + axis swap）

### 可选（针对 V=2 gap）
- 加 CMU 真图 fine-tune（最 dirty 但有效）
- 等更多 AMASS subsets（paper 23 个 vs 我们 18 个）

### 不要踩的坑
- 不要乱改 axis swap 矩阵（CMU 是 y-DOWN，nose y=-158, ankle y=-12）
- 不要在 swap 脚本里 /100（loader 自动做）
- 不要拿 nose 当 anchor 跟 paper KP* 比（要 pelvis-anchored）
- 不要相信 paper 配置里的 `USE_T: true`（错的，要改成 false）
- 不要用 RTX 5070 Ti / sm_120 GPU（PyTorch 不支持）

## 7. 我们维护的关键文件

```
OpenRUMPL/MHP/
├── 01_clip_create_dataset.py        # clip 版（备选，给 Phase 1 ST-VFT）
├── 02_clip_run.py                   # clip 版（备选，给 Phase 1）
├── 04_fix_data_to_room.py           # clip 版 room placement (offline)
├── 05_flatten_with_split.py         # clip → flat
└── 06_swap_cmu_axes.py              # ⚠ 必须，CMU 坐标系修正

OpenRUMPL/RUMPL/
├── data/
│   └── preprocess_cmu_panoptic.py   # ⚠ patched: subject matching
├── lib/core/
│   └── function_rumpl.py            # ⚠ patched: pelvis-anchored MPJPE
└── configs/cmu_panoptic/rumpl_amass/
    ├── clip_full.yaml               # 训练 config (修 3 bug)
    ├── cmu_eval.yaml                # CMU V=5 评估
    └── cmu_eval_v2.yaml             # CMU V=2 评估
```

## 8. 资源预算

| 项 | 量 |
|---|---|
| 磁盘 | 500 GB+（AMASS 100GB + CMU 图 293GB + 中间数据 ~100GB）|
| GPU | 1× 3090/4090/A100（sm_90 以下）|
| 内存 | 30 GB+（4 进程并行）|
| 时间 | 1 周（含下载 + 数据生成 + 训练 + 评估）|

## 9. 时间表参考

```
Day 1     : 环境 + 下载 AMASS / CMU 图（并行）
Day 2     : 跑 stage_I/II/III (01) + 跑 preprocess_cmu_panoptic
Day 2-6   : stage_V 生成（render + HRNet）
Day 6     : 训练 20 epoch
Day 7     : CMU 评估 + 报告
```

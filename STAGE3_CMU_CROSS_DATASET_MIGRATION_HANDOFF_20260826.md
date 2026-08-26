# Stage-3 迁移交接：真实 CMU 训练与 CMU→H36M 零样本测试

> 生成时间：2026-08-26。目标是把项目迁移到另一台电脑，继续第三阶段实验。
> 本文优先级高于旧计划中尚未完成的 CMU 描述；Stage-1/2 最终数值仍以
> `SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.md` 为准。

## 0. 审计结论

结论不是“clone 后立即可以正式训练”，而是：

- **核心模型代码已在 `/home/lixiaob/cjy` 仓库内**：canonical RUMPL、Global
  Joint-Query、E2 22-candidate scorer、H18 temporal、CMU dataset loader 和通用训练器均在；
- **Python 环境可重建**：根目录 `requirements.txt` 固定 Python 3.10、Torch
  2.1.0+cu118、MMCV 2.1.0 等版本；不上传 5.5 GB 虚拟环境二进制；
- **正式真实 CMU 数据不在 Git 仓库中，且本机下载尚未完成**；
- **Stage-3 专用一键流水线尚未冻结**：现有 CMU 配置包含历史服务器绝对路径，原始
  preprocessing 文件还保留多种历史协议，不能不审计就直接运行；
- 现有 `export_cmu_fourview_hypotheses_20260824.py` 和
  `evaluate_e2_cross_dataset_20260824.py` 是 **H36M-trained → CMU** 的早期诊断，
  不是论文要求的正式 **CMU-trained → H36M** 方向；
- 新电脑第一项工作应是完成真实 CMU 数据、冻结专用 YAML/launcher，并补齐
  CMU-trained checkpoint 的 H36M exporter/evaluator，之后才开始长时间训练。

## 1. 已完成的前两阶段

### Stage-1 H36M clean

完整 T=9：

| Input | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet-152 | 29.416 | 21.020 | 19.281 |
| HRNet-W32 | 37.392 | 29.501 | 27.713 |

### Stage-2 Human3.6M-Occ

| Input | Occ-2 V2/V3/V4 | Occ-3 V2/V3/V4 |
|---|---:|---:|
| ResNet-152 T=9 | 45.278/25.652/21.349 | 51.111/27.862/22.653 |
| HRNet-W32 T=9 | 53.966/32.204/28.705 | 58.852/33.970/29.695 |

完整模型链、checkpoint SHA256 和消融见：

- `SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.md`
- `SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.json`

## 2. Stage-3 要证明什么

第三阶段分成两个严格区分的实验：

1. **CMU→CMU 同域**：真实 CMU Panoptic 单人序列训练，在保留测试序列/测试相机上
   报告绝对 MPJPE 和 2/4/5/6/8 视角曲线；
2. **CMU→H36M 零样本**：冻结 CMU 训练的 generator/E2/H18，不在 H36M 微调，直接
   测试 H36M S9/S11，只在两数据集语义一致的肩、肘、腕、膝、踝 10 个关节上报告
   action-equal absolute MPJPE。

GBT 公开参考：CMU 四视角 HRNet 为 `17.2 mm`；CMU→H36M matched-joint 平均为
`38.9 mm`。这些是 reported values，不是本项目代码的严格复现。

## 3. 正式数据协议

### 3.1 序列划分

训练：

- `171026_pose1`
- `171026_pose2`
- `171026_pose3`
- `171204_pose1`
- `171204_pose2`
- `171204_pose3`
- `171204_pose4`

测试：

- `171204_pose5`
- `171204_pose6`

禁止把 pose5/pose6 加入训练。

### 3.2 相机协议不能混用

至少记录两套协议：

- **GBT reported-value comparison**：测试 HD cameras `2,13,10,19`；若严格沿用
  GBT 描述，训练使用其余 27 个 HD cameras，因此需要全部 31 个 HD camera 数据；
- **项目扩展曲线**：使用已选择的 20-camera pool，报告 V2/V4/V5/V6/V8；常用
  standard-five 为 `3,6,12,13,23`。

现有 downloader 的 20-camera pool 包含 GBT 四个测试相机，但不包含全部其余 27 个
训练相机。因此可以做 20-camera camera-holdout 实验，但不能把它称为严格 GBT 31-camera
训练协议。

### 3.3 本机数据当前并不完整

本机路径：`/mnt/data/cjydata/cmu_singleperson_real20`。

2026-08-26 审计到的视频数：

| Sequence | downloaded/linked videos |
|---|---:|
| 171026_pose1 | 0 |
| 171026_pose2 | 3 |
| 171026_pose3 | 5 |
| 171204_pose1 | 0 |
| 171204_pose2 | 0 |
| 171204_pose3 | 5 |
| 171204_pose4 | 0 |
| 171204_pose5 | 5 |
| 171204_pose6 | 5 |

没有 `TRAIN_STANDARD5_COMPLETE`、`TRAIN_REAL20_COMPLETE` 或 `DOWNLOAD_COMPLETE`。
因此当前 9.7 GB 目录不能作为正式训练集。

`/mnt/data/cjydata/mhp_workspace/paper_single_cmu` 是 AMASS/SMPL 投影到 CMU 相机的
合成数据，不得在论文中写成 real CMU train。

## 4. 环境迁移

GitHub 不应保存整个 `rumpl_venv310`。在新电脑执行：

```bash
git clone https://github.com/birdsallellingson110-star/8.18new.git
cd 8.18new
python3.10 -m venv rumpl_venv310
rumpl_venv310/bin/pip install -U pip
rumpl_venv310/bin/pip install -r requirements.txt
```

若 MMCV wheel 安装失败：

```bash
rumpl_venv310/bin/pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

随后运行只读预检：

```bash
bash OpenRUMPL_baseline_audit/preflight_stage3_cmu_cross_dataset_20260826.sh --code-only
```

环境参考：Python 3.10.20、PyTorch 2.1.0+cu118、torchvision 0.16.0+cu118、
MMCV 2.1.0、MMPose 1.3.2、NumPy 1.26.4。

## 5. 代码清单

### 5.1 模型与通用训练

- `OpenRUMPL/RUMPL/run/train_rumpl.py`
- `OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- `OpenRUMPL/RUMPL/lib/models/temporal_gbt_rumpl.py`
- `OpenRUMPL/RUMPL/lib/core/function_rumpl.py`
- `OpenRUMPL/RUMPL/lib/dataset/gbt_ray_augmentation.py`

### 5.2 CMU 数据

- `download_cmu_singleperson_real20_20260727.sh`
- `CMU_SINGLEPERSON_REAL20_PROTOCOL_20260727.md`
- `OpenRUMPL/RUMPL/data/preprocess_cmu_panoptic.py`
- `OpenRUMPL/RUMPL/lib/dataset/multiview_cmu_panoptic_rumpl.py`
- `OpenRUMPL/MHP/run_hrnet_cmu.py`
- `OpenRUMPL_baseline_audit/convert_cmu_coco_to_h36m_virtual_20260824.py`

注意：`run_hrnet_cmu.py` 仍有 `/mnt/data/...` 默认值；可以通过 CLI `--root/--out`
覆盖。`preprocess_cmu_panoptic.py` 同时保留历史 dataset settings，正式运行前必须由新的
Stage-3 launcher 显式指定七个 train pose 序列、两个 test pose 序列和 camera split，
不能依赖文件顶部的历史全局变量。

### 5.3 E2/H18

- `OpenRUMPL_baseline_audit/train_e2_camera_independent_22c_20260824.py`
- `OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py`
- `OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py`
- `OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py`
- `OpenRUMPL_baseline_audit/build_e2_temporal_uncertainty_20260825.py`
- `OpenRUMPL_baseline_audit/test_h18_generalization_time_20260825.py`

这些训练器可复用，但 CMU 正式 launcher 必须把 train holdout、test sequence 和 camera
holdout 分开，checkpoint 只能由 CMU train holdout 选择。

### 5.4 现有跨域诊断

- `OpenRUMPL_baseline_audit/export_cmu_fourview_hypotheses_20260824.py`
- `OpenRUMPL_baseline_audit/evaluate_e2_cross_dataset_20260824.py`
- `OpenRUMPL_baseline_audit/eval_gbt_aligned_cmu_metrics.py`
- `OpenRUMPL_baseline_audit/eval_cmu_viewgroups.py`

前两个脚本的当前方向是 H36M-trained → CMU，仅可作为 loader/camera/skeleton sanity
参考。正式论文方向 CMU-trained → H36M 仍需对应 exporter、matched-joint evaluator 和
结果 collector。

## 6. 新电脑上需要补齐的 Stage-3 专用代码

核心模块存在，但以下 orchestration 尚未冻结：

1. 参数化的 real-CMU preprocessing launcher，显式锁定 train/test sequences、camera
   split、抽帧和输出 PKL 名称；
2. canonical CMU generator 的正式 YAML 与双 seed launcher；
3. CMU 22-candidate cache 生成、E2 双 seed 训练与选择 launcher；
4. CMU dense centered T=9 cache、H18 训练与匹配 T=1/T=9 evaluator；
5. CMU-trained generator/E2/H18 → H36M S9/S11 的 exporter/evaluator；
6. CMU V2/V4/V5/V6/V8 与 CMU→H36M action-wise 最终 collector。

这些文件应在数据 schema 和 sanity checks 冻结后创建。不要直接把旧 H36M launcher 改几个
路径就跑长训练。

## 7. 必须按顺序执行

1. clone 仓库并重建环境；
2. 选择 GBT-31-camera 或项目 20-camera 协议，完成对应真实数据下载；
3. 解压 3D 标签、读取 calibration、抽取同步 HD frames；
4. 用与论文主线一致的冻结 YOLOX-X + HRNet-W32 生成 2D 坐标/置信度；ResNet-152
   作为第二输入分支后做；
5. 用确定性 COCO17→H36M17 adapter 统一骨架；
6. 检查样本数、同步、相机 ID、单位、world↔camera、左右关节和有效框；
7. 运行 GT2D triangulation sanity；理想误差应接近标定/量化误差，而不是几十毫米；
8. 小样本 overfit generator；
9. 做刚体变换等变性审计；
10. 正式训练 CMU canonical generator 两 seeds；
11. 训练 E2 两 seeds，再训练 selected H18；
12. CMU 同域报告全部 view groups；
13. 冻结全部 CMU 权重，零样本测试 H36M matched 10 joints；
14. 保存命令、commit、环境、PKL hash、checkpoint hash、每相机组合和 action-wise JSON。

## 8. 新对话首条指令

在新电脑打开仓库后，可直接告诉 Codex：

> 完整阅读 `STAGE3_CMU_CROSS_DATASET_MIGRATION_HANDOFF_20260826.md`、
> `SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.md` 和
> `conversation_exports/2026-08-25_to_2026-08-26_user_visible_dialogue.md`。
> 先运行 Stage-3 `--code-only` 预检，再只读审计真实 CMU 数据完整性。不要把
> `paper_single_cmu` 当真实训练集，不要运行当前 H36M→CMU 诊断作为正式跨数据集结果。
> 先冻结 real-CMU preprocessing schema、camera split 和 GT2D sanity，再创建正式
> canonical generator→E2→H18→CMU→H36M 流水线。

## 9. 不随 GitHub 上传的资产

- `rumpl_venv310/`：5.5 GB，使用 requirements 重建；
- `/mnt/data/cjydata/...`：真实/合成数据，另行复制或重新下载；
- `/mnt/data/cjyoutput/...`：checkpoint 和实验输出，另行复制；
- `backup_critical/baseline_conf_model_best.pth.tar`：454 MB，Git ignore；
- 其他 `.pth/.pt/.ckpt`：Git ignore。

## 10. 对话导出

用户可见对话已导出到：

`conversation_exports/2026-08-25_to_2026-08-26_user_visible_dialogue.md`

导出只包含用户消息和用户可见助手回复，不含 system/developer instructions、内部推理、
工具调用和工具输出。可用根目录 `export_codex_dialogue.py` 从原始 Codex JSONL 重新导出。

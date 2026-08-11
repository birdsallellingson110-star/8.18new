# RUMPL 项目完整 Chat 交接总结

更新时间：2026-07-26  
工作目录：`/home/lixiaob/cjy`  
当前主仓：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit`

本文是当前对话的单文件交接入口，覆盖严格基线、技术路线、关键结果、失败实验、数据、代码、论文资产、口径限制和下一步任务。接手后应先读本文，再按文中的路径查看原始日志与 JSON。

---

## 1. 项目目标

目标是在 RUMPL 基础上降低 CMU Panoptic 不同相机数量和相机组合下的 3D MPJPE，重点是少视角 V2，同时要求 V3/V4/V5 不退化，最终形成：

1. 一个训练完成即可直接测试的单模型；
2. 一套统一支持 V2-V5 的训练协议，而不是分别为每个视角数训练模型；
3. CMU 全相机组合平均、最佳组合、逐组合稳定性、KP* 和 All-17 的完整报告；
4. H36M、遮挡、噪声或时序等泛化实验；
5. 可以解释的模块、注意力/可靠性可视化和完整消融；
6. 一篇技术路线清楚、结果严谨的论文。

---

## 2. 最重要的口径纠正

### 2.1 7 月 22 日以前的旧 baseline 不正确

旧流水线存在两类关键偏差：

- CMU 抽帧协议错误：旧流程先按 frame id `%64` 抽图，官方流程先 grouping，再 `[::64]`；两套测试图像只有约 `65/3500` 重合。
- 模型配置偏离官方：包括 DIM、intersection、屏幕归一化等约 10 项差异。

旧 V2 baseline 约 `46.9 mm`，严格审计后的 R5 是 `30.885 mm`。因此以下旧结论不能作为相对正确 RUMPL 的论文结果：

- `hard-view + legw0.9` 的 V2 `10/10` 全降、平均 `-5.97 mm`；
- general/hard-view 蒸馏的大幅提升；
- prediction ensemble 的大幅提升；
- 旧 GBT/课程学习/token removal 相对 46.9 mm baseline 的数毫米收益。

这些实验可用于说明探索历史，但不能放进最终主结果表。

严格复现审计：

- `/home/lixiaob/cjy/2026-07-22_rumpl_baseline_reproduction_audit.md`

### 2.2 官方代码并非百分之百完整开源

论文 YAML 引用了若干未公开类。当前 R5 是基于公开代码、论文配置和三角化结果进行的 official-like 严格重建。三角化审计与论文表格仅约 `0.14 mm` 差异，因此当前复现口径是可信的，但论文中应写 official-like reproduction，不应声称拿到了所有未公开实现。

---

## 3. 当前硬基线 R5

### 3.1 代码与权重

- 仓库：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit`
- Python：`/home/lixiaob/cjy/rumpl_venv310/bin/python`
- 配置：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml`
- R5 checkpoint：`/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar`
- 标准评测：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_exact_multiview_20260723.sh`
- 汇总脚本：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/summarize_cmu_predictions.py`

### 3.2 严格 CMU clean 全组合平均

单位均为 Absolute MPJPE mm，输入为 MMPose HRNet 2D。

| Views | All-17 | KP* | 组合数 |
|---:|---:|---:|---:|
| V2 | 30.885 | 35.506 | 10 |
| V3 | 23.039 | 25.159 | 10 |
| V4 | 20.213 | 21.698 | 5 |
| V5 | 18.746 | 20.091 | 1 |

已知最佳 All-17 组合：

- V2：`(3,23)`，约 `25.218 mm`；
- V3：`(3,6,23)`，约 `20.460 mm`；
- V4：`(3,6,12,23)`，约 `19.316 mm`；
- V5：`18.746 mm`。

结果目录：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/multiview_model_best_eval/`
- V2 clean 也可见：`/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval/R5_v2_occ0.0_summary.json`

### 3.3 R5 泛化能力的真正来源

RUMPL 的泛化不能简化为“ray + Transformer”。关键是：

- AMASS 大规模合成姿态；
- 每个样本随机生成/抽取相机配置；
- 可变视角数训练；
- 输入只依赖 2D、confidence 和相机 ray；
- 测试时允许未见过的相机数量和布局。

后续融合图像方法时，不能用固定 Panoptic 相机端到端训练替代这套协议，否则会破坏 RUMPL 的核心泛化卖点。

---

## 4. RUMPL 当前前向与主要结构缺陷

### 4.1 主要文件

- 主模型：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/models/multiview_rumpl.py`
- CMU 数据：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/dataset/multiview_cmu_panoptic_rumpl.py`
- AMASS 随机相机：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/dataset/multiview_amass_rumpl.py`
- ray 生成：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/dataset/joints_dataset_rumpl.py`
- 几何函数：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/utils/calib.py`
- 训练循环：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/core/function_rumpl.py`
- 训练入口：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/train_rumpl.py`
- 验证入口：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/valid_rumpl.py`

### 4.2 前向过程

```text
2D keypoint + confidence + camera
             ↓
direction ray + camera origin/intersection + confidence
             ↓
VFT：每个 joint 独立做跨视角融合
             ↓
只取 fusion token，丢弃各 view token
             ↓
PFT：跨关节融合
             ↓
LayerNorm + Linear(D→3)
             ↓
绝对 3D 坐标
```

### 4.3 可改进的粗糙点

1. VFT 最终只读 fusion token，精炼后的 K 个视角 token 全部丢弃，且没有逐视角监督。
2. 输出头直接回归绝对坐标，没有利用已经计算出的几何解。
3. confidence 只经过一次浅层 Linear 后 concat，之后没有逐层校准或调制。
4. Miss20 是独立伯努利丢关节，与真实成组遮挡不一致。
5. 视角数常按 batch 采样，优化粒度较粗。
6. 官方 PFT 最后一个 block 被重复执行；修正实验表明这不是主要精度瓶颈。

重要数值限制：dataset 的 V2 skew-line midpoint 可用于审计；旧的 V3+ n-line midpoint 会出现几十米数值爆炸，不能直接作为多视角 anchor。V3+ 应使用稳定的 confidence-weighted ray least squares。

---

## 5. Geometry-Biased Transformer 与偏置路线

老师要求包括：confidence bias、geometry bias、训练噪声、将注意力绑定真实 3D 几何、观察 attention map。

已经实现过：

- confidence attention bias；
- ray-distance geometry bias；
- fusion-token geometry；
- learnable bias scale；
- oracle/learned reliability；
- attention 保存与可视化；
- global joint-view attention；
- token removal 和随机 attention mask。

严格 R5 上的结论：

- G0/G1/G2/G3 基本在 `±0.3 mm` 范围波动，没有稳定收益。
- G4 最好也只有 V2 约 `-0.011 mm`，没有论文价值。
- 置信度和 ray geometry 已经存在于 RUMPL 输入 token；简单在 logits 上再加一次属于信息冗余。
- GBT 原论文使用 global joint-view-time token、时序和含噪协议；只移植公式不能称完整复现。

严格 G0-G3 汇总：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/phase1_gbt_multiview_eval_summary_20260724.txt`

原论文：

- `/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`

---

## 6. Global Joint-View Attention

动机：原 VFT 在各 joint 内独立融合，跨关节上下文要等视角已经压缩后才进入 PFT。实现是在 VFT 前把 `J×V` token 展平做 global self-attention，再以残差形式送回原 VFT。

### 6.1 J0e：global-JV + confidence/geometry bias

相对 R5 All-17：

- V2 `-0.068 mm`，但 KP* `+0.106 mm`；
- V3 `-0.179 mm`；
- V4 `-0.177 mm`；
- V5 `-0.110 mm`。

### 6.2 J1e：global-JV，无 bias

- V2 `+0.165 mm`，KP* `+0.396 mm`；
- V3 `-0.155 mm`；
- V4 `-0.305 mm`；
- V5 `-0.334 mm`。

说明 global joint-view interaction 对 V3-V5 有小收益，但固定残差会伤 V2；固定 bias 在多视角下反而约束过强。

### 6.3 J2：视角数自适应 gate

J2 用同一个模型令 gate 随 V2→V5 从 0.05 连续增长到 0.12。

| Views | J2 | 相对 R5 | 改善组合 |
|---:|---:|---:|---:|
| V2 | 31.036 | +0.151 | 6/10 |
| V3 | 22.983 | -0.056 | 5/10 |
| V4 | 20.146 | -0.067 | 3/5 |
| V5 | 18.753 | +0.007 | 0/1 |

结论：没有实现统一提升，停止继续扫 gate。

文档与结果：

- `/home/lixiaob/cjy/2026-07-22_global_joint_view_ablation_manifest.md`
- `/home/lixiaob/cjy/2026-07-22_gated_global_joint_view_experiment_manifest.md`
- `/mnt/data/cjyoutput/baseline_reaudit_20260722/phase0_j2_s1_eval_summary_20260724.txt`

---

## 7. 时序路线

### 7.1 数据与泄漏修复

AMASS 时序数据：

- `/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl`

共 300 个 PKL、3000 clips、2214 条 AMASS source。491 条 source 被切成多个 clip。原随机 clip 切分会让同一 source 同时进入训练和验证；后续已改成按 `source_npz` 分组切分：2718 train、202 validation、149 validation sources、source overlap 为 0。

真实 CMU 时序标注：

- 正确坐标版本：`/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched/cmu_panoptic_validation.pkl`
- 禁用版本：`matched_swapv3` 改了坐标系，不能和 R5 直接比较。

### 7.2 T2：绝对 feature post-VFT temporal refinement

- 冻结 R5；
- 9 帧；
- per-joint temporal MHSA；
- AMASS 验证上看似大幅提升；
- 真实 CMU 上 V2/V3/V4/V5 全部退化约 `+6.26/+6.40/+6.41/+6.21 mm`，0 个组合改善。

根因：模型利用绝对中心特征学成了 AMASS 域校正器，而不是可迁移运动模型。

### 7.3 T3：motion-difference temporal refinement

- 输入改成 `F(t)-F(center)`；
- motion-energy gate；
- 静态序列严格退回 R5；
- residual penalty 0.05。

CMU V2/V3/V4/V5 平均变化约 `+0.119/+0.097/+0.096/+0.096 mm`。它成功消除了 T2 的 6 mm 域偏移，但仍无 clean MPJPE 收益。

结论：时序 clean-accuracy 主线停止。若以后使用，只能作为 jitter/noise robustness 独立指标，不能再拿 AMASS 内部验证宣称有效。

代码与记录：

- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/models/exact_temporal_rumpl.py`
- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/dataset/exact_temporal_clip.py`
- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/train_exact_temporal_rumpl.py`
- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/eval_exact_temporal_cmu.py`
- `/mnt/data/cjyoutput/temporal_exact_20260723/`
- `/home/lixiaob/cjy/2026-07-23_temporal_and_adaptive_global_jv_experiments.md`

---

## 8. 骨架对称先验 S1

按 SkelSplat 思路约束左右上臂、前臂、大腿、小腿长度相等，权重 0.5。

| Views | S1 | 相对 R5 |
|---:|---:|---:|
| V2 | 30.545 | -0.340 |
| V3 | 23.082 | +0.043 |
| V4 | 20.359 | +0.146 |
| V5 | 19.044 | +0.297 |

结论：人工先验在 V2 欠约束时略有帮助，视角充分后与观测冲突。不能作为统一 V2-V5 主模块，可保留为 V2 结构先验消融。

---

## 9. 蒸馏路线重新审计

旧 `hard-view + legw0.9 = -5.97 mm` 属于错误 baseline，作废。

在严格 R5 上重新训练：

### DS0 general full-to-sparse distillation

- V2：`30.285/34.920` All/KP*，相对 R5约 `-0.60/-0.59 mm`；
- V3：`+0.58/+1.05 mm`；
- V4：`+0.81/+1.42 mm`；
- V5：`+0.91/+1.62 mm`。

### DS1 hard-view + legw0.9

- V2 约 `-1.15 mm`；
- V3/V4/V5 约 `+0.16/+0.32/+0.36 mm`。

结论：多视角 teacher 对 V2 有真实但有限的帮助，却损害多视角 clean。老师质疑“5 视角 teacher 蒸馏 2 视角”的合理性是成立的，除非重新设计可靠性门控的 feature/relation/geometry distillation，否则不应作为唯一主故事。

结果目录：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/distill_r5_eval/`

---

## 10. 遮挡与鲁棒性路线

### 10.1 A2：当前遮挡轴最有效方案

配置：structural occlusion level 0.4 + occluded-joint soft loss boost 2.0。

| 设置 | R5 | A2 | A2-R5 |
|---|---:|---:|---:|
| V2 clean | 30.89 | 30.43 | -0.45 |
| V2 occ0.3 | 34.49 | 33.38 | -1.10 |
| V2 occ0.6 | 38.06 | 36.49 | -1.57 |
| V3 clean | 23.04 | 23.15 | +0.12 |
| V4 clean | 20.21 | 约20.67 | 约+0.46 |
| V5 clean | 18.75 | 约19.52 | 约+0.77 |

结论：A2 在少视角遮挡协议上成立，是当前最可信的鲁棒性结果；但 V4/V5 clean 掉点，不能作为统一 clean baseline。

结果目录：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval/`

### 10.2 AdaFuse ViewWeightNet B1

- B1 单独使用：clean 和 occlusion 均未改善；
- B1+A2：接近 A2，但未超过；
- 弱遮挡 B1occ02：近中性。

结论：简单 view weighting 不是瓶颈，停止该方向。

### 10.3 无图像 2D refine

实现：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib/utils/refine_2d_multiview.py`

- soft_fill：V2 clean/occ0.6、V3 occ0.6 均偏负；
- fill_only：clean 能保持不变，但遮挡仍略差。

原因：V2 遮挡后通常只剩一个有效视角，无法三角化；V3+ 重投影的错误方向会污染输入。MVGFormer 的 AM 有效依赖图像特征，不能只搬无图几何 fill。

---

## 11. 其他模块结果与失败原因

### 11.1 KPA / SemGCN

- 变体：KPA + A2；
- V2 约 `32.84 mm`，比 R5 差约 2 mm；
- 全面失败。

脚本：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_kpa_a2_20260725.sh`

### 11.2 MH3 多假设 fusion tokens

- 3 个 fusion token + Conv1d 融合；
- V2 约 `31.19 mm`；
- V5 约 `19.19 mm`；
- 未超过 R5。

脚本：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_mh3_a2_20260725.sh`

### 11.3 D3-PCT / UniCodebook-lite

- EMA codebook + DCSA-lite；
- PCT 版本发生 inplace backward 冲突并终止；
- 没有有效最终结果，修复 anomaly 前不要直接重训。

日志：`/mnt/data/cjyoutput/baseline_reaudit_20260722/D3PCT_a2_seed0_20260725.log`

### 11.4 Token removal / mask

- 10% 全程 removal 明显帮助旧坏基线 V2，但伤 V3-V5；
- 20% 更差；
- 前 5 轮 removal 比全程平衡，但严格 R5 审计后不再是主线。

### 11.5 重投影和 ray loss

- pixel reprojection loss：透视除法在 `z≈0` 时梯度爆炸；
- 3D point-to-ray loss：数值稳定，但会强制拟合噪声 ray，没有收益。

### 11.6 D1 triangulation anchor

旧实现：

```python
prediction = absolute_head + gate * absolute_anchor
```

这是两套绝对坐标相加，不是真正残差。V2 退化到约 `33.19 mm`，该实验不能用于否定几何 anchor。

错误位置约在 `multiview_rumpl.py` 的 tri-anchor 输出分支。正确形式应为：

```python
prediction = anchor + gate * residual
```

或：

```python
prediction = (1-gate) * anchor + gate * network_prediction
```

---

## 12. GT-2D 上限实验

脚本：

- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_gt2d_upperbound_20260725.sh`

结果：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/gt2d_upperbound/COMPARE_mmpose_vs_gt2d.json`

| Views | HRNet All/KP* | GT-2D All/KP* | All 改善 |
|---:|---:|---:|---:|
| V2 | 30.89 / 35.51 | 16.49 / 15.85 | -14.4 |
| V3 | 23.04 / 25.16 | 14.96 / 14.46 | -8.1 |
| V4 | 20.21 / 21.70 | 14.12 / 13.83 | -6.1 |
| V5 | 18.75 / 20.09 | 13.58 / 13.49 | -5.2 |

解释：V2 约一半误差来自 2D 检测，但 GT-2D 下仍有 16.5 mm，说明 fusion/lifting 也存在明显误差。

---

## 13. 分阶段几何误差审计

脚本：

- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/run/audit_stage_errors_gt2d.py`

结果：

- `/mnt/data/cjyoutput/baseline_reaudit_20260722/gt2d_upperbound/stage_audit/COMPARE_stages.json`

V2 All-17：

| 阶段 | HRNet 2D | GT-2D |
|---|---:|---:|
| ray 到 GT 距离 | 31.2 | 5.9 |
| ray oracle | 25.2 | 5.3 |
| V2 skew midpoint | 43.5 | 10.5 |
| RUMPL prediction | 30.9 | 16.5 |

核心发现：

- 坏 2D 时，几何 midpoint 很差，RUMPL 能从 43.5 修到 30.9；
- 好 2D 时，几何已有 10.5，RUMPL 却偏到 16.5；
- GT-2D 下网络只在约 15.5% 关节上优于 midpoint；
- 最大退化关节包括 lhip、lelb、rhip、nose，不只是脚踝等检测困难关节。

这说明 RUMPL 有“双重人格”：需要在几何可靠时相信几何，在几何不可靠时用网络先验修正。

---

## 14. 失败原因总分类

### F1：错误 baseline 造成虚假大提升

坏配置容易被任何模块修好；所有旧 46.9 mm 口径结果必须与 R5 分开。

### F2：重复信息

ray、相机中心、confidence 已在 token 中；简单 bias、view weighting 和 reliability 重复输入，没有新增监督。

### F3：先验与多视角观测冲突

对称 loss、mask、token removal 可帮助 V2，却会伤 V3-V5。

### F4：合成域到真实域偏移

T2 在 AMASS 上巨大提升、CMU 上退化 6 mm；蒸馏 teacher 也可能传递合成域偏差。

### F5：数值不稳定

透视重投影、旧 V3+ midpoint 和错误 anchor 公式都存在数值或语义问题。

### N1：固定 HRNet 2D 下 lifter 已接近软天花板

继续堆 attention/KPA/MH3 很难在 clean V2 获得稳定 1-2 mm。

### N2：无图像不能复制 appearance refinement

MVGFormer AM 的关键是图像特征，不是几何填点。

### N3：绝对坐标相加不等于残差学习

D1 失败不能用于否定正确的 anchor+residual。

---

## 15. H36M 与数据资产

### 15.1 H36M 格式 AMASS 合成训练集

Workspace：

- `/mnt/data/cjydata/mhp_workspace/paper_single_h36m/`

最终 combined train：

- `/mnt/data/cjydata/mhp_workspace/paper_single_h36m/datasets/_random_20_small_room_h36m_amass_mmpose_joints_train_dome_h36m_calibs_9_11_no_fit_rotated_triangulated__random_cameras_20_/amass_mmpose_joints_train.pkl`

状态：128,109 样本，99/99 分片和 combined 两级校验通过。

生成日志：

- `/mnt/data/cjydata/step2_logs_h36m_train_gpu1/`
- `/mnt/data/cjydata/step2_logs_h36m_train_gpu1_tail_98_87/`
- `/mnt/data/cjydata/step2_logs_h36m_train_watchdog/watchdog.log`

限制：真实 Human3.6M 官方测试 annotation 尚未形成完整严格闭环，不能把 synthetic validation 冒充论文 H36M test。

### 15.2 CMU 时序

- 图像：`/mnt/data/cjydata/cmu_temporal/`
- 正确 matched annotation：见第 7 节。

### 15.3 输出盘要求

服务器系统盘空间紧张。数据、checkpoint、日志和评测结果必须放在：

- 数据：`/mnt/data/cjydata/`
- 输出：`/mnt/data/cjyoutput/`

不要把大文件写入 `/home/lixiaob/cjy`。

---

## 16. 论文与参考代码

主要论文：

- RUMPL：`/home/lixiaob/cjy/reference/rumpl.pdf`
- Geometry-Biased Transformer：`/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`
- MVGFormer：`/home/lixiaob/cjy/reference/MVGformer.pdf`
- 7.15 三篇：`/home/lixiaob/cjy/reference/7.15/`
- 7.24 新论文：`/home/lixiaob/cjy/reference/7.24/`
- 蒸馏参考：`/home/lixiaob/cjy/reference/zszl/`

参考代码：

- `/mnt/data/cjydata/reference_code/`

重要代码方向：

- PlaneSweepPose：逐视角 ray-depth / depth-bin supervision；
- Learnable Triangulation：confidence-weighted differentiable triangulation；
- AdaFuse：view reliability 和 Occlusion-Person；
- MvP：projective attention，而不是标量 geometry bias；
- MVGFormer：显式 geometry module 与 appearance module 迭代；
- UniCodebook：离散姿态先验 + DCSA；
- UPose3D：不确定性感知、合成训练和跨域鲁棒性。

---

## 17. 当前论文故事应如何调整

不再使用：

- “5 视角 teacher 蒸馏 2 视角并大幅提升所有组合”；
- “简单 confidence/geometry bias 显著提升 RUMPL”；
- “时序模块显著降低真实 CMU MPJPE”；
- “所有视角数全部下降”。

目前更可信的科学发现是：

1. RUMPL 在 noisy 2D 下依靠学习先验修复坏几何；
2. 在高质量 2D 下，网络却会偏离已经正确的几何解；
3. 固定先验只帮助 V2，容易伤 V3-V5；
4. 需要 reliability-aware 的几何锚定残差，而不是重复注入 ray/conf 信息；
5. 结构化遮挡训练在 V2 robustness 轴上已有稳定收益。

候选论文主句：

> 当多视角几何可靠时，模型应保留几何解；当 2D 检测、遮挡或相机配置使几何失效时，学习模型才应介入修正。我们通过可靠性门控的几何残差融合，在任意相机配置下自适应切换显式几何与数据驱动先验。

---

## 18. 下一步主线：Reliability-Gated Geometric Residual

这是当前最有证据支撑、尚未被正确实验否定的方向。

```text
anchor = confidence-weighted differentiable ray least squares
residual = residual_head(VFT/PFT features)
reliability = MLP(confidence, ray consistency, geometry condition, visibility)
gate = sigmoid(reliability)
prediction = anchor + gate * residual
```

要求：

1. residual head 零初始化，模型初始严格等于 anchor；
2. 不能再做 `absolute + absolute`；
3. V2 可用稳定 midpoint 做对照，V3+ 使用 weighted ray LS；
4. 先冻结 backbone 只训 residual/gate，再小学习率联合训练；
5. 保留 AMASS 随机相机、随机视角数协议；
6. 第一轮必须同时报告：HRNet V2-V5、GT-2D V2、occ0.6；
7. 必须监控 gate 与 confidence/ray consistency 的相关性；
8. 若 clean 无提升但遮挡提升，可转为 robustness 论文，不应强说 clean SOTA。

推荐实验顺序：

1. 零训练验证 `prediction=anchor` 和 V2 GT-2D 约 10.5 mm 上限；
2. anchor + frozen-backbone residual；
3. 增加 reliability gate；
4. 加入 A2 结构化遮挡训练；
5. 做去 anchor、去 gate、去 A2、固定 gate 等消融；
6. 做 attention/gate/失败相机组合可视化；
7. 最后扩展 H36M 和 Occlusion-Person。

---

## 19. 明确不要重复的实验

- 不要继续扫简单 conf/geom attention bias 系数；
- 不要继续扫 global-JV gate；
- 不要把旧 46.9 mm baseline 的结果写进主表；
- 不要使用 `matched_swapv3` CMU 时序标注；
- 不要继续把 AMASS 时序内部提升当真实提升；
- 不要原样重启 D1 absolute+absolute anchor；
- 不要直接使用 V3+ 旧 n-skew midpoint；
- 不要在没有图像特征时继续做几何 2D soft-fill；
- 不要继续 KPA、MH3、AdaFuse view-weight 网格搜索；
- 不要在未修 inplace backward 前重跑 D3-PCT；
- 不要用固定 Panoptic 相机训练破坏 AMASS 随机相机泛化协议。

---

## 20. 原始文档索引

- 三周总结：`/home/lixiaob/cjy/2026-07-10_three_week_rumpl_summary_and_paper_plan.md`
- V2 hard-view 旧调试：`/home/lixiaob/cjy/2026-07-11_hardv_debug_cmu_v2_eval.md`
- strict baseline 审计：`/home/lixiaob/cjy/2026-07-22_rumpl_baseline_reproduction_audit.md`
- global-JV：`/home/lixiaob/cjy/2026-07-22_global_joint_view_ablation_manifest.md`
- gated global-JV：`/home/lixiaob/cjy/2026-07-22_gated_global_joint_view_experiment_manifest.md`
- 时序/J2/S1：`/home/lixiaob/cjy/2026-07-23_temporal_and_adaptive_global_jv_experiments.md`
- 失败分类与模块审计：`/home/lixiaob/cjy/2026-07-24_failure_analysis_and_rumpl_module_audit.md`
- GT-2D 与下一步交接：`/home/lixiaob/cjy/2026-07-26_chat_handoff_gt2d_stage_audit_and_next.md`
- 代码文献研究：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/NOTES_codebase_study_20260725.md`

---

## 21. 接手后的最小操作说明

```text
1. 只认 R5：V2/V3/V4/V5 = 30.885/23.039/20.213/18.746 mm。
2. 先读本文件和 7/26 GT-2D handoff。
3. 检查 multiview_rumpl.py 当前 tri-anchor 分支，禁止 abs+abs。
4. 实现 anchor + gate * residual，residual 零初始化。
5. 使用 eval_exact_multiview_20260723.sh 与 eval_gt2d_upperbound_20260725.sh。
6. 所有大文件写 /mnt/data/cjydata 或 /mnt/data/cjyoutput。
7. 每个模型必须报告 V2-V5 全组合平均、KP*、最佳组合、改善组合数和最坏退化。
```


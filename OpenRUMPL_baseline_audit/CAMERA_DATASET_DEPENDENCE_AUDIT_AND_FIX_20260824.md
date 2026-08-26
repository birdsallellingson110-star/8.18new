# 相机坐标系与跨数据集依赖审计、修复方案（2026-08-24）

## 0. 本阶段范围

- 正式跨数据集实验只做 **CMU 训练 → Human3.6M 测试**。
- 不把 H36M → CMU 作为论文实验，也不为该方向消耗训练资源。
- 本文当前数字仅来自 16–128 帧的小样本诊断，作用是定位依赖和验证修复的数学正确性，**不是论文主表结果**。
- 目前没有启动大规模训练；先证明问题在哪里、修复是否对，再决定正式训练。

## 1. 要回答的问题

当前 H76 + E2-C2 是否真正与相机位置、世界坐标系和数据集无关？如果同一组相机和人体只改变世界坐标系定义，网络输出是否只发生相同刚体变换？

对于任意旋转 `R` 和平移 `t`，同时变换所有输入射线、候选姿态与真值：

```text
d' = R d
o' = R o + t
y' = R y + t
```

物理观测没有变化，因此理想模型必须满足：

```text
f(Rd, Ro+t) = R f(d,o) + t
```

本文将变换后预测逆变换回原坐标系，再与原预测计算距离。该距离称为 **SE(3) 等变误差**。它不依赖 3D 标签，也不会混入 HRNet 精度；非零即表示模型使用了任意世界坐标统计。

## 2. 使用的模型与小样本

- 主干：H76，三角化锚点 + 中心射线 + Plücker 输入 + PFT/VFT。
- 评分融合：HRNet E2-C2 identity-hinge，22 个候选，soft fusion 温度 1.8。
- H36M 审计缓存：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/validation_c2_22c.npz`。
- SMPL 随机相机诊断：`strict_split90_9` 验证集的 16 帧，每帧从 20 台随机相机中用不依赖 3D 标签的置信度和射线条件数选 4 台。

审计脚本：

- `audit_camera_coordinate_equivariance_20260824.py`
- `evaluate_smpl_random_camera_dependency_20260824.py`

## 3. 结论一：当前主干明显依赖世界坐标系

64 帧 H36M 上，冻结 H76+C2 主干的 SE(3) 等变误差如下（mm）：

| 坐标变换 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 平移 x 方向 1 m | 7.48 | 2.79 | 2.78 |
| yaw 37° | 65.59 | 30.34 | 20.78 |
| yaw 90° | 86.95 | 43.65 | 39.79 |
| 任意三维旋转 | 92.60 | 46.76 | 39.20 |

这不是相机变差、2D 检测变差或标注误差，因为输入代表的物理射线完全相同，只是世界坐标轴换了定义。主干的旋转依赖是当前跨数据集泛化的首要结构性风险，量级远大于 E2 评分器。

### 3.1 平移依赖的精确来源

三角化锚点原实现有两个问题：

1. `A + lambda I` 的正则项隐式把解拉向世界原点。把 `lambda` 从 `1e-4` 降至 `1e-8` 后，V2 的异常平移误差大幅消失，证明这一来源存在。
2. 已训练权重中的 `tri_anchor_gate = 0.9973`。输入整体平移 1 m 时，仅该门控就会留下约 `(1-0.9973) × 1m = 2.7 mm` 的误差，与 V3/V4 实测值吻合。

### 3.2 旋转依赖的来源

中心射线和 Plücker 坐标只降低了平移敏感性，但后续仍把世界坐标 `x/y/z` 分量交给普通线性层与 Transformer。普通 MLP/Transformer 并不自动满足 SO(3) 等变性，因此换一个世界坐标轴，模型可能生成不同人体，而不是简单旋转原结果。

## 4. 结论二：E2 评分器也有依赖，但不是主因

128 帧 H36M 上，冻结 E2 soft scorer 的等变误差（mm）：

| 坐标变换 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 平移 x 方向 1 m | 0.325 | 0.631 | 0.443 |
| yaw 90° | 1.779 | 4.130 | 3.850 |
| 任意三维旋转 | 4.945 | 4.009 | 3.195 |

任意三维旋转还会使 MPJPE 改变 `+2.182/+0.558/+0.350 mm`。V4 在 yaw 90° 后，hard candidate 的选择有 58.5% 发生变化。

E2 的具体依赖包括：

- 使用 H36M 训练集逐坐标轴的 `mean/std`；
- 候选中保留绝对 root 和世界坐标轴方向；
- 候选、射线与几何残差经过轴相关的普通 MLP。

因此“只删除 camera ID”不能得到相机无关模型。另一方面，E2 的 1–5 mm 依赖明显小于主干的 20–90 mm；只把 GHT 规范化加在评分器上不能解决整个系统。

## 5. 结论三：随机相机 SMPL 暴露出数据集分布依赖

为避免随机抽到人体不可见或严重退化相机，16 帧诊断使用置信度 × 射线条件数的 label-free `quality4` 选相机。绝对 MPJPE 如下（mm）：

| 数据 | 模块 | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| H36M 小样本 | H76 generator | 41.83 | 34.80 | 34.05 |
| H36M 小样本 | triangulation | 103.81 | 56.97 | 53.68 |
| H36M 小样本 | E2 soft | 41.75 | 32.20 | 31.22 |
| SMPL 随机相机 | H76 generator | 112.74 | 69.20 | 59.23 |
| SMPL 随机相机 | triangulation | 125.94 | 73.92 | 65.37 |
| SMPL 随机相机 | E2 soft | 110.42 | 60.38 | 51.11 |

应谨慎解释：SMPL 的 2D 置信度和几何本身更难，三角化已经比 H36M 差约 11–22 mm；因此全部差距不能归因于网络。但主干在 SMPL 上丢失了大量相对纯三角化的优势，且 root-relative error 同样显著变差，说明除了相机几何，还存在姿态、2D 误差和置信度分布适配问题。

该表只作为小样本风险提示。正式跨数据集结论必须由 CMU 真实训练 → H36M 标准测试给出。

## 6. 已实现修复：将 GHT 式规范化移到整个生成器之前

只规范化 E2 太晚。当前实现对进入完整 RUMPL generator 的射线先构造由观测决定的身体坐标系：

1. 对激活视角做置信度加权射线相交，得到粗 3D 关节；
2. pelvis 为原点；
3. 左右肩方向为 x 轴；
4. pelvis→neck 在 x 轴正交平面内的方向为 y 轴；
5. 叉乘得到 z 轴；
6. 保留米制尺度，不用数据集骨长归一化；
7. 在该坐标系运行完整 H76，最后把预测逆变换回世界坐标。

正则项以 point-on-ray centroid 为中心，而不是世界原点，因此坐标系构造本身满足 SE(3) 等变。

代码位置：

- `OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
  - `equivariant_body_canonicalize_rays`
  - `body_canonical_pose_to_world`
- 开关：`RUMPL_BODY_CANONICAL_FRAME=1`
- 正则：`RUMPL_BODY_CANONICAL_REG=1e-4`
- 默认关闭，旧模型与旧实验路径不受影响。

### 6.1 数学正确性验证

16 帧、任意三维旋转和平移后：

- 原 H76 等变误差：约 `92.6/46.8/39.2 mm`；
- 外部 canonical wrapper：约 `0.150/0.003/0.002 mm`；
- 集成到模型内部后的最小回归：平均约 `0.004 mm`，P95 约 `0.009 mm`。

这说明坐标依赖已经从结构上消除，而不是靠数据增强记住更多相机。

但把旧权重直接套进新坐标系会得到约 `156/132/117 mm`，原因是旧权重是在 H36M 世界坐标中学习的。故该修复必须重新训练或逐阶段微调，不能把旧权重的 inference-only 结果当成方法性能。

## 7. E2 的对应修复

评分器增加同一套 ray-derived pelvis/shoulder/torso canonical geometry，并以固定各向同性米制尺度替代 H36M 逐轴 `mean/std`。候选最终仍在原世界坐标中加权融合，保持绝对 3D 输出。

代码位置：

- `OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py`
- `OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py`
- 固定当前 22 候选布局的入口：`train_e2_camera_independent_22c_20260824.py`
- 参数：`--canonical-geometry --fixed-metric-normalization`

旧 E2 权重套用 canonical geometry 后，任意旋转和平移等变误差已经降至约 `0.066/0.0003/0.0003 mm`，hard selection flip 为 0；但旧权重准确率为 `51.60/33.36/30.96 mm`，同样说明需要在新表征上重新训练。

已完成 2 batch、1 epoch smoke test：forward、backward、保存和重载正常，无 NaN。该结果只证明代码可训练，不代表精度。

## 8. 为什么不能仅“照搬 GHT 评分器”

GHT 式姿态规范化方向是正确的，但本系统与单纯 hypothesis scorer 的边界不同：

```text
当前依赖量级：H76 generator 20–90 mm  >>  E2 scorer 1–5 mm
```

所以正确顺序是：

```text
射线与置信度
  → 数据集无关身体坐标系
  → 完整 RUMPL PFT/VFT + 三角化锚点
  → canonical E2 候选效用评分
  → 世界坐标下融合和输出
```

保留的已有有效模块包括：RUMPL 射线建模、PFT/VFT、三角化锚点、E2 Set Transformer、逐关节候选效用和 soft fusion；替换的是它们所处的坐标表达和数据集统计归一化，而不是推倒整套模型。

## 9. 下一步（坚持小样本优先）

### A. 已完成：无训练结构诊断

- 主干和 E2 分别做刚体等变审计；
- 用 SMPL 随机相机做小样本分布检查；
- 确认主干是第一瓶颈，E2 是第二瓶颈；
- canonical 主干和 canonical E2 的数学等变性已通过。

### B. 下一项：小规模可学习性验证，不做完整训练

用少量现有训练样本做短程对照，只回答新坐标系能否从旧 H76 初始化并恢复有效精度：

| 组别 | generator frame | E2 frame | 目的 |
|---|---|---|---|
| B0 | 原世界坐标 | 原世界坐标 | 同数据、同迭代控制 |
| B1 | canonical | 暂不接 E2 | 验证主干能否学习并保持等变 |
| B2 | canonical | canonical | 验证端到端接口和评分学习 |

停止条件：若 B1 在很短训练内完全不能恢复、出现退化坐标系或肩轴不稳定，先修 frame estimator，不启动正式 CMU 训练。

### C. 正式实验：只做 CMU → H36M

必须使用真实 CMU 训练序列产生的 HRNet 2D、置信度和标定射线。当前本地检查到：

- `cmu_rumpl_official_eval_20260722` 只有真实 `171204_pose5/pose6` 验证数据；
- `paper_single_cmu` 的训练 PKL 是 AMASS/SMPL 投影到 CMU 相机布局，适合相机布局诊断，**不能称为真实 CMU 训练**。

因此正式 CMU→H36M 之前必须补齐真实 CMU 训练数据或等价的官方 2D cache。下载时只需选定训练序列和少量计划视角，不需要一次下载 30 台相机全部数据；但 HRNet 坐标协议需要相应 HD 图像/视频，不能只下载 3D 标注。

正式对照至少包含：

| 模型 | CMU 同域验证 | CMU→H36M | 世界坐标等变误差 |
|---|---:|---:|---:|
| 原 H76 + 原 E2 | 待测 | 待测 | 已证实较大 |
| canonical H76 | 待测 | 待测 | 约 0 |
| canonical H76 + canonical E2 | 待测 | 待测 | 约 0 |

## 10. 结果文件

- 主干 64 帧审计：`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/h76_c2_generator_64.json`
- E2 128 帧审计：`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/e2_identity_hinge_128.json`
- anchor 正则隔离：`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/h76_c2_generator_reg1e8_32.json`
- canonical wrapper：`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/h76_c2_bodycanonical_16.json`
- SMPL quality4 16 帧：`/mnt/data/cjyoutput/camera_generalization_20260824/smpl_random_camera_audit/smpl16_quality4_current_h76_e2.json`
- canonical E2 smoke：`/mnt/data/cjyoutput/camera_generalization_20260824/smoke_e2_canonical_22c`

## 11. 当前可下的结论与不可下的结论

可以下的结论：

1. 当前 H76 和 E2 都依赖 H36M 世界坐标，其中主干依赖更严重；
2. 问题不是简单 camera-ID embedding，而是坐标表征和 H36M 统计本身；
3. 将身体规范化放到完整 generator 前可在数值上消除任意世界刚体变换依赖；
4. 旧权重不能直接复用，必须训练适配；
5. SMPL 小样本提示仍有 2D/姿态/置信度分布差异，不能只修相机坐标。

暂时不能下的结论：

1. canonical 模型一定提高 H36M 或 CMU 的 MPJPE；
2. SMPL 16 帧数字等于正式跨数据集性能；
3. 本地 synthetic-CMU 可以替代真实 CMU 训练；
4. 尚未完成训练的 canonical 方案可以写入论文主结果表。

## 12. 2026-08-24 19:02 小样本训练后复核与正式 Stage 1 启动

在相同 HRNet 输入、seed 0、`8:1:1` 视角比例下，用 2048 个训练样本、
1024 个验证样本训练 5 轮。该实验只检查可学习性，不进入论文主表：

| 小样本 generator | 最后一轮 absolute MPJPE | SMPL random-camera V2/V3/V4 |
|---|---:|---:|
| 原世界坐标 H76 | 51.82 mm | 117.45/73.94/66.30 mm |
| body-canonical H76 | **47.22 mm** | **114.98/69.19/59.88 mm** |

因此新表征可以正常学习，并在这一小样本控制下没有以 H36M 精度换取
等变性。训练后的 canonical checkpoint 在任意三维旋转和平移下的平均
等变误差仍为 `0.066/0.0022/0.0017 mm`（V2/V3/V4）。SMPL 结果只提供
启动正式训练的正向证据，不能替代 CMU→H36M 跨数据集主实验。

正式 H36M Stage 1 已启动：

- 输出根目录：`/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend`
- tmux：`cjy_camgen_stage1`
- HRNet：body-canonical H76，20 epochs，`8:1:1`，当前先运行；
- ResNet-152：body-canonical H76 + Global Joint-Query，前 8 epochs 固定 K=2，
  后续 `3:1:1`，在 HRNet 后自动运行；
- 两条线均在 S1/S5/S6/S7/S8 训练，在 S9/S11 对全部 V2/V3/V4 组合评估；
- GPU1 上是其他用户的 RayMixSTE，正式任务只使用 GPU0。

启动器：`launch_stage1_canonical_dual_frontend_generators_20260824.sh`。

旧 H18 不能原样进入跨数据集路线，因为它显式使用 absolute root 和 11 个
H36M camera-subset task embedding。当前已增加 `--camera-independent` 模式：

- 用中心帧 pelvis/shoulder/torso 坐标系规范整个 T=9 pose window；
- root 输入变成相对中心帧的身体坐标运动，不再使用世界绝对位置；
- 不加入 camera-subset task embedding；
- 在 canonical 坐标预测 residual，再旋转回世界坐标；
- root 仍保持 E2 的绝对位置，不由时序模块重写。

随机非零 residual 权重下的刚体等变单元检查为 `0.000047 mm`。旧模式默认
行为不变；正式新路线只使用 `--camera-independent`。下游 E2/H18 已在 tmux
`cjy_camgen_stage1_downstream` 排队，等待两种 generator 完成后自动执行。

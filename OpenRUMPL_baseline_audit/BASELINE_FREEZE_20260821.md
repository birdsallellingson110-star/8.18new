# Baseline freeze（2026-08-21）

## 决定

当前主线冻结两个明确口径，禁止把不同输入、不同视角专用模型或不同评估协议拼接成一个结果。

### 论文公平主 baseline：E2-C2 soft-cal，T=1

| 方法 | V2 | V3 | V4 | 平均 |
|---|---:|---:|---:|---:|
| E2-C2 soft-cal（单一统一模型） | **38.700** | **29.486** | **27.274** | 31.820 |

这是当前唯一作为 GBT-style 单帧坐标级公平比较的主 baseline。它使用同一模型、同一
候选评分器和同一温度处理 V2/V3/V4，不包含时序、偏置、热图或图像特征。

输入固定为：GBT-aligned HRNet 输出的 2D 坐标、关节置信度和相机参数，经射线/候选
三角化进入 RUMPL/E2；不允许改成 A1D、CPN、热图、图像特征或 RIGR 中间特征。

主要缓存位置：

- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/`
- 统一实验记录：`/home/lixiaob/cjy/GBT_ALIGNED_SIX_STAGE_EXPERIMENT_PLAN_20260815.md`

### 当前输入不变条件下的最佳内部结果：H18-lowLR，T=9

| 方法 | V2 | V3 | V4 | 平均 |
|---|---:|---:|---:|---:|
| H18-lowLR clean temporal | **37.704** | **29.231** | **27.219** | **31.385** |

该结果仍使用完全相同的 HRNet 坐标、置信度、相机和 E2-C2 前端，只增加 T=9
时序残差，因此可以作为“当前输入不变时的最佳模型”记录。但 H18 使用中心帧窗口，
存在未来帧，不能直接作为 GBT 因果时序公平主表的 baseline；论文主表仍使用上面的
E2-C2 T=1，H18-lowLR 作为时序消融/内部最强参考。

H18-lowLR 输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/`。

## 不纳入 baseline 的历史结果

- RIGR→E2 的 `约 33.811/28.555/27.808`：使用了 RIGR/A1D 或中间特征输入，违反
  当前输入不变条件；只能作为历史上限，不可与本主表混列。
- E2-C1 `37.009/30.108/28.528`：V2 更好但 V3/V4 更差，整体不如统一 E2-C2。
- C2 K2-heavy `38.686/30.943/28.629`：同输入但平均及 V3/V4 不如 E2-C2。
- E5 ray-conditioned dual-stream：已停止。其 S8 epoch 0 的短暂改善没有在后续
  epoch 保持，未产生正式 S9/S11 结果，不纳入 baseline。

## 执行状态

2026-08-21 已停止 E5 ray-conditioned 及其 control 进程；checkpoint、日志和中间
结果保留。GPU0/GPU1 均已释放。后续模块实验默认从 E2-C2 T=1 baseline 开始，若研究
遮挡/时序，则单独报告 H18-lowLR，不得覆盖或替换 T=1 公平 baseline。

## ResNet-152 阶段执行口径

官方 Learnable Triangulation H36M-finetuned ResNet-152 前端已经导出并验证：384×384、
整图去畸变、H36M annotation bbox；训练集 312,188 条记录（S1/5/6/7/8），测试集
8,084 条记录（S9/11），每条记录均含相机 0/1/2/3。只向 RUMPL 提供 2D 坐标、置信度和
相机参数，没有热图或图像特征。

19:56 启动的 ResNet R0/H76 试跑已停止。原因是它没有接入已经冻结的 E2-C2→H18
主线，而且采用了旧的 `3:1:1+前8轮固定K=2` 配置，不能作为正式结果；未完成权重和日志
保留在 `gbt_aligned_resnet_20260817_gpu1/`，不删除、不混入表格。

正式 ResNet 管线改为：

1. ResNet-152 输入上训练 H76 源模型，严格使用 C2 已调好的 `8:1:1`（V2/V3/V4）
   采样、20 epoch、无固定视角覆盖；
2. 用该 ResNet-H76 checkpoint 重新导出 11 个候选，并追加同样的置信度加权候选形成
   22-candidate pool；
3. 按 E2-C2 原协议训练两个 seed 的统一 V2/V3/V4 scorer，采用相同的
   `T_V2=0.4,T_V3=T_V4=1.8`；
4. 用 seed-0 的冻结 E2-C2 输出构造帧级融合，再严格复用 H18-lowLR：T=9、stride=5、
   hidden=96、2层、lr=5e-5、wd=5e-4、12 epoch。

可恢复总启动脚本：
`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_resnet152_e2_c2_h18_lowlr_20260821.sh`

输出根目录：`/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/`。脚本每个阶段均有
独立缓存和完成标志，任何中断都从最近阶段恢复，不会复用 HRNet 的 E2 权重。

20:13 已重新启动正式管线；当前处于 ResNet-H76 的第 0 轮，日志已确认
`weighted-random-8,1,1`、`fixed_epochs=0`。旧 R0/H76 试跑不计入本阶段。

20:19 在 GPU1 追加启动同协议 seed1 复现实验，作为 ResNet-H76 的稳定性检查；它
不改变主线的 seed0→E2-C2→H18 顺序，也不与主线共享或覆盖 checkpoint。

21:10 进度：GPU0 seed0 已到第14/20轮，GPU1 seed1 已到第12/20轮；两张卡训练进程
正常、无 OOM 或数据错误。H76 源模型完成后才进入候选导出和 E2-C2，当前尚未产生
可报告的 ResNet MPJPE。

2026-08-22 00:10：两条 H76 均已完成，seed0/seed1 checkpoint 均已保存，候选缓存
（train/validation 11c→22c）已生成。首次进入 E2-C2 时发现并修复了 bash 中同一条
`local` 声明引用未初始化变量的问题；没有影响任何模型或缓存。当前两个 E2-C2
scorer 正在 GPU0/GPU1 并行训练，均已完成 direct epoch 4，尚未生成最终校准结果。
## 2026-08-22 ResNet-H76 direct evaluation and dense temporal validation fix

The frozen ResNet-152 H76 source checkpoint was evaluated directly under the
strict coordinate-level protocol (H36M S9/S11, action-equal All-17 absolute
MPJPE, no flip). Results are:

| seed | V2 | V3 | V4 |
|---|---:|---:|---:|
| 0 | 41.5703 | 26.2109 | 24.2924 |
| 1 | 41.3704 | 25.9503 | 24.0222 |
| mean | 41.4704 | 26.0806 | 24.1573 |

The ResNet E2-C2 scorer (the frozen H76 candidate pool plus confidence
candidate and per-cardinality calibration) remains the stronger ResNet line:
V2/V3/V4 = 40.9155/23.3164/20.6082 mm (mean of two seeds).

The first H18-lowLR ResNet attempt was correctly stopped as invalid: its
ordinary ResNet validation pkl contains only sparse image ids (step 65), so no
T=9, stride=5 windows can be formed. We are regenerating validation with the
same official LT ResNet-152 checkpoint and preprocessing, but using the dense
step-5 temporal record as the image/bbox index. This changes only frontend
density, not the input type or model; H18 will be rerun after this export.

The identity-hinge E2-C2 control is running in parallel with two seeds at
`e2_c2_scorer_hinge/`; it keeps the 2D/camera input and candidate pool fixed.

The two-seed identity-hinge control has now finished. With the same per-cardinality
calibration (V2 T=0.4, V3/V4 T=1.8), it gives V2/V3/V4 =
40.8620/23.1725/20.4261 mm (std 0.0104/0.0045/0.0142), versus the no-hinge
E2-C2 40.9155/23.3164/20.6082 mm. Thus the hinge improves all three
cardinalities by 0.054/0.144/0.182 mm; it is a small but consistent ablation,
not yet the final ResNet baseline.

The dense temporal ResNet frontend export has now completed with 105,076
validation records, and the H18-lowLR rerun is active at
`h18_lowlr_dense_validation/`. Its epoch-1 S8 holdout (not a final S9/S11
result) is already below the zero-residual holdout by 2.82/0.68/0.42 mm for
V2/V3/V4; training continues through the planned 12 epochs.

The dense-validation H18 run has since completed. Final strict S9/S11
temporal results are V2/V3/V4 = 38.5710/24.1900/21.6099 mm, improving its
matched dense-validation zero-residual baseline 43.6170/25.2243/22.1788 mm
by 5.046/1.034/0.569 mm. This is reported separately from the single-frame
E2-C2 table because its validation frontend is dense and its protocol is T=9.

**Important protocol correction:** this H18 ResNet run used the no-hinge
`e2_c2_scorer` checkpoint and its corresponding score/fused cache. It did not
use the later two-seed `e2_c2_scorer_hinge` baseline (40.8620/23.1725/20.4261
mm). Therefore 38.5710/24.1900/21.6099 mm must not be described as temporal
performance on the identity-hinge baseline. A separate H18 retraining from
the identity-hinge score cache is required before making that comparison.

## 2026-08-22 identity-hinge temporal and two-view geometry rerun

The corrected rerun is active. It uses the seed-0 checkpoint from the finished
two-seed identity-hinge E2-C2 scorer, the same 22-candidate cache, dense T=9
validation records, and the fixed H18-lowLR settings (stride 5, hidden 96,
2 layers, lr 5e-5, wd 5e-4, 12 epochs). Outputs are isolated at
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/h18_identity_hinge/` and are
launched by `launch_resnet_identity_hinge_h18_20260822.sh`.

At epoch 0 on S8 holdout, the corrected temporal branch is 19.434/10.736/9.383
mm (V2/V3/V4), compared with its matched T=9 center baseline 20.829/11.104/9.529
mm; the mean falls from 14.149 to 13.184 mm. This is an intermediate holdout
signal only; final S9/S11 will be read from `result.json` after all epochs.

In parallel, the ResNet-H76 geometry-uncertainty two-view diagnosis is still
running with two seeds under the fixed 8:1:1 sampling protocol at
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/h76_geom_uncertainty/`.

The corrected identity-hinge H18 run has now completed. Selecting the best
epoch on S8 holdout (epoch 11, 12.4822 mm mean), its strict S9/S11 result is
V2/V3/V4 = **38.5525/24.0819/21.4762 mm** (mean 28.0369 mm). Against its
matched dense T=9 center baseline 43.5837/25.0965/22.0174 mm, the gains are
5.0312/1.0146/0.5412 mm. This is the valid temporal result on the
identity-hinge branch; it is separate from the single-frame 40.8620/23.1725/
20.4261 mm table.

The two-view H76 geometry-uncertainty training and direct evaluation have also
completed. The two-seed action-equal All-17 results are:

| line | V2 | V3 | V4 |
|---|---:|---:|---:|
| seed 0 | 41.3769 | 26.0170 | 24.0893 |
| seed 1 | 40.7243 | 25.8568 | 23.9112 |
| mean | **41.0506** | **25.9369** | **24.0002** |

Relative to the preceding direct ResNet-H76 mean 41.4704/26.0806/24.1573 mm,
the geometry token gives only about 0.420/0.144/0.157 mm. It is therefore a
small two-view correction, not yet the final E2 result. The follow-up E2-C2
scorer on this geometry cache is still training at
`e2_c2_geom_uncertainty/scorer/`; its calibrated result is not available yet.

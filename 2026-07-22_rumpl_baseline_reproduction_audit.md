# RUMPL baseline 严格复现审计（2026-07-22）

## 结论

当前 CMU RUMPL baseline **没有复现论文结果**。

| 项目 | CMU V2 Absolute KP* MPJPE |
|---|---:|
| RUMPL 论文（全部 10 个相机对平均） | 35.000 mm |
| 当前 `run_conf`（全部 10 个相机对平均） | 46.913 mm |
| 绝对差距 | +11.913 mm |
| 相对差距 | +34.0% |

此前把 `[3,6] = 40.37 mm` 写成 CMU V2 baseline，是把单个相机对误当成全部组合平均，现正式作废。当前蒸馏、偏置和课程实验仍可作为同一内部实现上的相对消融，但在严格 baseline 恢复前，不能声称已经复现 RUMPL，也不能据此声称超过论文结果。

## 当前重跑状态（2026-07-22）

旧测试集上的第一条诊断训练已在 epoch 3 后主动停止。它的 MMPose Absolute MPJPE 从 epoch 0 的 `97.94 mm` 降到 epoch 2 的 `62.10 mm`，但由于随后确认测试帧集合错误，该曲线不能用于论文对比，只保留作排障记录。

精确测试协议现已恢复，并开始两条 **official-like public reconstruction** 正式训练。它们不是“严格复现完成”，而是对公开代码可达到结果的受控重建：

- 隔离代码：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit`，基于公开 commit `628bbe9`。
- 训练规模：128,109；测试规模：6,420（公开 loader 过滤后 642 帧 x 10 个双视角组合）。
- 公共配置：官方 YAML 的 `DIM=128`，ray/confidence 拼接后有效 hidden width 为 256；20 epochs、seed 0，scheduler 行为一致。
- `R0_public_original`：保留公开代码 PFT 末层重复执行行为。
- `R1_fix_pft_last_block`：唯一变量是 PFT 末层由执行两次修正为一次。
- 日志和输出根目录：`/mnt/data/cjyoutput/baseline_reaudit_20260722/`。
- 启动脚本：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh`。

正式比较将同时报告固定 epoch 20/final state 与 `model_best`，避免只挑最佳测试轮次造成选择偏差。scheduler 顺序修正不会混入这两条实验，而是后续独立变量。

两条实验的 epoch 0 精确协议验证已完成。MMPose Absolute 指标为：`R0` All-17 `94.29 mm`、KP* `98.97 mm`；`R1` All-17 `96.93 mm`、KP* `96.82 mm`。这些只是首轮收敛检查，不能作为最终模型比较。监控脚本会在两条实验完成后检查 `epoch.txt=20` 和 `final_state.pth.tar`，再自动生成十组合 JSON；日志为 `/mnt/data/cjyoutput/baseline_reaudit_20260722/watch_exact_baselines.log`。

2026-07-22 19:39 中间检查：两条实验均已完成 15/20 epochs。当前 epoch 14 的 MMPose Absolute 十组合平均为：`R0` All-17 `31.569 mm`、KP* `36.437 mm`；`R1` All-17 `31.449 mm`、KP* `36.412 mm`。两者已进入预设的宽松验收区间 `<=32.3/36.5 mm`，但与论文 `30.8/35.0 mm` 仍分别相差约 `0.6/1.4 mm`。当前最好 KP* 组合均为 `(3,23)`，约 `28.2 mm`；最差为 `(6,12)`，约 `42.2-42.8 mm`。中间文件会被下一轮验证覆盖，最终结论必须使用 epoch 20 和训练结束后独立重评的 `model_best`。

2026-07-22 20:18 最终检查：R0/R1 均正常完成 20 epochs，`epoch.txt=20`、`final_state.pth.tar` 和 `model_best.pth.tar` 均存在，自动检查状态为 0。独立重评结果如下：

| 实验与 checkpoint | All-17 | KP* | 相对论文 All-17/KP* |
|---|---:|---:|---:|
| R0 public-original, final epoch 20 | 31.410 | **36.075** | +0.610 / +1.075 mm |
| R0 public-original, model_best | **31.342** | 36.306 | +0.542 / +1.306 mm |
| R1 fix-PFT-last-block, final epoch 20 | 31.549 | 36.555 | +0.749 / +1.555 mm |
| R1 fix-PFT-last-block, model_best | 31.440 | 36.467 | +0.640 / +1.467 mm |
| 论文 | 30.800 | 35.000 | 0 / 0 |

R0 整体优于 R1，说明删除公开代码中的 PFT 末层重复执行并未改善该 seed。R0 final 的最好 KP* 组合为 `(3,23)=27.662 mm`，最差为 `(6,12)=41.968 mm`。四份正式汇总位于 `/mnt/data/cjyoutput/baseline_reaudit_20260722/*summary.json`。当前公开重建已通过预设宽松门槛，但仍不能写成精确达到论文数字；下一受控变量为 scheduler 调用顺序。

2026-07-22 22:56 scheduler 对照完成：R2（保留公开 PFT、修正 scheduler）优于 R0；R3（同时修正 PFT 与 scheduler）仍弱于 R2。结果如下：

| 实验与 checkpoint | All-17 | KP* | 相对论文 All-17/KP* |
|---|---:|---:|---:|
| R2 fix-scheduler, final epoch 20 | 31.134 | **35.871** | +0.334 / +0.871 mm |
| R2 fix-scheduler, model_best | **30.978** | 36.011 | +0.178 / +1.011 mm |
| R3 fix-scheduler+PFT, final epoch 20 | 31.366 | 36.386 | +0.566 / +1.386 mm |
| R3 fix-scheduler+PFT, model_best | 31.128 | 36.196 | +0.328 / +1.196 mm |

R2 是当前最接近论文的公开重建；scheduler 修正确有小幅正收益，而 PFT 末层修正仍无收益。R2 model_best 的 All-17 已距论文仅 `0.178 mm`，但同一 checkpoint 的 KP* 仍差 `1.011 mm`。此前 launcher 使用 `WORKERS=4`，官方 YAML 为 `WORKERS=16`；由于 dataset worker 内执行随机相机选择和 20% missing-keypoint noise，worker 数会改变训练随机序列，不能视为纯吞吐参数。已启动 R4/R5 workers=16 对照：R4 为公开 scheduler，R5 为修正 scheduler，两者均保留表现更好的公开 PFT 行为。

## 已确认的论文评估协议

1. 指标为 Absolute MPJPE；KP* 为肩、肘、腕、膝、踝 10 个 COCO 关键点。
2. 测试相机为 CMU HD cameras `(3, 6, 12, 13, 23)`。
3. 两视角结果对全部 `C(5,2)=10` 个相机组合取平均。
4. 官方 manifest 有 702 个完整五视角帧；公开 loader 的默认 `wrong_cases` 过滤后，各组合均有 642 个样本，因此逐组合算术平均与合并 6,420 个样本后计算均值等价。
5. 论文目标为 All-KP `30.8 mm`、KP* `35.0 mm`。

证据来自论文正文/补充材料：其明确说明 two-view MPJPE averaged on all possible camera pairs，并列出 CMU 的五个相机；公开 dataset loader 在 `TEST_ON_ALL_CAMERAS=true` 时同样枚举组合。

## 主要复现偏差

### 1. 我们运行的不是随仓库发布的 CMU 配置

| 项目 | 仓库 CMU YAML | 当前 run_conf |
|---|---|---|
| 模型入口 | `multiview_pose_3d_fuser` | `multiview_rumpl` |
| embedding 参数 | `DIM=128` | `DIM=256` |
| 屏幕坐标归一化 | false | true |
| 相机归一化 | false | true |
| ray intersection | Camera | Closest |
| direction/intersection | 先拼接后映射 | 分别映射后拼接 |
| 置信度 | concat embedding | third input + concat embedding |
| missing-keypoint noise | 20% | off |
| ray 3D positional encoding | off | on |
| spatial blocks | single stack | multiple-spatial flag on |
| 3D axis conversion | Y/Z swap | off |

这不是单个超参数偏差，而是输入表示和有效模型宽度均发生变化。

### 2. 官方配置引用的实现没有公开

当前公开仓库只有 `multiview_rumpl.py` 和公开的 `multiview_*_rumpl` dataset classes。官方 YAML 引用的以下模块在当前提交、完整 Git 历史及 Git objects 中均不存在：

- `multiview_pose_3d_fuser`
- `multiview_amass_dome_pose_3d_fuser`
- `multiview_cmu_panoptic_pose_3d_fuser`

因此官方 YAML 无法原样执行。只能用公开类做 official-like reconstruction，并明确记录映射假设；若要宣称完全复现，应向作者索取缺失代码或官方 checkpoint。

### 3. `DIM` 含义已澄清，但公开前向存在可疑行为

- 论文写 hidden embedding `D=256`，仓库 CMU YAML 写 `DIM=128`。公开模型 smoke test 表明，`DIM` 是每个分支的 embedding 宽度；ray 与 confidence 拼接后有效 token 维度为 `256`，因此这两者并不矛盾。
- 我们的 `DIM=256` 配合 direction/intersection/confidence 三路分别映射和拼接，实际有效 token 维度为 `768`，并非论文的 `D=256`。
- 公开 `multiview_rumpl.py` 的 PFT 循环会把最后一个 block 调用两次：配置 12 层，实际执行 13 次，且最后一次共享同一层参数。

official-like `DIM=128` 模型 smoke test 已通过，参数量 `12.661M`，训练/推理前向均输出 `(B,17,3)`。重复末层行为仍需作为受控变量验证。

### 4. 合成数据质量尚未通过论文门槛

当前数据规模与论文一致：train 128,109 poses、20 random cameras，validation 文件也存在。抽查 10 个 Stage-V 分片（12,940 poses）得到：

- MMPose 对渲染真值的 2D 误差：median `14.0 px`，mean `51.8 px`，P90 `103.4 px`。
- confidence median `0.885`，至少 10% 的 confidence 为 0。
- 使用关键点 bbox 近似 COCO area 计算时，pose-level `OKS >= 0.75` 比例约 `45.7%`。

最后一项不是完整 COCO AP，不能直接等同论文的 `AP@OKS=.75 = 93%`，但必须用渲染 mask/bbox 和 COCO evaluator 正式复算，并抽查失败样本是否存在检测错人、坐标变换或关键点映射问题。

### 5. CMU 测试 split 不一致（已修复并验证）

旧测试集包含 `171204_pose5/pose6` 和五个目标相机，但**不是论文仓库提供的精确帧集合**。公开 manifest `filtered_keys_cmu_7train_2val_pose_5_64_all_cams_no_bad_annot_validation.txt` 包含 3,510 张图、702 个五视角帧；旧 pkl 包含 3,495 张图、699 帧。两者仅重合 65 张图，即 13 个五视角帧：

| 集合比较 | 图像数 |
|---|---:|
| 当前 pkl | 3,495 |
| 官方 manifest | 3,510 |
| 交集 | 65 |
| 当前独有 | 3,430 |
| 官方独有 | 3,445 |

原因是旧脚本先按原始帧号 `% 64 == 0` 抽图，再以 `skip=1` 预处理；官方流程先形成完整、有效的多视角 grouping，再对 grouping 做 `[::64]`。两者样本数接近，但抽到的帧几乎完全不同。因此此前所有 CMU 数字都只能视为相邻协议上的内部结果，不能直接对照论文表格。

现已按官方 manifest 重建独立测试目录 `/mnt/data/cjydata/cmu_rumpl_official_eval_20260722`，没有覆盖旧数据。3,510 张图全部抽取成功，GT pkl 为 3,510 条记录、702 帧；使用公开 loader 的 `wrong_cases` 后为 642 帧。精确图像使用 MMPose 1.3.2、HRNet-W32 384x288 和 RTMDet-M 重新推理，3,510 个预期结果全部存在。

当前旧测试集 MMPose 相对原始 2D 标注误差 median `9.73 px`，KP* median `8.87 px`。official-like 配置必须使用未预先换轴的 pkl；旧 baseline 的 `swapv3` 只适用于关闭 AMASS loader 换轴的旧训练配置，两者不能混用。

精确数据上的两视角三角化审计为 All-KP `42.864 mm`、KP* `43.821 mm`，论文 Table 3 为 `43.0/44.0 mm`，差值仅 `-0.136/-0.179 mm`。这基本精确复现了论文的 2D 检测、相机、帧集合、过滤与评估协议，证明此前约 9 mm 的三角化差距主要来自抽帧顺序错误。报告位于 `/mnt/data/cjyoutput/baseline_reaudit_20260722/triangulation_exact_manifest.json`。

## 恢复计划与通过门槛

### Phase A：冻结论文结论

- 当前模块实验只保留为内部消融，不再更新论文主表。
- 新实验必须记录代码 commit、完整 YAML、数据 hash、checkpoint、逐组合日志和聚合脚本。

### Phase B：干净公开代码重建

已建立隔离 worktree：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit`，基于公开 commit `628bbe9`，不包含这几周的模块修改。

主配置固定为 official-like `DIM=128`（有效 hidden `D=256`）。受控变量仅为公开代码重复末层行为保留/修正；如需测试 `DIM=256`，只能标为容量消融，不能标为论文配置。其余输入、数据、optimizer、20 epochs、batch 32、seed 0 完全一致。先做单 batch/短跑验证，再跑完整实验。

当前官方仓库没有 checkpoint release；同时官方 CMU YAML 所引用的三个实现类缺失。因此需要向作者索取以下材料，才能消除“公开重建”与“论文严格复现”之间的不可观测差异：

- `multiview_pose_3d_fuser.py`；
- `multiview_amass_dome_pose_3d_fuser` 与 `multiview_cmu_panoptic_pose_3d_fuser`；
- 论文 checkpoint 或逐 epoch validation 记录；
- MMPose 精确版本、config、checkpoint、person bbox 来源和图像畸变处理；
- 论文训练时使用的 MMPose 精确依赖版本及 person bbox 细节（当前公开重建已通过三角化结果验证等价性）；
- PFT 最后一层重复调用是否为预期行为。

### Phase C：数据门槛

- 正式复算 MHP 合成图的 COCO AP@0.75；目标接近论文 CMU-style `93%`。
- 检查 128,109 样本、20 cameras、人物房间范围和关键点映射。
- 对 train/test 同时做 ray reprojection、axis conversion、camera-center intersection 数值单测。

### Phase D：正式验收

同一 checkpoint、十组合平均、Absolute 指标下同时报告 All-KP 与 KP*。建议门槛：

- All-KP `<= 32.3 mm`（论文 30.8 + 1.5）
- KP* `<= 36.5 mm`（论文 35.0 + 1.5）

未通过前，“严格复现 RUMPL”统一改为“公开代码上的内部 baseline”。通过后，再从该 checkpoint 重新训练 confidence/geometry bias、课程和其他模块，旧结果不直接搬入论文主表。

## 决策顺序

1. 精确 CMU 2D pipeline 已通过：三角化 `42.864/43.821 mm`，与论文 `43.0/44.0 mm` 对齐。
2. 并行完成 `R0_public_original` 与 `R1_fix_pft_last_block` 的 20 epochs，统一计算十组合 All-KP/KP*；不能用单一最好组合替代平均值。
3. 比较固定 epoch 20 与 `model_best`，并保存逐组合结果和全部 checkpoint。
4. 若网络结果仍差，再单独修正 scheduler 顺序；不能与 PFT 修正同时混入首轮对照。
5. baseline 达到验收门槛后，才恢复 geometry/confidence bias 主实验并重做全部消融。

## 2026-07-23：基线冻结与偏置实验恢复

精确 CMU 测试集、同一 seed 0、20 epochs、十个两视角组合平均的最终对照如下。`All17/KP*` 均为 absolute MPJPE，单位 mm。

| 运行 | workers | PFT 公开重复末层 | scheduler 顺序修正 | final | model_best |
|---|---:|---:|---:|---:|---:|
| R0 | 4 | 是 | 否 | 31.410 / 36.075 | 31.342 / 36.306 |
| R1 | 4 | 否 | 否 | 31.549 / 36.555 | 31.440 / 36.467 |
| R2 | 4 | 是 | 是 | 31.134 / 35.871 | 30.978 / 36.011 |
| R3 | 4 | 否 | 是 | 31.366 / 36.386 | 31.128 / 36.196 |
| R4 | 16 | 是 | 否 | 31.175 / 36.137 | 30.925 / 35.542 |
| R5 | 16 | 是 | 是 | 31.039 / 35.991 | **30.885 / 35.506** |
| 论文 | - | 未公开 | 未公开 | - | **30.8 / 35.0** |

R5 `model_best` 与论文只差 `+0.085/+0.506 mm`，达到预设复现门槛，冻结为后续模块实验的可信公开代码基线。由于论文配置引用的三个实现文件未公开，准确表述仍是“公开代码的近似严格复现”，不能称为私有代码逐位复现。

从 R5 协议恢复 Geometry-Biased Transformer 公式移植，其他变量不变：公开 PFT 行为、修正 scheduler、workers 16、seed 0、相同训练与精确测试数据、从头训练 20 epochs、无蒸馏、无 token dropout、无辅助损失。VFT 每层注意力改为：

`softmax(QK^T/sqrt(d_k) + eta_l^2 M_conf - gamma_l^2 M_dist)`

- `M_conf`：每个 key 的 MMPose confidence，沿 query 维复制；融合 token 的 confidence 设为 0。
- `M_dist`：论文的两射线 Pluecker distance，包括 skew 与 parallel 分支；融合 token 行列默认置 0。
- `eta_l/gamma_l`：每个 VFT block 独立可学习，平方后保证非负。
- 严格公式运行：`conf_init=0.1, geom_init=1.0, fusion_geom=0`。
- RUMPL 扩展运行：`conf_init=0.1, geom_init=0.1, fusion_geom=1`，仅令融合 query 到各 view key 的距离等于该 view 对其他视角的平均不一致度；该项明确标为本文扩展。

实现前验收已覆盖：射线距离解析例、偏置对 attention 的正确增减方向、偏置系数梯度、关闭偏置时与 R5 快照输出逐元素相同、开启偏置后的完整模型前向。

### 首轮精确偏置结果

| 模型（model_best） | All17 | KP* | 相对 R5 All17 | 相对 R5 KP* |
|---|---:|---:|---:|---:|
| R5 frozen baseline | **30.885** | **35.506** | 0.000 | 0.000 |
| G0：论文公式移植，`geom_init=1.0` | 31.182 | 36.436 | +0.297 | +0.930 |
| G1：fusion 几何扩展，`geom_init=0.1` | 31.020 | 35.666 | +0.135 | +0.160 |

两组都未超过 R5，不能作为精度提升。G1 更接近基线，并在 `03_06、06_13、06_23、12_23` 等组合的至少一个指标下降，说明直接 fusion 几何路径值得保留为消融候选，但尚未形成平均提升。

系数检查发现 G0 的浅层 `gamma_l^2` 增长到约 `2.8-3.5`，而 G1 多数层为约 `0.12-0.46`。此外，G0 最后一层 geometry scale 完全保持初始化值：RUMPL 在 VFT 后只读取 fusion token，而原论文式矩阵将 fusion token 几何行列设为 0，因此最后层 view-view 几何更新无法影响被读取的 token。G1 的 fusion query 几何行提供了直接路径，最后层系数可以学习。这证明论文的全局 joint-view-time encoder 公式不能在不考虑 fusion token 信息流的情况下机械照搬。

下一轮使用 G1 作为中心做等成本拆分：confidence-only，以及 geometry-only + fusion query。两者均保持 R5 的其余协议，以判断退化来自单项偏置还是两项交互。

## 2026-07-23：严格基线上的 post-VFT 时序重启

旧 ST-VFT v1 在 VFT 前融合每视角 noisy ray，真实 CMU 上 All17/KP* 分别约退化 `0.8/1.2 mm`，不再原样重跑。旧 ST-VFT v2 的 post-VFT 结构在合成 held-out V2 上曾把 relative All17 从约 `62.07` 降至 `53.27 mm`，但它加载旧 `46.9 mm` baseline，写死有效宽度 `768`，使用 normalized input、Closest intersection 和旧 CMU 数据，不能迁移为正式结论。

新实现以 R5 `model_best` 为唯一 backbone：有效宽度 `256`，6D ray 先映射后拼 confidence，Camera-center intersection，轴变换 `[x,y,z] -> [x,-z,y]`。数据流为 `per-frame VFT -> per-joint 9-frame Temporal Transformer -> center residual -> original PFT`。Temporal 输出层零初始化；回归测试确认 temporal/no-temporal 完全相同，批量九帧 VFT 与直接中心帧 R5 的最大浮点差约 `1e-6`。R5 全部冻结并保持 eval，只训练 temporal 参数，因此首轮无需额外 backbone 微调控制组。

现有 AMASS clips 共 300 pkl、3000 clips、每段 27 帧和 20 个随机相机。旧 ray dataloader 已弃用，改为从保存的 MMPose 2D、confidence 和相机参数重算 R5-compatible rays。10-clip smoke test 中 no-temporal R5 的中位误差为 V2 `31.9 mm`、V5 `16.8 mm`，但少量 detector failure 将均值拉至 `92.6/38.0 mm`。因此首轮并行：

- T0：保留全部自然检测噪声，Huber center-pose loss；
- T1：仅要求窗口中心的跨视角 median OKS `>=0.5`，同一 Huber loss。

两组均为 L=9、2 temporal blocks、random K=2..5、seed0、20 epochs。训练完成前不加入 GBT、蒸馏、人工帧破坏或 joint finetune。正式 CMU 结论必须另行重建连续帧 exact-MMPose 测试集，并在相同中心帧上报告十个 V2 相机组合的 Absolute All17/KP* 均值以及 MPJVE/jitter。

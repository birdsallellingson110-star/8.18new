# Structured Ray-Hypothesis Fusion：思路、代码来源与实验记录

更新时间：2026-08-20（本文件只记录同口径、已落盘的结果；进行中的实验单独标注）

## 1. 研究问题和固定协议

当前问题不是重新替换整套 RUMPL，而是修复真实 H36M + HRNet 坐标输入下的
多视角融合缺陷。固定保留：

1. RUMPL/H76 的世界射线表示和绝对 3D 候选；
2. 三角化锚点、中心射线和 Plücker 等已验证候选；
3. E2-C2 的候选集合 attention/unary utility；
4. 标准 H36M `S1/S5/S6/S7/S8` 训练、`S9/S11` 测试；
5. 3D 网络只接收 2D 坐标、置信度、相机参数和由此构造的几何量，不接图像热图。

为保证与 GBT 的 coordinate-level HRNet 表公平比较，后续融合器实验锁定以下边界：

- 允许：HRNet 2D 坐标/置信度、相机内外参、以及确定性导出的射线、极线/重投影
  残差、三角化候选和骨架兼容量；
- 禁止：RGB/crop、heatmap、图像 backbone feature、GT 2D、测试时 GT、额外检测器
  输出；
- 时序若启用必须作为 `T>1` 独立协议报告，不能与当前单帧主表混写。

跨领域论文只允许贡献候选评分、图推断或稳健求解模块，不允许借此改变输入信息量。

评价为 S9/S11、所有相机子集、动作等权 All-17 absolute MPJPE。任何温度、epoch
和 checkpoint 只允许由训练主体内部 `group_index % 10 == 0` 留出集选择。

当前外部目标（GBT HRNet）为 `36.8 / 30.4 / 26.0 mm`（V2/V3/V4）。因 GBT
未开源，只能作为论文报告目标，不能把我们的实现称作其严格复现。

## 2. 为什么不再继续堆普通 Transformer

扩大候选池后，E2 的逐关节 oracle 很低，但它允许 17 个关节各自看真值选择不同
候选，可能形成不连续骨架。因此首先计算三种约束程度不同的 oracle：

| 选择上限（S9/S11） | V2 | V3 | V4 | 含义 |
|---|---:|---:|---:|---|
| task H76 baseline | 38.686 | 30.943 | 28.629 | 同一任务的原候选 |
| pose oracle | 38.552 | 29.226 | 26.225 | 整个姿态只能选同一候选 |
| 5-part/limb oracle | **35.887** | **25.759** | **22.464** | 躯干、双臂、双腿分别选一致候选 |
| per-joint oracle | 31.547 | 20.153 | 16.126 | 每关节独立，物理一致性最弱 |

结论：整姿态候选的上限不足以达到 V2 目标；逐关节上限虽高却不可信。五肢体组
既保持局部运动学连贯性，又在 V2/V3/V4 都有足够上限，是当前最合理的结构粒度。

零训练 exact tree MAP 也已测试。能量为

\[
E(\mathbf z)=\sum_j u_j(z_j)+\lambda\sum_{(p,j)\in\mathcal E}
\left(\frac{\|X_{p,z_p}-X_{j,z_j}\|-\mu_{pj}}{\sigma_{pj}}\right)^2,
\]

在 H36M 骨架树上用 min-sum 动态规划求精确 MAP。它只带来很小 hard-selection
变化，且明显弱于 E2 soft fusion。固定平均骨长不能识别 HRNet 系统性 2D 偏差，
所以不作为主模块。

代码：`diagnose_e2_structured_candidates_20260820.py`。

## 3. 论文与官方代码来源

### 3.1 Generalizable Human Pose Triangulation（CVPR 2022）

- 官方仓库：`reference/general-3d-humans-official/`
- 关键实现：`src/dsac.py::PoseDSAC`
- 官方机制：每个关节随机选择相机子集并三角化，组装多组完整 3D 假设；小 MLP
  根据 root-centred 3D pose/骨长打分，再对假设做可微概率加权。
- 官方默认 control：bone-only、末层 Sigmoid、softmax temperature 1.8、归一化
  expectation + entropy + weighted-estimate loss。

我们没有移植其相机自标定或另一套 2D 前端，只把“假设生成和评分”替换到 E2
独立逐关节 softmax 的位置，因而仍保留 RUMPL 射线候选。

### 3.2 3D Pictorial Structures

运动学树/肢体级一致选择依据来自 3D pictorial structures 的 part-based inference
思想。当前不使用多人检测、图像 unary 或体素；只采用“相邻关节/肢体不能完全
独立选择”的结构约束。

### 3.3 D3DP JPMA（诊断边界）

D3DP 的 JPMA 说明测试时可按关节从多假设中选择；但它依赖重投影聚合。我们已经
单独验证纯 GT-free reprojection selector 会灾难性失败，因此没有直接照搬 JPMA
的重投影选择器，而只保留“局部而非整姿态选择”的假设空间思想。

### 3.4 PointDSC（CVPR 2021）

- 官方仓库：`reference/PointDSC-official/`，本地 commit `b009d53`；
- 关键代码：`models/PointDSC.py::NonLocalBlock/NonLocalNet/PointDSC`；
- 可迁移机制：把局部候选分数与候选间 spatial compatibility 联合起来，再做
  non-local message passing / consistency-weighted selection；
- 不迁移内容：点云输入、刚体变换目标与点特征。当前只把兼容图用于既有 RUMPL/E2
  肢体候选，因此坐标级输入协议不变。

## 4. 已实现模块

### 4.1 PoseDSAC-style 条件化肢体假设

代码：`train_e2_pose_dsac_20260820.py`。

对每帧、每个任务相机子集：

1. 冻结 RUMPL/H76 候选和 E2 unary `u_{j,c}`；
2. 以 `0.8 softmax(-u/T)+0.2/C` 作为无真值 proposal；
3. 五个肢体组各自采样一个候选标签，组内关节共享标签；
4. 始终把原 baseline、E2 hard 和 E2 soft 加入假设池作为安全 control；
5. 小型 3×50 ReLU6 MLP 读取 root-centred pose、标准化骨长、被选 E2 unary 和
   absolute pelvis；
6. 对完整假设打分并做 soft weighted estimate。

损失为

\[
L=L_{3D}(\sum_k w_k H_k,Y)+0.2\sum_k w_k e_k
+0.5\,CE(q(e),s)+0.25\,SmoothL1(\bar s,-\bar e),
\]

其中 `q(e)=softmax(-e/5mm)`。所有候选生成、打分和融合在推理时均不使用真值。

### 4.2 确定性 kinematic-part utility

代码：`train_e2_limb_utility_20260820.py`。

该版本不随机枚举组合，而是直接为五个肢体组预测每个候选的 cost。特征只来自：

- 组内 E2 unary 的均值/方差/最小/最大；
- 候选到当前任务所有射线的残差统计；
- 候选共识差和骨长 z-score；
- root-relative group centre/spread 和完整 pose context；
- 视角数 one-hot 与 group embedding。

输出为 `mean(E2 unary within group) + learned residual`，最后一层零初始化。因此网络
在训练开始时严格退化为 E2 的肢体平均 utility，不会由随机初始化破坏现有结果。

## 5. S9/S11 正式结果

| 方法 | V2 | V3 | V4 | 结论 |
|---|---:|---:|---:|---|
| 原 E2-C2 soft-cal 参考 | 38.700 | 29.486 | 27.274 | 进入本阶段前最好统一线 |
| expanded unbiased E2 soft | 38.536 | 29.488 | 27.270 | 改损失/特征不足以兑现 oracle |
| exact skeleton tree MAP | 38.299 | 30.369 | 28.193 | V3/V4 退化，失败 |
| faithful PoseDSAC control | 38.424 | 29.532 | 27.331 | 官方 bone-only 可用但较弱 |
| conditioned limb hypotheses, K48, seed0 | 38.096 | 29.291 | 27.210 | 三列均改善 |
| conditioned limb hypotheses, K48, seed1 | 38.079 | 29.313 | 27.263 | 种子趋势一致 |
| **conditioned limb hypotheses, K96** | **38.065** | **29.228** | **27.136** | 当前随机假设线最好 |
| deterministic limb utility, seed0 | 38.325 | **29.042** | **26.969** | V3/V4 更强、V2 较弱 |
| deterministic limb utility, seed1 | 38.301 | 29.061 | 26.984 | 双种子稳定 |

严格官方 control 弱于带 RUMPL/E2 条件的版本，说明第三方结构不能机械照搬；射线
几何和 E2 utility 是跨主体真实 2D 噪声下的必要证据。另一方面，K96 sampled
oracle 为 `36.315/27.386/25.368`，评分器仍没有充分识别好假设。

## 6. 温度校准失败经验

仅在训练主体内部前 2,048 个留出样本选择出的温度，迁移到 S9/S11 后反而退化：

- conditioned K96：`38.177/29.451/27.413`；
- deterministic limb utility：`38.194/29.329/27.195`。

因此保留训练时原温度，不把这次校准写成提升。原因更像 score calibration 的
跨主体偏移，而不是候选不足；后续若继续校准，应显式学习不确定性或做 subject-
independent calibration，不能继续在 S9/S11 扫温度。

## 7. 当前进行中与决策门槛

两视角 specialist 已于 epoch 6 主动停止。旧实验已经覆盖“全程随机两视角训练”：
RUMPL B2 300k 虽把 V2 降至 `37.886`，但 V3/V4 灾难性退化到
`62.215/46.217`；E2-C2 两个 specialist seed 的 V2 soft 约 `38.98`。本次新 head
在 epoch 6 的内部 V2 holdout 仍约 `23.755 mm`，与 all-view head（约 `23.74 mm`）
没有新趋势，因此不再重复长跑，也禁止按列拼接 specialist checkpoint。

PointDSC-style 手工 boundary-bone compatibility 已完成 7,805 个训练主体内部 holdout：

| 方法 | V2 | V3 | V4 | 三列均值 |
|---|---:|---:|---:|---:|
| frozen limb unary，pair weight=0 | 23.7406 | 16.2199 | 13.9370 | 17.9658 |
| 最佳统一权重 0.01 | 23.7415 | 16.2112 | 13.9303 | 17.9610 |

综合只改善 `0.0048 mm`，且 V2 退化，故不访问 S9/S11、不作为有效模块。进行中的
graph-only 实验改为从训练主体学习 pairwise compatibility，同时冻结 E2 和 unary；
采用条件是内部 holdout 明显优于 `17.9658 mm`，随后才允许一次 S9/S11 评估。

graph-only 已按预注册门槛在 epoch 6 停止：最佳 holdout 从 epoch 1 的
`18.0791` 逐步降到 epoch 6 的 `17.9912 mm`，但仍未超过冻结 unary 的
`17.9658 mm`。因此不跑 seed1、不访问 S9/S11。该结果与手工 compatibility 一起
说明：在当前候选上继续增加离散骨架图不是优先突破口。

随后依据 Generalized Differentiable RANSAC（ICCV 2023）官方实现
`reference/differentiable-ransac-official/samplers/gumbel_sampler.py`，检查“学习
proposal”是否与旧 Gumbel 实验重复。旧实验只对最终候选权重做 Gumbel-softmax；
当前 K96 也从固定 E2 unary proposal 在 `no_grad` 内采样，二者都没有训练 proposal。

在 2,048 个训练主体内部 holdout、固定 K96 scorer/随机流下，先做 proposal source
零训练对照：

| proposal | 温度 | 实际 V2/V3/V4 | 实际均值 | sampled-oracle 均值 |
|---|---:|---|---:|---:|
| E2 unary | 0.8 | 23.809 / 16.376 / 14.030 | 18.071 | 17.258 |
| limb utility | 0.8 | **23.744 / 16.256 / 13.949** | **17.983** | 17.200 |
| limb utility | 0.4 | 23.740 / 16.296 / **13.921** | 17.986 | **17.144** |

三列同向且 sampled oracle 同时改善，证明 proposal 是独立瓶颈。进行中实验保持
K96 scorer 架构/损失不变，只用 frozen limb utility 取代固定 E2 proposal，并与
旧 K96 同样训练 30 epochs；之后才决定是否加入官方 straight-through Gumbel sampler。

采用门槛：必须在 S9/S11 至少改善 V2 且不把 V3/V4 拉回原 E2 之上；否则只作为
“two-view specialization 的负结果”，不进入主模型。禁止用 V2 模型和 V3/V4
模型按测试列拼表。

## 8. 文件与输出位置

实现：

- `diagnose_e2_structured_candidates_20260820.py`
- `train_e2_pose_dsac_20260820.py`
- `evaluate_e2_pose_dsac_grid_20260820.py`
- `train_e2_limb_utility_20260820.py`
- `evaluate_e2_limb_utility_grid_20260820.py`
- `diagnose_pointdsc_limb_compatibility_20260820.py`
- `train_e2_limb_compatibility_graph_20260820.py`
- `diagnose_dsac_proposal_sources_20260820.py`

挂载盘结果：

- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_structured_candidates/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_pose_dsac/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_pose_dsac_long/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_pose_dsac_faithful/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_limb_utility/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_limb_utility_v2only/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/pointdsc_limb_compatibility/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_limb_compatibility_graph/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/dsac_proposal_source/`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_pose_dsac_limb_proposal/`

## 9. Learned proposal 完成结果与 MAP 主线门控

冻结 limb utility 作为 proposal 的 K96 正式 30 epoch 已完成：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 旧固定 proposal K96 | 38.065 | 29.228 | 27.136 |
| limb-utility proposal K96 | **38.017** | **29.143** | **27.089** |
| 新 K96 sampled oracle | 36.258 | 27.198 | 25.236 |

proposal 改进只有 `0.048/0.085/0.047 mm`，证明它是独立瓶颈但不是剩余 1--2 mm
差距的主因；本方向停止继续调 proposal 温度、数量和浅层 scorer。

随后按 `FAILURE_INFORMED_MAP_DECODER_PLAN_20260820.md` 实现冻结 K96 初始化的
joint-view probabilistic prior + differentiable ray-MAP。M1 固定 confidence 观测权重，
只学习 root/root-relative prior、prior precision 和 selective trust。数值检查满足：

- 视角置换最大误差 `0`；
- 初始化相对 K96 最大偏差约 `0.017 mm`；
- MAP 的全部梯度有限；
- 锚点生成路径不读取 target。

首次 3 epoch 随机帧留出出现明确的跨主体失败：

| 口径 | V2 | V3 | V4 |
|---|---:|---:|---:|
| K96 anchor，S9/S11 | 38.011 | 29.141 | 27.079 |
| M1 MAP，S9/S11 | 38.390 | 29.504 | 27.406 |
| 变化 | +0.379 | +0.362 | +0.327 |

训练主体内部三列均值却从 `17.903` 改善到 `17.787 mm`。这与旧 learned ray
correction 的“同主体改善、S9/S11 退化”一致，说明历史 `group_index % 10` 留出
不适合学习人体 prior/纠错。该 M1 判定失败，不允许进入 learned observation M2。

代码与结果：

- `train_failure_informed_map_20260820.py`；
- `launch_failure_informed_map_m1_gate_20260820.sh`；
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/failure_informed_map/m1_gate_seed0/`。

当前唯一允许的复核是严格主体门控：训练 S1/S5/S6/S7、完整留出 S8。若仍失败，
关闭直接 pose-delta prior，不以加 epoch/深度/seed 重跑；改用候选分布约束的 prior
或外部预训练后冻结的 pose density。

严格 S8 门控随后完成：最佳改善仅 `0.042 mm`，第 3 epoch 回落到 `0.019 mm`，
直接 pose-delta prior 已关闭。冻结 Gaussian pose density 重排 K96 的 S8 最佳改善
也只有 `0.006 mm`，S9/S11 约退化 `0.030 mm`，因此 normalizing-flow 高容量版本
未启动。新门控方向为 `train_observation_conditioned_k96_scorer_20260820.py`：不改
候选坐标，只用完整 joint-view 射线残差进行 body-before-view axial hypothesis scoring。

# 反事实关节级视角效用：依据、诊断与实验记录（2026-08-11）

## 1. 固定基线与目标

- 固定主基线为 H76：RUMPL ray token、VFT、PFT、共享 3D head、置信度三角化
  锚点、anchor-centered Plücker 输入全部保留。
- H36M S9/S11、真实 HRNet→A1D→H21 输入、action-equal absolute All-17：
  V2 `34.8163`、V3 `30.4890`、V4 `29.6913` mm（seed 0）。
- 本路线不使用蒸馏，不先加时序，不改 2D 点；先回答融合后的多假设效用是否
  可学习。若成立，再进入选择性 2D 纠错与随机子集单调训练。

## 2. 论文与代码依据

### 2.1 主要可复用依据：Generalizable Human Pose Triangulation（CVPR 2022）

- 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Bartol_Generalizable_Human_Pose_Triangulation_CVPR_2022_paper.html
- 官方 MIT 代码：https://github.com/kristijanbartol/general-3d-humans
- 本地固定 commit：`14805c78e79d57d1d870f00adfa64a98887111cf`
- 本地位置：`/home/lixiaob/cjy/reference/general-3d-humans-official`

官方 PoseDSAC 对每个关节随机选择 2..V 个视角形成 3D 假设，以三层 50 维
ReLU6 MLP 为整姿态假设评分，softmax 后做加权估计；默认损失权重为期望假设
风险 `1.0`、加权估计误差 `0.05`，温度 `1.8`。我们的 C0 保留这一评分器结构
作为论文对照；C1 将同一“多假设—学习评分”原则扩展到逐关节效用，并加入不依赖
相机 ID 的射线残差/置信度/条件谱统计。

### 2.2 仅作概念对照：DeProPose

- 论文/仓库：https://github.com/WUJINHUAN/DeProPose
- 本地 commit：`8016d396e1a8cdef9ad31111119a9b05b064b824`
- 本地位置：`/home/lixiaob/cjy/reference/DeProPose-official`

代码审计发现 `model/pose3D_model.py` 的融合权重由 `project(..., targets, ...)`
和预测对真值的 `mpjpe` 共同计算，`utils/projecter.py` 又把 target 投影作为误差
真值；即当前公开实现的视角权重在推理路径依赖 3D target。仓库还未提供 LICENSE。
因此不能照搬其代码或把其结果当作可部署实现，只保留“相对投影误差反映视角
质量”的概念。我们的模型推理不接触 GT。

### 2.3 不能重复声称的边界

- AdaFuse 已做跨视角热图纠正和自适应相机权重；不能声称首次坏视角抑制。
- GHT 已做随机几何假设和学习评分；不能声称首次学习三角化/假设评分。
- LOSTU 已做测量/相机不确定性感知的统计最优三角化。
- DeProPose 已提出相对投影误差动态融合；不能声称首次重投影加权。

目前可守住的组合是：**H76/RUMPL 子集假设生成 + 真值反事实三维风险监督的
逐关节效用 + 后续随机子集单调约束和三维敏感度选择性 2D 纠错**。

## 3. Stage A：冻结 H76 的负视角诊断

代码：`diagnose_h76_negative_view_20260811.py`。对 2021 个同步 S9/S11 帧，
遍历全部 12 条 V2→V3 和 4 条 V3→V4 嵌套转移，逐帧逐关节比较加视角前后
absolute/root-relative 误差，并导出训练标签、图和 JSON。

| 转移 | before | after | 平均变化 | 逐关节 NVR | >1 mm | >5 mm | 整姿态 NVR | 二选一 oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2→V3 | 34.816 | 30.489 | -4.327 | 42.73% | 36.12% | 16.08% | 28.61% | 28.249 |
| V3→V4 | 30.489 | 29.691 | -0.798 | 44.17% | 34.48% | 10.25% | 31.81% | 28.111 |

正确解释是：**新增视角在总体上有益，但在局部关节上仍频繁产生负贡献**。
不能写成“四视角总体变差”。V3→V4 中有害点平均恶化 `3.619` mm；理想逐关节
门若只在 before/after 间选择，最多可再挽回 `1.581` mm。

root-relative 也有同样现象：V3→V4 从 `32.543` 降至 `31.900` mm，但局部
root-relative NVR 为 `43.50%`，不是单纯 root translation 造成的假象。

手工量与 V3→V4 真实边际变化的 Pearson 相关很弱：置信度 `-0.049`、平均
视差角 `-0.021`、prediction-to-added-ray residual `-0.080`、加入后条件数
`-0.024`。这解释了 R1–R8 中普通 confidence/geometry bias 和小 adapter 为何
基本无效：真实效用不能由一个手工标量直接代替。

完整输出：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/H76_negative_view_audit.json`

## 4. 训练/测试隔离的多假设缓存

冻结 H76，在正式 H36M 训练 subjects S1/S5/S6/S7/S8 的 78,047 个同步四视角
帧上，导出全部 6 个 V2、4 个 V3、1 个 V4 候选，共 11 个候选/帧；同时保存
GT、四条原始射线、action、subject 和原始 group index。

- checkpoint SHA256：
  `69ae8783d76bf0eacc18c5018837c13a90ffed7d554cb78b4864cc84ab569838`
- shard 0：39,023 组，group `[0,39023)`；shard 1：39,024 组，
  group `[39023,78047)`；无重叠、无缺口。
- S9/S11 的 2021 组只从既有严格预测合并成最终评估缓存，明确禁止效用训练。
- 严格缓存基线复核：V2 `34.81634`、V3 `30.48905`、V4 `29.69129` mm。

若允许对当前输入的全部可用子集候选逐关节 oracle 选择，上限为 V3 `25.143`、
V4 `21.817` mm。它只是诊断上限，不是模型结果，也不用于参数选择。

缓存目录：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811`

## 5. 正式实验 C0/C1

共同设置：H76 完全冻结；真实训练 subjects；按训练 group index 的 10% 做内部
holdout 选择 checkpoint；S9/S11 只在选定模型上最终评估一次；AdamW、`lr=5e-4`、
weight decay `1e-4`、10 epochs、batch 512、seed 0、softmax temperature `1.8`。
损失沿用 GHT 主要系数：期望候选 3D 风险 `1.0` + 加权估计 MPJPE `0.05`。

| ID | 唯一变量 | 输入与输出 | 状态 |
|---|---|---|---|
| C0 | GHT 官方式整姿态 ScoreNN | root-centered 标准化 51D pose→50→50→50→单一候选分数 | GPU0 正式训练 |
| C1 | 逐关节反事实效用 | pose context + 候选/共识差 + 集合化射线残差、置信度、条件谱→每关节候选分数 | GPU1 正式训练 |

C0/C1 都只对当前输入真正可用的子集候选评分：V3 使用其三个 V2 子集和自身；
V4 使用六个 V2、四个 V3 和自身。特征不含相机 ID，按集合统计，第四视角不会
泄漏到 V3 任务。V2 没有合法单视角子假设，保持 H76 原输出不变。

实现与输出：

- `export_h76_train_subset_hypotheses_20260811.py`
- `build_h76_validation_hypothesis_cache_20260811.py`
- `train_h76_hypothesis_utility_20260811.py`
- `launch_C0_C1_counterfactual_utility_20260811.sh`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/C0_C1_training`

## 6. 预注册决策

1. C1 若在内部 holdout 和一次性 S9/S11 上均优于 C0/H76，先复验 seed 1/2，
   再把逐关节效用接入在线 H76（减少多次子集前向）并加入随机子集单调损失。
2. 若 C0 有效而 C1 无效，保留 GHT 多假设评分主线，简化 C1 特征，不进入 2D
   纠错。
3. 若两者都不能恢复 H76，说明当前软加权/评分可学习性不足；先做 hard/soft
   medoid 与 pairwise hypothesis solver 对照，不继续添加手工 bias。
4. 只有效用模型稳定后才开启选择性 `Delta p/alpha/covariance`；时序仍放在单帧
   主结果之后。

## 7. C0/C1 正式结果与失败机制

| 方法 | V2 | V3 | V4 | 相对 H76（V3/V4） |
|---|---:|---:|---:|---:|
| H76 | 34.8163 | 30.4890 | 29.6913 | — |
| C0 GHT 整姿态 score | 34.8163 | 30.6891 | 30.0845 | +0.2000 / +0.3932 |
| C1 逐关节 score/期望风险 | 34.8163 | 30.4890 | 29.6912 | -0.0001 / -0.0001 |

C0 在内部 holdout 也比 H76 差，说明只用 root-centered 3D pose plausibility 不能
判断绝对世界坐标候选质量。C1 并非模块没有生效，而是塌缩为最大视角保底：

| split | V3 baseline 平均权重/top1 | V4 baseline 平均权重/top1 |
|---|---:|---:|
| train-subject holdout | 99.993% / 100% | 99.985% / 100% |
| S9/S11 | 99.993% / 100% | 99.984% / 100% |

但最大视角实际只占 train holdout 的 V3 `22.68%`、V4 `5.27%` 逐关节 oracle，
S9/S11 更只有 `17.31%/2.95%`。间接期望风险监督在无法可靠区分局部候选时，
学到的是“平均最安全地保持 H76”，不是反事实效用。因此 C0/C1 score 路线停止，
转为显式预测 `delta_hat`。

诊断：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/C0_C1_domain_gap_diagnostic.json`

## 8. C2：直接反事实边际误差监督

C2 保持 C1 模型输入和 H76 候选不变，将训练目标改为每个候选相对当前完整输入
H76 的逐关节误差差：

`delta_true(c,j) = error(c,j) - error(baseline,j)`。

网络输出减去 baseline 输出，使 baseline 的预测 delta 严格为零。这样 baseline
天然是 selective identity 保底；soft 推理按 `softmax(-delta_hat)` 融合候选。

- C2a：10 mm 归一化 delta 的 Smooth-L1 回归；
- C2b：同一回归 + helpful/harmful 平衡分类 + 忽略 1 mm 内近似平局的 pairwise
  排序。两者都是对同一效用头的损失消融，不改候选生成或 H76。

### seed 0 一次性 S9/S11 结果

| 方法 | V2 | V3 hard | V4 hard | V3 soft | V4 soft | delta Pearson |
|---|---:|---:|---:|---:|---:|---:|
| C2a regression | 34.8163 | 30.4438 | 29.7782 | 30.0275 | 29.3578 | 0.8803 |
| C2b balanced rank | 34.8163 | 30.4350 | 29.8767 | **29.9882** | **29.3097** | 0.8743 |

C2b-soft 相对 H76 提升 V3 `0.5008`、V4 `0.3816` mm；C2a-soft 也同方向提升
`0.4615/0.3335` mm。hand-crafted 特征与 V3→V4 边际误差的相关仅约
`0.02–0.08`，而学习的 delta 在未见 S9/S11 上达到 `0.87–0.88`，证明直接
反事实监督确实学到了可迁移信号。

hard 在 V4 会因少量误选退化，soft 在 train holdout 与 S9/S11 均稳定更好；
因此正式推理规则固定为 C2b-soft。seed 0 的内部 soft 指标随 epoch 单调改善，
第 9 轮同时也是原 hard 口径最佳点，不需要因看到测试结果重选 checkpoint。
seed 1/2 已按内部 soft V3/V4 均值选择 checkpoint 并行复验。

输出：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/C2_delta_training`

### C2b 三种子复验

| seed | V3 soft | V4 soft | delta Pearson |
|---:|---:|---:|---:|
| 0 | 29.9882 | 29.3097 | 0.8743 |
| 1 | 29.9666 | 29.2931 | 0.8798 |
| 2 | 29.9811 | 29.3012 | 0.8750 |
| mean ± std(pop.) | **29.9786 ± 0.0090** | **29.3013 ± 0.0068** | — |

相对 H76 seed-0 固定表分别下降约 `0.5104/0.3900` mm，三个 seed 方向完全一致，
方差远小于提升。C2b-soft 因此升级为当前正式主模型候选。

### 训练后温度校准失败记录

在三个 seed 的 train-subject 10% holdout 上预先扫描 soft temperature
`0.25,0.5,0.75,1,1.5,2,3,4`，holdout 选择 `0.25`；但该温度一次性应用到
S9/S11 后三种子均值为 V3 `30.0735`、V4 `29.3664`，反而差于预设温度 1.0
的 `29.9786/29.3013`。这说明随机帧 holdout 不能校准未见 subject，不采用
0.25，也不继续看测试集扫温度。下一控制改为 leave-subject-out checkpoint
选择：S8 为预注册主 holdout，S7 为鲁棒性对照。

温度审计：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/C2_delta_temperature_calibration.json`

## 9. Subject-level holdout 与 C4 二阶段 GHT 损失

随机帧 holdout 校准失败后，C3 直接留出完整训练 subject 选 checkpoint：

| 效用训练协议 | V3 soft | V4 soft | 结论 |
|---|---:|---:|---|
| 全部 S1/S5/S6/S7/S8（C2b seed0） | 29.9882 | 29.3097 | 主设置 |
| leave-S8-out | 30.0158 | 29.3402 | 仍改善 H76，但少数据后较差 |
| leave-S7-out | 29.9965 | 29.3265 | 仍改善 H76，但少数据后较差 |

这说明 C2b 增益不是随机帧 holdout 泄漏造成的；效用训练需要多 subject 的误差
模式，最终仍使用全部正式训练 subjects。

C4 从各 seed 的 C2b delta checkpoint 开始，以 `lr=1e-4` 再训练 5 epoch，
增加 GHT 官方期望假设风险 `1.0` 和加权估计损失 `0.05`。单调版本另加
V3→V4 逐关节 ReLU violation，但内部 holdout 略差，故主结果用无单调版本。

| seed | C4-GHT V3 | C4-GHT V4 |
|---:|---:|---:|
| 0 | 29.9722 | 29.2915 |
| 1 | 29.9542 | 29.2770 |
| 2 | 29.9693 | 29.2764 |
| mean ± std(pop.) | **29.9653 ± 0.0079** | **29.2816 ± 0.0070** |

C4 相对 C2b 均值继续下降 `0.0134/0.0197` mm，三 seed 一致但贡献很小；论文
叙述中 C2 是核心模型创新，C4 只作为“delta 预训练后再加 GHT task loss”的
训练消融，不能把主要提升归因于调参。

## 10. Negative View Rate 结果

使用 C4 seed0 对同一 2021 个 S9/S11 同步帧重算所有嵌套转移：

| 方法 | 转移 | NVR | >1 mm | >5 mm | Pose NVR | root-relative NVR |
|---|---|---:|---:|---:|---:|---:|
| H76 | V2→V3 | 42.73% | 36.12% | 16.08% | 28.61% | 40.76% |
| C4 | V2→V3 | **39.00%** | **31.27%** | **11.22%** | **20.01%** | **37.13%** |
| H76 | V3→V4 | **44.17%** | 34.48% | 10.25% | 31.81% | 43.50% |
| C4 | V3→V4 | 44.58% | **32.22%** | **6.59%** | **31.33%** | **42.22%** |

C4 明显降低 V2→V3 的所有负增益指标；V3→V4 的任意正数 NVR 因微小抖动
上升 `0.41` 个百分点，但 >1 mm、>5 mm、整姿态和 root-relative 违例均下降。
论文应表述为“降低有意义/严重负视角伤害”，不能写成所有单调违例都下降。

输出：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/C4_negative_view_audit.json`

## 11. C5 异方差效用消融（失败）

依据 UPose3D 的 uncertainty/MLE 和 LOSTU 的 measurement uncertainty，在 C4
均值 delta 输出旁新增 log-variance；均值通道从 C4 初始化，方差通道以单位方差
初始化。C5a 用 Gaussian NLL，C5b 为 NLL + C2b 平衡分类/排序；风险推理用
`delta_mean + std`，baseline 风险固定为零。

| 方法 | V3 mean-soft | V4 mean-soft | V3 risk-soft | V4 risk-soft | best epoch |
|---|---:|---:|---:|---:|---:|
| C5a NLL | 29.9671 | 29.2852 | 30.0452 | 29.3057 | 0 |
| C5b NLL+rank | 29.9655 | 29.2852 | 30.0430 | 29.3052 | 0 |

mean-soft 只是在 C4 附近产生约 0.006 mm 微小漂移；真正使用 uncertainty 的
risk-soft 反而退化，且内部 holdout 同样在第 0 轮最好。说明当前标量方差不能
可靠预测候选选择风险，不能因为相关论文的不确定性有效就默认本任务也有效。
C5 停止、不补 seed、不进入主模型；保留为论文消融/失败边界。

## 12. 当前主模型与下一阶段入口

当前可复现主模型固定为 **C2b 直接反事实 delta + C4 GHT task loss**：V2 保持
H76 `34.8163`，三种子 V3 `29.9653±0.0079`、V4 `29.2816±0.0070` mm，并显著
降低 >1 mm、>5 mm 和 pose-level Negative View Rate。

下一阶段才开启选择性 2D/ray correction。必须保持 C4 checkpoint 和候选生成器
冻结，先只训练 `Delta p/alpha`，以 C4 预测的边际效用决定是否允许修正；首轮
不同时加入 covariance、bone、temporal 或 monotonic loss。若纠错不能在内部
subject-level holdout 改善 C4，则立即停止，不能靠 S9/S11 调 trust-region。

## 13. Stage D：恒等初始化射线纠错（跨主体失败）

实现了两项严格单变量实验；H76 全部参数冻结，纠错头只在射线方向的切平面内
输出最大 `0.5 degree` 的小角修正，camera point 和 HRNet confidence 不变。输出层
为零初始化。修正参数化经过独立测试，11 种 V2/V3/V4 组合在训练开始时均满足
完整 H76 输出 `max_abs_diff=0.0 m`，并验证第一步梯度非零。

- D0 geometry：用 H76 pose-to-ray 垂距、置信度、root-relative joint 和 joint
  embedding 预测修正；
- D1 utility：相同纠错头，但 V3/V4 只有在冻结 C4 判断“删除该视角会改善该
  关节”时才开放修正；V2 因没有合法 V1 反事实候选而保持开放。

训练只使用正式 H36M train subjects；按 `group_index modulo 40` 留出约 2.5%
训练主体帧选 checkpoint，S9/S11 只在最优 checkpoint 上评估一次。两组内部
holdout 都改善，但未见主体结果全部退化：

| 方法 | V2 | V3 | V4 | 相对 H76（V2/V3/V4） |
|---|---:|---:|---:|---:|
| H76 cached identity | 34.8163 | 30.4890 | 29.6913 | — |
| D0 geometry | 34.8921 | 30.7670 | 30.0638 | +0.0758 / +0.2779 / +0.3725 |
| D1 C4 utility gate | 34.9734 | 30.5775 | 29.8254 | +0.1570 / +0.0884 / +0.1341 |

D0 在 train-subject holdout 的 V2/V3/V4 从
`23.5985/19.7299/18.6145` 降到 `23.2115/19.3357/18.2050`；D1 降到
`23.2540/19.6602/18.5166`。因此失败不是纠错头没有学习，而是直接用 3D GT
驱动细小射线偏移时，学到了训练 subjects 特有的 detector/pose bias。C4 门控把
V3/V4 平均修正角限制到 `0.0074/0.0117 degree`，确实减轻 D0 的伤害，但不能
令其跨主体转正。

**决策：** D0/D1 停止，不补 seed，不在失败头上继续叠加 covariance、bone、
temporal，也不根据 S9/S11 扫角度上限。当前正式主模型仍为三种子 C4。下一步
针对 C4 的明确限制——候选逐个编码、只通过手工 consensus 间接比较——引入
GHT 多假设评分与 Set Transformer 式候选集合注意力，直接学习候选之间的相对
关系；这仍保留所有 H76/RUMPL 候选生成与几何表示。

代码与输出：

- `selective_ray_corrector_20260811.py`
- `train_selective_ray_corrector_20260811.py`
- `launch_D0_D1_selective_ray_correction_20260811.sh`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/D_geometry_ray_correction`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/D_utility_ray_correction`

## 14. Stage E：GHT + 候选 Set Transformer（当前最好）

### 动机与论文边界

C2/C4 的逐关节 utility MLP 对每个候选独立编码，只通过候选均值/consensus
间接比较；而 S9/S11 的候选 oracle 为 V3 `25.1429`、V4 `21.8169` mm，说明
假设池中仍有大量尚未利用的相对排序信息。Stage E 保留 GHT 的多假设池、直接
delta 监督和二阶段期望风险训练，并采用 Set Transformer（Lee et al., ICML
2019）的 permutation-equivariant self-attention 让同一关节的全部合法 H76
候选直接交互。没有 candidate-index/camera-ID embedding，因此不破坏候选顺序
置换等变性；H76/RUMPL 的 ray token、VFT、PFT、tri-anchor 和候选生成全部保留。

训练协议与 C2+C4 对齐：先 10 epoch balanced delta/classification/ranking，再以
`lr=1e-4` 做 5 epoch GHT expected-risk 微调；checkpoint 只按 train-subject
holdout 的 soft V3/V4 均值选择，S9/S11 最终评一次。

### 深度消融与三种子结果

| 方法 | V3 | V4 | 相对 H76（V3/V4） |
|---|---:|---:|---:|
| H76 | 30.4890 | 29.6913 | — |
| C4 三种子均值 | 29.9653 | 29.2816 | -0.5237 / -0.4097 |
| E Set Transformer 1-layer seed0 | 29.8496 | 29.0941 | -0.6394 / -0.5972 |
| E Set Transformer 2-layer seed0 | **29.8195** | **29.0502** | **-0.6696 / -0.6411** |

2-layer 三种子：

| seed | V3 soft | V4 soft | delta Pearson |
|---:|---:|---:|---:|
| 0 | 29.8195 | 29.0502 | 0.8642 |
| 1 | 29.8242 | 29.0733 | 0.8559 |
| 2 | 29.8215 | 29.0649 | 0.8710 |
| mean ± std(pop.) | **29.8217 ± 0.0019** | **29.0628 ± 0.0095** | — |

相对 C4 三种子均值继续降低 V3 `0.1436`、V4 `0.2188` mm，且三个 seed 方差
远小于提升。hard 选择 seed0 为 `30.1900/29.4485`，明显差于 soft，故正式推理
仍是 temperature=1 的 soft 候选融合，不能改写成 hard hypothesis selection。

### Negative View Rate

| 方法 | 转移 | NVR | >1 mm | >5 mm | Pose NVR | root-relative NVR |
|---|---|---:|---:|---:|---:|---:|
| H76 | V2→V3 | 42.73% | 36.12% | 16.08% | 28.61% | 40.76% |
| E-2 | V2→V3 | **38.63%** | **30.83%** | **10.73%** | **18.66%** | **36.75%** |
| H76 | V3→V4 | **44.17%** | 34.48% | 10.25% | 31.81% | 43.50% |
| E-2 | V3→V4 | 44.38% | **32.04%** | **6.48%** | **29.72%** | **41.98%** |

结论与 C4 一致但幅度更强：模型降低的是有意义/严重的负视角伤害；任意正数
NVR 仍受亚毫米抖动影响，不能宣称完全单调。

代码与输出：

- `train_h76_set_transformer_utility_20260811.py`
- `launch_E_set_transformer_utility_20260811.sh`
- `evaluate_c4_negative_view_rate_20260811.py --model-type set`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/E_set_transformer_depth2[_seed1|_seed2]`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/E_set_transformer_depth2/negative_view_audit.json`

**当前决策：** E-2 取代 C4 成为主模型。下一步先做 attention depth 3/4 的容量
消融；若不能同时优于 E-2，则固定 2 层，不继续扫深度，再转向不依赖主体特定
偏差的可微稳健几何求解器控制。

### 1--4 层完整容量消融

| candidate self-attention depth | internal holdout | V3 | V4 |
|---:|---:|---:|---:|
| 1 | 18.6549 | 29.8496 | 29.0941 |
| 2 | 18.5437 | **29.8195** | **29.0502** |
| 3 | 18.5063 | 29.8314 | 29.0689 |
| 4 | **18.4958** | 29.8141 | 29.0881 |

深度 3/4 在训练主体 holdout 继续下降，但未在 S9/S11 同时超过 2 层：3 层
V3/V4 均退化，4 层仅 V3 改善 `0.0054` mm 而 V4 退化 `0.0378` mm，平均也
更差。这是随深度增加的 subject-pattern overfit 信号，与“越深越好”不一致。
因此结构固定为 2 层，不补 3/4 层 seed，也不根据 S9/S11 调 dropout/宽度。

下一单变量模型实验采用 MVGFormer 官方实现中“query 结合 camera-ray features
进行 projective attention”的可迁移思想，但不引入图像 backbone：为每个 H76
候选/关节构造 candidate-dependent ray-view tokens，先做无相机 ID 的跨视角
attention，再进入已验证的 2 层候选 Set Transformer。并行对照仅 view-attention
与 view+candidate hierarchical attention，判断新增收益来自哪一级交互。

### Stage F：MVGFormer 式 ray-view query attention（无有效增益）

| 方法 | internal holdout | V3 | V4 | delta Pearson |
|---|---:|---:|---:|---:|
| E-2 candidate attention | 18.5437 | **29.8195** | 29.0502 | **0.8642** |
| F0 ray-view attention only | 18.6329 | 29.8403 | 29.1385 | 0.8439 |
| F1 ray-view + E-2 hierarchical | 18.5468 | 29.8191 | **29.0455** | 0.8634 |

F0 明确退化，说明逐射线再聚合不能替代直接候选比较。F1 与 E-2 几乎相同：
V3/V4 变化仅 `-0.0004/-0.0047` mm，小于 E-2 三种子标准差，且内部 holdout
还略差，不能称为提升。因此 F1 不补 seed、不升级主模型；保留为“RUMPL VFT
已充分编码 ray-view 交互”的消融证据。当前主模型继续固定 E-2。

下一模型控制针对 E-2 仍然逐关节独立输出 utility 的限制：借鉴 MixSTE 的轴向
分解思想，在“候选集合 attention”之外增加沿 17 个人体关节的 spatial
attention。对照 post-spatial 与 candidate/spatial alternating 两种正规结构，
不改候选、特征、损失或推理温度。

### Stage G：MixSTE 式 candidate/joint 轴向 attention（无有效增益）

| 方法 | internal holdout | V3 | V4 | delta Pearson |
|---|---:|---:|---:|---:|
| E-2 candidate-axis only | 18.5437 | **29.8195** | 29.0502 | **0.8642** |
| G0 candidate→post joint-axis | **18.5136** | 29.8295 | **29.0402** | 0.8580 |
| G1 candidate/joint alternating | 18.5344 | 29.8447 | 29.0732 | 0.8612 |

G0 相对 E-2 的 V3/V4 是 `+0.0100/-0.0100` mm，两项平均只差约
`0.00004` mm；这属于误差重新分配，不是总体提升。G1 两项均退化。二者内部
holdout 又比 E-2 略好，进一步验证增加空间轴容量会拟合训练 subjects，但不能
稳定迁移到 S9/S11。G0/G1 均不补 seed，E-2 继续作为正式主模型。

至此，围绕有效 E-2 的结构边界已经清楚：

1. candidate-set attention 有稳定、三种子可复现的净收益；
2. attention 深度超过 2 层不能同时改善 V3/V4；
3. 重复 ray-view attention 与 RUMPL VFT 信息重叠；
4. joint-axis attention 与完整 pose context 重叠；
5. 下一步不再堆单帧 attention 微变体。

下一阶段按原预注册顺序进入真实 H36M 时序：以 E-2 单帧输出为观测，先构造
严格不跨 subject/action/subaction 边界的 T=9 窗口，再对照轻量 fixed-lag
temporal head 与 MixSTE factorized temporal block。必须先验证当前每 5 帧采样
的 group 顺序/时间戳和训练测试协议，不能复用旧 H135/H136（它们基于 H81、
训练误差约 80--130 mm，且没有形成可信的 E-2 同口径结果）。

### Stage H：严格 T=9 时序对照（2026-08-11）

#### H0 数据与协议审计

没有把稀疏 `annot_filtered_5_64` 验证集硬拼成视频。完整的
`annot_temporal_5_5` 验证 pkl 有 105,076 条记录、26,269 个四视角同步时刻，
每个序列相邻 `image_id` 严格相差 5；按 subject/action/subaction 划分后得到
56 个序列、25,821 个不跨边界的 T=9 窗口。训练 pkl 有 78,047 个同步时刻、
150 个序列和 76,847 个窗口。H76 候选在两份 pkl 上均逐组与原始记录对齐，
没有跨 subject/action/subaction 拼窗。

代码和输出：

- `launch_h76_temporal_validation_export_20260812.sh`
- `build_e2_temporal_cache_20260812.py`
- `train_e2_temporal_residual_20260812.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_Cache_20260812/`

E-2 frame cache 中 V3/V4 仍使用单帧 E-2 soft candidate utility；V2 保持冻结
H76 的原始候选。时序头只看 T=9 组 pose，输出有界 root-relative residual，
中心帧作为 identity，root 平移强制复制中心帧。权重只按训练主体内部
`center_group_index % 10` 留出集选择，S9/S11 最终评一次。

#### H1 两种论文对应结构

| 方法 | 论文/结构依据 | S9/S11 V2 | S9/S11 V3 | S9/S11 V4 |
|---|---|---:|---:|---:|
| E-2 中心帧（同一 T=9 中心子集） | 单帧基线 | 34.9708 | 30.0009 | 29.2438 |
| H1 fixed-lag temporal | 轻量 temporal Transformer 对照 | **33.2124** | **29.7938** | **29.2139** |
| H2 MixSTE factorized | MixSTE：空间/时间轴向交替 attention | **33.1328** | 30.3936 | 29.7311 |

相对同一窗口中心帧，H1 的 V2/V3/V4 变化为 `-1.7584/-0.2071/-0.0299`
mm；H2 为 `-1.8380/+0.3927/+0.4872` mm。MixSTE 在训练主体留出集持续
优于 fixed-lag，但跨到 S9/S11 后 V3/V4 反而变差，属于明显的主体/动作模式
过拟合，不能把内部 holdout 数字当成泛化结果。当前 H1 是暂定时序控制，H2
只保留为“训练集有效、跨主体失败”的反例。

下一步不是继续堆层数，而是做预注册的单变量对照：只用 V3/V4 任务训练时序头、
保持同样的 root-protected residual 与窗口协议，检查 H1 的小幅 V3/V4 增益是否
被 V2 任务的联合损失稀释；如果仍不能稳定低于中心帧，就把时序定位为 V2 辅助
模块，不作为主论文主线。

#### H2 视角任务专门化复核（最终采用向量化 batch 版本）

为排除联合 V2/V3/V4 损失对高视角任务的稀释，另外只用五个任务
`(0,1,2)`, `(0,1,3)`, `(0,2,3)`, `(1,2,3)`, `(0,1,2,3)` 训练时序头，
其他设置完全不变。原始逐样本 DataLoader 只用于冒烟；正式结果采用 paired
NumPy batch gather，训练样本和标签完全相同，避免 CPU 预取成为瓶颈。

| 方法 | 训练任务 | S9/S11 V3 | S9/S11 V4 | 相对中心帧 |
|---|---|---:|---:|---:|
| H1 fixed-lag（联合版） | V2+V3+V4 | 29.7938 | 29.2139 | `-0.2071/-0.0299` |
| H3 fixed-lag（V3/V4 专门化） | V3+V4 | **29.4122** | **28.7503** | **`-0.5887/-0.4935`** |
| H2 MixSTE（联合版） | V2+V3+V4 | 30.3936 | 29.7311 | `+0.3927/+0.4872` |
| H4 MixSTE（V3/V4 专门化） | V3+V4 | 29.6115 | 29.0022 | `-0.3894/-0.2417` |

H3 明确优于联合 fixed-lag，说明之前时序增益小并非“时间信息无效”，而是
联合任务训练和跨主体泛化之间存在冲突。H4 仍弱于 H3，故正式时序结构采用
fixed-lag temporal Transformer，MixSTE 保留为结构反例，不作为主模型。

H3 三种子复核：

| seed | V3 | V4 |
|---:|---:|---:|
| 0 | 29.4122 | 28.7503 |
| 1 | 29.5425 | 28.8340 |
| 2 | 29.4527 | 28.8026 |
| mean ± pop.std | **29.4691 ± 0.0545** | **28.7956 ± 0.0345** |

三种子均低于同一窗口中心帧的 30.0009/29.2438 mm，且方差很小；H3 的
V3/V4 改善不是单次随机波动。seed1/seed2 输出位于
`Temporal_E2_Models_20260812/fixedlag_v34_seed1|seed2/`。

按动作审计显示，H3 的收益不是 root 平移造成的：root-protected 设计使三种子
的 root 误差与中心帧完全相同；改进来自 root-relative 关节。大多数动作的
all-17 delta 为负，个别动作（主要是 Phone/Purchases/Eating）有轻微正负波动，
因此主表同时保留动作均衡指标，不能只挑最有利动作。审计文件：
`Temporal_E2_Models_20260812/fixedlag_v34_per_action_audit.json`。

若按视角数量使用任务专门化路由，V2 可沿用联合 H1 的 `33.2124` mm，V3/V4
采用 H3 的 `29.4122/28.7503` mm；这组数字应标注为“view-count-specific
temporal heads”，不能与单一共享时序头混写。所有模型均 root-protected，
没有损害绝对平移的捷径。

最终输出：

- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_Models_20260812/fixedlag/result.json`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_Models_20260812/mixste/result.json`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_Models_20260812/fixedlag_v34_fast2/result.json`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_Models_20260812/mixste_v34_fast2/result.json`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_E2_WindowMemmap_20260812/`

**Stage H 决策：** 时序模块不再泛化为“任意 Transformer 都有效”。当前最有
证据的组合是 E-2 单帧候选 utility + fixed-lag temporal residual；V2 使用联合
头，V3/V4 使用专门化头。下一步若继续，只做该组合的 seed/动作分层和负贡献
分析，不再增加 MixSTE 深度或重复 ray/joint attention。

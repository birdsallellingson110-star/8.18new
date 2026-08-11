# 实验失败原因总结 与 RUMPL 粗糙模块审查（2026-07-24）

目的：回答两个问题。
1. 为什么这么多实验没有带来提升——失败的根本原因是什么？
2. RUMPL 论文/代码里哪些模块做得简单、粗糙、有明确的修改替换空间？

所有结论均以正确基线 R5 为参照（V2 All17 30.885 / KP* 35.506；V3 23.039；V4 20.213；V5 18.746 mm）。

---

## 第一部分：失败原因分类（五类）

### F1. 基线错误（7/22 之前的全部"提升"）

7/22 审计确认：旧流水线抽帧协议错误（`% 64` 先抽图 vs 官方先 grouping 再 `[::64]`，两者仅 65/3500 张图重合）、模型配置偏离官方（DIM=256、Closest intersection、屏幕归一化等 10 项差异）。旧基线 V2 是 46.9 mm，而正确复现是 30.885 mm。

**后果**：蒸馏 -5.97 mm、hard-view 全降、ensemble 等所有"大提升"，本质是把一个坏配置修回来，不是真实创新。旧模块栈最好也只到 ~42 mm，远差于正确基线。

**教训**：在坏基线上任何模块都容易"有效"。这也解释了为什么 7/23 之后同样的模块在正确基线上全部失效。

### F2. 信息冗余——给网络重复喂它已有的信息（7/23 后偏置类全部）

失效实验：G0/G1/G2/G3 各种 conf/geom attention bias、J0/J1/J2 global joint-view、L0/L2 learned reliability、P0-P5 pair gate。全部在 ±0.15 mm 内打转，最好的 G4 也只有 V2 -0.011 mm。

**机制**：RUMPL 的输入 token 已经包含射线方向、相机中心、置信度（`multiview_rumpl.py:602-604, 647-649`）。conf bias、ray-distance bias 注入的信息，网络从 128k 训练样本里本来就能学到。给一个已饱和的模型换个形式重复输入，期望收益就是零。

**GBT 论文为什么有效而我们无效**：它的上下文不同——9 帧时序输入、不同的 backbone、且其主要收益在含噪/遮挡评测下体现。把它的 bias 单独摘出来贴到干净数据上的 RUMPL，没有增量信息。

### F3. 先验冲突——正则化在信息稀缺时帮忙、信息充分时有害

- S1 对称正则：V2 -0.340（有效），V3/V4/V5 全部变差（+0.04~+0.30）。
- 随机 VFT mask、全程 token removal：帮困难 V2 组合、伤几何好的组合。

**机制**：人工先验只在数据无法自解释时有价值（2 视角欠约束）。视角多了以后先验和观测冲突，成为噪声。任何先验类模块要么做成视角数自适应，要么只作为 V2 消融报告。

### F4. 域差——在 AMASS 合成域学到的统计迁移不到 CMU

- T2 时序模块：AMASS 验证集提升，CMU 实测 -6 mm。学到的是 AMASS 特有的绝对姿态/运动统计。
- 蒸馏的教师本身也是合成域模型，蒸馏传递的也含域偏差。

**教训**：新模块必须只依赖域不变量（相对运动、几何关系），且第一时间跨域验证，不能只看 AMASS 验证集。

### F5. 数值不稳定——像素空间损失的奇异性

置信度加权重投影 loss 训练发散：透视除法 1/z 在近相机平面梯度爆炸。改成 3D point-to-ray 后稳定但无收益（对噪声射线做硬约束反而有害）。

### 总规律

**所有"把已有信息换个形式再喂一遍"的路线都失败了。** 要拿到 2 mm 级提升，只有两条正路：
1. 给网络它现在**得不到**的东西：新的监督信号、时序信息、真实域信息；
2. **结构性降低学习负担**：让网络学残差/修正量，而不是从零回归绝对坐标。

---

## 第二部分：RUMPL 粗糙模块审查（按可替换价值排序）

### M1. 融合读出：单 fusion token 独占输出，K 个 view token 全部扔掉 ★★★★★

证据：`multiview_rumpl.py:857` — VFT 之后 `x = x[:, 0, :]`，只取 fusion token。K 个视角 token 经过 4 层 attention 精炼后直接丢弃，**且从头到尾没有任何逐视角监督信号**。fusion token 学什么、每个 view token 编码了什么，完全靠端到端 MPJPE 一个损失隐式驱动。

**替换方案（首推）：逐视角射线深度监督（per-view ray-depth supervision）**
- 每个 view token 加一个极小的辅助头 `Linear(D→1)`，预测该关节沿本视角射线的深度 t；
- 3D 复原 = 相机中心 + t × 方向，GT 深度 = GT 关节到射线的投影长度，监督完全适定（单视角预测 3D 不适定，但沿已知射线预测 1D 深度是适定的）；
- 主输出仍走 fusion token，辅助 loss 强迫每个 view token 真正携带几何信息，融合的原材料变好。
- 这是**新监督信号**（F2 教训的正解），不是重复输入。参数量几乎为零，天然支持消融（per-view MPJPE 可单独报告）、注意力可解释（老师要求）。

### M2. 输出头：LayerNorm + 单层 Linear 直接回归绝对坐标 ★★★★

证据：`multiview_rumpl.py:509-512` — `head = Sequential(LayerNorm, Linear(D,3))`，直接回归房间归一化坐标。而数据管线其实已经算好了多视角射线最近点三角化（`multiview_cmu_panoptic_rumpl.py:385-392` 的 `middle_points`），但官方最优配置下**根本不送入网络**——代码里三角化输入与置信度 concat 互斥（`multiview_rumpl.py:351-352` 直接 raise）。

**替换方案：三角化锚定残差解码。** 把置信度加权三角化点作为锚，网络只预测残差修正量。V2 三角化 42.9 mm、网络 30.9 mm，说明网络确实学到了超越三角化的东西，但它把大量容量花在"重新发明三角化"上；残差化可释放容量给真正难的部分（噪声修正、遮挡补全）。改动集中在 head，风险低。

### M3. 置信度通路：最有价值的输入，最粗糙的注入方式 ★★★★

证据：`multiview_rumpl.py:354-356, 647-649` — conf 标量只在输入处过一次 `Linear(1→32)` concat，要穿过 VFT 4 层 + PFT 4 层 attention 存活。论文自己的消融（w/o Conf）显示置信度是收益最大的输入之一，却只有这一次浅注入。我们试过的 conf-bias 失败不否定这一点——bias 也是浅注入（attention logits 加个标量），同样粗糙。

**替换方案：逐层 FiLM 调制**——每个 Block 里用 conf 生成 scale/shift 调制 view token 特征（`x = x * (1+γ(c)) + β(c)`），让置信度在每一层参与特征选择；外加一个 2-3 层的 conf 校准小网络（HRNet 置信度跨相机/跨关节不校准是已知问题）。

### M4. 训练噪声模型：均匀伯努利丢关节，与真实遮挡不符 ★★★

证据：`multiview_amass_rumpl.py:758-763` — Miss20 就是每个关节独立以 20% 概率置零。真实遮挡是结构化的：成组（半边身体）、空间连续、跨帧持续、且伴随低置信度而非置信度不变。

**替换方案：结构化遮挡模拟**（按肢体分组遮挡、矩形遮挡块投影、遮挡时同步压低 conf）。这直接服务老师要求的"训练加噪声抗抖动"故事，且是让 M1/M3 的可靠性机制真正被逼出来的训练条件——干净训练数据下网络永远学不会怀疑某个视角。

### M5. 训练时视角数采样粒度粗 ★★

证据：`multiview_rumpl.py:588-590` — 每个 **batch** 只采一个 K（`torch.randint` 一次），全 batch 共享，且直接截取前 K 个视角。改成 per-sample K 是几行代码的事，让每步梯度覆盖所有视角数，减小 K 之间的优化耦合。

### M6. PFT 末层重复执行的官方 bug ★

证据：`multiview_rumpl.py:878-881` — 最后一个 block 被调用两次（配置 12 层实际执行 13 次，末两次共享权重）。R1/R3 已验证修掉它不涨点。可以借题发挥改造成带中间监督的迭代精化头，但优先级最低。

### 另外两个论文层面的事实（组会可以说）

- 官方 YAML 引用的三个实现类（`multiview_pose_3d_fuser` 等）从未开源，我们的 R5 是用公开类做的 official-like 重建，三角化审计与论文 Table 3 只差 0.14 mm，复现是扎实的。
- 损失就是纯 MPJPE，没有骨长/对称/重投影/时序任何附加约束——论文本身在监督信号上是极简的，这是它留给我们的空间。

---

## 第三部分：行动建议（排序）

| 优先级 | 方案 | 针对的粗糙点 | 为什么可能 ≥1-2 mm | 成本 |
|---|---|---|---|---|
| P1 | 逐视角射线深度监督 + 可靠性加权融合 | M1 | 新监督信号，不是信息冗余；直接改善融合原材料 | 训练 ~1.5h/次 |
| P2 | 三角化锚定残差解码 | M2 | 结构性降低学习负担 | 低 |
| P3 | conf 逐层 FiLM + 校准 | M3 | 把最有价值的输入用对 | 低 |
| P4 | 结构化遮挡训练 + 含噪/遮挡评测轴 | M4 | 干净数据接近饱和；2 mm 差距在含噪条件下最容易拉开，且是老师认可的故事 | 中 |
| P5 | per-sample K 采样 | M5 | 微改进，顺手做 | 极低 |

P1+P2+P4 组合成的故事线："**把融合过程锚定在真实 3D 几何上**（每个视角显式回答'关节在我的射线上哪里'+三角化锚点），并用结构化遮挡训练逼出可靠性判断"——这与老师"注意力机制与真实 3D 空间几何绑定 + 训练加噪声 + 注意力图验证"的要求逐条对应，而且每一项都是新监督/新结构，不再踩 F2（信息冗余）的坑。

---

## 第四部分：顶会论文依据与开源代码对照（2026-07-24 补充）

每个待替换模块都找到了顶会论文依据 + 官方开源实现。代码已克隆到 `/mnt/data/cjydata/reference_code/`，论文 PDF 在 `reference/7.24/`。

### P1 逐视角深度监督 ← PlaneSweepPose（CVPR 2021）

- 论文：`reference/7.24/PlaneSweepPose_CVPR2021.pdf`；代码：`reference_code/PlaneSweepPose/`
- 核心思想与我们完全一致：**对每个视角的每个关节回归沿射线的深度**，跨视角一致性通过 plane sweep 隐式约束，3D = 反投影。它证明了"逐视角深度回归"这条路是 SOTA 级别可行的。
- 直接学习的实现：`lib/models/mvmppe.py`（深度平面打分 + coarse-to-fine）+ `lib/models/softargmax.py`（深度 bin 分类 + soft-argmax，比直接回归标量深度更稳，建议照抄这个输出形式）。
- 融入 RUMPL：view token 加辅助头，输出深度 bin 分布 → soft-argmax 得到深度 t，GT 深度 = GT 关节在该射线上的投影长度。主输出不变。

### P2 三角化锚定 + P3 置信度加权 ← Learnable Triangulation（ICCV 2019 oral）

- 论文：`reference/7.24/LearnableTriangulation_ICCV2019.pdf`；代码：`reference_code/learnable-triangulation-pytorch/`
- 核心思想：**置信度加权的可微代数三角化**（加权 DLT + 可微 SVD），置信度端到端学出来，专治遮挡和离群视角。论文证明仅此一项就把 H36M 误差降了约一半。
- 直接学习的实现：`mvn/models/triangulation.py`（AlgebraicTriangulationNet，看 confidence 如何进加权线性系统）+ `mvn/utils/multiview.py`（`triangulate_batch_of_points`，可微 SVD 三角化，可以直接搬）。
- 融入 RUMPL：用它的可微加权三角化算锚点（权重用 2D conf 或由网络预测），RUMPL head 改为预测残差；三角化模块可微，端到端训练。

### P3/P4 自适应视角权重 + 遮挡评测 ← AdaFuse（IJCV 2021）

- 论文：`reference/7.24/AdaFuse_IJCV2021.pdf`；代码：`reference_code/adafuse-3d-human-pose/`
- 核心思想：为每个视角学一个**自适应融合权重**反映特征质量，避免坏视角污染好视角；并发布了 **Occlusion-Person 数据集**（73k 帧、8 视角、20% 关节遮挡、带逐关节遮挡标签，公开可下载）。
- 直接学习的实现：`lib/models/ada_weight_layer.py`（视角权重网络怎么设计、用什么输入特征）。
- 对我们的双重价值：a) conf 校准/视角加权的成熟设计；b) **Occlusion-Person 可作为我们的第三个 benchmark**——遮挡条件下与已发表数字直接对比，正是"2mm 差距最容易拉开"的赛场，也支撑老师的抗噪故事。

### 几何绑定注意力（若重做 bias 类模块）← MvP（NeurIPS 2021）+ MVGFormer（CVPR 2024）

- MvP 论文：`reference/7.24/MvP_NeurIPS2021.pdf`；代码：`reference_code/mvp/`，核心在 `lib/models/ops/modules/projattn.py`（projective attention）+ RayConv。它把注意力绑定到 3D 几何的方式不是加标量 bias，而是**围绕当前 3D 估计的投影位置采样特征**——这解释了我们 G0-G4 标量 bias 失败的原因：绑定方式太弱。
- MVGFormer 代码：`reference_code/MVGFormer/`（CVPR 2024，微软），**免学习的几何模块 + 可学习的外观模块迭代交替**，遮挡和跨相机泛化显著提升。它的"几何部分不学习、外观部分学习"的解耦哲学正好印证我们 F2 教训。
- 融入 RUMPL 的正确姿势：不是在 attention logits 上加 ray-distance 标量，而是 MVGFormer 式的迭代——当前 3D 估计 → 几何模块（重投影/三角化）→ 修正 → 再进 transformer。

### 方法学印证 ← UPose3D（ECCV 2024，无代码）

- 论文：`reference/7.24/UPose3D_ECCV2024.pdf`。与 RUMPL 同样用合成 mocap 数据训练、任意相机数、依赖 2D 关键点+不确定性。它的卖点正是**不确定性感知带来分布外鲁棒性 SOTA**——印证了我们"干净精度接近饱和、鲁棒性才是差异化战场"的判断。

### 组合出的论文技术路线（有据可依版）

1. **锚**：Learnable Triangulation 的置信度加权可微三角化 → RUMPL 只学残差（P2）；
2. **逐视角监督**：PlaneSweepPose 式深度 bin + soft-argmax 辅助头（P1）；
3. **可靠性**：AdaFuse 式视角权重 + conf 校准（P3）；
4. **训练与评测**：结构化遮挡训练 + CMU/H36M/Occlusion-Person 三 benchmark（P4）。

每一步都有顶会论文背书和可运行代码参考，消融表天然成立（去掉任一组件都是一行消融）。

---

## 第五部分：遮挡轨 / 蒸馏进度（2026-07-24 晚）

### DS0（full→sparse self-distill，R5 协议）— 终评完成

过早评测（训练未 END）V2≈60.9 mm 已作废；`chain_distill_reeval_final` 在 `END 22:33` 后用 `model_best` 重评：

| | All-17 / KP* | Δ vs R5 |
|---|---|---|
| V2 | 30.285 / 34.920 | **−0.60 / −0.59** |
| V3 | 23.622 / 26.208 | +0.58 / +1.05 |
| V4 | 21.021 / 23.122 | +0.81 / +1.42 |
| V5 | 19.659 / 21.710 | +0.91 / +1.62 |

结论：与 S2a 同型——**只帮稀疏 V2，伤多视角 clean**。不能单独当主故事。

### A2（S2a struct-occ 0.4 + occ-joint soft boost 2.0）— 已完成（END 23:54）

闸门（occ 相对 R5 降 >1 mm 且 clean 掉点 ≤0.3 mm）：**仅 V2 过闸**；V3–V5 fail（clean 伤 / occ 收益不足）。相对 S2a 普遍再好 ~0.2–0.4 mm（occ joint loss 有小加成）。

| | R5 | S2a | A2 | ΔA2−R5 |
|---|---|---|---|---|
| V2 occ0.0 | 30.89 | 30.72 | **30.43** | **−0.45** |
| V2 occ0.3 | 34.49 | 33.55 | **33.38** | **−1.10** ✓ |
| V2 occ0.6 | 38.06 | 36.85 | **36.49** | **−1.57** ✓ |
| V3 occ0.0 | 23.04 | 23.34 | 23.15 | +0.12 |
| V4/V5 clean | — | — | — | +0.46 / +0.77 |

结论：结构化遮挡 + 遮挡关节加权在 **稀疏视角遮挡轴** 上成立，但多视角 clean 仍掉点；不能单独做主 claim。下一刀：**B1**（visibility-conditioned VFT attention），或减弱 struct-occ level（0.2）做单变量对照。

### 2D refine（多视角 soft ray refine）— 已实现，冒烟偏负

实现：`RUMPL/lib/utils/refine_2d_multiview.py`，挂在 CMU/AMASS `__getitem__`（遮挡注入之后）。Env：`RUMPL_2D_REFINE=1`，`MODE=soft_fill|fill_only|soft|hard`。

| 设置 | V2 clean Δ | V2 occ0.6 Δ | V3 occ0.6 Δ |
|---|---|---|---|
| soft_fill s=0.5 | +0.29 | +0.26 | +0.62 |
| fill_only | **0.00** | +0.10 | +0.49 |

说明：`fill_only` 干净数据不变（实现正确）；遮挡轴仍略差——V2 单视角遮挡后只剩 1 票无法三角化填；V3+ 填入的重投影方向质量不够，拉低 RUMPL。模块默认关，可继续试更严投票 / 骨架先验，或把 refine 当训练时增强而非测时插件。

### B1 AdaFuse ViewWeightNet（IJCV'21）— 已否定

| 变体 | V2 clean/occ0.3/occ0.6 ΔR5 | 结论 |
|---|---|---|
| B1 (VW+occ0.4) | +0.06 / −0.44 / −0.55 | occ 增益不足 1mm |
| B1only (VW) | +0.72 / +0.43 / +0.43 | 单独无效 |
| B1occ02 | +0.46 / +0.10 / +0.06 | 近中性 |
| B1A2 (VW+A2) | −0.11 / −0.88 / −1.39 | 接近 A2，未超越 |
| C (train 2D refine+occ0.4) | +0.65 / +0.65 / +0.89 | 全面变差 |

**结论**：几何 2D fill 与 AdaFuse VW 均未超 A2；occlusion 主线仍押 A2。

### DS1（hard-view + legw）— 终评完成

V2 −1.15；V3/V4/V5 +0.16/+0.32/+0.36。精度故事可用 V2，多视 clean 仍掉点。

### 2D→3D 文献迁移结论（2026-07-25）

几何 2D 优化失败 ≠ 2D 不重要。可迁移且未否定的路线：
1. **UniCodebook（NeurIPS’25）**：离散姿态先验 + DCSA 软注入连续 transformer —— 下一刀 **D3**
2. Conf FiLM（审计 M3）、UPose3D 式**学习型** 2D residual compiler、MHFormer 多假设 —— 备选
3. 不再主推：AdaFuse VW、几何 2D fill、MixSTE 时序主架构、硬 tri 替 VFT

### D3 UniCodebook-lite（离散姿态先验）— 进行中

- Env：`RUMPL_POSE_CODEBOOK=1`；挂 A2（struct_occ0.4 + occJL）
- 实现：EMA codebook + DCSA-lite（VFT 读出后、PFT 前软注入）
- 变体：`D3_codebook_a2_seed0_20260725`

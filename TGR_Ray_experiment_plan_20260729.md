# TGR-Ray 实验路线（2026-07-29）

## 1. 论文的核心假设

TGR-Ray 不应被定义为“给 RUMPL 叠加偏置和时序模块”，而应验证下面的可证伪假设：

> 当多视角几何可靠时，模型应尽量保留解析几何解；当二维检测、遮挡或相机配置使几何退化时，学习模块才介入修正。

现有 CMU 阶段审计已经为该假设提供了先验证据：

- HRNet-2D 下，RUMPL V2 比几何 midpoint 好约 12.6 mm；
- GT-2D 下，几何 midpoint 约 10.5 mm，而 RUMPL 约 16.5 mm；
- 因此，RUMPL 擅长修复坏观测，但在好观测下会偏离已经正确的几何解。

正式论文需要在真实 Human3.6M 训练和统一评测协议下重新验证这一交叉关系。

## 2. 实验前必须修正的定义

### 2.1 射线距离

草稿式 (2) 当前使用带符号分子并除以叉积范数平方，不能直接送入
`log(1 + D/tau)`。两条非平行射线的欧氏最短距离应使用

```text
D = abs((ou - ov)^T (dv × du)) / (||dv × du|| + eps)
```

近乎平行时必须使用单独的稳定分支，并对无量纲距离 `D/tau_D` 做裁剪。

### 2.2 三角化夹角

`S = 1 - (dv^T du)^2` 可作为条件可靠性，但要注意：

- 自注意力对角项不能因 `log(S)` 变成负无穷；
- `S` 表示几何条件，不等于观测一致性；
- 近乎平行时应降低该视角对的深度可信度，而不是简单删除全部信息。

### 2.3 粗姿态软锚点

式 (10) 中的 `lambda ||q-q0||^2` 可能在 `lambda` 过大时让输出退化成自由回归
`q0`，从而绕过几何。`lambda` 必须由有效视角数、最小三角化角和求解矩阵条件数
单调控制或有上限，并报告：

- 几何项与锚点项的相对权重；
- 不同条件数分桶下的误差；
- 去掉锚点后的结果。

### 2.4 模块信息流

时序模块位于 VFT 后可以保留相机泛化，但逐视角射线修正头仍需同时读取：

```text
当前帧逐视角特征 + 当前关节时序上下文 + confidence/geometry statistics
```

不能只保留池化后的时序 token，否则无法区分应该修正哪一个视角。

### 2.5 修正角范围

以焦距约 1000 px 为例，5 度约对应几十像素偏差，作为默认上限偏大。第一轮敏感性应
使用 `0.5/1/2/5` 度，而不是直接使用 `2/5/10` 度。

## 3. 总体实验顺序

### P0：完成数据与论文复现审计

目的：冻结唯一可信的输入、划分、评测器和 RUMPL 实现。

1. 完成 16 个真实 H36M MMPose 分片；
2. 合并为保留帧号、动作边界、相机和置信度的训练 PKL；
3. 检查 312,188 条记录、78,047 个四相机同步组、S1/S5/S6/S7/S8；
4. 完成全图/裁剪、关节映射、左右腿交换、scheduler、worker 随机序列等复现排查；
5. 固定 S9/S11 测试协议和单一评测脚本。

通过门槛：

- 三角化、数据帧集合和坐标系能够确定性复现；
- 同一 checkpoint 重复评测结果一致；
- 未解释的论文差距尽量压到 1 mm 左右；若仍存在，明确记录为公开实现差异，后续所有消融使用同一内部基线。

预期：这一步可能解释当前剩余约 2 mm 的一部分，但不预设一定能完全消除。

### P1：真实 H36M 单帧 RUMPL 基线

目的：得到所有 TGR-Ray 模块的真实训练对照。

设置：

- 真实 H36M S1/S5/S6/S7/S8 训练；
- 冻结的同一 HRNet-W32 产生 2D；
- 随机使用 2/3/4 个视角；
- 不加 bias、时序、射线残差、不确定性或遮挡增强；
- 第一轮 seed 0 调通，正式结果至少 3 seeds。

报告：

- 标准相机 V2/V3/V4 全组合 Absolute MPJPE；
- GT-2D 对照；
- 线性三角化和 confidence-weighted ray LS；
- 每关节和每相机组合结果。

预期：

- 真实数据训练应更贴合 H36M 的动作和检测误差分布；
- 但相机 OOD 泛化可能弱于 AMASS 随机相机训练；
- 该实验的目标是建立真实训练基线，不要求立刻优于 AMASS-RUMPL。

### P2：解析几何与核心假设诊断（先于任何新网络）

目的：验证论文故事是否成立。

实现一个经过单元测试的 differentiable weighted ray least-squares solver，先不训练新模块。

对比：

1. 线性/等权 ray LS；
2. confidence-weighted ray LS；
3. RUMPL 自由回归；
4. GT-2D 下的上述三种方法。

按以下变量分桶：

- 2D reprojection error；
- ray pair distance；
- 最小三角化角；
- 求解矩阵条件数；
- 视角数。

关键通过门槛：

- 好 2D、好基线时 ray LS 应明显优于或至少不弱于 RUMPL；
- 坏 2D、低置信度或退化基线时 RUMPL 应存在互补优势；
- 必须观察到可学习门控有意义的“交叉区间”。

若没有交叉关系，TGR-Ray 的核心动机需要重写，不能直接继续堆模块。

### P3：单帧最小可行 TGR（核心实验）

先验证解析出口，不加入时序和 attention bias。

#### P3.0：原始射线 + 固定权重 + 粗姿态软锚点

```text
q0 = RUMPL coarse prediction
q* = weighted_ray_LS(original_rays, fixed_conf_weights, q0, lambda(condition))
```

验证求解器、梯度、退化回退和 `lambda` 行为。

#### P3.1：学习异方差权重，不修正射线

只预测每条观测的 `s=log variance`，由
`confidence + ray consistency + angle condition + per-view feature` 得到权重。

这是最可能先产生稳定收益的模块，因为它不改变成像几何，只决定哪些观测可信。

#### P3.2：切平面射线残差，等权求交

预测受限 `delta_theta`，输出层零初始化，角度上限先用 1 度。单独检查修正后的
reprojection/ray-to-GT distance 是否真的下降。

#### P3.3：射线残差 + 异方差 + 自适应软锚点

形成单帧完整版：

```text
per-view correction + learned uncertainty + closed-form solver + coarse anchor
```

训练顺序：

1. 冻结 RUMPL coarse backbone，只训练权重/残差头；
2. 确认恒等初始化与 baseline/solver 对照一致；
3. 再以 0.1 倍学习率联合微调最后若干层；
4. 监控修正角、权重、条件数和锚点占比，防止网络绕过几何。

预期：

- GT-2D 和几何良好样本应最先改善；
- clean HRNet 的收益可能只有 0–1.5 mm；
- 2D 噪声、少视角和退化相机下更可能有 2–10 mm 收益；
- 若仅 OOD/鲁棒性改善，应把论文主张定位为泛化与鲁棒性，而不是 clean SOTA。

### P4：几何–置信度 attention bias（可选增强，不作为地基）

已有实验表明，把未归一化 ray-distance bias 直接塞入 RUMPL VFT 通常无效或退化。
因此只在 P3 单帧核心成立后测试：

1. confidence-only；
2. 无量纲 normalized ray-distance；
3. angle condition；
4. 三者联合。

要求：

- bias 零初始化并暖启动；
- 直接作用到有明确几何含义的真实 view token/reliability head；
- 不再把 fusion token 的零距离行当作论文公式复现；
- 与无 bias 的同结构、同参数预算模型比较。

预期 clean 增益较小，约 0–0.5 mm；主要观察 OOD、低基线和异常观测分桶。若低于 seed
方差且没有鲁棒性收益，应从主模型删除，只保留负消融。

### P5：真实 H36M 时序

前提：P1 单帧基线与稠密 S9/S11 验证 clip 已固定。

数据：

- 当前 PKL 可直接按 `subject/action/subaction/image_id` 构造窗口；
- 每 5 个原始帧一张，约 10 fps；
- 不跨动作、subaction、缺帧和损坏序列；
- 主设置 `T=9`，预测中心帧或最新帧。

模型：

```text
frame-wise ray/VFT
→ per-joint temporal transformer
→ per-view correction/uncertainty calibration
→ closed-form solver
```

对照：

1. T=1；
2. 简单 1D/EMA 输出滤波；
3. T=5；
4. T=9；
5. 仅在 T=9 有效后测试 T=17。

第一轮不使用 T=27，也不照搬 MixSTE 的 81/243 帧。

预期：

- clean MPJPE：约 0–1.5 mm 的合理收益；
- GBT 的 1 帧到 9 帧差约 3.4 mm，可视为积极参考而不是承诺；
- MPJVE、抖动和连续遮挡应比 clean MPJPE 更明显改善；
- 若简单滤波达到相同 MPJVE 且 MPJPE 更好，删除学习式时序主张。

旧 T2 的 AMASS→CMU 约 +6 mm 域退化不否定本实验，因为本轮使用真实 H36M 连续帧训练；
但它要求我们禁止用合成域内部提升代替真实验证。

### P6：相机环境泛化（论文核心表）

先使用 H36M 3D 姿态重投影建立可控相机环境：

- ID：训练采样区间；
- Interpolation：训练区间内部但未出现的组合；
- Extrapolation：方位、俯仰、半径和焦距区间与训练互斥；
- 所有模型使用相同姿态、2D残差库和相机样本。

对比：

1. ray LS；
2. 真实 H36M RUMPL；
3. RUMPL + camera environment randomization；
4. 单帧 TGR；
5. 完整时序 TGR。

主指标：

```text
OOD absolute MPJPE
delta_gen = OOD - ID
```

通过门槛：

- 相对 `RUMPL + 同样环境增强` 仍降低 OOD，而不是把增强收益算到模块头上；
- OOD 至少有稳定、跨 seed 的相对改善，目标可先设为 5%；
- ID 退化最好不超过 0.5–1 mm。

环境随机化可能显著降低 OOD，但也可能轻微损伤 ID，这是需要真实测量的权衡。

### P7：鲁棒性、跨数据集和完整消融

在核心模型通过 P3/P5/P6 后再执行：

- 2/5/10 px 噪声；
- 10/20/30% joint dropout；
- 3/5/9 帧 burst occlusion；
- confidence temperature/miscalibration；
- CMU→H36M 或 H36M→CMU 零样本；
- TotalCapture 只在核心结果成立后下载和加入，不让它阻塞主线。

正式消融顺序应改为：

1. RUMPL；
2. analytic solver + constrained coarse anchor；
3. + learned uncertainty；
4. + tangent-plane ray residual；
5. + geometry/confidence bias；
6. + temporal context；
7. + camera environment randomization。

草稿原表中“先加射线残差、最后才加闭式求解”在因果上不完整，因为射线残差需要解析求解器
才能影响最终坐标。

## 4. 资源与运行优先级

当前只执行 P0。P0 完成后：

1. 用一张 GPU 做论文复现排查/评测；
2. 用另一张 GPU 跑 P1 seed 0；
3. P1 通过后先做 P2 的无训练几何诊断；
4. 只有 P2 支持核心假设，才开始 P3 网络实验；
5. P3 单帧核心有效后才占用资源跑时序与 OOD 大实验。

不要在基线、solver 和评测协议未冻结前并行启动多个模块组合。

## 5. 最终成功标准

论文不要求所有标准相机组合都获得大幅 clean 提升，但至少应满足：

1. 真实 H36M 同协议下，完整方法不显著损害 ID clean；
2. 相对同增强 RUMPL，OOD absolute MPJPE 和 `delta_gen` 稳定改善；
3. 在二维噪声、视角缺失或连续遮挡下退化更小；
4. learned uncertainty、ray correction 和 coarse anchor 的行为与几何条件相关；
5. 时序收益超过简单输出滤波；
6. 结果在至少 3 seeds 上超过方差，而不是依赖单次 0.1 mm 波动。

如果只满足鲁棒性/OOD而不提升 clean，论文仍可成立，但摘要、标题和主表应明确定位为
camera-generalizable robust lifting，而不宣称标准 H36M clean SOTA。

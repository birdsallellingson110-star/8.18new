# 下一阶段：Ray–Image Geometry Refiner 实验计划（2026-08-12）

## 1. 决策变化

不再要求完整保留 RUMPL 主干。只保留已由真实 H36M 严格实验支持的部件，允许
替换信息不足或尚未证明不可替换的模块。

当前固定评估协议仍为 S9/S11、absolute MPJPE、action-equal All-17、V2/V3/V4
全部相机组合。所有 checkpoint 只按训练 subjects 内部 holdout 选择，禁止用
S9/S11 调温度或挑 epoch。

## 2. 模块去留

| 模块 | 决策 | 实验证据 |
|---|---|---|
| HRNet→A1D→H21 二维输入 | 保留为当前输入基线 | 当前所有正式模型中效果最好，输入退回原 HRNet/其他修正均更差 |
| anchor-centered Plücker ray | 保留 | H76 的有效表示组合，兼顾弱相机对和相机无关性 |
| RUMPL 同关节跨视角 VFT | 暂保留，并设可移除消融 | skip/均值、GBT 替换、VFT 前跨关节混合和多类 attention bias 均失败 |
| tri-anchor | 只作为初始化/几何参考 | V4 anchor 单独约 41.4 mm，明显不是最终估计器 |
| 原 PFT + 共享 3D head | 允许替换 | GraFormer 的不匹配替换失败，但现有 PFT 没有利用图像局部证据；V3→V4 增益不足 |
| E2 多子集候选效用 | 保留为末端可选模块 | 22 候选版本 V3/V4 29.3765/28.6262，两个 seed 同向；但不再作为主干 |
| 时序模块 | 暂停 | 对齐协议后仅约 0.16/0.18 mm，当前主要瓶颈不在时间维 |

## 3. 新主线及论文依据

新模型暂称 **Ray–Image Geometry Refiner (RIGR)**：

```text
HRNet heatmap / feature map + camera
                 │
A1D→H21 2D → anchor-centered Plücker rays
                 │
        初始 3D query（H76 / VFT-only / tri-anchor）
                 │
       投影到各视角，在预测关节附近采样图像证据
                 │
  跨视角 epipolar attention → 2D offset + covariance/utility
                 │
           可微加权三角化更新 3D query
                 │
                 └── 重复 2--3 层，直接输出 3D
```

依据：

- Epipolar Transformer（CVPR 2020）：沿极线匹配并融合跨视角图像特征，证明
  修复 2D 特征本身可显著降低 H36M MPJPE；有官方代码。
- MVGFormer（CVPR 2024）：将当前 3D query 投影回各视角，读取图像特征、更新
  2D 点并三角化，迭代形成几何—外观闭环；有官方代码。
- AdaFuse（IJCV 2021）：跨视角热图融合和视角质量自适应，支持未知相机布局；
  有官方代码。
- DenseWarper（ICLR 2026）：密集极线热图交换；有官方代码，但其时序输入不作为
  第一阶段依赖，只借用单帧 dense warping。

与旧实验的关键区别：旧 query residual、bias、E2 都只读取已经生成的 ray/3D
候选；RIGR 每一层重新读取 HRNet 热图或中间图像特征，因此有机会修正错误观测，
不是在同一信息上继续堆 Transformer。

## 4. 实验顺序

### P0：零训练上限诊断（第一优先，先跑）

**状态（2026-08-12）**：已完成 S9/S11 全量 top-4、半径16 诊断，详见
`RIGR_P0_DECISION_20260812.md`。局部 HRNet heatmap 2D oracle 为
V2/V3/V4 = 99.095/67.034/63.074 mm，显著差于 H76 的
34.816/30.489/29.691 mm，因此不进入 heatmap-only P1；保留 P2 的
HRNet 中间 feature/官方 epipolar sampling 路线。

对 S1/S5/S6/S7/S8 内部 holdout 和 S9/S11 分开统计：

1. 将 H76、VFT-only、tri-anchor 的 3D 投影到各相机；
2. 在 HRNet heatmap 的 `3×3 / 5×5 / 9×9 / epipolar segment` 邻域生成二维候选；
3. 只用于诊断，计算局部候选 oracle 的 2D 与最终 3D MPJPE；
4. 报告 V2/V3/V4、各相机组合、各关节，尤其膝/踝和坏相机对；
5. 检查训练 holdout 与 S9/S11 的 oracle 趋势是否一致。

启动门槛：局部图像证据相对 H76 至少提供 `>2 mm` 的 V2 oracle headroom，或
`>1 mm` 的 V3/V4 headroom。达不到则停止该路线，不训练 RIGR。

### P1：轻量热图版 RIGR（两卡并行）

固定两层迭代、共享参数、相机 ID 无关、随机视角子集训练。

| GPU | 实验 | 初始 query | 用途 |
|---|---|---|---|
| 0 | RIGR-H76 | H76 完整输出 | 先追求最佳精度，验证图像闭环是否能修正当前最好模型 |
| 1 | RIGR-VFT | VFT 输出 + tri-anchor，不使用原 PFT/head | 检验 PFT/head 是否可以被几何闭环替代 |

首轮只读取已有 HRNet dense heatmap 和局部 epipolar support，不重新训练 2D
backbone。网络预测每关节每视角的二维 offset、2×2 covariance 和更新门控；通过
可微加权三角化得到下一层 3D。训练损失先只用 absolute 3D MPJPE + 2D offset
监督，不加入时序、骨长、蒸馏或多种正则。

首轮停损：seed0 若 V2/V3/V4 没有至少两项改善，且平均改善小于 `0.5 mm`，不做
调参和多 seed；转入 P2 的真正图像 feature 版本。

### P2：官方 Epipolar Transformer / MVGFormer feature 对齐

P1 使用热图只能利用 17 个通道的末端信息。P2 接 HRNet 倒数一层 feature map，
严格移植官方 epipolar sampling/attention：

| 实验 | 改动 |
|---|---|
| ET-H21 | 极线跨视角 feature 融合后重新解码 2D，再经过 H21→H76 |
| RIGR-Feat | H76/VFT query 投影后采样 HRNet feature，按 MVGFormer 更新 2D 并三角化 |

先冻结 HRNet，只训练融合与几何更新；若稳定改善，再允许解冻 HRNet 最后一 stage，
使用较小学习率。官方代码先做形状、坐标系、投影和 identity 单测，再迁入当前
pipeline；不凭论文描述重写核心采样公式。

### P3：结构消融，确定最终主干

只有 P1/P2 出现有效结果后才运行：

| 编号 | VFT | PFT/head | 图像几何迭代 | E2 |
|---|---|---|---|---|
| A0 | 原 | 原 | — | — |
| A1 | 原 | 原 | 2 层 residual refinement | — |
| A2 | 原 | 删除 | 2 层 direct output | — |
| A3 | 删除 | 删除 | 3 层、tri-anchor 初始化 | — |
| A4 | 最佳 | 最佳 | 最佳 | ✓ |

这组消融能严谨回答：RUMPL 到底保留了 VFT、PFT、tri-anchor 中的哪些部分，而
不是为了“融合创新”形式上保留整条主干。

## 5. 成功门槛与目标

- 第一阶段有效：相对 H76 seed0，V2/V3/V4 至少两项下降，平均下降 `≥0.5 mm`；
- 候选主模型：V2 `≤33.5`、V3 `≤29.0`、V4 `≤28.0`；
- 强结果目标：V2 `≤32`、V3 `≤28`、V4 `≤27`；
- 任何主张必须补 3 seeds、所有相机组合、参数/FLOPs/速度和训练集内部 holdout；
- 最后才把 E2 接到最佳单次前向模型，避免不同模块同时变化无法归因。

## 6. 明确停止的旧方向

不再继续：E2 深度/温度扫描、GHT canonical/骨长/Gumbel、sparsemax/top-k、简单
几何 bias、最终 3D query residual、GraFormer 直接替 PFT、MixSTE/固定滞后时序、
蒸馏。新建的 absolute-candidate-score 对照暂不启动，因为它仍未引入新观测信息，
优先级低于 P0/P1。

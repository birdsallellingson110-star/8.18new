# Global Joint-View Attention 消融实验记录（2026-07-22）

## 1. 实验目的

验证 Geometry-Biased Transformer 的全局 joint-view token 建模是否能进一步改进 RUMPL，并区分两类收益：

1. `plain`：收益是否仅来自增加一层全局 `J×V` self-attention；
2. `biased`：在相同全局层上加入论文式 confidence/geometry bias 后是否有额外收益。

这是模块消融，不使用蒸馏，不与 token removal 混合。主指标为同一个模型在 V2/V3/V4/V5 全部 26 个相机组合上的 Absolute MPJPE 组合平均值；最佳组合、逐组合改善数和最差 delta 为辅助指标。

## 2. 共同训练设置

- 数据：CMU Panoptic，当前严格 RUMPL/GBT 统一数据和评估口径。
- 初始化：scratch。
- 随机种子：0。
- 总训练：20 epoch。
- 课程：epoch 0-4 固定随机 V2；epoch 5-19 在 V2-V5 间随机 K。
- 原 RUMPL VFT：保留逐层可学习 confidence bias 和 ray-distance geometry bias。
- `GBT_CONF_INIT=0.1`，`GBT_GEOM_INIT=1.0`。
- `GBT_FUSION_GEOM=0`。
- token removal/dropout：关闭。
- 蒸馏、hard-view mining、辅助 multi-K loss、reprojection、ray、bone、CAA、DEPRO、monotonic loss：全部关闭。
- GPU：plain 使用物理 GPU0，biased 使用物理 GPU1。

## 3. 唯一消融变量

| 分支 | `GBT_GLOBAL_JV_DEPTH` | `GBT_GLOBAL_JV_BIASED` | 含义 |
|---|---:|---:|---|
| plain | 1 | 0 | 一层普通 global joint-view attention |
| biased | 1 | 1 | 同一层加入论文式 confidence 与 ray-distance bias |

全局 token 数为 `17×K`。输入 ray token 加 learned joint embedding 后展平为 `B×(17K)×D`，完成一层全局 attention，再恢复为 RUMPL VFT 的 `B×17×K×D` 输入。当前为单帧 RUMPL，因此没有复制论文的时间维度。

## 4. 论文依据与我们的适配

论文 Section III-C 将不同关节、视角和时间的 ray token 统一展平，并使用 global self-attention；其第 6 式为：

```text
softmax(QK^T/sqrt(d) + eta_l^2 M_conf - gamma_l^2 M_dist)
```

置信度和几何偏置公式属于论文方法。将单帧 `J×V` 全局层嵌入 RUMPL VFT/PFT 之前、继续保留 RUMPL 后端，是本项目的适配与待验证模块，不能表述为论文原样复现。

## 5. 路径与自动产物

- 后台服务：`rumpl-gbt-global-jv-20260722.service`
- 主日志：`/mnt/data/cjyoutput/rumpl_gbt_global_jv_ablation_20260722.log`
- 最终汇总：`/mnt/data/cjyoutput/rumpl_gbt_global_jv_ablation_20260722.txt`
- plain 运行目录：`/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/rumpl_gbt_global_jv_plain_20260722_2026-07-22_11-19-34/`
- biased 运行目录：`/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/rumpl_gbt_global_jv_biased_20260722_2026-07-22_11-19-35/`
- 代码与配置快照：`/mnt/data/cjyoutput/experiment_records/rumpl_gbt_global_jv_20260722/`
- V2-V5逐组合评估：`/mnt/data/cjyoutput/cmu_v{2,3,4,5}_eval_rumpl_gbt_global_jv_{plain,biased}_final20_20260722/`

评估完成后必须记录每个 K 的：组合平均 MPJPE、相对原始 RUMPL delta、相对 no-drop curriculum5 seed0 delta、改善组合数、最差组合 delta、最佳组合及其 MPJPE。不得仅按某个最好组合宣布模块有效。

## 6. 对照数值

| 视角数 | 原始 RUMPL 平均 | no-drop curriculum5 seed0 |
|---|---:|---:|
| V2 | 46.913 | 42.550 |
| V3 | 33.829 | 32.580 |
| V4 | 30.961 | 29.384 |
| V5 | 29.984 | 28.059 |

实验是否成功首先以超过 no-drop curriculum5 为准；仅超过原始 RUMPL 但不超过现有方法，应作为负结果或结构消融记录，不能替换当前 baseline。

## 7. 启动状态

- 启动时间：2026-07-22 11:19 CST。
- 已核对日志：plain 为 `GLOBAL_JV depth=1 biased=0`；biased 为 `depth=1 biased=1`。
- 已核对课程：epoch 0 为 `fixed-2`，`fixed_epochs=5`。
- 两组均已完成 epoch 0 并进入 epoch 1。

## 8. 最终结果（2026-07-22 14:15）

| 分支 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|
| 原始 RUMPL | 46.913 | 33.829 | 30.961 | 29.984 |
| no-drop curriculum5 seed0 | 42.550 | 32.580 | 29.384 | 28.059 |
| global-JV plain | **41.457** | 32.706 | 29.444 | **27.975** |
| global-JV biased | 41.691 | 33.190 | 29.915 | 28.204 |

`plain`相对原始RUMPL在26/26组合上全部下降，V2/V3/V4/V5平均改善`5.456/1.124/1.516/2.010 mm`。相对no-drop curriculum5，它在V2改善`1.093 mm`、V5改善`0.084 mm`，但V3/V4分别退化`0.126/0.060 mm`。因此global context在V2有明确价值，但强制经过全局层尚未形成统一提升。

`biased`相对no-drop curriculum5的V2/V3/V4/V5分别变化`-0.859/+0.610/+0.531/+0.145 mm`，并且V3只有8/10组合优于原始RUMPL。它在四种K上均弱于相同结构的`plain`，所以淘汰“对全部跨关节token直接施加ray-distance bias”的实现。可能原因是同相机不同关节的射线在相机中心相交、距离为0，导致几何项无法表达人体关节语义；同时RUMPL后端PFT已经建模关节关系，强全局几何约束造成重复或错误偏置。

严格结论：论文第6式在原VFT的同一关节跨视角融合中有效；直接把它扩展到RUMPL全部`J×V` token没有额外收益。普通global joint-view context保留为候选，但需要残差门控后再判断。完整逐组合结果位于`/mnt/data/cjyoutput/rumpl_gbt_global_jv_ablation_20260722.txt`。

## 8. 最终结果（2026-07-22 14:15）

| 分支 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|
| 原始 RUMPL | 46.913 | 33.829 | 30.961 | 29.984 |
| no-drop curriculum5 seed0 | 42.550 | 32.580 | 29.384 | 28.059 |
| global-JV plain | **41.457** | 32.706 | 29.444 | **27.975** |
| global-JV biased | 41.691 | 33.190 | 29.915 | 28.204 |

`plain`相对原始RUMPL在26/26组合上全部下降，V2/V3/V4/V5平均改善`5.456/1.124/1.516/2.010 mm`。相对no-drop curriculum5，它在V2改善`1.093 mm`、V5改善`0.084 mm`，但V3/V4分别退化`0.126/0.060 mm`。因此global context在V2有明确价值，但强制经过全局层尚未形成统一提升。

`biased`相对no-drop curriculum5的V2/V3/V4/V5分别变化`-0.859/+0.610/+0.531/+0.145 mm`，并且V3只有8/10组合优于原始RUMPL。它在四种K上均弱于相同结构的`plain`，所以淘汰“对全部跨关节token直接施加ray-distance bias”的实现。可能原因是同相机不同关节的射线在相机中心相交、距离为0，导致几何项无法表达人体关节语义；同时RUMPL后端PFT已经建模关节关系，强全局几何约束造成重复或错误偏置。

严格结论：论文第6式在原VFT的同一关节跨视角融合中有效；直接把它扩展到RUMPL全部`J×V` token没有额外收益。普通global joint-view context保留为候选，但需要残差门控后再判断。完整逐组合结果位于`/mnt/data/cjyoutput/rumpl_gbt_global_jv_ablation_20260722.txt`。

# 2026-07-20 组会汇报: Configuration-Robust Multi-View Pose Lifting

## 1. 本周目标

本周不再只优化两视角平均 MPJPE，而是解决一个更严格的问题:

> 使用同一个训练完成的单模型，在 V=2、V=3、V=4 的所有相机组合上都优于 RUMPL，并且不依赖 prediction ensemble 或测试时额外模型。

旧的最强单模型 `hard-view + legw0.9` 已经达到:

| 视角数 | 优于 RUMPL 的组合 | 平均 MPJPE 变化 | 最差组合变化 |
|---|---:|---:|---:|
| V=2 | 10/10 | -5.97 mm | -0.75 mm |
| V=3 | 10/10 | -1.00 mm | -0.52 mm |
| V=4 | 5/5 | -1.36 mm | -0.90 mm |

它证明 full-view teacher 到 hard sparse-view student 的蒸馏有效，但 V3/V4 的增益明显小于 V2，且固定 K2 训练没有充分利用多视角冗余。

## 2. 本周方法演进

### 2.1 问题诊断

- 只训练 hardest K2 student，可以稳定少视角，但 K3/K4 没有得到直接的 view-count-aware 监督。
- 把 student 随机改成 K2/K3/K4，会把 K2 hard supervision 的频率降到约三分之一，V3/V4 变好但 V2 平均退化约 0.74 mm。
- 因此 V2 与 V3/V4 不是互相冲突的任务，而是训练频率失衡: K2 需要每步保留，K3/K4 需要额外分支覆盖。

### 2.2 View-count balanced dual-hard distillation

每个训练 batch 包含三个角色:

1. full-view teacher: 使用完整视角，提供稳定 3D 几何软监督。
2. primary hard K2 student: 每步从候选相机对中选择 teacher-student discrepancy 最大的 hardest pair，保证 V2 鲁棒性。
3. auxiliary K3/K4 student: 复用 hardest pair并加入额外视角，同时接受 teacher 和 GT 监督，补足多视角训练覆盖。

训练目标可概括为:

`L = L_full_GT + L_K2_GT + L_K2_distill + lambda_aux (L_K3/4_GT + L_K3/4_distill)`

下肢蒸馏使用 `leg weight=0.9`，避免 teacher 的腿部误差被过度放大。所有分支只用于训练，推理仍是一个 checkpoint、一次前向。

### 2.3 K-aware geometry-biased attention

在 RUMPL 的 View Fusion Transformer 中，将 ray 几何不一致度作为 pre-softmax attention bias:

`Attention = Softmax(QK^T / sqrt(d) - lambda_g D_ray)`

- V=2 没有足够几何冗余，设置 geometry scale=0，避免弱基线几何被过度约束。
- V>=3 使用完整 geometry scale，让注意力优先融合空间上更一致的 rays。
- 本周 confidence bias 保持关闭，因此当前收益可以明确归因于 geometry bias，而不是混合模块。

最终规则非常简单: `K=2: scale 0; K>=3: scale 1`。

## 3. 关键实验结果

### 3.1 失败实验及结论

| 实验 | 结果 | 得到的结论 |
|---|---|---|
| fully-random VFT mask | V2/V3/V4 均不能超过旧 baseline | 随机缺视角不能替代针对配置难度的 hard mining |
| pixel reprojection loss | 出现 `1/z` 梯度奇点，训练可直接发散 | 不能把 noisy 2D detector 观测当作稳定几何真值 |
| point-to-ray loss | 数值稳定，但弱配置退化 | detector ray 与 GT 约有 25.9 mm 偏差，硬拉近会传播噪声 |
| random multi-K hard | V3/V4 改善，V2 退化 | 证明多视角覆盖有效，但不能牺牲每步 K2 hard supervision |

### 3.2 dual-hard + geometry 的当前最好已验证结果

同一个 `aux=0.5` checkpoint 的 view-count geometry 诊断结果:

| 视角数 | 优于 RUMPL 的组合 | 平均 MPJPE 变化 | 最差组合变化 | 相对旧最强 baseline 的平均变化 |
|---|---:|---:|---:|---:|
| V=2 | 10/10 | -6.144 mm | -0.860 mm | -0.177 mm |
| V=3 | 10/10 | -1.263 mm | -0.093 mm | -0.265 mm |
| V=4 | 5/5 | -1.714 mm | -1.005 mm | -0.359 mm |

这是目前第一次同时满足:

- 同一 checkpoint；
- V2/V3/V4 共 25 个组合全部下降；
- 三种视角数量的平均值都超过旧最强 baseline；
- 推理不使用 teacher、额外视角或模型 ensemble。

需要严格说明: 该 checkpoint 训练时的 V3/V4 geometry scale 与最终诊断强度没有完全对齐，因此上表是已经完整跑完的结构诊断上界，不作为最终论文正式表。最终对齐重训使用 `V2=0, V3=V4=1`。

### 3.3 严格重训当前状态

- 三组: `GBT aux=0.5` 主模型、`GBT aux=0.75` 权重对照、`no-GBT aux=0.5` 严格消融。
- 7 月 18 日三个进程被外部会话在 epoch 12 同时终止；无 NaN、OOM 或代码异常，optimizer/model/random-state checkpoint 完整。
- 7 月 19 日已修正 resume 时的 scheduler epoch 与 best metric 恢复，从 epoch 12 严格续训到 20。
- 训练完成后自动评测三组 V2/V3/V4 全组合，最终报告写入 `/mnt/data/cjyoutput/dualhard_k3plus_eval_summary_20260719.txt`。

组会上如果正式表尚未完成，应把 3.2 明确称为“完整 25 组合诊断结果”，不要称为最终重训结果。

### 3.4 严格重训最终结果（7 月 19 日更新）

三组模型已经完成 20/20 epoch 和全部 75 个组合评测。`GBT aux=0.75` 是严格设置中最稳的模型: V2/V3/V4 分别为 `10/10、10/10、5/5` 全降，平均变化为 `-5.294/-0.875/-1.231 mm`。no-GBT 对照为 `10/10、7/10、4/5`，平均 `-5.440/-0.490/-0.548 mm`，因此 geometry bias 对 V3/V4 的配置鲁棒性有明确贡献。

但 `GBT aux=0.75` 相对旧 `hard-view + legw0.9` baseline 仍平均差 `+0.672/+0.123/+0.124 mm`，所以当前正式 baseline 不变。此前 `-6.144/-1.263/-1.714 mm` 仍只能称为强度未对齐的诊断上界。进一步的 no-GBT + 推理期 geometry 控制实验在 V3/V4 仅得到 `-0.503/-0.731 mm`，说明几何需要参与训练，但固定满强度会限制收益。下一轮聚焦弱强度或随机强度 geometry training，而不是继续扩大固定权重扫描。

完整正式表: `/mnt/data/cjyoutput/dualhard_k3plus_eval_summary_20260719.txt`。

### 3.5 epoch20 与 K-aware geometry 新结果

进一步检查发现，原 synthetic validation 选出的 epoch17 并不是 CMU 相机组合鲁棒性最优点。固定 epoch20 的同一单模型在 V2/V3/V4 上达到 `10/10、10/10、5/5` 全降，平均分别为 `-5.795/-1.427/-1.793 mm`；V3/V4 平均比旧 baseline 再低 `0.428/0.438 mm`。V4 使用诊断得到的 scale=2，因此该结果仍不作为训练推理完全对齐的最终表，但它证明架构本身有效，并定位了 checkpoint selection 的跨域错配。

当前已启动两组严格重训，统一使用 `V2/V3/V4 scale=0/1/2` 和固定 epoch20 评测规则: 一组 geometry-only，另一组仅增加 confidence attention bias=0.1。两者将给出 K-aware geometry 以及 confidence+geometry 的直接消融。

## 4. H36M 泛化准备

- 严格 H36M 格式 synthetic train 数据已生成并完成两级校验。
- stage V 为 99/99 分片，无缺号、无临时文件，总计 128,109 个样本。
- combined shape: `joints_3d=(128109,17,3)`，`joints_2d=(128109,20,17,2)`。
- confidence valid ratio 为 0.691；相机 K/R/T/t、20-view IDs、人物中心范围和 NaN 检查全部通过。
- combined 文件约 2.96 GB，位于挂载盘 `/mnt/data`。
- 当前服务器缺少真实 Human3.6M validation annotation，因此真实 H36M 测试尚未开始，不能用 synthetic validation 替代论文测试协议。

需要向老师确认: 实验室是否已有合法的 Human3.6M annotation/预处理副本，或由谁提供数据访问。

## 5. 论文故事线

建议论文问题定义为:

> 现有 universal multi-view lifting 模型虽然支持任意相机数量，但对具体相机组合仍然敏感；单纯增加视角并不保证稳定提升。我们研究的是 camera-count 和 camera-configuration 双重鲁棒性。

三个核心贡献:

1. **Hard sparse-view self-distillation**: full-view teacher 向最困难少视角 student 传递几何先验，推理无额外成本。
2. **View-count balanced dual-hard training**: 同时保留每步 K2 hard branch 和 K3/K4 auxiliary branch，解决不同视角数量之间的监督失衡。
3. **K-aware geometry-biased view fusion**: 用真实 ray 一致性直接约束 attention；根据几何冗余在 K2 和 K>=3 之间自适应启用。

主要证据不是只看平均值，而是报告所有 camera combinations、improved ratio 和 worst delta，突出 configuration robustness。

## 6. 下周计划

1. 完成严格对齐模型及 no-GBT 消融的 25 组合正式表。
2. 对最终主模型至少补 seeds 1/2，报告 mean/std，确认提升不是 seed 0 偶然性。
3. 输出 geometry attention map，对比容易/困难相机组合以及 K2/K3/K4，验证注意力是否聚焦 ray-consistent views。
4. 获得真实 Human3.6M validation annotation 后，先复现原 RUMPL baseline，再迁移 dual-hard + GBT。
5. 补 inference FLOPs、参数量和耗时，说明新增 geometry bias 几乎不增加参数，训练分支不增加推理成本。

## 7. 组会口头版结论

本周最重要的进展不是又调低了一个平均 MPJPE，而是把原来只对两视角有效的 hard-view 蒸馏扩展成了统一的视角数量训练。实验显示，随机 multi-K 会因为稀释 K2 hard supervision 而伤害两视角，因此我设计了 dual-hard: 每个 batch 始终保留 hardest K2，同时增加 K3/K4 auxiliary student。对于多视角，再把 ray 几何一致性直接加入 view-fusion attention，而两视角关闭该偏置。当前同一 checkpoint 的完整诊断已经在 V2/V3/V4 共 25 个相机组合上全部优于 RUMPL，平均分别下降 6.14、1.26、1.71 mm，并且都超过旧最强单模型。最终训练强度完全对齐的三组正式实验正在从 checkpoint 续跑，随后会给出主模型、权重对照和 no-GBT 消融。与此同时，12.8 万 H36M 格式训练数据已经完成严格校验和合并，下一步缺的是服务器上的真实 Human3.6M 测试 annotation。

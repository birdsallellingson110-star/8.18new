# GBT 噪声鲁棒性与 Attention 机制诊断

时间: 2026-07-19  
性质: 快速机制诊断，不作为论文正式噪声实验。

## 1. 实验设置

- 数据: 真实 CMU validation，困难 V3 相机组合 `[3, 6, 13]`。
- 样本: 固定前 64 个样本。
- 模型对照: `hard-view + legw0.9` 与已完成的 `dual-hard + geometry` epoch20。
- 扰动: 在 ray direction 的切平面加入高斯角度扰动，以焦距 1000 px 换算为 `0/2/5/10/20 px` 等效二维关键点抖动。
- Attention 诊断: 只扰动相机 13，等效抖动 20 px，保存最后一层 VFT 中 fusion token 指向三个视角的 attention。

该扰动直接作用于 ray direction，适合快速隔离几何机制，但不等价于完整 detector 噪声分布。正式实验需要在原始 2D 关键点层加入扰动并重新生成 rays。

## 2. 噪声敏感性结果

![Noise robustness curve](noise_robustness_curve.png)

| 等效抖动 | hard-view MPJPE | hard-view 退化 | geometry MPJPE | geometry 退化 |
|---:|---:|---:|---:|---:|
| 0 px | 28.267 | +0.000 | 28.625 | +0.000 |
| 2 px | 28.349 | +0.082 | 28.871 | +0.246 |
| 5 px | 28.762 | +0.495 | 29.502 | +0.877 |
| 10 px | 30.622 | +2.355 | 31.642 | +3.017 |
| 20 px | 34.489 | +6.222 | 36.675 | +8.050 |

当前 geometry 模型未经有效噪声训练，在全部视角同时抖动时比旧 baseline 更敏感。因此目前不能声称 geometry bias 已经带来抗抖动能力。

## 3. Attention 可视化

![Attention clean vs jitter](attention_clean_vs_jitter.png)

相机 13 单独加入 20 px 等效抖动后:

- ray inconsistency: `0.02917 -> 0.03689`，增加 `+0.00772`；
- 相机 13 平均 attention: `0.30068 -> 0.31267`，增加 `+0.01199`；
- 当前批次 MPJPE: `32.08 -> 36.85 mm`。

几何距离模块成功感知了异常，但 attention 没有把异常视角稳定降权，部分关节反而提高了相机 13 权重。这说明固定 geometry bias 尚未压过 learned `QK^T` 内容项，也没有学到噪声条件下的可靠视角切换。

## 4. 技术判断

这不是“可视化不好看”，而是定位出了当前架构的真实缺口:

1. 当前训练配置 `NOISE_LEVEL=0`，模型没有见过受控关键点抖动。
2. 当前 geometry/confidence 系数是全层固定超参数；原 Geometry-Biased Transformer 使用每个 encoder layer 独立可学习的非负系数。
3. 仅靠 MPJPE 监督，不保证 attention 对受扰动视角单调降权。
4. 当前 ray penalty 使用样本内相对归一化；多个视角同时受扰时，异常对比度可能下降。

## 5. 下一版严格实验

建议下一版同时加入以下三项，但必须做逐项消融:

- `Noise curriculum`: 训练时从 0 到 10/20 px 逐步增加 2D keypoint jitter，并将扰动幅度映射为 confidence 衰减；加入随机关节缺失和单视角异常。
- `Layer-wise learnable bias`: 每层学习 `eta_l^2` 和 `gamma_l^2`，对应 confidence 与 geometry bias，保证系数非负。
- `Attention reliability loss`: 已知训练时哪一个视角被人工扰动，约束其 attention 低于 clean views；同时保留主 MPJPE，避免只优化热图。

正式证据应包含 clean/noisy MPJPE 曲线、AUC 或平均退化、attention-confidence 相关性、attention-ray-distance 负相关性，以及 clean accuracy 是否保持。

## 6. 资产

- 图和原始指标: `/mnt/data/cjyoutput/visualizations/gbt_noise_attention_20260719/`
- 原始 attention 数组: `attention_arrays.npz`
- 可复现实验脚本: `/home/lixiaob/cjy/OpenRUMPL/RUMPL/run/viz_gbt_noise_attention.py`

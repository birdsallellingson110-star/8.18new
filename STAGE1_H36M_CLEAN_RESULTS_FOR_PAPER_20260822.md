# 第一阶段 H36M clean 实验结果（论文实验章节事实源）

> 冻结日期：2026-08-22
> 用途：可将本文件直接交给网页版 Codex，作为论文实验章节中 H36M clean、组件消融和失败分析的数值事实源。
> 优先级：本文件中的第一阶段结论高于旧聊天记录和旧实验计划；不得用历史 A1D、CPN、热图、不同缓存或 oracle 数值替换这里的正式结果。

## 1. 第一阶段完成状态

第一阶段的目标是：在 H36M clean 上冻结两种二维坐标前端下的当前最佳模型，统一报告全部两相机、三相机和四相机组合，并为下一阶段 H36M-Occl 固定 clean-trained checkpoint。

截至 2026-08-22：

- ResNet-152 坐标线已经完成 `GQ-RUMPL -> E2-C2 -> H18`；
- HRNet 坐标线已经完成 C2、E2-C2、H18，并完成 Global Joint-Query 迁移审计；
- Global Joint-Query 在 ResNet 上显著有效，但在 HRNet 上没有稳定改善，因此 HRNet 最终线保留原 C2/E2/H18，不把失败的 Query 版本写成最终模型；
- 第一阶段的**当前实验结果已经冻结，可用于论文初稿**；这表示结果与权重已确定，并不表示所有列均已超过 GBT；
- 下一阶段只用这些 clean-trained 模型测试遮挡，不能根据 S9/S11 或遮挡测试结果回选第一阶段权重。

## 2. 公平评估协议

- 数据集划分：H36M `S1/S5/S6/S7/S8` 训练，`S9/S11` 测试；
- 按 GBT/Learnable Triangulation 的评估脚本排除 S9 中已知错误片段；
- 输入层级：3D 网络只读取冻结二维前端输出的 `(x,y,confidence)` 和相机参数/世界射线，不读取 RGB、heatmap、bbox 或二维网络中间特征；
- 二维前端：
  - `HRNet`：YOLOX-X person detector + HRNet-W32 COCO；
  - `ResNet-152†`：Learnable Triangulation 官方 H36M-finetuned ResNet-152；
- 指标：absolute All-17 MPJPE，单位 mm，无 root alignment、无 PA/Procrustes alignment；
- 汇总：action-equal 为主结果；V2 平均全部 6 个相机对，V3 平均全部 4 个三相机组合，V4 使用全部四个相机；
- `T=1` 与 `T=9` 必须分行；H18 当前为中心窗口，会使用未来帧，不能描述成 GBT 的 causal latest-frame 实现；
- checkpoint/epoch/温度只允许在训练主体 holdout 上选择，S9/S11 只做最终评估。

## 3. 第一阶段 clean 主结果

### 3.1 可直接放进论文的主表

| Method | 2D input | T | V2 | V3 | V4 |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation, GBT-reported | ResNet-152† | 1 | 51.1 | 23.4 | 19.1 |
| GBT, reported | ResNet-152† | 9 | **29.9** | 24.4 | 22.7 |
| Ours: `GQ-RUMPL` | ResNet-152† | 1 | 32.312 | 25.101 | 23.536 |
| Ours: `GQ-RUMPL-E2` | ResNet-152† | 1 | 32.319 | **22.558** | **20.272** |
| Ours: `GQ-RUMPL-E2-T` | ResNet-152† | 9 | **31.215** | **22.008** | **19.971** |
| Algebraic Triangulation, GBT-reported | HRNet | 1 | 120.7 | 50.9 | 44.2 |
| GBT, reported | HRNet | 9 | **36.8** | 30.4 | **26.0** |
| Ours: HRNet C2 generator | HRNet | 1 | 38.686 | 30.943 | 28.629 |
| Ours: HRNet C2 + E2-C2 | HRNet | 1 | 38.700 | 29.486 | 27.274 |
| Ours: HRNet C2 + E2-C2 + H18 | HRNet | 9 | **37.704** | **29.231** | **27.219** |

表内粗体只用于突出同一输入下我们的最终行或对应参考最优，排版时可按目标期刊规则重新处理。

### 3.2 与 GBT 报告值的差异

| 输入 | Ours T=9 | GBT T=9 | Ours - GBT (V2/V3/V4) |
|---|---:|---:|---:|
| ResNet-152† | 31.215/22.008/19.971 | 29.9/24.4/22.7 | `+1.315/-2.392/-2.729` |
| HRNet | 37.704/29.231/27.219 | 36.8/30.4/26.0 | `+0.904/-1.169/+1.219` |

正确结论是：

- ResNet 输入下，当前方法在 V3/V4 明显低于 GBT 报告误差，但 V2 仍高 1.315 mm；
- HRNet 输入下，V3 低于 GBT 报告误差，V2/V4 尚未超过；
- 因此不能写“全面超过 GBT”，只能按视角数逐列陈述；
- GBT 没有公开代码和完整前端细节，而且我们的 H18 为 centered、GBT 为 causal，表格属于同协议目标下的报告值比较，不得写成严格逐行复现或完全 matched implementation。

## 4. ResNet 模块消融

| ResNet-152† 组件 | T | V2 | V3 | V4 | 相对上一行变化 |
|---|---:|---:|---:|---:|---:|
| H76 direct reference | 1 | 41.470 | 26.081 | 24.157 | — |
| + Global Joint-Query | 1 | 32.312 | 25.101 | 23.536 | `-9.158/-0.980/-0.621` |
| + E2-C2 standard，两 seed 均值 | 1 | 32.331 | 22.646 | 20.361 | `+0.019/-2.455/-3.175` |
| + identity safeguard，两 seed 均值 | 1 | 32.319 | 22.558 | 20.272 | `-0.012/-0.088/-0.089` |
| + H18，matched dense center | 9 | 31.215 | 22.008 | 19.971 | `-1.222/-0.572/-0.335`* |

`*` H18 的严格增量以相同 25,821 个时序中心样本上的 dense T=1 baseline `32.437/22.581/20.306` 计算，不能直接用稀疏单帧均值相减。H18 在三种视角数上都改善，因此保留到遮挡阶段。

E2-C2 standard 的两 seed 明细：

| Seed | V2 | V3 | V4 |
|---|---:|---:|---:|
| 0 | 32.331 | 22.660 | 20.379 |
| 1 | 32.331 | 22.631 | 20.343 |
| Mean | 32.331 | 22.646 | 20.361 |

E2-C2 identity safeguard 的两 seed 明细：

| Seed | V2 | V3 | V4 |
|---|---:|---:|---:|
| 0 | 32.322 | 22.578 | 20.297 |
| 1 | 32.316 | 22.537 | 20.247 |
| Mean | **32.319** | **22.558** | **20.272** |

identity safeguard 只带来约 `0.01--0.09 mm` 的稳定修正。论文不能把 V3/V4 的主要提升归因于 hinge；主要提升来自 22 个学习/几何候选及其逐关节效用软融合。

## 5. HRNet 模块结果与 Query 迁移失败消融

### 5.1 保留的 HRNet 最终链

| HRNet 组件 | T | V2 | V3 | V4 | 相对上一行变化 |
|---|---:|---:|---:|---:|---:|
| C2 generator | 1 | 38.686 | 30.943 | 28.629 | — |
| + E2-C2 soft calibration | 1 | 38.700 | 29.486 | 27.274 | `+0.014/-1.457/-1.355` |
| + H18 | 9 | 37.704 | 29.231 | 27.219 | `-1.123/-0.407/-0.153`* |

`*` HRNet H18 增量以相同 dense center baseline `38.827/29.638/27.371` 计算。该时间模块在 V2/V3/V4 都没有退化，因此作为当前 HRNet clean baseline 保留。

### 5.2 Global Joint-Query 在 HRNet 上的受控审计

| HRNet Query 实验 | V2 | V3 | V4 | 结论 |
|---|---:|---:|---:|---|
| C2 source | 38.686 | 30.943 | 28.629 | 比较基准 |
| U1：冻结 RUMPL，仅训练 Query | 38.729 | 30.988 | 28.672 | 三列约退化 0.04 mm |
| U2：U1 后全网 `5e-6` 联合适配 | **38.487** | **30.913** | 28.676 | V2/V3 改善，V4 退化 0.047 mm |
| C2-direct A：直接联合适配 | 38.742 | 30.957 | 28.651 | 三列轻微退化 |
| C2-direct B：多视角恢复配比 | 38.921 | **30.870** | 28.712 | 只改善 V3，V2/V4 退化 |

高学习率 V2-specialist 源也没有被 Query 稳定改善：

| 实验 | V2 | V3 | V4 |
|---|---:|---:|---:|
| highLR source | **36.885** | **31.451** | **30.277** |
| + Query A | 37.026 | 31.612 | 30.417 |
| + Query B | 36.905 | 31.465 | 30.280 |

这些负结果说明 Joint-Query 并不是一个对任意二维前端都自动有效的外挂。ResNet 的成功来自全局 Query 在训练过程中补偿其两视角深度歧义；HRNet 的长程 C2 权重已形成不同的置信度/融合分布，冻结 adapter 无新信息，全网联合训练又会扰动已收敛的 VFT/PFT。论文可以把该表作为补充材料中的 detector-distribution ablation，不能把失败版本包装成最终 HRNet 方法。

## 6. 第一阶段可支持的论文结论

目前数据直接支持以下表述：

1. 在 ResNet-152 坐标输入上，VFT 前的 Global Joint-Query 绕过了 RUMPL 逐关节过早压缩，对 V2 的作用最明显（9.158 mm），并同时改善 V3/V4。
2. E2-C2 的主要贡献出现在 V3/V4（约 2.46/3.18 mm），说明可用视角增加后，不同相机子集和稳健几何候选具有可利用的互补性。
3. identity safeguard 不是主要精度来源，而是防止软融合相对 base 轻微负增益的稳定器。
4. H18 在 clean 上为两种前端都提供一致但随视角数减小的收益；ResNet 为 1.222/0.572/0.335 mm，HRNet 为 1.123/0.407/0.153 mm。这符合“视角越少、单帧深度歧义越大，历史信息越有价值”的解释。
5. 当前最好结果为 ResNet `31.215/22.008/19.971 mm`；两视角仍是主要短板，遮挡实验将检验时序与候选效用是否能在检测失败时产生更大收益。

目前数据**不支持**以下表述：

- 不支持“所有输入、所有视角数全面超过 GBT”；
- 不支持“Joint-Query 在 HRNet 和 ResNet 上都有效”；
- 不支持“identity-hinge 是 20 mm V4 的主要创新”；
- 不支持“当前 H18 是因果时序”或“严格复现 GBT 时序”；
- 不支持使用旧 A1D/热图/CPN/oracle 数值补充主表缺失项。

## 7. 可直接改写成论文实验分析的段落

> Under the H36M coordinate-only protocol, the proposed global joint-query branch substantially improves the ResNet-152 generator from 41.47/26.08/24.16 mm to 32.31/25.10/23.54 mm for two, three and four cameras, respectively. The largest gain occurs in the two-camera setting, where early per-joint view compression is most vulnerable to depth ambiguity. The multi-hypothesis utility fusion further reduces the three- and four-camera errors to 22.56 and 20.27 mm while preserving the two-camera result. A lightweight nine-frame residual refinement obtains 31.22/22.01/19.97 mm. The same temporal design also consistently improves the selected HRNet baseline, although transferring the global joint-query branch to the long-trained HRNet C2 generator does not yield a uniform gain. This detector-dependent ablation suggests that query-based residual fusion must be co-adapted with the spatial generator rather than treated as a universally effective post-hoc adapter.

中文写法：

> 在 H36M 坐标级协议下，全局关节查询将 ResNet-152 生成器在两、三、四视角上的误差由 41.47/26.08/24.16 mm 降至 32.31/25.10/23.54 mm，其中两视角收益最大，说明早期逐关节视角压缩在深度歧义较强时构成主要瓶颈。随后，多假设效用融合将三、四视角进一步降至 22.56 和 20.27 mm，同时基本保持两视角精度。九帧轻量时序残差最终得到 31.22/22.01/19.97 mm。相同的时序设计也稳定改善了 HRNet 基线，但全局关节查询迁移到已长期收敛的 HRNet C2 生成器时没有形成三列一致提升，说明该模块需要与空间生成器共同适配，而不能被视为对任意二维前端均有效的后处理插件。

## 8. 审计路径

### ResNet-152

- Global Joint-Query 根目录：`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/`
- T=1 direct：`eval/V2|V3|V4/table2.json`
- E2 standard：`e2_c2/scorer/calibrated_v2t04.json`
- E2 identity：`e2_c2_identity_hinge/calibrated_v2t04.json`
- 修正后的 H18：`h18_identity_hinge_v2_gtinput/result.json`

### HRNet

- C2/E2 主线根目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/`
- H18-lowLR：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/result.json`
- Query U1/U2：`/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/c2_initialized_query_staged/`
- Query C2-direct A/B：`/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/c2_direct_query_ab/`
- Query highLR A/B：`/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/highlr_initialized_query_ab/`

## 9. 下一阶段冻结输入

H36M-Occl 应至少评估以下 clean-trained 模型：

- ResNet：`GQ-RUMPL`、`GQ-RUMPL-E2`、`GQ-RUMPL-E2-T`；
- HRNet：C2、C2+E2、C2+E2+H18；
- 两种前端必须使用相同确定性遮挡 mask；
- 不允许先在遮挡测试集上选择 square size、seed、温度或 checkpoint；遮挡生成与校准细节见 `OpenRUMPL_baseline_audit/OCCLUSION_STAGE_PREPARATION_20260822.md`。

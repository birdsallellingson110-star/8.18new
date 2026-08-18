# RIGR P2：HRNet 中间特征级几何融合实现记录

## 目的

P0 已证明只从 HRNet heatmap 的局部 top-K 模式取二维候选不能解决真实 H36M 的误差：局部二维 oracle 反而达到约 V2/V3/V4 = 99.095/67.034/63.074 mm。因此不再继续调 top-K 半径、候选数或 ray-only Transformer。

P2 改为使用冻结 HRNet 的中间高分辨率特征，在 H76 初始三维查询投影到每个相机 crop 的位置采样 5×5 特征 patch，再进行视角注意力和关节注意力。这个接口对应 MVGFormer 的几何—图像交替条件化思路以及 Epipolar Transformer 的跨视角特征交互，但不改变 RUMPL 的相机泛化协议。

## 已实现文件

- `export_h36m_hrnet_features.py`：逐记录导出 HRNet stage-4 高分辨率分支（当前官方配置实际输出 `[1,32,96,72]`，与 heatmap head 的 `in_channels=32` 一致），保存为 float16 memmap。
- `prepare_rigr_feature_tokens_20260812.py`：将 4 个相机的特征在 H76 full-view 3D query 投影位置采样成 `[group,4,17,32,5,5]` 紧凑 token。
- `train_rigr_hrnet_feature_20260812.py`：冻结 H76 和 detector，只训练 view-shared Transformer + joint Transformer 的 residual refiner；最后输出层零初始化，epoch 0 是严格 H76。

## 数据与路径

- 验证特征：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/RIGR_P2_feature_export_20260812/val_shard*.{npz,features.npy}`，2021 完整四视角组。
- 验证 token：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/RIGR_P2_feature_tokens_20260812/val_tokens.npy`，shape `[2021,4,17,32,5,5]`。
- 训练首轮：4 个 GPU 导出任务各取前 10000 完整组，输出 `train10k_shard*.{npz,features.npy}`；导出完后生成 `train_tokens.npy`。

## 闭环检查

1. HRNet stage-4 hook 的实际形状为 `[1,32,96,72]`；不是配置注释中的 32+64+128+256 拼接，避免把不存在的通道误写入模型。
2. exporter 的输入 crop 元数据与已有 dense heatmap 导出完全一致（center/scale/decoded keypoints 均逐元素一致）。
3. 投影→crop 坐标→feature map 采样 smoke 通过。
4. refiner 零初始化输出与 H76 预测最大绝对差为 0。
5. 输入不读取 GT；GT 只在 supervised loss 和最终报告中使用。

## 首轮结果与采样诊断

首轮严格子集 token（10k 前缀组，全部 subject 1）两 seed 的 S9/S11 结果：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76 | 34.816 | 30.489 | 29.691 |
| feature seed0 | 35.904 | 31.673 | 30.732 |
| feature seed1 | 35.605 | 31.471 | 30.567 |

训练留出集从约 20/16/15 mm 降到约 15.6/11.8/10.7 mm，但 S9/S11 退化；检查发现 10k 前缀完全来自 subject 1，属于明确的 subject domain overfit，不能据此否定特征级结构。已改为 20k 覆盖采样（subjects 1/5/6/7/8，全部 15 actions），正在重新导出。

balanced 20k 使用相同模型、相同损失和相同 S9/S11 协议，区别只是训练组覆盖 subject/action；两 seed 均改善 H76：

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| H76 | — | 34.816 | 30.489 | 29.691 |
| HRNet feature + Transformer | 0 | **32.935** | **29.448** | **28.665** |
| HRNet feature + Transformer | 1 | 33.642 | 30.029 | 29.178 |
| HRNet feature + Transformer + gate | 0 | 33.179 | 29.737 | 28.921 |
| HRNet feature + Transformer + gate | 1 | 33.392 | 29.940 | 29.124 |

非门控两 seed 均值为 V2/V3/V4 = 33.288/29.738/28.921 mm；相对 H76 平均降低 1.528/0.751/0.770 mm。门控两 seed 均值为 33.286/29.839/29.023 mm，没有超过非门控，因此 gate 暂不作为主模型。20k balanced 是可行性规模，不是最终全量训练。

## 协议修正（正式训练前完成）

最初的 full-view token 只在 H76 full-view 预测投影处采样；它不能公平地直接用于 V2/V3，因为会把未参与当前子集的相机信息带入采样位置。该 token 仅保留作实现 smoke，不进入正式结果。

现已加入 `--all-combinations`：对 11 个 V2/V3/V4 子集分别使用对应的 H76 子集预测投影，输出 `[G,11,4,17,32,5,5]`，训练时按当前 combo 读取对应特征，未使用视角保持 padding+mask。1000 组前台 smoke 已通过。

## 判定门槛

- 首轮只看独立 holdout 与 S9/S11 一次性评估，不以训练集下降作为成功标准。
- 若 V3、V4 的 action-equal 平均没有至少约 0.5 mm 改善，停止该小模型，不扩展全量，也不继续盲目调层数/学习率。
- 若 V3/V4 有稳定改善，再做第二个 seed；只有两 seed 方向一致才导出全量训练特征并训练正式模型。
- V2 不能发生明显回退；如出现只改善 V2 而损害 V3/V4，记录为失败的局部修正，不作为主线。

## 2D 辅助量复核（2026-08-12）

为验证中间特征是否还需要显式接收检测器输出，增加了不读取 GT 的 5 维逐关节/逐视角辅助量：

1. 解码 2D 点相对 H76 投影点的归一化残差 `(dx, dy)`；
2. 检测器 score；
3. 解码 crop 坐标 `(x, y)`。

辅助量按当前 V2/V3/V4 子集单独生成，未使用视角为零并由 view mask 屏蔽；因此没有把 full-view 信息泄漏到子集实验。两个种子均使用 balanced20k、11 个子集、8 轮训练和同一 S9/S11 action-equal All-17 评估。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| H76 | — | 34.816 | 30.489 | 29.691 |
| HRNet feature（无显式 2D aux） | 0 | 32.935 | 29.448 | 28.665 |
| HRNet feature（无显式 2D aux） | 1 | 33.642 | 30.029 | 29.178 |
| HRNet feature + 2D aux | 0 | 33.146 | 29.704 | 28.910 |
| HRNet feature + 2D aux | 1 | 33.236 | 29.662 | 28.817 |

两 seed 平均：

- 无 aux：`33.288 / 29.738 / 28.921 mm`；
- 加 aux：`33.191 / 29.683 / 28.864 mm`。

相对于 H76，加 aux 平均降低 `1.625 / 0.806 / 0.827 mm`；相对于已经包含 HRNet 特征的模型，额外仅降低 `0.097 / 0.055 / 0.057 mm`。因此显式 2D aux 可作为轻量消融项，但不足以单独作为主要创新；主要贡献仍来自平衡覆盖训练和 HRNet 中间特征的跨视角/跨关节融合。下一步应测试显式视角质量/几何条件加权，而不是继续重复扩大 aux 维度。

## 学习式视角质量池化复核（2026-08-12）

把跨视角等权平均替换为一个零初始化的 learned view-quality softmax，作为 AdaFuse/视角质量自适应融合的最小结构对照。零初始化时严格退化为等权池化，训练协议、输入 token、H76 query 和 holdout 选点均不变。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + learned quality pooling | 0 | 33.450 | 29.936 | 29.088 |
| HRNet feature + learned quality pooling | 1 | 33.128 | 29.699 | 28.906 |

两 seed 均值为 `33.289 / 29.817 / 28.997 mm`，相对等权 feature 模型均值 `33.288 / 29.738 / 28.921 mm` 没有改善。因此不把“只增加一个 attention pooling”写成主创新，也不继续做温度/层数扫描；后续质量建模必须显式使用可解释的射线几何条件。

## 显式几何质量输入复核（2026-08-12）

又做了一个严格隔离实验：在 HRNet feature token 外加入由 H76 当前查询和标定射线直接计算的 5 维条件量（不读 GT）：检测置信度、查询到射线距离、与其它视角的平均/最大射线夹角正弦、查询深度。每个子集独立计算，未选视角为零并由 mask 屏蔽。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + geometry aux | 0 | 33.209 | 29.799 | 28.984 |
| HRNet feature + geometry aux | 1 | 33.365 | 30.058 | 29.266 |

两 seed 均值为 `33.287 / 29.929 / 29.125 mm`。相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm`，V2 基本不变，V3/V4 分别变差约 `0.19/0.20 mm`。这说明在当前“特征编码→视角 Transformer→等权池化→3D residual”结构中，几何标量拼接没有转化成更好的视角选择；失败原因更可能是融合器没有显式建模跨视角对应，而不是缺少几何输入。该分支停止，不做全量训练或参数扫描。

## Patch attention 复核（2026-08-12）

为检验 5×5 HRNet 局部特征中“中心像素等权平均”是否是瓶颈，加入了每个 patch 内的可学习空间权重。基础 HRNet feature encoder 保留不变，patch 分支最终投影零初始化并作为残差加入，因此训练初始点与无 patch 模型相同。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + patch attention | 0 | 33.056 | 29.626 | 28.779 |
| HRNet feature + patch attention | 1 | 33.446 | 30.007 | 29.180 |

两 seed 均值为 `33.251 / 29.817 / 28.979 mm`。相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm`，V3/V4 分别变差约 `0.079/0.058 mm`，低于预设的 `0.5 mm` 成功门槛。因此停止局部 patch 权重分支；当前结果支持的判断是：瓶颈不在单视角 5×5 patch 的简单空间池化，而在跨视角的几何对应/信息交互方式。

## Pairwise geometry-biased attention 复核（2026-08-12）

前面的 geometry aux 只是把几何标量拼到 token，不能改变视角间的注意力分配。这里改为在 view-axis attention logits 处加入由标定射线计算的 pairwise bias：绝对射线夹角、夹角正弦、归一化射线间距以及两端置信度。bias 的最后投影零初始化；并将四视角 qkv block 写成显式实现，避免通用 Transformer 的三维 mask 慢路径。H76、HRNet、训练数据和评估协议均不变，GT 只用于监督和最终评估。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + pairwise geometry bias | 0 | 32.880 | 29.269 | 28.387 |
| HRNet feature + pairwise geometry bias | 1 | 33.167 | 29.663 | 28.733 |

两 seed 均值为 `33.023 / 29.466 / 28.560 mm`，相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm`，表面改善 `0.265/0.272/0.361 mm`。该结果方向一致但没有达到预设的 `0.5 mm` 主线门槛；更重要的是，这个版本使用了专门的显式 qkv block，而无 bias feature 基线使用 PyTorch 原生 view Transformer，因此还不能把全部差异归因于 bias。下一步先跑“同一显式 qkv block、bias=0”的结构控制；只有 bias 控制差异仍稳定，才保留为几何偏置候选，否则停止该分支。

## 显式跨视角关系 token 复核（2026-08-12）

加入当前视角 token 相对其它有效视角均值的 relation token，并送入原有视角 Transformer。该分支最后一层零初始化，训练前与等权 feature 模型严格相同，作为 Epipolar Transformer“跨视角交互”的轻量结构对照。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + cross-view relation | 0 | 33.454 | 29.860 | 28.961 |
| HRNet feature + cross-view relation | 1 | 33.596 | 29.983 | 29.112 |

两 seed 均值为 `33.525 / 29.921 / 29.037 mm`，相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm` 分别变差约 `0.24/0.18/0.12 mm`。普通差分 MLP 没有提供有效跨视角增益，停止该分支；下一项改为在每个 5×5 feature patch 内学习空间位置权重，检验“特征定位”是否才是当前瓶颈。

## 显式几何质量输入复核（2026-08-12）

又做了一个严格隔离实验：在 HRNet feature token 外加入由 H76 当前查询和标定射线直接计算的 5 维条件量（不读 GT）：检测置信度、查询到射线距离、与其它视角的平均/最大射线夹角正弦、查询深度。每个子集独立计算，未选视角为零并由 mask 屏蔽。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + geometry aux | 0 | 33.209 | 29.799 | 28.984 |
| HRNet feature + geometry aux | 1 | 33.365 | 30.058 | 29.266 |

两 seed 均值为 `33.287 / 29.929 / 29.125 mm`。相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm`，V2 基本不变，V3/V4 分别变差约 `0.19/0.20 mm`。这说明在当前“特征编码→视角 Transformer→等权池化→3D residual”结构中，几何标量拼接没有转化成更好的视角选择；失败原因更可能是融合器没有显式建模跨视角对应，而不是缺少几何输入。该分支停止，不做全量训练或参数扫描。
## Patch attention 复核（2026-08-12）

为检验 5×5 HRNet 局部特征中“中心像素等权平均”是否是瓶颈，加入了每个 patch 内的可学习空间权重。基础 HRNet feature encoder 保留不变，patch 分支最终投影零初始化并作为残差加入，因此训练初始点与无 patch 模型相同。

| 方法 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet feature + patch attention | 0 | 33.056 | 29.626 | 28.779 |
| HRNet feature + patch attention | 1 | 33.446 | 30.007 | 29.180 |

两 seed 均值为 `33.251 / 29.817 / 28.979 mm`。相较无 aux feature 均值 `33.288 / 29.738 / 28.921 mm`，V3/V4 分别变差约 `0.079/0.058 mm`，低于预设的 `0.5 mm` 成功门槛。因此停止局部 patch 权重分支；当前结果支持的判断是：瓶颈不在单视角 5×5 patch 的简单空间池化，而在跨视角的几何对应/信息交互方式。
### 控制实现审计更正

首次启动的 `balanced_explicit_control_seed0/seed1` 命令是在补齐
`explicit_view_block` 构造参数之前执行的；其 `result.json` 中虽记录了 control
标记，但实际模型仍走原始 `nn.TransformerEncoder` 路径，因此不能作为 pairwise
bias 的结构控制，也不用于归因。代码已修正，后续使用新的
`balanced_explicit_control2_seed0/seed1` 目录重新训练；旧目录只保留作审计记录。

## E2 级联互补性诊断（2026-08-12）

E2 不是与 RIGR/HRNet 特征分支重复的 Transformer：E2 对冻结的多候选 3D
姿态做逐关节 utility scoring 和 soft fusion；RIGR 特征分支则从冻结 H76
query 投影位置读取 HRNet 中间图像特征并修正候选。因而两者理论上分别作用于
“候选内容”和“候选选择”，可以按

```text
H76 候选 → HRNet 特征逐候选修正 → E2 utility/soft fusion
```

级联。为先验证互补性，在完全相同的 H36M S9/S11 缓存、action-equal All-17
协议下做了推理级诊断：只替换 E2 22 候选池中的前 11 个 H76 候选，六个
pairwise 和五个 learned-triangulation 候选保持冻结；E2 checkpoint、温度
`T=1.8` 和评估均不变。该结果不是联合训练结果，不能直接作为最终论文主表。

| 级联输入 | seed | V3 | V4 |
|---|---:|---:|---:|
| E2（22 候选） | 0 | 29.3813 | 28.6382 |
| E2（22 候选） | 1 | 29.3717 | 28.6141 |
| HRNet 特征修正 → E2 | 0 | **28.6743** | **27.9253** |
| HRNet 特征修正 → E2 | 1 | **28.6713** | **27.9089** |
| 两 seed 均值的额外下降 | — | **0.704** | **0.709** |

为了排除几何偏置带来的结构混淆，又用同一显式 qkv、`bias=0` 的控制模型做
级联，结果为 V3 `28.7688/28.7672`、V4 `28.0055/27.9877`；相对 E2 仍有
约 `0.61/0.63 mm` 的稳定下降。因此主要互补收益来自 HRNet 中间图像证据，
不是更换 view Transformer 的偶然影响；几何偏置另有约 `0.10/0.08 mm` 的
级联增益。

诊断脚本和输出：

- `evaluate_rigr_e2_cascade_20260812.py`
- `.../RIGR_E2_Cascade_Diagnostic_20260812/result.json`
- `.../RIGR_E2_Cascade_Control_20260812/result.json`

下一步必须在训练集上重新导出逐候选 RIGR 结果，并在该结果上重新训练两个
E2 utility seed；只有联合训练后的增益仍保持，才把它作为正式的“图像证据
修正 + 候选效用融合”主线结果。直接使用当前推理级级联只能作为互补性证据，
不能声称 E2 已经与 RIGR 联合优化。

## RIGR→E2 候选级联正式重训（2026-08-12）

按上述要求，已使用之前固定的覆盖均衡 20k 训练组生成逐候选 RIGR 修正结果，
把前 11 个 H76 候选替换为 HRNet 特征修正候选，保留 6 个 pairwise 和 5 个
learned-triangulation 候选，随后完全重新训练 E2（10 direct + 5 GHT，
`T=1.8`，soft-oracle `5 mm`，同一 modulo-10 holdout）。两张 GPU 使用不同
seed 并行，未用 S9/S11 选 epoch。

正式输出：

- `.../RIGR_E2_Cascade_Retrain_20260812/seed0_full/result.json`
- `.../RIGR_E2_Cascade_Retrain_20260812/seed1_full/result.json`
- `.../RIGR_E2_Cascade_Retrain_20260812/manifest.json`

| 方法 | seed | V3 | V4 |
|---|---:|---:|---:|
| RIGR 特征候选（E2 输入前） | 0/1 | 29.2691 | 28.3869 |
| RIGR 修正候选 → E2（正式重训） | 0 | **28.5566** | **27.8036** |
| RIGR 修正候选 → E2（正式重训） | 1 | **28.5539** | **27.8121** |
| 两 seed 均值 | — | **28.5553** | **27.8078** |

E2 末端融合相对同一 RIGR 候选输入额外下降约 `0.714 mm (V3)` 和
`0.579 mm (V4)`；相对严格 H76 `30.4890/29.6913`，总下降约
`1.934/1.884 mm`。两 seed 方向一致，且正式重训比冻结 E2 的推理级级联
（约 `28.673/27.917`）进一步下降约 `0.12/0.11 mm`，说明 E2 重新适应修正
候选后确实学到了互补关系。

该结果已经足以把 E2 从“可选末端模块”提升为当前主线候选：

```text
H76/RUMPL 候选生成 → HRNet 中间特征逐候选修正
                  → E2 Set Transformer utility/soft fusion
```

仍需补做一项最终严谨性实验：用 RIGR feature seed1 重新导出一套训练/验证
候选并重训至少一个 E2 seed，确认 `28.55/27.81` 不依赖单个 RIGR checkpoint。
在此之前，正式表述应称为“两枚 E2 seed + 一枚固定 RIGR 候选修正器”，不应
声称已经完成三 seed 端到端联合优化。

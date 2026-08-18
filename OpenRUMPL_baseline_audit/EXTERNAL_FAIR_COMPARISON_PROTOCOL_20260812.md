# H36M 外部公平比较与官方代码复现协议（2026-08-12）

## 1. 目的

当前 RUMPL 真实 H36M 实验已经能用于同一代码库内的消融，但还不能把所有
数字直接与外部论文横向比较。根本原因不是 MPJPE 公式本身，而是不同工作使用
了不同的二维主干、人体框、图像预处理、抽帧和相机数量。

本文档把结果拆成两张互不混用的表：

1. **作者协议复现表**：官方代码、官方权重、官方预处理和官方指标；用于证明
   我们能运行并理解公开方法。
2. **统一输入受控表**：完全相同的 H36M split、帧、人体框、二维检测、相机组合
   和 absolute MPJPE；用于证明模块本身的增益。

内部使用 A1D/H21 修正二维输入的结果只放在第三张“增强输入消融表”，不能作为
外部公平主表。

## 2. 不再作为实现锚点的方法

### Geometry-Biased Transformer（GBT）

- 论文有结果和文字描述，但没有官方代码。
- 可以引用其公开实验结果和思想，但无法核验关键实现细节、训练细节和复现波动。
- 因此不再依据 GBT 猜测模块，也不把它作为首要复现基准。

## 3. 已获取并锁定的官方代码

| 方法 | 发表 | 官方仓库 | 本地目录 | 固定提交 | 用途 |
|---|---|---|---|---|---|
| Learnable Triangulation (LT) | ICCV 2019 Oral | `karfly/learnable-triangulation-pytorch` | `reference/learnable-triangulation-official` | `8dcc4e9` | 可微代数三角化、视角置信度、体素特征融合；第一复现锚点之一 |
| Cross View Fusion (CVF) | ICCV 2019 | `microsoft/multiview-human-pose-estimation-pytorch` | `reference/cross-view-fusion-official` | `3b42262` | 二维热图级跨视角融合与人体结构先验 |
| Epipolar Transformer (ET) | CVPR 2020 | `yihui-he/epipolar-transformers` | `reference/epipolar-transformers-official` | `21aa1f3` | 图像特征级极线注意力； correspondence 模块的权威来源 |
| AdaFuse | IJCV | `zhezh/adafuse-3d-human-pose` | `reference/adafuse-official` | `74fcb7d` | 极线热图融合、关节级视角质量权重；坏视角负增益主线 |
| Generalizable Human Pose Triangulation (GHT) | CVPR 2022 | `kristijanbartol/general-3d-humans` | `reference/general-3d-humans-official` | `14805c7` | 随机多假设与评分、未知相机布局泛化；E2/候选评分的权威来源 |

DeProPose 虽有公开仓库，但代码和实验资产的成熟度、可复现性弱于以上五项，暂不
进入第一批主表。

## 4. 不能直接横比的已核实差异

### 4.1 LT 官方协议

- H36M 训练主体 S1/S5/S6/S7/S8，验证 S9/S11；四台相机。
- `human36m-multiview-labels-GTbboxes.npy`：使用 H36M 分割标注生成的 GT 框。
- 图像预先去畸变，输入为 384×384。
- ResNet-152 二维主干先在 COCO 训练，再在 MPII 与 H36M 上联合微调。
- 官方 README 报告：Algebraic 绝对 MPJPE 19.2 mm，Volumetric 17.7 mm；
  README 模型表同时报告 pelvis-relative 22.5/20.4 mm。

所以 LT 的 19.2/17.7 mm 不能直接与“COCO-only HRNet + 其他框”的结果横比。

### 4.2 AdaFuse 官方协议

- 使用 H36M-Toolbox 生成的 PKL，训练 S1/S5/S6/S7/S8、测试 S9/S11，四相机。
- ResNet-152，输入 384×384、热图 96×96，`USE_GT_BBOX=true`。
- H36M 训练还使用 MPII 增强二维主干。
- 官方 README 报告同一模型内：NoFuse 22.94、HeuristicFuse 21.02、
  ScoreFuse 20.14、RANSAC 21.77、AdaFuse 19.54 mm。

这组同一仓库/同一输入下的消融非常有价值：AdaFuse 相对 NoFuse 的约 3.4 mm
提升是可核验目标，但它不等价于在 COCO-only HRNet 点上必然也提升 3.4 mm。

### 4.3 当前 RUMPL 真实 H36M 协议

- 训练主体和测试主体与上述论文一致。
- 当前 `annot_filtered_5_64` 训练 PKL 已经是原始视频每 5 帧采样；验证约每 65 帧。
- 训练共 78,047 个同步四相机时刻、312,188 条视图记录；验证共 2,021 个同步
  四相机时刻、8,084 条记录。
- AdaFuse 官方 loader 还会对原始 group 做训练 `::20`、验证 `::64`。若直接读取
  当前 PKL 会形成二次抽帧：训练约每 100 原始帧、验证约每 4,160 原始帧，协议错误。
- 当前常用内部最好输入含 annotation box 与 A1D/H21 后处理，不能作为 raw HRNet
  外部公平输入。

## 5. 固定比较口径

### 表 A：作者协议复现（不跨行宣称模块优劣）

每个方法严格按其官方代码、官方权重和官方数据预处理运行，记录：

- 原论文/README 数字；
- 本机复现数字；
- 相机数、输入主干、框、是否去畸变、absolute/root-relative；
- 代码提交、权重 hash、样本数和失败样本过滤规则。

优先级：LT Algebraic → AdaFuse → LT Volumetric → CVF/ET → GHT。

### 表 B：统一输入受控比较（论文主消融依据）

固定：

- S1/S5/S6/S7/S8 train，S9/S11 test；
- 相同物理帧和损坏序列过滤；
- 同一个未使用 A1D/H21 的原始 2D 检测缓存；
- 相同人体框；
- absolute MPJPE、17 joints、无刚性/尺度对齐；
- V2 平均 6 种组合、V3 平均 4 种组合、V4 唯一组合；
- action-equal 与 frame-weighted 同时保留，主表明确标注；
- 所有方法从同一 2D 坐标、置信度和相机参数开始。

首批方法：

1. 线性/DLT；
2. confidence-weighted DLT（LT）；
3. RANSAC；
4. GHT 多假设评分；
5. RUMPL；
6. RUMPL + correspondence；
7. RUMPL + geometry bias；
8. RUMPL + E2 候选评分；
9. 上述已证明互补模块的组合。

AdaFuse/ET/CVF 会改变二维热图或图像特征，需另设“同一 RGB/同一框/同一二维主干”
子表，不能假装它们只使用与坐标方法相同的信息。

### 表 C：增强输入消融（只说明系统上限）

放置 A1D/H21、annotation-box 修正、correspondence+E2 等结果。必须明确标成
`enhanced 2D`，不与外部 raw detector 数字混排。

## 6. 当前可保留的内部结果（仅表 C / 同流水线消融）

| 方法 | V2 | V3 | V4 | 口径 |
|---|---:|---:|---:|---|
| strict H76 / RUMPL | 34.8163 | 30.4890 | 29.6913 | action-equal, All-17 absolute；输入为 A1D/H21 enhanced 2D，**不是 raw HRNet** |
| correspondence + geometry bias + V234 E2（两种子均值） | 32.7701 | 28.6679 | 27.9273 | 与上行同输入/同评估 |
| V3/V4 specialist + correspondence+bias+E2（两种子均值） | — | 28.4148 | 27.6475 | 视角数专用头 |

这些数字证明当前模块在内部公平设置下有效，但在完成表 A/B 前不能声称优于外部方法。
尤其不能用 H76 数字与 raw COCO-HRNet 三角化直接计算“RUMPL 主干增益”，因为
H76 的训练和测试配置都明确使用
`mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap`。raw HRNet RUMPL 必须另行评估并
训练，才能进入统一输入表。

## 7. 执行顺序和通过条件

1. 下载官方权重到 `/mnt/data/cjyweights/external_multiview_h36m/`，不占系统盘。
2. 对 AdaFuse 数据 loader 禁止二次抽帧，仅作本地协议适配；保留原始官方代码提交。
3. 先运行 AdaFuse 官方权重，必须同时导出 NoFuse 与 AdaFuse，核对官方所述约
   22.94/19.54 mm 的量级与约 3.4 mm 内部提升。
4. 运行 LT Algebraic 官方权重；若缺少其 GT segment bbox/去畸变资产，明确标记为
   “近似作者协议”，不伪称精确复现。
5. 导出统一 raw detector 缓存，在同一输入上运行表 B。
6. 表 A/B 建立后，再决定优先移植 AdaFuse 的关节级视角权重、ET 的极线特征采样，
   还是 LT 的可微置信度三角化。

通过标准：

- 官方代码在本机复现与公开值差距可解释，样本数和指标一致；
- 统一输入下至少一个公开模块能稳定超过 RUMPL baseline，且两种子一致；
- 2→3→4 视角平均性能单调改善；
- 后续论文中的主表、消融表不再混用检测器和框。

## 8. 2026-08-13 执行记录

### 8.1 权重与严格加载

官方 LT 权重保存在挂载盘：

- `pose_resnet_4.5_pixels_human36m.pth`：275,353,380 bytes，SHA256
  `2fbab6c5a6b220fc10e1f945570a1b826bb1c217929bb5cdb44c1e99f04e74f2`；
- Algebraic `weights.pth`：SHA256
  `bd7927420ce60aec824e8e9473e3a1c20293c13f77d079f798a0bfedf1fffbed`。

Algebraic 检查点在官方网络上以 `strict=True` 加载，970 个张量全部匹配，无缺失、
无额外键、无形状适配。

### 8.2 统一 raw HRNet 坐标输入诊断

统一验证集为 2,021 个同步时刻，所有 V2/V3/V4 相机组合；输入为
`mmpose_hrnet_coco_legswap`，未使用 A1D/H21。

| 方法 | V2 | V3 | V4 | 说明 |
|---|---:|---:|---:|---|
| GT 2D + undistorted ray mean | 0.482 | 0.304 | 0.277 | action-equal absolute MPJPE；证明标定、同步、单位和射线正确 |
| raw HRNet + undistorted IRLS | 91.913 | 55.684 | 50.583 | 坐标级稳健几何上限诊断 |
| H15 RUMPL R0 | 91.521 | 72.443 | 63.954 | 真实 H36M、随机 2--4 视角训练、原始 RUMPL |
| H15 RUMPL + conf/geometry bias | **90.523** | **70.834** | **62.699** | 仅加入偏置，其余训练与输入不变 |

偏置相对原始 RUMPL 分别提升 0.999/1.609/1.254 mm，说明偏置有效，但不能弥补
COCO-only HRNet 观测与当前真实数据训练主干的巨大差距。H15 检查点还暴露了一个
公开代码细节：随机视角前向虽支持 K=2--4，检查点仍保存一个不参与该前向的
`weighted_mean` 2 视角张量。评估器现将“严格加载时的模型构造视角数”和“测试
数据视角数”解耦；所有权重仍为 `strict=True` 原样加载。

### 8.3 官方 LT 受控图像协议（已完成）

由于尚缺作者专用 `human36m-multiview-labels-GTbboxes.npy`，先运行受控子表：

- 官方 Algebraic 网络与官方完整权重；
- 当前 H36M annotation box；
- 官方 384x384、ImageNet normalization；
- 去畸变版本与不去畸变消融并行；
- V2 平均全部 6 对，V3 平均全部 4 组，V4 唯一组合；
- action-equal 与 frame-weighted、absolute 与 pelvis-relative 全部导出。

H36M-Toolbox PKL 相对
LT 预处理存在已知的下肢语义反向，17x17 通道诊断得到官方预测通道到目标通道为
`[5,4,3,2,1,0,6,7,8,9,10,11,12,13,14,15,16]`；躯干和双臂全部 identity。
这只用于标签语义转换，不修改网络输出坐标或权重。

全量 2,021 个同步时刻结果：

| 官方 LT Algebraic，当前 annotation box | V2 | V3 | V4 |
|---|---:|---:|---:|
| 去畸变，action-equal absolute | **53.592** | **24.339** | **19.921** |
| 去畸变，frame-weighted absolute | 52.758 | 24.566 | 20.046 |
| 不去畸变，action-equal absolute | 72.746 | 26.659 | 22.012 |

去畸变相对不去畸变在 V2/V3/V4 分别改善 19.154/2.320/2.091 mm。V4 19.921 mm
距离官方 README 19.2 mm 仅 0.721 mm；当前仍未使用作者专用 segment GT box，因此
应称为“官方权重 + 受控 annotation-box 协议”，不能伪称完全作者协议复现。

V2 的高均值来自相机几何差异，而不是所有相机对都差：

| V2 相机对 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|
| 去畸变 action-equal | 31.924 | 34.776 | 92.188 | 105.991 | 26.985 | 29.690 |

这与下一阶段的“几何条件/视差敏感度驱动视角效用”直接相关：固定平均所有相机对时，
两个退化相机对把 V2 拉到 53.592 mm；选到良好基线时已经约 27--35 mm。

可学习置信度的严格消融（完全相同 LT 2D 坐标）为：

| 三角化权重 | V2 | V3 | V4 |
|---|---:|---:|---:|
| uniform | 54.553 | 29.005 | 26.248 |
| LT learned joint/view confidence | **53.592** | **24.339** | **19.921** |

因此学习置信度在 V3/V4 独立贡献 4.666/6.327 mm；它是当前最强的有官方代码
支撑的模块证据，不能再将 LT 的全部提升笼统归因于 ResNet-152。

逐组合检查进一步说明了作用边界：学习置信度对六个 V2 相机对只改善
0.259--1.436 mm，而对四个 V3 组合改善 4.015--5.193 mm。两视角没有冗余观测可供
抑制，重加权不能修复 1-4、2-3 两个几何退化相机对；V2 后续应优先验证体积姿态
先验或相机对条件化融合，而不是继续单独放大 confidence bias。

冻结 H76 的 LT 输入分布迁移复核已经完成。第一轮临时关闭 H76 原 lower-body swap
得到 72.440/54.362/51.238 mm；恢复 H76 原协议后为
73.679/55.555/52.517 mm。随后投影语义审计确认：新 LT 缓存与 H36M 3D 真值已经
同序，验证时不应再执行旧缓存使用的 lower-body swap；不交换时 LT 2D 到未畸变
3D 投影的抽样均值为 4.80 px，交换后腿部均值升至 56.66 px。因此上述两次冻结
H76 结果都只作为分布失配诊断，不能进入最终公平表。旧 H76 权重不能直接拼接强
LT 观测；下一步必须用 `FLIP_LOWER_BODY_KP_TEST=false` 在完全相同 LT 训练缓存上
从头比较原始 RUMPL 与 tri-anchor/Plücker H76 结构。

另一个已修正的接口差异是置信度定义：LT 输出是正的跨视角权重，原始均值仅
0.0344，并非 detector visibility probability。若直接套 RUMPL 的 0.1 阈值，仅
0.55% 的关节/视角会被视为可见；官方做法是在当前相机子集内逐关节归一化。新
同输入训练通过显式 `RUMPL_NORMALIZE_VIEW_CONFIDENCE=1` 在模型完成 2/3/4 视角
子集选择后再归一化，原有 RUMPL 实验默认行为不变。

全量结果输出位置：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_official_controlled_full2021_undistort.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_official_controlled_full2021_distorted.json`

JSON 与日志均已写完；smoke 文件只用于管线检查，不进入论文主表。

### 8.4 官方 LT Volumetric：两视角退化修复

官方 Volumetric 权重已完整下载到挂载盘并严格加载：80,588,050 个参数、1,311 个
状态张量全部匹配；权重 SHA256 为
`cd1f443f23a173dba43b193165e187f2a4873073fea469b5c3bdac94d52e6757`。
受控适配对每个相机组合只使用该组合的 RGB/特征，并用同组合的官方 Algebraic
预测计算体积中心，未通过预计算 pelvis 泄漏未选视角。

全量 2,021 个同步时刻、六个 V2 相机对的 action-equal absolute MPJPE 为：

| 方法 | V2 平均 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LT Algebraic | 53.592 | 31.924 | 34.776 | 92.188 | 105.991 | 26.985 | 29.690 |
| LT Volumetric | **45.505** | **29.013** | 40.868 | **69.032** | **74.523** | 28.504 | 31.092 |

Volumetric 总体提升 8.087 mm，主要将两个退化对 1-4、2-3 分别改善 23.156、
31.468 mm；同时 1-3、2-4、3-4 略有退化。这验证了“人体姿态/图像体积先验能修复
两视角几何退化”，也说明最终模型应做条件化残差或门控，而非无条件用体积分支
替换所有相机对。当前 45.505 mm 已显著前进，但尚未达到 V2 < 40 mm 目标。

V3/V4 的全量结果为：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| LT Algebraic | 53.592 | 24.339 | 19.921 |
| LT Volumetric | **45.505** | **22.921** | **19.876** |

Volumetric 相对 Algebraic 在 V2/V3/V4 分别提升 8.087/1.418/0.045 mm，收益随视角
冗余增加迅速缩小。这是“体积人体先验主要修复少视角退化、不能视为通用替代”的
直接证据。V3 四个组合为 23.844/22.050/23.770/22.019 mm。

### 8.5 强 LT 输入上的 RUMPL 同输入结构消融（进行中）

训练和验证均使用新导出的官方 LT Algebraic 二维点、官方跨视角置信度语义、
去畸变 annotation-box 协议；三组实验 seed、损失、优化器和视角课程完全相同：

- R0：公开 RUMPL 主干，不加三角化锚点；
- TA：R0 + 置信度加权最小二乘三角化锚点；
- H76：TA + 锚点中心化射线 + Pluecker 线坐标。

新 LT 缓存已经通过投影语义审计，训练/验证明确
`FLIP_LOWER_BODY_KP_TEST=false`；置信度在实际选中相机子集内逐关节归一化。
旧缓存语义下启动的第一次临时试验不符合该协议，已停止且不进入任何结果表。

早期公平结果：

| 方法/检查点 | V2 action-equal | V4 frame-weighted | 说明 |
|---|---:|---:|---|
| R0 epoch 3 | 96.728 | 96.639 | 原始绝对坐标回归仍不稳定 |
| TA epoch 3 | 46.037 | 22.568 | 三角化锚点贡献主要提升 |
| H76 checkpoint 2（日志 epoch 1） | 45.489 | **21.232** | 当前 V4 最好 |
| H76 checkpoint 3（日志 epoch 2） | **44.796** | 31.023 | 当前 V2 最好 |

H76 的 V2 checkpoint 2/3/4/5 分别为 45.489/44.796/48.395/45.848 mm，说明 V2
与 V4 最优轮次并不相同，且 1e-4 学习率下存在明显波动；不能用 V4 单指标选模，
也不能默认末轮最好。还必须注意：训练课程前 8 个日志 epoch 固定只取两个视角，
因此 checkpoint 3 的 V4 恶化首先是视角数域偏移的直接证据，不应全归因于优化器。
现已启动完全相同配置、只将学习率从 1e-4 降至 3e-5 的稳定性对照；同时保留
“从 epoch 0 即随机训练 V2/V3/V4”的直接课程对照，二者都逐轮保存并分别评估
V2/V4。

H76 epoch 2 与 TA epoch 3 的 V2 分别为 45.489 和 46.037 mm，Pluecker/中心化射线
在三角化锚点之上仅再改善 0.548 mm，说明主要收益来自显式几何锚点。H76 epoch 3
逐相机对为 30.169/34.600/70.290/77.142/27.384/29.192 mm，剩余误差仍集中于
1-4、2-3 两个退化对。坏对的 root-relative MPJPE 也约 77--86 mm，因此后续必须
同时修复绝对深度和相对骨架，不能只做 root translation 校正。

所有逐轮检查点、V2 预测和 Table-2 JSON 位于：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_input_rumpl_ablation/epoch_checkpoints`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_input_rumpl_ablation/epoch_eval`

### 8.6 H76 与官方 Volumetric 的互补性诊断

两条推理管线的逐样本 GT 已严格对齐：2,021 帧、六个相机对，跨管线 GT 平均差
`3.81e-5` mm、最大差 `2.21e-4` mm。H76 checkpoint 3 与官方 Volumetric 的结果：

| 融合/选择方式 | V2 action-equal | 是否可作为最终结果 |
|---|---:|---|
| H76 | 44.796 | 是 |
| Volumetric | 45.505 | 是 |
| 全局固定坐标融合（验证集选 alpha） | 41.541 | 否，仅上限诊断 |
| 按相机对选分支 | 43.957 | 否，验证集 oracle |
| 逐帧二选一 | 39.612 | 否，GT oracle |
| 逐关节二选一 | 32.526 | 否，GT oracle |
| 17 个关节固定 alpha | 39.984 | 否，验证集拟合 |
| 17 个关节 alpha，留一动作拟合 | 40.066 | 否，仍使用 S9/S11 其余动作 |

这组结果否决了“整帧大门控”作为优先路线：其 oracle 仅刚过 40 mm。关节级互补
上限非常大，且固定关节权重就接近 40 mm，因此下一步先只在训练 subjects
S1/S5/S6/S7/S8 上拟合 17 个样本无关系数，再冻结到 S9/S11；只有该合法对照成立，
才进一步训练有论文依据的关节级条件融合头。

互补诊断和逐样本导出位于：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/h76_epoch3_volumetric_complementarity.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_vol_controlled_full2021_v2_predictions.npz`

### 8.7 训练主体拟合的合法融合与首次突破 40 mm

为避免测试集拟合，使用 S1/S5/S6/S7/S8 每动作随机 50 个同步时刻（共 750 帧、
4,500 个 V2 相机对）训练，S9/S11 全量 2,021 帧只作最终评估。两条预测管线的
GT 对齐平均差约 `3.8e-5` mm、最大差约 `2.2e-4` mm。

| 方法 | V2 action-equal | 训练/选择协议 |
|---|---:|---|
| H76 checkpoint 3 | 44.796 | 单分支 |
| LT Volumetric | 45.505 | 官方权重 |
| 17 个固定关节权重 | 40.271 | 只在训练 subjects 拟合 |
| 相机对×关节静态权重 | 40.118 | 只在训练 subjects 拟合 |
| 训练主体 CV 收缩静态权重 | 40.095 | 收缩系数只在训练主体留一 CV 选择 |
| AdaFuse-inspired adaptive joint gate，seed 0 | 39.016 | 架构只在训练主体留一 CV 选择 |
| 同上，seed 1 | **38.995** | 独立随机种子复核 |

动态门控不读取 GT、动作或测试主体身份。输入仅含 H76/Volumetric 的逐关节三维
分歧、各自 pelvis-relative pose、骨长偏差、LT confidence、joint ID 和 camera-pair
ID；输出逐样本逐关节的 0--1 凸融合权重。静态相机对×关节 logit 为初始化，小型
两层 MLP 只学习有限幅度残差。五个候选结构全部在 S1/S5/S6/S7/S8 留一主体 CV
中比较，选中 hidden=16、max-logit-delta=2、dynamic regularization=1e-3，然后才
冻结评估 S9/S11。

门控在六个相机对上均优于两条单分支；seed 1 的逐对结果为
26.849/32.154/59.134/64.576/24.991/26.266 mm。两个退化对仍是主要误差来源，但
相对 H76 的 70.290/77.142 mm 已分别改善 11.156/12.566 mm。

随后以“从 epoch 0 按 3:1:1 随机训练 V2/V3/V4”的统一 H76 checkpoint 6 替换
固定 V2 课程主分支：单分支 V2=44.734、V4=21.141 mm；相同训练主体 CV 门控后
V2 进一步为 **38.884 mm**。因此当前已同时满足原目标 V2<40、V4<30，且 V4 留有
很大余量。当前主候选是统一 H76 + 官方 Volumetric + 关节级自适应质量门控。

主要结果与权重：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_branch_gate_v234start_epoch6_seed0.json`
- `/mnt/data/cjyweights/external_multiview_h36m/adaptive_joint_branch_gate_v234start_epoch6_seed0.pth`

动态特征删除 camera-pair ID 后，V2 为 38.956 mm；第二随机种子为 38.886 mm。
审计随后发现该版本的静态初始化仍为 camera-pair×joint 权重，因此这些数字只能
证明“动态残差不依赖相机 ID”。修正后的严格版本同时删除静态与动态 camera-pair
ID，静态初始化只保留全局 17 个 joint 权重，得到 **38.993 mm**；相对 full 版本
只差 0.109 mm。因此主方法采用严格相机无关版本，full 版本只作 H36M 上限消融。
严格版本第二随机种子为 **38.962 mm**，两种子均值 38.978 mm、半极差 0.016 mm。

门控特征消融（统一 H76 checkpoint 6）：

| 动态门控输入 | V2 action-equal |
|---|---:|
| 仅两分支 3D disagreement + joint ID | 39.829 |
| + pelvis-relative pose + confidence（去 bone feature） | 39.229 |
| + bone consistency（去 confidence） | **38.944** |
| + confidence、严格去 camera-pair ID | 38.993 |
| full（含 camera-pair ID） | 38.884 |

几何/姿态上下文相对纯分歧门控贡献约 0.87 mm；骨长一致性独立贡献约 0.285 mm；
LT confidence 在 V2 上没有稳定额外收益（差 0.013 mm），与官方置信度在两视角
缺少冗余时只能改善约 0.3--1.4 mm 的前述观察一致。最终论文主线应强调分支间
几何分歧与人体结构一致性，不应把 confidence bias 夸大为核心创新。

### 8.8 跨视角数扩展：V3/V4 新最好与动态贡献边界

将同一严格相机无关门控扩展到 V3/V4。每个视角数仍只使用训练主体
S1/S5/S6/S7/S8 每动作 50 个同步时刻作拟合及留一主体结构选择，S9/S11 全量
2,021 帧只作最终评估；输入、相机组合顺序和 GT 已逐样本对齐。统一 H76 主干采用
从 epoch 0 按 3:1:1 随机训练 V2/V3/V4 的 checkpoint 6。

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 统一 H76 checkpoint 6 | 44.734 | 24.490 | 21.037 |
| 官方 LT Volumetric | 45.505 | 22.921 | 19.876 |
| 严格相机无关关节门控 | **38.962** | **21.543** | **18.792** |
| 训练主体静态关节融合 | 待统一 checkpoint 6 对照 | 21.687 | 18.797 |

V3 相对官方 Volumetric 改善 1.378 mm，相对 H76 改善 2.947 mm；V4 相对官方
Volumetric 改善 1.084 mm，相对 H76 改善 2.245 mm。V3 seed 0/1 分别为
21.5428/21.5433 mm，V4 seed 0/1 分别为 18.7927/18.7915 mm，随机种子波动极小。

必须区分“融合总收益”和“动态特征收益”：V3 动态模型相对静态关节融合仅再改善
0.144 mm，V4 仅再改善约 0.004 mm；V3 仅使用两分支 3D disagreement 与 joint ID
为 21.638 mm，完整无相机 ID 的结构上下文为 21.543 mm。因此动态判断的边际价值
呈 V2 > V3 > V4，符合视角冗余增加后退化组合减少的预期。V4 的 1.084 mm 总收益
主要来自训练主体拟合的关节级分支校准，不能夸大成动态门控贡献。

还从官方 Volumetric 的 64^3 逐关节概率体导出方差、熵和峰值概率。新旧 V4 导出
的 GT、动作及 3D 坐标逐元素完全一致（最大差 0）；概率体统计全部有限。加入这些
官方后验不确定性后 V4 为 18.796 mm，没有优于不含该特征的 18.793 mm，也几乎
等于静态融合，因此该特征实验记为负结果，不进入主方法。

主要结果：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_gate_v234e6_v3_strict_camera_agnostic_seed0.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_gate_v234e6_v3_strict_camera_agnostic_seed1.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_gate_v234e6_v4_strict_camera_agnostic_seed0.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_gate_v234e6_v4_strict_camera_agnostic_seed1.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/adaptive_joint_gate_v234e6_v4_uncertainty_strict_seed0.json`

### 8.9 单一跨视角数模型：当前主结果

为避免 V2/V3/V4 各训练一个后处理器，将三种视角数的训练主体样本共同训练一个
严格相机无关门控。V2/V3/V4 在损失中等权，避免六个 V2 组合天然获得六倍权重；
静态初始化也只有跨视角数共享的 17 个 joint logit。模型不输入相机编号；进一步
消融发现输入视角数 one-hot 也无益，因此主版本既不依赖相机身份，也不显式依赖
集合规模。hidden=16、max-delta=2、regularization=1e-3 沿用前述只在训练主体留一
CV 选出的结构，不在 S9/S11 上重新选结构。

| 单一模型 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 统一静态关节融合 | 40.495 | 21.846 | 19.062 |
| 动态门控，seed 0 | 39.083 | **21.556** | **18.737** |
| 动态门控，seed 1 | **39.053** | 21.559 | 18.740 |
| 显式输入视角数，seed 0 | 39.032 | 21.563 | 18.781 |

不输入视角数的单一动态模型相对统一静态融合分别改善 1.412/0.290/0.325 mm；两种子
V2/V3/V4 最大差分别只有 0.030/0.003/0.003 mm。它只比三个独立门控的 V2/V3
分别差 0.091/0.013 mm，却在 V4 再改善约 0.055 mm，因此比三个独立模型更适合作为
论文主版本。显式视角数只令 V2 偶然降低 0.051 mm，同时损害 V3/V4，且迁移叙事
更弱，故只保留为消融。

主要结果与权重：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_multiview_joint_gate_noviewcount_seed0.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_multiview_joint_gate_noviewcount_seed1.json`
- `/mnt/data/cjyweights/external_multiview_h36m/unified_multiview_joint_gate_noviewcount_seed0.pth`
- `/mnt/data/cjyweights/external_multiview_h36m/unified_multiview_joint_gate_noviewcount_seed1.pth`

### 8.10 合法 checkpoint 选择与射线质量门控（2026-08-13）

复核后修正了主 checkpoint 的选择口径：不能依据 S9/S11 的最终测试误差在
checkpoint 6/15/20 中挑选。对每个 checkpoint 只使用训练主体
S1/S5/S6/S7/S8 做 leave-one-subject-out CV，三种视角数等权的均值为：

| H76 checkpoint | 训练主体 CV 均值 | V2 CV | V3 CV | V4 CV |
|---|---:|---:|---:|---:|
| epoch 6 | 13.410 | 22.889 | 9.800 | 7.541 |
| epoch 15 | 12.875 | 22.014 | 9.481 | 7.131 |
| **epoch 20** | **12.847** | **21.948** | **9.477** | **7.115** |

因此后续正式主干固定为 epoch 20。使用原 50 帧/动作训练门控时，测试结果为
38.665/21.589/18.763 mm；epoch 15 在 S9/S11 上的局部优势只作为 post-hoc
诊断，不能用于模型选择。

随后把训练主体抽样从每动作 50 帧增加到 200 帧。单纯扩大数据量令训练主体 CV
从 12.847 降到 12.647 mm，但测试仅 V2 改善 0.051 mm，V3/V4 略退，说明收益
接近饱和。基于 AdaFuse 开源实现中“跨视角几何一致性驱动质量预测”的机制，额外
从当前预测 2D 与相机标定导出以下逐关节、相机身份无关特征：

- 无向射线的最小/平均/最大锐角；
- 射线对最短距离的均值与最大值；
- 置信度加权三角化法矩阵的三个归一化特征值。

特征导出器明确不读取 3D GT，并保存 `uses_ground_truth=false`；manifest 的图像、
record index、camera ID 均逐组核验。质量头采用 AdaFuse 官方代码对应的两层
`512→256→1` MLP，但输出控制 H76 与官方 Volumetric 的逐关节凸组合，而不是
照搬其热图融合。结构容量只按训练主体 CV 选择。

| 统一门控（200 帧/动作，seed 0） | 训练主体 CV | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| 原特征，16 隐层 | 12.647 | 38.615 | 21.602 | 18.777 |
| 射线特征，16 隐层 | 12.509 | 38.481 | 21.456 | 18.675 |
| 原特征，容量匹配 `512→256→1` | 12.214 | 37.355 | 21.311 | 18.628 |
| **射线特征，`512→256→1`** | **12.110** | **37.187** | **21.219** | **18.556** |

这项容量匹配对照说明，相对旧 16 维门控的总提升同时来自质量头容量与显式射线
几何，不能全部归因于几何模块。严格同容量下，射线特征相对无射线特征改善
V2/V3/V4 `0.168/0.092/0.072` mm，训练主体 CV 改善 0.104 mm。额外 confidence
min/mean/max 对训练主体 CV 没有收益，因此正式版本只保留纯射线几何。

正式结构三个随机种子测试均值（总体均值只作稳定性报告，不据此挑 seed）：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 射线质量门控，3-seed mean | **37.233** | **21.226** | **18.565** |
| 3-seed std | 0.032 | 0.009 | 0.006 |

主要代码与结果：

- `OpenRUMPL_baseline_audit/export_h36m_ray_geometry_features_20260813.py`
- `OpenRUMPL_baseline_audit/train_unified_multiview_joint_gate_20260813.py`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_gate_epoch20_train200fpa_rayonly_512x256_cv_seed{0,1,2}.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_gate_epoch20_train200fpa_rayonly_512x256_cv_seed{0,1,2}.pth`

当前可守住的描述是：**保留 RUMPL/H76 射线先验与坐标分支、保留官方 Volumetric
图像分支，用由跨视角射线条件和观测不一致性驱动的逐关节质量网络自适应融合。**
它有 AdaFuse 的开源机制依据，但融合位置和两个互补候选来自本项目，不声称首次
使用极线/射线一致性或首次学习视角权重。

剩余上限诊断：若允许测试 GT 为每个关节选择 H76--Vol 连线段上的最优凸组合，
V2/V3/V4 可分别达到 30.044/18.395/16.399 mm；仅在两分支中逐关节 hard oracle
选择也可达到 32.075/18.992/16.837 mm。约 35%--37% 的关节最优点位于两候选
内部，说明凸融合本身合理，但当前质量预测与 oracle 仍有约 7.14/2.82/2.16 mm
差距。下一阶段应提高逐关节效用预测或产生第三候选，而不是继续扩大同一门控容量。

### 8.11 第三候选：官方 Algebraic LT（2026-08-13）

先做零训练上限诊断。官方 Algebraic 本身测试为 53.592/24.339/19.921 mm，弱于
当前方法，但它与 H76/Vol 的误差并不完全重合。加入 Algebraic 后，逐关节 hard
oracle 从两候选的 32.075/18.992/16.837 降到三候选的
30.720/18.180/16.169 mm，证明第三候选有真实互补性。

随后在相同训练主体 200 帧/动作、相同纯射线特征、相同 `512→256` 质量头下，
将两分支 sigmoid 凸融合改为 H76/Vol/Algebraic 三分支 softmax 凸融合。仍然一个
模型覆盖 V2/V3/V4、三种视角数损失等权、不输入相机 ID/组合 ID/视角数，结构选择
只看训练主体 leave-one-subject-out CV。

| 方法（seed 0） | 训练主体 CV | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| 三候选，仅静态关节权重 | 13.299 | 40.148 | 21.899 | 19.109 |
| 三候选动态门控，无射线特征 | 12.001 | 36.859 | 21.185 | 18.511 |
| H76 + Vol 两候选射线门控 | 12.110 | 37.187 | 21.219 | 18.556 |
| **+ 官方 Algebraic 第三候选** | **11.869** | **36.561** | **21.050** | **18.424** |

三候选三种子正式测试均值为 **36.519/21.045/18.410 mm**，标准差为
0.031/0.013/0.013 mm；训练主体 CV 均值 `11.877±0.005` mm。收益在 CV 和三种
测试 seed 上同方向，因此保留为当前主版本。必须准确描述：Algebraic 不是新的
2D 输入，也不是单独达到最好精度；它是与 RUMPL 射线坐标候选、Volumetric 图像
候选互补的第三个几何候选，由统一质量网络按关节自适应选择/融合。
相对三候选无射线同容量控制，显式射线条件与一致性特征在 V2/V3/V4 再改善
0.298/0.134/0.087 mm；相对静态三候选，动态逐样本质量判断改善
3.586/0.849/0.685 mm。这两项消融分别隔离了射线特征与动态门控的贡献。

主要代码与结果：

- `OpenRUMPL_baseline_audit/train_unified_three_branch_gate_20260813.py`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_official_alg_train200fpa_predictions.npz`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_official_alg_full2021_predictions.npz`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_ray_gate_512x256_cv_seed{0,1,2}.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_ray_gate_512x256_cv_seed{0,1,2}.pth`

### 8.12 候选级射线解释残差（2026-08-13）

在相机集合级射线条件特征之外，为 H76/Vol/Algebraic 每个 3D 候选计算其到当前
预测 2D 世界射线的置信度加权距离、最大距离、加权角残差和最大角残差。距离使用
`log1p` 防止退化 V2 样本的极值支配标准化。全部特征只依赖候选 3D、预测 2D、
置信度和相机参数，不读取 GT。该设计与 AdaFuse 的 Sampson 一致性质量判断、
DeProPose 的投影误差动态融合机制一致，但这里在三种异构 3D 候选层面使用。

| 三候选模型（seed 0） | 训练主体 CV | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| 集合级射线质量 | 11.869 | 36.561 | 21.050 | 18.424 |
| + 仅候选距离残差 | 11.858 | 36.485 | 20.968 | 18.312 |
| + 仅候选角残差 | 11.844 | 36.532 | 20.946 | 18.286 |
| **+ 候选级射线解释残差** | **11.837** | **36.415** | **20.936** | **18.272** |

三种子测试均值为 **36.468/20.925/18.259 mm**，标准差为
0.076/0.007/0.010 mm；训练主体 CV `11.837±0.001` mm。V2 的 seed2 波动略高，
但 CV 与 V3/V4 均高度稳定，保留为当前主版本，不依据测试挑单一 seed。
距离与角度两组特征单独均有效，联合结果在训练主体 CV 和三个视角数上最好，故不
删除任一组；它们分别描述点到观测线的空间偏离和方向偏离，并非完全冗余。

主要代码与结果：

- `OpenRUMPL_baseline_audit/export_three_candidate_ray_residuals_20260813.py`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_candidate_residual_cv_seed{0,1,2}.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_candidate_residual_cv_seed{0,1,2}.pth`

固定结构/特征/损失后的训练充分性消融只按训练主体 CV 选择：

| epochs（seed 0） | 训练主体 CV | V2 | V3 | V4 |
|---:|---:|---:|---:|---:|
| 60 | 11.948 | 36.795 | 21.012 | 18.302 |
| 120 | 11.837 | 36.415 | 20.936 | 18.272 |
| **240** | **11.780** | **36.259** | **20.905** | **18.252** |

因此正式训练长度由 120 改为 240 epochs。240 轮三种子测试均值为
**36.277/20.898/18.248 mm**，标准差 0.026/0.005/0.008 mm；训练主体 CV
`11.783±0.003` mm。正式结果与权重：

- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_candidate_residual_e240_cv_seed{0,1,2}.json`
- `/mnt/data/cjyoutput/external_fair_comparison_20260813/unified_three_branch_candidate_residual_e240_cv_seed{0,1,2}.pth`

第四候选筛查只做 GT oracle，不进入训练：同输入的 confidence-ray/robust IRLS 自身
分别约为 V2 `47.9/47.8`、V3 `24.6/23.7`、V4 `20.6/19.6` mm。将 robust IRLS
加入当前三候选 hard oracle 仅额外改善 0.154/0.243/0.220 mm，明显小于 Algebraic
第三候选带来的 1.355/0.812/0.668 mm。考虑模型复杂度与论文叙事，正式候选数固定
为三，IRLS 只保留为被 oracle 排除的零训练控制。

训练目标也按既有论文/旧实验补了两个单变量控制。GHT 官方式 expected candidate
risk `1.0` + fused estimate `0.05` 令训练主体 CV 从 11.837 退化到 12.341，测试
为 37.346/21.111/18.338 mm；`tau=5 mm` 的 soft-oracle candidate target 令 CV
退化到 11.932，测试为 36.589/20.962/18.251 mm。后者仅 V4 测试偶然改善
0.021 mm，但未通过预先规定的训练主体 CV，因此两者均不保留、不补种子。正式
训练仍直接最小化三分支最终凸融合的 balanced V2/V3/V4 MPJPE。

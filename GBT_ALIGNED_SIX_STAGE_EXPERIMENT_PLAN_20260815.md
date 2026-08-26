# GBT 对齐的六阶段论文实验计划（2026-08-15 历史版）

> **2026-08-22 覆盖声明：**本文件保留旧 HRNet/E2-C2 路线和历史实验，不再作为
> 当前 baseline 的执行源。当前已升级为七阶段路线，统一模型为
> `GQ-RUMPL → GQ-RUMPL-E2 → GQ-RUMPL-E2-T`，HRNet/ResNet 只替换冻结 2D
> 前端。最新技术协议、状态和全部 GBT 表格见
> `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/JOINT_QUERY_MATCHED_FRONTENDS_PLAN_20260822.md`；
> 可直接交给 Codex 的母稿见
> `/home/lixiaob/cjy/PAPER_MASTER_STORY_METHOD_RESULTS_CODE_20260815.md` 文首 A/B 节。
> 下文数值只可作为历史失败/消融资料，不得与当前 matched pipeline 拼接。

## 0. 总体决定

论文实验故事按 Geometry-Biased Transformer（GBT）的组织方式展开，但由于 GBT
没有公开代码且若干前端细节未披露，全文使用“GBT-aligned / 按公开细节对齐”，
不写成严格复现。

最终必须保留两条完全独立的输入线：

1. **HRNet line**：冻结 COCO HRNet-W32，输入 2D 坐标、置信度和相机参数；
2. **ResNet-152 line**：冻结 Learnable Triangulation 官方提供、在 H36M 微调的
   ResNet-152，仍只向 3D 网络输入 2D 坐标、置信度和相机参数。

两条线使用同一套 3D 模型定义、数据划分、视角组合和评估代码，但分别训练，不能
把两种检测器的结果或特征混合。HRNet/ResNet 的比较是输入质量鲁棒性实验，不是
集成模型。

GBT 的 H36M 主表使用 9 帧模型。因此，我们的正式外部主表也必须报告 **T=9**；
同时保留 **T=1**，用于证明空间模块本身的贡献。不能用 T=1 结果声称公平超过
GBT T=9，也不能隐藏时序贡献。

## 1. GBT 公开协议与必须超过的目标

### 1.1 Human3.6M clean 主表

- 训练：S1/S5/S6/S7/S8；测试：S9/S11；
- 绝对世界坐标 All-17 MPJPE，不做 root/刚体/Procrustes 对齐；
- V2/V3/V4 枚举 4 台相机的全部 `6/4/1` 个组合，再做 action-equal 平均；
- GBT：训练随机两视角，输入/监督 9 帧，推理输入 9 帧、只取最新一帧；
- Adam，batch 256，300k iterations，初始 LR `1e-4`，warmup + 平滑衰减；
- scene centering、synthetic views、20% token dropout。

| 输入 | GBT V2 | GBT V3 | GBT V4 | 我们的成功门槛 |
|---|---:|---:|---:|---|
| HRNet-W32 COCO + YOLOX | 36.8 | 30.4 | 26.0 | 三列都严格低于 GBT，至少 2 seeds |
| H36M-finetuned ResNet-152 | 29.9 | 24.4 | 22.7 | 三列都严格低于 GBT，至少 2 seeds |

### 1.2 H36M-Occl

GBT 在原 H36M 测试图像上，以 `0.1` 概率在 2D 关节附近放置白色方块遮挡；模型
仍只在 clean H36M 上训练。论文未公开方块边长、随机种子和 YOLOX 细节，所以我们
必须固定生成脚本、mask 尺寸和 seed，并称为 GBT-aligned H36M-Occl。

| 输入 | GBT V2 | GBT V3 | GBT V4 | 我们的成功门槛 |
|---|---:|---:|---:|---|
| ResNet-152 | 39.1 | 33.4 | 31.3 | 三列均低于对应值 |
| HRNet-W32 | 42.3 | 34.5 | 31.6 | 三列均低于对应值 |

### 1.3 CMU 与跨数据集

- GBT 的 CMU 单人协议沿用 Iskakov 等人的单人 `pose` 序列，标准四个测试 HD
  相机为 `2,13,10,19`，其余 27 台用于训练；CMU 四视角 HRNet 结果为 `17.2 mm`。
- CMU→H36M 使用 CMU 训练模型直接测试 H36M，只评价两数据集共有的肩、肘、腕、
  膝、踝关节，GBT 平均为 `38.9 mm`。
- GBT 没有报告 CMU V2/V5/V6/V8。因此扩展曲线不能写成“超过 GBT 每一列”；
  只能在严格四视角点超过 `17.2 mm`，其余视角与公开可比方法和自身 baseline 比较。

## 2. 当前 HRNet 单帧结果与瓶颈

当前结果均为严格 S9/S11、action-equal、All-17 absolute MPJPE：

| 单一模型 | V2 | V3 | V4 | 相对 GBT HRNet T=9 |
|---|---:|---:|---:|---|
| C2 K2-heavy RUMPL/H76 | 38.686 | 30.943 | 28.629 | +1.886 / +0.543 / +2.629 |
| **E2-C2 soft-cal（当前统一 T=1 baseline）** | **38.700** | **29.486** | **27.274** | +1.900 / **-0.914** / +1.274 |
| E2-C1 soft-cal（V2 优先） | 37.009 | 30.108 | 28.528 | +0.209 / **-0.292** / +2.528 |
| B2→mixed，20E `5e-5` | **36.885** | 31.451 | 30.277 | +0.085 / +1.051 / +4.277 |
| B2→mixed，60E `1e-5` | 37.140 | 31.775 | 30.603 | +0.340 / +1.375 / +4.603 |

结论：

1. 当前最合理的统一单帧 baseline 仍是 **E2-C2 soft-cal**，因为它用同一个模型
   同时给出最好的 V3/V4，并且 V2 小于 40 mm；
2. 高 LR/长训证明，单靠 K2-heavy→mixed 的训练日程可以提高 V2，却会破坏
   V3/V4，不能升级为论文主模型；
3. E2-C2 候选 oracle 为约 `36.166/21.702/16.994 mm`，候选池仍有足够空间。
   V2 的剩余差距主要是候选质量/评分，V3/V4 的差距适合由候选效用和时序共同解决；
4. 当前 HRNet 输入虽按 GBT 公开描述使用 YOLOX + COCO HRNet-W32，但 GBT 未披露
   YOLOX 型号、阈值、bbox padding 等，输入线只能称 GBT-aligned。

## 3. 第一阶段：先冻结 HRNet baseline

### 3.0 H8 时序筛选（已完成，作为失败/边界证据）

在完成 GBT-aligned HRNet 前端的全量 S9/S11 验证导出后，先做了一个严格受控的
T=9 筛选。训练和评估均使用同一套 `gbt_yolox_x_score001_fallback_legswap`
坐标级输入、S1/S5/S6 训练缓存、S9/S11 验证缓存、随机 K=2 和 H76
`tri-anchor + anchor-centered Plücker` 主干；训练使用 30k optimizer steps、BF16，
仅改变时序分支。

| H8 分支 | T=1 V2/V3/V4 | T=9 V2/V3/V4 | 结论 |
|---|---:|---:|---|
| Pre-VFT temporal，冻结 RUMPL | 41.293 / 34.505 / 29.836 | 41.397 / 34.459 / 29.978 | T=9 与 T=1 基本持平，无 clean 增益 |
| Pre-VFT temporal，解冻 VFT/PFT/head | 55.887 / 57.888 / 46.935 | 56.071 / 58.040 / 47.124 | 破坏原有几何融合，明确失败 |

这些结果不是主表的最终 baseline（它们从 H76 checkpoint 继续训练，且时序模型训练
窗口为 T=9），但足以排除“直接在射线 VFT 前叠一个 Transformer”以及“低学习率解冻
RUMPL 融合器”两条重复路线。H8 的完整 JSON 位于
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_pre_vft_temporal/eval/`。
因此后续时序实验必须保留绝对几何路径，并换成有明确来源的姿态空间 residual 或
候选级门控，不能继续加深同一 Pre-VFT 模块。

### 3.1 H9/H10 时序筛选（已完成）

H9 使用 MixSTE（CVPR 2022）公开 STB→TTB 轴分解，但只把它放在 RUMPL 完整输出
之后的 root-relative articulation residual；pelvis correction 强制为零，RUMPL
VFT/PFT/head 及 absolute anchor 冻结。输入仍完全是 GBT-aligned HRNet 坐标、置信度和
相机射线，不使用图像特征。H9 的 12k-step 筛选已按同一 T=1/T=9、V2/V3/V4
全组合协议评估；H10 同步作为 VFT 后 TTB 控制线完成 T=9 评估。若 clean 仍不优于
H0，则只保留其遮挡鲁棒性候选价值，不再继续扩大时序搜索。

H9/H10 已完成正式 T=9 评估。H9（pose residual + MixSTE objective）为
`49.503/52.715/39.793 mm`；H10（VFT 后 per-joint TTB residual + RUMPL MPJPE）为
`46.041/41.882/33.940 mm`。两者都明显差于 H8 frozen `41.397/34.459/29.978 mm`，
也没有接近 E2-C2 的单帧折中线。这个结果把“时序必须放在候选级”从经验判断提升为
正式负对照：当前 H36M clean 的绝对射线误差不适合无条件 temporal residual。后续
只有在遮挡/缺失测试中做 uncertainty-gated temporal fallback 才有继续价值；clean
主线先回到空间候选/效用模块。

### 3.1 立即执行矩阵

| 编号 | 唯一改动 | 论文/代码依据 | 目的 | 继续门槛 |
|---|---|---|---|---|
| H0 | 冻结 E2-C2 soft-cal | 当前两 seed 正式结果 | T=1 统一参照 | 固定不再调测试集温度 |
| H1 | 在新完成的 HIGH-LR 单 checkpoint 上重建 22 候选并训练 E2，两 seeds | Generalizable Human Pose Triangulation 官方代码 + 当前 E2 | 检查 `36.885` 的 V2 与 E2 的 V3/V4 能否在同一模型中共存 | V2≤36.8 且 V3/V4 至少优于其原始 31.451/30.277 |
| H2 | 扩展随机三角化/RANSAC/IRLS 候选，只训练现有 utility，不改主干 | GHT CVPR 2022 官方代码；Learnable Triangulation ICCV 2019 官方 RANSAC/SVD | 提高 V2 candidate oracle，不靠更深 Transformer | 训练 holdout oracle 至少降 0.5 mm，正式模型 V2 至少降 0.3 mm |
| H3 | 轴分解 view block，T=1，零初始化 residual，无 camera-ID embedding | MTF-Transformer 官方 MFT；SVTformer AAAI 2025 官方 view block | 替换当前较弱的固定 VFT，而非在末端堆完整模型 | S8 holdout V2/V3/V4 均不退化，至少两列改善 ≥0.2 mm |
| H4 | 在 H1/H2/H3 中最强者上加入 T=9 candidate/ray temporal block | MTF-Transformer 官方 TFT；SVTformer temporal block；H8 已排除直接 Pre-VFT 叠加 | 公平对齐 GBT T=9，重点利用遮挡和短时运动 | clean 三列均优于对应 T=1；H36M-Occl V4 改善 ≥1 mm |
| H5 | scene centering + yaw + synthetic cameras，单变量加入 | GBT 公开方法；MTF 官方 DataAug；OpenMPL/MHP 相机合成代码 | 提升相机泛化和 CMU→H36M | clean 不明显退化，CMU→H36M 至少改善 1 mm |

H1 是最低成本、最高优先级；H2/H3 可以并行做 holdout 筛选。H4 不是“最后再说”的
附加项，而是外部主表公平性所必需，但必须建立在已经通过 T=1 门槛的空间模型上。

### 3.2 推荐的最终主干形式

```text
冻结 2D detector（HRNet 或 ResNet）
       │ 2D coordinates + confidence + calibrated rays
       ▼
anchor-centered Plücker ray tokens
       ▼
axis-factorized spatial/view encoder（MTF/SVT-style，无 camera ID）
       ▼
H76/RUMPL absolute 3D candidates + robust stochastic candidates
       ▼
E2 counterfactual joint-candidate utility
       ▼
T=9 candidate/ray temporal encoder（零初始化 residual，输出最新帧）
       ▼
soft candidate fusion → absolute world-coordinate 3D pose
```

这不是完整堆叠 RUMPL、GHT、MTF 三个模型：保留的是 RUMPL 的射线绝对定位和已验证
H76 候选；用 GHT 的随机假设生成替换单一求解器；用 MTF/SVT 的 view/temporal
attention 结构替换原 VFT/末端平滑；E2 负责统一的候选效用读出。

### 3.3 时序为什么现在应该加入

GBT 在 4 视角 HRNet 上 T=1→T=9 为 `29.4→26.0`（-3.4 mm），H36M-Occl 为
`41.5→31.6`（-9.9 mm）。因此遮挡表若没有时序，很难公平超过 GBT。

但本项目已经完成并失败的时序路线不能重复：

- 最终 3D pose residual 只改善约 0.4–0.5 mm；
- MixSTE 放在融合后的 pose 上出现跨主体过拟合；
- 冻结 RUMPL 的全局 JVT、简单 GBT bias、加深 attention 均没有稳定收益；
- T-CVU 只有约 0.1 mm。

新时序必须放在候选/射线仍保留视角来源的位置，并采用 identity-preserving
零初始化；不再直接平滑最终 3D，不再重复 MixSTE post-refiner。首版用 T=9 训练、
同时监督 9 帧、推理取最新帧；另报 T=1/T=2/T=3/T=6/T=9，直接形成 GBT Table VII
式消融。

## 4. 可借用的权威论文与代码优先级

| 优先级 | 方法 | 可借模块 | 适配性与限制 |
|---|---|---|---|
| A | Generalizable Human Pose Triangulation，CVPR 2022 Oral，官方 `kristijanbartol/general-3d-humans` | stochastic pose hypotheses、Gumbel/soft hypothesis scoring | 直接输入 2D 坐标和相机，适合 H2；现有 E2 已部分吸收，应继续用其官方生成/评分细节而不是重写新概念 |
| A | MTF-Transformer，TPAMI，官方 `lelexx/MTF-Transformer` | variable-view Relative Attention（MFT）、variable-length TFT、synthetic-view DataAug | 与坐标输入最接近；原论文评估为 root-relative，不能照搬其数值，只借模块到 absolute RUMPL/E2 |
| A | Learnable Triangulation，ICCV 2019 Oral，官方 `karfly/learnable-triangulation-pytorch` | H36M-finetuned ResNet-152、confidence、RANSAC、可微 SVD | 是第二阶段 ResNet 输入的首选官方来源，也可用于 H2 的 robust candidates |
| B | SVTformer，AAAI 2025，官方 `Rowenazhang/SVTformer` | spatial-view-temporal 轴分解顺序 | 代码固定 4 views、使用 view positional embedding 且做 root-relative；只借 axis block，必须改成 masked variable-K 且删除 camera ID |
| B | DeProPose，公开代码 `WUJINHUAN/DeProPose` | relative projection error view weighting、missing/noise generator | 放在第三阶段遮挡/缺失消融；不先替换 clean 主干，且需单独审计其代码与论文口径 |
| C | MixSTE，CVPR 2022，官方 `JinluZhang1126/MixSTE` | joint-wise temporal block | 单视角 root-relative、长窗口；本项目 post-pose 融合已失败，只能作为 H4 的局部实现参考，不再整模型接入 |
| C | MVGFormer，CVPR 2024，官方 `XunshanMan/MVGFormer` | 3D query→2D projection→image feature refinement | 会读取图像特征，属于 feature-level 协议，不能放入与 GBT 坐标输入相同的主表；可作为另表扩展 |
| 暂缓 | UPose3D，ECCV 2024 | 2D uncertainty、cross-view/temporal compiler、MLE | 方法高度契合但目前未确认完整官方代码；只作为方法依据，不作为第一版实现来源 |

## 5. 六阶段总计划表

| 阶段 | 数据/输入 | 训练与测试 | 主要输出 | 完成标准 |
|---|---|---|---|---|
| 1. HRNet baseline | H36M，YOLOX+COCO HRNet-W32，坐标+置信度 | S1/5/6/7/8→S9/11；T=1 与 T=9；V2/3/4 全组合 | clean 主表、时间帧消融、每相机组合 | T=9 `<36.8/30.4/26.0`；两 seeds |
| 2. ResNet baseline | H36M，官方 H36M-finetuned ResNet-152 | 完全复用阶段1的 3D 模型和协议，重新导出/训练 | clean ResNet 主表 | T=9 `<29.9/24.4/22.7`；两 seeds |
| 3. H36M-Occl | 阶段1/2两种 detector，各自 clean-trained checkpoint | 只改测试图像；固定 p=0.1、mask 尺寸、seed；V2/3/4 | 两种输入遮挡表、T=1/T=9、可见/遮挡关节拆分 | HRNet `<42.3/34.5/31.6`；ResNet `<39.1/33.4/31.3` |
| 4. CMU intra | 单人 CMU pose 序列，HRNet 为主 | 严格 GBT 四相机点 + 独立 8-camera extension；V2/4/5/6/8 | 视角数曲线、组合方差、NVR | GBT 四相机口径 `<17.2`；扩展曲线随视角单调下降 |
| 5. CMU→H36M | 阶段4 CMU checkpoint 直接测 H36M | H36M S9/S11；共有的肩肘腕膝踝；absolute、无对齐 | action-wise Table IV 式表格 | 平均 `<38.9`；不在 H36M 微调 |
| 6. 消融 | 阶段1–5同步产出 | 每个模块 checkpoint 统一评 H36M、Occl、CMU、CMU→H36M | GBT Table VI 式跨域矩阵 | full 在四行都最好或有可解释 trade-off；至少两 seeds 主结果 |

## 6. CMU 协议必须分成两张表

当前 `/mnt/data/cjydata/cmu_singleperson_real20` 只有约 9.7 GB，且不是完整的 GBT
31-camera 数据。若目标是严格对齐 GBT，必须补齐单人训练序列的 31 个 HD 相机。

### 6.1 CMU-GBT 四视角表（外部可比）

- 单人训练序列：`171026_pose1/2/3`、`171204_pose1/2/3/4`；
- 测试：`171204_pose5/6`；
- 测试相机：`2,13,10,19`；训练相机：其余 27；
- 报告四视角 absolute MPJPE，并与 GBT HRNet `17.2` 对比。

### 6.2 CMU-8 扩展曲线（本文扩展，不冒充 GBT）

- 固定 8 个空间覆盖均匀的 held-out test cameras；训练只使用其余 23 台；
- V2/V4/V5/V6/V8 在这 8 台中枚举组合。若全部枚举计算过大，预注册固定数量的
  均匀子集并固定 seed；
- 同时报告组合均值、标准差、最好/最差组合和 Negative View Rate；
- 不能把现有 real20 的同相机训练/测试结果与相机不相交协议混在同一列。

## 7. GBT Table VI 式消融设计

建议列定义如下，所有列都是一个真实 checkpoint，不拼接结果：

| 列 | Anchor-centered Plücker | Centering + synthetic views | Robust hypotheses | Geo/conf utility | Axis view block | T=9 temporal |
|---|---|---|---|---|---|---|
| A0 | ✓ | — | — | — | — | — |
| A1 | ✓ | ✓ | — | — | — | — |
| A2 | ✓ | ✓ | ✓ | — | — | — |
| A3 | ✓ | ✓ | ✓ | ✓ | — | — |
| A4 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

行定义复刻 GBT：

1. H36M→H36M clean，HRNet，V4；
2. H36M→H36M-Occl，HRNet，V4；
3. CMU→CMU，HRNet，严格四相机；
4. CMU→H36M，matched joints，HRNet。

同时补充三张小消融，避免主矩阵掩盖关键因素：

- T=1/2/3/6/9 时间帧表；
- V2/V3/V4 clean 主表；
- confidence-only、geometry-only、两者共同的 attention/utility bias 表。

## 8. 两张 GPU 的执行顺序

### 8.1 baseline 决策期

- GPU0：H1 seed0（HIGH-LR candidate + E2）；
- GPU1：H1 seed1；
- H1 完成后：GPU0 做 H2 robust candidate oracle/screen，GPU1 做 H3 T=1 axis-view
  block 的 S8 holdout；
- 只把通过门槛的一个空间模型升级为 T=9，避免同时长训多个已经退化的结构。

### 8.2 T=9 与输入双线

- GPU0/1 先跑 HRNet T=9 两 seeds；
- HRNet 达标后，冻结所有 3D 超参数，替换为 ResNet 输入，两卡跑两个 seeds；
- 遮挡评估不重训，直接并行跑两种 detector；
- CMU 训练期间同时生成消融所需的各 checkpoint 评估，不在最后重复训练。

## 9. 数据与实现阻塞项

1. 本地尚未找到 Learnable Triangulation 官方
   `pose_resnet_4.5_pixels_human36m.pth`；第二阶段开始前必须完整下载并校验哈希；
2. ResNet 官方输入使用 384×384、GT bbox、预先去畸变图像；这些都要按官方代码
   导出，不能用当前 HRNet bbox 近似；
3. GBT 未公开 H36M-Occl 的方块尺寸和随机 seed。我们的生成设置要预注册，并同时
   报告 detector 2D error，避免遮挡强度不等价；
4. 当前 H36M 图像/PKL 为原始 50 Hz 每 5 帧取一帧，T=9 约覆盖 0.8 s。必须把
   时间采样间隔写入论文；必要时再补完整 50 Hz 对照，不能默认与 GBT 完全相同；
5. CMU real20 不足以完成 GBT 27-train/4-test 和 23-train/8-test 两个相机不相交
   协议；下载计划需扩展到完整 31 HD cameras。

## 10. 当前 baseline 的命名与冻结结论

- **RUMPL baseline**：严格原始/公开代码近似复现结果，作为外部起点；
- **Ours-T1 baseline**：`E2-C2 soft-cal = 38.700/29.486/27.274 mm`；
- **V2 training control**：`36.885/31.451/30.277 mm`，只做训练协议消融；
- **Ours-T9**：尚未产生，只有同一统一模型在 HRNet clean 三列均超过 GBT 后，
  才能冻结为论文主 baseline；
- 不允许把 HIGH-LR 的 V2 与 E2-C2 的 V3/V4 拼成一个模型。

所有实验统一输出到：

`/mnt/data/cjyoutput/gbt_six_stage_20260815/`

建议子目录：`01_hrnet_clean/`、`02_resnet_clean/`、`03_h36m_occl/`、
`04_cmu_intra/`、`05_cmu_to_h36m/`、`06_ablation/`，每个目录保存 config、git diff、
checkpoint、逐组合 JSON、action-equal 主表和 seed 汇总。

## 11. H1 实验记录（2026-08-15）

### 11.1 目的与执行口径

H1 用已经完成的同一个高学习率 RUMPL checkpoint 生成全部 11 个视角组合的
H76 候选，再追加同组合的置信度加权三角化候选，形成 22-candidate cache；之后
使用与 E2-C2 完全相同的 utility scorer、损失、温度、holdout 和训练轮数，分别
训练 seed 0/1。这样验证的是“一个模型能否同时保留 V2 优势并改善 V3/V4”，而不
是把不同 checkpoint 的预测拼接成伪结果。

- checkpoint：`CARD2_HIGH_LR5E5_B2_MIXED_T1_20E_seed0_20260815_2026-08-15_18-45-38/model_best.pth.tar`
- 输入：`gbt_yolox_x_score001_fallback_legswap`，HRNet 坐标/置信度，四视角组合
- 输出目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_card2_high_input_protocol_v1/`
  和 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_card2_high_training_protocol_v1/`
- 启动脚本：`OpenRUMPL_baseline_audit/launch_e2_card2_high_cache_and_train_20260815.sh`
- 评估：H36M S9/S11，action-equal All-17 absolute MPJPE，T=1，seed 0/1

### 11.2 结果（最终一次评估，单位 mm）

| seed | V2 baseline | V2 soft | V3 baseline | V3 soft | V4 baseline | V4 soft |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 36.885 | 36.921 | 31.451 | 30.006 | 30.277 | 28.470 |
| 1 | 36.885 | 36.916 | 31.451 | 30.051 | 30.277 | 28.603 |
| mean | 36.885 | 36.918 | 31.451 | 30.029 | 30.277 | 28.536 |

同一缓存上的 oracle soft 下界为 V2/V3/V4 = **34.203/21.985/17.604 mm**；这不是
可部署结果，只用于判断候选池仍有多少可学习空间。

### 11.3 结论与下一步

H1 证实 E2 utility 可以在不破坏高学习率 checkpoint 的 V2 结果的情况下，把
V3/V4 从单模型的 31.451/30.277 降到 30.029/28.536（两 seed 均如此）。但它
仍未达到 E2-C2 soft-cal 的 29.486/27.274，因此“V2 专家 + E2 选择器”并未在
当前 22 候选池中完全统一；不能把 H1 宣称为最终 HRNet baseline。下一实验优先
做候选池/求解器的可解释对照（H2），再决定是否训练 T=9；时序暂不与 H1 同时改动。

## 12. H2 实验记录（2026-08-16）

### 12.1 候选池诊断

在 H1 的同一 validation cache 上零训练比较确定性射线求解器。结果是 action-equal
All-17 oracle（单位 mm）：

| 候选池 | V3 | V4 |
|---|---:|---:|
| H76 existing | 24.4635 | 20.0086 |
| + uniform all-view | 23.0891 | 19.0329 |
| + confidence all-view | 23.0633 | 19.0160 |
| + IRLS all-view | 22.9961 | 18.9781 |
| + pairwise-IRLS | **22.2094** | **18.0437** |

诊断脚本：`diagnose_h76_candidate_pool_20260812.py`；结果：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/h2_candidate_oracle_high_v1/`。
oracle 只用于判断候选池上限，不能当作模型精度。

### 12.2 33-candidate 训练

根据诊断新增每个 2/3/4 视角组合的 IRLS 候选，候选顺序为 H76(11) + confidence(11)
+ IRLS(11)，其他输入、RUMPL checkpoint、loss、10E+5E 和两 seed 不变。

| seed | V2 soft T=1.8 | V3 soft T=1.8 | V4 soft T=1.8 |
|---:|---:|---:|---:|
| 0 | 37.142 | 30.030 | 28.425 |
| 1 | 37.118 | 30.086 | 28.505 |
| mean | 37.130 | 30.058 | 28.465 |

复用预注册 V2 温度 `0.4` 后为 **36.891/30.058/28.465 mm**，仍比 H1 的
36.886/30.029/28.537 差约 `+0.005/+0.029/-0.072 mm`，也没有超过当前统一
E2-C2 soft-cal `38.700/29.486/27.274 mm`。因此 IRLS 候选虽然显著降低 oracle，
但当前 utility loss 没有把新增候选转化为同等幅度的测试提升；H2 暂停继续扩展
候选数量，下一步应优先检查训练目标/候选特征的可辨识性，而不是再盲目添加求解器。

## 13. H3 实验记录（2026-08-16，运行中）

H2 表明候选扩展不是当前最短路径，因此 H3 回到 RUMPL 主干，只做训练协议的
单变量 A/B：两组都从已完成的 300k B1 checkpoint 初始化，使用同一 HRNet 坐标/置信度
输入、同一 H76（tri-anchor + centered Plücker）表示、20E 和 `5e-5` 初始微调学习率；
GPU0 为 GBT 风格 K2-heavy `8:1:1`，GPU1 为当前 balanced `3:1:1`。两组均为
真实单模型训练，结果不会互相拼接。

- 脚本：`OpenRUMPL_baseline_audit/launch_b1_highlr_sampling_ab_20260816.sh`
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/b1_highlr_sampling_ab/`
- 当前状态：两组均已载入同一 B1 checkpoint，训练第 0 轮；自动完成 V2/V3/V4
  严格评估后再决定是否进入时序阶段。

预期判据：若 `8:1:1` 降低 V2 但明显牺牲 V3/V4，说明采样比例仍是 cardinality
trade-off；若 balanced 在高 LR 下同时改善三列，则将其冻结为 T=1 spatial baseline，
再做 T=9 单变量实验。无论哪种结果，都不把不同采样模型的列拼接成一个方法。

### 13.1 H3 最终结果

严格 H36M S9/S11、action-equal All-17 absolute MPJPE（mm）：

| 训练线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| B1 checkpoint + `8:1:1`, LR `5e-5` | **39.544** | **30.802** | **28.372** |
| B1 checkpoint + `3:1:1`, LR `5e-5` | 41.884 | 31.389 | 28.790 |
| C2 原统一模型（LR `1e-5`, `8:1:1`） | 38.686 | 30.943 | 28.629 |
| E2-C2 soft-cal（当前多视角 baseline） | 38.700 | **29.486** | **27.274** |

`8:1:1` 相对同预算 balanced 线改善 `2.340/0.587/0.418 mm`，说明 K2-heavy
采样在当前输入上确实有稳定作用；相对 C2，它的 V3/V4 还改善 `0.141/0.257 mm`，
但 V2 仍高 `0.858 mm`，且没有超过 E2-C2 soft-cal 的多视角结果。因此 H3 的
K2-heavy checkpoint 是更值得继续的 spatial 生成器，但暂不冻结为最终论文 baseline；
下一步可在该单模型上重新做 E2 utility，检验能否同时保留 V2 和多视角收益。

## 14. H4/H5 实验记录（2026-08-16，运行中）

以 H3 K2-heavy checkpoint 为唯一候选生成器，重新导出同一 22-candidate cache，
并做两条 utility loss A/B：

| 线 | 变化 | seed |
|---|---|---:|
| H4 standard | 原 E2 direct-ranking + GHT expected-risk | 0/1 |
| H5 hinge | H4 + identity-preserving hinge `0.25`，V2 权重 `4` | 0/1 |

两条线完全共享 HRNet 输入、H3 checkpoint、候选池、温度、训练预算和评估协议；
只比较 identity hinge 是否能保住两视角，不拼接不同线的结果。四个任务分布在两张
GPU 上，每卡同时运行 standard/hinge 一个 seed。输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_ab_protocol_v1/`。

## 15. 2026-08-16 H6：统一候选生成器的 cardinality curriculum（已完成）

H3/H4 证明 K2-heavy 生成器对少视角有利，但固定 `8:1:1` 仍不能同时保住 V3/V4；
因此 H6 不拼接模型输出，而是从同一个 B2 K=2 checkpoint 开始，在同一个模型内改变
训练阶段的视角数采样。

| 线 | 初始化 | epoch 0--6 | epoch 7--13 | epoch 14--19 |
|---|---|---|---|---|
| CURRICULUM | B2_FIXEDK2 | `8:1:1` | `3:1:1` | `3:2:2` |
| FIXED_MIXED | B2_FIXEDK2 | `3:1:1` | `3:1:1` | `3:1:1` |

两条线均使用 H76（tri-anchor + centered Plücker）、GBT-aligned HRNet 坐标/置信度、
20E、`5e-5` 初始微调学习率、`10/15` 衰减节点、seed 0，并严格评估所有 V2/V3/V4
相机组合。新增的 `RUMPL_CURRICULUM_VIEW_WEIGHTS` 只控制采样概率，不增加视角专用
参数、head 或后处理。

- 脚本：`OpenRUMPL_baseline_audit/launch_h6_cardinality_curriculum_ab_20260816.sh`
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/h6_cardinality_curriculum_ab/`
- 状态：两组均完成 20E 和严格 V2/V3/V4 评估。

严格 H36M S9/S11、action-equal All-17 absolute MPJPE（单位 mm）：

| 训练线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| CURRICULUM `8:1:1 → 3:1:1 → 3:2:2` | 36.939 | 31.631 | 30.484 |
| FIXED_MIXED `3:1:1` | **36.885** | **31.451** | **30.277** |

课程相对固定混合线反而退化 `+0.054/+0.181/+0.207 mm`，且固定混合线与已有
HIGH-LR B2 mixed 控制结果一致，没有产生新的统一模型增益。因此 cardinality
curriculum 停止，不升级为论文主线；H6 只保留为负结果和训练协议对照。

H6 的作用是确定单一空间 baseline；即使课程有效，也只作为训练协议控制，论文核心
仍需依赖后续候选判别器和 token 级时序模块。

## 16. 2026-08-16 H7：候选—视角几何判别器（已完成）

H2 的 oracle 改善没有转化为最终精度，说明候选池中有好解但 E2 评分器难以识别。H7
先在已完成的 H3 生成器和 22-candidate cache 上隔离验证评分器；待 H6 确定更好的
单模型生成器后，再用同一评分器复跑 H6 cache。H7 将 E2 的候选评分器换成带有
candidate-to-ray 几何 token 的视角交叉注意力版本；仍只输入坐标、置信度和相机射线，
不引入 heatmap 或图像特征。

该实现借鉴 Generalizable Human Pose Triangulation 的“多假设 + 学习评分”思想，代码
适配自已有的 ray-view attention 实现，保持相同 holdout、损失、温度和评估协议。

- 适配器：`OpenRUMPL_baseline_audit/train_h7_view_geometry_20260816.py`
- 排队脚本：`OpenRUMPL_baseline_audit/launch_h7_view_geometry_20260816.sh`
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/h7_view_geometry_ab/`

两 seed 均已完成。相同 H3 22-candidate cache 上的严格结果如下（未做跨模型拼接）：

| 评分方式 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H3 baseline（不训练评分） | 39.544 | 30.802 | 28.372 |
| H7 hard | 39.522 | 30.549 | 28.387 |
| H7 soft | 39.679 | **29.433** | **27.232** |

相对 H4 standard `39.483/29.456/27.223`，H7 soft 仅在 V3 改善 `0.023 mm`，
V2/V4 分别退化 `0.196/0.009 mm`，均不足以证明新判别器有效；hard 版本也没有
改善。因此 H7 不进入主线，停止继续增加注意力深度或候选数量。

### 14.1 H4/H5 最终结果

严格 H36M S9/S11、action-equal All-17 absolute MPJPE（温度校准后，mm）：

| 线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H4 standard，均值±std | 39.483±0.002 | **29.456±0.025** | **27.223±0.017** |
| H5 hinge，均值±std | **39.461±0.001** | 29.459±0.018 | 27.211±0.062 |
| E2-C2 soft-cal | 38.700 | 29.486 | 27.274 |

H4/H5 都把 H3 单模型的 V3/V4 从 `30.802/28.372` 降到约 `29.46/27.22`；相对
E2-C2 soft-cal，V3/V4 额外改善约 `0.030/0.051 mm`，但 V2 仍高约 `0.76 mm`。
H5 hinge 相对 H4 只改善 `0.022 mm`（V2）和 `0.013 mm`（V4），V3 基本不变，
低于 seed 波动，不能作为独立有效创新。因此 H5 停止，H4 仅作为“更强单模型生成器
+ utility”的多视角结果记录；最终统一 baseline 仍不能把 H4 的 V3/V4 与其他模型
的 V2 拼接。

## 46. H11：官方 GHT 候选评分与 V2 扩展（2026-08-17）

H11 在同一份 H3 K2-heavy、GBT-aligned HRNet 坐标/置信度缓存上验证
Generalizable Human Pose Triangulation 的“多候选 + 学习评分”思想。H11A 使用官方
whole-pose ScoreNN 结构；H11B 使用已有的绝对候选误差/expected-risk 评分器，只训练
V3/V4；H11C 将同一评分器扩展到全部 V2/V3/V4 组合，避免把不同模型的列拼接。
实现和启动脚本分别为：

- `OpenRUMPL_baseline_audit/train_h76_hypothesis_utility_20260811.py`
- `OpenRUMPL_baseline_audit/train_h76_pairwise_absolute_score_20260814.py`
- `OpenRUMPL_baseline_audit/launch_h11_ght_official_score_ab_20260817.sh`
- `OpenRUMPL_baseline_audit/launch_h11c_v2_absolute_20260817.sh`

严格 H36M S9/S11、action-equal All-17 absolute MPJPE（mm）：

| 线 | V2 | V3 | V4 | 结论 |
|---|---:|---:|---:|---|
| H11A official whole-pose ScoreNN | 39.544 | 33.363 | 32.119 | 评分器反而破坏多视角，停止 |
| H11B absolute pairwise（V3/V4） | 39.544 | 29.425 | 27.197 | V2 未训练，不能作为统一模型 |
| H11C absolute pairwise（V2/V3/V4，seed 0/1） | 39.62±0.01 | 29.43±0.01 | 27.14±0.02 | V3/V4 有效，V2 无效，停止升级 |
| H3 K2-heavy 单模型 | 39.544 | 30.802 | 28.372 | 同一候选池的生成器控制 |

H11 的关键结论是：候选池确实包含可用的多视角解，且绝对误差评分能稳定降低
V3/V4 约 `1.37/1.23 mm`；但 V2 的候选排序没有收益。因此该方向不能被叙述为
“统一 baseline 已解决”，也不能以 H11C 的 V2/V3/V4 混合其它模型结果。
完整输出位于 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h11_ght_official_score_ab/`
和 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h11c_v2_absolute_pairwise/`。

## 47. H12：MTF 源视角归一化融合（2026-08-17，已完成）

根据公开 MTF 实现的 source-normalized pairwise view aggregation，将标准 RUMPL
跨视角 VFT 替换为一个只使用目标视角 token、源视角 token、二者差值（可选 HRNet
置信度）的 pairwise message；不加入 heatmap、图像特征或 camera-ID embedding，后续
PFT 和 3D head 完全保留。该改动位于
`OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py` 的
`MTFSourceNormalizedFusion`，由环境变量显式打开，默认关闭，保证历史 baseline 不变。

| 线 | 唯一变化 | GPU/seed | 输出 |
|---|---|---|---|
| H12A MTF_PLAIN | pairwise source-normalized，不使用置信度 | 0/0 | `.../h12_mtf_source_norm_ab/MTF_PLAIN/` |
| H12B MTF_CONF | 同上，拼接 HRNet confidence | 1/1 | `.../h12_mtf_source_norm_ab/MTF_CONF/` |

两组使用同一 GBT-aligned HRNet 合并 pkl、同一 H35/H76 几何输入、同一随机视角采样、
20E 训练和 V2/V3/V4 全组合严格评估；输入文件和配置的 SHA256 写入各自日志。完整
结果如下（action-equal All-17 absolute MPJPE，mm）：

| 线 | V2 | V3 | V4 | 相对 H3 `39.544/30.802/28.372` |
|---|---:|---:|---:|---|
| H12A MTF_PLAIN | 46.813 | 33.403 | 29.798 | `+7.269/+2.601/+1.426` |
| H12B MTF_CONF | 41.378 | 31.266 | 28.140 | `+1.835/+0.464/-0.232` |

因此 H12 不是有效 baseline：无置信度线三列均退化，带置信度线只在 V4 小幅下降，
却损失 V2/V3。原因与实现路径一致：当前 MTF 替换器绕过了 RUMPL 原有的多层
fusion-token Transformer，将每个 target/source pair 的浅层消息直接做视角平均；
它没有保留原 VFT 的 learned fusion query、高阶跨视角交互和后续 token 归一化路径。
置信度只能部分缓解，不能恢复两视角的信息。H12 停止继续堆深度或调温度，保留代码和完整日志作为失败
消融。输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h12_mtf_source_norm_ab/`。

下一步若继续 MTF，只做一个有明确假设的修正：保留 target token，以零初始化的
residual gate 注入 source-normalized message，再在同一协议下做单变量 A/B；若仍不能
同时保持 V2 并改善 V3/V4，则关闭该方向，回到 H11 的候选池/几何求解器主线，不再
进行无依据的时序或融合堆叠。

## 48. H13：MTF residual-before-VFT（2026-08-17，已完成）

H13 只修正 H12 的结构性问题：MTF pairwise message 不再替换 RUMPL VFT，而是通过
零初始化 scalar gate 加到每个 view token，随后继续进入原 learned fusion-token VFT、
PFT 和 3D head。两组仍只比较是否使用 HRNet confidence：

| 线 | 唯一变化 | GPU/seed |
|---|---|---:|
| H13A RESIDUAL_PLAIN | MTF residual，不含 confidence | 0/0 |
| H13B RESIDUAL_CONF | MTF residual，含 confidence | 1/1 |

代码开关：`RUMPL_MTF_SOURCE_NORM_FUSION=1`、
`RUMPL_MTF_SOURCE_NORM_RESIDUAL=1`；启动脚本：
`OpenRUMPL_baseline_audit/launch_h13_mtf_source_norm_residual_ab_20260817.sh`；输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h13_mtf_source_norm_residual_ab/`。
该实验仍使用同一 GBT-aligned HRNet pkl、20E 和 V2/V3/V4 全组合评估；最终以同一
checkpoint 的三列是否同时下降作为进入主线的唯一判据。

### 48.1 H13 最终结果

两组均完成 20E、严格 H36M S9/S11 action-equal All-17 absolute MPJPE（mm）：

| 线 | V2 | V3 | V4 | 相对 H3 `39.544/30.802/28.372` |
|---|---:|---:|---:|---|
| H13A residual，无 confidence | 41.235 | 31.086 | 28.256 | `+1.691/+0.284/-0.116` |
| H13B residual，含 confidence | 40.846 | 30.943 | 27.671 | `+1.302/+0.141/-0.701` |

H13B 的 gate 最终约 `-0.081`，说明 residual 分支确实学到了非零修正，但它只对
V4 有效，V2/V3 退化；H13A 同样没有统一收益。因此 MTF 在当前坐标级 RUMPL 上
无论直接替换 VFT（H12）还是 residual-before-VFT（H13）都停止，不再进行深度、
温度或 gate 调参。H13 输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h13_mtf_source_norm_residual_ab/`。
空间主线回到 H11 暴露的候选池/几何求解问题。

## 49. H14：C2 候选池上的官方绝对误差评分（2026-08-17，已完成）

H14 将 H11 的 Generalizable Human Pose Triangulation 风格 absolute-error ScoreNN
迁移到 C2 的 22-candidate frozen cache。候选生成、HRNet 坐标/置信度、相机射线和
RUMPL 生成器均不改变；只把评分目标从 E2 的相对 delta 改为每个候选的绝对关节误差，
并同时训练 V2/V3/V4，避免把 V2 和 V3/V4 来自不同模型。两个 seed 使用相同的
10E direct + 5E expected-risk 预算、温度 1.8、ranking weight 0.25。

需要严格区分：H14 的 `baseline` 是 frozen C2 candidate cache 中的原始同视角候选，
不是已经经过 E2 soft scorer 的 C2 最终输出；因此应将 H14 soft 与 E2-C2 soft-cal
直接并列，而不能把 H14 的 baseline 误写成 C2 结果。

| 线（seed 0/1 均完成） | V2 | V3 | V4 |
|---|---:|---:|---:|
| H14 raw candidate baseline | 38.686 | 30.943 | 28.629 |
| H14 hard absolute score | 38.689 | 30.123 | 27.889 |
| H14 soft absolute score | 38.819 | 29.495 | 27.286 |
| E2-C2 soft-cal（统一空间参考） | **38.700** | **29.486** | **27.274** |

H14 soft 相对 E2-C2 变化为 `+0.119/+0.009/+0.012 mm`，在 V2、V3、V4 都没有
达到新的统一最优；H14 的绝对评分目标可以复现 V3/V4 的候选选择收益，但不能超过
已经校准的 E2-C2。H14 因此作为“官方评分目标的公平对照”保留，不进入主 baseline，
也不再继续增加评分器深度或温度搜索。

代码与结果：

- `OpenRUMPL_baseline_audit/train_h76_pairwise_absolute_score_20260814.py`
- `OpenRUMPL_baseline_audit/launch_h14_c2_absolute_score_ab_20260817.sh`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h14_c2_absolute_pairwise/`

## 50. 阶段切换：固定 HRNet 空间参考，进入 ResNet-152 输入线

截至 H14，HRNet 坐标级统一参考仍为 E2-C2 soft-cal `38.700/29.486/27.274 mm`；
它已经超过 GBT 的 V3 参考 `30.4 mm`，但尚未超过 GBT 给出的 V2/V4 参考 `36.8/26.0 mm`。
H4/H11/H14 的候选评分、H12/H13 的 MTF 替换和 residual 均未产生三列统一下降，
继续在 HRNet/RUMPL 上盲目调融合器的边际收益很低。因此按六阶段计划转入第二阶段：
保持 RUMPL 训练/评估协议、相机和严格 table2 不变，只替换冻结的二维前端为官方
Learnable Triangulation ICCV-2019 ResNet-152 checkpoint，先建立 R0（纯坐标）和
H76（同一三角化锚点/中心射线/Plücker 控制）的 ResNet-152 两条线，再决定是否把
ResNet 结果作为主 baseline。

已启动的执行脚本：
`OpenRUMPL_baseline_audit/launch_resnet152_rumpl_gbt_screen_gpu1_20260817.sh`；
结果根目录：`/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/`。
该线只占用空闲 GPU1，不触碰 GPU0 上已有的其他目录实验。导出阶段严格使用官方
LT 的去畸变、annotation-box crop、384×384 resize、相机更新和 joint mapping；
RUMPL 只接收导出的 2D 坐标、置信度和射线，不使用热图或图像特征。

## 51. 2026-08-18 GHT PoseDSAC 完整审计与后续执行修正

上一节的 ResNet 阶段切换暂缓：ResNet-152 前端导出已经完成，但在 HRNet 主线尚未
达到统一目标前不启动 R0/H76 长训。首先补齐了 GHT (Generalizable Human Pose
Triangulation, CVPR 2022) 公开代码中此前遗漏的 PoseDSAC 假设生成，而不是继续在
固定候选池上猜测。

脚本：`OpenRUMPL_baseline_audit/audit_ght_pose_hypotheses_20260818.py`。
它严格按官方 `src/dsac.py::PoseDSAC.__sample_hyp`，对每个关节从全部 2..K 视角
子集均匀采样，使用 HRNet 坐标与 H36M `K,R,t` 做无标签 DLT；标签只用于验证
oracle。全验证集为 2,021 帧、200 假设，输出：

| 候选池 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 官方 GHT PoseDSAC 随机候选 | 122.353 | 53.044 | 49.377 |
| H76 原有 11 候选 | 38.686 | 30.943 | 28.629 |
| H76 + confidence 22 候选 | 38.682 | 30.941 | 28.627 |
| H76-22 与 GHT 的 oracle 并集上限 | 38.682 | 30.870 | 28.590 |

即使加入 GHT 候选，理论上限只改善 `0/0.071/0.052 mm`；置信度加权 DLT 的并集
仍为 `38.682/30.860/28.575 mm`。因此 GHT 原始候选在当前 HRNet 噪声和 H36M
相机退化对下明显弱于 RUMPL 学习候选，不启动没有候选上限支撑的长程 GHT 评分器。
完整记录见 `OpenRUMPL_baseline_audit/GHT_POSEDSAC_AUDIT_20260818.md`，结果位于
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/ght_pose_hypothesis_audit_v1/`。

在此基础上启动两个可解释的单模型训练对照，均从 H1 高学习率 checkpoint
(`36.885/31.451/30.277 mm`) 继续，保持 H76 anchor + centered-Plücker、HRNet
坐标/置信度输入、相机和严格 table2 不变，只改变视角基数采样：

| 实验 | 采样权重 V2:V3:V4 | 输出 |
|---|---|---|
| G1 | 1:1:1（均衡） | `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h1_balanced_continuation/` |
| G2 | 5:1:1（两视角强化） | `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h1_v2heavy_continuation/` |

二者不是 ensemble 或后处理叠加；目的是判定当前 V2/V4 差距来自视角采样优化，还是
来自模型候选本身。完成后只保留严格三列均下降或在目标列有可复现收益的线，随后再
决定是否恢复 ResNet 或进入 T=9 时序实验。

### 51.1 G1/G2 最终严格结果与决策

G1/G2 均完成 20 轮训练。G1 原启动脚本在训练结束后因运行期间脚本行号变化导致
shell 解析错误，但训练权重完整保存；使用独立只评估脚本
`OpenRUMPL_baseline_audit/eval_g1_h1_balanced_strict_20260818.sh` 恢复了完全相同
环境下的 V2/V3/V4 严格结果，不需要重训。

| 方法 | V2 | V3 | V4 | 相对 H1 raw |
|---|---:|---:|---:|---:|
| H1 raw（继续训练起点） | **36.885** | **31.451** | **30.277** | — |
| G1，1:1:1 | 36.988 | 31.523 | 30.290 | `+0.104/+0.072/+0.013` |
| G2，5:1:1 | 37.034 | 31.548 | 30.426 | `+0.150/+0.097/+0.149` |
| E2-C2 soft-cal（冻结 T=1 参考） | 38.700 | **29.486** | **27.274** | 不同候选生成器 |
| GBT HRNet（T=9） | 36.800 | 30.400 | 26.000 | 外部参考 |

两种采样续训均在三列退化，说明 H1 已在该训练目标附近饱和；当前缺口不是
V2/V3/V4 出现频率不足。停止继续扫描采样权重、学习率或延长 H1，不对 G1/G2 再训练
E2。HRNet 单帧主参考保持 E2-C2 soft-cal `38.700/29.486/27.274 mm`，H1 raw 只作为
V2 优先消融。

下一步不切换 ResNet，先补齐与 GBT 公平比较缺失的 T=9：在冻结 E2-C2 候选与绝对
anchor 路径上，做 candidate-level temporal go/no-go。第一步先计算训练 holdout 的
T=9 temporal candidate oracle；只有 V2 或 V4 上限改善至少 `1.0 mm` 才训练 MTF-TFT
式、零初始化 identity residual/utility gate。该实验不再对最终 3D pose 做 MixSTE
平滑，也不重复已失败的 Pre-VFT temporal。若 temporal oracle 不足，则正式冻结 HRNet
线并恢复已导出的 ResNet-152 R0/H76 对照。

## 52. H15/H16：T=9 候选时序 go/no-go（2026-08-18，已完成）

根据当前决策，ResNet-152 阶段暂停；在 HRNet 线达到 GBT 目标前不得切换输入前端。
H15/H16 只在当前 C2 候选池上验证“时序是否能解决候选选择问题”，不改变 HRNet、
RUMPL H76、三角化候选、相机或 E2 单帧 scorer。

### 52.1 H15：零训练 temporal candidate oracle

输出目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle/`；
脚本：`audit_temporal_candidate_oracle_20260818.py`。验证使用完整 S9/S11 dense
T=9 pkl（26,269 个同步中心），窗口帧间隔 5；候选生成不读标签，标签只用于 oracle
评估。结果为 action-equal All-17 absolute MPJPE（mm）：

| 窗口 | V2 原始 C2 | V2 中心帧 oracle | V2 时序均值 oracle | V3 原始 | V3 中心 oracle | V3 时序 oracle | V4 原始 | V4 中心 oracle | V4 时序 oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T=1 | 38.718 | 36.205 | 36.205 | 30.947 | 21.732 | 21.732 | 28.623 | 17.044 | 17.044 |
| T=3 | 38.740 | 36.224 | 36.764 | 30.966 | 21.744 | 23.057 | 28.641 | 17.053 | 18.679 |
| T=5 | 38.763 | 36.244 | 37.007 | 30.987 | 21.756 | 23.640 | 28.658 | 17.063 | 19.371 |
| T=9 | 38.809 | 36.284 | 37.309 | 31.028 | 21.782 | 24.393 | 28.694 | 17.083 | 20.251 |

中心帧候选池仍有较大理论余量，但按时间平均误差选候选在 T=9 比中心 oracle
退化 `+1.025/+2.610/+3.168 mm`（V2/V3/V4）。这说明“跨帧平均误差”不是可用的
时序效用定义，不能把它写成时序提升证据。

### 52.2 H16：短程、零初始化 temporal utility residual

脚本：`train_temporal_e2_c2_screen_20260818.py`，启动脚本：
`launch_h16_temporal_c2_screen_20260818.sh`；输出目录：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h16_temporal_c2_screen/`。
冻结经过校准的 E2-C2 scorer（T2/T3/T4 temperature = 0.4/1.8/1.8），只训练一个
T=9 candidate-level Transformer residual；残差头零初始化，所以训练前严格等于
E2-C2。训练窗口均匀抽样 S1/S5/S6/S7，S8 作为内部 holdout，S9/S11 只最终评估。

T=9 baseline（完整验证集）为 `38.809/31.028/28.694 mm`。短程训练在 S8
holdout 的均值从 baseline `15.281` 变为最佳 `15.333 mm`，没有通过 identity floor；
因此没有保存可部署时序权重，最终验证仍为 baseline：

| 方法 | V2 | V3 | V4 | 结论 |
|---|---:|---:|---:|---|
| E2-C2 T=9 frozen | 38.809 | 31.028 | 28.694 | control |
| H16 residual（最终无增益，未保留） | 38.809 | 31.028 | 28.694 | stop |

H15/H16 的共同结论是：当前候选池的瓶颈是单帧候选效用可识别性，而不是缺少一个
时序平滑器；在没有单帧效用提升前继续扩大时序模型没有依据。该结论不等于“论文中
时序无效”，而是说明当前 HRNet/RUMPL 输入上的时序分支尚未看到可学习的正增益。

## 53. 下一步：H17 V2 pairwise candidate utility（不切 ResNet）

在用户明确要求“未达目标前不能转 ResNet”后，下一步固定 HRNet 前端，针对最紧迫的
V2 缺口 `38.700 -> 36.8` 做单变量、单帧实验。H17 只处理每个两视角任务中的两个
同视角候选（H76 与 confidence-weighted triangulation），以 GHT 的 hypothesis
scoring 思想为依据，但将输入限制为坐标级、置信度和射线几何：候选 root-relative
坐标、候选间位移/范数、视角置信度、候选到射线的垂距以及冻结 E2 score。

训练目标仍是训练集真值 3D 监督下的 candidate error-delta/ranking，不使用 teacher
prediction、蒸馏或验证标签；输出 residual 零初始化，严格保留 E2-C2 为 identity
control。只在 S8 holdout 选择，S9/S11 最终一次评估。通过门槛为：

1. V2 相对 E2-C2 至少下降 1.0 mm 且不超过 0.2 mm 退化于 V3/V4；
2. 六个两视角组合中至少四个组合下降，不能只靠单个相机对；
3. 若未达到门槛，停止继续调 scorer，转做有明确候选上限支持的稳健求解器/视角
   负增益分析；ResNet-152 仍保持冻结，直到 HRNet 线先达到既定 GBT 目标。

### 53.1 H17 结果（2026-08-18，已完成）

H17 使用完整训练缓存中的 S1/S5/S6/S7，S8 做选择，S9/S11 只做最终一次评估；
15E、batch 1024。冻结 E2-C2 的 T=1 baseline 在当前 dense 验证上的 V2 为
`38.718 mm`（六个相机对平均），pairwise residual 最终为 `38.737 mm`，即
`+0.019 mm`，没有收益；六个相机对也没有形成一致下降。结果目录：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h17_v2_pairwise_utility/`。

训练/验证分布还暴露出一个重要诊断：confidence candidate 在训练主体上的逐关节
胜率只有约 `7.4%--9.0%`，而 S9/S11 为 `10.5%--24.4%`（S9 明显更高）。因此
H17 scorer 在训练中学到“保留 H76”是合理的，但无法迁移到 S9，说明当前候选质量
存在 subject/domain shift；这不是再堆一个 Transformer 就能解决的。后续若再做
候选效用，必须先做无标签域稳定性/校准实验，并按相机对报告结果，不能把 H15
oracle 直接当作可实现精度。

H17 未达到门槛，保留 E2-C2 `38.700/29.486/27.274` 作为统一 HRNet T=1 参考，
同时保留 dense T=9 control `38.809/31.028/28.694`。时序和 V2 scorer 两条支线
均暂停；ResNet-152 仍不启动。

## 54. H18：正常帧优先的 E2-C2 temporal pose residual（2026-08-18，执行中）

### 54.1 重新启动的原因

用户要求时序不能只在遮挡下有效，正常 H36M 也必须有可测的下降。H16 不能作为
这个问题的否定证据：它只训练了 4096 个窗口、4 个 epoch，而且 temporal 分支
只改候选 utility logits，没有直接修正中心 3D pose。更关键的是，H16 的
`baseline` 字段记录的是原始 H76 候选误差，`temporal` 字段才是 E2 soft pose，
因此两者不是同一基线口径。H18 修正了这一评估错误。

### 54.2 实验定义

冻结当前 E2-C2 的 22 候选、评分器和已选温度（V2/V3/V4 = 0.4/1.8/1.8），
先离线生成与正式推理一致的每帧 soft-fusion 3D pose。新模块是 MixSTE 风格的
spatial block → temporal block 分解，输入 T=9 个连续稀疏帧的 11 个视角任务
输出，直接预测中心帧 3D residual；最后一层零初始化，step 0 严格等于 E2-C2。
clean control 中固定 root，避免收益来自重写相机几何的绝对平移。

训练使用完整 S1/S5/S6/S7 窗口，S8 整个 subject 只用于 checkpoint 选择，S9/S11
只做一次最终评估。代码与输出：

- `OpenRUMPL_baseline_audit/build_e2_fused_temporal_cache_20260818.py`
- `OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py`
- `OpenRUMPL_baseline_audit/launch_h18_clean_temporal_e2_20260818.sh`
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal/`

### 54.3 H18 最终结果（2026-08-18，已完成）

真正的 E2-C2 中心基线和第 0 epoch 结果如下（mm；8 epoch 仍在执行）：

| split | V2 baseline → temporal | V3 baseline → temporal | V4 baseline → temporal |
|---|---:|---:|---:|
| S8 holdout，第 0 epoch | 20.402 → **18.465** (-1.937) | 13.702 → **12.726** (-0.976) | 11.739 → **11.045** (-0.694) |
| S8 holdout，第 1 epoch（最佳） | 20.402 → **18.322** (-2.080) | 13.702 → **12.763** (-0.939) | 11.739 → **11.118** (-0.621) |
| S9/S11 dense E2-C2 baseline | 38.827 | 29.638 | 27.371 |
| S9/S11 H18 temporal（epoch 1） | **37.633** (-1.194) | **29.385** (-0.253) | 27.398 (+0.027) |

完整 S9/S11 的 action-equal 平均从 `31.945` 降到 `31.472 mm`（-0.473 mm）。
收益主要来自 V2，V3 为小幅稳定下降，V4 基本持平并有 `0.027 mm` 的噪声级退化；
因此 H18 可以作为“正常数据有效、尤其补强两视角”的时序控制，但不能声称三种
视角全部提升。第 2--7 epoch 在 S8 上继续下降，说明直接长训会跨主体过拟合，
后续固定使用 S8 选择的早停 checkpoint，不再延长训练。

结论：clean temporal branch 保留，下一步才允许在该 checkpoint 上加入遮挡增强和
uncertainty gate，并分别报告 clean/occlusion；若遮挡训练破坏上述 clean 结果，则
保留 H18 作为 clean temporal 消融，不把遮挡收益包装成主模型提升。

### 54.4 训练动力学复核

H18 的第 1 epoch 最优不是数据对齐错误。独立审计确认：训练与验证 pkl 的组键、
相机完整性、subject/action/frame 顺序均与 cache 一致。日志显示 train loss 从
`0.00840` 持续降至 `0.00699`，但 S8 holdout 均值在 epoch 1 的 `14.068 mm`
之后升至 epoch 7 的 `15.237 mm`，且 V4 先从 `-0.621 mm` 变为 `+0.688 mm`。
这是高学习率和高度重叠窗口下的小残差模型过拟合，不是“训练轮数不够”。

同时，H18 每个 epoch 包含 64,141 个训练窗口、约 1,003 个 optimizer step；8 epoch
约 8,024 次更新。它冻结了 E2-C2 主干，只训练约百万参数的 residual，不能与 GBT
从头训练完整网络的 300k iterations 直接类比。

为验证是否可以稳定训练，已启动 H18-lowLR（输出目录
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/`）：
学习率 `5e-5`、weight decay `5e-4`、12 epoch，其余协议完全不变。该结果确定前，
不启动遮挡增强。

### 54.5 H18-lowLR 最终结果（2026-08-18，已完成）

H18-lowLR 使用与 H18 完全相同的输入、窗口、冻结的 E2-C2 主干和 S8 选择协议，
只将学习率从 `3e-4` 降为 `5e-5`，weight decay 从 `1e-4` 提高为 `5e-4`，并跑
满 12 个 epoch。S8 的最佳 checkpoint 为第 3 epoch，而不是第 1 epoch；这说明原
H18 的早停主要受高学习率过拟合影响，而不是训练窗口不足或 cache 错位。

| 方法 | V2 | V3 | V4 | V2/V3/V4 均值 |
|---|---:|---:|---:|---:|
| T=9 E2-C2 baseline | 38.827 | 29.638 | 27.371 | 31.945 |
| H18 高 LR，第 1 epoch | 37.633 | 29.385 | 27.398 | 31.472 |
| H18-lowLR，第 3 epoch | **37.704** | **29.231** | **27.219** | **31.385** |

在正式 S9/S11 一次性评估中，H18-lowLR 相对同口径 T=9 E2-C2 baseline 的变化为：
V2 `-1.123 mm`、V3 `-0.407 mm`、V4 `-0.153 mm`，平均 `-0.561 mm`。与高 LR
版本相比，V2 略低 `0.071 mm`，但 V3/V4 分别再降低 `0.154/0.180 mm`，因此整体
更均衡。第 3 epoch 后 holdout 均值逐步回升，仍需按 S8 早停，不能把“跑满 epoch”
当作自动有效。

结论：正常帧上的时序残差分支已经有可重复的小幅收益，且不再只改善 V2；保留
H18-lowLR 第 3 epoch 作为 clean temporal checkpoint。下一步允许在该 checkpoint
上做 uncertainty-gated occlusion 训练，但必须同时报告 clean 表，并要求遮挡版在
clean 数据上的退化不超过预先设定阈值；否则只保留 clean temporal 消融，不把遮挡
收益包装成主模型收益。完整记录和逐 epoch 指标见
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/result.json`。

## 55. H19：修正为 GBT 公平协议的 causal seq2seq temporal（2026-08-18，已排队）

### 55.1 H18 不能直接与 GBT 时序结果比较的原因

复核代码后发现，H18 的 T=9 输出是窗口中间帧，因此使用了后 4 帧；GBT 在评估时
输入当前帧和 8 个历史帧，只输出最新帧。H18 还只监督中心帧，空间层与时序层成组
堆叠，而 MixSTE/GBT 都训练整段序列输出，MixSTE 明确采用 spatial/temporal 逐层
交替。因此 H18 只能证明当前数据中存在可利用的时序信息，其数值不能作为正式的
GBT 公平对比结果。

此外，H18 为 S8 holdout 选择 checkpoint 后直接用 S1/S5/S6/S7 权重测试，没有
像 GBT/H36M 标准协议那样把 S8 纳入最终训练。H19 改为两阶段：先用
S1/S5/S6/S7 -> S8 选择 epoch；固定 epoch 后重新初始化，在 S1/S5/S6/S7/S8 上
重训，S9/S11 只评一次。

### 55.2 root 瓶颈诊断

在严格 latest-frame T=9 窗口上，E2-C2 baseline 为约
`38.876 / 29.689 / 27.419 mm`。其 root 误差分别约为
`34.488 / 28.550 / 26.856 mm`。H18 完全禁止修改 root，而 GBT 的 V4 目标是
`26.0 mm`；因此仅靠 root-protected relative-pose residual 很难跨过 V4 目标。
H19 将 root translation 与 root-relative pose 分成两个零初始化输出头，并以小幅
trust region 单独控制 root residual。

### 55.3 H19 结构与消融

共同设置：冻结 E2-C2 单帧前端；T=9；只输出最新帧做正式评估；训练监督 9 个输出
帧；spatial/temporal block 逐层交替；初始函数严格等于 E2-C2；S8 选轮数后用五个
标准训练主体正式 refit。

| 实验 | root residual | temporal loss | 目的 |
|---|---|---|---|
| H19A | 禁止 | 0 | 隔离公平 causal/seq2seq/alternating 的结构收益 |
| H19B | 独立小幅 root head | 0 | 检验 absolute root 是否是 V4 天花板 |
| H19C | 独立小幅 root head | MixSTE-style velocity loss | 检验时序一致性损失的净收益 |

实现与启动器：

- `OpenRUMPL_baseline_audit/train_e2_causal_temporal_seq2seq_20260818.py`
- `OpenRUMPL_baseline_audit/launch_h19_causal_seq2seq_variant_20260818.sh`
- `OpenRUMPL_baseline_audit/run_h19_when_gpu_free_20260818.sh`

代码已通过 CPU 前向、训练、评估和 checkpoint 冒烟测试。由于两张 GPU 均被已有
任务占满，H19A/H19C 已进入 GPU0 安全等待队列，H19B 已进入 GPU1 安全等待队列；
只在对应卡显存低于 4 GB 时启动，不抢占现有任务。正式门槛仍为 GBT HRNet 的
`36.8 / 30.4 / 26.0 mm`，未跨过前不转 ResNet。

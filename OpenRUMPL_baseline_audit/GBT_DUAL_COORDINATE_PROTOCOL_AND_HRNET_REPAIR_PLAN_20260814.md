# GBT 双坐标输入协议与 HRNet 修复方案（2026-08-14 冻结版）

## 1. 今日决策

论文主表采用 GBT Table I 的两类输入，但三维网络只读取二维坐标、置信度和相机参数：

1. `C-Res152†`：官方 Learnable Triangulation ResNet-152，COCO 预训练后在
   MPII/H36M 微调，图像去畸变；
2. `C-HRNet`：COCO-only HRNet-W32 384x288，YOLOX-X 人框，图像去畸变。

禁止向三维网络输入 HRNet 热图、中间图像特征、Volumetric 特征或 A1D。两条前端
分别冻结缓存，同一个三维模型分别训练，不能把两个前端的坐标混合训练。

共同协议：H36M S1/S5/S6/S7/S8 训练、S9/S11 测试；删除 GBT/LT 指明的损坏
序列；absolute MPJPE、All-17、无对齐；V2 平均全部 6 个相机对，V3 平均全部
4 个三相机组合，V4 为全部四相机；action-equal 为主，同时保存 frame-weighted；
T=1 与 T=9 分列。

## 2. 当前可信结果与缺口

| 输入/模型 | T | V2 | V3 | V4 | 口径 |
|---|---:|---:|---:|---:|---|
| GBT ResNet-152† | 9 | 29.900 | 24.400 | 22.700 | 论文目标 |
| LT coords + H76 | 1 | 43.442 | 24.425 | 21.234 | 当前可信坐标基线 |
| GBT HRNet | 9 | 36.800 | 30.400 | 26.000 | 论文目标 |
| GBT HRNet | 1 | — | — | 29.400 | 论文时序消融 |
| 当前 raw HRNet + H76 | 1 | 52.281 | 37.348 | 34.016 | 仅诊断，前端不一致 |
| 当前 raw HRNet + 原 RUMPL | 1 | 84.235 | 64.868 | 55.460 | 仅诊断，前端不一致 |

当前 HRNet 训练缓存来自 annotation box，测试缓存来自 RTMDet-M 全图检测；二者
均未在检测前去畸变。因此这组 HRNet 数字不能作为最终外部公平结果。当前 H76 的
V2 逐对为 `40.258/46.966/74.683/74.096/37.717/39.967` mm，主要失败仍集中在
`1-4` 与 `2-3` 两个几何退化相机对。

## 3. GBT 评估与“坏相机对不影响”的澄清

GBT Sec. IV-C 明确写明：每个相机数均测试所有输入视角组合并取平均。因此 H36M
V2 是 6 对平均、V3 是 4 组平均。论文没有公布逐相机对结果，不能声称每一对都不受
影响；36.8 mm 只证明它的六对平均被压低。

GBT 的主要优势不是单独一个 bias，而是以下组合共同针对稀疏/坏几何输入：

- 训练始终随机采样两视角，300k iterations，使所有相机对成为直接训练任务；
- 全局 joint-view-time token 自注意力，以全身与历史姿态补偿两射线的深度退化；
- Plucker ray + 15-frequency harmonic embedding；
- confidence bias 与 ray-distance geometry bias 加在每层全局 encoder attention；
- joint-query decoder 直接回归完整 absolute 3D，不受逐关节三角化解唯一解的限制；
- T=9、20% token dropout、synthetic views 与 scene centering。

论文数据也表明它不是在所有场景都优于几何：ResNet-152† 下，GBT 相对 Algebraic
Triangulation 在 V2 从 51.1 降到 29.9，但 V3 从 23.4 变为 24.4、V4 从 19.1
变为 22.7。它最强的是低视角鲁棒先验，而不是高视角几何精度。

GBT 的 HRNet V4 消融为：无组件 39.0；centering 49.2；再加 synthetic 40.6；
在此前提下分别加 confidence/geometry bias 为 33.2/33.1；两个 bias 同时为 26.0。
T=1 到 T=9 又从 29.4 降到 26.0。故 26.0 不能全部归因于 bias，也不能把我们的
单帧局部 bias 实验与它等价。

## 4. C-HRNet 冻结实现

主协议命名为 `C-HRNet-YX`，准确表述为“对齐 GBT 已公开细节、采用官方 YOLOX-X
的可复现实例”，不是严格复现，因为 GBT 未公开 YOLOX 规格、阈值、box padding、
HRNet checkpoint/codec 和代码。

冻结如下：

1. 使用 H36M 相机 `K,k,p` 将整张 RGB 图像重映射到无畸变图像，保持 `K'=K`；
2. 在无畸变整图上运行官方 MMDetection `YOLOX-X 8xb8-300e COCO`；仅 person 类。
   GBT 正文只写 “YOLOX box proposals”，没有公开具体规模/配置，因此 YOLOX-X
   是我们为“最高官方 YOLOX 标准检测精度”选定的、可复现的明确实例，不得称为
   GBT 的严格 detector 复现。官方 YOLOX 标准表中 X 规格 COCO val AP 为 51.1，
   高于 S/M/L；其官方 demo 默认 conf=0.25、NMS=0.45，但本地 MMDetection
   配置的 NMS 为 0.65，二者必须在 manifest 中分别记录，不能混称。初始诊断阈值为
   `score_thr=0.25`；若验证集出现漏检，只能统一下调为 `0.10`（训练/验证同值），
   不允许按帧兜底；单人场景固定取 detector score 最高框；
3. 使用官方 MMPose `td-hm_hrnet-w32_8xb64-210e_coco-384x288` 与当前公共权重；
   `GetBBoxCenterScale padding=1.25`、MSRA 96x72 codec、flip-test 与 shift-heatmap
   保持官方配置；
4. 只导出解码后的 COCO-17 全图二维坐标、17 个 peak confidence、检测框和分数；
5. 首轮继续使用已核验的 RUMPL COCO-to-H36M 映射，以保证只改变前端处理；不得
   同时更换模型 token 语义；
6. 输出坐标已经处于无畸变像素系，射线构造必须使用相同 `K` 且 distortion=0；
7. 训练和测试必须运行完全同一个 exporter，禁止 annotation-box train 与 detector-
   box test；保存软件版本、config/checkpoint SHA256、所有阈值与逐帧选框 manifest。

`C-Res152†` 保持当前官方 LT 坐标缓存，不改其 annotation-box/384x384/去畸变和
H36M+MPII 微调权重，以匹配 GBT 的 dagger 输入。两个输入协议分别报告。

## 5. 实施和实验顺序

### 训练协议暂存（不在前端对齐阶段执行）

GBT 的训练细节单独冻结为后续实验变量：全程从 6 个相机对中均匀随机采样 V2，
训练约 300k optimizer updates，再测试 V2/V3/V4。当前阶段不把它与 HRNet 前端
同时改变；否则无法判断收益来自数据处理还是视角采样。前端缓存和几何门禁通过后，
再按 E0--E3 的 2×2 采样/训练量消融执行，并记录每个相机对的采样直方图。

### A. 前端单元测试与验证集门禁

先只导出 S9/S11 验证集，不训练三维网络。必须通过：

- train/val exporter 配置和代码 SHA 完全一致；
- 无畸变图像、坐标、内参三者投影一致，GT2D triangulation 仍约 0.5 mm；
- 无漏帧、重复帧和跨相机不同步；检测缺失率、多人误选率明确；
- 统计 direct-13 与四个虚拟关节的 2D pixel error、置信度校准、每相机误差；
- DLT、confidence-DLT、IRLS 对 V2/V3/V4 全组合及每个相机对出表。

这里使用 GT 只做一次性管线诊断，不以 S9/S11 选择 YOLOX 型号、阈值或 padding。

### B. 完整前端缓存

验证通过后，两卡分 shard 导出 S1/S5/S6/S7/S8 与 S9/S11。全部大文件放在
`/mnt/data/cjydata/`，日志与 manifest 放在
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/`。

### C. 同输入三维基线

两卡并行、同一 C-HRNet-YX 缓存：

- GPU0：原始 RUMPL，从头训练，作为模块消融基线；
- GPU1：H76（tri-anchor + anchor-centered Plucker + 当前视角 curriculum），从头训练。

先完成 T=1，均报告 V2/V3/V4 全组合、逐相机对和逐关节。任何 A1D、热图、图像
特征与旧泄漏缓存均不得读取。

### D. GBT 差距分解

在 H76 主干不变时，只做两个有直接问题指向的控制：

1. 当前 `fixed-V2 8 epochs -> 3:1:1 V2/V3/V4` curriculum；
2. GBT 对齐的“全程随机 V2、六对均衡”训练。

若修复后 T=1 V4 接近 GBT 的 29.4，但 V2 仍远高于 36.8，说明前端基本解决，
剩余瓶颈是坏几何对下的全身/时序先验；若 V3/V4 仍差 5 mm 以上，则继续检查
COCO-to-H36M 虚拟关节、confidence calibration 和训练/测试分布，不先堆模型。

### E. 后续模型选择（只在 C/D 有结果后）

- 首选有官方代码的 GHT/CVPR 2022 多假设整姿态评分，作为 H76 两视角退化分支；
- T=9 必须在原始 ray token 融合前建模并与 T=1 分列，不能再重复最终 3D pose
  residual、MixSTE、冻结 query residual 等已经失败的路线；
- 不重新做局部 confidence/geometry bias 微调。若采用 GBT bias，必须作为“全局
  JxVxT encoder 的结构消融”实现，不能把旧 RUMPL 局部 bias 冒充等价复现。

## 6. 论文初版可固定的主线

暂定主线：**稀疏二维坐标条件下的锚定 Plucker 射线融合**。当前可以写的三个模块：

1. confidence-weighted triangulation anchor 提供稳定 absolute 3D 初值；
2. anchor-centered Plucker line representation 保留相机无关的射线几何；
3. sparse-view curriculum 直接优化坏相机对与可变视角数。

主表严格分 `ResNet-152† coordinates` 与 `HRNet coordinates` 两个输入，分 T=1/T=9，
并增加逐相机对结果，弥补 GBT 只报组合平均的不足。当前 LT 结果已经证明 V3/V4
不弱，论文是否成立取决于修复 HRNet 后的 V2，以及后续多假设/时序是否能压低
`1-4`、`2-3` 两个坏对；在结果出来前不把未验证模块写成贡献。

## 7. 禁止项

- 不使用热图、HRNet feature、Volumetric、A1D；
- 不把 annotation-box、RTMDet、YOLOX 的结果混成同一个“HRNet”输入；
- 不用 S9/S11 调 detector 或训练超参数；
- 不复用旧 A1D V2/V3 泄漏结果；
- 不声称严格复现无代码的 GBT；
- 不重复已失败的最终 3D residual、普通 MixSTE、蒸馏和局部 bias 参数扫描。

## 8. 2026-08-14 数据对齐实现记录

已新增以下可复用脚本，均调用环境中官方 OpenMMLab API：

- `export_h36m_gbt_aligned_hrnet_20260814.py`：`mmdet.apis.inference_detector`
  + `mmpose.apis.inference_topdown`，整图去畸变、检测框、HRNet 解码坐标，分片
  输出预测和 SHA256 manifest；显式处理 mmdet/mmpose registry scope。
- `merge_h36m_gbt_aligned_hrnet_20260814.py`：严格检查每条记录恰好一个预测，
  COCO-17→H36M-17 映射、既有 lower-body swap、保留原始畸变参数并将 2D 坐标流的
  `camera.k/p` 置零。
- `audit_gbt_aligned_hrnet_cache_20260814.py`：检查坐标有限性、完整四相机分组、
  坐标系标记和 3D 标注针孔投影门禁。
- `launch_gbt_aligned_hrnet_export_20260814.sh`：两卡可配置分片导出并在全部分片
  成功后自动合并；阈值和 shard 数作为命令行参数写入 manifest。

两帧 RTMDet-M smoke 已通过。已有旧缓存经新 merge/audit schema 测试：8,084 条记录、
2,021 个完整四视角组，sample 中相机零畸变和协议门禁均通过；原始点对带畸变 3D
投影误差约 `0.049 px`，原始到无畸变平均位移约 `1.236 px`。这些是几何/输入诊断，
不是新的 3D MPJPE。

首次完整验证导出以 `0.30` 阈值在第 2,564 条记录停止，检测器最高 person 分数
`0.247872`；没有使用 annotation box 兜底。随后已按统一 `0.20` 阈值重新启动完整
验证导出，结果必须以该线的 manifest 和几何门禁为准，不能与 `0.30` 失败线混合。

## 10. 用公开坐标级方法替代 GBT 未公开前端细节（2026-08-14）

### 10.1 结论

GBT 只给出“YOLOX box proposals”和 HRNet 输入结果，没有公开 detector 规格、权重、
阈值、NMS、bbox padding、codec 及 flip-test。因此这些项不能作为“严格复现”的判据。
检索有论文和代码的多视角方法后，最适合借鉴的不是另一个未说明实现的模型，而是
下面三条公开坐标级数据链路：

| 方法 | 会议/代码 | 输入 | 可直接借鉴的处理 | 适合程度 |
|---|---|---|---|---|
| PlaneSweepPose | CVPR 2021，官方代码 | HRNet 解码的 `(x,y,confidence)` + 标定相机 | 全图像素坐标、显式畸变、COCO→统一骨架映射、置信度作为可见性/视角权重；CMU 使用固定 3/6/12/13/23 相机 | **最直接** |
| Generalizable Human Pose Triangulation (GHT) | CVPR 2022 Oral，官方代码 | 2D 坐标 + K/R/t（不依赖热图） | 检测框坐标的逆变换：`coord = crop_coord * bbox_height / 384 + bbox_top_left`；相机参数与坐标同一像素系；随机视角/子集 | **坐标和 crop 对齐** |
| Self-Supervised MV Geometry | FG 2021 Oral，官方代码 | 2D 坐标 + 相机 | 先固定 2D 检测，再做几何/重投影训练；可作单人坐标管线的辅助交叉检查 | 辅助 |

PlaneSweepPose 的 Panoptic 代码读取的预测就是每个关节的 `[x,y,c]`，保持相机
`K,R,t,distCoef`，再做统一关节映射；其训练还会以 2D 扰动幅度标定 confidence。
这正是我们需要的“HRNet 只作为坐标/置信度前端”的公开依据。它训练阶段为多人身份
监督把预测坐标与 GT 投影坐标做了 `0.5/0.5` 混合，这一项只用于其多人深度监督，不能
复制到我们的单人公平比较中。

GHT 官方数据集代码则把 384 尺度 crop 坐标显式还原回原图：先乘 bbox 高度除以
`384`，再加 bbox 左上角；它没有把热图或图像特征送入三维网络。我们将用此检查
MMPose HRNet 解码结果是否已正确逆仿射，避免“训练框坐标/测试框坐标”或“crop 坐标/
全图坐标”混用。

### 10.2 我们采用的公开参考协议

固定三维网络和损失，只替换输入处理，按以下顺序导出并比较：

1. **HRNet-C0（当前官方线）**：官方 MMPose top-down HRNet，输出解码后的 COCO-17
   全图像素坐标与 peak confidence；不输出热图/feature。
2. **HRNet-C1（GHT inverse-crop audit）**：记录 bbox、center/scale、affine 矩阵和
   逆仿射结果；用一个合成点 round-trip 单测确认 crop→全图误差小于 `1e-4 px`。
3. **HRNet-C2（PlaneSweep distortion-aware）**：保留畸变坐标和 `k,p`，射线构造
   直接调用畸变模型；与当前“先去畸变、再令 `k,p=0`”的 C0 结果分开报告。
4. **HRNet-C3（PlaneSweep confidence）**：不重新人为截断有效关节，只把 detector
   缺失/越界置为不可见；保存原始 confidence 分布、每关节校准曲线和视角缺失率。
5. **HRNet-C4（协议控制）**：训练、验证、测试使用同一个检测器 checkpoint、同一
   阈值/NMS、同一 crop/codec/flip 选项；训练不使用 annotation box，测试不使用 GT
   兜底。C4 是最终可用于外部论文比较的主线。

所有 C0--C4 均只输入 `(x,y,c)`、相机参数和 RUMPL 原有输入，不加入 A1D、热图、
Volumetric 或图像特征。检测器仍需记录为一个明确的可复现实例（当前 YOLOX-X）；
不能声称它就是 GBT 未公开的具体 detector。

### 10.3 只做数据处理的最小实验矩阵

| 实验 | 变化 | 目的 |
|---|---|---|
| P0 | 现有 HRNet-C0 | 基线，确认现有结果可重跑 |
| P1 | C0 + GHT 逆 crop 单测/修正 | 排除 crop 坐标未还原造成的系统误差 |
| P2 | C0 + distortion-aware 射线 | 判断“去畸变/不去畸变”是否与内参匹配 |
| P3 | C0 + PlaneSweep confidence/visibility | 排除置信度阈值和不可见关节处理误差 |
| P4 | P1+P2+P3 | 公开代码依据下的完整坐标级前端 |

每个实验都用同一 RUMPL 三维网络、同一 H36M S1/S5/S6/S7/S8→S9/S11 协议和同一
V2/V3/V4 全组合评估；先报告 2D 像素误差、三角化控制和逐相机对，再训练三维网络。
这样若 P4 提升，收益可归因于输入处理；若不提升，才进入模型结构/训练协议，而不
再把 GBT 未公开的阈值当作猜测变量。

### 10.4 参考链接

- [PlaneSweepPose CVPR 2021 论文](https://openaccess.thecvf.com/content/CVPR2021/html/Lin_Multi-View_Multi-Person_3D_Pose_Estimation_With_Plane_Sweep_Stereo_CVPR_2021_paper.html) 与 [官方代码](https://github.com/jiahaoLjh/PlaneSweepPose)
- [Generalizable Human Pose Triangulation CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Bartol_Generalizable_Human_Pose_Triangulation_CVPR_2022_paper.html) 与 [官方代码](https://github.com/kristijanbartol/general-3d-humans)
- [Self-Supervised 3D Human Pose Estimation with Multiple-View Geometry 官方代码](https://github.com/vru2020/Pose_3D)

SelfPose3D/CVPR 2024 和 MVGFormer/CVPR 2024 虽然使用 HRNet/PoseResNet 相关前端，
但分别依赖热图或图像特征，不纳入本次“坐标+置信度公平输入”参考线。

## 11. 2D 前端审计实现（2026-08-14）

已在 exporter 中加入可审计的公开协议开关，但没有改变正在运行的旧 C0 导出进程：

- `--bbox-padding`：默认保持官方 MMPose `GetBBoxCenterScale(padding=1.25)`；可单独
  运行 GHT 风格的紧 bbox 对照（`1.0`），不覆盖 C0 缓存；
- `--no-flip-test`、`--no-shift-heatmap`：只用于诊断，主线保留官方 HRNet TTA；
- 每条预测保存 `input_center`、`input_scale`、`input_size` 和逆仿射闭环误差，确认
  MMPose 输出是整图像素坐标而不是 288×384 crop 坐标；
- 保留每关节原始 peak confidence，另存 visibility（若 MMPose 版本提供），不把
  detector score 乘入关节 confidence；PlaneSweep 的置信度是每关节误差/可见性量，
  不能用单个框分数替代；
- merge/audit 已支持两种互斥相机协议：`undistorted_K_equals_K` 和
  `original_distorted`，并检查相机、坐标和逆仿射误差是否一致。

对应代码：

- `export_h36m_gbt_aligned_hrnet_20260814.py`；
- `merge_h36m_gbt_aligned_hrnet_20260814.py`；
- `audit_gbt_aligned_hrnet_cache_20260814.py`。

下一步先在验证集导出 P1（`padding=1.0`）和 P2（保留畸变坐标）小矩阵，比较 2D
投影误差和 GT-2D 三角化；只有通过几何门禁后，才用同一个 RUMPL 三维网络训练，避免
把 detector、crop、畸变和三维模块的收益混在一起。

训练集 YOLOX-X `score_thr=0.01` 导出过程中，S5/action09/cam1/frame3071 暴露出
一帧真实的严重弯腰/遮挡漏检：官方后处理没有 person；将 detector 内部
`test_cfg.score_thr` 暂时设为 `0` 后，去畸变图像上出现约 `0.0091` 的 person 候选。
这不是用 GT 框兜底，而是单独记录为低分候选协议（外部阈值和 detector 内部阈值同时
固定、训练/验证一致），与主 `0.01` 线分开比较。若低分候选带来更多错误，宁可报告
漏检率并保留主线，不在最终结果中按帧选择最优框。

统一 `score_thr=0.20` 的 RTMDet-M 工程验证线已经完成（仅作为管线单元测试，
不是最终 GBT/YOLOX 论文线）：

- merged cache：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/rtmdet_m_score020_validation_v1/validation/merged/h36m_validation.pkl`；
- 8,084/8,084 条预测，4×2,021 相机记录，2,021 个完整四视角组，0 errors；
- 几何门禁：camera `k/p` 全为零、坐标系标记完整、全量 finite，均通过；
- 2D 诊断：HRNet 输出相对无畸变 3D 标注投影平均 `10.35 px`，中位数 `8.54 px`；
- 独立三角化控制（仅说明输入几何，不是训练结果）：confidence-IRLS 为
  V2/V3/V4=`90.13/54.81/49.48 mm`；GT-2D 无畸变控制为
  `约 11.96/8.10/7.43 mm`（其无畸变射线误差约 `0.47/0.30/0.27 mm`）。

这条线使用本机已有官方 RTMDet-M checkpoint，不能写成 GBT 的 YOLOX 严格复现，
也不能和旧 annotation-box train/full-image test 缓存混合。它的作用是先验证“统一的
全图去畸变→检测框→HRNet 坐标→零畸变射线”工程链路；最终论文对比必须使用固定的
YOLOX-X checkpoint、NMS 和阈值重新导出 train 与 validation，并复用同一 merge/audit
过程。

RTMDet-M 的训练集导出曾以 `score_thr=0.20` 启动，验证集完整通过，但训练集在
两帧 person 分数为 `0.0974/0.1115` 的位置漏检，已停止该不完整线，不能作为训练
缓存或论文结果。RTMDet-M 保留为“全图去畸变→检测框→HRNet→零畸变射线”的单元
测试；最终训练缓存改用下面固定的 YOLOX-X 线。

## 9. YOLOX-X 选择与启动记录（2026-08-14）

GBT 正文仅给出 “YOLOX box proposals”，没有给出规模、权重、阈值或 NMS；因此
不能把任意 YOLOX 结果写成 GBT 严格复现。为保持 detector 家族一致，同时选用官方
YOLOX 标准模型中 COCO AP 最高的规格，固定候选为：

- MMDetection config：`mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py`；
- 官方 checkpoint：
  `/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth`；
- SHA256：`1ef88d67f9c912a7c3a6df4f4d9bdf391cf70df867e6c9d7f249c7a3990e3dec`；
- MMDetection test NMS：IoU `0.65`；最终统一外部 person score threshold `0.01`，
  与该官方 MMDetection YOLOX 配置的 test score threshold 一致；
- 仍只取单人场景 person 类最高分框，训练、验证和后续测试完全相同。

该 checkpoint 已在本地环境通过 `init_detector` CPU 初始化，80 类和 test config
正常。两卡已完成验证集候选导出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score025_validation_v1/`；
8084/8084 记录、2021 个完整四视角组、0 errors，HRNet 相对无畸变 GT 投影平均
`10.306 px`、中位数 `8.516 px`，camera distortion/坐标系门禁通过。

`0.25` 试跑在训练集 S1 act04/cam3 暴露了 person score `0.088--0.119` 的连续帧，
因此已停止该不完整线，不能用于训练。最终统一阈值改为 `0.01`（训练/验证/测试
相同，不按帧兜底），并重新导出为
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/`；不能
使用 RTMDet-M 或 score025 的训练缓存替代。

## 12. 漏检帧保留政策（2026-08-14）

当前训练导出中已确认一条 YOLOX-X `score_thr=0.01` 下的 person 漏检记录
（S5/action09/subact01/cam1/frame3071）。训练集共有 312,188 条相机图像记录、
78,047 个四视角组；由于其余 shard 尚未完成，`1/312,188=0.00032%` 和
`1/78,047=0.00128%` 只能作为当前已确认的下限，不能当作最终漏检率。

处理决策：

1. 不删除训练或测试中的原始帧，优先保留完整帧序列，为后续时序模型准备；
2. 主论文协议保留固定官方阈值线，并完整统计漏检数量，不用 GT 框悄悄兜底；
3. 另行补跑固定的低阈值候选线（内部和外部阈值同时固定），保留低分框及其置信度，
   与主线做消融；
4. 若阈值降低仍无候选，优先实现显式缺失视角 mask/零置信度输入，而不是删帧；
5. 只有在低阈值、缺失视角处理和几何/时序模型均无法改善后，才做“删除完整四视角组”
   的独立敏感性实验。该结果不能替代主结果，也不能按测试误差逐帧筛选。

文献核对：GBT 没有公开漏检过滤规则；GHT 主要使用 H36M 提供的 GT 框，绕开了检测
漏检；PlaneSweepPose 代码跳过缺失图像或空标注，但不是删除检测器漏检的困难测试帧。
因此，当前不把“直接删漏检帧”当作论文主协议。

## 13. 当前验证缓存的处理闭环状态（2026-08-14）

`yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl` 的数值处理
链路已完成：原图全图 `cv2.undistort(K_new=K)`、同一无畸变图上的 YOLOX-X 检测与
HRNet-W32 top-down、HRNet 坐标回到原图像素、COCO-17→H36M-17 映射（含既有
lower-body swap），并将相机 `k/p` 置零、保留原始畸变参数到审计字段。8,084 条记录、
2,021 个完整四视角组通过几何门禁，零畸变/坐标系/finite/形状检查均无失败。

门禁数值：GT 三维投影到无畸变图与预测 2D 的误差为均值 `10.306 px`、中位数
`8.516 px`；原始坐标与带畸变投影的差异均值 `0.051 px`，说明相机畸变约定和射线
坐标是闭合的。直接用预测 2D 做 IRLS 三角化的控制值为 V2/V3/V4=`90.405/54.833/49.485`
mm，这不是 RUMPL 学习后的 MPJPE。

注意：该缓存的 manifest/protocol 标记仍为 `v1`，是在新增逐条 MMPose inverse-affine
roundtrip 审计字段前导出的；当前代码已对新导出加入这些字段，故最终训练缓存要在漏检
处理后按同一数值协议重新导出并运行 v2 audit。GBT 未公开 detector 具体变体、阈值、
padding 和 flip-test，因而只能称为“内部处理闭环且可审计的 GBT-aligned 近似”，不能
声称严格复刻其未公开前端。

## 14. 2D 处理对精度的阶段性结论（2026-08-14）

当前 GBT-aligned YOLOX-X/HRNet 验证缓存与此前验证前端相比没有可见精度收益：
`score=0.01` 与 `score=0.25` 的 8,084 条验证记录逐条相同，2D 误差均值/中位数为
`10.4649/8.6556 px`；预测 2D 的 IRLS 三角化为 V2/V3/V4=`90.405/54.833/49.485`
mm。与 RTMDet-M `score=0.20` 线的 `10.4587 px` 和 `90.126/54.806/49.475 mm`
基本持平，V2 反而高约 `0.28 mm`。

解释：本阶段的全图去畸变、相机 `k/p` 置零、坐标回变换和 COCO/H36M 映射主要是
协议/几何一致性修复，不会凭空提高 YOLOX 或 HRNet 的关键点精度。去畸变前后使用
一致的 `K` 后，几何变换大部分相互抵消；因此“数值没下降”并不说明处理错误，而是
说明它不是有效的 2D 质量增强模块。后续精度提升必须单独比较检测框质量、bbox padding/
HRNet 推理策略、置信度校准或三维融合器，不能把当前协议修复宣称为提升点。

## 15. 与公开论文前端的等价性边界（2026-08-14）

不能把当前输入处理写成“与所有顶会论文完全相同”。可确认的等价性分三层：

| 对象 | 公开内容 | 与当前线关系 | 可否称严格相同 |
|---|---|---|---|
| GBT | 去畸变、HRNet-W32-COCO、YOLOX 框、坐标级 3D 网络输入 | 明示部分一致；当前使用官方 YOLOX-X/MMDetection、HRNet 384×288、默认 padding/flip-test，并显式保存相机协议 | 否，论文未给 YOLOX 变体/权重/阈值/NMS/crop/flip/漏检规则 |
| GHT | 官方代码读取预计算 VPose3D 2D 检测，使用其相机归一化/反畸变和 H36M-17 坐标 | 输入是坐标级，但不是当前在线 YOLOX+HRNet 前端 | 否，不能直接当 detector-level 公平比较 |
| PlaneSweepPose | 官方代码读取预计算 Mask R-CNN/HRNet COCO 2D 结果，多人 unified joints、可见性/置信度 | 都使用坐标和置信度，但关节定义、多人关联和前端缓存不同 | 否 |

因此当前线的准确表述是：**GBT 明示细节对齐、其未公开细节显式固定、内部几何闭环通过的
coordinate-only 近似协议**。它适合与 GBT 做“输入类型和已公开处理一致”的对比，但不能声称
严格复现 GBT 的 detector 前端；与 GHT/PlaneSweep 则应采用同一 2D 缓存后再比较模型，或把
结果标成不同 detector protocol。

当前 H36M 训练/测试主体已经是标准协议：训练 `S1/S5/S6/S7/S8`，测试 `S9/S11`。仍未
完全确定、需要单独消融的输入变量包括 detector 框、bbox padding、HRNet flip/shift、
置信度定义、COCO→H36M 虚拟关节映射和 raw-distorted 与 full-image-undistorted 两种
实现的等价性。后续必须在验证集固定这些变量后再导出训练集，不能把它们混在 3D 网络收益中。

## 16. 可完整照搬的数据预处理候选（2026-08-14）

严格满足“单人 H36M + HRNet/YOLOX + 2D 坐标/置信度/相机、不给 3D 网络图像或热图”
的公开论文目前没有一篇能完全照搬：GBT 缺少 detector 前端代码和关键参数；其他公开
代码通常发布的是预计算 2D 检测，或使用不同 detector/多人关节定义。

最有价值的两个可复制基线：

1. **Learnable Triangulation (ICCV 2019)**：官方 H36M 预处理脚本完整公开了 bbox 来源
   （GT/MRCNN/SSD）、全图去畸变、`scale_bbox=1.0`、裁剪、`384×384` resize、相机
   内参随 crop/resize 更新和标准 S1/S5/S6/S7/S8→S9/S11 协议。它的网络输入是图像/热图，
   所以只能把这套前端作为“LT-style bbox/crop/相机控制线”，不能宣称与 GBT 坐标级输入
   相同。[官方代码](https://github.com/karfly/learnable-triangulation-pytorch)
2. **Generalizable Human Pose Triangulation (CVPR 2022)**：官方代码是坐标级，明确实现
   bbox 坐标恢复 `preds_2d *= bbox_height/384; preds_2d += bbox_top_left`，并提供 H36M
   预计算 2D 结果、bbox 和相机处理。但它没有完整的 2D detector 提取脚本；要称“完全
   照搬”必须使用其发布的 2D cache，而不能把我们的 YOLOX/HRNet 输出直接叫作 GHT 输入。
   [官方代码](https://github.com/kristijanbartol/general-3d-humans)

OpenMPL/T-CAP ECCV24 虽然给出 H36M+HRNet 的预处理脚本和 COCO→H36M 映射，但使用
MMPose inferencer，detector/crop 部分仍不如 LT 明确；PlaneSweepPose 使用预计算
Mask-RCNN/HRNet、多人与 unified joints，也不能作为当前单人 H36M 的完全同协议线。

因此后续采用双控制：保留当前 **GBT-aligned YOLOX/HRNet coordinate-only** 现实线，
另做 **LT-style 完整 bbox/crop/undistort 控制线**；两条线只替换 2D 前端，RUMPL 三维
网络、训练主体、视角组合和评估完全相同。这样才能判断“数据处理是否是瓶颈”，而不把
论文不同模型的热图/图像特征收益误算到我们的输入处理上。

## 17. LT-style HRNet 控制线首轮结果（2026-08-14）

已实现并完成验证集导出：

- 导出脚本：`export_h36m_lt_style_hrnet_20260814.py`；
- 合并脚本：`merge_h36m_lt_style_hrnet_20260814.py`；
- 启动脚本：`launch_lt_style_hrnet_export_20260814.sh`；
- 输出：`/mnt/data/cjyoutput/lt_style_hrnet_20260814_validation/validation/merged/h36m_validation.pkl`；
- 8,084/8,084 记录、2,021 个四视角组、0 errors；
- RUMPL 输入仍只有 HRNet COCO-17 坐标、HRNet peak confidence，转换为 H36M-17；
  没有写入 LT heatmap、ResNet 特征或 LT 三角化结果。

前端流程为：原图 `cv2.undistort(K_new=K)` → LT 官方 PIL crop（整数 bbox）→
`INTER_AREA` resize 到 384×384 → HRNet top-down → 坐标保留在 crop frame；相机 K
严格执行 LT `update_after_crop` → `update_after_resize`，并将去畸变后的 k/p 置零。
GT 2D 也在审计脚本中用同一变换映射到 crop frame，未直接把不同坐标系相减。

结果（所有相机组合、绝对 MPJPE、控制三角化，不是 RUMPL 学习结果）：

| 前端 | 2D 均值/中位数 (px) | V2 IRLS (mm) | V3 IRLS (mm) | V4 IRLS (mm) |
|---|---:|---:|---:|---:|
| GBT-aligned full-image HRNet | 10.358 / 8.535 | 90.405 | 54.833 | 49.485 |
| LT-style crop HRNet | 11.582 / 9.214 | 90.158 | 54.716 | 49.595 |

结论：在当前 HRNet-W32（384×288 codec）和现有 H36M `box` 下，LT crop 没有带来
可重复的观测质量提升；2D 反而约差 `1.22 px`，V3 仅改善 `0.12 mm`，V4 轻微退化
`0.11 mm`，属于噪声级别。相机/坐标变换本身通过了几何等价审计（GT 点投影到 crop
K 的误差约 `0.16 px`），所以不能把结果解释为相机更新错误。

重要协议边界：本地 RUMPL PKL 的 `box` 是 H36M-Toolbox 的方形 center/scale 框；LT
官方仓库默认读取其 `human36m-multiview-labels-GTbboxes.npy` 中由 segmentation mask
生成的 GT bbox。当前机器没有该 LT segmentation bbox 文件，因此这轮是“LT crop/camera
语义严格、bbox 数值采用现有正式 PKL”的控制上限，不能写成 LT 官方 bbox 的严格复现。
若要继续，先补齐 LT 官方 bbox 文件再做一次；在当前结果下不启动昂贵的 LT-style 全量
RUMPL 训练，避免把没有前端收益的线误报为模型改进。

## 18. GBT-style HRNet coordinate training gate（2026-08-14）

在决定回到 GBT 风格全图输入后，重新检查了 YOLOX-X 训练缓存。训练集 312,188 条
相机记录中，YOLOX 在一帧严重弯腰姿态上没有输出 COCO `person` 类；这不是坐标/相机
变换错误，而是 detector coverage failure。该记录的图像、检测输出和原始 H36M 框均已
保存为审计证据。为避免静默丢帧，新增显式 `--fallback-record-box`：仅在无 person
proposal 时使用该记录原始 H36M 框，写入 `fallbacks` manifest 和
`source_2d_detector_fallback` 字段。该线应称为 **GBT-style YOLOX + explicit
record-box fallback**，不能称为无例外的严格 YOLOX-only 复现。

为缩短等待，将原 shard 2 拆为 `shard2a`/`shard2b`（16-way partition）在两张卡上
并行重导出；完成后自动合并训练缓存，并在完全相同的缓存上启动坐标级 RUMPL R0 与
H76。启动脚本为 `launch_gbt_aligned_rumpl_training_20260814.sh`，输出位于
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/rumpl_training/`。这两组结果只读取
HRNet 解码坐标、peak confidence 和相机射线，不读取热图、A1D 或 LT 特征。

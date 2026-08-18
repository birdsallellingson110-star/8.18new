# 坐标级双前端协议、当前结果与目标（2026-08-13）

## 1. 固定任务定义

本文只记录坐标级方法：三维网络仅可读取二维关节坐标、关节置信度和相机参数，
不得读取 RGB、HRNet 热图/中间特征或 Volumetric 图像特征。A1D 属于热图级输入
修正，不进入本文件的坐标级主表。

共同评估口径：H36M S1/S5/S6/S7/S8 训练，S9/S11 测试；删除公认损坏序列；
absolute MPJPE、All-17、无对齐；V2 平均全部 6 个相机对，V3 平均全部 4 个
三相机组合，V4 为四相机组合。action-equal 为主指标，同时保留 frame-weighted。

## 2. 两条前端协议必须分表

### C-HRNet：OpenMPL/RUMPL 公开可复现 HRNet 坐标协议（主协议）

HRNet 主协议不再以无代码的 GBT 为实现依据，而以与本项目任务和输入完全匹配、
且公开了 H36M 预处理代码的 OpenMPL/RUMPL 为依据：整张 H36M 图像输入
`MMPoseInferencer`，由 MMPose 默认 person detector（在已冻结的 MMPose 1.3.2
环境中为 RTMDet-M COCO-person）产生框，再运行 COCO 预训练 HRNet-W32
384x288，保存二维坐标和 detector score。随后严格使用公开 COCO-17 到 H36M-17
映射，并对二维坐标和相机内参做一致的屏幕归一化。

必须固定并记录 MMPose/MMDetection 版本、RTMDet/HRNet config、checkpoint SHA、
默认 `bbox_thr=0.3`、`nms_thr=0.3`、多人时的选择规则、COCO-H36M 映射和验证帧
清单。所有坐标级方法共享同一缓存，不允许每个方法重新检测。

GBT 仅作为外部目标：其公开文字说明为图像去畸变、COCO HRNet-W32、YOLOX
proposal、T=9、随机两视角训练并测试所有视角组合；但未公开 YOLOX 型号、阈值、
box padding、HRNet checkpoint/codec 和代码，故不能拿猜测实现替代公开主协议，也
不能标为严格复现。

当前 annotation-box raw HRNet 诊断不是该协议，不能进入正式 HRNet 主表：

| 当前临时诊断 | V2 | V3 | V4 |
|---|---:|---:|---:|
| raw HRNet + IRLS | 91.913 | 55.684 | 50.583 |
| raw HRNet + RUMPL H15 | 91.521 | 72.443 | 63.954 |
| raw HRNet + RUMPL H15 + bias | 90.523 | 70.834 | 62.699 |

这些结果使用 annotation box，且 H15 不是 H76。它们与公开 OpenMPL 的整图
Inferencer 检测链不同。此前冻结的公开 RUMPL H36M 复现已经证明这一差异是实质性
的：整图 RTMDet-M + HRNet-W32 缓存上，RUMPL 2-view 得到 51.874 mm，论文为
52.500 mm。当前尚无“该公开 HRNet 缓存 + 真实 H36M 全训练 subjects + H76”正式
结果，必须在冻结缓存上从头训练。

### C-LT：LT/ResNet-152 坐标协议

使用官方 Learnable Triangulation Algebraic 网络及严格加载的官方权重；二维主干为
ResNet-152（COCO 预训练后在 MPII/H36M 联合微调），图像去畸变，当前使用 H36M
annotation box 和官方 384x384 变换。由于尚缺作者 segment-GT-bbox 文件，准确
名称为“官方权重 + 受控 annotation-box 协议”。

| 当前坐标级方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 官方 LT Algebraic | 53.592 | 24.339 | 19.921 |
| **H76 only，epoch 20** | **43.442** | **24.425** | **21.234** |

H76 only 不含 Algebraic 3D 输出融合、Volumetric、A1D、图像特征或时序；它只使用
LT 输出的二维坐标/跨视角置信度和相机参数。

## 3. 目标线（来自 GBT 表 I，仅作目标）

| GBT 论文方法 | T | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| Ours (ResNet-152) dagger | 9 | 29.9 | 24.4 | 22.7 |
| Ours (HRNet) | 9 | 36.8 | 30.4 | 26.0 |

这两行来自无代码论文，作为目标而非可复现基线。其 T=9，不能与本项目 T=1
结果隐去时间窗口后直接宣称完全公平。项目目标为：在各自固定前端协议下，单一
坐标级模型达到或超过相应行，并同时报告 T=1/T=9 消融。

当前 C-LT H76 相对目标的差值为 V2 `+13.542`、V3 `+0.025`、V4 `-1.466` mm；
即 V3 基本持平、V4 已超过目标，主要缺口集中在 V2。

## 4. LT 下 V2 退化的实证原因

H76 epoch 20 的逐相机对 action-equal MPJPE：

| 相机对 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|
| LT + H76 | 29.116 | 33.528 | 68.885 | 74.785 | 26.176 | 28.162 |
| 旧 A1D/H21 + H76 | 31.199 | 32.302 | 42.435 | 41.642 | 30.523 | 30.797 |

LT 在四个正常几何相机对中的三个反而更好；平均退化几乎完全来自 `1-4` 和
`2-3` 两个退化相机对。旧 A1D V2 缓存在修正二维点时读取过全部四个视角，随后才
在 RUMPL 端删为两个视角，因此对这两个坏对使用了未选视角的信息，不能作为公平
V2 基线。两视角也没有第三个观测供 LT learned confidence 抑制离群点；到了 V3/V4，
额外射线和跨视角置信度能够压制坏观测，故误差大幅下降。

因此不能得出“更好的检测器让 V2 变差”。正确结论是：LT 坐标更准确，但当前
H76 对几何退化相机对仍不稳；旧 34.816 mm 又包含不合法的四视角热图辅助。

## 5. HRNet 输入论文及公开代码审计

| 方法 | 3D 阶段实际输入 | HRNet/框与坐标处理 | 是否可直接作为本项目 HRNet 坐标协议 |
|---|---|---|---|
| OpenMPL/RUMPL | 2D 坐标、置信度、相机 | MMPose HRNet-W32 COCO 384x288；整图 Inferencer 的默认 detector 框；公开 COCO→H36M 映射；二维与内参一致归一化；公开代码未启用去畸变段 | **是，主依据** |
| GBT（FG 2024） | 2D 坐标、置信度、射线，T=9 | 先去畸变；HRNet-W32 COCO；YOLOX 框；无代码且未给完整 detector/codec 参数 | 仅作目标和补充协议，不能严格复现 |
| UPose3D（ECCV 2024） | 2D 坐标及 RLE 不确定性、相机，含时序 | H36M 主表实际用 CPN/ResNet-152 384x384 并在 H36M+MPII 微调；HRNet-W48 384x288 只用于 RICH/CMU/HUMBI 的 OoD；用 reference-view subject bbox 归一化跨视角点云 | 不可作为 H36M-HRNet 主协议；可借鉴 covariance/RLE 与 bbox 归一化消融 |
| Generalizable Human Pose Triangulation（CVPR 2022） | 2D 坐标、置信度、相机 | 使用 Learnable Triangulation ResNet-152 预测和 GT bbox；公开代码把 crop 坐标还原至整图坐标 | 不是 HRNet 前端；可借鉴多假设评分器 |
| Cross View Fusion / Epipolar Transformer / AdaFuse | 热图或图像中间特征 | 通常用 GT bbox，并在热图/特征层跨视角融合 | **否**，只能放热图级表，不能与 raw-coordinate HRNet 混表 |
| UPose3D 等论文表中的“HRNet”行 | 依论文而异 | 有些 HRNet 只用于 OoD、检测基线或图像分支，不能因表格写 HRNet 就视为同一输入协议 | 必须逐项审计，不能直接并入 |

因此，经论文与代码筛选后，当前最可靠且最接近我们输入形式的公开数据处理实现
就是 OpenMPL/RUMPL，而不是 GBT。并不存在一批可直接照抄、同时满足“HRNet
坐标+置信度+相机、H36M absolute、多视角、公开代码”的顶会方法；大多数 HRNet
方法融合的是热图/图像特征，或报告的是 root-relative/单目/OoD 结果。

## 6. 执行顺序

1. 冻结 `C-HRNet-OpenMPL`：直接复用已核验的整图 RTMDet-M + COCO HRNet-W32
   384x288 检测链；补齐正式训练 subjects S1/S5/S6/S7/S8，并保存全部 config、
   checkpoint SHA、阈值、codec、选择规则和索引。
2. 在该缓存上零训练评估 DLT、confidence-DLT、IRLS/RANSAC，并逐相机对核验；
   同时报告 2D pixel error 与跨视角重投影误差，先确认输入质量再启动模型训练。
3. 在完全相同缓存上从头训练真实 H36M RUMPL baseline、H76 及后续模块；A1D
   禁止进入 raw-coordinate 主表。
4. 只在需要回答“GBT 输入差异”时另做 `C-HRNet-GBT-aligned` 控制：去畸变 +
   明确固定的 YOLOX + 同一 HRNet。它与主协议分表，不能混用输入缓存。
5. C-LT 保持现有 H76 43.442/24.425/21.234 为固定基线。优先针对退化相机对做
   有公开依据的 GHT 多假设评分、几何条件化残差/关节效用；禁止融合图像分支。
6. 单帧结构稳定后再统一加入 T=9，分别报告 T=1 与 T=9，最终对照 GBT 目标线。

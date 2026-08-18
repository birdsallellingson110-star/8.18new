# AdaFuse 超越目标：2026-08-14 进度与下一步

## 目标口径

AdaFuse 官方仓库的 H36M README 报告四视角结果：NoFuse 22.94、HeuristicFuse 21.02、ScoreFuse 20.14、RANSAC 21.77、AdaFuse 19.54 mm。其公开测试代码在 H36M 上先把预测骨盆平移到 GT 骨盆，再计算 MPJPE；因此本项目从本文件起始同时保留 absolute MPJPE 和 root-aligned MPJPE，不能混报。

参考：

- 论文/仓库：<https://arxiv.org/abs/2010.13302>、<https://github.com/zhezh/adafuse-3d-human-pose>
- 官方实现：`/home/lixiaob/cjy/reference/adafuse-official`

## 已完成的可审计工作

1. 复用当前真实 H36M S9/S11、4 个固定同步视角、HRNet-W32 COCO 冻结热图，严格按视角子集重新枚举 V2（12126）、V3（8084）、V4（2021）组合。热图没有先看四视角再删视角的泄漏。
2. 全热图极线支持与训练式 A1D 的严格热图控制已经完成：HRNet 当前热图坐标下，A1D-line action-equal absolute MPJPE 为 V2/V3/V4 = 87.685/52.631/46.896 mm；固定极线规则几乎没有收益。这个结果是融合诊断，不是 AdaFuse 复现。
3. 方形 384×384、96×96 的 ResNet-152 公共 COCO 权重已导出全验证热图（4×604 MB），并检查了解码一致性。该权重不是 AdaFuse 的 H36M 微调权重，因此只能称为 public ResNet-152 front-end control。
4. 评估器新增 `--solver dlt`，实现 AdaFuse 公开代码中的无权 DLT 三角化；`--solver robust_ray` 保留 RUMPL 鲁棒射线求解。两个 solver 的差异已单独记录。

## 当前方形 ResNet-152 + DLT 全量结果

文件：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Res152_square_val_20260814/eval_dlt/`

| 输入/求解器 | V2 absolute | V3 absolute | V4 absolute | V4 root-aligned |
|---|---:|---:|---:|---:|
| Res152 square top-1 + DLT | 126.620 | 62.206 | 58.033 | 68.490 |
| Res152 square dense-PoE α=1 + DLT | 99.880 | 60.119 | 55.725 | 66.969 |

这远高于 AdaFuse 的 19.54，不能宣称已超过，也不能归因于“融合器不够深”。主要原因是：当前 ResNet 权重仍为 COCO 预训练，只改了采样形状；且当前评估数据经过 RUMPL 的 COCO→H36M 映射，和官方 H36M 微调前端并不相同。

## 关键诊断

- 当前 HRNet heatmap 的预测相对本项目 H36M 记录平均约 10.35 px；ResNet square 约 10.60 px，二者都不是官方 AdaFuse H36M 2D backbone。
- 官方 AdaFuse 2D 配置是 mixed(H36M+MPII) 训练、ResNet-152、384×384/96×96、GT bbox；官方 3D 融合配置使用已在 H36M 微调的 `h36m_4view_d87025.pth.tar`。这两个官方权重本地没有，SharePoint 下载不可用，故目前不能写“严格 AdaFuse 复现”。
- DLT 对低基线/异常 2D 尤其是 V2 很敏感；因此后续主表必须明确 solver，先统一“官方 DLT 对照”和“RUMPL robust-ray 主线”，不把 solver 差异写成模块收益。

## 下一步实验顺序

1. 已完成 DLT 全量控制；接下来将修复/准备 AdaFuse-style 的 H36M 2D 训练数据接口，优先使用本地真实 H36M 训练图像和标注，训练 square ResNet-152（先 2D-only，小规模 smoke，再全量 10 epoch）。
2. 同一前端导出 train/validation 热图后，先跑官方式 NoFuse、ScoreFuse、DLT，再训练/评估 AdaFuse-style view-weight + epipolar warp；所有结果同时输出 absolute/root-aligned。
3. 若官方口径仍未到 20 mm，再将最有效的 A1D/RUMPL 几何残差作为独立模块接入，保持 `HRNet/ResNet front-end → heatmap fusion → DLT` 的有机替换关系，不堆叠三个完整模型。
4. 只有在同口径 4-view 接近或低于 19.54 后，才扩展 V2/V3、随机视角和跨相机泛化表；否则先定位 2D 前端与相机/畸变协议，不调时序。

## 运行产物

- 评估器：`eval_h36m_dense_epipolar_heatmaps.py`
- 方形配置：`configs/td-hm_res152_8xb32-210e_coco-384x384.py`
- 方形热图：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Res152_square_val_20260814/`
- DLT 结果：`.../Res152_square_val_20260814/eval_dlt/v2.json`、`v34.json`

## 2026-08-14 LT-H36M 前端控制（已完成）

为避免把“未在 H36M 上训练的 COCO 前端”误当成融合器结论，补充了
Learnable Triangulation 官方公开的 H36M 2D 权重。该权重在其 README 中明确
说明为 ResNet-152，先 COCO 预训练，再联合 MPII/H36M 微调；它不是 AdaFuse
私有权重，因此本实验只标记为 **LT-H36M frontend control**，不称为严格
AdaFuse 复现。

- 来源：Learnable Triangulation 官方仓库的公开 Yandex 文件
  `pose_resnet_4.5_pixels_human36m.pth`；文件大小 275,353,380 bytes。
- SHA256：`2fbab6c5a6b220fc10e1f945570a1b826bb1c217929bb5cdb44c1e99f04e74f2`。
- 转换：保留官方 ResNet-152 / 384x384 / 96x96 结构，将 `module.*` 命名空间
  转为 MMPose 命名；官方 checkpoint 的 final layer 为 33 通道，按其 H36M
  17-joint 配置保留前 17 通道，并在独立脚本中记录该转换。
- 验证：16 组（64 张）smoke 和完整 S9/S11 四个 shard 均输出
  `[17,96,96]` 热图；手工峰值解码相对 MMPose 的平均误差约 `3e-5 px`。
- 评估：使用与 COCO-ResNet/HRNet 完全相同的 H36M PKL、视角组合
  （V2=12,126、V3=8,084、V4=2,021）、absolute/root-relative 双口径，
  并分别跑 RUMPL robust-ray 与 AdaFuse 同构 unweighted DLT。LT 官方前端
  的 robust-ray 结果为 V2/V3/V4 = 50.21/30.24/28.06 mm（root-relative，
  action-equal）；它验证了前端和相机协议，但还没有达到 AdaFuse 的 19.54 mm。
- 已修正 LT 的专有输出排列：`lt_h36m` 映射为
  `[6,3,4,5,2,1,0,7,8,16,9,13,14,15,12,11,10]`，未经映射的旧评估会把
  下肢和躯干通道混用，不能使用。
- 在修正通道后，AdaFuse 风格全热图极线支持（V4、DLT、2-sample line）结果：
  原始 DLT root-relative frame-weighted 约 27.78 mm（action-equal 约
  27.53 mm）；PoE/几何支持的最佳 `alpha=2` 为约 26.01 mm
  （action-equal 约 25.85 mm，absolute action-equal 约 24.01 mm）。这是
  training-free control，说明跨视角热图支持有效（约 1.9 mm），但仍距
  19.54 mm 约 6.3 mm，不能把固定规则写成 AdaFuse 的学习权重。
- 输出：`LT_H36M_official_full_20260814/learned/result.json`、
  `LT_H36M_noflip_bgr_val_20260814/eval_ada_controls/lt_v4_dlt_line.json`。

### 已废弃的 LT 评估

`LT_H36M_frontend_val_20260814/eval_robust.json`、`eval_dlt.json` 以及
其中的 smoke 文件使用了 COCO 默认翻转/通道解释，数值异常（数千 mm）；
这些文件仅保留作审计记录，不进入任何结果表。

## 2026-08-14 AdaFuse 2D 训练线（进行中）

- 官方 `pose2d/h36m.yaml` 的关键设置已落实：真实 H36M 训练 subjects
  S1/S5/S6/S7/S8、官方 `[::20]` multiview group 采样、384×384 输入、
  96×96 sigma=3 高斯热图、scale=0.15、rotation=20、H36M 17 通道、
  COCO ResNet-152 初始化、末层重置、20-joint union head（缺失的 3 个
  union joint 权重为 0）、官方逐关节 MSE、Adam 无 weight decay、epoch 8
  学习率衰减；3D 融合之外只训练 2D 前端。
- 输入顺序按官方 OpenCV + `ToTensor` 处理保留 BGR；RGB 仅作为后续对照，
  不混入主结果。仿射变换已经与官方 `get_affine_transform` 逐项核对，
  四个旋转角最大矩阵差为 0。
- 当前全量训练：
  `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Ada2D_H36M_union20_train_final_20260814/`
  （3903 groups、15612 images、10 epochs、双 RTX 4090、batch=16）。此前
  `Ada2D_H36M_bgr_train_r2_20260814` 使用连续 17 通道且未设官方学习率
  调度，已停止并标记为流程验证，不进入主结果。
  训练结束后会用同一导出器重新生成验证热图，再跑 NoFuse/ScoreFuse/line-DLT
  和可学习 view-weight，避免把检测器收益和融合器收益混在一起。

新增可复现实验脚本：

- `convert_lt_h36m_resnet_to_mmpose_20260814.py`
- `launch_lt_h36m_frontend_val_20260814.sh`
- `train_h36m_res152_heatmap_ada_protocol_20260814.py`（真实 H36M 2D-only
  微调 smoke 已通过；后续用于完整 H36M 2D 训练，不使用 3D 标签）。

## 2026-08-14 官方 AdaFuse 核心严格控制（已完成）

在同一份 Ada2D H36M-only ResNet-152 前端上，补齐了公开 AdaFuse 的关键实现
细节：`official_line` 最近邻整条极线采样、投影 GT heatmap 监督、官方无放回
multiview group sampler、DLT 求解器，并比较非负热图与保留原始符号热图。每个
版本用两个 seed 训练 3904 steps，正式结果均使用 **S9/S11 validation pkl**
和四视角完整组（2021 groups）；训练 pkl 只用于训练，不能用于结果表。

| 版本 | V4 absolute action-equal | V4 root-relative action-equal | 备注 |
|---|---:|---:|---|
| Ada2D + official AdaFuse，非负热图，seed 0/1 | 24.66/24.64 | 27.59/27.54 | 严格控制最佳约 27.56 |
| Ada2D + official AdaFuse，保留符号，seed 0/1 | 24.66/24.69 | 27.54/27.60 | 与非负版基本相同 |

因此“有符号热图”没有解决泛化差距；先前训练集上的约 15 mm 是路径误指向
训练 pkl 造成的数据泄漏，已明确排除，不进入任何结论。当前可复现实验表明：
公开 AdaFuse view-weight 核心在我们的 H36M-only Ada2D 前端上能把 V4
absolute 从约 28.73 降到约 24.65 mm，但距离论文 19.54 mm 仍约 5.1 mm。
下一主线应转向与论文更接近的 2D 前端/训练协议，而不是继续微调同一个
融合器；后续所有实验必须继续使用独立 validation pkl。

## 2026-08-14 LT 前端 + 官方 AdaFuse view-weight（V4 已完成）

为验证上述判断，使用 Learnable Triangulation 官方公开的
`pose_resnet_4.5_pixels_human36m.pth` H36M 前端，导出同一训练 stride-20
和 S9/S11 validation 热图；只训练公开 AdaFuse ViewWeightNet，融合器、线采样、
投影 GT 目标、DLT 和评估协议均不变。LT 的 17 通道先重排到 RUMPL H36M 顺序，
评估结束再映回原顺序。

| 前端 + 融合 | V4 absolute action-equal | V4 root-relative action-equal |
|---|---:|---:|
| LT top-1 DLT | 25.946 | 27.530 |
| LT + 官方 AdaFuse view-weight，seed 0 | 22.994 | 25.377 |
| LT + 官方 AdaFuse view-weight，seed 1 | 22.944 | 25.273 |

相对于同一 LT 前端 top-1，学习视角权重约带来 **2.2 mm root-relative** 收益；
相对于 Ada2D H36M-only + AdaFuse 的约 27.56 mm，LT 前端进一步降低约 **2.3 mm**。
这证明前端和融合器两个变量都有效，但当前仍比 AdaFuse 论文 19.54 mm 高约
5.7 mm，不能声称已经复现/超过作者结果。LT validation 的 decoded 2D 误差
约 4.82 px（对投影 GT），仅比 Ada2D 的约 5.25 px 好约 0.43 px，因此剩余
差距不应简单归结为单纯 2D 像素误差，需继续核对官方训练权重/融合实现及训练
协议。

新增输出：

- LT 训练热图：`LT_H36M_noflip_bgr_train_stride20_20260814/`
- LT+AdaFuse 权重：`OfficialAdaFuse_LT_H36M_officialloader_seed{0,1}_20260814/`
- V4 验证 JSON：各目录下 `eval_val_v4/eval_official_loader_dlt.json`

随后完成同一模型的全组合 V2/V3 验证（V2=12,126、V3=8,084 组合）。两 seed 的
action-equal root-relative MPJPE 如下：

| 视角数 | LT top-1 | LT + AdaFuse seed 0 | seed 1 |
|---|---:|---:|---:|
| V2 | 51.084 | 49.929 | 49.681 |
| V3 | 29.983 | 29.133 | 28.876 |
| V4 | 27.530 | 25.377 | 25.273 |

V2→V3→V4 保持单调下降，且 view-weight 在三个视角数上均有收益；这条线可以
作为可靠的“前端替换 + 有代码依据的 AdaFuse 融合”对照，但尚未达到论文目标。

### 官方优化器对照

按官方 `h36m_4view.yaml` 加入 `weight_decay=0.001` 后，LT+AdaFuse V4
root-relative action-equal 为 seed0/1=`25.253/25.246 mm`，相对无 weight
decay 的 `25.377/25.273 mm` 仅改善约 `0.02--0.12 mm`。该旋钮不作为新的
论文贡献，后续不再围绕它重复训练。

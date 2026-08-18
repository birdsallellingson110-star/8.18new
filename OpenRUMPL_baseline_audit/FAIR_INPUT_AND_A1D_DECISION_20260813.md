# H36M 公平输入与 A1D 使用决策（2026-08-13）

## 1. 训练主体纠正

此前仅查看 `H76_train_all_subsets_shard0of2.npz`，看到其中包含
S1/S5/S6，误判为训练主体不完整。实际数据如下：

| 数据 | 数量 | 主体 |
|---|---:|---|
| 原始 H36M train PKL | 312,188 条相机记录 | S1/S5/S6/S7/S8 |
| 完整四视角同步组 | 78,047 组 | S1/S5/S6/S7/S8 |
| H76 cache shard 0 | 39,023 组 | S1/S5/S6（按顺序切片） |
| H76 cache shard 1 | 39,024 组 | S6/S7/S8（按顺序切片） |
| 两个 cache 合并 | 78,047 组 | S1/S5/S6/S7/S8 |
| 新输入 20k 均衡子集 | 20,000 组 | 4005/4005/4005/3995/3990 |

因此基础 H76 和新输入试验均没有漏掉 S7/S8，不需要重新下载、补生成或重训来
“补主体”。标准测试仍为 S9/S11。

## 2. “都是 HRNet”为什么仍不一定公平

HRNet 只是二维主干名称；论文实际输入至少分为四档：

| 输入档位 | 模型实际可见信息 | 典型方法 | 应如何比较 |
|---|---|---|---|
| C：坐标级 | 2D 坐标、置信度、相机参数 | RUMPL、GBT、DLT/RANSAC、GHT | 同一冻结检测缓存、同一框 |
| H：热图级 | HRNet 每关节完整热图 | AdaFuse、CVF、A1D | 同一 HRNet 权重/热图/框 |
| F：特征级 | HRNet 中间图像特征、热图 | Epipolar Transformer | 同一 RGB、框、主干权重 |
| V：图像/体积级 | 多视图图像特征和三维体积 | LT Volumetric、部分 MVGFormer | 单独图像方法子表 |

即使两篇文章都写“HRNet-W32”，以下差异仍会显著影响 MPJPE：

- COCO-only 还是在 MPII/H36M 微调；
- YOLOX 检测框还是 H36M annotation/GT 框；
- 输入 256x192、384x288 还是 384x384；
- 是否去畸变、flip-test、UDP；
- 只使用 argmax 2D 点，还是还读取完整热图/中间特征；
- 单帧还是 9 帧。

## 3. 当前输入与公开论文协议

当前冻结前端是：MMPose HRNet-W32、COCO 权重、384x288、96x72 热图、非 UDP、
flip-test、H36M annotation box。坐标映射与 MMPose MSRA codec 的最大差仅
`8.6e-6` feature pixel。

| 方法 | 论文输入 | 与当前是否同输入 |
|---|---|---|
| GBT | HRNet-W32 COCO + YOLOX；2D pose/confidence/rays；T=9 | 仅主干名称相同；框、时序、信息层级不同 |
| AdaFuse | ResNet-152，MPII/H36M增强，GT bbox，完整热图 | 不同2D主干；但属于与A1D相同的热图级类别 |
| LT Algebraic/Volumetric | ResNet-152，COCO→MPII/H36M微调，GT bbox，去畸变 | 不同主干及训练数据；放作者协议表 |
| Epipolar Transformer | ResNet中间特征的极线融合 | 与当前新极线试验同属特征级，但骨干/指标协议不同 |
| raw RUMPL | 2D坐标、置信度、相机射线 | 可在统一raw HRNet坐标表中直接消融 |

GBT没有官方代码，因此其公开数字可作参考，但不作为“完全公平、可核验”的首要
基准。优先使用 LT、AdaFuse、ET、GHT 等有官方代码的方法建立受控表。

## 4. A1D 是否可以使用

可以，但必须把它写成方法模块，而不能把 A1D 后的点称为 `raw HRNet input`。

A1D 的定位：

```text
冻结 HRNet 完整热图
  -> 相机标定定义的极线支持
  -> 共享的 dense residual fusion
  -> 修正后的 2D 热图/坐标
  -> RUMPL 射线主干
```

它有 AdaFuse/CVF/ET 的公开方法依据，且不需要相机 ID；因此可以作为“热图级几何
输入修正 + RUMPL 射线回归”的融合模块。公平的贡献证明应为：

1. raw HRNet + 同一 RUMPL；
2. raw HRNet heatmap + A1D + 同一 RUMPL；
3. AdaFuse-style/CVF-style + 同一 RUMPL（统一热图对照）；
4. A1D + H21 + H76；
5. 上一步 + E2（仅在前面独立有效后）。

## 5. 旧 A1D 结果的协议问题

`train_dense_geometry_residual_fusion.py` 训练时会随机采样 2/3/4 视角，训练设计本身
支持变视角。问题在 `export_a1d_refined_mmpose_pkl.py`：

- 每个同步组固定读取四个视角；
- 调用 `a1d_corrected_coco()` 时四视角共同生成所有修正点；
- 再把这套四视角修正永久写入一个 PKL；
- 后续所谓 V2/V3 只是在 RUMPL 端减少射线，输入点此前已经看过未选视角。

所以旧 A1D/H21/H76 的 34.816/30.489/29.691 及其后续 RIGR/E2结果仍有工程和
方法探索价值，但 V2/V3 不能进入严格外部公平主表。V4不存在未选视角泄漏，但仍属于
`enhanced HRNet heatmap`，不能和只用raw坐标的方法伪装成相同输入。

旧 A1D 单独接几何三角化的结果为 87.705/54.222/48.085 mm，并没有证明 A1D 单独
有效；后续好结果来自 A1D、H21、H76 的联合训练分布。新实验必须逐级拆开消融，不能
继续将整个链条的收益全部归给A1D。

## 6. 正确重做方案

### 表 B-C：统一raw坐标输入

固定完整五主体、annotation box、COCO-only HRNet-W32、相同原始2D/置信度：

- DLT / confidence DLT / RANSAC；
- raw RUMPL；
- raw RUMPL + bias；
- GHT/E2候选效用。

这张表不使用A1D，以证明3D融合模块贡献。

### 表 B-H：统一HRNet热图输入

仍固定同一HRNet权重、框、帧和热图：

- NoFuse；
- AdaFuse-style；
- A1D；
- A1D + RUMPL/H76。

这张表允许A1D，因为所有被比较方法都获得相同完整热图信息。

### A1D实现修正

不能再把唯一四视角修正烘焙成PKL。应预计算11个相机子集各自的A1D 2D/射线缓存，
训练和评估时按实际采样的组合读取对应缓存。必须验证：

- V2的任意组合只读取两张热图；
- V3只读取三张热图；
- 视角顺序置换前后结果一致；
- raw identity对照逐元素一致；
- V2平均6组合、V3平均4组合、V4唯一组合；
- S7/S8包含在训练缓存中。

### 检测框子表

annotation box适合与LT/AdaFuse受控版本比较。若要对照GBT公开HRNet结果，另导出
YOLOX box + 同一HRNet-W32 COCO缓存；不把两种框的数字混在同一主表。


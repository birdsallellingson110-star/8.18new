# 超过 AdaFuse 的实验主线与记录（2026-08-13）

## 0. 目标定义

当前阶段的目标是：在不混用检测器、人体框、输入信息层级和指标协议的前提下，
逐步超过 AdaFuse 的公开结果。AdaFuse 官方仓库 README 的同一模型消融为：

| 官方输出 | H36M MPJPE（mm） |
|---|---:|
| NoFuse | 22.94 |
| HeuristicFuse | 21.02 |
| ScoreFuse | 20.14 |
| AdaFuse | **19.54** |

这 19.54 mm 是作者协议目标，不是当前 COCO-only HRNet 坐标级输入可以直接承诺的
数字。当前论文主线必须同时维护两张表：

1. **统一输入受控表**：相同 H36M 帧、框、HRNet 权重、热图/坐标和绝对 MPJPE，
   用来证明模块增益；
2. **作者协议表**：官方数据处理、主干和权重，只有协议相同时才与 19.54 mm
   做直接比较。

## 1. 已知基线和差距

| 线 | V2 | V3 | V4 | 输入/状态 |
|---|---:|---:|---:|---|
| RUMPL raw HRNet（严格重评） | 84.235 | 64.868 | 55.460 | COCO-only HRNet-W32，坐标+置信度+射线；新训练权重 |
| H76 enhanced（严格重评） | 52.281 | 37.348 | 34.016 | 当前 raw-HRNet 训练的锚点+居中射线+Plücker 对照 |
| H76 enhanced（历史统一输入） | 34.816 | 30.489 | 29.691 | A1D/H21 enhanced 2D，不能作为 raw 外部表 |
| RIGR→E2 specialist（历史） | — | 28.555 | 27.808 | enhanced 输入，V3/V4 专用头 |
| AdaFuse 官方公开值 | — | — | **19.54** | ResNet-152、384×384、完整热图、MPII/H36M 微调 |

不能把历史 27--29 mm 数字直接宣称“只差一个模块就能达到 19.54 mm”。AdaFuse
同时改变了热图信息、二维主干、框和训练数据；首先要把这些变量拆开。

### 1.1 2026-08-13 raw checkpoint 评估纠正

此前一次手工评估没有传入 H76 的 tri-anchor/Plücker 环境，并且把 checkpoint
文件名解析成了 `model_best.pth.tar` 目录名，产生了日志中看似 8 mm 的无效数字。
本次重新使用训练时的全部环境、同一 checkpoint、`--model-num-views 4` 后，得到上表
的 84/65/55 与 52/37/34 mm。预测字典、Table-2 JSON 和组合记录数分别为
V2=12126、V3=8084、V4=2021，因而这三项才是当前 raw-coordinate 训练基线。

## 2. 当前选择的论文依据

- [AdaFuse 官方仓库](../reference/adafuse-official)：极线热图融合和关节/视角质量
  自适应权重，作为热图级对照与最终上限目标。
- [Epipolar Transformer 官方仓库](../reference/epipolar-transformers-official)：
  从目标关节沿极线读取其他视角特征，作为当前新实验的特征对应模块。
- [Learnable Triangulation 官方仓库](../reference/learnable-triangulation-official)：
  可学习的视角/关节置信度和可微三角化，作为受控热图/坐标融合对照。
- Cross View Fusion、GHT 和 TEMPO 只作为相邻方法参考；它们的输入层级或时序协议
  不同，不直接混入 raw 坐标表。

## 3. 当前正在启动的实验：Epipolar correspondence → frozen H76

实验目录：

`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Epipolar_Ray_Input_20260813/`

流程为：

```text
冻结 H76/RUMPL 3D 主干
  + HRNet stage-4 中间特征
  + H76 三维查询投影到目标视角
  + 沿目标射线在其他视角极线采样 64 个深度点
  + Epipolar Transformer 风格对应描述子
  -> 视角置换等变的 bounded ray correction
  -> 原 H76 求解器
```

重要边界：该实验不使用测试 GT、世界体素或相机 ID；但首轮复用了历史 H76
checkpoint 和 20k balanced feature subset，因此属于**结构筛选/可行性实验**，
不能把首轮结果当作最终外部公平数字。若它稳定有效，下一步必须在同一 raw HRNet
输入和全训练主体上重做；若无效，保留官方 Epipolar Transformer 的负结果证据，
不再盲目堆叠 Transformer。

### 3.1 管线自检（已完成）

- 20k 组对应的 80k 条四视角记录与四个 HRNet feature shard 一一匹配，缺失 0；
- 11 个相机子集均可生成描述子，输出形状为 `[N, 11, 4, 17, 70]`；
- `ground_truth_used=false`；
- 修正头零初始化：射线改变量为 0，重新跑 H76 与导出缓存最大漂移约
  `6.5e-5 m`（来自缓存浮点/重算差异），视角置换误差约 `1.2e-7 m`；
- 16 组 smoke 训练已成功，首个 holdout 仅作通路检查，不作为性能结论。

## 4. 后续实验顺序

### P1：先完成无泄漏的统一热图表

在验证/测试每个实际相机子集分别导出 A1D/NoFuse 输入，不再把四视角修正点烘焙
后再删视角。固定同一 HRNet、框和物理帧，比较：

1. NoFuse/直接三角化；
2. confidence-DLT/IRLS；
3. AdaFuse-style 极线支持+质量权重；
4. A1D；
5. 上述输入接同一 RUMPL。

只有这张表能回答“热图输入是否真的优于坐标输入”，并能公平报告 A1D 的增益。

2026-08-13 已启动严格评估：验证集的每个同步四视角组在线枚举 6 个 V2、4 个
V3 和 1 个 V4 组合；A1D 模型在当前组合内才读取对应热图，不读取被排除视角。
输出目录为
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A1D_strict_heatmap_eval_20260813/`。
该评估结束前不把旧的 34/30/29 或 RIGR→E2 数字宣称为 AdaFuse 公平比较。

### 1.2 严格热图评估首轮结果（2026-08-13）

验证集在线按当前视角组合计算，action-equal All-17 absolute MPJPE（mm）：

| 热图方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| HRNet top-1 + robust ray intersection | 93.221 | 56.707 | 51.458 |
| 固定极线规则（add, α=0.25） | 92.200 | 56.683 | 51.409 |
| 固定极线规则（residual, α=0.25） | 92.355 | 56.677 | 51.414 |
| 固定极线规则（PoE, α=0.25） | 91.634 | 56.696 | 51.403 |
| **训练式 A1D dense residual** | **86.032** | **52.700** | **47.169** |

这组结果的每个 V2/V3 组合只读取所选热图，V2/V3/V4 的组合数为
12126/8084/2021；没有测试 GT 参与修正器，GT 只用于最终 MPJPE。相对 top-1，
A1D 分别下降 7.189/4.007/4.289 mm，证明完整热图的训练式极线支持有真实作用；
固定规则只有约 0.03--1.59 mm（不同 α/形式），不能把收益归给“加一个极线采样”
本身。该表是热图级几何诊断，不是 RUMPL 端到端结果，下一步必须把同一个 A1D
模块接入 H76 并重新训练/评估，才能判断是否能超过当前 RUMPL 主干。

### P2：特征级极线对应

对本记录的 Epipolar Ray Input 做两种成对训练：

- `feature`：使用 HRNet 极线对应描述子；
- `geometry`：同结构但屏蔽描述子，作为参数量匹配控制。

要求 V2/V3/V4 都按全部相机组合平均，两种 seed，记录 absolute MPJPE、负视角率、
视角置换误差和修正角度分布。

### P3：AdaFuse 风格视角效用

不先修改 2D 点，只学习关节-视角质量，接同一个鲁棒三角化器。若 V3/V4 有稳定
增益，再打开选择性二维残差；若质量头无增益，停止继续增加 gating。

### P4：作者协议复现

官方 AdaFuse/LT 权重、框、去畸变、训练增强和采样分别锁定，导出 NoFuse 与 AdaFuse
输出，验证 19.54 mm 的量级。作者协议结果与统一输入结果分栏，绝不混写。

## 5. 通过/停止标准

- 通过：同一输入、两 seed、V2→V3→V4 不恶化，且至少一个公开模块稳定超过 raw
  RUMPL；
- 进入论文主线：统一热图表相对 NoFuse/RUMPL 有可复现增益，并在未见组合/噪声下
  降低 Negative View Rate；
- 冲击 AdaFuse：作者协议表与官方 19.54 mm 同口径，或在统一输入表明确优于
  AdaFuse-style control；
- 停止：只在 V3/V4 专用头提升、V2 明显恶化，或需要测试 GT/视角 ID/世界体素才
  能提升。此时不把结果包装成通用模块。

## 6. 文件与日志约定

- 代码：`OpenRUMPL_baseline_audit/`
- 训练/评估大文件：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/`
- 当前双卡控制基线日志：
  `/mnt/data/cjyoutput/hrnet_openmpl_coordinate_protocol_20260813/{rumpl,h76}/logs/`
- 当前极线实验：上述 `Epipolar_Ray_Input_20260813/{descriptors,feature_seed0,geometry_control_seed0,logs}/`

任何结果进入论文前，都必须在 JSON 中写明 `input_level`、`bbox_source`、
`detector_weight`、`views_evaluated`、`absolute/root_relative`、`ground_truth_used`。

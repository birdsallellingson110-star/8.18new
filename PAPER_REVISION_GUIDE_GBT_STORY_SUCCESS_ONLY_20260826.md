# 论文全面修改指导：规范射线、多假设与时序鲁棒融合

> 依据旧稿 `viz_attn/main.pdf`、2026-08-26 冻结模型和最终实验结果整理。
> 本文档只组织已经成功并冻结的技术路线与实验，不写失败分支，不使用待完成结果。

## 1. 修改后的论文定位

旧稿标题和正文把“Global Joint-Query”放在核心位置，但当前最终模型中只有
ResNet-152 分支启用该模块，HRNet-W32 分支没有启用。因此，新稿不应继续把
Global Joint-Query 写成适用于两种前端的唯一核心贡献，也不应继续用旧 world-frame
实验中的 `41.470 → 32.312 mm` 作为新 canonical 模型的主消融。

建议把整篇论文重新定位为：

> 面向少视角、可变相机子集和部分视角遮挡的坐标级多视角三维人体姿态重建。
> 方法先在身体规范坐标中完成相机身份无关的射线提升，再显式生成并逐关节评价
> 学习式与几何式三维假设，最后用轻量、相机无关的时序残差补偿当前帧缺失证据。

推荐标题：

> **面向相机子集泛化的规范射线多假设时空融合三维人体姿态重建**

英文可写为：

> **Canonical-Ray Multi-Hypothesis Spatio-Temporal Fusion for Camera-Subset-Generalizable Multi-View 3D Human Pose Reconstruction**

如果希望保留“相机配置泛化”，可以使用：

> **面向相机配置泛化的规范射线多假设时空融合三维人体姿态重建**

但正文必须把当前实验结论限定为“不同相机数量、相机子集与受污染视角下的鲁棒性”。
在没有冻结的跨数据集或全新物理相机布局结果时，不要把实验结论扩大成“已证明对任意
未见相机位置均泛化”。

## 2. 与 GBT 的关系应该怎样写

论文可以沿用 GBT 的问题动机和实验叙事，但不要写“复现 GBT”或“严格遵循 GBT
实现”。GBT 没有公开代码；当前能严格引用的是论文公开的模型设计、训练策略和报告值。

### 2.1 可以继承的叙事原则

GBT 将现实部署难点概括为少视角、宽基线、遮挡、有限重叠区域和未见场景。其泛化性
主要来自：

1. 把二维关节转换为世界三维 Plücker 射线，而不是让网络从像素坐标隐式学习相机投影；
2. 将时间、视角和关节 token 放入全局 Transformer，用置信度和射线距离引导注意力；
3. 不使用固定相机身份作为输入语义；
4. 通过场景中心化、随机旋转、合成相机视角和 token dropout 降低固定训练布局依赖；
5. 在 H36M 上系统报告两、三、四视角，并用遮挡和跨数据集实验检验鲁棒性。

我们的论文应沿用“现实问题 → 几何显式化 → 相机身份无关 → 少视角/遮挡验证”的逻辑，
但技术解法要明确区别于 GBT。

### 2.2 我们与 GBT 的核心区别

| 方面 | GBT | 本文 |
|---|---|---|
| 主体结构 | 关节、视角、时间 token 的统一编码器–解码器 | 分阶段的规范空间生成器、显式候选效用融合和轻量时序残差 |
| 几何表示 | 世界坐标 Plücker 射线及射线距离注意力偏置 | 先由人体锚点建立 body-canonical 坐标，再在规范坐标中编码射线与姿态 |
| 绝对位置处理 | 场景中心化；推理时由三角化/前一帧确定中心 | 置信度加权射线交会给出 pelvis 锚点，并在输出时逆变换回世界坐标 |
| 鲁棒融合 | 全局注意力隐式融合观测 | 11 个学习式候选 + 11 个置信度三角化候选，逐关节预测相对风险并软融合 |
| 时间模块 | 9 帧历史输入，预测最新帧，因果 | 过去 4 帧 + 当前帧 + 未来 4 帧，预测中心帧，离线 centered T=9 |
| 缺失增强 | 20% token dropout | HRNet 分支训练期使用 10% token dropout；测试关闭 |
| 当前训练数据 | 论文包含合成视角策略 | 当前最终模型只用 clean H36M；不要声称使用 AMASS 或随机相机训练 |

论文中可写：

> Inspired by the geometry-explicit and camera-identity-free principles of GBT,
> we pursue a different staged solution. Instead of regressing the final pose from
> a single global spatio-temporal latent, we canonicalize the ray geometry in a
> body-attached metric frame, explicitly construct complementary learned and
> geometric hypotheses, estimate their joint-wise utility, and apply a bounded
> temporal residual only after spatial fusion.

这里的 “inspired by” 是方法论关系；不要写 “following the GBT implementation”。

## 3. 新的整篇故事主线

### 3.1 问题

坐标级多视角方法比图像特征或体素方法更适合带宽受限、前端异构和相机端独立部署，
但仍面临三个泛化瓶颈：

1. **世界坐标依赖**：直接在世界 XYZ 中学习可能记住训练捕获空间的朝向、原点和人体
   朝向分布；使用射线并不自动消除后续网络的坐标轴偏置。
2. **证据压缩与候选歧义**：少视角时逐关节证据不足；多视角时新增视角也可能带来遮挡
   或离群二维检测，单个融合结果无法显式表达不同相机子集的可靠性。
3. **当前帧证据缺失**：空间模块只能利用同一时刻的观测，无法区分瞬时检测错误、持续
   遮挡和真实运动。

### 3.2 方法回答

与上述三个问题一一对应：

1. **Body-canonical ray generator**：用置信度加权射线交会和人体轴建立刚体一致的身体
   规范坐标，RUMPL 的 VFT/PFT 在规范坐标中运行，最终逆变换到世界系。
2. **22-candidate joint-wise utility fusion**：从全部合法相机子集分别产生 11 个生成器
   候选和 11 个置信度三角化候选，集合评分器逐关节预测风险并软融合。
3. **Camera-independent temporal residual**：把九帧窗口统一到中心帧身体坐标，利用规范
   姿态、相对运动和融合不确定性预测有界中心帧残差。

### 3.3 实验回答

论文只需要两大实验：

1. **H36M clean 少视角实验**：与 GBT 相同，报告全部 V2/V3/V4 组合，回答“可用相机
   数量减少时，模型是否仍能稳定恢复绝对三维姿态”。
2. **Human3.6M-Occ VOC 物体遮挡实验**：clean-only 训练、zero-shot 遮挡测试，回答
   “部分相机受到未见物体遮挡时，空间候选和时间上下文能否抑制坏观测”。

不要在当前稿中保留空白的 CMU→H36M 表、未完成的 Occlusion-Person 表或没有最终值的
时间窗口 2/3/6 帧列。它们会削弱已经完整的两阶段故事。

## 4. 摘要如何重写

旧摘要中的以下数值全部替换：

| 二维输入 | 旧稿 | 新冻结值 |
|---|---|---|
| ResNet-152 | 31.215/22.008/19.971 | **29.416/21.020/19.281** |
| HRNet-W32 | 37.704/29.231/27.219 | **37.392/29.501/27.713** |

摘要不要再写“Global Joint-Query 是两种输入的统一核心”，也不要只写 clean 结果。建议
加入遮挡主结果和 zero-shot 设置。

### 4.1 建议摘要文本

> 本文研究少视角、可变相机子集和部分视角遮挡条件下的坐标级多视角三维人体姿态
> 重建。现有射线提升方法虽然显式使用相机几何，但后续网络仍可能依赖训练捕获空间的
> 世界坐标轴；单次视角融合也难以显式判断不同相机子集在遮挡下的可靠性。为此，本文
> 提出一种规范射线多假设时空融合框架。模型首先由置信度加权射线交会和人体轴建立
> 保持米制尺度的身体规范坐标，并在该坐标中执行可变视角集合提升；随后，从全部合法
> 相机子集产生 11 个学习式候选和 11 个置信度三角化候选，由置换等变评分器预测逐关节
> 相对风险并完成软融合；最后，轻量九帧时序分支在中心帧规范坐标中利用相对运动和融合
> 不确定性预测有界残差。整个三维网络仅接收二维坐标、检测置信度和标定射线，不使用
> 相机编号或图像特征。在 Human3.6M clean 协议上，完整模型使用 ResNet-152 输入取得
> 29.416/21.020/19.281 mm 的两、三、四视角绝对 MPJPE，三列均低于 GBT 的公开报告值；
> 使用通用 HRNet-W32 输入取得 37.392/29.501/27.713 mm。在仅使用 clean Human3.6M
> 训练并直接测试未见物体遮挡时，ResNet-152 模型在 Human3.6M-Occ 的 Occ-2/Occ-3
> 四视角设置下分别达到 21.349/22.653 mm，低于相同 ResNet-152 输入族的公开方法。
> 消融结果表明，多假设融合主要利用三、四视角的空间冗余，而时序上下文在遮挡下提供
> 比 clean 场景更显著的补偿。

摘要中的“低于相同输入族公开方法”比“SOTA”更稳妥，因为我们的 T=9 是离线中心窗口，
而公开遮挡方法多为 T=1。

## 5. 引言修改指导

### 5.1 第一段：实际需求

保留旧稿对单目深度歧义和多视角优势的介绍，但更快进入现实部署：少量宽基线相机、
相机覆盖不完整、家具/人员遮挡、二维骨架传输。不要在第一段堆叠所有模块名。

### 5.2 第二段：为什么已有方法不够

按三类方法组织：

1. 三角化依赖至少两条可靠射线，少视角、退化基线或二维离群点会导致绝对位置发散；
2. 图像/热图/体素融合精度高，但通信和计算代价高，而且通常绑定具体图像主干；
3. 射线 Transformer 具有更好的相机泛化潜力，但“世界射线输入”并不等于后续网络
   自动摆脱世界坐标轴、固定场景和候选污染。

### 5.3 第三段：关键观察

建议把旧稿的“过早逐关节压缩”改成更完整的观察：

> 相机泛化不仅要求输入对相机排列置换不敏感，还要求中间三维表征不把捕获空间的绝对
> 方向和原点当作可学习捷径；遮挡鲁棒性也不仅依赖一次全局注意力，还需要显式比较不同
> 相机子集产生的互补三维解释。

这样能够自然引出 canonical frame 和 E2，而不是把论文全部押在只对 ResNet 开启的
Global Joint-Query 上。

### 5.4 第四段：总体方法

用一段依次介绍 canonical generator、22 候选效用融合和 T=9 residual。Global
Joint-Query 只写成 ResNet 生成器中的联合关节–视角更新分支；HRNet 使用不带该分支的
生成器。两者共享 body-canonical、E2 和 H18 主框架。

### 5.5 贡献列表

建议写成三项技术贡献加一项实验贡献：

1. 提出保持米制尺度的身体规范射线提升，把相机射线、生成姿态和后续残差统一到由
   当前人体观测确定的局部坐标，降低世界原点和轴向分布对坐标级网络的影响。
2. 提出逐关节多假设效用融合，在统一集合中评价 11 个学习式与 11 个几何式候选，
   不使用相机编号，并利用相机子集冗余抑制遮挡视角的负贡献。
3. 提出相机无关的轻量中心帧时序残差，在规范坐标中结合姿态运动与 label-free 融合
   不确定性，在不进行遮挡训练的情况下补偿当前帧缺失证据。
4. 在两种冻结二维前端上，按照全部 V2/V3/V4 组合系统评估 clean 少视角和 VOC 物体
   遮挡；ResNet-152 clean 三列均低于 GBT 报告值，遮挡四视角结果低于相同输入族的
   SkelSplat 和 AdaFuse。

## 6. 相关工作如何调整

建议保留三节，但改变侧重点：

### 6.1 几何约束的多视角重建

介绍 Learnable Triangulation、Cross-view Fusion、Epipolar Transformer、AdaFuse。
强调这些方法分别使用解析投影、热图/图像特征融合和自适应视角权重。最后指出本文从
冻结坐标和置信度出发，不读取图像特征。

### 6.2 相机配置泛化与射线表示

把 GHT、GBT、RUMPL 放在这一节：

- GHT：随机相机子集产生假设并学习评分，启发我们的显式候选评价；
- GBT：Plücker 射线、几何/置信度注意力偏置、中心化、合成相机和 token dropout；
- RUMPL：VFT/PFT 的可变视角集合提升，是本文空间生成器的基础。

最后明确本文区别：body-canonical metric frame + 学习/几何候选逐关节效用 + 后置
camera-independent temporal residual。

### 6.3 遮挡鲁棒性与时序

介绍 AdaFuse、Multi-view Pose Fusion、SkelSplat 和 GBT 的时序建模。指出：

- Human3.6M-Occ 的公开路线来自 Multi-view Pose Fusion，并被 SkelSplat 继续使用；
- 它通过 Pascal VOC 前景物体形成跨相机受污染观测，适合测试 clean-trained 模型的
  zero-shot robustness；
- 本文时序不是对最终姿态做无约束平滑，而是在规范坐标中预测有界中心帧残差。

## 7. 方法章节重构

建议将旧稿第 3 节改为以下结构。

### 7.1 问题定义和框架

定义相机集合 `V_t`、二维关节 `x_tvj=(u,v,1)`、置信度 `c_tvj`、内参 `K_v`、外参
`(R_v,t_v)` 和目标绝对世界坐标姿态 `P_t`。明确：

- 冻结二维检测器；
- 三维网络不读取 RGB、heatmap、bbox 特征或相机 ID；
- 相机数量和排列可变；
- 输出为无对齐的绝对三维坐标。

框架图建议画成单条主链：

```text
Frozen 2D keypoints/confidences + camera calibration
                        │
                        ▼
        Body-canonical ray construction
                        │
                        ▼
       Canonical spatial pose generator
      (ResNet variant includes Joint-Query)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
  11 learned hypotheses   11 confidence-triangulated hypotheses
            └───────────┬───────────┘
                        ▼
       Joint-wise candidate utility fusion
                        │
                        ▼
    Camera-independent centered T=9 residual
                        │
                        ▼
             Absolute world 3D pose
```

图中不要再出现未进入最终链的模块，也不要把 MOR 画成一个独立贡献，除非最终模型确实
加载了对应权重并有冻结消融。

### 7.2 身体规范射线表示

由像素和标定得到世界射线：

```math
d_{tvj}=\frac{R_v^\top K_v^{-1}x_{tvj}}
              {\|R_v^\top K_v^{-1}x_{tvj}\|_2},\qquad
o_v=-R_v^\top t_v.
```

用置信度加权射线交会得到 pelvis 锚点 `a_t`。由左右肩方向构造横轴 `e_x`，将
pelvis→neck 向量去除 `e_x` 分量并归一化得到 `e_y`，再令 `e_z=e_x×e_y`，组成
`B_t=[e_x,e_y,e_z]`。世界点和射线转换为：

```math
x^c=B_t^\top(x^w-a_t),\qquad d^c=B_t^\top d^w.
```

必要时以规范射线原点构造 Plücker 矩 `m^c=o^c×d^c`。输出通过
`x^w=B_t x^c+a_t` 返回世界坐标。强调只移除刚体平移/旋转分布，不做尺度归一化，
因此仍预测 metric absolute pose。

### 7.3 规范空间生成器

保留 RUMPL 的 VFT/PFT：先用共享视角集合编码器处理同一关节的可变视角观测，再建模
全身结构。ResNet-152 实例额外启用 depth=2、max residual=0.5 的 Global
Joint-Query，使每个关节在输出前访问完整关节–视角记忆。HRNet-W32 实例关闭该分支，
使用 pelvis prior 和训练期 10% token dropout。两种实例都在同一个 body-canonical
框架中输出学习式姿态候选。

正文不要把两种实例写成完全相同参数。Implementation Details 中用一个小表透明列出：

| 设置 | ResNet-152 | HRNet-W32 |
|---|---:|---:|
| body-canonical regularization | `1e-4` | `1e-2` |
| pelvis prior | off | on |
| Global Joint-Query | on | off |
| train token dropout | 0 | 10% |

### 7.4 多假设逐关节效用融合

四个原始相机的全部两、三、四视角合法子集共有 `6+4+1=11` 个。对每个子集同时产生：

- 一个 canonical generator 候选；
- 一个 confidence-weighted triangulation 候选。

因此完整候选集合共有 22 个。对候选 `c` 和关节 `j`，使用根相对姿态、候选共识偏差、
点到射线距离、重投影统计、置信度统计、几何条件和关节语义构造相机身份无关特征。
集合编码器预测风险 `r_cj`，融合为：

```math
\alpha_{cj}=\frac{\exp(-r_{cj}/\tau_K)}
{\sum_{c'}\exp(-r_{c'j}/\tau_K)},\qquad
\hat p_j=\sum_c\alpha_{cj}p_{cj}.
```

使用 `τ_2=0.4`、`τ_3=τ_4=1.8`。逐关节权重允许躯干、腕和踝使用不同候选。identity
protection 只描述为防止可靠基线被无故破坏的稳定约束，不把它包装成主要精度来源。

### 7.5 相机无关时序残差

使用 `T=9`、frame stride=5 的中心窗口。将整段窗口统一变换到中心帧身体坐标，输入
规范姿态、相对运动和融合统计；先做帧内骨架编码，再沿每个关节的时间轴编码轨迹，
只预测中心帧有界残差。最终残差旋回世界系。

ResNet 最终实例使用 7 维 label-free uncertainty gate、stage-balanced loss 和
sequence loss weight=0.25；HRNet 使用 continuous-nowarp 版本。共同设置为 hidden=96、
两层空间/时间建模、residual scale=0.10 m。

必须明确：这是 centered/offline T=9，不是 GBT 的 causal latest-frame T=9。

### 7.6 训练与模型选择

按生成器 → E2 → H18 分阶段训练和冻结。E2 报告两个 seed 的均值。H18 checkpoint 只按
clean H36M S8 holdout 选择；S9/S11 和 Human3.6M-Occ 均不参与 epoch、温度或模型选择。
遮挡实验不重新训练任何模块。

## 8. 实验一：H36M clean 少视角

### 8.1 实验目的

这一实验与 GBT 的 Table I 对齐，回答：

- 相机从四个减少到三个、两个时，绝对深度歧义如何变化；
- 同一完整模型能否处理不同相机数量和全部相机子集；
- 相同二维输入来源下，与 GBT 的公开结果相比处于什么位置。

### 8.2 协议必须写清

- train：S1/S5/S6/S7/S8；test：S9/S11；
- 排除 S9 中公开已知错误片段；
- V2/V3/V4 分别平均全部 `6/4/1` 个相机组合；
- All-17 absolute MPJPE，action-equal，无 root/scale/Procrustes alignment；
- ResNet-152† 为 Learnable Triangulation 发布并在 H36M 微调的前端；
- HRNet 为 YOLOX-X + COCO HRNet-W32 通用前端；
- Ours 为 centered T=9，GBT 为 causal T=9；必须保留 `T` 列。

### 8.3 建议 clean 主表

| Method | 2D input | T | V2 | V3 | V4 |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation | ResNet-152† | 1 | 51.1 | 23.4 | **19.1** |
| GBT | ResNet-152† | 9 causal | 29.9 | 24.4 | 22.7 |
| **Ours** | **ResNet-152†** | **9 centered** | **29.416** | **21.020** | 19.281 |
| Algebraic Triangulation | HRNet-W32 | 1 | 120.7 | 50.9 | 44.2 |
| GBT | HRNet-W32 | 9 causal | **36.8** | 30.4 | **26.0** |
| **Ours** | **HRNet-W32** | **9 centered** | 37.392 | **29.501** | 27.713 |

粗体按同一输入 block 的最佳值标。注意 ResNet V4 的 Algebraic 19.1 比我们的 19.281
低 0.181 mm，不能把整张表写成全部方法全列最优。正确主结论是：

- 相比 GBT，ResNet-152 下我们在 V2/V3/V4 分别降低 `0.484/3.380/3.419 mm`；
- HRNet 下我们在 V3 降低 `0.899 mm`，V2/V4 报告客观数值，不声称领先；
- 相比相同 HRNet 坐标的 Algebraic，三列均显著降低，说明提升不依赖 H36M 微调前端。

### 8.4 建议 clean 空间消融表

不要继续使用旧稿的 H76 → Global Query → identity 的 world-frame 表。改成当前 canonical
链的成功累积消融：

| Input / cumulative spatial model | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet canonical GQ generator | 30.617 | 23.712 | 22.434 |
| + 22-candidate utility fusion | 30.643 | **21.528** | **19.566** |
| HRNet canonical token10 generator | 38.412 | 31.376 | 28.900 |
| + 22-candidate utility fusion | 38.423 | **29.776** | **27.708** |

分析写成：E2 对 V2 基本保持（+0.027/+0.011 mm），但 ResNet V3/V4 降低
`2.184/2.868 mm`，HRNet V3/V4 降低 `1.600/1.192 mm`。这与候选冗余解释一致：
三、四视角提供足够交叉验证证据，两个视角则难以从共享的系统性二维误差中恢复信息。

### 8.5 建议 clean 时序消融表

时序增量必须使用匹配中心帧，不能直接拿稀疏 E2 均值相减：

| Input | matched T=1 V2/V3/V4 | selected T=9 V2/V3/V4 | gain |
|---|---:|---:|---:|
| ResNet-152 | 30.671/21.540/19.587 | **29.416/21.020/19.281** | **1.255/0.519/0.306** |
| HRNet-W32 | 38.529/29.893/27.802 | **37.392/29.501/27.713** | **1.137/0.392/0.089** |

结论：clean 下时序收益稳定但适中，而且视角越少收益越大；不要把它写成主要 clean
精度来源，其更强价值在第二实验的遮挡条件下体现。

## 9. 实验二：Human3.6M-Occ VOC 物体遮挡

### 9.1 为什么不使用 GBT 的遮挡数字直接比较

GBT 使用的是 H36M-Occl：在二维关节附近随机放置白色方块，概率为 0.1。论文有文字
说明和结果，但没有公开生成代码、固定 mask 或可逐样本对齐的资产。我们不能把自己
生成的白色方块数据称为严格复现，也不能把 VOC 遮挡结果与 GBT Table II 数字直接相减。

因此第二实验采用可复查的 Human3.6M-Occ 路线。该基准由 Multi-view Pose Fusion 提出，
使用 Pascal VOC 2012 前景物体遮挡 H36M 人物；SkelSplat 进一步系统报告 Occ-2、Occ-3
和 Occ-3-Hard，并汇总多个公开方法。我们的协议沿用其 Occ-2/Occ-3 设定。

### 9.2 实验目的

这一实验不是再次证明 clean 精度，而是检验三点：

1. 只在 clean H36M 上训练的模型能否 zero-shot 迁移到未见物体遮挡；
2. E2 的相机子集候选能否利用未受污染视角和候选共识；
3. H18 是否能用相邻帧补偿当前帧被遮挡的关节，并且收益是否大于 clean。

### 9.3 协议

- 四个源视角中随机遮挡两个（Occ-2）或三个（Occ-3）；
- 每个被选视角粘贴两个 Pascal VOC 前景物体；
- object scale 为人体框短边的 `0.2–0.5`，seed=42；
- 所有 V2/V3/V4 相机组合均评估；
- 26,269 个同步组提供时序上下文，2,021 个中心组评分；
- T=9 为过去 4 帧、中心帧、未来 4 帧，frame stride=5；
- All-17 action-equal absolute MPJPE，无对齐；
- 所有 checkpoint 只在 clean H36M 训练和选择，遮挡集不参与调参。

### 9.4 建议公开方法主表：只比较有精确公开值的 V4

| Method | 2D input family | T | Occ-2 V4 | Occ-3 V4 |
|---|---|---:|---:|---:|
| Algebraic Triangulation | ResNet-152 | 1 | 43.2 | 48.9 |
| RANSAC | ResNet-152 | 1 | 33.7 | 38.6 |
| AdaFuse | ResNet-152 | 1 | 27.9 | 31.2 |
| SkelSplat | ResNet-152 | 1 | 24.6 | 27.0 |
| **Ours, spatial fusion** | **ResNet-152** | **1** | **23.383** | **26.092** |
| **Ours, full model** | **ResNet-152** | **9 centered** | **21.349** | **22.653** |

这张表能支持两层结论：

- 不使用时序时，我们的 T=1 空间模型已分别比 SkelSplat 低 `1.217/0.908 mm`；
- 完整 T=9 模型进一步比 SkelSplat 低 `3.251/4.347 mm`，比 AdaFuse 低
  `6.551/8.547 mm`。

表注必须写“same ResNet-152 detector family/source”，除非已经逐文件证明所有公开方法
使用完全相同的缓存、crop 和预处理，否则不要写“identical 2D coordinates”。

公开论文没有给出同一 VOC 协议下精确的 V2/V3 数字，因此不要从曲线估算，也不要拿
GBT 白方块遮挡的 V2/V3 数字填入这张表。

### 9.5 建议完整遮挡结果表

| Input | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-152, T=9 | **45.278** | **25.652** | **21.349** | **51.111** | **27.862** | **22.653** |
| HRNet-W32, T=9 | **53.966** | **32.204** | **28.705** | **58.852** | **33.970** | **29.695** |

ResNet 是外部主比较线；HRNet 用来验证相同 canonical/E2/H18 框架不绑定单一二维前端。
不要跨前端给数字加共同排名。

### 9.6 建议遮挡时序消融表

| Input / setting | matched T=1 V2/V3/V4 | T=9 V2/V3/V4 | H18 gain |
|---|---:|---:|---:|
| ResNet Occ-2 | 49.739/27.886/22.672 | **45.278/25.652/21.349** | **4.461/2.235/1.323** |
| ResNet Occ-3 | 56.699/30.897/24.607 | **51.111/27.862/22.653** | **5.588/3.035/1.954** |
| HRNet Occ-2 | 57.996/34.088/29.667 | **53.966/32.204/28.705** | **4.030/1.884/0.962** |
| HRNet Occ-3 | 63.893/36.498/31.153 | **58.852/33.970/29.695** | **5.041/2.528/1.458** |

可以直接写的结果分析：

> 与完全匹配的中心帧 T=1 基线相比，九帧时序残差在两种二维输入、两种遮挡强度和
> 三种视角数下均降低误差。最大收益出现在 ResNet Occ-3 V2，为 5.588 mm。随着可用
> 视角增加，时间收益逐步减小；随着遮挡从 Occ-2 加强到 Occ-3，时间收益增大。这说明
> 时序模块主要在当前帧空间证据不足时发挥作用，而不是对已经可靠的多视角结果进行
> 无差别平滑。

## 10. 表格和图的排版建议

### 10.1 主表数量

正文控制在四张核心表：

1. H36M clean 外部比较（两种输入、V2/V3/V4）；
2. clean 模块消融（空间 E2 + matched temporal，可做上下两个 panel）；
3. Human3.6M-Occ V4 外部比较（ResNet-152）；
4. Human3.6M-Occ 全视角时序消融（两种输入）。

完整遮挡结果可与第 4 张合并，或放补充材料。不要再保留多个空表。

### 10.2 表格视觉规则

- 用横向分组标题区分 ResNet-152 与 HRNet-W32，禁止跨输入比较粗体；
- `V2/V3/V4` 上方可加 grouped header “Number of input cameras”；
- 外部公开值统一保留一位小数，我们的内部表统一三位小数；不要在同一比较差值中使用
  不一致精度；
- `†` 只表示二维前端在 H36M 微调，并在表注一次性解释；
- `T=9 causal` 与 `T=9 centered` 直接写在 T/Temporal 列，不能只藏在正文；
- 粗体表示同协议 block 最优，underline 表示第二；
- 若输入、监督或评价不同，增加 `2D source`、`3D GT train` 或脚注，不用视觉粗体制造
  不公平领先印象；
- 表内使用 `Occ-2`、`Occ-3`，不要同时混用 H36M-Occl、H36M-Occ 和 Occlusion-Person。

### 10.3 框架图

旧图应突出五个信息：

1. 输入只有 `(x,y,confidence)` 和 calibration；
2. body-canonical 坐标轴如何由 pelvis、shoulders、neck 构造；
3. 11+11 候选从不同相机子集产生；
4. 每个关节具有独立候选权重；
5. T=9 只对中心帧输出残差，然后 inverse canonical transform。

建议用三种颜色：蓝色为几何/坐标变换，橙色为可学习空间模块，绿色为时间模块。候选
分支画成两个并行框，在 “22 hypotheses” 处合流。不要在主图中画训练日志名 E2/H18；
使用论文语义名称，代码名只在补充材料映射。

### 10.4 建议增加一张定性图

选择 Occ-3 中同一时刻的四视角图像和骨架：

- 展示三个视角被 VOC 物体覆盖、一个视角较完整；
- 展示单帧空间结果和 T=9 最终结果；
- 在 3D 图中突出腕、踝等改善关节；
- 图注写明模型只在 clean H36M 训练。

这张图比旧稿的噪声注意力曲线更贴合最终故事。

## 11. 旧稿逐项修改清单

### 首页与摘要

- 更换标题，弱化仅对 ResNet 使用的 Global Joint-Query；
- 替换两组 clean 最终数值；
- 加入 Occ-2/Occ-3 四视角最终数值；
- 把贡献重点改为 body-canonical、22-candidate utility 和 temporal residual。

### 引言

- 保留现实相机部署动机；
- 增加“世界射线输入仍可能保留世界轴依赖”的问题；
- 从“推迟视角压缩”扩展为“规范坐标 + 显式候选 + 时间补证据”；
- 删除空白跨数据集实验承诺；
- 更新贡献列表。

### 相关工作

- 增强 GBT、GHT、RUMPL 的联系与区别；
- 加入 Multi-view Pose Fusion 和 SkelSplat 对 Human3.6M-Occ 的定义及公开比较；
- 不把 GBT 的 H36M-Occl 与 VOC Human3.6M-Occ 视为同一数据集。

### 方法

- 在旧“锚点中心化令牌”之前加入完整 body-canonical 坐标构造；
- 把方法主线由 Global Joint-Query 改成 canonical generator；
- 保留 22 候选逐关节风险融合公式；
- 将旧 H18 输入中的绝对世界 root 表达改为中心帧规范坐标中的姿态、相对运动和融合
  统计；
- 写明 ResNet/HRNet 两个生成器实例的差异；
- 写明 centered/offline T=9。

### 实验

- 表 1 替换为本指导第 8.3 节的新 clean 主表；
- 删除旧 H36M-Occl 空白表，不能用 VOC 数据填入 GBT 白方块协议表；
- 删除旧 Occlusion-Person 空白表；
- 删除旧 CMU→H36M 空白表；
- 用第 8.4、8.5 节替换旧 world-frame 核心消融；
- 新增 Human3.6M-Occ 外部 V4 表、完整结果表和 matched 时序消融表；
- 删除时间长度 2/3/6 的空列，只保留有冻结数据的 matched T=1/T=9。

### 结论

结论只概括已被两组实验直接支持的内容：

> 本文通过身体规范射线表示、逐关节多假设效用融合和相机无关时序残差，提高了坐标级
> 多视角三维姿态在少视角和部分视角遮挡下的稳定性。ResNet-152 输入下，clean
> V2/V3/V4 均低于 GBT 报告值；在 clean-only 训练的 VOC 物体遮挡测试中，T=1 空间
> 模型和 T=9 完整模型均在四视角上低于相同输入族公开结果。多假设融合的收益集中在
> 具有空间冗余的三、四视角，而时序收益在重遮挡和两视角条件下最大。

不要在结论中加入尚未由当前两组实验验证的 AMASS、随机相机、CMU→H36M 或任意新
相机位置结论。

## 12. 可直接用于正文的两段结果分析

### 12.1 Clean 少视角

> Table X reports action-equal All-17 absolute MPJPE over all camera subsets.
> With the H36M-finetuned ResNet-152 input, our centered nine-frame model
> obtains 29.416, 21.020, and 19.281 mm for two, three, and four cameras,
> respectively. These values are 0.484, 3.380, and 3.419 mm lower than the
> corresponding GBT results. With the off-the-shelf HRNet-W32 input, our model
> obtains 37.392/29.501/27.713 mm and improves the three-view GBT result by
> 0.899 mm. The two detector blocks are reported separately to avoid attributing
> detector accuracy to the 3D reconstruction module.

### 12.2 VOC 物体遮挡

> We train all components exclusively on clean Human3.6M and directly evaluate
> the frozen models on Human3.6M-Occ. Under the ResNet-152 input, the spatial
> T=1 model obtains 23.383/26.092 mm for Occ-2/Occ-3 with four cameras,
> already improving over SkelSplat by 1.217/0.908 mm. The complete centered
> T=9 model further reduces the error to 21.349/22.653 mm. On matched center
> frames, temporal refinement improves all view counts and both detector inputs;
> its largest gain is 5.588 mm for ResNet Occ-3 with two cameras. This pattern
> supports the intended division of labor: candidate utility exploits spatial
> redundancy, whereas temporal context compensates for insufficient or corrupted
> current-frame evidence.

## 13. 公平比较边界

论文最终提交前逐条核对：

1. clean 主表可以逐列比较 GBT，因为数据集、二维输入类别、绝对 MPJPE 和相机数对齐；
   但 GBT 为 causal T=9、我们为 centered T=9，因此称“reported-value comparison”，
   不称“strict matched implementation”。
2. VOC 遮挡外部表只比较 V4 精确公开值；V2/V3 不与外部论文做数值排名。
3. “same ResNet-152 input family”不等于“identical cached keypoints”。只有同一模型、
   crop、去畸变、样本和二维坐标文件全部核验后才能使用 identical。
4. 不把 HRNet 与 ResNet 行直接比较为三维方法优劣。
5. 不把 T=9 与 T=1 的差异隐藏；必须有 matched T=1 消融。
6. 不把遮挡集用于 checkpoint 或超参数选择；正文明确 clean-only training。
7. 所有最终数字以三位小数书写，内部计算保留完整精度。

## 14. 公开来源

- GBT 原文与设计/clean 报告值：
  https://arxiv.org/abs/2312.17106
- Human3.6M-Occ 的 Multi-view Pose Fusion 原始路线：
  https://arxiv.org/abs/2408.15810
- SkelSplat WACV 2026 原文、Human3.6M-Occ Table 4 和公开比较值：
  https://openaccess.thecvf.com/content/WACV2026/html/Bragagnolo_SkelSplat_Robust_Multi-view_3D_Human_Pose_Estimation_with_Differentiable_Gaussian_WACV_2026_paper.html
- AdaFuse：
  https://arxiv.org/abs/2010.13302

## 15. 本项目冻结事实源

- 成功模型链、clean 消融和最终数值：
  `/home/lixiaob/cjy/SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.md`
- 机器可读版本：
  `/home/lixiaob/cjy/SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.json`
- clean H18 选择：
  `/mnt/data/cjyoutput/camera_generalization_20260824/final_temporal_selection_20260825.json`
- VOC 遮挡最终表：
  `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.json`

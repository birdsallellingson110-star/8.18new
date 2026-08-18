# RUMPL 论文支撑的模型创新计划（2026-08-09）

## 1. 约束与当前基线

- 主基线固定为 H76：保留 RUMPL 的 ray token、VFT、PFT、相机标定输入、
  confidence、triangulation anchor 和 anchor-centered Plücker 表示。
- H76 seed0 严格 H36M S9/S11 action-equal All-17：V2 `34.8163`、
  V3 `30.4890`、V4 `29.6913` mm。
- H76 三种子均值：V2 `34.677±0.150`、V3 `30.501±0.350`、
  V4 `29.710±0.549` mm。
- 导师不接受蒸馏，因此 D1/D2 及其 fixed-teacher/dual-hard 后续全部停止。
- 本阶段不先改变损失、视角采样、学习率、epoch 或数据增强；先单独验证模型
  结构。模型成立以后才允许调训练方式。

## 2. 论文与官方代码筛选

### 2.1 采用：SGraFormer（AAAI 2024）

- 论文：L. Zhang et al., *Deep Semantic Graph Transformer for Multi-View
  3D Human Pose Estimation*, AAAI 2024。
- 官方论文：https://ojs.aaai.org/index.php/AAAI/article/view/28549
- 官方代码：https://github.com/z0911k/SGraFormer
- 本地论文：`/home/lixiaob/cjy/reference/SGraFormer_official/SGraFormer_AAAI2024.pdf`
- 本地官方代码：`/home/lixiaob/cjy/reference/SGraFormer_official`
- 代码许可证：MIT。

采用理由：

1. 它与当前任务输入一致：先用离线 2D detector，再从多视角 2D pose 做 3D
   lifting；实验包含真实 Human3.6M 和 CPN 检测输入。
2. 它是模型结构创新，不是训练技巧。核心是把 joint position、multi-hop
   skeleton structure 和 bone edge 写进 Transformer 编码器。
3. 论文 Table 5 的同一训练口径消融为：position-only `29.4` mm；加入 spatial
   graph 为 `28.2` mm；position+spatial+edge 完整模型为 `27.6` mm，说明骨架
   结构贡献并非只有更大网络容量。
4. 论文 Figure 6 在时间感受野 `TRF=1` 时仍为约 `29.6` mm，说明语义骨架与
   多视角模块在没有时序时仍可工作；时序不是本轮成立的前提。
5. 论文 Table 7 随视角数从 V1/V2/V3/V4 得到 `47.8/32.0/31.1/27.6` mm，
   与我们需要解决的“V4 增益不足”直接相关。

### 2.2 暂不采用的候选

- MVGFormer（CVPR 2024）：官方代码存在，但它需要图像 backbone 特征，执行
  3D query 投影、局部图像特征采样、2D residual 和可微三角化迭代。当前
  RUMPL 只有检测后的关键点/射线，直接接入会同时改变输入和完整框架，无法
  做单变量模型判断。
- GraFormer（CVPR 2022）：有 Apache-2.0 官方代码，证明 GraAttention 与
  ChebGConv 对骨架关系有效；但它是单视角 2D-to-3D。若 SGraFormer 主实验
  失败，才把它作为第二优先级的 PFT 完整替换方案，而不是再做单层 graph
  residual。
- MPL：它是 RUMPL 的直接前身，继续使用其 SPT/FPT 不能形成新的融合创新。
- HMVformer++：包含 iterative feature distillation，与导师明确排除的蒸馏
  冲突；当前也不作为主线。

## 3. RUMPL 缺陷与接入位置

H76 当前的顺序是：

`每条 ray 独立嵌入 -> 同一 joint 跨 view 的 VFT -> 17 joints 的 PFT -> 3D head`

因此 VFT 判断某一视角的膝、踝 ray 是否可靠时，还没有看到该视角内完整人体
骨架。H76 audit 又显示坏 V2 pair 主要由局部下肢误差主导，而单纯 view bias、
joint bias 和 PFT 后单层 graph residual 均失败。

本轮按 SGraFormer 原论文所在位置，把语义骨架编码放在跨视角融合之前：

`ray embedding -> 每个 view 内 SGra semantic graph encoder -> 原 RUMPL VFT
-> 原 RUMPL PFT -> 原 tri-anchor residual head`

这样：

- 保留 RUMPL 的相机泛化核心和完整输出路径；
- 在 VFT 选择/融合各视角以前引入人体结构；
- 不重复 R7c。R7c 只在 PFT 以后做一次零初始化邻接 MLP；本轮在 VFT 前使用
  四层 semantic graph attention，让 multi-hop skeleton/edge 参与每层 attention
  与 feature update。

## 4. 首轮公平实验

| ID | 模型唯一变量 | 其余设置 |
|---|---|---|
| SG-M0 | VFT 前加入与主模型同深度的 position-only spatial Transformer | 完全沿用 H76 |
| SG-M1 | VFT 前加入完整 SGraFormer semantic graph encoder（position + 1--4 hop spatial graph + edge attention） | 完全沿用 H76 |

SG-M0 是论文对应的结构消融，用于排除“只是多了四层 Transformer/参数”的解释；
SG-M1 才是候选论文模型。两组同时从 scratch 训练，使用相同 seed0、真实 H36M
matched-H21 输入、H76 视角采样、优化器、学习率、epoch、loss 和严格
V2/V3/V4 评估。

首轮不做：蒸馏、temporal、bone loss、reprojection loss、额外数据增强、视角
采样调权、学习率 sweep、模块深度 sweep。

## 5. 判定标准

1. SG-M1 必须优于 H76，而不只是优于 SG-M0。
2. 首轮 seed0 若 V2/V3/V4 同向改善，并至少一个主指标改善超过约 `0.5 mm`，
   才补 seed1/seed2；最终以三种子均值和方差判断，不用单次幸运 seed 写结论。
3. 若 SG-M0 与 SG-M1 都退化，停止在这个结构上调学习率，说明 VFT 前增加
   joint-context 与 RUMPL ray 表示不兼容；转入 GraFormer 完整替换 PFT 的第二
   论文方向。
4. 若 SG-M1 明显优于 SG-M0 但仍未超过 H76，只保留为机制证据，不作为最终
   模型；后续仍先换论文模型，不靠训练超参掩盖结构失败。

## 6. 实现与运行状态（2026-08-09 08:37）

- 实现：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/semantic_graph_encoder.py`
- 接入：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- 测试：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/tests/test_semantic_graph_encoder.py`
- 启动器：`launch_SG_M0_M1_semantic_graph_20260809.sh`
- 输出根目录：
  `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/SGraFormer_preVFT_20260809`

实现审计：

- `off/position/full` 三种完整模型前向均输出有限的 `(B,17,3)`；full 模型反向
  中 90 个 semantic-graph 参数张量均获得有限梯度。
- position/full 都对 view permutation 等变；position-only 不依赖 ray 几何，
  full 对 ray 几何变化有响应。
- 新模块构造时保存并恢复 RNG；同 seed 下，SG-M0/SG-M1 与 H76 共有的 308 个
  state tensors 最大逐元素差均为 `0.0`，排除了下游 VFT/PFT 初始化漂移。
- SG-M0 参数量 `14,773,641`；SG-M1 参数量 `16,521,014`；H76 为
  `12,660,361`。两组均为全模型从 scratch 训练，不使用 adapter scope。

当前任务：

| GPU | tmux | 实验 | 状态 |
|---:|---|---|---|
| 0 | `cjy_sgm0_position` | SG-M0 position-only | epoch 0 训练中 |
| 1 | `cjy_sgm1_full` | SG-M1 full semantic graph | epoch 0 训练中 |

两组均为 20 epochs；结束后启动器自动运行严格 V2/V3/V4 Table-2，并把
checkpoint、训练日志和评估 JSON 写入上述挂载硬盘目录。

## 7. SG-M0/SG-M1 严格结果与结构结论（2026-08-09）

两组均已完成 20 epochs，并完成正式 H36M S9/S11、action-equal All-17 的
V2/V3/V4 严格评估。数值单位为 mm，越低越好。

| 方法 | V2 | V3 | V4 | 相对 H76（V2/V3/V4） |
|---|---:|---:|---:|---:|
| H76 baseline | 34.8163 | 30.4890 | 29.6913 | — |
| SG-M0 position-only pre-VFT | 40.5288 | 35.6152 | 35.0060 | +5.7125 / +5.1261 / +5.3147 |
| SG-M1 full semantic graph pre-VFT | 46.4494 | 37.3064 | 36.6286 | +11.6330 / +6.8174 / +6.9373 |

SG-M1 相对 SG-M0 仍退化 `+5.9205 / +1.6913 / +1.6226` mm。两组没有崩溃、
NaN 或模块未激活问题；新模块梯度有限且训练损失正常下降。SG-M1 后期训练损失
甚至略低于 SG-M0，但验证与严格测试更差，说明主要问题是表示/接入位置失配并伴随
过拟合，而不是没有训练收敛。

机制判定：RUMPL 的 VFT 依赖“同一 joint 的多视角 ray token”这一明确对应关系。
在 VFT 以前先用四层网络混合 17 个 joint，position-only 已使这层 ray identity
被纠缠并退化约 5 mm；继续加入 multi-hop skeleton 与 edge attention 又进一步
退化。因此本结果不否定 SGraFormer 本身，只否定“把其关节编码器适配到 world-ray
token 并置于 RUMPL VFT 前”这一融合位置。

按第 5 节预注册标准，本路线停止：不补 seed、不做学习率/深度/epoch sweep，也不
合并 checkpoint。下一模型方向应保留原 RUMPL VFT 直至同关节跨视角融合完成，再
尝试由 GraFormer 官方 GraAttention + ChebGConv 完整替换 PFT；这与此前 PFT 后
单层零初始化 graph residual 不同，属于有论文和官方代码支撑的完整模型替换。

严格结果目录：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/SGraFormer_preVFT_20260809/eval`

## 8. 第二模型方向：GraFormer 完整替换 PFT（2026-08-09）

依据为 GraFormer（CVPR 2022）论文及 Apache-2.0 官方代码。论文 Table 4 在
Human3.6M 上报告：删除重复 ChebGConv block 的 model-AT 为 `37.78` mm，完整
GraFormer 为 `35.17` mm；因此首轮直接复现这个论文内结构消融，不做参数搜索。
官方默认设置为 5 层、4 heads、dropout 0.25，每层按 GraAttention 后接二阶
Chebyshev 骨架图卷积。

本轮位置改为：

`原 RUMPL ray embedding -> 原 VFT -> GraFormer feature encoder -> 原 3-D head
-> 原 tri-anchor`

SG-M0/M1 失败后确认必须先保留逐关节跨视角 ray identity；因此 GraFormer 只在
VFT 已经完成同关节跨视角融合后处理 17 个 joint。它完整替换原 PFT，不是 PFT
后的单层零初始化 residual。

| ID | 模型唯一变量 | 论文对应 |
|---|---|---|
| GF-M0 | input/output ChebGConv + 5 层 GraAttention，不含层间 residual ChebGConv | Table 4 model-AT |
| GF-M1 | input/output ChebGConv + 5×(GraAttention + residual ChebGConv) | 完整 GraFormer |

输入/输出 ChebGConv 的通道从官方 2→96→3 适配为 RUMPL PFT feature
`256→256→256`（配置 DIM=128 的 ray embedding 与 confidence embedding 拼接
后为256维），使原 VFT、共享 3-D head 和三角化锚点保持不变；其余公式、
二阶 Chebyshev 多项式、LAM-Gconv 可学习邻接、模块顺序和官方默认层数均照官方
代码。首轮从 scratch、seed0、20 epochs，沿用 H76 的真实 H36M matched-H21
输入、视角采样、优化器、学习率、loss 和严格 V2/V3/V4 评估，不使用蒸馏。

实现与审计：

- 模块：`OpenRUMPL/RUMPL/lib/models/graformer_pft.py`
- 接入：`OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- 测试：`OpenRUMPL/RUMPL/tests/test_graformer_pft.py`
- 启动器：`OpenRUMPL_baseline_audit/launch_GF_M0_M1_graformer_pft_20260809.sh`
- 单元测试、完整模型前反向和有限梯度均通过；VFT 的 view permutation 最大误差
  为 `1.04e-6`（GF-M0）和 `8.34e-7`（GF-M1）。
- 相同 seed 下，H76/GF-M0 共有 164 个 state tensors、GF-M0/GF-M1 共有
  255 个 state tensors，最大逐元素差均为 `0.0`。
- 参数量：H76 `12,660,359`；GF-M0 `9,365,804`；GF-M1 `11,334,444`。
- 输出根目录：
  `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/GraFormer_PFT_20260809`

运行状态（2026-08-09 15:27）：

| GPU | tmux | 实验 | 状态 |
|---:|---|---|---|
| 0 | `cjy_gfm0_attention` | GF-M0 model-AT | epoch 0 训练中，首100 iterations 有限 |
| 1 | `cjy_gfm1_full` | GF-M1 full GraFormer | epoch 0 训练中，首100 iterations 有限 |

两组启动器会在20轮结束后自动执行严格 V2/V3/V4 Table-2，并写入上述输出根
目录的 `eval/`、`checkpoints/` 和 `completed/`。

### GF-M0/GF-M1 最终严格结果（2026-08-11 补录）

| 方法 | V2 | V3 | V4 | 相对 H76（V2/V3/V4） |
|---|---:|---:|---:|---:|
| H76 baseline | 34.8163 | 30.4890 | 29.6913 | — |
| GF-M0 GraAttention/model-AT | 42.3116 | 37.2964 | 34.7511 | +7.4953 / +6.8074 / +5.0598 |
| GF-M1 full GraFormer | 45.0645 | 35.9863 | 34.9645 | +10.2482 / +5.4973 / +5.2732 |

两种完整 PFT 替换均正常完成训练但显著退化。GF-M1 没有稳定优于 GF-M0，且
两者都远差于原 RUMPL PFT。因此“用单视角 2D-to-3D 的 GraFormer 关节建模完整
替换 RUMPL PFT”判定失败，不做参数搜索或多种子。SGraFormer pre-VFT 与
GraFormer post-VFT/PFT 两个论文结构方向至此均已严谨排除；后续转向保留 H76
融合器、在其输出的多视角子集假设上学习评分，而不是继续替换主干。

严格结果目录：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/GraFormer_PFT_20260809`

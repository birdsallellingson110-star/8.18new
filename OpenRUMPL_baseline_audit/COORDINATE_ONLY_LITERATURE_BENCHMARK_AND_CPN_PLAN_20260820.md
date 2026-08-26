# 坐标级论文对比档案与 CPN 输入路线（2026-08-20）

## 1. 用途与当前锚点

本文档固定保存只在 3D 阶段使用 2D 关键点坐标（可选置信度、相机和时序）
的方法、协议差异和后续 CPN 实验路线。后续论文表格、实验设计和代码实现均以本文
为核对入口，禁止把 RGB/热图特征方法的结果混入坐标级主表。

当前 HRNet 坐标级最好模型为 K96 limb-utility proposal：

| 方法 | 2D 输入 | 时序 | V2 | V3 | V4 |
|---|---|---:|---:|---:|---:|
| K96（当前） | HRNet 坐标+置信度+相机射线 | T=1 | 38.011 | 29.141 | 27.079 |

口径：H36M S9/S11、All-17、action-equal、absolute world MPJPE、所有相机组合
平均。正式结果位于：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/e2_pose_dsac_limb_proposal/seed0_30e_tmux/result.json`

## 2. CPN 坐标到底是什么

CPN 是 Chen et al., CVPR 2018 的 Cascaded Pyramid Network，是一个 top-down 2D
人体关键点检测器：先由人体检测器给 bbox，再由 GlobalNet 和 RefineNet 在裁剪图像
上预测每个关节的二维热图。

需要严格区分两个阶段：

```text
图像 -> bbox -> CPN -> 关节热图 -> 解码得到 (x,y[,score]) -> 保存离线坐标
                                                        |
                                                        v
                                      坐标级 3D 网络只读取这里
```

- CPN 内部确实使用热图，这是任何 heatmap-based 2D detector 的内部实现。
- 但 VideoPose3D、MTF、SGraFormer、SVTformer、HMVformer 等坐标级 3D 方法读取的
  是离线解码后的坐标；3D 网络不读取图像，也不读取完整热图。
- 因此采用 CPN 坐标仍属于“只输入 2D 坐标”的 3D 协议，不等于 AdaFuse/TransFusion
  那种在 3D 或跨视角模块中继续融合整张热图。

最常用公共文件为：

`data_2d_h36m_cpn_ft_h36m_dbb.npz`

其命名和内容为：

- `cpn_ft_h36m`：CPN 在 H36M 训练主体 S1/S5/S6/S7/S8 上微调；
- `dbb`：bbox 来自同样在 H36M 上微调的 Detectron；
- `positions_2d`：每条序列、每个相机的数组形状为 `(T,17,2)`；
- 官方公共 NPZ 默认只有 `x,y`，没有完整热图，也没有每关节 confidence。

MTF-Transformer 的官方工程另外读取 `score.pkl`/`vis_ada.pkl`，将每关节可见性或
置信度拼为第三通道。因此必须固定两条 CPN 子协议：

| 子协议 | 输入 | 可直接对齐的论文 |
|---|---|---|
| CPN-XY | 官方 NPZ 的 `(x,y)` | SGraFormer、SVTformer、HMVformer、ECTFormer、FusionFormer、PoseIRM 等 |
| CPN-XYC | `(x,y)` + 官方同源 `score.pkl` | MTF-Transformer/MTF-Transformer+ |

不得由当前 HRNet confidence 冒充 CPN confidence；若拿不到 MTF 同源 score，则
CPN-XYC 结果只能标为“our reconstructed confidence”，不能标为严格官方协议。

### 2.1 本机已有 MTF/CPN 数据审计

无需重新生成 CPN。挂载盘已经保存 MTF 官方 H36M 数据包：

`/mnt/data/cjydata/datasets/mtf_transformer_official/data`

- 总大小约 `2.2 GB`；包含 S1/S5/S6/S7/S8/S9/S11；
- `score.pkl` 约 `145 MB`，840 个 `subject_action.camera` 条目，每项形状为
  `(T,17)` 的 `float32`；
- 每个 `h36m_sub*.npz` 的 `positions_2d` 按 subject/action/4 cameras 组织；
- 本地 MTF 处理文件的每个关节不是公共 CPN NPZ 的纯两通道，而是 7 通道组合：
  `GT 2D (2) + predicted CPN 2D (2) + camera-space 3D (3)`；官方运行代码再从
  `score.pkl` 拼接第 8 个 visibility/confidence 通道；
- 例如 S1/Smoking 每相机数组为 `(2478,17,7)`，`run_h36m.py` 明确使用
  `[...,2:4]` 作为 CPN prediction、`[...,4:7]` 作为 camera-space 3D、
  `[...,7:8]` 作为拼接后的 score。

所以正式转换器必须显式取 `2:4`，不得误把前两维 GT 2D 当作检测坐标，也不得把
后 3 维 3D 泄露进输入。

本机还已运行过官方 MTF T=7 checkpoint。现有日志在：

`/mnt/data/cjyoutput/mtf_transformer_official/logs/official_t7_v2_v4_eval.log`

日志报告其自身配置下：T=1 时 V2/V4 为 `35.034/27.566`，T=7 时为
`33.845/26.651`。这证明 CPN 数据、score 和官方模型能够跑通，也显示时序在该
CPN/P1 模型里约改善 V2 `1.189 mm`、V4 `0.915 mm`。但该配置启用了多项
rotation/translation/scale alignment，必须在 P0 双评估器审计后才能作为论文正式
对比数字，不能直接与 absolute K96 相减。

## 3. 可直接比较的 absolute 坐标级结果

以下来自 GBT 的相同 absolute world MPJPE、所有相机组合平均协议，和当前结果
最接近：

| 方法 | 输入 | T | V2 | V3 | V4 | 相对 K96 |
|---|---|---:|---:|---:|---:|---|
| K96 | HRNet | 1 | 38.011 | 29.141 | 27.079 | — |
| GBT | HRNet-W32 COCO | 9 | 36.8 | 30.4 | 26.0 | V2/V4 更好；V3 更差 |
| GBT | ResNet-152 H36M FT | 9 | 29.9 | 24.4 | 22.7 | 三列更好 |
| Algebraic Triangulation | ResNet-152 H36M FT | 1 | 51.1 | 23.4 | 19.1 | V3/V4 更好，V2 崩溃 |
| Algebraic Triangulation | HRNet | 1 | 120.7 | 50.9 | 44.2 | 三列更差 |

## 4. CPN/坐标级但不同评估协议的较低结果

下表主要采用 H36M 标准 Protocol #1：预测和 GT 先去 pelvis/root，通常使用固定
四视角和 CPN，和 absolute world MPJPE 不可直接横向宣称胜负。

| 方法 | 3D 网络输入 | T | V2 | V3 | V4/主结果 | 备注 |
|---|---|---:|---:|---:|---:|---|
| CV-UGCN | CPN 坐标+相机 | 1 | 27.4 | — | — | 特定两相机对，非 all-pair absolute |
| Progressive Multi-View Fusion | CPN-XY | 27 | 31.4 | 30.0 | 26.8 | 无相机 |
| MTF-Transformer | CPN-XYC | 27 | 33.26 | 28.27 | 26.21 | 无相机、可变视角 |
| MTF-Transformer+ | CPN-XYC+相机约束 | 27 | — | — | 25.8 | 主表四视角 |
| SGraFormer | CPN-XY | 27 | 32.0 | 31.1 | 27.6 | V2 数值低；V3/V4 不如 K96 absolute 数字 |
| HMVformer | CPN-XY | 27 | — | — | 26.8 | 固定四视角 |
| FusionFormer | CPN-XY | 27 | — | — | 26.6 | 固定四视角 |
| PoseIRM | CPN-XY | 27 | — | — | 25.1 | camera-free、相机域增强 |
| SVTformer | CPN-XY | 27 | 30.6 | 29.1 | 26.0 | 空间-视角-时序轴分解 |
| ECTFormer | CPN-XY | 27 | 32.2 | 29.9 | 26.5 | CNN-Transformer 融合 |
| UniCodebook | CPN-XY | 27 | — | — | 26.0 | 离散 2D/3D pose prior |
| UPose3D | 坐标+RLE 分布不确定性 | 1/27 | — | — | 26.9/26.4 | 超出标量 confidence，单列说明 |

## 5. 不进入坐标级主表的方法

AdaFuse、Learnable Triangulation 完整网络、Epipolar Transformer、TransFusion、
Volumetric、MvP、MVGFormer 和 2026 Algebraic Priors 等在 3D/跨视角阶段继续使用
图像、热图、局部图像描述子或体素特征。它们可作为相关工作或模块灵感，但不能放进
“只输入坐标/置信度”的公平主表。

## 6. 是否值得建立 CPN 线

结论：值得，但 CPN 是一条独立的外部公平协议，不替换现有 HRNet absolute 主线。

CPN 线的价值：

1. 大量开源坐标级论文共享同一个 VideoPose3D NPZ，输入差异最少，适合公平复现；
2. CPN 在 H36M 上微调，域内 2D 坐标通常比 COCO-only HRNet 更适配 H36M；
3. 可以把“输入质量”和“3D 模块能力”分开：同一 CPN 缓存下模型差异才真正来自
   跨视角/骨架/时序模块；
4. 论文可同时报告 GBT 风格 HRNet absolute 表和 CPN Protocol-1 表，覆盖两类
   文献，而无需把热图送入 3D 网络。

风险和边界：

- CPN 在 H36M 上微调，域内精度高但跨数据集泛化可能弱于 COCO-only HRNet；
- 换 CPN 本身不是模型创新，只是输入协议和公平对比基准；
- CPN P1 数字不能直接与当前 absolute 数字比较；
- 旧论文固定四视角/固定相机 ID 的模型，不能直接证明任意相机泛化；
- 最终模型必须同时保留 HRNet/camera-generalization 结果，否则论文故事会变成
  只在 H36M 固定相机上刷分。

## 7. 失败经验约束

后续借模块不能重复已经判定失败的接法：

| 已做失败 | 结论 | CPN 线约束 |
|---|---|---|
| SGraFormer 四层图编码直接插到 RUMPL VFT 前 | V2/V3/V4 退化约 11.6/6.8/6.9 mm | 不再把 joint mixer 放到逐关节 ray 对应建立之前 |
| GraFormer 完整替换 PFT | V2/V3/V4 退化约 10.2/5.5/5.3 mm | 不再整体替换已有效的 RUMPL PFT |
| 重复 ray-view attention | 与 VFT 信息重复，接近零收益或退化 | 只在 observation scorer/utility 中使用，不再串联第二个 VFT |
| 普通后端 T=9 pose refiner/MixSTE | clean 仅约 0.1--0.2 mm，跨主体易退化 | 时序不得只做最终 3D 平滑 |
| learned 2D/ray correction | 训练主体有效、S9/S11 退化 | 不把 CPN 换输入变成新的 2D correction 网络 |
| 固定骨长/DLT/IRLS 替换主干 | 显著弱于 RUMPL | 几何解只作 anchor/candidate，不替换主输出 |

因此正确方式不是“完整堆叠 MTF+SGra+SVT”，而是先在各自官方 CPN 协议复现，
用同一数据确认模块净增益，再把一个最小模块放进 RUMPL 尚未覆盖的位置。

## 8. 分阶段执行计划

### P0：冻结数据与双评估器

1. 下载官方 `data_2d_h36m_cpn_ft_h36m_dbb.npz`，记录 URL、SHA256 和文件大小；
2. 获取 MTF 官方 H36M data 包，审计 `score.pkl` 与 CPN 序列、帧数和相机顺序；
3. 建立不可修改的 CPN 缓存 manifest；
4. 同一预测同时输出：
   - `absolute/action-equal/all-combination`：延续我们的 GBT 主表；
   - `P1/root-relative/frame-weighted/fixed-4-view`：对齐 CPN 论文；
5. 检查 S1/S5/S6/S7/S8 训练、S9/S11 测试、17 joints、序列排除、下采样、
   相机顺序完全一致。

通过标准：同一预测由两个评估器转换后差异可由 root translation 和聚合方式完全
解释；不得存在单位、joint mapping、帧错位。

### P1：零训练输入审计

在同一 S9/S11 帧上比较 HRNet 和 CPN：

- 2D pixel MPJPE、逐关节误差、每相机误差；
- confidence/score 校准（仅 CPN-XYC）；
- DLT/triangulation absolute MPJPE；
- 六个 V2 相机对、V3/V4；
- CPN 相对 HRNet 的改善究竟来自 bbox、关节坐标还是特定相机；
- CPN 与 GT 的系统偏差、左右关节交换和尾部误差。

只有 CPN 在相同三角化器下明显降低 2D 或 3D 误差，才把“换 CPN 可提升 absolute
精度”作为假设；否则 CPN 只用于公平对比，不期待它自动改善主线。

### P2：四个必要基线

| ID | 输入 | 模型 | 目的 |
|---|---|---|---|
| C0 | CPN-XY | 官方 MTF/SVT 类最小坐标 lifting baseline | 验证 P1 管线 |
| C1 | CPN-XY | 当前 RUMPL/K96，confidence 固定为 1 | 测纯坐标迁移 |
| C2 | CPN-XYC | 当前 RUMPL/K96，使用同源 CPN score | 测 confidence 净贡献 |
| C3 | HRNet-XYC | 当前冻结 K96 | 原主线控制，38.011/29.141/27.079 |

每组必须同时报告 absolute 与 P1；C1/C2 只换输入，模型、训练预算、采样和评估均
不变。这样可回答精度差异来自 detector 还是 3D fusion。

### P3：先严格复现开源坐标模型

优先级按“代码完整度、输入匹配、未重复失败”排序：

1. **MTF-Transformer**：官方 CPN-XYC、官方 checkpoint，复现 T=1/T=27、
   V2/V3/V4；审计 CAA、Relative Attention、Random Block Mask 的独立贡献。
2. **SVTformer**：官方 CPN-XY、T=27 checkpoint，复现 30.6/29.1/26.0；重点
   读取其 S->V->T 顺序消融，不直接把整网叠到 RUMPL。
3. **SGraFormer/HMVformer**：只作为 CPN 官方复现和结构对照。由于相同图模块
   已在 RUMPL 上失败，不再重复插入实验。
4. **PoseIRM/UniCodebook**：在前两项复现后再评估。PoseIRM 主要是训练域增强，
   UniCodebook 引入额外 3D/AMASS prior，必须单独标明额外训练数据。

官方模型若不能在官方 CPN/P1 口径接近论文数字，停止模块迁移，先解决数据和评估。

### P0/P1 已完成记录（2026-08-20）

官方 MTF 数据已转换为 RUMPL 原生 pkl，严格按 H36M 标准主体划分：训练
S1/S5/S6/S7/S8（312,188 条相机记录），验证/测试 S9/S11（8,468 条记录，
2,117 个四相机同步组）。转换器为
`build_mtf_native_cpn_rumpl_pkl_20260820.py`，输出目录为
`/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict/`。模型输入只取官方
`positions_2d[..., 2:4]` 的 CPN 坐标；C1 置信度固定为 1，C2 使用同源
`score.pkl`，没有把 MTF 的 GT 3D 或其余通道泄露到输入。

双评估器审计输出：
`/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict/cpn_dual_audit.json`。
在同一 2,117 个四视角组上，C1/C2 坐标逐元素完全一致（最大差 0 px），
CPN 重投影误差均值 6.686 px、P95 20.962 px，跨相机世界坐标一致性
0.00052 mm。零训练 confidence-weighted DLT（C1/C2 因未使用 score 而相同）为：

| 输入 | absolute world All-17（action-equal） | P1 root-relative（action-equal） |
|---|---:|---:|
| CPN C1/C2，四视角 DLT | **35.504 mm** | **31.794 mm** |

这一步证明 CPN 坐标和相机/单位/关节映射是可用的；它不是 RUMPL 学习结果。
HRNet 参考使用的是另一套采样帧，不能与 CPN 作逐帧 detector 优劣结论，只作为
管线交叉检查保留。

随后进行了严格的“只换输入、不改模型、不重训”屏幕：固定 H76 checkpoint、
固定 tri-anchor/centered-ray/Plücker 配置和评估代码，只替换 C1 或 C2 pkl。双
评估器包括 RUMPL 原生 absolute/relative 日志，以及独立的
`eval_h36m_table2.py` action-equal All-17/KP* 统计。结果如下（All-17，
action-equal）：

| frozen H76 + 输入 | V2 | V3 | V4 |
|---|---:|---:|---:|
| C1：CPN-XY，conf=1 | 63.024 | 54.537 | 52.401 |
| C2：CPN-XYC，官方 score | 79.326 | 64.881 | 54.302 |

产物目录分别为
`frozen_h76_eval_rerun2/C1/` 和 `frozen_h76_eval_rerun3/C2/`。
这一屏幕的解释是“输入分布迁移”而不是 CPN 无效：H76 是在 HRNet 置信度/误差
分布上训练的，直接喂 CPN 坐标没有适配；C2 的 score 语义更进一步改变了融合权重，
所以冻结模型反而退化，不能拿它冒充 CPN 训练 baseline。下一步必须保持完全相同
RUMPL 结构和训练预算，分别在 C1、C2 上从头训练，再与 HRNet 控制线比较。

同结构重训已于 2026-08-20 启动，脚本为
`launch_cpn_c1_c2_train_20260820.sh`，输出目录为
`/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict/trained_same_model/`。
C1/C2 均使用 20 epoch、前 8 epoch 固定 K=2、随后 3:1:1 视角采样；当前两条线均已
通过数据加载并在 epoch 0/1 正常训练。最终结论以该重训后的三列结果为准。

### P2 C1/C2 同结构重训结果（2026-08-21）

两条线均已完成 20 epoch，并使用同一 `eval_h36m_table2.py` 输出 action-equal
All-17：

| 输入/模型 | V2 | V3 | V4 |
|---|---:|---:|---:|
| GBT-style HRNet + H76（现有控制） | 46.227 | 31.334 | 27.964 |
| C1：CPN-XY + 同结构重训 | 46.973 | 35.973 | 33.364 |
| C2：CPN-XYC(score) + 同结构重训 | 53.376 | 37.802 | 35.627 |

相对现有 HRNet 控制，C1 为 `+0.746/+4.639/+5.400` mm，C2 为
`+7.149/+6.468/+7.663` mm；因此在当前 RUMPL/H76 融合器和官方 CPN 输入下，
CPN 没有带来净提升，C2 的 score 还会进一步拖累融合。结果文件位于：
`trained_same_model/C1/eval/` 和 `trained_same_model/C2/eval/`。

注意：CPN native validation 在 RUMPL 的 damaged-frame 过滤后为 1,976 个组，
现有 GBT 控制为 2,021 个组（CPN MTF 原生序列采样 stride=65，GBT 缓存为 stride=64），
所以这张表是“同模型/同训练预算”的输入对照，但不是逐帧完全相同的最终 detector 主表。
结论方向已经明确；若写论文，下一步应先生成 stride=64 的 CPN 验证缓存，在共同帧数上
复核一次，再决定是否彻底放弃 CPN 主线。

### P4：只迁移有净贡献且不重叠的模块

第一优先：MTF 的 confidence-aware view masking，不复制完整 MTF。

```text
原 RUMPL ray/VFT/PFT/K96 candidates 保持不变
       + CPN confidence/visibility
       + 随机 view-block mask（训练）
       + observation-conditioned candidate utility residual（零初始化）
                         -> K96 soft candidate fusion
```

理由：当前瓶颈是坏 V2 pair 和负视角贡献；MTF 的 confidence 和随机 view mask
直接作用于视角可靠性与可变视角训练，而不会再次破坏逐关节 ray identity。

对照实验：

| ID | 唯一变化 |
|---|---|
| M0 | C2：CPN K96 基线 |
| M1 | 只加官方分布的 random view-block mask |
| M2 | 只加 CAA 式 confidence gate 到 candidate utility logits |
| M3 | M1+M2 |

第二优先：SVT 的轴顺序只进入 scorer，不改 backbone。

- 对每个 K96 hypothesis 构造 `joint x view x time` 观测残差；
- scorer 严格按 Spatial -> View -> Temporal 顺序；
- 输出只是不确定性/候选效用 residual，不直接回归 3D；
- T=1 时严格退化为 M3；T=9/27 只在 M3 已通过 clean 门槛后启动。

这与过去失败的最终 3D pose temporal refiner不同：时间在候选选择和视角压缩前参与
判断，但仍不移动 2D 坐标、不替换 RUMPL 主干。

第三优先：root 与 articulation 双头，而不是再堆图网络。

- absolute root translation 保留 RUMPL/K96 几何路径；
- root-relative articulation 使用 CPN 文献的 P1 supervision；
- 最终 `X_abs = root_abs + pose_rel`；
- 同时优化小权重 absolute 与 root-relative loss，先做双头恒等初始化；
- 用诊断分别验证 pelvis error 和 non-root root-relative error。

这是把 CPN/P1 模型擅长的相对人体结构与 RUMPL 擅长的绝对相机几何有机组合，
比完整替换 PFT 更符合已有失败经验。

### P5：训练和选择规则

- 结构筛选只用训练主体严格留一主体 S8；S9/S11 只做最终一次报告；
- 首轮 3 epoch gate，S8 三列均值至少改善 0.15 mm、V2 不退化才进入正式训练；
- 正式训练固定 epoch 后使用 S1/S5/S6/S7/S8 重训；
- 第一个 seed 通过后才补第二/第三 seed；
- 不允许按 V2/V3/V4 拼 specialist checkpoint；
- 每个模型必须报告六个 V2 相机对、root、root-relative、逐关节和 Negative View Rate；
- 模块实验完成前不扫学习率、深度、温度和多损失权重。

## 9. 推荐实验优先级与成功门槛

| 优先级 | 实验 | 预计用途 | 继续门槛 |
|---:|---|---|---|
| 1 | P0/P1 CPN 数据与输入审计 | 确认 CPN 是否真优于 HRNet | 管线零错位；出完整 2D/tri 表 |
| 2 | 官方 MTF、SVT 复现 | 建立可信外部 baseline | 接近论文/官方 checkpoint |
| 3 | C1/C2 CPN-RUMPL/K96 | 得到公平 CPN baseline | P1/absolute 双表稳定 |
| 4 | M1/M2/M3 view mask + CAA utility | 修复坏视角/坏相机对 | absolute 三列均值改善 >=0.3 mm，V2 不退化 |
| 5 | root/articulation 双头 | 同时利用相机几何与 P1 pose prior | root 和 articulation 至少一项显著改善且另一项不退化 |
| 6 | scorer 内 S->V->T | clean+遮挡时序 | clean >=0.5 mm；遮挡 >=1 mm |

最终论文目标必须分两张主表：

1. **HRNet absolute/all-combination**：直接对齐 GBT，目标低于 `36.8/30.4/26.0`；
2. **CPN root-relative/fixed-four-view**：对齐 MTF/SVT/PoseIRM 等，至少低于
   `25.1--26.0 mm` 的强方法区间，或在 V2/可变视角/遮挡鲁棒性上形成明确优势。

若 CPN 线只改善 P1、不改善 absolute，它仍可作为外部公平表和 articulation 消融；
不能据此替换 HRNet 主线或宣称相机泛化提升。

## 10. 主要来源

- CPN: Chen et al., *Cascaded Pyramid Network for Multi-Person Pose Estimation*, CVPR 2018.
- VideoPose3D official dataset protocol and CPN NPZ.
- MTF-Transformer official code: `/home/lixiaob/cjy/reference/MTF-Transformer`.
- SVTformer official code: `/home/lixiaob/cjy/reference/SVTformer`.
- SGraFormer official paper/code: `/home/lixiaob/cjy/reference/SGraFormer_official`.
- GBT local paper: `/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`.
- ECTFormer local paper:
  `/home/lixiaob/cjy/reference/transformer/(2026)ECTFormer_Efficient_CNN-Transformer_Network_for_Uncalibrated_Multiview_3-D_Human_Pose_Estimation.pdf`.

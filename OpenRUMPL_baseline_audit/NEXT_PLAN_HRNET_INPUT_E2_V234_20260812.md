# HRNet 图像证据与 E2 V2/V3/V4 完整化计划（2026-08-12）

## 1. 当前结论

当前所谓“HRNet 修正”并不修改 HRNet 参数，也不直接修改 HRNet 输出的二维坐标。
准确名称应为：

> **冻结 HRNet 图像特征辅助的 H76/RUMPL 三维候选修正器**

流程是：H76 子集候选投影到各相机 crop，在冻结 HRNet stage-4 高分辨率特征图
采样 5×5 patch，融合图像特征、射线和当前 3D query，最后预测有界 3D residual。
目前它是受 MVGFormer、Epipolar Transformer 启发的轻量 probe，不是这两篇论文
的严格复现：没有沿极线搜索对应、没有显式预测 2D offset/covariance，也没有经过
可微三角化迭代更新。

当前正式数字（H36M S9/S11，action-equal All-17 absolute MPJPE）：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76/RUMPL | 34.8163 | 30.4890 | 29.6913 |
| HRNet feature + pairwise geometry bias，seed0 | 32.8802 | 29.2691 | 28.3869 |
| HRNet feature + pairwise geometry bias，seed1 | 33.1667 | 29.6634 | 28.7333 |
| 固定 RIGR seed0 → E2 重训，2 个 E2 seed 均值 | — | **28.5553** | **27.8078** |

级联相对 H76 在 V3/V4 分别下降 1.934/1.884 mm。最后一行只覆盖两枚 E2 seed
和一枚固定 RIGR seed0；还不是两条完整 pipeline seed。

## 2. 公平比较边界

当前结果不能直接写成“与所有 HRNet 输入论文完全公平”，原因有两项：

1. 当前 H76 稀疏输入实际为 `HRNet-W32 COCO → A1D/H21 → H76`，不是原始
   HRNet 坐标/置信度。
2. 当前图像和 HRNet 特征 crop 的框来自 H36M 注释 H5 的 `center/scale`，不是
   GBT 明确使用的 YOLOX proposals。

论文结果必须拆成两类表：

### 表 A：稀疏 2D-only

- 输入只能是冻结 detector 的二维坐标、置信度和相机参数；
- 对照 GBT、GHT、代数/可学习三角化、RUMPL；
- 原始 HRNet 和 A1D/H21 增强 HRNet 分列报告；
- E2-V234 的主消融放在此表。

### 表 B：RGB/中间图像特征

- 明确标注使用 intermediate feature；
- 对照 Epipolar Transformer、AdaFuse、Learnable Triangulation、MVGFormer；
- backbone、是否 H36M fine-tune、box 来源、输入尺寸、T、指标和相机组合必须逐项标注。

不能把 COCO-only HRNet + 注释框、H36M-finetuned detector + GT box、COCO-only
HRNet + YOLOX 三种协议混在一列只写“HRNet”。

## 3. 是否达到极限

### E2

当前 22 候选池与现有 loss/模型容量已经局部饱和：延长训练、改深度、温度、
source bias、dropout、bone loss、Gumbel、简单几何偏置和 top-k 均未产生稳定增益。
但不是全局极限：RIGR→E2 候选池 oracle 为 V3 20.94、V4 17.83 mm，说明主要
瓶颈仍是候选可辨识性和候选覆盖，不是数值上没有空间。

E2 当前只训练 V3/V4 是论文完整性弱点。它源于早期实验专门诊断“新增视角反而
变差”，不是理论限制。当前每个 V2 task 只有两个候选：对应子集 H76/RIGR 和
明显较差的裸 pairwise ray 候选。零训练诊断如下：

| V2 候选池 | baseline | 裸 pairwise | 等权平均 | 逐关节 oracle |
|---|---:|---:|---:|---:|
| 原 H76 候选池 | 34.816 | 52.352 | 39.778 | 31.213 |
| RIGR 候选池 | 32.880 | 52.352 | 38.423 | **29.142** |

因此先造出有互补性的 V2 候选，再把 E2 改为统一 V2/V3/V4 训练；直接把现有两个
候选塞进训练很可能只会学恒等映射。

### HRNet 图像特征修正器

远未达到结构极限。当前只是 balanced 20k/78,047 组、冻结 HRNet、单尺度 32 通道、
一次 5×5 局部采样和一次 unconstrained 3D residual。它证明“图像证据有效”，但
没有真正实现论文中最有价值的跨视角对应和几何闭环。简单 gate、quality pooling、
patch attention、几何标量拼接已经局部失败，不再重复调这些小模块。

## 4. 论文支持的输入处理路线

| 来源 | 输入处理/融合 | 适合借鉴的部分 | 协议注意 |
|---|---|---|---|
| GBT (FG 2024) | 图像去畸变；COCO HRNet-W32；YOLOX box；2D 坐标+置信度；随机 2 视角；scene centering、synthetic views、token dropout | 建立 raw HRNet+YOLOX 的稀疏输入公平线；随机子集训练 | T=9，不是纯单帧；不能与当前注释框+A1D/H21直接混表 |
| Epipolar Transformer (CVPR 2020) | 在特征层沿极线搜索/匹配，把邻视角证据送回当前视角再解码 2D | 真正的 epipolar correspondence，而非普通 view attention | RGB feature 方法，检测器训练协议需单列 |
| AdaFuse (IJCV 2021) | 保留完整热图；沿极线聚合；外观质量+几何一致性产生逐关节视角权重 | full-heatmap epipolar fusion 和 joint-view reliability | 其强结果常使用目标数据集训练的 2D detector |
| Learnable Triangulation (ICCV 2019) | soft-argmax；逐相机/逐关节置信度；加权可微三角化；另有中间特征 3D volume | 让图像分支预测 2D offset/uncertainty，再经过几何求解 | COCO 预训练后还可能联合 MPII/H36M fine-tune、使用 GT box |
| MVGFormer (CVPR 2024) | 3D query 投影；图像特征更新 2D 点和 confidence；学习外观模块与无学习几何三角化交替迭代 | 作为下一代 RIGR 的首选结构，官方代码可直接核对 | 官方公开重点为 CMU/多人；迁移 H36M 时要保留我们的单人协议 |
| UPose3D (ECCV 2024) | RLE 预测二维分布/不确定性；bbox normalization；跨视角和时序 pose compiler | 2D covariance/NLL、尺度归一和异常点鲁棒性 | 主训练依赖合成多视角动作，不作为我们的真实数据主协议 |
| UDP (CVPR 2020) | 连续坐标一致的 affine、flip、heatmap encode/decode | 审计半像素、crop→feature 坐标和 flip 对齐 | 若改 detector 解码，必须对所有比较方法使用同一版本 |

## 5. 下一轮执行顺序

### P0：先锁公平协议与输入坐标（必须先做）

1. 输出 detector/input manifest：HRNet config+SHA、checkpoint+SHA、box 来源、
   图像是否去畸变、crop affine、UDP、flip-test、joint mapping、输入分辨率。
2. 用点阵/关键点做完整 round-trip 测试：原图 → MMPose affine → 96×72 特征图
   → 原图；检查当前 `align_corners=True` 和半像素约定，目标误差 <0.05 feature px。
3. 建三个不混用的缓存：
   - `P_raw_yolox`：原始 COCO HRNet-W32 + YOLOX；
   - `P_raw_annbox`：原始 COCO HRNet-W32 + 当前注释框；
   - `P_enhanced_annbox`：A1D/H21 + 当前注释框（当前内部上限线）。
4. 三个协议都报告 Algebraic Triangulation 和 H76/RUMPL，避免将输入提升误认为
   新融合器提升。

Go/No-Go：如果 box/坐标修正改变 MPJPE ≥0.3 mm，所有后续模型基于修正缓存重跑；
否则固定 manifest，不再反复改输入协议。

### P1：补齐 E2-V234

先做无训练候选诊断，再训练：

1. V2 候选扩展：H76、RIGR、confidence-weighted DLT、IRLS、RANSAC/soft-medoid，
   以及同一双视角观测的 uncertainty perturbation hypotheses；全部只读同一 2D 输入。
2. 只保留满足以下至少一项的候选：自身均值不比 baseline 差太多，或逐关节 oracle
   额外降低至少 0.3 mm；去掉纯重复候选。
3. 将 `TASK_COMBINATIONS` 扩展到全部 V2/V3/V4；按 view count 和 action 均衡采样，
   加 view-count embedding；训练选择指标改为 V2/V3/V4 平均，而不是只看 V3/V4。
4. 严格对照：V34 specialist、V234 universal、V2 specialist。主表使用 universal；
   specialist 只作诊断。

GPU 分配：GPU0 跑 V234 universal seed0；GPU1 跑 V2 specialist/候选消融。首轮通过后
再同时跑 universal seed1 和 seed2。

成功门槛：V2 相对 32.880 至少降低 0.3 mm，且 V3/V4 相对 28.555/27.808 回退
不超过 0.1 mm。否则 E2 保留为 V3/V4 消融，不再称“任意视角数统一模块”。

### P2：把轻量 HRNet residual 升级成几何闭环

优先实现官方代码可核对的 **MVGFormer-style 单人迭代模块**：

1. H76/RUMPL 提供初始 3D query；
2. 投影到各视角，使用精确 affine 在 HRNet 多尺度特征采样；
3. appearance head 输出每关节/每视角 2D offset、confidence/covariance；
4. 使用 confidence-weighted differentiable triangulation 更新 3D；
5. 重复 1–2 层；保留 H76 residual/identity skip；
6. 与当前 direct-3D-residual 做同输入、同 20k、同训练轮数的单变量比较。

并行第二路线只做一个强对照：AdaFuse/Epipolar Transformer 风格的 full-heatmap
沿极线融合。过去失败的 local top-K 不等于 full epipolar heatmap fusion，不能据此
否定该路线。

GPU 分配：GPU0 为 MVGFormer-style 1-layer；GPU1 为 full-heatmap epipolar/AdaFuse
对照。20k 筛选达到 ≥0.5 mm 增益后才扩到全部 78,047 训练组；否则停止，不做层数、
学习率大扫参。

### P3：完整复验和论文口径

1. 用 RIGR seed1 导出候选并重训 E2，补齐两条完整 pipeline seed；最终至少 3 seed。
2. raw-HRNet 稀疏表、enhanced-input 消融表、image-feature 表分开。
3. 报 V2/V3/V4 所有组合、均值/标准差、参数量、FLOPs、速度；T=1 和后续 T=9 分开。
4. 当前可暂定的模型故事为：

   `RUMPL camera-generalizable ray hypotheses → image-conditioned geometry-consistent
   candidate refinement → counterfactual per-joint hypothesis utility fusion`。

   在完成 P1/P2 前，不把简单 3D residual 或微小 geometry bias 单独包装成主创新。

## 6. 执行时禁止重复的失败路线

- 不再重复普通 temporal residual/MixSTE 后处理；
- 不再扫简单 gate、quality pooling、patch attention、几何标量拼接；
- 不再用 local heatmap top-K 冒充 AdaFuse/epipolar full-heatmap；
- 不蒸馏；
- 不以 S9/S11 测试结果选择 checkpoint 或超参数；
- 不把不同 detector fine-tune、box、T、指标的论文数字放在同一“公平”列。

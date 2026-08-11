# Single-frame GBT × RUMPL 架构与实验计划（2026-07-27）

## 1. 目标与约束

目标：在不使用时序模块的前提下，尽可能忠实地迁移 Geometry-Biased Transformer（GBT）中真正有效的部分，并适配 RUMPL 的单帧、多视角绝对 3D 姿态任务。

约束：

- 不加入时间帧、temporal encoding 或 temporal attention；
- 正式结果只在 clean CMU MMPose V2–V5 上与 RUMPL baseline 比较；
- 不使用遮挡结果作为主比较；
- 保持任意视角数和视角排列不变性；
- 所有训练与评测输出继续写入 `/mnt/data/cjyoutput/baseline_reaudit_20260722/`。

## 2. 为什么单帧仍可使用 GBT

GBT 的 confidence/geometry bias 定义在 encoder self-attention 上，与时间维度没有必然绑定。去掉时间后，token 集合从 `T×J×V` 变为 `J×V`：

`A_l = softmax(QK^T / sqrt(d) + eta_l^2 M_conf - gamma_l^2 M_dist)`

仍然保留：

- 同相机不同关节的结构关系；
- 同关节跨相机的射线一致性；
- 低置信度 key 抑制；
- 几何离群观测抑制；
- 对视角顺序和相机数量的泛化。

不加入伪时序，也不复用此前无法迁移的 temporal 数据管线。

## 3. 当前 RUMPL 与目标结构

当前：

`ray(d, camera_center, conf) → per-joint VFT(12) → fusion token → PFT(12) → xyz`

目标适配版：

`Plücker ray token → global biased encoder(3) → joint-query decoder(2) → short PFT(4) → xyz`

其中：

- global encoder 完全在 `J×V` token 上工作；
- bias 只加入 encoder，与论文一致；
- 不再使用 fusion token；
- learned joint queries 直接 cross-attend 全部 biased ray tokens；
- 保留较短 PFT，作为 RUMPL 已验证的身体结构先验；
- 纯 GBT（不带 PFT）作为结构对照。

## 4. 输入与 bias 的忠实实现

### 4.1 Ray token

对每个 2D detection 构造单位方向 `d` 和相机中心 `o`：

`m = o × d`

使用 Plücker ray：

`r = [d, m]`

采用论文的 harmonic encoding（15 frequencies）后线性投影到 256 维。加入 learned joint embedding，不加入 camera-id/view-id embedding，保持对相机排列的等变性。

主模型中不再把 confidence concat 到 ray embedding；confidence 只通过 `M_conf` 进入 attention。这样可以更忠实地验证论文 bias，避免 confidence 信息重复。

保留一个 `ConfConcat` 对照，判断移除原 RUMPL confidence 输入是否损失信息。

### 4.2 Confidence bias

对所有 query 重复 key confidence：

`M_conf[q,k] = confidence[k]`

每层一个非负参数：

`eta_l^2`

不按 head 设置参数，先严格遵循论文。

### 4.3 Geometry bias

在所有 `J×V` rays 间计算 Plücker line distance：

- 同相机 rays 在相机中心相交，距离为 0；
- 同关节跨相机的正确 rays 应接近相交；
- 错误 detection 对应的 ray 通常与其他观测不一致。

每层一个非负参数：

`gamma_l^2`

需要同时实现两种距离：

1. `raw`：论文原始距离；
2. `normalized`：适合当前“米”制输入：

   `d_norm = d / (median_nonzero(d) + eps)`

或对每个样本使用 robust MAD 标准化。normalized 版本是主实验，raw 版本用于判断是否复现此前尺度过小的问题。

### 4.4 Mask

无效/缺失 token 使用 attention key mask，而不是将 token 置零后仍允许其他 token attend。clean 主评测不制造额外遮挡，但正确 mask 是架构必要条件。

## 5. 三层实验路线

### Route A：低风险 global-bias adapter

保留原 RUMPL VFT/PFT，在 ray embedding 后插入 2 层 `J×V` global biased encoder：

`x_out = x + gate × global_encoder(x)`

- output projection 和 gate 零初始化；
- 从 R5 checkpoint 初始化；
- 初始预测必须与 R5 bitwise/数值等价；
- 第一阶段只训练 global encoder、bias scale 和 gate；
- 第二阶段解冻 VFT/PFT 最后若干层。

实验：

- `A0`：已有 J1e，无 bias，作为结构对照；
- `A1`：global confidence-only；
- `A2`：global confidence + normalized geometry；
- `A3`：A2 + raw geometry，对照距离归一化。

目的：最快判断 bias 在正确的全局位置是否有效。

### Route B：GBT 替换 VFT，保留短 PFT（主架构）

移除 12 层 per-joint VFT 和 fusion token：

1. 3 层 global biased encoder；
2. 2 层 learned joint-query cross-attention decoder；
3. 4 层 PFT；
4. 原 3D head。

实验：

- `B0`：无 bias；
- `B1`：confidence-only；
- `B2`：confidence + raw geometry；
- `B3`：confidence + normalized geometry。

这是最推荐的最终结构，因为：

- bias 位置和论文一致；
- query decoder 解决当前 fusion token 不受 geometry 直接控制的问题；
- 短 PFT 保留 RUMPL 的身体结构建模；
- 参数量和计算量不会比原 12+12 Transformer 更大。

### Route C：纯 single-frame GBT

结构：

`Plücker encoder(3) → joint-query decoder(2) → xyz head`

实验：

- `C0`：无 bias；
- `C1`：confidence + normalized geometry。

用途不是优先追求最好结果，而是回答：

“论文主体在我们的单帧数据协议上能否独立工作？”

如果 C1 明显优于 C0，说明 bias 可复现；如果 B3 又优于 C1，说明短 PFT 提供了额外人体先验。

## 6. 训练策略

### Route A

- R5 初始化；
- epoch 0–2：只训练新 global encoder、bias scale、gate；
- epoch 3–10：解冻 VFT 后 4 层和 PFT 后 4 层；
- 若验证仍下降，再完整训练至 20 epoch；
- gate 初始 0，防止新模块一开始破坏 baseline。

### Route B/C

- ray embedding可尝试从 R5 输入投影部分初始化；
- decoder、global encoder 重新初始化；
- B 的短 PFT/head从 R5相应参数初始化；
- 先保持当前 random V2–V5 训练协议，保证与 baseline 公平；
- 20 epoch 主实验；
- winner 再追加 30 epoch 和第二 seed；
- 暂不加入 token dropout、scene centering和时序，避免同时改变过多因素。

## 7. 实验调度

### Wave 0：实现验证

- bias 公式单元测试；
- raw/normalized distance 分布测试；
- view permutation 测试；
- V2–V5 可变 token 数测试；
- key mask 测试；
- Route A 的 R5 零初始化等价性测试；
- 保存各层 `QK`、confidence bias、geometry bias 的 logit 均值/P90，确保 bias 不是数值上接近 0。

### Wave 1：正确位置的 bias

两张 GPU 并行：

- GPU0：A1 global confidence-only；
- GPU1：A2 global confidence + normalized geometry。

已有 A0/J1e 作为 no-bias 对照。

### Wave 2：主架构

- GPU0：B0 no-bias；
- GPU1：B3 confidence + normalized geometry。

若 B3 优于 B0，再补：

- B1 confidence-only；
- B2 raw geometry。

### Wave 3：纯论文结构验证

- C0；
- C1。

仅在 B 系列无法判断 bias 原因或需要论文式对照时运行。

## 8. 评价与停止规则

主指标：clean CMU MMPose All-17 MPJPE，V2/V3/V4/V5。

RUMPL baseline：

- V2 30.885
- V3 23.039
- V4 20.213
- V5 18.746

每组同时报告：

- 每个 view count 的绝对 MPJPE；
- 相对 baseline 的 delta；
- V2–V5 macro average；
- 参数量、FLOPs和推理显存；
- learned `eta_l^2/gamma_l^2`；
- bias logit 与 content logit 的实际比例。

晋级条件：

1. V2–V5 macro average 优于 baseline；
2. 至少 3 个 view count 优于 baseline；
3. 任一 view count 退化不超过 0.15 mm；
4. bias 版本必须优于同架构 no-bias 版本，才能声称“偏置有效”。

停止条件：

- A2 不优于 A0/A1：停止在旧 VFT 前继续堆 bias，转 Route B；
- B3 不优于 B0：说明论文 bias 在当前 clean 数据协议中收益不足，不再扩大 Transformer；
- normalized geometry 有效而 raw 无效：后续只保留 normalized；
- confidence-only 最好：保留 confidence bias，移除 geometry，避免为了形式忠实而牺牲精度。

## 9. 推荐优先级

1. A1：global confidence-only；
2. A2：global confidence + normalized geometry；
3. B0/B3：GBT encoder-decoder 替换 VFT并保留 4 层 PFT；
4. B1/B2：拆分消融；
5. C0/C1：纯论文结构验证。

预期最有希望的最终结构是：

`Plücker harmonic tokens → 3-layer global biased encoder → 2-layer joint-query decoder → 4-layer PFT → 3D head`

它既把 bias 放在论文证明有效的全局位置，又保留 RUMPL 对人体关节结构的建模能力，同时完全不依赖时序。

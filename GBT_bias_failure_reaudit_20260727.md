# GBT bias 失败复审（2026-07-27）

## 结论

现有证据不支持“Geometry-Biased Transformer 的 bias 在 RUMPL 上无效”这一笼统结论。

更准确的结论是：

1. confidence bias 已经产生小而稳定的收益；
2. 原 VFT 内的 geometry bias 与论文结构并不等价，并且其直接作用于最终 fusion token 的有效强度过小；
3. 最接近论文的 global joint-view biased encoder 实验 J0 没有跑完，只运行到 epoch 0 的 600/4003；
4. 同一 global joint-view 结构的无偏置版本 J1e 已在 V3–V5 超过 RUMPL baseline，因此“global token mixing + bias”仍是尚未完成的高价值实验。

## 论文和 RUMPL 的关键差异

论文的 encoder token 是 `joint × camera × time`，在全局 self-attention 的每一层加入：

`softmax(QK^T / sqrt(d) + eta_l^2 M_conf - gamma_l^2 M_dist)`

随后由 joint query decoder 直接读取所有 encoder token。论文默认使用 9 帧、随机 2 视角训练、3 层全局 encoder、2 层 decoder，并配合 token dropout。

当前 RUMPL G0–G4：

- 只有单帧；
- bias 主要位于逐关节 VFT，每次 attention 只含同一个关节的多个视角；
- 最终输出读取 fusion token；
- baseline 已将 confidence 拼入 token embedding，显式 confidence bias 的信息存在冗余；
- 默认 geometry matrix 中 fusion-token 行为 0，因此 geometry bias 不直接改变最终 fusion query；
- `GBT_FUSION_GEOM=1` 时使用“该视角与其他视角的平均射线距离”作为额外启发式，这不是论文原公式。

## 已有结果重新解释

All-17 MPJPE（mm，越低越好）：

| 方法 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|
| RUMPL baseline | 30.885 | 23.039 | 20.213 | 18.746 |
| G0 conf+geom，fusion 不受 geom | 31.182 | 23.211 | 20.480 | 19.225 |
| G1 conf+geom，fusion geom | 31.020 | 23.307 | 20.448 | 18.994 |
| G2 confidence only | 31.058 | **22.951** | **20.056** | **18.679** |
| G3 geometry only，fusion geom | 30.947 | 23.199 | 20.463 | 19.149 |
| G4 conf+geom，较小 geom init | 30.874 | 23.095 | 20.292 | 18.882 |
| J1e global joint-view，无 bias | 31.050 | **22.884** | **19.907** | **18.413** |

所以：

- confidence bias 不是完全失败：G2 在 V3/V4/V5 分别改善约 0.088/0.156/0.067 mm；
- geometry bias 是主要退化来源；
- global joint-view 结构对 V3–V5 很有效，但 V2 仍差 0.165 mm；
- 最自然的未完成实验是用论文式 bias 改善 J1e 的低视角和异常观测处理。

## 为什么当前 geometry bias 没成功

### 1. 读出路径不匹配

论文没有 fusion token：decoder query 直接 cross-attend 全部 biased encoder tokens。

RUMPL 默认从 fusion token 输出，但默认 geometry distance 对 fusion-token query 是 0。几何偏置只能先改变 view-view attention，再间接传递到 fusion token，路径较弱。

### 2. bias 数值尺度太小

CMU 验证集真实统计：

- 同关节跨视角 pairwise ray distance 中位数约 1.98 cm；
- 90% 分位约 7.17 cm；
- mean pairwise distance 与真实 point-to-ray error 的 Pearson 相关系数约 0.846，Spearman 约 0.508。

说明几何 cue 本身有效，但模型使用米作为单位。G1/G3/G4 学到的 geometry scale 均值约 0.15–0.19，典型 attention logit 改变量只有约 `0.15 × 0.02 = 0.003`，远小于内容 attention logit，几乎不起作用。

G0 的 geometry scale 较大（均值约 1.62），但它没有给 fusion query 加 geometry bias，因此强度和读出没有同时满足。

### 3. confidence 已经被 baseline 使用

当前配置是 `ConfConcat`：confidence 已拼入 ray token。论文的 ray embedding只编码 Plücker ray，confidence bias 是新增信息。因此把论文 confidence bias 移植到 RUMPL 后，主要是改变信息注入方式而不是增加信息，收益自然小于论文消融。

### 4. 论文 bias 的价值来自全局消歧

论文在同一个 attention 图中混合不同关节、视角和时间：

- 同相机的不同关节射线距离为 0；
- 同关节跨相机的正确射线应接近相交；
- confidence 和 geometry 用来约束一个很大的全局候选集合。

RUMPL VFT 已预先按 joint 分组，候选 key 都是同一个关节，任务简单很多，geometry bias 提供的信息更冗余。

### 5. 训练预算和配套增强不等价

论文训练 300k iterations，使用 9 帧、synthetic views、20% token dropout 和 scene centering。当前训练约 80k iterations、单帧、RUMPL 自己的随机相机合成协议。表 VI 是整套架构中的移除消融，不能理解为“把 bias 单独加到任意 Transformer 都应得到同样提升”。

## 尚未真正尝试的方案

### P0：完成论文式 global joint-view bias（首选）

在已经有效的 J1e global joint-view 分支上：

- 全局 token 为 `J × V`；
- confidence 和 pairwise Plücker distance 加入每层 global attention；
- 保留 RUMPL 后续 VFT/PFT；
- 先跑与 J1e 完全相同的训练协议；
- 同时跑 global confidence-only，分离 geometry 的贡献。

这条路线之前的 J0c 在 epoch 0 被中断，不能计为失败。

建议两组：

- `J2_global_conf_only`：验证 confidence 是否能修复 J1e 的 V2；
- `J3_global_conf_geom_norm`：加入归一化 geometry bias。

### P0：对 geometry distance 做无量纲归一化

不要直接将“米”加到 attention logits。建议每个样本/关节使用：

`d_norm = (d - median(d)) / (MAD(d) + eps)`

或：

`d_norm = d / temperature`，temperature 初始值取 0.02 m 并可学习。

这能让典型几何偏置达到 0.1–1 的 logit 量级，同时避免场景尺度影响。

### P1：fusion-query reliability bias

对每个 view 构造：

`r_i = median_{j != i} d(ray_i, ray_j)`

再将 `-alpha × standardized(r_i)` 只加到 fusion query 对各 view key 的 logits。它直接实现“几何离群视角降权”，比当前在 view-view 矩阵中绕一圈更契合 RUMPL 读出。

可结合 confidence：

`reliability_i = a × log(conf_i + eps) - b × standardized(r_i)`

其中 `a,b` 分层学习并保持非负。

### P1：bias warm-start / zero-init

从 J1e 或 R5 checkpoint 初始化，令 bias gate 为 0，先只训练 bias scale 1–2 epoch，再联合微调。这样可以保证初始输出不劣于已有模型，并直接判断 bias 是否能提供增益。

### P2：补齐论文的 token dropout

训练时随机删除而不是置零 10%–20% 的 joint-view tokens，迫使 global encoder 使用 confidence/geometry。先在固定干净评测上验证，再单独检查少视角鲁棒性。

## 推荐执行顺序

1. `J2_global_conf_only`；
2. `J3_global_conf_geom_norm`；
3. 若 J3 不如 J2，改跑 `fusion-query median-ray reliability`；
4. 只有上述方案有效后，再加入 token dropout。

成功标准应是：

- 首先要求 V2–V5 平均优于 RUMPL baseline；
- 不能只改善 V2 而持续损害 V4/V5；
- 至少复现 J1e 在 V3–V5 的优势，并把 V2 从 31.050 降到 30.885 以下。

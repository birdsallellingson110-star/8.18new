# GBT 两视角差距诊断与下一阶段实验计划（2026-08-15）

## 1. 结论先行

当前证据不支持“HRNet 数据处理仍是 20 mm 差距的主因”。GBT 在 H36M 上的强项
是：当两条观测射线本身无法稳定确定 3D 时，用全局人体姿态与历史上下文直接回归
合理 3D，而不是只做更精确的两射线融合。

GBT 的 `HRNet` 行使用 9 帧输入，不能与当前 RUMPL/H76 的单帧结果直接相减。
论文没有代码，且未公开 YOLOX 型号/阈值、时序帧间隔、隐藏维度、synthetic camera
分布等细节，因此 `36.8 mm` 只能作为目标，不能称为已严格复现的可归因结果。

## 2. 为什么输入处理不是主要解释

### 2.1 同为 HRNet 坐标时，GBT 的输入三角化并不更好

| 方法/输入 | V2 | V3 | V4 |
|---|---:|---:|---:|
| GBT 论文 HRNet + Algebraic Triangulation | 120.7 | 50.9 | 44.2 |
| GBT 论文完整模型，T=9 | 36.8 | 30.4 | 26.0 |
| 本地 GBT-style HRNet + IRLS 控制 | 90.405 | 54.833 | 49.485 |

本地 V2 三角化甚至比论文三角化低约 30 mm，但完整网络远没有自动得到 GBT 的
36.8 mm。这说明 GBT 的主要收益来自学习式 3D 推断/人体先验，而不是一套天然更准
的 HRNet 坐标。

### 2.2 我们的误差集中在退化相机对

LT 输入下 H76 的六个 V2 相机对为：

| 相机对 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|
| H76 | 30.169 | 34.600 | 70.290 | 77.142 | 27.384 | 29.192 |

四个正常相机对已经约 27--35 mm，均值主要被 `1-4` 和 `2-3` 拉高。官方 LT
Volumetric 可把这两个坏对分别改善约 23.2/31.5 mm，证明人体/图像体积先验确实
能修复几何退化；单纯 confidence 重加权对 V2 每个相机对只改善约 0.3--1.4 mm。

因此下一阶段的首要指标不是只看 V2 平均，而是同时报告六个相机对，要求重点降低
`1-4`、`2-3`，并保护另外四个正常相机对。

## 3. GBT 真正包含哪些优势

GBT 不是“RUMPL 加 geometry bias”。它同时使用：

1. Plucker ray 的 15-frequency harmonic embedding；
2. 将 joint、view、time 全部 token 放入 3 层 global self-attention encoder；
3. 用 2 层 learned joint-query decoder 从全局集合直接解码 3D，而不是先按 joint
   压缩 view，再在压缩结果上建模人体；
4. 每层可学习的 confidence 正偏置和 pairwise ray-distance 负偏置；
5. 训练固定随机两视角，输入/输出 9 帧；测试同一模型的 2/3/4 视角；
6. 300k iterations、batch 256、MSE；
7. scene centering、synthetic views、20% token dropout。

论文自己的 H36M V4 消融为：无组件 `39.0`；centering `49.2`；再加 synthetic
`40.6`；confidence-only `33.2`；geometry-only `33.1`；两种 bias 同时为
`26.0 mm`。同一完整模型把时间从 T=1 增加到 T=9 时，V4 从 `29.4` 降至
`26.0 mm`，即时间贡献约 3.4 mm。

这说明：

- bias 在其完整 global encoder-decoder 中作用明显，但不能推出把同一公式塞进
  RUMPL VFT 也会得到相同增益；
- 时序解释不了全部 20 mm，但确实贡献了其中一部分，且 V2 可能比 V4 更依赖历史；
- centering/synthetic 主要用于跨场景泛化，在 H36M 同分布下不是首要降误差手段；
- 20% token dropout 没有单独消融，不能假定它一定提高 clean H36M。

## 4. 为什么以前的 GBT/时序移植没有得到论文增益

### 4.1 RUMPL 存在先压缩、后建模的瓶颈

RUMPL/H76 的主路径是：

`同一 joint 跨 view 的 VFT -> 每个 joint 一个融合 token -> PFT -> 3D`

当某个膝/踝在两个视角中都不可靠时，VFT 在看到完整人体和历史之前就完成了视角
压缩。GBT 则在压缩以前让所有 joint-view-time token 相互作用，再由 joint query
读取全局证据。两视角没有第三条冗余射线时，这个结构差异尤其重要。

### 4.2 以前的全局 bias 接入位置不等价

旧 global-JV 实验中，plain global context 相对课程控制在 V2 改善约 1.09 mm，
说明全局上下文方向有效；但对所有 `J x V` token 直接加 ray-distance bias 后反而
退化。原因是同一相机发出的不同关节射线共享相机中心，线距离为 0，若无完整
query-decoder 语义配合，会把“同相机不同关节”错误地当成强几何对应。

旧 pre-VFT SGraFormer 和 PFT 替换也均明显退化，说明不能继续把新模块随意插入
RUMPL 的中间表征。下一次必须把 global encoder-decoder 当成完整替代主干做对照，
而不是继续在 VFT/PFT 前后堆层。

### 4.3 以前的时序看到的是已经压缩的信息

旧 ST-VFT、T-CVU、candidate fixed-lag、MixSTE 大多冻结或保留单帧 H76/E2，
只对已压缩 3D/candidate 做后处理；严格对齐后增益约 0.1--0.2 mm。GBT 的时间
token 在 view fusion 和 3D 解码以前参与全局 attention，并用 9 帧 3D 监督端到端
训练，信息通路不同。因此不重复旧后处理，只允许对新的 early-token 主干做一次
端到端 T=9 对照。

## 5. 下一阶段实验顺序

### P0：完成当前 GBT-style HRNet 坐标基线

当前双卡正在从头训练同一新缓存上的：

- R0：原始 RUMPL；
- H76：tri-anchor + anchor-centered Plucker。

完成后必须输出：V2/V3/V4 action-equal、frame-weighted、六个 V2 相机对、
absolute/root-relative、逐关节误差。该结果是后续唯一可用的坐标级基线。

### P1：训练单帧 GBT 结构控制，不加时序

不再保留完整 RUMPL VFT/PFT。建立一个单一新主干：

`H76 triangulation anchor -> anchor-centered Plucker harmonic tokens ->`
`3-layer global encoder -> 2-layer joint-query decoder -> anchor + 3D residual`

保留的是 RUMPL 的相机无关 ray 表示和已验证的三角化锚点；替换的是在坏相机对上
不足的 VFT/PFT 融合器。首轮两组：

| 实验 | 唯一差异 | 目的 |
|---|---|---|
| G0 | global encoder-decoder，无 attention bias | 测全局 set/query 结构本身 |
| G1 | G0 + 论文 confidence/ray-distance learnable bias | 测论文 bias 在正确作用层的贡献 |

共同设置：T=1、固定随机 K=2、所有六个相机对均匀采样、MSE、无 dropout、无
synthetic views。同一 checkpoint 测 V2/V3/V4。先跑约 50k-step screening；只有
验证曲线仍下降且坏对改善，才继续到 100k/200k/300k，不用 20 epoch 提前判死刑。

成功门槛：相对同输入 H76，V2 至少降低 3 mm；`1-4`、`2-3` 各至少降低 8 mm；
四个正常对平均退化不超过 1 mm；V3/V4 不明显退化。

### P2：若 G1 的全局 ray bias 仍伤害，做语义受限 bias

只运行一个有诊断依据的修正版：

- ray-distance bias 只作用于“同一 joint、同一时刻、跨 view”的 token 对；
- 同一 view 的不同 joint 由普通 attention + joint embedding 建模，不对其施加
  线距离奖励；
- confidence 仍作为 key reliability；
- 其余结构、参数量和训练完全等同 G1。

这是针对旧 global-biased 失败机制的最小修复，不能与 G1 同时调其他参数。

### P3：在最强单帧模型上做一次真正的 early-token T=9

仅当 P1/P2 单帧已经明显超过 H76 时才启动。训练缓存已按原视频每 5 帧采样，可
构造 T=9@10Hz 窗口；验证需要在已有 `annot_temporal_5_5` 上重新导出同一
GBT-style YOLOX-X+HRNet 坐标。论文未公开 temporal stride，因此必须明确标为
`T=9, stride=5`，不能声称完全复制论文时间协议。

对照只保留 T=1 与 T=9；9 帧 token 在 global encoder 前进入，端到端监督 9 帧，
推理只输出最后一帧。成功门槛：V2 再降低至少 1 mm，且 V3/V4 不退化；若能接近
论文 V4 的 3.4 mm 增益，再补 T=3 消融。

### P4：最后才测正则与泛化组件

按优先级只做：

1. no dropout vs 论文 20% true token removal；
2. 若 20% 伤 clean，复核已在 RUMPL 中有效的 10% warmup-only；
3. scene centering/synthetic views 留到跨相机/跨数据集泛化表，不作为 H36M
   clean 主表的首轮优化。

## 6. 停止重复的实验

- 不再单独扫 confidence weight：V2 没有冗余，已有官方 LT 证据表明收益很小；
- 不再做冻结 H76 后的 MixSTE/fixed-lag/candidate temporal 后处理；
- 不再把 SGraFormer/GraFormer 插入 VFT 前或替换 PFT；已完成严格负结果；
- 不再对所有 global joint-view token直接使用无语义 mask 的 ray-distance bias；
- 不以历史 A1D/H21 的 `34.816/30.489/29.691` 与 raw HRNet GBT 表混比；该线输入
  和子集信息协议不同。

## 7. 目标表

| 阶段 | V2 目标 | V3 目标 | V4 目标 | 解释 |
|---|---:|---:|---:|---|
| P0 | 以新 H76 实测为准 | 同左 | 同左 | 锁定公平基线 |
| P1/P2 T=1 | H76 -3 mm，坏对各 -8 mm | 不退化 | 不退化 | 证明新融合主干有效 |
| P3 T=9 | <=40 mm | <=30.4 mm | <=26.0 mm | 首个可投稿级里程碑 |
| 最终外部目标 | <36.8 mm | <30.4 mm | <26.0 mm | 在相同 HRNet/T=9 表中超过 GBT |

任何结果都同时报告 all-pair average 和六个 V2 pair。只有坏对真实下降而非挑选
相机对，才算解决 GBT 所对应的两视角问题。

## 8. 训练量核算：不是由 T=9 自动产生的差异

当前 RUMPL 日志显示每 epoch 有 `2438` 个 batch，训练 batch size 为 `32`，共
`20` 个 epoch，因此实际优化更新数为：

```text
2438 × 20 = 48,760 updates
```

这与 78,047 个同步四相机训练时刻相符（每个 epoch 基本遍历一次训练组）。GBT
论文明确写的是 `300,000 iterations`、batch size `256`，因此按论文口径是：

```text
300,000 updates × 256 = 76,800,000 sequence samples
```

若只按优化更新次数比较，GBT 是当前 RUMPL 的约 `6.15×`；若按 batch 内序列样本
数比较，则约 `49.2×`。GBT 每个样本又包含 9 个输入/输出时间帧，且论文训练的
MSE 对 9 帧都计算，因而每次更新包含约 9 倍的时间监督项；这不等于 9 倍独立数据，
但确实增加了每步的信息量。

因此：

- `T=9` 不会把 48k 自动变成 300k；iteration 数是独立的优化超参数；
- T=9 影响的是每个 token 序列的长度、显存/计算量和每步监督项；
- 300k 主要意味着 GBT 进行了更长的优化，并且其 batch 更大；
- 当前训练前 8 个 epoch 还固定 K=2，之后恢复 V2/V3/V4 混合，而 GBT 论文描述为
  全程随机两视角，这也是另一个独立变量。

要拆分这三个因素，后续应增加如下控制：

| 控制 | T | 更新预算 | 视角采样 | 目的 |
|---|---:|---:|---|---|
| B1 | 1 | 300k | 当前 RUMPL 采样 | 只测训练预算 |
| B2 | 1 | 300k | 全程随机 K=2 | 测预算+GBT 视角协议 |
| G0 | 1 | 300k | 全程随机 K=2 | 测 GBT global encoder-decoder |
| G0-T9 | 9 | 300k | 全程随机 K=2 | 在同一主干上单独测时序贡献 |

B1/B2 应先于 G0-T9。若 B1 已接近 GBT 目标，差距主要是优化预算；若 B1/B2
仍在平台而 G0 明显下降，才可把收益归因于融合主干；G0-T9 相对 G0 的差值才是
时序的独立贡献。

## 9. 2026-08-15 实际执行记录

01:35 已启动持久化队列
`launch_gbt_budget_controls_20260815.sh`（tmux：`cjy_gbt_budget_20260815`）。
当前 R0/H76 先行对照仍在 GPU0/GPU1 上运行；队列等待两组的 `.done` 标记，避免
四个训练进程互相争用数据加载和 GPU。两组完成后自动接续并评估 V2/V3/V4：

| 实验 | 主干 | T | 训练轮数/更新数 | 视角采样 | GPU |
|---|---|---:|---:|---|---:|
| B1_CURRENTK | H76（锚点+中心射线+Plücker） | 1 | 123 / 299,874 | 前 8 轮 K=2，之后 RUMPL 当前 3:1:1 混合 | 0 |
| B2_FIXEDK2 | 同上 | 1 | 123 / 299,874 | 全程随机 K=2 | 1 |

两组使用同一 GBT-style YOLOX-X+HRNet 坐标缓存、seed=0、batch=32 和 RUMPL
损失；将原 20 轮的 `[10,15]` 学习率节点按训练进度缩放为 `[62,93]`，并由
`RUMPL_LR_STEPS` 显式记录。这是“预算/视角协议”控制，不是新模块，结果不能
直接当作 GBT 主干结果。当前 `train_rumpl.py` 已增加该环境变量解析，不影响已在
运行的 R0/H76 进程。

同日 01:49 又启动 `launch_gbt_global_screen_20260815.sh`（tmux：
`cjy_gbt_global_screen_20260815`），在两张卡上并行进行短筛选：G0 为一层
ReZero global joint-view attention，G1 再加该层的 confidence/ray-distance
bias。两组都是 H76、T=1、全程 K=2、20 轮（约 48,760 updates），使用 4 个
worker，以便不完全挤占 B1/B2。两组已打印 `[GLOBAL_JV]` 配置并进入第 0 轮训练；
它们只用于筛选方向，不能替代后续 300k 公平主实验。

### G0/G1 筛选结果（2026-08-15 03:27 完成）

| 实验 | V2 | V3 | V4 | 相对 H76 46.227/31.334/27.964 |
|---|---:|---:|---:|---:|
| G0 plain global-JV | 78.336 | 47.536 | 43.232 | +32.109/+16.202/+15.268 |
| G1 global-JV+bias | 78.192 | 47.383 | 43.050 | +31.965/+16.049/+15.086 |

两组均显著差于 H76；G1 相对 G0 只改善 0.144/0.153/0.182 mm，不能说明 bias
本身有效。训练日志中 G0/G1 的末轮平均 loss 约为 0.032/0.039，而 H76 约为
0.024，说明当前实现的 global-JV 分支已改变优化轨迹，不能继续直接扩展到 300k
或与 E2 叠加。该结果作为失败消融保留：GBT 的 global encoder-query decoder
仍需重新设计为完整替代融合器，不能把一层 attention 插到 H76 中就等价复现。

### B1/B2 预算与视角协议结果（2026-08-15 09:54--10:14 完成）

| 实验 | V2 | V3 | V4 | 解释 |
|---|---:|---:|---:|---|
| H76 20E 基线 | 46.227 | 31.334 | 27.964 | 8E 固定 K=2 后混合 3:1:1 |
| B1 299,874 updates | **43.456** | **30.732** | **27.818** | 只延长预算，保留当前采样 |
| B2 299,874 updates | **37.886** | 62.215 | 46.217 | 全程 K=2 |

B1 说明训练量确有收益，但幅度有限（V2/V3/V4 分别 -2.771/-0.602/-0.146 mm），
不能解释 H76 与 GBT 的全部差距。B2 把 V2 降到 37.886 mm，但 V3/V4 严重恶化，
说明 RUMPL 在只看 K=2 的训练分布下发生了明显的视角数量分布偏移；固定两视角并不
能直接迁移为 RUMPL 的多视角模型优势。B2 的六个 V2 对分别为
`33.855/38.112/47.825/45.930/29.701/31.895 mm`，所以它确实修复了两个坏
相机对，却牺牲了 V3/V4 的组合泛化。

因此后续不再把“增加训练轮数”或“全程固定 K=2”单独包装成模型创新；这两组只用于
解释 GBT 训练协议的作用。E2 当前输入候选池应优先针对 B1/H76 的两个坏对重建，
再进行 V234 universal 融合。

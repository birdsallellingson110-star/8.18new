# 时序候选效用路线：去重决策与实验冻结（2026-08-12）

## 1. 最终决策

下一阶段不再训练普通 3D pose temporal refiner，也不再重做 GBT 式全局
joint-view-time Transformer。正式新方向固定为：

> **Temporal Counterfactual Candidate Utility（T-CVU）**：保留 RUMPL 的射线、
> VFT/PFT、tri-anchor 和多视角子集候选，保留 E-2 已验证的逐关节反事实效用；
> 时序模块只利用相邻帧证据修正目标帧的候选效用 logits，最终仍以 soft 权重融合
> 目标帧的 RUMPL 候选，不直接回归任意 3D 位移。

该决定的核心依据是：单帧候选池在 S9/S11 上仍有很大 oracle 空间（V3
`25.1429`、V4 `21.8169` mm），而普通时序 3D 残差的三种子收益只有约
`0.532/0.448` mm。下一步应改善“选择哪个几何候选”，而不是继续平滑已经融合
完且丢失视角来源的 3D 输出。

## 2. 已完成时序实验和禁止重复项

以下方向已经有正式结果，不能换名字后重复：

| 已做方向 | 代表实验 | 结果/结论 |
|---|---|---|
| T=9 全局 joint-view-time，冻结 RUMPL | H91/H92、H100--H102 | V2/V4 变化仅约 `0.00--0.02` mm；confidence/geometry bias 几乎无差异 |
| 全 9 帧监督 vs latest 监督 | H100/H101 | V4 `29.9640/29.9475`，全帧监督没有产生 GBT 式大收益 |
| token dropout 0 vs 20% | H104--H107 | V4 `29.985--30.002`，无有效区别 |
| 解冻 head/VFT/backbone | H103、H136、H172 | 均退化；不能再以“解冻更多层”作为新实验 |
| GBT confidence/geometry bias | H92/H102/H105/H107/H109/H111 等 | 在既有 RUMPL/JVT 位置没有稳定增益 |
| query decoder / joint query | H108--H111 | 相对同协议 H76，V2/V3/V4 全部退化 |
| MixSTE TTB、残差、交替 STB/TTB、MixSTE loss | H169--H174、Stage H2/H4 | 训练主体内改善但 S9/S11 退化或弱于 fixed-lag，属于跨主体过拟合 |
| 加深 attention | 3/4/12 层及 E-2 depth 3/4 | 内部 holdout 更好、S9/S11 更差；禁止继续堆深度 |
| 最终 3D pose 的 T=9 residual | Stage H1--H4 | H3 有稳定但有限收益；V3/V4 三种子均值 `29.4691/28.7956`，作为时序控制保留 |
| 重复 ray-view attention | Stage F | 与 RUMPL VFT 信息重叠，F0 退化、F1 仅 `0.0004/0.0047` mm 漂移 |
| 增加 joint-axis attention | Stage G | 不能稳定迁移，平均无增益 |

同时禁止在结构有效前加入 bone/velocity/acceleration 等多损失，也禁止依据
S9/S11 扫温度、深度、窗口或学习率。

## 3. 对 GBT 3.4 mm 时序收益的重新解释

GBT Table VII 的 H36M 四视角结果从 T=1 的 `29.4` 降到 T=9 的 `26.0`，但：

1. 论文训练的是 T=9、输出 9 帧的模型，T=1 是同一模型测试时减少输入，不是
   独立优化的强单帧基线；`3.4 mm` 包含训练/测试时窗不匹配。
2. GBT 在射线 token 被融合之前联合建模时间、视角和关节；我们的 H3 只看到
   已经融合后的 3D pose。
3. GBT 回归完整 absolute 3D；H3 强制复制 root，只允许 root-relative residual。
4. 本地 H36M 图像包只包含帧号 `1,6,11,...`，即原始 50 Hz 的每 5 帧一张。
   当前 T=9 覆盖约 0.8 秒。论文没有开源，也没有明确说明 Table VII 是否使用
   同样的下采样，因此不能把其 `3.4 mm` 当成我们的保证收益。

## 4. 新路线为何不重复旧实验

旧 JVT 在 RUMPL VFT 前重写全部 ray token，容易破坏已经校准的单帧几何表示；
旧 pose refiner 在融合后看不到候选来源。T-CVU 位于二者之间：

```text
每帧 RUMPL 多子集候选 + E-2 单帧 delta logits
                       │
          同一关节/候选沿时间轴的小型 encoder
                       │
        目标帧 delta_logit + 零初始化 temporal residual
                       │
                  softmax(-delta)
                       │
              只融合目标帧 RUMPL 候选
```

- 目标帧 temporal residual 零初始化，step 0 与 E-2 完全一致。
- 不直接移动 3D 点，输出始终位于已有几何候选的 soft convex combination 中。
- 第一版不重复加入 ray-view attention、joint-axis attention或相机 ID。
- 第一版只使用候选轨迹、候选间分歧和 E-2 delta logits；raw confidence/ray
  distance 仅在第一版成立后做单变量消融。
- 延续 C2/E-2 已验证的 delta regression + balanced classification/ranking，
  不首先发明新损失。
- V3/V4 专门训练；V2 暂时保留 H1 的独立时序头，避免已证实的任务冲突。

## 5. 实验执行顺序

### I0：零训练可行性审计

在 train subjects 和 S8 holdout 上统计：

1. 每关节 oracle candidate identity 在相邻帧的保持率；
2. candidate true delta 在 `t-1/t/t+1` 的自相关；
3. E-2 预测误差是否可由相邻帧 delta/disagreement 解释；
4. T=3/5/9 的有效样本数及完全相同 center 集合。

若 oracle identity 与 delta 几乎没有时间相关性，T-CVU 立即停止，不训练模型。

### I1：时间跨度控制（不改模型结构）

使用现有 E-2 pose-only fixed-lag 控制，只比较 T=3、T=5、T=9。所有模型：

- 使用相同的 T=9 可用中心帧集合；
- 使用相同 V3/V4 任务、损失、宽度、深度和训练样本顺序；
- 只改变窗口长度；现有采样间隔均为 5 个原始帧。

T=3 覆盖约 0.2 秒，最接近假设中的 GBT 连续 9 帧约 0.16 秒。先在完整 S8
subject holdout 选择时间跨度，不使用重叠窗口随机 holdout；固定后才训练全部
S1/S5/S6/S7/S8，并对 S9/S11 评估一次。

### I2：T-CVU 最小模型

按 I0/I1 选出的窗口，只训练一个 1 层小型 temporal utility encoder。GPU0/GPU1
分别跑无 temporal utility 的严格 identity 对照和 T-CVU；确认数值对齐后，两卡
并行跑 T=3/T=5 候选版本。T=9 只有在较短窗口没有明显优势时才补。

### I3：单变量信息消融

仅当 I2 在 S8 holdout 同时改善 V3/V4 后，依次加入：

1. candidate disagreement/方差；
2. HRNet confidence summary；
3. pairwise ray-distance summary。

三者不得同批一起加入。旧 Stage F 已证明完整 ray-view attention 重复，因此这里只
允许 summary 特征，不再创建第二个 ray Transformer。

### I4：正式复验

选定结构用全部训练 subjects 重训，S9/S11 只评一次。若 seed 0 同时优于 E-2
和 H3，则补 seed 1/2；否则停止，不对测试主体调参。

## 6. 预注册成功与停止门槛

| 阶段 | 继续条件 | 停止条件 |
|---|---|---|
| I0 | delta/最佳候选存在明确相邻帧相关性 | 时间相关性接近 0 |
| I1 | 某个时窗在 S8 上同时改善 V3/V4 | 仅训练窗口随机 holdout 改善或两项冲突 |
| I2 | S8 上 V3/V4 均至少改善 `0.2 mm`，并超过三种子噪声 | 任一视角稳定退化；不扫测试集补救 |
| I4 seed 0 | S9/S11 同时低于 H3 seed0 的 `29.4122/28.7503` | 未同时超过则不补 seed |
| 三种子主结果 | 均值稳定下降且每个 seed 同方向 | 仅单 seed 或低于约 `0.1 mm` 的漂移 |

长期目标仍是 V2 `<40 mm`、V4 `<30 mm` 的基础上继续压低 MPJPE；若要接近或
超过 GBT-HRNet 四视角 `26.0 mm`，T-CVU 需要证明它能利用候选 oracle 空间，
而不能靠测试协议、root 解冻或挑选动作获得表面提升。

## 7. 暂缓事项

- 不立即重新下载完整 50 Hz H36M。先用 T=3 的相近时间跨度验证收益；只有
  T=3 明显优于 T=9，才证明更密邻帧值得额外数据成本。
- 暂不做 root 解冻。旧 H173 的 unrestricted global residual 已明显失败；待
  候选时序成立后，再做 root-frozen/root-only 的独立诊断。
- 不与 H3 pose residual 立即融合。T-CVU 必须先单独超过 E-2，之后才允许做
  两个已独立有效模块的组合消融。

## 8. 执行记录（2026-08-12）

### I0 已完成

I0 在固定中心集合上通过。验证集（S9/S11）相邻窗口的 candidate-delta
相关性约为 `0.697/0.700`（V3/V4），E-2 误差相关性约为 `0.813/0.826`；
最佳候选 identity persistence 也保持正相关。因此没有因“时序无信号”提前停止。
完整数值见：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Temporal_Candidate_Utility_20260812/I0_temporal_candidate_audit.json`。

### I1 已完成

在相同 S8 留出、相同中心集合和 pose-only fixed-lag 控制下：

| 窗口 | V3（mm） | V4（mm） | 相对中心基线 |
|---|---:|---:|---:|
| T3 | 29.7862 | 28.9541 | -0.1576 / -0.2428 |
| T5 | 29.7433 | 29.0094 | -0.2035 / -0.1799 |

T3/T5 的优势不一致，不能据此宣称最佳窗口；它们只证明短窗口存在约
`0.2--0.3 mm` 的可迁移 pose-only 信号。I2 因此同时保留 T3/T5，不使用 S9/S11
选择窗口。

### I2 当前执行状态

T-CVU 已完成数值对齐修正：评估基线定义为冻结 E-2 soft-fusion，而不是原始
全视角 H76 候选。零初始化时与导出的 E-2 输出最大差异为
`4.8e-7 m`。T3/T5 已在 GPU0/GPU1 以 S8 留出并行训练（8 epochs、batch 128）；
第一次 batch 512 触发 CUDA fused Transformer 的配置上限，已改为 128 重启，
不改变模型或数据变量。当前输出目录分别为：

- `.../Temporal_Candidate_Utility_20260812/T-CVU_T3_full_v3/`
- `.../Temporal_Candidate_Utility_20260812/T-CVU_T5_full_v3/`

最终结果（各自最佳 S8 留出 checkpoint）如下：

| 方法 | V3（mm） | V4（mm） | 相对 E‑2 |
|---|---:|---:|---:|
| E‑2 frozen soft-fusion | 30.0009 | 29.2438 | — |
| T‑CVU T3 | 29.9036 | 29.1339 | -0.0972 / -0.1099 |
| T‑CVU T5 | **29.8964** | **29.1256** | **-0.1045 / -0.1182** |

两组均同方向改善，说明路线没有失效，但没有达到 I2 预注册的 `0.2 mm`
双视角门槛，也没有超过已有 H3 fixed-lag 控制。因此 I2 结论为“弱正信号、
不足以作为主线”，不进入无依据的深度/窗口扫描。后续若继续，只允许做 I3 的
单变量几何质量摘要（先 confidence，再 ray-distance），并保持相同 checkpoint
选择和 S9/S11 一次性评估；若摘要也小于 `0.2 mm`，将时序候选效用作为负结果
归档，而不是包装成论文主贡献。

### I3 confidence 已完成

每个 H76 候选增加其包含视角的 detector confidence 均值，其他输入不变。结果：

| 方法 | V3（mm） | V4（mm） | 相对 E‑2 |
|---|---:|---:|---:|
| T‑CVU T3 + confidence | 29.9063 | 29.1378 | -0.0946 / -0.1060 |
| T‑CVU T5 + confidence | **29.8917** | **29.1209** | **-0.1091 / -0.1229** |

与无 confidence 的 T5（29.8964/29.1256）相比只额外下降约 `0.005 mm`，
属于噪声级差异；confidence 不是剩余时序瓶颈。confidence 与候选真实误差的
验证集相关性约 `-0.178`，而跨相邻帧本身高度平滑（T3 中心-邻帧相关约
`0.985`），可解释其增益有限。

### I3 ray-distance 当前状态

已按 E‑2 的点到射线残差定义准备候选摘要缓存；验证集原始平均残差与候选误差
相关性约 `0.101`，弱于 confidence，但包含独立几何信息。T3/T5 的单变量复验
随后运行；若同样不超过 `0.2 mm`，I3 结束，转回单帧候选/三角化求解器或更密
真实 50 Hz 输入，而不再增加时序特征。

最终 ray-distance 结果：

| 方法 | V3（mm） | V4（mm） | 相对 E‑2 |
|---|---:|---:|---:|
| T‑CVU T3 + ray-distance | 29.9100 | 29.1412 | -0.0909 / -0.1027 |
| T‑CVU T5 + ray-distance | **29.8954** | **29.1240** | **-0.1055 / -0.1198** |

这与无摘要 T5（29.8964/29.1256）和 confidence T5（29.8917/29.1209）
处于同一噪声级，未达到 `0.2 mm`。因此 I3 正式停止：T‑CVU 在当前稀疏帧、
候选效用接口上只有稳定但极小的正信号，不能支撑“时序显著提升”的论文主张。

## I4：pairwise 候选池上的 fixed-lag temporal residual（2026-08-12）

为避免把旧 H76 候选和新增 pairwise 候选混在一起，使用当前单帧最佳
`pairwise E2 + soft oracle target (tau=5 mm)` 的冻结输出，重新导出 17 候选，
在官方 T=9、stride=5 时序缓存上训练同一个 zero-initialized fixed-lag utility
residual。两张 GPU、seed 0/1、8 epochs、leave-subject-8 选择，S9/S11 仅最终
评估。评估阶段由于 V4 的 `512*17*17` token 行触发 CUDA fused attention
grid-size 限制，推理前向按 16384 行分块；该修复不改变模型函数。

| 方法 | V3 时序 | V3 中心 E2 | V4 时序 | V4 中心 E2 |
|---|---:|---:|---:|---:|
| seed0 | 29.5758 | 29.7357 | 28.8568 | 29.0297 |
| seed1 | 29.5796 | 29.7357 | 28.8644 | 29.0297 |
| 均值 | **29.5777** | **29.7357** | **28.8606** | **29.0297** |

两个 seed 均带来约 `0.16/0.17 mm` 的小幅改善，但远低于 3 mm，不能证明
“时序显著增益”。此外，时序窗口验证集为所有中心帧（26269 组），而单帧 E2
缓存为 2021 个重采样帧，因此中心基线不应与单帧 29.42/28.73 直接比较；这正是
后续需要优先完成的协议对齐审计。

**I4 结论：**时序在更强 pairwise 候选基础上仍只提供稳定小修正，固定滞后路线
暂不继续加深或换 MixSTE。下一步转向同帧采样的单帧/时序对齐与候选融合器、
稳健三角化求解器诊断；只有在对齐后仍有明确收益，才保留时序作为辅助消融。

### I4 协议对齐复核（2026-08-12）

按四元组物理帧键对齐稀疏 `annot_filtered_5_64` 与密集
`annot_temporal_5_5`。2021 个稀疏中心中 1986 个有完整 T=9 上下文，35 个是
序列边界；相同物理帧的 pairwise 候选预测最大差为 `4.8e-7`，GT 完全相同。

| 方法 | V3 | V3 中心 E2 | V4 | V4 中心 E2 |
|---|---:|---:|---:|---:|
| seed0（1986 对齐中心） | 29.4618 | 29.6229 | 28.7332 | 28.9081 |
| seed1（1986 对齐中心） | 29.4652 | 29.6229 | 28.7408 | 28.9081 |

对齐后时序仍只改善约 `0.16/0.18 mm`，证明密集协议中的小幅收益不是采样
错位造成的；原先中心基线与单帧全量数值的差别来自边界剔除和验证采样分布。
对齐脚本：`audit_pairwise_temporal_protocol_alignment_20260812.py`；结果：
`Pairwise_Temporal_Alignment_Audit_20260812/`。

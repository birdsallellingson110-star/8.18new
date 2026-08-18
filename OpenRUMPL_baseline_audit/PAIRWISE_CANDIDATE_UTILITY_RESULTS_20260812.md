# Pairwise candidate utility 实验（2026-08-12）

## 动机

E2 长训到 30 轮后，S9/S11 只改善约 0.01 mm，说明主要瓶颈不是优化轮数。
零训练诊断发现：在冻结 RUMPL/H76 的四条世界射线上，用每个视角对构造新的
pairwise closest-point hypotheses 后，逐关节候选 oracle 明显下降：

| 候选池 | V3 oracle | V4 oracle |
|---|---:|---:|
| 原 H76 11 个候选 | 25.1429 | 21.8169 |
| 加 6 个 pairwise ray hypotheses | **22.1772** | **19.0446** |

这里的 oracle 只回答“候选池里是否存在更好的候选”，不是真实模型结果，也没有
用于 checkpoint 选择。

## 实现

- 保留原 H76 11 个候选；
- 额外加入六个固定视角对 `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)` 的
  uniform closest-point ray hypotheses；
- 用 E2 depth=2 Set Transformer 做逐关节 counterfactual utility scoring；
- 不修改 2D 输入、不改 RUMPL 主干、不使用 GT 推理；
- 训练主体内部 `group_index % 10 == 0` holdout 选 checkpoint，S9/S11 最终评估一次；
- 两个 seed 并行训练 15 轮（10 direct + 5 GHT）。

扩展缓存：

`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/pairwise_candidate_cache_20260812/`

## 结果

| 方法 | seed | V3 | V4 |
|---|---:|---:|---:|
| H76 原始基线 | — | 30.4890 | 29.6913 |
| E2 原候选池 | 0 | 29.8195 | 29.0502 |
| E2 + pairwise hypotheses | 0 | **29.5109** | **28.8666** |
| E2 + pairwise hypotheses | 1 | **29.5238** | **28.8389** |
| 两 seed 均值 ± pop.std | — | **29.5173 ± 0.0064** | **28.8527 ± 0.0139** |

相对原 E2 的平均改善为 V3 `0.3022 mm`、V4 `0.1975 mm`；相对 H76 的平均
改善为 V3 `0.9717 mm`、V4 `0.8385 mm`。两个 seed 方向一致，说明新增候选
不是单次随机波动。但距离目标和 oracle 仍有明显差距，不能声称已经解决融合器
瓶颈。

输出：

- `.../Pairwise_E2_depth2_seed0/result.json`
- `.../Pairwise_E2_depth2_seed1/result.json`
- `.../Candidate_Pool_Diagnostic_20260812/result.json`

## 当前判断

这条路线是目前最有证据的改进方向：问题同时包含“候选池覆盖不足”和“效用评分
不足”。但 pairwise 候选本身平均质量较差，提升来自评分器对少数有用关节/视角对
的选择，不能直接把 pairwise 三角化当作最终输出。

下一步优先级：

1. 导出 pairwise 模型的候选使用率、负视角率和按关节收益，确认提升是否由新增
   候选真实贡献；
2. 在不增加新模型的情况下，把 IRLS/置信度 pairwise 作为第二种候选池对照；
3. 若新增候选的使用率和泛化稳定，再做候选池扩展 + E2 长训/第三 seed；否则
   停止继续堆候选，转向训练目标（集合单调/反事实风险）而不是时序模块。

## 使用率与温度审计

seed0/seed1 在 S9/S11 上给新增 pairwise 候选的平均 soft 权重约为 V3
`15.8%/15.5%`、V4 `11.8%/11.6%`，但 pairwise 成为 hard top-1 的比例只有
V3 `3.16%/2.93%`、V4 `1.64%/1.47%`。被 hard 选中的 pairwise 候选平均误差仍为
约 `36–41 mm`，说明新增候选只在少数关节起作用，不能直接 hard-select。

对 softmax temperature 做无训练审计后，训练主体 holdout 偏好较低温度，但
S9/S11 的最低点反而在 `T=2.0` 附近，说明这是主体/数据域校准差异，不能用测试集
调温度作为正式结果。正式表仍使用预注册的 `T=1.0`。

使用率输出：

- `.../Pairwise_E2_depth2_seed0/usage_audit.json`
- `.../Pairwise_E2_depth2_seed1/usage_audit.json`
- `.../Pairwise_E2_depth2_seed0/temperature_audit.json`
- `.../Pairwise_E2_depth2_seed1/temperature_audit.json`

因此下一步不再优先做 temperature sweep 或 hard top-1，而是研究训练目标如何
在保持 baseline identity 的同时，对“少数有益 pairwise 候选”进行更可靠的逐关节
风险排序。

## GHT 温度 1.8 严格对照

GHT 论文默认 softmax temperature 为 1.8。为避免把测试集温度扫描误当作结果，
重新从头训练两个 seed，并在 direct/GHT 损失、holdout 选择和最终融合中统一使用
`T=1.8`。

| 方法 | seed | V3 | V4 |
|---|---:|---:|---:|
| pairwise E2, T=1.8 | 0 | 29.4303 | 28.7697 |
| pairwise E2, T=1.8 | 1 | 29.4445 | 28.7651 |
| 两 seed 均值 ± pop.std | — | **29.4374 ± 0.0071** | **28.7674 ± 0.0023** |

相对 pairwise E2 `T=1.0` 均值，T=1.8 再改善约 V3 `0.0799` mm、V4 `0.0853` mm；
相对原 E2，累计改善约 V3 `0.3821` mm、V4 `0.2828` mm。两个 seed 一致，说明
这不是单次温度偶然性。但 T=1.8 仍只是一项校准改进，不能解释 oracle 与模型
结果之间的大 gap。

输出：

- `.../Pairwise_E2_depth2_T18_seed0/result.json`
- `.../Pairwise_E2_depth2_T18_seed1/result.json`

## 风险排序与随机子集对照（2026-08-12）

为验证“候选池 oracle 有空间，但评分器把少数错误 pairwise 候选赋予过高权重”
这一诊断，进行了两个不改变 RUMPL 主干、2D 输入和推理候选池的单变量实验。两者
均使用 T=1.8、depth=2、10 个 direct epoch + 5 个 GHT epoch、同一 modulo-10
holdout 和 seed 0。

1. **Safe-Rank**：对真实误差高于当前任务 baseline 的候选加入 1 mm hinge，要求
   其预测 delta 至少为正，风险项权重 4。结果为 V3 `29.4905`、V4 `28.8499`
   mm，较标准 pairwise T=1.8 均值 `29.4374/28.7674` 退化 `0.0531/0.0825`
   mm。简单地把所有 harmful candidate 推离 baseline 会同时压低少数有益候选的
   软融合权重，不能作为主线。

2. **Random-Subset**：训练时以 0.5 概率随机移除六个新增 pairwise hypotheses，
   保留原 H76 候选和任务 baseline，推理时仍使用完整 17 候选。结果为 V3
   `29.4716`、V4 `28.7955` mm，较标准 pairwise T=1.8 均值退化
   `0.0342/0.0281` mm。它提高了候选子集鲁棒性，但没有改善完整候选集合的泛化，
   暂不继续扫 dropout 比例。

输出：

- `.../Pairwise_E2_safe_margin1mm_seed0/result.json`
- `.../Pairwise_E2_random_subset50_seed0/result.json`

**决策**：停止简单风险 hinge 和随机 dropout；下一步针对新增 pairwise 与原 H76
候选具有相同视角 mask、但生成分布不同这一事实，引入显式候选来源标识，验证
“异质候选的来源不可辨识”是否是评分器剩余误差的瓶颈。该实验只增加来源先验，
不改几何候选、不改 RUMPL 输入，使用两个 seed 做严格对照。

## 显式候选来源先验对照（2026-08-12）

在 E2 Set Transformer 输出后增加 source×joint bias：原 H76 候选为 source 0，
六个 pairwise ray hypotheses 为 source 1；bias 零初始化，因此初始函数与 E2
完全相同。T=1.8、depth=2、15 轮，seed 0/1。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| pairwise E2 T=1.8 | 29.4303 | 29.4445 | **29.4374** | 28.7697 | 28.7651 | **28.7674** |
| + source×joint bias | 29.4729 | 29.4499 | **29.4614** | 28.7944 | 28.7829 | **28.7886** |

source bias 相对标准 pairwise E2 退化约 `0.0240/0.0212` mm，两个 seed 方向一致，
故“来源不可辨识”不是主要瓶颈；新增 pairwise 候选的收益需要依赖几何/候选间
交互，而不是一个全局来源先验。

输出：

- `.../Pairwise_E2_source_seed0/result.json`
- `.../Pairwise_E2_source_seed1/result.json`

**候选池阶段结论**：确定性 pairwise 候选 oracle 确实把上限从 `25.1429/21.8169`
降到 `22.1772/19.0446` mm，但三种评分器训练改动（risk hinge、随机子集、来源
先验）都无法再接近该上限；T=1.8 的普通 E2（29.4374/28.7674）仍是当前候选
融合主结果。继续堆固定 pairwise 候选或微调评分先验缺乏证据，下一步转向
**view-count 专门化 utility head** 或已验证的 fixed-lag temporal residual，
并保持单帧 E2 作为严格基线。

## Soft oracle-target ranking（2026-08-12）

GHT 的 expected-risk 项直接对 soft 权重提供梯度，但对候选之间的相对排序约束较
弱。新增一个标准多假设评分式监督：以每个候选真实 3D 误差构造
`q_c = softmax(-e_c / 5 mm)`，用交叉熵拟合预测权重，同时保留原 balanced
counterfactual delta、GHT expected-risk 和 weighted-estimate 项。推理仍使用
T=1.8 的 17 候选，未使用 GT。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| pairwise E2 T=1.8 | 29.4303 | 29.4445 | 29.4374 | 28.7697 | 28.7651 | 28.7674 |
| + soft oracle target (5 mm) | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |

相对标准 pairwise E2 均值改善约 `0.0166/0.0405` mm，两个 seed 同方向；该
方向可作为候选效用学习的正式消融，但还不是达到论文目标的决定性提升。输出：

- `.../Pairwise_E2_oracle_rank_tau5mm_seed0/result.json`
- `.../Pairwise_E2_oracle_rank_tau5mm_seed1/result.json`

### Soft-target temperature control

为了区分“oracle ranking 本身”与“5 mm 标签过尖”，在完全相同的训练协议下将
target temperature 改为 10 mm，仍使用 T=1.8 推理和两个 seed。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| soft oracle target (5 mm) | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |
| soft oracle target (10 mm) | 29.4356 | 29.4198 | 29.4277 | 28.7442 | 28.7430 | 28.7436 |

10 mm 仍略优于普通 E2，但比 5 mm 退化约 `0.0069/0.0167` mm；因此正式保留
5 mm，后续不再继续扫 target temperature。

输出：

- `.../Pairwise_E2_oracle_rank_tau10mm_seed0/result.json`
- `.../Pairwise_E2_oracle_rank_tau10mm_seed1/result.json`

## Pairwise E2 + fixed-lag temporal residual（2026-08-12）

单帧候选效用的最佳配置（soft oracle target，`tau=5 mm`，seed 0 checkpoint）已
冻结，仅在其导出的 17 个候选上训练 T=9、frame stride=5 的 fixed-lag temporal
utility residual。时序模块保持根节点不变、中心帧监督、零初始化；因此其零步
输出应等于冻结 E2 soft-fusion，而不是重新训练一个 3D 回归器。训练使用官方
时序 H36M 窗口，S9/S11 只在最终一次评估。

实现与缓存：

- `train_temporal_candidate_utility_20260812.py`
- `Pairwise_Temporal_Memmap_20260812/{train,validation}/`
- `Pairwise_Temporal_Models_20260812/fixedlag_seed0_rerun/`
- `Pairwise_Temporal_Models_20260812/fixedlag_seed1_rerun/`

首次完整运行在验证阶段触发 PyTorch fused Transformer 的 CUDA grid-size 限制：
V4 的 `512*17*17` token 行超过当前内核上限。已在推理模式按 16384 行分块，
不改变数值和网络结构；两个 seed 均从头重跑 8 轮并成功完成。

| 方法 | V3 时序 | V3 中心 E2 | V4 时序 | V4 中心 E2 | 最佳轮次 |
|---|---:|---:|---:|---:|---:|
| pairwise E2 + fixed-lag，seed0 | 29.5758 | 29.7357 | 28.8568 | 29.0297 | 7 |
| pairwise E2 + fixed-lag，seed1 | 29.5796 | 29.7357 | 28.8644 | 29.0297 | 7 |
| 两 seed 均值 | **29.5777** | **29.7357** | **28.8606** | **29.0297** | — |

在严格相同的时序验证协议内，时序残差稳定改善约 `0.158 mm`（V3）和
`0.169 mm`（V4），两个 seed 同方向；但它不是论文中所需的 3 mm 级提升，
也没有达到把时序作为主贡献的证据门槛。需要注意：时序验证缓存包含所有稀疏
时序窗口（26269 个中心组），而单帧 E2 验证缓存是 2021 个重采样帧；两者
的中心基线分别约 29.74/29.03 与单帧 29.42/28.73，不能直接混报。时序结果
只能与同一时序缓存中的“中心 E2”比较。

**阶段结论**：pairwise 候选池 + soft oracle ranking 的单帧收益仍是当前更强
的方向；在该基础上接入固定滞后时序后只获得小幅、可复现的后处理收益。暂不
继续堆叠 MixSTE/更深时序或调窗口，下一步优先做时序/单帧帧采样协议完全对齐，
随后转向候选融合器或几何求解器瓶颈。

### 稀疏单帧协议对齐审计（2026-08-12）

为排除“密集 temporal 验证集与稀疏论文单帧验证集分布不同”的影响，按
`(subject, action, subaction, image_id)` 将两个 pkl 精确匹配。两套缓存对相同
物理帧的候选预测和 GT 一致（候选最大差 `4.8e-7`）；稀疏 2021 个中心中有
1986 个具备完整 T=9 上下文，35 个为序列边界而无法形成 9 帧窗口。

在这 1986 个相同中心上重新评估：

| 方法 | V3 时序 | V3 中心 E2 | V4 时序 | V4 中心 E2 |
|---|---:|---:|---:|---:|
| fixed-lag seed0 | 29.4618 | 29.6229 | 28.7332 | 28.9081 |
| fixed-lag seed1 | 29.4652 | 29.6229 | 28.7408 | 28.9081 |
| 两 seed均值 | **29.4635** | **29.6229** | **28.7370** | **28.9081** |

因此采样协议对齐后仍只有 `0.159/0.176 mm` 的时序收益，结论不变；原先密集
验证的基线偏高不是实现错误，而是包含所有帧和不同边界分布。对齐脚本及结果：

- `audit_pairwise_temporal_protocol_alignment_20260812.py`
- `.../Pairwise_Temporal_Alignment_Audit_20260812/manifest.json`

## 冻结候选的 robust aggregation 对照（2026-08-12）

针对候选 oracle（约 22/19 mm）与 soft fusion（约 29/29 mm）之间的差距，保持
17 个候选和 E2 utility logits 完全冻结，只将最终加权均值替换为无需训练的
utility-weighted geometric median（8 次 Weiszfeld）和 weighted medoid。该实验
不使用 GT 推理，仅用候选间距离做鲁棒聚合。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 原 soft mean | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |
| weighted geometric median | 29.7117 | 29.6860 | 29.6989 | 28.9569 | 28.9441 | 28.9505 |
| weighted medoid | 29.9731 | 29.9526 | 29.9629 | 29.1268 | 29.1246 | 29.1257 |

两种鲁棒聚合均在两个 seed、两个视角阶段退化，说明剩余误差不是简单的候选
离群点拉动；停止该 solver 分支。脚本与输出：

- `evaluate_pairwise_robust_fusion_20260812.py`
- `.../Pairwise_Robust_Fusion_20260812*/result.json`

## View-count specialist utility head（2026-08-12）

为检验共享 V3/V4 utility head 是否存在梯度冲突，保持 17 候选、Set Transformer
depth=2、soft oracle target `tau=5 mm`、T=1.8、10 direct + 5 GHT epochs 和
相同 modulo-10 holdout，仅将训练任务拆成 V3（四个三视角任务）与 V4（一个四视角
任务）两个独立 checkpoint。该实验不改变 RUMPL 或候选生成。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 共享 soft-oracle E2 | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |
| V3/V4 specialist | 29.4251 | 29.4014 | **29.4132** | 28.6668 | 28.7061 | **28.6864** |
| specialist 相对共享 | -0.0102 | -0.0049 | **-0.0076** | -0.0675 | -0.0134 | **-0.0405** |

两 seed 的 V4 方向一致，V3 只有约 `0.008 mm` 的均值收益。它证明共享 head
存在轻微跨视角数冲突，但收益不足以作为论文主创新；如果后续组合，最多作为
“view-count-specific utility head”消融或实现细节。输出目录：
`.../Pairwise_Stage_Specialist_20260812/v{3,4}_seed{0,1}/`。

## View-count specialist utility + fixed-lag temporal（2026-08-12）

为检验时序收益是否被共享 V3/V4 utility head 的梯度冲突掩盖，V3、V4 分别使用
独立 specialist utility checkpoint，并在完全相同的 T=9 pairwise candidate
窗口上训练固定滞后 residual。最终按稀疏单帧协议与 dense temporal cache 的共同
1986 个中心重新评估，避免把采样分布差异误判为时序收益。

| 方法 | V3 时序 | V3 中心 E2 | V4 时序 | V4 中心 E2 |
|---|---:|---:|---:|---:|
| specialist + fixed-lag，seed0 | **29.4477** | 29.6025 | **28.7082** | 28.8235 |
| 相对中心帧 | `-0.1548` | — | `-0.1153` | — |

共享 head 的 aligned gain 为约 `0.159/0.176 mm`，专用 head 后仍为同量级，说明
时序小收益不是由 V3/V4 梯度冲突造成；没有证据继续增加 MixSTE 深度、窗口或时序
seed。输出：

- `.../Pairwise_Specialist_Temporal_Models_20260812/v3_seed0/`
- `.../Pairwise_Specialist_Temporal_Models_20260812/v4_seed0_rerun/`
- `.../Pairwise_Specialist_Temporal_Alignment_Audit_20260812/v{3,4}_seed0/`

## GHT whole-pose score + E2 hybrid（2026-08-13）

为检验 E2 逐关节效用是否缺少完整姿态层面的候选评分，新增一个 GHT-style
whole-pose score branch。该支路输入保留绝对世界坐标（不是 C0 的纯 root-relative
输入），最后一层零初始化，因此初始输出与 E2 完全一致；其余候选、损失、温度和
训练协议不变，两个 seed 从 scratch 训练。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| pairwise E2 + soft oracle ranking | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |
| + zero-init GHT whole-pose branch | 29.4396 | 29.4040 | **29.4218** | 28.7448 | 28.7355 | **28.7401** |

混合支路相对 E2 的均值变化为 `+0.0010/+0.0132 mm`，两个 seed 没有同向提升，
因此“整体姿态评分与逐关节效用简单相加”停止，不把它作为主模型。它排除了
继续堆叠独立 GHT score 的必要性；下一步改查逐关节软融合产生的骨架一致性问题。

代码与输出：

- `train_h76_pairwise_hybrid_utility_20260813.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Pairwise_Hybrid_GHT_E2_20260813/`

## Staged bone-length consistency（2026-08-13）

E2 的 soft fusion 为每个关节独立混合候选，理论上可能产生骨长不一致。依据
FLEX/图结构姿态方法，在 GHT 阶段加入训练 GT 的 16 条骨长 Smooth-L1，direct
阶段保持完全原 E2；推理时不输入 GT 或骨长。为避免 seed 偏差，两个权重都做了
seed-paired 对照。

| 权重 | seed0 V3/V4 | seed1 V3/V4 | 两 seed均值 V3/V4 | 相对同 seed E2 |
|---:|---:|---:|---:|---:|
| `λ=0.1` | 29.4354 / 28.7344 | 29.4063 / 28.7195 | 29.4209 / 28.7269 | 约 `+0.0001/+0.0001` |
| `λ=0.5` | 29.4361 / 28.7350 | 29.4074 / 28.7205 | 29.4217 / 28.7278 | 约 `+0.0010/+0.0010` |

骨长项对 holdout 训练损失有响应，但跨主体 S9/S11 没有改善，较大权重还有轻微
退化。因此停止 bone-loss 分支，不再叠加 symmetry/temporal；它作为“逐关节
融合误差不是主要由骨长不一致造成”的负消融保留。

代码与输出：

- `train_h76_pairwise_bone_utility_20260813.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Pairwise_Bone_E2_20260813/`

## GHT Gumbel-Softmax hypothesis exploration（2026-08-13）

GHT（CVPR 2022）使用 Gumbel-Softmax 进行候选假设探索，以减轻训练早期评分器
过早塌缩到单一假设的问题。这里仅在 5 轮 GHT fine-tuning 中替换训练时的
candidate softmax 为 `gumbel_softmax(tau=1.0)`；直接效用预训练、17 个候选、
soft-oracle target（`tau=5 mm`）、推理温度 `T=1.8` 和评估协议全部保持 E2
不变。测试时仍用确定性 softmax，因此没有随机推理噪声。

| 方法 | V3 seed0 | V3 seed1 | V3 均值 | V4 seed0 | V4 seed1 | V4 均值 |
|---|---:|---:|---:|---:|---:|---:|
| pairwise E2 + soft oracle ranking | 29.4353 | 29.4063 | **29.4208** | 28.7343 | 28.7195 | **28.7269** |
| + GHT Gumbel-Softmax（tau=1.0） | 29.4389 | 29.4109 | **29.4249** | 28.7269 | 28.7134 | **28.7201** |

相对同 seed E2，均值变化为 `+0.0041 mm`（V3）和 `-0.0067 mm`（V4），远小于
seed 波动且没有两个视角阶段同向提升。说明在当前候选池和 E2 评分器下，GHT 的
采样式候选探索不是主要瓶颈；它没有把 22/19 mm 的候选 oracle 优势转化为最终
融合精度。因此停止继续扫描 Gumbel 温度、采样次数或更多 fine-tuning 轮数，
不把该项作为主创新。

代码与输出：

- `train_h76_pairwise_gumbel_utility_20260813.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Pairwise_Gumbel_E2_20260813/`

## Geometry-biased candidate utility residual（2026-08-14）

GBT 的核心是把 calibrated ray geometry 显式用于注意力/可靠性，而不是只依赖
候选坐标。基于候选池诊断中视线夹角与条件谱对真实候选误差的相关性，新增一个
零初始化的关节级 residual head：对每个候选的 active views 汇总 Plucker/ray
方向、ray moment、视线夹角、射线间距和条件谱；E2 Set Transformer、候选池、
损失、温度和推理均不变。`full` 使用全部摘要，`angle_distance` 只使用夹角/
间距与条件谱。两个模式各跑一个 seed。

| 方法 | V3 | V4 | 相对 pairwise E2 |
|---|---:|---:|---:|
| pairwise E2（同 seed） | 29.4208 | 28.7269 | — |
| + geometry bias residual（full） | 29.4072 | 28.7327 | −0.0136 / +0.0058 |
| + angle/distance residual | 29.4247 | 28.7455 | +0.0040 / +0.0186 |

`full` 只在 V3 产生约 0.014 mm 的单 seed 漂移，V4 反向；轻量模式两项均退化，
没有达到双视角数同向、超过 seed 波动的门槛。因此停止继续堆 Plucker 统计量或
geometry head，不把该项作为主创新。它排除了“E2 只缺少一个简单几何摘要”这一
解释，下一步应改变最终候选融合机制，而非继续增加输入特征。

代码与输出：

- `train_h76_pairwise_geometry_bias_20260814.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Pairwise_Geometry_Bias_E2_20260814/`

## E2 final fusion solver diagnostic（2026-08-14）

在不重新训练、不使用 GT 推理的前提下，直接对两个正式 E2 checkpoint 的 logits
比较了 full softmax、sparsemax 和 top-k masked softmax。若问题只是低置信候选被
平均进最终结果，稀疏求解器应能在同一 checkpoint 上改善；实际两 seed 测试均
显示 sparsemax 与 top-k 都退化：

| 求解器 | V3 两 seed均值 | V4 两 seed均值 |
|---|---:|---:|
| E2 softmax T=1.8 | **29.4208** | **28.7269** |
| sparsemax | 29.7637 | 29.1860 |
| top-3 masked softmax | 29.7068 | 29.3268 |
| top-5 masked softmax | 29.4469 | 29.1128 |

硬选择、top-2/top-4 也没有同时超过 E2。说明 E2 的收益依赖保留多个候选的
连续加权，不能靠稀疏化或换聚合器解决；下一步不实现 sparsemax/top-k 训练，转向
直接学习候选生成/几何求解过程。

评估脚本：`/tmp/eval_sparse_topk.py`（只读已有 E2 checkpoint，未产生训练权重）。

## Canonical GHT ScoreNN preprocessing（2026-08-13）

前面的 whole-pose hybrid 只把绝对/相对坐标送入新分支，尚未严格复刻 GHT 官方
`PoseDSAC.__score_nn` 的输入。因此补做严格控制：对每个 pairwise 候选先按训练
统计量标准化、以 pelvis 居中、用肩线/骨盆法向旋转到 canonical frame，再按 GHT
官方 H36M connectivity 计算 16 条 bone lengths；完整姿态评分器使用官方
`50-50-50-1 ReLU6` 结构。`mode=1` 输入 canonical pose+bone lengths，
`mode=2` 只输入 bone lengths。新增分支末层零初始化，E2、候选池、损失和温度
均不变，两个模式各跑一个 seed。

| 方法 | V3 | V4 | 相对 pairwise E2 |
|---|---:|---:|---:|
| pairwise E2（同协议均值） | 29.4208 | 28.7269 | — |
| canonical GHT `mode=1`（pose+length） | 29.4350 | 28.7473 | +0.0142 / +0.0204 |
| canonical GHT `mode=2`（length only） | 29.4165 | 28.7298 | −0.0043 / +0.0029 |

`mode=2` 的 V3 微小变化没有在 V4 同向出现，且远低于 seed/训练波动；严格复刻
GHT 的 canonical/body-length score 仍不能缩小 E2 的候选 oracle gap。因此停止
继续扫描 GHT body-length mode、canonical 旋转细节或 GHT 轮数，不将该分支作为
主创新。它同时说明问题不在“是否加入一个 whole-pose 骨架先验”，而在当前
逐关节候选效用如何利用视角观测。

代码与输出：

- `train_h76_pairwise_canonical_ght_20260813.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Pairwise_Canonical_GHT_E2_20260813/`

## Learnable Triangulation candidate extension（2026-08-14）

为改变“只改候选打分、不改候选生成”的瓶颈，按照 Learnable Triangulation
（ICCV 2019）的可微加权射线交点、AdaFuse 的视角可靠性思想，以及 RUMPL 已有
tri-anchor 的同一加权线交点，新增一个只读取 `[ray direction, ray origin,
confidence]` 的可靠性网络。网络直接预测逐关节逐视角权重，再求解正则化加权
DLT；不读取 GT、不读取 H76 候选，推理接口保持真实可用。独立视角 MLP 和一层
跨视角 Transformer 两个版本各训练 12 轮。

### 直接输出诊断

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76/RUMPL baseline | 34.8163 | 30.4890 | 29.6913 |
| 固定置信度 DLT | 50.9928 | 42.3455 | 41.4305 |
| learned triangulation（independent） | 50.7637 | 41.8525 | 40.5845 |
| learned triangulation（cross-view） | **50.7109** | **41.5628** | **40.1844** |

跨视角版本相对固定 DLT 的改善为 V2/V3/V4 `0.282/0.783/1.246 mm`，说明可靠性
网络确实能学习一部分视角权重；但它仍比 H76 高 `15.9/11.1/10.5 mm`，不能替换
RUMPL 的候选生成。该结果也排除了“仅把 DLT 换成可学习权重即可解决正式数据误差”
的解释。

### 作为新候选加入 E2

将 5 个 V3/V4 学习三角化输出追加到原 17 个 H76+pairwise 候选中，先计算候选
oracle，再用完全相同的 E2 Set Transformer、soft-oracle ranking、温度 `T=1.8`
和 train/holdout/test 协议重训两个 seed。

| 方法 | V3 | V4 |
|---|---:|---:|
| 原 pairwise E2 两 seed均值 | 29.4208 | 28.7269 |
| 扩展候选 oracle（验证集） | 22.0819 | 18.9005 |
| 扩展候选 E2 零样本 | 29.4890 | 28.7464 |
| 扩展候选 E2 重训 seed0 | 29.3813 | 28.6382 |
| 扩展候选 E2 重训 seed1 | 29.3717 | 28.6141 |
| 扩展候选 E2 重训均值 | **29.3765** | **28.6262** |

候选 oracle 只下降约 `0.096/0.144 mm`；零样本评分反而退化，说明旧评分器不
会自动识别新候选。重训后两个 seed 均同向改善，相对原 E2 约 `0.044/0.101 mm`，
但幅度仍不足以作为主线模型创新，只能作为“候选多样化”辅助消融。后续不再扫描
该候选的温度、稀疏求解器或更多训练轮次，主线转向真正提升 H76 候选质量的模型化
观测/三角化过程。

代码与输出：

- `train_h76_learnable_triangulation_20260814.py`
- `export_h76_learned_candidates_20260814.py`
- `eval_h76_learned_candidate_e2_20260814.py`
- `train_h76_learned_candidate_e2_20260814.py`
- `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Learnable_Triangulation_20260814/`

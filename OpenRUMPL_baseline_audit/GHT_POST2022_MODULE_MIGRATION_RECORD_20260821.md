# GHT 后续模块严谨移植记录（2026-08-21）

## 1. 固定实验边界

- 输入不变：HRNet 2D 坐标、关节置信度、相机参数；不使用热图或图像特征。
- 数据协议不变：H36M `S1/S5/S6/S7/S8` 训练，`S9/S11` 测试。
- 短程模型选择：训练主体中整主体 `S8` 留出；测试集不用于调参。
- 冻结基座：E2 candidate generator、limb utility 初始 proposal、K96 scorer。
- 当前正式参考：limb-utility proposal K96，`38.017/29.143/27.089 mm`。
- 当前 sampled oracle：`36.258/27.198/25.236 mm`。

## 2. E3：Generalized Differentiable RANSAC

### 官方来源

- 论文：Wei et al., *Generalized Differentiable RANSAC*, ICCV 2023。
- 官方仓库：`reference/differentiable-ransac-official/`。
- 官方 commit/远端：`https://github.com/weitong8591/differentiable_ransac.git`。
- 对照文件：`samplers/gumbel_sampler.py`、`ransac.py`。

### 保留的官方关键细节

1. Gumbel-Max 随机采样；
2. `hard - soft.detach() + soft` straight-through 梯度；
3. 硬前向只选择合法的离散候选；
4. 对采样模型的期望风险直接训练 proposal；
5. proposal 与可微求解/最终风险相连，而非仅扰动最终融合权重。

### 针对 RUMPL/K96 的必要适配

- 最小样本不再是点匹配，而是五个人体 part 各自的 candidate label；
- 每个 hypothesis 的五个 part 独立采样，part 内保持结构一致；
- proposal 前向分布严格保留历史 K96：
  `0.8 * softmax(-cost / 0.8) + 0.2 / candidate_count`；
- 保留 baseline、E2-hard、E2-soft 三个安全控制候选；
- E2、几何候选及 K96 scorer 冻结，只训练 proposal。

### 代码与验证

- 训练：`train_e3_differentiable_ransac_proposal_20260821.py`；
- 并行门控：`launch_e3_dransac_screens_20260821.sh`；
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e3_dransac/`。
- 冒烟测试：梯度范数 `0.3171`，参数 L2 变化 `0.00199`，无 in-place 错误。
- 正式短筛：官方退火与保守强锚定两组，GPU 0 并行运行。

### 停止规则

- S8 三个视角数平均改善 `<0.15 mm`：不进入 S9/S11；
- sampled oracle 没有改善：proposal 方向关闭；
- 仅 S8 过门后才跑完整验证和第二种子。

### 首轮门控结果（持续更新）

| 配置 | S8 初始均值 | S8 最佳均值 | 改善 | 门槛 |
|---|---:|---:|---:|---|
| conservative anchor | 14.7082 | 14.6771 | 0.0311 mm | 未过 0.15 mm |
| official temperature annealing | 14.7082 | 14.6653（epoch 2 暂定） | 0.0428 mm | 运行中 |

保守配置证明 proposal 梯度有效、三列同向改善，但幅度不足；不会因为正向
结果就越过预设门槛直接测试 S9/S11。

## 3. E4：PCT/UniCodebook 两阶段离散先验

### 官方来源

- PCT 官方仓库：`/mnt/data/cjydata/reference_code/PCT`；
- 远端：`https://github.com/Gengzigang/PCT.git`；commit `6f356f6`；
- 对照文件：`models/pct_tokenizer.py`、`models/modules.py`、
  `models/pct_loss.py`、`configs/pct_base_tokenizer.py`；
- 新方法依据：Chen et al., *Unified 2D-3D Discrete Priors for
  Noise-Robust and Calibration-Free Multiview 3D Human Pose Estimation*,
  NeurIPS 2025。

### Stage-I 已实现细节

1. joint/channel MLP-Mixer encoder；
2. 34 个 compositional tokens；
3. hard nearest-code assignment；
4. EMA codebook，decay `0.9`；
5. straight-through quantization；
6. 4 层 encoder、1 层 decoder；
7. 随机关节 mask `0.2`；
8. Smooth-L1 `beta=0.05`；
9. commitment 权重 `15`；
10. 整主体 S8 留出，root-relative 3D pose 训练，不看测试集。

与官方不同之处必须在论文中明确：PCT 原始 tokenizer 针对 2D COCO pose，
这里依据 UniCodebook 思路改成 H36M root-relative 3D tokenizer；由于
UniCodebook 未公开代码，因此属于论文支撑的适配，而非严格复现。

### 代码与队列

- Stage-I：`train_e4_pct_3d_tokenizer_stage1_20260821.py`；
- 并行队列：`launch_e4_pct_stage1_screens_20260821.sh`；
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_pct/`；
- 两组容量：34 tokens，codebook 512/1024；其余训练机制相同；
- E3 完成后自动启动，避免三个重计算任务争用同一张 GPU。

Stage-II 必须冻结完整 tokenizer，仅将离散-连续交互特征作为 K96 scorer
的零初始化残差，禁止用测试主体更新 codebook。

### Stage-I 首轮失败与防塌缩修复

原 PCT EMA 结果发生 representation collapse：

| 量化器 | S8 RR 重建 MPJPE | 活跃码 | perplexity |
|---|---:|---:|---:|
| EMA-512 | 62.31 mm | 6/512 | 5.90 |
| EMA-1024 | 65.64 mm | 4/1024 | 3.94 |

因此未接 Stage-II。后续修复严格来自两篇带官方实现的量化论文：

1. SimVQ，ICCV 2025：冻结 Gaussian coefficient codebook，仅学习共享线性
   basis transformation；官方仓库 `reference/SimVQ-official/`，commit
   `d8bd94d`，核心 `taming/modules/vqvae/quantize.py`。
2. FSQ，ICLR 2024：固定有限标量 levels、bound + round-STE，不需要 EMA、
   commitment 或 dead-code 重置；Google 官方 JAX 代码保存于
   `reference/fsq-google-official/fsq.ipynb`。

四组正式短筛由 `launch_e4_anticollapse_screens_20260821.sh` 并行运行，输出
在 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_anticollapse/`。epoch 2
初步结果：SimVQ 使用 `469/512`、`865/1024` 个码；FSQ 使用 `932/1000`、
`3003/4375` 个码，确认已消除原始塌缩机制。最终选择仍以完整 12 epoch 的
真实毫米重建误差、利用率和 S8 下游门控为准。

完整 12 epoch 与毫米审计结果：

| tokenizer | 最佳 epoch | S8 RR MPJPE | active/size | perplexity |
|---|---:|---:|---:|---:|
| FSQ `[8,5,5,5]` | 10 | 21.754 mm | 961/1000 | 637.7 |
| **FSQ `[7,5,5,5,5]`** | **11** | **20.453 mm** | **3445/4375** | **1665.1** |
| SimVQ-1024 | 12 | 26.531 mm | 1024/1024 | 623.6 |
| SimVQ-512 | 12 | 26.823 mm | 512/512 | 324.8 |

审计文件：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_anticollapse/rr_mpjpe_audit.json`。
FSQ-4375 为主实验，FSQ-1000 与 SimVQ-1024 为量化器对照。

### Stage-II DCSA-style K96 scorer

- 代码：`train_e4_discrete_continuous_k96_scorer_20260821.py`；
- 启动器：`launch_e4_stage2_dcsa_screens_20260821.sh`；
- tokenizer、E2、proposal、K96 scorer 全部冻结；
- continuous tokens 查询 discrete tokens，再向原 K96 score 加零初始化残差；
- 冒烟测试已验证初始近似恒等、反向梯度存在；
- FSQ-4375、FSQ-1000、SimVQ-1024 三组在 GPU 0 并行训练；
- 仍执行 S8 `0.15 mm` 门控，未过门不评 S9/S11。

## 4. 已关闭方向

`observation_conditioned_k96_screen_seed0` 已完成：S8 平均退化
`0.00688 mm`，未过 `0.15 mm` 门槛，不进入正式验证，也不继续扩深网络。

## 5. E5：射线条件化 ST/TS 双流（2026-08-21）

### 适配性结论与论文边界

UniCodebook 的“dual-stream”严格指并行的 `Spatial→Temporal` 与
`Temporal→Spatial` 两条分支，并非“2D 流+3D 流”。其 DCSA 才负责以连续特征为
query、离散 token 为 key/value 的跨表示交互。E5 保留这一有论文依据的计算结构，
但将无公开实现的离散 token 换成 RUMPL 已有的逐视角观测证据：HRNet 2D 所形成的
世界射线、置信度、K96 锚点到射线的有符号垂距和深度。因此 E5 是针对 RUMPL 的
模块适配，不标为 UniCodebook 严格复现。

该方向与已失败实验的区别：

- H19 只输入融合后的 3D 姿态，无法知道哪一个视角/关节在当前帧不可靠；
- 单帧 observation-conditioned K96 只重排冻结候选，无法跨帧修正姿态；
- E5 在每层将逐视角观测证据注入姿态 token，再并行执行 ST/TS，并由逐 token
  自适应权重融合；最终直接预测有界 3D residual。

### 固定协议

- 输入仍只有 HRNet 坐标、置信度和相机参数；不使用热图或图像特征；
- 禁止固定相机组合 embedding，只编码视角数量 2/3/4；相机差异完全由射线表达，
  保持对未见相机组合的定义和置换兼容性；
- 冻结 K96/E2-C2 作为绝对坐标锚点，输出头零初始化，step 0 严格恒等；
- T=9 causal latest-frame，训练时监督窗口全部帧；
- `S1/S5/S6/S7 → S8` 选择，S8 平均提升至少 `0.15 mm` 才允许
  `S1/S5/S6/S7/S8` refit 和 S9/S11 模型评估；
- `control` 与 `ray-cross` 仅相差射线观测交互，用于隔离双流和条件化的贡献。

### 实现与状态

- 训练：`train_e5_ray_conditioned_dual_stream_20260821.py`；
- 启动：`launch_e5_ray_conditioned_dual_stream_20260821.sh`；
- 测试：`test_e5_ray_conditioned_dual_stream_20260821.py`；
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e5_ray_dual_stream/`；
- 两项单元测试通过：初始输出逐元素等于 K96；未选择视角对任务输出严格无影响；
- 首次启动在训练前发现 `h18_clean_temporal_cache` 实际是旧的 22 候选 E2-C2
  锚点（dense validation `38.876/29.689/27.419 mm`），不是当前正式 K96。两组在
  首个 epoch 完成前立即停止，结果作废，避免在较差基线上得出结论。
- 新增 `build_k96_temporal_anchor_cache_20260821.py`，严格使用正式 K96 的 seed=10000、
  batch=192、11 个任务顺序，生成 train/dense validation 的逐帧锚点；16 帧端到端
  冒烟通过。
- `run_e5_after_k96_cache_20260821.sh` 已在 GPU 0 启动：先完成严格 K96 缓存，随后
  自动并行启动 `ray_seed0`（batch 32）和 `control_seed0`（batch 64）。GPU 1 的已有
  偏置实验不受影响。

严格 dense temporal K96 缓存已完成，S9/S11 action-equal 为
`38.026/29.186/27.133 mm`，与原稀疏正式 K96 `38.017/29.143/27.089` 仅差
`0.009/0.043/0.045 mm`，说明新时序锚点口径正确。ray/control 随后已自动进入
metadata/window 构建阶段。

FSQ 不在首轮同时加入。只有 ray-conditioned 双流通过 clean S8 门槛，才允许把冻结
FSQ token 作为第二组 key/value 做单变量消融；这避免把已证明在 K96 重评分上无效
的离散先验与新结构混在一起，无法判断收益来源。

### 首轮运行审计（14:11 更新）

旧的串行 ray encoder 在 44 分钟内仍未完成第 1 epoch；GPU 利用率低，确认是逐11任务
调用 view attention 的 kernel-launch 瓶颈，不是训练卡死。该运行与同时被父 shell
管理的 control 均在完成前停止，目录移动到 `aborted_prevector_*`，不计正式结果。

停止前 control 的 S8 中间结果可作为趋势诊断：

| epoch | V2 delta | V3 delta | V4 delta | V234 平均改善 |
|---:|---:|---:|---:|---:|
| 0 | -0.244 | -0.015 | +0.059 | 0.067 mm |
| 1 | -0.334 | -0.032 | +0.057 | 0.103 mm |

尚未达到 `0.15 mm` 门槛，且 V4 有轻微退化，因此不能称为有效结果。

ray observation encoder 已改为11任务完全向量化，未改变特征、mask、attention 或损失；
单元测试再次通过。现以两个独立 tmux 重启，避免一组中止连带另一组：

- `cjy_e5_ray_vec_20260821`：ray batch 16，约 5.85 GB；
- `cjy_e5_control_20260821`：control batch 64，约 6.26 GB。

### 15:23 重启说明

两组未完成的 memmap 版本已分别移入 `aborted_memmap_*`，没有删除。当前版本增加
`--preload-fused`，将约 0.7 GB 的 train/validation K96 时序锚点预加载到 RAM；主机
剩余内存约 205 GB，GPU0 两进程显存约 12.6 GB，训练稳定运行。新版本尚未完成第一
个 epoch，因而目前没有新的正式精度数字。

### 15:31 偏置实验处置

GPU1 的 `cjy_raymix_gbt_trainable_20260821` 是本项目早前启动的 GBT trainable
view-bias 任务，不属于其他用户任务。按当前主线“不加入偏置”的决定，已停止该进程
和 tmux 会话；其 checkpoint、日志和输出均保留，未删除，后续如需复核可恢复。当前
GPU1 已释放。GPU0 的 E5 ray-conditioned 双流与 no-ray control 两组继续运行。

随后为避免 GPU0 两进程争用，将尚未完成首个 epoch 的 control 组连同其未完成目录
保存为 `control_seed0_interrupted_gpu0_20260821_1536`，重新置于 GPU1 启动为同一
配置、同一 seed、同一输出口径；ray 组继续在 GPU0。迁移不产生结果，不改变实验协议。

### 16:45 首轮中间结果

两组仍未结束全部 4 个 epoch，以下仅为 S8 holdout 选择结果，不能替代最终 S9/S11：

| 配置 | 已完成 epoch | 最佳 S8 均值 | 相对基线 | V2/V3/V4 绝对变化 |
|---|---:|---:|---:|---|
| no-ray control | 0--2 | 14.7126 mm（epoch 1） | -0.1031 mm | -0.334 / -0.032 / +0.057 mm |
| ray-conditioned | 0 | 14.6605 mm（epoch 0） | -0.1552 mm | -0.455 / -0.054 / +0.043 mm |

当前 `ray-conditioned` 首轮已刚好通过 0.15 mm 门槛，但仍需完成其余 epoch；
control 暂未通过门槛。两组最终是否进入正式 S9/S11 评估，以完整选择过程和门控结果为准。

### 18:04 进度更新

`no-ray control` 已完成 4 个选择 epoch。最佳仍为 epoch 1：S8 均值
`14.7126 mm`，相对基线仅改善 `0.1031 mm`，未通过 `0.15 mm` 门槛，因此没有
进行正式 refit；其 S9/S11 输出保持冻结 K96 基线 `38.164/29.308/27.242 mm`。

`ray-conditioned` 已完成 epoch 0--1。epoch 0 的 `14.6605 mm` 仍是当前最佳，
epoch 1 回升到 `14.8270 mm`，说明暂时的提升不稳定；该进程继续完成剩余 epoch，
尚未产生正式 S9/S11 结果。

### E5 停止与 baseline 冻结（2026-08-21）

按用户决定停止 ray-conditioned dual-stream；其后续 epoch 和正式 S9/S11 不再运行。
综合历史结果，论文公平主 baseline 冻结为 E2-C2 soft-cal T=1：
`38.700/29.486/27.274 mm`。同一 HRNet 坐标输入下的最佳时序内部结果为
H18-lowLR T=9：`37.704/29.231/27.219 mm`，但因中心帧使用未来上下文，只作为
时序消融参考，不替代 T=1 公平主 baseline。完整冻结口径见
`BASELINE_FREEZE_20260821.md`。

### 进度追加：ray epoch 2

ray-conditioned 的 epoch 2 已完成，S8 均值升至 `14.9925 mm`，继续劣于冻结基线
`14.8157 mm`；V3/V4 分别退化 `0.313/0.405 mm`。当前最佳仍为 epoch 0 的
`14.6605 mm`，训练尚剩最后一个 epoch，正式测试尚未启动。该现象说明前期小幅
改善尚不稳定，不能直接作为时序有效结论。

### ResNet 输入阶段纠正（2026-08-21 20:13）

19:56 启动的 ResNet R0/H76 试跑在完成前停止，原因是它没有接入已冻结的
E2-C2→H18-lowLR 主线，且使用旧的 `3:1:1 + 前8轮固定K=2` 配置。其日志和未完成
权重保留但不计结果。

正式脚本为 `launch_resnet152_e2_c2_h18_lowlr_20260821.sh`，固定执行：

1. ResNet-152 H76 源模型，C2 调好的 `8:1:1` V2/V3/V4 比例，20 epoch，取消固定
   视角覆盖；
2. ResNet-H76 候选导出、22 候选 E2-C2 scorer（V2/V3/V4 同一模型、两 seed、
   `T=0.4/1.8/1.8`）；
3. 与 HRNet 完全相同的 H18-lowLR：T=9、stride=5、hidden=96、2层、
   lr=5e-5、wd=5e-4、12 epoch。

20:13 已重新启动，当前日志确认采样器为
`weighted-random-8,1,1 capacity=2-4 fixed_epochs=0`，处于 ResNet-H76 第0轮。
正式输出根目录为 `/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/`。

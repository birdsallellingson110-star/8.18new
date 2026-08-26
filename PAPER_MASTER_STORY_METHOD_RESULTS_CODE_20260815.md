# 论文初稿生成母文档：全局关节查询与多假设效用融合

> 版本：2026-08-22（第一阶段 H36M clean 结果冻结版）
> 用途：将本文件整体交给网页版 Codex，生成中文或英文论文初稿。  
> 重要规则：本文件把“已完成的事实”“历史探索结果”“正在运行”“计划/目标”严格分开。写论文时不得把 oracle、计划值、不同输入协议或正在运行的实验写成正式主结果；不得虚构缺失数值。

---

## 最高优先级路线覆盖声明（2026-08-22）

本节及其后紧接的“当前冻结技术方案”和“GBT 对齐实验总表”是本文档的
**最高优先级事实源**。若后面的 2026-08-15 历史稿与本节冲突，必须以本节
为准。历史稿中的 RIGR/A1D 图像特征、HRNet patch correspondence、可训练
view bias、蒸馏、FSQ/codebook、旧双流时序等均属于探索记录，**不是当前模型
baseline 的组成部分**，不得在摘要、方法图或主结果中写成已经采用的方法。

当前路线冻结为三个逐级版本：

- `GQ-RUMPL`：RUMPL/H76 射线主干 + 全局 Joint-Query residual，单帧；
- `GQ-RUMPL-E2`：在同一生成器上增加 E2-C2 多假设效用软融合，单帧；
- `GQ-RUMPL-E2-T`：在冻结空间模型后增加 H18，9 帧中心窗口。

这三个名称是实验记录名，不是最终论文标题。最终题目和“核心创新模块”必须等
H36M clean、H36M-Occl 和消融表完成后再决定，不能提前把尚未证实的模块写成
主要贡献。

以上三段链是当前 **ResNet 主方法线**。HRNet 上的 Global Joint-Query 迁移没有
稳定改善，因此 HRNet 第一阶段冻结为 `C2 -> E2-C2 -> H18`，作为相同坐标输入
层级下的 detector-specific 对照；不得把 HRNet 行误标成 `GQ-RUMPL`。

### 给网页版 Codex 的当前写作指令

1. 当前总目标是严格沿用 GBT 的实验组织方式：H36M 双输入 clean baseline、
   H36M-Occl、Occlusion-Person、CMU、CMU→H36M、组件消融和时序长度。
2. 第一阶段 H36M clean 已完成并冻结；第二阶段 H36M-Occl 尚未产出结果，后续
   阶段不得被写成已完成。第一阶段独立事实源为
   `STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`。
3. 3D 网络始终只接收 2D 坐标、置信度和相机标定射线，不接收 RGB、heatmap、
   bbox、HRNet/ResNet 中间特征；两条线只有冻结 2D 前端不同。
4. HRNet 与 ResNet 使用相同输入层级、训练/测试 subjects、指标和相机组合口径；
   允许针对冻结二维前端选择不同训练超参数。Global Joint-Query 在 ResNet 上有效，
   但 HRNet 迁移实验未形成统一提升，因此最终 HRNet 行保留 C2+E2+H18，不能虚构
   成两条前端具有完全相同的最终模块链。
5. 所有 `TBD` 都必须保留，不能由语言模型补齐；GBT 数字只能标成
   `GBT-reported`，不能冒充我们的结果或严格复现结果。
6. 单帧 `T=1` 与时序 `T=9` 必须分行。当前 H18 是中心窗口、非因果模块，不能
   写成 GBT 的 causal latest-frame 实现；T=2/T=6 需另写因果版本后才能填表。
7. 不使用 S9/S11 调温度、选 checkpoint 或挑 seed。主指标为 absolute All-17
   MPJPE，无 root alignment 或 Procrustes alignment，并平均全部相机组合。
8. “identity-hinge”不是让 V4 降至约 20 mm 的主要原因；主要收益来自 E2-C2
   候选池与软效用融合，hinge 只用于抑制相对 base 的小幅退化。

---

## A. 当前冻结技术方案

### A.1 研究目标与输入协议

给定同步多视角帧，冻结二维前端为每个时间 (t)、视角 (v)、关节 (j)
输出像素坐标与置信度：

\[
\mathbf x_{tvj}=(u,v,1)^\top,\qquad c_{tvj}\in[0,1].
\]

相机标定为 \((\mathbf K_v,\mathbf R_v,\mathbf t_v)\)，相机中心和世界坐标
单位射线为：

\[
\mathbf o_v=-\mathbf R_v^\top\mathbf t_v,
\qquad
\mathbf d_{tvj}=\operatorname{normalize}
(\mathbf R_v^\top\mathbf K_v^{-1}\mathbf x_{tvj}).
\]

模型输出无对齐的绝对世界坐标 3D pose。当前正式对比包含两条坐标输入：

- `HRNet`：HRNet-W32 COCO 关键点与 YOLOX-X person box；
- `ResNet-152†`：Learnable Triangulation 官方 H36M-finetuned ResNet-152
  关键点与置信度。

二者的 3D 网络不读取图像或热图。因此这仍是坐标级方法，前端替换不会把我们
变成 heatmap/volumetric 方法；但论文表中必须明确 detector，不能隐藏输入差异。

### A.2 置信度三角化锚点与中心化 Plücker 射线

每条射线的法平面投影矩阵为

\[
\mathbf P_{vj}=\mathbf I-\mathbf d_{vj}\mathbf d_{vj}^{\top},
\qquad w_{vj}=c_{vj}+\epsilon.
\]

以加权射线最小二乘得到关节锚点：

\[
\mathbf A_j=\sum_vw_{vj}\mathbf P_{vj}+\lambda\mathbf I,
\quad
\mathbf b_j=\sum_vw_{vj}\mathbf P_{vj}\mathbf o_v,
\quad
\mathbf a_j=\operatorname{solve}(\mathbf A_j,\mathbf b_j),
\]

其中 \(\epsilon=0.05\)、\(\lambda=10^{-4}\)。以 root 锚点平移相机原点
\(\mathbf o'_v=\mathbf o_v-\mathbf a_{root}\)，中心化 Plücker moment 为

\[
\mathbf m_{vj}=\mathbf o'_v\times\mathbf d_{vj}.
\]

共享 ray encoder 编码方向、moment 与置信度，不使用 camera-ID lookup 或固定
camera-slot embedding，因而保留 RUMPL 的相机布局泛化性。

### A.3 保留的 RUMPL/H76 主干

RUMPL 先用 VFT 对每个关节跨视角融合，再用 PFT 建模 17 个关节：

\[
\mathbf h_j=\operatorname{VFT}(\{\mathbf z_{vj}\}_{v\in\mathcal V}),
\quad
\mathbf H=\operatorname{PFT}([\mathbf h_1,\ldots,\mathbf h_{17}]),
\quad
\hat{\mathbf p}^{base}_j=\mathbf a_j+f_{3D}(\mathbf H_j).
\]

这一分支保留 RUMPL 最重要的相机无关射线表示、视角共享权重和可变相机数量
能力。其瓶颈是 VFT 会过早把每个关节的多视角 token 压成一个 token，使腕、踝
等困难关节无法在压缩前利用完整人体与全部视角证据。

### A.4 全局 Joint-Query residual

在 VFT 之前保留全部 `joint × view` token 作为 memory：

\[
\mathcal M=\{\mathbf z_{vj}+\mathbf e_j^{mem}\mid
j=1\ldots17,\ v\in\mathcal V\}.
\]

每个 learned joint query 由其三角化锚点条件化：

\[
\mathbf q_j=\mathbf q_j^{learn}+\mathbf W_a\mathbf a_j,
\qquad
\tilde{\mathbf q}_{1:17}=\operatorname{Decoder}
(\mathbf q_{1:17},\mathcal M).
\]

两层 Transformer decoder 允许每个关节同时访问所有关节、所有视角，然后只
预测一个有界残差：

\[
\Delta\mathbf p_j=0.5\tanh
(\mathbf W_o\operatorname{LN}(\tilde{\mathbf q}_j)),
\qquad
\hat{\mathbf p}^{GQ}_j=\hat{\mathbf p}^{base}_j+\Delta\mathbf p_j.
\]

输出层 \(\mathbf W_o\) 零初始化，使新增支路在训练第 0 步是严格 identity。
与已失败的“冻结主干后接 query adapter”不同，这里 Joint-Query 和完整 RUMPL
主干联合训练。该支路修复的是 RUMPL 的早期逐关节视角压缩，而不是改变输入。

### A.5 生成器训练协议

- H36M 训练：S1/S5/S6/S7/S8；测试：S9/S11；
- seed 0，20 epochs，学习率 `1e-4`；
- epoch 0--7：均匀随机采样 K=2 相机组合；
- epoch 8--19：V2:V3:V4 任务比例 `3:1:1`；
- 同一个 checkpoint 评估 6 个 V2、4 个 V3 和 1 个 V4 组合；
- clean baseline 不使用遮挡增强；损失直接监督 absolute 3D coordinate。

前 8 轮强化两射线人体先验，随后回放多视角基数，避免历史上 K2-only 训练造成
V3/V4 崩溃。当前协议不是 300k iteration 的 GBT 严格复现，而是对我们主干经
审计后冻结的 matched protocol。

### A.6 E2-C2 多假设效用软融合

H36M 四相机共有 11 个合法子集：6 个 V2、4 个 V3、1 个 V4。每个子集生成两
类候选：

1. 冻结 `GQ-RUMPL` 的学习型 3D 预测；
2. 相同观测上的 confidence-weighted robust triangulation/IRLS 候选。

总计 22 个候选；测试某个可用相机集合时，包含不可用相机的候选被 mask。评分器
不使用相机编号，并在 V2/V3/V4 之间共享。

候选特征包括：绝对/根相对 pose、与候选共识的距离、点到射线距离、包含与排除
视角的残差和置信度统计、subset fraction、加权射线法矩阵特征值谱、全身上下文
和 joint identity。两层 Set Transformer 输出每个候选、每个关节的相对风险
\(r_{cj}\)，按视角数设温度进行软融合：

\[
\alpha_{cj}=\frac{\exp(-r_{cj}/\tau_K)}
{\sum_{c'}\exp(-r_{c'j}/\tau_K)},
\qquad
\hat{\mathbf p}^{E2}_j=\sum_c\alpha_{cj}\hat{\mathbf p}_{cj},
\]

其中 \(\tau_2=0.4\)，\(\tau_3=\tau_4=1.8\)。温度只能由训练 holdout 选择。
训练为 10 个 direct-risk epochs（`5e-4`）加 5 个 expected-risk epochs
（`1e-4`）。最终评分器增加 identity safeguard：

\[
L_{id}=0.25\sum_K\gamma_K
\max(0,E_{soft}^{K}-E_{base}^{K}),
\qquad \gamma_2=4,\quad\gamma_3=\gamma_4=1.
\]

这一项只避免融合器在 base 已正确时产生小幅伤害；它不是 E2 主要提升来源，也
不能被包装成单独核心创新。

### A.7 H18 时序残差

H18 放在冻结的 E2-C2 输出之后。每关节输入 12 维信号：

\[
[\mathbf p_t-\mathbf p_{t,root},\ \mathbf p_{t,root},\
\mathbf p_t-\mathbf p_{t-1},\
(\mathbf p_t-\mathbf p_{t-1})-(\mathbf p_{t-1}-\mathbf p_{t-2})].
\]

先做每帧两层 spatial Transformer，再做每关节两层 temporal Transformer，
预测中心帧有界残差：

\[
\Delta\mathbf p^{temp}=0.1\tanh(f_{ST}(\mathbf p_{t-4:t+4})),
\qquad
\hat{\mathbf p}^{T}_t=\hat{\mathbf p}^{E2}_t+\Delta\mathbf p^{temp}.
\]

输出层零初始化，root correction 强制为 0。冻结配置为 T=9、stride=5、
hidden=96、2 个 spatial + 2 个 temporal layers、12 epochs、`lr=5e-5`、
`weight_decay=5e-4`。它用于稳定正常序列并为后续遮挡实验提供历史信息；若 clean
结果退化则不能作为最终 baseline。当前实现是 centered/non-causal，必须如实写。

### A.8 当前已核实结果与状态

第一阶段完整结果另见
`/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`。本节只保留
论文主线所需的冻结结果。

ResNet-152 坐标输入下，逐级结果为：

| 方法 | T | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| ResNet H76 direct reference | 1 | 41.4704 | 26.0806 | 24.1573 |
| ResNet `GQ-RUMPL` | 1 | 32.3121 | 25.1006 | 23.5364 |
| `GQ-RUMPL-E2` identity-preserving | 1 | 32.3193 | 22.5576 | 20.2718 |
| `GQ-RUMPL-E2-T` | 9 | **31.2151** | **22.0084** | **19.9710** |

Global Joint-Query 相对 H76 在 V2/V3/V4 改善 `9.158/0.980/0.621 mm`。E2 的
主要收益位于 V3/V4；H18 在相同 dense center 子集上相对 T=1 基线
`32.437/22.581/20.306` 改善 `1.222/0.572/0.335 mm`。

在相同 22-candidate cache 上，**不含 identity-hinge 的标准 E2-C2** 已完成两个
seed，S9/S11 严格 action-equal All-17 结果为：

| 中间消融 | V2 | V3 | V4 |
|---|---:|---:|---:|
| E2-C2 standard, seed 0 | 32.3312 | 22.6603 | 20.3791 |
| E2-C2 standard, seed 1 | 32.3305 | 22.6308 | 20.3429 |
| E2-C2 standard, two-seed mean | **32.3309** | **22.6456** | **20.3610** |

它相对直接 `GQ-RUMPL` 为 `+0.019/-2.455/-3.175 mm`：V3/V4 明显改善，但
V2 微退化 0.019 mm。加入 identity safeguard 后，两 seed 均值为
`32.319/22.558/20.272 mm`，相对 standard 再改善 `0.012/0.088/0.089 mm`。
因此 identity hinge 是稳定器，不是 V3/V4 大幅提升的主要来源。

HRNet 最终保留历史强 C2/E2/H18 链，而不是失败的 Joint-Query 迁移版本：

| 方法 | T | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| HRNet C2 generator | 1 | 38.686 | 30.943 | 28.629 |
| HRNet C2 + E2-C2 | 1 | 38.700 | 29.486 | 27.274 |
| HRNet C2 + E2-C2 + H18 | 9 | **37.704** | **29.231** | **27.219** |

HRNet Query 审计的最好折中 U2 为 `38.487/30.913/28.676 mm`，但 V4 相对 C2
退化 0.047 mm；C2-direct A/B 和 highLR A/B 也均未同时改善三列。因此不能声称
Global Joint-Query 对 HRNet 有效，完整负结果表放在第一阶段独立事实源中。

上述已完成结果的审计文件与当前流水线输出位置：

- ResNet matched 根目录：
  `/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/`
- ResNet direct V2/V3/V4：上述目录下 `eval/V2|V3|V4/table2.json`
- 标准 E2-C2 两 seed 汇总：
  `e2_c2/scorer/calibrated_v2t04.json`
- identity-hinge：上述目录下 `e2_c2_identity_hinge/calibrated_v2t04.json`
- 修正后的 ResNet H18：上述目录下
  `h18_identity_hinge_v2_gtinput/result.json`
- matched HRNet 根目录：
  `/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/`
- 冻结 HRNet H18：
  `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/result.json`
- ResNet/HRNet 自动启动脚本：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_resnet_query_best_e2_h18_20260822.sh`、
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_query_matched_pipeline_20260822.sh`

关键实现与记录：

- 主干、anchor、Plücker、Joint-Query：
  `/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- E2-C2 candidate wrapper：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_current_e2_confidence_20260815.py`
- E2 loss/evaluation：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py`
- Set Transformer features：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py`
- H18：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py`
- 完整技术方案：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/JOINT_QUERY_MATCHED_FRONTENDS_PLAN_20260822.md`

### A.9 当前论文故事线与暂定贡献

当前证据支持的故事不是“堆叠三个模型”，而是一条围绕 RUMPL 信息瓶颈的递进式
修复链：

1. **问题定位**：RUMPL 的相机无关 world-ray 表示值得保留，但逐关节 VFT 在
   看见全身前就压缩所有 view token；真实 H36M 训练下，这会放大困难两相机组合
   的深度歧义。
2. **生成器修复**：Global Joint-Query 在 VFT 前保留完整 `joint × view` memory，
   让每个关节利用人体结构与所有视角证据，同时以零初始化残差保护原 RUMPL 解。
   现有 ResNet 结果已证明该修复可同时改善 V2/V3/V4，而不是用牺牲某个视角数换取
   另一列提升。
3. **集合推理修复**：不同相机子集与稳健几何求解器产生互补假设。E2-C2 不按
   固定 camera ID 选择，而是按候选与射线的一致性、人体共识和几何条件逐关节软
   融合；当前结果已证明 V3/V4 有 2--3 mm 级增益。
4. **时序鲁棒性**：H18 只对最终空间结果做 identity-initialized 小残差，不改变
   单帧几何主线。只有在 clean 不退化且 occlusion 明显获益时，才可列为正文贡献；
   否则降为遮挡增强或补充实验。
5. **泛化证据**：无 camera-ID、可变候选 mask、全组合训练/评估与 CMU→H36M
   零样本共同证明方法不是记忆 H36M 四个固定机位。

在完整实验完成前，建议暂定三项贡献为：

- 一种保留 RUMPL camera-generalizable rays、但绕过早期逐关节视角压缩的全局
  joint-query residual；
- 一种统一 V2/V3/V4 的、逐关节多假设效用软融合机制，并以 identity safeguard
  限制负融合；
- 一套在两种坐标前端、clean/occlusion、跨数据集和可变相机数下严格 matched 的
  评估与诊断。

不得声称“首次使用 Transformer、三角化、Plücker、Set Transformer、时序或
多假设评分”。最终创新表述要落在这些公开组件如何针对 **RUMPL 的早期信息压缩与
负候选融合**形成统一、可审计的结构修复上。

---

## B. GBT 对齐实验总路线与结果表模板

### B.1 七阶段路线

| 阶段 | 数据集/协议 | 必须产出的结果 | 当前状态 |
|---|---|---|---|
| 1 | H36M clean，HRNet/ResNet，全部 V2/V3/V4 组合 | 冻结空间与时序 baseline | **已完成并冻结** |
| 2 | H36M-Occl；只用 clean H36M 训练模型 | 两种输入的遮挡主表 | **协议已准备，待运行** |
| 3 | Occlusion-Person，2/3/4/8 cameras | 与 RANSAC/ScoreFuse/AdaFuse 对比 | 待数据审计 |
| 4 | CMU Panoptic in-domain | H36M/CMU 主表及 2/4/5/6/8 扩展 | 待开始 |
| 5 | CMU 训练、H36M 零样本测试 | matched joints 分动作表 | 待开始 |
| 6 | clean/occlusion/cross-dataset 组件消融 | GBT Table VI 风格累加消融 | 随各阶段同步生成 |
| 7 | T=1/2/3/6/9 | H36M 与 H36M-Occl 时序长度表 | 待写 causal/even-T 版本 |

### B.2 冻结评估协议

- H36M：S1/S5/S6/S7/S8 train，S9/S11 test；
- 按 GBT/Learnable Triangulation 口径排除 S9 的 Greeting、SittingDown、
  Waiting 错误片段；
- absolute All-17 MPJPE，单位 mm，无 root/PA alignment；
- action-equal 作为主报告，frame-weighted 结果保留供审计；
- V2 平均全部 6 组合，V3 平均全部 4 组合，V4 为全部 4 相机；
- clean 与 occlusion 必须使用同一 clean-trained checkpoint；
- 同一输入下 T=1 与 T=9 分行；不同 detector 不合并或择优拼接。

以下 `GBT-reported` 数据直接转录自
`/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`，只用作
表格结构与参考目标。我们的方法行必须从对应评估 JSON 自动填入。

### Table I. H36M clean，较少相机，absolute MPJPE (mm)

| Method | 2D input | T | 2 cams | 3 cams | 4 cams |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation [10], GBT-reported | ResNet-152† | 1 | 51.1 | 23.4 | 19.1 |
| GBT, reported | ResNet-152† | 9 | 29.9 | 24.4 | 22.7 |
| `GQ-RUMPL` (verified direct) | ResNet-152† | 1 | **32.312** | **25.101** | **23.536** |
| `+E2-C2 standard` (intermediate ablation) | ResNet-152† | 1 | 32.331 | 22.646 | 20.361 |
| `GQ-RUMPL-E2` (identity-preserving final) | ResNet-152† | 1 | **32.319** | **22.558** | **20.272** |
| `GQ-RUMPL-E2-T` | ResNet-152† | 9 | **31.215** | **22.008** | **19.971** |
| Algebraic Triangulation, GBT-reported | HRNet | 1 | 120.7 | 50.9 | 44.2 |
| GBT, reported | HRNet | 9 | 36.8 | 30.4 | 26.0 |
| HRNet C2 generator | HRNet | 1 | 38.686 | 30.943 | 28.629 |
| HRNet C2 + E2-C2 | HRNet | 1 | 38.700 | 29.486 | 27.274 |
| HRNet C2 + E2-C2 + H18 | HRNet | 9 | **37.704** | **29.231** | **27.219** |

### Table II. H36M-Occl，absolute MPJPE (mm)

| Method | 2D input | T | 2 cams | 3 cams | 4 cams |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation [10], GBT-reported | ResNet-152† | 1 | 163.3 | 39.5 | 27.9 |
| GBT, reported | ResNet-152† | 9 | 39.1 | 33.4 | 31.3 |
| `GQ-RUMPL` | ResNet-152† | 1 | 47.750 | 33.503 | 31.259 |
| `GQ-RUMPL-E2` | ResNet-152† | 1 | **47.866** | **29.236** | **24.511** |
| `GQ-RUMPL-E2-T` (centered H18, internal) | ResNet-152† | 9 | **43.416** | **26.991** | **22.985** |
| Algebraic Triangulation, GBT-reported | HRNet | 1 | 217.3 | 72.4 | 54.1 |
| GBT, reported | HRNet | 9 | 42.3 | 34.5 | 31.6 |
| HRNet C2 generator | HRNet | 1 | 47.213 | 36.095 | 32.668 |
| HRNet C2 + E2-C2 | HRNet | 1 | **47.252** | **33.509** | **29.739** |
| HRNet C2 + E2-C2 + H18 (centered, internal) | HRNet | 9 | **43.996** | **31.944** | **28.794** |

Our rows use a **GBT-described, algebraically calibrated reconstruction**
(`p=0.1`, white square side `0.15` of the longer annotation-box side, seed
`20260822`).  GBT did not release the H36M-Occl generator, square size, or seed,
so the numerical comparison is protocol-aligned but must not be called an exact
official-payload reproduction.  This qualifier is about the missing
implementation parameters (including the unspecified mask insertion point),
not about the novelty of synthetic H36M occlusion:
Sárándi's public code uses Pascal-VOC/shaped occluders, Banik et al. mask joints
over selected frame spans, and Bragagnolo et al. release a different
Human3.6M-Occluded generator with two random VOC objects in three of four views.
Those are related but non-equivalent benchmarks and their numbers should not be
merged into GBT Table II.  All T=1 rows use frozen clean checkpoints and no
H36M-Occl labels or metrics for model selection.

The H18 rows use the existing centered T=9 model (four past and four future
frames), so they are internal robustness results rather than strict causal
GBT comparisons.  The matched dense center baselines are ResNet
`47.422/29.201/24.473` and HRNet `47.741/33.821/29.961`; H18 improves all
three cardinalities for both frontends.  A past-only T=9 clean-trained control
remains a separate pending experiment.

### Table III. Occlusion-Person，absolute MPJPE (mm)

| Method | 2 cams | 3 cams | 4 cams | 8 cams |
|---|---:|---:|---:|---:|
| RANSAC† [42], GBT-reported | 33.7 | 87.5 | 35.0 | 15.5 |
| ScoreFuse† [42], GBT-reported | 32.7 | 25.7 | 21.4 | 15.0 |
| AdaFuse† [42], GBT-reported | — | 26.2 | 19.7 | 12.6 |
| GBT (HRNet), reported | 30.8 | 22.9 | 19.6 | 14.2 |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD |

### Table IV. CMU→H36M 泛化，matched joints，absolute MPJPE (mm)

| Method | Dir | Disc | Eat | Greet | Phone | Pose | Purch | Sit | SitD | Smoke | Photo | Wait | Walk | WalkD | WalkT | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Algebraic Triangulation (HRNet), GBT-reported | 44.6 | 43.5 | 42.2 | 42.5 | 43.7 | 41.6 | 48.5 | 46.8 | 51.7 | 45.4 | 45.7 | 42.3 | 40.3 | 45.6 | 39.8 | 44.3 |
| GBT (HRNet), reported | 33.9 | 32.5 | 32.6 | 33.7 | 38.3 | 30.8 | 35.1 | 56.9 | 85.9 | 36.0 | 37.9 | 33.3 | 29.1 | 38.7 | 28.7 | 38.9 |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Table V. H36M 与 CMU 主结果，absolute MPJPE (mm)

| Method | H36M | CMU |
|---|---:|---:|
| Cross View Fusion [27]†, GBT-reported | 26.2 | — |
| RANSAC Baseline [10], GBT-reported | — | 33.4 |
| Algebraic Triangulation [10]†, GBT-reported | 19.2 | 21.3 |
| Volumetric [10]†, GBT-reported | 17.7 | 13.7 |
| Remelli et al. [29]†, GBT-reported | 30.2 | — |
| Epipolar Transformers [8]†, GBT-reported | 27.1 | — |
| TransFusion [17]†, GBT-reported | 25.8 | — |
| GBT (ResNet-152)†, reported | 22.7 | — |
| Algebraic Triangulation (HRNet), GBT-reported | 44.2 | 26.2 |
| GBT (HRNet), reported | 26.0 | 17.2 |
| `GQ-RUMPL-E2-T` (ResNet-152)† | **19.971** | TBD/— |
| HRNet C2 + E2-C2 + H18 | **27.219** | TBD |

### Table VI-A. GBT 原文组件消融（仅作参考布局）

| Train→Test | none | +Centering | +Synthetic | +Conf. bias | +Geom. bias | all |
|---|---:|---:|---:|---:|---:|---:|
| CMU→CMU | 22.4 | 18.9 | 19.3 | 17.7 | 17.1 | 17.2 |
| H36M→H36M | 39.0 | 49.2 | 40.6 | 33.2 | 33.1 | 26.0 |
| H36M→H36M-Occl | 43.1 | 52.3 | 44.0 | 35.5 | 37.2 | 31.6 |
| CMU→H36M | 101.2 | 57.2 | 55.7 | 43.0 | 43.1 | 38.9 |

### Table VI-B. 我们的累加组件消融

| Train→Test | RUMPL rays | +Tri-anchor/Plücker | +Global Query | +22-cand. E2 | +Identity hinge | +H18 |
|---|---:|---:|---:|---:|---:|---:|
| H36M→H36M, HRNet | TBD | **38.686/30.943/28.629** | not selected | **38.700/29.486/27.274** | — | **37.704/29.231/27.219** |
| H36M→H36M, ResNet | **41.470/26.081/24.157** | included in H76 | **32.312/25.101/23.536** | **32.331/22.646/20.361** | **32.319/22.558/20.272** | **31.215/22.008/19.971** |
| H36M→H36M-Occl, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |
| H36M→H36M-Occl, ResNet | TBD | TBD | TBD | TBD | TBD | TBD |
| CMU→CMU, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |
| CMU→H36M, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |

H36M clean 的每个格子记录 `V2/V3/V4`。正式排版时可按数据集拆表，但不得
删除中间组件，否则无法判断提升来自 generator、candidate fusion 还是 temporal。

### Table VII. 时序帧数，四相机，absolute MPJPE (mm)

| Dataset/method | 1 | 2 | 3 | 6 | 9 |
|---|---:|---:|---:|---:|---:|
| H36M, GBT-reported | 29.4 | 27.9 | 27.7 | 27.3 | 26.0 |
| H36M-Occl, GBT-reported | 41.5 | 34.9 | 34.0 | 32.5 | 31.6 |
| H36M, ours HRNet | **27.274** | TBD | TBD | TBD | **27.219** |
| H36M-Occl, ours HRNet | TBD | TBD | TBD | TBD | TBD |
| H36M, ours ResNet | **20.272** | TBD | TBD | TBD | **19.971** |
| H36M-Occl, ours ResNet | TBD | TBD | TBD | TBD | TBD |

### Supplementary Table S1. 严格 matched 双前端比较

| frontend | `GQ-RUMPL` T=1 V2/V3/V4 | `+E2` T=1 V2/V3/V4 | `+H18` T=9 V2/V3/V4 |
|---|---:|---:|---:|
| HRNet-W32/YOLOX-X | Query not selected; C2 **38.686/30.943/28.629** | **38.700/29.486/27.274** | **37.704/29.231/27.219** |
| ResNet-152† | **32.312/25.101/23.536** | **32.319/22.558/20.272** | **31.215/22.008/19.971** |

### Supplementary Table S2. CMU 视角数量扩展

| Method | 2 cams | 4 cams | 5 cams | 6 cams | 8 cams |
|---|---:|---:|---:|---:|---:|
| Algebraic Triangulation (HRNet) | TBD | TBD | TBD | TBD | TBD |
| `GQ-RUMPL-E2` (HRNet) | TBD | TBD | TBD | TBD | TBD |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD | TBD |

### B.3 第一阶段冻结结论

GBT clean 参考值为 ResNet `29.9/24.4/22.7`、HRNet
`36.8/30.4/26.0`。第一阶段已经完成并冻结，但“完成”不等于全面超过参考值：
ResNet 最终结果为 `31.215/22.008/19.971`，HRNet 最终结果为
`37.704/29.231/27.219`。所有 checkpoint、温度和 epoch 均未依据 S9/S11 选择；
下一阶段遮挡测试不得用于反向挑选 clean 权重。详细事实、差值、负消融和路径见
`/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`。

---

## 历史稿说明（2026-08-15，冲突内容不得作为当前方法）

下面从“历史稿给网页版 Codex 的直接指令”开始的内容记录了此前大量探索、诊断、
失败实验和候选论文故事。它可以用于 Related Work、实验动机和失败经验，但任何
与上面 2026-08-22 冻结路线冲突的模型描述、数值或结论都必须忽略。

## 0. 历史稿给网页版 Codex 的直接指令

请依据本文件撰写一篇完整的多视角 3D 人体姿态估计论文初稿。论文可以先用中文写，也可以直接写成英文 CVPR/ICCV 风格。写作时遵守以下要求：

1. 主问题是：**在不记忆固定相机编号和布局的前提下，如何抑制低质量视角的负贡献，并修复双视角退化相机对造成的深度歧义。**
2. 当前推荐的证据版方法名称为 **Geometry-Anchored Ray Refinement with Counterfactual Hypothesis Utility**，中文为“几何锚定的射线精化与反事实候选效用”。项目旧名 `TGR-Ray` 可以保留为内部名称，但在时序未成为稳定主贡献前，不建议让标题以 Temporal 开头。
3. 论文方法由一个统一流水线组成，而不是三个完整模型的简单堆叠：
   - RUMPL 的相机无关世界射线表示；
   - 置信度加权三角化锚点与锚点中心化 Plücker 射线；
   - 在粗 3D 查询附近读取冻结 HRNet 局部特征并做跨视角 correspondence；
   - 在真实 attention logit 上加入连续几何偏置；
   - 生成可用视角子集的多个 3D 假设；
   - 用逐关节反事实效用和 Set Transformer 做软候选融合；
   - 时序只作为可选第二阶段，不应在当前摘要中宣称是主要性能来源。
4. 论文应把“输入协议”写清楚。当前存在两条不能混报的实验线：
   - **正式坐标级公平线**：YOLOX-X/HRNet-W32 只输出 `(x,y,c)`，3D 网络读取坐标、置信度和相机参数；这是与 GBT/RUMPL 坐标级比较的主线。
   - **历史增强输入/特征级线**：A1D/H21 或 HRNet 中间特征；它证明模块机制有效，但旧 A1D 的 V2/V3 存在先看四视角再删视角的问题，只能放增强输入表或消融表。
5. 对外部论文数字必须标注 detector、是否使用热图/图像特征、是否 T=9、是否 root-aligned、是否使用 GT box。不能把输入和指标不同的方法直接按数值排序后宣称超过。
6. 所有 `待填`、`正在运行` 或 `目标` 必须保留占位符，不得由语言模型猜数值。
7. 论文的可靠贡献应围绕以下完整故事，而不是“加了几个 Transformer”：
   - 诊断并量化 Negative View Problem；
   - 用相机无关的几何锚点稳定绝对位置；
   - 用图像局部证据与连续射线几何改善候选内容；
   - 用反事实效用改善候选选择，降低严重负视角事件；
   - 用随机视角子集训练和全组合评估保持可变视角能力。

---

## 1. 推荐论文身份

### 1.1 推荐标题

当前证据版标题：

> **Geometry-Anchored Ray Refinement with Counterfactual Hypothesis Utility for Robust Multi-view 3D Human Pose Estimation**

中文：

> **面向鲁棒多视角三维人体姿态估计的几何锚定射线精化与反事实候选效用**

若后续未见相机配置实验成立，可改为：

> **Camera-Generalizable Multi-view 3D Human Pose Estimation via Geometry-Anchored Ray Refinement and Counterfactual View Utility**

若最终时序只有约 0.1--0.6 mm 的提升，不应继续使用“Temporal”作为标题第一关键词。旧题目 “Temporal Geometry-Refined Ray Lifting” 可作为项目演化记录，而不是当前最稳妥标题。

### 1.2 一句话核心

> 我们保留 RUMPL 中与相机编号无关的世界射线表示，用解析三角化提供稳定绝对位置，再让网络仅负责读取局部图像证据、校准跨视角关系并评估多个几何假设的逐关节边际效用，从而抑制新增低质量视角对多视角融合的负贡献。

### 1.3 推荐缩写

可选：`GACU`（Geometry-Anchored Counterfactual Utility）或 `GARU`（Geometry-Anchored Ray Utility）。若希望延续已有代码与草稿名称，可在正文中称整体为 `TGR-Ray`，但应把 T 解释为 `Triangulation-guided`，而非在没有强时序结果时解释为 `Temporal`。

### 1.4 当前最可信的论文定位

这篇论文不应声称“首次使用几何偏置”“首次使用 HRNet 特征”“首次使用多假设评分”或“首次使用不确定性三角化”。这些分别已被 GBT、Epipolar Transformer/MVGFormer、Generalizable Human Pose Triangulation、AdaFuse/Learnable Triangulation 等工作覆盖。

当前可守住的组合差异是：

> **在相机无关射线提升主干上，以置信度三角化锚定绝对位置；在粗查询附近做局部跨视角 correspondence，并用连续射线几何修正 attention；最后以“删除/加入某个视角对每个关节 3D 风险的真实边际影响”为监督，学习集合化的候选效用并进行软融合。**

这三个部分共同针对同一个可量化问题：`Negative View Problem`，即增加视角后总体均值可能下降，但大量帧/关节反而变差。

---

## 2. 研究问题、观察和论文故事线

### 2.1 研究问题

给定同步的多视角图像、冻结二维姿态检测器和已标定相机，希望预测绝对世界坐标中的 3D 人体姿态。模型应满足：

- 输入视角数量可在 2/3/4 或更广范围变化；
- 视角顺序置换不改变输出；
- 不使用相机 ID lookup，不记忆固定输入槽位；
- 在正常相机组合上利用解析几何，在退化相机组合、误检和遮挡下利用人体/图像先验；
- 新增视角平均应有帮助，并尽量降低新增视角造成局部恶化的概率；
- 主指标为 absolute MPJPE，不通过 root alignment 隐藏绝对位置误差。

### 2.2 RUMPL 的优点与缺陷

RUMPL 的优点：

1. 将二维点和相机标定转换成世界射线，而不是把相机编号作为 embedding；
2. 参数在视角间共享，天然支持不同数量和顺序的视角；
3. View Fusion Transformer（VFT）先融合同一关节的不同视角，再由 Pose Fusion Transformer（PFT）建模 17 个关节；
4. 随机相机环境/MHP 合成训练可迁移到不同相机布局；
5. 在其论文的 AMASS/MHP→H36M 协议上，2-view All-KP MPJPE 为 52.5 mm，公开代码近似复现也已达到相近量级。

RUMPL 的关键缺陷：

1. 路径是 `同一关节跨视角 VFT → 每关节一个 token → PFT → 3D`；视角在网络看到完整人体或历史之前就被压缩。
2. 当两个视角都对某个腕、踝、头部关节不可靠时，VFT 没有第三条射线冗余，也没有足够全局证据修复深度。
3. 单纯延长训练可改善，但不能自动解决视角数量分布迁移；只训练 K=2 会使 V3/V4 崩溃。
4. 把通用 graph/global/temporal 模块随意插入 VFT 前后通常破坏 ray identity，已有大量严格负结果。

### 2.3 最关键的实证观察：误差不是均匀分布

在当前正式 GBT-style YOLOX-X+HRNet 坐标输入上，H76 的 V2 六个相机对为：

| 相机对 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H76, mm | 34.731 | 39.046 | **70.439** | **67.945** | 32.346 | 32.855 | **46.227** |

正常四对平均为 34.744 mm，退化两对 `1-4`、`2-3` 平均为 69.192 mm。这两个退化组合把六对总体均值拉高约 11.48 mm。因此问题不是“所有两视角都差”，而是**少数相机几何与二维误差共同形成的灾难性负组合**。

V3 四个组合为 33.098、30.642、31.979、29.615 mm，V4 为 27.964 mm。加入第三/第四视角可在平均意义上缓解退化，但逐帧逐关节仍存在 Negative View：

| 转移 | before | after | 逐关节 NVR | 恶化 >1 mm | 恶化 >5 mm | 整姿态 NVR |
|---|---:|---:|---:|---:|---:|---:|
| V2→V3（历史 H76 同协议诊断） | 34.816 | 30.489 | 42.73% | 36.12% | 16.08% | 28.61% |
| V3→V4 | 30.489 | 29.691 | 44.17% | 34.48% | 10.25% | 31.81% |

这说明只报告平均 MPJPE 会掩盖大量局部负贡献。论文应将 NVR 与 Monotonic Violation 作为补充指标。

### 2.4 由观察导出的完整故事

1. **几何锚定**：用置信度加权射线最小二乘给出粗 3D 绝对位置，避免网络从零学习深度和平移。
2. **候选内容修正**：在粗 3D 查询投影位置读取冻结 HRNet 局部特征；跨视角 patch correspondence 为误检/遮挡提供图像证据；连续射线几何直接进入 view attention logit。
3. **候选集合构造**：对当前可用相机集合的 2/3/4 视角子集生成多个 3D 假设，使模型能做反事实比较，例如“删掉第 4 个视角后此关节是否更好”。
4. **反事实效用学习**：不只预测 detector confidence，而是监督每个候选相对于当前全视角基线的真实 3D 边际误差；Set Transformer 在候选集合内比较共识、射线残差和几何条件。
5. **软融合而非硬选择**：逐关节 softmax 融合保留多个互补候选，已有实验显示 hard top-1、sparsemax、top-k、medoid/median 均不如软融合稳定。
6. **可选时序**：只在当前单帧方法稳定后，用轻量 fixed-lag residual 对候选/3D 做第二阶段修正；当前时序主要帮助 V2，不能解释全部性能。

---

## 3. 问题定义与符号

给定长度为 (T) 的同步序列、最多 (V) 个相机和 (J=17) 个关节。二维检测器输出：

\[
\mathcal X=\{(\mathbf x_{t,v,j},c_{t,v,j})\},\qquad
\mathbf x_{t,v,j}=(u,v,1)^\top,\quad c_{t,v,j}\in[0,1].
\]

相机内参、世界到相机旋转和平移分别为 \(\mathbf K_v,\mathbf R_v,\mathbf t_v\)。相机中心和世界单位射线方向为：

\[
\mathbf o_v=-\mathbf R_v^\top\mathbf t_v,
\qquad
\mathbf d_{t,v,j}=\operatorname{normalize}
\left(\mathbf R_v^\top\mathbf K_v^{-1}\mathbf x_{t,v,j}\right).
\]

若图像/二维坐标已经去畸变，则相机模型必须同步使用零畸变参数；若保留原始畸变坐标，则射线构造必须显式调用畸变模型。不能混用。

对应世界射线：

\[
\ell_{t,v,j}(s)=\mathbf o_v+s\mathbf d_{t,v,j},\qquad s\ge 0.
\]

模型输出绝对世界坐标姿态：

\[
\hat{\mathbf Q}_t=[\hat{\mathbf q}_{t,1},\ldots,\hat{\mathbf q}_{t,J}]\in\mathbb R^{J\times 3}.
\]

---

## 4. 方法总览

```text
同步多视角 RGB
  │
  ├─ 冻结 person detector + HRNet ──> 2D (x,y), confidence c
  │                                      │
  │                                      └─ 可选 HRNet stage-4 局部特征
  │
  └─ 相机内外参 ──────────────────────> 世界射线 (o,d)
                                             │
                           置信度加权三角化锚点 q_tri
                                             │
                    锚点中心化 Plücker ray token + RUMPL VFT/PFT
                                             │
                                      粗 3D 候选 q_base
                                             │
            q_base 投影到各视角 HRNet 特征图，采样 5×5 patch
                                             │
        patch correspondence + geometry-biased view attention + joint context
                                             │
                                  精化候选 q_c = q_base + Δq
                                             │
      对当前相机集合的所有合法子集生成候选；加入置信 DLT/其他候选
                                             │
        逐关节 Set Transformer 预测反事实候选风险 Δe_{c,j}
                                             │
                       soft candidate fusion 得到最终 3D pose
                                             │
                      可选 T=9 fixed-lag temporal residual
```

该结构中 RUMPL 不是被完全丢弃：保留世界射线、跨视角共享参数、VFT/PFT 基础表示和相机泛化接口；改进集中在解析锚点、局部图像证据和候选效用。最终稿若改成完整新主干，应重新画图并明确哪些 RUMPL 层被替换。

---

## 5. 模块一：置信度加权三角化锚点

### 5.1 解析锚点

对关节 \(j\)，令到射线法平面的投影矩阵：

\[
\mathbf P_{v,j}=\mathbf I-\mathbf d_{v,j}\mathbf d_{v,j}^{\top}.
\]

使用检测置信度构造正权重：

\[
w_{v,j}=\operatorname{clip}(c_{v,j},0,1)+\epsilon_c.
\]

三角化锚点为：

\[
\mathbf q^{\text{tri}}_j
=\arg\min_{\mathbf q}
\sum_{v\in\mathcal V}w_{v,j}
\left\|\mathbf P_{v,j}(\mathbf q-\mathbf o_v)\right\|_2^2
+\lambda\|\mathbf q\|_2^2.
\]

闭式线性系统：

\[
\mathbf A_j=\sum_v w_{v,j}\mathbf P_{v,j}+\lambda\mathbf I,
\qquad
\mathbf b_j=\sum_vw_{v,j}\mathbf P_{v,j}\mathbf o_v,
\qquad
\mathbf q_j^{\text{tri}}=\mathbf A_j^{-1}\mathbf b_j.
\]

实现中用 `torch.linalg.solve`，不显式求逆。当前默认 \(\epsilon_c=0.05\)，\(\lambda=10^{-4}\)。

### 5.2 为什么锚点有效

- 为网络提供粗略但稳定的绝对三维位置；
- 将学习问题从“从射线自由回归 3D”改为“预测锚点残差”；
- 保留解析几何对新相机参数的适配；
- 在 V3/V4 有冗余射线时，置信度可抑制异常视角；
- 但 V2 没有冗余，单纯重加权无法修复两个共同退化的相机，因此还需要人体/图像先验和候选比较。

### 5.3 代码

- 核心实现：`OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`，约 1557--1595 行。
- 环境开关：`RUMPL_TRI_ANCHOR=1`、`RUMPL_TRI_ANCHOR_REG=1e-4`、`RUMPL_TRI_ANCHOR_CONF_EPS=0.05`。
- 输出残差：约 2359--2363 行，形式为 `prediction = learned_residual + gate * tri_anchor`。

---

## 6. 模块二：锚点中心化 Plücker 射线表示

一条 3D 直线可用 Plücker 坐标表示：

\[
\mathbf L=(\mathbf d,\mathbf m),\qquad \mathbf m=\mathbf o\times\mathbf d.
\]

Plücker moment 对同一直线上的点选择不变，因为对任意 \(\mathbf p=\mathbf o+s\mathbf d\)：

\[
\mathbf p\times\mathbf d=(\mathbf o+s\mathbf d)\times\mathbf d
=\mathbf o\times\mathbf d.
\]

为减少场景绝对原点变化，先以三角化锚点构造局部坐标，再计算 moment。可写为：

\[
\tilde{\mathbf o}_{v,j}=\mathbf o_v-\mathbf c,
\qquad
\tilde{\mathbf m}_{v,j}=\tilde{\mathbf o}_{v,j}\times\mathbf d_{v,j},
\]

其中 \(\mathbf c\) 可为人体 root anchor 或关节 anchor。当前 H76 使用 subject-root frame；实现也支持 per-joint anchor，但不是当前正式结果默认。

射线 token：

\[
\mathbf z_{v,j}^{(0)}=phi([\mathbf d_{v,j},\tilde{\mathbf m}_{v,j},c_{v,j}])
+\mathbf e_j,
\]

其中 \(\phi\) 是共享 MLP/位置编码，\(\mathbf e_j\) 是 joint embedding，不包含 camera ID。

代码：

- `multiview_rumpl.py` 约 1600--1616 行。
- 开关：`RUMPL_ANCHOR_CENTERED_RAYS=1`、`RUMPL_INPUT_PLUCKER=1`。

---

## 7. 模块三：RUMPL 粗射线提升主干

RUMPL 的分解融合保留为稳定起点：

1. VFT：对固定关节 \(j\)，在视角轴上融合 \(\{\mathbf z_{v,j}\}\)；
2. PFT：对 17 个已经视角融合的 joint token 建模人体结构；
3. 共享 3D head 输出相对三角化锚点的残差。

可简写：

\[
\mathbf h_j=\operatorname{VFT}(\{\mathbf z_{v,j}\}_{v\in\mathcal V}),
\qquad
\mathbf H=\operatorname{PFT}([\mathbf h_1,\ldots,\mathbf h_J]),
\]

\[
\mathbf q_j^{\text{base}}=\mathbf q_j^{\text{tri}}+g_j\,f_{3D}(\mathbf H_j).
\]

该主干在正常相机对上已达到约 32--39 mm，但对 `1-4`/`2-3` 退化。后续模块不是重复另跑一个完整 pose estimator，而是用同一粗姿态查询局部证据并修正候选。

---

## 8. 模块四：粗查询引导的 HRNet 局部特征采样

### 8.1 动机

只使用 HRNet argmax 坐标会丢失热图形状、局部纹理和遮挡证据。Learnable Triangulation、Epipolar Transformer、AdaFuse、MVGFormer 的共同启示是：在 2D 坐标之外保留图像/热图证据，可显著改善跨视角融合。

本项目不构造大体素，也不使用相机 ID，而是在粗 3D 查询投影附近采样紧凑局部特征：

\[
\hat{\mathbf u}_{v,j}=\pi_v(\mathbf q_j^{\text{base}}),
\]

\[
\mathbf F_{v,j}=\operatorname{GridSample}
\left(\mathbf H_v^{\text{stage4}},\mathcal N_{5\times5}(\hat{\mathbf u}_{v,j})\right),
\]

其中当前冻结 HRNet stage-4 高分辨率分支实际为 `[32,96,72]`，每个关节/视角得到 `[32,5,5]` patch。

### 8.2 防止视角泄漏

每个 V2/V3/V4 子集必须使用**该子集自己的 H76 预测**产生投影点。不能先用四视角预测采样特征，再在 3D 网络中删除视角。正式缓存形状按 11 个视角组合分别保存，例如 `[G,11,4,17,32,5,5]`。

### 8.3 代码

- 特征导出：`OpenRUMPL_baseline_audit/export_h36m_hrnet_features.py`。
- 采样 token：`OpenRUMPL_baseline_audit/prepare_rigr_feature_tokens_20260812.py`。
- 精化模型：`OpenRUMPL_baseline_audit/train_rigr_hrnet_feature_20260812.py`。

---

## 9. 模块五：跨视角 patch correspondence

### 9.1 结构

对每个当前视角，粗投影中心的 feature 作为 query，所有有效视角的 5×5 patch token 作为 key/value：

\[
\mathbf q_{v,j}=f_q(\mathbf F_{v,j}^{\text{center}}),
\]

\[
\mathcal K_j=\mathcal V_j=
\{f_k(\mathbf F_{u,j,p})+\mathbf e_p:
u\in\mathcal V,\ p\in\{1,\ldots,25\}\},
\]

\[
\mathbf a_{v,j}=\operatorname{MHA}(\mathbf q_{v,j},\mathcal K_j,\mathcal V_j).
\]

未选视角在 key/value 侧 mask。使用零初始化门控保持训练初始与原 feature stream 完全一致：

\[
\tilde{\mathbf f}_{v,j}=\mathbf f_{v,j}
+\tanh(g_{\text{corr}})
\left(\operatorname{LN}(\mathbf f_{v,j}+\mathbf a_{v,j})-\mathbf f_{v,j}\right),
\quad g_{\text{corr}}(0)=0.
\]

### 9.2 来源与边界

- Epipolar Transformer：跨视角图像特征对应；
- MVGFormer：3D query 投影并读取局部图像特征，几何与特征迭代；
- 本方法只是针对稀疏 HRNet patch 和 RUMPL 查询的轻量适配，不应声称逐行复现上述官方模型。

### 9.3 代码

`train_rigr_hrnet_feature_20260812.py` 约 218--347 行，类 `RIGRHRNetFeature`。

---

## 10. 模块六：真实 attention logit 上的连续几何偏置

### 10.1 几何特征

对同一关节的视角对 \((v,u)\)，定义：

\[
s_{vu}=\|\mathbf d_v\times\mathbf d_u\|_2,
\qquad
r_{vu}=|\mathbf d_v^\top\mathbf d_u|,
\]

\[
D_{vu}=
\frac{|(\mathbf o_u-\mathbf o_v)^\top(\mathbf d_v\times\mathbf d_u)|}
{\|\mathbf d_v\times\mathbf d_u\|_2+\epsilon}.
\]

注意：旧草稿中把分母写成叉积范数平方且没有绝对值，不应继续使用。近乎平行时需用 point-to-line 稳定分支。距离再按当前样本有效视角对的均值归一化并裁剪到 `[0,10]`。

几何偏置输入：

\[
\mathbf g_{vu,j}=
[r_{vu,j},s_{vu,j},\bar D_{vu,j},c_{v,j},c_{u,j}].
\]

每个 attention head 学习一个 bias：

\[
b_{vu,j}^{(h)}=\operatorname{MLP}^{(h)}(\mathbf g_{vu,j}),
\]

\[
\boxed{
a_{vu,j}^{(h)}=
\frac{(\mathbf W_q^{(h)}\mathbf z_{v,j})^\top
(\mathbf W_k^{(h)}\mathbf z_{u,j})}{\sqrt{d_h}}
+b_{vu,j}^{(h)}
}
\]

为避免歧义，LaTeX 实际应写：

```latex
a_{vu,j}^{(h)}=
\frac{(W_q^{(h)}z_{v,j})^\top (W_k^{(h)}z_{u,j})}{\sqrt{d_h}}
+b_{vu,j}^{(h)}.
```

偏置 MLP 最后一层零初始化，因此 epoch 0 是严格 no-bias control。偏置只作用于同一关节的跨视角关系，不把同相机不同关节的共中心射线错误当成强对应。

### 10.2 为什么以前偏置失败、这里才有依据

以前把 ray-distance bias 直接加到 RUMPL fusion token 或所有 joint-view token：

- fusion token 没有真实射线，零距离行不对应论文几何；
- 同一相机不同关节射线共享中心，线距离可能为零，却不是同一 3D 关节；
- VFT 已经或即将压缩视角，bias 无法与局部图像对应协同。

当前偏置位于显式 view attention logit，并限制为同一 joint、跨 view，且与 HRNet correspondence 联合。因此这是“贴近论文作用位置”的版本。

### 10.3 代码

`train_rigr_hrnet_feature_20260812.py` 约 259--269、376--414 行。

---

## 11. 模块七：三维残差精化

经过 view transformer 和 joint transformer 后，预测受限 3D residual：

\[
\Delta\mathbf q_j=\delta_{\max}\tanh(f_{\Delta}(\mathbf h_j)),
\qquad \delta_{\max}=0.25\ \text{m},
\]

\[
\mathbf q_j^{\text{rigr}}=\mathbf q_j^{\text{base}}+\Delta\mathbf q_j.
\]

输出线性层零初始化，确保 epoch 0 的输出逐元素等于 H76 候选。可选 trust gate：

\[
\Delta\mathbf q_j\leftarrow
\sigma(f_g(\mathbf h_j))\Delta\mathbf q_j,
\]

但已有两 seed 结果显示 gate 没有超过无 gate，不作为默认主模型。

代码：`train_rigr_hrnet_feature_20260812.py` 约 430--435 行。

---

## 12. 模块八：视角子集多假设集合

四相机时，所有 2/3/4 视角组合为：

\[
\mathcal C=
\binom{\{1,2,3,4\}}{2}
\cup
\binom{\{1,2,3,4\}}{3}
\cup
\binom{\{1,2,3,4\}}{4},
\]

候选总数为 \(6+4+1=11\)。可扩展候选包括：

- H76/RIGR 对每个子集的 learned pose；
- confidence-weighted ray/DLT；
- pairwise hypotheses；
- learned triangulation 候选；
- IRLS 仅在 oracle 证明有用时加入。当前 GBT-style 新输入诊断显示 IRLS 明显差，应排除。

给定当前任务相机集合 \(\mathcal V_t\)，只允许使用满足 \(C_k\subseteq\mathcal V_t\) 的候选：

\[
\mathcal H(\mathcal V_t)=
\{\mathbf Q_k\mid C_k\subseteq\mathcal V_t\}.
\]

这是防止“V3 任务偷看第 4 个相机”的核心 mask。

Generalizable Human Pose Triangulation（CVPR 2022）为“从多个几何假设中学习评分”提供权威依据；本方法的差异是逐关节反事实效用、候选间 Set Transformer 和 Negative View 目标。

---

## 13. 模块九：反事实逐关节候选效用

### 13.1 监督目标

对任务 \(\mathcal V_t\)，令直接使用该任务全部视角的候选为 baseline \(b\)。候选 \(k\) 对关节 \(j\) 的真实误差：

\[
e_{k,j}=\|\mathbf q_{k,j}-\mathbf q_j^{gt}\|_2.
\]

定义相对 baseline 的反事实边际误差：

\[
\Delta e_{k,j}=e_{k,j}-e_{b,j}.
\]

若 \(\Delta e_{k,j}<0\)，说明删掉某些视角或使用另一求解器可改善该关节；若大于 0，则候选有害。网络预测 \(\widehat{\Delta e}_{k,j}\)，而不是直接照抄 2D confidence。

为消除候选分数整体平移不确定性，预测也相对 baseline：

\[
\widehat{\Delta e}_{k,j}
=s_{k,j}-s_{b,j}.
\]

### 13.2 候选特征

每个候选/关节 token 使用：

1. root-relative 归一化候选 pose；
2. candidate root/绝对位置；
3. 候选相对集合共识的位移和范数；
4. 候选到包含射线/排除射线的点线残差统计（mean/min/max）；
5. 包含/排除视角的 confidence 统计；
6. 当前候选视角占任务视角的比例；
7. 加权射线 normal matrix 的特征值谱，反映几何条件；
8. joint embedding；
9. pose-level context。

候选到射线距离：

\[
r_{k,j,v}=\|(mathbf q_{k,j}-\mathbf o_v)
\times\mathbf d_{v,j}\|_2.
\]

数值特征使用：

\[
\tilde r=\log(1+r/0.005).
\]

射线 normal matrix：

\[
\mathbf N_{k,j}=\sum_{v\in C_k}(c_{v,j}+0.05)
(\mathbf I-\mathbf d_{v,j}\mathbf d_{v,j}^{\top}).
\]

其归一化特征值对数谱用于表征可三角化条件。

### 13.3 Set Transformer 候选交互

把候选 token 编码到 64 维：

\[
\mathbf h_{k,j}^{(0)}=f_{enc}(\mathbf f_{k,j}).
\]

对每个关节，在候选轴上进行 2 层 self-attention：

\[
[\mathbf h_{1,j},\ldots,\mathbf h_{K,j}]
=\operatorname{SetEnc}([\mathbf h_{1,j}^{(0)},\ldots,\mathbf h_{K,j}^{(0)}]).
\]

不使用 candidate index 或 camera ID embedding，因此对候选排列等变。深度消融显示 2 层最佳；3/4 层没有稳定提升。

### 13.4 软融合

以预测风险的负值构造权重：

\[
\alpha_{k,j}=
\frac{\exp(-\widehat{\Delta e}_{k,j}/\tau)}
{\sum_{l\in\mathcal H(\mathcal V_t)}
\exp(-\widehat{\Delta e}_{l,j}/\tau)}.
\]

最终逐关节融合：

\[
\hat{\mathbf q}_j=\sum_k\alpha_{k,j}\mathbf q_{k,j}.
\]

历史 E2 最佳推理温度为 \(\tau=1.8\)。hard top-1、sparsemax、top-3/top-5、weighted medoid 和 geometric median 均不稳定或退化，因此当前默认 softmax soft fusion。

### 13.5 损失

直接反事实监督可由回归和排序组成：

\[
\mathcal L_{delta}
=\operatorname{SmoothL1}(\widehat{\Delta e},\Delta e),
\]

\[
\mathcal L_{rank}
=\sum_{k,l}\log\left(1+exp[-y_{kl}(s_l-s_k)]\right),
\quad
y_{kl}=\operatorname{sign}(e_l-e_k).
\]

GHT-style expected risk：

\[
\mathcal L_{risk}=
\frac{1}{BJ}\sum_{b,j,k}\alpha_{b,k,j}e_{b,k,j},
\]

融合姿态监督：

\[
\mathcal L_{fuse}=
\frac{1}{BJ}\sum_{b,j}
\|\hat{\mathbf q}_{b,j}-\mathbf q_{b,j}^{gt}\|_2.
\]

当前训练代码中 GHT 阶段使用：

\[
\mathcal L=\mathcal L_{direct}
+\frac{\mathcal L_{risk}+0.05\mathcal L_{fuse}}{0.01}.
\]

训练分两阶段：10 epochs direct delta/rank，再 5 epochs GHT expected-risk。继续到 30 epochs 只带来约 0.01 mm，说明旧 E2 已收敛。

### 13.6 代码

- Set Transformer：`OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py`，类 `SetTransformerJointUtility`。
- 通用 V234 训练：`train_e2_v234_universal_20260812.py`。
- 候选缓存：`export_h76_train_subset_hypotheses_20260811.py`、`build_e2_v234_candidate_cache_20260812.py`。
- 当前 GBT-style oracle：`build_current_input_e2_oracle_20260815.py`。

---

## 14. 可选模块：fixed-lag 时序残差

当前数据由原视频每 5 帧采样一帧，仍保留 subject/action/subaction/image_id，可构造连续窗口。T=9 时约覆盖 0.8 秒（以约 10 Hz 缓存计），不可跨 action/subaction/缺帧边界。

推荐时序不是在最终 3D 上盲目平滑，而是读取：

\[
[\hat{\mathbf Q}_{t-k:t},\ \text{candidate utility},\
\text{view confidence},\ \text{geometry spectrum}]
\]

预测 root-protected residual：

\[
\hat{\mathbf Q}^{temp}_t
=\hat{\mathbf Q}^{frame}_t
+g_t\odot\Delta\mathbf Q_t.
\]

当前严格结果表明：

- 联合 V2/V3/V4 fixed-lag 相对同窗口中心帧：V2 改善 1.758 mm，V3 0.207 mm，V4 0.030 mm；
- V3/V4 专门化 fixed-lag 三种子：V3 29.469±0.055，V4 28.796±0.035，改善约 0.59/0.49 mm；
- MixSTE 式 factorized 模块训练内有效但跨 S9/S11 过拟合；
- pairwise E2 后时序在严格对齐协议下只改善约 0.16/0.18 mm。

因此时序可作为补充消融或 V2 辅助，不应在当前摘要中声称带来 GBT 论文同等的 3.4 mm。GBT 的时序发生在 global joint-view-time encoder 之前，旧实验多为候选/3D 后处理，信息通路不同。

---

## 15. 指标：不只看平均 MPJPE

### 15.1 Absolute MPJPE

\[
\operatorname{MPJPE}=
\frac1{NJ}\sum_{n=1}^{N}\sum_{j=1}^{J}
\|\hat{\mathbf q}_{n,j}-\mathbf q_{n,j}^{gt}\|_2.
\]

主表必须说明：All-17、absolute world coordinates、无刚体/尺度/root 对齐。

### 15.2 Action-equal 与 frame-weighted

H36M 动作帧数不均。主结果当前采用 action-equal：先对每个 action 求均值，再平均动作。另报告 frame-weighted 作为补充，不能混报。

### 15.3 Negative View Rate（NVR）

对嵌套视角集合 \(S\subset S'\)：

\[
\operatorname{NVR}(S\to S')=
\frac1{NJ}\sum_{n,j}
\mathbb 1[e_{n,j}(S')>e_{n,j}(S)].
\]

同时报告恶化超过 1 mm、5 mm 的比例，以及整姿态 NVR。

### 15.4 Monotonic Violation

\[
\operatorname{MV}_{2\to3}=\Pr[operatorname{MPJPE}(S_3)>
\operatorname{MPJPE}(S_2)],
\]

按所有嵌套组合平均。论文故事应强调减少严重负视角，而不是保证每个微小误差都严格单调。

### 15.5 其他必须保留的指标

- 六个 V2 相机对和四个 V3 组合；
- root translation error；
- root-relative MPJPE；
- 每关节、每动作结果；
- 参数量、FLOPs、提升器推理时间；
- 若加入时序：MPJVE、加速度误差、静态抖动；
- 若加入不确定性：NLL、coverage、误差-不确定性相关性、ECE。

---

## 16. Human3.6M 数据和正式评估协议

### 16.1 数据划分

- 训练主体：S1/S5/S6/S7/S8；
- 测试主体：S9/S11；
- 训练 PKL：312,188 个相机记录，78,047 个四相机同步时刻；
- 验证/测试：8,084 条记录，2,021 个四相机同步时刻；
- 删除公认损坏序列；
- 四个相机，V2 枚举 6 组，V3 枚举 4 组，V4 枚举 1 组；
- 视角组合必须全部平均，不能只选好相机对。

### 16.2 正式坐标级前端

当前 GBT-style 主线：

- person detector：YOLOX-X，score threshold 0.01（未检测时有记录的 fallback）；
- pose detector：HRNet-W32 COCO；
- 输入：坐标、置信度、相机参数，不使用热图/图像特征；
- 图像/坐标去畸变与内参匹配；
- 当前缓存类型：`gbt_yolox_x_score001_fallback_legswap`；
- 训练与验证共享同一前端定义；
- 这只是按公开细节构造的 `GBT-style` 输入，因 GBT 未公开 YOLOX 型号、阈值、checkpoint 等，不能写成严格 GBT 前端复现。

### 16.3 历史增强输入协议

旧 H76/RIGR/E2 使用 `mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap`、annotation box 和增强二维/HRNet 特征。旧 A1D 导出时先读取四视角热图生成修正点，再在 RUMPL 端删成 V2/V3，因此：

- 旧 V2/V3 结果不能进入严格外部公平主表；
- V4 没有未选视角泄漏，但属于 enhanced heatmap/input；
- 这些结果可用于模块内部配对消融和机制分析；
- 正式版本必须按 11 个实际相机子集分别生成前端/候选缓存。

### 16.4 两种输入层级必须分表

| 层级 | 3D 模型可见内容 | 可比较方法 |
|---|---|---|
| 坐标级 C | `(x,y,c)` + camera | RUMPL、GBT、DLT/RANSAC、GHT/E2 |
| 热图级 H | 完整 joint heatmap | AdaFuse、CVF、A1D |
| 特征级 F | HRNet/ResNet 中间特征 | Epipolar Transformer、RIGR、部分 MVGFormer |
| 图像/体积级 V | RGB/体素特征 | LT Volumetric、MVGFormer |

不能仅因为都写 HRNet 就认为输入公平。

---

## 17. 当前真实结果总表

### 17.1 正式 GBT-style 坐标级主线（可进入当前论文主实验草稿）

指标均为 H36M S9/S11、absolute、All-17、action-equal、全组合平均，单位 mm。

| 实验 | 主干/训练协议 | V2 | V3 | V4 | 状态与解释 |
|---|---|---:|---:|---:|---|
| R0 | 原始 RUMPL，20E | 75.250 | 56.737 | 48.539 | 同一 GBT-style 前端 |
| H76 | tri-anchor + anchor-centered rays + Plücker，20E | 46.227 | 31.334 | 27.964 | 当前短预算平衡基线 |
| B1 | H76，299,874 updates；前8E K=2，后3:1:1 | **43.456** | **30.732** | **27.818** | 延长预算，三视角/四视角保持 |
| B2 | H76，299,874 updates；全程随机 K=2 | **37.886** | 62.215 | 46.217 | V2 接近目标，但发生 cardinality collapse |
| G0 | H76 + plain global-JV，20E K=2 | 78.336 | 47.536 | 43.232 | 严格负结果 |
| G1 | G0 + global bias | 78.192 | 47.383 | 43.050 | bias 相对 G0 仅约0.15 mm，整体失败 |
| C1 | 从 B2 微调，K2/K3/K4=3:1:1，20E | **37.007** | 32.103 | 30.930 | 已完成；保留 V2 优势，但仍有 cardinality 专门化 |
| C2 | 从 B1 微调，K2/K3/K4=8:1:1，20E | 38.686 | **30.943** | **28.629** | 已完成；当前最好的统一折中模型 |

主要结论：

- H76 的解析锚点/局部表示相对 R0 提升巨大；
- 300k 预算对 B1 只有中等收益，不足以解释 GBT 全部优势；
- 全程 K=2 能把 V2 从 46.23 降到 37.89，但模型无法泛化到未训练视角数量；
- “更多训练”与“视角基数分布”必须分开；
- 新论文需要 cardinality-balanced 或 utility-based 统一模型，而不是分别报告互不兼容的 specialist。

### 17.2 H76 V2 相机对诊断

| 模型 | 1-2 | 1-3 | 1-4 | 2-3 | 2-4 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|
| H76 20E | 34.731 | 39.046 | 70.439 | 67.945 | 32.346 | 32.855 |
| B1 300k | 33.800 | 40.108 | 58.268 | 62.808 | 32.295 | 33.459 |
| B2 K2-only 300k | 33.855 | 38.112 | 47.825 | 45.930 | 29.701 | 31.895 |

B2 的主要收益正是修复 `1-4`、`2-3`，不是只在简单相机对上过拟合。问题在于未学习 K=3/K=4。

### 17.3 旧协议 GBT-style E2 候选 oracle（已作废，仅保留审计痕迹）

使用 B1 冻结候选、当前 GBT-style 输入：

| 候选池（旧 `FLIP_LOWER_BODY_KP_TEST=true`） | V2 oracle | V3 oracle | V4 oracle |
|---|---:|---:|---:|
| 原始 11 个 H76 子集候选 | 51.127 | 28.946 | 22.964 |
| + confidence-weighted triangulation | **45.657** | **25.237** | **19.672** |
| IRLS 相关池 | 90.226 | 45.047 | 36.993 |
| 全部 33 候选 oracle | 45.529 | 25.127 | 19.569 |

解释：这一整张表来自下肢关节语义未对齐的旧缓存，已被第 32 节协议审计判为无效，不能作为候选互补性证据，也不能用于论文主表。保留它只为解释为什么旧 E2 试跑会出现虚假的 oracle 增益；正确协议结果见第 32.3 节。

### 17.4 历史增强输入/特征级结果（机制证据，不能与 17.1 混表）

| 方法 | V2 | V3 | V4 | 说明 |
|---|---:|---:|---:|---|
| H76 enhanced baseline | 34.8163 | 30.4890 | 29.6913 | 旧 A1D/H21；V2/V3 有四视角预修正泄漏 |
| RIGR feature + geometry bias, seed0 | 32.880 | 29.269 | 28.387 | HRNet 中间特征级 |
| RIGR→E2 specialist，两 seed均值 | — | **28.5553** | **27.8078** | V3/V4 专用模型 |
| correspondence no-bias + V234 E2 | 33.3333±0.0201 | 29.1050±0.0249 | 28.4034±0.0273 | 同一增强输入 |
| correspondence+bias + V234 E2 | **32.7701±0.0096** | **28.6679±0.0083** | **27.9273±0.0042** | 统一 V234 模型 |
| correspondence no-bias→E2 specialist | — | 28.8783 | 28.1465 | 两 seed均值 |
| correspondence+bias→E2 specialist | — | **28.4148** | **27.6475** | bias 配对增益 0.4635/0.4990 |

这一表支持：

- HRNet 图像证据与 E2 候选效用具有互补性；
- 在相同 correspondence/E2 结构和配对种子下，geometry bias 带来约 0.44--0.56 mm 的一致收益；
- 统一 V234 模型优于 no-bias，但 specialist 的 V3/V4 更低；
- 这些数值不能对外声称超过 GBT HRNet，因为信息层级、框、时序和旧缓存协议不同。

### 17.5 反事实效用/NVR 的历史严格机制结果

| 方法 | V3 | V4 | 说明 |
|---|---:|---:|---|
| H76 | 30.4890 | 29.6913 | baseline |
| C4 delta + GHT，3 seeds | 29.9653±0.0079 | 29.2816±0.0070 | 反事实效用 |
| E2 Set Transformer depth2，3 seeds | **29.8217±0.0019** | **29.0628±0.0095** | 候选交互进一步提升 |
| E2 续训到 30E | 29.8108 | 29.0380 | 仅约0.01 mm额外收益 |

NVR：

| 方法 | 转移 | NVR | >1 mm | >5 mm | Pose NVR | root-relative NVR |
|---|---|---:|---:|---:|---:|---:|
| H76 | V2→V3 | 42.73% | 36.12% | 16.08% | 28.61% | 40.76% |
| E2 | V2→V3 | **38.63%** | **30.83%** | **10.73%** | **18.66%** | **36.75%** |
| H76 | V3→V4 | 44.17% | 34.48% | 10.25% | 31.81% | 43.50% |
| E2 | V3→V4 | 44.38% | **32.04%** | **6.48%** | **29.72%** | **41.98%** |

E2 不一定减少所有“微小正误差”事件，但显著减少超过 1/5 mm 的严重负贡献和整姿态 NVR。这正是论文应强调的机制结果。

### 17.6 H8/H9 时序筛选（当前实验记录）

H8 是在当前 GBT-aligned HRNet 坐标级输入上进行的受控 T=9 筛选，不与旧 A1D、热图或
特征级实验混表。验证为 S9/S11 全量、All-17 absolute、action-equal、V2/V3/V4 全
组合；单位 mm。

| 分支 | T=1 V2 | T=1 V3 | T=1 V4 | T=9 V2 | T=9 V3 | T=9 V4 |
|---|---:|---:|---:|---:|---:|---:|
| H8 frozen Pre-VFT temporal | 41.293 | 34.505 | 29.836 | 41.397 | 34.459 | 29.978 |
| H8 joint VFT/PFT/head | 55.887 | 57.888 | 46.935 | 56.071 | 58.040 | 47.124 |

H8 的冻结分支在 T=9 相对 T=1 的变化为 `+0.104/-0.046/+0.142 mm`，不能声称 clean
提升；联合分支三列均明显退化，作为“解冻 RUMPL 融合器不稳定”的失败消融。完整
JSON 在 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_pre_vft_temporal/eval/`。

H9 已完成：采用 MixSTE 的 spatial-temporal block，但只对 RUMPL 已解码的
root-relative articulation 预测 residual，并将 pelvis residual 置零；主干、射线
anchor 和 absolute 定位路径冻结。H9 使用同一 GBT-aligned HRNet 输入和 T=9 训练协议，
不是旧 A1D MixSTE 结果。它的正式结果见下表；两条 MixSTE 路线均为负对照。

截至 2026-08-17，H9/H10 已完成正式 T=9 全量验证：

| 分支 | V2 | V3 | V4 | 对 H8 frozen |
|---|---:|---:|---:|---|
| H9 MixSTE pose residual（MixSTE objective） | 49.503 | 52.715 | 39.793 | 全部更差 |
| H10 MixSTE TTB residual（RUMPL MPJPE） | 46.041 | 41.882 | 33.940 | 全部更差 |

这两条线均为严格 GBT-aligned HRNet 坐标输入，不能与旧 A1D MixSTE 数值混用。它们
说明在当前 absolute world-coordinate RUMPL 中，直接对 pose 或 VFT token 做无条件
时序 residual 会破坏射线定位；时序若继续，只能作为遮挡/缺失时由置信度触发的回退，
而不应作为 clean 主干的无条件增强。

---

## 18. 与外部论文公开结果的对比

### 18.1 可以引用的目标/参考值

| 方法 | 输入与协议 | T | V2 | V3 | V4 | 是否能与当前主表直接比 |
|---|---|---:|---:|---:|---:|---|
| RUMPL paper | MMPose COCO 2D；MHP/AMASS 随机相机训练；All-KP absolute | 1 | 52.5 | — | — | 只适合 RUMPL 复现线 |
| GBT Ours (HRNet) | HRNet-W32 COCO + YOLOX；坐标/conf/rays；absolute | 9 | 36.8 | 30.4 | 26.0 | 目标值；无代码，前端细节不全 |
| GBT Ours (ResNet-152) | H36M fine-tuned ResNet-152；absolute | 9 | 29.9 | 24.4 | 22.7 | 不同 2D detector |
| GBT Algebraic (HRNet) | 同论文 HRNet 坐标；absolute | 1/未说明 | 120.7 | 50.9 | 44.2 | 输入目标线参考 |
| GBT Algebraic (ResNet-152) | fine-tuned detector；absolute | — | 51.1 | 23.4 | 19.1 | 不同 detector |
| LT Algebraic paper | ResNet-152、GT/segment box、去畸变、图像模型；absolute | 1 | — | — | 19.2 | 作者协议表，不是 raw HRNet |
| LT Volumetric paper | 图像/体素特征；absolute | 1 | — | — | 17.7 | 图像级方法 |
| AdaFuse paper | ResNet-152、GT box、完整热图；其官方代码口径含 pelvis/root alignment | 1 | — | — | 19.54 | 只能在热图/root-aligned表比较 |
| SGraFormer | CPN，多视角时序，H36M P1 | 27 | 32.0 | 31.1 | 27.6 | root-relative/时序协议，不直接比 absolute |

### 18.2 GBT 关键消融，可作为方法动机

GBT H36M V4、HRNet、absolute MPJPE：

| Centering | Synthetic | Confidence bias | Geometry bias | V4 |
|---|---|---|---|---:|
| × | × | × | × | 39.0 |
| ✓ | × | × | × | 49.2 |
| ✓ | ✓ | × | × | 40.6 |
| ✓ | ✓ | ✓ | × | 33.2 |
| ✓ | ✓ | × | ✓ | 33.1 |
| ✓ | ✓ | ✓ | ✓ | 26.0 |

时间帧消融：T=1/2/3/6/9 对应 V4 29.4/27.9/27.7/27.3/26.0 mm。说明偏置和时序在 GBT 的完整 global encoder-decoder 中有效，但不能推出把公式插入 RUMPL VFT 会获得同等收益。

### 18.3 AdaFuse 内部同输入消融

官方 README/代码线 V4 约为：NoFuse 22.94、HeuristicFuse 21.02、ScoreFuse 20.14、RANSAC 21.77、AdaFuse 19.54 mm。价值在于同前端下 AdaFuse 相对 NoFuse 约 3.4 mm；但它使用完整热图、ResNet-152、GT box 和 root-aligned 实现口径，不能与当前 absolute HRNet 坐标结果直接相减。

### 18.4 本机公开代码控制

- LT Algebraic 官方权重 + 当前 annotation box + 去畸变：53.592/24.339/19.921 mm（absolute V2/V3/V4）；V4 距官方 19.2 约 0.72 mm。
- 同 LT 坐标 uniform vs learned confidence：54.553/29.005/26.248 → 53.592/24.339/19.921；学习置信度主要在 V3/V4 有效，V2 只有 0.26--1.44 mm，因为两视角没有冗余。
- LT + 官方 AdaFuse view-weight 的 root-relative V2/V3/V4 两 seed约为 49.93/29.13/25.38 和 49.68/28.88/25.27；同输入下均有提升，但仍与 absolute 主表不同。

---

## 19. 已完成失败实验及其论文价值

论文可在附录/消融中选择性报告这些结果，主文不必全部堆表，但必须吸收其结论。

### 19.1 把骨架 Transformer 放错位置

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76 | 34.816 | 30.489 | 29.691 |
| SG-M0 position-only pre-VFT | 40.529 | 35.615 | 35.006 |
| SG-M1 SGraFormer pre-VFT | 46.449 | 37.306 | 36.629 |
| GF-M0 GraAttention 替换 PFT | 42.312 | 37.296 | 34.751 |
| GF-M1 full GraFormer 替换 PFT | 45.065 | 35.986 | 34.965 |

结论：VFT 前混合关节会破坏同一关节跨视角 ray identity；用单视角 2D-to-3D graph 模块替换 PFT 也与 world-ray feature 不匹配。

### 19.2 简单 global/query residual

- G0/G1 在正式 GBT-style 输入上显著退化；
- H76 旁路 GBT query residual 的 global/per-joint memory 仅得到 34.8387/30.4943/29.6876 和 34.8243/30.4797/29.6929，变化 <0.03 mm；
- 说明最终 3D 旁路残差读出没有新信息，必须在候选内容或早期视角对应处改进。

### 19.3 直接射线/3D 纠错跨主体失败

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76 cached identity | 34.816 | 30.489 | 29.691 |
| D0 geometry correction | 34.892 | 30.767 | 30.064 |
| D1 utility-gated correction | 34.973 | 30.578 | 29.825 |

训练主体 holdout 改善，但 S9/S11 退化，说明直接用 3D GT 训练自由纠错头会学习主体/动作偏差。后续必须有更强 identity/trust-region、图像证据或显式解析出口。

### 19.4 候选融合的失败选择

- hard top-1 在 V4 会因少量误选退化；
- sparsemax：29.764/29.186，top-3：29.707/29.327，top-5：29.447/29.113，均差于 softmax 29.421/28.727；
- weighted geometric median 29.699/28.951、medoid 29.963/29.126，也退化；
- source×joint bias、Gumbel exploration、bone loss、canonical GHT length score 均无稳定同向提升；
- 结论是剩余 gap 主要来自候选内容和评分泛化，而不是把 softmax 改成稀疏选择。

### 19.5 时序失败经验

- 冻结/后置 MixSTE 容易在训练主体下降、S9/S11 退化；
- 对已经压缩的 3D/candidate 做后处理通常只有 0.1--0.2 mm；
- GBT 的时序在 joint-view-time token 压缩前参与并对 9 帧端到端监督，因此两者不等价；
- 下一次若做时序，应只在最强 early-token 主干上做 T=1/T=9 单变量对照，不再重复后处理深度/window sweep。

---

## 20. 推荐贡献写法

当前初稿可以写为三点，措辞需保守：

1. **问题与诊断贡献**：提出 Negative View Rate 和嵌套相机子集诊断，发现 H36M 的双视角误差主要由少数退化相机对和局部关节负贡献造成；平均 MPJPE 不能充分描述多视角融合质量。
2. **几何锚定的局部跨视角精化**：以置信度加权射线交点锚定绝对位置，用锚点中心化 Plücker 表示保持相机无关性；在粗查询附近读取冻结 HRNet patch，并将连续射线几何作为真实 view-attention bias，改善候选内容。
3. **反事实候选效用**：对所有合法视角子集生成 3D 假设，以逐关节真实边际误差监督一个 permutation-equivariant Set Transformer，并以软风险融合降低严重 Negative View 事件。

若当前 GBT-style 公平输入的 RIGR/E2 实验还未完成，摘要最后一句只能写“实验表明各模块在受控内部协议中有效，并在严格坐标协议上完成了基线与瓶颈定位”；不要写“达到 SOTA”。

若未来 C1/C2/E2 当前输入结果达到目标，可改为：

> 在 H36M absolute MPJPE 的全部相机组合评估上，我们的方法同时改善 2/3/4 视角，并显著降低退化相机对和严重 Negative View Rate；在未见相机组合和噪声测试中保持更稳定的性能。

---

## 21. 推荐摘要草稿（结果占位版）

多视角三维人体姿态估计通常假设新增相机能够单调改善重建，但真实二维检测中，低质量或几何退化的视角可能反而破坏局部关节。我们发现，在 Human3.6M 上，两视角误差主要集中于少数退化相机对；即使平均误差随视角数下降，超过四成的逐帧关节仍可能在增加视角后恶化。为此，本文提出一种几何锚定的射线精化与反事实候选效用框架。首先，我们把二维关键点转换为相机无关的世界射线，并以置信度加权射线交点提供绝对三维锚点，在锚点中心坐标系中构造 Plücker 射线表示。随后，以粗三维姿态为查询，在冻结 HRNet 的局部特征图上采样跨视角 patch，并将射线夹角、线间距离和检测置信度作为同关节跨视角注意力偏置，以修正几何退化下的三维候选。最后，我们从当前相机集合的合法子集生成多个三维假设，直接监督每个候选相对全视角基线的逐关节边际风险，并通过集合 Transformer 和软候选融合抑制负视角贡献。该方法不使用相机编号嵌入，并支持可变数量和任意顺序的输入视角。在统一的 Human3.6M S1/S5/S6/S7/S8→S9/S11 协议下，当前完整结果为 `待填`；已有受控实验显示，几何偏置在配对种子下为 V2/V3/V4 带来约 0.56/0.44/0.48 mm 的稳定收益，反事实候选融合显著降低超过 1 mm 和 5 mm 的 Negative View Rate。最终稿将补充严格坐标输入下的主表、未见相机配置、噪声鲁棒性和效率评估。

注意：摘要中的 0.56/0.44/0.48 来自历史增强输入的 V234 配对实验。若最终主表使用 GBT-style raw coordinate 输入，应在当前输入复验完成后再决定是否保留这句话。

---

## 22. 推荐论文结构

### 1. Introduction

- 多视角通常被认为“更多视角更好”，但坏视角存在负增益；
- 解析三角化相机泛化好但怕误检/退化基线；自由回归可利用人体先验但易记忆相机；
- RUMPL 用射线解决表示泛化，但过早 VFT 压缩使坏双视角难修复；
- 由 H36M 六个相机对和 NVR 引出方法；
- 概述几何锚点、局部 correspondence、反事实效用三部分；
- 给出三点贡献。

### 2. Related Work

2.1 解析与可学习三角化：DLT、Learnable Triangulation、Probabilistic Triangulation、LOSTU。  
2.2 热图/特征级跨视角融合：Cross View Fusion、Epipolar Transformer、AdaFuse、MVGFormer。  
2.3 射线与相机泛化 lifting：MPL、RUMPL、GBT、GHT、PoseIRM。  
2.4 时序姿态：PoseFormer、MixSTE、MTF-Transformer；说明时序是补充而非当前核心。  
2.5 不确定性和负视角：AdaFuse、UPose3D、DeProPose；明确本文的反事实 3D 边际效用差异。

### 3. Method

3.1 Problem formulation and world rays。  
3.2 Confidence-weighted triangulation anchor。  
3.3 Anchor-centered Plücker ray lifting。  
3.4 Query-guided local HRNet evidence。  
3.5 Geometry-biased cross-view correspondence。  
3.6 Counterfactual hypothesis utility and Set Transformer。  
3.7 Soft fusion and training objectives。  
3.8 Optional fixed-lag temporal refinement。

### 4. Experiments

4.1 H36M data, input protocols, metrics。  
4.2 Fair coordinate-level comparison。  
4.3 Enhanced heatmap/feature-level comparison。  
4.4 Main results V2/V3/V4。  
4.5 Pairwise camera and Negative View analysis。  
4.6 Ablation。  
4.7 Unseen camera/noise/occlusion robustness。  
4.8 Efficiency and limitations。

### 5. Discussion

- 为什么 confidence weighting 主要改善 V3/V4 而非 V2；
- 为什么全局偏置、骨架模块和后置时序失败；
- 为什么 soft candidate fusion 优于硬选择；
- camera-generalizable 的边界：仍依赖同步和标定；
- 多人扩展需要额外的跨视角身份关联，不属于本稿。

---

## 23. 论文应画的图和表

### 图 1：方法总览

画出 RGB/HRNet→世界射线→三角化锚点→H76 粗姿态→投影采样 HRNet patch→geometry-biased correspondence→多个子集候选→E2 soft fusion。用颜色区分解析几何、冻结前端、可学习模块。

### 图 2：Negative View 问题

- 左：六个 V2 相机对柱状图，突出 1-4、2-3；
- 中：V2→V3、V3→V4 的逐关节 delta 分布；
- 右：新增视角后某腕/踝误差上升的例子。

### 图 3：几何锚点与局部 correspondence

显示 q_tri、q_base 投影、5×5 patch、跨视角 attention 和 ray distance/angle bias。

### 图 4：候选效用

显示 V4 任务中 6 个 V2、4 个 V3、1 个 V4 候选；每个关节有不同 soft 权重。展示“全视角候选对右腕有害，某 V3 子集更好”。

### 表 1：外部结果协议表

列 Method、Input、BBox、T、Alignment、V2/V3/V4。先把协议列清楚，再放数字。

### 表 2：统一坐标输入主表

至少包含 DLT/conf-DLT/IRLS、RUMPL R0、H76、B1/B2、Ours 各消融。C1/C2 当前待填。

### 表 3：模块消融

| Anchor | Plücker | HRNet patch | Corr. | Geo bias | Utility | Temporal | V2 | V3 | V4 |

### 表 4：V2 相机对

六个组合全部报告。

### 表 5：NVR/Monotonic

报告 NVR、>1、>5、Pose NVR、root-relative NVR。

### 表 6：鲁棒性

2D Gaussian noise 2/5/10 px、joint dropout 10/20/30%、视角缺失、标定扰动、H36M-Occl。

### 表 7：效率

Params、FLOPs、lifter latency、FPS、peak memory；2D detector 与 3D lifter 分开。

---

## 24. 关键代码和产物索引

### 24.1 RUMPL/H76 主干

- 主模型：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- 三角化锚点：约 1557--1595 行；
- 锚点中心化/Plücker：约 1596--1616 行；
- anchor residual 输出：约 2359--2363 行；
- 训练入口：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/run/train_rumpl.py`
- 评估：`run/eval_rumpl_checkpoint.py`、`run/eval_h36m_table2.py`。

### 24.2 GBT-style 输入

- 导出：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/export_h36m_gbt_aligned_hrnet_20260814.py`
- train merged：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl`
- validation merged：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl`
- 当前结果计划：`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/GBT_V2_GAP_DIAGNOSIS_AND_NEXT_EXPERIMENTS_20260815.md`
- 相机对诊断：`H76_GBT_INPUT_PAIRWISE_ANALYSIS_AND_E2_DECISION_20260815.md`。

### 24.3 RIGR 特征精化

- HRNet 特征导出：`export_h36m_hrnet_features.py`
- query-guided token：`prepare_rigr_feature_tokens_20260812.py`
- correspondence/geometry bias/refiner：`train_rigr_hrnet_feature_20260812.py`
- 实验记录：`RIGR_P2_FEATURE_IMPLEMENTATION_LOG_20260812.md`
- correspondence 配对结果：`RIGR_CORRESPONDENCE_IMPLEMENTATION_LOG_20260812.md`。

### 24.4 E2 候选效用

- 原始候选导出：`export_h76_train_subset_hypotheses_20260811.py`
- Set Transformer：`train_h76_set_transformer_utility_20260811.py`
- 通用 V234：`train_e2_v234_universal_20260812.py`
- candidate build：`build_e2_v234_candidate_cache_20260812.py`
- 当前输入 oracle：`build_current_input_e2_oracle_20260815.py`
- 完整结果：`COUNTERFACTUAL_VIEW_UTILITY_PLAN_AND_RESULTS_20260811.md`
- V234 实现：`E2_V234_IMPLEMENTATION_LOG_20260812.md`
- 收敛：`E2_CONVERGENCE_AUDIT_20260812.md`。

### 24.5 当前运行

- C1/C2 启动：`launch_cardinality_recovery_20260815.sh`
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/cardinality_recovery/`
- 当前 E2：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input/`
- 挂载盘结果根：`/mnt/data/cjyoutput/`。

### 24.6 本地论文与官方代码

- RUMPL：`/home/lixiaob/cjy/reference/rumpl.pdf`
- GBT：`/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`
- 旧中文草稿：`/home/lixiaob/cjy/reference/AAAmy/TGR_Ray_Chinese_CVPR_Draft.pdf`
- Learnable Triangulation：`reference/learnable-triangulation-official/`
- AdaFuse：`reference/adafuse-official/`
- Epipolar Transformer：`reference/epipolar-transformers-official/`
- GHT：`reference/general-3d-humans-official/`
- MVGFormer：`reference/MVGFormer_official/`
- SGraFormer：`reference/SGraFormer_official/`
- MixSTE：`reference/Mixste(1).pdf`。

### 24.7 可放入论文附录或交给 Codex 理解实现的关键代码

三角化锚点的核心逻辑（与当前实现一致的精简版）：

```python
unit_dir = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
weight = confidence.clamp(0, 1) + 0.05
eye = torch.eye(3, device=unit_dir.device, dtype=unit_dir.dtype)
projector = eye - unit_dir.unsqueeze(-1) * unit_dir.unsqueeze(-2)
weighted_projector = weight.unsqueeze(-1) * projector
A = weighted_projector.sum(dim=2) + 1e-4 * eye
b = (weighted_projector @ camera_center.unsqueeze(-1)).sum(dim=2)
q_tri = torch.linalg.solve(A, b).squeeze(-1)
```

Plücker token：

```python
direction = F.normalize(direction, dim=-1, eps=1e-7)
centered_origin = center_ray_points_on_anchor(origin, q_tri)
moment = torch.cross(centered_origin, direction, dim=-1)
ray_token_input = torch.cat([direction, moment, confidence], dim=-1)
```

局部 correspondence 的核心逻辑：

```python
# features: [B,V,J,C,P,P], P=5
patch_tokens = patch_encoder(features).reshape(B * J, V * P * P, d_model)
query = base_view_feature.reshape(B * J * V, 1, d_model)
key_value = repeat_for_each_query_view(patch_tokens)
attended, _ = cross_view_attention(
    query, key_value, key_value,
    key_padding_mask=invalid_view_patch_mask,
    need_weights=False,
)
mixed = layer_norm(base_view_feature + attended.reshape(B * J, V, d_model))
view_feature = base_view_feature + torch.tanh(corr_gate) * (mixed - base_view_feature)
```

几何 pair bias：

```python
d = F.normalize(rays[..., :3], dim=-1)
o = rays[..., 3:6]
c = rays[..., 6].clamp(0, 1)
cross = torch.cross(d.unsqueeze(-2), d.unsqueeze(-3), dim=-1)
sine = torch.linalg.vector_norm(cross, dim=-1)
cosine = (d.unsqueeze(-2) * d.unsqueeze(-3)).sum(-1).abs()
baseline = o.unsqueeze(-2) - o.unsqueeze(-3)
distance = (baseline * cross).sum(-1).abs() / sine.clamp_min(1e-5)
pair_feature = torch.stack([cosine, sine, normalized(distance), c_i, c_j], -1)
attention_bias = zero_initialized_mlp(pair_feature)  # [B,J,heads,V,V]
```

反事实效用和软融合：

```python
candidate_error = torch.linalg.vector_norm(
    candidates - target[:, None], dim=-1
)                              # [B,K,J]
true_error = candidate_error.permute(0, 2, 1)  # [B,J,K]
true_delta = true_error - true_error[..., baseline_id:baseline_id + 1]

score = utility_set_transformer(candidate_features)  # lower is better
pred_delta = score - score[..., baseline_id:baseline_id + 1]
weights = torch.softmax(-pred_delta / temperature, dim=-1)
fused = torch.einsum("bjk,bkjd->bjd", weights, candidates)
expected_risk = (weights * true_error).sum(-1).mean()
loss = balanced_rank_loss(pred_delta, true_delta)
loss = loss + (expected_risk + 0.05 * mpjpe(fused, target)) / 0.01
```

相机子集 mask 的核心约束：

```python
available = [
    k for k, candidate_views in enumerate(candidate_combinations)
    if set(candidate_views).issubset(current_task_views)
]
```

该约束必须在训练和测试同时存在；否则 V2/V3 会间接使用未选视角。

---

## 25. 当前论文绝对不能写错的事项

1. **旧 A1D/H21 V2/V3 不是严格公平外部结果。** 前端先看了四视角热图；只能作历史增强输入消融。
2. **oracle 不是模型结果。** 25/19 mm 等数字使用 GT 选择候选，只能表示潜在上限。
3. **B2 不是统一最好模型。** 它 V2=37.886，但 V3/V4=62.215/46.217，不能只截取 V2 写 Ours。
4. **GBT HRNet 是 T=9。** 当前多数结果 T=1，不能隐去时间维度。
5. **GBT 没代码。** 只能称 paper-reported target 或 GBT-style approximation。
6. **AdaFuse 19.54 与当前 absolute 指标不同。** 其官方 H36M evaluator做 pelvis/root alignment，并使用完整热图、ResNet-152 和 GT box。
7. **SGraFormer 32.0/31.1/27.6 是 CPN、T=27、P1 口径。** 不应直接宣称 absolute 超越。
8. **HRNet 中间特征方法不再是坐标级。** 如果完整模型读取 stage-4 patch，必须放 feature-level 表。
9. **时序不是当前主要创新。** 当前强证据只有 V2 约 1.76 mm和 V3/V4 0.1--0.6 mm，不能声称统一 3 mm。
10. **偏置不是无条件有效。** 它在 RIGR correspondence 后配对有效，在简单 RUMPL/global 插入时失败。论文必须强调作用层与语义 mask。
11. **不能声称 end-to-end 联合训练，除非最终代码确实联合训练。** 当前许多最佳结果采用冻结 H76→RIGR→离线候选→E2 分阶段训练。
12. **不能声称多人。** 当前是单人多视角；多人需要跨视角身份关联和实例查询。

---

## 26. 初稿中可以诚实写出的阶段性结论

1. 数据、相机、同步和坐标系已经由 GT-2D 三角化约 0.3--0.5 mm 的误差验证，当前几十毫米误差来自二维检测和融合，不是标定单位错误。
2. H76 的三角化锚点、中心化射线和 Plücker 表示是当前坐标级 RUMPL 的关键提升来源。
3. 两视角平均的主要瓶颈是 `1-4` 和 `2-3`，不是所有相机对都差。
4. 延长到约 300k updates 有益，但不足以单独达到 GBT；固定 K=2 可修复坏对，却造成视角数量分布崩溃。
5. 简单 global attention、全 token 几何偏置、SGraFormer pre-VFT、GraFormer PFT replacement、末端 query residual 均失败，说明接入位置比模块名字重要。
6. HRNet 中间特征的 query-guided refinement 与候选 utility 在历史受控协议上互补。
7. 几何偏置只有在同一关节跨视角 correspondence 的 attention logit 上才得到稳定配对收益。
8. 反事实 Set Transformer 能降低严重 Negative View Rate，soft fusion 比 hard/sparse/robust selection 更稳定。
9. 时序后处理作用有限；若要复现 GBT 的 3.4 mm，需要在 view fusion 前建模 joint-view-time token，并端到端训练。

---

## 27. 完成论文前还必须补的实验

按重要性排序：

1. **C1/C2 当前坐标协议结果**：已完成；C2 是当前统一折中模型，仍需多 seed 确认。
2. **当前 GBT-style 候选训练与 E2-V234**：使用修正 `flip=false` 缓存重新验证候选；旧协议的 confidence oracle 增益不再采信，IRLS 仍排除。
3. **统一 raw-coordinate 主表**：R0、H76、B1/B2、C1/C2、confidence-DLT、E2；三种子至少用于最终 Ours 和主 baseline。
4. **严格特征级 RIGR 重做**：按实际视角子集采样 HRNet patch；不使用旧四视角 A1D 烘焙缓存。
5. **完整消融**：anchor、Plücker、correspondence、bias、utility，保持同输入、同训练预算。
6. **NVR/相机对分析**：当前正式输入上的 Ours vs B1/H76。
7. **未见视角组合/顺序置换**：训练排除部分相机组合，测试未见组合；视角 permutation 必须数值一致。
8. **鲁棒性**：2D noise、joint dropout、camera dropout、轻微标定扰动；至少 H36M-Occl 或人工遮挡一项。
9. **时序 T=1/T=9**：只在最强单帧模型上做一次，若收益仍小则作为补充。
10. **效率和参数量**：所有模块的 Params/FLOPs/latency。

建议投稿门槛：

- 坐标级统一模型 V2 < 36.8、V3 < 30.4、V4 < 26.0 是超过 GBT HRNet 公开目标的理想门槛；
- 若 clean MPJPE 未完全超过，至少需在 Negative View、坏相机对、噪声/缺失、未见相机组合中有显著优势；
- 论文不能只靠 0.1 mm clean 改善，需用机制和鲁棒性证据形成完整贡献。

---

## 28. 历史稿旧写作要求（已被文首 2026-08-22 指令覆盖）

> 本段仅保存旧稿演化历史，不再是给 Codex 的有效指令。正式写作必须使用文首
> “最高优先级路线覆盖声明”、A 节冻结技术方案和 B 节 GBT 对齐表，不能优先使用
> 下述旧稿第 3--14 节，也不能把旧 RIGR/图像特征/view-bias 路线写成当前方法。

旧要求曾计划生成一篇包含标题、摘要、引言、相关工作、问题定义、方法公式、训练
目标、实验协议、主结果表、消融、失败分析、局限性和结论的论文，并将创新集中到
Negative View Problem、反事实逐关节效用和严格相机子集评估。此目标结构仍可参考，
但技术事实和结果必须完全服从文首 2026-08-22 路线。

---

## 29. 历史路线中相关论文的作用（不得直接当作当前模块表）

| 论文 | 本稿借鉴内容 | 本稿不能声称的内容 | 本稿差异 |
|---|---|---|---|
| RUMPL | world ray、VFT/PFT、可变视角、MHP 相机随机化 | 首次射线 lifting、首次相机泛化 | 增加解析锚点、局部图像精化与反事实效用 |
| Geometry-Biased Transformer | confidence/ray-distance bias、global query、T=9/random K2 训练 | 严格复现其模型或前端 | bias 只用于同关节 correspondence，并配合候选效用 |
| Learnable Triangulation | learned confidence、图像特征三角化、官方 H36M protocol | 首次可学习三角化 | 解析锚点作为 residual origin，最终还做集合效用 |
| Epipolar Transformer | 跨视角图像特征对应 | 首次跨视角 feature attention | 只在粗 3D 查询附近使用紧凑 patch，不做全极线图像模型 |
| MVGFormer | 3D query 投影、局部图像证据、geometry-feature refinement | 首次 2D/3D iterative query | 保留稀疏 ray lifter，并针对负视角候选选择 |
| AdaFuse | 低质量视角会有负贡献；关节级视角权重 | 首次 adaptive view weighting | 用真实反事实 3D 边际风险，而非只按热图质量加权 |
| Generalizable Human Pose Triangulation | 多几何假设、ScoreNN、相机布局泛化 | 首次学习假设评分 | 逐关节 utility、候选 self-attention、嵌套视角 Negative View |
| UPose3D | 2D uncertainty、时序 compiler、MLE | 首次 uncertainty/temporal correction | 当前主线不依赖其完整 compiler；重点是负视角反事实效用 |
| DeProPose | reprojection-error-aware dynamic fusion | 首次按投影误差做视角权重 | 使用射线条件谱和删除视角的真实 3D 边际贡献 |
| MixSTE | 分解空间/时间轴 attention | 首次 factorized temporal transformer | 当前只作时序对照；实验证明直接移植有跨主体过拟合 |
| SGraFormer/GraFormer | 骨架图语义、关节 Transformer | 首次 graph pose modeling | 本项目将其作为失败位置消融，证明 ray identity 需要先保护 |

建议参考文献至少包含：Human3.6M、RUMPL、MPL、GBT、Learnable Triangulation、AdaFuse、Epipolar Transformer、MVGFormer、Generalizable Human Pose Triangulation、UPose3D、MixSTE、PoseFormer、PoseIRM、Probabilistic Triangulation、SGraFormer。最终 BibTeX 应从本地 PDF/官方仓库导出，避免由语言模型猜作者或页码。

---

## 30. 2026-08-15 C1/C2 cardinality recovery 完成补录

### 30.1 运行有效性

此前两组恢复实验的早期启动脚本曾遗留 `TRAIN_FIXED_NUM_VIEWS=2`，那批进程已停止且不纳入结果。修正脚本后重新完成的正式运行日志明确显示：C1 使用 `weighted-random-3,1,1`，C2 使用 `weighted-random-8,1,1`；两组均为 H76、T=1、20E、LR=1e-5，并分别从 B2/B1 checkpoint 微调。以下数值来自自动生成的严格评估 `table2.json`，不是中途日志或手工抄录。

### 30.2 结果

| 实验 | 初始化与 K 分布 | V2 | V3 | V4 | 结论 |
|---|---|---:|---:|---:|---|
| B1 | 299,874 updates；前 8E K=2，后 3:1:1 | 43.456 | 30.732 | 27.818 | 原统一折中基线 |
| B2 | 299,874 updates；全程 K=2 | 37.886 | 62.215 | 46.217 | K=2 专家，发生 cardinality collapse |
| C1 | B2 → K2/K3/K4=3:1:1，20E | **37.007** | 32.103 | 30.930 | V2 保留最好水平，但 V3/V4 尚未恢复 |
| C2 | B1 → K2/K3/K4=8:1:1，20E | 38.686 | **30.943** | **28.629** | 当前最好的统一折中模型 |

相对 B1，C2 将 V2 降低 4.770 mm，同时 V3 略降 0.211 mm，V4 上升 0.811 mm；相对 H76 短预算基线（46.227/31.334/27.964），C2 分别改善 7.541/0.391 mm，但 V4 仍高 0.665 mm。相对 GBT 的 HRNet 公开目标 36.8/30.4/26.0，C2 仍有约 +1.886/+0.543/+2.629 mm 差距。因此 C2 可作为当前主线 baseline/统一模型，不能写成已经超越 GBT。

C1 的含义是：从 K=2 专家出发，加入少量 K=3/K=4 训练即可避免完全崩溃，但 3:1:1 仍不足以恢复多视角；C2 说明较强的 K=2 权重可以在不牺牲 V3 的情况下改善 V2，但 V4 仍存在训练分布权衡。这一结果支持“cardinality-aware training 是必要条件”，但尚未证明仅靠采样即可解决负视角问题。

### 30.3 输出位置与下一步

- C1/C2 日志与 checkpoint：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/cardinality_recovery/`；
- C1 checkpoint：`/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C1_B2_TO_MIXED_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-03-00/model_best.pth.tar`；
- C2 checkpoint：`/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar`；
- E2 旧输入训练/验证候选缓存位于 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input/`，因 `flip=true` 仅作废审计记录；修正缓存位于 `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input_protocol_v2/`，正式 E2 只能使用后者。

下一步应先完成 confidence-candidate-only 的 E2 训练与严格 V2/V3/V4 评估，再用 C2 和 E2 做同输入、同预算、多 seed 对照；当前 C1/C2 训练均已结束，GPU 空闲。

## 31. 2026-08-15 当前公平输入 E2 实验启动记录

为避免把历史 RIGR/热图/IRLS 候选混入 GBT-style 坐标级比较，当前 E2 使用独立的 22 候选池：前 11 个是 H76 原始候选，后 11 个是同一组相机子集上的 confidence-weighted closest-point triangulation。两部分的相机组合顺序完全一致，模型仍使用逐关节 Set Transformer utility scorer、V2/V3/V4 任务掩码、direct ranking → GHT-style expected-risk 两阶段训练。

- 训练缓存：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input/train_h76_22c.npz`；
- 验证缓存：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input/validation_h76_22c.npz`；
- 候选生成脚本：`OpenRUMPL_baseline_audit/append_confidence_candidates_current_input_20260815.py`；
- 训练适配器：`OpenRUMPL_baseline_audit/train_current_e2_confidence_20260815.py`；
- 两 seed 启动脚本：`OpenRUMPL_baseline_audit/launch_current_e2_confidence_20260815.sh`。

验证集零训练 oracle 仅用于候选池诊断，原先记录的 `51.127/28.946/22.964 → 45.657/25.237/19.672 mm` 属于旧的下肢关节交换协议，已被协议审计判为无效，不再用于主表。正式 E2 结果必须来自修正后的严格缓存，不能用 oracle 替代。

## 32. 2026-08-15 E2 协议审计、C2 基线层级与修正重跑

### 32.1 当前模型层级

C2 K2-heavy 是当前同一坐标级输入、同一 RUMPL 主干下的最佳统一模型，应作为当前 Ours baseline；B1 仍保留为原始训练协议基线，不能把 B1 称为当前最好模型。

| 层级 | 含义 | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| B1 | RUMPL/H76 原始随机视角训练基线 | 43.456 | 30.732 | 27.818 |
| C2 K2-heavy | 保留 RUMPL 主干，仅改变 K=2/3/4 采样权重（8:1:1） | **38.686** | **30.943** | **28.629** |
| E2-B1 | B1 候选池 + 逐关节 Set Transformer utility（两 seed 均值） | 43.217 | **29.101** | **26.309** |

### 32.2 旧 E2 缓存为何无效

旧 E2 缓存由 `H35_a1d_h21_refined_rumpl_tri_anchor.yaml` 的默认
`FLIP_LOWER_BODY_KP_TEST=true` 导出，而严格 B1/C2 评估显式使用
`--flip-lower-body-kp-test false`。这会交换 H36M 内部的左右下肢关节语义，导致缓存中的 H76 预测、confidence 候选和目标不能与主表比较。旧 E2 训练得到的约 `48.55/32.83/29.73` 只记录为“协议错误技术试跑”，禁止写入主结果。

### 32.3 修正后的逐项协议验证

导出脚本新增 `--flip-lower-body-kp-test {true,false}`，严格线固定为 `false`。修正缓存目录为：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input_protocol_v2/`

在验证集上，修正缓存中的 H76 原始候选复算为：

| 候选池 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76 原始候选（逐组合平均） | 43.456 | 30.732 | 27.818 |
| confidence-weighted 三角化候选 | 90.634 | 57.300 | 52.905 |
| 两者 oracle | 43.433 | 30.725 | 27.818 |

H76 原始候选数值与严格 B1 表一致（四视角逐样本输出最大差约 `2.4×10^-7`）。confidence 候选在正确关节语义下没有 oracle 增益，说明旧缓存的“候选互补收益”是协议错配假象；E2 的修正重跑用于确认效用头是否会学习恒等选择以及验证该失败结论。

### 32.4 修正 E2 重跑

- 训练缓存：`train_h76_22c.npz`；验证缓存：`validation_h76_22c.npz`；
- 22 个候选按相同 11 个相机子集排列：前 11 个 H76，后 11 个 confidence solver；
- 训练：direct ranking 10 epochs + GHT expected-risk 5 epochs，batch 256，温度 1.8，seed 0/1；
- 输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_confidence_training_protocol_v2/seed{0,1}/`；
- 当前状态：seed0/seed1 已完成；正式均值和标准差见第 33 节。该线以 B1 候选池为冻结输入，不能直接称为 C2-E2。

本节规则：任何 E2 数字只有在修正缓存、严格 B1 复算和两 seed 结果全部通过后，才能进入第 17 节主结果表。

## 33. 2026-08-15 修正协议 E2 正式结果

### 33.1 结果与统计

模型使用严格 `FLIP_LOWER_BODY_KP_TEST=false` 的 B1/H76 候选池；seed0 和 seed1 使用相同训练/验证缓存、相同 direct-ranking 10E + GHT 5E 预算。下表为 S9/S11、action-equal、All-17、绝对 MPJPE，单位 mm；`soft` 是论文主推的 utility soft fusion，`hard` 是只取最低预测风险候选，`oracle` 仍然使用 GT 选择，仅作上限。

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| B1 严格基线 | 43.456 | 30.732 | 27.818 |
| E2-B1 hard，seed0/1 均值±std | 43.332±0.023 | 30.632±0.202 | 27.776±0.267 |
| **E2-B1 soft，seed0/1 均值±std** | **43.217±0.003** | **29.101±0.030** | **26.309±0.024** |
| E2-B1 oracle（不可作为模型结果） | 40.329 | 21.462 | 16.451 |
| C2 K2-heavy（当前 V2 最佳统一模型） | **38.686** | 30.943 | 28.629 |

相对 B1，E2 soft 改善 `0.240 / 1.631 / 1.509 mm`（V2/V3/V4）；两 seed 的标准差不超过 `0.031 mm`（hard 除外）。相对 C2，E2-B1 在 V3/V4 分别改善 `1.842 / 2.320 mm`，但 V2 退化 `4.531 mm`。因此当前证据支持两条互补路线：C2 的 K2-heavy 采样保留两视角优势，E2 的反事实候选效用改善多视角；以 C2 为冻结 H76 候选池的 E2-C2 已完成，结果见第 34 节，不能把两组数值拼成单一模型结果。

### 33.2 可复现输出

- seed0：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_confidence_training_protocol_v2/seed0/result.json`，best epoch 14；
- seed1：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_confidence_training_protocol_v2/seed1/result.json`，best epoch 14；
- 修正候选缓存：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input_protocol_v2/`；
- 修正导出脚本：`OpenRUMPL_baseline_audit/export_h76_train_subset_hypotheses_20260811.py`，新增显式 `--flip-lower-body-kp-test`；
- E2 训练适配器：`OpenRUMPL_baseline_audit/train_current_e2_confidence_20260815.py`。

### 33.3 论文叙事边界

E2 的正式贡献可以写成“候选级反事实风险排序在保持相同坐标级输入的情况下显著降低 V3/V4 误差”，但不能写成“已全面超过 C2”或“全面超过 GBT”。C2 与 E2-C2 分别代表 cardinality-aware baseline 与 utility-enhanced 多视角模型；两者的列不能手工拼成一个不存在的模型。

## 34. 2026-08-15 E2-C2：当前最佳基线上的有机融合结果

### 34.1 正式结果

E2-C2 以 C2 K2-heavy checkpoint 为冻结 H76 候选生成器，严格使用 `flip=false` 的同一 22 候选缓存、同一 S9/S11 评估和两 seed 训练。结果为 action-equal All-17 absolute MPJPE（mm）：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| C2 原始候选 | 38.686 | 30.943 | 28.629 |
| E2-C2 hard，seed0/1 均值±std | 38.731±0.006 | 30.580±0.007 | 28.318±0.082 |
| **E2-C2 soft，seed0/1 均值±std** | **38.959±0.014** | **29.486±0.030** | **27.274±0.004** |
| E2-C2 oracle（不可作为模型结果） | 36.166 | 21.702 | 16.994 |

E2-C2 soft 相对 C2 的变化为 `+0.273 / −1.457 / −1.355 mm`（V2/V3/V4）。这证明候选效用模块确实能在 C2 的多视角候选之间学习到有用的反事实排序，但当前训练目标仍会牺牲少量两视角性能；因此不能把 E2-C2 直接标为全面最佳。

### 34.2 输出位置

- seed0：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2/seed0/result.json`，best epoch 14；
- seed1：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2/seed1/result.json`，best epoch 14；
- C2 候选缓存：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/`；
- C2 缓存启动脚本：`OpenRUMPL_baseline_audit/launch_e2_c2_cache_20260815.sh`。

### 34.3 下一步：identity-preserving utility

E2-C2 的失败模式不是候选池没有信息，而是 utility soft fusion 在 V2 任务上没有严格保持原始 C2 候选。下一组实验只改训练目标，不改输入或主干：

1. 对每个任务加入 `max(0, E_soft - E_baseline)` 的 identity-preserving hinge；
2. 保留 GHT expected-risk 和 direct ranking，先只在 V2 任务加权，避免破坏 V3/V4；
3. 两 seed、同一 C2 22 候选缓存、同一 15E 预算；
4. 只有当 V2 不退化且 V3/V4 保持 E2-C2 收益时，才进入论文主模型，否则将 E2-C2 作为“多视角改善但两视角 trade-off”的消融。

## 35. 2026-08-15 identity-preserving hinge：是否能在 C2 上统一提升

### 35.1 实验目的

C2 K2-heavy 是当前最强的统一基础模型，因此 identity-preserving 实验不能再以 B1 为参照，而必须约束新模型不要劣于 C2。训练中对每个任务加入

\[
L_{id}=\max(0, E_{soft}-E_{C2}),
\]

并额外提高 V2 的权重。这里的 \(E_{C2}\) 是冻结的 C2 原始候选误差，不能使用测试集标签；它只作为训练缓存中候选风险的基准。两组实验均保持同一 22 候选缓存、`flip=false`、15E 预算和两粒度种子。

### 35.2 结果

下表为 S9/S11、action-equal、All-17、绝对 MPJPE（mm），均为两 seed 均值±标准差。

| 方法 | identity hinge | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| C2 K2-heavy（冻结统一基线） | — | **38.686** | 30.943 | 28.629 |
| E2-C2 普通 soft | 0 | 38.959±0.014 | 29.486±0.030 | **27.274±0.004** |
| E2-C2 moderate | 0.25，V2×4 | 38.908±0.017 | **29.432±0.015** | **27.265±0.046** |
| E2-C2 strong | 0.5，V2×8 | 38.834±0.009 | 29.471±0.001 | **27.271±0.029** |

strong 版本相对普通 E2-C2 将 V2 退化从 `+0.273 mm` 降至 `+0.149 mm`，但仍未达到 C2；V3/V4 也没有超过 moderate 版本。因此 identity hinge 只能缓解 utility soft fusion 的 V2 损失，尚不能形成“全面超过 C2”的主模型。当前结论是：

1. **C2 K2-heavy 应作为论文中的 Ours baseline/内部主基线**，因为它在相同坐标级输入和 RUMPL 主干下给出最强 V2，且定义清楚、可复现；
2. B1 仍作为“原始 RUMPL 严格复现基线”，用于量化 C2 的训练协议收益，不能用 B1 代替 C2 做新模块对比；
3. E2-C2 是基于 C2 的多视角 utility 扩展，当前最佳 V3/V4，但存在可量化的 V2 trade-off，应作为模块消融/多视角分支报告；不能把 C2 的 V2 与 E2 的 V3/V4 拼成一个不存在的模型；
4. 下一步若继续，只能在 C2 参照下解决 V2 保持问题（例如硬选择/门控而非无约束 soft 融合），否则应停止继续堆 identity 权重。

### 35.3 可复现输出

- moderate：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_identity_protocol_v2/seed{0,1}/result.json`；
- strong：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_identity_strong_protocol_v2/seed{0,1}/result.json`；
- 训练脚本：`OpenRUMPL_baseline_audit/train_current_e2_confidence_20260815.py`，参数分别为 `--identity-hinge 0.25 --identity-v2-weight 4.0` 和 `--identity-hinge 0.5 --identity-v2-weight 8.0`。

## 36. 2026-08-15 按 GBT 实验路线重排基线与 V2 保持

### 36.1 GBT 原文实验路线核对

逐段核对 `reference/Geometry-Biased Transformer(1).pdf` 后，GBT 的 H36M 路线是：

- 训练 subjects 为 S1/S5/S6/S7/S8，测试为 S9/S11；V2/V3/V4 都枚举全部相机组合并取平均；指标为不做对齐的 absolute MPJPE；
- 输入为冻结的 COCO HRNet-W32 坐标/置信度和标定相机，正文只写 YOLOX box proposals，没有公开 YOLOX 型号、阈值、bbox padding 或代码；
- 训练时固定随机采样两视角，输入 9 帧、输出 9 帧；推理输入 9 帧但只输出最新一帧；
- Adam、batch size 256、300,000 iterations、初始学习率 `1e-4`，warmup 后平滑衰减；
- 训练增强包括 scene centering、random synthetic views 和 20% token dropout；
- 网络不是把 bias 插入 RUMPL 的 VFT/PFT，而是先把全部 joint-view-time ray token 放入 global encoder，再由 joint-query decoder 直接回归绝对 3D。confidence bias 和 ray-distance bias 只是这个完整结构中的两项 attention bias。

因此 GBT 的 `36.8/30.4/26.0` 是 T=9 完整模型的 HRNet 表，不能直接当作单帧坐标模型目标；其 Table VII 只在 V4 报告 T=1 到 T=9 的变化（`29.4 → 26.0`）。后续时序实验必须单独报告 T=1/T=9，不能把时序收益提前算入当前单帧 baseline。

### 36.2 V2 校准控制

E2-C2 soft 原来所有视角数使用 GHT 的 `T=1.8`。训练集内部 `group_indices % 10 == 0` holdout 显示 V2 的最优融合温度约为 `0.4`；不改候选、不改 RUMPL 主干、不重训，仅将 V2 温度改为 0.4，V3/V4 仍使用预注册的 GHT `T=1.8`。最终严格 S9/S11 结果为：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| E2-C2 soft，统一 T=1.8 | 38.959±0.014 | 29.486±0.030 | 27.274±0.004 |
| **E2-C2 soft，V2 holdout calibration（T=0.4/1.8/1.8）** | **38.700±0.005** | **29.486±0.021** | **27.274±0.003** |
| C2 K2-heavy 原始候选 | 38.686 | 30.943 | 28.629 |

相对原 E2-C2 soft，V2 降低约 `0.259 mm`，V3/V4 保持不变；相对 C2，V2 只差约 `0.014 mm`，同时保留 E2 的 V3/V4 优势。因此当前主线应保留两种名称：

1. **E2-C2 soft（T=1.8）**：原始 E2 消融结果，便于与前文完全对应；
2. **E2-C2 soft-cal（T=0.4/1.8/1.8）**：当前最强统一 baseline 候选，用于后续模块比较。温度只在训练 holdout 选择，不能用 S9/S11 调整。

输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_calibrated_protocol_v2/result.json`；复现实验脚本：`OpenRUMPL_baseline_audit/evaluate_e2_c2_calibrated_20260815.py`。

### 36.3 当前执行中的 V2 专门化对照

为了判断 V2 退化是否来自一个共享 utility calibration head 的尺度冲突，正在运行 **E2-C2 stage-heads**：共享候选/几何 Set Transformer，只为 V2/V3/V4 使用三个独立输出 head；候选缓存、输入、loss、15E 预算和两粒度 seed 均不变。它是一个结构消融，不改变 GBT 的输入协议；只有当 V2 继续下降且 V3/V4 不退化，才考虑把它作为 baseline，否则保留 soft-cal 并停止继续堆 V2 权重。

输出目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_stage_heads_protocol_v2/`；启动脚本：`OpenRUMPL_baseline_audit/launch_e2_c2_stage_heads_20260815.sh`。

### 36.4 后续顺序

1. 先完成 stage-heads 两 seed，确定单帧 baseline；
2. 固定该 baseline 后，再按 GBT 的全局 joint-view encoder/decoder 做 T=1 结构对照，不能把局部 RUMPL bias 称为 GBT bias；
3. 最后在同一最强单帧结构上加入 T=9、token dropout 和遮挡测试。时序的价值重点放在 H36M-Occl，clean H36M 与 T=1 必须单独列出；
4. 任何新模型都必须同时满足：V2 不高于 `38.700` 的校准 baseline，且 V3/V4 不损失当前 E2-C2 soft 的收益，否则只作为消融记录。

## 37. 2026-08-15 E2-C2 stage-heads 结果：停止拆分输出头

stage-heads 与普通 E2-C2 使用完全相同的 C2 22 候选缓存、训练 loss、10E direct + 5E GHT、batch 256 和 seed 0/1；唯一变化是共享 Set Transformer 编码器后，为 V2/V3/V4 使用三个独立 utility 输出 head。最终严格 S9/S11 结果如下：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| E2-C2 soft（统一 head，T=1.8） | 38.959±0.014 | 29.486±0.030 | 27.274±0.004 |
| E2-C2 stage-heads（T=1.8） | 38.962±0.002 | **29.473±0.004** | 27.318±0.008 |

stage-heads 没有解决 V2 退化：V2 与普通 head 几乎相同，V3 只改善 `0.013 mm`，V4 退化 `0.044 mm`。因此不能把它包装成有效模型创新，也不再继续扫 head 数量或 V2 loss 权重。当前冻结的单帧主 baseline 仍为 `E2-C2 soft-cal = 38.700/29.486/27.274 mm`；stage-heads 只保留为负结果消融。

输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_stage_heads_protocol_v2/seed{0,1}/result.json`。

## 38. 2026-08-15 GBT 路线的单帧结构筛选与 V2 基线策略

### 38.1 GBT 路线不能被简化为“给 RUMPL 加一个 bias”

对 `reference/Geometry-Biased Transformer(1).pdf` 的训练和网络描述再次核对后，GBT 的可迁移实验路线应拆成三个独立因素：

1. **训练分布**：训练时每个样本随机取两台相机，推理时枚举 2/3/4 台相机组合；模型没有针对每个测试组合单独拟合。这样做的目的，是让同一个两视角模型学习视角缺失和几何变化，而不是让 V2/V3/V4 各自拥有一个专门模型。
2. **输入和结构**：9 帧的 joint-view-time token 进入全局 encoder，再由 joint-query decoder 回归绝对 3D；confidence bias 与 ray-distance bias 是 attention 中的几何先验，不是 RUMPL 的局部 VFT/PFT 后处理。
3. **优化预算**：论文写明 batch 256、300k iterations、初始学习率 `1e-4`、warmup 和平滑衰减，并配合 scene centering、synthetic views、20% token dropout。因而当前 T=1 的局部结构筛选不能直接声称复现了论文的 `36.8/30.4/26.0`，后者是 T=9 完整模型结果。

### 38.2 T=1 全局结构 5E 筛选

为了先检查“全局 joint-view encoder + joint-query decoder”是否能在不加入时序的情况下保留现有几何模型收益，固定 `flip=false`、H76 三角化锚点、anchor-centered Plücker 射线、15 阶 harmonic、随机 K=2 训练，只比较是否启用 GBT 风格 confidence/geometry bias。该实验只训练 5 个 epoch（约 12k updates），是结构 go/no-go 筛选，不是 300k-iteration 的 GBT 复现。

| T=1 结构筛选 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 全局 set encoder/decoder，plain | 80.295 | 50.740 | 46.967 |
| 同结构 + confidence/geometry bias | 78.728 | 48.267 | 43.732 |

bias 在这个短训筛选中相对 plain 改善 `1.567/2.473/3.235 mm`，说明偏置方向并非没有信号；但绝对误差远高于当前 C2/E2，原因首先是训练预算和训练分布仍与论文不一致，不能据此把 GBT 结构判定为失败，也不能把该筛选结果写成正式主表。输出：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/gbt_singleframe_screen/{PLAIN,BIASED}/eval/V{2,3,4}/table2.json`。

### 38.3 当前 baseline 的冻结口径与 V2 专门化诊断

当前论文内部 baseline 采用两层口径：

- **C2 K2-heavy**：单帧坐标级、H76 输入、原始 RUMPL 主干，在 V2 为 `38.686 mm`，作为 V2 保持性的硬参照；
- **E2-C2 soft-cal**：在同一 C2 候选上训练 utility soft fusion，V2/V3/V4 为 `38.700/29.486/27.274 mm`，作为当前统一多视角 baseline 候选。它相对 C2 的 V2 只差 `0.014 mm`，同时保留 V3/V4 的 `1.457/1.355 mm` 收益。

为判断 V2 是否受“同时优化 V2/V3/V4”影响，新增 **V2-specialist**：候选池、H76 输入、RUMPL 生成器、训练/测试缓存全部不变，只把 utility 训练和 checkpoint 选择限制为六个两视角组合；结果必须单独报告，不能把该模型的 V2 与 E2-C2 的 V3/V4 拼接成一个模型。脚本：

`OpenRUMPL_baseline_audit/launch_e2_c2_v2_specialist_20260815.sh`；输出：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_v2_specialist_protocol_v1/`。

最终两 seed 结果为：原始 `T=1.8` soft `38.983±0.003 mm`；只在训练 holdout 上选 `T=0.4` 后为 `38.749±0.008 mm`。二者都没有超过 C2 的 `38.686 mm`，也没有超过统一 E2-C2 soft-cal 的 `38.700±0.005 mm`。因此“两视角专门训练”作为 V2 单独优化路线停止；它说明当前瓶颈不是简单的 V2/V3/V4 任务混合，而更可能来自候选三角化误差和 soft utility 的可用修正范围。

### 38.4 E2-C1：以 C1 为候选生成器的单模型扩展

C1 是从 B2 的全程 K=2 长预算 checkpoint 出发，再按 `3:1:1` 混合视角恢复 20E 得到的单一 H76/RUMPL 模型（其原始结果为 `37.007/32.103/30.930 mm`）。在不拼接不同 checkpoint 输出的前提下，将同一 E2 utility scorer 作用于 C1 候选，严格使用 `flip=false`、相同 22 候选定义、两 seed 和 10E direct + 5E GHT 训练。最终结果为：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| C1 原始单模型 | 37.007 | 32.103 | 30.930 |
| **E2-C1 soft，seed0/1 均值±std** | **37.046±0.004** | **30.108±0.028** | **28.528±0.022** |
| E2-C1 soft，V2 `T=0.4` holdout calibration | **37.009±0.002** | 30.108±0.028 | 28.528±0.022 |
| E2-C2 soft-cal（统一 baseline） | 38.700±0.005 | **29.486±0.021** | **27.274±0.003** |

E2-C1 相对 E2-C2 soft-cal 将 V2 降低约 `1.691 mm`（校准后），同时把 C1 的 V3/V4 分别降低约 `1.995/2.402 mm`；但它仍比 E2-C2 的 V3/V4 高 `0.622/1.254 mm`。因此 E2-C1 是当前“V2 优先、仍具多视角能力”的单模型候选，不能与 E2-C2 的 V3/V4 拼接成一个主结果。输出：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c1_training_protocol_v1/`；校准输出：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c1_calibrated_protocol_v1/result.json`。

为判断 C1 的 V2 优势是否能通过更充分的多视角恢复保留，已启动 `launch_cardinality_recovery_v2_20260815.sh`：`LONG_LR1E5` 为 60E、`1e-5`、30/45 学习率节点；`HIGH_LR5E5` 为 20E、`5e-5`、10/15 节点。两者均从同一个 B2 checkpoint 和 `3:1:1` 采样开始，属于单模型训练控制。

### 38.5 后续决策门槛

1. V2-specialist 若不能在两 seed 下低于 `38.686 mm`，停止 V2 专门化，冻结 E2-C2 soft-cal 为统一 baseline；
2. 若能稳定提升 V2，再用同一 checkpoint 评估 V3/V4，只有三种视角均不退化才考虑将其升级为主模型；
3. GBT 全局结构只在 T=1 保持 C2/E2 后再做长预算控制；时序 T=9 放到单帧 baseline 固定之后，重点验证 H36M-Occl 和视角缺失，clean H36M 的 T=1/T=9 必须分表。

### 38.6 Cardinality recovery v2：高学习率对照已完成，长周期仍在运行

为判断 C1 的 V2 优势能否通过更充分的多视角恢复保留，两个控制实验都从同一个 B2 checkpoint 初始化，均使用同一 RUMPL 主干、H76 输入、`flip=false`、`3:1:1` 的 K=2/3/4 采样和单模型评估；没有拼接不同模型的视角结果。

| 控制 | 训练预算 | V2 | V3 | V4 | 状态 |
|---|---:|---:|---:|---:|---|
| B2→mixed，高 LR | 20E，`5e-5`，10/15 衰减 | 36.885 | 31.451 | 30.277 | 已完成，单 seed |
| B2→mixed，长低 LR | 60E，`1e-5`，30/45 衰减 | 37.140 | 31.775 | 30.603 | 已完成，单 seed |

高 LR 对照相对 B2 原始 `37.886/62.215/46.217` 恢复了 V3/V4 的灾难性退化，并将 V2 再降约 `1.001 mm`；与当前 E2-C2 soft-cal 的 `38.700/29.486/27.274` 相比，V2 反而低 `1.815 mm`，但 V3/V4 分别高 `1.965/3.003 mm`；V2 也比 E2-C1 校准结果 `37.009 mm` 低 `0.124 mm`。长低 LR 的 `37.140/31.775/30.603` 没有超过高 LR，说明继续延长同一 mixed-cardinality 微调不能解决统一模型问题。两者均只作为训练协议控制，不升级为论文主 baseline。

输出目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/cardinality_recovery_v2/`；严格评估分别在 `HIGH_LR5E5/eval/V{2,3,4}/table2.json` 和 `LONG_LR1E5/eval/V{2,3,4}/table2.json`。

## 39. 2026-08-15 GBT 对齐六阶段实验路线

后续实验按 GBT 的论文表格组织成六阶段：H36M HRNet clean、H36M ResNet-152
clean、H36M-Occl 双输入、CMU V2/V4/V5/V6/V8、CMU→H36M、跨数据集组件消融。
完整协议、目标、代码来源、成功门槛和两卡执行顺序见：

`/home/lixiaob/cjy/GBT_ALIGNED_SIX_STAGE_EXPERIMENT_PLAN_20260815.md`。

Cardinality recovery v2 的长低学习率控制也已完成：

| 控制 | V2 | V3 | V4 |
|---|---:|---:|---:|
| B2→mixed，20E，`5e-5` | **36.885** | **31.451** | **30.277** |
| B2→mixed，60E，`1e-5` | 37.140 | 31.775 | 30.603 |

长训没有同时恢复 V3/V4，高学习率对照也只是 V2 优先的训练控制。因此当前统一
T=1 baseline 仍冻结为 E2-C2 soft-cal `38.700/29.486/27.274 mm`。下一项最高
优先级实验是在 HIGH-LR 的单 checkpoint 上重新生成完整候选池并训练 E2 两 seed，
检验同一模型能否同时保留 `36.885` 的 V2 和 E2 的多视角收益；不得拼接不同模型。

## 40. 2026-08-15 H1：HIGH-LR 单 checkpoint + E2 utility

H1 已完成。它从同一个 HIGH-LR B2→mixed checkpoint 重新导出训练/验证的 11 个
视角组合，再追加 11 个置信度加权候选，得到 22-candidate cache；E2 scorer 使用
与 E2-C2 相同的 10E direct + 5E GHT 训练、温度 `T=1.8` 和两粒度 seed。没有
把 HIGH-LR 的 V2 与其他 checkpoint 的 V3/V4 拼接。

| H1 seed | V2 baseline | V2 soft | V3 baseline | V3 soft | V4 baseline | V4 soft |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 36.885 | 36.921 | 31.451 | 30.006 | 30.277 | 28.470 |
| 1 | 36.885 | 36.916 | 31.451 | 30.051 | 30.277 | 28.603 |
| mean | 36.885 | 36.918 | 31.451 | 30.029 | 30.277 | 28.536 |

H1 说明单一 checkpoint + utility scorer 可以保持 HIGH-LR 的 V2 优势，并把同一
模型的 V3/V4 分别降低 `1.422/1.741 mm`；但仍未超过冻结的 E2-C2 soft-cal
`38.700/29.486/27.274`（V3/V4 还差 `0.543/1.262 mm`）。因此 H1 是有效的
“统一模型”控制实验，但不是论文最终 baseline。候选池 oracle 为
`34.203/21.985/17.604 mm`，表明候选中仍有较大可学习空间，不过 oracle 不能作为
可部署结果。

脚本：`OpenRUMPL_baseline_audit/launch_e2_card2_high_cache_and_train_20260815.sh`；
输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_card2_high_*_protocol_v1/`。
下一步按计划先做 H2 的候选/稳健求解器对照，暂不因 H1 的 soft 结果直接引入时序。

## 41. 2026-08-16 H2：IRLS 候选池诊断与训练

H1 验证缓存上的零训练 oracle 显示新增几何候选确有信息：V3/V4 从 existing
`24.4635/20.0086` 降至 pairwise-IRLS `22.2094/18.0437 mm`。因此按预注册方案
生成 33-candidate cache（H76、confidence、IRLS 各 11 个组合），只改变候选池，
不改变 RUMPL checkpoint、HRNet 输入、损失和训练预算。

| H2 seed | V2 soft T=1.8 | V3 soft T=1.8 | V4 soft T=1.8 |
|---:|---:|---:|---:|
| 0 | 37.142 | 30.030 | 28.425 |
| 1 | 37.118 | 30.086 | 28.505 |
| mean | 37.130 | 30.058 | 28.465 |

复用预注册 V2 温度 `0.4` 后，均值为 **36.891/30.058/28.465 mm**。因此 IRLS
候选池的 oracle 增益没有被当前 utility scorer 学出来；相比 H1
`36.886/30.029/28.537`，仅 V4 有约 `0.072 mm` 改善，仍落后统一 E2-C2
soft-cal `38.700/29.486/27.274` 的 V3/V4。结论是当前瓶颈不是“候选数量不够”，
而是 utility loss/几何特征无法辨别新增候选；H2 不升级为主 baseline，也暂停继续
堆 IRLS 变体。

脚本与结果：
`OpenRUMPL_baseline_audit/launch_e2_card2_high_robust_cache_and_train_20260815.sh`、
`train_current_e2_robust_20260815.py`、
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/h2_robust_training_protocol_v1/`。

## 42. 2026-08-16 H3：B1 checkpoint 高学习率采样 A/B（运行中）

H2 之后不再继续堆几何候选。H3 从已完成的 300k B1 checkpoint 开始，固定 H76 与
GBT-aligned HRNet 输入，只比较视角 cardinality 采样：GPU0 `8:1:1` K2-heavy，
GPU1 `3:1:1` balanced；两者均为 `20E`、初始微调学习率 `5e-5`、同一 `10/15` 学习率
里程碑和 seed 0。两组结果分别评估 V2/V3/V4，不做跨模型拼接。

脚本：`OpenRUMPL_baseline_audit/launch_b1_highlr_sampling_ab_20260816.sh`。
输出：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/b1_highlr_sampling_ab/`。
启动时两组均确认加载：
`GBTCTRL_B1_CURRENTK_H76_123E_T1_seed0_20260815_2026-08-15_01-40-28/model_best.pth.tar`。
截至本记录，两组均处于第 0 轮训练，尚无最终精度；完成后以严格 table2 JSON 更新本节。

### 42.1 H3 最终结果

两组均完成 20E 微调，并按严格 H36M S9/S11、action-equal All-17 absolute MPJPE
评估：

| 训练线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| B1 + `8:1:1`, LR `5e-5` | **39.544** | **30.802** | **28.372** |
| B1 + `3:1:1`, LR `5e-5` | 41.884 | 31.389 | 28.790 |
| C2（`1e-5`, `8:1:1`） | 38.686 | 30.943 | 28.629 |
| E2-C2 soft-cal | 38.700 | **29.486** | **27.274** |

K2-heavy 相对同预算 balanced 线改善 `2.340/0.587/0.418 mm`，证明视角采样比例
仍是主要训练因素；相对 C2，V3/V4 额外改善 `0.141/0.257 mm`，但 V2 仍差
`0.858 mm`，且未超过 E2-C2 soft-cal 的多视角结果。因此 H3 K2-heavy 是下一轮
空间实验的候选生成器，但不能把它和 E2-C2 的其他列拼接。结果文件：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/b1_highlr_sampling_ab/`。

## 43. 2026-08-16 H4/H5：H3 K2-heavy 上的 utility A/B（运行中）

H3 K2-heavy 是当前较好的单模型生成器，因此重新导出其 22-candidate cache，比较：

- H4 standard：原 E2 direct-ranking + GHT expected-risk；
- H5 hinge：在同一损失上增加已注册的 identity-preserving hinge（`0.25`，V2×4）。

两条线各跑 seed 0/1，输入、checkpoint、候选池、训练预算和评估协议完全一致；四个
任务分摊到两张 GPU，禁止跨线拼接。输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_ab_protocol_v1/`。

截至记录时四个任务均在第 2/15 轮，最终结果待训练和温度校准完成后补充。

### 43.1 H4/H5 最终结果

温度 `V2=0.4,V3=V4=1.8` 校准后的严格结果（单位 mm）：

| 线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H4 standard，均值±std | 39.483±0.002 | **29.456±0.025** | **27.223±0.017** |
| H5 identity hinge，均值±std | **39.461±0.001** | 29.459±0.018 | 27.211±0.062 |
| E2-C2 soft-cal | 38.700 | 29.486 | 27.274 |

H4/H5 相对 H3 单模型 `39.544/30.802/28.372` 明显改善 V3/V4；相对 E2-C2，
H4 的 V3/V4 仅改善 `0.030/0.051 mm`，V2 退化约 `0.783 mm`。H5 相对 H4 的
变化只有 `−0.022/+0.002/−0.013 mm`，没有超过 seed 波动，因此 identity hinge
不能作为新的有效模块。H4 是当前更强的单模型多视角控制，但仍不能与其他模型的 V2
拼接为统一主结果。

结果目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_ab_protocol_v1/`。

## 44. H6：单模型 cardinality curriculum（2026-08-16，已完成）

H6 用同一个 B2 K=2 checkpoint 训练两条单模型控制线：CURRICULUM 在 20 个 epoch
内按 `8:1:1 → 3:1:1 → 3:2:2` 调整 V2/V3/V4 的采样概率，FIXED_MIXED 全程使用
`3:1:1`。除采样课程外，H76、HRNet 坐标/置信度、相机射线、学习率和严格评估均相同。
该实验不拼接不同模型的输出，也不引入视角专用 head，目的只是得到能兼顾 V2 和
V3/V4 的统一 spatial generator。

实现与日志：

- `OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`：增加可审计的
  `RUMPL_CURRICULUM_VIEW_WEIGHTS=start:w2,w3,w4;...` 开关；未设置时历史行为不变；
- `OpenRUMPL_baseline_audit/launch_h6_cardinality_curriculum_ab_20260816.sh`；
- `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/h6_cardinality_curriculum_ab/`。

两组均完成 20E 和 V2/V3/V4 全组合评估。严格 action-equal All-17 absolute MPJPE：

| 训练线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| CURRICULUM `8:1:1 → 3:1:1 → 3:2:2` | 36.939 | 31.631 | 30.484 |
| FIXED_MIXED `3:1:1` | **36.885** | **31.451** | **30.277** |

课程线相对固定混合线退化 `+0.054/+0.181/+0.207 mm`；固定混合线只是复现已有
HIGH-LR 控制结果，没有新增收益。该方向停止，不把课程作为论文创新点。

## 45. H7：几何候选判别器（已完成）

H7 针对 H2 暴露的 oracle gap：先在相同 H3 22-candidate cache 上隔离验证，E2 评分器加入
candidate-to-ray 的视角几何 token，并用视角交叉注意力判别候选可靠性；输入协议仍是
HRNet 坐标、置信度和相机射线，不使用 heatmap/图像特征。H7 保持原 E2 的候选生成、
训练/验证划分、holdout、温度和 action-equal All-17 absolute MPJPE 口径，只替换评分器。

代码：`OpenRUMPL_baseline_audit/train_h7_view_geometry_20260816.py`；启动：
`OpenRUMPL_baseline_audit/launch_h7_view_geometry_20260816.sh`；输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/h7_view_geometry_ab/`。

两 seed 已完成。严格结果（H3 22-candidate cache，action-equal All-17 absolute MPJPE）：

| 评分方式 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H3 baseline | 39.544 | 30.802 | 28.372 |
| H7 hard | 39.522 | 30.549 | 28.387 |
| H7 soft | 39.679 | **29.433** | **27.232** |

相对 H4 standard `39.483/29.456/27.223` 没有稳定的全视角收益：soft 仅 V3
改善 `0.023 mm`，V2/V4 分别退化 `0.196/0.009 mm`；hard 也未改善。因此
H7 不作为论文主线，后续不继续堆候选判别器复杂度；时序只在稳定的空间 baseline
之后单独报告，不能用来掩盖单帧空间退化。

## H11/H12 后续验证与失败经验（2026-08-17）

H11 直接验证了 Generalizable Human Pose Triangulation 的多候选学习评分思想。官方
whole-pose ScoreNN（H11A）得到 `39.544/33.363/32.119` mm；绝对候选误差评分器
（H11B/H11C）在同一候选池上得到约 `39.62/29.43/27.14` mm，说明候选池中确实有
更好的多视角解，但 V2 排序仍无收益。因此 H11 只能作为“候选池有 oracle gap、评分
存在视角数偏置”的诊断，不能把不同模型的 V2 与 V3/V4 拼成主结果。

H12 将公开 MTF 的 source-normalized pairwise view aggregation 适配到 RUMPL，保持
HRNet 坐标、置信度、相机射线和 RUMPL PFT/3D head 不变；仅替换 VFT。两条线完整
20E、同一输入 hash、同一 V2/V3/V4 评估：

| 线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H12A，不含置信度 | 46.813 | 33.403 | 29.798 |
| H12B，含 HRNet 置信度 | 41.378 | 31.266 | 28.140 |
| H3 控制 | 39.544 | 30.802 | 28.372 |

H12 不能作为有效模块。失败不是检测器或评估协议变化，而是直接替换 VFT 后绕过了
RUMPL 原有的 learned fusion query、高阶跨视角 Transformer 交互和 token 归一化路径；
置信度只缓解部分退化。若要继续，只允许
做保留 target token 的零初始化 residual-gated MTF A/B；若不能同时保持 V2 并改善
V3/V4，应关闭该方向。完整代码与日志：
`OpenRUMPL_baseline_audit/launch_h12_mtf_source_norm_ab_20260817.sh`、
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h12_mtf_source_norm_ab/`。

H13 对 H12 做了唯一结构修正：MTF pairwise message 以零初始化 gate 加到每个 view
token，再进入原 RUMPL fusion-token VFT，而不是替换 VFT。完整 20E 结果：

| 线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H13A residual，不含 confidence | 41.235 | 31.086 | 28.256 |
| H13B residual，含 confidence | 40.846 | 30.943 | 27.671 |
| H3 控制 | 39.544 | 30.802 | 28.372 |

H13B 的 V4 改善 `0.701 mm`，但 V2/V3 分别退化 `1.302/0.141 mm`；H13A 同样没有
统一收益。由此可写入论文的结论是：在当前坐标级 HRNet 输入和 RUMPL 主干中，MTF
pairwise 信息即使以 residual 形式注入，也不能解决少视角与多视角之间的 cardinality
trade-off；该方向停止，不作为主线。输出：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h13_mtf_source_norm_residual_ab/`。

## H14：C2 候选池上的官方绝对误差评分（2026-08-17）

为公平检验 Generalizable Human Pose Triangulation 的核心训练目标，而不再改变
RUMPL 主干，H14 在 C2 的 22-candidate frozen cache 上训练 absolute candidate-error
ScoreNN。输入仍是 HRNet 坐标、置信度和相机射线；候选生成、训练/验证划分、温度和
严格 H36M S9/S11 action-equal All-17 absolute MPJPE 均固定。与只训练 V3/V4 的
H11B 不同，H14 同时训练全部 V2/V3/V4 组合，避免跨模型拼接。

| 线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H14 frozen candidate baseline | 38.686 | 30.943 | 28.629 |
| H14 hard absolute score | 38.689 | 30.123 | 27.889 |
| H14 soft absolute score | 38.819 | 29.495 | 27.286 |
| E2-C2 soft-cal（当前统一空间参考） | **38.700** | **29.486** | **27.274** |

H14 soft 相对 E2-C2 为 `+0.119/+0.009/+0.012 mm`，没有形成新的统一最优，因此
不应把它写成有效改进。它只证明绝对候选评分可以在 V3/V4 复现候选选择收益，但当前
评分器还不能解决 V2 与多视角之间的 cardinality trade-off。完整代码和输出分别为：
`OpenRUMPL_baseline_audit/train_h76_pairwise_absolute_score_20260814.py`、
`OpenRUMPL_baseline_audit/launch_h14_c2_absolute_score_ab_20260817.sh`、
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h14_c2_absolute_pairwise/`。

## 阶段切换：ResNet-152 输入线

HRNet 坐标级空间参考暂定为 E2-C2 soft-cal `38.700/29.486/27.274 mm`。在 H4、H11、
H14 候选评分以及 H12/H13 MTF 替换/残差均未同时改善三列后，论文实验转入预先规划的
第二阶段：冻结 RUMPL 训练/评估协议，只替换二维前端为官方 Learnable Triangulation
ICCV-2019 ResNet-152。先建立 R0 纯坐标和 H76（三角化锚点、中心射线、Plücker）两条
ResNet 线，再用相同 table2 协议比较 V2/V3/V4。ResNet 导出严格包含官方去畸变、
annotation-box crop、384×384 transform、相机内参更新和 joint mapping；RUMPL 仍只
接收坐标、置信度和射线，不使用热图/图像特征。

执行脚本：`OpenRUMPL_baseline_audit/launch_resnet152_rumpl_gbt_screen_gpu1_20260817.sh`；
输出根目录：`/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/`。

## 2026-08-22：Global Joint-Query 后的严格时序修复结果

ResNet Global Joint-Query + E2 identity-hinge 的稠密时序前端已从原始 H36M
temporal GT PKL 重新生成，修复了旧缓存继承 HRNet 去畸变记录、导致相机畸变参数被
提前置零的问题。修正后的 T=9 中心帧空间基线为
`32.437/22.581/20.306 mm`，与稀疏单帧 identity-hinge
`32.319/22.558/20.272 mm` 基本对齐。

H18 使用 9 帧、stride 5、hidden 96、2 层、`lr=5e-5`，只在训练主体 holdout
选择 epoch 3，随后对 S9/S11 评估一次：

| ResNet 路线 | V2 | V3 | V4 |
|---|---:|---:|---:|
| matched dense T=1 center baseline | 32.437 | 22.581 | 20.306 |
| + H18，T=9 | **31.215** | **22.008** | **19.971** |
| 改善 | **1.222** | **0.572** | **0.335** |

三种视角数均改善，因此该时序模块保留用于 clean/occlusion 表格。正式结果：
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/h18_identity_hinge_v2_gtinput/result.json`。

HRNet C2→Query-only→全网 `5e-6` 的 U2 结果为
`38.487/30.913/28.676 mm`，相对 C2 改善 `0.199/0.030 mm`，但 V4 退化
`0.047 mm`，不作为统一最优。直接从 C2 初始化并联合训练的 A/B 已完成：A 为
`38.742/30.957/28.651 mm`，B 为 `38.921/30.870/28.712 mm`，均未同时改善
三列。highLR 初始化 A/B 分别为 `37.026/31.612/30.417 mm` 和
`36.905/31.465/30.280 mm`，也未超过其 source
`36.885/31.451/30.277 mm`。因此 HRNet 第一阶段不采用 Global Joint-Query，
最终保留 C2+E2-C2+H18 的 `37.704/29.231/27.219 mm`。完整冻结结论见
`/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`。

## 2026-08-23：Human3.6M-Occluded 官方生成器对齐与遮挡主结果

> **论文更新入口（2026-08-24）**：本节的最终表格、论文可用描述、消融逻辑和
> T=9 回填规则统一维护在
> `/home/lixiaob/cjy/STAGE2_H36M_OCC_RESULTS_FOR_PAPER_20260824.md`。后续生成论文时
> 应以该文件和 `final_occ23_table.json` 为准；本节早期 T=1 数字仅作空间消融，不能
> 替代完整的 Stage-2 T=9 主结果。

为了避免使用 GBT 未开源白方块协议作为唯一遮挡证据，本阶段增加了 Bragagnolo
Human3.6M-Occluded 官方生成器和 SkelSplat WACV 2026 Table 4 路线。模型仍只在
clean H36M 上训练；测试时不使用遮挡微调、遮挡验证集选 epoch、温度或 seed。

### 公开协议设置

遵循公开 Human3.6M-Occ 生成流程，在 H36M 图像上粘贴 Pascal VOC 物体，并固定
Occ-2/Occ-3 的遮挡视角数、物体尺度、随机种子和全部相机组合。所有方法使用同一批
遮挡输入；模型只在 clean H36M 上训练，遮挡测试集不参与权重或超参数选择。协议恢复
诊断和控制差值仅保留在内部审计文件，不在论文正文展开。

### 正式冻结结果

所有数字为 H36M S9/S11、All-17、absolute MPJPE、action-equal；V2/V3 平均全部
相机组合，V4 为四相机。E2 是两个 clean-trained scorer seed 的均值。

| input / frozen chain | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|
| HRNet C2 direct | 55.510 | 38.286 | 34.175 | 64.060 | 42.003 | 36.894 |
| HRNet C2 + E2 | 55.576 | 33.840 | **29.406** | 64.143 | 37.122 | **31.600** |
| ResNet GQ direct | 50.409 | 36.722 | 34.534 | 61.120 | 42.832 | 40.286 |
| **ResNet GQ + E2** | **50.511** | **29.336** | **23.383** | **61.267** | **34.067** | **26.092** |

论文主比较只能使用 V4，因为 SkelSplat Table 4 没有 V2/V3：

| Table-4 method, ResNet-152 or stated input | Occ-2 V4 | Occ-3 V4 |
|---|---:|---:|
| Algebraic Triangulation (paper) | 43.2 | 48.9 |
| RANSAC | 33.7 | 38.6 |
| AdaFuse | 27.9 | 31.2 |
| MV Pose Fusion | 33.4 | 36.7 |
| SkelSplat (MeTRAbs) | 29.6 | 31.1 |
| SkelSplat (ResNet-152) | 24.6 | 27.0 |
| **Ours ResNet GQ + E2, T=1 ablation** | **23.383** | **26.092** |

Ours T=1 比 SkelSplat ResNet-152 低 `1.217/0.908 mm`；最终主表以完整 T=9
baseline 替换该行，并增加 `T` 列。T=1 只保留在消融表。

### 对方法故事的意义

E2 在 clean 上主要改善 V3/V4，而在遮挡下作用显著放大。相对 matched direct
generator，ResNet V4 在 Occ-2/Occ-3 分别降低 `11.151/14.194 mm`，HRNet 降低
`4.769/5.294 mm`；ResNet 两 seed std 仅 `0.044/0.039 mm`。这支持论文故事：
Global Joint-Query/RUMPL 保留跨视角射线表征并生成直接姿态，E2 将直接候选和
confidence-triangulation 候选作为同一关节的多种几何解释，用 set scorer 预测效用并
soft fuse；当部分视角被污染时，候选差异和 oracle gap 增大，因此 E2 的鲁棒收益远大于
clean 主表的 0.x mm。

相对完全相同链的 clean E2 V4，ResNet 从 `20.272` 上升到
`23.383/26.092 mm`，退化仅 `+3.111/+5.820 mm`；HRNet 从 `27.274` 上升到
`29.406/31.600 mm`，退化 `+2.132/+4.326 mm`。这组同链 clean→occlusion
退化比跨论文绝对排名更严格，因为输入模型、3D 权重、scorer 和指标完全一致。

分层结果也支持该解释。ResNet E2 的 Occ-2 V2 在子集中含 `0/1/2` 个遮挡视角时为
`32.420/50.921/66.961 mm`；Occ-3 V3 含 `2/3` 个遮挡视角时为
`32.360/39.189 mm`。结果随真实污染视角数单调变差，并非某个固定相机带来的偶然值。

关键代码与结果：

- 数据适配：`OpenRUMPL_baseline_audit/generate_h36m_occ_official_adapter_20260823.py`；
- 前端：`OpenRUMPL_baseline_audit/launch_h36m_occ_official_frontends_20260823.sh`；
- 冻结 3D 链：`OpenRUMPL_baseline_audit/launch_h36m_occ_official_spatial_eval_20260823.sh`；
- 遮挡分层：`OpenRUMPL_baseline_audit/evaluate_e2_occlusion_stratified_20260823.py`；
- Occ-2：`/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ2/eval/`；
- Occ-3：`/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ3/eval/`；
- 完整审计：`OpenRUMPL_baseline_audit/OCCLUSION_STAGE_PREPARATION_20260822.md`。

### Occ-3-Hard 的协议置信度

Hard 的论文 Algebraic control 为 `120.4 mm`。公开生成器 `0.5--1.0` 在我们子集上
为 `229.934 mm`；有限候选 `0.3--0.8 / 0.4--0.75 / 0.4--0.78` 分别为
`203.958 / 111.351 / 337.885 mm`。最大遮挡物会触发离散式离群误差，因此停止继续围绕
测试控制调上界，冻结最接近的 `0.4--0.75`，差 `-9.049 mm`。Hard 结果必须标成
**approximate Hard control-aligned reconstruction**，其证据等级低于普通 Occ-2/Occ-3。

## 2026-08-24：官方强遮挡协议的 V2/V3/V4 少视角补充实验

为给少视角遮挡结论增加一个完全由公开代码定义、且确实包含 V2/V3/V4 的协议，
补充运行 Bragagnolo 等人 *Multi-view Pose Fusion for Occlusion-Aware 3D Human
Pose Estimation* 随附的 Human3.6M-Occluded 生成器。该实验与上一节的
SkelSplat control-aligned ordinary Occ-2/Occ-3 结果是两个独立协议，不能混表：

- 官方生成器 commit `4a2625805d6979e92283ee9c93ee476a1d8c6a82`；
- S9/S11 共 2021 个同步组，沿用 damaged-sequence filter；
- 每组随机遮挡 4 个视角中的 3 个，每个遮挡视角粘贴 2 个 Pascal VOC 物体；
- 物体尺度为人体框短边的 `0.5--1.0`，随机种子 `42`；
- V2/V3/V4 穷举 `6/4/1` 个相机子集；
- action-equal All-17 absolute MPJPE，无对齐；所有学习模型只在 clean H36M
  训练，不做遮挡微调或遮挡集选参。

同一 2D 坐标、置信度和相机参数下的 T=1 消融结果为：

> 口径更正：下表只用于分解 direct generator、几何控制和 E2 的作用，不能作为
> 第二阶段最终 Ours。第二阶段主结果必须继承第一阶段冻结的完整 baseline：ResNet
> `GQ-RUMPL-E2-H18` 和 HRNet `C2-E2-H18`。由于本次官方缓存是 stride-64，尚不能
> 构造 T=9；完整主方法需要在同一官方生成器的 dense stride-5 版本上另行评估。

| ResNet-152 matched coordinates | V2 | V3 | V4 |
|---|---:|---:|---:|
| Algebraic confidence DLT | 1249.311 | 240.155 | 203.438 |
| Confidence ray intersection | 259.889 | 143.507 | 129.557 |
| IRLS confidence rays | 259.882 | 118.661 | 87.111 |
| AdaFuse public RANSAC | 1249.732 | 179.786 | 111.966 |
| Frozen direct generator | **208.422** | 132.789 | 124.119 |
| **Ours E2** | 209.345 | **100.427** | **64.486** |

| HRNet-W32 matched coordinates | V2 | V3 | V4 |
|---|---:|---:|---:|
| Algebraic confidence DLT | 1700.274 | 578.572 | 295.643 |
| Confidence ray intersection | 428.720 | 160.219 | 118.136 |
| IRLS confidence rays | 426.362 | 143.910 | 86.799 |
| AdaFuse public RANSAC | 3183.242 | 460.475 | 210.466 |
| Frozen direct generator | **298.312** | 138.207 | 106.724 |
| **Ours E2** | 299.117 | **122.752** | **80.942** |

E2 相对 direct generator 在 ResNet V3/V4 改善 `32.362/59.633 mm`，在
HRNet V3/V4 改善 `15.455/25.782 mm`；相对最强的非 E2 matched control，
ResNet 仍改善 `18.234/22.626 mm`，HRNet 改善 `15.455/5.857 mm`。但 V2
分别退化 `0.923/0.804 mm`。这是后续方法需要正面解决的边界：3/4 原始视角被
遮挡时，两视角子集可能两条射线同时受污染，缺少第三条冗余射线进行离群判别，
所以候选重评分在 V3/V4 非常有效，却不能凭空恢复 V2 中缺失的观测信息。

分层统计进一步排除了“某个固定坏相机”的解释。ResNet E2 在 V2 子集中含
`1/2` 与 `2/2` 个遮挡视角时分别为 `161.016/257.675 mm`，HRNet 为
`215.084/383.150 mm`；V3 子集中含 `2/3` 与 `3/3` 个遮挡视角时 ResNet 为
`88.270/136.899 mm`，HRNet 为 `105.571/174.295 mm`。只要子集中仍有 clean
view，误差就明显更低；V2 无 clean view 时才是当前核心失败模式。

Pose Fusion Table 4 仅给出精确 V4：RANSAC `80.7`、Algebraic `127.4`、
AdaFuse `41.1`、TransFusion `96.5`、Pose Fusion `37.8 mm`。我们的 ResNet E2
外部参考上超过 RANSAC、Algebraic、TransFusion，但没有超过 AdaFuse 和 Pose
Fusion。由于其各自使用不同图像前端，这些数字只放在 separate external reference
table，不声明 matched-input SOTA。Pose Fusion Figure 6 的 V2/V3 只有曲线，不能
伪造为精确表格数据。

本表保持单帧。H18 需要 stride-5 的连续 T=9 缓存，而当前官方强遮挡缓存是
stride-64 且遮挡随机独立；若做时序遮挡实验，必须另行生成并审计稠密序列，不能把
两种口径混在主表中。

审计文件和结果：

- `OpenRUMPL_baseline_audit/POSEFUSION_OFFICIAL_OCC_V234_20260824.md`；
- `OpenRUMPL_baseline_audit/collect_posefusion_occ_v234_table_20260824.py`；
- `/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3/eval/posefusion_v234_table.json`；
- `/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3/eval/posefusion_v234_table.md`。

### GBT 中 Algebraic Triangulation 数字的来源

GBT 的 `Alg. Tri. [10]` 指向 Iskakov 等人的 ICCV 2019 *Learnable
Triangulation of Human Pose*。原论文报告了 clean filtered H36M 上学习置信度的
Algebraic 分支，在可用全部相机下 absolute MPJPE 为 `19.2 mm`；它没有报告 GBT
所列的全部 V2/V3 相机组合，也没有 GBT 的白方块 H36M-Occl 实验。

GBT 正文明确说明：ResNet 行使用 Iskakov 公开的 H36M-finetuned ResNet-152 输出，
再应用 Algebraic Triangulation；HRNet 行则把 GBT 自己提取的 HRNet 坐标送入同一
三角化基线；所有视角数都平均全部相机组合。因此 Table I 的 V2/V3 和 Table II 的
全部遮挡数字属于 GBT 自己重跑的结果，不是从2019论文直接摘录。GBT clean V4 的
`19.1 mm` 与原论文 `19.2 mm` 接近，也支持这一实现来源。

Iskakov 的网络、checkpoint 和代数求解器有公开代码，我们已在本地保存并使用；但
GBT 没有公开白方块尺寸、精确放置实现和随机种子，所以其 Table II 仍不能 byte-level
复现。论文中只能称 GBT-reported reference，不能把我们猜测生成的遮挡集称为 GBT
官方测试集。

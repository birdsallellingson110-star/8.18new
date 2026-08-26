# OpenRUMPL 多视角单人 3D 姿态项目完整交接（2026-08-24）

> 用途：把本文件直接发给新的 Codex 对话。新对话应先完整阅读本文，再检查实时进程和输出，不要从头重新猜测、下载、复现或重复已经失败的实验。
>
> 状态快照时间：2026-08-24 19:53 CST。训练会继续推进，因此本文中的“正在运行”状态只代表该时刻；路径、方法、历史冻结结果和实验原则仍长期有效。

---

## 0. 新 Codex 接手后的第一组动作

先执行只读检查，不要立刻重启训练，也不要占用 GPU1：

```bash
cd /home/lixiaob/cjy

date '+%F %T %Z'
nvidia-smi
tmux ls
pgrep -af 'CAMGEN_STAGE1|launch_stage1_canonical|train_rumpl.py|train_current_e2|train_e2_clean_temporal'

tail -n 30 /mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/hrnet/generator/train.log

find /mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend \
  -maxdepth 5 -type f \
  \( -name 'table2.json' -o -name 'result.json' -o -name 'COMPLETED' -o -name 'STAGE1_COMPLETED' \) \
  -print | sort
```

当前预期 tmux：

- `cjy_camgen_stage1`：GPU0，先训练 HRNet canonical generator，再排队训练 ResNet generator。
- `cjy_camgen_stage1_downstream`：等待两个 generator 完成，随后自动跑 22 候选、E2 双 seed 和 camera-independent H18。
- GPU1 在快照时属于 `/home/lixiaob/czj` 的 `czj_raymix_coco17_full6000_20260824`，**不是本项目任务，禁止停止或抢占**。

如果上述两个 `cjy_camgen_stage1*` 会话仍在，绝对不要重复启动。只有确认 tmux 不存在、对应进程不存在，并检查输出后，才能决定恢复方式。

---

## 1. 项目总目标和不可改变的约束

### 1.1 总目标

做一篇可发表的、单人多视角 3D 姿态估计论文。实验叙事大体仿照 GBT（Geometry-Biased Transformer）：

1. Human3.6M clean，HRNet 和 ResNet-152 两种 2D 输入，报告 2/3/4 视角。
2. 遮挡条件下报告 2/3/4 视角，证明少视角和证据受损时的鲁棒性。
3. CMU Panoptic 同域训练测试，报告 2/4/5/6/8 视角。
4. 用真实 CMU 训练模型零样本测试 H36M，证明跨相机、跨数据集泛化。
5. 做完整消融，解释 RUMPL、候选效用融合、时序和相机无关规范化各自的贡献。

旧的粗略门槛是 V2 < 40 mm、V4 < 30 mm；现在目标已升级为 **MPJPE 越低越好，并尽可能超过更多公开方法**。不能达到某个目标就随意转方向，也不能通过改变输入协议获得不公平优势。

### 1.2 输入和公平性约束

论文主线是坐标级协议：3D 网络只接收

- 每个视角的 2D 关节坐标；
- 2D 置信度；
- 相机参数构造的射线/几何量。

3D 网络不接收 RGB、热图或图像特征。2D 检测器冻结。主表同时报告 HRNet 与 ResNet-152 前端，使其可与 GBT 等相同输入类别的方法比较。

可以分别为 HRNet 和 ResNet 使用同一网络结构下不同的训练参数，因为两种检测器误差分布不同；但不能为了某个视角数训练专门模型再拼接成一行，也不能悄悄换输入。

### 1.3 用户明确否决或限制的事项

- 不做蒸馏，老师不认可。
- 不再使用 A1D/热图作为论文主线；A1D 可保留为历史内部结果，但不能与坐标级方法混为公平比较。
- 不使用 CPN 路线；已验证分布不适配，V2 明显恶化。
- 不允许删除 AMASS 数据、RUMPL baseline 权重、Python 环境；也不要动其他人的目录和任务。
- 正式跨数据集只计划 **真实 CMU 训练 → H36M 测试**，不要把 H36M→CMU当正式主线。
- GBT 没有开源代码，不能声称“严格复现 GBT”；只能称按论文公开细节对齐输入/协议，并引用其报告数值。

---

## 2. 正式评估口径

### 2.1 Human3.6M clean

- 训练 subjects：S1/S5/S6/S7/S8。
- 测试 subjects：S9/S11。
- 4 个标准相机。
- V2：全部 6 个两相机组合取平均。
- V3：全部 4 个三相机组合取平均。
- V4：全部 4 相机。
- 指标：absolute MPJPE、All-17 joints、action-equal，不做 Procrustes 对齐。
- T=1 和 T=9 必须分开标注。

任何只测试单一相机对、单个 holdout、S8 内部验证、某个 early epoch 的数值都不能当论文主表结果。

### 2.2 H36M-Occluded

采用公开 Human3.6M-Occluded 的 VOC 前景物体粘贴协议：

- Occ-2：4 个视角中选择 2 个视角遮挡。
- Occ-3：4 个视角中选择 3 个视角遮挡。
- 每个被选中视角贴 2 个 VOC 物体。
- 尺度 0.2–0.5，seed 42。
- 模型只在 clean H36M 训练，遮挡测试为 zero-shot。
- 同样报告所有 V2/V3/V4 组合。

这个协议比根据 GBT 一句话自行重建白色方块更有可复查性。论文中无需反复解释本地 Alg. Tri. 与公开数字的小差异，但内部必须保留验证记录。

### 2.3 时序口径

历史 H18 是 T=9 居中窗口：过去 4 帧 + 当前帧 + 未来 4 帧，stride=5。它不是因果推理，但这不是作弊，只是离线时序设定；论文必须明确写 centered/offline T=9，不能写 causal。

新 camera-independent H18 仍保持这一窗口，目的是在 clean 数据不退化并在遮挡条件明显获益。

---

## 3. 当前最终技术路线

当前最合理的论文方法不是堆叠三个完整模型，而是一个有明确分工的模块化系统：

```text
冻结 2D 检测器（HRNet 或 ResNet-152 坐标/置信度）
        │
        ├── 相机内外参 → 世界射线 / Plücker 几何
        │
        ▼
SE(3)-canonical RUMPL generator
  - 置信度加权射线交会得到 pelvis 原点
  - shoulder 与 pelvis→neck 定义身体规范坐标轴
  - 完整 RUMPL VFT/PFT 在规范坐标内工作
  - 预测后逆变换回世界坐标
        │
        ├── 11 个生成器候选
        └── 11 个置信度三角化候选
                    ▼
canonical E2 候选效用评分（22 candidates）
  - Set Transformer / 逐关节效用
  - identity protection
  - 不用绝对 root、H36M mean/std 和相机 ID
                    ▼
camera-independent H18，T=9
  - 中心帧身体坐标规范化整段窗口
  - 输入姿态、相对运动和融合信息
  - 预测规范坐标残差，再旋回世界系
                    ▼
              绝对世界坐标 3D pose
```

### 3.1 保留了 RUMPL 的哪些优秀部分

RUMPL 不是被完全替换：

- 保留其从相机射线出发的坐标级表示；
- 保留 VFT（view fusion transformer）和 PFT（pose fusion transformer）；
- 保留其对任意视角数和相机排列的 set-style 融合框架；
- 保留生成器作为多假设的基础。

新的 canonical frame 是对 RUMPL 输入/输出坐标系的结构性修复，不是另起一个体素或图像模型。

### 3.2 H76 / 三角化锚点 / centered Plücker

旧 H76 的有效部分：

- confidence-weighted triangulation anchor 提供稳定绝对位置；
- 射线原点相对 anchor 中心化；
- 使用 Plücker 射线表达；
- 保留 RUMPL VFT/PFT。

但旧 H76 仍把 canonical 化后的向量直接送进普通 XYZ MLP/Transformer，存在世界坐标旋转依赖。新的 body-canonical generator 修复了这个问题。

### 3.3 Global Joint-Query

Global Joint-Query 对 ResNet 前端非常有效，尤其是 V2；它在全局关节—视角 token 上做联合更新，让两个视角时不必只靠局部逐关节融合。

它对 HRNet 曾失败/退化，说明它是检测器分布相关的优化。因此当前统一架构允许：

- ResNet：启用 Global Joint-Query。
- HRNet：关闭 Query，使用 C2/K2-heavy generator。

不要为了“完全同参数”强行让 HRNet 使用失败的 Query。

### 3.4 E2 22 候选效用融合

E2 不等于时序。它对单帧多个 3D 候选逐关节评分：

- 11 个 RUMPL generator hypotheses；
- 11 个 confidence triangulation hypotheses；
- 逐关节 soft utility / candidate weighting；
- identity hinge 约束输出不要无故破坏强基线。

E2 在 3/4 视角提升显著，在 2 视角提升很小甚至几乎持平。原因不是“2 视角只有一个候选”，而是候选来自不同生成/三角化方案；但两视角本身几何冗余低，候选之间往往共享同一个系统性 2D 误差，评分器无法凭空恢复缺失信息。3/4 视角有更多交叉验证证据，因此效用评分更能排除坏候选/坏视角。

### 3.5 H18 时序

H18 是轻量残差时序模块，不是完整 MixSTE 堆叠：

- T=9、stride 5；
- hidden 96、2 layers；
- 小学习率 5e-5；
- residual scale 0.10 m；
- 目标是在 clean 上不退化/略升，在遮挡下利用相邻帧补证据。

历史大量时序实验只提升 0.1–0.2 mm甚至退化。最终 H18 的稳定收益也不大，但它提供遮挡叙事。不能把它夸大为主要 clean 精度创新；应描述为“不确定/缺失证据时的轻量状态修正”。

### 3.6 新的相机无关 canonical 处理

这是 2026-08-24 已实现并正在正式重训的关键结构修复。

对每一帧，利用观测射线和置信度构造身体坐标：

1. 置信度加权的等变射线交会得到 pelvis/anchor `o`；
2. 左右肩方向构造横轴 `e_x`；
3. pelvis→neck 方向去除横轴分量后构造 `e_y`；
4. `e_z = e_x × e_y`；
5. 对世界点/射线应用 `x_c = B^T(x_w-o)`；
6. RUMPL/E2/H18 在 canonical 坐标处理；
7. 输出 `x_w = B x_c + o`。

保持米制尺度，不做会破坏绝对 MPJPE 的尺度归一化。

结构审计使用刚体变换：

```text
d' = R d
o' = R o + t
y' = R y + t
```

理想模型应满足 `f(Rd, Ro+t) = R f(d,o)+t`。

---

## 4. 已冻结的历史 Stage-1 clean 结果

权威记录：

`/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`

这些是旧 world-frame 模型的已完成正式 S9/S11 数据。新的 canonical 结果未完成前，不得覆盖它们。

| 2D 前端 | 模型 | T | V2 | V3 | V4 |
|---|---|---:|---:|---:|---:|
| ResNet-152 | GQ-RUMPL | 1 | 32.312 | 25.101 | 23.536 |
| ResNet-152 | + E2 identity | 1 | 32.319 | 22.558 | 20.272 |
| ResNet-152 | + H18 | 9 | **31.215** | **22.008** | **19.971** |
| HRNet | C2 generator | 1 | 38.686 | 30.943 | 28.629 |
| HRNet | + E2 | 1 | 38.700 | 29.486 | 27.274 |
| HRNet | + H18 | 9 | **37.704** | **29.231** | **27.219** |

GBT 论文报告参考：

| GBT 报告 | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet-152† | 29.9 | 24.4 | 22.7 |
| HRNet | 36.8 | 30.4 | 26.0 |

正确解读：

- 我们旧 ResNet 最终 T=9 在 V3/V4 明显优于 GBT 报告，V2 仍差 1.315 mm。
- 我们旧 HRNet 在 V3 优于 GBT，V2/V4 仍差。
- 不能说“全面超过 GBT”。
- 两种方法的时序/实现细节并非严格相同，比较时要清楚标注输入和 T。

关键消融变化：

- ResNet Global Joint-Query：旧 H76 约 41.470 → 32.312（V2），是两视角最大来源；V3/V4只约 1 mm收益。
- ResNet E2：32.312/25.101/23.536 → 32.319/22.558/20.272。
- HRNet E2：38.686/30.943/28.629 → 38.700/29.486/27.274。
- ResNet H18：E2 T=1 → T=9 约提升 1.104/0.550/0.301 mm（以冻结表为准，其他 matched 统计约 1.222/0.572/0.335，引用时不要混口径）。
- HRNet H18：约提升 0.996/0.255/0.055 mm（不同 matched cache 曾统计 1.123/0.407/0.153；论文只引用最终冻结表）。
- identity hinge 本身通常只有约 0.01–0.09 mm，不能作为主要创新点。

---

## 5. 已冻结的历史 Stage-2 遮挡结果

权威记录：

`/home/lixiaob/cjy/STAGE2_H36M_OCC_RESULTS_FOR_PAPER_20260824.md`

当前已冻结的主要是旧 world-frame T=1 空间结果；新的 canonical Stage-1 完成后应再决定是否全量重跑 Stage-2。

### 5.1 HRNet

| 设置 | 模型 | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| Occ-2 | direct | 55.510 | 38.286 | 34.175 |
| Occ-2 | +E2 | 55.576 | 33.840 | 29.406 |
| Occ-3 | direct | 64.060 | 42.003 | 36.894 |
| Occ-3 | +E2 | 64.143 | 37.122 | 31.600 |

### 5.2 ResNet-152

| 设置 | 模型 | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| Occ-2 | direct | 50.409 | 36.722 | 34.534 |
| Occ-2 | +E2 | 50.511 | 29.336 | 23.383 |
| Occ-3 | direct | 61.120 | 42.832 | 40.286 |
| Occ-3 | +E2 | 61.267 | 34.067 | 26.092 |

### 5.3 本地 Alg. Tri. 协议校验

| 输入 | 设置 | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| ResNet | Occ-2 | 153.150 | 45.657 | 41.855 |
| ResNet | Occ-3 | 140.948 | 68.883 | 50.716 |
| HRNet | Occ-2 | 249.877 | 68.385 | 55.986 |
| HRNet | Occ-3 | 360.170 | 110.317 | 60.622 |

公开 V4 参考：

- AdaFuse ResNet-152：Occ-2 27.9，Occ-3 31.2。
- SkelSplat ResNet-152：Occ-2 24.6，Occ-3 27.0。

旧 ResNet E2 T=1 的 V4 已达到 23.383/26.092，优于上述公开 V4 数值。论文比较表必须注明输入前端、是否使用 T=9，以及对方是否只报告 V4。

预计最终聚合文件（接手时检查是否已生成）：

```text
/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.json
/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.md
```

快照时这两个文件尚未发现。

---

## 6. 2026-08-24 相机/数据集依赖审计与修复

总记录：

`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/CAMERA_DATASET_DEPENDENCE_AUDIT_AND_FIX_20260824.md`

### 6.1 旧 generator 的 SE(3) 依赖

结果：

`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/h76_c2_generator_64.json`

| 变换 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 平移 x=1m | 7.48 | 2.79 | 2.78 |
| yaw 37° | 65.59 | 30.34 | 20.78 |
| yaw 90° | 86.95 | 43.65 | 39.79 |
| 任意完整旋转 | 92.60 | 46.76 | 39.20 |

任意完整旋转会使实际 MPJPE 额外恶化约 56.28/18.30/12.24 mm。说明旧生成器严重依赖 H36M 世界轴，不能直接宣称相机布局/坐标系泛化。

根因：

1. 原三角化求解 `A + λI` 的正则中心固定在世界原点，破坏平移等变性。
2. 学到的 `tri_anchor_gate≈0.9973`，所以 1m 平移残留约 `(1-gate)×1m≈2.7mm`。
3. anchor-centered Plücker 虽缓解平移，但普通 XYZ MLP/Transformer 不具备 SO(3) 等变性。

### 6.2 旧 E2 的依赖

结果：

`/mnt/data/cjyoutput/camera_generalization_20260824/equivariance_audit/e2_identity_hinge_128.json`

| 变换 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 平移 x=1m | 0.325 | 0.631 | 0.443 |
| yaw 90° | 1.779 | 4.130 | 3.850 |
| 任意旋转 | 4.945 | 4.009 | 3.195 |

V4 yaw90 下 hard candidate selection flip rate 达 58.5%。根因是旧 E2 使用 H36M mean/std、绝对 root 和轴向 XYZ 特征。

### 6.3 已实现的 canonical generator

修改文件：

`/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`

关键函数：

- `equivariant_body_canonicalize_rays`
- `body_canonical_pose_to_world`

环境变量：

```bash
export RUMPL_BODY_CANONICAL_FRAME=1
export RUMPL_BODY_CANONICAL_REG=1e-4
```

默认关闭，旧实验路径保持不变。正式新训练脚本会显式打开。

### 6.4 已实现的 canonical E2

修改/新增文件：

```text
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_camera_independent_22c_20260824.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/evaluate_e2_c2_calibrated_20260815.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/build_e2_base_scores_20260818.py
```

关键选项：

```text
--canonical-geometry
--fixed-metric-normalization
```

checkpoint 会保存 canonical flags，后续评分/导出脚本已经同步读取，避免训练规范化、评估却走旧特征。

### 6.5 已实现的 camera-independent H18

文件：

`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py`

关键选项：

```text
--camera-independent
```

实现细节：

- 用中心帧 pelvis/shoulder/torso 轴规范化整个 T=9 窗口；
- 不把世界绝对 root 当输入，改用中心身体坐标下的相对 root motion；
- 跳过 11 个 H36M camera-subset task embedding；
- canonical 空间预测残差，旋回世界坐标；
- 保持 root protection。

随机非零残差单元测试的 SE(3) 等变误差为 `0.000047 mm`。

### 6.6 小规模可学习性证据（不是论文主表）

launcher：

`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_body_canonical_hrnet_learnability_20260824.sh`

输出：

`/mnt/data/cjyoutput/camera_generalization_20260824/learnability_hrnet_small`

同一 HRNet 输入、2048 train、1024 val、5 epochs：

- world-frame 最后 absolute MPJPE：51.82 mm；
- body-canonical：47.22 mm。

这只证明 canonical 路径能训练且方向合理，不能写成正式性能结果。

训练后等变审计：

`/mnt/data/cjyoutput/camera_generalization_20260824/postlearn_audit`

- 完整旋转+平移等变误差：V2 0.066 mm，V3 0.0022 mm，V4 0.0017 mm。
- 16 帧随机相机 SMPL：
  - world：117.45/73.94/66.30；
  - canonical：114.98/69.19/59.88；
  - 改善：2.47/4.75/6.42 mm。

仍然只是少量数据验证。

### 6.7 当前能说和不能说的结论

可以说：

- 已发现并修复旧网络对任意世界坐标平移/旋转的结构依赖；
- 新 canonical 路径在数值测试中接近 SE(3) 等变；
- 少量随机相机 SMPL 验证显示方向有效。

不能说：

- “模型已经与数据集无关”；
- “已经证明真实跨数据集泛化”；
- “CMU→H36M 已完成”。

数据分布依赖仍可能来自人体骨架、动作、2D 检测器误差、置信度分布。最终必须用真实 CMU 训练→H36M 测试验证。

早期 SMPL 审计文件：

`/mnt/data/cjyoutput/camera_generalization_20260824/smpl_random_camera_audit/smpl16_quality4_current_h76_e2.json`

它显示除了坐标系依赖外还有分布依赖。不要用“修了坐标系”代替跨数据集实验。

---

## 7. 当前正在运行的正式 Stage-1 canonical 双前端实验

总输出：

`/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend`

### 7.1 Generator launcher

```text
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_generators_20260824.sh
```

tmux：`cjy_camgen_stage1`

顺序在 GPU0 串行执行：

1. HRNet canonical H76：20 epochs，V2/V3/V4 采样比 8:1:1，不启用 Query。
2. ResNet canonical H76 + Global Joint-Query：20 epochs，前 8 epochs 固定 V2，之后 3:1:1。
3. 每个前端自动对 S9/S11 的所有 V2/V3/V4 组合做正式评估，输出 `table2.json`。

快照状态：HRNet 正在 Epoch 11（即第 12 个 epoch），日志约 0.10 秒/batch、2438 batches/epoch；训练/验证分别加载 78,047/2,021 个 group。GPU0 显存约 1.8 GB。每轮日志中的约 29–31 mm validation 只是训练脚本内部小型 sanity 评估，不是正式 all-combinations Table 2。

日志：

`/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/hrnet/generator/train.log`

### 7.2 Downstream launcher

```text
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_downstream_20260824.sh
```

tmux：`cjy_camgen_stage1_downstream`

它等待：

`/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/GENERATORS_COMPLETED`

之后对 HRNet 和 ResNet 依次完成：

1. 导出 11 个 generator candidates；
2. 追加 11 个 confidence triangulation candidates，得到 22c；
3. canonical E2 identity-hinge，seed0/seed1：10 epoch direct + 5 epoch GHT-risk；
4. V2 temperature=0.4，V3/V4=1.8；
5. 构造时序 fused cache；
6. camera-independent H18：T=9、stride5、12 epochs。

输出目录：

```text
.../hrnet/canonical_e2
.../hrnet/canonical_h18
.../resnet152/canonical_e2
.../resnet152/canonical_h18
```

### 7.3 只在会话确实不存在时启动

```bash
cd /home/lixiaob/cjy

tmux new-session -d -s cjy_camgen_stage1 \
  'cd /home/lixiaob/cjy && bash OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_generators_20260824.sh'

tmux new-session -d -s cjy_camgen_stage1_downstream \
  'cd /home/lixiaob/cjy && bash OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_downstream_20260824.sh'
```

注意：launcher 会依据 `checkpoint.txt`、`table2.json`、`result.json` 跳过已完成阶段；但 generator 若在训练中途死掉且还没有 `checkpoint.txt`，直接重跑可能从头开始。此时先检查模型目录中的 `checkpoint.pth.tar` / `model_best.pth.tar` 和训练器 resume 机制，不要同时启动重复进程。

### 7.4 监控命令

```bash
tmux attach -t cjy_camgen_stage1
# 离开 tmux：Ctrl-b 后按 d

tmux attach -t cjy_camgen_stage1_downstream

tail -f /mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/hrnet/generator/train.log

watch -n 2 nvidia-smi
```

完成标志：

```text
.../GENERATORS_COMPLETED
.../hrnet/STAGE1_COMPLETED
.../resnet152/STAGE1_COMPLETED
.../STAGE1_COMPLETED
```

### 7.5 正式结果读取

Generator T=1：

```text
.../<frontend>/generator/eval/V2/table2.json
.../<frontend>/generator/eval/V3/table2.json
.../<frontend>/generator/eval/V4/table2.json
```

E2：

```text
.../<frontend>/canonical_e2/identity_hinge/seed0/result.json
.../<frontend>/canonical_e2/identity_hinge/seed1/result.json
.../<frontend>/canonical_e2/identity_hinge/calibrated_v2t04.json
```

H18：

```text
.../<frontend>/canonical_h18/model/result.json
```

新结果出来后必须整理成与第 4 节完全一致的六列表，并同时保留 old world-frame 对照，回答三个问题：

1. canonical 是否不损失 clean H36M；
2. 是否在 V2/V3/V4 中至少保持旧最好结果；
3. 等变性提升是否与跨相机/遮挡泛化提升一致。

---

## 8. 数据、模型、代码和输出位置

### 8.1 Python 和代码仓库

```text
Python: /home/lixiaob/cjy/rumpl_venv310/bin/python
核心仓库: /home/lixiaob/cjy/OpenRUMPL/RUMPL
实验/审计脚本: /home/lixiaob/cjy/OpenRUMPL_baseline_audit
基础配置: /mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
模型输出根: /mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
```

运行 Python 脚本通常需要：

```bash
export PYTHONPATH=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
export CUDA_VISIBLE_DEVICES=0
```

### 8.2 H36M HRNet 前端

```text
type: gbt_yolox_x_score001_fallback_legswap
train: /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
validation: /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
temporal validation: /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl
```

HRNet 的 `yolox_x...` 名称说明检测框来自本地 YOLOX-X 对齐线，2D 关键点仍是 HRNet 输出。它不是热图输入到 3D 网络。

### 8.3 H36M ResNet-152 前端

```text
type: res152_lt_alg_undistorted_annbox
train: /mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/train/h36m_train_res152.pkl
validation: /mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/validation/h36m_validation_res152.pkl
temporal validation: /mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/validation/h36m_validation_res152_temporal.pkl
```

这是 LT/Alg. Tri. 风格去畸变、annotation bbox 的 ResNet-152 坐标输入。

### 8.4 CMU 数据现状

真实 CMU 本地只有有限验证：

```text
/mnt/data/cjydata/cmu_rumpl_official_eval_20260722
```

主要是 pose5/pose6 验证，不足以作为真实 CMU 正式训练集。

下面目录不是“真实 CMU 训练数据”：

```text
/mnt/data/cjydata/mhp_workspace/paper_single_cmu
```

它约 46 GB，是 AMASS/SMPL 投影到 CMU 相机的合成数据。可用于 RUMPL 复现/预训练/诊断，但论文不得写成 real CMU train。

正式 CMU→H36M 需要：

- 选定与论文一致的 CMU Panoptic 单人序列；
- 下载足够的 HD 图像或视频和校准；
- 对图像运行与主表一致的冻结 HRNet/ResNet 前端；
- 建立真实时序 PKL；
- 训练 canonical 模型；
- 不在 H36M 微调，直接测试 H36M。

CMU 数据很大且官方单连接曾只有几十 KiB/s。之前暂停 CMU 全量下载，不要未经确认再次盲目下载全部 9 序列×30 视角。可以按正式实验需要选定视角/序列并行下载。

### 8.5 H36M-Occluded

主要记录和生成/评估脚本集中在：

```text
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/POSEFUSION_OCC23_DENSE_FINAL_PLAN_RESULTS_20260824.md
/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
```

---

## 9. 代码改动清单（接手时不要丢失）

当前仓库是 dirty worktree，用户和过去实验已有大量修改。**禁止 git reset --hard、checkout 覆盖或清理未知文件。**

本轮相机无关主线相关改动：

```text
/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py
/home/lixiaob/cjy/OpenRUMPL/RUMPL/run/train_rumpl.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/audit_camera_coordinate_equivariance_20260824.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/evaluate_smpl_random_camera_dependency_20260824.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_camera_independent_22c_20260824.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/evaluate_e2_c2_calibrated_20260815.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/build_e2_base_scores_20260818.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_body_canonical_hrnet_learnability_20260824.sh
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_generators_20260824.sh
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_downstream_20260824.sh
```

`run/train_rumpl.py` 新增 `--train-n-samples` 和 `--test-n-samples`，主要用于小样本 smoke test；正式 launcher 未裁剪样本。

---

## 10. 已经成功的经验

### 10.1 数据和评估管线已基本可信

过去 Human3.6M 复现差约 2 mm，经过 subject split、全部相机组合、action-equal、All-17、单位、去畸变、相机参数等长期核查，已经确认主要评估管线没有几十毫米级错误。GT 2D 下仍有误差是合理的，因为 RUMPL 不是解析纯三角化，它经过学习网络/射线融合；GT 2D 很低说明标定、同步、单位和射线构造基本正确，不表示网络必须绝对 0 mm。

### 10.2 更好 2D 不保证 V2 自动更好

ResNet 相比 HRNet 在 V3/V4 一度大幅改善，但 V2 反而变差。原因不是相机夹角改变，而是融合器在低冗余条件下对检测器置信度与系统误差分布失配。3/4 视角能通过冗余抵消误差；2 视角没有足够证据。Global Joint-Query 的成功说明瓶颈确实在融合器，而非只在 2D 输入。

### 10.3 Global Joint-Query 是 ResNet V2 的主要成功模块

它把 ResNet V2 从约 41.47 拉到 32.31 mm，同时保持 V3/V4。这是目前最明显的结构提升。它不是简单调温度或训练技巧。

### 10.4 E2 的价值是多视角负证据抑制

E2 对 clean V3/V4 和 Occ V3/V4 都有显著收益，尤其遮挡时提升可达数毫米至十余毫米。它适合写成“候选级、逐关节视角效用融合”，而不是说所有视角数都同等提升。

### 10.5 时序应服务于遮挡，不应夸大 clean 提升

H18 是众多时序失败中唯一较稳定的版本。它在 clean 上只小幅提升，但能为遮挡时恢复证据提供合理机制。论文中把 clean 不退化和 occlusion 增益一起展示。

### 10.6 必须先做坐标等变性再谈泛化

旧网络在任意旋转下几十毫米变化，说明仅说“输入射线所以与相机无关”是不成立的。新的 body canonical frame 把结构误差降到接近 0，是正式 CMU→H36M 之前必须做的基础。

---

## 11. 失败实验与禁止重复的路线

详细来源：

```text
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/FAILURE_INFORMED_MAP_DECODER_PLAN_20260820.md
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/GHT_POST2022_MODULE_MIGRATION_RECORD_20260821.md
/home/lixiaob/cjy/OpenRUMPL_baseline_audit/BASELINE_FREEZE_20260821.md
```

### 11.1 简单 bias / geometry bias

- confidence bias、geometry bias、learnable view bias、手工夹角/条件数标量等大多无提升或只有极小提升。
- 原因：单个标量与真实 3D 错误相关性很弱；检测器误差是关节相关、视角相关且具有系统偏差。
- GBT 新主干无偏置 B0 vs 有偏置 B3 的数字不是 RUMPL baseline 消融，不能据此声称我们的 bias 实现应该同幅提升。

### 11.2 直接替换图卷积/大 Transformer

- SGraFormer/GraFormer 插入或替换出现严重退化。
- frozen output 后加 query residual 收益 <0.03 mm，因为压缩后的 3D 已丢失可恢复信息。
- 早期 global joint-view Transformer 可到 78/47/43 mm，优化与表示不匹配。
- 不要仅因为论文模块知名就整个堆上去。

### 11.3 解析几何求解器并非万能

- DLT、confidence DLT、IRLS、RANSAC/reprojection selection、pairwise hypotheses、soft medoid 多数弱于学习融合。
- “更贴近输入射线”不等于更接近 GT，因为 2D 检测存在跨视角相关的系统偏差。
- 固定骨长、PointDSC 等几乎无增益或退化。

### 11.4 候选评分局部平台

- 反复 K96/scorer/proposal/temperature 调参常只有 0.05–0.09 mm。
- K2-only 长训练得到 V2 37.886，但 V3/V4 崩到 62.215/46.217，属于 cardinality collapse。
- 禁止训练 V2 专家、V3专家、V4专家然后拼表；论文必须一个模型自然支持不同视角数。

### 11.5 时序失败

- 直接 post-fusion smoothing、普通 TCN、MixSTE 大模块、冻结空间模型后粗暴接时序，多数仅 0.1–0.2 mm或退化。
- 最佳 epoch 总在第 1 epoch 时通常说明模型只学接近 identity，而不是数据太少需要无限加 epoch。
- 旧时序使用绝对 root、H36M camera subset embedding，会损害相机泛化。
- 不要重复大权重速度/加速度/骨长损失；必须逐项小权重消融。

### 11.6 学习 2D/ray correction

- 内部 holdout 变好但 S9/S11 失败，说明学到 subject/detector bias。
- failure-informed MAP M1：内部约 +0.117 mm，但 S9/S11 恶化 +0.379/+0.362/+0.327；S8 strict 仅 +0.042。
- 低秩 2D bias oracle 上限仅约 0.075 mm，不值得继续堆网络。
- 选择性 2D correction 只有出现新的、可泛化 observable/监督证据时才重开。

### 11.7 先验/离散码本/跨领域模块

- pose density / normalizing flow gate 收益约 0.006 mm，跨测试更差。
- joint residual PCA oracle 太小。
- DSAC/可微 RANSAC、PCT/FSQ/K96、PointDSC、2D观测+射线双流等已尝试或不稳定，尚无正式增益。
- FSQ 能以约 20.45 mm 重建姿态只表示码本重建上限，不代表从噪声 2D 输入能正确选择 code；不能把 oracle/reconstruction 当真实结果。

### 11.8 输入路线失败

- CPN 转换后 V2 曾到 46.227 mm，RUMPL 融合器无法直接适配该置信度/误差分布，已放弃。
- A1D/热图输入的 33.811/28.555/27.808 等历史结果不是坐标级公平主表。
- LT-style bbox/crop/去畸变复制本身没有显著降低 2D 或三角化误差，说明前端处理不是唯一主瓶颈。

### 11.9 论文与工程原则

- 没有代码的 GBT 只能用于想法/对比，不能凭猜测把所有细节伪装成官方复现。
- 优先移植有官方代码、输入协议相符的模块；但仍必须做最小验证，不等于整模型堆叠。
- 每次实验记录输入、checkpoint、视角采样、seed、指标口径和输出路径。失败也记录，避免下一对话重复。

---

## 12. 论文主线建议

### 12.1 核心问题

多视角模型常假设增加视角必然有益，但真实 2D 检测包含关节级系统偏差与遮挡污染；在低冗余或跨相机/跨数据集条件下，额外视角可能产生负贡献。普通 RUMPL 又把世界 XYZ 作为学习坐标，产生隐式相机/数据集依赖。

### 12.2 方法故事

建议论文暂用以下统一故事，最终名称可再定：

> **身体规范坐标中的射线生成与候选效用融合，用轻量时序在证据受损时补偿，实现坐标级、跨相机、遮挡鲁棒的多视角 3D 姿态估计。**

三层贡献：

1. **SE(3)-canonical ray generator**：保留 RUMPL VFT/PFT，但移除世界轴依赖，使任意坐标旋转/平移下输出一致。
2. **canonical joint-wise candidate utility fusion**：融合学习生成与几何三角化候选，按关节抑制坏视角/坏候选，重点解决 3/4 视角和遮挡下的 negative evidence。
3. **camera-independent temporal residual**：在中心身体坐标中利用 T=9 邻域，避免相机 ID 和绝对 root 泄漏，在遮挡时恢复缺失证据。

Global Joint-Query 可作为 generator 内部对低冗余 ResNet 输入的增强，但因为 HRNet 不一定启用，论文需谨慎定位：可以描述为可选的低冗余 joint-view interaction，而不是整篇唯一核心。

### 12.3 不要过度声称

- body canonicalization 单独并非前所未有；创新应落在它与 RUMPL ray generator、候选效用和时序的一体化设计及系统验证。
- E2 不应声称“首次候选评分”或“首次不确定性三角化”。
- 时序不应声称因果、实时或巨大 clean 增益。
- 在 CMU→H36M 未完成前不要写“dataset-independent”。可以写“designed to remove coordinate-frame dependence”。

### 12.4 建议论文贡献表述草案

1. 揭示坐标级射线 Transformer 仍会因世界 XYZ 正则、绝对 root 和轴向特征产生显著 SE(3) 依赖，并提出端到端 body-canonical ray fusion 修复。
2. 提出规范坐标下的逐关节多假设效用融合，将学习式 RUMPL 候选与几何候选有机结合，在视角冗余和遮挡条件下抑制负贡献证据。
3. 提出与同一规范坐标共享的轻量 T=9 residual refinement，在不更改 2D 输入协议的前提下增强遮挡鲁棒性。
4. 在 H36M clean、H36M-Occluded、CMU 多视角和 CMU→H36M 零样本上系统验证 2/3/4（及 CMU 2/4/5/6/8）视角表现与相机泛化。

最后一条只有对应实验完成后才能保留为已验证贡献。

---

## 13. 后续实验顺序

### 阶段 A：完成当前 canonical H36M 双前端（正在执行）

必须得到 HRNet/ResNet：

- Generator T=1：V2/V3/V4；
- +E2 T=1：V2/V3/V4；
- +H18 T=9：V2/V3/V4；
- 两个 E2 seed 的均值/方差或至少一致性；
- 与旧 world-frame frozen rows 一一比较。

决策规则：

- canonical 等变性是硬约束；不能为了 0.x mm clean 收益退回明显世界轴依赖。
- 如果 canonical clean 退化 >1 mm，先排查 canonical frame 稳定性、左右肩/torso 低置信度、训练 schedule 和 checkpoint 读取，不要直接废弃。
- 若某前端 Query 退化，允许该前端关闭 Query，但其余 canonical/E2/H18结构保持一致。

### 阶段 B：canonical H36M-Occluded

固定阶段 A 最佳 checkpoint，不重新在遮挡数据训练：

- HRNet/ResNet；
- Occ-2/Occ-3；
- V2/V3/V4；
- direct、+E2、+H18。

重点指标：

- clean degradation；
- Occ V4 与 AdaFuse/SkelSplat；
- E2 对 V3/V4 的增益；
- H18 是否在遮挡下比 clean 提升更明显；
- Negative View Rate：增加一个视角后误差反而上升的关节/样本比例。

### 阶段 C：相机泛化的低成本验证

在正式下载 CMU 前，继续做少量但严格的测试：

- 随机全局 SE(3) 坐标变换；
- 未见 H36M 相机子集/相机顺序置换；
- 随机虚拟相机位置的 SMPL 少量帧；
- 去除/保留 canonical 的 matched ablation。

这些验证用于防止正式 CMU 训练几天后才发现结构泄漏，但不能替代跨数据集主表。

### 阶段 D：准备真实 CMU 训练数据

先从目标论文常用的单人 Panoptic 序列和 20 个视角列表确定最小下载集合；必须包含之后用于 2/4/5/6/8 的固定/随机视角策略。历史用户特别提到过 `3,6,12,13,23` 是常见测试视角，正式方案需再对照权威论文。

工作项：

1. 查清真实 CMU 训练/测试序列，不把 AMASS 合成目录混入。
2. 下载视频/图像、校准和 3D 标签。
3. 用同一 HRNet/ResNet 前端生成 coordinate PKL。
4. 校验同步、米/毫米、关节映射、camera-to-world/world-to-camera。
5. 小样本 overfit 和 GT2D triangulation sanity。
6. 正式 CMU 训练和同域 2/4/5/6/8 评估。
7. 不微调直接 H36M S9/S11 测试。

### 阶段 E：消融和论文表格

至少包含：

| ID | canonical generator | Query | 22c E2 | canonical E2 stats | H18 | 用途 |
|---|---|---|---|---|---|---|
| A0 | — | 前端原设置 | — | — | — | 旧 RUMPL/H76 |
| A1 | ✓ | — | — | — | — | 坐标无关 generator |
| A2 | ✓ | ResNet可选 | — | — | — | 低冗余融合 |
| A3 | ✓ | 同A2 | ✓ | ✓ | — | 候选效用 |
| A4 | ✓ | 同A2 | ✓ | ✓ | ✓ | 最终模型 |

附加消融：

- world XYZ vs body canonical；
- H36M mean/std vs fixed metric normalization；
- absolute root/camera task embedding on/off；
- 11 generator vs 11 triangulation vs 22 candidates；
- identity hinge on/off；
- T=1 vs centered T=9；
- fixed V2 vs mixed cardinality schedule；
- SE(3) equivariance error与MPJPE共同报告。

---

## 14. 下一位 Codex 的明确工作清单

1. 持续监控当前 Stage-1，不能重复开任务或抢 GPU1。
2. HRNet generator 完成后立即读取正式 V2/V3/V4 `table2.json`，与 38.686/30.943/28.629 比较；不要用训练日志内部 validation 替代。
3. ResNet generator 完成后与 32.312/25.101/23.536 比较。
4. 检查 downstream 是否自动从等待状态进入 cache/E2/H18；任何错误先看对应 log 和 manifest。
5. 汇总新的 canonical 六行 Stage-1 表，更新本文件和：
   - `STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`
   - `PAPER_MASTER_STORY_METHOD_RESULTS_CODE_20260815.md`
6. 若 canonical clean 合格，使用同一最终模型重跑 H36M-Occ Stage-2，更新遮挡表。
7. 在 Stage-2 同时补 Negative View Rate / monotonic violation，强化 E2 的论文证据。
8. 完成少量相机变换/随机相机审计后，再准备真实 CMU 数据；正式方向只做 CMU→H36M。
9. 所有新实验先检查历史 failure docs，避免重复 K96、粗暴时序、2D correction、CPN、蒸馏等失败路线。
10. 每得到一个正式结果，记录：代码 commit/diff、完整命令/launcher、env、数据 PKL、checkpoint、seed、全部视角组合和输出路径。

---

## 15. 重要文档索引

按阅读优先级：

1. 本交接：
   `/home/lixiaob/cjy/2026-08-24_chat_handoff_camera_generalization_stage1_and_next.md`
2. 项目早期总交接（了解 7 月阶段历史用，不代表当前状态）：
   `/home/lixiaob/cjy/2026-07-26_chat_handoff_gt2d_stage_audit_and_next.md`
3. 相机/数据集依赖修复：
   `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/CAMERA_DATASET_DEPENDENCE_AUDIT_AND_FIX_20260824.md`
4. Stage-1 clean 冻结结果：
   `/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`
5. Stage-2 occlusion 冻结结果：
   `/home/lixiaob/cjy/STAGE2_H36M_OCC_RESULTS_FOR_PAPER_20260824.md`
6. 论文母稿/方法与代码：
   `/home/lixiaob/cjy/PAPER_MASTER_STORY_METHOD_RESULTS_CODE_20260815.md`
7. GBT 六阶段路线：
   `/home/lixiaob/cjy/GBT_ALIGNED_SIX_STAGE_EXPERIMENT_PLAN_20260815.md`
8. 双前端 Joint-Query 计划：
   `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/JOINT_QUERY_MATCHED_FRONTENDS_PLAN_20260822.md`
9. 遮挡 dense 计划：
   `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/POSEFUSION_OCC23_DENSE_FINAL_PLAN_RESULTS_20260824.md`
10. 失败驱动 MAP 记录：
   `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/FAILURE_INFORMED_MAP_DECODER_PLAN_20260820.md`
11. GHT/后2022模块迁移记录：
    `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/GHT_POST2022_MODULE_MIGRATION_RECORD_20260821.md`

参考论文目录：

```text
/home/lixiaob/cjy/reference
/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf
/home/lixiaob/cjy/reference/Mixste(1).pdf
/home/lixiaob/cjy/reference/time
/home/lixiaob/cjy/reference/transformer
/home/lixiaob/cjy/reference/AAAmy/TGR_Ray_Chinese_CVPR_Draft.pdf
```

---

## 16. 给新对话的最终提醒

当前项目已经不是“找一个 Transformer 随便替换 RUMPL”的阶段。最重要的进展是：

1. 已有一套在 H36M clean 和遮挡上很强的旧冻结模型与可信评估口径；
2. 已查出旧模型隐含依赖世界坐标和 H36M统计，泛化故事此前并不成立；
3. 已完成 generator、E2、H18 三层相机无关改造，并通过小样本/等变性测试；
4. 正在进行完整 H36M 双前端重训，结果尚未冻结；
5. 下一关键里程碑是 canonical clean 不退化，然后 canonical occlusion，最后真实 CMU→H36M。

接手时不要把历史好数字、当前中间日志、小样本审计、合成 CMU 和真实 CMU混在一起。所有论文结论都必须对应一个明确协议、输入前端、T、checkpoint 和结果文件。

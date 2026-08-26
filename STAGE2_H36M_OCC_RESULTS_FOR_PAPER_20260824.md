# Stage 2：Human3.6M-Occ 遮挡实验——论文更新事实源

> 用途：本文件可直接交给 Codex，用于更新论文的 Experiments、Ablation、
> Robustness 和 Implementation Details。若本文件与旧聊天记录或旧母稿中的遮挡数字
> 冲突，以本文件和最终生成的 `final_occ23_table.json` 为准。
>
> 当前状态（2026-08-24）：T=1 空间消融、Algebraic Triangulation 和公开论文对比值
> 已冻结；完整 T=9 主方法仍在运行。所有 `{{...}}` 均为待自动回填项，**不得用 T=1
> 数字代替 T=9 主结果**。

## 1. 这一阶段要证明什么

第一阶段已经在无遮挡 Human3.6M 上确定了完整模型。第二阶段不重新训练模型，而是把
第一阶段冻结的权重直接用于未见过的物体遮挡，检验以下三个命题：

1. 模型在少视角和部分视角受污染时仍能稳定融合，而不是依赖固定、干净的相机集合；
2. E2 候选评分在遮挡下可以抑制不可靠的多视角几何解释；
3. H18 时序残差利用相邻帧补充当前帧缺失信息，在不改变二维输入来源的条件下进一步
   提高遮挡鲁棒性。

这部分在论文中的核心故事应写成：

> 我们不是通过遮挡数据增强重新适配测试集，而是将只在 clean Human3.6M 上训练的
> 完整模型零样本迁移到 Human3.6M-Occ。空间候选重评分处理跨视角污染，轻量时序残差
> 则利用邻帧上下文恢复当前帧不稳定的姿态，两者分别覆盖空间与时间上的观测退化。

## 2. 冻结的完整方法与 clean 参考结果

遮挡主表必须继承第一阶段的完整 baseline，不能临时换成表现较好的单个模块：

| 二维输入 | 第一阶段冻结链 | T | Clean V2 | Clean V3 | Clean V4 |
|---|---|---:|---:|---:|---:|
| ResNet-152 | Global Joint-Query RUMPL + E2 identity safeguard + H18 | 9 | **31.215** | **22.008** | **19.971** |
| HRNet-W32 | C2 RUMPL + E2-C2 + H18 | 9 | **37.704** | **29.231** | **27.219** |

其中 ResNet 链是主结果；HRNet 链用于和采用 HRNet 坐标输入的工作比较，并验证方法不
依赖单一二维检测器。Direct 与 E2 的 `T=1` 结果只用于消融，不得称为最终 Ours。

## 3. Human3.6M-Occ 实验协议

- 数据划分：Human3.6M 的 S1/S5/S6/S7/S8 用于 clean 训练，S9/S11 用于测试；
- 遮挡生成：遵循公开 Human3.6M-Occ 生成流程，在人物区域粘贴 Pascal VOC 前景物体；
- Occ-2：四个原始相机中随机遮挡两个；Occ-3：四个原始相机中随机遮挡三个；
- 每个被选中的视角粘贴两个物体，尺度范围为人体框短边的 `0.2--0.5`，随机种子 42；
- V2/V3/V4 分别穷举四个相机的全部 `6/4/1` 种组合，然后取平均；
- 时序上下文由 26,269 个同步四相机组构成；最终在与空间实验一致的 2,021 个同步
  目标中心组上评分，保证 T=1/T=9 的样本完全匹配；
- T=9 使用 `4` 帧过去、当前帧和 `4` 帧未来，帧间隔为 5；只在上述目标中心帧评分；
- 指标：17 关节 absolute MPJPE（mm），按动作等权平均，不做 root alignment 或
  Procrustes alignment；
- 训练约束：所有模型只在无遮挡 H36M 上训练；遮挡集不参与训练、epoch 选择、温度
  调节或超参数选择；
- 两种输入分别报告，不能混合：ResNet-152 行只和同类 ResNet-152 输入作最强公平比较，
  HRNet-W32 行单独报告。

### 论文建议用语

正文无需展开内部的生成器恢复和数值控制过程。Implementation Details 中写：

> Following the public Human3.6M-Occ generation procedure, we paste two
> Pascal-VOC foreground objects into each selected camera view. Occ-2 and
> Occ-3 occlude two and three of the four source views, respectively. We use
> an object-scale range of 0.2--0.5 and a fixed random seed of 42. All models
> are trained exclusively on clean Human3.6M and evaluated on every 2-, 3-,
> and 4-camera subset without occlusion-specific fine-tuning or model
> selection.

## 4. 表 II：完整遮挡主结果（最终论文主表）

表头中的 V2/V3/V4 表示测试时使用的相机数；数值为 absolute MPJPE，越低越好。
Algebraic 行和 Ours 行使用相同的遮挡图像、二维输入和相机组合。完整 Ours 必须是
`T=9`。

| Method / 2D input | T | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alg. Tri. / ResNet-152 | 1 | 153.150 | 45.657 | 41.855 | 140.948 | 68.883 | 50.716 |
| **Ours / ResNet-152** | **9** | **{{RES_OCC2_V2_T9}}** | **{{RES_OCC2_V3_T9}}** | **{{RES_OCC2_V4_T9}}** | **{{RES_OCC3_V2_T9}}** | **{{RES_OCC3_V3_T9}}** | **{{RES_OCC3_V4_T9}}** |
| Alg. Tri. / HRNet-W32 | 1 | 249.877 | 68.385 | 55.986 | 360.170 | 110.317 | 60.622 |
| **Ours / HRNet-W32** | **9** | **{{HR_OCC2_V2_T9}}** | **{{HR_OCC2_V3_T9}}** | **{{HR_OCC2_V4_T9}}** | **{{HR_OCC3_V2_T9}}** | **{{HR_OCC3_V3_T9}}** | **{{HR_OCC3_V4_T9}}** |

### 结果出来后的自动描述模板

不要提前填写“提升显著”之类结论。根据最终 JSON 计算后，把下面模板中的占位符替换：

> Under the ResNet-152 input, our complete model obtains
> `{{RES_OCC2_V2_T9}}/{{RES_OCC2_V3_T9}}/{{RES_OCC2_V4_T9}}` mm on Occ-2 and
> `{{RES_OCC3_V2_T9}}/{{RES_OCC3_V3_T9}}/{{RES_OCC3_V4_T9}}` mm on Occ-3 for
> two, three, and four views, respectively. Compared with Algebraic
> Triangulation using the same 2D input, this reduces the four-view error by
> `{{RES_ALG_GAIN_OCC2_V4}}` mm and `{{RES_ALG_GAIN_OCC3_V4}}` mm. The same
> trend is observed with HRNet-W32, demonstrating that the robustness gain is
> not tied to a particular 2D detector.

如果 HRNet 某个视角数没有优于 Algebraic，只写“保持一致趋势”前必须先核对；应改为
客观陈述各列数值，不能用 ResNet 的结论代替 HRNet。

## 5. 表 III：与公开方法的四视角比较

公开论文只提供了精确的 V4 数字，因此这张表只比较 V4，不从曲线估算 V2/V3。
ResNet-152 block 是主要公平对比；其他二维前端只作为外部参考。T 是模型使用的时序
长度，透明列出即可。时序是我们的方法模块，不是更换输入，因此完整 T=9 可以进入主表，
同时必须在消融表提供 matched T=1 结果。

### 5.1 相同 ResNet-152 输入族的主要比较

| Method | T | Occ-2 V4 | Occ-3 V4 |
|---|---:|---:|---:|
| Algebraic Triangulation | 1 | 43.2 | 48.9 |
| RANSAC (as in AdaFuse) | 1 | 33.7 | 38.6 |
| AdaFuse | 1 | 27.9 | 31.2 |
| SkelSplat | 1 | **24.6** | **27.0** |
| **Ours** | **9** | **{{RES_OCC2_V4_T9}}** | **{{RES_OCC3_V4_T9}}** |

### 5.2 不同二维前端的补充参考

| Method / input reported by the method | Occ-2 V4 | Occ-3 V4 |
|---|---:|---:|
| Algebraic Triangulation / MeTRAbs | 36.0 | 39.0 |
| TransFusion / method-specific frontend | 40.8 | 76.3 |
| MV Pose Fusion / method-specific frontend | 33.4 | 36.7 |
| SkelSplat / MeTRAbs | 29.6 | 31.1 |

以上公开数值来自 SkelSplat WACV 2026 Table 4。正文主要讨论 5.1；5.2 可放补充材料，
不能把不同前端的数值说成严格 matched-input 比较。

### 可直接使用的英文比较段落

> Table X compares robustness under object occlusion. With the same
> ResNet-152 keypoint source, our complete model reaches
> `{{RES_OCC2_V4_T9}}` mm on Occ-2 and `{{RES_OCC3_V4_T9}}` mm on Occ-3.
> This is `{{GAIN_VS_SKEL_OCC2}}/{{GAIN_VS_SKEL_OCC3}}` mm lower than
> SkelSplat and `{{GAIN_VS_ADA_OCC2}}/{{GAIN_VS_ADA_OCC3}}` mm lower than
> AdaFuse. We explicitly report the temporal length of each method: our T=9
> model uses the same ResNet-152 2D observations as its T=1 spatial counterpart,
> while exploiting neighboring frames through a lightweight residual temporal
> head.

只有当差值为正（我们的误差更低）时才使用 “lower than”；否则改成 “compared with”
并仅报告数值。

## 6. 表 IV：空间模块消融（已冻结的 T=1 结果）

这张表用于回答“遮挡提升来自哪里”。Direct 是冻结空间主干的直接输出；`+E2` 加入
候选评分与软融合，但不使用时序。

| Input / spatial model | T | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HRNet C2 direct | 1 | **55.510** | 38.286 | 34.175 | **64.060** | 42.003 | 36.894 |
| HRNet C2 + E2 | 1 | 55.576 | **33.840** | **29.406** | 64.143 | **37.122** | **31.600** |
| ResNet GQ direct | 1 | **50.409** | 36.722 | 34.534 | **61.120** | 42.832 | 40.286 |
| ResNet GQ + E2 | 1 | 50.511 | **29.336** | **23.383** | 61.267 | **34.067** | **26.092** |

相对同输入 Direct，E2 的变化为：

| Input | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|
| HRNet：Direct − E2 | -0.066 | **4.446** | **4.769** | -0.083 | **4.881** | **5.294** |
| ResNet：Direct − E2 | -0.102 | **7.386** | **11.151** | -0.147 | **8.765** | **14.194** |

正数表示 E2 降低误差。E2 在 V3/V4 的提升很大，在 V2 轻微退化约 `0.07--0.15 mm`。
正确解释不是“E2 对两视角完全无效”，而是：两视角缺少第三条冗余观测，若两个输入
视角均被污染，候选评分无法从不存在的信息中恢复关节；三、四视角有更多几何候选，
因此可靠候选识别和软融合的收益随冗余增加而放大。

### 可直接使用的英文消融描述

> The benefit of candidate scoring becomes substantially larger under
> occlusion. For ResNet-152, E2 reduces the V4 error by 11.151 mm on Occ-2
> and 14.194 mm on Occ-3 over the matched direct generator. The corresponding
> gains with HRNet-W32 are 4.769 and 5.294 mm. In contrast, V2 changes by less
> than 0.15 mm. This cardinality-dependent behavior is expected: three and
> four views provide redundant geometric hypotheses that allow the scorer to
> identify corrupted observations, whereas a two-view subset may contain no
> clean view and offers insufficient redundancy for outlier discrimination.

## 7. 表 V：H18 时序消融（等待最终回填）

这一表必须比较**完全相同目标中心帧**上的 E2 T=1 和完整 H18 T=9，不能直接拿旧稀疏
T=1 表和稠密 T=9 表相减。

| Input / protocol | matched T=1 V2/V3/V4 | H18 T=9 V2/V3/V4 | H18 gain V2/V3/V4 |
|---|---:|---:|---:|
| ResNet, Occ-2 | {{RES_OCC2_CENTER_T1}} | {{RES_OCC2_T9}} | {{RES_OCC2_H18_GAIN}} |
| ResNet, Occ-3 | {{RES_OCC3_CENTER_T1}} | {{RES_OCC3_T9}} | {{RES_OCC3_H18_GAIN}} |
| HRNet, Occ-2 | {{HR_OCC2_CENTER_T1}} | {{HR_OCC2_T9}} | {{HR_OCC2_H18_GAIN}} |
| HRNet, Occ-3 | {{HR_OCC3_CENTER_T1}} | {{HR_OCC3_T9}} | {{HR_OCC3_H18_GAIN}} |

结果出来后按三种情况写：

- 三列都改善：可以说 H18 稳定提高遮挡鲁棒性，并报告平均和最大收益；
- V2 改善、V3/V4 持平：重点写时序补足少视角冗余不足，避免夸大总体收益；
- 有列退化：如实报告，并把时序价值限定在改善的遮挡强度/视角数，不能称为普遍提升。

建议英文模板：

> Adding H18 changes the matched center-frame T=1 results from
> `{{RES_OCC2_CENTER_T1}}` to `{{RES_OCC2_T9}}` mm on Occ-2 and from
> `{{RES_OCC3_CENTER_T1}}` to `{{RES_OCC3_T9}}` mm on Occ-3. The largest gain
> appears at `{{H18_BEST_SETTING}}`, where temporal context reduces the error
> by `{{H18_BEST_GAIN}}` mm. Since the temporal head is trained only on clean
> sequences and initialized as an identity residual, this improvement reflects
> zero-shot temporal robustness rather than occlusion-specific adaptation.

## 8. 方法故事与表格之间的对应关系

| 论文主张 | 证据表格 | 当前状态 |
|---|---|---|
| 少视角无遮挡精度 | Stage-1 clean V2/V3/V4 主表 | 已完成 |
| E2 能抑制受污染的跨视角候选 | 本文件表 IV，Direct vs E2 | 已完成，V3/V4 强成立 |
| 完整模型对未见遮挡有泛化能力 | 本文件表 II，Occ-2/Occ-3 V2/V3/V4 | 等待 T=9 |
| 与相同 ResNet-152 输入方法相比具有竞争力 | 本文件表 III | 等待 T=9 后计算排名 |
| 时序对遮挡有额外贡献 | 本文件表 V，matched T=1 vs T=9 | 等待 T=9 |
| 改进不依赖单一二维检测器 | ResNet 与 HRNet 两个 block | 主空间结论已成立；完整结论待 T=9 |

整体叙事顺序应为：clean 少视角能力 → 未见遮挡零样本测试 → 空间候选评分消融 → 时序
消融 → 跨二维输入验证。不要把 Algebraic Triangulation 的协议核验过程写成方法贡献。

## 9. 可以宣称与暂时不能宣称的内容

### 现在已有数据支持

- E2 在 Occ-2/Occ-3 的 V3/V4 上对两种二维输入均显著优于 matched direct generator；
- ResNet T=1 空间消融的 V4 已达到 `23.383/26.092 mm`，分别比公开
  SkelSplat ResNet-152 的 `24.6/27.0 mm` 低 `1.217/0.908 mm`；
- 提升来自冻结的 clean-trained 模型，未用遮挡标签或遮挡验证集调参；
- 相同模块在 ResNet 与 HRNet 两种坐标输入上呈现一致的 V3/V4 鲁棒收益。

### 最终 T=9 出来前不能写

- “完整模型在 Occ-2/Occ-3 上达到 SOTA”；
- “H18 在所有视角数和两种输入上均有效”；
- “完整模型超过 SkelSplat/AdaFuse 多少 mm”；
- 用当前 T=1 `23.383/26.092` 冒充完整 T=9 主结果。

## 10. 给 Codex 的明确更新指令

把本文件交给 Codex 时，可以直接附上以下指令：

> 请以 `STAGE2_H36M_OCC_RESULTS_FOR_PAPER_20260824.md` 为唯一的 Stage-2
> 遮挡实验事实源，更新论文的实验设置、主结果、消融和鲁棒性分析。保留输入来源和 T
> 列，ResNet 与 HRNet 分开比较。不要展开内部协议恢复过程，不要把不同前端称为严格
> 公平比较，不要从图中估算缺失的 V2/V3 数字。若 `{{...}}` 尚未回填，保留占位符并
> 标注 pending，绝不能用 T=1 消融值替代完整 T=9 主结果。英文描述可以润色，但不能
> 改变本文件中的实验口径、数值或结论强度。

## 11. 最终结果回填来源与关键文件

最终运行结束后，以以下两个文件为数值真值源：

- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.json`；
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.md`。

自动汇总脚本：

- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/collect_posefusion_occ23_dense_final_table_20260824.py`。

其他可追溯来源：

- 第一阶段 clean 冻结结果：
  `/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`；
- 公开协议与内部审计：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/POSEFUSION_OFFICIAL_OCC_V234_20260824.md`；
- 稠密时序实验计划：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/POSEFUSION_OCC23_DENSE_FINAL_PLAN_RESULTS_20260824.md`；
- SkelSplat WACV 2026 原文：
  `/home/lixiaob/cjy/reference/7.15/Bragagnolo_SkelSplat_Robust_Multi-view_3D_Human_Pose_Estimation_with_Differentiable_Gaussian_WACV_2026_paper.pdf`。

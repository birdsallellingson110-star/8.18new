# Chat 交接文档（2026-07-25 ~ 07-26）

供下一轮对话直接读取。覆盖：硬基线、本 chat 讨论与实验、失败原因、RUMPL 关键细节、文件位置、下一步该做什么。

相关旧文档（勿重复踩坑）：
- `/home/lixiaob/cjy/2026-07-24_failure_analysis_and_rumpl_module_audit.md`（F1–F5 失败分类 + M1–M4 粗糙模块）
- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/NOTES_codebase_study_20260725.md`（文献代码对照 + KPA/MH3/D3）
- Agent transcript：`/home/lixiaob/.cursor/projects/home-lixiaob-cjy/agent-transcripts/693d1558-c64c-45ec-9c1d-816ce08a2bba/`

---

## 0. 一句话现状

**干净 CMU + 固定 HRNet-2D 上再拧 VFT/模块基本到软天花板；GT-2D 上限实验证明约一半 V2 误差来自 2D，但去掉 2D 后最大非 2D 误差是「网络偏离好几何」而非三角化做不到。下一主线：正确写的 reliability-gated geometric residual（不是已失败的 D1 加法式 tri_anchor）。**

---

## 1. 环境与硬基线（R5）

| 项 | 路径/值 |
|----|---------|
| 工作仓 | `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/` |
| 模型代码 | `RUMPL/lib/models/multiview_rumpl.py`（~1500 行） |
| 官方式配置 | `RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml` |
| Python | `/home/lixiaob/cjy/rumpl_venv310/bin/python` |
| 输出根 | `/mnt/data/cjyoutput/baseline_reaudit_20260722/` |
| R5 ckpt | `.../output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar` |
| 评测脚本模板 | `eval_exact_multiview_20260723.sh`（`--use-mmpose-val`） |

**R5 干净 CMU Absolute MPJPE（mmpose HRNet，All-17 / KP*）：**

| Views | All-17 | KP* |
|------:|-------:|----:|
| V2 | 30.885 | 35.506 |
| V3 | 23.039 | 25.159（以 summary 为准） |
| V4 | 20.213 | 21.698 |
| V5 | 18.746 | 20.091 |

汇总文件示例：
- V2：`occlusion_eval/R5_v2_occ0.0_summary.json`
- V3–V5：`multiview_model_best_eval/R5_v{3,4,5}_summary.json`

**泛化主因（用户强调，已确认代码）：**  
不是单靠 ray/VFT 结构，而是 **AMASS + 每样本随机抽视角（常配到 20）+ 可变视角数 + 纯 2D+conf**。  
代码：`multiview_amass_rumpl.py` 中 `max_random_n_views` / `np.random.choice(self.all_camera_ids, n_views)`；模型 `random_num_views` 分支在 `multiview_rumpl.py` ~608+。  
**融合任何图像方法时禁止用固定相机端到端重写这条训练协议。**

---

## 2. RUMPL 前向流水线（评测/审计用）

数据（CMU val）：
`multiview_cmu_panoptic_rumpl.py` `__getitem__` →  
`get_rays`（`joints_dataset_rumpl.py`）→ 每视角 `(direction, intersection/origin, conf)` →  
`closest_points_on_n_skew_lines`（`lib/utils/calib.py:579`）→ `middle_points = mean` →  
`rays = cat(dir, origin, conf)` shape `(17, V, 7)` 量级。

模型：
- 输入：`APPLY_VIEW_FUSION=True` 时吃 **rays**（不是 middle_points）
- VFT（跨视角）→ 取 fusion token（`x[:,0,:]`）→ PFT（跨关节）→ `head = LN + Linear(D→3)` **直接回归绝对 3D**
- 官方最优：**不**把 middle_points 送进网络

评测入口：`RUMPL/run/valid_rumpl.py`  
- mmpose：`--use-mmpose-val`  
- GT/org 2D：`--not-use-mmpose-val`（conf→1；org `joints_2d`）  
汇总：`RUMPL/run/summarize_cmu_predictions.py`

**注意：** dataset 的 `middle_points`（n-skew solve）在 **V=2 可靠**；**V>2 数值会炸到几十米**（本 chat 审计已证实）。推理路径不依赖 mid；几何锚请用 **V=2 mid** 或 **可微 weighted ray LS**（D1 里那套 3×3 solve），不要用坏的 n-line mid。

---

## 3. 本 chat 时间线与实验

### 3.1 讨论（未全部落地代码）

| 话题 | 结论 |
|------|------|
| MVGFormer 融合（单人/多人） | 正确姿势：**AM 留给图像侧（query+遮挡 2D），GM 换成 RUMPL ray-VFT**；不要三角化替换 VFT；单人=P=1。级联 P0 最稳。 |
| AM 抗遮挡机制 | 不是单图瞎猜；是 **AM↔GM 迭代**：3D 投影→deformable 采特征→修 2D+conf→三角化/几何→再投影。无图像则借不来。 |
| 「误差是不是全是 2D」 | 否。但固定 HRNet 时改 fusion 空间极小；动 2D 或换 stress 协议才有空间。 |
| 创新点 | 单纯换 2D 检测器不够发论文；无图修 2D 已失败；应打「好几何时网络乱跑」的结构性问题。 |

参考代码仓：`/mnt/data/cjydata/reference_code/MVGFormer/`（本地 clone 可能缺完整 models/ops；论文 arXiv 2311.10983）。

### 3.2 文献模块轨（本 chat 前后并行，结果）

实现均在 `multiview_rumpl.py`，env 开关：

| 实验 | Env | 状态 | 相对 R5/A2 |
|------|-----|------|------------|
| **A2**（struct_occ 0.4 + occJL soft×2） | `RUMPL_TRAIN_STRUCT_OCC=1 LEVEL=0.4`, `RUMPL_OCC_JOINT_LOSS=1` | **保留/赢家** | V2 clean 约 30.43（略优于 R5 30.89）；遮挡协议有用 |
| **KPA** SemGCN（禁 softmax）+A2 | `RUMPL_KPA=1` | **失败** | V2 32.84（+~2mm）；全面差 |
| **MH3** H=3 fusion tokens + Conv1d | `RUMPL_MULTI_HYP=3` | **偏负/未超** | V2 31.19；V5 19.19 |
| **D3-PCT** codebook+EMA+CE | `RUMPL_POSE_CODEBOOK=1` | **训练炸** | inplace grad：`D3PCT_a2_seed0_20260725.log` FAILED |
| DS1 hard-view+legw | （旧） | 弱正于 V2 | 可作辅助故事 |
| GBT / AdaFuse VW / 几何 2D refine / LT 替 VFT / 时序 T2 | 见 7/24 文档 | **失败** | F2/F4 类 |

脚本：
- `run_kpa_a2_20260725.sh`, `run_mh3_a2_20260725.sh`, `run_d3_pct_a2_20260725.sh`
- `launch_kpa_mh3_d3pct_20260725.sh`, `chain_module_waitlog_20260725.sh`
- 日志：`/mnt/data/cjyoutput/baseline_reaudit_20260722/{KPA,MH3,D3PCT}_a2_seed0_20260725.log`
- 评测：`occlusion_eval/{KPA,MH3}_v*_occ*_summary.json`

### 3.3 GT-2D 上限（本 chat 新做，关键）

脚本：`OpenRUMPL_baseline_audit/eval_gt2d_upperbound_20260725.sh`  
结果目录：`/mnt/data/cjyoutput/baseline_reaudit_20260722/gt2d_upperbound/`  
对比：`COMPARE_mmpose_vs_gt2d.json`

| Views | mmpose All/KP* | GT-2D All/KP* | ΔAll |
|------:|---------------:|--------------:|-----:|
| V2 | 30.89 / 35.51 | **16.49 / 15.85** | **−14.4** |
| V3 | 23.04 / … | **14.96 / 14.46** | **−8.1** |
| V4 | 20.21 / … | **14.12 / 13.83** | **−6.1** |
| V5 | 18.75 / … | **13.58 / 13.49** | **−5.2** |

解读：V2 约一半误差随完美 2D 消失；仍剩 ~16.5mm → 非纯 2D。

### 3.4 分步误差审计（本 chat 核心新发现）

脚本：`RUMPL/run/audit_stage_errors_gt2d.py`  
结果：`.../gt2d_upperbound/stage_audit/`  
总表：`COMPARE_stages.json`

**V2 All-17 mm：**

| 阶段 | mmpose | GT-2D |
|------|-------:|------:|
| 1 ray→GT 距离 | 31.2 | 5.9 |
| 2 ray-oracle（各射线最近点再均） | 25.2 | 5.3 |
| 3 mid-geom（V=2 斜线中点） | 43.5 | **10.5** |
| 4 RUMPL pred | 30.9 | **16.5** |
| 网络相对 mid | **改善 −12.6** | **变差 +6.0** |
| 网络优于 mid 的关节比例 | 61% | **15.5%** |

GT-2D 下网络拉垮最狠关节（rumpl−mid）：**lhip +17.6, lelb +15.9, rhip +15.7, nose +11.3**（偏全局/躯干，不像纯脚踝 2D 噪声）。

**双重人格：**
- 坏 2D：几何崩，网络在救人（这是 RUMPL 存在价值）。
- 好 2D：几何已 ~10.5，网络跑到 16.5（容量浪费在「重新发明三角化」+先验偏置）。

V3/V5 的 dataset mid 数值炸了（忽略）；只信 ray≈5–6、RUMPL≈14–15。

---

## 4. 失败原因（本 chat 增量 + 继承）

### 已继承（7/24 F1–F5）
- F1 坏基线幻觉；F2 信息冗余 bias；F3 先验与多视角冲突；F4 合成域时序；F5 投影 loss 奇异。

### 本 chat 新增/钉死

| ID | 现象 | 根因 | 勿再做 |
|----|------|------|--------|
| **N1** | 固定 mmpose 上改 VFT/KPA/MH3… 不动点 | 误差大头在 2D+少视角；lifter 近饱和 | 再堆 attention bias / 换 fusion 花活刷干净 V2 −1mm |
| **N2** | 无图几何 2D refine 偏负 | AM 有效靠**图像特征**；无外观 fill 只会引入坏几何 | `RUMPL_2D_REFINE` 几何 soft_fill 当主线 |
| **N3** | D1 `RUMPL_TRI_ANCHOR` 全面变差（V2 33.19） | 实现是 `pred = head_abs + gate * anchor_abs`（两套绝对坐标相加），**不是残差** | 原样重开 D1 |
| **N4** | D3PCT 训练崩溃 | codebook 路径 inplace 与 backward 冲突 | 未修 anomaly 前勿重训 |
| **N5** | 与 MVGFormer 整网端到端混训 | 会毁掉 AMASS 随机视角泛化 | 必须分域：AM 图像域 / RUMPL 合成多视角 |

D1 错误代码位置：`multiview_rumpl.py` ~1368–1369：
```python
if tri_anchor_point is not None:
    x = x + self.tri_anchor_gate * tri_anchor_point  # BUG: abs+abs
```
正确形式应近似：`pred = anchor + gate * resid`，且 resid 头零初始化；或 `pred = (1-g)*anchor + g*head`。

D1 结果：`depth_anchor_eval/D1_tri_anchor_seed0_20260724_v*_summary.json`。

---

## 5. 重要文件索引

### 代码
| 用途 | 路径 |
|------|------|
| 主模型 | `OpenRUMPL_baseline_audit/RUMPL/lib/models/multiview_rumpl.py` |
| CMU 数据/rays/mid | `.../lib/dataset/multiview_cmu_panoptic_rumpl.py` |
| AMASS 随机相机 | `.../lib/dataset/multiview_amass_rumpl.py` |
| get_rays | `.../lib/dataset/joints_dataset_rumpl.py` |
| skew mid | `.../lib/utils/calib.py` (`closest_points_on_n_skew_lines`) |
| train/val | `.../run/train_rumpl.py`, `valid_rumpl.py` |
| 训练循环 | `.../lib/core/function_rumpl.py` |
| GT2D 评测 | `eval_gt2d_upperbound_20260725.sh` |
| 分步审计 | `RUMPL/run/audit_stage_errors_gt2d.py` |
| 标准多视角评 | `eval_exact_multiview_20260723.sh` |

### 结果
| 用途 | 路径 |
|------|------|
| 输出根 | `/mnt/data/cjyoutput/baseline_reaudit_20260722/` |
| GT2D | `gt2d_upperbound/COMPARE_mmpose_vs_gt2d.json` |
| 分步 | `gt2d_upperbound/stage_audit/COMPARE_stages.json` |
| 遮挡评 | `occlusion_eval/` |
| 参考文献代码 | `/mnt/data/cjydata/reference_code/`（MVGFormer, KTPFormer, MHFormer, PCT, AdaFuse, LT…） |

---

## 6. 下一对话建议起点（已达成共识、未实现）

**主线：Reliability-gated geometric residual（修对的 D1）**

```
anchor = weighted ray LS（可微；V2 可用 mid 对照）
resid  = Head(VFT)          # 只学偏移，init≈0
gate   = σ(MLP([conf, ray_consist, ...]))
pred   = anchor + gate ⊙ resid
```

建议步骤：
1. 零训练：V2 上 `pred=mid` 对照 R5（应接近 All≈10.5）。
2. 改公式 + 冻 VFT 只训 resid 头 → 看 GT-2D 是否 16→~11、mmpose 是否不崩过 R5。
3. 加 gate → 小 lr 全网。
4. 必报：mmpose V2–V5、GT-2D V2、occ0.6。

故事句：
> 坏 2D 时网络优于三角化；好 2D 时三角化优于网络——用可靠性门控在两者间切换，而不是绝对坐标回归。

**不要：** 再开 KPA/MH3/GBT bias；无图 2D refine；用 Panoptic 固定相机重训掉随机视角；未修 inplace 前重跑 D3PCT。

**可选支线（故事向，非抠干净 mm）：**  
Ray-VFT 作为 geometry adapter 插到 FusionFormer/MVGFormer-AM 级联；遮挡/少视角 stress 协议（A2 已有基础）。

---

## 7. 给下一 agent 的最小指令模板

```
读 /home/lixiaob/cjy/2026-07-26_chat_handoff_gt2d_stage_audit_and_next.md
硬基线 R5 V2=30.885；仓 OpenRUMPL_baseline_audit；python rumpl_venv310。
任务：把 multiview_rumpl.py 的 RUMPL_TRI_ANCHOR 从 abs+abs 改成 pred=anchor+gate*resid，
零初始化 resid，先冻骨干训头，用 eval_gt2d_upperbound + eval_exact_multiview 报
mmpose V2 与 GT-2D V2。保留 AMASS 随机视角训练协议。
```

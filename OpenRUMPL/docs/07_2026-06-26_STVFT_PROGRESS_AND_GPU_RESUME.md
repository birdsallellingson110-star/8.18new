# ST-VFT 进度清单 + 文件位置 + GPU 恢复操作指引

> 更新: 2026-06-24 · 给 GPU 空出来后的接续操作用。
> 配套: [EXPERIMENT_PROGRESS_AND_PLAN.md](EXPERIMENT_PROGRESS_AND_PLAN.md)(总览)

---

## 〇、排错记录: "64mm 初始精度差" 真因 (2026-06-24)

**现象**: ST-VFT eval 里 baseline(gate=0)= 60-64mm,而 baseline 在 12w synth val 只有 14mm,差 4.5×,疑似数据有问题。

**逐个排除(每步实测,多次推翻自己的猜测):**
1. ❌ "动作难/数据差" — 实测 HRNet vs GT 像素误差: clip median 14.3px vs 12w 14.4px **几乎相同**(仅尾部 p90 150 vs 109 略差)。
2. ❌ "相机设置不同" — 实测相机到人距离 clip 3.05m vs 12w 3.06m **相同**, fx 同一 CMU 相机池。
3. ❌ "骨架定义不一致毁了权重" — 实测 baseline 预测骨架(躯干0.513)= clip GT(0.512)**完全一致**; 模型从射线三角测量、自适应输入骨架, 不强加先验。(注: clip 用 J_regressor_coco=正确coco; 12w 髋偏高是它当时用了不同regressor, 但**无害**)
4. ❌ "14mm 是 GT-2D 不公平" — 查训练 log: `USE_MMPOSE_VAL: True`, 14mm **也是 HRNet**。
5. ✅ **真因 = 视角数口径不同**: 实测 clip gate=0 中心帧 **V=5=14.0mm(中位)= 12w 的 14mm**; **V=2=42.7mm = baseline CMU V=2 的 42mm**。12w synth val 用 `TEST_VIEWS=[0,1,2,3,4]`=**V=5**; ST-VFT eval 用 k∈[2,5] 随机(混大量 V=2)。**拿少视角的数比多视角的数,当然差。**

**结论**: **clip 数据 = 12w 数据(V=5 都 14mm),完全没问题,不用重新生成。** 64mm 是少视角(V=2)难场景——而那正是 ST-VFT 要解决的(少视角下用时序补几何)。
**教训**: 比数字前先对齐口径(视角/2D源/关节法/单位);先实测再下结论。

## 〇.5、评估口径已对齐 12w (2026-06-24)

排错坐实"64 是少视角口径"后,把 ST-VFT 评估改成**和 12w/baseline 完全同口径,只保留时序训练加项**:
- **固定视角数**(不再 k∈[2,5] 随机): `--eval-views 2 5`。V=5 对照 12w synth val(TEST_VIEWS=5→14mm);V=2 对照 CMU(40.4)=研究口径。`make_collate_fixed_views`。
- **中心帧单帧**(不是 per-t 平均): per-t/MPJVE 是时序训练 **loss 加项**, 只在训练用; eval 用中心帧才和 12w 单帧可比。
- **Abs MPJPE All-17 + KP\***(KP* 用固化的 `lib/utils/kp_star.py`,与 baseline/B/C 一致)。
- best-ckpt 判据 = `eval_views[0]`(默认 V=2)的 All-17。
- 验证: gate=0 在 V=5 = 11.5mm(≈12w 14mm,口径对上); 每 epoch 报 `V2[all17/kp] V5[all17/kp]`。
- 代码: `run/train_stvft.py` (`eval_center`/`eval_all_views`/`--eval-views`), `lib/dataset/stvft_dataset.py` (`make_collate_fixed_views`)。

**之前所有 batch2 诊断数(60→46 等)是旧口径(per-t k∈[2,5]),作废为正式数**——正式训练用新口径重出 V2/V5 的 All-17+KP*。

## 〇.7、CMU 金标准结果 + 架构结论 (2026-06-26)

**生成了 CMU 连续帧 eval 数据**(pose5/6, skip=1 抽 300 连续帧 → preprocess → HRNet → mmpose匹配 → 轴swap),数据在 `/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/..._swapv3/cmu_panoptic_validation.pkl`。生成脚本: `MHP/extract_frames_cmu.py --skip 1 --max-frames 300`,后续步骤同 `run_cmu_singleperson_eval.sh`。eval 脚本 `RUMPL/run/cmu_eval_bvc.py`(按 pose_id+image_id 分组, 滑窗 L=5, V=2/V=5, Abs, median+mean, KP*, B(gate=0) vs C(时序))。

**B vs C 结果(cold-start 训练的 ST-VFT, stvft/coldstart_aligned):**
- **数据管线验证 ✓**: B(gate=0) V=2 = All-17 med 37.5 / KP* med 43.6 ≈ 已确立 baseline(37.9/40.4)→ 连续帧数据正确。
- **金标准判决: 时序在 CMU 真实数据上不帮、略伤**: V=2 增益 B-C = All-17 **-0.8** / KP* **-1.2**mm; V=5 ~-0.2。

**诊断(根因不是可修的bug):**
- 帧率 bug **已排除**: cmu_eval dt 用真实 fps=30, dt_encoder 连续正弦, 不是硬编码120。
- 真因 = **domain gap + 架构位置**: 当前 TFT 在 **view-fusion(VFT)之前、per-view 的 2D 派生 ray 特征**上做时序 → 继承 per-view 2D 噪声; 真实数据 per-view 信噪比~1, 时序信号噪声主导 → TFT 放大噪声 → 略伤。合成数据(HRNet误差小,时序信号干净)训练的 TFT 迁移不到真实数据(时序信号带噪)。

**架构结论(下一方向, 路B):**
> 当前 ST-VFT 把时序融合放在 VFT 之前、在 per-view 2D 派生特征上做; 真实数据 per-view 信噪比不支撑。时序融合应放在 **VFT 之后**, 在多视角去噪后的特征空间(VFT输出 = `(B,J,768)` 富特征, 非3D坐标)做 —— 那里 SNR 更高、且能从邻帧稳健借信息补欠定帧。
> 实现 = 换序: `encode → VFT(逐帧跨V融合, 得(B,J,L,768)) → TFT(跨L) → PFT → 3D`(当前是 encode→TFT→VFT→PFT)。
> **保留(待验, 非保证)**: 真实V=2"有无可恢复欠定"仍需实验答; 若坏帧在所有视角同时坏(真实遮挡常见), 邻帧VFT输出也坏, 时序借不到 —— 这个 corner case 实验里要盯。

## 〇.8、路B (STVFT v2) 架构决策 — 钉死再动手 (2026-06-26)

学自 PoseFormer(`/home/lixiaob/cjy/reference/PoseFormer`, RUMPL 本就基于它): 顺序 Spatial→Temporal→`weighted_mean`(Conv1d L→1, 收成中心帧)→head; 无 gate。

**决策1 — VFT 逐帧, 路B 天然成立**: 当前 `vft_forward(x)` 吃单帧 `(B,J,V,768)`→`(B,J,768)`, forward 里只在中心帧调一次。路B = 对全部 L 帧各调一次(可 batch: `(B·L,J,V,768)` 一次过)→ `(B,J,L,768)`。不用拆, 直接多调。

**决策2 — Temporal 维度**: C_vft=**768**。Temporal **按关节做**(token=768, 跨L), **不**学 PoseFormer 展平关节(17×768=13k 太大)。`weighted_mean = Conv1d(L→1)` 在 L 维、对每 (B,J)。

**决策3 — 训练策略 + 替代 gate 的关键 trick**: finetune(继承 baseline encode/VFT/PFT 权重, Temporal 随机初始化)。**weighted_mean 初始化成"只选中心帧"(one-hot [0,0,1,0,0])** → 初始输出==中心帧VFT==baseline → 从 baseline 起步学时序加权。**这干净给了"从baseline起步"的性质, 无 gate 冷启动/退化解病理**(weighted_mean 自学各帧权重, 不需要"要不要用时序"的门控)。

**路B 数据流**: `rays→encode→VFT(逐帧跨V, (B,J,L,768))→Temporal(per-joint 跨L, 自适应)→PFT→head`。

**决策4 — 防退化解(核心, 钉死)**:
- **退化解根因**: 当前固定/全局聚合在"帧质量不均"数据上必向均值平滑(易帧被邻帧稀释)。合成数据帧质量太均匀→模型没见过"某帧崩邻帧好"→学不会选择性借→迁移不到CMU(真实帧质量不均)。
- **★命门 = 帧质量扰动训练**(制造"落差", 非加噪): 随机让窗口里**一帧**崩、其余干净。
  - 扰动覆盖两种失败模式: **模式1 低conf崩**(大偏移+conf×0.1) + **模式2 高conf错**(大偏移+conf不变), 各~50%。[防模型只学"信conf"; 真实HRNet会自信地错]
  - 两种情景分开(非同时多帧崩, 否则无帧可借): **情景A 中心帧(t=2)崩**(学"中心崩借邻帧") + **情景B 某邻帧([0,1,3,4])崩**(学"邻帧崩别被污染"), 各~50%。
- **自适应机制 = 时序一致性检测为主**(某帧与邻帧严重不一致=可疑→降权; 对模式1/2都成立) **+ conf(b1: conf作attention对K/V降权)为辅**。**不**用固定 weighted_mean(不均数据上会退化), 用 **center-query 时序注意力**(逐样本自适应)。
- **Loss = Huber/robust**(防离群帧主导, 不学"压离群"退化解)。

**保留(待验, 诚实)**: ① 真实V=2有无可恢复欠定 ② 坏帧全视角同坏时邻帧VFT也坏借不到 ③ 模拟扰动分布≠CMU真实HRNet失败分布, 最终仍需CMU验证。

**PoseFormer 精髓复用 vs 我们的新增**: PoseFormer(H36M, 单视角, 均匀质量)验证过的 = **架构骨架(spatial→temporal→收中心帧)+ 训练recipe(dims/depth/lr/aug)** → 直接抄, 不重造。但 PoseFormer **没面对"帧质量不均"**(H36M均匀)→ 决策4(扰动+时序一致性选择性)是**我们超出PoseFormer的新增, 它给不了现成解**。即: 骨架学PoseFormer, 质量不均处理是自研(未验证领域)。

## 一、核心结论(已用真实 stdout 踩实)

**时序有用 — 硬证据(synthetic val):** 强制 gate=1 满接入、冻结 VFT/PFT、只训 TFT,
held-out val per-t pos 从 **gate=0 地板 60.10mm 稳定单调降到 43.90mm(~27%)**,train/val 同步、不过拟合、无 NaN。
- 区别于早期假信号(那是 gate≈0=baseline+噪声);这次 gate=1 是 TFT 真接入。
- best 权重: `/mnt/data/cjyoutput/stvft/full_gate1/stvft_best.pth`(val 43.90)

**严格界定(别越界):**
- ✅ 证明: 时序天花板存在且可观(~27%)。
- ❌ 未证: learnable gate 自达天花板(冷启动中);CMU 真实有用;`44mm per-t synthetic` ≠ `40.4mm KP* CMU`(不同指标/数据,别混)。

**baseline 复现(早先已踩实):** CMU V=2 带 conf, Abs All-17=37.9mm / KP*=40.40mm
(脚本打印+交叉验证)。详见 memory `stvft-progress-and-next`。

---

## 二、任务清单(✅完成 / 🔄进行 / ⏳待GPU / 📋待办)

| 状态 | 任务 | 备注/位置 |
|---|---|---|
| ✅ | baseline 复现(RUMPL conf) | model_best 见下;CMU V=2 KP*=40.40 |
| ✅ | 写死 center 缺陷修复 | forward 加 `t_target`,残差基随 t(per-t 对齐) |
| ✅ | gate=0 旁路恒等检验 | `run/check_gate0_identity.py`,1.6e-5 |
| ✅ | ray 路径一致性核验 | `run/check_ray_paths.py`,两路逐元素 1e-6 一致 |
| ✅ | 46mm 之谜结案 | clip 数据更难(HRNet 在极端姿态渲染上检测差 ~34mm + 几何),非 bug |
| ✅ | **时序有用(天花板)** | gate=1 满接入 val 60→44;`stvft/full_gate1/stvft_best.pth` |
| ✅ | **NaN 根因坐实+根治** | = PyTorch SDPA **efficient 后端 backward 的 nan bug**(autograd anomaly 点名 `ScaledDotProductEfficientAttentionBackward0`);修=tft.py 强制 math 后端(`sdp_kernel(enable_math=True,其余False)`)。已验证: 临界态ckpt 80 batch(含 pos<15mm)零 nan。(sqrt-eps 是误判,非根因) |
| ✅ | **破 ReZero 冷启动成功**(math修复后) | learnable gate init 0.1, 40ep 零 nan; gate 自爬 0.1→**0.50**(单调,不靠手动设gate); **val best 45.86mm**(baseline 64.36, 天花板~44, 差~1.9mm; 末态噪声 45.86-50.32)。权重 `stvft/coldstart_mathfix/stvft_best.pth`。结论: learnable gate 自达近天花板=方法自洽。**注: batch2 诊断数字, 非可报数字** |
| 🔄 | h36m 3D 转换 — **时序clip**(CPU) | `MHP/05_add_h36m_3d.py`;Kabsch 残差 0.0000 验证通过;clip pkl 自带 smplh 参数 |
| ⏳ | h36m 转换 — **单帧12.8w**(GPU) | 最终 pkl 无 smplh 参数;stage_IV 有但 117k≠128k 难匹配 → **不能 CPU 强匹配(会配错)**;干净做法见步骤4 |
| ⏳ | **正式训练(batch 16)** | 等 GPU 空 ≥13GB;现 batch 2 仅诊断,数字不可报 |
| ⏳ | CMU 连续帧 eval 数据生成 | 需 GPU(render+HRNet);金标准验证前置 |
| 📋 | CMU 金标准验证(B vs C) | `run/cmu_temporal_eval.py`(需上面的连续帧 pkl) |
| 📋 | h36m 2D(训练输入) | HRNet→h36m 映射 or 重检测,需 GPU/决策 |
| 📋 | 三数据集对标(CMU+RICH+H36M) | 论文终局 |

---

## 三、关键文件位置

**数据:**
- clip 训练数据(3000 clip,coco): `/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl`(300 个)
  - 每 pkl 含: joints_3d(N,27,17,3米), joints_2d_mmpose(HRNet), confs_2d_mmpose, joints_2d_amass(GT投影), camera_parameters_all, smplh_*(参数), frame_rate, source_npz
  - validation 目录**空** → 训练时从 train 固定种子留 200 当 held-out val(`--val-clips`)
- h36m 3D GT 输出(CPU 转换中): `/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train_h36m3d/*_h36m3d.pkl`

**权重:**
- baseline(RUMPL conf, 40.4mm): `/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar`
- ST-VFT 天花板(gate=1): `/mnt/data/cjyoutput/stvft/full_gate1/stvft_best.pth`(val 43.90)
- 破冷启动(进行中): `/mnt/data/cjyoutput/stvft/full_coldstart/stvft_best.pth`

**代码(本轮新增/改动):**
- `run/train_stvft.py` — 改: `t_target`/`--fixed-gate`/`--gate-init`/`--grad-clip`/NaN守卫/best-ckpt/held-out val/`eval_val`
- `lib/models/stvft/stvft_pretrained.py` — 改: forward `t_target`+任意 `gate_override`;gate `gate_init`
- `lib/utils/kp_star.py` + `run/eval_kp_star.py` — KP* 固化(baseline/B/C 共用)
- 诊断脚本: `run/check_gate0_identity.py`, `check_pert_gate0.py`, `check_ray_paths.py`, `check_2d_source.py`, `check_clip_continuity.py`
- `MHP/05_add_h36m_3d.py` — h36m 3D 转换(纯 CPU)

**日志:**
- 破冷启动: `/mnt/data/cjyoutput/stvft_coldstart_train.log`
- gate=1 天花板: `/mnt/data/cjyoutput/stvft_gate1_train.log`
- h36m 转换: `/mnt/data/cjyoutput/h36m_3d_convert.log`

---

## 四、GPU 空出来后怎么做(按优先级)

**环境(每次先 source):** `source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh`;`cd OpenRUMPL/RUMPL`
**显存:** batch 16≈13.3GB / batch 4≈4.4GB / batch 2≈2.9GB(实测)。一张空 4090 够 batch 16;两张空可 batch 32(DataParallel,快一倍,但 train_stvft 当前单卡,要先改)。

### 步骤1 — 正式破冷启动训练(batch 16,learnable gate)
等 GPU ≥13GB 空闲,把诊断的 batch 2 换成 batch 16 正经训:
```bash
CUDA_VISIBLE_DEVICES=<空卡> python run/train_stvft.py \
  --data-glob "/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl" \
  --save-dir /mnt/data/cjyoutput/stvft/full_coldstart_b16 \
  --pretrained-ckpt /mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar \
  --pretrained-cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --freeze-backbone 1 --gate-init 0.1 --gate-lr 1e-3 --lr 3e-5 --grad-clip 0.5 \
  --epochs 40 --batch-size 16 --L 5 --val-clips 200 --eval-views 2 5
```
(SDPA math 后端已修 nan; eval 与 12w 同口径: 固定视角 V2/V5 + 中心帧 + Abs All-17/KP*)
判据: gate 从 0.1 自爬(>0.05+) + **V=2 的 All-17/KP\* 降到 baseline V=2 以下**(时序在少视角下的增益=研究点);V=5 应≈12w 14mm。
- 若爬不起来 → 上两阶段: 先 `--fixed-gate 0.3`(或更高)训 TFT 几 epoch,再用其 ckpt 续训 learnable gate。

### 步骤2 — gate=1 天花板正式复现(batch 16,出可报数字)
```bash
# 同上但: --fixed-gate 1.0 --lr 3e-5(去掉 --gate-init/--gate-lr),save-dir 改 full_gate1_b16
```
batch 2 的 43.90 是指示性;batch 16 出可报的天花板数字。

### 步骤4 — 单帧 12.8w 的 h36m 转换(GPU,为 h36m baseline 备料)
单帧最终 pkl 不带 smplh 参数,CPU 无法可靠匹配(117k stage_IV ≠ 128k 最终)。干净做法 = 重跑单帧 pipeline 出 h36m:
```bash
# 02 已支持 --regressor both/h36m (line 116). 重跑单帧 render 出 coco+h36m 3D + h36m 2D.
# 数据: paper_single_cmu/stage_IV(有完整 pose/trans/betas), 输出到新 exp 名
bash MHP/run_step2_*.sh  # 加 --regressor both, 详见 02_clip_run.py / run_mmpose_02
```
注: 时序 clip 的 h36m 2D(训练输入)同理也待定 — GT 投影 or HRNet→h36m 映射,需决策。

### 步骤3 — CMU 金标准验证(最终证据)
前置(需 GPU): 生成 CMU 连续帧 eval pkl(skip=1 + HRNet),供 `run/cmu_temporal_eval.py --cmu-pkl`。
- CMU 测试图/GT: `/mnt/data/dataset/cjy/tempo try/data/panoptic/test/`(4 序列, 含 calib+hdPose3d GT)
- 注意 Δt 尺度: CMU 30fps vs 训练 120fps(脚本已按 fps 算 Δt)
- 跑出 B(gate=0) vs C(完整) 的 KP*,这是论文核心数字(用 `lib/utils/kp_star.py` 算 KP*,保证可比)

---

## 五、纪律(贯穿,见 memory `discipline-script-output-is-truth`)
脚本打印的数才算真返回;任何聚合(均值/换算)标明是算的+附算式;不把"应完成"当"已完成";
判据先定死再跑;一次只动一个旋钮。

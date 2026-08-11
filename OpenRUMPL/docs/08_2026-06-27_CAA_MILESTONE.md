# CAA 里程碑固化(2026-06-27)

## TL;DR
**方向从"时序优化"转向"conf-aware 视角融合(CAA)"。CAA conf-attention bias(零训练)在 CMU V=5 上拿到 −2.9mm 真实增益(两序列验证)。V=2 受几何限制(2视角无法降权)只 −0.54mm。VFT 微调是陷阱(过拟合 AMASS、伤 CMU)。**

---

## 1. 核心结果:CAA 零训练 CMU 增益

机制:在 RUMPL 的 VFT 跨视角 attention 上乘 `conf_weights = 1 − λ(1−conf)`(用 backbone 预留的 `Attention.conf_weights` 接口,baseline 从未启用)。λ=0 → conf_weights=1 → 精确==baseline。**无需训练**,手设 λ 即生效。

口径(紧贴官方 `function_rumpl.evaluate`):**pelvis-anchored(COCO mid-hip=(J11+J12)/2)+ MEAN + cm(OUTPUT_IN_METER ×100)**,官方 `calc_mpjpe`。单帧(L=1)。

**CMU 官方 rel-mean (cm),原始 baseline + CAA(冻结 VFT,零训练):**

| | baseline λ=0 | CAA λ=1 | 增益 |
|---|---|---|---|
| V=2 (全体) | 5.223 | 5.169 | −0.54mm |
| V=5 (全体) | 3.704 | 3.414 | **−2.90mm** |

**按序列拆分(可靠性验证,排除"幸运相机"):**

| | pose5 λ=1 | pose6 λ=1 |
|---|---|---|
| V=5 ΔAll17 | −1.97mm | −3.88mm |
| V=2 ΔAll17 | +1.13mm(伤!) | −2.27mm |

- **V=5:两个独立序列在 λ=1 都改善 → 真增益**(若是偶然会一改善一变差)。量级随序列变(−2~−4mm)。每相机 conf 无系统性偏低(3:0.876/6:0.825/12:0.833/13:0.843/23:0.882)→ 非"恒定 downweight 某相机"。
- **V=2:脆弱,最优 λ 在序列间冲突**(pose5 喜欢 0.5、pose6 喜欢 1;λ=1 伤 pose5)。固定 λ 无解 → 需自适应(见第 5 节 A)。
- **λ≥2 普遍崩**(冻结 VFT OOD)→ 可用范围窄,也证明不能微调 VFT。

---

## 2. 反面结论(同样重要)

### VFT 微调是陷阱(不是简单过拟合,是坐标系迁移)
微调 VFT(rel-loss)在 AMASS V=2 涨 **−11.6mm**,但 CMU V=2 **+1.85mm 变差**、V=5 仅 −0.69mm。
推测:rel-loss 把 VFT 拉向 AMASS 的坐标系约定(pelvis 定义/骨架比例),CMU 坐标系不同 → 偏移。V=2 缺多视角冗余去抵消,伤得更重。**结论:冻结 VFT。**

### V=2 是几何限制,不是实现问题
V=2 downweight 一个视角 = 退化成单视角 → MPJPE 爆炸 → CAA 不敢真降权 → 增益天然弱。CAA 在视角越多时越强(V=5 有 3-4 个好视角可依靠)。

### 时序在多视角下天花板低(已封存)
- attention map 实证:我们的时序模块只学到 ≈均匀/位置性平均,非内容自适应。
- 上限实验(零训练):平均类聚合 ~1-2mm,oracle 借帧 9mm 但 **conf 选帧 +15mm(失效)** → 大头卡在"缺可靠性信号"。
- 根因:多视角已消解单帧歧义,时序信息大体冗余;PoseFormer/MTF/SVTformer 文献一致时序天花板 0.7-2mm。
- 我们的"−8.8mm AMASS 时序增益"是 **eval window 随机化 bug**(baseline 算一次/C 每 epoch 重算,比在不同窗口)虚高的;修复后(`eval_fixed_window`)≈ 平均水平。

---

## 3. 评估口径(必须一致,否则数字不可比)
- 指标:**pelvis-relative(COCO mid-hip)MEAN MPJPE,cm**。官方 `calc_mpjpe`,`OUTPUT_IN_METER ×100`。**不是 absolute、不是 median**(早先用错过,导致误判)。
- KP* = 排除头(0-4)+髋(11,12)的 10 关节。
- CMU 数据:`.../annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl`(pose5+pose6,5相机[3,6,12,13,23])。
- V=2=[3,6],V=5=全 5 相机。窗口 B/C 必须同窗口(`eval_fixed_window=True`)。

## 4. 复现命令
```bash
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh; cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
export CUDA_VISIBLE_DEVICES=1
# CAA λ-sweep(零训练, 按序列): 复现 V=5 −2.9mm
python -m run.caa_sweep \
  --rumpl-ckpt /mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar \
  --L 1 --lams 0 0.5 1 2 3 --per-seq
# 时序上限实验
python -m run.temporal_upperbound --rumpl-ckpt <同上> --L 9
```
关键文件:`lib/models/stvft/stvft_v2.py`(CAA: caa_scale + vft_forward conf_weights),`run/caa_sweep.py`,`run/cmu_eval_v2.py`,`run/temporal_upperbound.py`。

## 5. 下一步 A:V=2 自适应学习版(冻结 VFT)
动机:V=2 最优 λ 随样本差 2 倍(pose5 0.5 / pose6 1)→ 固定标量无解 → 学每样本自适应强度。
**预期 V=2 −1~−1.5mm(几何限制下的上限,到不了 V=5 的 −3mm)。−0.8mm 也算预期内,非失败。**

**预防措施(防 AMASS-specific 陷阱,同 VFT 微调教训):**
1. 只训 conf→weight 小网络(几百参数),**不动 VFT**。
2. **CMU 序列做 val**(哪怕只 pose5/pose6),早停看 CMU val 不是 AMASS。
3. 训练用 view-perturbation 造"哪个视角该降权"的样本。
4. 回退保护:若 conf 网络在 CMU 上比固定 λ=0.5 还差,回退固定 λ。

## 6. 给老师简报
> 时序方向阶段结论:多视角场景下时序贡献天花板 0.7-2mm(PoseFormer/MTF/SVTformer 文献一致),我们 ST-VFT 持平符合 trend。
> 新发现:CAA conf-attention bias(零训练)在 CMU V=5 上 −2.9mm 真实增益,两独立序列验证(pose5 −1.97mm, pose6 −3.88mm)。VFT 微调反而过拟合 AMASS、伤 CMU,证明冻结 VFT 正确。
> 下一步:V=2 受几何限制(2视角无法降权),固定 λ 在不同样本最优值差 2 倍,正做自适应学习版,预期 V=2 −1~−1.5mm。

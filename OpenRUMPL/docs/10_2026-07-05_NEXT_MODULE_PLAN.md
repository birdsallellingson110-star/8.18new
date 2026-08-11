# 下一模块规划(在蒸馏baseline之上)— 2026-07-05

## 0. 当前主线(baseline for next module)

**方法 = 多视角→少视角自蒸馏**(train时 teacher=全视角 no_grad, student=随机k视角, loss=`‖student−teacher‖`输出级 + `‖student−GT‖`)。
**故事 = 对相机配置鲁棒的少视角3D姿态**(RUMPL对相机俯视几何敏感 → 蒸馏注入满视角先验 → 跨配置鲁棒)。

**结果(CMU pose5/6, V=2, 全10种相机对, abs KP\*):** 配置平均 -2.02mm(3-seed)/ seed1单模型 -2.42mm 全10配置降。最坏几何配置(高俯视相机对如[6,12]base67, [12,13])改善最多(-5~-6mm)。

**已定诊断:** ① config敏感=相机俯视驱动(corr高度0.68, 低相机2D 10px vs高17px)② 蒸馏helps上肢hurts下肢 ③ 瓶颈=per-view 2D质量, RUMPL用VFT学习融合+scalar conf。

**进行中:** leg-weight扫描(0.0/0.3/0.5/0.7, 腿降蒸馏权重治[3,13]微退+0.45)。兜底=seed1(全降)。

---

## 1. 三个候选模块(实际读了代码, repo已克隆到 /mnt/data/cjyoutput/{DeProPose,S2DHand,DVGaze})

### ① DeProPose — 重投影误差可靠度加权融合 ⭐推荐先做
- **代码**: `DeProPose/model/pose3D_model.py` forward(~183-253)。机制:每视角各出3D→重投影回该视角2D算误差`losses[j]`→`weights[j]=(1/losses[j])/归一化`→`fused=Σ weights[j]·views[j]`。**哪个视角3D重投影越准=越可靠=权重越大,自动降权坏视角。**
- **对RUMPL**: VFT融合时按几何可靠度加权。两版:
  - **A(简, input端)**: 可靠度=ray几何一致性(`ray_pairwise_residual`已有)+conf → 喂VFT的`conf_weights`hook加权cross-view attention。无需3D,便宜gate。
  - **B(忠实, 2-pass)**: RUMPL前向出初步3D→重投影每视角→per-view重投影误差→重加权→refine。更强更对症,代价第2次前向。
- 类型架构/推理; 直接对症config故事; 有现成基础设施(conf_weights hook + ray_pairwise_residual 都在 stvft_v2.py)。

### ② S2DHand — 无监督测试时适配 ⭐更大贡献的备选
- **代码**: `S2DHand/adapt_detnet_dual.py` + `consistency.py`。机制:测试时(无GT)用**跨视角consensus**(两视角对同一3D达成一致→consensus当伪标签)+**transformation invariance**(`R_from_2poses`恢复视角间旋转,该变换应一致)→自监督fine-tune→per-config适配。
- **对RUMPL**: CMU测试时用2视角consensus适配到**每个具体相机配置**→直击config鲁棒性,跨数据集(AMASS→CMU)友好无需标注。
- 类型测试时(新贡献类别); novelty高; 与蒸馏**很互补**(蒸馏=通用先验, TTA=config专属适配)。工程量中(per-config fine-tune循环)。

### ③ DVGaze — pass
- `DVGaze/Code/eth/model.py`: 相机pos编码transformer融合 + 双视角一致性loss(MSE)。RUMPL的VFT本就是ray几何编码融合(相似); 可借的"双视角一致性loss"和我们蒸馏的跨视角一致**重叠**, novelty弱。**不做。**

---

## 2. 推荐路径(两步)
1. **先 DeProPose 可靠度融合(方案A gate)**: 最便宜、架构、stacks蒸馏、基础设施现成。验证"可靠度加权对config有增益"。有信号→上方案B(2-pass重投影,更强)。
2. **要更大故事 → S2DHand 测试时适配**: "训练期蒸馏(通用先验)+ 测试期consensus适配(config专属)" = 完整强故事,直击config鲁棒性。

**判定口径**: 全部沿用 CMU pose5/6 V=2 全10配置 abs KP\*(配置平均 + 最差配置 + 是否全降), 叠在蒸馏baseline之上比。

## 3. 已排除的死路(勿重试)
骨长约束(2D噪声地板)、时序、换2D检测器、RCG(输出级冲突太多)、特征级/合并蒸馏(更差,伤[3,12])、DVGaze一致性loss(与蒸馏重叠)。

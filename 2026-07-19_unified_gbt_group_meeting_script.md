# Distill-RUMPL + Unified Geometry-Biased Transformer 组会讲稿

## 一句话概括

我们已经严格复现 RUMPL，并验证了 full-view 到 sparse-view 的蒸馏能够提升困难相机组合。下一阶段不是重新设计一套网络，而是把 Geometry-Biased Transformer 已经验证有效的 confidence bias 和 geometry bias，严格移植到 Distill-RUMPL 的 View Fusion Transformer 中，让同一个模型统一处理任意少视角输入。

---

## 一、研究基础

老师，我们目前的工作可以分成三个阶段。

第一阶段是 RUMPL 复现。RUMPL 把二维关节和相机参数转换成三维 ray，再通过 View Fusion Transformer 融合不同相机，因此理论上能够处理任意相机数量和相机配置。这个 baseline 我们已经完成严格复现。

第二阶段是 sparse-view distillation。训练时使用完整视角预测作为 teacher，再让少视角 student 学习 full-view 的三维输出。进一步通过 hard-view mining，选择与 teacher 差距最大的相机组合进行训练。实验已经证明，蒸馏对困难两视角组合有明确提升。

因此现在的研究起点不是原始 RUMPL，而是已经验证有效的 Distill-RUMPL。

第三阶段就是本周开始的工作: 把 Geometry-Biased Transformer 的核心 attention 模块加入 Distill-RUMPL。

---

## 二、当前要解决的问题

原始 RUMPL 虽然把 confidence 和 ray 都作为输入特征，但 attention 权重仍主要由网络学习到的 QK 相似度决定。

这存在两个问题。

第一，网络不一定会稳定相信高置信度视角。confidence 虽然进入了 token embedding，但经过多层特征映射后，它对最终 attention 的作用不可控。

第二，attention 没有被强制满足真实三维几何。两个视角的特征即使在三维空间中不一致，只要 QK 特征相似，仍可能获得较高权重。

Geometry-Biased Transformer 的核心价值，就是把这两种已知先验直接加入 softmax 前的 attention logits。

---

## 三、论文模块的严格公式

普通 Transformer attention 是:

`A = Softmax(QK^T / sqrt(d))`。

论文将第 l 层 attention 改成:

`A_l = Softmax(QK^T / sqrt(d) + eta_l^2 M_conf - gamma_l^2 M_dist)`。

这里有三部分。

第一部分 `QK^T` 是网络学习到的特征相关性。

第二部分 `M_conf` 是 confidence bias。对于所有 key token，把二维检测器给出的置信度沿 query 维重复。这样无论当前 query 是什么，高置信度 key 都会提高 attention logit。

第三部分 `M_dist` 是 geometry bias。它由任意两条三维 ray 的最近距离组成。两条 ray 越不一致，距离越大，attention 惩罚越强。

`eta_l` 和 `gamma_l` 不是人工固定超参数，而是每个 Transformer layer 独立学习。论文通过平方保证系数非负，因此 confidence 永远是奖励项，ray distance 永远是惩罚项。

这也是我们这次重新实现的重点。

---

## 四、为什么在 softmax 前简单加减是合理的

这个公式虽然形式上只是加减，但经过 softmax 后可以写成:

`A_ij proportional to exp(Q_i K_j) * exp(eta_l^2 c_j) * exp(-gamma_l^2 d_ij)`。

也就是说:

- 学习特征决定基础相关性；
- confidence 形成乘法可靠性增益；
- geometry distance 形成指数衰减。

所以它不是简单的经验加权，而是在 attention 概率中显式加入观测可靠性和三维几何先验。

---

## 五、怎样移植到 RUMPL

我们不复现 Geometry-Biased Transformer 的完整 encoder-decoder，也不替换 RUMPL 主体结构。

具体做法是，在 RUMPL 每个关节的 View Fusion Transformer 中加入论文的 biased attention。

对于同一个关节，不同相机提供多条 ray。我们先构建这些 ray 两两之间的 `M_dist`，再把 confidence 和 geometry matrix 传入每一个 VFT block。

每一层分别学习自己的 `eta_l` 和 `gamma_l`。浅层和深层可以根据任务自动决定依赖 confidence、geometry 还是 learned QK，而不是所有层共用一个手工系数。

RUMPL 的 fusion token 没有对应的真实 ray，因此严格版本不人为给 fusion token 构造虚拟距离。geometry bias 首先约束真实 view token 之间的交互，fusion token 通过多层 attention 间接读取经过几何约束的 view features。

后续如果增加 fusion consensus bias，它必须作为我们自己的扩展单独消融，不能和原论文模块混在一起。

---

## 六、统一处理少视角，而不是分别针对 V2/V3/V4

原 Geometry-Biased Transformer 使用同一个模型处理不同相机数量，并没有为两视角、三视角和四视角手工设置三套 scale。

因此我们新的严格实现也遵循同样原则:

- K=2、K=3、K=4 以及训练中的其他视角数量共用同一个模型；
- 共用同一组逐层学习的 `eta_l` 和 `gamma_l`；
- 不读取任何 `V2_SCALE`、`V3_SCALE` 或 `V4_SCALE`；
- 模型根据输入 token 和训练数据自行学习不同视角数量下的融合策略。

之前测试的 K-aware 固定 scale 只作为 preliminary diagnostic，不进入这一阶段的正式主结果。

---

## 七、Ray geometry 如何计算

对于非平行 ray，我们使用两条三维直线的最近距离。它与论文的 Pluecker ray distance 在数学上等价。

这次实现还补齐了平行和近似平行 ray 的特殊分支。之前直接使用 `cross_norm + epsilon`，可能把平行但不重合的 ray 错误计算为接近零距离。新实现会计算两条平行线之间的真实垂直距离。

同时，严格版本不再对每个样本的距离矩阵除以自身均值。原始 ray distance 直接进入 attention，尺度由每层可学习的 `gamma_l` 自己调整。

---

## 八、严格消融怎样设计

我们固定已经验证的 `hard-view + legw0.9 Distill-RUMPL` 作为统一 baseline。

下一轮只训练三组模型:

1. Distill-RUMPL + learnable confidence bias；
2. Distill-RUMPL + learnable geometry bias；
3. Distill-RUMPL + confidence bias + geometry bias。

三组保持完全相同的:

- 训练数据；
- 蒸馏损失；
- hard-view mining；
- leg distillation weight；
- epoch 数量；
- seed；
- checkpoint 选择规则；
- V2、V3、V4 评测协议。

这一阶段不加入 auxiliary multi-K、K-aware scale、fusion consensus 或新的 noise loss，保证能够准确判断论文两个模块各自的贡献。

每组训练过程中，每1000个 iteration 会输出12层当前学到的 confidence 和 geometry 有效系数。训练结束后不仅比较 MPJPE，也要分析不同层主要依赖哪一种 bias。

---

## 九、噪声与 Attention 诊断说明了什么

我们已经对之前的固定 geometry bias 模型做了快速噪声和 attention 诊断。

在 CMU 困难三视角组合 `[3,6,13]` 上，对 ray direction 加入20 px等效二维抖动后:

- 旧 hard-view baseline 的 MPJPE 退化约 6.22 mm；
- 固定 geometry 模型退化约 8.05 mm。

单独扰动相机13时，它的 ray inconsistency 从0.02917增加到0.03689，说明几何模块成功检测到了异常；但相机13的平均 attention 没有下降，反而从0.30068上升到0.31267。

这个负结果说明，固定全层 geometry coefficient 不足以让模型形成稳定的异常视角抑制。学习到的 QK 项仍可能压过固定几何惩罚。

因此这次严格回到原论文的逐层 learnable coefficient 是必要的，而不是形式上的修改。

但是需要强调，当前训练配置的关键点 `NOISE_LEVEL=0`，所以这些结果只能说明现有模型的噪声敏感性，不能用于评价噪声训练是否有效。

---

## 十、Noise training 放在什么时候

正确顺序是先完成 clean 条件下的 confidence、geometry 和二者联合消融。

确认统一 learnable bias 在正常输入上有效后，再增加独立的 noise-training 实验，包括:

- 二维关键点高斯抖动；
- 扰动幅度与 confidence 衰减联动；
- 随机关节缺失；
- 单个相机异常；
- 人体中心平面扰动和绕竖直轴旋转。

最后比较 clean、轻噪声和重噪声下的 MPJPE 曲线，确保鲁棒性不是以牺牲 clean accuracy 为代价。

---

## 十一、Attention map 如何验证

模型已经支持保存每一层 softmax 后的 attention。

正式实验中需要比较:

1. 高置信度和低置信度视角的 attention；
2. ray distance 小和 ray distance 大的 token 对；
3. clean 输入和人工扰动输入；
4. confidence-only、geometry-only 和二者联合模型；
5. 浅层、中层和深层 attention。

除了热图，还要报告:

- attention 与 confidence 的正相关性；
- attention 与 ray distance 的负相关性；
- 人工异常视角的 attention suppression ratio；
- attention 变化与 MPJPE 变化的关系；
- 每层最终学习到的 `eta_l^2` 和 `gamma_l^2`。

这样才能证明模型真正使用了 confidence 和三维几何，而不是只看最终误差下降。

---

## 十二、当前进度

统一 learnable GBT 已完成代码实现和测试。

目前已经验证:

- 相交、平行、异面 ray distance 正确；
- confidence 增大会提高对应 key attention；
  - geometry distance 增大会降低对应 attention；
- 12个 VFT layer 均有独立参数和有效梯度；
- 同一个模型可以直接前向 K=2、K=3 和 K=4；
- 默认关闭新模块时，旧 Distill-RUMPL checkpoint 可以 `strict=True` 加载。

三组严格消融已经进入持久实验队列。当前 preliminary 固定偏置实验完成并释放 GPU 后，confidence-only、geometry-only 和二者联合会自动并行训练，随后使用固定 epoch20 对 V2、V3、V4 共75个相机组合进行评测。

---

## 十三、最后总结

目前论文故事可以这样概括:

RUMPL 解决的是任意相机表示问题，但对困难相机配置仍然敏感。我们首先通过 full-view 到 hard sparse-view distillation 改善少视角监督。然后严格引入 Geometry-Biased Transformer 的 confidence 和 geometry-biased attention，让视角融合显式受到检测可靠性和真实三维 ray 几何约束。

这一阶段使用同一个模型、同一套逐层可学习参数统一处理不同少视角数量。等已有论文模块的贡献被严格验证后，再研究 fusion consensus、noise reliability supervision 等真正属于我们的扩展。

最终推理仍然是一个 checkpoint、一次前向，不需要 teacher、额外相机或模型 ensemble。

---

## 老师可能追问的问题

### 1. 这部分是不是直接使用了别人的创新？

是。Confidence bias 和 geometry bias 属于已有论文模块，所以必须明确引用，并通过严格消融验证它在 Distill-RUMPL 中的作用。我们的贡献不能写成重新发明这两个 bias，而应来自蒸馏框架、RUMPL 迁移方式以及后续经过实验验证的新扩展。

### 2. 为什么不直接使用原论文完整网络？

因为我们的研究基础是已经严格复现并优化的 RUMPL。当前目标是验证其 biased-attention 模块能否进一步改进 RUMPL，而不是换掉 backbone 后重新比较两个完全不同的系统。

### 3. 为什么不继续使用 K-aware scale？

因为原论文是统一模型。K-aware 是我们从 preliminary 结果提出的假设，必须排在忠实模块迁移之后。只有统一 learnable bias 的结果明确后，才能把 K-aware 作为独立创新做公平对照。

### 4. 现在能否声称抗抖动？

不能。当前噪声诊断显示固定 bias 模型仍然敏感，而且训练噪声实际为零。必须完成明确的 noise-training 和 clean/noisy 对照后才能下结论。

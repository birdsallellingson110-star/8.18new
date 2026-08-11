# 论文调研总结 + RUMPL V=2 优化方向(2026-06-28)

读完 reference/ 下全部 10 篇 + 3 代码库。**一个压倒性的收敛主题:能跨域泛化 + 处理少视角的多视角方法,几乎都用"显式三角化 + 骨长/结构约束 + 可学习refine",而 RUMPL 是唯一的异类(纯学习回归3D)。**

---

## 一、按对"RUMPL V=2"的有用度分级

### Tier 1 — 直接对症(显式几何/结构约束/少视角泛化)⭐⭐⭐

| 论文 | 核心 | 对我们的用处 |
|---|---|---|
| **Structural Triangulation**(136650685, Chen SJTU, 有代码) | **闭式解**:多视角2D+相机+**骨长**→约束优化→闭式三角化 | **直接解V=2深度歧义**:骨长约束住欠定的深度自由度。可微、可plug。**最对症。** |
| **MVGFormer**(CVPR, 有代码) | **混合**:AM(可学习,refine各视角2D)+ GM(**learning-free显式三角化**出3D),迭代coarse-to-fine | **out-of-domain显著领先**。证明"3D用显式三角化、2D用学习"是跨域正道。RUMPL正缺这个。 |
| **MV-SSM**(CMU/NUS CVPR25) | SSM(Mamba)多视角融合 + **可微代数三角化**出3D + 残差refine | **在CMU上跨相机/跨数据集泛化强**(+10.8/+15.3)。又一个"显式三角化+学习refine"。 |
| **Generalizable Triangulation**(2110.00280, Bartol, 有代码) | 生成3D假设池 + 打分网络(输入=**归一化姿态+16骨长**=姿态先验)+ 加权平均 | 姿态先验打分**否决深度定错的解**,骨长dataset-invariant→跨域。8.8%提升。 |
| **MoViSense**(IEEE Sensors26) | MSFF空间+DHF-DMA时序 + **BCM骨长约束** + **显式三角化** | **专攻sparse camera(=少视角!)**,CMU MPJPE 27.85。又一个三角化+骨长约束。 |

### Tier 2 — 结构/图(V=2结构信息)⭐⭐
| **SGraFormer**(Deep Semantic Graph, AAAI24) | 把**骨骼图(关节邻接+骨边)注入transformer attention**→结构感知→缓解深度歧义 | 结构先验注入attention(非loss)。可借鉴改RUMPL的PFT。 |

### Tier 3 — 时序(已证此范式下死路,但记录)⭐
| **SVTformer**(00757, AAAI25, 已克隆) | 空间-视角-时序解耦,Attended SVT PE(+1.5mm) | 同域H36M。时序结论:多视角下冗余、跨域不迁移(我们已实测)。 |
| **MTF-Transformer**(2110.05092, 已克隆) | 多视角时序融合 + view-dropout训练 + CAA(conf) + T_ij旋转对齐 | view-dropout/CAA可借;时序部分增益小(V=2才1.87mm)。 |
| **TEMPO**(CMU) | recurrent多视角时序,tracking+forecasting | volumetric,架构差异大。 |
| **PoseFormer**(已克隆) | 单视角时序lift(消歧)→多视角下此机制冗余 | 时序的"种子",已验证多视角下无效。 |

### Baseline
**RUMPL**:ray编码(几何)→VFT融合→PFT→**Linear头直接回归3D**(无显式三角化,我们查过代码)。优点:ray使跨域比纯学习好;缺点:**学习回归3D在V=2下深度歧义、且仍overfit运动分布**。

---

## 二、核心洞察(收敛主题)

```
能"跨域泛化 + 处理少视角"的方法(MVGFormer/MV-SSM/MoViSense/Structural/Generalizable Tri)都用:
  ① 3D = 显式(可微)三角化 —— 几何, learning-free/闭式 → 数据集/相机无关 → 跨域
  ② + 骨长/结构约束 —— 解剖普适 → 解V=2深度歧义
  ③ + 可学习refine(2D/特征) —— 抗遮挡

RUMPL = 唯一异类: 学习回归3D(无显式三角化、无骨长约束)
  → V=2深度歧义(极端姿态灾难误差, 我们实测mean被尾巴拉到58)
  → 这就是RUMPL V=2弱的根因
```

**和我们所有实验结论吻合**:时序(运动分布特定)不迁移;V=2瓶颈是深度歧义;需要"几何/解剖普适"的信息(像ray)才跨域——**骨长/显式三角化正是这种普适信息。**

---

## 三、给 RUMPL V=2 的方案(按投入排序)

**层次0 — 验证(便宜,先做)**:骨长诊断——V=2高误差预测的骨长是否崩。崩→结构约束有救空间。

**层次1 — 训练创新:骨长一致性loss**。RUMPL训练加 `L_bone=||预测骨长−GT骨长||` + 对称性。不改架构,骨长dataset-invariant→约束跨域。最便宜。

**层次2 — 替换/增强3D head 为显式结构化三角化**(最对标论文,最novel):
```
现在: RUMPL ray特征 → Linear头回归3D(V=2弱)
改成: RUMPL ray特征 → 出refined 2D/置信度 → Structural Triangulation(骨长约束闭式解)→ 3D
  = MVGFormer/MV-SSM思路: 2D用RUMPL学习, 3D用显式结构化三角化
  → V=2深度歧义被骨长约束解决, 且显式三角化跨域(像它们在CMU泛化领先)
```

**层次3 — 假设框架(Generalizable Triangulation思路)**:V=2沿深度歧义方向生成假设→骨长姿态先验打分→加权平均。

**推荐路径**:层次0验证 → 层次1(便宜试) → 不够上层次2(真创新,有5篇论文背书)。

**为什么这方向对(对比时序)**:① 攻V=2真瓶颈(深度歧义)② 骨长/显式三角化几何普适→跨域(像ray,不像时序的运动分布)③ 真architectural创新(RUMPL无显式三角化)④ 5+篇论文一致证明这条路在CMU/跨域有效。

相关:[[v2-extreme-pose-tail-optimization-angle]] [[caa-milestone-and-direction]]。代码可参考:Structural Triangulation、MVGFormer、MV-SSM、Generalizable Triangulation 都开源。

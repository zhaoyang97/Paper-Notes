---
title: >-
  [论文解读] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness
description: >-
  [NeurIPS 2025][人体理解][群体公平性] 本文首次系统研究了优化算法选择对深度学习群体公平性的影响，通过随机微分方程（SDE）分析和两个新定理证明，自适应优化器（RMSProp/Adam）比SGD更容易收敛到公平的极小值点，特别是在数据严重不平衡时。 机器学习在决策系统、风险评估等社会敏感领域的广泛应用使公平性…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "群体公平性"
  - "优化器"
  - "自适应梯度"
  - "随机微分方程"
  - "深度学习公平"
---

# Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness

**会议**: NeurIPS 2025  
**arXiv**: [2504.14882](https://arxiv.org/abs/2504.14882)  
**代码**: [GitHub](https://github.com/Mkolahdoozi/Some-Optimizers-Are-More-Equal)  
**领域**: 人体理解  
**关键词**: 群体公平性, 优化器, 自适应梯度, 随机微分方程, 深度学习公平

## 一句话总结

本文首次系统研究了优化算法选择对深度学习群体公平性的影响，通过随机微分方程（SDE）分析和两个新定理证明，自适应优化器（RMSProp/Adam）比SGD更容易收敛到公平的极小值点，特别是在数据严重不平衡时。

## 研究背景与动机

机器学习在决策系统、风险评估等社会敏感领域的广泛应用使公平性成为关键问题。现有促进群体公平性的技术主要分为三类：预处理（数据增强）、训练中处理（修改损失函数）和后处理（校准输出），但这些方法往往引入额外的计算开销或对训练流程的干扰。

作者提出一个被完全忽视的问题：**优化算法的选择本身是否影响模型的群体公平性？** 优化器是每个深度学习训练管线的基本组成部分，如果特定优化器天然更有利于公平，那就无需额外的公平性增强手段。

已有研究表明SGD训练的模型比自适应方法更鲁棒（Rebuffi et al.），而公平性与鲁棒性常常是竞争目标，这暗示自适应优化器可能在公平性上具有优势。然而，这一关系从未被正式探究。

## 方法详解

### 整体框架

作者采用"理论分析 → 模拟验证 → 实验验证"的三阶段方法论。首先在一个可解析的设定中用SDE分析SGD和RMSProp的公平性行为，推导出闭式解；然后证明两个一般性定理；最后在三个数据集上实验验证。

### 关键设计

1. **SDE分析（Theorem 1）**: 考虑两个子群的简单损失函数 $\mathcal{L}_0(w)=\frac{1}{2}(w-1)^2$ 和 $\mathcal{L}_1(w)=\frac{1}{2}(w+1)^2$，最公平的极小值为 $w^*_{pop}=0$。当子群采样概率分别为 $p_0$ 和 $p_1$ 时，通过Fokker-Planck方程求解SGD和RMSProp的稳态分布，得到：

   SGD稳态分布：$p_{sgd}(w)=\sqrt{\frac{\vartheta}{\pi}}\exp(-\vartheta(w-(p_0-p_1))^2)$，其中 $\vartheta=\frac{1}{8\eta p_0 p_1}$

   RMSProp稳态分布：$p_{rms}(w)=\sqrt{\frac{\kappa}{\pi}}\exp(-\kappa(w-(p_0-p_1))^2)$，其中 $\kappa=\frac{1}{4\eta\Theta\sqrt{p_0 p_1}}$

   **关键结论**：当采样偏差 $|p_0-p_1|$ 超过阈值 $\Delta(p_1p_2,\eta)$ 时，RMSProp收敛到公平极小值的概率高于SGD，即 $\frac{p_{rms}(w^*_{pop})}{p_{sgd}(w^*_{pop})}>1$。

2. **更公平的参数更新（Theorem 2）**: 在各向同性噪声梯度假设下，RMSProp的子群间参数更新差异 $\|D(\nabla\mathcal{L}_0-\nabla\mathcal{L}_1)\|$ 以SGD的对应差异 $\|\nabla\mathcal{L}_0-\nabla\mathcal{L}_1\|$ 为上界。这是因为RMSProp的归一化矩阵 $D$ 的对角元素 $D_{jj}<1$（当 $\Theta^2>\mu^2$ 时），相当于"压缩"了子群间的梯度差异，防止大梯度的子群主导训练动态。

3. **人口统计均衡保证（Theorem 3）**: 在单步优化中，RMSProp引起的人口统计均衡差距的最坏情况增加量，其上界不超过SGD的对应上界。这意味着RMSProp的自适应学习率通过基于历史梯度平方缩放更新，有助于缓解训练中出现的人口统计均衡差距。

### 公平性度量

使用三种广泛采用的公平性标准：
- **均等化赔率（Equalized Odds）** $F_{EOD}$：要求各子群的真正率和假正率相等
- **均等机会（Equal Opportunity）** $F_{EOP}$：要求各子群的真正率相等
- **人口统计均衡（Demographic Parity）** $F_{DPA}$：要求预测标签分布与敏感属性无关

所有指标越高越公平。

## 实验关键数据

### 主实验（ViT骨干，三个数据集）

| 数据集 | 敏感属性 | 指标 | Adam | RMSProp | SGD | 提升(vs SGD) |
|--------|---------|------|------|---------|-----|-------------|
| CelebA | 性别(G) | $F_{EOD}$ | 65.21 | 65.18 | 62.66 | +2.55 |
| CelebA | 性别(G) | $F_{EOP}$ | 99.90 | 99.91 | 96.60 | +3.31 |
| CelebA | 性别(G) | $F_{DPA}$ | 73.50 | 73.68 | 60.80 | +12.88 |
| CelebA | 年龄(A) | $F_{EOD}$ | 72.34 | 71.99 | 68.40 | +3.59 |
| FairFace | 种族(R) | $F_{EOD}$ | — | +9%(vs SGD) | 基准 | RMSProp优势明显 |

### 准确率对比（不以牺牲精度为代价）

| 数据集 | SGD Acc | RMSProp Acc | Adam Acc | SGD F1 | RMSProp F1 | Adam F1 |
|--------|---------|-------------|----------|--------|------------|---------|
| CelebA | 91.23 | 91.54 | 92.08 | 92.12 | 91.17 | 92.09 |
| MS-COCO | 89.62 | 89.71 | 90.03 | 68.35 | 71.03 | 74.10 |
| FairFace | 89.41 | 91.37 | 92.20 | 91.13 | 92.07 | 92.17 |

### 统计显著性检验（Wilcoxon检验p值）

| 指标 | SGD vs RMSProp (性别) | SGD vs Adam (性别) | SGD vs RMSProp (年龄) | SGD vs Adam (年龄) |
|------|---------------------|-------------------|---------------------|-------------------|
| $F_{EOD}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ |
| $F_{EOP}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $5\times10^{-3}$ |
| $F_{DPA}$ | $2\times10^{-3}$ | $1\times10^{-3}$ | $7\times10^{-3}$ | $3\times10^{-3}$ |

### 与公平性增强方法的互补性

| 方法 | 指标 | Adam | RMSProp | SGD |
|------|------|------|---------|-----|
| 有公平性增强 | 均等机会Gap↓ | 0.45 | 0.48 | 0.71 |
| 有公平性增强 | 人口统计均衡Gap↓ | 0.86 | 0.86 | 2.60 |
| 无公平性增强 | 均等机会Gap↓ | 13.99 | 13.90 | 15.19 |
| 无公平性增强 | 人口统计均衡Gap↓ | 11.49 | 11.45 | 11.80 |

### 关键发现

- 数据集越不平衡（如FairFace种族少数群体仅0.9%），自适应优化器的公平性优势越明显，与Theorem 1的理论预测一致
- CelebA中男性比例从42%降到2%时，RMSProp与SGD的 $F_{DPA}$ 差距持续扩大
- 自适应优化器的公平性提升不依赖于整体分类性能，F1分数更高时不一定更公平
- 自适应优化器的公平性优势可与现有公平性增强方法叠加使用

## 亮点与洞察

- **问题的重要性和新颖性**: "优化器影响公平性"这一发现简单但深刻，具有很强的实践指导意义——在公平性敏感场景中，仅换用Adam/RMSProp就能获得显著改善
- **理论扎实**: SDE分析给出了闭式解，Fokker-Planck方程的推导过程严谨，两个后续定理进一步扩展到一般情况
- **实验设计全面**: 10次重复实验+Wilcoxon检验确保统计显著性，多个骨干网络/数据集/敏感属性组合

## 局限与展望

- 理论分析在简单二次损失下进行，虽然模拟和实验验证了一般性，但理论推广到高维仍有困难
- 仅关注群体公平性，未讨论个体公平性
- 所有定理基于各向同性噪声假设（附录中有各向异性扩展，但受限于更强的条件）
- 未探讨自适应优化器在公平性最优超参数选择方面的指导原则

## 相关工作与启发

- 与Rebuffi等人的研究（SGD更鲁棒）形成互补：鲁棒性与公平性的权衡值得进一步研究
- Zeng等人(2024)的工作表明平衡数据集有利于更紧的公平性保证，本文的SDE分析提供了在数据不平衡场景下的替代方案
- 启发：可以进一步探索优化器设计空间，开发专门面向公平性的优化算法

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次从优化器角度理解公平性，洞察独到且实用
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集、多个敏感属性、统计检验、消融实验、与公平性方法互补实验
- **写作质量**: ⭐⭐⭐⭐ 从简单例子出发逐步建立理论，pedagogical风格好
- **价值**: ⭐⭐⭐⭐⭐ 对从业者的直接指导——只需换优化器即可改善公平性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)
- [\[CVPR 2025\] SocialGesture: Delving into Multi-Person Gesture Understanding](../../CVPR2025/human_understanding/socialgesture_delving_into_multi-person_gesture_understanding.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2026\] MOFA-VTON: More Fashion Possibilities with Fine-Grained Adaptations in Virtual Try-On](../../CVPR2026/human_understanding/mofa-vton_more_fashion_possibilities_with_fine-grained_adaptations_in_virtual_tr.md)
- [\[CVPR 2026\] Bridging Facial Understanding and Animation via Language Models](../../CVPR2026/human_understanding/bridging_facial_understanding_and_animation_via_language_models.md)

</div>

<!-- RELATED:END -->

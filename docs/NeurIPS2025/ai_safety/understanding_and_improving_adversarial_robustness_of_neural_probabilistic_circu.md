---
title: >-
  [论文解读] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits
description: >-
  [NeurIPS 2025][AI安全][概念瓶颈模型] 理论分析神经概率电路（NPC）的对抗鲁棒性仅取决于属性识别模型而与概率电路无关，并提出 RNPC 通过类级推理集成方式实现可证明的鲁棒性提升，在保持良性准确率的同时显著增强对抗鲁棒性。 概念瓶颈模型（CBM）：通过引入人类可理解的中间概念层提供可解释性…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "概念瓶颈模型"
  - "概率电路"
  - "对抗鲁棒性"
  - "类级推理"
  - "可解释性"
---

# Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits

**会议**: NeurIPS 2025  
**arXiv**: [2509.20549](https://arxiv.org/abs/2509.20549)  
**代码**: [https://github.com/uiuctml/RNPC](https://github.com/uiuctml/RNPC)  
**领域**: AI 安全 / 对抗鲁棒性  
**关键词**: 概念瓶颈模型, 概率电路, 对抗鲁棒性, 类级推理, 可解释性

## 一句话总结

理论分析神经概率电路（NPC）的对抗鲁棒性仅取决于属性识别模型而与概率电路无关，并提出 RNPC 通过类级推理集成方式实现可证明的鲁棒性提升，在保持良性准确率的同时显著增强对抗鲁棒性。

## 研究背景与动机

**概念瓶颈模型（CBM）** 通过引入人类可理解的中间概念层提供可解释性，但传统 CBM 在概念层上使用线性预测器，不仅性能有损还会**损害鲁棒性**。**神经概率电路（NPC）** 是新一代 CBM，由属性识别模型和概率电路两个模块组成：属性识别模型预测可解释的类别属性，概率电路学习属性与类别的联合分布并支持可扩展推理。NPC 在性能和可解释性之间取得了良好平衡。

然而，NPC 中的属性识别模型仍是神经网络黑盒，容易受到**对抗攻击**——通过微小不可察觉的输入扰动操纵属性预测，进而影响最终分类。

核心矛盾：NPC 的估计误差是组合性的（各模块误差线性叠加），那么其对抗鲁棒性也是如此吗？如果不是，能否设计更鲁棒的推理方式？

**关键发现**：NPC 的对抗鲁棒性仅取决于属性识别模型，加入概率电路"免费"获得鲁棒性（与传统 CBM 中线性层损害鲁棒性形成对比）。这一发现启发了进一步提升鲁棒性的 RNPC 设计。

## 方法详解

### 整体框架

RNPC 与 NPC 共享**相同的模型架构和训练**（属性识别模型 + 概率电路），唯一区别在于**推理方式**：NPC 使用节点级（node-wise）集成，RNPC 使用类级（class-wise）集成。

### 关键设计

1. **NPC 对抗鲁棒性分析（Theorem 3.4）**：

    - 定义预测扰动 $\Delta_{\theta,w}^{NPC}$ 为对抗攻击下类别分布变化的最坏情况 TV 距离
    - 证明：$\Delta_{\theta,w}^{NPC} \leq \sum_{k=1}^K \mathbb{E}_X[\max_{\tilde{X}} d_{TV}(\mathbb{P}_{\theta_k}(A_k|X), \mathbb{P}_{\theta_k}(A_k|\tilde{X}))]$
    - 含义：鲁棒性上界仅由属性识别模型决定，概率电路不影响鲁棒性——这与 NPC 估计误差的组合性形成鲜明对比

2. **属性空间的类级分区**：

    - 将高概率属性节点集 $V$ 按最可能的类别分区：$V = \bigcup_y V_y$
    - 定义类间 Hamming 距离 $d_{i,j} = \min_{v_i \in V_i, v_j \in V_j} \text{Ham}(v_i, v_j)$
    - 定义半径 $r = \lfloor \frac{d_{min}-1}{2} \rfloor$
    - 定义类邻域 $\mathcal{N}(y,r)$：$V_y$ 加上距 $V_y$ 不超过 $r$ 步的低概率节点

3. **RNPC 类级集成推理（Equation 2）**：
    $\Phi_{\theta,w}(Y|X) = \sum_{\tilde{y}} (\mathbb{P}_\theta(A_{1:K} \in \mathcal{N}(\tilde{y},r)|X) \cdot \sum_{a_{1:K} \in V_{\tilde{y}}} \mathbb{P}_w(Y|A_{1:K}=a_{1:K}))$
    - 核心直觉：当攻击扰动属性预测时，概率从正确节点 $a_{1:K}^*$ 流向相邻节点。如果这些节点仍在 $\mathcal{N}(y,r)$ 内（即攻击属性数 $m \leq r$），那么**整个类的权重几乎不变**，从而维持正确预测
    - 对比 NPC 的节点级集成：单个节点的权重下降直接影响预测

4. **理论保证**：

    - **鲁棒性上界**（Lemma 4.6）：RNPC 的扰动上界 $\Lambda_{RNPC} \leq \alpha_\epsilon$，而 NPC 的上界 $\Lambda_{NPC} \leq \frac{|A_1|\cdots|A_K|}{2} \alpha_\epsilon$（Theorem 4.7）——RNPC 的上界比 NPC 小**指数级**
    - **组合估计误差**（Theorem 4.10）：RNPC 的良性误差仍可分解为两个模块的线性组合
    - **鲁棒性-准确率 tradeoff**（Theorem 4.11）：最优 RNPC 与真实分布之间的距离由 $V_y$ 分区质量决定

### 损失函数 / 训练策略

属性识别模型：最小化所有属性的交叉熵损失之和。概率电路：LearnSPN 学结构 + CCCP 优化参数。两者独立训练，NPC 和 RNPC 共享训练好的模型。

## 实验关键数据

### 主实验（良性准确率）

| 数据集 | CBM | DCR | NPC | RNPC |
|--------|------|------|------|------|
| MNIST-Add3 | 99.02 | 98.54 | 99.32 | **99.37** |
| MNIST-Add5 | 99.37 | 99.21 | 99.40 | **99.51** |
| CelebA-Syn | 99.83 | 99.45 | 99.95 | **99.95** |
| GTSRB-Sub | 99.42 | 99.42 | **99.57** | 99.49 |

RNPC 在良性输入上与 NPC 持平甚至略优，验证了鲁棒性-准确率 tradeoff 在实际中可忽略。

### 消融实验（对抗攻击 PGD-∞, ε=0.11）

| 配置 | MNIST-Add5 对抗准确率 | 说明 |
|------|---------|------|
| CBM | <20% | 线性预测器损害鲁棒性 |
| DCR | <25% | 类似 CBM |
| NPC | <40% | 概率电路不损害但也不增强 |
| **RNPC** | **>80%** | 类级集成显著提升 |
| RNPC (r=0) | ~60% | 半径过小，容错不足 |
| RNPC (r=r*=2) | **>80%** | 自然半径最优 |
| RNPC (r=5=K) | 29.9% | 覆盖全空间，丧失区分力 |

### 关键发现
- NPC 和 RNPC 的鲁棒性显著优于 CBM 和 DCR，证实概率电路对鲁棒性"免费"
- RNPC 在 MNIST-Add5 上对抗准确率比 NPC 高 40%+，验证了指数级的理论优势
- 半径 $r = r^*$ 是最优选择：减小增大均损害对抗准确率
- **攻击传播**问题：在 GTSRB 上因属性间虚假相关，攻击单个属性会间接影响其他属性的预测，削弱 RNPC 优势
- 对抗训练可有效缓解攻击传播（解耦属性间的虚假关联）

## 亮点与洞察
- **"概率电路对鲁棒性免费"**这一发现非常深刻，与估计误差的组合性形成了有趣的对比
- RNPC 的类级集成思想直觉上很自然——当属性预测被扰动时，只要还落在正确类的"邻域"内，就不影响最终预测
- 理论推导与实验验证的一致性很强，尤其是半径 $r$ 的消融实验完美匹配理论预测
- 推理时间复杂度反而降低（$|V| \leq \prod_k |A_k|$）

## 局限与展望
- 攻击传播（属性间虚假相关）在实际数据中普遍存在，RNPC 的优势会被削弱
- 数据集较小且合成成分多，需在更大规模的真实数据集上验证
- 属性空间的分区和半径依赖于数据的内在结构，需已知或可计算
- 仅考虑白盒 norm-bounded 攻击，未涉及其他攻击类型（如语义攻击）

## 相关工作与启发
- **vs 标准 CBM [Koh et al.]**：CBM 的线性预测器损害鲁棒性，NPC/RNPC 的概率电路不影响鲁棒性
- **vs Sinha et al. (2023)**：他们的目标是让属性概率在攻击下不变（通过对抗训练），RNPC 的目标是即使概率变了也能正确预测
- **vs DCR**：DCR 用嵌入层替代线性层，但仍受线性预测器的鲁棒性限制

## 评分
- 新颖性: ⭐⭐⭐⭐ 类级集成推理是自然的设计，但"概率电路对鲁棒性免费"的理论发现很有价值
- 实验充分度: ⭐⭐⭐⭐ 多个数据集、多种攻击、丰富消融，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 理论与实验紧密结合，定义和定理清晰
- 价值: ⭐⭐⭐⭐ 为可解释模型的鲁棒性提供了理论框架和实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](../../ICML2025/ai_safety/solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[CVPR 2026\] Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks](../../CVPR2026/ai_safety/towards_reliable_evaluation_of_adversarial_robustness_for_spiking_neural_network.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)

</div>

<!-- RELATED:END -->

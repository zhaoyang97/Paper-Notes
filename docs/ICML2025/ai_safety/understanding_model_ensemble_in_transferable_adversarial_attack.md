---
title: >-
  [论文解读] Understanding Model Ensemble in Transferable Adversarial Attack
description: >-
  [ICML 2025][AI安全][对抗迁移性] 首次为模型集成对抗攻击建立理论框架，定义 transferability error 并将其分解为脆弱性（vulnerability）与多样性（diversity），再利用信息论工具给出上界，从理论上验证了"更多模型+更高多样性+更低复杂度"三条实践指南。
tags:
  - "ICML 2025"
  - "AI安全"
  - "对抗迁移性"
  - "模型集成攻击"
  - "Rademacher复杂度"
  - "脆弱性-多样性分解"
  - "信息论"
---

# Understanding Model Ensemble in Transferable Adversarial Attack

**会议**: ICML 2025  
**arXiv**: [2410.06851](https://arxiv.org/abs/2410.06851)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 对抗迁移性, 模型集成攻击, Rademacher复杂度, 脆弱性-多样性分解, 信息论

## 一句话总结

首次为模型集成对抗攻击建立理论框架，定义 transferability error 并将其分解为脆弱性（vulnerability）与多样性（diversity），再利用信息论工具给出上界，从理论上验证了"更多模型+更高多样性+更低复杂度"三条实践指南。

## 研究背景与动机

对抗样本的**迁移性**（adversarial transferability）使得攻击者无需接触目标模型即可发起黑盒攻击，对安全关键系统构成严重威胁。现有提升迁移性的方法分为三类：输入变换、梯度优化、模型集成攻击。其中模型集成攻击最为有效——利用多个代理模型同时生成对抗样本，可与前两类方法叠加使用。

经验上存在两条广为人知但缺乏理论解释的现象：(1) 集成更多代理模型可提升迁移性；(2) 使用更多样的模型架构进一步增强迁移性。然而，对抗迁移性的理论研究多集中在数据分布、代理模型泛化、优化平坦性等方面，**模型集成攻击的理论基础几乎空白**。

本文的核心动机是：**能否建立一套完整的理论框架来解释模型集成攻击中的迁移性来源，并为未来算法设计提供可操作的指南？** 作者从学习理论中的偏差-方差分解出发，将类似思想迁移到对抗攻击领域，并借助信息论前沿工具处理代理模型之间的非独立性问题。

## 方法详解

### 整体框架

本文提出三个核心定义——**Transferability Error**、**Diversity**、**Empirical Model Ensemble Rademacher Complexity**——构成理论框架的基础。在此之上给出两个关键理论结果：(1) Vulnerability-Diversity 分解定理；(2) Transferability Error 上界定理。框架逻辑为：

1. **定义 transferability error** 度量迁移性差距 → 2. **分解**为 vulnerability + diversity 揭示来源 → 3. 因存在 vulnerability-diversity 权衡，进一步用 **Rademacher 复杂度 + 信息论** 给出上界 → 4. 上界推导出三条实践指南

### 关键设计一：Vulnerability-Diversity Decomposition

**核心思想**：类比偏差-方差分解，将 transferability error 拆为两个有直观含义的项。

给定对抗样本 $z = (x, y)$，population risk $L_P(z) = \mathbb{E}_{\theta \sim \mathcal{P}_\Theta}[\ell(f(\theta; x), y)]$，最优对抗样本 $z^*$ 使 $L_P(z^*)$ 最大。Transferability error 定义为：

$$TE(z, \epsilon) = L_P(z^*) - L_P(z)$$

在均方误差损失 $\ell(f(\theta; x), y) = [f(\theta; x) - y]^2$ 下，令 $\tilde{f}(\theta; x) = \mathbb{E}_{\theta \sim \mathcal{P}_\Theta} f(\theta; x)$ 为集成预测期望，则：

$$TE(z, \epsilon) = L_P(z^*) - \underbrace{\ell(\tilde{f}(\theta; x), y)}_{\text{Vulnerability}} - \underbrace{\text{Var}_{\theta \sim \mathcal{P}_\Theta} f(\theta; x)}_{\text{Diversity}}$$

- **Vulnerability**：集成平均预测偏离真实标签的程度，越大说明集成攻击越有效
- **Diversity**：集成成员预测的方差，越大说明成员间分歧越大，有助于防止对抗样本对集成过拟合

该分解揭示了一个基本权衡：**vulnerability 和 diversity 不可能同时最大化**，类似于经典的偏差-方差权衡。这促使作者寻求更实用的上界分析。

### 关键设计二：基于信息论的 Transferability Error 上界

为突破分解中的权衡限制，作者定义了**经验模型集成 Rademacher 复杂度**：

$$\mathcal{R}_N(\mathcal{Z}) = \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{z \in \mathcal{Z}} \frac{1}{N} \sum_{i=1}^{N} \sigma_i \ell(f(\theta_i; x), y) \right]$$

其中 $\sigma_i$ 为 Rademacher 随机变量。该量度量了 $N$ 个集成分类器相对于输入空间的"拟合随机标签"能力，反映模型复杂度。

**MLP 的复杂度上界（Lemma 4.2）**：对 $l$ 层 MLP，令 $T = \prod_{j=1}^{l} \sup_{i \in [N]} \|W_{i,j}\|_F$，$\|x\|_F \leq B$，则：

$$\mathcal{R}_N(\mathcal{Z}) \leq \frac{(\sqrt{(2 \ln 2) l} + 1) BT}{\sqrt{N}}$$

这说明增加模型数 $N$ 或减小权重范数 $T$（降低模型复杂度）均可收紧上界。

**Transferability Error 上界（Theorem 4.3）**：损失有界于 $\beta$ 时，以概率 $\geq 1 - \delta$：

$$TE(z, \epsilon) \leq 4\mathcal{R}_N(\mathcal{Z}) + \sqrt{\frac{18\gamma \beta^2}{N} \ln \frac{2^{2+1/\gamma} H_\alpha^{1/\alpha}(\mathcal{P}_{\Theta^N} \| \mathcal{P}_{\bigotimes_{i=1}^{N} \Theta})}{\delta}}$$

其中 $H_\alpha(\cdot \| \cdot)$ 为 Hellinger 积分，衡量联合分布 $\mathcal{P}_{\Theta^N}$ 与边际积 $\mathcal{P}_{\bigotimes \Theta}$ 的分歧。第一项由复杂度和模型数量控制，第二项通过 Hellinger 积分刻画模型间的依赖程度。**代理模型越独立（越多样），Hellinger 项越小，上界越紧**。

证明中的关键技术难点在于：代理模型通常在相似任务上训练，参数之间并非独立。作者引入 Esposito & Mondelli (2024) 的信息论最新结果，绕过了传统泛化理论的独立性假设。

### 三条实践指南

综合两个理论结果，作者得出三条可操作指南：

1. **使用更多代理模型**：$N$ 增大时，$\mathcal{R}_N(\mathcal{Z}) \propto 1/\sqrt{N}$ 下降，第二项也随 $N$ 收紧
2. **增加代理模型多样性**：模型越独立，联合分布越接近边际积，Hellinger 项越小
3. **降低模型复杂度（防止过拟合）**：权重范数 $T$ 减小→复杂度下降，对应正则化或更简单的架构

## 实验关键数据

实验使用 MNIST、Fashion-MNIST、CIFAR-10 三个数据集，构建 54 个模型（3 种 MLP × 3 种 CNN × 3 种数据变换 × 3 种 weight decay），以额外训练的 ResNet-18 作为黑盒目标模型。

### 攻击动态验证（MI-FGSM, 20步攻击）

| 数据集 | 模型 | ASR 趋势 | Loss 趋势 | Variance 趋势 |
|--------|------|----------|-----------|---------------|
| MNIST | MLP | 随步数单调上升 | 持续增大 | 先升后降 |
| MNIST | CNN | 随步数单调上升 | 持续增大 | 先升后降 |
| Fashion-MNIST | MLP | 随步数单调上升 | 持续增大 | 先升后降 |
| Fashion-MNIST | CNN | 随步数单调上升 | 持续增大 | 先升后降 |
| CIFAR-10 | MLP | 随步数上升 | 持续增大 | 持续增大 |
| CIFAR-10 | CNN | 随步数上升 | 持续增大 | 小 $\lambda$ 下降，大 $\lambda$ 上升 |

关键观察：vulnerability（loss）的量级约为 diversity（variance）的 **10 倍**，在分解中占主导地位。ASR 的上升趋势与 vulnerability 增大一致，验证了分解的有效性。

### 集成模型数量影响（逐步从 1 增至 18 个模型）

| 数据集 | 1 模型 ASR | 9 模型 ASR | 18 模型 ASR | Vulnerability | Diversity |
|--------|-----------|-----------|------------|---------------|-----------|
| MNIST | ~40% | ~75% | ~85% | 持续增大 | 持续增大 |
| Fashion-MNIST | ~25% | ~55% | ~65% | 持续增大 | 持续增大 |
| CIFAR-10 | ~15% | ~40% | ~50% | 持续增大 | 量级远小于 vulnerability |

所有数据集上，增加集成模型数量均显著提升 ASR，验证了理论预测。MNIST 和 Fashion-MNIST 上 vulnerability 和 diversity 双增；CIFAR-10 上 diversity 偶有下降但量级仅为 vulnerability 的 1/100，对 ASR 影响可忽略。

## 关键发现

1. **Vulnerability 主导迁移性**：在 vulnerability-diversity 分解中，vulnerability 项的量级远大于 diversity 项，是提升 ASR 的主要驱动力
2. **Diversity 的复杂行为**：diversity 不总是单调变化——MNIST/Fashion-MNIST 上呈"钟形"先升后降，CIFAR-10 上受模型类型和正则化强度影响，行为各异
3. **复杂度-多样性权衡**：实验中发现增大 weight decay $\lambda$（降低复杂度）可能使 variance 增大，暗示在过拟合阶段降低复杂度有助于同时提升迁移性
4. **泛化-迁移性类比成立**：Transferability error 的数学形式与泛化误差高度平行，从理论上验证了"对抗样本迁移性类比于模型泛化性"这一长期启发式观点

## 亮点与洞察

- **首个模型集成攻击理论框架**：填补了该方向理论空白，三个定义（transferability error、diversity、ensemble Rademacher complexity）简洁且有良好数学性质
- **偏差-方差分解的巧妙迁移**：将学习理论的经典工具从"模型的泛化"迁移到"对抗样本的迁移"，视角新颖
- **信息论处理非独立性**：传统泛化界依赖数据独立性假设，但代理模型间不独立。引入 Hellinger 积分和最新信息论技术是关键创新
- **实用指南有理论支撑**：三条指南（更多模型、更多样、更低复杂度）虽然经验上已被广泛使用，但本文首次给出了严格的理论依据

## 局限性

1. **理论-实践间距**：vulnerability-diversity 分解基于均方误差损失，实际攻击多用交叉熵损失（虽然附录中有 KL 散度版本，但推导更粗糙）
2. **模型规模有限**：实验仅使用浅层 MLP 和 CNN，未在 ViT、大规模 ImageNet 预训练模型等现代架构上充分验证
3. **Hellinger 项不可计算**：上界中的 $H_\alpha(\mathcal{P}_{\Theta^N} \| \mathcal{P}_{\bigotimes \Theta})$ 在实际中无法直接计算，限制了上界的实用性
4. **未给出最优集成策略**：理论揭示了 vulnerability-diversity 权衡和 complexity-diversity 权衡的存在，但未提出具体的平衡策略或新算法
5. **Same parameter space 假设**：主定理假设代理和目标模型共享同一参数空间，虽然附录有扩展讨论，但实际中架构差异很大时适用性待验证

## 相关工作与启发

- **偏差-方差分解** (Geman et al., 1992)：本文 vulnerability-diversity 分解直接类比 bias-variance decomposition
- **Rademacher 复杂度** (Bartlett & Mendelson, 2002)：经典泛化理论工具，本文将其推广到模型集成攻击场景
- **信息论泛化界** (Esposito & Mondelli, 2024)：处理非独立样本的最新数学工具，是本文证明的核心技术依赖
- **集成学习多样性** (Wood et al., 2024; Ortega et al., 2022)：Diversity 定义借鉴了集成学习理论的最新进展
- **对抗迁移性理论** (Wang & Farnia, 2023; Fan et al., 2024)：前者从泛化误差角度分析单模型迁移性，后者分解迁移性为局部有效性与迁移损失

本文启发了一个有趣方向：**将泛化理论的各种工具系统性地迁移到对抗迁移性分析**。例如，PAC-Bayes 界、VC 维理论等是否也能给出有价值的迁移性洞察？

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 理论深度 | ⭐⭐⭐⭐⭐ | 首个完整理论框架，三个定义+两个核心定理+信息论新技术 |
| 实验充分性 | ⭐⭐⭐ | 54 模型验证充分，但模型规模和数据集较简单 |
| 创新性 | ⭐⭐⭐⭐⭐ | 将学习理论工具首次系统引入模型集成攻击的理论分析 |
| 实用价值 | ⭐⭐⭐⭐ | 三条指南有实用意义，但缺少具体新算法 |
| 写作清晰度 | ⭐⭐⭐⭐ | 逻辑清晰，图示直观，符号较多但定义明确 |
| 总评 | ⭐⭐⭐⭐ | 理论导向的高质量工作，开创了模型集成攻击理论研究新方向 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Identifying and Understanding Cross-Class Features in Adversarial Training](identifying_and_understanding_cross-class_features_in_adversarial_training.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[AAAI 2026\] Transferable Hypergraph Attack via Injecting Nodes into Pivotal Hyperedges](../../AAAI2026/ai_safety/transferable_hypergraph_attack_via_injecting_nodes_into_pivotal_hyperedges.md)
- [\[CVPR 2025\] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](../../CVPR2025/ai_safety/mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)

</div>

<!-- RELATED:END -->

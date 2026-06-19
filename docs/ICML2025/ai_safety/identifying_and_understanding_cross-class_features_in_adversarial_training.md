---
title: >-
  [论文解读] Identifying and Understanding Cross-Class Features in Adversarial Training
description: >-
  [ICML2025][AI安全][对抗训练] 从类别级特征归因的角度揭示对抗训练(AT)中的"跨类特征"如何先被学习后被遗忘，统一解释了鲁棒过拟合和软标签训练优势两大现象。 对抗训练(AT)是使深度网络抵抗对抗攻击最有效的方法之一，其核心为最小-最大优化： $$\min_{\boldsymbol{\theta}} \frac…
tags:
  - "ICML2025"
  - "AI安全"
  - "对抗训练"
  - "跨类特征"
  - "鲁棒过拟合"
  - "知识蒸馏"
  - "特征归因"
---

# Identifying and Understanding Cross-Class Features in Adversarial Training

**会议**: ICML2025  
**arXiv**: [2506.05032](https://arxiv.org/abs/2506.05032)  
**代码**: [PKU-ML/Cross-Class-Features-AT](https://github.com/PKU-ML/Cross-Class-Features-AT)  
**领域**: 对抗训练 / AI安全  
**关键词**: 对抗训练, 跨类特征, 鲁棒过拟合, 知识蒸馏, 特征归因

## 一句话总结

从类别级特征归因的角度揭示对抗训练(AT)中的"跨类特征"如何先被学习后被遗忘，统一解释了鲁棒过拟合和软标签训练优势两大现象。

## 研究背景与动机

对抗训练(AT)是使深度网络抵抗对抗攻击最有效的方法之一，其核心为最小-最大优化：

$$\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{i=1}^{N} \max_{\|\delta_i\|_p \leq \epsilon} \ell(f(\boldsymbol{\theta}, x_i + \delta_i), y_i)$$

然而 AT 存在两个未被充分理解的现象：

**鲁棒过拟合(Robust Overfitting)**：训练中期模型达到最佳测试鲁棒精度，之后测试鲁棒精度逐渐下降，而训练鲁棒误差持续降低，形成巨大泛化鸿沟

**软标签优势**：用知识蒸馏等软标签替代 one-hot 标签可显著提升 AT 性能（如 CIFAR-10 上从 41%→48%），但原因不明

现有解释分别从数据级损失、标签噪声等角度切入，但缺乏统一视角。本文首次从**类别级特征归因**提出统一假设。

## 方法详解

### 核心概念：跨类特征 vs 类特定特征

- **跨类特征(Cross-class Features)**：多个类别共有的特征，如 CIFAR-10 中汽车和卡车共享的"车轮"特征
- **类特定特征(Class-specific Features)**：仅属于单一类别的特征，如青蛙的"蛙眼"

### 特征归因度量

设分类器 $f(\cdot) = Wg(\cdot)$，其中 $g$ 为特征提取器，$W \in \mathbb{R}^{K \times n}$ 为线性层。对于样本 $x$ 在第 $i$ 类上的归因向量定义为：

$$A_i(x) = (g(x)_1 W[i,1], \cdots, g(x)_n W[i,n])$$

每个分量 $g(x)_j W[i,j]$ 表示第 $j$ 个特征对第 $i$ 类 logit 的贡献。

### 跨类特征相关矩阵

构建类间特征归因相关矩阵，使用余弦相似度衡量：

$$C[i,j] = \frac{A_i \cdot A_j}{\|A_i\|_2 \cdot \|A_j\|_2}$$

高 $C[i,j]$ 值表示类 $i$ 和类 $j$ 共享更多特征。

### 数值指标 CAS（Class Attribution Similarity）

$$\text{CAS}(C) = \sum_{i \neq j} \max(C[i,j], 0)$$

CAS 定量反映模型对跨类特征的使用程度，仅考虑正相关项。

### 主要假设：AT 的两阶段动态

1. **初期阶段**：模型同时学习类特定特征和跨类特征，两者共同降低鲁棒损失
2. **后期阶段**：当鲁棒损失降到一定程度时，跨类特征因在非目标类上产生正 logit 而阻碍损失进一步下降，模型开始放弃跨类特征，转而依赖类特定特征 → 导致鲁棒过拟合

### 理论分析：合成数据模型

设三类分类任务，每类有独占特征 $x_{E,i}$ 和跨类特征 $x_{C,j}$，数据分布：

$$x_{E,j} \sim \begin{cases} \mathcal{N}(\mu, \sigma^2), & j=i \\ 0, & j \neq i \end{cases}, \quad x_{C,j} \sim \begin{cases} \mathcal{N}(\mu, \sigma^2), & j \neq i \\ 0, & j=i \end{cases}$$

**定理1**（跨类特征对鲁棒损失更敏感）：存在阈值 $\epsilon_0 \in (0, \mu/2)$，当 $\epsilon > \epsilon_0$ 时 AT 优化结果 $w_2 = 0$（放弃跨类特征），但任意 $\epsilon \in (0, \mu/2)$ 始终 $w_1 > 0$（保留类特定特征）。

**定理2**（跨类特征有助鲁棒分类）：在 $w_2 \in [0, w_1]$ 范围内，增大 $w_2$ 单调提升模型在对抗攻击下的正确分类概率。

**定理3**（软标签保留跨类特征）：使用标签平滑的 AT 阈值 $\epsilon_1 > \epsilon_0$，且在 $\epsilon \in (0, \epsilon_1)$ 时有 $w_2^{\text{LS}}(\epsilon) > w_2^*(\epsilon)$，即软标签保留更多跨类特征。

## 实验关键数据

### CIFAR-10 上 AT 不同阶段的 CAS 和鲁棒精度（PreActResNet-18）

| 阶段 | Epoch | 鲁棒精度(RA) | CAS |
|------|-------|-------------|-----|
| 欠拟合 | 70 | 42.6% | 18.2 |
| 最佳 | 108 | 47.8% | 25.6 |
| 过拟合 | 200 | 42.5% | 9.0 |

→ 最佳检查点的 CAS 远高于过拟合检查点，验证跨类特征与鲁棒泛化的正相关。

### 不同扰动强度 ε 下的 ΔCAS（Best - Last）

| ε | ΔCAS | 过拟合程度 |
|---|------|----------|
| 2/255 | 4.1 | 轻微 |
| 4/255 | 8.9 | 中等 |
| 6/255 | 13.8 | 较重 |
| 8/255 | 16.6 | 严重 |

→ 更大的 ε 导致更多跨类特征被遗忘，对应更严重的鲁棒过拟合。

### 极大 ε 下的 CAS 变化

| ε | Epoch 10 CAS/RA | Best CAS/RA | Last CAS/RA |
|---|----------------|-------------|-------------|
| 8/255 | 16.7/36.9% | 25.6/47.8% | 9.0/42.5% |
| 12/255 | 15.6/29.8% | 18.9/38.7% | 8.7/34.1% |
| 16/255 | 14.4/23.8% | 17.5/31.3% | 8.4/28.1% |

→ 极大 ε 下初始阶段已学很少跨类特征，因此遗忘效应减弱，鲁棒过拟合反而缓解。

### 知识蒸馏 AT 对比

| 方法 | 阶段 | RA | CAS |
|------|------|-----|-----|
| AT+KD | Best | 48.1% | 25.7 |
| AT+KD | Last | 46.2% | 24.1 |

→ KD 在整个训练过程中保持高 CAS，CAS 差距从 16.6 降到 1.6，鲁棒过拟合大幅缓解。

### 跨数据集/架构验证

- **CIFAR-100**: Best CAS=569, Last CAS=352
- **TinyImageNet**: Best CAS=1548, Last CAS=998
- **ℓ₂-AT**: Best CAS=22.1, Last CAS=10.7
- **DeiT-Ti (Transformer)**: Best CAS=25.4, Last CAS=16.6

→ 所有设置下结论一致，跨类特征使用量与鲁棒泛化强正相关。

## 亮点与洞察

1. **全新视角**：首次从"跨类特征"角度统一解释鲁棒过拟合和软标签优势两大 AT 谜题
2. **直观量化指标 CAS**：简洁有效地量化模型对跨类特征的依赖程度，基于最后一层线性层的特征归因
3. **理论-实验闭环**：合成数据上的三个定理精确刻画跨类特征的敏感性和有用性，实验完美验证
4. **Saliency Map 可视化**：通过 GradCAM 直观展示最佳检查点关注整体特征（车轮+车身），过拟合检查点仅关注局部独有特征（圆形车顶），解释力强
5. **覆盖面广**：跨 ℓ∞/ℓ₂ 范数、CNN/Transformer 架构、多数据集、FAT(快速AT) 场景均验证假设

## 局限与展望

1. **CAS 依赖线性层假设**：归因向量 $A_i(x) = g(x) \odot W[i]$ 仅适用于最后一层为线性层的架构，无法直接推广到更复杂的分类头
2. **理论局限于合成模型**：三类线性模型的理论分析较简化，与实际深度网络的训练动态仍有差距
3. **未提出新防御方法**：文章主要是分析和理解，没有基于跨类特征假设设计新的对抗训练算法
4. **因果性不确定**：CAS 与鲁棒精度的相关性已确认，但是否为直接因果关系仍需更严格论证
5. **大规模数据集验证不足**：实验主要在 CIFAR-10/100 和 TinyImageNet 上进行，缺少 ImageNet 规模验证

## 相关工作与启发

- **Ilyas et al., 2019**：提出鲁棒特征 vs 非鲁棒特征的框架，本文在此基础上进一步区分跨类 vs 类特定鲁棒特征
- **Rice et al., 2020**：首次系统研究鲁棒过拟合，本文为其提供了特征级别的新解释
- **Chen et al., 2021 (ARD)**：知识蒸馏改善 AT 的代表工作，本文从跨类特征角度解释了其成功原因
- **启发**：跨类特征的遗忘机制暗示，可通过显式鼓励跨类特征学习（如特征正则化、跨类对比学习）来缓解鲁棒过拟合

## 评分

- 新颖性: ⭐⭐⭐⭐ — 跨类特征视角新颖，统一解释两大现象
- 实验充分度: ⭐⭐⭐⭐⭐ — 多数据集、多架构、多范数全面验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论实验结合紧密
- 价值: ⭐⭐⭐⭐ — 提供深刻的 AT 理解，但未直接产出新方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](../../NeurIPS2025/ai_safety/distributional_adversarial_attacks_and_training_in_deep_hedging.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](../../CVPR2026/ai_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[ACL 2025\] Efficiently Identifying Watermarked Segments in Mixed-Source Texts](../../ACL2025/ai_safety/watermark_segment_detection.md)

</div>

<!-- RELATED:END -->

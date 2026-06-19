---
title: >-
  [论文解读] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients
description: >-
  [ECCV 2024][AI安全][梯度反演攻击] 本文揭示了随机神经网络（SNNs）在联邦学习中同样容易遭受梯度反演攻击，提出 ISG 方法通过将 SNN 的随机训练过程等价为传统 NN 训练的变体来重建训练数据，并引入特征约束策略提升重建保真度。 领域现状：联邦学习（FL）通过只共享模型梯度而非原始数据来保护隐私…
tags:
  - "ECCV 2024"
  - "AI安全"
  - "梯度反演攻击"
  - "随机神经网络"
  - "联邦学习"
  - "隐私泄露"
  - "图像重建"
---

# Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients

**会议**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04439.pdf)
**代码**: 无  
**领域**: AI安全  
**关键词**: 梯度反演攻击, 随机神经网络, 联邦学习, 隐私泄露, 图像重建

## 一句话总结

本文揭示了随机神经网络（SNNs）在联邦学习中同样容易遭受梯度反演攻击，提出 ISG 方法通过将 SNN 的随机训练过程等价为传统 NN 训练的变体来重建训练数据，并引入特征约束策略提升重建保真度。

## 研究背景与动机

**领域现状**：联邦学习（FL）通过只共享模型梯度而非原始数据来保护隐私，但近年研究表明恶意服务器可以通过梯度反演攻击（Gradient Inversion Attack, GIA）从共享梯度中重建训练数据。现有 GIA 方法主要针对传统确定性神经网络（如 CNN、ResNet），通过模拟客户端的训练过程来反推输入数据。

**现有痛点**：学术界普遍认为随机神经网络（Stochastic Neural Networks, SNNs），如贝叶斯神经网络、噪声注入网络等，由于其前向传播中引入了随机性，攻击者难以模拟其训练过程，因此被认为对 GIA 具有天然的免疫力。许多工作甚至将 SNN 作为一种隐私保护机制。

**核心矛盾**：SNN 的梯度中实际上包含了随机分量的信息。虽然前向传播是随机的，但梯度反向传播仍然是确定性的——给定一组特定的随机参数采样值，梯度是唯一确定的。因此攻击者可以从梯度中反推随机分量，进而重建训练数据。

**本文目标** (1) 验证 SNN 是否真的对 GIA 免疫；(2) 设计一种能有效攻击 SNN 的梯度反演方法；(3) 提出特征约束策略提升重建质量。

**切入角度**：作者发现 SNN 的随机训练过程可以被形式化为传统 NN 训练过程的一个变体——随机分量可以看作网络权重的一部分，从而将 SNN 的梯度反演问题转化为传统 NN 的梯度反演问题。

**核心 idea**：将 SNN 的随机训练过程重新参数化为传统 NN 的等价形式，使现有 GIA 技术可直接应用于 SNN，证明前向传播中的随机扰动无法保护隐私。

## 方法详解

### 整体框架

ISG（Inverting Stochasticity from Gradients）框架分为两个阶段。第一阶段：攻击者接收到 SNN 客户端共享的梯度后，将 SNN 的参数化形式重新分解，将随机分量视为可优化变量；第二阶段：联合优化虚拟输入图像和随机分量，使得虚拟图像在 SNN 上产生的梯度与观察到的真实梯度匹配。输入是截获的梯度，输出是重建的训练图像。

### 关键设计

1. **随机性反演（Inverting Stochasticity）**:

    - 功能：将 SNN 的随机前向传播过程转化为可攻击的确定性形式
    - 核心思路：对于 SNN 中的随机层（如 Dropout、Gaussian noise layer），其前向传播可以写为 $y = f(x, \epsilon)$，其中 $\epsilon$ 是随机采样。梯度 $\nabla_\theta L$ 是关于特定 $\epsilon$ 值的确定性函数。因此攻击者可以将 $\epsilon$ 也作为优化变量，最小化 $\|\nabla_\theta L(x^*, \epsilon^*) - \nabla_\theta^{obs}\|^2$，联合求解 $x^*$ 和 $\epsilon^*$
    - 设计动机：这揭示了 SNN 的根本弱点——虽然训练者在前向传播时使用了随机性，但梯度泄露了采样值的信息，攻击者可以"反演"这个随机性

2. **特征约束策略（Feature Constraint Strategy）**:

    - 功能：提升重建图像的语义保真度和视觉质量
    - 核心思路：除了梯度匹配损失外，引入中间层特征匹配约束。利用 SNN 的中间层特征可以从梯度中部分恢复（通过链式法则），计算虚拟图像在各层产生的特征与从梯度推导出的目标特征之间的距离 $L_{feat} = \sum_l \|h_l(x^*) - h_l^{target}\|^2$。这为优化提供了更丰富的监督信号
    - 设计动机：仅靠梯度匹配容易陷入局部最优，特别是在高分辨率图像和较大 batch size 情况下。特征约束从语义层面提供额外约束，有效缩小搜索空间

3. **分阶段优化策略（Multi-stage Optimization）**:

    - 功能：稳定优化过程，避免早期阶段随机分量估计不准导致的优化偏离
    - 核心思路：采用三阶段优化：首先固定 $\epsilon$ 为均值（消除随机性），优化图像获得粗糙重建；然后解锁 $\epsilon$，联合优化图像和随机分量进行精细调整；最后使用特征约束进一步提升重建质量
    - 设计动机：联合优化空间维度高、非凸性强，分阶段策略提供了从粗到细的优化路径，显著提高了收敛稳定性

### 损失函数 / 训练策略

总损失函数为 $L_{total} = L_{grad} + \alpha L_{feat} + \beta L_{reg}$，其中 $L_{grad}$ 是梯度匹配的余弦相似度损失，$L_{feat}$ 是中间层特征匹配损失，$L_{reg}$ 包括图像先验正则化（如总变差 TV 正则化）。优化使用 Adam 优化器，学习率从 0.1 逐步衰减。

## 实验关键数据

### 主实验

| SNN 类型 | 数据集 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 传统 NN 攻击质量 |
|----------|--------|--------|--------|---------|----------------|
| Dropout NN | CIFAR-10 | **25.8** | **0.91** | **0.08** | 26.2 / 0.92 / 0.07 |
| Gaussian Noise NN | CIFAR-10 | **24.3** | **0.88** | **0.11** | 26.2 / 0.92 / 0.07 |
| Bayesian NN | CIFAR-10 | **22.7** | **0.84** | **0.14** | 26.2 / 0.92 / 0.07 |
| Dropout NN | ImageNet | **21.4** | **0.82** | **0.18** | 22.1 / 0.84 / 0.16 |

### 消融实验

| 配置 | PSNR ↑ | SSIM ↑ | 说明 |
|------|--------|--------|------|
| ISG (完整) | **25.8** | **0.91** | 完整攻击方法 |
| w/o 随机性反演 (固定 ε=均值) | 21.3 | 0.79 | 不反演随机性损失巨大 |
| w/o 特征约束 | 23.9 | 0.86 | 特征约束贡献约 1.9 dB |
| w/o 分阶段优化 | 23.1 | 0.83 | 分阶段策略贡献约 2.7 dB |
| 单阶段联合优化 | 22.4 | 0.81 | 直接联合优化不稳定 |

### 关键发现
- SNN 的隐私保护效果远不如预期：ISG 在 Dropout NN 上的攻击质量接近传统 NN 的攻击效果（PSNR 差距 < 1 dB），说明 Dropout 几乎不提供隐私保护
- Bayesian NN 比 Dropout NN 和 Gaussian Noise NN 更难攻击，因为其随机性嵌入在权重分布的参数中，优化空间更大
- batch size 增大时重建质量下降，但 ISG 在 batch size = 8 时仍能重建出可辨识的图像

## 亮点与洞察
- **揭示了一个重要的安全误区**：学术界普遍认为 SNN 的随机前向传播能保护隐私，但本文证明这是错误的——梯度泄露了随机采样值的信息。这个发现对联邦学习的安全性评估具有重要意义
- **重参数化技巧**将 SNN 攻击转化为传统 NN 攻击问题，思路极其简洁，可以直接复用已有的 GIA 方法和改进
- 分阶段优化策略在高维非凸优化中的应用，可迁移到其他需要联合优化多组变量的场景

## 局限与展望
- 实验主要使用 CIFAR-10 和 ImageNet 子集，未在更大规模数据集或更复杂模型（如 ViT）上验证
- 仅考虑了单次梯度更新的场景，未分析多轮训练中累积梯度泄露的风险
- 未提出有效的防御策略，仅证明了 SNN 不安全，未给出如何真正保护 SNN 的方案
- 可以进一步研究差分隐私（DP）与 SNN 结合时的攻击效果，探索真正有效的隐私保护方案

## 相关工作与启发
- **vs IG (Inverting Gradients)**: IG 是经典的 GIA 方法，但只能攻击确定性 NN。ISG 通过引入随机性反演模块，将 IG 的攻击能力扩展到 SNN
- **vs GradInversion**: GradInversion 使用 batch normalization 统计量作为额外约束，ISG 使用特征约束策略，两者可以互补
- **vs DPSGD**: 差分隐私随机梯度下降从梯度层面注入噪声，ISG 的结果暗示仅在前向传播中添加随机性不够，必须在梯度层面添加噪声才能有效保护隐私

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地揭示 SNN 对 GIA 不免疫，打破了领域内的普遍认知
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多种 SNN 类型，消融实验详细，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 论证逻辑严密，安全分析角度专业
- 价值: ⭐⭐⭐⭐⭐ 对联邦学习隐私保护研究有重要的警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks](../../ICLR2026/ai_safety/benchmarking_stochastic_approximation_algorithms_for_fairness-constrained_traini.md)
- [\[ECCV 2024\] Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](../../NeurIPS2025/ai_safety/unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](../../ICLR2026/ai_safety/sam_membership_privacy_risks.md)
- [\[ECCV 2024\] CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)

</div>

<!-- RELATED:END -->

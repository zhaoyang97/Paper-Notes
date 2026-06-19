---
title: >-
  [论文解读] Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss
description: >-
  [ICML 2025 Spotlight][AI安全][Lipschitz Neural Networks] 本文提出了一种高效的低秩正交层参数化方法（BRO Layer）和一种退火机制的损失函数（Logit Annealing Loss），用于构建具有更强认证鲁棒性的 Lipschitz 神经网络 BRONet，在 CIFAR-10/100、Tiny-ImageNet 和 ImageNet 上达到 SOTA。
tags:
  - "ICML 2025 Spotlight"
  - "AI安全"
  - "Lipschitz Neural Networks"
  - "Certified Robustness"
  - "Orthogonal Layers"
  - "Logit Annealing"
---

# Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss

**会议**: ICML 2025 Spotlight  
**arXiv**: [2505.15174](https://arxiv.org/abs/2505.15174)  
**代码**: [https://github.com/](https://github.com/)  
**领域**: 其他  
**关键词**: Lipschitz Neural Networks, Certified Robustness, Orthogonal Layers, Logit Annealing

## 一句话总结
本文提出了一种高效的低秩正交层参数化方法（BRO Layer）和一种退火机制的损失函数（Logit Annealing Loss），用于构建具有更强认证鲁棒性的 Lipschitz 神经网络 BRONet，在 CIFAR-10/100、Tiny-ImageNet 和 ImageNet 上达到 SOTA。

## 研究背景与动机
**领域现状**：深度学习模型容易受到对抗攻击。认证防御方法中，Lipschitz 神经网络因能通过单次前向传播计算认证半径而具有推理时效率优势。

**现有痛点**：现有正交层构造方法（如 SOC、LOT）计算代价高昂，依赖迭代近似算法（如牛顿法、Taylor 展开），且可能因近似误差违反 1-Lipschitz 约束。

**核心矛盾**：正交层的计算开销限制了其在更复杂架构中的应用；交叉熵损失无法有效增大 Lipschitz 网络的分类间距。

**本文切入角度**：利用 Block Reflector 的低秩参数化思想，构造无需迭代近似的正交层；通过理论分析 Lipschitz 网络的有限模型容量，设计退火损失函数。

**核心idea**：$W = I - 2V(V^TV)^{-1}V^T$ 形式的低秩正交参数化 + 退火机制逐步增大分类间距。

## 方法详解

### 整体框架
输入图像 → BRO 正交卷积层组成的 Lipschitz 网络 BRONet → logit 输出 → Logit Annealing Loss 训练。整个网络保证全局 1-Lipschitz 约束，推理时可直接计算认证半径 $\varepsilon = \mathcal{M}_f(x) / \sqrt{2}$。

### 关键设计

1. **BRO Layer（Block Reflector Orthogonal Layer）**:

    - 功能：构造正交权重矩阵实现 1-Lipschitz 约束的线性层
    - 核心思路：给定非方阵参数 $V \in \mathbb{R}^{m \times n}$，正交矩阵参数化为 $W = I - 2V(V^TV)^{-1}V^T$。$W$ 自动满足 $W^TW = I$，无需任何迭代近似
    - 对于卷积层，利用 2D 卷积定理在傅里叶域进行正交卷积：$\tilde{W}_{:,:,i,j} = I - 2\tilde{V}_{:,:,i,j}(\tilde{V}^*_{:,:,i,j}\tilde{V}_{:,:,i,j})^{-1}\tilde{V}^*_{:,:,i,j}$
    - 设计动机：避免 SOC 的 Taylor 展开误差和 LOT 的牛顿法数值不稳定问题，同时由于低秩参数化（$n \leq c$），大幅节省计算和内存

2. **Logit Annealing Loss**:

    - 功能：替代交叉熵+CR正则的训练目标，更有效地增大分类间距
    - 核心思路：通过 Rademacher Complexity 分析发现 Lipschitz 网络模型容量有限，难以最小化经验间距损失风险。Logit Annealing 通过退火机制逐步增大对大多数数据点的间距要求
    - 设计动机：解决 CR 正则项的不连续梯度和梯度支配问题

3. **BRONet 架构**:

    - 功能：基于 BRO Layer 构建完整的 Lipschitz 网络
    - 核心思路：将 BRO 层集成到 ResNet 风格架构中，利用 MaxMin 激活函数保持梯度范数
    - 设计动机：BRO 的计算效率使得构建更深更宽的 Lipschitz 网络成为可能

### 损失函数 / 训练策略
Logit Annealing Loss 结合退火机制，在训练初期允许较小间距，随训练推进逐步增大间距要求，避免一开始就施加过难的约束。

## 实验关键数据

### 主实验

| 数据集 | 扰动半径 | BRONet | 之前SOTA | 提升 |
|--------|----------|--------|----------|------|
| CIFAR-10 | ε=36/255 | 70.1% | 68.5% (LOT) | +1.6% |
| CIFAR-100 | ε=36/255 | 40.0% | 38.3% | +1.7% |
| Tiny-ImageNet | ε=36/255 | 28.2% | 26.1% | +2.1% |
| ImageNet | ε=36/255 | 40.5% | - | 首次报告 |

### 消融实验

| 配置 | CIFAR-10 (ε=36/255) | 说明 |
|------|---------------------|------|
| BRONet + Logit Annealing | 70.1% | 完整模型 |
| BRONet + CE+CR | 67.8% | 去掉 Logit Annealing，掉2.3% |
| LOT + Logit Annealing | 68.9% | 换正交层，掉1.2% |
| SOC + Logit Annealing | 68.5% | 换正交层，掉1.6% |

### 关键发现
- BRO 在计算时间和内存上均优于 SOC 和 LOT（训练速度提升约 2-3 倍）
- LOT 使用 Kaiming 初始化时会数值不稳定导致非正交，BRO 无此问题
- Logit Annealing Loss 对不同正交层架构均有效

## 亮点与洞察
- 低秩正交参数化思想非常优雅：通过 Block Reflector 直接构造正交矩阵，完全避免迭代近似，既保证理论正确性又提升效率
- 从 Rademacher Complexity 角度分析 Lipschitz 网络的有限容量，为损失函数设计提供理论指导
- BRO 虽然不是万能正交参数化（单层不能表示所有正交矩阵），但深层网络的表达能力可以弥补

## 局限与展望
- 单个 BRO 层的表达能力受限（只有 $n$ 个特征值为 -1），虽然实验表明深层网络可弥补，但理论上不如 LOT/SOC 灵活
- 仅在 $\ell_2$ 范数下的认证鲁棒性，未涉及 $\ell_\infty$ 等其他范数
- ImageNet 规模的实验仍然有较大提升空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 低秩正交参数化思路巧妙但不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多模型、详细消融
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 对认证鲁棒性领域有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](../../NeurIPS2025/ai_safety/enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/ai_safety/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[ICML 2026\] REFLECTOR：把"边走边自省"内化进生成轨迹以抵御间接越狱](../../ICML2026/ai_safety/reflector_internalizing_step-wise_reflection_against_indirect_jailbreak.md)

</div>

<!-- RELATED:END -->

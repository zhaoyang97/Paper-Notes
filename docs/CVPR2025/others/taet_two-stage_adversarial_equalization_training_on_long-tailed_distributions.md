---
title: >-
  [论文解读] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions
description: >-
  [CVPR 2025][长尾分布] 提出TAET两阶段对抗均衡训练框架：先用交叉熵损失稳定早期训练，再用层级对抗鲁棒学习(HARL)联合BCL/HDL/RCEL三种损失均衡各类性能，并引入平衡鲁棒性(Balanced Robustness)评估指标，解决长尾分布下对抗训练的尾部类鲁棒性不足问题。 对抗训练在平衡数据集（CIF…
tags:
  - "CVPR 2025"
  - "长尾分布"
  - "对抗训练"
  - "均衡化损失"
  - "鲁棒性过拟合"
  - "平衡鲁棒性指标"
---

# TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions

**会议**: CVPR 2025  
**arXiv**: [2503.01924](https://arxiv.org/abs/2503.01924)  
**代码**: [GitHub](https://github.com/BuhuiOK/TAET-Two-Stage-Adversarial-Equalization-Training-on-Long-Tailed-Distributions)  
**领域**: 其他  
**关键词**: 长尾分布, 对抗训练, 均衡化损失, 鲁棒性过拟合, 平衡鲁棒性指标

## 一句话总结

提出TAET两阶段对抗均衡训练框架：先用交叉熵损失稳定早期训练，再用层级对抗鲁棒学习(HARL)联合BCL/HDL/RCEL三种损失均衡各类性能，并引入平衡鲁棒性(Balanced Robustness)评估指标，解决长尾分布下对抗训练的尾部类鲁棒性不足问题。

## 研究背景与动机

对抗训练在平衡数据集（CIFAR-10、ImageNet）上表现良好，但现实数据常呈长尾分布——少数头部类样本丰富而大量尾部类样本稀缺。现有长尾对抗训练方法（如AT-BSL）存在两个关键问题：

1. **BSL对弱类改善不足**: Balanced Softmax Loss仅根据样本数量调整logit，无法识别真正的弱类（如CIFAR-10-LT中class 3虽样本多但性能差）。尾部类的对抗鲁棒性远低于头部类。

2. **鲁棒性过拟合严重**: BSL使模型在第~25个epoch达到鲁棒性峰值后逐渐下降，自然准确率上升但对抗鲁棒性不升反降，特别是在学习率调整阶段不稳定。

此外，长尾鲁棒性研究一直忽视了**平衡准确率**这一长尾识别中的关键指标，导致评估不够全面。

## 方法详解

### 整体框架

两阶段训练：阶段一（初始稳定化）——使用标准交叉熵损失和PGD对抗训练快速收敛、稳定准确率；阶段二（HARL均衡化）——切换到层级均衡损失$\mathcal{L}_{\text{HEL}} = \alpha \cdot \mathcal{L}_{\text{BCL}} + \beta \cdot \mathcal{L}_{\text{HDL}} + \gamma \cdot \mathcal{L}_{\text{RCEL}}$，动态识别并增强弱类性能。

### 关键设计1：层级对抗鲁棒学习(HARL)的三组件损失

- **功能**: 全面均衡各类别在对抗训练中的性能
- **核心思路**: 三个互补的损失：(a) **BCL (Balanced Cross-Class Loss)** — 对所有类的损失取均值$\mathcal{L}_{\text{BCL}} = \frac{1}{S_c}\sum_{c=1}^C \mathcal{L}_c$，防止头部类主导总损失；(b) **HDL (Hierarchical Deviation Loss)** — 惩罚各类损失偏离均值的程度$\mathcal{L}_{\text{HDL}} = \frac{1}{S_c}\sum_{c=1}^C (\mathcal{L}_c - \bar{\mathcal{L}})^2$，减少类间性能差距；(c) **RCEL (Rare Class Emphasis Loss)** — 对稀有类（高损失类）赋予更高权重$\mathcal{L}_{\text{RCEL}} = \sum_{c=1}^C (\frac{\mathcal{L}_c}{\sum_j \mathcal{L}_j})^2$
- **设计动机**: 不同于BSL仅根据样本数量识别尾类，HARL基于训练中的平均损失动态识别弱类——这更准确（如class 3样本多但性能差的情况）。三个损失从不同角度促进均衡化

### 关键设计2：两阶段训练策略

- **功能**: 缓解鲁棒性过拟合，同时优化准确率和鲁棒性
- **核心思路**: 阶段一用交叉熵损失在对抗样本上训练若干epoch，让模型快速收敛到稳定的准确率基线；阶段二转用HARL损失进行均衡化对抗训练
- **设计动机**: 实验发现交叉熵损失在早期训练中能提供稳定的梯度信号，比直接使用BSL更不容易鲁棒性过拟合。早期CE→后期HARL的切换实现了稳定性和均衡性的兼顾

### 关键设计3：平衡鲁棒性(Balanced Robustness)评估指标

- **功能**: 在长尾场景下公平评估各类的对抗鲁棒性
- **核心思路**: 将平衡准确率的概念扩展到对抗样本：$\text{BR} = \frac{1}{S_C}\sum_{i=1}^C \frac{TP_i^{x'}}{TP_i^{x'} + FN_i^{x'}}$，计算各类在对抗样本上的类别召回率均值
- **设计动机**: 传统整体鲁棒性会被头部类主导，掩盖尾部类脆弱性。平衡鲁棒性确保各类贡献相等，对医疗等安全关键领域尤其重要

### 损失函数

阶段一: $\mathcal{L}_{\text{CE}}$（标准交叉熵）
阶段二: $\mathcal{L}_{\text{HEL}} = \alpha \cdot \mathcal{L}_{\text{BCL}} + \beta \cdot \mathcal{L}_{\text{HDL}} + \gamma \cdot \mathcal{L}_{\text{RCEL}}$

## 实验关键数据

### 主实验：CIFAR-10-LT (IR=10) ResNet-18 平衡准确率/鲁棒性

| 方法 | Clean BA↑ | PGD-20 BR↑ | PGD-100 BR↑ | AA BR↑ |
|------|----------|-----------|------------|-------|
| AT | 69.00 | 25.69 | 24.55 | 24.28 |
| AT-BSL | 72.74 | 26.86 | 25.62 | 25.26 |
| RoBal | 73.18 | 27.12 | 26.98 | 24.13 |
| REAT | 74.56 | 24.02 | 22.52 | 22.69 |
| **TAET** | **76.22** | **30.12** | **28.45** | **27.31** |

### 消融实验：各组件贡献

| 配置 | 效果 |
|------|------|
| Only BCL | 均衡基础改善 |
| BCL + HDL | 类间差距显著缩小 |
| BCL + HDL + RCEL | 最优均衡效果 |
| 两阶段 vs 直接HARL | 两阶段显著减少鲁棒性过拟合 |

### 关键发现

- TAET在balanced accuracy和balanced robustness上同时超越所有基线
- t-SNE可视化显示TAET在尾部类的特征分离度明显优于AT、TRADES、AT-BSL
- 基于损失的弱类识别比基于样本数量的更准确——能捕获样本数多但性能差的类别
- 交叉熵预训练阶段有效缓解了BSL方法的鲁棒性过拟合现象
- 内存和计算效率也优于RoBal等两阶段方法

## 亮点与洞察

1. **Balanced Robustness指标的引入**: 填补了长尾鲁棒性评估的空白，使尾部类的脆弱性visible
2. **基于损失的弱类识别**: 比基于样本数量更灵活准确，能发现"隐藏的弱类"
3. **两阶段策略的简洁性**: CE预训练→HARL均衡化的切换简单有效，无需复杂的课程学习设计

## 局限与展望

- 目前主要在CIFAR-10-LT上验证，大规模数据集（ImageNet-LT）的结果需补充
- HARL的三个超参数$\alpha, \beta, \gamma$需要调节
- 未探索与其他对抗训练方法（如TRADES）组合的可能性

## 相关工作与启发

- 将平衡准确率从长尾识别引入对抗鲁棒性评估的思路可推广到其他安全关键的不均衡场景
- 两阶段训练策略对其他需要在训练过程中切换目标的场景有参考价值

## 评分

⭐⭐⭐⭐ — 问题实际且重要（真实世界数据确实长尾），Balanced Robustness指标是有价值的贡献。方法简洁有效，各组件有清晰的设计动机。消融实验充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Confusion-Aware Spectral Regularizer for Long-Tailed Recognition](../../CVPR2026/others/confusion-aware_spectral_regularizer_for_long-tailed_recognition.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](../../ICML2025/others/fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)

</div>

<!-- RELATED:END -->

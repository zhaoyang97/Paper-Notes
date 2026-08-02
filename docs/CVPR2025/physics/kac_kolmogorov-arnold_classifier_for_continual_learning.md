---
title: >-
  [论文解读] KAC: Kolmogorov-Arnold Classifier for Continual Learning
description: >-
  [CVPR 2025][物理/科学计算][KAN] 首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。
tags:
  - "CVPR 2025"
  - "物理/科学计算"
  - "KAN"
  - "continual learning"
  - "RBF"
  - "Kolmogorov-Arnold"
  - "classifier"
---

# KAC: Kolmogorov-Arnold Classifier for Continual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.21076](https://arxiv.org/abs/2503.21076)  
**代码**: [https://github.com/Ethanhuhuhu/KAC](https://github.com/Ethanhuhuhu/KAC)  
**领域**: LLM评测  
**关键词**: KAN, continual learning, RBF, Kolmogorov-Arnold, classifier

## 一句话总结
首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

## 研究背景与动机

**领域现状**：基于预训练 ViT 的持续学习方法（L2P、DualPrompt、CODAPrompt、CPrompt）在增量学习中取得显著进展，但通常使用简单的线性分类头，限制了模型在新旧任务间的判别能力。

**现有痛点**：线性分类器表达能力有限，无法充分建模复杂类别边界。任务数增多时问题尤为严重——40-step 增量学习场景中性能急剧下降。用 MLP 替代效果反而更差（相同参数量下准确率降低 0.36%），说明简单增加非线性层不是解决方案。

**核心矛盾**：持续学习需要分类器既有强大表达能力以区分累积的大量类别，又要足够轻量以避免在有限数据上过拟合。线性分类器太弱，MLP 又容易过拟合。

**本文目标** 设计一个轻量但表达力强的分类器，即插即用地改善现有持续学习方法。

**切入角度**：Kolmogorov-Arnold 表示定理表明任何连续多变量函数可用单变量函数的组合和加法表示。KAN 通过在边上放置可学习激活函数提供更灵活的函数逼近能力。

**核心 idea**：用 RBF 替代 KAN 中的 B-spline，利用 RBF 的高斯混合解释和局部化特性，构建参数极少但表达力强的持续学习分类器。

## 方法详解

### 整体框架
KAC 以即插即用方式替换现有持续学习方法中的线性分类头。输入为 ViT backbone 提取的特征，经 Layer Normalization 后进入 KAC 分类器输出类别预测。仅改变分类头，不修改 backbone 或 prompt 机制。

### 关键设计

1. **RBF-based KAN Layer**

    - 功能：用径向基函数替代 KAN 中的 B-spline 激活函数
    - 核心思路：paper_notes/docs/CVPR2025/ai_safety/deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md(x) = \sum_p \Phi_p \sum_{i=1}^N \omega_{p,i} arphi(||x_p - c_i||)$，其中 $arphi(r) = \exp(-r^2/(2\sigma_i^2))$ 为高斯 RBF
    - 分布解释：$\sum_i \omega_{p,i} \mathcal{N}(c_i, \sigma_i^2)$ 等价于高斯混合模型
    - 设计动机：B-spline 在持续学习中需频繁更新网格点不够稳定；RBF 中心固定（-2到2均匀分布），宽度固定（$\sigma=1$），仅需学习权重 $\omega$

2. **KAC 架构设计**

    - 功能：将 RBF-KAN 具体化为分类器
    - 核心思路：输入经 LayerNorm → N=4 个高斯 RBF 映射 → $\text{Diag}(W_C \cdot \Phi(\text{LN}(F(x))) \cdot W_q)$
    -  \in \mathbb{R}^{C \times n}$ 为输出权重， \in \mathbb{R}^{N \times C}$ 为类别特定 RBF 权重
    - 设计动机：对角化确保每个类别有独立激活模式，仅增加 0.23M 参数（vs ViT-B/16 的 86M）

### 损失函数 / 训练策略
- 直接使用原始持续学习方法的损失函数，KAC 只替换分类头
- 不引入额外正则化或蒸馏损失
- 与 L2P、DualPrompt、CODAPrompt、CPrompt 四种方法均兼容

## 实验关键数据

### 主实验

**ImageNet-R（40-step）：**

| 方法 | Baseline | +KAC | 提升 |
|------|---------|------|------|
| L2P | 74.28 | 76.34 | +2.06 |
| DualPrompt | 74.51 | 76.87 | +2.36 |
| CODAPrompt | 76.80 | 79.79 | +2.99 |
| CPrompt | 78.98 | 80.89 | +1.91 |

**CUB200（40-step，提升最显著）：**

| 方法 | Baseline | +KAC | 提升 |
|------|---------|------|------|
| L2P | 46.84 | 66.08 | **+19.24** |
| DualPrompt | 50.61 | 71.31 | **+20.70** |
| CODAPrompt | 52.57 | 71.36 | +18.79 |
| CPrompt | 77.34 | 85.11 | +7.77 |

### 消融实验

| 配置 | ImageNet-R 20-step |
|------|-------------------|
| Linear classifier | 80.92% |
| MLP (same params) | 80.56% |
| MLP (fixed) | 65.87% |
| **KAC (RBF)** | **83.59%** |

| N (RBF数) | 表现 |
|-----------|------|
| 2 | 最低 |
| 4 | 最优 ✓ |
| 8 | 与N=4相当 |
| 16 | 轻微下降 |

### 关键发现
- **任务步数越多，KAC 增益越大**：5-step 提升 0.2-4.4%，40-step 提升 2-20%
- **细粒度数据集增益最大**：CUB200 提升远超 ImageNet-R，RBF 的局部化特性擅长精细类别边界
- **稳定性显著提升**：L2P 在 CUB200 上标准差从 5.06 降至 0.57
- MLP 不如线性分类器（-0.36%），证明问题在于 KAN 特有的可学习激活设计

## 亮点与洞察
- **极简但极效**：仅 0.23M 参数的改动就在所有方法上获得一致提升
- **RBF 的高斯混合解释巧妙**：将分类转化为特征在高斯混合空间中的定位
- **步数越多增益越大**：现实应用中增量步数往往很多，这是现有方法最薄弱的环节
- 可迁移到 few-shot learning、open-set recognition 等需要轻量有效分类器的场景

## 局限与展望
- 所有实验基于 ViT-B/16 backbone，未验证与其他 backbone 兼容性
- RBF 中心和宽度手动设定，自适应确定可能进一步提升
- 未与 exemplar-based 方法（DER、BiC）结合验证
- DomainNet 上提升较小（+1-2%），domain incremental 可能需额外机制

## 相关工作与启发
- **vs Linear Classifier**: 表达力不足，尤其长序列和细粒度任务
- **vs MLP**: 相同参数量下性能更差，证明 KAN 优势不在于简单加深网络
- **vs 原始 KAN (B-spline)**: B-spline 需网格更新，RBF 固定中心只学权重，更适合增量场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 KAN 引入持续学习，RBF 替代 B-spline 动机清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 四种方法、四个数据集、多步数设置
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验丰富
- 价值: ⭐⭐⭐⭐ 即插即用特性使其很容易被采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Catastrophic Forgetting in Kolmogorov-Arnold Networks](../../AAAI2026/physics/catastrophic_forgetting_in_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/physics/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/physics/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/physics/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/physics/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)

</div>

<!-- RELATED:END -->

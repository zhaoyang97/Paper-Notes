---
title: >-
  [论文解读] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks
description: >-
  [CVPR 2025][概率边际攻击] 本文提出 Probability Margin Attack (PMA)，在概率空间而非 logits 空间定义对抗边际损失函数，其梯度等价于无目标和有目标交叉熵损失的自适应加权组合，一致性地超越现有个体攻击方法；基于此构建百万级评估数据集 CC1M，首次开展对抗训练模型的百万规模白盒鲁棒性评估。
tags:
  - "CVPR 2025"
  - "概率边际攻击"
  - "对抗鲁棒性评估"
  - "白盒攻击"
  - "百万级评估"
  - "损失函数设计"
---

# Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks

**会议**: CVPR 2025  
**arXiv**: [2411.15210](https://arxiv.org/abs/2411.15210)  
**代码**: 无  
**领域**: 其他（对抗鲁棒性）  
**关键词**: 概率边际攻击, 对抗鲁棒性评估, 白盒攻击, 百万级评估, 损失函数设计

## 一句话总结

本文提出 Probability Margin Attack (PMA)，在概率空间而非 logits 空间定义对抗边际损失函数，其梯度等价于无目标和有目标交叉熵损失的自适应加权组合，一致性地超越现有个体攻击方法；基于此构建百万级评估数据集 CC1M，首次开展对抗训练模型的百万规模白盒鲁棒性评估。

## 研究背景与动机

**领域现状**：白盒对抗攻击是评估模型对抗鲁棒性的核心工具。PGD 是强效的一阶攻击，AutoAttack (AA) 是集成四种攻击的标准评估方法。RobustBench 排行榜上的模型均使用 AA 进行鲁棒性验证。

**现有痛点**：PGD 在遇到混淆梯度时效果不佳，AA 计算成本极高（运行时间常超过训练时间），在大规模评估场景下不可行。现有损失函数存在缺陷：交叉熵同时向所有非目标方向探索但难以集中于最优方向；有目标交叉熵集中于 $z_{max}$ 但抑制了其他攻击方向的探索；margin loss ($z_{max} - z_y$) 只看两个 logit 忽略其他类别信息；DLR 虽额外考虑第三大 logit 但仍遗漏其他维度。

**核心矛盾**：个体攻击效率高但强度不足，集成攻击强度高但效率低——在百万级评估需求下，需要接近集成攻击强度的高效个体攻击方法。

**本文目标**：(1) 设计更强的个体攻击损失函数；(2) 构建高效的集成攻击方案；(3) 开展百万级规模的鲁棒性评估。

**切入角度**：将对抗边际从 logits 空间搬到概率空间——概率空间的分母自然包含所有类别的 logit 信息，开辟更多攻击方向的探索可能。

**核心 idea**：提出概率边际损失 $\mathcal{L}_{pm} = p_{max} - p_y = (e^{z_{max}} - e^{z_y}) / \sum_i^N e^{z_i}$，通过梯度分析证明 PM loss 的梯度是无目标 CE 和有目标 CE 梯度的概率加权组合，天然平衡探索与聚焦。

## 方法详解

### 整体框架

PMA 采用 Margin Decomposition (MD) 攻击的两阶段管线，替换损失函数为概率边际损失。阶段一交替使用 $p_{max}$ 和 $-p_y$ 两个分解项进行充分探索（每次 restart 交替选择），阶段二使用完整的 $p_{max} - p_y$ 进行精细优化。步长采用余弦衰减策略。在此基础上构建两种集成攻击（PMA+AA、PMA+APGD）以平衡效率和强度，同时建立 CC1M 百万级评估数据集。

### 关键设计

1. **概率边际损失 (PM Loss)**:

    - 功能：在概率空间定义对抗边际，统一无目标和有目标攻击的优势
    - 核心思路：$\mathcal{L}_{pm}(z, y) = p_{max} - p_y = (e^{z_{max}} - e^{z_y}) / \sum_i^N e^{z_i}$。其梯度可分解为 $\nabla_x \mathcal{L}_{pm} = p_y \nabla_x \mathcal{L}_{ce} + p_{max} \nabla_x \mathcal{L}_{cet}$，即无目标 CE（多方向探索）和有目标 CE（集中攻击 $z_{max}$）的加权组合。权重由 $p_y$ 和 $p_{max}$ 动态调节：当 $p_{max}$ 接近 $p_y$ 时，其他方向的贡献被正则化抑制，攻击自动聚焦于 $z_{max}$ 方向
    - 设计动机：相比 margin loss 只用两个 logit，PM loss 的分母 $\sum e^{z_i}$ 包含所有类别信息，计算成本不增加但攻击方向更丰富。理论分析表明 margin loss 等价于 CE + 有目标 CE 的简单加法，而 PM loss 是其自适应加权版本

2. **两阶段攻击策略（PMA 攻击流程）**:

    - 功能：通过分阶段交替优化 PM loss 的两个分量，充分探索攻击空间
    - 核心思路：阶段一（$k < K^1$）：奇数次 restart 优化 $p_{max}$（鼓励最大非目标类概率上升），偶数次优化 $-p_y$（鼓励真实类概率下降），通过 restart 次数的奇偶交替实现方向探索。阶段二（$K^1 < k < K$）：使用完整 PM loss $p_{max} - p_y$，从阶段一提供的良好初始化出发做精细优化。步长余弦衰减：$\alpha = \epsilon \cdot (1 + \cos(k/K \cdot \pi))$
    - 设计动机：借鉴 MD 攻击的边际分解思想，但将 logits 空间的分解替换为概率空间的分解。通过阶段一的充分探索为阶段二提供更好的初始扰动

3. **CC1M 百万级评估数据集**:

    - 功能：构建首个百万规模的白盒对抗鲁棒性评估基准
    - 核心思路：从 CC3M 数据集出发，筛除异常值后保留 100 万张图像。用对抗训练的 ImageNet 模型对 CC1M 进行评估，对比在标准 ImageNet-1k 测试集（5万张）上的评估结果。构建 PMA+APGD 和 PMA+AA 两种集成攻击以在效率和强度间取得平衡
    - 设计动机：现有评估都在 5 万张测试集上进行，可能不反映真实世界的对抗风险分布。百万级评估能揭示小规模评估遗漏的鲁棒性差距

### 损失函数 / 训练策略

PMA 使用概率边际损失 $\mathcal{L}_{pm} = p_{max} - p_y$ 作为对抗目标，配合 $L_\infty$ 约束和 PGD 投影。CIFAR-10/100 扰动预算 $\epsilon = 8/255$，ImageNet $\epsilon = 4/255$。攻击步数 100，使用随机初始化和余弦步长衰减。

## 实验关键数据

### 主实验

PM loss vs 现有损失函数（PGD 策略，鲁棒准确率 %，越低攻击越强）：

| 数据集 | 模型 | CE | DLR | Margin | **PM (Ours)** | 降低 |
|--------|------|-----|-----|--------|-------------|------|
| CIFAR-10 | WRN-70-16 | 73.60 | 71.61 | 71.56 | **71.34** | -0.22 |
| CIFAR-10 | WRN-70-16 (Rebuffi) | 69.56 | 67.96 | 67.79 | **67.55** | -0.24 |
| CIFAR-100 | WRN-70-16 | 48.19 | 43.93 | 43.90 | **43.66** | -0.24 |
| ImageNet | Swin-L | 59.47 | 59.80 | 59.31 | **57.97** | -1.34 |
| ImageNet | ConvNeXt-L | 58.42 | 59.17 | 58.63 | **57.24** | -1.18 |

PMA vs 现有个体攻击方法（鲁棒准确率 %）：

| 数据集 | 模型 | MD | FAB | **PMA (Ours)** | vs AA |
|--------|------|-----|-----|---------------|-------|
| CIFAR-10 | WRN-70-16 (Peng) | 71.14 | 72.31 | **71.10** | 71.10 |
| CIFAR-100 | WRN-70-16 | 43.53 | 44.73 | **43.39** | 43.40 |
| ImageNet | Swin-L | 56.89 | 60.05 | **56.49** | 56.56 |

### 消融实验

| 配置 | CIFAR-10 Avg. | ImageNet Avg. | 说明 |
|------|-------------|-------------|------|
| PM loss (full) | **最佳** | **最佳** | 完整概率边际 |
| 仅 $p_{max}$ 项 | 次之 | 次之 | 缺少 $p_y$ 方向 |
| 仅 $-p_y$ 项 | 最差 | 最差 | 缺少目标聚焦 |
| Margin loss | 介于之间 | 介于之间 | logits 空间边际 |

### 关键发现

- PM loss 在所有 29 个模型上一致性地超越 CE、DLR 和 margin loss，优势在类别数越多的数据集上越明显（ImageNet 平均降低 1.22% 鲁棒准确率）
- PMA 作为个体攻击已接近甚至匹配 AutoAttack 的性能（集成 4 种方法），同时计算效率高数倍
- 百万级评估揭示了重要的鲁棒性缺口：在 CC1M 上的评估鲁棒性显著低于 ImageNet-1k 测试集上的结果，说明小规模评估存在高估风险
- PM loss 的自适应加权机制是其优势的核心——当 $p_{max}$ 接近决策边界时自动增强攻击聚焦

## 亮点与洞察

- PM loss 的理论分析非常优雅：一个简单公式优雅地统一了无目标和有目标 CE 的双重优势，梯度推导 $\nabla \mathcal{L}_{pm} = p_y \nabla \mathcal{L}_{ce} + p_{max} \nabla \mathcal{L}_{cet}$ 清晰揭示了其自适应机制
- 百万级评估首次揭示了小规模测试集可能高估模型鲁棒性，对安全关键应用有重要警示意义
- MD 攻击管线 + PM loss 的组合思路值得借鉴——好的攻击策略和好的损失函数是正交的改进维度

## 局限与展望

- PM loss 的优势在 CIFAR-10 上较小（平均 ~0.2%），主要优势体现在类别多的数据集上
- CC1M 数据集的分布与 ImageNet 不完全一致，鲁棒性差距可能部分来自分布偏移而非鲁棒性评估不足
- 仅关注 $L_\infty$ 攻击，未验证在 $L_2$、$L_1$ 等其他范数约束下的效果
- 未来方向：(1) 将 PM loss 扩展到生成模型的对抗评估；(2) 研究更大规模（千万级）评估的实际影响；(3) 探索 PM loss 在对抗训练中的应用

## 相关工作与启发

- **vs AutoAttack**：AA 集成 4 种攻击（2×APGD + FAB + Square），计算成本高。PMA 作为单一攻击已能匹配 AA 效果（部分模型上 PMA 鲁棒准确率甚至低于 AA）
- **vs Margin Decomposition (MD)**：PMA 沿用 MD 的两阶段管线但替换了损失函数，在 MD 基础上进一步降低评估鲁棒准确率
- **vs DLR loss**：DLR 在 logits 空间考虑第三大 logit 做归一化，PM loss 在概率空间通过 softmax 分母天然纳入所有类别信息

## 评分

- **新颖性**: ⭐⭐⭐⭐ — PM loss 公式简洁优雅，梯度统一性分析有理论深度
- **实验充分度**: ⭐⭐⭐⭐⭐ — 29 个模型、3 个数据集、个体/集成/百万级评估覆盖全面
- **写作质量**: ⭐⭐⭐⭐ — 梯度分析推导清晰，五种损失函数的对比表格直观
- **价值**: ⭐⭐⭐⭐ — 对对抗鲁棒性评估社区有实际贡献，PM loss 可直接用于更精准的鲁棒性测量

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](../../CVPR2026/others/irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[CVPR 2025\] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)
- [\[CVPR 2025\] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)
- [\[ICML 2025\] Probably Approximately Global Robustness Certification](../../ICML2025/others/probably_approximately_global_robustness_certification.md)

</div>

<!-- RELATED:END -->

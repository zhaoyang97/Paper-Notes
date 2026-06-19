---
title: >-
  [论文解读] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI
description: >-
  [NeurIPS 2025][AI安全][algorithmic fairness] 通过因果图模型分析表明，分组评估（disaggregated evaluation）中跨子群体的性能差异不一定意味着不公平，而可能是数据生成过程中分布差异的自然结果，建议结合因果假设和加权评估补充标准分组评估。 领域现状 领域现状：分组评估…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "algorithmic fairness"
  - "disaggregated evaluation"
  - "causal inference"
  - "distribution shift"
  - "subgroup performance"
---

# Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI

**会议**: NeurIPS 2025  
**arXiv**: [2506.04193](https://arxiv.org/abs/2506.04193)  
**代码**: [GitHub](https://github.com/google-research/google-research/tree/master/causal_evaluation)  
**领域**: AI安全  
**关键词**: algorithmic fairness, disaggregated evaluation, causal inference, distribution shift, subgroup performance

## 一句话总结

通过因果图模型分析表明，分组评估（disaggregated evaluation）中跨子群体的性能差异不一定意味着不公平，而可能是数据生成过程中分布差异的自然结果，建议结合因果假设和加权评估补充标准分组评估。

## 研究背景与动机

### 领域现状

**领域现状**：分组评估是评估 ML 模型公平性的标准做法：将模型性能按子群体（如种族、性别）拆分，将性能差异视为公平性问题的证据。然而，本文指出这种做法可能产生误导：

### 现有痛点

**现有痛点**：即使是 Bayes 最优模型（零估计误差），在不同子群体间通常也不会获得相等的性能

### 核心矛盾

**核心矛盾**：性能差异可能是数据在子群体间分布差异的自然结果，而非模型缺陷

### 解决思路

**解决思路**：当存在选择偏差时，分组评估和基于条件独立性的替代方法都可能失效

### 补充说明

**补充说明**：强制实现跨子群体性能相等的算法策略可能直接引入伤害

这些发现对如何设计和解释模型评估有广泛影响。

## 方法详解

### 整体框架

本文使用因果有向无环图（DAGs）描述子群体间异质性的不同数据生成过程，分析在每种设定下 Bayes 最优模型的公平性属性和性能指标稳定性，进而提出加权评估作为控制混杂的方法。

### 关键设计

1. **因果图描述子群体异质性**: 定义了因果方向（X→Y）和反因果方向（Y→X）各三种+选择偏差设定：

    - **因果方向**：协变量偏移（$P(X|A)$ 不同但 $P(Y|X,A)=P(Y|X)$）、结果偏移（$P(X|A)$ 相同但 $P(Y|X,A) \neq P(Y|X)$）、复合偏移
    - **反因果方向**：标签偏移（$P(Y|A)$ 不同但 $P(X|Y,A)=P(X|Y)$）、表现偏移（$P(Y|A)$ 相同但 $P(X|Y,A) \neq P(X|Y)$）、复合偏移
    - 用双向边表示未观测混杂因子的影响，而非将 A 视为 X 或 Y 的直接原因（反映社会结构决定因素的间接作用）

2. **Bayes 最优模型的公平性属性分析**: 

    - **Sufficiency** 准则（$Y \perp A | R$）：子群体 Bayes 最优预测器总是满足 sufficiency，但人口级 Bayes 最优预测器仅在 $Y \perp A | X$（协变量偏移）下才满足
    - **Separation** 准则（$R \perp A | Y$）：仅在标签偏移下且子群体不可知预测时才被满足
    - 关键结论：$Y \perp A | X$ 是决定性条件——当其成立时，人口级最优=子群体最优，无需子群体感知建模

3. **性能指标稳定性分析（Table 2）**: 核心条件是 $\{R,Y\} \perp A | V$ 对控制变量 $V \in \{X, Y, R\}$：

    - 在所有设定中，$\{R,Y\} \perp A$ 均不成立→不应期望任意指标的跨子群体相等性
    - 协变量偏移下：控制 X 即可解释性能差异（对子群体不可知模型）
    - 标签偏移下：控制 Y 即可解释性能差异
    - 结果偏移/表现偏移/复合偏移下：X 和 Y 单独都不能解释差异
    - 满足 sufficiency 的模型：控制 R 可解释差异

4. **加权评估方法**: 当差异可由 X 的分布差异解释时，使用逆概率加权（类似于倾向得分方法）构造加权性能估计，使不同子群体的 X 分布对齐。这等价于一类可配置的条件独立性检验。

### 损失函数 / 训练策略

本文不提出新的训练方法。核心贡献是分析框架和评估方法论。加权评估使用标准的逆概率加权/倾向得分技术。

## 实验关键数据

### 主实验（理论属性验证）

| 因果设定 | Y⊥A\|X? | Sufficiency(f*) | Separation(f*) | X可解释差异? |
|---------|---------|----------------|----------------|------------|
| 协变量偏移 | ✓ | ✓ | ✗ | ✓ |
| 结果偏移 | ✗ | ✗(人口级) | ✗ | ✗ |
| 标签偏移 | ✗ | ✗(人口级) | ✓(不可知) | ✗ |
| 表现偏移 | ✗ | ✗(人口级) | ✗ | ✗ |
| 复合偏移 | ✗ | ✗(人口级) | ✗ | ✗ |

所有设定中子群体Bayes最优预测器均满足 sufficiency。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不加控制的分组评估 | 性能差异≠不公平 | 在协变量偏移下差异完全由X分布造成 |
| 控制X的加权评估 | 消除混杂 | 仅在Y⊥A\|X时有效 |
| 控制Y的加权评估 | 消除标签分布差异 | 仅在标签偏移下有效 |
| 子群体感知 vs 不可知 | 不可知更稳定 | 但在Y⊥̸A\|X时牺牲精度 |
| 子群体可分性 | 行为类似Y⊥A\|X | 极端分离比下人口级≈子群体级 |
| 选择偏差扩展 | 评估可能失效 | 需明确偏差机制的假设 |

### 关键发现

- 跨子群体性能相等不是公平的可靠度量：Bayes 最优模型在存在分布差异时本就不会实现相等性能
- 强制约束（如等化赔率约束优化）可能直接引入伤害，而非实现公平
- 协变量偏移是唯一可通过控制 X 完全解释性能差异的情况
- 标签偏移是唯一 separation 准则自然成立的情况
- 当因果结构不清楚时，简单的分组评估不足以支撑公平性判断

## 亮点与洞察

- 对一个被广泛使用但很少被质疑的评估范式提出了系统性的理论挑战
- 用因果图统一了多种分布偏移类型，使得不同场景下的属性可以对比分析
- Table 2 的综合分析提供了极有价值的实践参考：告诉从业者在什么因果假设下可以使用什么控制变量
- 强调公平不是模型属性而是部署政策的效果属性，这一视角对政策制定者尤为重要

## 局限与展望

- 理论分析主要在 Bayes 最优假设下，实际模型性能差异可能同时包含估计误差和分布差异两个来源
- 因果图选择本身需要领域知识，而实际中因果结构往往未知
- 加权评估依赖于准确的倾向得分估计，在高维协变量空间中可能不准确
- 仅考虑二值标签和简单因果图，复杂场景（多标签、连续输出、多层因果结构）未覆盖
- 实验以合成和简单真实数据为主，大规模复杂 ML 系统的验证不足

## 相关工作与启发

- Liu et al. (2019) 关于 sufficiency/separation/calibration 不可兼得的不可能定理是本文的直接理论基础
- Cai et al. 的加权评估方法被本文进一步发展和因果解读
- Mhasawade et al. 使用因果模型分析算法公平性是方法论上的先驱
- 与临床预测公平性文献（心血管风险预测、医学影像诊断）有直接的应用联系

## 评分

- 新颖性: ⭐⭐⭐⭐ 对广泛使用的评估范式提出理论挑战，分析框架完整
- 实验充分度: ⭐⭐⭐ 以理论分析为主，实验主要用于验证理论性质
- 写作质量: ⭐⭐⭐⭐⭐ 论证严谨缜密，因果图的使用使复杂问题可视化
- 价值: ⭐⭐⭐⭐⭐ 对 AI 公平性评估实践有深远影响，应成为所有做分组评估者的必读文献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D](ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)
- [\[NeurIPS 2025\] Matchings Under Biased and Correlated Evaluations](matchings_under_biased_and_correlated_evaluations.md)
- [\[ICML 2025\] Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape](../../ICML2025/ai_safety/distributed_and_decentralised_training_technical_governance_challenges_in_a_shif.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)

</div>

<!-- RELATED:END -->

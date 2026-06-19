---
title: >-
  [论文解读] To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers
description: >-
  [ICML 2025][hierarchical classification] 本文提出了针对概率层次分类器的后处理最优解码框架，为不同评价指标（如层次 $F_\beta$）推导了最优决策规则，在候选集限于节点集时给出通用算法，对子集预测推导了专门的层次 $hF_\beta$ 最优策略。 领域现状：层次分类利用标签的树状结…
tags:
  - "ICML 2025"
  - "hierarchical classification"
  - "decision rules"
  - "optimal decoding"
  - "F-beta score"
  - "probabilistic classifiers"
---

# To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers

**会议**: ICML 2025  
**arXiv**: [2506.01552](https://arxiv.org/abs/2506.01552)  
**代码**: [https://github.com/RomanPlaud/hierarchical_decision_rules](https://github.com/RomanPlaud/hierarchical_decision_rules)  
**领域**: 分类学习 / 层次分类  
**关键词**: hierarchical classification, decision rules, optimal decoding, F-beta score, probabilistic classifiers

## 一句话总结
本文提出了针对概率层次分类器的后处理最优解码框架，为不同评价指标（如层次 $F_\beta$）推导了最优决策规则，在候选集限于节点集时给出通用算法，对子集预测推导了专门的层次 $hF_\beta$ 最优策略。

## 研究背景与动机
**领域现状**：层次分类利用标签的树状结构来区分错误严重程度（如将"狗"错分为"猫"比错分为"汽车"更可接受）。现有方法通常训练概率模型后用启发式规则做预测解码。

**现有痛点**：启发式解码（如选择最大概率的叶节点，或沿树贪心搜索）未必与评价指标对齐。不同指标需要不同的最优策略，但缺乏系统的最优解码理论。

**核心矛盾**：概率模型提供了丰富的不确定性信息，但解码时未充分利用这些信息来优化目标指标。

**本文目标**：给定一个已训练的概率层次分类器和目标评价指标，如何后处理（解码）以最优化该指标。

**切入角度**：Bayes 最优决策理论——条件期望损失最小化。

**核心idea**：不同指标的最优解码规则不同，作者为一系列指标推导了解析或算法化的最优规则。

## 方法详解

### 整体框架
输入：训练好的概率层次分类器 $P(y|x)$（$y$ 是层次结构中的节点）、目标指标 $\Psi$
输出：对新样本 $x$ 的最优预测 $\hat{y}^*(x)$

### 关键设计

1. **节点预测的最优规则**:

    - 功能：当预测候选是层次中的单个节点时，推导最优预测
    - 核心思路：对于给定概率分布 $P(y|x)$ 和度量 $\Psi$，最优预测为 $\hat{y}^* = \arg\max_{v \in \mathcal{V}} \mathbb{E}[\Psi(v, y) | x]$。对于不同指标具体化：
        - 0-1 层次损失：选概率最大的叶节点
        - 树距离损失：选加权深度最优的节点
        - 层次 $F_1$：选使期望 $F_1$ 最大的节点（需搜索所有节点）
    - 设计动机：提供普适算法，确保对任何指标都能找到最优节点预测

2. **子集预测的最优规则（$hF_\beta$ 专用）**:

    - 功能：当预测可以是一组节点时（如预测一条从根到叶的路径），推导层次 $F_\beta$ 的最优策略
    - 核心思路：层次 $F_\beta$ 度量考虑预测的节点集合和真实标签的祖先集合之间的重叠：
    $hF_\beta = \frac{(1+\beta^2) |A(\hat{y}) \cap A(y)|}{(1+\beta^2)|A(y)| + |A(\hat{y})|}$
      其中 $A(v)$ 是节点 $v$ 的祖先集。最优子集预测通过动态规划在树上高效计算
    - 设计动机：子集预测在不确定时更有用（如只预测"动物"而非具体物种），需要任务指标导向的策略

3. **高效计算算法**:

    - 功能：对所有推导出的最优规则提供多项式时间算法
    - 核心思路：利用树的结构进行动态规划。对每个节点 $v$，计算以 $v$ 为根的子树中的最优贡献，自底向上合并
    - 设计动机：暴力搜索所有可能的子集是指数级的，需要利用树结构加速

### 损失函数 / 训练策略
本方法是后处理方法，不改变模型训练。分类器用标准交叉熵训练，解码时使用推导的最优规则。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 最优解码(本文) | 贪心解码 | Top-1叶节点 | 提升 |
|---|---|---|---|---|---|
| iNaturalist (8K类) | $hF_1$ | **0.685** | 0.641 | 0.612 | +7.3% |
| CIFAR100-H | $hF_1$ | **0.762** | 0.731 | 0.718 | +4.3% |
| ImageNet-H | $hF_\beta$ ($\beta=2$) | **0.801** | 0.768 | 0.745 | +4.3% |
| 文本分类 (DBpedia) | tree dist | **0.34** | 0.41 | 0.52 | +17% |

### 消融实验

| 解码策略 | $hF_1$ (iNat) | 计算时间 | 说明 |
|---|---|---|---|
| 最优解码（本文） | **0.685** | 2.3ms/样本 | 动态规划 |
| 贪心沿树搜索 | 0.641 | 0.5ms/样本 | 启发式 |
| 最大概率叶节点 | 0.612 | 0.1ms/样本 | 最简单 |
| 随机按概率采样 | 0.595 | 0.1ms/样本 | 基线 |
| 不同 $\beta$ ($F_0$) | 0.712 | — | 精度导向 |
| 不同 $\beta$ ($F_\infty$) | 0.638 | — | 召回导向 |

### 关键发现
- 最优解码在所有指标和数据集上均显著优于启发式方法
- 改进在"欠定场景"（模型不确定、层次较深）中最大——此时贪心方法的次优性最明显
- 不同 $\beta$ 值导致完全不同的最优策略：高 $\beta$ 倾向于预测较高层（更保守），低 $\beta$ 倾向于预测更具体的叶节点
- 计算开销可接受（动态规划只需毫秒级）

## 亮点与洞察
- "每个指标配一个解码器"的理念简洁有力：同一个模型可以为不同需求提供最优预测
- 理论驱动的实用方法：在不重新训练的情况下，仅通过更好的解码即可提升性能
- 层次 $F_\beta$ 的动态规划解法优雅，充分利用了树结构

## 局限与展望
- 依赖概率模型的校准质量——如果概率估计不准确，最优解码也不是真正最优
- 有向无环图（DAG）层次结构的推广需要额外工作
- 与端到端训练方法（直接优化层次指标）的对比尚未展开，后续可验证后处理方法是否可与端到端训练互补

## 相关工作与启发
- 与 Deng et al. (2014) 的层次分类错误度量相关
- 后处理方法与校准（Platt scaling、温度缩放）有类似精神
- 在生物分类学、电商类目等深层层次结构场景中直接适用

## 评分
- 新颖性: ⭐⭐⭐⭐ 最优层次解码规则的系统推导
- 实验充分度: ⭐⭐⭐⭐ 多种指标和数据集
- 写作质量: ⭐⭐⭐⭐⭐ 理论与实践结合好
- 价值: ⭐⭐⭐⭐ 实用且理论扎实的后处理方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[AAAI 2026\] TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](../../AAAI2026/others/taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/others/potential_field_based_deep_metric_learning.md)
- [\[ICML 2025\] Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model](revisiting_instance-optimal_cluster_recovery_in_the_labeled_stochastic_block_mod.md)

</div>

<!-- RELATED:END -->

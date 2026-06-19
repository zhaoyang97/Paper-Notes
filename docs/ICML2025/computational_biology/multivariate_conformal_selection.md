---
title: >-
  [论文解读] Multivariate Conformal Selection
description: >-
  [ICML2025][计算生物][Conformal Selection] 将 Conformal Selection 从单变量响应推广到多变量设定，提出区域单调性 (Regional Monotonicity) 概念，设计距离型 (mCS-dist) 和学习型 (mCS-learn) 两种非一致性分数，在有限样本下保证 FDR 控制并提升选择功效。
tags:
  - "ICML2025"
  - "计算生物"
  - "Conformal Selection"
  - "多变量响应"
  - "FDR 控制"
  - "非一致性分数"
  - "BH 过程"
  - "区域单调性"
  - "可微排序"
---

# Multivariate Conformal Selection

**会议**: ICML2025  
**arXiv**: [2505.00917](https://arxiv.org/abs/2505.00917)  
**代码**: 无  
**领域**: 优化/理论  
**关键词**: Conformal Selection, 多变量响应, FDR 控制, 非一致性分数, BH 过程, 区域单调性, 可微排序

## 一句话总结

将 Conformal Selection 从单变量响应推广到多变量设定，提出区域单调性 (Regional Monotonicity) 概念，设计距离型 (mCS-dist) 和学习型 (mCS-learn) 两种非一致性分数，在有限样本下保证 FDR 控制并提升选择功效。

## 研究背景与动机

### 现有痛点

**现有痛点**：选择问题的普遍性**：药物发现（筛选高结合亲和力化合物）、精准医学（识别正向治疗效应）、LLM 输出认证（筛选可信生成内容）都需要从候选集中选出满足标准的子集。

### 领域现状

**领域现状**：现有 CS 的局限**：Conformal Selection (Jin & Candès, 2023) 仅支持单变量响应 $y > c$ 的阈值选择，无法处理多维标准（如 LLM 输出需同时满足公平性、安全性、正确性）。

### 解决思路

**解决思路**：多变量 CP 不直接适用**：多变量共形预测构建的置信集形状与预定义目标区域 $R$ 可能不兼容，且仅控制 PCER 而非 FDR。

### 核心矛盾

**核心矛盾**：目标**：在多变量响应设置下，构建同时满足以下条件的选择框架：(1) 有限样本 FDR 控制；(2) 最大化选择功效（Power）；(3) 模型无关。

## 方法详解

### 整体框架 (Algorithm 1)

1. **训练**：构建多变量预测模型 $\hat{\mu}$。
2. **校准**：计算区域单调的非一致性分数 $V_i = V(\bm{x}_i, \bm{y}_i)$，构建 conformal p-values。
3. **阈值化**：应用 BH 过程进行多重检验校正，输出选择集 $\mathcal{S}$。

### 关键设计一：区域单调性 (Definition 3.1)

$$V(\bm{x}, \bm{y}') \leq V(\bm{x}, \bm{y}), \quad \forall \bm{y}' \in R^c, \bm{y} \in R$$

保证 conformal p-value 的保守性 (Proposition 3.2)，从而保证 FDR 控制 (Theorem 3.5)。

### 关键设计二：mCS-dist（距离型分数）

$$V(\bm{x}, \bm{y}) = D_1(\bm{y}, R^c) - D_2(\hat{\mu}(\bm{x}), R^c)$$

- **Regular score**：$D_1 = D_2 = \inf_{\bm{s} \in R^c} \|\cdot - \bm{s}\|_p$
- **Clipped score（更优）**：$D_1 = M \cdot \mathbb{1}\{\bm{y} \notin R^c \cup \partial R\}$，Theorem 4.1 证明 clipped score 在渐近功效上优于 regular score。

### 关键设计三：mCS-learn（学习型分数）

$$V^\theta(\bm{x}, \bm{y}) = M \cdot \mathbb{1}\{\bm{y} \notin R^c \cup \partial R\} - f_\theta(\bm{x}, \bm{y}; R)$$

- 使用可微排序 (soft-rank) 近似 conformal p-value，通过反向传播优化 $f_\theta$。
- 损失函数 $L_2$：直接惩罚 p-value，对目标区域内样本最小化 p-value，对区域外样本增大 p-value。
- Proposition 4.2 证明该分数族包含最优非一致性分数。

## 实验关键数据

### 模拟数据

- 在 2D/5D/10D 高斯混合和各种目标区域（凸/非凸/不规则）上测试。
- mCS-dist 和 mCS-learn 在所有设定下均维持 FDR ≤ $q$，功效显著优于基线方法。
- mCS-learn 在非凸区域和高维场景中优势最为明显。
- 维度从 2 增至 10 时，mCS-dist 功效下降但仍保持 FDR 控制，mCS-learn 下降幅度更小。

### 真实数据

- 药物发现数据集：mCS 在 FDR 控制下实现最高选择功效。
- LLM 对齐：多维对齐分数的选择任务中，mCS 成功筛选出同时满足多维标准的输出。

### 基线方法对比

- **Marginal CS (逐维度独立 CS + Bonferroni 校正)**：功效极低，因多重校正过于保守。
- **CP-based 选择**：仅控制 PCER，FDR 可能超标。
- **Oracle 选择**：已知真实响应的上界参考。
- mCS-dist 和 mCS-learn 均显著优于 Marginal CS，接近 Oracle 选择。

## 亮点与洞察

1. **区域单调性**是核心创新，优雅地将单变量单调性推广到任意维度和目标区域。
2. **mCS-learn 的表达力**：Proposition 4.2 从理论上保证最优分数可被学习型家族覆盖。
3. **实用价值**：提供了从药物发现到 LLM 认证的通用不确定性量化框架。
4. **模块化设计**：预训练模型 $\hat{\mu}$ 可独立于选择过程，灵活集成。

## 局限与展望

- 需要在训练集中划分校准集，减少了用于训练的数据量。
- mCS-learn 需要额外的训练-验证-校准三路划分，数据效率较低。
- 目标区域 $R$ 需要预先定义，自适应目标区域的探索有待开展。
- 计算 $\inf_{\bm{s} \in R^c} \|\cdot\|$ 在复杂区域上可能代价高。
- 当 $|R|$ 非常小或非常大时，功效可能不理想。
- 未探索条件密度估计器与 conformal p-value 的结合。
- 对分布外测试数据的鲁棒性未讨论。

## 相关工作与启发

- **Conformal Selection (Jin & Candès, 2023)**：本文的直接推广，从 $y > c$ 到 $\bm{y} \in R$。
- **Conformal Prediction 多变量推广**：Bates et al. (2021)、Feldman et al. (2023) 等的多变量 CP 方法面向预测集构建而非选择。
- **BH 过程 (Benjamini & Hochberg, 1995)**：用于 FDR 控制的多重检验校正基础。
- **可微排序 (Blondel et al., 2020)**：mCS-learn 中 soft-rank 的技术基础。
- **启发**：可将学习型分数思想推广到条件密度估计或 Bayesian 非参数框架中，亦可探索自适应目标区域的学习。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 区域单调性是简洁而有力的推广
- 实验充分度: ⭐⭐⭐⭐ — 模拟+真实数据全面验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论和算法描述极为清晰
- 价值: ⭐⭐⭐⭐ — 为多标准选择问题提供了严格的统计保证框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)
- [\[ICLR 2026\] ConfHit: Conformal Generative Design with Oracle Free Guarantees](../../ICLR2026/computational_biology/confhit_conformal_generative_design_with_oracle_free_guarantees.md)
- [\[NeurIPS 2025\] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](../../NeurIPS2025/computational_biology/a_unified_framework_for_variable_selection_in_modelbased_clu.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/computational_biology/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2026\] Active Timepoint Selection for Learning Measure-Valued Trajectories](../../ICML2026/computational_biology/active_timepoint_selection_for_learning_measure-valued_trajectories.md)

</div>

<!-- RELATED:END -->

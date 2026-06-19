---
title: >-
  [论文解读] Post Hoc Regression Refinement via Pairwise Rankings
description: >-
  [NeurIPS 2025][计算生物][regression refinement] 提出 RankRefine，一种模型无关的后处理回归改进方法，通过将基础回归器的预测与基于成对排序的估计进行逆方差加权融合，在无需重训练的情况下显著降低预测误差，仅需 20 次成对比较和通用 LLM 即可实现分子性质预测中高达 10% 的 MAE 相对减少。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "regression refinement"
  - "pairwise ranking"
  - "inverse-variance weighting"
  - "LLM ranker"
  - "few-shot learning"
---

# Post Hoc Regression Refinement via Pairwise Rankings

**会议**: NeurIPS 2025  
**arXiv**: [2508.16495](https://arxiv.org/abs/2508.16495)  
**代码**: [ktirta/regref](https://github.com/ktirta/regref)  
**领域**: 计算生物  
**关键词**: regression refinement, pairwise ranking, inverse-variance weighting, LLM ranker, few-shot learning

## 一句话总结

提出 RankRefine，一种模型无关的后处理回归改进方法，通过将基础回归器的预测与基于成对排序的估计进行逆方差加权融合，在无需重训练的情况下显著降低预测误差，仅需 20 次成对比较和通用 LLM 即可实现分子性质预测中高达 10% 的 MAE 相对减少。

## 研究背景与动机

连续属性的准确预测在科学与工程领域至关重要。以分子性质预测（MPP）为例，可靠的物理/化学属性估计能加速药物发现、材料设计和催化剂开发。然而，与计算机视觉或 NLP 不同，许多专业领域面临根本性瓶颈：**获取标注需要专家主导的实验，既昂贵又缓慢**，现实任务常在数据稀缺场景下运行（甚至 50 个标注样本已算充足）。

一个被低估的资源是**专家知识的相对比较形式**：成对判断（"哪个分子的溶解度更高？"）比精确数值标注容易得多。关键洞察：

**人类擅长相对比较而非绝对估计**（Thurstone, 1927）：成对任务降低认知负荷，避免尺度解释偏差

**通用 LLM 继承了这一优势**：预训练语料中富含相对陈述，RLHF 对齐进一步强化了比较能力

**现有方法要么需要重训练，要么只处理排序不处理回归**：缺乏即插即用的后处理方案

## 方法详解

### 整体框架

RankRefine 的核心思想：将基础回归器的预测与基于成对排序的估计进行融合。

给定：
- 基础回归器输出 $\hat{y}_0^{\text{reg}}$ 及其不确定性 $\sigma_{\text{reg}}^2$
- 参考集 $\mathbb{D} = \{(x_i, y_i)\}_{i=1}^k$（可直接从训练集抽取）
- 外部排序器（LLM 或人类专家）提供 $x_0$ 与每个 $x_i$ 的成对比较

RankRefine 算出基于排序的估计 $\hat{y}_0^{\text{rank}}$，然后通过逆方差加权融合两个估计。

### 关键设计

**逆方差加权融合**（RankRefine Fusion Theorem）：

若 $\hat{y}_0^{\text{reg}}$ 和 $\hat{y}_0^{\text{rank}}$ 为独立无偏高斯估计量，最小方差无偏估计量为：

$$\hat{y}_0^* = \sigma_{\text{post}}^2 \left(\frac{\hat{y}_0^{\text{reg}}}{\sigma_{\text{reg}}^2} + \frac{\hat{y}_0^{\text{rank}}}{\sigma_{\text{rank}}^2}\right), \quad \sigma_{\text{post}}^2 = \left(\frac{1}{\sigma_{\text{reg}}^2} + \frac{1}{\sigma_{\text{rank}}^2}\right)^{-1}$$

**核心保证**：$\sigma_{\text{post}}^2 < \sigma_{\text{reg}}^2$，即任何具有有限方差的信息性排序器都能降低期望 MAE。

**基于 Bradley-Terry 模型的排序估计**：

成对概率 $P(x_i \succ x_j) = s(y_i - y_j)$，其中 $s(z) = (1 + e^{-z})^{-1}$。通过最小化负对数似然得到排序估计：

$$\hat{y}_0^{*\text{rank}} = \arg\min_{\hat{y}_0^{\text{rank}}} \left[-\sum_{x_i \in \mathbb{A}} \log s(\hat{y}_0^{\text{rank}} - y_i) - \sum_{x_j \in \mathbb{B}} \log(1 - s(\hat{y}_0^{\text{rank}} - y_j))\right]$$

其中 $\mathbb{A}$ 是排在 $x_0$ 之后的参考集，$\mathbb{B}$ 是排在 $x_0$ 之前的参考集。

**排序估计的方差**（Lemma 3.2）：

$$\sigma_{\text{rank}}^2 \approx \left[\sum_{y_i \in \mathbb{A} \cup \mathbb{B}} s(\Delta_i)(1 - s(\Delta_i))\right]^{-1}, \quad \Delta_i = \hat{y}_0^{*\text{rank}} - y_i$$

这通过逆 Fisher 信息近似得到。

**改进保证的定量条件**（Implication 4）：

要使 $\text{MAE}_{\text{post}} \leq \alpha \cdot \text{MAE}_{\text{reg}}$，只需：

$$\sigma_{\text{rank}}^2 \leq \frac{\alpha^2 \sigma_{\text{reg}}^2}{1 - \alpha^2}$$

### 损失函数 / 训练策略

RankRefine **无需任何训练**——它是纯后处理方法。仅需：
- 基础回归器能输出预测值和不确定性估计（如随机森林、高斯过程）
- 外部排序器提供成对比较结果
- 正则化：当排序器过度自信时，温度化方差 $\sigma_{\text{rank}}^2 \leftarrow \max(\sigma_{\text{rank}}^2, c \cdot \sigma_{\text{reg}}^2)$

## 实验关键数据

### 主实验

**LLM 作为排序器的分子性质预测**（ChatGPT-4o，20 次成对比较）：

| 数据集 | 成对排序准确率 | β (MAE_post/MAE_reg) |
|--------|-------------|---------------------|
| Lipophilicity | 0.622±0.008 | 0.957±0.012 |
| Solubility | 0.693±0.035 | **0.934±0.048** |
| VDss | 0.605±0.010 | **0.895±0.053** |
| Caco2 | 0.660±0.013 | 0.970±0.027 |
| Half Life | 0.602±0.014 | 0.971±0.005 |
| FreeSolv | 0.681±0.050 | **0.937±0.012** |

即使排序准确率仅 60%-69%，RankRefine 仍持续降低 MAE（β < 1）。

**人类排序器实验**（年龄估计，6 名参与者，15 名参考人物）：

| MAE_reg | 成对排序准确率 | β |
|---------|-------------|---|
| 6.343±0.610 | 0.759±0.052 | 0.954±0.046 |

人类成对判断（准确率 76%）可将年龄估计 MAE 降低约 5%，验证了人在环场景的实用性。

### 消融实验

**排序准确率和参考集大小的影响**（9 个 TDC 数据集，oracle 排序器）：

在大多数数据集上，排序准确率低至 0.55、仅 k=10 次比较时 RankRefine 就已有改善（β < 1）。k=20 通常是最优选择，进一步增加到 k=30 的改善微小。

**与 baseline 对比**（k=30）：

| 比较方法 | 排序准确率 0.5-0.95 | 排序准确率 >0.95 |
|---------|-------------------|-----------------|
| RankRefine vs Projection（Yan et al. 2024） | **RankRefine 胜** | Projection 胜 |
| RankRefine vs RbR（Gonçalves et al. 2023） | **RankRefine 全面胜** | **RankRefine 胜** |

RankRefine 在实际可行的 0.50-0.95 排序准确率范围内优于 projection baseline，仅在近乎完美排序时略逊。

**抗偏差实验**：
- 回归器偏差高达 60% SD 时，β 仍 < 1（改进有效）
- 参考集偏采样（覆盖范围仅 10%）时，RB=90% + 60% 排序准确率下 β = 0.884（仍有 11.6% 改善）
- 分布偏移（参考集与查询集不重叠）时，≥65% 准确率仍可改善

### 关键发现

1. **排序准确率门槛极低**：仅 55% 准确率就能带来改善，远低于直觉期望
2. **LLM 并非纯记忆**：在未公开的私有化合物-活性数据集上，ChatGPT-4o 仍达到 60.14% 成对排序准确率
3. **跨领域泛化**：从分子预测到表格数据（农业、教育、国际费用），RankRefine 均有效
4. **过于准确的排序器反而略有退化**：因 Fisher 信息的过度自信和边界值外推问题

## 亮点与洞察

1. **理论优雅性**：任何有限方差的排序器都能改善回归的保证（Corollary 3.2.1），为方法提供了坚实的理论基础
2. **极度实用**：无需重训练、无需改变模型架构、计算成本几乎为零（仅需 LLM API 调用）
3. **人机协作自我校正**：人类+RankRefine 可在不改变任何模型的情况下提升预测质量
4. **认知科学启发**：利用"人类（和 LLM）擅长相对比较"这一认知科学发现
5. **稀缺数据场景价值特别高**：仅 50 个训练样本 + 20 次 LLM 成对比较就能获得显著改善

## 局限与展望

1. 理论假设（高斯无偏独立误差）在实践中可能不成立，尤其面对重尾或偏斜噪声
2. 依赖回归器和排序器方差的良好校准，误校准可能抵消或削弱收益
3. Oracle 实验假设均匀随机排序错误，但真实排序器可能有系统性偏差（如在极端值上持续失败）
4. 使用 Bradley-Terry 模型处理真实属性值（vs 学习隐空间得分），引入了建模不匹配
5. 仅处理标量回归，向多变量/结构化目标（如完整药代动力学曲线）的扩展有待探索

## 相关工作与启发

- 与 RankUp（Huang et al. 2024，联合训练回归+排序）互补：RankRefine 是纯后处理，不改变训练
- 与 Pairwise Difference Regression（Tynes et al. 2021）的区别：后者预测配对差异需要训练，RankRefine 无需
- 启发方向：利用 LLM 的推理能力（rationale）提供可解释的排序依据，增强决策关键领域的信任度

## 评分

- 新颖性: ⭐⭐⭐⭐ 逆方差加权融合本身不新，但将成对排序注入后处理回归的框架设计巧妙且理论完备
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 9 个分子数据集 + 3 个表格数据集 + 人工实验 + LLM 实验 + 大量消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论-实验-分析逻辑链完整，公式推导清晰，实验设计周到
- 价值: ⭐⭐⭐⭐ 对数据稀缺领域极具实用价值，但适用场景可能集中在科学计算领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](self_iterative_label_refinement_via_robust_unlabeled_learning.md)
- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](../../ICML2026/computational_biology/neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)
- [\[ICML 2026\] Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback](../../ICML2026/computational_biology/influence-guided_symbolic_regression_scientific_discovery_via_llm-driven_equatio.md)
- [\[ICML 2026\] STRIDE: Post-Training LLMs to Reason and Refine Bio-Sequences via Edit Trajectories](../../ICML2026/computational_biology/stride_post-training_llms_to_reason_and_refine_bio-sequences_via_edit_trajectori.md)
- [\[ICLR 2026\] CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](../../ICLR2026/computational_biology/cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)

</div>

<!-- RELATED:END -->

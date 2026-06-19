---
title: >-
  [论文解读] Evaluating Multiple Models Using Labeled and Unlabeled Data
description: >-
  [NeurIPS 2025][计算生物][半监督学习] 提出 **SSME (Semi-Supervised Model Evaluation)**，利用少量标注数据和大量未标注数据，通过半监督混合模型估计多个分类器联合分布 $P(y, \mathbf{s})$，实现精确的分类器性能评估，误差降低至仅用标注数据的 1/5。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "半监督学习"
  - "model evaluation"
  - "mixture model"
  - "unlabeled data"
  - "classifier performance"
---

# Evaluating Multiple Models Using Labeled and Unlabeled Data

**会议**: NeurIPS 2025  
**arXiv**: [2501.11866](https://arxiv.org/abs/2501.11866)  
**代码**: 待确认  
**领域**: 计算生物  
**关键词**: semi-supervised evaluation, model evaluation, mixture model, unlabeled data, classifier performance

## 一句话总结

提出 **SSME (Semi-Supervised Model Evaluation)**，利用少量标注数据和大量未标注数据，通过半监督混合模型估计多个分类器联合分布 $P(y, \mathbf{s})$，实现精确的分类器性能评估，误差降低至仅用标注数据的 1/5。

## 研究背景与动机

机器学习评估的核心困境：**大规模标注数据是评估的基础，但在许多领域中获取标注数据成本极高甚至不可行**（如医疗、内容审核、分子性质预测等）。与此同时，未标注数据通常非常充裕。

现代 ML 实践使得问题更加突出：模型中心（如 HuggingFace）提供了海量现成分类器，实践者面对大量训练好的模型却缺少足够的标注数据来评估它们。

现有方法的局限：
- 仅利用标注数据：样本量小导致评估方差大
- 单分类器方法（如 SPE、Active-Testing）：没有利用多分类器之间的互信息
- Dawid-Skene 等标注者模型：仅使用离散标注，丢弃了概率预测信息
- AutoEval / Pseudo-Labeling：仅从标注数据学习映射再外推，忽略了联合学习的优势

SSME 是首个同时利用三个信息源的方法：**(i) 多个分类器、(ii) 连续概率分数、(iii) 未标注数据**。

## 方法详解

### 整体框架

SSME 分两步：

**Step 1: 估计联合分布 $P(y, \mathbf{s})$**

使用半监督混合模型，每个混合成分对应一个真实类别。联合利用标注数据和未标注数据最大化对数似然：

$$\underset{\theta}{\text{argmax}}\left[\sum_{i=1}^{n_\ell}\log[P_\theta(\mathbf{s}_i|y_i)P_\theta(y_i)] + \lambda_U\sum_{j=1}^{n_u}\log\sum_{k=1}^{K}[P_\theta(\mathbf{s}_j|y_j=k)P_\theta(y_j=k)]\right]$$

其中 $\lambda_U$ 控制未标注数据权重（固定为 1），$\mathbf{s}_i = [f_1(x_i), \ldots, f_M(x_i)]$ 为所有分类器的拼接分数。

**Step 2: 利用 $P(y, \mathbf{s})$ 估计分类器性能**

拟合密度后，通过 $P_\theta(y|\mathbf{s})$ 为未标注样本采样标签，然后计算任意标准指标（Accuracy、AUC、AUPRC、ECE 等）。

### 关键设计

**加性对数比变换 (Additive Log-Ratio Transform)**：分类器输出为概率单纯形上的值，存在边界偏差问题。SSME 通过 ALR 变换将其映射到无界实数空间 $\mathbb{R}^{K-1}$，使密度估计更准确。

**核密度估计 (KDE)**：使用高斯核参数化类条件分布 $P_\theta(\mathbf{s}|y)$，带宽通过改进的 Sheather-Jones 算法估计。KDE 不做参数化假设，适合分类器预测分布多样性。

**EM 优化**：1000 轮迭代。E 步计算每个数据点属于各成分的后验 $\gamma_{ik}$（标注数据直接固定为真实标签），M 步更新类先验和密度参数。

### 损失函数 / 训练策略

理论分析在二元高斯混合模型设定下推导误差界：

$$|\text{AUC}_k - \widehat{\text{AUC}}_k| \leq \Phi\left(\frac{\mathbf{c}_k}{\sqrt{2}}\right) - \Phi\left(\frac{\mathbf{c}_k - \epsilon_\mathbf{c}}{\sqrt{2}}\right)$$

其中 $\epsilon_\mathbf{c} \lesssim \frac{1}{p}\left(\sqrt{\frac{d}{\|c\|^2 n_u}} + \|c\|e^{-\frac{1}{2}n_l\|c\|^2(\cdot)}\right)$

关键含义：
- $n_u$ 增加 → 误差下降（未标注数据有用）
- $\|c\|$ 增大 → 误差下降（分类器越准估计越好）
- $d$ 增加（更多分类器）→ 若分离度增长快于维度代价，误差下降

## 实验关键数据

### 主实验

**实验设置**：5 个二分类数据集（MIMIC-IV 三个任务、CivilComments、OGB-SARS-CoV、MultiNLI、AG News），20/50/100 标注样本 + 1000 未标注样本，与 8 个基线对比。

**核心结果（$n_\ell=20, n_u=1000$）**：

| 方法 | 相对标注数据的误差降低倍数 |
|------|---------------------------|
| SSME (本文) | **5.1×** |
| 次优基线 | 2.4× |
| Labeled only | 1.0× (基线) |

**分指标表现**：
- Accuracy 估计：SSME 误差降低 5.6×，次优方法 2.0×
- ECE 估计：SSME 误差降低 7.2×（最大优势）
- AUC 估计：SSME 误差降低 2.9×，次优 2.6×
- AUPRC 估计：SSME 误差降低 2.2×（最小优势）

**绝对误差**：20 标注 + 1000 未标注，SSME 估计 accuracy 误差仅 1.5 个百分点（次优 3.4 百分点）。

### 消融实验

**SSME-M（边际拟合）**：仅对单个分类器拟合边际 $P(y|s)$，效果明显不如联合拟合。证实了多分类器提供互补信息。

**有效样本量 (ESS)**：20 标注 + 1000 未标注下，SSME 在 ECE 估计上等价于 **539** 个标注样本；次优方法仅等价于 110 个。

**标注数据量影响**：20 → 50 → 100 标注样本时，SSME 相对优势从 5.6× 降至 3.0× 降至 1.6×，但始终保持领先。

### 关键发现

1. SSME 在 60 个（数据集 × 指标 × 标注量）组合中的 51 个上达到最优或持平
2. ECE 估计场景受益最大——因为 ECE 需要分桶统计，少量标注下方差极大
3. 联合拟合多分类器远优于逐个拟合（SSME vs SSME-M）
4. 理论预测的三个趋势在真实实验中全部得到验证
5. Case Study：LLM 分类器评估误差降低 2.3×；子群体评估中性别维度误差降低 5.3×

## 亮点与洞察

- **统一框架**：首次将多分类器、连续概率分数、未标注数据三个信息源统一利用
- **理论与实践匹配**：误差界推导清晰，并在实验中得到验证
- **强实用性**：在仅 20 个标注样本下就能获得接近 500+ 标注样本的评估精度
- **广泛适用性**：跨医疗、NLP、化学、内容审核等多领域验证
- **公平性应用**：可直接用于子群体性能评估，对算法公平性审查有价值

## 局限与展望

- KDE 在高维空间（分类器数量多 / 类别数多）可能效果下降
- 假设标注和未标注数据同分布——分布偏移场景下未验证
- $\lambda_U = 1$ 的选择是否最优未做深入探索
- 仅关注分类任务，未扩展到回归或生成任务的评估
- 核密度估计的计算效率在超大规模场景下可能成为瓶颈

## 相关工作与启发

- 与 Dawid-Skene 类方法的核心区别：SSME 利用连续概率分数而非离散标注，信息量更大
- 与 Prediction-Powered Inference (Angelopoulos et al., 2023) 互补：后者关注置信区间
- 对 LLM-as-evaluator 的启示：在缺乏大量标注时，SSME 比"让 LLM 评 LLM"更可靠且更通用
- 未来可与 active learning 结合：SSME 识别出高不确定性样本后优先标注

## 评分

- **创新性**：⭐⭐⭐⭐ — 巧妙整合三个信息源，解决实际问题
- **理论深度**：⭐⭐⭐⭐⭐ — 完整的误差界推导和 UL+ 分析
- **实验充分度**：⭐⭐⭐⭐⭐ — 5 个数据集、8 个基线、多指标多场景
- **实用性**：⭐⭐⭐⭐⭐ — 标注稀缺场景价值极大
- **综合评价**：8.5/10

## 与相关工作的对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](self_iterative_label_refinement_via_robust_unlabeled_learning.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] Compressing Biology: Evaluating the Stable Diffusion VAE for Phenotypic Drug Discovery](compressing_biology_evaluating_the_stable_diffusion_vae_for_phenotypic_drug_disc.md)
- [\[NeurIPS 2025\] Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)

</div>

<!-- RELATED:END -->

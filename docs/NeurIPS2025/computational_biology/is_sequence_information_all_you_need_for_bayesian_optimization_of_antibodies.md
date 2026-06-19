---
title: >-
  [论文解读] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][计算生物][贝叶斯优化] 本文系统比较了序列信息和结构信息在抗体贝叶斯优化中的作用，发现通过蛋白质语言模型（pLM）软约束，纯序列方法可以匹配结构方法的性能，质疑了结构信息在抗体贝叶斯优化中的必要性。 领域现状： 治疗性抗体是一类重要的药物…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "计算生物"
  - "贝叶斯优化"
  - "抗体设计"
  - "蛋白质语言模型"
  - "结构信息"
  - "高斯过程"
---

# Is Sequence Information All You Need for Bayesian Optimization of Antibodies?

**会议**: NeurIPS 2025 (AI for Science Workshop)  
**arXiv**: [2509.24933](https://arxiv.org/abs/2509.24933)  
**代码**: 无  
**领域**: 生物信息学 / 抗体工程  
**关键词**: 贝叶斯优化, 抗体设计, 蛋白质语言模型, 结构信息, 高斯过程

## 一句话总结

本文系统比较了序列信息和结构信息在抗体贝叶斯优化中的作用，发现通过蛋白质语言模型（pLM）软约束，纯序列方法可以匹配结构方法的性能，质疑了结构信息在抗体贝叶斯优化中的必要性。

## 研究背景与动机

**领域现状**: 治疗性抗体是一类重要的药物，开发过程需要多轮迭代优化（结合亲和力、热稳定性等）。贝叶斯优化（BO）因其不确定性引导的特性，非常适合这类高成本、小数据场景。

**现有痛点**: BO的成功依赖代理模型和采集函数的选择，但现有工作尚未系统探索结构信息在抗体BO中的作用。结构扩散模型虽在非迭代设计中有用，但生成的抗体仍需进一步优化。

**核心矛盾**: 结构信息对哪些抗体性质有帮助？是抗体内在属性（如稳定性）还是抗体-抗原特异性属性（如结合亲和力）？尤其在结合位姿未知的常见场景下。

**本文目标**: 系统评估不同方式引入结构信息的效果，并与序列方法对比。

**切入角度**: 提出基于pLM的"软约束"机制，引导优化至有希望的序列空间区域。

**核心idea**: 适当引入序列先验信息（pLM软约束）后，纯序列方法可匹配甚至消除结构方法的优势。

## 方法详解

### 整体框架

基于Pareto感知的批量贝叶斯优化框架，使用qHSRI采集函数配合NSGA-II遗传算法进行离散空间优化。每轮获取约80个候选分子，使用GP代理模型建模。

### 关键设计

1. **序列方法**:

    - **OneHot-T**: 使用one-hot编码 + Tanimoto核的GP，作为基线
    - **BLO-T**: 使用BLOSUM-62矩阵编码 + Tanimoto核，在亲和力优化中表现最优
    - **ESM-M**: 使用ESM-2 650M模型的mean-pooled嵌入 + Matérn-5/2核

2. **结构方法**:

    - **IgFold-M**: 使用IgFold预测结构的α碳坐标作为输入，捕获纯3D几何信息
    - **IgFold-ESM-M/IgFold-BLO-T**: 将结构特征与序列特征拼接或核函数加权求和
    - **Kermut-T**: 结合ProteinMPNN结构信息和序列核的复合模型，核函数为：
    $k(\mathbf{x}, \mathbf{x}') = \pi k_{\text{struct}}(\mathbf{x}, \mathbf{x}') + (1-\pi) k_{\text{seq}}(\mathbf{x}, \mathbf{x}')$

3. **抗体特异性改进（AbMPNN-Kermut-T）**: 将通用ProteinMPNN替换为抗体特异性的AbMPNN，在亲和力优化中提升了早期迭代性能。

4. **pLM软约束（核心贡献）**: 受约束BO启发，用pLM伪似然作为采集函数的软约束：
    $a_{\text{pLM}}(\mathbf{x}) = p_{\text{pLM}}(\mathbf{x}) \cdot a(\mathbf{x})$
   其中 $p_{\text{pLM}}(\cdot)$ 来自Sapiens pLM（轻量级抗体特异性语言模型）。这避免了BO探索"不自然"的突变导致表达失败。

### 训练策略

- 从真实优化实验的早期阶段取50个样本作为初始数据集
- 每轮获取80个分子，随机丢弃30个（模拟实验失败）
- 共执行9轮采集，每个实验重复3次

## 实验关键数据

### 主实验

实验基于内部oracle（训练于真实优化campaign数据），优化解离常数 $K_D$（亲和力）和熔点温度 $T_m$（热稳定性）。

| 方法 | 亲和力（$K_D$）表现 | 热稳定性（$T_m$）表现 |
|------|---------------------|----------------------|
| OneHot-T | 优秀 | 中等，但最终追平 |
| BLO-T | **最优**（序列方法中） | 与OneHot-T相当 |
| ESM-M | 不如Tanimoto核方法 | 早期强但最终持平 |
| IgFold-M | 早期迭代表现好 | 中等 |
| Kermut-T | 亲和力最差 | **最优**（结构方法中） |
| AbMPNN-Kermut-T | 比Kermut-T提升较大 | 与Kermut-T相当 |

### 软约束效果

| 方法 | 亲和力变化 | 热稳定性变化 |
|------|-----------|------------|
| C-OneHot-T vs OneHot-T | 变化不大 | **match结构方法** |
| C-BLO-T vs BLO-T | 偶尔有负面影响 | 有改善 |
| C-Kermut-T vs Kermut-T | 无显著变化 | 无显著变化 |

### 消融实验

| 消融项 | 亲和力 | 热稳定性 |
|--------|--------|---------|
| Kermut原始 → Kermut-M (数值精度改进) | 提升 | 提升 |
| Kermut-M → Kermut-T (Tanimoto核替代ESM核) | 持平或微升 | 持平 |
| ProteinMPNN → AbMPNN | 亲和力提升 | 无变化 |
| ESM-2先验均值 → 常数均值 | 无变化 | 下降 |

### 关键发现

1. **亲和力优化**: 纯序列方法（BLO-T）达到最优渐近性能，结构信息仅在早期迭代提供数据效率优势
2. **热稳定性优化**: Kermut-T（结构方法）最优，但加入pLM软约束后，C-OneHot-T（纯序列）可匹配其性能
3. **没有单一方法在两个属性上同时最优**，表明不同性质需要不同特征
4. IgFold-M的早期优势源于其保持接近亲本结构的倾向（RMSD分析）

## 亮点与洞察

- **pLM软约束**是一个简洁优雅的设计，仅需一行公式修改即可引导BO远离不可行区域
- 深入分析了"纯结构"信息（IgFold坐标）与"统计结构"信息（ProteinMPNN概率）的本质区别
- 结论具有实用价值：在结合位姿未知的常见场景下，序列方法配合pLM约束足够使用
- BLOSUM编码出人意料地强于ESM-2嵌入，说明领域特异性简单特征有时优于通用大模型

## 局限与展望

1. 未考虑抗体-抗原复合物结构（结合位姿），这可能改善结构方法在亲和力优化上的表现
2. 结构信息的融合方式较简单，更复杂的融合策略值得探索
3. 实验基于in silico oracle，尚未进行体外验证
4. 仅评估了亲和力和热稳定性两个属性，其他可开发性属性待验证

## 相关工作与启发

- **LaMBO/LaMBO-2**：基于VAE潜空间的序列BO方法，本文发现其在小初始数据集下表现不佳
- **Kermut模型**：将ProteinMPNN结构信息与序列核结合的GP方法，本文对其进行了多项改进
- **GAUCHE**：蛋白质/化学分子BO工具库，提供了Tanimoto核等基础方法
- 启发：在蛋白质优化中，简单但领域适配的方法（BLOSUM编码）可能比复杂的通用方法更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ pLM软约束idea简洁有效，系统性比较有价值
- 实验充分度: ⭐⭐⭐⭐ 对比方法全面，但仅用in silico oracle
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，逻辑严密，研究问题明确
- 价值: ⭐⭐⭐⭐ 对抗体工程实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](../../ICLR2026/computational_biology/extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICML 2025\] Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks](../../ICML2025/computational_biology/empower_structure-based_molecule_optimization_with_gradient_guided_bayesian_flow.md)
- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] TabStruct: Measuring Structural Fidelity of Tabular Data
description: >-
  [ICLR 2026 Oral][表格数据生成] 提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。 领域现状：表格数据生成是增强训练、缺失填补等…
tags:
  - "ICLR 2026 Oral"
  - "表格数据生成"
  - "结构保真度"
  - "因果结构"
  - "全局效用"
  - "条件独立性"
---

# TabStruct: Measuring Structural Fidelity of Tabular Data

**会议**: ICLR 2026 Oral  
**arXiv**: [2509.11950](https://arxiv.org/abs/2509.11950)  
**代码**: [https://github.com/SilenceX12138/TabStruct](https://github.com/SilenceX12138/TabStruct)  
**领域**: 数据生成 / 表格数据 / 因果结构  
**关键词**: 表格数据生成, 结构保真度, 因果结构, 全局效用, 条件独立性

## 一句话总结

提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。

## 研究背景与动机

**领域现状**：表格数据生成是增强训练、缺失填补等任务的基础。现有评估主要关注三个维度：密度估计（分布是否接近）、ML 效用（下游预测性能）、隐私保护（合成数据与真实数据距离）。

**现有痛点**：这三个维度都来自同质模态（文本/图像），没有针对表格数据的异质特性做专门评估。一个典型问题是 SMOTE 在密度估计和 ML 效用上可以很好，但生成的数据严重违反了变量间的因果关系——比如物理定律所蕴含的条件独立性被破坏了。

**核心矛盾**：表格数据的核心先验是结构因果模型（SCM），变量间存在因果依赖关系。传统指标只评估边缘分布或某个 target 的预测，无法捕捉特征间的全局因果交互。且现有唯一考虑结构保真度的 CauTabBench 仅限于玩具 SCM 数据集，因为量化结构保真度需要真实因果图，而真实世界数据集几乎不可能获取。

**本文目标**
   - 如何在没有真实因果图的情况下评估结构保真度？
   - 现有生成器在结构保真度上表现如何？
   - 结构保真度与传统评估维度是什么关系？

**切入角度**：作者观察到如果一个生成器真正学到了数据的因果结构，那么用合成数据训练的模型应该能以每个变量为目标、用其他变量来预测，且性能都与真实数据接近。这个"全变量可预测"的性质与 SCM 中的 Markov blanket 概念紧密相关。

**核心 idea**：将每个变量轮流作为预测目标，聚合所有变量的预测性能比值作为 global utility，无需因果图即可评估表格数据的全局结构保真度。

## 方法详解

### 整体框架

TabStruct 把表格生成器的评估统一到一个框架里：给定参考数据集 $\mathcal{D}_{\text{ref}}$ 和某个生成器产出的合成数据集 $\mathcal{D}_{\text{syn}}$，它沿四个维度打分——密度估计（Shape/Trend）、隐私保护（$\alpha$-precision/$\beta$-recall/DCR/$\delta$-Presence）、ML 效用（local utility），以及本文新增的核心维度结构保真度。前三个维度沿用已有指标，整个框架的新意全在结构保真度这一块。

结构保真度怎么算，取决于有没有真实因果图，框架因此走两条路。在能拿到真实 SCM 的数据集上，直接用条件独立性（CI）测试逐条核对合成数据有没有保住该有的因果关系；在拿不到因果图的真实数据集上，没法直接核对，就用一个叫 global utility 的代理指标来间接刻画全局结构保持程度。下面三个设计点分别支撑这两条路，以及说明为什么传统 ML 效用不够用。

### 关键设计

**1. 条件独立性（CI）评分：有因果图时直接量化结构保真度**

当数据集附带专家验证的真实 SCM 时，可以把"结构有没有保住"翻译成一组可检验的命题。具体做法是从真实 SCM 的 CPDAG 里枚举出所有 CI 语句 $\mathcal{C}_{\text{global}}$，既包括 d-separation（该独立的变量对）也包括 d-connection（该依赖的变量对），然后在合成数据上逐条做统计检验（显著性 $\alpha=0.01$），统计通过率 $CI(\mathcal{C}, \mathcal{D}) = \frac{1}{|\mathcal{C}|}\sum \mathbb{1}[\hat{\mathcal{I}}_\alpha = 1]$。

之所以在 CPDAG 层面而非完整 DAG 层面评估，是因为现有因果发现方法在特征数 >10 时就不可靠，逼到完整 DAG 反而引入噪声；而退到骨架层面又会丢掉边的方向信息，所以 CPDAG 是精度和可行性的折中点。这套评分还区分局部和全局：local CI 只看与预测目标 $y$ 相关的那些 CI 语句，global CI 则覆盖所有变量对——后者才是这篇真正关心的全局结构。

**2. Global Utility：没有因果图时的结构保真度代理指标**

真实世界数据集几乎拿不到因果图，CI 评分就用不上了，于是需要一个不依赖 SCM 的间接指标。作者的切入点是：如果生成器真学到了因果结构，那么把每个变量 $x_j$ 轮流当成预测目标、用其余变量去预测它，合成数据训练出的模型性能应该都贴近真实数据。把单变量效用定义为这个预测性能与参考数据的比值（分类用 balanced accuracy，回归用 RMSE 取倒数做归一），再对所有变量求平均，就得到全局效用 $\text{Global Utility}(\mathcal{D}) = \frac{1}{D+1}\sum_{j=1}^{D+1}\text{Utility}_j(\mathcal{D})$。

这个设计同时解掉两件事：轮流预测每个变量避免了 local utility（只预测 $y$）对单一目标的偏差，归一化后的比值再聚合则让不同类型的任务能放在一起比较。为了不让结果被某个预测模型的脾气带偏，每个预测子任务用 AutoGluon 集成 9 种预测器。它的理论依据是：高保真生成器应当让每个变量的条件分布 $p(x_j | \mathcal{X} \setminus \{x_j\})$ 与真实数据一致，而这正对应 SCM 里的 Markov blanket 概念——把所有变量的条件可预测性都对齐，等价于把整张因果结构对齐。

**3. Local Utility vs. Global Utility：用对照锚定 global utility 才是对的代理**

光提出 global utility 还不够，得说清它凭什么比传统指标更能反映结构。作者把传统 ML efficacy 重新命名为 local utility——它只把预测目标 $y$ 当作回归对象，本质是 global utility 在"只预测一个变量"时的特例。两者一对照，结论很硬：local utility 与 local CI 强相关（$r_s=0.78$），却与 global CI 几乎无关（$r_s=0.14$）；而 global utility 与 global CI 强相关（$r_s=0.84$）。这说明 ML efficacy 只反映目标变量附近的局部结构，撑不起对生成器全局因果保真度的判断，必须聚合到全变量的 global utility 才能对齐 global CI。

## 实验关键数据

### 主实验——13 种生成器在 SCM 数据集上的结构保真度

| 生成器 | Global CI ↑ | Global Utility ↑ | Local Utility ↑ | Shape ↑ |
|--------|------------|-------------------|-----------------|---------|
| $\mathcal{D}_\text{ref}$ | 1.00 | 0.99 | 0.99 | 1.00 |
| TabSyn | **0.70** | 0.76 | 0.76 | 0.50 |
| TabDDPM | **0.69** | **0.80** | 0.29 | 0.62 |
| TabDiff | 0.57 | 0.75 | 0.80 | 0.69 |
| SMOTE | 0.30 | 0.39 | **0.92** | 0.82 |
| CTGAN | 0.08 | 0.26 | 0.80 | 0.46 |
| GReaT | 0.16 | 0.25 | 0.27 | 0.62 |
| NRGBoost | 0.11 | 0.16 | 0.75 | 0.65 |

关键发现：SMOTE 的 local utility 最高（0.92），但 global CI 仅 0.30——说明传统 ML 效用会误导评估。扩散模型（TabDDPM/TabSyn/TabDiff）在全局结构保真度上一致最优。

### 真实数据集上 Global Utility 排名

| 生成器 | Global Utility ↑ | Local Utility ↑ |
|--------|-------------------|-----------------|
| $\mathcal{D}_\text{ref}$ | 0.99 | 0.96 |
| TabSyn | **0.73** | 0.76 |
| TabDiff | **0.73** | 0.78 |
| TabDDPM | **0.72** | 0.27 |
| ARF | 0.56 | 0.54 |
| TVAE | 0.53 | 0.70 |
| BN | 0.44 | 0.38 |
| SMOTE | 0.41 | **0.91** |
| CTGAN | 0.13 | 0.70 |
| GReaT | 0.20 | 0.23 |

在真实数据集上，Global Utility 的 Top-3 仍然是扩散模型（TabSyn/TabDiff/TabDDPM），与 SCM 数据集一致，验证了 global utility 的泛化能力。

### 相关性分析

| 指标对 | Spearman $r_s$ | p 值 |
|--------|----------------|------|
| Global Utility ↔ Global CI | **0.84** | <0.001 |
| Local Utility ↔ Local CI | 0.78 | <0.001 |
| Local Utility ↔ Global CI | 0.14 | <0.001 |

Global utility 与 global CI 的强相关性（0.84）是核心实证结果，证明其作为无 SCM 代理指标的有效性。

## 亮点与创新

- **填补评估空白**：首次系统性地将结构保真度纳入表格生成器评估框架，且提出的 global utility 不需要真实因果图
- **规模化基准**：13 种生成器 × 29 个数据集 × 超 15 万次评估，远超现有基准的覆盖范围
- **揭示扩散模型优势的本质**：扩散模型因噪声独立添加到每个特征、去噪时同时重建所有特征，天然学习置换不变的条件分布，与表格数据结构先验对齐
- **打破 SMOTE 的"幻觉"**：实验清楚展示 SMOTE 在传统指标上表现好但严重违反因果结构，揭示了评估偏差

## 局限性

- Global utility 与 global CI 的强相关是经验发现，缺乏理论证明
- 评估在 CPDAG 层面进行，比完整 DAG 层面粗糙，可能遗漏某些方向性因果关系
- 依赖 AutoGluon 集成预测器，计算开销随特征数增长（虽然 Tiny-default 变体效率较好，0.64s/1000 样本）
- SCM 数据集使用专家验证的因果图，但这类数据集数量有限（仅 6 个），泛化性有待扩展

## 相关工作

- **表格生成基准**：Synthcity（Qian et al., 2024）、SynMeter（Du & Li, 2024）覆盖密度/隐私/ML 效用，但无结构保真度；CauTabBench（Tu et al., 2024）评估结构保真度但仅限玩具 SCM
- **表格生成器**：从 SMOTE、BN 到 TVAE、CTGAN、NFlow、ARF，再到扩散模型 TabDDPM/TabSyn/TabDiff/TabEBM，以及 LLM-based 的 GReaT 和树模型 NRGBoost
- **因果发现**：DAG 学习方法在特征数 >10 时困难重重（Zanga et al., 2022），本文因此选择 CPDAG 层面评估
- **表格基础模型**：Hollmann et al. (2025) 经验证明 SCM 是表格数据的有效结构先验

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次提出无需因果图的全局结构保真度指标 |
| 技术深度 | ⭐⭐⭐⭐ | 理论分析清晰，CI 框架 + global utility 设计合理 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 13 模型 × 29 数据集，15 万次评估，极其充分 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，动机示例直观 |
| 实用价值 | ⭐⭐⭐⭐ | 开源框架，可直接用于评估新生成器 |
| 总评 | ⭐⭐⭐⭐ | 表格数据生成评估的重要贡献，global utility 有望成为标准指标 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](../../ACL2025/others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](../../ICML2026/others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)

</div>

<!-- RELATED:END -->

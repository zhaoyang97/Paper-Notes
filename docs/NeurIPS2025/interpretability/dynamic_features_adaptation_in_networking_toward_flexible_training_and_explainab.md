---
title: >-
  [论文解读] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference
description: >-
  [NeurIPS 2025][可解释性][Adaptive Random Forest] 提出 DAFI（Drift-Aware Feature Importance）算法，利用分布漂移检测动态切换 SHAP/MDI 两种特征重要性方法，结合自适应随机森林（ARF）实现通信网络场景下特征动态增加时的灵活训练与高效可解释推理。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "Adaptive Random Forest"
  - "Feature Importance"
  - "Explainable AI"
  - "Data Stream"
  - "6G Networks"
  - "Drift Detection"
  - "SHAP"
  - "MDI"
---

# Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference

**会议**: NeurIPS 2025  
**arXiv**: [2510.08303](https://arxiv.org/abs/2510.08303)  
**代码**: 未公开  
**领域**: 可解释性  
**关键词**: Adaptive Random Forest, Feature Importance, Explainable AI, Data Stream, 6G Networks, Drift Detection, SHAP, MDI

## 一句话总结

提出 DAFI（Drift-Aware Feature Importance）算法，利用分布漂移检测动态切换 SHAP/MDI 两种特征重要性方法，结合自适应随机森林（ARF）实现通信网络场景下特征动态增加时的灵活训练与高效可解释推理。

## 研究背景与动机

**6G 网络 AI 原生化需求**：未来通信网络被设想为 AI 原生架构，AI 需嵌入基站和网络节点进行移动模式识别与 KPI 优化，但不同厂商、硬件代际导致各节点暴露的特征集不一致。

**特征异构性挑战**：多厂商部署下，不同基站可能提供不同的 KPI 测量指标，难以训练统一的全局模型，且硬件升级会引入新特征，模型需要动态适应特征空间的变化。

**数据分布漂移问题**：流量模式、用户行为和无线信道条件随时间演变，静态模型无法应对持续漂移，需要增量式/流式学习方案。

**可解释性的关键地位**：AI 控制关键基础设施时，运营商信任、监管合规要求模型具备透明的决策解释能力，Feature Importance（FI）是主要手段。

**SHAP 计算开销过高**：SHAP 是最广泛使用的 FI 方法，但计算代价随特征数和模型规模指数增长，难以满足基站实时、短间隔重复计算的需求。

**MDI 与 SHAP 不一致**：MDI 作为树模型内置的轻量 FI 方法速度极快，但在数据漂移且模型尚未更新时，其排名与 SHAP 显著不一致，可靠性不足。

## 方法详解

### 整体框架

本文提出两个互补组件：(1) 使用自适应随机森林（ARF）进行迭代流式训练，适应特征动态增加的通信网络场景；(2) 提出 DAFI 算法，在 ARF 训练过程中根据分布漂移信号智能切换 SHAP 和 MDI，兼顾 FI 精度与计算效率。

### 关键设计 1：自适应随机森林（ARF）迭代训练

- **功能**：在数据流上增量训练 ARF，随训练进行渐进引入新特征（epoch 10 加 1 个特征，epoch 20 加 2 个特征）。
- **核心思路**：ARF 内置漂移检测器（基于 Hoeffding Tree），当检测到概念漂移时自动替换性能退化的子树，使模型在特征空间扩展时保持稳定性。采用 prequential 80:20 划分，按顺序在 40-50 个 batch 上迭代训练和评估。
- **设计动机**：通信网络中特征集随新硬件和配置变化而扩展，ARF 的自适应机制天然匹配这种动态环境，且相比重新训练全局模型成本更低。

### 关键设计 2：DAFI（Drift-Aware Feature Importance）

- **功能**：对每个样本，根据当前训练 batch 与上一 batch 之间的分布漂移状态，决定使用 SHAP（漂移时）还是 MDI（无漂移时）计算 FI。
- **核心思路**：使用 Kolmogorov-Smirnov (KS) 检验对每个特征检测前后 batch 间的分布差异。若 KS 统计量超过阈值 η，判定存在漂移，触发 SHAP 计算；否则使用快速的 MDI。阈值 η 需根据数据集 batch 大小调整（Network 数据集 η=0.125，其他 η=1）。
- **设计动机**：MDI 在模型已适应当前数据时与 SHAP 基本一致，仅在漂移发生、模型尚未调整时偏差较大。因此仅在漂移时"付出" SHAP 的高计算代价，正常时段使用 MDI 节省资源。

### 关键设计 3：动态 Top-k 评估指标

- **功能**：设计自适应的 Top-k 特征评估方法，通过累积 SHAP 权重至阈值 0.8 来确定 k 的值，而非固定 k。
- **核心思路**：在特征数量动态变化的流式场景中，固定 k 不合理。按 SHAP 权重降序累加，达到 80% 累积阈值时确定要比较的 Top-k 集合，再用 Set Match（集合匹配）、Exact Match（精确排序匹配）和 Spearman 秩相关评估 MDI/DAFI 与 SHAP 的一致性。
- **设计动机**：归一化的 FI 分数依赖当前特征数，动态 k 能更公平地反映不同方法在变化特征空间下的排名准确性。

## 损失函数与训练策略

- ARF 采用 river 库实现，基于 Hoeffding Tree 增量学习，无需显式损失函数优化，而是通过信息增益/Gini 指数进行节点分裂。
- 内置 ADWIN 漂移检测器，检测到漂移时重置受影响的子树。
- 训练为 prequential（先测后训）流式设置，ntrees=10，每次 FI 计算采样 nsamples=50 个测试样本。

## 实验

### 实验 1：ARF 在动态特征下的性能（Network 数据集）

| 阶段 | 特征数 | 测试准确率 |
|------|--------|-----------|
| Epoch 0-9 | 3 个特征 | ~0.64 |
| Epoch 10-19 | 4 个特征（+1） | ~0.64→逐步提升 |
| Epoch 20-40 | 6 个特征（+2） | ~0.72 |

新特征引入不会损害模型性能，反而持续提升准确率，从 0.64 提升至 0.72，证明 ARF 在动态特征空间下的稳定性。

### 实验 2：DAFI 运行时效率与 FI 准确性（Network 数据集）

| FI 方法 | 运行时间 (s) | 节省 (%) | Top-k Set | Top-k Exact | Spearman |
|---------|-------------|---------|-----------|-------------|----------|
| SHAP | 1488.32 | 0.00 | 1.00 | 1.00 | 1.00 |
| MDI | 9.37 | 99.37 | 0.27 | 0.09 | 0.53 |
| **DAFI** | **674.49** | **54.68** | **0.55** | **0.40** | **0.67** |

DAFI 在 Network 数据集上节省约 55% 运行时间，同时在所有 FI 准确性指标上全面超越 MDI。在 Electricity 数据集上 Top-k Exact 达 0.76、Spearman 达 0.85，在 Weather 上 Spearman 达 0.83，展示了跨域泛化能力。

## 亮点

- **问题切入精准**：准确识别了 6G 网络 AI 部署中特征异构、动态增长与实时可解释性的实际需求。
- **DAFI 思路简洁有效**：仅通过 KS 检验判断漂移来切换 SHAP/MDI，无需复杂架构即获得显著的效率-精度平衡。
- **运行时间大幅削减**：在保留较高 FI 一致性的前提下，将 SHAP 的计算开销降低约 50-65%。
- **实验验证完整**：在 3 个不同领域数据集上验证，且提供了 SHAP 随特征数/模型规模的 runtime 增长分析作为支撑。

## 局限性

- **Vision Paper 缺乏深度技术贡献**：论文自我定位为 vision paper，ARF 和 KS 检验均为现有方法的组合，方法新颖性有限。
- **阈值 η 需手动调优**：KS 检验的漂移阈值在不同数据集上差异很大（0.125 vs 1.0），缺乏自动化选择策略。
- **评估规模较小**：实验仅使用 ntrees=10、nsamples=50，在大规模实际部署场景下的效果未验证。
- **仅验证特征增加场景**：未涉及特征删除/替换等更复杂的动态特征变化场景。
- **分类为 object_detection 存疑**：本文核心为网络流量/KPI 分类，与目标检测领域关联不大。

## 相关工作

- **SHAP / TreeSHAP**：Lundberg & Lee (2017) 提出的模型无关 FI 方法，精确但计算开销大。
- **MDI (Mean Decrease in Impurity)**：Scornet (2021) 形式化的树模型内置 FI 方法，快但不稳定。
- **增量 XAI**：iPFI (Fumagalli et al., 2023) 和 Muschalik et al. (2022) 探索流式场景下的增量解释方法。
- **ARF**：Gomes et al. (2017) 提出的自适应随机森林，内置漂移检测的流式集成学习方法。
- **网络领域 XAI**：Brik et al. (2024) 综述了 6G O-RAN 中的可解释 AI 方法。

## 评分

- 新颖性: ⭐⭐⭐ — 方法为已有技术的组合，但问题定义和漂移驱动切换思路有价值
- 实验充分度: ⭐⭐⭐ — 3 个数据集验证充分，但规模偏小且缺乏消融实验
- 写作质量: ⭐⭐⭐⭐ — 定义清晰、结构合理、motivation 论述充分
- 价值: ⭐⭐⭐⭐ — 对 6G 网络 AI 场景的实际部署有实用参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)
- [\[ICML 2025\] Concept-Based Unsupervised Domain Adaptation](../../ICML2025/interpretability/concept-based_unsupervised_domain_adaptation.md)
- [\[NeurIPS 2025\] Dense SAE Latents Are Features, Not Bugs](dense_sae_latents_are_features_not_bugs.md)
- [\[ICML 2025\] Towards Flexible Perception with Visual Memory](../../ICML2025/interpretability/towards_flexible_perception_with_visual_memory.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Finding Time Series Anomalies using Granular-ball Vector Data Description
description: >-
  [AAAI 2026][时间序列] 提出 Granular-ball One-Class Network (GBOC)，通过在潜在空间中自适应构建密度引导的粒球向量数据描述 (GVDD)，取代传统聚类或单一超球体假设，实现对时间序列正常行为的灵活建模和鲁棒异常检测。 时间序列异常检测在工业监控、数据中心、智能工厂等复杂物理-…
tags:
  - "AAAI 2026"
  - "时间序列"
---

# Finding Time Series Anomalies using Granular-ball Vector Data Description

**会议**: AAAI 2026  
**arXiv**: [2511.12147](https://arxiv.org/abs/2511.12147)  
**作者**: Lifeng Shen, Liang Peng, Ruiwen Liu, Shuyin Xia, Yi Liu
**代码**: [https://github.com/notshine/GBOC](https://github.com/notshine/GBOC)

## 一句话总结

提出 Granular-ball One-Class Network (GBOC)，通过在潜在空间中自适应构建密度引导的粒球向量数据描述 (GVDD)，取代传统聚类或单一超球体假设，实现对时间序列正常行为的灵活建模和鲁棒异常检测。

## 研究背景与动机

时间序列异常检测在工业监控、数据中心、智能工厂等复杂物理-网络系统中至关重要。现有方法主要包括三类：

**最近邻方法**（如 KNN）：依赖局部密度或邻近性，在群体异常场景下容易将一组异常点误判为正常（异常点之间互为"邻居"），且无法捕捉全局/时序结构。

**聚类方法**（如 KShapeAD、KMeansAD）：需预定义簇数，假设正常数据形成离散、边界清晰的簇结构。但时间序列滑动窗口的表示通常具有**结构连续性**，呈现平滑过渡而非离散分界，刚性的簇划分不适合此类数据。

**单类分类方法**（如 DeepSVDD、SVDD）：将正常数据建模为单个超球体，过于简化的假设难以捕捉多模态正常模式的多样性。

**基于记忆的方法**（如 MEMTO）：依赖存储的原型质量和代表性，原型覆盖不全时效果显著下降。

**核心动机**：时间序列潜在表示具有连续拓扑结构，传统方法的刚性假设（预定义簇数、离散边界、单一超球体）不适合此类数据。需要一种自适应、无需预定义参数、能灵活建模复杂数据分布的方法。

## 方法详解

### 整体框架

GBOC 由四个步骤组成：(1) 时间序列编码 → (2) 粒球构建 → (3) 粒球表示优化 → (4) 异常推理。

### 1. 时间序列编码

使用滑动窗口将输入时间序列分割为重叠窗口 $\{\mathbf{w}_1, \mathbf{w}_2, \ldots\}$，每个窗口经三层 LSTM 编码器映射到 $d'$ 维潜在表示 $\mathbf{z}_i = f_\theta(\mathbf{w}_i)$。各层最终隐藏状态拼接形成多层级时序依赖建模。编码器可替换为 Transformer 等其他架构。

### 2. 粒球向量数据描述 (GVDD)

在潜在空间中执行粒球构建：

- **初始化**：用 $k_0 = \lfloor\sqrt{n}\rfloor$ 的 K-Means 将数据粗分为初始粒球
- **递归分裂**：对含 $\geq s_{\min}=8$ 个点的粒球，用 2-Means 分裂为两个子球
- **分裂判据**：基于粒球分布度量 $DM = s / |GB|$（$s$ 为所有点到中心距离之和），只有加权子球 $DM_w < DM$ 时才接受分裂
- 递归直到无法再进行质量改进的分裂

每个粒球由中心 $\mathbf{c}$（均值）和半径 $\mathbf{r}$（最大欧氏距离）定义，自然处于单个样本与全局簇之间的粒度，保留数据的局部拓扑结构。

### 3. 粒球表示优化

**低质量粒球剪枝**：设定动态阈值 $r_{th} = \mu \cdot \max\{\text{median}(r), \text{mean}(r)\}$（$\mu=2$），剔除半径过大（过于弥散或噪声区域）的粒球，保留结构紧凑、高置信度的粒球。

**联合优化目标**：

- **粒球对齐损失** $\mathcal{L}_{gb}$：将每个样本拉向其最近粒球中心，增强潜在空间的紧凑性和判别性：
$$\mathcal{L}_{gb} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{z}_i - \mathbf{c}_{s(i)}\|_2^2$$

- **重构损失** $\mathcal{L}_{rec}$：通过轻量 MLP 解码器保持时序保真度，防止表示坍缩：
$$\mathcal{L}_{rec} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i - g_\phi(\mathbf{z}_i)\|_2^2$$

- **总损失**：$\mathcal{L} = \lambda \cdot \mathcal{L}_{rec} + (1-\lambda) \cdot \mathcal{L}_{gb}$，$\lambda=0.5$

### 4. 异常推理

测试样本编码后，计算到最近粒球中心的欧氏距离作为异常分数：
$$\text{Score}(\mathbf{z}) = \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{z} - \mathbf{c}\|_2$$

采用经验 $3\sigma$ 规则确定无监督阈值：异常分数超过评估集均值 3 个标准差即标记为异常。

## 实验

### 实验设置

- **数据集**：覆盖 7 个单变量 + 5 个多变量数据集，跨越工业系统（SMD）、Web 服务（IOPS, WSD）、医疗（UCR, LTDB, SVDB）、环境（TAO, SMAP, MSL）、合成（YAHOO）等领域
- **基线**：14 个方法，包括非深度学习（PCA, KNN, IForest, MatrixProfile, KShapeAD）和深度学习（CNN, LSTMAD, TranAD, USAD, TimesNet, AnomalyTransformer, DeepSVDD, THOC, MEMTO）
- **指标**：VUS-PR、VUS-ROC、Affiliation-F1
- **硬件**：NVIDIA RTX 4090 GPU, 128GB RAM

### 主要结果

**表1：单变量异常检测 (VUS-PR)**

| 方法 | SMD | TAO | YAHOO | UCR | IOPS | WSD |
|------|-----|-----|-------|-----|------|-----|
| KNN | 0.766 | 0.940 | 0.281 | 0.856 | 0.222 | 0.011 |
| DeepSVDD | 0.812 | 0.945 | 0.967 | 0.996 | 0.236 | 0.404 |
| TimesNet | 0.680 | 0.932 | 0.577 | 0.023 | 0.184 | 0.354 |
| THOC | 0.272 | 0.938 | 0.048 | 0.513 | 0.407 | 0.025 |
| MEMTO | 0.314 | 0.932 | 0.074 | 0.630 | 0.180 | 0.021 |
| **GBOC** | **0.831** | **0.978** | **0.991** | **0.996** | **0.604** | **0.963** |

GBOC 在 6 个单变量数据集上均取得最优 VUS-PR，特别是在 YAHOO（0.991 vs 次优 0.967）、IOPS（0.604 vs 0.407）和 WSD（0.963 vs 0.404）上优势显著。

**表3：漂移与噪声鲁棒性 (VUS-PR)**

| 方法 | I: 干净 | II: 漂移 | III: 噪声 | IV: 漂移+噪声 |
|------|---------|---------|----------|--------------|
| KShapeAD | 1.000 | 0.982 | 0.802 | 0.624 |
| DeepSVDD | 0.824 | 0.153 | 0.833 | 0.893 |
| MEMTO | 0.782 | 0.028 | 0.121 | 0.031 |
| **GBOC** | **1.000** | **0.977** | **0.952** | **0.921** |

GBOC 在所有四种场景下均表现最佳。在最严峻的"漂移+噪声"场景（Type IV），GBOC（0.921）远超 KShapeAD（0.624）和 MEMTO（0.031），展现出极强的鲁棒性。

### 消融实验

**表4：粒球组件消融 (VUS-PR)**

| GBC | 剪枝 | SMD | IOPS | UCR | YAHOO |
|-----|------|-----|------|-----|-------|
| ✗ (K-Means) | ✗ | 0.755 | 0.554 | 0.921 | 0.823 |
| ✓ | ✗ | 0.781 | 0.566 | 0.972 | 0.795 |
| ✓ | ✓ | **0.831** | **0.604** | **0.996** | **0.991** |

- 移除粒球计算（用 K-Means 替代）导致性能大幅下降，说明自适应密度感知的粒球构建优于固定簇结构
- 移除剪枝保留噪声区域，也导致性能退化

**表5：损失函数消融**

仅用 $\mathcal{L}_{rec}$ 或仅用 $\mathcal{L}_{gb}$ 均不如两者联合。在噪声较大的 YAHOO 数据集上效果最为明显（联合 0.991 vs 单一 0.869/0.701）。

## 亮点与创新

- **粒球计算首次引入单类异常检测**：提出 GVDD，在单类方法中融入粒球计算，填补了粒球计算在时间序列异常检测中的空白
- **自适应无参数建模**：无需预定义簇数或邻居数，粒球自动根据数据密度分裂和剪枝，天然适配连续时序结构
- **噪声/漂移场景下鲁棒性极强**：通过聚焦高密度紧凑区域，有效过滤噪声和低质量区域，在四种不同复杂度场景下均保持高性能
- **推理高效**：粒球数量远少于训练样本数，异常评分仅需计算到最近球心的距离

## 局限性

- LSTM 编码器对极长时间序列的建模能力有限，虽论文提到可替换 Transformer 但未展开对比
- 粒球构建依赖 K-Means 初始化和 2-Means 递归分裂，对初始化敏感性未深入分析
- 剪枝阈值 $\mu=2$ 和最小支持 $s_{\min}=8$ 均为经验设定，不同数据规模下最优值可能不同
- 仅在无监督场景下评估，未探索半监督或少量已知异常的场景
- 推理时需遍历所有粒球中心求最小距离，粒球数量很大时可能影响实时性

## 相关工作

- **最近邻方法**：KNN (SubKNN)、LOF — 依赖局部密度，不适应群体异常
- **聚类方法**：KShapeAD、KMeansAD、SAND — 需预定义簇数，假设离散模式
- **单类分类**：SVDD、DeepSVDD — 单一超球体，难以捕捉多模态
- **记忆增强**：MEMTO — 固定记忆结构，原型覆盖不全时退化
- **层次化聚类**：THOC — 多尺度向量数据描述，但仍受固定结构限制
- **重构/预测方法**：USAD、LSTMAD、TranAD — 在噪声/非平稳环境中易失效
- **粒球计算 (GBC)**：已用于聚类加速、密度聚类、点云配准、意图分类等场景，本文首次将其引入时间序列异常检测

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](../../ICLR2026/time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[AAAI 2026\] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICML 2026\] Embedding Hybrid Systems into Continuous Latent Vector Fields](../../ICML2026/time_series/embedding_hybrid_systems_into_continuous_latent_vector_fields.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)

</div>

<!-- RELATED:END -->

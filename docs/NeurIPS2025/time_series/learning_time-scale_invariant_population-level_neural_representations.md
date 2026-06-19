---
title: >-
  [论文解读] Learning Time-Scale Invariant Population-Level Neural Representations
description: >-
  [NeurIPS 2025][时间序列][神经时间序列] 提出时间尺度增强预训练（TSAP）策略，通过在预训练阶段引入多种时间窗口长度的数据增强，使群体级神经信号基础模型对输入时间尺度具有不变性，在匹配和未见时间尺度上均显著提升解码性能。 领域现状：构建神经时间序列的通用表征是神经科学和脑机接口（BCI）研究的基础目标…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "神经时间序列"
  - "基础模型"
  - "时间尺度不变性"
  - "群体级表征"
  - "脑机接口"
---

# Learning Time-Scale Invariant Population-Level Neural Representations

**会议**: NeurIPS 2025  
**arXiv**: [2511.13022](https://arxiv.org/abs/2511.13022)  
**代码**: 无  
**领域**: 时间序列 / 神经信号基础模型  
**关键词**: 神经时间序列, 基础模型, 时间尺度不变性, 群体级表征, 脑机接口

## 一句话总结

提出时间尺度增强预训练（TSAP）策略，通过在预训练阶段引入多种时间窗口长度的数据增强，使群体级神经信号基础模型对输入时间尺度具有不变性，在匹配和未见时间尺度上均显著提升解码性能。

## 研究背景与动机

**领域现状**：构建神经时间序列的通用表征是神经科学和脑机接口（BCI）研究的基础目标。颅内脑电图（iEEG）等高保真神经记录捕获多脑区的复杂活动模式，但因受试者间和session间变异性及数据集规模有限等挑战，建模难度极大。

**现有痛点**：近期群体级预训练方法（如 Population Transformer, PopT）在冻结的时间编码器之上学习空间聚合表征，取得了良好的下游解码性能，但这些模型对预处理参数——尤其是时间尺度——非常敏感。当预训练和下游任务使用不同长度的时间窗口时，性能会显著下降。

**核心矛盾**：神经记录在不同数据集和任务间的长度差异很大（1秒到5秒不等），但现有模型仅在固定时间窗口上预训练，无法泛化到不同长度的输入。

**本文目标**：量化时间尺度不匹配导致的性能下降，并提出一种策略让模型对任意输入时间尺度均能达到最优性能。

**切入角度**：从数据增强角度出发，在预训练阶段暴露模型于多种时间窗口长度的数据。

**核心 idea**：通过在预训练中混合多种时间尺度的iEEG片段（TSAP），让PopT学到时间尺度不变的群体级表征。

## 方法详解

### 整体框架

基于 Population Transformer（PopT）框架：对每个电极通道的给定时间区间，先通过冻结的时间编码器（BrainBERT）获取时间嵌入，再叠加由3D电极坐标生成的位置嵌入，最后通过 Transformer 编码器得到空间上下文化的通道表征和聚合的 [CLS] 输出，用于下游解码。

### 关键设计

1. **Time-scale Augmented Pretraining（TSAP）**:

    - **功能**：修改数据生成流程，使模型在预训练时接触多种时间窗口长度的iEEG信号
    - **为什么**：消除模型对特定时间尺度的过拟合，建立时间尺度不变性
    - **怎么做**：采样长度 $l \in \{1, 2, 4, 5\}$ 秒的记录片段（3秒作为held-out），对每个通道独立编码为BrainBERT嵌入。不同窗口长度的嵌入包含重叠窗口，时间编码器会将其映射为不同的表征
    - **区别**：原始PopT仅在固定5秒窗口上预训练，TSAP通过多尺度暴露鼓励模型跨时间尺度泛化

2. **嵌入空间分析（PCA + K-Means）**:

    - **功能**：可视化分析不同时间尺度下的时间嵌入和[CLS] token表征分布
    - **为什么**：验证TSAP是否确实消除了时间尺度相关的聚类
    - **怎么做**：对特定受试者-session取100个样本，跨1-5秒时间尺度，进行2D PCA投影和K-Means聚类分析
    - **发现**：5秒预训练的PopT产生强烈的时间尺度聚类，而TSAP模型的聚类明显混合，表明更强的时间尺度不变性

### 训练策略

- 预训练步数从500,000翻倍到1,000,000以处理更大的增强数据集
- 学习率固定为 $1 \times 10^{-4}$ 以提高训练稳定性
- 基于验证损失选择最佳检查点
- 下游微调时对每个受试者随机选择90个电极，每个实验重复5个种子

## 实验关键数据

### 主实验

使用公开的 BrainTreeBank 数据集（10名受试者，1688个电极），评估两个听觉-语言分类任务：Word Onset 和 Sentence Onset。

| 模型 | 1s | 2s | 3s (held-out) | 4s | 5s |
|------|-----|-----|------|-----|-----|
| Non-Pretrained | 0.645 | 0.665 | 0.663 | 0.671 | 0.678 |
| 1s Pretrained | **0.770** | 0.807 | 0.809 | 0.817 | 0.819 |
| 5s Pretrained | 0.717 | 0.801 | 0.846 | 0.879 | **0.901** |
| **TSAP** | **0.777** | **0.843** | **0.866** | **0.893** | **0.907** |

*Word Onset ROC-AUC（均值±标准误，跨受试者和5个种子）*

TSAP 在所有时间尺度上匹配或超越"最优"基线（即预训练与微调使用相同时间尺度的模型），包括 held-out 的3秒时间尺度。

### 消融实验

| 对比 | 统计显著性 (p值) |
|------|-----------------|
| TSAP vs 1s Optimal (1s) | p=0.017* |
| TSAP vs 4s Optimal (4s) | p=0.00005* |
| TSAP vs 5s Optimal (5s) | p=0.004* |
| TSAP vs 3s Optimal (3s, held-out) | p=0.442 |

配对 t-test 显示 TSAP 在大多数时间尺度上显著优于最优基线，held-out 3秒虽未显著但也偶有提升。

### 关键发现

- 时间尺度不匹配会导致显著的性能下降：例如用1秒预训练的模型在5秒输入上的性能远低于5秒预训练模型
- 即使不匹配，任何预训练模型都优于非预训练模型，说明预训练仍学到了有价值的信息
- TSAP 不仅恢复了不匹配的性能损失，还在多数情况下超越了"最优"匹配基线
- PCA 分析证实 TSAP 显著减少了嵌入空间中的时间尺度聚类

## 亮点与洞察

- **简洁有效**：TSAP 是一种纯数据增强策略，不需要修改模型架构，仅在预训练阶段混合多种时间尺度即可
- **物理直觉清晰**：不同时间窗口虽然包含重叠信息，但经过时间编码器后会产生非常不同的嵌入，这是性能下降的根本原因
- **在 held-out 时间尺度上泛化**：3秒未参与训练但模型仍能良好泛化，表明TSAP学到的不变性具有泛化性
- **实用价值高**：对BCI等实际应用来说，不同任务和实验范式使用不同长度的神经记录是常态

## 局限与展望

- 当前仅在iEEG数据上验证，未测试EEG等其他模态
- 仅探索了数据增强策略，未与时间编码器层面的不变性方法（如TF-C、BioFAME等频域方法）结合
- 时间尺度范围有限（1-5秒），更大范围的泛化性需要进一步验证
- 计算开销翻倍（预训练步数从50万增至100万），但考虑到效果提升仍可接受

## 相关工作与启发

- **PopT (chau2025population)**：群体级Transformer，本文的基础框架
- **BrainBERT (wang2023brainbert)**：通道独立的时间编码器，提供冻结的时间嵌入
- **TS-Rep (somaiya2022ts)**：通过triplet目标鼓励时长无关表征
- **TF-C (zhang2022self)**：利用频域一致性促进时间尺度不变性
- 启发：数据增强是解决预处理多样性问题的轻量级有效方案，可推广到其他传感器数据领域

## 评分

- 新颖性: ⭐⭐⭐ 方法本身是多尺度数据增强，较为朴素，但问题发现具有价值
- 实验充分度: ⭐⭐⭐⭐ 多种时间尺度、两个任务、统计检验、嵌入分析全面
- 写作质量: ⭐⭐⭐⭐ Workshop paper 篇幅短小但结构清晰、论述严谨
- 价值: ⭐⭐⭐⭐ 对神经信号基础模型的工程化落地有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)

</div>

<!-- RELATED:END -->

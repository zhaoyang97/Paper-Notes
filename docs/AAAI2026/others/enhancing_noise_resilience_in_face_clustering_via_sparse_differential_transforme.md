---
title: >-
  [论文解读] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer
description: >-
  [AAAI 2026][人脸聚类] 提出预测驱动的 Top-K Jaccard 相似度系数提升邻居纯度，配合稀疏差分 Transformer（SDT）消除噪声注意力，在 MS-Celeb-1M 等大规模人脸聚类数据集上达到 SOTA 性能。 领域现状：基于 GCN 的人脸聚类方法通过图消息传播学习特征…
tags:
  - "AAAI 2026"
  - "人脸聚类"
  - "Jaccard相似度"
  - "Transformer"
  - "噪声边"
  - "自适应邻居发现"
---

# Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer

**会议**: AAAI 2026  
**arXiv**: [2512.22612](https://arxiv.org/abs/2512.22612)  
**代码**: 无  
**领域**: 其他  
**关键词**: 人脸聚类, Jaccard相似度, 稀疏差分Transformer, 噪声边, 自适应邻居发现

## 一句话总结
提出预测驱动的 Top-K Jaccard 相似度系数提升邻居纯度，配合稀疏差分 Transformer（SDT）消除噪声注意力，在 MS-Celeb-1M 等大规模人脸聚类数据集上达到 SOTA 性能。

## 研究背景与动机

**领域现状**：基于 GCN 的人脸聚类方法通过图消息传播学习特征，但构建人脸图时基于 kNN 的余弦距离产生大量噪声边（连接不同身份的节点），消息沿噪声边传播会污染特征。Ada-NETS 和 FC-ESER 引入 Jaccard 相似度系数替代余弦距离，但引入过多无关节点导致 Jaccard 系数区分度不足。

**现有痛点**：
   - FC-ESER 计算的不同人脸之间 Jaccard 系数非常接近——阈值稍低就混合不同身份，稍高就切碎同一身份
   - Ada-NETS 对最优邻居数 $k$ 的预测不准确，偏离最优值
   - Vanilla Transformer 在关系预测中过度关注所有特征关系（包括无关和噪声特征），导致错误聚类

**核心矛盾**：如何精确确定每个节点的有效邻居范围，同时在 Top-K 边界附近可靠地判断节点关系？

**本文目标** 提升 Jaccard 相似度计算的可靠性 + 处理 Top-K 边界的不确定性 + 消除 Transformer 的噪声注意力。

**切入角度**：(1) 用 Transformer 预测每个节点的最优邻居数 Top-K，仅用 Top-K 内邻居计算 Jaccard (2) 用 SDT 处理 Top-K 边界不确定性。

**核心 idea**：预测驱动的 Top-K Jaccard 提纯邻居 + 稀疏差分注意力消除噪声关系判断。

## 方法详解

### 整体框架
构建人脸图 → Transformer 预测 Top-K 邻居边界 → 用 Top-K 精炼人脸图 → SDT 判断 Top-K 附近节点关系 → Map Equation 聚类。

### 关键设计

1. **预测驱动 Top-K Jaccard 相似度**:

    - 功能：动态预测每个节点的最优邻居数，提升 Jaccard 计算质量
    - 核心思路：用 Transformer 替换 Ada-NETS 的 LSTM 预测 Top-K，仅用 Top-K 之前的邻居计算 Jaccard
    - 距离变换改进：$p_{ij} = \frac{1}{1 + e^{\delta d_{ij} + \epsilon}}$（sigmoid 形式，$\delta=7.5, \epsilon=-5$），放大小距离差异
    - 设计动机：FC-ESER 的指数距离变换压缩了相似度差异导致不同身份的 Jaccard 系数过于接近

2. **稀疏差分 Transformer（SDT）**:

    - 功能：处理 Top-K 边界附近的不确定关系
    - 核心思路：基于 Differential Transformer 的差分注意力消噪 + Top-K 稀疏 mask 屏蔽无关节点
    - 差分注意力：计算两个独立 softmax 注意力图的差值来消除噪声注意力
    - 稀疏 mask：只关注 Top-K 之前的相关节点，屏蔽 Top-K 之后的无关节点
    - 还有 MoE-SDT 变体进一步增强能力
    - 设计动机：Vanilla Transformer 对所有特征关系分配注意力，包括不相关或噪声特征，导致误判

### 损失函数 / 训练策略
二分类交叉熵损失。先训练 Transformer 预测 Top-K，再用 SDT 精化关系，最后 Map Equation 聚类。

## 实验关键数据

### 主实验（MS-Celeb-1M，5种规模）

| 方法 | 584K $F_P$/$F_B$ | 5.21M $F_P$/$F_B$ |
|------|---------|---------|
| K-Means | 79.21/81.23 | 66.47/69.42 |
| GCN(V+E) | 87.93/86.09 | 79.30/79.25 |
| Ada-NETS | ~89/~87 | ~81/~80 |
| **Ours** | **SOTA** | **SOTA** |

### 消融实验
- Top-K Jaccard vs 标准 Jaccard：Top-K 显著提升聚类精度
- SDT vs Vanilla Transformer：SDT 在所有规模上更好
- 距离变换改进（sigmoid vs exponential）：sigmoid 更好地区分相似/不相似样本
- MoE-SDT 进一步提升但增加计算量

### 关键发现
- 预测 Top-K 的精度直接影响聚类质量——Top-K 太大引入噪声，太小丢失信息
- SDT 的稀疏 mask 利用了 Top-K 的先验信息，比通用去噪更有效
- 在最大规模（5.21M 图像）上优势更明显——噪声问题随规模增大而加剧

## 亮点与洞察
- **"预测邻居数+精化关系"的两阶段设计**很实用——分层处理不同粒度的问题
- **差分注意力+稀疏 mask 的组合**巧妙利用了聚类任务的先验信息
- **sigmoid 距离变换**简单但有效，放大了区分度

## 局限与展望
- Top-K 预测本身的准确性仍然有限
- SDT 增加了模型复杂度
- 仅在人脸聚类上验证，通用图聚类有待测试
- MoE-SDT 的计算开销需要考虑

## 相关工作与启发
- **vs Ada-NETS**: Ada-NETS 的 $k_{off}$ 预测不准确；本文用 Transformer 替换 LSTM 更可靠
- **vs FC-ESER**: FC-ESER 的 Jaccard 区分度不足；本文的 Top-K + sigmoid 距离变换显著改善
- **vs Differential Transformer**: DiffTransformer 在 NLP 中提出；本文将其扩展到图聚类并加入稀疏 mask

## 评分
- 新颖性: ⭐⭐⭐⭐ 预测驱动 Top-K + SDT 消噪组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5种规模的大规模实验 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，图示直观
- 价值: ⭐⭐⭐⭐ 大规模人脸聚类的实用方案

## 补充分析
- 本文提出的方法在其特定子领域代表了一种有意义的技术进步
- 核心创新点在于将领域特有的结构性先验知识编码到模型设计中，而非完全依赖数据驱动的端到端学习
- 与同期发表的其他顶会工作相比，本文在问题定义和方法设计的系统性上展现了较高水平的研究素养
- 在实际部署场景中，还需综合考虑计算效率、实时性要求、数据隐私保护以及系统可扩展性等工程因素
- 方法的核心思想具有一定的可迁移性——类似的设计范式可能在相关但不同的任务和数据模态上发挥作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)
- [\[ICML 2026\] Vision Transformer 微调中的非光滑分量优势](../../ICML2026/others/vision_transformer_finetuning_benefits_from_non-smooth_components.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)

</div>

<!-- RELATED:END -->

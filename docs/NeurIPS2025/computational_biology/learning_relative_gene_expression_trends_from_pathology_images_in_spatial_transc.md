---
title: >-
  [论文解读] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics
description: >-
  [NeurIPS 2025][计算生物][空间转录组] 提出 STRank 损失函数，将病理图像基因表达估计重新定义为排序分数估计任务，利用二项分布/多项分布建模表达计数的随机噪声特性，使模型能从包含批次效应和随机波动的空间转录组数据中学习到鲁棒的相对表达关系。 空间转录组技术（如 Visium、Xenium）能在组织切片上…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "空间转录组"
  - "基因表达估计"
  - "排序学习"
  - "病理图像"
  - "批次效应"
---

# Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics

**会议**: NeurIPS 2025  
**arXiv**: [2512.06612](https://arxiv.org/abs/2512.06612)  
**代码**: [GitHub](https://github.com/naivete5656/STRank)  
**领域**: 医学图像 / 空间转录组学  
**关键词**: 空间转录组, 基因表达估计, 排序学习, 病理图像, 批次效应

## 一句话总结

提出 STRank 损失函数，将病理图像基因表达估计重新定义为排序分数估计任务，利用二项分布/多项分布建模表达计数的随机噪声特性，使模型能从包含批次效应和随机波动的空间转录组数据中学习到鲁棒的相对表达关系。

## 研究背景与动机

空间转录组技术（如 Visium、Xenium）能在组织切片上获取高分辨率的基因表达谱，但测序成本高昂。从病理图像直接估计基因表达是一种低成本替代方案，然而面临两大核心挑战：

**批次效应（Batch Effects）**：不同试剂批次、设备等技术因素导致不同组织样本间的表达值存在系统性偏差（缩放和偏移）。使用 MSE 损失的模型会学到这些伪关联而非生物信号。

**随机噪声（Stochastic Noise）**：由于细胞异质性和时间动态，即便外观相同的图像块，观测到的基因表达值也会随机波动。低表达基因的信噪比特别低，噪声可能改变样本间的相对排序。

现有方法主要使用 MSE 逐点损失，逐样本优化绝对表达值，无法处理批次效应。虽然 Ranking Loss 等成对损失能缓解批次效应，但不考虑计数数据的概率特性，在低信号条件下无法有效区分信号与噪声。

本文的核心假设是：**即使绝对表达值受批次效应和噪声影响，基因在不同图像块之间的相对表达趋势在独立实验中仍保持一致**。例如，癌区域相对于非癌区域的癌症特异基因表达总是更高。

## 方法详解

### 整体框架

将传统的"预测绝对表达值"任务重新定义为"预测排序分数"任务。模型 $f: x^{n,i} \to r^{n,i}$ 从图像块预测与尺度无关的排序分数 $r$，反映同一组织内的相对表达关系。特征提取器固定（使用 CONCH 病理基础模型），只训练分类头以评估损失函数效果。

### 关键设计

1. **Pairwise STRank 损失**：给定同一组织的一对图像块 $(x^i, x^j)$，假设表达计数 $e^i_g$ 服从二项分布 $\text{Binomial}(t_g^{i,j}, p_g^i)$，其中 $t_g^{i,j} = e_g^i + e_g^j$ 是基因 $g$ 的总表达量，$p_g^i$ 是斑点 $i$ 的频率参数。模型输出分数 $\hat{r}^i, \hat{r}^j$ 经 softmax 转为概率估计 $\hat{p}_g^i$，通过最小化负对数似然训练：

$$L_{\text{STRank}}^{\text{pair}} = -\sum_{g=1}^{N^g} (e_g^i \log \hat{p}_g^i + e_g^j \log \hat{p}_g^j)$$

其中 $\hat{p}_g^i = \frac{\exp(\hat{r}_g^i)}{\exp(\hat{r}_g^i) + \exp(\hat{r}_g^j)}$。关键优势在于：当 $\hat{r}^i \gg \hat{r}^j$ 时，该损失退化为传统 Ranking Loss 的形式，但在排序不确定时，它会根据计数值大小自适应调整权重——高表达基因的排序更可信，低表达基因的排序允许更大的不确定性。

2. **Listwise STRank 损失**：将成对扩展为列表，假设 $N^k$ 个斑点的表达计数服从多项分布 $\text{Multinomial}(T_g^{(n)}, p_g^i)$。使用 softmax 计算所有斑点的概率：

$$L_{\text{STRank}}^{\text{List}} = -\sum_{g}^{N^g} \sum_{i}^{N^k} e_g^i \log p_g^i$$

列表式学习能捕获全局表达模式，优于成对比较。

3. **总量校正机制**：引入每个斑点的总表达量 $l^i = \sum_g e_g^i$ 作为校正因子，调整概率估计 $p_g^i = \frac{\exp(\hat{r}_g^i) l^i}{\sum_j \exp(\hat{r}_g^j) l^j}$，在保持计数数据离散结构的同时考虑斑点间检测能力差异。

### 损失函数 / 训练策略

- 样本对通过组内随机排列构建，每个参考样本随机配对同组织的另一样本
- 小批量损失聚合来自不同组织的相对信号
- 使用 AdamW 优化器，学习率 $5 \times 10^{-5}$，batch size 256
- 早停策略：patience = 30 epochs

## 实验关键数据

### 合成数据实验

| 损失函数类型 | 方法 | Uniform SCC ↑ | Imbalanced SCC ↑ |
|------------|------|---------------|------------------|
| 逐点 | MSE | 0.748 | 0.583 |
| 逐点 | Poisson | 0.777 | 0.603 |
| 逐点 | Negative Binomial | 0.788 | 0.601 |
| 成对 | Rank | 0.835 | 0.738 |
| 成对 | **PairSTRank** | **0.907** | **0.818** |
| 列表 | PCC | 0.858 | 0.560 |
| 列表 | **ListSTRank** | **0.945** | **0.828** |

### 真实数据集实验（HEST-1k 基准，SCC ↑）

| 损失函数 | IDC | PRAD | PAAD | COAD | READ | ccRCC | IDC-L | 平均 |
|---------|-----|------|------|------|------|-------|-------|------|
| MSE | 0.393 | 0.484 | 0.307 | 0.556 | 0.140 | 0.093 | 0.168 | 0.306 |
| Rank | 0.317 | 0.317 | 0.181 | 0.566 | 0.047 | 0.059 | 0.110 | 0.228 |
| PCC | 0.472 | 0.459 | 0.307 | 0.640 | 0.105 | 0.102 | 0.198 | 0.326 |
| PairSTRank | 0.494 | 0.458 | 0.346 | 0.613 | 0.136 | 0.127 | 0.228 | **0.343** |
| ListSTRank | **0.510** | 0.459 | 0.343 | 0.597 | 0.140 | 0.125 | 0.238 | **0.345** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| PairSTRank vs Rank | SCC +0.072/+0.080 | 建模计数分布优于简单排序 |
| ListSTRank vs PCC | SCC +0.087/+0.268 | 多项分布建模在不平衡条件下优势巨大 |
| Listwise vs Pairwise | ListSTRank > PairSTRank | 全局上下文信息有助于批次效应更强的场景 |
| 带/不带总量校正 | 略有提升 | 解决斑点间检测能力差异 |

### 关键发现

- 相对表达学习（排序方法）一致优于绝对表达学习（逐点方法），特别在有批次效应时
- STRank 的概率建模在低信号条件下优势显著——当基因表达稀疏时，计数数据的随机噪声影响更大
- ListSTRank 在合成数据上表现最佳（能捕获全局模式），但在真实数据上与 PairSTRank 相当
- 真实数据评估本身包含噪声，但 STRank 在平均性能上仍然最优

## 亮点与洞察

- **问题重定义的巧妙**：将表达值估计转为排序分数估计，直接绕过了批次效应问题的根源
- **概率建模的合理性**：用二项/多项分布建模计数数据是统计学上的自然选择，使损失函数能自适应地根据表达量调整权重
- **与传统 Ranking Loss 的统一**：证明了当分数差足够大时 STRank 退化为传统排序损失，建立了优雅的理论联系
- **下游分析友好**：相对表达关系正是差异表达分析等常用下游任务所需的信息

## 局限与展望

- 真实数据上改进幅度有限（平均 SCC 从 0.326 提升到 0.345）
- 只评估了 50 个高变基因，实际应用中基因数可能更多
- 特征提取器固定（CONCH），未探索端到端训练
- 未在跨平台（Visium → Xenium）迁移场景中验证
- 评估指标本身受噪声影响，难以确定真正的性能上限

## 相关工作与启发

- **Learning to Rank**: 经典信息检索方法，本文将其引入空间转录组领域
- **HEST-1k**: 空间转录组基准数据集
- **CONCH**: 病理视觉-语言基础模型，提供特征表示
- 启发：损失函数设计是一个被低估的研究方向，合理的概率建模能在不改变模型架构的情况下带来显著提升

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将排序学习与计数数据概率模型结合的损失函数设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 合成数据验证假设 + 7 个真实数据集，但改进幅度有限
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐ 损失函数方法通用性强，可扩展到其他计数数据场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[CVPR 2026\] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images](../../CVPR2026/computational_biology/from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](../../ICML2025/computational_biology/scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[NeurIPS 2025\] Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow](modeling_microenvironment_trajectories_on_spatial_transcriptomics_with_nicheflow.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)

</div>

<!-- RELATED:END -->

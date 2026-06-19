---
title: >-
  [论文解读] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases
description: >-
  [NeurIPS 2025][医学图像][脑图谱] DCA（Deep Cluster Atlas）提出图引导深度嵌入聚类框架，结合预训练 Swin-UNETR 的体素级时空嵌入和 KNN 图空间正则化，通过 KL 散度对齐软分配与图谱聚类辅助标签，生成功能一致且空间连续的个性化脑图谱，在 HCP 数据集上同态性提升 98.8%、轮廓系数提升 29%，并在自闭症诊断、认知解码等下游任务中超越现有图谱。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "脑图谱"
  - "深度聚类"
  - "Swin-UNETR"
  - "图正则化"
  - "个性化分区"
---

# DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases

**会议**: NeurIPS 2025  
**arXiv**: [2509.01426](https://arxiv.org/abs/2509.01426)  
**代码**: [https://github.com/ncclab-sustech/DCA](https://github.com/ncclab-sustech/DCA)  
**领域**: 脑影像 / 脑图谱构建  
**关键词**: 脑图谱, 深度聚类, Swin-UNETR, 图正则化, 个性化分区

## 一句话总结
DCA（Deep Cluster Atlas）提出图引导深度嵌入聚类框架，结合预训练 Swin-UNETR 的体素级时空嵌入和 KNN 图空间正则化，通过 KL 散度对齐软分配与图谱聚类辅助标签，生成功能一致且空间连续的个性化脑图谱，在 HCP 数据集上同态性提升 98.8%、轮廓系数提升 29%，并在自闭症诊断、认知解码等下游任务中超越现有图谱。

## 研究背景与动机

**领域现状**：脑图谱是神经影像数据降维和可解释分析的基础工具。数十年来提出了数百种图谱（Yeo、Schaefer、AAL、MMP 等），基于解剖、功能或细胞构筑学标准，分辨率从 <10 到 >1000 区域不等。

**现有痛点**：(a) 大多数图谱基于皮层表面，忽略皮层下和白质结构；(b) 分辨率固定且预定义，用户无法灵活控制；(c) 组级模板基于平均数据，忽略个体间的显著变异；(d) 传统聚类（K-Means、层次聚类、谱聚类）不考虑空间连续性，产生碎片化的不合理分区。

**核心矛盾**：fMRI 数据信噪比低、维度极高（灰质 mask 含数万体素），使功能相似性和空间连续性难以同时优化。距离惩罚策略需要精心调参，否则会牺牲功能一致性。

**本文目标** 设计一个可扩展的深度聚类框架，能生成体素级、个性化的脑图谱，同时保证功能一致性和空间连续性。

**切入角度**：利用 Swin-UNETR 通过掩码重建预训练学习体素级时空嵌入，然后用 KNN 图的空间先验约束聚类过程——让功能相近且空间相邻的体素归为同一区域。

**核心 idea**：预训练 Swin-UNETR 提取体素嵌入 + 26-邻域图正则化 + KL 散度联合优化 = 功能一致且空间连续的个性化脑图谱。

## 方法详解

### 整体框架
**输入**：4D fMRI 数据（$96\times96\times96\times300$，空间+时间）+ 二值 ROI mask $M$。**输出**：$K$ 类体素级分区标签。**Pipeline**：(A) Swin-UNETR 掩码重建预训练 → (B) 冻结编码器提取体素嵌入 → 可学习聚类中心 + 软分配 → 26-邻域图构建+谱聚类辅助标签 → KL 散度优化（交替更新嵌入、图权重、聚类中心）。

### 关键设计

1. **4D fMRI 掩码重建预训练**：

    - 功能：学习包含局部和全局上下文的体素级时空嵌入
    - 核心思路：在 Swin-UNETR 编码器-解码器结构上，随机掩码 80% 的时空 patch，训练编码器重建被掩盖区域。编码器输出分辨率不变的特征图 $96\times96\times96\times256$，每个体素对应一个 256 维嵌入
    - 设计动机：Swin Transformer 的层级结构和窗口注意力保留空间层次关系，比标准 ViT 更适合体素级建模。80% 的高掩码率迫使模型学习远程依赖

2. **可学习聚类中心 + 软分配**：

    - 功能：将体素嵌入与 $K$ 个聚类中心关联
    - 核心思路：维护可训练矩阵 $\{\boldsymbol{\mu}_j\}_{j=1}^K \subset \mathbb{R}^d$（正交初始化+L2归一化）。计算每个体素到所有中心的欧氏距离 $\Delta_{ij} = \|\mathbf{z}_i - \boldsymbol{\mu}_j\|_2$，经 min-max 归一化和 softmax 得到软分配 $\mathbf{q}_i \in \Delta^{K-1}$
    - 设计动机：软分配允许梯度反传，使聚类中心和编码器可以联合优化

3. **26-邻域图构建与空间正则化**：

    - 功能：强制空间相邻体素倾向归入同一区域
    - 核心思路：对 ROI mask 内的体素建立 26-邻域图 $G=(V,E)$（3×3×3 立方体内所有邻居，排除中心）。边权为去均值嵌入的余弦相似度 $a_{ij} = \cos(\mathbf{z}_i - \bar{\mathbf{z}}_i, \mathbf{z}_j - \bar{\mathbf{z}}_j)$。对加权图做**谱聚类**（计算 Laplacian $L=D-W$ 的最小 $K$ 个特征向量 + K-Means）得到硬辅助标签 $\mathbf{p}$
    - 设计动机：谱聚类天然考虑图结构（空间邻域），产生的标签具有空间连续性。用匈牙利算法在迭代间对齐标签，避免标签置换问题

4. **KL 散度目标函数**：

    - 功能：让软分配（来自嵌入距离）向硬辅助标签（来自图谱聚类）靠拢
    - 核心思路：$\mathcal{L} = \text{KL}(\mathbf{P} \| \mathbf{Q}) = \frac{1}{N} \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$，其中 $\mathbf{P}$ 是辅助标签的 one-hot 编码，$\mathbf{Q}$ 是软分配矩阵。梯度仅反传到聚类中心 $\{\boldsymbol{\mu}_j\}$ 和 Swin-UNETR 最后投影层，冻结其余编码器权重
    - 设计动机：KL 散度是深度聚类的标准目标函数（源自 DEC）；辅助标签引入空间先验，使优化后的软分配自然产生空间连续的分区

5. **组级图谱生成**：

    - 功能：从个性化图谱汇聚为可比较的组级图谱
    - 核心思路：三步法——(a) 选 $K$ 个模板标签向量；(b) 每个灰质体素分配到标签相似度最高的模板；(c) 保留每个 parcel 的最大连通分量，孤立小区域重分配给相邻 parcel

### 损失函数 / 训练策略
- 预训练：8 epochs，2× A100，batch size 4，Adam lr=0.01，掩码率 0.8
- 聚类微调：仅更新聚类中心和最后投影层，Adam lr=0.01
- 交替迭代：更新嵌入 → 重新计算图权重 → 重新谱聚类得辅助标签 → KL 优化
- 支持 $K \in \{41, 100, 200, 360, 400, 500, 800\}$ 不同分辨率

## 实验关键数据

### 主实验 — 同态性和轮廓系数（100 HCP 被试）

| 图谱 | Parcels | Homogeneity ↑ | Silhouette ↑ |
|------|---------|---------------|--------------|
| Yeo | 7 | ~0.02 | ~0.005 |
| Brodmann | 41 | 低 | 低 |
| Schaefer200 | 200 | 基线 | 基线 |
| MMP | 360 | 基线 | 基线 |
| **DCA200** | 200 | **+77.7%** vs Schaefer200 | **+19.5%** vs Schaefer200 |
| **DCA（均值）** | 41-800 | **+98.8%** 均值提升 | **+29%** 均值提升 |

### 消融实验 — 各组件贡献

| 方法 | Homogeneity ↑ | Silhouette ↑ | 连通分量/parcel ↓ |
|------|---------------|--------------|-------------------|
| fMRI + K-Means | 0.074±0.019 | 0.006±0.005 | 447.9 |
| fMRI + Graph Cut | 0.086±0.020 | 0.018±0.005 | 8.9 |
| Embedding + K-Means | 0.079±0.020 | 0.015±0.007 | 322.3 |
| Embedding + Graph Cut | 0.090±0.021 | 0.021±0.006 | 4.9 |
| **DCA (完整)** | **0.100±0.022** | **0.030±0.007** | **1.005** |

### 下游任务 — 分类准确率

| 任务 | DCA100 | DCA200 | DCA360 | Schaefer 对应 | 说明 |
|------|--------|--------|--------|--------------|------|
| 性别预测 (HCP) | 竞争力 | 优势 | 优势 | 基线 | 静息态 FC |
| 流体智力 (HCP) | 竞争力 | 优势 | 优势 | 基线 | 静息态 FC |
| 7-任务解码 (HCP) | 优势 | 顶级 | 顶级 | 基线 | 任务 FC |
| ASD 诊断 (ABIDE) | 优势 | **最优** | 优势 | 基线 | 静息态 FC |

### 关键发现
- 所有分辨率下 DCA 均优于对应的最佳 baseline，提升在低分辨率时更显著
- K-Means 直接应用于原始 fMRI 数据平均产生 447.9 个连通分量/parcel——严重碎片化；DCA 仅 1.005
- 图正则化是空间连续性的关键：Graph Cut 将碎片从 447.9 降到 8.9，DCA 进一步降到 1.005
- 预训练嵌入比原始 fMRI 提升约 5-10% 的同态性，说明特征学习很重要
- 任务特异性图谱（DCA$^{\text{gender}}_{100}$）在性别分类上提升高达 +12%（CNN）和 +10%（k-GNN）

## 亮点与洞察
- **深度聚类首次用于脑图谱**：将 DEC 范式引入神经影像，通过 KL 散度+图先验联合优化嵌入和聚类——技术迁移优雅
- **空间连续性几乎完美**：平均 1.005 连通分量/parcel，无需后处理即可得到空间连续图谱，远超传统方法
- **灵活性极强**：支持任意 ROI mask、任意分辨率 $K$、个性化/组级切换——通用框架
- **可迁移思路**：Swin-UNETR 掩码预训练+图正则化聚类的组合可直接用于其他 3D/4D 医学图像的区域分割（如心脏功能分区、肺区划分）

## 局限与展望
- 体素级聚类的内存和计算成本高，全脑规模困难
- 固定 26-邻域 KNN 图可能抑制远程功能连接
- 仅使用单模态 fMRI，未整合结构 MRI、扩散 MRI 或电生理数据
- 不同任务最优分辨率不同（无通用最优 $K$），用户需要自行选择
- 谱聚类步骤计算 Laplacian 特征向量，对大规模图（全脑 ~10 万体素）可能成为瓶颈

## 相关工作与启发
- **vs Schaefer（梯度驱动聚类）**：Schaefer 用空间加权功能连接做组级聚类；DCA 用深度嵌入+图先验做个性化聚类，同态性提升 77.7%
- **vs GIANT（遗传驱动图谱）**：GIANT 整合遗传信息；DCA 仅用 fMRI，但在功能指标上更优
- **vs 脑分割方法（Swin-UNETR/DDParcel）**：分割是有监督的（已知标签），图谱构建是无监督的（需要发现新分区）；DCA 利用预训练分割模型的特征学习能力但做无监督聚类

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将深度聚类应用于脑图谱构建，图正则化+KL 优化设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 12 种 baseline 图谱对比、多分辨率、6 个下游任务、详尽消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，benchmark 平台有社区价值，但符号较多
- 价值: ⭐⭐⭐⭐ 脑图谱领域的重要进步，可配置分辨率+个性化+开源代码实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities](../../CVPR2026/medical_imaging/virtual_nodes_guided_dynamic_graph_neural_network_for_brain_tumor_segmentation_w.md)
- [\[NeurIPS 2025\] EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks](evobrain_dynamic_multi-channel_eeg_graph_modeling_for_time-evolving_brain_networ.md)
- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)

</div>

<!-- RELATED:END -->

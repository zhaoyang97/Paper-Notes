---
title: >-
  [论文解读] Multi-Label Cluster Discrimination for Visual Representation Learning
description: >-
  [ECCV 2024][信息检索/RAG][视觉表征学习] 提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。
tags:
  - "ECCV 2024"
  - "信息检索/RAG"
  - "视觉表征学习"
  - "聚类判别"
  - "多标签分类"
  - "CLIP"
  - "大规模预训练"
---

# Multi-Label Cluster Discrimination for Visual Representation Learning

**会议**: ECCV 2024  
**arXiv**: [2407.17331](https://arxiv.org/abs/2407.17331)  
**代码**: 有 ([https://github.com/deepglint/unicom](https://github.com/deepglint/unicom) + Hugging Face)  
**领域**: 社会计算  
**关键词**: 视觉表征学习, 聚类判别, 多标签分类, CLIP, 大规模预训练

## 一句话总结

提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。

## 研究背景与动机

### 问题一：实例判别无法捕获语义结构

CLIP 等语言监督视觉预训练方法采用实例判别（instance discrimination）：将每对图文视为唯一实例，不同实例始终作为负样本对推远。当 mini-batch 中大量语义相似的实例被当作负对时，语义相近的样本会在嵌入空间中被不合理地推远。这导致**实例判别难以编码训练数据的语义结构**。

### 问题二：单标签聚类判别忽略多标签信号

为解决实例判别的局限，聚类判别方法（DeepCluster、SwAV、UNICOM 等）通过迭代聚类和分类来探索语义结构。将相似实例聚到同一簇可以拉近语义相近的样本。然而，大多数聚类判别只给每张图像分配**单一伪标签**。

### 核心观察与动机

自然图像常常包含多个视觉对象或属性（如一张图中同时有建筑、天空、行人）。CLIP 的自然语言监督可以为单张图像提供多粒度标签（对象、场景、动作、关系），而单标签聚类无法覆盖一张图中的所有视觉信号。

因此，作者提出：**为每张图像分配多个聚类标签**（多标签聚类判别），同时设计专门的消歧多标签损失来处理大规模自动聚类中的噪声问题。

## 方法详解

### 整体框架

MLCD 由两步组成：(1) **聚类步骤**：基于预训练 CLIP 特征对 LAION-400M 做离线 k-means 聚类为100万个类别，为每张图像分配多个最近聚类中心作为正类标签；(2) **判别步骤**：设计消歧多标签分类损失训练图像编码器。

### 关键设计

#### 1. **多标签聚类分配 (Multi-label Clustering)**

**功能**：为每张训练图像分配 $l$ 个正类标签（而非传统的1个），捕获图像中的多粒度视觉信号。

**核心思路**：利用预训练 CLIP ViT-L/14 的特征，对 LAION-400M 做一步离线 k-means 聚类（$k=1M$ 类别，约10分钟完成）。对每张图像计算其嵌入与所有聚类中心的余弦相似度，选取最近的 $l$ 个中心（默认 $l=8$）作为正类标签，其余为负类。

**设计动机**：由于 CLIP 模型判别力有限，单一伪标签可能无法覆盖图像中所有视觉信号。通过选取多个最近中心，可以多粒度地描述一张图像的语义内容（类似于CLIP文本可提供多粒度标签的能力）。优先保证类内纯度（聚100万类），通过 PartialFC 采样缓解类间冲突。

#### 2. **多标签分类基础损失 (MLC Loss)**

**功能**：将多标签分类形式化为减小所有负类-正类相似度差 $(s_j - s_i)$ 的优化问题。

**核心思路**：令 $\{s_i\}$ $(i=1,...,l)$ 为正类相似度，$\{s_j\}$ $(j=1,...,k-l)$ 为负类相似度，基础多标签损失为：

$$\mathcal{L}_\text{MLC} = \log\left(1 + \sum_{j \in \Omega_n} \exp(s_j) \sum_{i \in \Omega_p} \exp(-s_i)\right)$$

等价于遍历每对 $(s_j, s_i)$ 优化减小 $(s_j - s_i)$。同时引入 PartialFC 负类采样，仅随机采样 $r=10\%$ 的负类中心参与计算。

**设计动机**：继承 Circle Loss 的成对比较思想，自然扩展到多标签场景。PartialFC 采样在百万级类别下既节约计算又缓解类间冲突。

#### 3. **多标签分类消歧损失 (MLCD Loss)**

**功能**：解决基础 MLC 损失中 $(s_j - s_i)$ 优化导致的决策边界歧义问题。

**核心思路**：MLC 优化 $(s_j - s_i)$ 的决策边界为 $s_j - s_i = m$，但这允许歧义：$\{s_j, s_i\}=\{0.1, 0.4\}$ 和 $\{0.5, 0.8\}$ 都满足 $m=0.3$，但后者的 $s_j=0.5$ 仍然偏高。因此额外引入两个优化目标——**最大化 $s_i$**（正类高相似度）和**最小化 $s_j$**（负类低相似度）：

$$\mathcal{L}_\text{MLCD} = \underbrace{\log\left(1 + \sum_{i \in \Omega_p} \exp(-s_i)\right)}_{\text{正类损失}} + \underbrace{\log\left(1 + \sum_{j \in \Omega'_n} \exp(s_j)\right)}_{\text{负类损失}}$$

关键优雅之处在于：引入额外的两项后，**正类损失和负类损失被自然分离**（数学上可证），各自独立优化互不干扰。

**设计动机**：(1) 实验可视化（Fig.3）显示 MLCD 比 MLC 在正类余弦相似度 $s_i$ 上提升更快、分布更集中，负类 $s_j$ 更趋近零（更正交）；(2) 分离的正负类损失使优化更稳定、收敛更好；(3) 相比引入阈值的 TLPR 损失，MLCD 更简洁且适合大规模噪声数据。

### 损失函数 / 训练策略

- **预训练**：LAION-400M 数据集，ViT-L/14 backbone，32 epochs，batch size 32K，80×A100
- **优化器**：AdamW，学习率 0.001，权重衰减 0.2
- **加速**：混合精度训练 + Flash Attention + DALI 数据加载
- **文本编码器**：冻结图像编码器后用 LiT 方式从头训练32 epochs（用于 zero-shot 任务）
- **默认超参**：$k=1M$ 类别，$r=0.1$ 负类采样率，$l=8$ 正标签数

## 实验关键数据

### 主实验（Linear Probe）

26个下游数据集上的 linear probe 性能（ViT-L/14 backbone）：

| 方法 | 训练数据 | 26数据集平均 | 代表性数据集 |
|------|---------|------------|------------|
| CLIP | WIT-400M | 84.2 | IN1K: 83.9 |
| OpenCLIP | LAION-400M | 82.3 | IN1K: 82.1 |
| UNICOM | LAION-400M | 83.3 | IN1K: - |
| **MLCD (Ours)** | **LAION-400M** | **84.6** | **IN1K: 84.6** |

- 比 OpenCLIP 平均提升 **2.3%**（25/26 数据集上胜出）
- 比 UNICOM 平均提升 **1.3%**（23/26 数据集上胜出）
- 甚至超越使用私有WIT数据的CLIP

### 主实验（Zero-shot 分类 + 检索）

| 方法 | 训练数据 | Zero-shot 25数据集平均 | MSCOCO I2T R@1 | MSCOCO T2I R@1 |
|------|---------|---------------------|----------------|----------------|
| CLIP | WIT-400M | 66.9 | 56.2 | 35.8 |
| OpenCLIP | LAION-400M | 63.6 | 58.0 | 41.3 |
| FLIP | LAION-400M | 66.0 | 60.2 | 44.2 |
| **MLCD (Ours)** | **LAION-400M** | **67.5** | **60.8** | **44.5** |

Zero-shot 分类比 OpenCLIP 提升 **3.9%**，比 FLIP 提升 **1.5%**。MSCOCO 检索全面领先。

### 消融实验

在 ViT-B/32 + LAION-400M（5 epochs）上的消融：

| 消融维度 | 配置 | IN1K Linear Probe | 说明 |
|---------|------|-------------------|------|
| 类别数 $k$ | 100K / 200K / 500K / **1M** / 2M / 5M | 66.9 / 71.1 / 74.4 / **75.2** / 74.9 / 74.7 | 1M最优，过多则类间冲突加剧 |
| 负类采样率 $r$ | 0.01 / 0.05 / **0.1** / 0.2 / 0.5 / 1.0 | 73.4 / 75.1 / **75.2** / 74.9 / 68.3 / 63.2 | 0.1最优，1.0性能骤降 |
| 正标签数 $l$ | 1 / 2 / 4 / **8** / 16 / 32 | 71.4 / 72.9 / 73.2 / **75.2** / 72.1 / 68.7 | 8个最优，过多引入噪声标签 |
| MLC vs MLCD | MLC / **MLCD** (32 epochs, ViT-B/32) | FT: 80.9/81.2, LP: 76.9/78.1, ZS: 63.9/64.5 | MLCD 在所有设置下均优于 MLC |

### ImageNet 鲁棒性评估

| 方法 | 数据 | Finetune | Linear | Zero-Shot | IN-V2 | IN-A | IN-R |
|------|------|----------|--------|-----------|-------|------|------|
| OpenCLIP | LAION | 86.2 | 82.1 | 72.8 | 64.0 | 48.3 | 84.3 |
| FLIP | LAION | - | - | 74.6 | 66.8 | 51.2 | 86.5 |
| **Ours** | **LAION** | **87.1** | **84.6** | **75.6** | **68.9** | **56.4** | **85.1** |

在所有鲁棒性基准（IN-V2/A/R/ObjectNet）上均显著优于 OpenCLIP。

### 关键发现

1. **多标签显著优于单标签**：$l=8$ vs $l=1$ 提升 3.8%（75.2 vs 71.4），证明多标签信号的价值
2. **消歧损失MLCD一致优于MLC**：三个评估设置（FT/LP/ZS）全面提升，正类分布更紧凑、负类更正交
3. **负类过采样有害**：$r=1.0$ 时性能从 75.2 暴跌到 63.2，说明百万级类别下类间冲突严重
4. **固定标签数优于自适应阈值**：全局相似度阈值难搜索，固定 $l=8$ 利用了"日常图像统计上包含几个视觉概念"的先验
5. **跨数据集一致性**：LAION-400M → COYO-700M 转换时 MLCD 仍一致优于 UNICOM

## 亮点与洞察

1. **简洁有效的多标签策略**：仅需一步离线聚类 + 选Top-$l$近邻中心，不增加额外聚类开销
2. **MLCD 损失的数学优雅性**：通过额外两个优化目标，正负类损失自然分解为两个独立的 log-sum-exp 项，实现简洁而高效
3. **工程实用性强**：配合 PartialFC 和负类采样，在百万级类别下高效训练；代码和模型已开源，具备即插即用价值
4. **实验覆盖极广**：26个 linear probe 数据集 + 25个 zero-shot 数据集 + 检索 + 鲁棒性，说服力强

## 局限与展望

1. **聚类质量依赖预训练模型**：使用 CLIP ViT-L/14 做一步离线聚类，聚类质量受限于该模型的特征质量，未探索迭代聚类-训练
2. **正标签数固定不灵活**：$l=8$ 对所有图像统一设置，不同图像的视觉复杂度差异未被考虑
3. **仅验证ViT架构**：未报告CNN backbone的结果
4. **文本编码器需额外训练**：zero-shot 需 LiT 额外训练文本编码器32 epochs，增加了流程复杂度
5. **未与最新方法对比**：如 SigLIP、EVA-CLIP 等

## 相关工作与启发

- **UNICOM (ICLR 2023)**：同一团队前序工作，单标签聚类判别，MLCD 在此基础上引入多标签
- **Circle Loss**：MLCD 损失的理论基础，揭示了 $(s_j - s_i)$ 优化的歧义问题
- **PartialFC**：同一团队的人脸识别工作，负类采样策略在百万级分类中至关重要
- 启发：多标签聚类思想可与自蒸馏、MAE 等结合，或扩展到视频/3D 表征学习

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多标签聚类判别概念自然且有效，MLCD消歧损失设计优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 51个下游数据集评估 + 详尽消融 + 鲁棒性 + 跨数据集验证
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法推导流畅，消融设计系统
- **价值**: ⭐⭐⭐⭐⭐ — 代码模型开源，方法简洁通用，对CLIP系列改进有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)
- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](../../ICLR2026/information_retrieval/digging_deeper_learning_multi-level_concept_hierarchies.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](../../NeurIPS2025/information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)

</div>

<!-- RELATED:END -->

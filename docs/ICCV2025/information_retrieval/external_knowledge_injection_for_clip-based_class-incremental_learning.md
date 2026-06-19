---
title: >-
  [论文解读] External Knowledge Injection for CLIP-Based Class-Incremental Learning
description: >-
  [ICCV 2025][信息检索/RAG][类增量学习] 提出 Engine（ExterNal knowledGe INjEction）框架，通过双分支注入调优（视觉分支用数据增强、文本分支用 GPT-4 生成判别性描述）和推理时后调优知识注入（成对判别特征重排序），在无需存储历史样本的条件下，在 9 个基准数据集上以 3-10% 的优势超越所有 CLIP-based 类增量学习方法。
tags:
  - "ICCV 2025"
  - "信息检索/RAG"
  - "类增量学习"
  - "CLIP"
  - "外部知识注入"
  - "GPT-4"
  - "视觉语言模型"
---

# External Knowledge Injection for CLIP-Based Class-Incremental Learning

**会议**: ICCV 2025  
**arXiv**: [2503.08510](https://arxiv.org/abs/2503.08510)  
**代码**: [GitHub](https://github.com/LAMDA-CL/ICCV25-ENGINE)  
**领域**: 信息检索  
**关键词**: 类增量学习, CLIP, 外部知识注入, GPT-4, 视觉语言模型

## 一句话总结

提出 Engine（ExterNal knowledGe INjEction）框架，通过双分支注入调优（视觉分支用数据增强、文本分支用 GPT-4 生成判别性描述）和推理时后调优知识注入（成对判别特征重排序），在无需存储历史样本的条件下，在 9 个基准数据集上以 3-10% 的优势超越所有 CLIP-based 类增量学习方法。

## 研究背景与动机

类增量学习（CIL）需要模型在持续接收新类别数据时不遗忘旧知识。基于预训练 CLIP 的 CIL 方法利用了 CLIP 的强大零样本能力，通过匹配视觉嵌入和类名文本嵌入进行分类。然而现有方法存在三个核心问题：

**模板文本信息不足**：CLIP 默认使用 "a photo of a [CLASS]" 作为文本匹配目标，忽略了丰富的细粒度视觉特征。例如，"cat" 可以被分解为 "long thin tail"、"soft fur"、"round face with large eyes" 等特征，这些特征作为匹配目标能提供更准确的分类信号。

**视觉 Prompt 的局限**：L2P、DualPrompt 等方法通过学习视觉 Prompt 适配下游任务，但无法利用文本侧信息，且模型需要将类名自动分解为细粒度概念去匹配视觉特征——这种分解能力在增量更新中会退化。

**文本 Prompt 的过拟合**：CoOp 等方法学习可训练的文本 Prompt，但容易对训练实例的特定分布过拟合，在测试分布变化时表现不佳。

本文的核心思想是：利用 GPT-4 等大语言模型作为外部知识源，为每个类别提供通用、全面的视觉特征描述，并通过轻量级注入单元将这些知识编码到 CLIP 中，避免了 Prompt 学习的过拟合问题。

## 方法详解

### 整体框架

Engine 包含两个阶段的知识注入：
- **On-the-fly 知识注入**（训练阶段）：冻结 CLIP 编码器，学习视觉和文本两个分支的轻量级注入单元
- **Post-tuning 知识注入**（推理阶段）：利用 GPT-4 生成的成对判别特征重排序 top-k 预测

### 关键设计

1. **文本知识注入单元**:

    - 功能：利用 GPT-4 生成每个类别的判别性视觉特征描述，将这些外部知识编码到线性注入单元中
    - 核心思路：用 GPT-4 问每个类别的独特视觉特征（如 "What are unique visual features of [CLASS] in a photo?"），得到描述集合 $\mathbf{d}_i$。在冻结的文本编码器 $\bar{g_t}$ 后添加线性层 $u_t(\cdot): \mathbb{R}^d \to \mathbb{R}^d$，通过最大化相似度损失注入知识：
    $\mathcal{L}_t = -\text{Sim}\left(u_t\left(\bar{g_t}(\mathbf{t}_i)\right), \bar{g_t}(\mathbf{d}_i)\right)$
      使得经过注入单元处理的模板文本特征 $u_t(\bar{g_t}(\mathbf{t}_i))$ 接近 GPT-4 生成的详细描述特征 $\bar{g_t}(\mathbf{d}_i)$。
    - 设计动机：通过这种方式，即使推理时使用简单的类名模板，注入单元也能自动将其映射到包含 "long thin tail" 等细粒度特征的嵌入空间，比直接学习可训练 Prompt 更通用且不易过拟合。

2. **视觉知识注入单元**:

    - 功能：通过数据增强增强视觉特征的多样性
    - 核心思路：在冻结的视觉编码器 $\bar{g_i}$ 后添加线性层 $u_i(\cdot): \mathbb{R}^d \to \mathbb{R}^d$，最大化原始图像与增强图像的特征相似度：
    $\mathcal{L}_i = -\text{Sim}\left(u_i\left(\bar{g_i}(\mathbf{x})\right), \bar{g_i}(\mathcal{A}(\mathbf{x}))\right)$
      其中 $\mathcal{A}$ 是 AutoAugment 数据增强策略。
    - 设计动机：视觉注入单元学习将原始特征映射到对增强变换不变的空间，使得视觉嵌入包含更丰富、更鲁棒的视觉信息。

3. **任务特定注入单元扩展 + 原型重放**:

    - 功能：为每个增量任务创建独立的注入单元，防止遗忘
    - 核心思路：学习第 $b$ 个任务时，初始化新的 $u_t^b, u_i^b$，冻结所有之前的注入单元。最终表示是所有注入单元的求和：
    $G_i(\mathbf{x}) = \sum_{p=1}^{b-1} \bar{u}_i^p(\bar{g_i}(\mathbf{x})) + u_i^b(\bar{g_i}(\mathbf{x}))$
      同时利用每个类别的视觉原型 $\mathbf{p}_k$（类别平均嵌入）作为旧类代理参与训练，并添加高斯噪声 $\epsilon \sim \mathcal{N}(0, \alpha^2 \mathbf{I})$ 增强多样性。
      注入单元是线性层，可重参数化为单个等效层 $\sum_p u_i^p$，不增加推理复杂度。
    - 设计动机：由于 CLIP 图像编码器全程冻结，视觉原型在不同阶段是一致的，可以作为历史类别的可靠替代，无需存储原始数据。

4. **Post-tuning 知识注入（推理时重排序）**:

    - 功能：利用 GPT-4 生成的成对判别特征对 top-k 预测进行局部精炼
    - 核心思路：提取模型 top-k 预测的类别集合 $\{y_{i_1}, \ldots, y_{i_k}\}$，对每对类使用 GPT-4 生成判别性差异描述（如 "cat 有 long thin tail, 而 lion 有 stockier tail with tuft"），构建判别矩阵 $\mathbf{D} = [\mathbf{d}_{ij}]$。用零样本 CLIP 匹配查询图像到这些成对描述：
    $f_{\text{pt}, y_i}(\mathbf{x}) = \frac{1}{k-1} \sum_{j=1}^{j \neq i} f_{y_i}(\mathbf{x}, \mathbf{d}_{ij})$
      最终预测是注入特征匹配和后调优匹配的和：$f(\mathbf{x}) = f_{\text{inj}}(\mathbf{x}) + f_{\text{pt}}(\mathbf{x})$
    - 设计动机：后调优使用零样本 CLIP（不随增量更新），提供了对预测的局部精细化能力，尤其擅长区分视觉相似的类别（如 cat vs lion）。

### 损失函数 / 训练策略

总训练目标结合三项损失：
$$\min_{\{u_t^b, u_i^b\}} \sum_{(\mathbf{x}, y) \in \mathcal{D}^b \cup \mathbf{P}} \mathcal{L}_t + \mathcal{L}_i + \mathcal{L}_c$$

其中 $\mathcal{L}_c$ 是基于注入特征的对比损失（Eq. 9），$\mathbf{P}$ 是原型集。使用 SGD 优化器，batch size 64，训练 10 epochs，学习率从 0.05 余弦退火。超参数 $\alpha = 0.25$（原型噪声），$k = 5$（后调优候选数）。

## 实验关键数据

### 主实验

9 个基准数据集上的最终准确率 $\mathcal{A}_B$（CLIP ViT-B/16，无 exemplar 存储）：

| 方法 | Aircraft B0I10 | CIFAR100 B0I10 | Cars B0I10 | ImageNet-R B0I20 | CUB B0I20 | UCF B0I10 |
|------|---------------|---------------|-----------|-----------------|----------|----------|
| SimpleCIL | 48.09 | 76.63 | 86.85 | 74.48 | 77.52 | 85.68 |
| L2P | 28.29 | 73.03 | 61.82 | 66.52 | 57.93 | 76.43 |
| CODA-Prompt | 27.69 | 73.43 | 66.47 | 68.95 | 62.98 | 80.14 |
| RAPF | 23.61 | 78.04 | 62.85 | 70.48 | 62.77 | 80.33 |
| **Engine** | **58.69** | **79.22** | **90.08** | **80.37** | **80.20** | **90.03** |

Engine 在所有数据集上都显著领先，特别是在细粒度数据集 Aircraft 上超过 RAPF 达 35%。

### 消融实验

各组件贡献（CIFAR100 B0 Inc10 最终准确率）：

| 配置 | $\mathcal{A}_B$ | 说明 |
|------|----------------|------|
| ZS-CLIP（基线） | ~71.38 | 零样本CLIP |
| + Visual Injection | ~74 | 视觉增强立即见效 |
| + Dual Injection（视觉+文本） | ~77 | 双模态注入进一步提升 |
| + Post-tuning Injection（完整Engine） | **79.22** | 后调优最终精炼 |

外部知识公平对比（给所有方法相同的 GPT-4 描述和数据增强）：

| 方法 | 模板 | ImageNet-R $\mathcal{A}_B$ | CIFAR100 $\mathcal{A}_B$ |
|------|------|--------------------------|------------------------|
| ZS-CLIP | GPT描述 | 77.59 | 71.70 |
| RAPF | GPT描述 | 71.04 | 78.52 |
| **Engine** | **GPT描述** | **80.37** | **79.22** |

即使所有方法使用相同的外部知识，Engine 仍然最优，说明其知识利用方式更高效。

### 关键发现

- **无 exemplar 存储仍超越有 exemplar 方法**：Engine（0 exemplar）在 Aircraft 上 58.69%，超越 PROOF（20/class exemplar）的 53.59%
- **每个组件都有贡献**：视觉注入、文本注入、后调优层层递进，消融验证了每个组件的必要性
- **参数鲁棒性**：$\alpha \in [0.15, 0.35]$ 和 $k \in [3, 7]$ 范围内性能稳定
- **t-SNE 可视化证实**：Engine 实现了视觉-文本特征的对齐（同类聚类），且学习新任务时不遗忘之前任务的聚类结构
- **后调优能修正错误预测**：可视化显示 post-tuning 能够将错误预测（如 acorn→birdhouse）通过关注局部判别特征纠正

## 亮点与洞察

1. **外部知识的优雅利用**：GPT-4 为每个类别提供通用、全面的描述，比直接学习 Prompt 更不容易过拟合于训练数据分布
2. **注入单元设计精妙**：单层线性层即可编码外部知识到冻结编码器中，且可通过权重求和重参数化，不增加推理开销
3. **双阶段知识注入**：训练时注入全局类别知识，推理时注入局部成对判别知识，两者互补
4. **无 exemplar 存储设计**：利用冻结编码器产出的一致性原型作为旧类替代，巧妙避开了数据存储需求

## 局限与展望

- 依赖 GPT-4 提供外部知识，对 GPT-4 知识库覆盖不到的特定领域（如高度专业化的医学影像类别）可能失效
- 后调优需要对 top-k 的每个类对调用 GPT-4 生成判别描述，在类别数非常大时（如 10,000+ 类）组合数可能爆炸
- 注入单元数量随任务数线性增长（虽然可重参数化），但原型数量也线性增长
- 数据增强策略（AutoAugment）较为通用，针对特定领域设计的增强可能更有效

## 相关工作与启发

- 外部知识注入思路可推广至其他增量学习场景（如目标检测的类增量学习）
- GPT-4 生成的成对判别描述可用于零样本分类中的困难类对消歧
- 注入单元的设计理念（冻结编码器 + 轻量级线性映射）可应用于任何需要适配下游任务的预训练模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 双阶段外部知识注入的思路新颖，尤其是后调优的成对判别重排序
- 实验充分度: ⭐⭐⭐⭐⭐ 9个数据集×多种设定的全面评估，消融和对比实验非常充分
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻，方法表示清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 在CLIP-based CIL领域取得大幅领先，外部知识注入思路有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SuperCLIP: CLIP with Simple Classification Supervision](../../NeurIPS2025/information_retrieval/superclip_clip_with_simple_classification_supervision.md)
- [\[ICML 2026\] CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels](../../ICML2026/information_retrieval/care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->

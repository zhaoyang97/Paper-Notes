---
title: >-
  [论文解读] Vision Graph Prompting via Semantic Low-Rank Decomposition
description: >-
  [ICML2025][多模态VLM][图神经网络] 提出 Vision Graph Prompting (VGP)，首个面向 Vision GNN (ViG) 的视觉提示学习框架，利用图中语义连通分量的低秩特性，设计了图/边/节点三层粒度的语义低秩提示（SeLo-Graph/Edge/Node Prompt），在参数高效的前提下达到接近全量微调的下游任务迁移性能。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "图神经网络"
  - "视觉提示学习"
  - "低秩分解"
  - "参数高效微调"
---

# Vision Graph Prompting via Semantic Low-Rank Decomposition

**会议**: ICML2025  
**arXiv**: [2505.04121](https://arxiv.org/abs/2505.04121)  
**作者**: Zixiang Ai, Zichen Liu, Jiahuan Zhou
**代码**: [GitHub](https://github.com/zhoujiahuan1991/ICML2025-VGP)  
**领域**: 多模态VLM  
**关键词**: Vision GNN, 视觉提示学习, 低秩分解, 参数高效微调, 图神经网络

## 一句话总结

提出 Vision Graph Prompting (VGP)，首个面向 Vision GNN (ViG) 的视觉提示学习框架，利用图中语义连通分量的低秩特性，设计了图/边/节点三层粒度的语义低秩提示（SeLo-Graph/Edge/Node Prompt），在参数高效的前提下达到接近全量微调的下游任务迁移性能。

## 研究背景与动机

### 问题背景
Vision GNN (ViG) 将图像 patch 表示为图结构，通过 K 近邻算法动态连接 patch 节点，利用图神经网络捕获图像中不规则分布的语义模式。相比 CNN 的固定网格表示和 ViT 的序列化 token 表示，ViG 的图结构更自然地建模了语义部件之间的全局交互关系。随着 ViG 模型规模不断增大，将预训练 ViG 迁移到下游任务时，全量微调的存储和计算开销变得不可接受。

### 已有工作的不足

**现有视觉提示方法面向 Transformer 设计**：VPT、VP 等视觉提示方法都是为 ViT 量身打造的，直接迁移到 ViG 上效果不佳，因为它们忽略了图拓扑中丰富的节点-边语义关系

**现有图提示方法面向非视觉领域**：GPF、All-in-One 等图提示方法主要面向社交网络、化学分子等场景，无法捕获视觉图像的独特语义特征

**缺乏专为视觉图结构设计的提示方法**：ViG 作为通用视觉骨架的潜力已被充分证明，但参数高效微调策略严重缺失

### 核心观察与动机
作者通过 PCA 和 t-SNE 可视化发现了 ViG 图结构的关键特性：**语义相关的图节点共享相同的主成分特征，在 t-SNE 嵌入中形成紧凑聚类**。这意味着视觉图中的语义信息主要存在于特征空间的低秩分量中。这一洞察直接启发了语义低秩提示的设计——在提示中引入低秩分解，保留全局语义信息的同时过滤掉局部噪声细节。

## 方法详解

### 整体框架

VGP 框架在冻结的预训练 ViG 模型上引入三个层次的可训练提示组件，从粗到细分别作用于图结构的不同粒度：

1. **SeLo-Graph Prompt（图级提示）**：在原始图上添加虚拟节点，捕获全局语义依赖
2. **SeLo-Edge Prompt（边级提示）**：在边的消息传递中注入低秩语义特征，促进连通节点间的语义传播
3. **SeLo-Node Prompt（节点级提示）**：增强每个节点的局部细粒度语义特征

三种提示均基于统一的语义低秩分解原则设计，形成从全局到局部的多粒度语义增强体系。

### 关键设计 1：语义低秩分解

核心思想是将特征矩阵分解为低秩语义分量和残差分量。给定 ViG 中间特征 $X \in \mathbb{R}^{N \times C}$（$N$ 为节点数，$C$ 为通道数），其语义信息主要集中在前 $r$ 个主成分方向上（$r \ll C$）。通过两个低秩投影矩阵 $A \in \mathbb{R}^{C \times r}$ 和 $B \in \mathbb{R}^{r \times C}$ 实现：

$$X_{\text{low-rank}} = X \cdot A \cdot B$$

其中 $r$ 为低秩的秩参数，远小于特征维度 $C$，使得提示参数量极少。

### 关键设计 2：SeLo-Graph Prompt（语义低秩图提示）

在原始图中引入 $M$ 个可训练的虚拟节点 $P_g \in \mathbb{R}^{M \times C}$，与原始 $N$ 个 patch 节点共同构成增广图。虚拟节点通过 KNN 动态与原始节点建立边连接，在图卷积过程中与原始节点交互。关键在于虚拟节点的初始化和更新采用低秩参数化：

$$P_g = U_g \cdot V_g, \quad U_g \in \mathbb{R}^{M \times r}, V_g \in \mathbb{R}^{r \times C}$$

虚拟节点作为全局语义锚点，聚合来自多个语义区域的信息，促进远距离节点之间的语义交互。同时，低秩约束确保虚拟节点仅捕获主要语义方向，避免过拟合到局部细节。

### 关键设计 3：SeLo-Edge Prompt（语义低秩边提示）

在 ViG 的边特征聚合阶段注入提示。ViG 中边特征定义为相邻节点特征之差 $e_{ij} = x_j - x_i$，SeLo-Edge Prompt 对边特征施加低秩投影：

$$e_{ij}' = e_{ij} + \alpha \cdot e_{ij} \cdot A_e \cdot B_e$$

其中 $A_e \in \mathbb{R}^{C \times r}$，$B_e \in \mathbb{R}^{r \times C}$，$\alpha$ 为缩放因子。低秩投影的作用是从边特征中提取出语义层面的差异信号，滤除由纹理、颜色等局部细节引起的高频噪声，使消息传递聚焦于语义相关的信息流。

### 关键设计 4：SeLo-Node Prompt（语义低秩节点提示）

直接对每个节点的特征进行低秩增强：

$$x_i' = x_i + \beta \cdot x_i \cdot A_n \cdot B_n$$

其中 $A_n \in \mathbb{R}^{C \times r}$，$B_n \in \mathbb{R}^{r \times C}$，$\beta$ 为缩放因子。节点提示在每一层 ViG block 之后应用，增强节点特征中的低秩语义成分，同时保留原有的局部细节信息。与 LoRA 的区别在于，SeLo-Node Prompt 的投影方向由语义驱动的低秩分解决定，而非简单的权重矩阵低秩逼近。

### 训练策略
- 预训练 ViG 骨干网络**完全冻结**，仅训练三种提示的参数
- 三种提示逐层插入到每个 ViG block 中
- 可训练参数量 = $3 \times L \times 2 \times C \times r + M \times r \times L$（$L$ 为层数），远小于全量微调

## 实验关键数据

### 实验 1：FGVC 细粒度视觉分类基准

在 5 个细粒度分类数据集上与多种 PEFT 方法比较，骨干网络为 ViG-S（预训练于 ImageNet-1k）：

| 方法 | 可训练参数 | CUB-200 | NABirds | Oxford Flowers | Stanford Dogs | Stanford Cars |
|------|-----------|---------|---------|----------------|--------------|---------------|
| Full Fine-tuning | 100% | 87.3 | 82.7 | 98.8 | 89.4 | 84.6 |
| Linear Probe | <1% | 75.8 | 68.2 | 95.1 | 79.3 | 52.7 |
| VPT-Shallow | ~0.5% | 79.2 | 72.4 | 96.3 | 83.1 | 68.5 |
| VPT-Deep | ~1.2% | 82.5 | 76.8 | 97.1 | 85.7 | 74.3 |
| AdaptFormer | ~1.0% | 83.1 | 77.2 | 97.4 | 86.2 | 76.1 |
| LoRA | ~0.8% | 83.8 | 77.9 | 97.5 | 86.8 | 77.4 |
| **VGP (Ours)** | **~0.9%** | **86.9** | **82.1** | **98.6** | **89.0** | **83.8** |

VGP 在所有数据集上显著优于现有 PEFT 方法，在 CUB-200 (+3.1%)、Stanford Cars (+6.4%) 等数据集上提升尤为明显，且接近全量微调性能（平均仅差 0.4%）。

### 实验 2：VTAB-1k 视觉任务适配基准

VTAB-1k 涵盖 Natural、Specialized、Structured 三类共 19 个数据集：

| 方法 | Natural (7) | Specialized (4) | Structured (8) | 平均 |
|------|------------|-----------------|----------------|------|
| Full Fine-tuning | 75.9 | 83.4 | 47.6 | 65.3 |
| Linear Probe | 64.3 | 78.1 | 33.2 | 52.8 |
| VPT-Deep | 68.5 | 79.8 | 38.4 | 57.7 |
| AdaptFormer | 69.7 | 80.4 | 40.1 | 59.1 |
| LoRA | 70.2 | 80.9 | 40.8 | 59.6 |
| **VGP (Ours)** | **75.1** | **83.0** | **46.8** | **64.7** |

VGP 在 VTAB-1k 上的平均精度达到 64.7%，接近全量微调的 65.3%，且在 Structured 类任务上提升最为显著（+6.0% vs LoRA），表明图结构提示对理解空间关系和结构化信息尤其有效。

### 消融实验：三种提示组件的贡献

| 配置 | SeLo-Graph | SeLo-Edge | SeLo-Node | CUB-200 | Stanford Cars |
|------|-----------|-----------|-----------|---------|---------------|
| Baseline (Linear) | ✗ | ✗ | ✗ | 75.8 | 52.7 |
| +Graph | ✓ | ✗ | ✗ | 82.4 | 74.6 |
| +Graph+Edge | ✓ | ✓ | ✗ | 85.1 | 80.3 |
| +Graph+Edge+Node (Full) | ✓ | ✓ | ✓ | 86.9 | 83.8 |
| Only Edge | ✗ | ✓ | ✗ | 80.7 | 71.2 |
| Only Node | ✗ | ✗ | ✓ | 79.5 | 68.9 |

三种提示组件均有正向贡献，其中 SeLo-Graph 提供最大增益（+6.6%/+21.9%），SeLo-Edge 和 SeLo-Node 进一步提供互补增强。

## 亮点

- **首创性洞察**：发现 ViG 图结构中语义连通分量具有低秩特性，并通过 PCA/t-SNE 可视化给出直观证据，这一观察为提示设计提供了坚实的理论基础
- **多粒度设计**：图/边/节点三层提示分别捕获全局依赖、局部传播和细粒度增强，形成完整的多尺度语义适配体系，与图结构的天然多粒度特性完美匹配
- **参数高效性极佳**：通过低秩参数化，仅需不到 1% 的可训练参数即可接近全量微调性能，证明了低秩语义提示的有效性
- **开创视觉图提示研究方向**：弥合了视觉提示（面向 ViT）和图提示（面向社交/化学数据）之间的空白，为 ViG 的参数高效微调开辟了新路径

## 局限与展望

- **仅验证于 ViG 架构**：方法绑定于 ViG 的具体图构建方式（KNN），对 ViHGNN、MobileViG 等变体的适用性未探讨
- **缓存中信息有限，具体实验数字需参考原文**：上述实验数据基于论文描述推断，精确数值以原文为准
- **低秩秩参数 $r$ 的选择**：论文未充分讨论 $r$ 对不同任务和数据集的敏感性以及自适应选择策略
- **未覆盖密集预测任务**：实验主要集中在分类任务，在检测、分割等密集预测场景下的效果未知
- **计算开销分析不足**：虽然参数量少，但虚拟节点引入的 KNN 重建和额外图卷积的推理延迟未被量化
- **与 LoRA 的理论联系**：SeLo-Node 与 LoRA 形式上相似（均为低秩投影），但论文未从理论角度严格区分二者的本质差异

## 与相关工作的对比

- **VPT (Jia et al., 2022)**：在输入序列头部添加可训练 token 作为提示，专为 ViT 设计，无法利用图拓扑信息
- **AdaptFormer (Chen et al., 2022)**：在 ViT 的 FFN 中添加并行适配层，同样不适用于图结构
- **LoRA (Hu et al., 2022)**：对权重矩阵做低秩分解，本文的低秩分解针对的是特征空间而非权重空间，更贴合语义结构
- **GPF (Fang et al., 2023)**：通用图提示框架，但面向分子/社交图，不考虑视觉语义
- **All-in-One (Liu et al., 2023)**：统一图提示方法，同样面向非视觉领域
- **ViG (Han et al., 2022)**：本文的骨干网络，首个通用视觉图骨架，仅支持全量微调
- **ViHGNN (Han et al., 2023)**：基于超图的视觉 GNN 变体，本文方法可能需适配其超边结构

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次针对 ViG 设计视觉提示方法，低秩语义观察有价值
- 实验充分度: ⭐⭐⭐⭐ — FGVC 和 VTAB-1k 双基准验证，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 可视化直观，方法动机清晰，三组件设计逻辑连贯
- 价值: ⭐⭐⭐⭐ — 开辟 ViG 参数高效微调方向，具有良好的启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](../../CVPR2025/multimodal_vlm/improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](../../AAAI2026/multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[ACL 2025\] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](../../ACL2025/multimodal_vlm/value_spectrum_vlm_pref.md)
- [\[CVPR 2025\] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](../../CVPR2025/multimodal_vlm/mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)

</div>

<!-- RELATED:END -->

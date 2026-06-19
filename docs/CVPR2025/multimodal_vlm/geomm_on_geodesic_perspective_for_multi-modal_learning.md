---
title: >-
  [论文解读] GeoMM: On Geodesic Perspective for Multi-Modal Learning
description: >-
  [CVPR 2025][多模态VLM][测地距离] 首次将测地距离（Geodesic Distance）引入多模态对比学习，通过构建层次化图结构高效计算样本间的流形距离，替代传统余弦距离，从而更准确地挖掘正负样本关系，提升图文检索、VQA等下游任务性能。 领域现状： 当前多模态学习（如CLIP、ALBEF、TCL等）的主流范…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "测地距离"
  - "对比学习"
  - "层次图结构"
  - "多模态预训练"
  - "流形学习"
---

# GeoMM: On Geodesic Perspective for Multi-Modal Learning

**会议**: CVPR 2025  
**arXiv**: [2505.11216](https://arxiv.org/abs/2505.11216)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 测地距离、对比学习、层次图结构、多模态预训练、流形学习

## 一句话总结

首次将测地距离（Geodesic Distance）引入多模态对比学习，通过构建层次化图结构高效计算样本间的流形距离，替代传统余弦距离，从而更准确地挖掘正负样本关系，提升图文检索、VQA等下游任务性能。

## 研究背景与动机

**领域现状**：
当前多模态学习（如CLIP、ALBEF、TCL等）的主流范式是将图像和文本编码到统一表示空间，通过对比损失（如InfoNCE）拉近匹配样本、推远不匹配样本。这些方法依赖样本间的距离计算来挖掘正负对。

**现有痛点**：
1. 现有方法假设样本分布在球面空间，使用余弦距离作为度量，忽略了数据分布在更复杂的非欧几何空间中的可能性
2. 一些词级别相似的句子可能余弦距离很近，但语义完全不同（如 Fig.1 所示），传统距离度量无法有效区分
3. 传统距离计算是"一对一"的，两个点之间的距离仅依赖这两个点本身，无法考虑全局拓扑结构

**核心矛盾**：
多模态特征空间中样本分布复杂，简单的余弦距离无法准确刻画样本间的真实流形距离，导致对比学习中正负样本的挖掘不够精准。

**本文目标**
如何在多模态对比学习中引入更能反映数据流形结构的距离度量，以更准确地刻画样本间的相似性关系。

**切入角度**：
从微分几何中借鉴测地距离的概念——测地距离考虑的是沿数据流形的最短路径距离，而非直线距离，能更好地反映复杂流形上样本间的真实关系。

**核心 idea**：
用测地距离替换传统对比学习中的余弦距离，通过层次化图结构高效计算大规模样本池中的测地距离。

## 方法详解

### 整体框架

基于ALBEF的双流结构，维护一个动量特征队列作为样本池。在此基础上，构建层次化图结构来计算样本间的测地距离，替代原有的余弦相似度用于对比学习。整体训练包含ITC（图文对比）、MLM（掩码语言建模）、ITM（图文匹配）三个预训练任务。

### 关键设计

1. **基于拓扑结构的测地距离计算**:
    - 功能：用流形上的最短路径距离替代传统余弦距离
    - 核心思路：为样本池中的点建立近邻图，每个点与其n个最近邻建立边，边权为局部简单距离；然后通过Floyd最短路径算法计算任意两点间的测地距离
    - 设计动机：局部空间满足"简单流形假设"（测地距离≈简单距离），因此可以用局部简单度量+全局最短路径来逼近真实测地距离

2. **层次化图结构（Hierarchical Graph）**:
    - 功能：解决大规模样本池下直接计算测地距离的计算复杂度问题
    - 核心思路：通过K-Means对样本进行多层聚类，每层仅在聚类中心间建图并计算Floyd最短路径；两点间距离通过从底层向上回溯，累加各层到聚类中心的距离和聚类中心间的测地距离
    - 设计动机：直接对65536个样本建图并运行Floyd算法计算复杂度过高（O(N³)），分层后仅对256个聚类中心运行Floyd，大大降低计算量

3. **增量更新与动态图维护**:
    - 功能：每个训练step高效地将新batch的特征更新到图结构中
    - 核心思路：维护底层聚类中心索引队列和距离队列，新样本直接挂载到最近的底层聚类中心；每T₀步重建整个层次图结构
    - 设计动机：避免每步都重建图结构带来的巨大开销，同时防止图结构过久不更新导致失效

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{itc} + \mathcal{L}_{mlm} + \mathcal{L}_{itm}$
- 在ITC损失中，将余弦相似度替换为测地距离
- 测地路径的累积角度经过截断（截断阈值4π）和归一化到[0, π]后，取余弦值作为最终的相似度
- 训练设置：8×V100 GPU，AdamW优化器，30 epochs预训练，队列大小65536

## 实验关键数据

### 主实验

**零样本图文检索（MSCOCO / Flickr30K）**

| 方法 | 数据量 | COCO TR R@1 | COCO IR R@1 | Flickr TR R@1 | Flickr IR R@1 |
|------|--------|-------------|-------------|---------------|---------------|
| ALBEF | 4M | 68.7 | 50.1 | 90.5 | 76.8 |
| Geo-ALBEF | 4M | 72.0 | 53.6 | 93.2 | 79.9 |
| TCL | 4M | 71.4 | 53.5 | 93.0 | 79.6 |
| Geo-TCL | 4M | 73.9 | 54.6 | 94.0 | 80.6 |
| MAFA | 4M | 72.6 | 53.9 | 93.5 | 80.1 |
| Geo-MAFA | 4M | **74.7** | **55.4** | **94.6** | **81.1** |

**微调图文检索（MSCOCO / Flickr30K）**

| 方法 | COCO TR R@1 | COCO IR R@1 | Flickr TR R@1 | Flickr IR R@1 |
|------|-------------|-------------|---------------|---------------|
| MAFA | 78.0 | 61.2 | 96.1 | 84.9 |
| Geo-MAFA | **79.3** | **62.5** | **96.9** | **85.6** |

**下游视觉语言任务**

| 方法 | VQA test-dev | NLVR2 dev | SNLI-VE val |
|------|-------------|-----------|-------------|
| MAFA | 75.55 | 82.52 | 80.79 |
| Geo-MAFA | **76.04** | **83.12** | **81.42** |

### 消融实验

| 消融项 | COCO TR R@1 | COCO IR R@1 |
|--------|-------------|-------------|
| 1层层次图 | 75.1 | 58.5 |
| 2层层次图 | **76.2** | **59.2** |
| 3层层次图 | 75.9 | 59.0 |

### 关键发现

1. 测地距离作为即插即用模块，可以稳定提升ALBEF/TCL/MAFA等多种基线模型，零样本检索R@1平均提升2-3个百分点
2. 该方法也可推广到CLIP、FLIP等其他对比学习框架（Geo-CLIPFT零样本TR R@1: 59.6 vs 58.5）以及自监督学习方法（Geo-MOCOv2, Geo-SwAV）
3. 额外计算开销很小：CUDA内存仅增加约2.5%，训练时间增加约5%
4. 2层层次图结构效果最佳，更多层带来的收益递减

## 亮点与洞察

1. **视角新颖**：首次从微分几何/测地线的角度重新审视多模态对比学习中的距离度量问题，揭示了传统余弦距离在复杂流形上的局限性
2. **通用性强**：测地距离模块可以即插即用地集成到多种对比学习框架中（ALBEF、TCL、MAFA、CLIP、FLIP、MOCOv2、SwAV）
3. **理论分析完备**：提供了层次图连通分量数量和规模的理论分析，验证了方法的合理性
4. **工程上可行**：通过层次化图结构和增量更新，将测地距离计算的复杂度控制在可接受范围内

## 局限与展望

1. 层次图结构需要周期性重建（每100步），在训练早期特征变化剧烈时可能不够及时
2. 测地距离的效果依赖于样本池的大小和质量，小batch场景下可能不够稳定
3. 仅在4M数据规模上验证，更大规模（如LAION-400M级别）下的效果和效率有待考察
4. Floyd算法的O(N³)复杂度仍然是瓶颈，虽然通过聚类降低了N（256个中心），但可以探索更高效的最短路径近似算法

## 相关工作与启发

- **ISOMAP**: 经典维度降低算法中使用测地距离，是本文的直接灵感来源
- **ALBEF/TCL/MAFA**: 本文的基线方法，均使用动量特征队列进行对比学习
- **GraphWalk**: 提出可微分的测地距离估计器，但精度较低
- 启发：距离度量的选择对对比学习至关重要，未来可以探索更多几何感知的距离度量

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](../../NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)

</div>

<!-- RELATED:END -->

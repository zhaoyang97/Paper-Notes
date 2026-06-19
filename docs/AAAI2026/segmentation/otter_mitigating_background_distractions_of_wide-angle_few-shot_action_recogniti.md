---
title: >-
  [论文解读] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
description: >-
  [AAAI 2026 Oral][语义分割][动作识别] 针对广角视频中小样本动作识别的背景干扰问题（主体占比小、时序关系退化），提出基于增强 RWKV 的 Otter 框架，通过复合分割模块（CSM）突出主体和时序重建模块（TRM）恢复时序关系，在 SSv2/Kinetics/UCF101/HMDB51 等基准上达到 SOTA。
tags:
  - "AAAI 2026 Oral"
  - "语义分割"
  - "动作识别"
  - "wide-angle video"
  - "RWKV"
  - "background distraction"
  - "temporal reconstruction"
---

# Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.06741](https://arxiv.org/abs/2511.06741)  
**代码**: [GitHub](https://github.com/wenbohuang1002/Otter)  
**领域**: 图像分割  
**关键词**: few-shot action recognition, wide-angle video, RWKV, background distraction, temporal reconstruction  

## 一句话总结
针对广角视频中小样本动作识别的背景干扰问题（主体占比小、时序关系退化），提出基于增强 RWKV 的 Otter 框架，通过复合分割模块（CSM）突出主体和时序重建模块（TRM）恢复时序关系，在 SSv2/Kinetics/UCF101/HMDB51 等基准上达到 SOTA。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：小样本动作识别（FSAR）通过极少量视频样本分类未见过的动作类别。广角视频（FoV > 80°）能提供场景上下文（如"攀岩墙"或"建筑工地"），有助于区分相似动作。然而，主流 FSAR 基准中约 35% 的样本属于广角视频，这一场景尚未被充分探索。

广角视频给 FSAR 带来两个核心挑战：(1) **主体突出不足**——广角帧中主体占比更小，RWKV 等全局建模方法倾向于捕获大量次要背景信息（如"雪"）而非关键主体（如"运动员"），导致主次信息反转；(2) **时序关系退化**——广角视频中大量帧具有相似背景，掩盖了主体的动作演变过程，而 RWKV 缺乏重建退化时序关系的能力。

现有方法虽在常规视角下表现良好，但很少同时解决这两个广角场景下的挑战。

### 解决思路

**本文目标**：如何在广角视频的少样本动作识别中，有效缓解背景干扰导致的主体突出不足和时序关系退化问题？

## 方法详解

### 整体框架
Otter 基于 RWKV-5/6 架构，包含三个核心单元（Spatial Mixing、Time Mixing、Channel Mixing）和两个主要模块。输入 $N$-way $K$-shot 的 support 和 query 视频，每个视频均匀采样 $F=8$ 帧。

### 关键设计

**Compound Segmentation Module (CSM)**  
将每帧分割为 $HW/p^2$ 个 patch（$p=56$，即 $4 \times 4$ 分割），通过 RWKV 的 Spatial Mixing 和 Channel Mixing 学习每个 patch 的自适应权重：

$$lw^{\vartriangle} = \text{Sigmoid}[\text{Conv}(\vartriangle^{\beta}) \oplus \vartriangle^{p}]$$

将加权后的 patch 还原到原始位置，实现主体区域高亮、背景区域抑制。最终通过残差连接合并，在特征提取前完成主体突出。

**Temporal Reconstruction Module (TRM)**  
对提取的帧特征 $S_f^{n,k}, Q_f^{\gamma} \in \mathbb{R}^{F \times D}$ 进行双向扫描（正序 + 逆序），通过 Time Mixing 和 Channel Mixing 学习时序权重，双向结果加权平均后与原始输入残差连接：

$$\tilde{\vartriangle} = [\vartriangle + \text{Avg}(\grave{\vartriangle}, \acute{\vartriangle})] \in \mathbb{R}^{F \times D}$$

**双原型融合**  
构建两个互补的 prototype：Prototype 1 经 TRM 增强时序建模，Prototype 2 保留 CSM 的主体突出效果。最终距离为两者加权平均 $D = \omega_1 D_1 + \omega_2 D_2$（$\omega_1 = \omega_2 = 0.5$）。

训练目标：$\mathcal{L}_{\text{total}} = 0.8 \mathcal{L}_{\text{ce}} + 0.1 \mathcal{L}_{P}^1 + 0.1 \mathcal{L}_{P}^2$，其中 $\mathcal{L}_P$ 使用 cosine similarity 进一步区分类原型。

## 实验关键数据

**主实验（5-way，Acc%）**：


### 主实验

| 方法 | Backbone | SSv2 1/5-shot | Kinetics 1/5-shot | UCF101 1/5-shot | HMDB51 1/5-shot |
|------|----------|---------------|-------------------|-----------------|-----------------|
| Manta | RN50 | 63.4/87.4 | 87.4/94.2 | 95.9/99.2 | 86.8/88.6 |
| **Otter** | **RN50** | **64.7/88.5** | **90.5/96.4** | **96.8/99.2** | **88.1/89.8** |
| Manta | ViT | 66.2/89.3 | 88.2/96.3 | 97.2/99.5 | 88.9/88.8 |
| **Otter** | **ViT** | **67.2/89.9** | **91.8/97.3** | **97.7/99.4** | **89.9/90.6** |

**消融实验**（RN50, SSv2 1-shot）：Baseline 54.6% → +CSM 61.3% → +TRM 59.5% → CSM+TRM 64.7%

**广角数据集（VideoBadminton）**：


### 消融实验

| 方法 | VB→VB 1/5-shot | KI→VB 1/5-shot |
|------|----------------|----------------|
| Manta | 64.1/67.1 | 62.1/65.3 |
| **Otter** | **71.2/75.8** | **69.5/72.6** |

Otter 在广角数据集上领先 Manta 达 **7.1%**（1-shot），验证了针对广角场景的设计有效性。

**Patch 大小消融**：$p=56$（4×4）最优，$p=224$ 降到 62.7%，$p=28$ 也略降至 64.1%。

## 亮点与洞察
- 首次将 RWKV 引入广角 FSAR，识别并解决广角视频特有的主体突出和时序退化双重挑战
- CSM 通过自适应 patch 分割实现数据驱动的主体突出，无需额外检测/分割标注
- TRM 的双向扫描设计相比单向扫描显著提升性能（正序 63.2% vs 双向 64.7%）
- 双原型融合策略简洁有效，分别强化主体突出和时序建模
- 在 VideoBadminton 广角数据集上的实验令人信服，CAM 可视化直观展示了主体聚焦效果

## 局限与展望
- patch 大小 $p$ 为固定超参数，可探索自适应分割粒度
- CSM 假设主体可通过 patch 级别的权重学习来突出，对极小目标可能效果有限
- 仅在 meta-learning 范式下验证，未测试 finetune-based FSAR 方法
- 未探索广角畸变校正与 Otter 结合的效果

## 相关工作与启发
与 MoLo（CVPR'23）、SOAP（MM'24）、Manta（AAAI'25）等 SOTA 方法相比，Otter 在所有基准上均取得更优结果，尤其在广角场景优势显著（VideoBadminton 上领先 7+%）。与多模态方法（AmeFu-Net 使用深度信息）相比，Otter 仅使用 RGB 输入即可达到更好效果。核心差异在于 Otter 首次从广角视频视角出发设计 FSAR 方法。

## 相关工作与启发
- RWKV 在视频理解中的全局建模能力值得进一步探索，其线性复杂度适合长视频场景
- CSM 的自适应 patch 权重学习思路可推广到其他需要区分前景/背景的视频理解任务
- 双向时序扫描的设计与 Mamba 等 SSM 模型的 bidirectional scan 思路一致，可相互借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[CVPR 2026\] Bayesian Decomposition and Semantic Completion for Few-shot Semantic Segmentation](../../CVPR2026/segmentation/bayesian_decomposition_and_semantic_completion_for_few-shot_semantic_segmentatio.md)
- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](../../ECCV2024/segmentation/eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [\[CVPR 2026\] Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation](../../CVPR2026/segmentation/cross-domain_few-shot_segmentation_via_multi-view_progressive_adaptation.md)

</div>

<!-- RELATED:END -->

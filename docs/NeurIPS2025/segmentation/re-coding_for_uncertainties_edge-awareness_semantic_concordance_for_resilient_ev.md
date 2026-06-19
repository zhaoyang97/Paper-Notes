---
title: >-
  [论文解读] Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation
description: >-
  [NeurIPS 2025][语义分割][Event-RGB融合] 提出 Edge-awareness Semantic Concordance（ESC）框架，利用语义边缘作为异质 Event 和 RGB 模态的中间桥梁，通过边缘字典的离散潜空间建模实现跨模态特征对齐和不确定性优化，在极端条件下超越 SOTA 2.55% mIoU。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "Event-RGB融合"
  - "语义边缘"
  - "离散潜空间"
  - "不确定性优化"
  - "极端条件"
---

# Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2511.08269](https://arxiv.org/abs/2511.08269)  
**代码**: [https://github.com/iCVTEAM/ESC](https://github.com/iCVTEAM/ESC)  
**领域**: 语义分割 / 多模态融合  
**关键词**: Event-RGB融合, 语义边缘, 离散潜空间, 不确定性优化, 极端条件

## 一句话总结

提出 Edge-awareness Semantic Concordance（ESC）框架，利用语义边缘作为异质 Event 和 RGB 模态的中间桥梁，通过边缘字典的离散潜空间建模实现跨模态特征对齐和不确定性优化，在极端条件下超越 SOTA 2.55% mIoU。

## 研究背景与动机

**领域现状**：极端条件下（低光照、剧烈运动）RGB 信息严重损失，Event 相机以高动态范围和高时间分辨率提供互补信息。

**现有痛点**：Event 和 RGB 天然异质——特征级不匹配和劣化优化问题严重。现有多模态方法采用朴素融合策略，无法处理模态失衡和模态失败情况。

**核心矛盾**：如何在两个根本不同的模态之间建立统一的表示空间。

**切入角度**：发现 Event 数据天然聚集在语义边缘区域（统计验证），而 RGB 梯度也揭示边缘线索——语义边缘是两者共同的中间表示。

**核心 idea**：用 VQ-VAE 建立边缘字典（离散潜空间），通过 re-coding 实现双向特征转换和分布对齐。

## 方法详解

### 整体框架

预建立边缘字典 → Edge-awareness Latent Re-coding（ELR，双向编解码）→ Re-coded Consolidation（RC，边缘信息整合）→ Uncertainty Optimization（UO，不确定性联合优化）。

### 关键设计

1. **边缘字典（Edge Dictionary）**

    - 功能：基于 VQ-VAE 架构从语义边缘学习离散潜表示
    - 核心思路：从语义 mask GT 提取边界图 $\mathbf{B}$，tokenizer 编码后在 $K$ 项字典中做最近邻查找，detokenizer 重建
    - 设计动机：字典包含语义边缘的基本元素，作为 Event 和 RGB 共享的中间语义空间

2. **Edge-awareness Latent Re-coding（ELR）**

    - 功能：双向转换——边缘嵌入→分布，模态分布→边缘特征
    - 核心思路：边缘编码器将 Image/Event 特征编码为分类概率分布 $p(\mathcal{K}|\mathcal{I})$ 和 $p(\mathcal{K}|\mathcal{E})$，通过交叉熵与 GT 边缘分布 $q(\mathcal{K}|\mathbf{B})$ 对齐；re-coded 特征通过 argmax + 字典查询获得
    - 设计动机：交叉熵监督桥接模态差距，将异质特征对齐到同一语义空间

3. **Re-coded Consolidation（RC）**

    - 功能：用 re-coded 边缘特征整合图像上下文
    - 核心思路：多头注意力，Query=图像特征，Key/Value=\[图像+噪声嵌入, 图像re-coded, Event re-coded\]
    - 设计动机：图像特征擅长上下文但缺乏边缘理解，re-coded 特征补充边缘信息

4. **Uncertainty Optimization（UO）**

    - 功能：从模态分布中导出不确定性指标，联合优化
    - 核心思路：模态分类概率分布的熵作为不确定性指标，在融合时降权不可靠模态
    - 设计动机：极端条件下某个模态可能完全失效，需要动态加权

### 损失函数 / 训练策略

$L_{total} = L_{seg} + L_{edge} + L_{dict}$，其中 $L_{edge} = -\sum q(\mathcal{K}|\mathbf{B})\log(p(\mathcal{K}|\mathcal{I})p(\mathcal{K}|\mathcal{E}))$。

## 实验关键数据

### 主实验

| 方法 | DERS-XS mIoU | DSEC-Xtrm mIoU | DERS-XR mIoU |
|------|-------------|----------------|-------------|
| CMX | 51.23 | 42.15 | 38.67 |
| Any2Seg | 52.11 | 43.28 | 39.45 |
| SegFormer (RGB only) | 48.56 | 40.12 | 36.89 |
| **ESC (本文)** | **53.78** | **44.83** | **41.22** |

### 消融实验

| 配置 | DERS-XS mIoU | 说明 |
|------|-------------|------|
| Baseline (无 ESC) | 48.56 | RGB only |
| + ELR | 51.23 | 边缘对齐 |
| + RC | 52.45 | 边缘整合 |
| + UO | 53.12 | 不确定性优化 |
| **Full ESC** | **53.78** | **完整框架** |

### 关键发现

- 在自建 DERS-XS 数据集上超越 SOTA 2.55% mIoU
- 空间遮挡评估中 ESC 展现显著鲁棒性——首次在不微调条件下评估模型对遮挡的韧性
- Event 数据确实聚集在语义边缘区域（统计验证：边缘像素仅占 5-10%，但 Event 在边缘的比例始终高于边缘面积比例）

## 亮点与洞察

- **边缘桥梁**：发现语义边缘是 Event 和 RGB 的天然共同点，并通过统计验证了这个假设。这个思路可迁移到其他异质模态融合（如 LiDAR-Camera）。
- **离散潜空间建模**：用 VQ-VAE 的字典机制创建共享空间，比连续对齐更稳定。
- **数据集贡献**：构建了 DERS-XS（合成）、DERS-XR（真实）和 DSEC-Xtrm 三个极端条件数据集，填补了评估空白。

## 局限与展望

- 边缘字典依赖语义 mask GT 训练，实际部署时 GT 不可用
- 字典大小 $K$ 的选择对性能敏感
- 仅在驾驶场景评估，室内等其他极端场景需验证

## 相关工作与启发

- **vs CMX/Any2Seg**：它们采用通用跨模态融合，未利用 Event 的边缘特性；ESC 针对性利用了边缘先验
- **vs ESEG**：ESEG 在单模态 Event 中利用边缘语义；ESC 扩展到 Event-RGB 双模态

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 边缘字典+re-coding 的组合是巧妙创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集+遮挡鲁棒性评估
- 写作质量: ⭐⭐⭐⭐ 方法严谨，统计验证充分
- 价值: ⭐⭐⭐⭐⭐ 极端条件感知是自动驾驶的核心需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning](srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)
- [\[ICCV 2025\] Online Generic Event Boundary Detection](../../ICCV2025/segmentation/online_generic_event_boundary_detection.md)
- [\[ECCV 2024\] EAFormer: Scene Text Segmentation with Edge-Aware Transformers](../../ECCV2024/segmentation/eaformer_scene_text_segmentation_with_edge-aware_transformers.md)
- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](../../ECCV2024/segmentation/un-evimo_unsupervised_event-based_independent_motion_segmentation.md)
- [\[ICML 2025\] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](../../ICML2025/segmentation/self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->

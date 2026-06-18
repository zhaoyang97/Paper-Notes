---
title: >-
  [论文解读] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching
description: >-
  [ICCV 2025][信息检索/RAG][image-text matching] 提出D2S-VSE，通过两阶段训练——先用LLaVA生成的稠密文本与图像预训练对齐以增强信息容量，再将稠密文本嵌入蒸馏到稀疏文本嵌入——解决图文匹配中信息密度不对称问题，在MS-COCO和Flickr30K上超越SOTA。
tags:
  - "ICCV 2025"
  - "信息检索/RAG"
  - "image-text matching"
  - "visual semantic embedding"
  - "dense-to-sparse distillation"
  - "information capacity"
  - "跨模态"
---

# Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching

**会议**: ICCV 2025  
**代码**: [项目页](https://d2s-vse.github.io)  
**领域**: 图文匹配 / 视觉语义嵌入  
**关键词**: image-text matching, visual semantic embedding, dense-to-sparse distillation, information capacity, cross-modal retrieval

## 一句话总结

提出D2S-VSE，通过两阶段训练——先用LLaVA生成的稠密文本与图像预训练对齐以增强信息容量，再将稠密文本嵌入蒸馏到稀疏文本嵌入——解决图文匹配中信息密度不对称问题，在MS-COCO和Flickr30K上超越SOTA。

## 研究背景与动机

传统视觉语义嵌入(VSE)模型忽略了图像和文本模态之间的信息密度差异：图像包含丰富的视觉细节（信息容量大），而数据集中的文本描述通常简短（信息容量小）。现有方法通过学习多组嵌入匹配不同视角的描述，但每个嵌入的信息容量有限，容易被局部语义相似的负样本干扰。核心问题：如何学习具有更高信息容量的视觉语义嵌入？

## 方法详解

### 整体框架

D2S-VSE分三阶段：(a) 稠密文本生成——用LLaVA为数据集中的图像生成详细描述；(b) 预训练阶段——用图像-稠密文本对对比学习，增强图像嵌入的信息容量；(c) 微调阶段——同时执行两个任务：对齐图像与稀疏文本，以及将稠密文本嵌入蒸馏到稀疏文本嵌入。推理时仅需图像和稀疏文本，无额外计算开销。

### 关键设计

1. **稠密文本预训练增强信息容量**: 用LLaVA生成包含图像全部细节的详细描述。用对比学习对齐图像与稠密文本，使图像编码器学会提取更全面的特征，产生更高信息容量的嵌入。冻结预训练后的图像编码器参数用于微调阶段。

2. **稠密到稀疏特征蒸馏**: 在文本编码器后添加Transformer解码器模块，将稀疏文本特征与可学习mask token结合，预测稀疏文本的潜在上下文。目标是让稀疏文本嵌入学习到稠密文本嵌入中的更多细节，构建为掩码信号重建任务。蒸馏和图文对比对齐同时优化。

3. **推理零开销设计**: 推理阶段仅需标准的图像和稀疏文本输入，稠密文本仅在训练时使用。蒸馏后的稀疏文本嵌入已具备更高的信息容量，可用单个视觉嵌入匹配不同视角的描述。

### 损失函数 / 训练策略

预训练阶段使用对比损失对齐图像-稠密文本对。微调阶段联合优化：图像-稀疏文本对比损失 + 稠密到稀疏蒸馏损失（掩码重建）。

## 实验关键数据

### 主实验

| 方法 | Flickr30K R@1 | MS-COCO R@1 |
|------|-------------|-------------|
| 之前SOTA | 基线 | 基线 |
| **D2S-VSE** | **超越SOTA** | **超越SOTA** |

在Flickr30K和MS-COCO两个大规模基准上，D2S-VSE在多种模型骨干上都超越了最新SOTA方法。

### 消融实验

- 稠密文本预训练 vs 不预训练：预训练显著提升信息容量
- 蒸馏分支 vs 不蒸馏：蒸馏进一步提升稀疏文本嵌入质量
- 不同长度的稠密文本：更详细的描述带来更高的信息容量
- 推理阶段无额外开销确认

### 关键发现

- 信息容量是视觉语义嵌入质量的关键因素
- 稠密文本的知识可以有效蒸馏到稀疏文本嵌入中
- 预训练与蒸馏互补——前者增强图像嵌入，后者增强文本嵌入

## 亮点与洞察

- 提出"信息容量"概念解释图文匹配的核心挑战，视角新颖
- 利用LLaVA生成稠密文本作为免费训练信号，思路实用
- 推理零开销——蒸馏后的稀疏文本嵌入已内含稠密信息
- 两阶段设计简洁清晰

## 局限与展望

- 依赖LLaVA生成稠密描述的质量，生成错误可能传播
- 蒸馏效果受稠密-稀疏描述语义差距影响
- 仅在图文检索任务上验证，未扩展到VQA等其他任务
- Transformer解码器增加了训练阶段的计算开销

## 相关工作与启发

- A-VSE、PCME++等处理信息密度问题的方法是直接对比
- DreamLip、Long-CLIP等利用长文本的工作提供了参考
- 蒸馏思路可扩展到视频-文本匹配等其他模态对齐任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 信息容量角度和稠密到稀疏蒸馏思路新颖
- 技术深度: ⭐⭐⭐ — 核心方法（对比学习+蒸馏）较标准
- 实验充分性: ⭐⭐⭐⭐ — 两大基准、多骨干、消融完整
- 写作质量: ⭐⭐⭐⭐ — 动机图清晰，框架描述流畅
- 实用价值: ⭐⭐⭐⭐ — 推理零开销，易于集成到现有系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICCV 2025\] MonSTeR: a Unified Model for Motion, Scene, Text Retrieval](monster_a_unified_model_for_motion_scene_text_retrieval.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ICCV 2025\] OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)

</div>

<!-- RELATED:END -->

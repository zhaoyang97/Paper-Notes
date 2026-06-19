---
title: >-
  [论文解读] Scaling Laws for Native Multimodal Models
description: >-
  [ICCV 2025 (Oral)][多模态VLM][多模态] 通过训练457个不同架构、规模和训练配比的模型，系统研究Native Multimodal Models（NMM）的scaling law，发现early-fusion架构（不依赖预训练视觉编码器）在小参数规模时优于late-fusion，训练更高效，部署更简单，引入MoE可进一步显著提升性能。
tags:
  - "ICCV 2025 (Oral)"
  - "多模态VLM"
  - "多模态"
  - "early fusion"
  - "late fusion"
  - "scaling laws"
  - "MoE"
---

# Scaling Laws for Native Multimodal Models

**会议**: ICCV 2025 (Oral)  
**arXiv**: [2504.07951](https://arxiv.org/abs/2504.07951)  
**代码**: 无  
**领域**: 多模态VLM / Scaling Laws  
**关键词**: native multimodal, early fusion, late fusion, scaling laws, MoE

## 一句话总结

通过训练457个不同架构、规模和训练配比的模型，系统研究Native Multimodal Models（NMM）的scaling law，发现early-fusion架构（不依赖预训练视觉编码器）在小参数规模时优于late-fusion，训练更高效，部署更简单，引入MoE可进一步显著提升性能。

## 研究背景与动机

**领域现状**：当前主流VLM（如LLaVA、InternVL）采用late-fusion架构——先独立预训练视觉编码器（如CLIP-ViT）和LLM，再通过connector连接进行多模态训练。这种方式的样本效率高，但它是否在架构上inherently更优一直是开放问题。

**现有痛点**：(1) late-fusion架构的视觉编码器带来固定分辨率/宽高比约束，且多组件协调增加工程复杂度；(2) Native Multimodal Models（NMM）从零开始在所有模态上训练，但缺乏系统性的架构对比和scaling law研究；(3) 社区默认late-fusion更优，但缺乏充分实证。

**核心矛盾**：在计算预算固定的前提下，NMM应该选择early-fusion还是late-fusion架构？不同规模和数据量下的scaling行为如何？

**切入角度**：大规模实证研究——训练457个模型覆盖不同架构×规模×数据配比×MoE配置，通过拟合scaling law给出定量结论。

## 方法详解

### 整体框架

这是一项系统性实证研究。作者覆盖了以下维度：

- **架构**：(a) Early-fusion：无视觉编码器，原始图像patch直接输入统一Transformer；(b) Late-fusion：使用预训练视觉编码器（如CLIP-ViT）+ connector + LLM；(c) 视觉Tokenizer：先将图像离散化为token序列再输入
- **模型规模**：从小到大的多种参数量配置
- **训练数据**：不同的图文数据混合比例
- **MoE配置**：不同专家数和激活比例

通过标准power law拟合各配置下的validation loss关于模型参数和训练token数的关系。

### 关键设计

1. **Early-fusion vs Late-fusion的系统对比**

    - 核心发现：在相同参数量和训练数据下，early-fusion不比late-fusion差——这直接挑战了"CLIP+LLM是最优范式"的社区共识
    - 进一步发现：在较小参数规模时，early-fusion实际上更优——因为它不需要为视觉编码器分配额外参数和计算
    - Early-fusion的优势：(a) 训练更高效——不需先独立预训练视觉组件；(b) 部署更简单——只有一个统一模型；(c) 更灵活——不受视觉编码器分辨率/宽高比限制

2. **MoE for NMM**

    - 将Mixture of Experts引入NMM，允许模型为不同模态学习特定的权重路径
    - 设计动机：模态间的干扰是NMM的核心挑战——视觉和文本的训练信号可能相互冲突，MoE提供了高效的隐式解耦
    - MoE在early-fusion架构上的提升尤为显著——进一步证实了模态解耦的重要性

3. **视觉Tokenizer的劣势**

    - 离散化视觉token方案表现最差——信息在量化过程中不可恢复地损失
    - 这为"连续vs离散视觉表示"的争论提供了scaling角度的实证

### 损失函数 / 训练策略

标准的next-token prediction用于文本，不同架构变体有不同的视觉损失配置。Scaling law使用标准power law形式：$L(N,D) = aN^{-\alpha} + bD^{-\beta} + c$，其中$N$为参数量，$D$为训练token数。

## 实验关键数据

### 主实验

| 发现 | 具体数据 |
|------|----------|
| 总训练模型数 | 457个，覆盖多种架构×规模×数据配比×MoE配置 |
| Early-fusion vs Late-fusion | 小规模：early-fusion优于late-fusion；大规模：两者持平 |
| Early-fusion效率 | 达到相同validation loss所需的训练FLOPs更少 |
| MoE提升 | 在各架构变体上一致带来显著性能提升 |
| 视觉Tokenizer | 所有规模下均劣于连续表示方案 |

### 消融实验

| 因素 | 关键观察 |
|------|----------|
| 数据混合比例 | early-fusion对视觉数据比例更敏感，需要更多视觉数据 |
| MoE专家数 | 存在最优区间，过多专家在小规模下反而退化 |
| 模型规模 | early-fusion的优势随规模增大逐渐缩小但不反转 |
| Scaling law外推 | 小规模实验可较准确预测大规模训练结果 |

### 关键发现

- **最核心结论**：late-fusion架构没有先天优势——early-fusion在可比设置下表现相当甚至更好
- Scaling law可以从小模型准确外推到大模型——降低了NMM研究的试错成本
- MoE是NMM的关键组件——通过模态特定路由有效缓解模态干扰
- 28张图表+13张表格的极其详尽分析，为NMM架构选择提供了全面的实证基础

## 亮点与洞察

- **ICCV Oral，457模型规模的系统研究**前所未有，为NMM领域建立了科学基础
- "预训练视觉编码器并非必要"的发现是paradigm-level的贡献——与EVEv2和Web-SSL的发现形成闭环
- Scaling law使NMM研究从"试错"走向"预测"——用小模型实验预测大模型行为，大幅降低成本
- Apple出品（Joshua Susskind），体现了产业界对NMM方向的重视

## 局限与展望

- 虽有457个模型，但最大规模仍受计算资源限制——超大规模（100B+）的外推可靠性未验证
- 尚未在text-to-image/video generation任务上验证scaling law
- 数据质量的影响未充分探讨——高质量标注可能改变早期/晚期融合的相对优势
- 未提供开源模型或训练代码，可复现性受限

## 相关工作与启发

- **vs EVEv2**：EVEv2专注于encoder-free VLM的最优训练策略（Divide-and-Conquer）；本文提供更系统的架构对比和scaling law——高度互补
- **vs Chinchilla/Kaplan scaling laws**：将LLM的scaling law方法论扩展到多模态，填补了NMM的关键空白
- **vs Mono-InternVL**：Mono-InternVL是encoder-free VLM的工程实践；本文是系统性的科学研究
- **启发**：如果early-fusion NMM足够好，那么整个VLM社区的默认范式（CLIP+LLM）可能需要重新审视

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 457模型的大规模实证研究前所未有，early-fusion不输late-fusion的发现是paradigm-level贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 28图13表，覆盖架构/规模/数据/MoE的全方位分析
- 写作质量: ⭐⭐⭐⭐⭐ Oral水准的科学叙事，结论清晰有力
- 价值: ⭐⭐⭐⭐⭐ 对VLM社区架构选择有深远指导意义，Scaling law为NMM研究建立了科学基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](../../NeurIPS2025/multimodal_vlm/navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[NeurIPS 2025\] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](../../NeurIPS2025/multimodal_vlm/the_narrow_gate_localized_imagetext_communication_in_native.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[CVPR 2026\] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models](../../CVPR2026/multimodal_vlm/tuna_taming_unified_visual_representations_for_native_unified_multimodal_models.md)
- [\[ICCV 2025\] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](vq-vla_improving_vision-language-action_models_via_scaling_vector-quantized_acti.md)

</div>

<!-- RELATED:END -->

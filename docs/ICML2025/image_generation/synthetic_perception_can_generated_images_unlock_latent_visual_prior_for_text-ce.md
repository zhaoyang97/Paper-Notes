---
title: >-
  [论文解读] Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?
description: >-
  [ICML 2025][图像生成][synthetic perception] 系统研究"合成感知"——利用T2I模型为纯文本数据即时生成合成图像作为互补模态，通过三阶段评估框架（生成→融合→评估）证明该策略在讽刺检测和隐式情感分析等困难任务上可为Llama-3/Qwen-2.5等强LLM带来显著提升（+3.9% Acc），但在简单事实分类任务上增益边际。
tags:
  - "ICML 2025"
  - "图像生成"
  - "synthetic perception"
  - "T2I augmentation"
  - "多模态"
  - "text classification"
  - "modality gap"
---

# Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?

**会议**: ICML 2025  
**arXiv**: [2506.17623](https://arxiv.org/abs/2506.17623)  
**代码**: 待发布  
**领域**: 图像生成  
**关键词**: synthetic perception, T2I augmentation, multimodal fusion, text classification, modality gap

## 一句话总结

系统研究"合成感知"——利用T2I模型为纯文本数据即时生成合成图像作为互补模态，通过三阶段评估框架（生成→融合→评估）证明该策略在讽刺检测和隐式情感分析等困难任务上可为Llama-3/Qwen-2.5等强LLM带来显著提升（+3.9% Acc），但在简单事实分类任务上增益边际。

## 研究背景与动机

**领域现状**：大量真实世界数据仅以纯文本形式存在，造成"模态缺口"限制了多模态模型的直接应用。同时T2I合成技术已达到前所未有的图像质量和语义保真度。

**现有痛点**：

1. 标准多模态学习依赖高质量图文对数据，但纯文本数据天然没有视觉对应

2. 已有数据增强方法将T2I作为离线数据工厂扩充训练集，但未探索在线、实例级的模态补全策略

3. 多模态微调可能导致"模态诱导遗忘"——模型原有的文本处理能力退化

**核心问题**：T2I生成的合成图像作为动态新模态，究竟能在什么条件下（if）、通过什么机制（how）、以多大程度（to what extent）增强文本理解？

## 方法详解

### 整体框架

三阶段评估流水线：

1. **合成视觉模态生成**：原始文本 → 提示工程策略 → T2I模型 → 合成图像

2. **多模态表示与融合**：文本编码器提取文本特征 + 图像编码器提取视觉特征 → 融合模块 → 统一表示

3. **下游任务评估**：融合表示 → 分类头 → 与纯文本基线、文本扩展基线、知识检索基线对比

### 关键设计

1. **多层次提示工程策略**

    - P1（Direct）：直接使用原始文本作为T2I prompt
    - P2（Keyword-Enhanced）：提取核心语义元素（名词/形容词/动词）插入模板，提高信噪比
    - P3（Task-Aligned Stylization）：注入任务相关的风格关键词
    - P4（LLM-Elaborated）：使用Llama-3-8B作为"提示工程师"重写文本为丰富的视觉描述
    - 结论：P2和P4最优，P2效率最高，P4靠LLM重写质量取胜

2. **三种融合架构对比**

    - F1（Late Fusion）：简单特征拼接
    - F2（Cross-Attention）：Transformer解码层中文本特征查询图像特征——最优
    - F3（Deep Fusion / MMBT）：早期注入视觉token到文本编码器
    - 关键发现：Cross-Attention在大多数配置下最优，因为它允许选择性地提取相关视觉信息

### 损失函数 / 训练策略

- 文本编码器：Llama-3-8B-Instruct / Qwen-2.5-7B / Mistral-7B / BERT-base
- 图像编码器：SigLIP（最优）> CLIP ViT-B/32 > DINOv2（最差，因缺乏文本对齐能力）
- T2I模型：Flux.1-schnell（效率最优）/ SDXL / DALL-E 3（质量最优）
- AdamW优化器，早停防止过拟合，NVIDIA A100 GPU

## 实验关键数据

### 主实验

**跨四个数据集的总体性能（Accuracy%）**

| 方法 | Amazon Reviews | AG News | SARC | Implicit Sentiment |
|------|---------------|---------|------|-------------------|
| Llama-3 Text-Only | 81.20 | 95.10 | 76.50 | 79.50 |
| + Textual Expansion | 81.65 | 95.15 | 77.10 | 80.10 |
| + Knowledge Retrieval | 81.40 | 95.30 | 76.80 | 79.80 |
| **+ Gen. Image (Ours)** | **82.90** | 95.25 | **80.40** | **83.20** |
| Qwen-2.5 Text-Only | 82.50 | 95.40 | 78.20 | 81.10 |
| **Qwen-2.5 + Gen. Image** | **84.10** | 95.55 | **82.10** | **84.80** |

### 消融实验

**T2I模型与提示策略对比（Amazon Reviews, Macro-F1%）**

| T2I模型 | 提示策略 | Ma-F1 | CLIP Score |
|---------|---------|-------|------------|
| SD1.5 | P2 | 76.24 | 0.28 |
| SDXL | P2 | 77.02 | 0.32 |
| SDXL | P4 | 77.35 | 0.33 |
| Flux.1-schnell | P2 | 77.15 | 0.34 |
| DALL-E 3 | P2 | 77.80 | 0.35 |

### 关键发现

- **突破"文本扩展"天花板**：在SARC数据集上，文本扩展仅提升0.6%，而合成图像提升3.9%——证明增益来自视觉模态本身而非"更多文本"
- **任务难度决定增益**：困难任务（讽刺检测+3.9%、隐式情感+3.7%）增益远大于简单任务（AG News+0.15%）
- **SigLIP >> DINOv2**：语义对齐能力比纯视觉几何特征重要得多
- **Flux.1-schnell**：性能接近SDXL（77.15% vs 77.35%）但推理速度快10倍（0.8s vs 8s）
- CLIP Score与下游性能高度相关，可作为合成图像质量的代理指标

## 亮点与洞察

- 首次将T2I生成图像定位为"在线实例级模态补全"而非离线数据增强，提出了一种全新的跨模态探测范式
- 关键洞察：合成感知对需要视觉接地的深层推理任务最有价值（讽刺需要"看到"字面与实际情境的冲突）
- 设计了3个"关键基线"（文本扩展/知识检索/Oracle）系统排除混淆因素，方法论严谨
- Flux.1-schnell的效率分析为实际部署提供了可行性论证

## 局限与展望

- 仅在文本分类任务上验证，未涉及生成任务（如摘要、QA）
- 合成图像的计算开销是实际部署的主要瓶颈，即便Flux.1-schnell也需0.8s/样本
- 对"模态诱导遗忘"问题仅提及未深入定量分析
- 数据集选择偏英文，缺少多语种评估
- 未探索视频或多图像作为合成模态的可能性

## 相关工作与启发

- **vs TTIDA/DIAGen**：这些方法将T2I做离线数据增强扩充训练集，本文做在线实例级模态补全，目标和使用方式不同
- **vs Missing Modality方法**：传统缺失模态方法是被动应对，本文主动为纯文本数据合成视觉模态
- **vs CLIP/SigLIP**：图像编码器的语义对齐能力是合成感知成功的关键前提
- 启发：对于文本中含有隐喻、讽刺、暗示等需要"看见"场景才能理解的任务，合成视觉是有价值的补充

## 评分

- 新颖性: ⭐⭐⭐ 研究问题新颖但方法框架以评估为主，无模型架构创新
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、四类编码器、五种T2I模型、四种提示策略的全面消融
- 写作质量: ⭐⭐⭐⭐ 研究问题明确，三阶段框架清晰
- 价值: ⭐⭐⭐ 对合成感知范式的系统评估有参考价值，但缺少方法创新

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[ICML 2025\] Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)
- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)
- [\[NeurIPS 2025\] AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition](../../NeurIPS2025/image_generation/auggen_synthetic_augmentation_using_diffusion_models_can_imp.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](../../CVPR2025/image_generation/opensdi_spotting_diffusion-generated_images_in_the_open_world.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions
description: >-
  [NeurIPS 2025][多模态VLM][CLIP] 提出 CLIP-IN 框架，利用指令编辑数据集作为硬负样本和长描述增强 CLIP 的细粒度视觉理解能力，在 MMVP 等基准上显著提升且不损害零样本性能，集成到 MLLM 中可减少视觉幻觉。 CLIP 等视觉-语言模型在粗粒度图文对齐上表现出色，但在细粒度视觉理解上存…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "CLIP"
  - "细粒度视觉理解"
  - "指令编辑数据"
  - "硬负样本"
  - "长描述"
---

# VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions

**会议**: NeurIPS 2025  
**arXiv**: [2508.02329](https://arxiv.org/abs/2508.02329)  
**代码**: 无  
**领域**: 机器人  
**关键词**: CLIP, 细粒度视觉理解, 指令编辑数据, 硬负样本, 长描述

## 一句话总结

提出 CLIP-IN 框架，利用指令编辑数据集作为硬负样本和长描述增强 CLIP 的细粒度视觉理解能力，在 MMVP 等基准上显著提升且不损害零样本性能，集成到 MLLM 中可减少视觉幻觉。

## 研究背景与动机

CLIP 等视觉-语言模型在粗粒度图文对齐上表现出色，但在细粒度视觉理解上存在明显短板：

**粗粒度对齐**: CLIP 的对比学习倾向于学习高层语义对齐，忽略细微差异

**短文本限制**: 标准 CLIP 使用短文本描述，丢失丰富的语义细节

**缺乏硬负样本**: 训练中缺少高度相似但语义不同的图文对

本文的两个核心创新：
- 利用**图像编辑指令数据集**作为天然的硬负样本源
- 引入**长描述**和**旋转位置编码**来捕获丰富语义

## 方法详解

### 整体框架

CLIP-IN 包含两个核心创新：
1. 基于指令编辑数据的硬负样本对比学习
2. 融合长描述的旋转位置编码

### 关键设计

1. **指令编辑数据作为硬负样本**:

    - 利用已有的图像编辑数据集（如 InstructPix2Pix）
    - 编辑前后的图像对构成天然的硬负样本
    - 例如：原图(猫在红色沙发上) vs 编辑图(猫在蓝色沙发上)
    - 配对的编辑指令提供语义差异的精确描述

2. **对称硬负样本对比损失**:

    - 不仅让模型匹配正确图文对，还要区分细微编辑差异
    - 对称设计：图→文和文→图两个方向都进行硬负样本对比
    $\mathcal{L}_{\text{HN}} = -\log \frac{e^{s(I, T^+)}}{e^{s(I, T^+)} + \sum_k e^{s(I, T_k^-)}}$

3. **长描述 + 旋转位置编码 (RoPE)**:

    - 引入详细的长文本描述（通常 100-300 tokens）
    - 标准 CLIP 的文本编码器限于 77 tokens
    - 使用 RoPE 扩展上下文长度，保持位置感知能力

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{CLIP}} + \alpha \mathcal{L}_{\text{HN}} + \beta \mathcal{L}_{\text{long}}$$

- $\mathcal{L}_{\text{CLIP}}$: 标准对比损失（保持零样本能力）
- $\mathcal{L}_{\text{HN}}$: 硬负样本对比损失（提升细粒度）
- $\mathcal{L}_{\text{long}}$: 长描述对齐损失

## 实验关键数据

### 主实验（细粒度视觉理解）

| 方法 | MMVP ↑ | Winoground ↑ | ARO-Relation ↑ | SugarCrepe ↑ | IN-1K 零样本 ↑ |
|------|--------|-------------|---------------|-------------|-------------|
| CLIP (ViT-L/14) | 28.5 | 35.2 | 62.8 | 75.3 | 75.5 |
| CLIP + NegCLIP | 32.1 | 38.5 | 68.5 | 78.2 | 74.8 |
| CLIP + SigLIP | 30.8 | 37.1 | 66.2 | 77.5 | 76.2 |
| CLIP + DAC | 35.2 | 40.8 | 70.5 | 80.1 | 74.5 |
| **CLIP-IN** | **42.8** | **46.5** | **75.8** | **84.5** | **75.8** |

### MLLM 集成实验

| 视觉编码器 | LLaVA-1.5 MMVP ↑ | LLaVA 幻觉率 ↓ | POPE Acc ↑ | MMBench ↑ |
|-----------|------------------|-------------|----------|----------|
| CLIP-ViT-L | 32.5 | 45.2 | 83.5 | 64.8 |
| SigLIP | 35.8 | 42.1 | 85.2 | 66.5 |
| **CLIP-IN** | **45.2** | **32.5** | **88.8** | **68.2** |

### 消融实验

| 组件 | MMVP ↑ | Winoground ↑ | IN-1K ↑ |
|------|--------|-------------|--------|
| CLIP-IN 完整 | 42.8 | 46.5 | 75.8 |
| 去掉硬负样本 | 33.5 | 39.2 | 76.1 |
| 去掉长描述 | 38.2 | 43.1 | 75.5 |
| 去掉 RoPE | 36.5 | 41.8 | 75.2 |
| 随机负样本(非指令编辑) | 35.8 | 40.5 | 75.5 |
| 仅用短描述 | 37.2 | 42.5 | 76.0 |

### 关键发现

1. CLIP-IN 在 MMVP 上提升 14.3%（28.5 → 42.8），证明硬负样本策略的巨大价值
2. 指令编辑数据作为硬负样本远优于随机负样本（+7.0 MMVP）
3. 关键：零样本 ImageNet 性能不降反升（75.5 → 75.8），说明细粒度提升不损害通用能力
4. 集成到 MLLM 后，视觉幻觉率从 45.2% 降至 32.5%，实际价值显著

## 亮点与洞察

- **数据源创新**: 图像编辑数据的"废物利用"，低成本获取高质量硬负样本
- **不损害通用性**: 在提升细粒度的同时保持零样本能力，这一点难能可贵
- **下游价值**: 减少 MLLM 幻觉的效果非常显著，直接提升了实际应用质量
- **RoPE 扩展**: 优雅地解决了 CLIP 文本长度限制

## 局限与展望

1. 指令编辑数据集主要覆盖视觉属性编辑，对抽象概念差异的覆盖不足
2. 长描述的生成依赖外部模型，可能引入噪声
3. RoPE 虽扩展了长度，但极长文本（>1000 tokens）的效果未验证
4. 训练成本高于标准 CLIP 微调

## 相关工作与启发

- **NegCLIP**: 基于负样本增强 CLIP 理解的先驱
- **InstructPix2Pix**: 本文使用的图像编辑数据源
- **SigLIP**: Google 的改进对比学习
- **DAC**: 描述增强的对比学习

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 3 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 5 |
| 总体推荐 | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](../../ICCV2025/multimodal_vlm/visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->

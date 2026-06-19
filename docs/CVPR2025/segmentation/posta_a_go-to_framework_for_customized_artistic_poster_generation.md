---
title: >-
  [论文解读] POSTA: A Go-to Framework for Customized Artistic Poster Generation
description: >-
  [CVPR 2025][语义分割][海报生成] 提出 POSTA，一个由扩散模型和多模态大语言模型驱动的模块化艺术海报生成框架，通过背景生成、版式设计规划和艺术文字风格化三个模块实现高度可定制的专业级海报创作。 领域现状 领域现状：海报设计是视觉传达的关键媒介，在广告、教育和艺术领域有广泛需求 现有痛点 现有痛点：现有自动海…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "海报生成"
  - "扩散模型"
  - "多模态大语言模型"
  - "艺术文字"
  - "版式设计"
---

# POSTA: A Go-to Framework for Customized Artistic Poster Generation

**会议**: CVPR 2025  
**arXiv**: [2503.14908](https://arxiv.org/abs/2503.14908)  
**代码**: [项目主页](https://haoyuchen.com/POSTA)  
**领域**: 图像分割 / 图像生成  
**关键词**: 海报生成, 扩散模型, 多模态大语言模型, 艺术文字, 版式设计

## 一句话总结

提出 POSTA，一个由扩散模型和多模态大语言模型驱动的模块化艺术海报生成框架，通过背景生成、版式设计规划和艺术文字风格化三个模块实现高度可定制的专业级海报创作。

## 研究背景与动机

### 领域现状

**领域现状**：海报设计是视觉传达的关键媒介，在广告、教育和艺术领域有广泛需求

### 现有痛点

**现有痛点**：现有自动海报生成方法面临四大挑战：文字生成不准确（拼写错误、字符扭曲）、设计元素控制不灵活（"一键生成"范式缺乏细粒度调整）、美学质量不足、缺乏系统化工作流

### 核心矛盾

**核心矛盾**：艺术海报（电影、展览、演唱会等）对内容准确性和视觉冲击力有双重要求

### 解决思路

**解决思路**：已有端到端方法如 COLE 虽然涵盖完整流程，但过程复杂、需要大量中间检查

### 补充说明

**补充说明**：现有文字生成模型在处理长文字序列时仍容易产生乱码和不可读字符

### 补充说明

**补充说明**：需要一个模块化、可控、高美感的系统化解决方案

## 方法详解

### 整体框架

POSTA 由三个模块组成：(1) Background Diffusion 基于 FLUX 和多个风格 LoRA 生成主题背景；(2) Design MLLM 基于 LLaVA 架构（CLIP ViT-L/14 + Llama3 7B）预测版式和排版属性（位置、大小、字体、颜色、对齐、旋转角度），然后直接渲染100%准确的文字；(3) ArtText Diffusion 基于 BrushNet（ControlNet 类 inpainting 架构）对标题文字进行艺术风格化。每个阶段完全可控可定制，符合专业设计师的工作流程。

### 关键设计

**1. 风格化背景生成 (Background Diffusion)**
- **功能**: 根据用户输入生成多种艺术风格的高质量背景图像
- **核心思路**: 基于 FLUX 模型训练多个专业 LoRA 块（每种风格约50张图片，嵌入维度64，分辨率1024），分别控制极简主义、复古、现代艺术等风格。集成 MLLM 驱动的 Magic Prompter 自动补充用户输入提示的细节
- **设计动机**: 专业级背景是高质量海报的基础，LoRA 微调让模型在保持通用能力的同时掌握特定艺术风格，模块化设计允许用户上传自定义背景

**2. 版式与排版规划 (Design MLLM)**
- **功能**: 基于背景图和用户需求智能规划布局、文字位置和排版属性
- **核心思路**: 通过视觉指令微调 LLaVA，模型理解背景图语义后为每个文字元素预测完整属性（坐标、字体、大小、颜色、对齐、旋转角度）。文字被分为标题/副标题/信息三类。**关键创新**：不使用生成模型"画"文字，而是基于预测的属性直接渲染文字，确保100%文字准确性
- **设计动机**: 直接渲染完全消除了生成模型的文字错误问题，矢量格式文字元素支持生成后的二次编辑，这与专业设计工作流一致

**3. 艺术文字风格化 (ArtText Diffusion)**
- **功能**: 对标题文字施加艺术效果（3D、金属质感、渐变色、轮廓等），使文字与背景风格融合
- **核心思路**: 基于 SDXL 的 BrushNet 训练 mask 指导的 inpainting 模型（分辨率1216），通过文字蒙版实现局部生成。融合公式：$I_{\text{blended}} = M \odot I_1 + (1-M) \odot I_2$，对蒙版应用高斯核实现平滑边界
- **设计动机**: 文字风格必须与背景语义和风格协调（如复古海报的文字应有做旧效果），上下文感知的训练数据使模型学会生成与背景和谐的文字效果

### 损失函数

各模块使用各自的标准训练损失：Background Diffusion 使用扩散模型标准去噪损失；Design MLLM 使用语言模型的交叉熵损失；ArtText Diffusion 使用 inpainting 扩散损失。

## 实验关键数据

### 主实验：人类与 GPT-4V 评估

| 方法 | 视觉吸引力 | 文字可读性 | 提示相关性 |
|------|-----------|-----------|-----------|
| SD3 | ~4.5 | ~3.0 | ~4.5 |
| FLUX-dev | ~5.5 | ~4.0 | ~5.5 |
| Recraft V3 | ~6.0 | ~5.0 | ~6.0 |
| Ideogram v2 | ~6.5 | ~5.5 | ~6.0 |
| **POSTA** | **~8.0** | **~9.0** | **~8.0** |

*60位有AI工具经验的用户和GPT-4V打分（1-10），POSTA 在所有指标上大幅领先*

### 文字准确率（OCR 评估）

| 方法 | 精确率 | 召回率 |
|------|--------|--------|
| AnyText | ~0.3 | ~0.25 |
| TextDiffuser-2 | ~0.35 | ~0.3 |
| FLUX | ~0.4 | ~0.35 |
| **POSTA** | **~1.0** | **~1.0** |

*POSTA 通过直接渲染实现近乎完美的文字准确率*

### 关键发现

- 所有其他模型（包括商业产品 Recraft、Ideogram）在处理较长文字时均出现严重的准确率下降
- POSTA 的模块化设计允许独立调整背景、布局、字体和风格的每一个方面
- 艺术文字风格化能根据不同背景区域自动调整文字的光照和颜色效果
- 参考海报生成和海报编辑任务展示了框架的多功能性

## 亮点与洞察

1. **"不生成文字"的巧妙策略**: 通过预测排版属性+直接渲染完全绕过了生成模型的文字错误问题，实用性远超端到端生成方法
2. **模块化与专业工作流对齐**: 三阶段流程与设计师的实际工作习惯一致，每个阶段可独立调整，大幅提升实用价值
3. **PosterArt 数据集**: 由专业设计师精心策划的高质量艺术海报数据集，包含详细排版标注和像素级文字分割

## 局限与展望

- Design MLLM 目前仅能生成较简单的布局设计，字体类型选择有限
- 数据集规模（约2000张背景+2500个文字分割）相对较小，限制了设计多样性
- ArtText Diffusion 对标题的风格化可能与某些极端背景不够协调
- 未来可通过扩展数据集和更先进的模型架构提升可扩展性

## 相关工作与启发

- 与 COLE/OpenCOLE 的系统化设计流程相比，POSTA 更简洁高效且美学质量更高
- BrushNet 的 mask 指导 inpainting 范式在局部生成任务中表现优异
- 将 MLLM 用于设计规划（而非直接生成）的思路可推广到其他创意任务

## 评分

⭐⭐⭐⭐ — 对海报生成问题的实用解法非常出色，模块化设计理念先进，文字准确性的解决方案优雅巧妙。实验对比全面（包含开源和商业模型），用户评估有说服力。但训练数据规模和设计复杂度仍有较大提升空间。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)
- [\[CVPR 2025\] EditAR: Unified Conditional Generation with Autoregressive Models](editar_unified_conditional_generation_with_autoregressive_models.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->

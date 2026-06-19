---
title: >-
  [论文解读] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit
description: >-
  [AAAI 2026][多模态VLM][视觉语言模型] 本文提出 LLMC+，一个全面的视觉语言模型（VLM）压缩基准和即插即用工具包，支持 5 个代表性 VLM 家族的 20+ 种压缩算法，系统研究了 token 级和模型级压缩的独立及联合效果，揭示了三大关键发现。 领域现状：大型视觉语言模型（VLM…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "视觉语言模型"
  - "模型压缩"
  - "token裁剪"
  - "量化"
  - "基准测试"
---

# LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit

**会议**: AAAI 2026  
**arXiv**: [2508.09981](https://arxiv.org/abs/2508.09981)  
**代码**: [GitHub](https://github.com/ModelTC/LightCompress)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: 视觉语言模型, 模型压缩, token裁剪, 量化, 基准测试

## 一句话总结

本文提出 LLMC+，一个全面的视觉语言模型（VLM）压缩基准和即插即用工具包，支持 5 个代表性 VLM 家族的 20+ 种压缩算法，系统研究了 token 级和模型级压缩的独立及联合效果，揭示了三大关键发现。

## 研究背景与动机

**领域现状**：大型视觉语言模型（VLM，如 LLaVA、InternVL、Qwen-VL）展现了强大的多模态理解能力，但其计算和内存需求巨大——超长的视觉 token 序列和巨量参数是两大瓶颈。近期出现了多种免训练压缩方法，包括 token 裁剪（减少视觉 token 数量）和模型量化（降低参数精度）。

**现有痛点**：（1）现有方法未将技术解耦为可比较的模块，导致空间冗余和时间冗余的方法无法公平对比；（2）评估局限于简单的单轮任务，无法反映多轮对话等真实场景的表现；（3）各种压缩技术独立使用，未探索联合压缩的潜力。

**核心矛盾**：缺乏统一的评估框架导致压缩方法之间"苹果比橘子"，难以给出可靠的方法选择建议。

**本文目标**：构建统一的 VLM 压缩基准，支持公平评估和系统研究。

**切入角度**：开发模块化工具包，将各种压缩方法解耦为可组合的模块。

**核心 idea**：通过统一的基准和工具包实现 VLM 压缩方法的公平对比和联合优化。

## 方法详解

### 整体框架

LLMC+ 包含：（1）统一接口——所有压缩方法统一为可组合的模块；（2）全面评估——涵盖单轮/多轮、视觉问答/推理/细节描述等多种任务；（3）联合压缩——探索 token 裁剪 + 模型量化的组合。

### 关键设计

1. **模块化压缩框架**:

    - 功能：使不同压缩方法在统一接口下可比较和可组合。
    - 核心思路：将压缩方法分为两大类——token 级压缩（视觉 token pruning/merging）和模型级压缩（权重量化/剪枝）。每类内部解耦为可替换的模块，如 token 重要性评估模块、token 裁剪策略模块等。
    - 设计动机：过去不同方法使用不同的评估设置和数据集，导致论文间的性能数值不可比。统一框架消除了评估偏差。

2. **多维度评估基准**:

    - 功能：全面评估压缩后VLM的能力保持。
    - 核心思路：评估覆盖：单轮视觉问答、多轮对话、细节敏感任务（如OCR、fine-grained recognition）、视觉推理。特别加入了多轮对话测试——现有基准几乎不测这个维度，但它对实际应用至关重要。
    - 设计动机：单轮VQA可能掩盖信息损失——模型即使丢失了一些视觉细节仍能猜对答案，但在需要持续理解的多轮对话中就会暴露问题。

3. **联合压缩探索**:

    - 功能：研究 token+模型双重压缩的可行性。
    - 核心思路：先进行 token 裁剪减少序列长度，再进行模型量化降低参数精度。测试不同压缩级别的组合，找到极致压缩下性能损失最小的配置。
    - 设计动机：单一压缩维度的收益有限（如4-bit量化已接近极限），联合压缩可以在多个维度同时降低成本。

### 损失函数 / 训练策略

所有压缩方法均为免训练（training-free），使用少量校准数据即可完成。评估使用标准的VLM推理pipeline。

## 实验关键数据

### 主实验

覆盖5个VLM家族，20+种压缩算法。

| 发现 | 详情 | 说明 |
|------|------|------|
| 发现1 | 空间和时间冗余需要不同技术策略 | Token级和模型级互相不可替代 |
| 发现2 | Token裁剪在多轮对话和细节任务上显著退化 | 单轮评估的假象 |
| 发现3 | Token+模型联合压缩可实现极致压缩且性能损失最小 | 1+1>2的效果 |

### 消融实验

| 压缩方式 | 压缩比 | 性能保持 | 说明 |
|----------|--------|---------|------|
| 仅token裁剪50% | 中等 | 单轮好/多轮差 | 信息累积损失 |
| 仅4-bit量化 | 中等 | 均匀下降 | 不依赖任务类型 |
| Token50%+4-bit量化 | 极致 | 接近单一压缩 | 联合效果好 |

### 关键发现

- Token 裁剪方法在多轮对话中的退化比预期严重得多——这暴露了现有评估的盲点。
- 联合压缩的关键是两种技术作用于不同的冗余维度——token裁剪减少空间冗余，量化减少精度冗余，两者几乎正交。
- 工具包的价值在于促进了公平比较，揭示了之前论文中被掩盖的方法优劣。

## 亮点与洞察

- **系统性基准构建**对VLM压缩领域有长期价值——后续研究可以直接在此基准上公平对比。
- **多轮对话评估**的加入是重要贡献——揭示了token裁剪的隐含风险。
- **联合压缩的发现**有直接的部署指导意义。

## 局限与展望

- 5个VLM家族可能不覆盖所有架构类型。
- 免训练压缩的上限可能低于量化感知训练方法。
- 可以扩展到视频理解VLM的压缩。

## 相关工作与启发

- **vs LLM压缩**: VLM比LLM多了视觉token这一冗余维度，压缩策略需要更多考虑。
- **vs 单一压缩方法论文**: LLMC+不是提出新方法，而是为所有方法提供公平舞台。

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性基准和联合压缩发现新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5个VLM家族+20+算法+多维度评估
- 写作质量: ⭐⭐⭐⭐ 发现总结清晰有洞察
- 价值: ⭐⭐⭐⭐⭐ 对VLM压缩领域有基础设施级贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)
- [\[NeurIPS 2025\] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression](../../NeurIPS2025/multimodal_vlm/breaking_the_compression_ceiling_data-free_pipeline_for_ultra-efficient_delta_co.md)
- [\[CVPR 2026\] VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression](../../CVPR2026/multimodal_vlm/vlic_vision-language_models_as_perceptual_judges_for_human-aligned_image_compres.md)

</div>

<!-- RELATED:END -->

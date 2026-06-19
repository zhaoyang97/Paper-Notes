---
title: >-
  [论文解读] CoLLM: A Large Language Model for Composed Image Retrieval
description: >-
  [CVPR 2025][多模态VLM][组合图像检索] 提出 CoLLM——利用大语言模型进行组合图像检索（CIR）的一站式框架，通过从图文对即时生成训练三元组、用 LLM 生成联合多模态嵌入，以及构建 340 万样本的 MTCIR 大规模数据集，在多个 CIR 基准上取得 SOTA 性能，MTCIR 最高带来 15% 的性能提升。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "组合图像检索"
  - "大语言模型"
  - "多模态融合"
  - "零样本检索"
  - "三元组生成"
---

# CoLLM: A Large Language Model for Composed Image Retrieval

**会议**: CVPR 2025  
**arXiv**: [2503.19910](https://arxiv.org/abs/2503.19910)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 组合图像检索, 大语言模型, 多模态融合, 零样本检索, 三元组生成

## 一句话总结

提出 CoLLM——利用大语言模型进行组合图像检索（CIR）的一站式框架，通过从图文对即时生成训练三元组、用 LLM 生成联合多模态嵌入，以及构建 340 万样本的 MTCIR 大规模数据集，在多个 CIR 基准上取得 SOTA 性能，MTCIR 最高带来 15% 的性能提升。

## 研究背景与动机

**领域现状**：组合图像检索（Composed Image Retrieval, CIR）是一种多模态检索任务，给定一张参考图像和一段描述期望修改的文本，目标是检索出满足修改要求的目标图像。例如给定一张"红色连衣裙"的图片和"改为蓝色"的文本，需要检索到"蓝色连衣裙"的图像。主流方法需要（参考图像, 修改文本, 目标图像）三元组数据来训练联合嵌入。

**现有痛点**：（1）**数据瓶颈严重**——标注 CIR 三元组极其昂贵耗时，现有数据集规模有限（如 CIRR 仅含约 3.6 万三元组），严重限制模型泛化能力；（2）**零样本方法的局限**——为绕过数据稀缺，一些方法使用合成三元组或 VLM 将 CIR 转化为图文检索，但合成三元组规模小、多样性差、修改文本不自然，而纯图文对方法由于缺乏三元组结构无法学习有效的联合嵌入；（3）**多模态融合不充分**——复杂的修改指令需要对视觉和语言的深度融合理解，现有方法（简单拼接、浅层交叉注意力等）无法处理细致入微的语义修改。

**核心矛盾**：CIR 需要三元组数据来训练联合嵌入，但三元组标注成本极高；零样本方法要么质量低（合成数据）要么缺乏联合学习能力（图文对）。如何在不依赖昂贵标注的情况下实现高质量的多模态联合嵌入学习？

**本文目标**：设计一个不需要人工三元组标注、能处理复杂修改指令、在监督和零样本设置下都表现优异的 CIR 方法。

**切入角度**：大语言模型天然擅长理解和融合复杂的多模态输入——如果能将参考图像和修改文本作为 LLM 的输入，让 LLM 直接输出联合嵌入，就能实现深层次的多模态融合。同时，可以利用 LLM 的文本处理能力从图文对中自动生成三元组。

**核心 idea**：用 LLM 同时解决数据和模型两个问题——（1）从现成的图文对即时生成 CIR 三元组用于训练；（2）用 LLM 的隐状态作为参考图像+修改文本的联合嵌入，实现深度多模态融合。

## 方法详解

### 整体框架

CoLLM 包含三个核心组件：（1）三元组即时生成模块——从网络爬取的图文对（image, caption）中自动构造（reference image, modification text, target image）三元组；（2）LLM 联合嵌入器——将参考图像和修改文本送入多模态 LLM，提取其隐状态作为联合查询嵌入；（3）检索模块——在预训练的视觉嵌入空间中用联合嵌入检索目标图像。训练时用生成的三元组进行对比学习。

### 关键设计

1. **即时三元组生成（On-the-fly Triplet Generation）**:

    - 功能：从图文对自动构造 CIR 训练三元组，消除人工标注需求
    - 核心思路：给定一个 mini-batch 中的多个（图像, 标题）对，将其中两个配对——图像 A 作为参考图像，图像 B 作为目标图像，然后用 LLM 分析两张图的标题差异，自动生成描述"如何从 A 修改到 B"的修改文本。这个过程在训练时实时进行（on-the-fly），不需要预先构建三元组数据集。关键在于利用同一 batch 中语义相关但不完全相同的图像对作为伪三元组
    - 设计动机：纯合成三元组需要额外的生成模型且规模有限；从图文对即时构造则可以利用互联网上几乎无限的图文数据，且生成的修改文本更加自然，因为它基于真实图像之间的差异

2. **LLM 联合嵌入（LLM Joint Embedding）**:

    - 功能：用 LLM 的深层推理能力生成参考图+修改文本的联合表示
    - 核心思路：将参考图像通过视觉编码器提取特征后作为 visual tokens、修改文本作为 text tokens，一起送入 LLM。取 LLM 特定层的隐状态（或最后一个 token 的输出）作为联合查询嵌入。这个嵌入通过投影层映射到与目标图像相同的特征空间中，使用对比损失（如 InfoNCE）训练投影层和（可选的）LLM 适配器
    - 设计动机：传统方法用简单的拼接或浅层交叉注意力融合图文，无法处理"把红色改为蓝色并缩小裙摆"这样需要理解语义依赖关系的复杂修改指令。LLM 的多层 Transformer 天然适合处理这种需要深层推理的融合任务

3. **MTCIR 大规模数据集 + 基准修正**:

    - 功能：提供 340 万样本的大规模 CIR 训练集，并修正现有基准的评估问题
    - 核心思路：MTCIR（Multi-Text CIR）包含约 340 万三元组样本，基于多个来源的图文对构建。每个三元组包含多条修改文本描述（Multi-Text），覆盖不同粒度和风格的修改。此外，论文发现现有基准 CIRR 和 Fashion-IQ 存在标注噪声和评估偏差（如 CIRR 有约 X% 的测试对有歧义），提出了修正版本以提高评估可靠性
    - 设计动机：数据规模对深度学习至关重要；MTCIR 比现有最大 CIR 数据集大一个数量级以上。基准修正则确保了评估结果的可靠性——在有噪声的基准上比较方法可能得出误导性结论

### 损失函数 / 训练策略

主要使用 InfoNCE 对比损失训练：将 LLM 生成的联合嵌入与正确目标图像的视觉嵌入拉近，与负样本的嵌入推远。训练时可以冻结 LLM 主体只训练投影层（参数高效），也可以用 LoRA 微调 LLM（更高性能）。视觉编码器通常冻结（使用 CLIP 等预训练权重），确保目标图像的嵌入空间稳定。

## 实验关键数据

### 主实验

| 方法 | 设置 | CIRR R@5 | CIRR R@10 | Fashion-IQ R@10 | 说明 |
|------|------|----------|-----------|-----------------|------|
| Pic2Word | 零样本 | 基准 | 基准 | 基准 | 图文对方法 |
| SEARLE | 零样本 | 中等 | 中等 | 中等 | 合成三元组 |
| CompoDiff | 零样本 | 较高 | 较高 | 较高 | 扩散模型 |
| **CoLLM** | **零样本** | **SOTA** | **SOTA** | **SOTA** | LLM 联合嵌入 |
| ARTEMIS | 监督 | 基准 | 基准 | 基准 | 传统方法 |
| **CoLLM** | **监督** | **SOTA** | **SOTA** | **SOTA** | +三元组数据 |

### MTCIR 贡献

| 训练数据 | 性能 | 说明 |
|----------|------|------|
| 原始小规模数据 | 基准 | 现有数据集 |
| + MTCIR | 最高 +15% | 大规模数据带来显著提升 |
| 修正基准 vs 原始基准 | 排名变化 | 更可靠的评估减少了噪声影响 |

### 关键发现

- **LLM 联合嵌入显著优于浅层融合**：使用 LLM 隐状态作为联合嵌入比传统的拼接/交叉注意力方法性能更高，尤其在处理复杂修改文本时优势明显
- **即时三元组生成可行且有效**：从图文对即时生成的三元组质量足以支撑有效训练，性能接近甚至超过使用人工标注三元组的方法
- **数据规模至关重要**：MTCIR 的 340 万样本带来最高 15% 的性能提升，证明 CIR 领域的性能瓶颈很大程度上在数据而非模型
- **基准修正有意义**：在修正后的 CIRR 和 Fashion-IQ 上，不同方法的相对排名出现变化，说明原始基准的噪声确实影响了公平比较

## 亮点与洞察

- **一站式解决方案**：CoLLM 同时解决了 CIR 的数据问题（即时三元组生成）、模型问题（LLM 联合嵌入）和评估问题（基准修正），这种系统性的解决思路值得学习
- **LLM 作为特征融合器的新范式**：不将 LLM 用于生成文本，而是利用其内部表示作为多模态联合嵌入——这种"LLM as Encoder"的思路可以迁移到其他需要深度多模态融合的检索任务中
- **三元组即时生成**：利用 batch 内图像对的语义差异自动构造三元组是一个巧妙的自监督策略，这个思路可以推广到其他需要关系型训练数据的任务

## 局限与展望

- 即时三元组生成的质量依赖于 batch 内图像对的语义相关性，如果 batch 中图像差异太大或太小，生成的修改文本可能不够有意义
- LLM 推理的计算成本较高，在大规模检索的在线场景中可能成为瓶颈
- MTCIR 的构建过程依赖于自动化程序，可能包含噪声样本
- 论文主要在时尚和通用场景评估，在专业领域（如医学影像、卫星图像）的泛化性未验证
- 未来可探索：将 CoLLM 扩展到视频 CIR、3D CIR 等更复杂的检索场景

## 相关工作与启发

- **vs Pic2Word**: Pic2Word 将参考图像映射为文本 token 与修改文本拼接，用 CLIP 文本编码器生成查询嵌入——融合深度有限。CoLLM 用更强大的 LLM 替代 CLIP 文本编码器，实现更深层的融合
- **vs SEARLE**: SEARLE 通过反转 CLIP 图像嵌入为伪文本 token 进行零样本 CIR。CoLLM 的即时三元组生成策略让模型能做真正的监督学习，而非依赖近似
- **vs CompoDiff**: CompoDiff 用扩散模型生成目标图像再做检索。CoLLM 直接在嵌入空间操作，避免了生成步骤的计算开销和误差累积
- CoLLM 的"LLM 作为嵌入器"思路与 E5-Mistral 等文本检索工作异曲同工，值得关注这一趋势在多模态领域的发展

## 评分

- 新颖性: ⭐⭐⭐⭐ LLM 联合嵌入和即时三元组生成均有创新，但各自的基础组件（对比学习、LLM 嵌入）并非全新
- 实验充分度: ⭐⭐⭐⭐ 多基准多设置验证全面，但 HTML 不可用导致具体数字无法完全核实
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐⭐ 对 CIR 领域有全方位贡献（数据+模型+评估），MTCIR 数据集价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](../../ACL2026/multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](../../CVPR2026/multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] Self-guided Semantic Inspection for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/self-guided_semantic_inspection_for_zero-shot_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->

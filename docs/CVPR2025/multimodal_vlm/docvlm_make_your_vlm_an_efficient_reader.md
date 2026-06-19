---
title: >-
  [论文解读] DocVLM: Make Your VLM an Efficient Reader
description: >-
  [CVPR 2025][多模态VLM][文档理解] 提出一种模型无关的 OCR 编码模块，将 OCR 提取的文本和布局信息压缩为 64 个 learned query token 并注入冻结的 VLM，在极低视觉 token 数量下大幅提升文档理解能力（DocVQA 最高 +30.6 分），并零样本泛化到多页文档。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "文档理解"
  - "OCR"
  - "token压缩"
  - "模型无关"
  - "多页文档"
---

# DocVLM: Make Your VLM an Efficient Reader

**会议**: CVPR 2025  
**arXiv**: [2412.08746](https://arxiv.org/abs/2412.08746)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 文档理解、OCR编码器、token压缩、模型无关、多页文档

## 一句话总结
提出一种模型无关的 OCR 编码模块，将 OCR 提取的文本和布局信息压缩为 64 个 learned query token 并注入冻结的 VLM，在极低视觉 token 数量下大幅提升文档理解能力（DocVQA 最高 +30.6 分），并零样本泛化到多页文档。

## 研究背景与动机

**领域现状**：文档理解任务（DocVQA、InfoVQA 等）要求模型能识别文档中的文字并理解其布局关系。当前主流 VLM（LLaVA-OneVision、InternVL2、Qwen2-VL）主要依赖高分辨率视觉 token 来"看清"文字，但高分辨率意味着大量视觉 token。

**现有痛点**：当视觉 token 数量受限时（低分辨率或 token 预算有限），VLM 在文档任务上性能急剧下降——InternVL2 在 256 token 时 DocVQA 仅 56.0%，而 1280 token 时达 85.7%。另一方面，简单地将 OCR 文字作为文本 token 输入 LLM 虽然有效，但 800+ 个 OCR token 带来巨大的推理开销。

**核心矛盾**：文档理解需要精确的文字和布局信息，但从像素中提取这些信息需要大量视觉 token；OCR 可以直接提供文字，但缺乏高效的编码和压缩方式将其融入 VLM。

**本文目标** 设计一个高效的 OCR 编码模块，用极少的 token（64 个）将文档的文字和布局信息注入冻结的 VLM，在不修改 VLM 权重的前提下显著提升文档理解。

**切入角度**：利用 DocFormerV2 的编码器处理 OCR 文本 + 2D 位置信息，通过 instruction-aware 的 learned query 压缩机制将 800+ 个 OCR token 压缩为 64 个，然后与视觉 token 拼接送入 VLM。

**核心 idea**：用 learned query 压缩的 OCR 编码器作为即插即用的文档理解增强模块，以 64 个 token 的开销实现接近 800 token 完整编码的文档理解能力。

## 方法详解

### 整体框架
输入文档图像 → OCR 系统提取文字和 bounding box → DocFormerV2 编码器处理 OCR embedding + 指令 embedding + 64 个 learned query → 只保留 query 输出作为压缩表征 → 线性投影到 VLM 隐藏维度 → 与视觉 token 拼接 → 送入冻结的 LLM 解码器。

### 关键设计

1. **OCR 编码器（基于 DocFormerV2）**:

    - 功能：将 OCR 文本和 2D 布局信息编码为连续表征
    - 核心思路：使用 DocFormerV2 的 T5 编码器（344M 参数），去掉视觉分支（避免与 VLM 视觉编码器冗余）。输入为 OCR 文字 token 的 embedding + 2D bounding box 位置编码，输出为包含文字语义和空间布局信息的连续表征
    - 设计动机：比直接将 OCR 文字作为 LLM 输入高效得多——编码器能捕获布局上下文（如表格结构、段落层次），消融实验显示编码器输出比原始 OCR 文字在 DocVQA 上高 9+ 个点

2. **Instruction-Aware Query 压缩**:

    - 功能：将数百个 OCR token 压缩为固定数量（64）的紧凑表征
    - 核心思路：初始化 $M=64$ 个 learnable query，其分布与 OCR encoder embedding 匹配。query 与 OCR embedding 和指令 embedding 一起送入编码器进行联合注意力处理，编码器输出中只保留 query 对应的 64 个特征。query 不仅聚合 OCR 信息，还根据指令内容自适应地关注相关文档区域
    - 设计动机：类似 BLIP-2 的 Q-Former 思路，但关键区别在于 instruction-aware——query 在编码阶段就看到了问题，可以有选择地保留与问题相关的信息，比无条件压缩更高效

3. **两阶段训练（VLM 完全冻结）**:

    - 功能：逐步对齐 OCR 编码器和 VLM 的表征空间
    - 核心思路：Stage I（OCR-LLM 对齐）：不使用图像输入，强制模型仅依赖 OCR 编码；先冻结编码器 10K 步只训练 query + 投影层，再解冻编码器训练 130K 步。Stage II（视觉对齐）：重新引入视觉特征，100K 步微调，学习 OCR 和视觉的互补
    - 设计动机：Stage II 对压缩表征特别关键——16 query 时 Stage II 带来 +6.2 的提升，而 800 full token 时仅 +0.7。压缩表征需要更多训练来学会与视觉特征的互补

### 损失函数 / 训练策略
标准的 next-token prediction 损失。VLM 权重全程冻结，只训练 OCR 编码器 + learned query + 投影层。训练数据包括多种文档理解数据集。

## 实验关键数据

### 主实验

| 方法 | Token 数 | DocVQA | InfoVQA | MP-DocVQA | TextVQA |
|------|---------|--------|---------|-----------|---------|
| InternVL2 baseline | 256 | 56.0 | 38.4 | 51.0 | 65.7 |
| DocVLM+InternVL2 | 320 | **86.6** | **57.6** | **76.2** | **71.2** |
| Qwen2-VL baseline | 320 | 84.4 | 54.1 | 73.0 | 78.0 |
| DocVLM+Qwen2-VL | 320 | **91.2** | **61.2** | **81.7** | **79.6** |
| Qwen2-VL baseline | 576 | 91.5 | 65.3 | 82.1 | 82.3 |
| DocVLM+Qwen2-VL | 576 | **92.8** | **66.8** | **84.5** | **82.8** |

### 消融实验

| 配置 | DocVQA | 说明 |
|------|--------|------|
| OCR 原始文字 (800 tok) | 76.4 | 直接输入 OCR 文字 |
| OCR 编码 (800 tok) | 89.2 | 编码后大幅提升 |
| 64 压缩 query | 85.5 | 仅用 8% token 保留大部分增益 |
| 64 压缩 + 视觉特征 | 90.2 | 视觉互补提升 +4.7 |
| 800 编码 + 视觉特征 | 91.9 | 全编码 + 视觉互补 +2.7 |
| 16 query (Stage I only) | 81.7 | 低 query 数 |
| 16 query (+ Stage II) | 87.9 | Stage II 关键 (+6.2) |

### 关键发现
- **低 token 下增益最大**：InternVL2 从 256 token 的 56.0 跳到 320 token(+64 OCR) 的 86.6，提升 30.6 个绝对点，这是最惊人的结果
- **64 query 是最佳平衡点**：保留了 800 token 完整编码 ~97% 的信息，但 token 数量仅为 8%
- **视觉和 OCR 高度互补**：对压缩编码的提升（+4.7）大于对完整编码的提升（+2.7），说明压缩过程丢失的信息恰好被视觉特征弥补
- **零样本多页泛化**：仅用单页数据训练，MP-DocVQA 达到 86.3% ANLS（超越之前 SOTA GRAM 的 80.3%），验证了压缩编码的泛化能力

## 亮点与洞察
- **"即插即用 OCR 模块"的设计思路极具工程价值**：VLM 完全冻结，不影响原始通用能力，仅通过外接 OCR 编码器增强文档理解。这种模块化设计可以推广到其他专业能力的增强（如表格理解、图表理解）
- **64 个 token 的极致压缩**：类似 BLIP-2 的 Q-Former 但加入了 instruction-awareness，使压缩更智能。这个 idea 可以迁移到视频 VLM 中压缩长视频的 token
- **零样本多页扩展**：只在单页上训练但能处理多页，通过 global/page-wise 两种编码策略灵活应对不同需求

## 局限与展望
- 依赖外部 OCR 系统的质量，OCR 错误会直接传播到编码器
- OCR 编码器额外增加 344M 参数，加上 64 个 query 的推理开销，在极端实时场景下可能不可忽略
- 仅在文档理解任务上验证，对通用图像理解任务是否有影响未讨论
- Instruction-aware 压缩意味着不同问题需要重新编码，不能像固定视觉 token 那样复用缓存

## 相关工作与启发
- **vs GRAM**：之前的多页文档 SOTA，使用外部文本模块但方法更复杂。DocVLM 以更简洁的方式超越它（86.3 vs 80.3）
- **vs TextMonkey / UReader**：这些方法通过修改视觉编码器来增强 OCR 能力，需要重新训练 VLM。DocVLM 保持 VLM 冻结，更灵活
- **vs 直接 OCR 文字输入**：消融实验清晰展示了编码器比原始文字高 9+ 个点，布局编码是关键差异

## 评分
- 新颖性: ⭐⭐⭐⭐ instruction-aware query 压缩是 Q-Former 思路的有效改进，即插即用设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 VLM 基线、7 个 benchmark、详尽的压缩率/训练阶段消融
- 写作质量: ⭐⭐⭐⭐ 方法讲解清楚，实验组织合理，但部分表格重复信息较多
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 文档理解有即时的工程应用价值，模块化设计特别实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](../../AAAI2026/multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[CVPR 2026\] Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation](../../CVPR2026/multimodal_vlm/is_your_vlm_sky-ready_a_comprehensive_spatial_intelligence_benchmark_for_uav_nav.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[ICML 2025\] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning](../../ICML2025/multimodal_vlm/towards_efficient_online_tuning_of_vlm_agents_via_counterfactual_soft_reinforcem.md)

</div>

<!-- RELATED:END -->

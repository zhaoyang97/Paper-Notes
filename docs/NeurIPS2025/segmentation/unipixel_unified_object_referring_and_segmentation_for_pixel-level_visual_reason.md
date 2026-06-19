---
title: >-
  [论文解读] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
description: >-
  [NeurIPS 2025][语义分割][多模态大模型] UniPixel 提出了首个端到端统一对象引用 (referring) 和分割 (segmentation) 的大型多模态模型，通过创新的 Object Memory Bank 设计将稀疏视觉提示转化为稠密对象掩码特征并注入推理过程，在 10 个基准上实现 SOTA，还引入了需要同时完成引用、分割和问答的 PixelQA 新任务。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "多模态大模型"
  - "像素级推理"
  - "对象引用与分割统一"
  - "Object Memory Bank"
  - "视频理解"
---

# UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2509.18094](https://arxiv.org/abs/2509.18094)  
**代码**: [项目主页](https://polyu-chenlab.github.io/unipixel/)  
**领域**: 图像分割  
**关键词**: 多模态大模型, 像素级推理, 对象引用与分割统一, Object Memory Bank, 视频理解

## 一句话总结

UniPixel 提出了首个端到端统一对象引用 (referring) 和分割 (segmentation) 的大型多模态模型，通过创新的 Object Memory Bank 设计将稀疏视觉提示转化为稠密对象掩码特征并注入推理过程，在 10 个基准上实现 SOTA，还引入了需要同时完成引用、分割和问答的 PixelQA 新任务。

## 研究背景与动机

大型多模态模型 (LMMs) 在整体图像/视频理解任务上表现出色，但在细粒度像素级理解方面存在两个根本性限制：

**交互形式单一**：用户只能通过文本交互，缺乏更直观的沟通方式（如画点/框作为引用，或用掩码来 ground 模型响应）

**推理粒度粗糙**：模型内部推理主要在整体层面进行，直接感知全部内容而非对特定对象/区域进行推理，难以理解细粒度细节

现有方法（如 LISA、VISA 等）虽然探索了 LMM 驱动的分割，但存在根本局限：它们只能独立地执行引用或分割任务，依赖于刚性的输入/输出模板（如 LISA 的 "It's \<SEG\>."），无法灵活地同时理解用户引用的概念并生成掩码 grounded 的响应。更关键的是，这些方法无法将细粒度感知能力与通用的多模态推理能力融合，导致在通用理解基准上性能退化。

UniPixel 的核心创新在于通过 Object Memory Bank 统一了引用和分割的内部表示，使模型能够在推理过程中动态地分割关键对象、编码其特征、并基于这些对象级信息进行后续推理。

## 方法详解

### 整体框架

UniPixel 基于 Qwen2.5-VL 框架构建，包含 LLM 骨干和支持动态分辨率的 ViT 视觉编码器。在此基础上引入三个关键组件：Prompt Encoder（支持点/框/掩码三种视觉提示）、Object Memory Bank（存储和注入对象信息）、Mask Decoder（基于 SAM 2.1 生成时空掩码）。扩展 LLM 词汇表，添加 `<REF>`、`<MEM>`、`<SEG>` 三个特殊 token。

### 关键设计

1. **Prompt Encoder（视觉提示编码器）**：将每种视觉提示编码为单个 token 送入 LLM。对于稀疏提示（点和框），使用 2D Fourier 嵌入编码空间坐标加可学习类型嵌入，创新性地扩展加入 1D Fourier 时间编码表示帧索引，最后通过 GELU→Linear 投影到 LLM 嵌入空间。对于稠密提示（掩码），直接在视觉编码器输出上做 masked pooling，通过 M→L 投影器映射。设计动机：受 SAM 启发但有两个关键差异——加入时间信息并去掉负向点。

2. **Object Memory Bank（对象记忆库）**：这是核心创新——一个以对象 ID 为键、时空掩码为值的 hashmap，每个对话会话初始化为空，按需动态更新。包含两个操作：(a) **Memory Pre-filling**：当输入中检测到 `<REF>` token 时触发，模型生成对象 ID 和 `<SEG>` token 预测时空掩码并存入记忆库；(b) **Memory Injection**：将存储的对象掩码下采样后做 masked pooling，每帧掩码压缩为单个特征 token 通过投影器映射后替换 `<MEM>` token，将对象级信息注入后续推理。设计动机：直接在 `<REF>` 后追加 `<SEG>` 的替代方案存在两个问题——因果自注意力使掩码无法获取完整上下文导致质量差，以及引用和分割无法解耦训练。

3. **Mask Decoder（掩码解码器）**：采用 SAM 2.1 解耦离散语言建模和连续掩码预测。对每个 `<SEG>` token 提取最后层隐状态，通过 L→M 投影器下采样并 reshape 为两个 token（确保信息在高→低维通道空间下采样时更好保留），用这些 token 提示 SAM 2.1 在首帧预测掩码后传播到其他帧。

### 损失函数 / 训练策略

总损失为语言建模损失和掩码解码损失的线性组合：
- 语言建模：标准 cross-entropy，权重 1
- 掩码预测：focal loss (权重 100) + dice loss (权重 5) + IoU 预测 MAE (权重 5) + objectness cross-entropy (权重 5)

三阶段渐进训练：
1. 阶段一：用 851K 区域描述数据预训练稀疏提示编码器
2. 阶段二：用 87K 引用分割数据训练 L→M 投影器对齐 LLM 和掩码解码器
3. 阶段三：解冻 M→L 投影器和掩码解码器，对视觉编码器和 LLM 应用 LoRA，在 ~2M 多任务样本上联合训练

## 实验关键数据

### 主实验 — 推理视频目标分割 (ReVOS)

| 方法 | 模型大小 | Overall 𝒥&ℱ | 之前SOTA | 提升 |
|------|---------|-------------|---------|------|
| VISA | 13B | 50.9 | — | — |
| ViLLa | 6B | 57.0 | — | — |
| UniPixel | 3B | **62.1** | 57.0 | +5.1 |
| UniPixel | 7B | **63.7** | 57.0 | +6.7 |

### 消融实验

| 配置 | 𝒥&ℱ | Acc | 说明 |
|------|------|-----|------|
| 仅 Referring | — | 64.6 | 无分割能力 |
| 仅 Segmentation | 47.5 | — | 无引用能力 |
| Refer + Segment (无Memory) | 48.2 | 67.4 | 统一但无记忆 |
| Refer + Segment + Memory | **49.0** | **68.5** | 完整 UniPixel |
| ① 单 token 引用 | 46.8 | 64.5 | 最简引用 |
| ② \<REF\>\<SEG\> | 47.8 | 64.9 | 加辅助分割 |
| ③ + Pooling | 47.5 | 66.3 | 加 pooled 特征 |
| ④ Object Memory Bank | **49.0** | **68.5** | 解耦设计最优 |

### 关键发现

- **引用与分割的互增强效应**：联合训练引用和分割能力使两个任务都有提升（分割从 47.5→48.2，引用 QA 从 64.6→67.4）
- **3B 模型超越 7-13B 竞品**：在 ReVOS 上，3B 的 UniPixel 超越所有 7B-13B 参数的竞品，说明统一设计比简单增大模型更有效
- **时间编码至关重要**：去掉提示编码器中的时间编码后 𝒥&ℱ 从 49.0 降至 44.3
- **Ref-SAV 大幅领先**：在复杂的长视频数据集 Ref-SAV 上，UniPixel-3B 达到 67.2 𝒥&ℱ，远超 Sa2VA-8B 的 41.3（不做微调）

## 亮点与洞察

- **首个端到端统一引用+分割的方法**：通过 Object Memory Bank 的优雅设计，避免了外部帧采样器、掩码生成器或目标追踪器
- **PixelQA 新任务**：提出了需要同时完成引用、分割和 QA 的新范式，弥合了像素级感知和语言推理的鸿沟
- **对象级测试时扩展**：可以被视为一种 object-centric test-time scaling 方法——先分割关键对象再编码以辅助推理
- **记忆库的解耦设计**：解决了因果自注意力限制下 `<SEG>` token 无法获取完整上下文的根本问题

## 局限与展望

- 7B 版本在 PixelQA 分割质量上反而低于 3B（𝒥&ℱ 44.6 vs 60.9），可能存在大模型下分割能力退化的问题
- 推理分割数据 (ReasonSeg) 仅 239 样本，容易被大规模数据淹没
- 掩码传播依赖 SAM 2.1，对极端运动或遮挡的鲁棒性受限于外部模块

## 相关工作与启发

- 与 LISA（开创 LMM 分割范式）相比，UniPixel 统一了引用和分割，避免了刚性模板
- 与 Sa2VA（同用 SAM2 解码器）相比，UniPixel 的 Object Memory Bank 提供了更强的对象感知能力
- Object Memory Bank 的预填充-注入机制可推广到其他需要对象级推理的多模态任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ Object Memory Bank 设计新颖，PixelQA 任务定义有前瞻性
- **实验充分度**: ⭐⭐⭐⭐⭐ 10 个基准、9 个任务全面覆盖，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，研究问题引导式叙述
- **价值**: ⭐⭐⭐⭐⭐ 统一框架有广泛应用前景，PixelQA 开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)

</div>

<!-- RELATED:END -->

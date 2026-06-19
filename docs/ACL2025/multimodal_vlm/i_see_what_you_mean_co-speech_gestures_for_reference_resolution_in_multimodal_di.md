---
title: >-
  [论文解读] I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue
description: >-
  [ACL 2025][多模态VLM][表征性手势] 提出自监督预训练方法学习表征性共语手势（co-speech iconic gestures）的嵌入表示，将骨骼动作 grounded 到语言中，在面对面对话的指称消解任务上证明手势与语音的互补性——手势+语音准确率 31% 远超单独语音 24% 或手势 19%。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "表征性手势"
  - "指称消解"
  - "多模态对话"
  - "自监督预训练"
  - "手势表示学习"
---

# I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue

**会议**: ACL 2025  
**arXiv**: [2503.00071](https://arxiv.org/abs/2503.00071)  
**代码**: [https://github.com/EsamGhaleb/ReferenceResolution](https://github.com/EsamGhaleb/ReferenceResolution)  
**领域**: 多模态理解 / 人机交互  
**关键词**: 表征性手势, 指称消解, 多模态对话, 自监督预训练, 手势表示学习

## 一句话总结
提出自监督预训练方法学习表征性共语手势（co-speech iconic gestures）的嵌入表示，将骨骼动作 grounded 到语言中，在面对面对话的指称消解任务上证明手势与语音的互补性——手势+语音准确率 31% 远超单独语音 24% 或手势 19%。

## 研究背景与动机

**领域现状**：指称消解（reference resolution）在对话中主要研究文本/语言表达如何指向对象。在非言语方面，指示性手势（pointing）和节奏性手势（beat gestures）已有一些计算研究，但**表征性手势**（iconic gestures，如用手模拟物体形状）几乎未从计算角度研究。

**现有痛点**：(1) 表征性手势不像 pointing 那样有明确的方向信息，其语义需要更复杂的理解；(2) 缺少学习手势表示的有效方法——手势数据稀缺且标注成本高；(3) 多模态人机交互系统没有利用这一重要沟通渠道。

**核心矛盾**：认知科学研究表明表征性手势帮助听者更快识别所指对象，但计算方法无法有效建模这种手势-对象的映射关系。

**本文目标**：(1) 如何学习表征性手势的鲁棒嵌入？(2) 手势在指称消解中的作用有多大？与语音是互补还是冗余？

**切入角度**：利用手势与共现语音的语义关联（手势和对应语音描述同一对象），通过自监督对比学习将骨骼动作 grounded 到语言空间。

**核心 idea**：多模态预训练（骨骼+语音/文本语义）学到的手势表示，即使推理时没有语音输入，也比纯骨骼表示更具区分力。

## 方法详解

### 整体框架
分两阶段：(1) **预训练阶段**：在 CABB 数据集上用自监督方法学习手势嵌入（骨骼+语言对齐）；(2) **下游任务**：用预训练的手势表示做指称消解（70 类物体子部分分类）。

### 关键设计

1. **三种模态编码器**:

    - **骨骼编码器**：改造 DSTFormer（时空 Transformer），保留时间→空间和空间→时间两个并行分支，各保留一层，第二层替换为可选的跨模态注意力模块
    - **语音编码器**：多语言 wav2vec-2（xlsr-300），用可学习加权平均聚合所有 Transformer 层 + 两层 CNN 融合时间维度
    - **语义编码器**：Dutch BERTje 提取词嵌入，提供更丰富的语义信息

2. **三种预训练架构**:

    - **Unimodal**：仅用骨骼——masked reconstruction loss + unimodal contrastive loss（两个增强视角的对比）
    - **Multimodal**：骨骼 + 语音/语义——增加 CLIP 式多模态对比损失，将手势全局表示和共现语音/文本映射到共同空间
    - **Multimodal-X**（最强）：在 Multimodal 基础上加 cross-attention——将语义 tokens 作为 key/value 注入骨骼编码器的跨注意力层，再加一个 crossmodal contrastive loss 对齐纯骨骼表示与融合表示
    - 设计动机：Multimodal-X 通过 cross-attention 实现更细粒度的时间对齐，且对齐纯骨骼表示使得推理时无需语音输入也能受益于预训练时的语言 grounding

3. **指称消解模型**:

    - 功能：给定手势嵌入，预测手势指称的 70 个物体子部分之一
    - 核心思路：两层 MLP（300, 150），直接在冻结的预训练表示上训练，leave-one-round-out 交叉验证
    - 两种推理场景：(1) 仅手势输入（测试预训练表示质量）；(2) 手势+共现语音拼接（测试多模态互补性）

### 损失函数 / 训练策略
- Unimodal loss = masked reconstruction + unimodal contrastive
- Multimodal loss = unimodal losses + CLIP-style multimodal contrastive
- Multimodal-X loss = multimodal losses + crossmodal contrastive（对齐纯骨骼 vs 融合表示）
- 预训练数据：CABB-XL（~400k 样本），过采样 1 秒窗口覆盖 >50% 自动分割的手势

## 实验关键数据

### 主实验（指称消解准确率，70 类，随机 1.4%）

| 输入模态 | 模型 | 准确率 |
|---------|------|--------|
| 骨骼 only | Unimodal | 16% |
| 骨骼 only | Multimodal (语义预训练) | 19% |
| 骨骼 only | Multimodal-X (语义预训练) | ~19% |
| 语义 only (BERTje) | - | 24% |
| 骨骼 + 语义 | Unimodal 骨骼 + BERTje | ~27% |
| 骨骼 + 语义 | **Multimodal-X + BERTje** | **31%** |

### 消融实验（预训练表示质量，Spearman 相关性 with 专家标注）

| 模型 | CABB-L | CABB-XL |
|------|--------|---------|
| Unimodal (骨骼) | ~0.32 | ~0.33 |
| Multimodal (语音) | ~0.35 | ~0.34 |
| Multimodal (语义) | ~0.34 | ~0.37 |
| Multimodal-X (语义) | ~0.34 | **~0.38** |
| Ghaleb et al. 2024 (ST-GCN) | ~0.28 | ~0.29 |

### 关键发现
- **多模态预训练表示"免费"提升**：Multimodal-X 预训练后，即使推理时只输入骨骼，准确率从 16%→19%（+18.8%），语言 grounding 信息被"蒸馏"进了骨骼表示中
- **手势与语音互补而非冗余**：语音 only 24%，手势+语音 31%（+29.2%），说明表征性手势携带了语音未传达的信息
- **对话历史促进手势消解**：随着对话轮次增加，利用历史手势的 dialogue-specific 模型准确率持续上升（手势 entrainment 效应），且多模态预训练模型受益更大
- **语义嵌入优于原始语音**：使用 BERTje 文本语义作为对齐目标比 wav2vec-2 语音效果更好，特别是在大数据量（CABB-XL）训练时

## 亮点与洞察
- **"训练时用语音，推理时不用"**：通过多模态预训练将语言知识注入骨骼表示，实现推理时零语音依赖的性能提升。这种预训练范式可迁移到其他"训练时多模态、推理时单模态"的场景
- **表征性手势的计算化研究**：填补了指示性/节奏性手势之外、语义最丰富的表征性手势的计算研究空白
- **Cross-attention 融合 > late fusion**：Multimodal-X 的 cross-attention 比 Multimodal 的 CLIP 式对比学习提供了更细粒度的时间对齐

## 局限与展望
- 仅在荷兰语任务导向对话数据集（CABB）上验证，跨语言/跨任务/开放域泛化未知
- 70 类物体子部分的分类任务仍然偏简单，真实场景中候选对象更多
- 使用 2D 骨骼（ViTPose），3D 骨骼可能提供更好的空间信息
- 预训练数据虽过采样到 400k，但相比 NLP/CV 领域的预训练数据量仍很小
- 未将手势表示链接到对象的视觉属性（如形状、尺寸），这可能进一步提升指称消解

## 相关工作与启发
- **vs Abzaliev et al. 2022**: 他们从 TED 演讲学习手势-词嵌入，但关注非表征性手势（功能词/discourse markers）。本文专注表征性手势并用于指称消解
- **vs Ghaleb et al. 2024b**: 前作用 ST-GCN + 语音对比学习。本文用 Transformer + cross-attention + 语义嵌入，相关性从 ~0.29 提升到 ~0.38
- **vs vision-language 指称消解**: ReferIt/Visual Genome 研究文本→图像区域映射，本文研究手势→对象映射，是不同但互补的方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 表征性手势的计算化研究非常新颖，多模态预训练框架设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融全面，对比了多种架构和训练数据量，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，实验设计严谨，统计检验完整
- 价值: ⭐⭐⭐⭐ 为多模态人机交互提供了新的研究方向和实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding](../../CVPR2026/multimodal_vlm/see_what_i_mean_aligning_vision_and_language_representations_for_video_fine-grai.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](../../ACL2026/multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->

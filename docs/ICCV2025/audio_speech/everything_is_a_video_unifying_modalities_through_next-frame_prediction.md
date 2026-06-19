---
title: >-
  [论文解读] Everything is a Video: Unifying Modalities through Next-Frame Prediction
description: >-
  [ICCV 2025][音频/语音][多模态统一] 本文将多模态学习中的文本、图像、音频、视频等不同模态任务统一重构为下一帧预测问题（所有输入输出都渲染为 64×64 视频帧序列），用单一 Transformer 模型无需模态特定编码器即可处理跨模态任务，验证了"everything is a video"这一激进但可行的统一表征范式。
tags:
  - "ICCV 2025"
  - "音频/语音"
  - "多模态统一"
  - "下一帧预测"
  - "任务重构"
  - "视频生成"
  - "模态统一表征"
---

# Everything is a Video: Unifying Modalities through Next-Frame Prediction

**会议**: ICCV 2025  
**arXiv**: [2411.10503](https://arxiv.org/abs/2411.10503)  
**代码**: 无  
**领域**: 音频语音  
**关键词**: 多模态统一, 下一帧预测, 任务重构, 视频生成, 模态统一表征

## 一句话总结

本文将多模态学习中的文本、图像、音频、视频等不同模态任务统一重构为下一帧预测问题（所有输入输出都渲染为 64×64 视频帧序列），用单一 Transformer 模型无需模态特定编码器即可处理跨模态任务，验证了"everything is a video"这一激进但可行的统一表征范式。

## 研究背景与动机

**领域现状**：多模态学习需要整合文本、图像、音频、视频等信息来完成视觉问答、跨模态检索、字幕生成等任务。当前主流方案依赖模态特定编码器（如 ViT 处理图像、Transformer 处理文本）加后期融合，需要为每种模态专门设计编码器并处理模态间对齐。

**现有痛点**：模态特定编码器的设计限制了可扩展性和灵活性——每增加一种新模态就需要设计新的编码器和融合策略；不同模态的表征空间不同，跨模态知识迁移困难；架构复杂度随模态数量增长。

**核心矛盾**：NLP 领域已经实现了"统一接口"的范式——所有 NLP 任务都可以重构为文本生成问题（prompt-based learning），这使得单一 LLM 能处理翻译、摘要、问答等多种任务。但多模态领域尚未实现类似的统一——不同模态仍然需要不同的处理方式。

**本文目标**：将 NLP 的任务重构思想扩展到多模态领域，找到一种能统一所有模态的"超级任务"（supertask），使单一模型不需要任何模态特定组件就能处理各种多模态任务。

**切入角度**：作者观察到文本可以渲染为图像帧（每个 token 一帧），音频可以转换为频谱图，因此所有模态理论上都可以无损地转换为视觉帧序列。

**核心 idea**：将所有多模态任务重构为下一帧预测问题——输入和输出统一表示为 64×64 RGB 视频帧序列，用分隔符帧区分输入和输出，使模型只需学习预测下一帧即可处理跨模态任务。

## 方法详解

### 整体框架

任务重构 + 视频预测模型。首先将各模态任务统一编码为视频帧序列：文本 token 渲染为单帧图像（固定宽度字体 64×64），音频转为频谱图帧，图像缩放到 64×64，视频保持原始帧序列。输入帧 + 分隔符帧 + 输出帧构成完整序列，模型自回归预测下一帧。推理时最后生成的帧通过 OCR 或直接使用得到答案。

### 关键设计

1. **模态统一重构（Modality-Unified Reformulation）**:

    - 功能：将不同模态的输入输出统一编码为 64×64 RGB 视频帧序列
    - 核心思路：文本 → 每个 token 渲染为一帧（固定宽度字体，填满 64×64）；图像 → 缩放到 64×64 作为一帧或多帧；音频 → 转换为频谱图作为一帧；视频 → 直接使用原始帧（可能下采样）。每个任务序列格式为 [输入帧...] [分隔符帧] [输出帧...]。例如 SST2 情感分类：[每个词渲染为帧] [|] [positive/negative渲染为帧]；CIFAR-10：[64×64图像] [|] [类别名帧]
    - 设计动机：文本和音频都可以无信息损失地转换为视觉表示（文本渲染、频谱图），因此视频帧序列是一种理论上可以涵盖所有模态的通用表征。分隔符帧让模型清楚地知道输入结束、需要开始生成输出

2. **空间-时间 Transformer 视频预测模型**:

    - 功能：自回归地预测视频序列的下一帧
    - 核心思路：输入视频帧被分割为 8×8 不重叠的 patch，线性嵌入后加空间和时间位置编码。模型采用 U-Net 风格的编码器-解码器结构：编码侧通过 patch merge 操作逐步降低分辨率，在最低分辨率处使用全局空间-时间 Transformer，解码侧通过 patch unmerge 恢复分辨率，并有 skip connection。时间注意力使用因果掩码确保自回归性质。嵌入维度 $K=512$
    - 设计动机：U-Net 风格设计允许在低分辨率处进行全局注意力（降低计算成本），同时通过 skip connection 保留高分辨率细节。因果时间注意力确保预测只依赖过去帧

3. **跨模态知识迁移机制**:

    - 功能：通过共享的帧级表征，实现不同模态间的隐式知识迁移
    - 核心思路：由于所有模态共享同一个视觉输入空间（64×64 帧），且模型没有任何模态特定组件，模型在学习一种模态任务时获得的帧级预测能力可以自然迁移到其他模态。例如学习文本渲染帧→帧的规律有助于其他涉及文本输出的任务
    - 设计动机：这正是统一表征的核心优势——不同模态共享表征空间消除了跨模态对齐问题，模型可以在一个统一的框架内积累不同任务的通用知识

### 损失函数 / 训练策略

使用 Multi-Scale Structural Similarity（MS-SSIM）损失训练，学习率 $3 \times 10^{-4}$，AdamW 优化器，dropout 0.1，batch size 8-32（根据序列长度调整）。每个任务独立训练（单任务），无预训练权重，所有模型在单张 A100 上训练最多 7 GPU 天。

## 实验关键数据

### 主实验

| 任务 | 数据集 | OCR F1/Acc | 其他指标 | 对比方法 |
|------|--------|-----------|---------|---------|
| 文本分类 | SST-2 | 76.8 / 75.5 | - | SOTA 91.3（有预训练），无预训练基线接近 |
| 图像分类 | CIFAR-10 | 89.1 / 89.1 | - | ViT+预训练 99.5，PCANet 77.1 |
| 视频分类 | TinyVIRAT | 30.4*(macro F1) | - | ResNet50 29.1，WideResNet 32.6 |
| 音频分类 | AudioMNIST | 96.9 / 97.1 | - | AlexNet 95.82 |
| 视频问答 | CLEVRER | 52.4 / 52.5 | - | LSTM 34.7，LSTM+CNN 51.8 |
| 目标追踪 | LaSOT | - | IoU 0.63 | 直接追踪，表现稳定 |
| 视频着色 | TinyVIRAT | - | CDC 0.0169, 色彩丰富度 73.1 | 原始数据 70.6 |

### 消融/分析

| 分析维度 | 发现 |
|---------|------|
| 文本截断影响 | SST-2 限制 ≤20 tokens 后 F1 提升到 80.0 |
| 注意力可视化 | 空间注意力聚焦于关键目标/文字，时间注意力关注信息量大的帧 |
| 音频错误分析 | 最常见混淆：four vs five（语音相似） |
| 追踪退化 | 长序列末段边界框侵蚀，因自回归像素误差累积 |
| 着色权衡 | 色彩多样性高于真值（73.1 vs 70.6）但时间一致性较差 |

### 关键发现

- **音频分类表现最好**：97.1% 准确率超越 AlexNet 基线，说明频谱图作为视觉表征对音频分类非常有效
- **无预训练也能达到可用水平**：所有任务从零训练，多数接近或超越无预训练的基线方法，验证了重构范式的可行性
- **注意力分析证实了跨模态理解**：模型在情感分类中关注情感关键词（nightmare, painful），在 CLEVRER 中关注问题关键词（color, metal）和物体轨迹
- **主要瓶颈在于文本输出的 OCR 解码**：大量训练时间花在学习如何输出可读文本上，预训练可以解决这个问题

## 亮点与洞察

- **极致的统一思想**：把"所有东西都是视频"推到了极端——文本一个字一帧、音频变频谱图。虽然看起来"暴力"，但论证了一个重要观点：视觉帧序列确实可以作为跨模态的通用表征，无信息损失
- **无模态特定组件的简洁性**：模型中没有任何专为某种模态设计的组件——同一个 Transformer 处理文本帧和图像帧。这种彻底的统一性是其他多模态模型（FLAVA、GPT-4V）没有达到的
- **为多模态基础模型提供了理论支撑**：如果下一帧预测可以作为跨模态的"超级任务"，那么类似 LLM 在 NLP 中的角色，一个足够大的视频预测模型理论上可以成为跨所有模态的基础模型

## 局限与展望

- 当前是单任务分别训练，未验证多任务联合训练的效果——联合训练是真正实现"统一模型"的关键
- 64×64 分辨率限制了处理细粒度任务的能力（如小目标检测、长文本理解）
- 文本渲染→OCR 解码的管线引入了额外误差，且 OCR 本身不完美
- 没有使用任何预训练——论文承认大量训练时间花在学习"如何输出文字"上，预训练可以大幅提效
- 未来方向：在大规模多任务数据上预训练、提升分辨率、增加更多模态（如触觉、点云）

## 相关工作与启发

- **vs UNITER/UniT**: UNITER 用 RNN 预处理图像再输入 Transformer，UniT 用独立编码器+共享解码器。本文完全没有模态特定组件，是更彻底的统一
- **vs data2vec**: data2vec 将多模态投射到共享隐空间。本文直接将多模态转换为同一视觉输入空间，是在"输入侧"而非"表征侧"做统一
- **vs GPT-4V/FLAVA**: 这些基础模型也通过 ViT 处理视觉输入，但文本和视觉仍然走不同的处理流程。本文展示了在模型层面可以做到完全统一

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "everything is a video"的统一范式非常大胆新颖，从根本上重新思考了模态表征
- 实验充分度: ⭐⭐⭐ 覆盖 7 种任务，但均为简单基准，且无多任务联合训练实验
- 写作质量: ⭐⭐⭐⭐ 思路清晰，各任务重构方式描述详细
- 价值: ⭐⭐⭐⭐ 概念验证价值高，为多模态统一基础模型提供了新方向，但实际性能还有很大提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](../../ICML2025/audio_speech/ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)
- [\[ICML 2026\] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](../../ICML2026/audio_speech/group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](../../ICML2025/audio_speech/flam_frame-wise_language-audio_modeling.md)
- [\[ACL 2025\] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](../../ACL2025/audio_speech/clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)

</div>

<!-- RELATED:END -->

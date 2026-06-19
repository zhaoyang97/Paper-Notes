---
title: >-
  [论文解读] Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator
description: >-
  [ICCV 2025][人体理解][手语生成] 提出 SOKE，一种基于预训练语言模型的多语言手语生成框架，通过解耦式 tokenizer 将连续手语动作离散化为 token 序列，结合多头解码和检索增强策略，实现从文本到多语种 3D 手语 avatar 的高质量生成。 手语是聋人和听障群体的主要交流方式…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "手语生成"
  - "多语言手语"
  - "自回归语言模型"
  - "检索增强生成"
  - "动作离散化"
---

# Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator

**会议**: ICCV 2025  
**arXiv**: [2411.17799](https://arxiv.org/abs/2411.17799)  
**代码**: [项目主页](https://2000zrl.github.io/soke/)  
**领域**: 人体理解  
**关键词**: 手语生成, 多语言手语, 自回归语言模型, 检索增强生成, 动作离散化

## 一句话总结

提出 SOKE，一种基于预训练语言模型的多语言手语生成框架，通过解耦式 tokenizer 将连续手语动作离散化为 token 序列，结合多头解码和检索增强策略，实现从文本到多语种 3D 手语 avatar 的高质量生成。

## 研究背景与动机

手语是聋人和听障群体的主要交流方式，具备自然语言的所有语言学特征（离散语义单元、语法结构）。手语处理分为两大方向：手语翻译（Sign-to-Text, SLT）和手语生成（Text-to-Sign, SLG）。虽然预训练语言模型在 SLT 上已取得显著成功，但在 SLG 方向仍**严重不足**。

现有 SLG 方法的主要问题：

**忽略手语的语言学本质**：多数方法将 SLG 视为视觉内容生成任务（视频/关键点/运动），使用 GAN 或扩散模型，未能利用预训练 LM 的泛化性和可扩展性

**gloss 依赖**：传统方法依赖 gloss（手语的书写形式）作为中间表示，但 gloss 需要大量标注且造成信息瓶颈

**单语言局限**：现有方法通常只处理一种手语（如 ASL），缺乏统一的多语言模型

**解码效率低**：使用解耦 tokenizer 时，将多个身体部位的 token 展平为单一序列，导致解码步数成倍增加

**词级精度不足**：句子级生成难以保证每个手语词的精确性

这些问题共同导致了现有 SLG 方法在质量、效率和覆盖范围上的不足。

## 方法详解

### 整体框架

SOKE 由两个阶段组成：(1) **解耦式 Tokenizer (DETO)** — 将连续手语动作离散化为身体各部位的 token；(2) **自回归多语言生成器 (AMG)** — 基于预训练 mBART 模型，从文本输入生成动作 token 序列。训练完成后，文本输入 → LM 编码器 → 多头解码器生成各部位 token → 部分解码器重建 3D 手语动作。

### 关键设计

1. **解耦式 Tokenizer（DETO）**：手语具有多信道特性——语义同时由身体动作和手部动作传达。DETO 使用三个独立的 VQ-VAE 分别建模上半身（$B$）、左手（$LH$）、右手（$RH$），将每部分的 SMPL-X 参数映射为独立的离散 token 序列。量化过程为：

$$\hat{z}_i^p = Q(s_{f,i}^p) = \arg\min_{z_j \in \mathbf{Z}^p} \|s_{f,i}^p - z_j\|_2$$

Codebook 大小设置为：身体 $N_Z^B = 96$，左右手 $N_Z^{LH} = N_Z^{RH} = 192$，code 维度 $C = 512$。解耦设计利用了手语的运动学先验，使每个部位可以独立精确建模。

2. **多头解码策略（Multi-Head Decoding）**：这是本文的核心创新之一。现有方法有两种解码方式：(a) **顺序解码**将所有部位 token 展平为 $3K$ 长度序列，逐一预测，效率极低；(b) **并行解码**完全独立地解码三个部位，缺乏信息融合。SOKE 提出的多头解码在每个时间步使用三个独立的 LM head 同时预测三个部位的 token，同时输入嵌入为各部位 token 嵌入的加权平均：

$$\mathbf{E} = (1-2\lambda)\mathbf{E}^B + \lambda\mathbf{E}^{LH} + \lambda\mathbf{E}^{RH}$$

其中 $\lambda = 1/3$。这实现了条件独立假设下的高效融合——解码步数从 $3K$ 降为 $K$，同时通过加权输入嵌入保留了部位间的信息交互。推理延迟从 3.26s/视频降至 1.46s/视频。

3. **检索增强手语生成（Retrieval-Enhanced SLG）**：受 RAG 在 NLP 中的成功启发，SOKE 利用外部手语词典（从孤立手语识别数据集构建）提供精确的词级手语作为辅助条件。具体流程：(a) 构建词典：将 RGB 视频转为 SMPL-X 姿态，用 DETO 离散化为 token 四元组 $\{(w, m^B, m^{LH}, m^{RH})\}$；(b) 给定输入文本，检索所有匹配词并获取其动作 token；(c) 将原始文本 token 与检索到的动作 token 拼接后送入 LM 编码器。这种方式避免了 gloss-based 方法直接拼接词典手语导致的共发音不自然问题，因为模型学会了在词典手语的条件下自然地生成连贯的句子级动作。

### 损失函数 / 训练策略

- **DETO 阶段**：标准 VQ-VAE 损失 $\mathcal{L}_{vq}^p = \mathcal{L}_{rec}^p + \mathcal{L}_{emb}^p + \mathcal{L}_{com}^p$，包括重建损失、嵌入损失和承诺损失
- **AMG 阶段**：标准交叉熵损失 $\mathcal{L}_{LM} = -\log P(\mathbf{Y}|\mathbf{h}_{en})$
- DETO 在合并的多语言数据集 + 词典数据上训练 500 epochs，AdamW，cosine scheduler，lr=2e-4
- AMG 使用 mBART-large-cc25 作为骨干，微调 150 epochs
- 推理时使用贪心解码，任意一个 head 预测到 EOS 即终止
- 训练使用 6 张 RTX 3090 GPU

## 实验关键数据

### 主实验

**跨三个手语数据集的 DTW-PA-JPE（手部）对比**

| 方法 | 多语言? | How2Sign ↓ | CSL-Daily ↓ | Phoenix-2014T ↓ |
|------|---------|------------|-------------|-----------------|
| NSA（扩散模型） | × | 7.33 | — | — |
| S-MotionGPT | × | 4.39 | 3.78 | 3.41 |
| **SOKE（本文）** | **✓** | **2.35** | **1.71** | **1.38** |

**改进幅度**（相对 S-MotionGPT）：How2Sign -46.5%，CSL-Daily -54.8%，Ph14T -59.5%

SOKE 在 Back-Translation BLEU-4 上也显著领先：14.48 vs 11.45（How2Sign）

### 消融实验

| 解码方法 | 检索增强 | How2Sign DTW↓ | 延迟(s/视频) | 说明 |
|----------|----------|---------------|--------------|------|
| Sequential | × | 4.64 | 3.26 | 展平所有 token |
| Parallel | × | 5.06 | 1.39 | 完全独立解码 |
| Multi-head | × | 4.17 | 1.46 | 本文方法（无检索） |
| Multi-head | ✓ | **3.34** | 1.55 | 本文完整方法 |

### 关键发现

- 多头解码相比顺序解码将延迟降低 55%（3.26→1.46s），同时 DTW 误差降低 10%
- 检索增强将 DTW 误差进一步降低 20%（4.17→3.34），且仅增加 0.09s 延迟
- 并行解码虽然快但效果最差，证明部位间信息融合的必要性
- 单一统一模型同时处理 ASL/CSL/DGS 三种手语，无需为每种语言单独训练
- 手部动作的离散化质量至关重要——手部 codebook 需比身体大（192 vs 96）

## 亮点与洞察

- **将手语视为"语言"而非"视频"**的建模范式是核心洞察——手语的离散结构天然适合语言模型方法
- 多头解码在效率和质量之间取得了优雅的平衡，条件独立假设在实践中工作良好
- 检索增强策略巧妙地将词级精度和句子级自然性统一——不是简单拼接词典手语，而是作为条件送入 LM
- 构建了 CSL-Daily 和 Phoenix-2014T 的 SMPL-X 标注，填补了多语言 3D 手语数据的空白

## 局限与展望

- 检索依赖于手语词典的覆盖率——对于不在词典中的罕见词汇可能效果有限
- SMPL-X 模型的面部表情参数较少（10个），而面部表情在手语中承载重要语义
- mBART-large 的模型规模（约 600M）对于边缘设备部署可能偏大
- 当前仅使用贪心解码，beam search 或采样策略可能进一步提升质量
- 未评估实际聋人用户对生成手语的理解率

## 相关工作与启发

- **T2S-GPT** 和 **MotionGPT** 是基于 tokenizer-LM 的先驱工作，但仅支持单语言且使用耦合 tokenizer
- **RAG** 在 NLP 的成功直接启发了检索增强 SLG 的设计
- 启发：多头解码的思路可推广到其他多通道动作生成任务（如舞蹈生成中的上下肢协调）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多头解码和检索增强 SLG 均为首创，多语言统一模型有意义
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集、多种基线、完整消融、定性+定量分析齐全
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示直观，方法表述规范
- **价值**: ⭐⭐⭐⭐⭐ 对手语无障碍技术有重要社会价值，开源数据和代码利于社区

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[ECCV 2024\] A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](../../ECCV2024/human_understanding/a_simple_baseline_for_spoken_language_to_sign_language_trans.md)
- [\[CVPR 2025\] Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](../../CVPR2025/human_understanding/lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)
- [\[CVPR 2026\] SignPR: A Progressive Vector-Quantized Diffusion Framework for Sign Language Production](../../CVPR2026/human_understanding/signpr_a_progressive_vector-quantized_diffusion_framework_for_sign_language_prod.md)

</div>

<!-- RELATED:END -->

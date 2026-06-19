---
title: >-
  [论文解读] Goldfish: Vision-Language Understanding of Arbitrarily Long Videos
description: >-
  [ECCV 2024][视频理解][长视频理解] 提出 Goldfish 框架，通过将长视频分割为短 clip 并利用基于文本相似度的检索机制选取与问题最相关的 top-k 片段，实现对任意长度视频的高效理解，同时提出 MiniGPT4-Video 短视频模型和 TVQA-long 长视频评测基准。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "长视频理解"
  - "检索增强生成"
  - "视觉语言模型"
  - "视频问答"
  - "MiniGPT4-Video"
---

# Goldfish: Vision-Language Understanding of Arbitrarily Long Videos

**会议**: ECCV 2024  
**arXiv**: [2407.12679](https://arxiv.org/abs/2407.12679)  
**代码**: [有 (GitHub)](https://github.com/Vision-CAIR/Goldfish)  
**领域**: 视频理解  
**关键词**: 长视频理解, 检索增强生成, 视觉语言模型, 视频问答, MiniGPT4-Video

## 一句话总结

提出 Goldfish 框架，通过将长视频分割为短 clip 并利用基于文本相似度的检索机制选取与问题最相关的 top-k 片段，实现对任意长度视频的高效理解，同时提出 MiniGPT4-Video 短视频模型和 TVQA-long 长视频评测基准。

## 研究背景与动机

当前基于 LLM 的视频理解模型主要面向短视频（分钟级），处理长视频（如电影、电视剧）时面临三大核心挑战：

**噪声与冗余**：类似 NLP 领域的 "needle in a haystack" 问题，LLM 在过长上下文中容易忽略关键信息。长视频中大量片段与问题无关，尤其在空间和时间分辨率压缩后，模型更难提取有意义内容。

**计算与内存限制**：视频越长，处理所需的计算和内存开销越大。现有视频 LLM（如 Video-LLaMA、Video-ChatGPT）对可处理的最大视频长度存在固有限制。

**缺乏有效的长视频评测基准**：现有长视频基准（如 LLaMA-VID）主要通过将电影摘要和脚本输入语言模型来生成问题，忽略了视觉内容，导致问题仅依赖文本即可回答。

已有尝试如 MovieChat 通过记忆整合模块、LLaMA-VID 通过将每帧压缩为两个 token 来扩展上下文窗口，但这些压缩策略导致空间和时间视觉细节丢失，且仍使用全视频特征预测答案，无法有效过滤噪声。本文的关键洞察是：**准确识别与查询相关的视频片段是理解长视频的关键**，由此提出基于检索的框架来解决上述所有挑战。

## 方法详解

### 整体框架

Goldfish 框架包含三个核心模块：(1) Video Descriptor（视频描述器），(2) Retrieval Module（检索模块），(3) Answer Module（回答模块）。工作流程为：将长视频切分为短 clip → 对每个 clip 生成详细描述 → 根据用户查询检索最相关的 top-k clip → 将检索结果送入回答模块生成最终响应。

### 关键设计

1. **Video Descriptor（MiniGPT4-Video）**：

    - **功能**：将长视频分割为不重叠的短 clip，为每个 clip 生成详细文本描述并编码为嵌入向量
    - **核心思路**：将视频帧序列 $V = \{v_1, v_2, \ldots, v_T\}$ 分成 $m$ 个 chunk，每个 chunk 最多包含 $L$ 帧（由 MiniGPT4-Video 上下文窗口决定）。使用 EVA-CLIP 编码每帧，通过投影层将视觉特征映射到 LLM 文本空间，并将每 4 个相邻视觉 token 拼接为 1 个（从 256 降至 64 token/帧，减少 75%）。生成的描述 $S_1, S_2, \ldots, S_m$ 及对应字幕通过文本编码器（OpenAI text-embedding-3-small）编码为嵌入 $\{T_{s_i}\}$ 和 $\{T_{u_i}\}$
    - **设计动机**：需要一个能处理多帧输入的短视频模型来为每个 clip 提供高质量的语义描述，同时 token 压缩使得在有限上下文窗口内可容纳更多帧

2. **Retrieval Module（检索模块）**：

    - **功能**：根据用户查询从所有 clip 中检索最相关的 top-k 个片段
    - **核心思路**：将用户查询 $Q$ 编码为 $T_Q \in \mathbb{R}^d$，计算其与所有 clip 描述及字幕嵌入的余弦相似度：$\frac{K_i \cdot T_Q}{|K_i||T_Q|}$，其中 $K_i \in \{T_{u_1}, \ldots, T_{u_m}, T_{s_1}, \ldots, T_{s_m}\}$，选择相似度最高的 top-k 个 clip
    - **设计动机**：通过检索机制过滤无关内容，仅保留与问题相关的信息，有效解决噪声冗余和计算成本问题，使框架可扩展到任意长度视频

3. **Answer Module（回答模块）**：

    - **功能**：融合检索到的 clip 信息和原始查询生成最终答案
    - **核心思路**：将原始用户查询与检索到的 clip 描述（及字幕）作为上下文输入 Llama2-chat 生成最终回答。消融实验表明直接将描述+字幕输入 LLM（方案 A，准确率 41.78%）优于重新过视频模型获取新描述（方案 B/C，约 27.6%），因为后者会产生幻觉
    - **设计动机**：避免重复处理视频帧导致的幻觉问题，直接利用已有高质量描述

### 训练策略

MiniGPT4-Video 采用三阶段训练：
1. **大规模图文预训练**：在 LAION、Conceptual Captions、SBU 上训练线性投影层，对齐视觉和 LLM 文本空间
2. **大规模视频文本预训练**：使用 CMD 和 WebVid 数据集，每视频最多采样 45 帧，学习理解短视频
3. **视频问答指令微调**：使用 Video-ChatGPT 数据集的高质量 QA 对进行指令微调

## 实验关键数据

### 主实验

**长视频基准对比（Table 3）**：

| 数据集 | 方法 | 模态 | Acc.↑ | Score↑ |
|--------|------|------|-------|--------|
| TVQA-Long | LLaMA-VID | V | 24.63 | 2.16 |
| TVQA-Long | **Ours** | **V** | **28.61** | **2.78** |
| TVQA-Long | LLaMA-VID | V+T | 26.81 | 2.21 |
| TVQA-Long | **Ours** | **V+T** | **41.78** | **3.21** |
| Movie QA | LLaMA-VID | V | 24.42 | 2.19 |
| Movie QA | **Ours** | **V** | **28.49** | **2.80** |
| MovieChat | LLaMA-VID | V | 53.2 | 3.81 |
| MovieChat | **Ours** | **V** | **67.6** | **4.23** |

**短视频基准对比（Table 5）**：

| 数据集 | 本文 (Ours-7B) | LLaMA-VID-7B | Video-ChatGPT | 提升 |
|--------|---------------|-------------|----------------|------|
| MSVD (Acc.) | **72.93** | 69.7 | 64.9 | +3.23% |
| MSRVTT (Acc.) | **58.83** | 57.7 | 49.3 | +2.03% |
| TGIF (Acc.) | **67.9** | — | 51.4 | +16.5% |
| TVQA (Acc.) | **36.45** | — | 23.35 | +23.59% |

### 消融实验

**检索模块重要性**：

| 配置 | TVQA-Long Acc. | 说明 |
|------|---------------|------|
| 无检索（直接下采样45帧） | 25.07% | 接近随机猜测（5选1基线20%） |
| **有检索（top-k）** | **41.78%** | 检索带来 +16.71% 绝对提升 |

**检索输入消融（Table 1）**：

| 检索输入 | TVQA | TVR-Text | TVR-Vision |
|----------|------|----------|------------|
| 仅字幕 | **39.7** | 66.4 | 48.4 |
| 仅摘要 | 12.1 | 41.2 | 51.2 |
| 字幕 \| 摘要 | 39.5 | **67.2** | **50.8** |

**文本编码器消融（Table 2）**：

| 文本编码器 | 检索Acc. | 总体Acc. |
|-----------|---------|---------|
| bert-base-nli | 19.0 | 28.4 |
| paraphrase-MiniLM | 31.9 | 38.03 |
| all-mpnet-base-v2 | 32.5 | 38.33 |
| **OpenAI-text-embedding-3-small** | **46.6** | **41.78** |

### 关键发现

- 检索精度与最终准确率呈线性正相关，更好的文本编码器直接带来更好的整体性能
- 字幕对文本类问题更有效，视频摘要对视觉类问题更有效，两者结合（"or" 方式）效果最佳
- 框架在不同视频长度（6/12/24分钟）下保持鲁棒，检索和整体准确率不随视频时长增加而显著下降

## 亮点与洞察

- **RAG 思想迁移到视频理解**：将文本领域的检索增强生成（RAG）思路创造性地应用于长视频理解，是一个优雅的工程方案
- **模块化设计的灵活性**：Video Descriptor (MiniGPT4-Video) 既可作为 Goldfish 的组件，也可独立用于短视频任务，且短视频性能同样达到 SOTA
- **实用价值高**：框架可处理任意长度的视频，支持电影、电视剧等真实场景

## 局限与展望

- 检索依赖文本相似度，对纯视觉问题（无文本线索）的检索效果可能不佳
- 回答模块使用 Llama2-chat 而非视频模型，无法利用检索到的视频帧的视觉信息（消融实验表明直接用视频帧反而下降）
- clip 在 TVQA-Long 上准确率为 41.78%，虽超过 SOTA 14.94%，但绝对值仍有较大提升空间
- 未验证 clip 分割粒度和 top-k 设置对不同类型问题的影响

## 相关工作与启发

- 与 MovieChat（记忆整合）和 LLaMA-VID（token 压缩）相比，Goldfish 选择"检索"而非"压缩"的路线，在保留信息完整性方面更有优势
- MiniGPT4-Video 的三阶段作，延续并改进了 MiniGPT-v2 的架构设计
- 检索与生成分离的架构便于各模块独立升级

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 RAG 应用于长视频理解的思路有新意，但整体是工程组合创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个长视频基准+5个短视频基准，消融实验全面覆盖检索输入、文本编码器、回答模块、视频长度
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述到位，但部分公式符号略显冗余
- **价值**: ⭐⭐⭐⭐ — 实用价值高，框架简洁有效，对长视频理解研究有重要参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](../../CVPR2025/video_understanding/rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[CVPR 2025\] BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](../../CVPR2025/video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)

</div>

<!-- RELATED:END -->

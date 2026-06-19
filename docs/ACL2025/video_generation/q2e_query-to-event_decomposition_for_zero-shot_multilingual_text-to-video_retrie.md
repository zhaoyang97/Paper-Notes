---
title: >-
  [论文解读] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval
description: >-
  [ACL 2025][视频生成][文本到视频检索] Q2E 提出了一种零样本的查询到事件分解方法，利用 LLM 和 VLM 的参数化世界知识将简单查询分解为前因/当前/后果事件，并结合视频的视觉描述和语音转录，通过逆熵融合排序实现 SOTA 的多语言文本到视频检索性能。 问题背景 文本到视频检索是一个重要的多媒体任务…
tags:
  - "ACL 2025"
  - "视频生成"
  - "文本到视频检索"
  - "事件分解"
  - "零样本"
  - "LLM知识迁移"
  - "多模态融合"
---

# Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval

**会议**: ACL 2025  
**arXiv**: [2506.10202](https://arxiv.org/abs/2506.10202)  
**代码**: [有](https://dipta007.github.io/Q2E/)  
**领域**: 视频生成  
**关键词**: 文本到视频检索, 事件分解, 零样本, LLM知识迁移, 多模态融合

## 一句话总结

Q2E 提出了一种零样本的查询到事件分解方法，利用 LLM 和 VLM 的参数化世界知识将简单查询分解为前因/当前/后果事件，并结合视频的视觉描述和语音转录，通过逆熵融合排序实现 SOTA 的多语言文本到视频检索性能。

## 研究背景与动机

### 问题背景

文本到视频检索是一个重要的多媒体任务，但存在几个核心挑战：

**查询过于简洁**：用户通常输入简短查询（如"2025 LA fire"），但期望系统理解事件的各个方面

**信息分散在多个视频中**：单个视频可能只展示事件的一部分

**多语言障碍**：视频可能不是用户所知的语言

### 核心动机

现有视频检索系统通常依赖平台元数据或人工标注数据（如标题、搜索优化描述），无法处理复杂的真实世界事件查询。经典数据集如 MSR-VTT 和 MSVD 只包含通用、高层级查询（如"a person is explaining something"），而非复杂事件。

### 三个关键洞察

1. 通过 LLM 查询分解可以增强对粗粒度查询的理解，检索到原本会被忽略的前因/后果相关视频
2. VLM 字幕和 ASR 输出虽然有噪声和重复，但 LLM 精炼器可以有效去噪
3. 从多种相似度/相关性判断中聚合排名时，基于熵的融合方法优于简单方法

## 方法详解

### 整体框架

Q2E 系统（图 2）包含四个核心模块：

1. **事件分解模块（蓝色部分）**：将查询分解为前因/当前/后果事件
2. **视频分解模块（绿色部分）**：从视频中提取多模态描述
3. **音频分解模块（橙色部分）**：多层翻译管道处理语音
4. **融合排名模块（紫色部分）**：逆熵融合所有评分

### 关键设计

1. **事件分解（Event Decomposition）**：

    - 使用 LLM 将查询分解为三种子事件：**前因（Prequel）**——可能导致当前事件的前置事件；**当前（Current）**——事件发生时可观察到的具体子事件；**后果（Sequel）**——事件可能产生的后续结果
    - 每种类型生成 5 个子事件
    - 使用同一 LLM 提取时间、空间和主体事件信息进行**分解精炼**，生成更自然的查询

2. **视频分解（Video Decomposition）**：

    - **上下文化帧描述**：均匀采样 16 帧，使用滑动窗口（window=2）的 VLM 为每帧生成条件于前一帧描述的上下文化字幕
    - **视频描述**：将所有 16 帧描述输入 LLM，总结为单个视频密集字幕，保留时序信息并关注整体画面

3. **音频分解（Audio Decomposition）**：多层翻译管道——

    - 第一层：Whisper-v3 多语言 ASR（转录原始语言+英文翻译）
    - 第二层：NLLB 翻译器（从原始转录翻译为英文）
    - 第三层：Llama-70B 精炼器（精炼两种英文翻译结果）

4. **评分与融合**：

    - 计算 5 种评分：(a) 查询 vs 视频、(b-d) 前因/当前/后果 vs 多模态描述、(e) 查询 vs 多模态描述
    - 查询-视频使用 MultiCLIP 图像编码器的余弦相似度
    - 查询-描述使用 ColBERT 文本相似度（优于 SBERT，因 ColBERT 在 token 级最大聚合可降低噪声影响）
    - 事件-描述使用 many-to-many 全局最大相似度
    - **逆熵融合排序**：将每种评分转为 softmax 分布，低熵=高置信度，用逆熵加权融合：$\hat{S} = \sum_{i}^{5} \frac{1}{H(P_i)} \cdot P_i$

### 训练策略

- **完全零样本**：不对任何模型进行微调，利用现有 LLM/VLM 的参数化知识
- 方法跨数据集、领域、LLM 和 VLM 均可适配

## 实验关键数据

### 主实验（Table 1，NDCG 指标）

| 数据集 | 编码器 | Baseline | +Event | +ASR+Event |
|--------|--------|----------|--------|------------|
| MultiVENT | MultiCLIP | 75.34 | 80.04 | **83.24** |
| MultiVENT | InternVideo2-1B | 50.43 | 69.15 | **76.10** |
| MSR-VTT-1kA | MultiCLIP | 59.72 | 61.51 | **63.59** |
| MSR-VTT-1kA | InternVideo2-1B | 66.07 | 67.16 | **69.53** |
| MSVD | MultiCLIP | 71.69 | **74.10** | - |
| MSVD | InternVideo2-1B | 77.51 | **77.84** | - |

### 消融实验

| 融合方法 | NDCG↑ |
|---------|-------|
| Neg. Exp. Entropy | 73.20 |
| RRF | 76.29 |
| Max | 80.04 |
| Mean | 82.44 |
| **Inv. Entropy (Q2E)** | **83.24** |

| LLM 大小 | NDCG↑ |
|----------|-------|
| Baseline (无分解) | 75.34 |
| 1B | 82.50 |
| 3B | 83.03 |
| 8B | 82.91 |
| **70B** | **83.24** |

### 关键发现

1. **事件分解有效**：仅添加事件分解（不含 ASR），MultiVENT NDCG 提升 5-19 点；加上 ASR 额外提升 3-7 点，总提升 8-26 点
2. **跨语言一致提升**：阿拉伯语、中文、韩语等低资源语言提升更大（分别 +6、+9、+10 NDCG）
3. **小模型也有效**：即使 1B 参数的 LLM 也能比基线提升至少 8 NDCG 点
4. **逆熵融合最优**：超过 Mean、Max、RRF 等方法
5. **各组件互补**：移除视频评分影响最大（-9.28 NDCG），其次是事件（-1.49）和查询（-1.70）

## 亮点与洞察

- **事件因果知识迁移**：创造性地利用 LLM 的世界知识将简单查询展开为前因/当前/后果事件结构，这是一种新颖的查询增强范式
- **多层 ASR 管道**：三层处理（ASR +翻译+精炼）有效解决了多语言视频的语音质量问题
- **零样本 plug-and-play**：无需微调，可直接替换底层编码器（MultiCLIP/InternVideo2），实用性强
- **全局最大策略**：对事件-描述匹配使用全局最大而非平均，有效降低 LLM 幻觉的负面影响

## 局限与展望

- **计算开销大**：需运行多个大模型（LLM、VLM、ASR、翻译器），推理时间和成本高
- **LLM 幻觉风险**：事件分解和字幕精炼过程中可能产生虚假信息
- **偏见传播**：依赖模型参数化知识，可能传播模型内在偏见
- **仅关注文本到视频方向**：未探索视频到文本检索
- 未来可探索如何利用事实和反事实信息进行正/负对齐

## 相关工作与启发

- 延续了事件因果推理在 NLP 中的应用，将时序因果推理（前因/后果）首次应用于视频检索领域
- 受 Yin & Jiang (2024) 启发使用逆熵融合
- 对 RAG 系统有参考意义：查询分解+多源信息融合的范式可迁移到其他检索场景
- 多层翻译管道思想可应用于其他多语言多模态任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 事件分解+逆熵融合是新颖组合，将 LLM 世界知识创造性地迁移到视频检索
- **实验充分度**: ⭐⭐⭐⭐⭐ 3 个数据集、2 个编码器、5 种语言、多种消融，极为详尽
- **写作质量**: ⭐⭐⭐⭐ 框架清晰，图示直观，但部分公式排版略显拥挤
- **价值**: ⭐⭐⭐⭐ 零样本方法在事件密集型视频检索上大幅提升，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](../../CVPR2025/video_generation/zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](../../CVPR2025/video_generation/identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[CVPR 2025\] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](../../CVPR2025/video_generation/video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](../../CVPR2025/video_generation/conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)

</div>

<!-- RELATED:END -->

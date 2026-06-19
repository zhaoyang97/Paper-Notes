---
title: >-
  [论文解读] Two Causally Related Needles in a Video Haystack
description: >-
  [NeurIPS 2025][视频理解][长视频理解] 提出Causal2Needles基准（4,100个问答对），通过设计"桥接实体"将两个因果相关事件的理解绑定在一起，强制VLM必须联合检索和推理两个分散在长视频中的"针"，揭示现有最强模型在因果双针问题上的严重不足（ChatGPT-4o双针Both准确率仅13.4%）。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "长视频理解"
  - "因果推理"
  - "needle-in-haystack"
  - "视频语言模型"
  - "benchmark"
---

# Two Causally Related Needles in a Video Haystack

**会议**: NeurIPS 2025  
**arXiv**: [2505.19853](https://arxiv.org/abs/2505.19853)  
**代码**: [项目页面](https://limiaoyu.github.io/Causal2Needles)  
**领域**: 视频理解 / 因果推理  
**关键词**: 长视频理解, 因果推理, needle-in-haystack, 视频语言模型, benchmark

## 一句话总结

提出Causal2Needles基准（4,100个问答对），通过设计"桥接实体"将两个因果相关事件的理解绑定在一起，强制VLM必须联合检索和推理两个分散在长视频中的"针"，揭示现有最强模型在因果双针问题上的严重不足（ChatGPT-4o双针Both准确率仅13.4%）。

## 研究背景与动机

**领域现状**: 长视频理解基准不断涌现，但大多只评估单针信息提取能力（从视频中一个位置提取答案），或仅通过视觉外观匹配来追踪对象。

**现有痛点**: (1) NLP研究已表明模型在多针问题上表现远差于单针问题，但多模态领域缺乏系统的多针评估；(2) 现有基准对VLM是否具备"世界模型"的评估仅限于物理运动预测，忽略了人类行为因果推理；(3) 叙事文本输入可能导致"文本偏见"——模型直接从文本回答而不真正理解视频。

**核心矛盾**: 单针问题上表现优异的模型是否真正理解了长视频的因果关系，还是仅仅在做局部信息检索？

**本文目标**: 构建一个同时评估双针联合理解能力和因果推理能力的长视频理解基准。

**切入角度**: 利用电影摘要视频中的因果事件对，通过"桥接实体"设计强制模型必须先检索效果事件才能定位原因事件。

**核心 idea**: 用模糊化的桥接实体将双针问题变成不可分解的联合推理任务，防止退化为两个独立的单针问题。

## 方法详解

### 整体框架

Causal2Needles基于192个电影摘要视频（YMS和SyMoN数据集）构建，包含四类问题：(1) 非因果单针（1,704题）——询问事件细节；(2) 因果单针（902题）——从效果推因，但答案在单一位置；(3) 因果双针-视觉定位格式（747题）——要求选择包含答案的视频片段；(4) 因果双针-图像描述格式（747题）——要求描述因事件视频的视觉细节。整个构建流程：LLM提取因果关系 → 生成桥接实体 → 生成双部分问题 → 自动+人工质量评估。

### 关键设计

1. **桥接实体驱动的双针问题设计**:

    - 功能：设计一种问题结构，强制模型必须联合理解原因事件和效果事件
    - 核心思路：识别因果事件对共享的桥接实体（如"Superman之死"），在问题中用模糊表述替代（如"tragedy"）。Part 1要求从效果事件解析桥接实体内容，Part 2要求基于解析结果检索原因事件对应的视频片段
    - 设计动机：如果桥接实体被明确写出，双针问题会退化为可独立回答的单针问题——模糊化是确保联合推理的关键

2. **双互补问答格式（Visual Grounding + Image Description）**:

    - 功能：设计两种互补的问题格式来抵消各自的评估偏差
    - 核心思路：视觉定位格式要求选择正确视频片段，防止纯文本作答但可能OOD；图像描述格式要求回答视觉细节的多选题，规避OOD但可能受预训练知识影响。同时使用两种格式综合评估
    - 设计动机：单一格式会高估或低估模型能力——视觉定位可能低估（OOD），图像描述可能高估（电影知识泄露）

3. **全局+局部因果关系提取**:

    - 功能：用LLM从视频叙事中提取全面的因果关系图
    - 核心思路：全局图从完整叙事提取长距离因果关系，滑动窗口（15句、步长5）提取局部图捕获细粒度关系，合并后仅保留距离$\geq 3$个事件的因果对
    - 设计动机：LLM对长文本中部内容注意力不足（"lost in the middle"），局部窗口弥补遗漏

### 损失函数 / 训练策略

纯评估基准，不涉及训练。问题生成采用LLM（GPT-4o-mini, Gemini-2.0-flash），质量评估由ChatGPT-4.1和Gemini-2.0-flash及5名人工标注员完成。桥接实体共享存在率达95%+，问题事实正确性达4.6+/5分。

## 实验关键数据

### 主实验

VLM在Causal2Needles上的准确率（%，取Forward/Reverse平均）：

| 模型 | 非因果1针 | 因果1针 | VG 2针Both | ID 2针 |
|------|:---------:|:-------:|:----------:|:------:|
| Human | - | 78.2 | 79.3 | 88.2 |
| ChatGPT-4o | 56.8 | 39.2 | 13.4 | 59.2 |
| Gemini-1.5-pro | 55.4 | 35.6 | 8.4 | 60.9 |
| ChatGPT-4o-mini | 39.9 | 33.4 | 5.2 | 52.3 |
| Qwen2.5VL-32B | 30.7 | 11.7 | 1.9 | 53.5 |
| LLaVA-OneVision-7B | 12.3 | 18.0 | 0.1 | 28.3 |

### 消融实验

因果性与双针距离的影响：

| 分析维度 | 发现 |
|---------|------|
| 非因果→因果（1针） | ChatGPT-4o: 56.8% → 39.2%（因果推理显著更难） |
| 1针→2针Both | ChatGPT-4o: 39.2% → 13.4%（双针联合推理极大下降） |
| 距离关联 | 双针距离越远，性能越差（负相关显著） |
| 正序vs逆序 | 多数模型在正序输入上更佳，但差异因模型而异 |

### 关键发现

1. **因果推理远难于非因果检索**: 所有模型从非因果到因果均大幅下降
2. **双针联合理解是真正瓶颈**: 最强模型ChatGPT-4o的VG双针Both准确率仅13.4%，远低于单针
3. **开源模型几乎完全失败**: Qwen2.5VL-32B在双针Both上仅1.9%
4. **图像描述格式普遍显著高于视觉定位格式**: 可能因为模型在回答关于外观的多选题时可利用预训练知识

## 亮点与洞察

- **桥接实体的设计巧妙**: 通过模糊化一个参考确保双针不可拆分——这是评估联合推理的核心创新
- **揭示"高分陷阱"**: 模型可能在现有基准上取得高分但完全不具备因果推理能力
- **双互补格式的实验设计**: 同时控制OOD偏差和知识泄露偏差
- **数据集质量极高**: 自动+人工评估显示>95%的桥接实体有效，4.6+/5事实正确性

## 局限与展望

- 视频来源为电影摘要（经过剪辑），可能不完全代表自然长视频
- 叙事文本作为辅助输入可能引入难以完全消除的文本偏见
- 人类基线评估规模较小（5人标注员）
- 因果关系提取本身依赖LLM，可能遗漏某些隐性因果链
- 仅评估了有限的VLM模型，更多架构（如视频原生模型）待验证

## 相关工作与启发

- **vs VideoMME/MLVU**: 这些基准包含多针但针之间仅视觉关联、可独立理解——Causal2Needles要求因果联合理解
- **vs EgoSchema**: 无诊断类别标签——Causal2Needles精准隔离单针vs双针、因果vs非因果
- **启发**: "能做检索≠能做推理"——长视频理解的下一个挑战是从信息提取到因果推理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 桥接实体驱动的双针因果推理设计在长视频基准中独树一帜
- 实验充分度: ⭐⭐⭐⭐ 覆盖10+模型、4类问题、正逆序对比、质量评估全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机论证有力
- 价值: ⭐⭐⭐⭐⭐ 揭示现有VLM在联合推理上的根本性弱点，高度有临床诊断价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](../../CVPR2026/video_understanding/rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)
- [\[CVPR 2026\] Asynchronous Temporal Modeling with Two-Agent Framework for Streaming Dense Video Captioning](../../CVPR2026/video_understanding/asynchronous_temporal_modeling_with_two-agent_framework_for_streaming_dense_vide.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)

</div>

<!-- RELATED:END -->

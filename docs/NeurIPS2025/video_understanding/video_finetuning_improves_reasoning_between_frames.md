---
title: >-
  [论文解读] Video Finetuning Improves Reasoning Between Frames
description: >-
  [NeurIPS 2025][视频理解][视频微调] 本文通过提出视觉思维链（vCoT）方法，系统地比较了图像LLM与视频微调LLM在帧间推理能力上的差异，发现视频微调使模型隐式学会了帧间过渡推理，且这种能力可迁移到静态图像的关系推理任务中。 多模态大语言模型（LLM）在视觉理解方面取得了显著进展，但从图像扩展到视频理解时…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "视频微调"
  - "多模态大语言模型"
  - "帧间推理"
  - "视觉思维链"
  - "时序理解"
---

# Video Finetuning Improves Reasoning Between Frames

**会议**: NeurIPS 2025  
**arXiv**: [2511.12868](https://arxiv.org/abs/2511.12868)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频微调, 多模态大语言模型, 帧间推理, 视觉思维链, 时序理解

## 一句话总结

本文通过提出视觉思维链（vCoT）方法，系统地比较了图像LLM与视频微调LLM在帧间推理能力上的差异，发现视频微调使模型隐式学会了帧间过渡推理，且这种能力可迁移到静态图像的关系推理任务中。

## 研究背景与动机

多模态大语言模型（LLM）在视觉理解方面取得了显著进展，但从图像扩展到视频理解时，大多数方法仍然采用朴素的逐帧token拼接策略，缺乏真正的时序理解能力。这导致模型在需要推理帧间隐式过渡的任务上表现不佳，往往只能依赖表面的视觉线索。

视频LLM通过额外的视频数据微调和时序位置编码（如RoPE）等归纳偏置来增强视频理解，但一个核心问题始终未得到系统回答：**视频微调到底为模型带来了什么？它在多大程度上增强了模型超越图像模型的推理能力？**

本文的核心切入点是：如果视频微调确实让模型学会了帧间推理，那么显式提供帧间过渡描述对视频模型的增益应该很小（因为模型已经隐式学会了），而对图像模型的增益应该很大（因为图像模型缺少这种能力）。基于这一假设，作者设计了vCoT来验证这一机制。

## 方法详解

### 整体框架

作者的研究框架是一个比较实验设计：选取结构相同的图像LLM和视频LLM对（如LLaVA-NeXT vs LLaVA-NeXT-Video），唯一区别是是否经过视频微调。通过在有无vCoT条件下比较两类模型的性能变化，揭示视频微调的实际作用。

### 关键设计

1. **视觉思维链（vCoT）生成**: vCoT的核心思路是为相邻帧之间生成显式的文本过渡描述。具体分两步：

    - **Step 1 — 共同视觉属性识别**: 给模型展示两帧图像，询问"这两张图有什么共同点？"，让模型识别出共享的场景元素（如物体、背景、空间配置），建立跨帧的稳定上下文。
    - **Step 2 — 桥接事件推理**: 基于两帧和识别出的共同元素，提示模型推理可能发生在两帧之间的中间事件（如"这个人把球踢向房子"）。为保持简洁，使用Qwen-2.5模型对描述进行精简改写。

2. **模态扰动实验（Modality Shuffling）**: 为了区分模型对视觉和文本线索的依赖程度，设计了两种扰动：

    - **视觉扰动**: 将每个视频帧替换为不相关视频的帧，保持文本infill不变。
    - **文本扰动**: 保持原始帧不变，但用其他视频的文本infill替换。
   通过观察不同模型对两种扰动的敏感度，揭示视频模型与图像模型的模态依赖差异。

3. **静态图像推理迁移实验**: 在i-RAVEN基准上测试视频模型是否能将帧间推理能力迁移到非时序的关系推理任务中。RAVEN是一个抽象视觉推理任务（类似智力测试中的渐变矩阵），要求从一组面板中推断抽象规则并选择正确的补全。

### 训练策略

本文不涉及新的训练方法，而是一个分析性研究。使用的模型对包括：
- LLaVA-NeXT（图像） vs LLaVA-NeXT-Video（视频）
- InternVL-Image vs InternVL-Video
所有模型对共享相同的视觉编码器、语言骨干和跨模态投影器。

## 实验关键数据

### 主实验：EgoSchema上的vCoT效果

| 模型 | 帧数 | 基线准确率 | +vCoT准确率 | 提升 |
|------|------|-----------|------------|------|
| LLaVA-NeXT (图像) | 5 | 44.0% | 51.4% | **+7.4%** |
| LLaVA-NeXT-Video (视频) | 5 | 47.0% | 48.6% | +1.6% |
| LLaVA-NeXT (图像) | 10 | 49.2% | 55.4% | **+6.2%** |
| LLaVA-NeXT-Video (视频) | 10 | 49.0% | 51.4% | +2.4% |
| InternVL-Image | 5 | 38.4% | 40.4% | +2.0% |
| InternVL-Video | 5 | 44.6% | 42.4% | **-2.2%** |
| InternVL-Image | 10 | 37.4% | 42.6% | **+5.2%** |
| InternVL-Video | 10 | 45.8% | 49.0% | +3.2% |

### 模态扰动消融实验

| 模型 | 帧数 | vCoT基线 | 视觉扰动 | 文本扰动 |
|------|------|---------|---------|---------|
| LLaVA-NeXT (图像) | 5 | 51.4% | 39.8% (-11.6) | 42.0% (-9.4) |
| LLaVA-NeXT-Video (视频) | 5 | 48.6% | 41.6% (-7.0) | 47.0% (-1.6) |
| LLaVA-NeXT (图像) | 10 | 55.4% | 51.8% (-3.6) | 45.0% (-10.4) |
| LLaVA-NeXT-Video (视频) | 10 | 51.4% | 46.4% (-5.0) | 45.4% (-6.0) |

### i-RAVEN静态推理迁移结果

| 模型 | center | dist_4 | dist_9 | in/out | indist4/out | L/R | U/D | 平均 |
|------|--------|--------|--------|--------|-------------|-----|-----|------|
| InternVL-Image | 14.8 | 14.4 | 15.2 | 11.6 | 13.2 | 15.2 | 14.4 | 14.1 |
| InternVL-Video | 15.6 | 16.0 | 15.8 | 13.8 | **17.0** | 14.0 | 14.2 | **15.2** |
| LLaVA-Image | 7.0 | 8.0 | 15.0 | 7.0 | 9.0 | 12.0 | 14.0 | 10.3 |
| LLaVA-Video | 7.0 | 14.0 | 16.0 | 8.0 | **13.0** | 14.0 | **21.0** | **13.3** |

### 关键发现

- vCoT对图像模型提升显著（最高+7.4%），对视频模型提升有限甚至为负——说明视频微调使模型隐式学会了帧间过渡推理
- 视频模型对文本噪声的鲁棒性显著优于图像模型——视频模型更依赖视觉信息而非文本
- 视频模型在i-RAVEN上的表现优于图像模型——帧间推理能力可迁移到静态关系推理

## 亮点与洞察

- 提出了一种巧妙的实验设计来"探测"视频微调的内在效果：通过观察显式帧间描述的边际效益来推断模型是否已经隐式学会了帧间推理
- vCoT本身也是一种实用的图像模型增强手段——对于无法使用视频模型的场景，可以通过vCoT来弥补帧间推理能力的不足
- 发现视频微调带来的推理能力具有迁移性，可以从动态视频迁移到静态图像的关系推理中
- 扰动实验揭示了视频模型和图像模型在模态依赖上的本质差异

## 局限与展望

- 实验仅在EgoSchema（第一人称视频）上进行，缺乏更多样化的视频理解基准的验证
- 使用的模型规模相对较小（7B级别），未验证在更大模型上是否仍有类似发现
- vCoT的生成本身依赖一个较强的VLM，引入了额外的计算开销
- i-RAVEN上的绝对准确率仍然很低（均<21%），说明当前VLM在抽象推理上仍有很大提升空间
- 未探索不同的视频微调策略（如不同数据量、不同任务类型）对帧间推理能力的影响

## 相关工作与启发

- 该工作与CoT思维链系列工作一脉相承，将思维链的概念从纯文本推理扩展到视觉时序推理
- 与Video-ChatGPT等工作不同，本文不关注如何提升性能，而是关注"理解"视频微调为何有效
- 启发：在评估视频模型时，可以用vCoT增益作为一种"探针"来衡量模型的隐式时序推理能力
- 对多模态模型设计有重要启示——视频微调不仅提升了视频任务性能，还增强了通用的关系推理能力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 实验设计巧妙，vCoT作为探测工具的思路新颖
- 实验充分度: ⭐⭐⭐ — 基准和模型覆盖有限，但控制变量设计严谨
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，故事讲述流畅
- 价值: ⭐⭐⭐⭐ — 对理解视频微调机制有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](../../CVPR2025/video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)

</div>

<!-- RELATED:END -->

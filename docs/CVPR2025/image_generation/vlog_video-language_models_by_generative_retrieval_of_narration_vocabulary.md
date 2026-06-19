---
title: >-
  [论文解读] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary
description: >-
  [CVPR 2025][图像生成][视频理解] 提出 VLog，将视频叙事（narration）定义为词汇表单元，通过生成式检索架构（GPT-2 推理 + SigLIP 检索）实现比生成式 VideoLLM 快 10-20 倍的高效视频理解。 现有 VideoLLM 继承了 LLM 的子词词汇表（如 LLaMA-3 的 12…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频理解"
  - "生成式检索"
  - "叙事词汇表"
  - "视频语言模型"
  - "高效推理"
---

# VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary

**会议**: CVPR 2025  
**arXiv**: [2503.09402](https://arxiv.org/abs/2503.09402)  
**代码**: [GitHub](https://github.com/showlab/VLog)  
**领域**: 图像生成  
**关键词**: 视频理解, 生成式检索, 叙事词汇表, 视频语言模型, 高效推理

## 一句话总结

提出 VLog，将视频叙事（narration）定义为词汇表单元，通过生成式检索架构（GPT-2 推理 + SigLIP 检索）实现比生成式 VideoLLM 快 10-20 倍的高效视频理解。

## 研究背景与动机

现有 VideoLLM 继承了 LLM 的子词词汇表（如 LLaMA-3 的 128K 词汇，包含大量无视觉意义的子词如 'happ'）和逐 token 自回归解码方式，导致推理速度慢，难以实时处理视频流。

实际应用（如 AR 眼镜助手）更需要简洁、上下文相关的实时响应，而非详尽描述。人类在回忆日常活动时，自然地将经验组织为一系列叙事事件（如"关闭闹钟"、"洗碗"），形成行为"词汇表"。

**核心问题**: 如何构建以叙事为最小单元的词汇表替代子词词汇表，同时保留 LLM 的推理能力？检索模型（CLIP）可支持灵活的词汇更新但缺乏推理能力，生成模型推理强但解码慢。如何结合二者优势？

## 方法详解

### 整体框架

VLog 基于轻量级 GPT-2-medium 和 SigLIP 构建。核心创新包括：(1) 生成式检索架构——在 GPT-2 序列末尾引入检索 token，融合视觉和查询信息后与词汇嵌入做点积检索；(2) 叙事对编码（NPE）构建层次化词汇表，支持前缀+后缀两级检索；(3) 基于 LMM+LLM 的 agent 工作流实现词汇表自动扩展。

### 关键设计1: 生成式检索架构

**功能**: 结合生成模型的推理能力和检索模型的效率与灵活性。

**核心思路**: 在 GPT-2 语言模型的输入序列末尾添加一个检索 token $\mathbf{t}$，它通过自注意力机制关注前面的视觉输入和查询输入。经过 GPT-2 后，输出嵌入 $\tilde{\mathbf{t}}$ 编码了视觉和查询信息，用于与词汇嵌入做点积检索：$\Pr(\mathcal{X} = \tilde{o_i} | \mathcal{V}, \mathcal{Q}) = \tilde{\mathbf{t}}^T \tilde{\mathbf{o}_i}$。词汇嵌入由 SigLIP 预计算并缓存，无需经过 GPT-2，形成不对称结构降低计算量。

**设计动机**: 纯检索模型（SigLIP）缺乏推理能力，无法回答"下一个动作是什么？"等因果查询。纯生成模型逐 token 解码太慢。通过检索 token 桥接两者，既保留 GPT-2 的因果推理能力，又实现叙事级别的一步检索。

### 关键设计2: 叙事对编码（NPE）与层次化索引

**功能**: 从大规模叙事数据构建结构化词汇表并实现高效检索。

**核心思路**: 类似 BPE 的分词思想，将叙事分解为前缀集（核心动作如"切土豆"）和后缀集（修饰信息如"用左手"）。检索时先通过场景层级（如"厨房"）缩小前缀搜索范围，再匹配后缀。形成三级层次：场景 → 前缀叙事子集 → 后缀。

**设计动机**: 百万级词汇表暴力搜索不可行。人类活动天然与场景关联（"切土豆"在厨房），层次化索引将搜索空间压缩数个数量级。前缀+后缀分离让词汇表更紧凑且表达力更强。

### 关键设计3: 词汇表自动扩展

**功能**: 在推理时处理未见过的新事件。

**核心思路**: 当检索 token 与最佳匹配词汇的相似度低于阈值 0.4 时，判定为 OOV 事件。启动 agent 工作流：(1) 用 LLaVA-OV-0.5B 生成视觉场景描述；(2) 用 Qwen2.5-0.5B 根据场景描述推理可能的事件，解析为新词汇条目。这是一种"生成增强检索"（Generative-Augmented Retrieval）范式。

**设计动机**: 初始词汇表无论多大都无法覆盖所有新场景。检索模型的优势在于词汇嵌入独立于模型权重（由 SigLIP 直接编码），添加新词汇无需重新训练。

### 损失函数

标准对比学习损失：$\mathcal{L} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \log \frac{\exp(\tilde{\mathbf{t}}_i^T \tilde{\mathbf{o}_i}/\tau)}{\sum_{j \in \mathcal{B}} \exp(\tilde{\mathbf{t}}_i^T \tilde{\mathbf{o}_j}/\tau)}$，温度 $\tau=0.05$。

## 实验关键数据

### 主实验结果 (Vidcab-Eval 上的检索性能)

| 方法 | CIDEr(Naive) | R@1(Naive) | CIDEr(Causal) | R@1(Causal) | 解码时间(s) |
|------|-------------|-----------|--------------|------------|-----------|
| Generative GPT2 | 64.8 | 7.9 | 53.7 | 3.1 | 0.362 |
| Retrieval (FT) | 95.8 | 11.8 | 48.9 | 2.1 | 0.016 |
| **VLog** | **96.9** | **12.4** | **87.3** | **5.0** | **0.018** |

### COIN 基准测试 (动作感知)

| 方法 | 模型大小 | Step Acc | Task Acc | Next Acc |
|------|---------|----------|----------|----------|
| VideoLLM-online | 7B | 59.8 | 92.1 | 48.1 |
| GPT2 (生成式) | 355M | 44.6 | 82.4 | 32.1 |
| **VLog** | 355M | 56.1 | 93.0 | 46.0 |
| **VLog+Ego4D预训练** | 355M | **57.4** | **94.4** | **48.4** |

### 关键发现

1. **因果检索远超对手**: 在 Causal 设定下（需要推理"之前/之后"关系），VLog CIDEr=87.3 远超检索模型的 48.9 和生成模型的 53.7，证明生成式检索有效融合了推理与检索。
2. **20x 加速**: VLog 解码时间 0.018s vs 生成模型 0.362s，接近纯检索模型速度。
3. **轻量模型媲美大模型**: 355M 参数的 VLog 在 COIN 上性能与 7B VideoLLM-online 相当。
4. **词汇表可迁移**: Ego4D 预训练词汇成功迁移到 COIN 数据集，提升所有指标。

## 亮点与洞察

- **范式创新**: "叙事即词汇"的思想将视频理解从逐 token 生成转变为叙事级检索，从根本上解决速度瓶颈。
- **架构优雅**: 检索 token 作为生成与检索之间的桥梁，设计简洁，不对称结构避免重复计算词汇嵌入。
- **Generative-Augmented Retrieval**: 与 RAG 相反的新范式，用生成模型扩展检索词汇。

## 局限与展望

- **封闭词汇假设**: 仍依赖预定义词汇表，复杂开放场景描述能力受限。
- **Ego4D 偏向**: 词汇主要来自第一人称视频，第三人称场景适用性未充分验证。
- **后缀表达力有限**: 后缀集合无法捕捉所有细粒度差异，如具体数量、颜色等属性。
- 未来可探索词汇表持续学习、多模态词汇、与大 LLM 集成等方向。

## 相关工作与启发

- **CLIP 检索**: SigLIP 提供了灵活的开放词汇嵌入能力，但缺乏推理。VLog 通过检索 token 补全了推理能力。
- **BPE 分词**: NPE 将 BPE 的子词分解思想扩展到叙事级别，启发如何构建领域特定的词汇表。
- **启发**: "检索 token"的设计可推广到其他需要结合推理和检索的多模态任务。

## 评分

⭐⭐⭐⭐ — 问题定义新颖（叙事词汇表替代子词），生成式检索架构设计优雅，20x 加速效果显著。轻量模型媲美大模型的效果令人印象深刻。但封闭词汇假设限制了开放世界适用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[ECCV 2024\] IRGen: Generative Modeling for Image Retrieval](../../ECCV2024/image_generation/irgen_generative_modeling_for_image_retrieval.md)
- [\[CVPR 2025\] Font-Agent: Enhancing Font Understanding with Large Language Models](font-agent_enhancing_font_understanding_with_large_language_models.md)

</div>

<!-- RELATED:END -->

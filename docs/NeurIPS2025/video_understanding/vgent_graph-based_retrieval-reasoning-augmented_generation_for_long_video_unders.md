---
title: >-
  [论文解读] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding
description: >-
  [NeurIPS 2025 Spotlight][视频理解][graph RAG] 提出 VGEnt，一个基于图的检索-推理增强生成框架，通过构建视频知识图谱保留跨片段语义关系，并引入结构化推理步骤过滤噪声、聚合信息，在多个长视频理解基准上一致提升开源 LVLM 3.0%~5.4%，超越现有视频 RAG 方法 8.6%。
tags:
  - "NeurIPS 2025 Spotlight"
  - "视频理解"
  - "graph RAG"
  - "structured reasoning"
  - "retrieval-augmented generation"
  - "video language model"
---

# VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.14032](https://arxiv.org/abs/2510.14032)  
**代码**: [GitHub](https://xiaoqian-shen.github.io/Vgent)  
**领域**: Video Understanding  
**关键词**: long video understanding, graph RAG, structured reasoning, retrieval-augmented generation, video language model

## 一句话总结

提出 VGEnt，一个基于图的检索-推理增强生成框架，通过构建视频知识图谱保留跨片段语义关系，并引入结构化推理步骤过滤噪声、聚合信息，在多个长视频理解基准上一致提升开源 LVLM 3.0%~5.4%，超越现有视频 RAG 方法 8.6%。

## 研究背景与动机

长视频理解面临的核心挑战：

**上下文窗口限制**：30 分钟视频可超过 200K tokens，超出大多数模型上下文限制。现有方法通过稀疏采样或 token 压缩应对，但不可避免地丢失细粒度时间信息

**传统 RAG 的局限**：朴素 RAG 将视频切成独立片段检索，破坏了实体的连续性和时序依赖；约 40% 的失败案例中正确片段已被检索到，但模型仍给出错误答案——因为无关信息的干扰

**对闭源模型的依赖**：VideoAgent、DrVideo 等方法依赖 GPT-4 做多轮交互，成本高且不灵活

## 方法详解

### 整体框架

VGEnt 包含四个阶段：(1) 离线视频图构建；(2) 基于图的检索；(3) 结构化推理；(4) 多模态增强生成。整个流程无需训练，可直接应用于任意开源 LVLM。

### 关键设计

1. **视频知识图构建 (Video Graph Construction)**：

    - 将视频按每 $K=64$ 帧切成片段，每个片段作为图中一个节点
    - 利用 LVLM 从每个片段提取关键实体（主体、动作、场景）及其描述
    - 通过文本嵌入的相似度计算（阈值 $\tau=0.7$）进行跨片段实体合并：语义等价的实体统一为同一实体，包含相同实体的节点之间建立边
    - 图构建是离线且查询无关的：一旦建好可复用于同一视频的多个问题

2. **基于图的检索 (Graph-based Retrieval)**：

    - 从用户问题中提取关键词 $\mathcal{K}$
    - 计算关键词与全局实体集 $\mathcal{U}$ 中每个实体描述的相似度，超过阈值 $\theta=0.5$ 的实体所关联的所有节点作为候选
    - 通过重排序选取 Top-$N$（$N=20$）个最相关片段
    - 相比朴素 RAG 逐片段独立检索，图结构天然保留了实体间的时序关联

3. **结构化推理 (Structured Reasoning)**：

    - 核心发现：约 40% 失败案例中正确片段已被检索但模型仍回答错误（信息过载问题）
    - **分治验证**：让 LVLM 生成结构化子查询（yes/no 或数值型），对每个检索到的片段逐一验证
    - **噪声过滤**：只保留至少通过一个子查询验证的片段（最多保留 $r=5$ 个），有效消除硬负例
    - **信息聚合**：对过滤后的片段，聚合所有子查询结果生成辅助上下文

### 损失函数 / 训练策略

VGEnt 是无训练 (training-free) 的框架，不涉及额外微调或损失函数。图构建使用 BAAI/bge-large-en-v1.5 嵌入进行相似度计算，字幕提取使用 openai/whisper-large。

## 实验关键数据

### 主实验

| 模型 | 尺寸 | MLVU 提升 | VideoMME (w/ sub.) 提升 | LVB 提升 |
|------|------|----------|----------------------|---------|
| InternVL2.5 + VGEnt | 2B | +4.4 | +1.6 | +2.8 |
| Qwen2.5-VL + VGEnt | 3B | +4.2 | +2.0 | +3.6 |
| LongVU + VGEnt | 7B | +5.4 | +2.8 | +2.5 |
| Qwen2-VL + VGEnt | 7B | +4.6 | +2.0 | +2.8 |
| LLaVA-Video + VGEnt | 7B | +3.0 | +1.9 | +2.9 |
| Qwen2.5-VL + VGEnt | 7B | +3.3 | +3.2 | +3.7 |

亮点：Qwen2.5-VL (3B) + VGEnt 达到 70.4% MLVU，**超越其 7B 版本** (68.8%)。

### 消融实验

| 配置 | MLVU | VideoMME | LVB | 说明 |
|------|------|---------|-----|------|
| Qwen2.5-VL 基线 | 68.8 | 71.1 | 56.0 | 无 RAG |
| + NaïveRAG | 65.4 | 68.3 | 56.2 | 朴素 RAG 反而降低 MLVU |
| + GraphRAG | 69.5 | 72.7 | 57.1 | 图检索比朴素 RAG 好 |
| + NaïveRAG + SR | 68.6 | 69.8 | 57.3 | 结构化推理帮助有限 |
| + GraphRAG + SR | **72.1** | **74.3** | **59.7** | 两者协同最优 |

### 关键发现

- **NaïveRAG 可能有害**：在 MLVU 上从 68.8 降到 65.4，说明不当检索引入噪声比不检索更差
- **GraphRAG vs NaïveRAG**：平均提升 2.9%，MLVU 上差距达 4.1%
- **结构化推理的必要性**：GraphRAG + SR 比 GraphRAG 多提升约 2.6%~2.6%
- 在长视频场景（VideoMME long 子集）提升最为显著，达 5.4%

## 亮点与洞察

- 图结构的优势在于保留了实体级时序依赖，这是朴素 chunk-based RAG 根本无法做到的
- 结构化推理的"分治验证"策略非常巧妙：将复杂问题拆解为简单的 yes/no 子问题，让模型按片段逐一回答，降低了 LVLM 的推理难度
- 框架的模块化设计使其可即插即用到任何开源 LVLM，且图构建是一次性的离线成本

## 局限与展望

- 图构建依赖 LVLM 提取实体和描述，对 LVLM 本身的视觉理解能力有要求
- 实体合并使用固定相似度阈值，可能导致同义不同词的实体未被合并
- 结构化推理引入多轮 LVLM 调用，推理延迟较高
- 对于无明确实体的抽象推理问题（如情感、风格分析），图结构的优势可能有限

## 相关工作与启发

- 与 GraphRAG (NLP 领域) 的关系：将文本领域的图增强 RAG 思路迁移到视频领域，但增加了视觉实体合并和多模态验证
- 与 VideoAgent 的关系：解决类似问题但不依赖闭源 API，实现了自包含的开源方案
- 启发：结构化后验证的思路可推广到其他多模态 RAG 场景，如文档理解、多图推理

## 评分
- 新颖性: ⭐⭐⭐⭐ 图 RAG + 结构化推理的组合在视频领域是新尝试
- 实验充分度: ⭐⭐⭐⭐⭐ 7个模型 × 3个基准 × 充分消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图文并茂
- 价值: ⭐⭐⭐⭐ 即插即用框架，实用性强；3B 模型超 7B 的结果很有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[CVPR 2026\] RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](../../CVPR2026/video_understanding/ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](../../CVPR2025/video_understanding/drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->

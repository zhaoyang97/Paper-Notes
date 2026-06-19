---
title: >-
  [论文解读] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps
description: >-
  [ACL 2025][信息检索/RAG][Adaptive-RAG] 提出 IDR²，一种模型无关的自适应RAG加速框架，通过消除多轮检索间重叠文档的冗余表示并利用检索内容指导并行解码，实现端到端约2倍加速且不损失生成质量。 领域现状：检索增强生成(RAG)通过引入外部知识弥补LLM的知识不足。自适应RAG(A-RAG)进一…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "Adaptive-RAG"
  - "KV Cache共享"
  - "推理加速"
  - "推测解码"
  - "检索增强生成"
---

# Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps

**会议**: ACL 2025  
**arXiv**: [2505.12731](https://arxiv.org/abs/2505.12731)  
**代码**: 无  
**领域**: 信息检索 / RAG加速  
**关键词**: Adaptive-RAG, KV Cache共享, 推理加速, 推测解码, 检索增强生成

## 一句话总结

提出 IDR²，一种模型无关的自适应RAG加速框架，通过消除多轮检索间重叠文档的冗余表示并利用检索内容指导并行解码，实现端到端约2倍加速且不损失生成质量。

## 研究背景与动机

**领域现状**：检索增强生成(RAG)通过引入外部知识弥补LLM的知识不足。自适应RAG(A-RAG)进一步通过多轮检索-生成交互来处理复杂查询，显著提升了回答质量。

**现有痛点**：A-RAG的多轮交互机制加剧了RAG固有的效率问题。**核心矛盾在于**：相邻轮次的检索结果存在大量重叠（实验发现约60-80%的文档在相邻轮次中重复出现），但现有方法每轮都从头处理所有检索内容，导致大量冗余计算。

**本文目标**：消除A-RAG中因检索结果重叠而产生的冗余表示计算，同时加速自回归解码过程。

**切入角度**：分别在prefilling和decoding两个阶段引入加速机制——prefilling阶段通过缓存复用和指令引导消除重叠文档的冗余表示，decoding阶段利用检索文档构建近似语言模型实现并行生成。

**核心idea**：A-RAG多轮检索结果高度重叠，可以缓存已处理文档的KV表示直接复用；同时，RAG生成的内容与检索文档高度相关，可据此构建draft token实现类speculative decoding的并行生成。

## 方法详解

### 整体框架

IDR² 将每轮A-RAG的生成过程分为检索、prefilling、decoding三个阶段，分别在后两个阶段引入加速模块：CICS+IDGR加速prefilling，IGPG加速decoding。

### 关键设计

1. **跨迭代缓存共享 (CICS)**:
    - 功能：建立共享缓存空间 $\mathbb{C}$ 存储每轮已处理文档的KV表示
    - 核心思路：在第 $t$ 轮，先检查检索到的文档集 $D_t$ 中哪些已有缓存。对于已缓存的文档 $D_t^o$，直接加载其 $K_t^o, V_t^o$；仅对新文档 $D_t^n = D_t \setminus D_t^o$ 执行prefilling。形式化为 $a_t^1, K_t, V_t = \text{LLM}_P(q_t, D_t^n, A_{<t}, K_t^o, V_t^o)$
    - 设计动机：A-RAG相邻轮次文档重叠率极高（实验中3篇检索设置下达60-80%），缓存复用可大幅减少prefilling计算量

2. **指令驱动去重引导强化 (IDGR)**:
    - 功能：通过自然语言指令引导LLM正确处理缓存中的冗余信息
    - 核心思路：由于self-attention机制，缓存文档的KV表示中包含了上一轮其他文档的信息（如文档A的表示中融合了文档B的信息）。IDGR自动生成指令 $I_t$ 告诉LLM：(a) 哪些文档与当前轮次相关/无关；(b) 文档的相关性排名。例如："#5881721 ...是相关文档。#10028469 是无关文档。相关性分数为..."
    - 设计动机：直接使用缓存的KV表示会引入旧轮次的无关信息噪声，导致生成质量下降（实验显示EM下降2-3%）。IDGR利用LLM的指令遵循能力过滤噪声，不仅恢复还提升了性能

3. **信息引导并行生成 (IGPG)**:
    - 功能：利用检索文档中的内容构建draft token序列，实现并行验证和生成
    - 核心思路：RAG生成的内容与检索文档高度相关（2-token组合约70%出现在检索文档中）。IGPG用检索文档 $D_t$ 构建近似N-gram语言模型 $P(x_t|x_{t-N+1},...,x_{t-1})$，在每步自回归前查询匹配的后续短语片段作为draft。LLM一次前向传播验证draft的M个token，验证通过则一次生成多个token
    - 设计动机：不同于传统speculative decoding需要训练小模型，IGPG直接利用RAG场景中现成的检索文档作为draft来源，零训练成本

### 损失函数 / 训练策略

IDR² 是一种推理时加速方法，不涉及模型训练。使用BM25或SGPT作为检索器，兼容多种A-RAG方法（FL-RAG, FS-RAG, FLARE, DRAGIN）。

## 实验关键数据

### 主实验

在4个A-RAG方法 × 3个模型（LLaMA2-7B/13B, Vicuna-13B） × 4个数据集上的平均加速比：

| 加速阶段 | 最小加速 | 最大加速 | 平均加速 |
|----------|---------|---------|---------|
| Prefilling | 1.75× | 4.72× | 2.79× |
| Decoding | 1.49× | 4.00× | 2.33× |
| End-to-End | 1.31× | 3.53× | 2.0× |

DRAGIN+IDR² 在 LLaMA2-7B 上的具体延迟（2WikiMultihopQA, n=3）：

| 阶段 | 原始(s) | IDR²(s) | 加速比 |
|------|---------|---------|--------|
| Prefilling | 3.71 | 1.18 | 3.14× |
| Decoding | 12.55 | 6.07 | 2.07× |
| End-to-End | 19.31 | 9.56 | 2.02× |

### 消融实验

各模块对DRAGIN（2WikiMultihopQA）的影响：

| CICS | IGPG | IDGR | LLaMA2-7B EM↑ | LLaMA2-13B EM↑ |
|------|------|------|---------------|----------------|
| ✗ | ✗ | ✗ | 22.5 | 30.4 |
| ✓ | ✗ | ✗ | 20.3 | 28.0 |
| ✗ | ✓ | ✗ | 22.4 | 30.4 |
| ✓ | ✓ | ✓ | **25.4** | **34.4** |

### 关键发现

1. CICS单独使用会引入冗余信息噪声导致性能下降，但加入IDGR后不仅恢复还超越了基线
2. IGPG对生成质量几乎无影响（EM变化<0.1），因为它本质是带验证的并行生成
3. DRAGIN因query refinement机制使得相邻轮次query更相似，文档重叠率更高，因此从IDR²中获益最大（prefilling最高达4.72×）
4. 不同检索器（BM25 vs SGPT）下IDR²均有效，加速比>2.6×

## 亮点与洞察

- **关键洞察**：A-RAG多轮检索结果的重叠被现有方法完全忽视，这是一个"隐性"的效率瓶颈
- **方法优雅性**：IDR²是模型无关的，可以即插即用地应用于任何A-RAG方法
- **IDGR的创新**：用自然语言指令解决KV cache复用中的信息冲突问题，巧妙利用了LLM的指令遵循能力
- **IGPG的效率**：利用RAG场景的特殊性（生成与检索文档高度相关），零成本构建draft模型

## 局限与展望

- 仅适用于开源LLM，不适用于仅支持文本接口的LLM API
- KV cache存储带来额外显存开销，对超长文档可能成为瓶颈
- IGPG的加速效果取决于生成内容与检索文档的重合度，在创造性任务中可能减弱
- 未探索KV cache的进一步压缩以减少存储开销

## 相关工作与启发

- **与TurboRAG的区别**：TurboRAG预处理整个知识库的KV表示，适用于单轮RAG；IDR²针对A-RAG的多轮间重叠问题，动态缓存
- **与Speculative Decoding的联系**：IGPG借鉴了speculative decoding的验证思想，但不需要额外的draft模型，直接利用检索文档
- 启发：在多轮对话、multi-agent等多次调用LLM的场景中，类似的缓存复用策略可能同样有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地发现并解决A-RAG的检索重叠冗余问题，IDGR和IGPG设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4种A-RAG方法×3个模型×4个数据集，消融实验和案例分析详尽
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，公式化定义严谨
- 价值: ⭐⭐⭐⭐ 实用价值高，即插即用加速A-RAG推理，约2倍加速非常可观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] VISA: Retrieval Augmented Generation with Visual Source Attribution](visa_retrieval_augmented_generation_with_visual_source_attribution.md)
- [\[ACL 2025\] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)

</div>

<!-- RELATED:END -->

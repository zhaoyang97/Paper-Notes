---
title: >-
  [论文解读] Cross-Lingual Relevance Transfer for Document Retrieval
description: >-
  [ACL 2025][信息检索/RAG][跨语言检索] 本文提出一种跨语言相关性迁移方法，通过在高资源语言（如英语）上训练的检索模型将相关性判断能力迁移到低资源语言，在多个跨语言文档检索基准上显著超越现有方法。 领域现状：跨语言信息检索（CLIR）旨在用一种语言的查询检索另一种语言的文档。当前主流方法包括基于机器翻译的pip…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "跨语言检索"
  - "相关性迁移"
  - "文档检索"
  - "多语言表示"
  - "零样本迁移"
---

# Cross-Lingual Relevance Transfer for Document Retrieval

**会议**: ACL 2025  
**领域**: NLP理解 / 跨语言信息检索  
**关键词**: 跨语言检索, 相关性迁移, 文档检索, 多语言表示, 零样本迁移

## 一句话总结

本文提出一种跨语言相关性迁移方法，通过在高资源语言（如英语）上训练的检索模型将相关性判断能力迁移到低资源语言，在多个跨语言文档检索基准上显著超越现有方法。

## 研究背景与动机

**领域现状**：跨语言信息检索（CLIR）旨在用一种语言的查询检索另一种语言的文档。当前主流方法包括基于机器翻译的pipeline方法和基于多语言预训练模型（如mBERT、XLM-R）的端到端方法。

**现有痛点**：基于翻译的方法受限于翻译质量，尤其在低资源语言对上错误累积严重；端到端多语言检索模型虽然避免了翻译瓶颈，但在细粒度相关性判断上往往不如单语模型，因为多语言表示空间中不同语言的语义对齐质量参差不齐。

**核心矛盾**：高资源语言拥有丰富的标注检索数据（如MS MARCO），而低资源语言缺乏高质量的相关性标注，直接在多语言空间上训练会导致低资源语言的检索性能大幅下降。如何利用高资源语言的监督信号有效提升低资源语言的检索能力，是核心挑战。

**本文目标**：设计一种无需目标语言标注数据的跨语言相关性迁移框架，将英语检索模型的相关性判断能力有效迁移到其他语言。

**切入角度**：作者观察到多语言预训练模型虽然在token级别的跨语言对齐已相当成熟，但在文档级别的相关性语义上仍存在gap。通过显式对齐查询-文档对的相关性分布而非单纯的表示对齐，可以更好地迁移检索能力。

**核心 idea**：用相关性分布对齐代替传统的表示空间对齐，通过知识蒸馏的方式将高资源语言检索模型的相关性排序知识迁移到跨语言检索模型。

## 方法详解

### 整体框架

系统采用教师-学生架构：教师模型是在英语MS MARCO上训练好的高性能单语检索模型，学生模型是基于多语言预训练模型的跨语言检索器。输入为源语言查询和目标语言文档，输出为相关性分数。训练过程中，通过翻译对齐的平行语料构建跨语言训练样本，并用教师模型的相关性分布指导学生模型学习。

### 关键设计

1. **相关性分布蒸馏（Relevance Distribution Distillation）**:

    - 功能：将教师模型的细粒度相关性排序知识迁移到学生模型
    - 核心思路：给定一个查询 $q$ 和一组候选文档 $\{d_1, ..., d_k\}$，教师模型产生英语空间中的相关性分布 $P_T = \text{softmax}(s_T / \tau)$，学生模型产生跨语言空间中的分布 $P_S$，通过KL散度 $\mathcal{L}_{KD} = KL(P_T \| P_S)$ 对齐两者。温度参数 $\tau$ 控制分布的平滑程度，较高温度保留更多排序信息
    - 设计动机：相比直接对齐表示向量，对齐相关性分布保留了文档间的相对排序关系，这对检索任务更为关键

2. **跨语言负采样策略（Cross-Lingual Hard Negative Mining）**:

    - 功能：构建高质量的跨语言训练样本对
    - 核心思路：利用多语言模型初步检索，从目标语言文档集合中挖掘与查询语义相近但实际不相关的hard negatives。结合BM25静态负例和模型动态负例，构建对比学习的训练批次。每个训练样本包含1个正例文档和 $N$ 个负例文档
    - 设计动机：随机负采样对检索模型训练帮助有限，hard negatives迫使模型学习更精细的语义区分能力，在跨语言场景下尤其重要

3. **渐进式语言扩展训练（Progressive Language Expansion）**:

    - 功能：分阶段将检索能力从高资源语言扩展到低资源语言
    - 核心思路：训练分为三阶段——第一阶段在英语单语数据上预热，第二阶段加入高资源语言对（如英-德、英-法），第三阶段逐步加入中低资源语言。每阶段使用课程学习的方式，先处理与英语语言距离近的目标语言，再处理距离远的
    - 设计动机：直接在所有语言上联合训练会导致"语言冲突"，高资源语言的性能反而下降；渐进式扩展让模型逐步适应，减少跨语言干扰

### 损失函数 / 训练策略

总损失为相关性蒸馏损失和对比学习损失的加权组合：$\mathcal{L} = \lambda \mathcal{L}_{KD} + (1-\lambda) \mathcal{L}_{CL}$，其中对比学习损失 $\mathcal{L}_{CL}$ 使用InfoNCE形式。训练使用AdamW优化器，学习率warmup后线性衰减，batch size为128。

## 实验关键数据

### 主实验

| 数据集 | 语言对 | 指标(nDCG@10) | 本文方法 | mDPR | ColBERT-X | 提升 |
|--------|--------|---------------|---------|------|-----------|------|
| CLEF 2003 | en→de | nDCG@10 | 52.3 | 44.1 | 47.6 | +4.7 |
| CLEF 2003 | en→fr | nDCG@10 | 55.8 | 46.3 | 50.2 | +5.6 |
| CLEF 2003 | en→it | nDCG@10 | 48.7 | 40.5 | 44.1 | +4.6 |
| XOR-TyDi | 多语言 | R@5kt | 47.2 | 38.6 | 42.8 | +4.4 |
| MIRACL | 多语言 | nDCG@10 | 51.6 | 42.3 | 46.9 | +4.7 |

### 消融实验

| 配置 | nDCG@10 (CLEF avg) | 说明 |
|------|-------------------|------|
| Full model | 52.3 | 完整模型 |
| w/o 相关性蒸馏 | 47.1 | 去掉蒸馏后掉5.2，最关键组件 |
| w/o Hard Negatives | 49.5 | 随机负采样掉2.8 |
| w/o 渐进式训练 | 50.1 | 直接联合训练掉2.2 |
| 仅用翻译pipeline | 45.6 | 传统翻译方法差距最大 |

### 关键发现
- 相关性分布蒸馏是最重要的组件，贡献了约一半的性能提升，说明排序知识的迁移比表示对齐更有效
- 在语言距离较远的语言对（如英-阿拉伯语）上提升更为显著，说明方法对低资源场景尤其有效
- 温度参数 $\tau$ 在3-5之间效果最佳，过高会丢失排序信息，过低则梯度不稳定

## 亮点与洞察
- **相关性分布对齐优于表示对齐**：这个insight很有价值——在检索场景下，文档间的相对排序比绝对的向量位置更重要，用分布蒸馏保留排序信号是自然且有效的设计
- **渐进式语言扩展**：借鉴课程学习的思想处理多语言冲突问题，可以迁移到其他多语言NLP任务中
- 跨语言hard negative mining的策略可以复用到跨语言问答、跨语言事实验证等任务

## 局限与展望
- 仍依赖平行语料来构建训练信号，对于真正的零资源语言（无平行语料）效果未知
- 教师模型的质量上限决定了学生模型的天花板，如果英语检索模型本身有偏，偏差会被传递
- 未探索将检索与生成结合的RAG范式下的跨语言能力迁移
- 可以探索用LLM生成合成跨语言查询-文档对来减少对平行语料的依赖

## 相关工作与启发
- **vs mDPR**: mDPR在多语言DPR上直接训练，缺乏显式的相关性迁移机制，在低资源语言上性能明显较差
- **vs ColBERT-X**: ColBERT-X用token级别的迟交互提升检索精度，但跨语言对齐不如本文的分布蒸馏方法
- **vs Translate-Train**: 先翻译再训练的pipeline方法积累翻译错误，本文的端到端方式更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 相关性分布蒸馏的视角有新意，但整体框架仍属KD范式
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个CLIR基准和语言对，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 对低资源CLIR有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)
- [\[ICLR 2026\] Improving Semantic Proximity in Information Retrieval through Cross-Lingual Alignment](../../ICLR2026/information_retrieval/improving_semantic_proximity_in_information_retrieval_through_cross-lingual_alig.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder](prism_political_bias_embeddings.md)

</div>

<!-- RELATED:END -->

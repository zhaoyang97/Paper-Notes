---
title: >-
  [论文解读] LOCORE: Image Re-ranking with Long-Context Sequence Modeling
description: >-
  [CVPR 2025][LLM效率][图像重排序] 提出 LoCoRe（Long-Context Re-ranker），首次实现基于局部描述子的列表级（list-wise）图像重排序，利用 Longformer 长上下文序列模型同时处理查询图像和整个候选列表的局部描述子，通过捕获候选图像间的传递关系显著提升重排序性能。
tags:
  - "CVPR 2025"
  - "LLM效率"
  - "图像重排序"
  - "长上下文序列模型"
  - "局部描述子"
  - "Longformer"
  - "列表级学习"
---

# LOCORE: Image Re-ranking with Long-Context Sequence Modeling

**会议**: CVPR 2025  
**arXiv**: [2503.21772](https://arxiv.org/abs/2503.21772)  
**代码**: [GitHub](https://github.com/MrZilinXiao/LongContextReranker)  
**领域**: LLM效率  
**关键词**: 图像重排序, 长上下文序列模型, 局部描述子, Longformer, 列表级学习

## 一句话总结

提出 LoCoRe（Long-Context Re-ranker），首次实现基于局部描述子的列表级（list-wise）图像重排序，利用 Longformer 长上下文序列模型同时处理查询图像和整个候选列表的局部描述子，通过捕获候选图像间的传递关系显著提升重排序性能。

## 研究背景与动机

**领域现状**：图像检索通常分为两阶段——首先通过全局描述子快速检索候选列表，然后使用更精细的重排序方法对候选列表进行二次排序。重排序阶段常使用局部特征描述子进行成对（pair-wise）相似度估计。

**现有痛点**：
- **成对重排序**（如 RRT、CVNet、AMES）每次只比较查询与单个候选图像，无法利用候选图像之间的关系
- **列表级重排序**（如 SSR Rerank）虽然能考虑候选间关系，但仅使用全局描述子，缺乏局部特征的细粒度信息
- 成对重排序器处理 K 个候选需要 K 次前向传播，计算开销大

**核心矛盾**：局部描述子提供了细粒度匹配能力，但每张图像有多个描述子，将所有候选图像的局部描述子一起处理面临序列长度的巨大挑战。

**本文目标** 如何在不牺牲局部描述子细粒度优势的前提下，实现列表级重排序以利用候选图像间的传递关系。

**切入角度**：借鉴 NLP 中的序列标注和抽取式问答范式，将图像重排序转化为长序列的 token 级分类问题。

**核心 idea**：将查询和所有候选图像的局部描述子拼接为超长序列，用 Longformer 建模序列中的上下文依赖，通过 token 级分类实现列表级重排序。

## 方法详解

### 整体框架

LoCoRe 将查询图像和 K 个候选图像的局部描述子拼接成一个长序列，输入预训练的 Longformer 模型。模型对每个 token 进行二分类（属于正样本或负样本图像），推理时将同一图像的 token 得分聚合作为该图像的相似度分数。

### 关键设计

1. **局部描述子序列化与分隔标记**:
    - 功能：将多张图像的局部描述子组织为可处理的长序列
    - 核心思路：序列格式为 [query, SEP, gallery_1, SEP, ..., gallery_K, SEP]，其中每张图像贡献 L 个局部描述子，SEP 是可学习的分隔 token。总序列长度为 M = (L+1)(K+1)，默认 L=50, K=100 时为 5,050 tokens
    - 设计动机：分隔 token 既标记图像边界，又作为全局注意力的锚点

2. **查询全局注意力机制**:
    - 功能：在 Longformer 的滑动窗口注意力基础上，确保长距离依赖建模
    - 核心思路：查询图像的所有 token 和所有 SEP token 被设置为全局注意力 token（对称地 attend 所有 token），其余 token 仅参与局部窗口注意力。这样保证线性计算复杂度的同时不丢失全局信息
    - 设计动机：去除全局查询注意力后 R@1 从 82.4% 暴跌至 60.7%，证明其不可或缺

3. **画廊随机打乱训练 + Token级分类**:
    - 功能：防止位置偏差捷径，实现端到端训练
    - 核心思路：全局检索往往将正样本排在前面，直接使用该顺序会让模型学到"位置=标签"的捷径。因此训练时随机打乱候选顺序。所有 (L+1)×K 个 token 使用 BCE 损失训练，推理时聚合同一图像的 token得分
    - 设计动机：不打乱训练时模型完全退化（mAP与全局检索相同）

### 损失函数 / 训练策略

- **损失函数**：对所有 gallery token 的二元交叉熵损失（BCELoss）
- **推理聚合**：SEP token 得分、平均 token 得分或首 token 得分（效果相当）
- **滑动窗口策略**：推理时若候选数 N > K，从列表末尾向前滑动，窗口大小 K、步长 S，重叠区域取平均
- **模型初始化**：LoCoRe-small 从 longformer-base-4096 前6层初始化，base 从全部 12层初始化，位置编码线性插值从 4096 扩展到 5120
- **训练配置**：AdamW 优化器，学习率 5e-5，8 × A100 GPU，全局 batch size 128

## 实验关键数据

### 主实验

**地标检索（ROxf, RPar）**：

| 设置 | 方法 | ROxf+1M Hard | RPar+1M Hard |
|------|------|-------------|-------------|
| RN50-DELG | CVNet Reranker | +13.4 mAP | +8.1 mAP |
| RN50-DELG | LoCoRe-base | **+17.8 mAP** | **+13.8 mAP** |

**度量学习基准**：

| 数据集 | 指标 | Global | RRT | LoCoRe-base |
|--------|------|--------|-----|-------------|
| CUB-200 | R@1 | 68.9 | 68.7 | **78.3** |
| CUB-200 | mAP@R | 49.8 | 55.6 | **64.8** |
| SOP | R@1 | 80.8 | 81.9 | **83.8** |
| SOP | mAP@R | 65.1 | 67.2 | **71.0** |
| In-Shop | R@1 | 86.3 | 88.3 | **89.4** |

### 消融实验

| 消融项 | R@1 (SOP) | mAP@R (SOP) |
|--------|-----------|-------------|
| 全局检索基线 | 80.8 | 65.1 |
| LoCoRe-tiny | 82.4 | 68.0 |
| w/o 画廊打乱训练 | 80.7 | 65.1 |
| w/o 全局查询注意力 | 60.7 | 53.0 |
| LoCoRe-base | **83.8** | **71.0** |

### 关键发现

- **画廊打乱至关重要**：不打乱时模型退化为全局检索的复读机
- **模型规模有收益**：tiny→small→base 性能持续提升
- **传递关系有效**：定性分析显示模型确实利用了候选图像间共享的局部特征
- **延迟优势显著**：LoCoRe-small 24.7ms vs RRT 74.4ms（重排100张）
- **循环模型不适用**：Mamba和RWKV表现不如Transformer

## 亮点与洞察

1. **范式创新**：首次将局部描述子的列表级重排序变为可行，开创了新的重排序范式
2. **传递关系建模**：通过长上下文捕获候选间的传递关系——两个正样本共享的局部特征可以相互增强置信度
3. **NLP 启发**：巧妙地将图像重排序转化为 NER/QA 的 token 级序列标注任务
4. **效率优势**：一次前向传播处理 100 张候选，而成对方法需要 100 次
5. **训练技巧重要性**：画廊随机打乱这个简单技巧是方法成功的关键

## 局限与展望

1. **上下文窗口限制**：Longformer 的最大上下文长度限制了单次可处理的候选数（默认100张）
2. **循环模型效果不佳**：Mamba、RWKV 无法有效捕获列表级重排序依赖
3. **未来方向**：可探索 decoder-only 大模型（更长上下文）、context parallelization（如 RingAttention）
4. **跨模态扩展**：可推广到文档检索、视频重排序等

## 相关工作与启发

- **Longformer**：线性复杂度长序列建模的核心骨干
- **RRT / CVNet / AMES**：代表性成对重排序方法
- **序列标注任务（NER, QA）**：NLP 中 token 级分类的设计灵感来源
- **对后续研究的启发**：列表级学习信号在排序/推荐任务中的潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ACL 2025\] Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](../../ACL2025/llm_efficiency/sliding_windows_full_ranking.md)
- [\[ICML 2025\] Curse of High Dimensionality Issue in Transformer for Long-context Modeling](../../ICML2025/llm_efficiency/curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->

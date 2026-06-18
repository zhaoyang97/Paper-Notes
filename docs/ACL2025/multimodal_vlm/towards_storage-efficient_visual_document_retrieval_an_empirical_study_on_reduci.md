---
title: >-
  [论文解读] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings
description: >-
  [ACL 2025][信息检索/RAG][Visual Document Retrieval] 系统性研究了视觉文档检索（VDR）中 patch 级别嵌入的压缩策略，发现 pruning 在 VDR 中本质不适用（简单随机剪枝反而最优），而 token merging 结合微调可在仅保留 2.8% 存储量时维持 94.6% 的检索性能（Light-ColPali/ColQwen2）。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "Visual Document Retrieval"
  - "ColPali"
  - "Token Merging"
  - "剪枝"
  - "存储效率"
---

# Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings

**会议**: ACL 2025  
**arXiv**: [2506.04997](https://arxiv.org/abs/2506.04997)  
**代码**: 无（提及但未明确链接）  
**领域**: 信息检索  
**关键词**: Visual Document Retrieval, ColPali, Token Merging, Token Pruning, 存储效率

## 一句话总结

系统性研究了视觉文档检索（VDR）中 patch 级别嵌入的压缩策略，发现 pruning 在 VDR 中本质不适用（简单随机剪枝反而最优），而 token merging 结合微调可在仅保留 2.8% 存储量时维持 94.6% 的检索性能（Light-ColPali/ColQwen2）。

## 研究背景与动机

视觉文档检索（VDR）将文档页面作为图像编码为嵌入向量进行检索，保留了布局结构和视觉元素。当前最先进的 ColPali/ColQwen2 检索器为每个页面生成大量 patch 级别嵌入（ColPali: 1024 个，ColQwen2: 768 个），虽然实现了细粒度感知和卓越检索性能，但导致了巨大的存储开销：

- 一个 50 页的中等规模文档需要约 10 MB 存储嵌入向量
- 在大规模实际部署场景中，存储瓶颈严重制约了 VDR 系统的可扩展性

核心问题：**如何在大幅减少每页 patch 嵌入数量的同时最小化检索性能下降？**

## 方法详解

### 整体框架

本文从两个维度系统探索 token 减少策略：
1. **Token Pruning**（直接丢弃部分嵌入）→ 证明不适用于 VDR
2. **Token Merging**（将多个嵌入合并为一个）→ 从三个维度搜索最优组合 → Light-ColPali/ColQwen2

### 关键设计

1. **Token Pruning 实验（证明其不可行）**：

    - 测试三种策略：随机剪枝、分数导向（基于合成查询的响应潜力排序）、注意力导向（基于 [EOS] token 注意力权重）
    - **反直觉发现**：随机剪枝竟然优于精心设计的策略
    - **原因分析**：(a) 同一页面被不同查询激活的 patch 几乎不重叠（仅比随机略多），无法预测哪些 patch 重要；(b) patch 嵌入存在冗余分组，精心设计的策略会整组丢弃导致信息缺失
    - 核心结论：pruning 在没有查询信息的离线阶段本质上是不可行的

2. **Token Merging 三维度探索**：

    - **合并方式**（三种）：1D 空间池化、2D 空间池化、语义聚类（层次聚类）
    - **微调与否**：训练时也使用合并后的嵌入进行微调
    - **合并位置**（四种）：Pre-Encoder / Post-Encoder / Post-LLM / Post-Projector
    - 每个维度独立实验后组合最优方案

3. **Light-ColPali/ColQwen2 的最终方案**：

    - 合并方式：语义聚类（略优于空间池化）
    - 微调：是（在合并因子 49 时恢复 67% 性能损失）
    - 合并位置：Post-Projector（尽可能晚地合并，保留最多视觉信息）
    - 设计动机：VDR 场景关注存储而非推理延迟，因此可以在最后阶段合并

### 损失函数 / 训练策略

- 微调使用与 ColPali/ColQwen2 相同的对比学习损失
- 训练和推理阶段都使用合并后的嵌入计算相关性分数
- 微调消耗约 72 A100-GPU 小时

## 实验关键数据

### 主实验（Light-ColQwen2 在三个 benchmark 上的表现）

| 方法 | 合并因子 | 相对存储 | ViDoRE-Info | ViDoRE-Doc | ViDoRE-Avg | 相对性能 |
|------|---------|---------|-------------|------------|------------|---------|
| ColQwen2 (原版) | - | 64.4x | 91.5 | 55.4 | 81.4 | 100% |
| ColQwen2+Pruning | 9x | 7.6x | 85.6 | 48.3 | 74.0 | 90.9% |
| Light-ColQwen2 | 4x | 16.4x | 89.5 | 56.6 | 80.6 | **99.0%** |
| Light-ColQwen2 | 9x | 7.6x | 90.4 | 56.1 | 79.9 | **98.2%** |
| Light-ColQwen2 | 25x | 3.0x | 88.9 | 54.6 | 78.4 | 96.3% |
| Light-ColQwen2 | 49x | 1.8x | 86.9 | 52.6 | 77.0 | **94.6%** |

### 合并位置消融实验（合并因子=9）

| 位置 | Info | Doc | Arxiv | TabF | TAT | Shift | 平均 |
|------|------|-----|-------|------|-----|-------|------|
| Pre-Encoder | 70.2 | 29.8 | 80.0 | 74.1 | 50.5 | 49.7 | 59.1 |
| Post-Encoder | 79.5 | 41.7 | 81.9 | 80.8 | 54.1 | 54.4 | 65.4 |
| Post-LLM | 89.7 | 55.2 | 87.6 | 88.6 | 79.5 | 85.7 | **81.0** |
| Post-Projector | 90.4 | 56.1 | 86.7 | 88.8 | 79.1 | 87.3 | **81.4** |

### 关键发现

1. **Pruning 在 VDR 中不可行**：随机剪枝反而优于精心设计的策略，在 95% 剪枝率时性能仅保留 58-85%
2. **语义聚类略优于空间池化**：在合并因子 9 和 25 时分别保留 97.5% 和 92.6% 的性能
3. **微调至关重要**：在合并因子 49 时，微调恢复了 67% 的性能损失（绝对提升 8.4%）
4. **越晚合并越好**：Post-Projector 位置的平均得分（81.4）远超 Pre-Encoder（59.1）
5. **极端压缩可行**：Light-ColQwen2 在仅 2.8% 存储开销下仍保持 94.6% 的检索性能

## 亮点与洞察

- **反直觉+深度分析**：随机剪枝优于精心设计策略的发现令人惊讶，且分析原因很到位（激活 patch 的不可预测性和冗余性）
- **系统性实验设计**：三个维度（合并方式×微调×位置）的组合搜索方法论值得借鉴
- **VDR vs 生成的关键差异洞察**：在生成任务中关注延迟和 FLOPs，所以要在早期层做 token 减少；但在 VDR 中关注存储，所以可以在最后阶段做合并
- **实用性强**：Light-ColPali/ColQwen2 可直接作为存储高效 VDR 的基线方案

## 局限与展望

- 微调仍需 72 A100-GPU 小时的计算开销
- 仅在 ColPali/ColQwen2 两个检索器上验证，通用性需验证
- 语义聚类的在线计算开销（层次聚类）未被充分讨论
- 未结合维度压缩（如量化）进行联合优化
- 在更大规模的文档语料库上的实际部署效果未测试

## 相关工作与启发

- 与 Clavié et al. (2024) 的 TokenPooling 工作相比，本文增加了系统性的 pruning 分析和微调验证
- 与 LVLM 生成效率工作（FastV, ToMe 等）的关键区别在于 VDR 场景不关心延迟而关心存储
- 与 ColBERTv2 的乘积量化维度压缩是正交的，两者可以组合使用
- 启发：VDR 场景下的 token 减少需要完全不同于生成场景的思路，离线/无查询条件是核心约束

## 评分

- **新颖性**: ⭐⭐⭐⭐ — pruning不可行的发现很有洞察力，系统性的三维度探索方法论好
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个benchmark、两个基础模型、多种策略对比、充分的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，从pruning的失败到merging的成功的叙事很自然
- **价值**: ⭐⭐⭐⭐ — 为VDR系统的实际部署提供了直接可用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](../../ACL2026/information_retrieval/a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](../../ACL2026/information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] Cross-Lingual Relevance Transfer for Document Retrieval](cross-lingual_relevance_transfer_for_document_retrieval.md)
- [\[ACL 2025\] VISA: Retrieval Augmented Generation with Visual Source Attribution](visa_retrieval_augmented_generation_with_visual_source_attribution.md)

</div>

<!-- RELATED:END -->

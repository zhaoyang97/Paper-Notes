---
title: >-
  [论文解读] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation
description: >-
  [ACL 2025][信息检索/RAG][自适应检索] SeaKR 利用 LLM 内部隐藏层的自感知不确定性（通过多次采样 EOS token 隐藏表示的 Gram 行列式度量）来自适应地决定何时检索、如何重排检索结果、以及选择何种推理策略，在复合 QA 上 F1 比 DRAGIN 提升 6%，比 IRCoT 提升 9.5%。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "自适应检索"
  - "自感知不确定性"
  - "内部状态"
  - "知识重排"
  - "迭代推理"
---

# SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2406.19215](https://arxiv.org/abs/2406.19215)  
**代码**: 已开源（GitHub）  
**领域**: RAG / 检索增强生成  
**关键词**: 自适应检索, 自感知不确定性, 内部状态, 知识重排, 迭代推理

## 一句话总结

SeaKR 利用 LLM 内部隐藏层的自感知不确定性（通过多次采样 EOS token 隐藏表示的 Gram 行列式度量）来自适应地决定何时检索、如何重排检索结果、以及选择何种推理策略，在复合 QA 上 F1 比 DRAGIN 提升 6%，比 IRCoT 提升 9.5%。

## 研究背景与动机

**领域现状**：RAG 通过将外部知识注入 LLM 上下文来缓解幻觉问题。自适应 RAG 进一步优化为仅在需要时才检索，避免不相关知识干扰。现有自适应方法如 FLARE 和 DRAGIN 通过输出 token 概率判断何时检索，Self-RAG 通过微调让模型生成特殊标记来指示检索需求。

**现有痛点**：（1）现有方法仅基于 LLM 输出来判断检索需求，但 LLM 存在系统性的自信偏差——即使参数知识不足也会自信地输出错误答案，输出概率并不能可靠反映真实的知识缺口。（2）现有方法忽略了检索后的知识整合问题——检索到多条知识后如何选择最有用的？多条知识如何整合推理？这些问题几乎未被系统性地解决。

**核心矛盾**：自适应 RAG 需要准确感知"我不知道什么"，但 LLM 的输出是经过离散化和自信偏差过滤后的，信息损失严重。更底层的内部状态理论上承载更丰富的不确定性信息，但尚未被利用于 RAG。

**本文目标** 利用 LLM 内部隐藏状态的自感知不确定性，在自适应 RAG 的三个关键环节——何时检索、如何整合、如何推理——都进行不确定性驱动的决策。

**切入角度**：基于认知科学研究中"LLM 对不确定内容的多次采样在隐空间中表现不一致"的观察，SeaKR 用 Gram 行列式度量多次生成的 EOS 隐藏表示的一致性作为不确定性分数，高不确定性触发检索，低不确定性的知识片段被优先选择。

**核心 idea**：用 LLM 中间层 EOS token 隐藏表示的多次采样一致性作为自感知不确定性度量，驱动检索触发、知识重排和推理策略选择。

## 方法详解

### 整体框架

SeaKR 采用 CoT 风格的迭代推理。每一步：（1）生成候选推理步骤，用自感知不确定性评估器判断是否需要检索；（2）如需检索，从搜索引擎返回的 top-N 片段中选择最能降低不确定性的知识；（3）迭代完成后，用自感知推理在"直接生成"和"基于所有检索知识的 CoT 推理"两个策略中选择不确定性更低的结果。

### 关键设计

1. **自感知不确定性评估器**:

    - 功能：从 LLM 内部状态量化其对当前生成内容的确信程度
    - 核心思路：对同一输入上下文 $\mathbf{c}$ 采样 $k=20$ 次生成，收集每次生成的 EOS token 在中间层 $l=L/2$ 的隐藏表示 $\mathbf{H}^{(l)}_{\langle EOS \rangle}$，计算这 $k$ 个向量的正则化 Gram 矩阵的行列式作为不确定性分数 $U(\mathbf{c})$。行列式越大表示向量越分散、模型越不确定
    - 设计动机：相比输出 token 概率，内部状态级别的一致性度量避免了自然语言的多义性干扰——相同语义可以有不同的表面表达导致 token 概率低，但隐藏状态在语义空间中保持一致

2. **自感知重排（Self-aware Re-ranking）**:

    - 功能：从搜索引擎返回的 top-N 知识片段中选择最能降低 LLM 不确定性的片段
    - 核心思路：将每个候选知识片段分别拼接到输入上下文中，独立计算 $N$ 个增强上下文的不确定性 $U(\mathbf{c} + \mathbf{k}_i)$，选择不确定性最低的片段。这从"检索相关性"视角转变为"对模型最有帮助"视角
    - 设计动机：传统 RAG 按查询相关性排序知识，但高相关性不等于高实用性——可能与模型已有知识冲突或无法降低特定的不确定性

3. **自感知推理（Self-aware Reasoning）**:

    - 功能：在迭代推理结束后，选择最优的知识整合策略生成最终答案
    - 核心思路：提供两种推理策略——（1）直接从 CoT 推理链的最后一步生成答案；（2）将所有重排后的检索知识拼接为上下文做新的 CoT 推理。分别计算两种策略的答案不确定性，选择更确信的
    - 设计动机：不同问题的最优推理路径不同——有些问题的 CoT 推理链已经足够，额外知识反而干扰；有些问题需要综合所有知识重新推理

### 损失函数 / 训练策略

SeaKR 是无需训练的方法（tuning-free）。所有组件都基于 LLaMA-2-7B-chat 的原始推理能力和内部状态，不需要额外的微调。不确定性阈值 $\delta$ 在 NQ 训练集的小子集上搜索确定。

## 实验关键数据

### 主实验

复合 QA（多跳推理，LLaMA-2-7B-chat + BM25）：

| 方法 | 2WikiMultiHop F1 | HotpotQA F1 | IIRC F1 |
|------|------------------|-------------|---------|
| CoT (无检索) | 22.3 | 27.5 | 17.3 |
| IRCoT (每步检索) | 26.5 | 30.4 | 21.6 |
| Self-RAG (微调) | 19.6 | 17.5 | 5.7 |
| FLARE | 21.3 | 22.1 | 16.4 |
| DRAGIN | 30.0 | 34.2 | 22.9 |
| **SeaKR** | **36.0** | **39.7** | **23.5** |

### 消融实验

| 消融配置 | 2Wiki F1 | HotpotQA F1 | 说明 |
|---------|----------|-------------|------|
| SeaKR (完整) | **36.0** | **39.7** | 基线 |
| w/o 自感知检索 (每步检索) | 33.5 | 37.1 | -2.5/-2.6，验证自适应检索有效 |
| w/o 自感知重排 (用 top-1) | 31.2 | 35.6 | -4.8/-4.1，重排贡献最大 |
| w/o 自感知推理 (直接生成) | 33.8 | 37.9 | -2.2/-1.8 |
| 用输出概率替代内部状态 | 27.8 | 33.1 | -8.2/-6.6，验证内部状态优于输出概率 |

### 关键发现

- **自感知重排贡献最大**：消融显示去掉重排后 F1 下降 4-5%，大于去掉自感知检索（-2.5%）和推理（-2.2%），说明"如何整合知识"比"何时检索"更重要
- **内部状态远优于输出概率**：用输出概率替代内部状态后 F1 下降 6-8%，验证了内部状态承载更丰富的不确定性信息
- **Self-RAG 在复合 QA 上崩塌**：基于 NQ 微调的 Self-RAG 在多跳 QA 上 F1 仅 17.5-19.6%，甚至低于无检索的 CoT，说明微调类方法的分布偏移问题严重
- **自适应检索 vs 全量检索**：SeaKR 检索次数约为 IRCoT 的 60%，但 F1 高出 9.5%，证明减少不必要检索确实能提升性能

## 亮点与洞察

- **内部状态驱动的 RAG 决策是全新范式**：SeaKR 首次将 LLM 内部状态的自感知不确定性引入 RAG 的每个环节（检索、重排、推理），建立了"不确定性驱动的自适应 RAG"范式
- **Gram 行列式的不确定性度量**：通过多次采样隐藏表示的一致性来量化不确定性，这个度量方式兼顾了理论优雅性（Gram 行列式度量线性无关程度）和实践有效性（无需额外训练）
- **无需训练即可泛化**：相比 Self-RAG 的微调方案在分布偏移时崩塌，SeaKR 的 tuning-free 设计在不同类型 QA 上都保持稳健

## 局限与展望

- **推理开销大**：每次不确定性评估需要 20 次采样，每次重排需要 N×20 次推理，计算成本显著高于基线方法
- **仅在 LLaMA-2-7B 上验证**：缺少对更大规模和更新 LLM 的验证
- **BM25 检索器限制**：使用 BM25 而非密集检索器，检索质量可能是性能瓶颈
- **简单 QA 上优势不明显**：Self-RAG 在简单 QA 上仍有优势，说明 SeaKR 的优势主要体现在多跳推理场景

## 相关工作与启发

- **vs DRAGIN**：DRAGIN 也做自适应检索但仅用输出 token 概率，SeaKR 用内部状态 F1 高出 6%
- **vs Self-RAG**：Self-RAG 通过微调让模型自判断检索需求，但存在严重的分布偏移问题；SeaKR 无需微调，泛化性更强
- **vs INSIDE (Chen et al. 2023a)**：SeaKR 将 INSIDE 的内部状态不确定性度量从幻觉检测扩展到 RAG 全流程

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将内部状态自感知引入 RAG 的检索/重排/推理三环节
- 实验充分度: ⭐⭐⭐⭐ 多数据集、详细消融、组件贡献分析清晰
- 写作质量: ⭐⭐⭐⭐ 问题定义明确，框架结构清晰
- 价值: ⭐⭐⭐⭐ 开创了不确定性驱动的自适应 RAG 范式，尤其在复合 QA 上效果显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)

</div>

<!-- RELATED:END -->

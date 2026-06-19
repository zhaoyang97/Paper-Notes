---
title: >-
  [论文解读] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models
description: >-
  [ACL 2025][幻觉检测][检索增强] 提出 REFIND 框架，通过计算每个 token 在有无检索文档条件下的生成概率之比（Context Sensitivity Ratio, CSR），实现对 LLM 输出中幻觉片段的高效检测，在 SemEval-2025 Task 3 的 9 种语言上显著超越基线。
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "检索增强"
  - "上下文敏感度"
  - "多语言"
  - "token级别分析"
---

# REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.13622](https://arxiv.org/abs/2502.13622)  
**代码**: [https://github.com/oneonlee/REFIND](https://github.com/oneonlee/REFIND)  
**领域**: 幻觉检测  
**关键词**: 幻觉检测, 检索增强, 上下文敏感度, 多语言, token级别分析

## 一句话总结

提出 REFIND 框架，通过计算每个 token 在有无检索文档条件下的生成概率之比（Context Sensitivity Ratio, CSR），实现对 LLM 输出中幻觉片段的高效检测，在 SemEval-2025 Task 3 的 9 种语言上显著超越基线。

## 研究背景与动机

LLM 生成的幻觉内容（即与事实不符的输出）严重限制了其在知识密集型任务中的可靠性。现有幻觉检测方法存在明显不足：

**Token级分类器（如基于 XLM-RoBERTa）**：仅依赖模型内部知识进行二分类，不利用外部证据，在低资源语言上表现极差

**FAVA（检索增强编辑方法）**：虽然引入外部知识，但采用多步流水线（检索→比较→编辑），步骤间的对齐容易引入误差，且流程复杂

核心问题在于：**如何更直接、高效地利用检索到的外部文档来定位 LLM 输出中的幻觉片段？**

REFIND 的关键洞察是：如果一个 token 是幻觉（虚构的），那么当提供了正确的外部证据后，模型对该 token 的生成概率应当发生显著变化。反之，如果一个 token 是事实性的，外部证据不会大幅改变其生成概率。

## 方法详解

### 整体框架

REFIND 的三步流程：(1) 给定问题 q，使用检索器 R 检索相关文档集 D；(2) 用冻结的 LLM 分别计算每个 token 在有/无检索上下文条件下的生成概率；(3) 计算 CSR，超过阈值 δ 的 token 被标记为幻觉。

### 关键设计

1. **Context Sensitivity Ratio (CSR)**：核心指标，定义为：

    $CSR(t_i) = \frac{\log p_\theta(t_i | D, q, t_{<i})}{\log p_\theta(t_i | q, t_{<i}) + \varepsilon}$

    - 分子：在问题 q、历史 token t_{<i} 和检索文档 D 条件下的 log 概率
    - 分母：仅在问题和历史 token 条件下的 log 概率（+ ε 防除零）
    - CSR 高意味着检索上下文对该 token 的生成产生了强影响，即该 token 可能是幻觉
    - **设计动机**：与其让另一个模型去"判断"某段文本是否幻觉，不如直接观察原始 LLM 在获得正确证据后的概率变化——这是一种更本质、更直接的信号

2. **混合检索策略**：采用稀疏+稠密混合检索。先用 BM25 从预处理的多语言 Wikipedia 语料库中检索 Top-10 文档，再用 multilingual-e5-large 重排选出最终 5 篇文档。为保持跨语言一致性，统一使用多语言嵌入模型。

3. **阈值判定**：CSR ≥ δ 则判定为幻觉。δ 是可调超参数，用于平衡精确率和召回率。实验表明大多数语言在 δ=0.1~0.4 范围内表现稳定。

### 技术细节

- 无上下文概率 p(t_i|q, t_{<i}) 直接使用 Mu-SHROOM 数据集提供的 token 概率
- 有上下文概率 p(t_i|D, q, t_{<i}) 通过 PyTorch 2 计算
- 不需要训练，是一种 zero-shot 的检测方法

## 实验关键数据

### 主实验（IoU 指标，越高越好）

| 方法 | AR | CS | DE | EN | ES | EU | FI | FR | IT | 平均 |
|------|------|------|------|------|------|------|------|------|------|------|
| XLM-R | 0.042 | 0.096 | 0.032 | 0.031 | 0.072 | 0.021 | 0.004 | 0.002 | 0.010 | 0.035 |
| FAVA | 0.217 | 0.235 | 0.386 | 0.281 | 0.235 | 0.387 | 0.230 | 0.212 | 0.326 | 0.279 |
| **REFIND** | **0.374** | **0.276** | 0.352 | **0.353** | 0.215 | **0.407** | **0.506** | **0.473** | 0.313 | **0.363** |

### 消融实验（阈值敏感性分析）

| 阈值 δ | 整体表现 | 说明 |
|--------|---------|------|
| 0.1 | 高 | 高资源语言稳定，低资源语言波动略大 |
| 0.2 | 高 | 整体最稳定区间 |
| 0.3 | 中高 | 英语、德语保持~0.35 |
| 0.4 | 中 | 部分低资源语言性能下降 |

### 关键发现

- **REFIND 平均 IoU 达 0.363**，比 FAVA（0.279）高出 30%，比 XLM-R（0.035）高出 10 倍
- **低资源语言提升尤为显著**：芬兰语从 0.230 到 0.506，法语从 0.212 到 0.473，阿拉伯语从 0.217 到 0.374
- XLM-R 在芬兰语和法语上几乎完全失效（IoU<0.01），说明仅靠模型内部知识难以应对低资源语言
- REFIND 在多数语言上对阈值选择不敏感，表明方法的鲁棒性
- 检索质量直接影响 CSR 计算的可靠性

## 亮点与洞察

- **CSR 的核心思想简洁而有效**：不需要训练额外模型，直接利用 LLM 自身的概率分布变化来检测幻觉，计算成本远低于需要微调的方法
- **多语言零样本能力突出**：得益于多语言检索器和 CSR 的语言无关性，在 9 种语言（含低资源）上均表现良好
- **比 FAVA 更直接**：避免了多步流水线中的对齐误差，CSR 在 token 级别提供了清晰的幻觉信号
- **case study 清晰**：例如对 "Chance the Rapper 何时出道" 的回答中，"2011" 被正确识别为幻觉，因为检索文档给出了不同日期

## 局限与展望

- **依赖检索质量**：如果检索器未能返回高质量文档，CSR 计算会不准确甚至产生误判
- **计算开销**：需要分别计算有无上下文的 token 概率，在低延迟场景下可能是瓶颈
- **仅关注事实性幻觉**：未测试在非事实问答（如观点类、创意类问题）中的表现
- **阈值为固定超参数**：虽然实验表明鲁棒性较好，但自适应阈值机制（如基于语言或领域）可能进一步提升性能
- **西班牙语表现异常**：REFIND 在西班牙语上（0.215）反而低于 FAVA（0.235），原因未深入分析

## 相关工作与启发

- 与 SelfCheckGPT 类似，REFIND 也利用 LLM 自身信息检测幻觉，但引入了外部检索证据，克服了纯内部方法的局限
- 语义熵方法（Farquhar et al., 2024）判断整个回答是否幻觉，而 REFIND 定位到具体 span，粒度更细
- CSR 的理念可推广到其他需要判断"信息来源可靠性"的场景，如自动事实核查

## 评分

- **新颖性**: ⭐⭐⭐⭐ — CSR 指标的设计思路新颖直观，直接量化上下文敏感度是一个优雅的解决方案
- **实验充分度**: ⭐⭐⭐ — 9种语言的评测覆盖面好，但消融有限（仅阈值分析），缺少对检索器、LLM选择等变量的消融
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰简洁，图示直观
- **价值**: ⭐⭐⭐⭐ — 提供了一种轻量级、无需训练的幻觉检测范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ACL 2025\] Removal of Hallucination on Hallucination: Debate-Augmented RAG](removal_of_hallucination_on_hallucination_debate-augmented_rag.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](../../ACL2026/hallucination/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)

</div>

<!-- RELATED:END -->

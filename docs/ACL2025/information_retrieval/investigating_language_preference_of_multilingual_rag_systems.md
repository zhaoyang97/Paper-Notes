---
title: >-
  [论文解读] Investigating Language Preference of Multilingual RAG Systems
description: >-
  [ACL 2025][信息检索/RAG][multilingual RAG] 系统研究多语言 RAG 系统在检索和生成两个阶段的语言偏好问题，提出 MLRS 指标量化检索器对特定语言的偏好程度，揭示检索器偏好高资源语言和查询语言、生成器偏好查询语言和拉丁字母语言的现象，并设计 DKM-RAG 框架通过融合翻译段落与模型内部知识有效缓解偏好问题。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "multilingual RAG"
  - "language preference"
  - "MLRS"
  - "DKM-RAG"
  - "cross-lingual retrieval"
---

# Investigating Language Preference of Multilingual RAG Systems

**会议**: ACL 2025  
**arXiv**: [2502.11175](https://arxiv.org/abs/2502.11175)  
**代码**: [GitHub](https://github.com/jeonghyunpark2002/LanguagePreference.git)  
**领域**: NLP / 多语言检索增强生成  
**关键词**: multilingual RAG, language preference, MLRS, DKM-RAG, cross-lingual retrieval

## 一句话总结

系统研究多语言 RAG 系统在检索和生成两个阶段的语言偏好问题，提出 MLRS 指标量化检索器对特定语言的偏好程度，揭示检索器偏好高资源语言和查询语言、生成器偏好查询语言和拉丁字母语言的现象，并设计 DKM-RAG 框架通过融合翻译段落与模型内部知识有效缓解偏好问题。

## 研究背景与动机

多语言检索增强生成（mRAG）系统通过整合多语言外部知识来增强 LLM 的回答能力，但面临严重的**语言偏好**问题，导致检索到的文档不是最优的、生成的答案不准确或不一致。

**两个核心问题**：

1. **检索器语言偏好**：检索器倾向于优先选择高资源语言（如英语）或与查询同语言的文档，即使低资源语言中包含更相关的信息。例如，韩语查询搜索多语言知识库时，英语文档可能因语言优势排名更高，而真正相关的韩语文档反而被埋没。这导致生成器因缺少相关输入而产生错误答案或拒绝回答。
2. **生成器语言偏好**：即使检索到了多语言相关文档，生成器也可能偏爱查询语言或拉丁字母语言的段落，忽视其他语言中的关键证据，导致跨语言回答不一致。

**此前研究的不足**：

- 主要关注有限的语言组合，未能反映不同语言文档的真实排序动态
- Language-Preference-Based Re-ranking 等方法仅聚焦单一 mRAG 阶段
- 缺乏量化检索器语言偏好的标准指标

本文围绕三个研究问题展开：检索器偏好哪些语言？生成器偏好哪些语言以及如何影响性能？如何缓解语言偏好？

## 方法详解

### 整体框架

本文是一项系统性的实证研究 + 方法提出的工作，结构为：
1. 提出 MLRS 指标量化检索器语言偏好（RQ1）
2. 通过多语言答案一致性评估生成器语言偏好（RQ2）
3. 分析语言偏好与 mRAG 性能的相关性
4. 提出 DKM-RAG 框架缓解语言偏好（RQ3）

### 关键设计

1. **MLRS（MultiLingualRankShift）指标**:

    - 功能：量化检索器对特定语言的偏好程度，回答"如果消除语言差异，排名会提升多少？"
    - 核心思路：三阶段计算——(i) 对查询 $q$ 从多语言知识库检索文档，获得初始排名 $r_d^{init}$；(ii) 将非查询语言文档翻译为查询语言；(iii) 对翻译后的文档重排序，衡量排名变化 $\Delta r_d = \max(r_d^{init} - r_d^{re\text{-}rank}, 0)$
    - 归一化：$MLRS_q = \frac{\Delta r_q}{\Delta r_q^{max}} \times 100$，其中 $\Delta r_q^{max}$ 为最大可能提升（所有文档都升到第一名），最终对所有查询取平均
    - 核心洞察：如果翻译到某种目标语言后排名大幅上升，说明是语言差异（而非内容差异）导致了原始排名的下降——即存在对该目标语言的偏好
    - 设计动机：现有方法仅用统计等价性测试或简单的公平性指标，无法精确捕捉排名动态中的语言偏好

2. **生成器语言偏好评估**:

    - 功能：评估 LLM 在不同语言下回答的一致性，揭示生成器的语言倾向
    - 核心思路：对同一查询和同一检索文档集，让生成器分别用 8 种语言（en, ko, zh, fr, ja, it, pt, es）回答，计算所有语言对之间回答的嵌入相似度（使用 LaBSE），得到 8×8 相似度矩阵。某语言的偏好 = 该语言回答与其余语言回答的平均相似度
    - 设计动机：如果生成器不受语言偏好影响，那么同一输入下不同语言的回答应当语义一致

3. **DKM-RAG（Dual Knowledge Multilingual RAG）**:

    - 功能：缓解 mRAG 中的语言偏好，提升跨语言回答质量
    - 四步流程：
        - Step 1（检索+重排序）：从多语言知识库检索文档，使用 BGE-m3 重排序
        - Step 2（翻译）：将所有检索文档翻译为查询语言，得到 $P_{translated}$
        - Step 3（精炼）：用 Rewriter LLM 结合内部知识重写翻译段落——去除冗余、过滤不相关信息、补充可靠内容，得到 $P_{refined}$
        - Step 4（融合）：拼接 $P_{translated}$ 和 $P_{refined}$ 作为生成器的最终输入
    - 设计动机：仅翻译可纠正语言不匹配但无法过滤高资源语言中的不相关内容；精炼利用模型内部知识可以进一步提升信息质量。双知识源（外部翻译 + 内部精炼）互补

### 损失函数 / 训练策略

本文不涉及模型训练——MLRS 是一个评估指标，DKM-RAG 是 inference-time 的框架，所有组件（检索器、翻译模型、重写者、生成器）直接使用现成模型：

- **检索器**：BGE-m3（主要）、p-mMiniLM、p-mMpNet
- **翻译模型**：NLLB-200-distilled-600M（主要）、GPT-4o-mini（翻译质量实验）
- **生成器**：aya-expanse-8B、Qwen2.5-7B-Instruct、Phi-4 14B、Llama-3.1-8B-Instruct
- **语义相似度**：LaBSE（多语言句子嵌入）
- **评估指标**：character 3-gram recall
- **数据集**：MKQA（10k 样本，25 种语言，基于英文 Wikipedia），取 2.7k 与 KILT NQ 重叠子集

## 实验关键数据

### 主实验

**检索器语言偏好（MLRS 得分，BGE-m3）**：

| 查询语言 | 同语言匹配 | → en | → ko | → zh | → fr |
|---------|-----------|------|------|------|------|
| en | 56.03 | — | 33.02 (-23.0) | 33.10 (-22.9) | 36.61 (-19.4) |
| ko | 43.49 | 41.15 (-2.3) | — | 34.42 (-9.1) | 36.42 (-7.1) |
| zh | 45.26 | 44.98 (-0.3) | 34.52 (-10.7) | — | 36.34 (-8.9) |
| fr | 43.18 | 47.23 (+4.1) | 33.29 (-9.9) | 33.58 (-9.6) | — |

**DKM-RAG 性能对比（character 3-gram recall）**：

| 查询语言 | 模型 | all | 最优单语言 | DKM-RAG |
|---------|------|-----|----------|---------|
| en | aya-expanse-8B | 80.09 | 79.34 (en) | **82.60** |
| zh | aya-expanse-8B | 32.55 | 38.31 (zh) | **44.57** |
| ko | aya-expanse-8B | 40.60 | 49.66 (ko) | **55.01** |
| en | Phi-4 | 79.69 | 78.89 (en) | **82.59** |
| zh | Phi-4 | 16.75 | 36.76 (zh) | **44.56** |
| ko | Phi-4 | 26.80 | 49.25 (ko) | **54.82** |

### 消融实验

DKM-RAG 消融（aya-expanse-8B）：

| 配置 | Lq=en | Lq=zh | Lq=ko |
|------|-------|-------|-------|
| DKM-RAG (完整) | **82.60** | **44.57** | **55.01** |
| w/o $P_{refined}$ | 79.34 (-3.26) | 38.31 (-6.26) | 49.66 (-5.35) |
| w/o $P_{translated}$ | 81.10 (-1.50) | 39.44 (-5.13) | 46.15 (-8.86) |

两个组件都对性能至关重要：移除精炼段落 $P_{refined}$ 导致非英语查询性能大幅下降（-5~6 点），移除翻译段落 $P_{translated}$ 影响更大（-5~9 点），说明外部翻译知识和内部精炼知识的互补效应是 DKM-RAG 成功的关键。

### 关键发现

1. **检索器强烈偏好同语言和高资源语言**：同语言匹配时 MLRS 最高（56.03 for en-en），英语作为文档语言时几乎总获得最高偏好——甚至在非英语查询时也能超过同语言匹配（如法语查询时英语文档偏好 47.23 > 法语同语言 43.18）
2. **语族相近的语言偏好差距较小**：罗曼语族（法、意、葡、西）之间的跨语言偏好下降仅 1-6 点，而到东亚语言（中、日、韩）的下降达 7-23 点
3. **文档语言资源比查询语言资源更关键**：文档语言的资源水平对 MLRS 有显著影响（高资源 > 中资源 > 低资源），而查询语言的资源水平影响有限
4. **生成器偏好拉丁字母语言和查询语言**：拉丁字母语言（en, fr, it, pt, es）间的回答一致性远高于非拉丁语言（ko, zh, ja）；查询语言的回答一致性略高但提升有限
5. **高偏好≠高性能**：检索器偏好英语并不意味着用英语文档回答所有查询最好——非英语查询时，用查询语言文档回答效果更优
6. **DKM-RAG 全面且大幅提升性能**：在所有查询语言和所有生成器上均为最优，非英语查询的提升尤为显著（中文 +6~28 点，韩语 +5~28 点）

## 亮点与洞察

- **MLRS 指标设计精巧**：通过"翻译后重排序"的反事实实验思路，巧妙地将语言因素与内容因素解耦，为量化语言偏好提供了标准化的工具
- **全面覆盖 mRAG 管线两侧**：既分析检索器偏好又分析生成器偏好，并探究两者的交互关系，比此前只聚焦单一阶段的工作更有体系
- **"高偏好≠高性能"的反直觉发现**：挑战了"用高资源语言主导 mRAG 即可提升性能"的简单假设
- **DKM-RAG 方法简洁有效**：无需训练任何新模型，仅通过翻译+精炼+拼接的推理时流程即可大幅缓解语言偏好，易于即插即用

## 局限与展望

- **依赖翻译质量**：MLRS 指标和 DKM-RAG 框架都高度依赖翻译模型的准确性，翻译错误可能扭曲原始语义
- **计算开销增加**：MLRS 需要翻译+重排序，DKM-RAG 需要翻译+重写+拼接，在大规模实时系统中延迟和成本可能是瓶颈
- **语言覆盖有限**：仅实验了 8 种语言（以高中资源语言为主），缺乏对低资源语言（如阿拉伯语、印地语、斯瓦希里语等）的验证
- **数据集局限**：MKQA 基于英文 Wikipedia，知识根基偏向英语，可能不能完全反映真正多语言知识场景下的偏好
- **DKM-RAG 的精炼步骤黑箱化**：Rewriter LLM 如何决定过滤/补充哪些信息缺乏可控性和可解释性
- **未探索可训练的融合机制**：当前简单拼接翻译和精炼段落，动态加权或注意力融合可能更优

## 相关工作与启发

- **vs Bergen**（mRAG 基线）：Bergen 探索了构建有效 mRAG 管线的各组件选择，本文在其基础上深入分析语言偏好问题并提出缓解方案
- **vs Language-Preference-Based Re-ranking**：仅在检索阶段重排序，未考虑生成器偏好；本文全面覆盖两个阶段
- **vs 跨语言 IR 研究**（Yang et al., Telemala & Suleman）：主要关注公平性指标或有限语言对，MLRS 提供了更精确的排名动态度量
- **启发**：(1) mRAG 系统设计应将语言偏好作为显式优化目标而非忽略；(2) 翻译虽简单但在 mRAG 中效果显著，是成本效益比很高的跨语言桥梁；(3) 内部知识（LLM parametric knowledge）与外部知识（retrieved passages）的融合是提升 RAG 质量的重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ MLRS 指标设计新颖，首次系统研究 mRAG 全管线的语言偏好问题
- 实验充分度: ⭐⭐⭐⭐ 8 种语言、3 种重排序编码器、4 种生成器，实验覆盖检索+生成两个阶段，消融完整
- 写作质量: ⭐⭐⭐⭐ RQ 驱动的结构清晰，分析层层递进，从现象到解决方案逻辑完整
- 价值: ⭐⭐⭐⭐ 对多语言 RAG 系统的设计和部署有实际指导意义，MLRS 和 DKM-RAG 均可直接应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](knowshiftqa_rag_knowledge_shifts.md)

</div>

<!-- RELATED:END -->

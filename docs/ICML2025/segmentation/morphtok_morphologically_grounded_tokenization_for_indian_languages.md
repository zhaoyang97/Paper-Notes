---
title: >-
  [论文解读] MorphTok: Morphologically Grounded Tokenization for Indian Languages
description: >-
  [ICML 2025 (TokShop Workshop)][语义分割][分词] 提出 MorphTok 框架，通过形态学感知的预分词步骤（查找表/语言模型）和约束 BPE（CBPE）算法处理印度语言中的依存元音问题，在机器翻译和语言建模任务上提升下游性能，并引入人类评估指标 EvalTok。 大语言模型（LLM）普遍依赖字…
tags:
  - "ICML 2025 (TokShop Workshop)"
  - "语义分割"
  - "分词"
  - "预分词"
  - "形态学感知分割"
  - "BPE"
  - "Constrained BPE"
  - "印度语言"
  - "连音拆分"
  - "人类评估指标"
---

# MorphTok: Morphologically Grounded Tokenization for Indian Languages

**会议**: ICML 2025 (TokShop Workshop)

**arXiv**: [2504.10335](https://arxiv.org/abs/2504.10335)

**作者**: Maharaj Brahma, N J Karthika, Atul Kumar Singh, Devaraja Adiga, Smruti Bhate, Ganesh Ramakrishnan, Rohit Saluja, Maunendra Sankar Desarkar

**领域**: 分割 (子词分割 / 文本分词)

**关键词**: 分词, 预分词, 形态学感知分割, BPE, Constrained BPE, 印度语言, 连音拆分, 人类评估指标

## 一句话总结

提出 MorphTok 框架，通过形态学感知的预分词步骤（查找表/语言模型）和约束 BPE（CBPE）算法处理印度语言中的依存元音问题，在机器翻译和语言建模任务上提升下游性能，并引入人类评估指标 EvalTok。

## 研究背景与动机

大语言模型（LLM）普遍依赖字节对编码（BPE）进行子词分词。BPE 贪心地合并高频字符二元组（bigram），产生的分割往往与语言学上有意义的单元不对齐。这一问题在**形态学丰富的印度语言**（如印地语 Hindi、马拉地语 Marathi）中尤为严重，主要体现在：

**连音（Sandhi）现象**：复合词在词素边界发生音变，标准 BPE 无法识别这些形态边界，导致分词结果缺乏语言学意义

**依存元音（Dependent Vowels）**：印度语言使用元音附标文字（Abugida），元音符号依附于辅音，标准 BPE 可能将依存元音拆分为独立 token，破坏字符的内聚性

**多语言分词偏差**：现有多语言模型（如 IndicTrans2）使用统一的 BPE，对印度语言的 fertility（生育率，即每词平均 token 数）偏高，增加计算开销

现有解决方案如 Morfessor 等无监督形态分割方法对印度语言效果有限，且缺乏针对元音附标文字特性的专门设计。

## 方法详解

### 整体框架

MorphTok 由三个核心组件构成：

1. **形态学感知预分词（Morphology-Aware Pre-tokenization）**：在 BPE 之前，先将词语拆分为形态学上有意义的单元
2. **约束 BPE（CBPE）**：扩展标准 BPE 算法，加入脚本特定的约束条件
3. **EvalTok 评估指标**：面向人类评估的分词质量度量

### 关键设计一：形态学预分词

提出两种实现方式：

- **查找表（Lookup Table）方式**：基于人工标注的形态分割数据集（Hindi 54k 条、Marathi 58k 条），包含连音拆分（Sandhi Splitting）。将复合词拆分为词根形态素，还原音变前的原始形式
- **语言模型（LM）方式**：使用语言模型自动进行形态学分割，减少对人工标注的依赖

预分词的形式化描述：给定词 $w$，预分词函数 $f_{\text{pre}}$ 将其映射为形态素序列：

$$f_{\text{pre}}(w) = (m_1, m_2, \ldots, m_k)$$

其中每个 $m_i$ 为语言学上有意义的形态单元。对于连音现象，还需进行**归一化**（normalization），使分割后的形态素能匹配已有词汇。

### 关键设计二：约束 BPE（CBPE）

标准 BPE 的合并规则为：在每一步选择频率最高的字符对 $(a, b)$ 进行合并。CBPE 在此基础上增加脚本特定约束：

$$\text{Merge}(a, b) = \begin{cases} \text{允许合并} & \text{if } (a, b) \text{ 满足约束 } C \\ \text{禁止合并} & \text{otherwise} \end{cases}$$

核心约束 $C$：**依存元音必须与其依附的辅音形成内聚单元**，即禁止将依存元音（如 ा, ि, ी 等）作为独立 token 拆分出来。这确保了 CBPE 产生的 token 在文字系统层面是合理的。

### 损失函数

下游任务使用标准训练损失。对于**机器翻译**，使用交叉熵损失：

$$\mathcal{L}_{\text{MT}} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, X; \theta)$$

对于**语言建模**，使用自回归语言模型的困惑度（perplexity）作为评估指标：

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i})\right)$$

### 评估指标：EvalTok

提出人类评估指标 EvalTok，评估分词结果的形态学和语义质量。通过人工标注者对 100 个采样词语的分词结果进行评分，衡量分词是否与有意义的词素边界对齐，实现更贴近人类判断的评估。

## 实验关键数据

### 主实验：机器翻译性能（En↔Hi, En↔Mr）

| 分词方法 | BLEU (En→Hi) | chrF (En→Hi) | COMET (En→Hi) | BLEU (Hi→En) | chrF (Hi→En) |
|---------|-------------|-------------|--------------|-------------|-------------|
| BPE (baseline) | — | — | — | — | — |
| CBPE | ≈ BPE | ≈ BPE | ≈ BPE | ≈ BPE | ≈ BPE |
| Lookup + BPE | **↑** | **↑** | **↑** | **↑** | **↑** |
| Lookup + CBPE | **↑** | **↑** | **↑** | **↑** | **↑** |

**关键发现**：Lookup + BPE 和 Lookup + CBPE 在所有翻译方向上均优于标准 BPE，形态学预分词带来的增益大于 CBPE 约束本身。

### 语言建模性能

| 分词方法 | Perplexity (Hindi) | Loss (Hindi) | Perplexity (Marathi) | Loss (Marathi) |
|---------|-------------------|-------------|---------------------|---------------|
| BPE (baseline) | 高 | 高 | 高 | 高 |
| CBPE | 中 | 中 | 中 | 中 |
| Lookup + BPE | **低** | **低** | **低** | **低** |
| Lookup + CBPE | **最低** | **最低** | **最低** | **最低** |

**关键发现**：Lookup + CBPE 在语言建模中一致取得最低困惑度和损失，验证形态学感知分词对语言理解的提升。

### Fertility 分析

| 分词方法 | Fertility (Hindi) | Fertility (Marathi) | 较 BPE 降幅 |
|---------|-------------------|--------------------|-----------| 
| BPE (baseline) | 1.8858 | — | — |
| CBPE | 1.8174 | — | **-1.68%** |
| Lookup + BPE | — | — | — |
| Lookup + CBPE | — | — | — |

**关键发现**：CBPE 将 fertility 降低 1.68%（约每 100 词节省 7 个 token），虽然改进幅度不大但在保持下游性能的同时提升了分词效率。

### 人类评估（EvalTok）

| 评估维度 | BPE | CBPE | Lookup + BPE | Lookup + CBPE |
|---------|-----|------|-------------|--------------|
| 形态学对齐度 | 低 | 中 | **高** | **最高** |
| 语义一致性 | 低 | 中 | **高** | **最高** |

**关键发现**：EvalTok 人类评估确认形态学预分词方法（Lookup 系列）在形态学和语义层面产生更优质的分词结果。

## 亮点与洞察

1. **语言学驱动的分词优化**：不同于纯统计方法，利用形态学知识（连音拆分 + 词素边界）指导分词，语言学意义明确
2. **CBPE 的脚本适配设计**：针对元音附标文字（Abugida）的依存元音特性引入约束，思路可推广至其他使用类似书写系统的语言（如泰语、缅甸语）
3. **EvalTok 填补评估空白**：现有分词评估主要依赖 fertility 等自动指标，EvalTok 提供人类视角的评估，更全面衡量分词质量
4. **数据集贡献**：发布 Hindi 54k + Marathi 58k 的形态分割标注数据集，是印度语言 NLP 的重要资源

## 局限性

1. **仅评估两种语言**：实验限于 Hindi 和 Marathi，对其他印度语言（如 Tamil、Bengali、Telugu）的效果未验证
2. **查找表依赖人工标注**：Lookup 方式需要语言学专家提供形态分割标注，扩展到新语言成本较高
3. **Fertility 改进有限**：CBPE 的 fertility 降低仅 1.68%（约每 100 词节省 7 个 token），实际计算收益较小
4. **缺乏与 LLM 规模实验**：实验仅在中小规模模型上进行，在大规模 LLM 上的效果未知
5. **单向翻译任务为主**：缺少从英语到其他印度语言方向的评估（审稿人也提出此问题）

## 相关工作与启发

- **BPE (Sennrich et al., 2016)**：标准子词分词方法，贪心合并策略忽略语言学结构
- **Morfessor**：无监督形态分割方法，对形态学复杂语言效果有限
- **IndicTrans2**：多语言印度语言翻译模型，使用统一 BPE 导致 token 过多
- **German Compound Splitting**：德语复合词拆分工作面临类似的词素边界音变问题，方法思路可互相借鉴

**启发**：形态学感知分词的思路可推广至所有形态学丰富的语言。CBPE 的约束设计模式可扩展为通用的"脚本感知 BPE"框架，根据不同文字系统的特性自动选择约束规则。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3 | 预分词 + 约束 BPE 思路直观，但形态学分词并非全新概念 |
| 技术深度 | 3 | Workshop 论文，方法较为直接，缺乏深层理论分析 |
| 实验充分性 | 3 | 仅两种语言、中小规模模型实验，缺乏大规模验证 |
| 写作质量 | 3 | 结构清晰但有冗余（审稿人指出重复描述 EvalTok） |
| 实用价值 | 4 | 数据集发布 + 即插即用方法组合，可直接用于印度语言系统 |
| 总评 | 3.2 | 实用导向的 workshop 工作，解决真实问题但深度有限 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](../../ICCV2025/segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](../../NeurIPS2025/segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](actionpiece_contextually_tokenizing_action_sequences_generative_recommendation.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)

</div>

<!-- RELATED:END -->

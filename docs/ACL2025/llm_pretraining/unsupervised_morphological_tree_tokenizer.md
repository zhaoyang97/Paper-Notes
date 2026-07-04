---
title: >-
  [论文解读] Unsupervised Morphological Tree Tokenizer
description: >-
  [ACL2025][预训练][tokenization] 提出 TreeTok，一种基于无监督神经形态结构归纳的分词器，通过 MorphOverriding 机制和自监督目标学习字符级树结构，以自顶向下词表匹配方式进行分词，在形态分割和语言建模任务上均优于 BPE/WordPiece。 领域现状：BPE 和 WordPiec…
tags:
  - "ACL2025"
  - "预训练"
  - "tokenization"
  - "morphology"
  - "unsupervised parsing"
  - "composition model"
  - "BPE"
---

# Unsupervised Morphological Tree Tokenizer

**会议**: ACL2025  
**arXiv**: [2406.15245](https://arxiv.org/abs/2406.15245)  
**代码**: [GitHub](https://github.com/martianmartina/TreeTokenizer)  
**领域**: LLM预训练  
**关键词**: tokenization, morphology, unsupervised parsing, composition model, BPE

## 一句话总结

提出 TreeTok，一种基于无监督神经形态结构归纳的分词器，通过 MorphOverriding 机制和自监督目标学习字符级树结构，以自顶向下词表匹配方式进行分词，在形态分割和语言建模任务上均优于 BPE/WordPiece。

## 研究背景与动机

**领域现状**：BPE 和 WordPiece 是当前语言模型的事实标准分词器，被 GPT、BERT 等广泛采用。

**现有痛点**：这些统计分词器经常破坏词内的语素边界，产生不自然的碎片化 token。例如 BPE 将 "windsurfing" 切成 "wind/sur/fing"，破坏了 "surf" 和 "ing" 的语素完整性。

**核心矛盾**：语言学理论认为词和句子一样具有内部结构，但现有分词器在自底向上合并过程中必须保留所有中间"垃圾" token（如 "lo", "fing"），导致词表空间浪费且语义信息丢失。

**本文目标**：如何在无监督条件下归纳出符合形态学规则的字符级树结构，并基于此实现更好的分词？

**切入角度**：将句法组合模型（composition model）从词级别降到字符级别，引入 MorphOverriding 机制处理语素不可分解性，用自监督目标训练。

**核心 idea**：用深度组合模型联合编码词的内部结构和表示，通过 MorphOverriding 保证语素的不可分解性，再以自顶向下词表匹配进行分词。

## 方法详解

### 整体框架

TreeTok 分为三个阶段：(1) 训练组合模型归纳字符级树结构；(2) 基于树结构构建精简词表；(3) 自顶向下词表匹配完成分词。

### 关键设计 1：MorphOverriding 组合模型

- **功能**：对给定词的字符序列，自底向上计算所有子串的组合表示和兼容性得分，归纳出最优二叉树结构。
- **为什么**：传统组合模型总是将子词表示分解为其组件的组合，但语素（如 "wind"）是最小意义单元，不应被进一步分解。如果 "wind" 的表示由 "w"+"ind" 组成，会导致与 "win" 等不相关语素的表示纠缠。
- **怎么做**：构建一个启发式语素词表 $\mathbb{V}$（用 BPE 生成），每个词条关联可学习嵌入。当子串命中词表时，将其语素嵌入 $\mathbf{s}_{i,j}$ 注入组合函数，使模型可以学习用语素嵌入混合或覆盖（override）组合向量。未命中则用空嵌入。组合函数基于多层 Transformer 实现。

### 关键设计 2：双重自监督训练目标

- **功能**：同时利用词内信息和词间上下文信息训练组合模型。
- **为什么**：仅预测字符（从几十个字符中选择）太简单，学不到好的结构；上下文信息可帮助消歧（如 "asking" 在连续时态语境下更倾向 "ask+ing" 而非 "as+king"）。
- **怎么做**：
    - **自编码损失 $\mathcal{L}_{ae}$（词内）**：预测词内所有命中词表的子串（不只是单字符），利用外部表示（outside representation）从上下文预测被遮盖的语素。每个子串还附加一个 constituency 权重以区分真正的成分与偶然重叠的子串。
    - **自回归损失 $\mathcal{L}_{ar}$（词间）**：将组合模型生成的词嵌入送入因果语言模型（GPT2），通过 batch 内采样构建候选词表进行下一个词预测。

### 关键设计 3：TreeTok 分词算法

- **分词过程**：自顶向下遍历解析树，若子词命中词表则保留并回溯；对解析错误导致的可合并碎片，用动态规划搜索使总概率最大的合并方案进行后处理。
- **词表构建**：先用树结构约束的 BPE 变体构建启发式词表（只合并树中同父节点的相邻 token 对），再用树结构约束的 Unigram 变体基于 delta 熵进行剪枝，去除不重要子词直到目标大小。
- **优势**：自顶向下匹配不需要保留自底向上合并的中间 token，可构建更紧凑的词表。

## 实验关键数据

### 主实验：形态分割准确率

| 方法 | Morpho (Acc.) ↑ | Compound (Acc.) ↑ | 词表大小 |
|------|:---:|:---:|:---:|
| BPE | 19.50 | 62.98 | 30,000 |
| WordPiece | 26.20 | 62.19 | 30,000 |
| Unigram | 27.10 | 53.10 | 30,000 |
| **TreeTok** | **37.90** | **68.07** | 30,000 |

### 语言建模与统计指标

| 方法 | Rényi ↑ | PPL ↓ | BLEU ↑ | 平均 token 数 |
|------|:---:|:---:|:---:|:---:|
| BPE | 44.66 | 107.76 | 26.55 | 26.58 |
| WordPiece | 44.54 | 110.97 | - | 26.60 |
| Unigram | 45.07 | 106.91 | - | 31.68 |
| **TreeTok** | 44.82 | 107.26 | **26.68** | **25.99** |

### 消融实验：树结构质量（语素召回率）

| 变体 | Morpho ↑ | Compound ↑ |
|------|:---:|:---:|
| Fast R2D2 | 67.69 | 48.96 |
| Neural PCFG | 39.87 | 58.33 |
| **TreeTok（完整）** | **90.10** | **86.20** |
| w/o context | 70.00 | 63.02 |
| w/o MorphOverriding | 75.99 | 46.35 |
| w/o span loss | 89.42 | 78.39 |

### 关键发现

- TreeTok 在形态分割上大幅超越 BPE（+18.4 Morpho, +5.1 Compound），同时平均 token 数最少（25.99），意味着更高效的编码。
- 去除 MorphOverriding 在 Compound 任务上性能暴跌近 50%（86.20→46.35），证实该机制对处理复合词至关重要。
- 去除上下文（自回归损失）使 Morpho 性能从 90.10 降至 70.00，说明词间上下文信息对消歧结构归纳极为关键。
- 中文实验中，无 MorphOverriding 即可达到 99.24% 词边界召回，因为中文字符即语素，不存在不可分解性问题。

## 亮点与洞察

1. **MorphOverriding 机制**：精巧地解决了组合模型与语素不可分解性之间的矛盾，是全文最关键的创新。当子串是语素时，其表示可以脱离组件，这一语言学洞察被优雅地编码为可学习的注入机制。
2. **自顶向下分词 + 词表剪枝**：打破了 BPE 自底向上合并的范式，避免了"垃圾 token"问题，能构建更紧凑有效的词表。
3. **跨语言验证**：中文实验（无需 MorphOverriding 即可成功）反向验证了英语等语言中 MorphOverriding 的必要性，增强了理论解释力。

## 局限与展望

1. **推理速度**：TreeTok 单 token 处理时间约为 BPE 的 80 倍（1.98e-3 vs 2.49e-5 秒），虽然可以通过 GPU 批处理和高频词缓存（hit rate 98.11%）缓解，但仍是实际部署的障碍。
2. **训练成本**：组合模型需要额外训练（WikiText-103 需 8×A100 不到一天），对于快速迭代场景不够轻量。
3. **主要在英语上验证**：除中文结构实验和德语翻译实验外，缺乏对形态更丰富语言（如土耳其语、芬兰语）的系统评估。
4. **启发式词表依赖**：MorphOverriding 的语素词表由 BPE 启发式构建，词表大小的选择对性能影响大（图 3 呈倒 U 形），最优值需要搜索。
5. **下游任务提升有限**：在 PPL 和 BLEU 上 TreeTok 相对 BPE 仅有微弱提升，形态对齐的收益尚未在大模型预训练中得到验证。

## 相关工作与启发

### vs BPE / WordPiece / Unigram
传统统计分词器通过频率驱动的自底向上合并（BPE/WordPiece）或自顶向下剪枝（Unigram）构建词表，完全不考虑词的内部结构。TreeTok 的核心差异在于引入了神经归纳的字符级树结构，使分词与形态学对齐。BPE 产生语义无关的中间 token，而 TreeTok 通过自顶向下匹配避免了这一问题。

### vs Morfessor
Morfessor 是经典的无监督形态分割方法，使用 MDL 代价函数和层次化分割策略。但它无法控制词表大小，不适合作为分词器。TreeTok 结合了 Morfessor 的形态动机和 BPE 的压缩能力，同时利用现代神经方法更好地捕获上下文和词内语义信息。

### vs 神经句法归纳（R2D2 / Neural PCFG）
R2D2 和 Neural PCFG 是句子级别的结构归纳模型。TreeTok 将它们降维到字符级别，并发现直接应用效果不佳（R2D2 在 Compound 仅 48.96%），根本原因是缺少 MorphOverriding。这一发现本身就是重要的学术贡献。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MorphOverriding 机制巧妙解决了组合模型在字符级别的根本困难，将形态学约束优雅地融入神经结构归纳中
- **实验充分度**: ⭐⭐⭐⭐ — 包含形态分割、语言建模、机器翻译、中文验证和详细消融，但缺少对更多形态丰富语言和大模型预训练的验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 论文结构清晰，motivation 层层递进，从语言学洞察到技术方案到实验验证非常流畅
- **价值**: ⭐⭐⭐⭐ — 为分词领域提供了新范式，但推理效率和大模型适配性仍需后续验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](../../ICML2025/llm_pretraining/evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[ACL 2025\] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)

</div>

<!-- RELATED:END -->

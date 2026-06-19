---
title: >-
  [论文解读] Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing
description: >-
  [ACL 2025][自监督学习][Constituency Parsing] 提出 LLM 反向生成 (LLM Back Generation) 方法，将不完整的跨领域句法树作为输入让 LLM 补全缺失词生成 treebank，并设计 span 级别对比学习预训练策略，实现跨领域成分句法分析的 SOTA 性能。
tags:
  - "ACL 2025"
  - "自监督学习"
  - "Constituency Parsing"
  - "Cross-domain"
  - "LLM Back Generation"
  - "对比学习"
  - "Treebank Generation"
---

# Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing

**会议**: ACL 2025  
**arXiv**: [2505.20976](https://arxiv.org/abs/2505.20976)  
**代码**: [GitHub](https://github.com/guopeiming/Back_Parsing_LLM)  
**领域**: 句法分析 / 跨领域迁移  
**关键词**: Constituency Parsing, Cross-domain, LLM Back Generation, Contrastive Learning, Treebank Generation  

## 一句话总结

提出 LLM 反向生成 (LLM Back Generation) 方法，将不完整的跨领域句法树作为输入让 LLM 补全缺失词生成 treebank，并设计 span 级别对比学习预训练策略，实现跨领域成分句法分析的 SOTA 性能。

## 研究背景与动机

**研究问题：** 跨领域成分句法分析因缺乏多领域标注 treebank 仍是未解决的挑战。现有的新闻领域 treebank (PTB) 训练的 parser 在其他领域上性能显著下降。

**现有方法的不足：** (1) LLM 直接进行句法分析性能很差（ChatGPT 仅 22.99% F1），因其自回归生成难以保证有效树结构；(2) Li et al. (2023) 用 LLM 生成原始文本再用 parser 标注伪 treebank，这种两阶段流水线间接利用 LLM，引入噪声和误差传播。

**核心动机：** 虽然 LLM 做正向句法分析（句子→树）很差，但如果反转这个过程——给定树的骨架和领域关键词，让 LLM 补全缺失的词——就能利用 LLM 的语言生成能力同时保证句法结构的有效性。

## 方法详解

### 整体框架

整体分为两个阶段：(1) **LLM 反向生成** 从目标领域句子中提取句法树骨架和领域关键词，掩码非关键词后让 LLM 补全生成跨领域 treebank；(2) **对比学习预训练** 在生成的 treebank 上进行 span 级对比学习，训练 constituent span 表示模型，再 fine-tune 到跨领域句法分析。

### 关键设计

1. **跨领域句法树准备：** 首先用基础 chart-based parser 解析目标领域无标注句子的句法树（获取领域句法结构），然后用 KeyBERT 提取与原句最相似的 25% 词作为领域关键词保留，掩码其余词。掩码树同时携带领域句法结构和领域词汇两个跨领域关键要素。

2. **LLM 反向生成：** 将掩码句法树与少量示例 (ICL demonstrations) 输入 LLM (GPT-4)，让 LLM 在保持树结构不变的前提下补全缺失词，生成 $(\hat{X}, Y)$ 对。这从输入端保证了句法树的有效性，避免了 LLM 直接解析时生成无效括号结构的问题。

3. **Span 级对比学习预训练：** 为每个 constituent span $(i,j)$，以其左子、右子、父节点和兄弟节点作为正例（4 个），以边界相邻的无效 span 作为负例（15 个）。对比目标拉近有效 constituent span 的表示、推远无效 span，显著扩展预训练数据量（平均每棵树约 25 个 span，10K 树 → 250K 预训练样本）。

### 损失函数

对比学习损失：

$$\mathcal{L} = -\sum_{m \in (i,j)^+} \log \frac{e^{f(\boldsymbol{r}, \boldsymbol{r}_m^+)}}{\sum_{n \in (i,j)^-} e^{f(\boldsymbol{r}, \boldsymbol{r}_m^+)} + e^{f(\boldsymbol{r}, \boldsymbol{r}_n^-)}}$$

其中 $f$ 为余弦相似度除以温度因子 $\tau$。Fine-tune 阶段使用标准 tree-based max-margin loss。

## 实验

### 主实验结果

在 MCTB 五个目标领域上的 F1 分数：

| 方法 | Dia | For | Law | Lit | Rev | Avg |
|------|-----|-----|-----|-----|-----|-----|
| ChatGPT (valid) | 70.38 | 70.36 | 80.70 | 74.74 | 69.08 | 73.05 |
| GPT-4 (valid) | 77.64 | 76.27 | 84.49 | 79.58 | 75.63 | 78.72 |
| Kitaev & Klein (2018) | 86.10 | 86.92 | 92.07 | 86.28 | 84.32 | 87.14 |
| Li et al. (2023) | 87.59 | 87.55 | 93.29 | 87.54 | 85.58 | 88.31 |
| Natural Corpus + CTPT | 87.33 | 87.80 | 92.54 | 86.91 | 84.35 | 87.79 |
| **LLM Back Gen + CTPT** | **87.92** | **88.13** | **93.22** | **87.50** | **85.86** | **88.52** |

### 消融实验

| 掩码率 | DAPT | NOPT | CTPT |
|--------|------|------|------|
| 0% (自然语料) | 87.15 | 87.38 | 87.79 |
| 25% (最佳) | 87.39 | 87.81 | **88.52** |
| 50% | — | — | 88.2 (约) |
| 100% | — | — | 87.8 (约) |

| 预训练策略 | Avg F1 |
|-----------|--------|
| DAPT (Domain Adaptive) | 87.15 |
| NOPT (No Pre-training) | 87.38 |
| **CTPT (Contrastive)** | **87.79** |

### 关键发现

- LLM 反向生成的 treebank 在所有预训练策略下均显著优于自然语料 treebank，表明反向生成有效引入了目标领域句法多样性。
- 25% 掩码率效果最佳——保留足够领域关键词引导 LLM，同时给予足够自由度引入变化。掩码率过高会导致生成偏离目标领域。
- 对比学习预训练 (CTPT) 收敛速度显著快于 DAPT 和 NOPT（600 步 vs 1000 步），且最终性能最高。
- 仅需 8K 棵树即可达到性能饱和，span 级对比学习有效放大了有限数据的价值。

## 亮点

- 巧妙地反转了句法分析任务方向：利用 LLM 的语言生成优势而非其不擅长的结构预测能力。
- 首次将对比学习引入成分句法分析，span 级设计显著放大训练数据量。
- 正/负例的构造紧密结合句法树的层次结构特性，语言学动机清晰。

## 局限性

- 仅在英语 MCTB 数据集上验证，缺乏其他语言的跨领域句法 treebank 进行多语言验证。
- 依赖 GPT-4 进行反向生成，其他 LLM（ChatGPT、LLaMA-3）多数无法生成有效句法树或错误率高。
- 在法律和文学等长句/正式领域表现略弱于 Li et al. (2023)，可能因单一 parser 处理全部领域需要平衡不同分布。

## 相关工作

- **跨领域句法分析：** Yang et al. (2022) 标注了多领域 treebank MCTB；Li et al. (2023) 用 LLM 生成目标领域原始文本再伪标注。
- **LLM 用于句法分析：** Bai et al. (2023) 全面评估了 ChatGPT/GPT-4/LLaMA 的句法分析能力，证明直接标注效果差。
- **对比学习：** SimCSE (Gao et al., 2021) 在句子表示中引入对比学习；本文首次将其移植到 span 级别的句法分析任务。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 实用性 | 7 |
| 实验充分度 | 8 |
| 写作质量 | 8 |
| 总体评分 | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2025/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](../../CVPR2026/self_supervised/learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)
- [\[ECCV 2024\] WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](../../ECCV2024/self_supervised/wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)

</div>

<!-- RELATED:END -->

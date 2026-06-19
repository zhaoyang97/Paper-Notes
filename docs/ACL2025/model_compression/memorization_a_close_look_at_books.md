---
title: >-
  [论文解读] Memorization: A Close Look at Books
description: >-
  [ACL 2025][模型压缩][LLM 记忆化] 系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。 LLM 的记忆化能力引发了严重的隐…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "LLM 记忆化"
  - "训练数据提取"
  - "版权"
  - "微调攻击"
  - "LoRA 权重分析"
---

# Memorization: A Close Look at Books

**会议**: ACL 2025  
**arXiv**: [2504.12549](https://arxiv.org/abs/2504.12549)  
**代码**: 无  
**领域**: 其他  
**关键词**: LLM 记忆化, 训练数据提取, 版权, 微调攻击, LoRA 权重分析

## 一句话总结

系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。

## 研究背景与动机

LLM 的记忆化能力引发了严重的隐私和版权问题。虽然已有大量研究证明 LLM 可以被提取训练数据中的敏感片段，但几个关键问题仍未得到充分回答：

**能否提取完整作品？** 现有研究主要关注短文本片段（如电话号码、邮箱）的提取，但对版权诉讼更重要的是：LLM 是否记住了整本书？

**什么决定了记忆化程度？** 训练数据的重复度与记忆化之间的定量关系在完整作品层面尚未被研究。

**对齐能否真正消除记忆化？** Nasr et al. (2025) 已证明可通过微调"撤销"对齐模型的反反刍措施，但其背后的权重变化模式仍不清楚。

研究选择书籍作为分析对象有两个原因：(a) 书籍处于多起版权诉讼的核心，(b) 书籍长且独特，是技术上最具挑战性的提取目标。

## 方法详解

### 整体框架

研究设计三组实验：
1. 基线提取：在 Llama 3 / 3.1 预训练和指令微调模型上测试书籍提取
2. SFT 攻击：通过 Nasr et al. 的微调技术尝试恢复 instruct 模型的记忆
3. 扩展研究 + 权重分析：32 本书的大规模分析 + LoRA 权重更新的深入剖析

### 关键设计

1. **数据集构建**：从 Project Gutenberg 收集 32 本英文书籍，沿两个维度控制：

    - **添加日期**：区分训练截止日期（2023年12月）前后的书籍
    - **流行度**：用 Goodreads 评分数作为代理（从 0 到 100 万+评分）
    - 预处理：去掉每本书前 2000 和后 5000 tokens（前言、版权信息等）

2. **Prefix Prompting 提取方法**：

    - 以 500 tokens 作为前缀提示
    - 让模型续写 30 tokens
    - 将续写内容与原文对应的 30 tokens 计算相似度
    - 全部使用贪心解码，确保输出确定性
    - 两种模式：逐片段提取（通过滑动窗口）和自回归生成（将输出循环喂回作为新的输入）

3. **SFT 微调攻击**：

    - 使用 43 本不在测试集中的 Gutenberg 书籍作为微调数据
    - 使用 QLoRA（LoRA rank=16）进行高效微调
    - 分别微调 500 和 1000 个样本
    - 微调格式：将前缀放在 user content，后缀放在 assistant content，教会模型"续写书籍"的行为

4. **权重更新分析**：这是本文最独特的贡献。对 SFT-1000 的 LoRA 模型，重建完整的权重更新矩阵：

    - 计算 $W_{update} = \alpha r^{-1} \cdot BA$
    - 计算相对更新 $W_{rel} = |W_{update} \oslash W_{original}|$（Hadamard 除法）
    - 分析更新分布：哪些层、哪些模块接收了最大的相对变化

### 评估指标

使用多种相似度指标：Jaccard 相似度、余弦相似度、Levenshtein 距离、BLEU、ROUGE-L、Sequence Matcher Similarity。主要结果以 Jaccard 相似度的中值报告。

## 实验关键数据

### 主实验：基线模型自回归提取 (Jaccard Similarity)

| 书籍 | GoodReads 评分数 | Llama 3 | Llama 3.1 | Llama 3.1 Instruct |
|------|-----------------|---------|-----------|-------------------|
| Alice in Wonderland | 413,400 | **~0.95** | ~0.7 | ~0.05 |
| The Time Machine | 546,286 | ~0.6 | ~0.4 | ~0.05 |
| Peter Pan | 362,694 | ~0.3 | ~0.2 | ~0.05 |
| Ethics | 19,734 | ~0.1 | ~0.05 | ~0.05 |
| Rosin the Beau | 2 | ~0.02 | ~0.02 | ~0.02 |
| A girl and her ways* | 0 | ~0.02 | ~0.02 | ~0.05 |

*: 截止日期后添加的书籍

### 消融实验：SFT 对 Instruct 模型的影响

| 书籍 | Instruct 基线 | +SFT 500 | +SFT 1000 | 预训练基线 |
|------|--------------|----------|-----------|-----------|
| Alice | ~0.05 | ~0.7 | **~0.91** | ~1.0 |
| Time Machine | ~0.05 | ~0.3 | ~0.4 | ~0.5 |
| Peter Pan | ~0.05 | ~0.2 | ~0.3 | ~0.4 |
| Ethics | ~0.05 | ~0.1 | ~0.15 | ~0.2 |
| Rosin the Beau | ~0.02 | ~0.02 | ~0.02 | ~0.02 |

扩展到 32 本书的相关性分析（Instruct SFT-1000）：
- Jaccard 相似度与 log(GoodReads 评分数) 的相关系数 = **0.5**（中强正相关）
- 最高提取率：The Communist Manifesto (0.95)、Alice (0.91)、Romeo and Juliet (0.76)
- 截止日期后的书籍（红点）：无论流行度如何，提取率均极低

### 权重更新分析关键发现

| 更新阈值 | 受影响权重比例 |
|---------|--------------|
| > 1% 相对变化 | ~14% |
| > 100% 相对变化 | ~0.15% |

更新在网络中的分布：
- **集中在底层 transformer blocks**（前 10-20 层）
- 自注意力层接收的更新比例约为 MLP 层的 **7 倍**
- 后层几乎不需要任何改变即可恢复记忆检索

### 关键发现

1. **Llama 3 可以自回归生成整本《爱丽丝梦游仙境》**：仅从前 500 tokens 出发，生成的全书内容与原文 Jaccard 相似度高达 ~0.95。
2. **流行度是记忆化的强预测因子**：GoodReads 评分数与提取率的相关性为 0.5，反映了流行作品在训练数据中更可能被重复。
3. **指令微调大幅降低但未消除记忆化**：Instruct 模型的提取率近乎为零，但仅需 ~1000 样本的 SFT 即可恢复大部分记忆。
4. **记忆抑制仅涉及极少量权重**：只有 ~0.15% 的权重发生了超过原始值 100% 的变化。
5. **底层是关键**：恢复记忆主要需要修改底层 transformer blocks，尤其是自注意力层。这暗示对齐训练可能通过修改底层的"检索门控"来抑制记忆输出，而非真正删除记忆。

## 亮点与洞察

1. **从片段提取到完整作品提取的跨越**：证明了 LLM 不仅记住了训练数据的片段，而是完整地记住了整本书，这对版权讨论有重大意义。
2. **流行度 → 重复度 → 记忆化的因果链**：虽然无法直接观测 Llama 的训练数据，但通过 GoodReads 评分作为代理，建立了令人信服的间接证据。
3. **权重分析视角独特且深入**：不仅揭示了"什么被改变了"，还揭示了"在哪里被改变了"。底层集中更新的发现暗示了一种新的模型安全研究方向——能否通过锁定底层权重来防止记忆恢复？
4. **实验设计中 cutoff date 控制很巧妙**：截止日期后添加的书籍作为天然的阴性对照，进一步验证了提取率确实反映了训练数据的存在。

## 局限与展望

1. **仅限 Llama 3.x 70B**：未验证其他模型族（GPT、Gemini等）和不同规模的模型。
2. **仅使用公共领域书籍**：更具争议性的受版权保护作品（如哈利波特）未被测试。
3. **流行度代理的局限**：GoodReads 评分适用于书籍，但对新闻、论文等内容缺乏类似指标。
4. **前缀长度固定 500 tokens**：更短的前缀对实际攻击场景更有意义，但本文未探索。
5. **未讨论防御对策**：既然底层权重是关键，是否可以通过冻结底层或增加底层正则化来防止记忆恢复？

## 相关工作与启发

- **Carlini et al. (2021/2023)**：LLM 记忆化和训练数据提取的先驱工作，证明了记忆化随模型规模、数据重复和提示长度增加
- **Nasr et al. (2025)**：提出 divergence attack 和 finetuning attack，本文在此基础上扩展到完整书籍并增加权重分析
- **Karamolegkou et al. (2023)**：研究内容流行度与记忆化的关系，但仅关注最长公共子序列，本文关注完整重建
- **LoRA (Hu et al., 2022) / QLoRA**：高效微调方法，本文将其用于"攻击"分析而非任务适应

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 完整书籍提取 + 权重更新分析的组合提供了新的视角
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多书籍、多样本量的系统性实验，32 本书的规模化分析
- **写作质量**: ⭐⭐⭐⭐ — 实验设计清晰，图表直观，结论有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 对 AI 安全和版权讨论有直接且重大的影响，权重分析为防御研究开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](../../ICLR2026/model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

</div>

<!-- RELATED:END -->

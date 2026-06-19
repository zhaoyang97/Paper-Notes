---
title: >-
  [论文解读] 500xCompressor: Generalized Prompt Compression for Large Language Models
description: >-
  [ACL 2025][模型压缩][提示学习] 提出 500xCompressor，将最多约 500 个自然语言 token 压缩为最少 1 个特殊 token 的 KV 值，实现 6x 到 480x 的压缩比，仅增加约 0.25% 的参数，LLM 在压缩后保留 62.26%-72.89% 的原始能力，显著超越 ICAE 基线。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "提示学习"
  - "软提示"
  - "KV缓存"
  - "高压缩比"
  - "自编码器"
---

# 500xCompressor: Generalized Prompt Compression for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2408.03094](https://arxiv.org/abs/2408.03094)  
**代码**: [https://github.com/ZongqianLi/500xCompressor](https://github.com/ZongqianLi/500xCompressor)  
**领域**: Model Compression / Prompt Compression  
**关键词**: Prompt压缩, 软提示, KV缓存, 高压缩比, 自编码器

## 一句话总结

提出 500xCompressor，将最多约 500 个自然语言 token 压缩为最少 1 个特殊 token 的 KV 值，实现 6x 到 480x 的压缩比，仅增加约 0.25% 的参数，LLM 在压缩后保留 62.26%-72.89% 的原始能力，显著超越 ICAE 基线。

## 研究背景与动机

长提示在 NLP 应用中带来多重挑战：推理速度降低、计算成本增加、用户体验下降，且上下文长度限制制约了模型的应用场景。

现有的 Prompt 压缩方法分为两类：
- **硬提示（Hard Prompt）**：如 SelectiveSentence、LLMLingua，删除低信息句子/词/token，但只是选择性保留
- **软提示（Soft Prompt）**：如 GIST、AutoCompressor、ICAE，将自然语言 token 压缩为少量特殊 token

现有方法的主要问题：

**压缩比低**：ICAE 最高仅约 15x

**信息损失不清晰**：win rate 等指标无法定量刻画压缩信息损失

**数据泄漏风险**：评估文本来自 Pile 数据集可能与 LLaMA 训练数据重叠，无法确定 LLM 输出来自压缩 token 还是模型记忆

## 方法详解

### 整体框架

500xCompressor 采用类自编码器架构，包含编码器和解码器：
- **编码器**：冻结的 LLM + 可训练 LoRA 参数（约 0.25% 额外参数）
- **解码器**：原始冻结 LLM（无任何额外参数）

核心流程：原始文本输入编码器 → 通过注意力机制将信息编码到压缩 token 的 KV 值 → KV 值传入解码器 → 解码器重生成原文或回答问题。

### 关键设计

**1. 使用 KV 值而非 Embedding**

与 ICAE 使用压缩 token 的 embedding 不同，500xCompressor 使用压缩 token 在 LLM 每一层的 KV 值。KV 值能封装更多信息，且不增加推理时间，对 GPU 显存的影响也很小。

这一设计类似于 Prefix Tuning 与 Prompt Tuning 的关系：Prompt Tuning 仅训练前缀 token 的 embedding，而 Prefix Tuning 同时训练 KV 值，能编码更丰富的信息。

**2. BOS token 触发机制**

使用 [BOS] token 触发 LLM 重生成压缩文本，而 ICAE 需要创建可训练的新 token。这种设计更简洁，且与 LLM 原有的生成机制兼容。

**3. 无数据泄漏的训练设计**

编码器和解码器中的原始 LLM 参数均冻结，解码器中不引入任何额外参数。因此没有信息保存在解码器中，确保推理时的输出完全来自压缩 token 的 KV 值。

### 损失函数 / 训练策略

**预训练阶段**：
- 输入：压缩 token 的 KV 值 + [BOS] + 原始文本
- 损失：解码器输出与原始文本的交叉熵
- 目标：学会将文本信息编码到压缩 token 中

$$\mathcal{L}_P = -\sum_{i=1}^{l} \log P(t_i | H_C, [BOS], t_{1:i-1}; \Theta_{LLM}, \Theta_{Lora})$$

**微调阶段**：
- 输入：压缩 token 的 KV 值 + 问题 + 答案
- 损失：答案部分的交叉熵
- 目标：学会从压缩 token 的 KV 值中检索信息并生成答案

$$\mathcal{L}_F = -\sum_{j=1}^{n} \log P(a_j | H_C, q_{1:m}, a_{1:j-1}; \Theta_{LLM}, \Theta_{Lora})$$

**训练数据**：
- 预训练：Arxiv Corpus（2023 年 7 月前的摘要作为训练集）
- 微调：ArxivQA 数据集（由 LLaMA-3-70b-chat 从摘要生成的抽取式 QA 对）
- 测试集：2024 年 1 月后发表的论文摘要（严格未见数据）

**训练配置**：
- 学习率：预训练 1e-4，微调 5e-5
- 批次大小：4
- 优化器：AdamW
- 压缩 token 数：1 / 4 / 16

## 实验关键数据

### 主实验

**文本重生成（Arxiv Corpus 测试集）**：
- 500xCompressor 在所有压缩比下均超越 ICAE
- Rouge-l-f 差异范围：12.18%-18.96%
- BLEU 差异范围：12.41%-26.50%
- 从 16 token 减至 4 token 质量下降较小，从 4 减至 1 下降明显

**ArxivQA 数据集问答**：
- F1 提升：2.06%-9.23%
- EM 提升：0.56%-7.20%
- 压缩比越高，500xCompressor 相对 ICAE 的优势越大

**跨任务泛化（5 个基准）**：

| 任务 | 500→16 F1 | 500→4 F1 | 500→1 F1 |
|---|---|---|---|
| ArxivQA（信息提取）| 40.77 vs 38.70 | 38.30 vs 33.31 | 29.73 vs 20.50 |
| SQuAD（信息提取）| 50.01 vs 51.94 | 49.66 vs 47.48 | 42.86 vs 27.20 |
| RelationExtraction | 68.73 vs 65.94 | 63.72 vs 67.28 | 63.09 vs 44.46 |
| HotpotQA（多跳）| 41.68 vs 42.11 | 36.86 vs 40.39 | 37.47 vs 22.44 |
| RACE（阅读理解）| 35.42 vs 23.69 | 21.49 vs 20.06 | 21.75 vs 13.71 |

500→1 压缩时，500xCompressor 在所有基准上全面超越 ICAE，最大提升 18.62%（RelationExtraction F1）。

**对比 gold standard**：
- Instruct full context（完整文本 + 指令）：5 基准平均 F1=61.36
- 500xCompressor 500→16：平均 F1=45.32，保留约 73.8% 能力
- 500xCompressor 500→1：平均 F1=38.98，保留约 63.5% 能力

### 消融实验

**压缩 token 数量的影响**：
- 500xCompressor 并非均匀利用所有 token：16→4 token 时重生成质量变化小，4→1 时明显下降
- 说明 500xCompressor 能更高效地使用少量压缩 token
- ICAE 未表现出这种两阶段下降特性

**不同压缩比下的退化速度**：
- ICAE 在高压缩比下退化更快
- 500xCompressor 的可扩展性更好

### 关键发现

1. **KV 值优于 Embedding**：在高压缩比下保存信息的优势显著
2. **并非所有压缩 token 被均匀利用**：存在信息分布不均现象
3. **重生成错误不一定影响 QA**：即使重生成有误，QA 仍可能正确（反之亦然）
4. **高压缩比下优势更明显**：500→1 时 500xCompressor 优势最大
5. **多跳推理和阅读理解受压缩影响最大**：HotpotQA 和 RACE 性能下降显著
6. **严格未见数据验证了泛化性**：2024-01 后的数据排除了模型记忆干扰

## 亮点与洞察

- 达到 480x 的极端压缩比，远超前人 <50x 的上限，实质性推进了 prompt 压缩的上限探索
- 使用 KV 值而非 embedding 的设计巧妙利用了 Transformer 的注意力机制特性
- ArxivQA 数据集的构建思路值得借鉴：利用论文时间戳严格控制数据泄漏
- 将压缩 token 解读为"新的 LLM 语言"的视角有启发性（编码信息、传输信息、适应性评估三要素）
- 只训练 LoRA 参数（~0.25%）即实现高压缩性能，参数效率极高

## 局限与展望

- 预训练和微调仅在较小的 Arxiv 语料上进行，扩大训练数据有望进一步提升
- 当前仅支持约 500 token 的输入长度，对更长文本的扩展需要验证
- 多跳推理（HotpotQA）和阅读理解（RACE）任务下性能损失较大
- 压缩 token 的信息分布不均问题需进一步研究
- 未探索在 RAG、in-context learning、角色扮演等实际应用场景中的效果
- 仅基于 LLaMA-3-8b-chat，对其他 LLM 架构的适用性未验证

## 相关工作与启发

- 与 ICAE 的核心差异在 embedding vs KV values，类似 Prompt Tuning vs Prefix Tuning 的关系
- 与 LLMLingua 等硬提示方法互补：硬提示选择性保留，软提示无差别压缩
- xRAG 和 COCOM 将软提示应用于 RAG，是自然的下游扩展方向
- Function Vectors 的研究表明特殊 token 可编码高级语义功能，与压缩 token 的"新语言"视角呼应
- 对 KV Cache 压缩和长上下文推理加速有启发意义

## 评分

- **新颖性**: 7/10 — 架构改进（KV 值替代 embedding）虽非颠覆但效果显著
- **技术深度**: 7/10 — 实验全面，消融充分，但理论分析偏少
- **实用性**: 8/10 — 高压缩比对推理加速和成本降低有直接价值，已开源
- **写作质量**: 7/10 — 结构清晰但部分内容与 ICAE 对比的行文稍显重复

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)
- [\[ACL 2025\] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)

</div>

<!-- RELATED:END -->

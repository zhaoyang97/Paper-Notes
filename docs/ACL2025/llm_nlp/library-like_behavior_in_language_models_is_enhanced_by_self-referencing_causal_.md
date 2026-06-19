---
title: >-
  [论文解读] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles
description: >-
  [ACL2025][LLM 其他][reversal curse] 提出"自引用因果循环"（ReCall）概念，揭示 LLM 预训练数据中自然存在的重复 token 序列如何形成循环引用，使自回归模型能够绕过单向因果限制、克服逆向诅咒（reversal curse），并据此设计了两步 ReCall-aware prompting 策略。
tags:
  - "ACL2025"
  - "LLM 其他"
  - "reversal curse"
  - "causal cycle"
  - "cycle tokens"
  - "autoregressive model"
  - "information retrieval"
  - "提示学习"
---

# ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles

**会议**: ACL2025  
**arXiv**: [2501.13491](https://arxiv.org/abs/2501.13491)  
**代码**: [samunaai/remember](https://github.com/samunaai/remember)  
**领域**: LLM/NLP  
**关键词**: reversal curse, causal cycle, cycle tokens, autoregressive model, information retrieval, prompting  

## 一句话总结

提出"自引用因果循环"（ReCall）概念，揭示 LLM 预训练数据中自然存在的重复 token 序列如何形成循环引用，使自回归模型能够绕过单向因果限制、克服逆向诅咒（reversal curse），并据此设计了两步 ReCall-aware prompting 策略。

## 研究背景与动机

**逆向诅咒（Reversal Curse）**是自回归语言模型的一个已知缺陷：模型在"A is B"上训练后无法正确推断"B is A"。例如，LLM 可以正确说出"Gave proof through the night..."之后的一行是"O say does that star-spangled banner yet wave"，但被问到前一行时却给出错误答案。

这一现象源于自回归模型的单向因果性——模型基于前面的 token 生成后续 token，要求知识以一致的 token 顺序被学习和再现。公式上：

$$S_r = \arg\max_{s \in \mathcal{S}} P_\mathcal{M}(s|S_l) \quad \text{（容易）}$$
$$S_l \neq \arg\max_{s \in \mathcal{S}} P_\mathcal{M}(s|S_r) \quad \text{（困难）}$$

现有解决方案主要依赖数据增强（token 排列、反转训练等人工干预）。作者提出了一个全新视角：**逆向诅咒并非总是障碍——预训练数据中自然存在的模式足以缓解这一问题**。

核心比喻：将 LLM 类比为一座图书馆，prompt 如同图书馆的交叉引用索引。预训练数据中反复出现的 token 序列（如诗歌标题、歌曲名等）就像"超链接"，连接文本的不同部分。

## 方法详解

### 1. 自引用因果循环（Self-Referencing Causal Cycle）

**核心概念——Cycle Token**：

考虑原始序列 $\mathcal{S}_{seq} = [e_1, e_2, ..., e_n]$，在位置 i 处切分为左段 $S_l = [e_1, ..., e_i]$ 和右段 $S_r = [e_{i+1}, ..., e_n]$。

从 $S_r$ 直接恢复 $S_l$ 需要遍历所有可能序列，计算上不可行。关键洞察是：如果 $e_1$（或某个 token 序列）在文本中**多次出现**，它就充当了"循环 token"——将序列末尾连接回序列开头，形成因果循环。

构造修改序列：$S_r' = [e_{i+1}, ..., e_n, e_1]$（将 $e_1$ 附加到 $S_r$ 末尾）。从 $S_r'$ 出发，模型可以继续预测 $S_l' = [e_2, e_3, ..., e_i]$，且当 i 足够大时 $S_l' \approx S_l$。

**直觉**：cycle token 就像网页中的"锚点"或书中的"交叉引用"，让模型在单向生成流中"跳转"回文本的早期部分。

### 2. 形式化框架

给定右段 $S_r$，通过 cycle token 生成候选集 $S_{l_c}$，然后选择最优：

$$S_l = \arg\max_{s \in S_{l_c}} P_\mathcal{M}(S_r|s) P_\mathcal{M}(s)$$

这将原本需要遍历所有序列的搜索问题转化为在有限候选集上的选择问题。

### 3. Few-Token 实验设计

**确定性实验系列**（6 种设置）：

| 实验 | 训练序列 | 测试路径 | 验证目标 |
|------|---------|---------|---------|
| Baseline | (e1, e2, e3, e1) | e3→e1→e2 | 基本 cycle 能力 |
| Length of Path | (e1, e2, e3, E4, e1) | e3→E4→e1→e2 | 路径长度影响 |
| Length of 'Out-of' Path | (e1, e2, E3, e4, e1) | e4→e1→e2 | 路径外噪声 |
| Cycle Composability | 两个序列共享 e3 | e3→e1→e2 | 跨样本组合 |
| Hyperlink Composability | 两个序列共享 e3 | e2→e3→e1→e4 | 超链接跳转 |

所有实验使用 2 层 8 头的小型 Transformer，证明 cycle token 机制的基本可行性。

**随机性实验**：扩展为每个 cycle token 对应多个候选后续 token（n 个），验证在歧义情况下 cycle token 的选择行为。

### 4. ReCall-aware Prompting 策略

针对实际 LLM，设计两步提示策略：

**Step 1 - 召回上下文**：
> "Tell me the lines surrounding this line 'X'."

让模型利用自引用因果循环，输出目标行周围的所有相关文本，形成候选集。

**Step 2 - 提取答案**：
利用模型的 in-context learning 能力，从 Step 1 的输出中提取正确的前一行。

## 实验

### 确定性 Few-Token 实验

**核心结果**：在所有 6 种实验设置中，模型均在训练后达到**100% 验证准确率**（除 Cycle Composability 外）。这直接证明了 cycle token 使模型能从右段恢复左段。

**序列长度消融**：将路径长度 N 从 4 增加到 64，默认 embedding 维度 36 下泛化变慢，但增加到 256 维后所有长度均快速泛化，说明 ReCall 机制在不同序列长度下保持一致。

**Cycle Composability 的例外**：当左侧上下文改变了 cycle token 的语义时（如 [e3, e1, e4] 中 e3 在 e1 前改变了 e1 的含义），模型倾向于按训练模式预测 e4 而非 e2。这实际上是**正确的行为**——self-attention 自然地根据上下文调整 token 语义。

### 随机性实验

当候选集大小为 n 时，预测下一个 token 的准确率精确地遵循 **1/n** 的规律。例如：
- n=2 → 50% 准确率
- n=3 → 33% 准确率  
- n=4 → 25% 准确率

这比随机猜测（1/V，V 为词表大小，V >> n）好得多，说明 cycle token 有效地将搜索空间缩小到了语义相关的候选集。

### 预训练语料中的 Cycle Token 分析

选取 50 篇文化经典文本（诗歌、演讲、童谣）进行分析：
- **高频出现**：如"Star-Spangled Banner"在对应 Wikipedia 文章中出现 73 次
- **分布均匀**：cycle token 序列在文本中分布广泛，形成丰富的因果路径网络
- 这些重复的标题或关键短语天然形成了"超链接"，无需人工干预

### ReCall-aware Prompting 效果

在 GPT-4o (2024-12-23) 和 LLaMA-3.3-70B 上测试：
- **100% 的成功率**：对所有 50 篇经典文本中的任意完整句子，两步 ReCall-aware prompting 都能正确检索前一行
- 对比之下，传统 prompting（直接问"前一行是什么"）在大多数情况下失败
- 即使使用 chain-of-thought、排除法等高级 prompting 策略也失败

## 亮点与洞察

1. **图书馆比喻的精妙性**：将 LLM 比作图书馆、prompt 比作索引目录、cycle token 比作交叉引用——这个类比既直观又深刻，从根本上重构了我们对 reversal curse 的理解
2. **从"bug"到"feature"的视角转换**：reversal curse 传统上被视为需要修复的缺陷，但本文展示了预训练数据中天然存在的结构就足以缓解它
3. **理论与实践的桥梁**：从公理级别的 few-token 实验到实际 LLM 的 prompting 策略，提供了完整的从理论到应用的路径
4. **1/n 随机选择规律**：在歧义情况下 cycle token 精确遵循均匀分布，说明模型内部确实形成了结构化的候选集
5. **轻量级解决方案**：ReCall-aware prompting 无需修改模型或训练数据，仅通过两步 prompt 即可解决 reversal curse

## 局限性

1. **受控实验的简化性**：few-token 实验使用极小模型和简化数据，与真实 LLM 的复杂性存在差距
2. **归因困难**：在大模型中，难以精确将信息检索归因于特定的 cycle token——大模型的训练数据通常是闭源的
3. **隐私和安全约束**：从模型中提取预训练数据本身是被刻意限制的，这增加了验证 cycle token 机制的难度
4. **实际部署的可靠性**：虽然在 50 篇经典文本上达到 100% 成功率，但对任意文本的泛化性未充分验证
5. **依赖数据中的自然重复**：如果预训练数据中缺乏足够的重复模式，cycle token 机制可能不适用

## 相关工作

- **Reversal Curse**：Berglund et al. (2023) 首次系统识别，Allen-Zhu & Li (2023) 的理论分析
- **数据增强方案**：Guo et al. (2024) 的 token 排列，Golovneva et al. (2024) 的反转训练，Springer et al. (2024) 的 token 重复
- **双向模型**：BERT (Devlin et al., 2019) 通过 masked token 目标避免了 reversal curse，但不适用于自回归生成
- **LLM 作为知识库**：Petroni et al. (2019), Heinzerling & Inui (2020), Lederman & Mahowald (2024) 的图书馆类比

## 评分

⭐⭐⭐⭐ (4/5)

立意新颖，从全新视角审视 reversal curse 问题，提出的 cycle token 概念直觉上优雅且理论上严谨。Few-token 实验设计精心且说服力强。但从小模型实验到大模型应用的推广仍需更多验证，实际 prompting 策略的适用范围也有待拓展到更广泛的文本类型。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs](sqlong_enhanced_nl2sql_for_longer_contexts_with_llms.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ACL 2025\] ProgCo: Program Helps Self-Correction of Large Language Models](progco_program_helps_self-correction_of_large_language_models.md)

</div>

<!-- RELATED:END -->

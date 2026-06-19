---
title: >-
  [论文解读] Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities
description: >-
  [ACL 2025][自监督学习][解码器增强] 提出 Magnet 方法，通过混合注意力机制（双向+因果）和三个自监督目标（掩码预测+对比学习+缺失片段生成），将纯解码器 LLM 同时增强为文本编码器和填充模型，在 token 级和句子级表示学习任务上超越 LLM2Vec 等专用方法，同时避免了双向化带来的严重文本重复问题。
tags:
  - "ACL 2025"
  - "自监督学习"
  - "解码器增强"
  - "双向注意力"
  - "表示学习"
  - "文本填充"
  - "联合训练"
  - "重复问题"
---

# Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities

**会议**: ACL 2025  
**arXiv**: [2501.08648](https://arxiv.org/abs/2501.08648)  
**代码**: 未公开  
**作者**: Savya Khosla, Aditi Tiwari, Kushal Kafle, Simon Jenni, Handong Zhao, John Collomosse, Jing Shi  
**机构**: Adobe Research, University of Illinois Urbana-Champaign  
**领域**: 自监督学习 / 语言模型统一  
**关键词**: 解码器增强, 双向注意力, 表示学习, 文本填充, 联合训练, 重复问题

## 一句话总结

提出 Magnet 方法，通过混合注意力机制（双向+因果）和三个自监督目标（掩码预测+对比学习+缺失片段生成），将纯解码器 LLM 同时增强为文本编码器和填充模型，在 token 级和句子级表示学习任务上超越 LLM2Vec 等专用方法，同时避免了双向化带来的严重文本重复问题。

## 研究背景与动机

**解码器 LLM 的局限**：
   - 因果注意力限制了对双向上下文的理解，在情感分析、NER 等需要全局上下文的任务中表现不佳
   - 文本填充任务需要同时理解上文和下文，因果解码器天然缺乏此能力

**现有增强方法的割裂**：
   - 针对表示学习的方法（LLM2Vec、Echo Embeddings）将因果注意力改为双向，但破坏了生成能力
   - 针对文本填充的方法（GLM、InCoder）增加填充能力，但不能生成好的文本表示
   - **从未有方法同时赋予 LLM 表示学习和填充两种能力**

**重复问题**：将 LLM 改为双向模型后，文本生成会出现严重的句子/短语重复（LLM2Vec 使重复率增加 36.5 倍），根本原因是训练中缺乏自回归目标

**核心动机**：在单一框架中统一文本理解和生成，利用联合训练的协同效应提升各项能力

## 方法详解

### 整体框架

Magnet 对预训练 LLM（如 Llama-2-7B）进行微调，核心创新在两方面：(1) 混合注意力掩码，(2) 三个自监督训练目标。

### 3.1 混合注意力机制

将输入 token 分为两类：

1. **Context tokens（蓝色）**：彼此之间使用**完全双向注意力**，每个 context token 可以看到所有其他 context token。这实现了编码器式的双向理解
2. **Span tokens（绿色）**：可以看到**所有 context tokens**（双向），但 span tokens 之间使用**因果注意力**。这实现了填充式生成

**关键洞察**：span tokens 之间保持因果注意力是避免重复问题的关键——纯双向化的 LLM2Vec 恰恰因为完全打开双向注意力而导致生成退化。

推理时支持三种模式：
- 纯因果：传统左到右文本生成
- 纯双向：表示学习任务
- 混合：文本填充

### 3.2 三个训练目标

#### 目标 1：Masked Next Token Prediction (MNTP)

- 随机选择 20% 输入 token 进行掩码（80% 替换为 [MASK]，10% 随机词，10% 不变，继承 BERT 策略）
- **关键设计**：使用位置 l 的输出预测位置 l+1 的掩码 token（而非 BERT 式的位置 l 预测位置 l），保持与 LLM 预训练时"预测下一个 token"的一致性
- 损失函数：交叉熵，仅在掩码位置计算
- 仅作用于 context tokens

#### 目标 2：Self-Supervised Contrastive Learning (SSCL)

- 使用 paraphrase 模型生成输入的增强视图 x⁺
- 使用最后一个 token [EOS] 的表示作为句子编码
- InfoNCE 损失 + batch 内负样本
- **设计巧妙**：选用最后一个 token 是为了与 MNTP 解耦（MNTP 中位置 l 的输出用于预测 l+1 的 token，最后一个 token 的输出不参与 MNTP）
- 添加指令前缀："Given the sentence, find its representation:"

#### 目标 3：Missing Span Generation (MSG)

- 从输入中删除一个或多个连续片段，要求模型自回归地生成填充内容
- 每个 span token yₗ 条件于所有 context tokens 和前面的 span tokens
- 交叉熵损失，仅在 span 位置计算
- **附带好处**：当所有 token 都是 span token 时，退化为标准的下一 token 预测任务，保持生成能力

#### 总损失

ℒ = λ₁ℒ_MNTP + λ₂ℒ_SSCL + λ₃ℒ_MSG

### 3.3 训练流程

对每个训练样本 x：
1. 生成三个视图：(a) 带掩码和 span 的 xᵐ，(b) 原始 x 用于 SSCL，(c) paraphrase 增强的 x⁺
2. 通过同一基础模型处理三个视图，使用不同的注意力掩码
3. 分别计算三个损失并加权求和

## 实验结果

### Token 级表示学习（CoNLL-2003）

| 模型 | Chunking | NER | POS-Tags |
|------|:---:|:---:|:---:|
| BERT-Large | 71.77 | 90.09 | 75.12 |
| DeBERTa-Large | 85.74 | 94.97 | 86.49 |
| StructBERT-Large | 89.99 | 97.31 | 90.86 |
| Llama-2-7B (原始) | 88.23 | 96.59 | 91.53 |
| LLM2Vec^{MNTP} | 91.61 | 97.16 | 92.61 |
| **Magnet** | **92.64** | **98.31** | **93.34** |

**关键发现**：Magnet 在三个任务上全面超越 LLM2Vec^{MNTP}（仅用 MNTP 训练的版本），证明联合训练的协同效应。注意 Magnet 比 LLM2Vec 多了 SSCL 和 MSG 目标，但这些"额外"目标不仅没有干扰表示学习，反而促进了它。

### 句子级表示学习（STS 基准）

| 模型 | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RoBERTa-Large + SimCSE | 72.86 | 83.99 | 75.62 | 84.77 | 81.80 | 81.98 | 71.26 | 78.90 |
| Llama-2-7B (原始) | 50.98 | 74.02 | 62.86 | 67.09 | 71.03 | 63.56 | 67.22 | 65.25 |
| LLM2Vec | 65.39 | 79.26 | 72.98 | 82.72 | 81.02 | 78.32 | 71.77 | 75.92 |
| **Magnet** | **67.98** | **84.66** | **77.67** | **84.17** | 79.44 | **82.88** | **78.77** | **79.36** |

Magnet 在 STS 平均分上超越 LLM2Vec 约 3.4 个百分点，甚至超过了 RoBERTa-Large + SimCSE 这类专门训练的编码器。

### 文本填充（Perplexity）

| 方法 | ROC Stories PPL | Wikitext-103 PPL |
|------|:---:|:---:|
| Llama-2-7B | 13.93 | 22.04 |
| **Magnet** | **9.52** | **15.46** |

Magnet 显著降低了填充困惑度。人工评估中，Magnet 生成的填充在 62% 的情况下被评为"上下文合适"（vs. Llama-2-7B 原始 53.5%、zero-shot 5.5%、five-shot 54.5%）。

### 重复问题分析（核心贡献之一）

| 方法 | Rep-Sen (Wiki) | Rep-4 (Wiki) | Rep-Sen (ROC) | Rep-4 (ROC) |
|------|:---:|:---:|:---:|:---:|
| Llama-2-7B | 0.0056 | 0.0601 | 0.0381 | 0.0163 |
| LLM2Vec | 0.2044 | 0.4747 | 0.2945 | 0.5243 |
| **Magnet** | **0.0151** | 0.2047 | **0.0737** | 0.2573 |

- LLM2Vec 使 Llama-2-7B 的句子重复率增加 **36.5 倍**（Wikitext），而 Magnet 仅增加 2.7 倍
- 随着训练迭代增加，LLM2Vec 的重复问题持续恶化，Magnet 则没有这种趋势
- **原因分析**：LLM2Vec 仅用双向注意力训练，使解码器退化为类 BERT 模型；Magnet 的 MSG 目标保持了自回归生成能力

### 知识与推理能力保持

| 模型 | HellaSwag | BBH | ARC-Easy | ARC-Challenge | MMLU (平均) |
|------|:---:|:---:|:---:|:---:|:---:|
| Llama-2-7B | 75.51 | 33.57 | 73.95 | 44.28 | 46.81 |
| Magnet | 75.08 | 32.22 | 74.33 | 44.52 | 45.98 |

Magnet 对预训练知识的影响极小，各基准测试变化在 1-2 个百分点以内。

## 亮点与洞察

1. **统一框架的优越性**：通过联合训练不同目标，各能力之间产生正向协同——Token 级表示学习因为有 MSG 目标的存在而更好，这颠覆了"多任务会互相干扰"的直觉
2. **重复问题的根因分析**：首次系统分析了双向化导致文本生成重复的问题，指出纯双向注意力使 LLM 退化为 BERT 式模型，而 BERT 本身就有生成重复的问题
3. **混合注意力的优雅设计**：context tokens 双向、span tokens 因果的分区策略，用一个注意力掩码同时服务于理解和生成
4. **最后一个 token 作为句子表示的巧妙理由**：与 MNTP 目标解耦，避免两个任务在同一位置竞争
5. **参数高效**：在现有 LLM 上微调即可，无需从头预训练

## 局限性

1. 仅在 Llama-2-7B 上验证，未测试更大模型（13B/70B）或其他架构（Mistral、GPT）
2. SSCL 依赖 paraphrase 模型生成增强视图，增强质量可能影响句子表示效果
3. Rep-4 指标上 Magnet 仍比原始模型有显著退化（Wikitext 从 0.06 到 0.20），重复问题并未完全解决
4. 填充推理需要知道缺失位置和上下文，实际应用场景有限
5. 未与 Mistral/Qwen 等更新模型或 instruction-tuned 版本进行比较

## 相关工作

- **表示学习**: LLM2Vec (BehnamGhader et al., 2024) 用 MNTP+SimCSE 将 LLM 变为编码器，但破坏生成能力；Echo Embeddings (Springer et al., 2024) 通过重复输入获取双向信息
- **文本填充**: GLM (Du et al., 2021) 用自回归空白填充，InCoder (Fried et al., 2022) 重排训练样本；FIM (Bavarian et al., 2022) 用 Fill-in-the-Middle 训练
- **理解与生成统一**: XLNet (Yang et al., 2019) 用排列目标，UniLM (Dong et al., 2019) 用多方向注意力掩码，但这些方法需从头预训练

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐⭐ 首次在单一框架中同时赋予 LLM 编码、填充和生成三种能力，混合注意力设计优雅
- **实验充分性**: ⭐⭐⭐⭐ 涵盖 token/句子级表示、填充、生成、知识保持等多维度评估，重复问题分析深入
- **实用价值**: ⭐⭐⭐⭐ 为 LLM 增加多种能力而不损失原有能力，实用性强
- **写作质量**: ⭐⭐⭐⭐ 图表设计清晰，方法动机阐述充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Residual Connections Harm Generative Representation Learning](../../CVPR2026/self_supervised/residual_connections_harm_generative_representation_learning.md)
- [\[ACL 2025\] QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](qaencoder_aligned_representation.md)
- [\[ACL 2025\] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)
- [\[CVPR 2026\] OpenVision 2: A Family of Generative Pretrained Visual Encoders for Multimodal Learning](../../CVPR2026/self_supervised/openvision_2_a_family_of_generative_pretrained_visual_encoders_for_multimodal_le.md)
- [\[CVPR 2025\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2025/self_supervised/representation_learning_for_spatiotemporal_physical_systems.md)

</div>

<!-- RELATED:END -->

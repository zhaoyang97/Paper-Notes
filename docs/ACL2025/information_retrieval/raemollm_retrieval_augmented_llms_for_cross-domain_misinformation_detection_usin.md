---
title: >-
  [论文解读] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information
description: >-
  [ACL2025][信息检索/RAG][跨域虚假信息检测] 提出 RAEmoLLM，首个基于情感信息检索的 RAG 框架，利用情感 LLM 的隐式嵌入构建检索数据库，为跨域虚假信息检测提供情感相关的 few-shot 示例，在三个基准上最高分别提升 15.64%、31.18% 和 15.73%（对比其他 few-shot 方法），无需微调。
tags:
  - "ACL2025"
  - "信息检索/RAG"
  - "跨域虚假信息检测"
  - "检索增强生成"
  - "情感信息"
  - "上下文学习"
  - "少样本学习"
---

# RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information

**会议**: ACL2025  
**arXiv**: [2406.11093](https://arxiv.org/abs/2406.11093)  
**代码**: [lzw108/RAEmoLLM](https://github.com/lzw108/RAEmoLLM)  
**领域**: 信息检索  
**关键词**: 跨域虚假信息检测, 检索增强生成, 情感信息, 上下文学习, 少样本学习

## 一句话总结

提出 RAEmoLLM，首个基于情感信息检索的 RAG 框架，利用情感 LLM 的隐式嵌入构建检索数据库，为跨域虚假信息检测提供情感相关的 few-shot 示例，在三个基准上最高分别提升 15.64%、31.18% 和 15.73%（对比其他 few-shot 方法），无需微调。

## 研究背景与动机

**虚假信息泛滥**：互联网上的虚假信息在教育、政治、健康等多个领域广泛传播，对社会生活和稳定造成严重危害，人们需要投入大量时间和精力辨别真伪。

**跨域检测的困难**：在特定领域训练的模型面对新领域样本时往往表现脆弱、预测不准确。跨域虚假信息检测成为紧迫的全球性问题。

**现有方法的代价高**：当前跨域检测方法依赖费时费力的微调和复杂的模型结构（如 MoE 多域模型），难以快速适应新领域。

**LLM 方法局限于域内**：虽然 LLM 在很多任务上表现优异，但现有 LLM 虚假信息检测方法仅关注域内任务，缺乏跨域能力。

**情感信息被忽视**：虚假信息作者往往刻意选择特定情感来吸引读者注意和共鸣。然而现有方法很少利用情感（sentiment）和情绪（emotion）特征，唯一的 ConspEmoLLM 也需要微调且无跨域能力。

**ICL + 情感检索的空白**：上下文学习（ICL）无需微调，只需任务指令和少量示例即可工作。但目前尚无工作将基于情感信息检索的 ICL 应用于跨域虚假信息检测。

## 方法详解

### 整体框架

RAEmoLLM 包含三个模块，形成完整的检索增强推理流水线：
1. **索引构建模块**（Index Construction）：利用情感 LLM 提取所有域数据的情感嵌入，构建检索数据库
2. **检索模块**（Retrieval）：基于情感相似度从源域中检索 Top-K 相关示例
3. **推理模块**（Inference）：将检索到的示例作为 few-shot 示范，指导 LLM 对目标域内容进行虚假信息判断

### 关键设计

**情感信息的双重利用**：
- **隐式情感信息**：利用 EmoLLaMA-chat-7B 的最后隐藏层（4096维向量）作为情感嵌入，捕捉文本中深层的情感语义
- **显式情感信息**：提取五种情感维度的标签（EIreg 情绪强度、EIoc 情绪分类、Vreg 情感极性强度、Voc 情感极性分类、Ec 情绪检测），可选择性地添加到 prompt 中

**情感分析验证**：论文通过 t-test 验证了真实信息和虚假信息在情感分布上存在显著差异：
- AMTCele 数据集中，虚假信息的愤怒和恐惧强度显著更高，快乐程度显著更低
- 隐式嵌入的 t-test 表明同类别内 Top-K 相似度显著高于跨类别，PCA 可视化也显示不同类别在潜空间中明显分离

**检索过程**：
1. 通过 EmoLLaMA 对目标域和源域数据分别编码为情感嵌入 $E_T$ 和 $E_S$
2. 对每个目标域嵌入 $e_t$，计算与所有源域嵌入 $e_s$ 的余弦相似度
3. 选取相似度最高的 K 个源域样本（含文本和标签）作为 few-shot 示例

**推理模板**：
- Template 1（仅隐式）：[任务指令] + [目标文本] + [检索示例] → [输出]
- Template 2（隐式+显式）：在 Template 1 基础上为目标文本和示例都添加显式情感标签

### 训练策略

- **无需微调**：整个框架基于 in-context learning，不对任何 LLM 进行微调
- **灵活的基座模型**：支持多种 LLM 作为推理引擎（ChatGPT、GPT-4o、Mistral、Llama、Gemma、Vicuna 等）
- **检索示例数量**：默认使用 4 个 few-shot 示例，实验显示增加示例不一定提升性能，过多示例可能引入噪声

## 实验关键数据

### 数据集

| 数据集 | 域数 | 类型 | 规模 |
|:---:|:---:|:---:|:---:|
| AMTCele | 7 域 | 假新闻检测 | 980 篇 |
| PHEME | 9 域 | 谣言检测 | 6425 条推文 |
| COCO | 12 域 | 阴谋论检测 | 2581 条推文 |

### 主实验结果（F1 Score）

| 模型 | AMTCele | PHEME | COCO |
|:---:|:---:|:---:|:---:|
| BERT (微调) | 0.5322 | 0.7208 | 0.6356 |
| RoBERTa (微调) | 0.4730 | 0.7204 | 0.6388 |
| MDFEND | 0.5815 | 0.5829 | 0.7793 |
| EDDFN | 0.6951 | 0.6816 | 0.5917 |
| Mistral-7b-zs | 0.6926 | 0.5936 | 0.4673 |
| Mistral-7b-random | 0.6889 | 0.6227 | 0.7287 |
| **Mistral-7b-Vreg** | **0.7404** | **0.6788** | **0.7898** |
| **Mistral-7b-Vreg-addexpl** | **0.7717** | **0.6920** | **0.7931** |
| GPT4o-zs | 0.8813 | 0.6228 | 0.7396 |
| **GPT4o-Vreg** | **0.8884** | **0.6992** | **0.8326** |

### 不同嵌入检索的消融（Mistral-7b，F1）

| 检索方式 | AMTCele | PHEME | COCO |
|:---:|:---:|:---:|:---:|
| Vreg (情感) | **0.7404** | **0.6788** | **0.7898** |
| Vreg-addexpl | **0.7717** | **0.6920** | **0.7931** |
| Semantic (RoBERTa) | 0.6904 | 0.6718 | 0.7771 |
| SentiBERT | 0.6984 | 0.6663 | 0.7687 |

### 关键发现

1. **情感检索显著优于随机 few-shot**：在所有 LLM 骨干上，基于 Vreg 情感嵌入检索的示例均优于随机采样的示例，最大提升分别为 15.64%（AMTCele/Gemma-2b）、31.18%（PHEME/Llama3.2-1b）、15.73%（COCO/Vicuna-7b）。
2. **隐式+显式情感信息互补**：添加显式 Vreg 标签（Vreg-addexpl）在大多数情况下进一步提升性能，说明隐式嵌入和显式标签提供了互补的情感信号。
3. **情感嵌入优于语义嵌入**：相比 RoBERTa 语义嵌入和 SentiBERT 情感嵌入，EmoLLaMA 的情感嵌入检索效果最好，验证了专门情感 LLM 的价值。
4. **检索数量非越多越好**：增加检索示例到 8/16/32/64 并不总是提升性能，过多示例可能引入跨类别噪声。默认 4 个示例是较好的平衡点。
5. **单纯添加显式情感无效**：random-addexpl（随机采样+显式情感）几乎不优于 random，说明关键在于情感驱动的**检索**而非简单地附加情感标签。

## 亮点与洞察

- **情感视角的独特价值**：将情感信息作为跨域虚假信息检测的桥梁，基于"虚假信息往往带有特定情感模式"的洞察，利用情感嵌入空间的分布差异来选择高质量 few-shot 示例
- **无需微调的实用方案**：完全基于 ICL 的框架对新域有即插即用的适应能力，实际部署成本远低于需要微调的方案
- **充分的统计验证**：通过 t-test 和 PCA 可视化严谨地验证了情感信息与虚假信息类别之间的统计关联，增强了方法的理论基础
- **广泛的 LLM 兼容性**：在 9 种不同 LLM（从 1B 到 GPT-4o）上验证有效，说明框架的通用性

## 局限与展望

1. **PHEME 数据集上表现相对较弱**：短文本谣言检测任务中，微调模型有时优于 RAEmoLLM，可能因为短文本的情感信号不够丰富
2. **情感 LLM 的依赖**：框架核心依赖 EmoLLaMA-chat-7B 的嵌入质量，如果情感 LLM 在新语言/领域表现不佳可能影响效果
3. **计算开销**：需要对所有数据先通过情感 LLM 编码，再逐一计算余弦相似度，大规模数据集上效率可能成为瓶颈
4. **仅使用 Vreg 维度**：五种情感维度中主要实验基于 Vreg（情感极性强度），其他维度（如具体情绪类型）的检索策略未被充分探索
5. **领域划分假设**：框架假设源域和目标域是明确划分的，对边界模糊的领域可能需要额外处理

## 相关工作与启发

### vs ConspEmoLLM (Liu et al., 2024)
ConspEmoLLM 是唯一利用情感 LLM 检测虚假信息的先前工作，但它需要微调、无跨域能力、对情感信息的利用不充分。RAEmoLLM 通过 ICL + 情感检索实现了无微调的跨域检测，且同时利用了隐式和显式情感信息，方案更轻量、适用范围更广。

### vs MDFEND (Nan et al., 2021) / CANMD (Yue et al., 2022)
这些是专门的跨域虚假信息检测方法，采用 MoE 等复杂结构。MDFEND 在 AMTCele 和 PHEME 上 F1 仅 0.58，CANMD 在 PHEME 上 0.73。RAEmoLLM (Mistral-7b-Vreg-addexpl) 在 AMTCele 和 COCO 上均大幅超越这些方法，且无需训练任何参数。

### vs 标准 RAG（语义检索）
使用 RoBERTa 语义嵌入进行检索的变体（Mistral-7b-semantic）F1 分别为 0.690/0.672/0.777，而情感嵌入检索（Vreg）达到 0.740/0.679/0.790，差距在 AMTCele 上尤为明显（+5.0%），说明情感维度比纯语义维度更适合虚假信息检测的跨域迁移。

## 评分

- **新颖性**: 7/10 — 情感信息+RAG 用于跨域虚假信息检测的组合有新意，但各组件（RAG、情感 LLM、ICL）均为已有技术
- **实验充分度**: 8/10 — 3个数据集、9种 LLM、多种消融实验（嵌入类型、检索数量、显式/隐式）全面详尽
- **写作质量**: 7/10 — 结构清晰、情感分析的统计验证严谨，但表格过多略显冗长
- **价值**: 7/10 — 提供了实用的无微调跨域虚假信息检测方案，情感驱动检索的思路有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Contradiction Detection in RAG-Based Chatbots](contradiction_detection_in_rag-based_chatbots.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ACL 2025\] On Synthetic Data Strategies for Domain-Specific Generative Retrieval](on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)
- [\[ACL 2025\] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->

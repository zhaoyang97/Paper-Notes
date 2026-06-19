---
title: >-
  [论文解读] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs
description: >-
  [ACL 2025][LLM效率][长上下文建模] LADM提出了一种基于注意力机制的长上下文训练数据选择框架，通过训练一个小型Long Attention Calculator来计算span间的注意力依赖分数（PFS → AFS → CDS），从大规模语料中高效筛选具有强长程依赖的高质量样本用于持续预训练，仅用1B tokens即可显著提升LLM的长上下文能力。
tags:
  - "ACL 2025"
  - "LLM效率"
  - "长上下文建模"
  - "数据选择"
  - "注意力机制"
  - "上下文依赖"
  - "持续预训练"
---

# LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs

**会议**: ACL 2025  
**arXiv**: [2503.02502](https://arxiv.org/abs/2503.02502)  
**代码**: [ZNLP/LADM](https://github.com/ZNLP/LADM)  
**领域**: LLM效率  
**关键词**: 长上下文建模, 数据选择, 注意力机制, 上下文依赖, 持续预训练  

## 一句话总结

LADM提出了一种基于注意力机制的长上下文训练数据选择框架，通过训练一个小型Long Attention Calculator来计算span间的注意力依赖分数（PFS → AFS → CDS），从大规模语料中高效筛选具有强长程依赖的高质量样本用于持续预训练，仅用1B tokens即可显著提升LLM的长上下文能力。

---

## 研究背景与动机

**研究背景：** 长上下文建模是LLM领域的热点，GPT-4支持128K、Gemini 1.5支持1M token。通过持续预训练长上下文数据是赋予LLM长输入处理能力的标准方法。然而，训练数据的质量度量尚未受到足够关注。

**现有方法的局限性：** (1) 简单拼接短文本会导致样本内缺乏跨段依赖，模型倾向于忽略远距离信息；(2) ProLong (Chen et al., 2024a) 将样本分段计算delta perplexity，但忽略了完整上下文中的内在结构和关系；(3) 基于相似度的文档聚合方法（Staniszewski et al., 2023）无法精确度量已有样本内的依赖强度。

**核心动机：** 注意力机制本身具有内在的检索能力——对当前token更相关的历史token分配更高权重。利用这一特性可以精确度量长上下文中的跨段依赖关系。

---

## 方法详解

### 整体框架

LADM包含四个步骤：(1) 训练Long Attention Calculator → (2) 计算Pairwise Focus Score (PFS) → (3) 聚合为Aggregated Focus Score (AFS) → (4) 合并为Contextual Dependency Score (CDS) 进行样本排序和选择。

### 关键设计

**1. Long Attention Calculator：** 基于TinyLlama-1.1B训练的小型长上下文模型，用5B随机采样token训练32K上下文能力。验证表明它能通过注意力分数区分不同依赖强度的样本。

**2. Pairwise Focus Score (PFS)：** 度量样本中第 $j$ 个span对第 $i$ 个span的注意力依赖：
$$\text{PFS}(i,j) = \text{Sum}\left(\text{Softmax}\left(\frac{Q_j K_{0:j}^T}{\sqrt{d_k}}\right)[:,i]\right)$$
即span $j$ 分配给span $i$ 的累积注意力权重。

**3. Aggregated Focus Score (AFS)：** 对每个span聚合其与所有前序span的PFS，排除开头 $m$ 个和局部 $n$ 个span（避免初始token和近邻的偏置），加入距离权重和方差加权：
$$\text{AFS}(j) = \sigma_j \sum_{i=m}^{j-n-1} \frac{j-i}{N} \cdot \text{PFS}(i,j)$$

### 样本级评分

Contextual Dependency Score (CDS) 通过加权求和所有span的AFS得到：
$$\text{CDS}(S) = \sum_{j=n_0}^{N-1} \frac{j}{N} \cdot \text{AFS}(j)$$

位置越大的span权重越高，鼓励长程依赖。

---

## 实验

### 主实验：Perplexity评估

| 模型 | 方法 | 2K | 4K | 8K | 16K | 32K |
|------|------|-----|-----|-----|------|------|
| L-7B | Random | 4.515 | 3.900 | 3.264 | 2.780 | 2.458 |
| L-7B | ProLong | 4.516 | 3.906 | 3.275 | 2.792 | 2.470 |
| L-7B | **LADM** | **4.481** | **3.878** | **3.252** | **2.773** | **2.453** |
| M-7B | Random | 4.620 | 3.936 | 3.267 | 2.775 | 2.455 |
| M-7B | ProLong | 4.293 | 3.696 | 3.095 | 2.644 | 2.346 |
| M-7B | **LADM** | **4.266** | **3.673** | **3.076** | **2.629** | **2.332** |

### LongBench实评（部分）

| 模型 | 方法 | Single-Doc QA Avg | Multi-Doc QA Avg |
|------|------|-------------------|------------------|
| L-7B | Random | 30.13 | 25.80 |
| L-7B | ProLong | 29.50 | 29.04 |
| L-7B | **LADM** | **32.24** | **31.10** |
| M-7B | Random | 24.08 | 24.63 |
| M-7B | ProLong | 23.76 | 24.43 |
| M-7B | **LADM** | **33.85** | **29.09** |

### 消融与分析

| 分析维度 | 发现 |
|----------|------|
| 拼接数据长度 | 4K拼接→32K的平均检索准确率仅0.57，原生32K数据为0.88 |
| 注意力分数验证 | Long Attention Calculator能区分不同依赖强度的样本 |
| 训练效率 | LADM用0.5B tokens即超越Random的1B tokens效果 |
| Needle-in-Haystack | L-7B/13B和M-7B几乎达到100%检索率 |

### 关键发现

- **依赖性是核心：** 拼接短文本的训练数据因缺乏跨段依赖，显著降低长上下文建模能力
- **注意力=依赖探针：** 注意力分布可有效作为长上下文依赖的度量信号
- **数据效率提升：** LADM用半数训练token即超越随机采样的完整训练效果
- **跨模型鲁棒性：** 在OpenLlama-3B/Llama2-7B/13B/Mistral-7B四个模型上一致有效

---

## 亮点

- 提出了清晰且可操作的长上下文数据质量度量框架（PFS → AFS → CDS）
- 创新性地利用注意力机制的内在检索能力来度量上下文依赖
- 使用低成本的TinyLlama作为数据筛选器，实现高效的数据选择流水线
- 仅1B tokens持续预训练即实现显著的长上下文能力提升

## 局限性

- 实验仅限32K token长度，更长上下文（64K/128K）的适用性未验证
- Long Attention Calculator本身需要5B tokens训练，存在一定的前置成本
- 数据选择保持原始领域分布，未探索领域分布调整的潜在收益
- CDS评分中的多个超参数（$m$, $n$, $d$, $n_0$）缺乏系统的敏感性分析

## 相关工作

- **长上下文建模：** RoPE扩展 (PI, YaRN, NTK)、LongLoRA、S²-Attn 等训练方法
- **长上下文数据：** ProLong (Chen et al., 2024a) 基于delta perplexity的数据筛选
- **文档聚合：** Staniszewski et al. (2023) 基于相似度的文档组合
- **注意力检索：** Wu et al. (2024) 揭示注意力机制中的检索操作

---

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总评 | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[NeurIPS 2025\] LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](../../NeurIPS2025/llm_efficiency/loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
description: >-
  [代码智能] 提出CL4D对比学习框架，通过继续预训练将decoder-only代码生成模型适配到代码理解任务（代码搜索、克隆检测），在不重新训练encoder模型的前提下实现了与同等规模encoder-only模型相当甚至更优的性能。 问题背景 - 大规模decoder-only代码生成模型（如StarCoder、Code…
tags:
  - "代码智能"
---

# Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning

## 元数据
- **标题**: Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
- **作者**: Jiayi Lin, Yanlin Wang, Yibiao Yang, Lei Zhang, Yutao Xie
- **会议**: AAAI 2026
- **arXiv**: [2406.12326](https://arxiv.org/abs/2406.12326)
- **代码**: [GitHub](https://github.com/JiayiLin1024/CL4D)
- **领域**: 代码理解, 对比学习, 自监督学习

## 一句话总结

提出CL4D对比学习框架，通过继续预训练将decoder-only代码生成模型适配到代码理解任务（代码搜索、克隆检测），在不重新训练encoder模型的前提下实现了与同等规模encoder-only模型相当甚至更优的性能。

## 研究背景与动机

### 问题背景
- 大规模decoder-only代码生成模型（如StarCoder、CodeLlama、DeepSeek-Coder）在代码生成方面取得了巨大成功，但在代码理解任务（代码搜索、克隆检测）上表现不佳
- 这主要源于其自回归训练目标（next-token prediction）侧重于生成而非语义理解
- encoder-only模型（如CodeBERT、UniXcoder、CodeSage）在理解任务上更强，但规模远小于最新decoder-only模型

### 核心矛盾
- Decoder-only模型拥有更大规模、更丰富的训练数据，但缺乏双向注意力机制，限制了细粒度代码理解能力
- 从头预训练同等规模的encoder-only模型计算成本极高（当前最大的CodeSage也仅1.3B，CoLSBERT仅1.5B）
- 核心问题：**能否在不重新训练的情况下，增强现有decoder-only模型的代码理解能力？**

### 研究动机
- 利用已有decoder-only模型中丰富的代码知识，通过对比学习进行知识迁移
- 探索统一代码生成和代码理解的可能性，用单一decoder-only架构同时服务两类任务

## 方法详解

### 整体框架：CL4D (Contrastive Learning for Decoder-only)
CL4D是一个对比学习框架，对预训练的decoder-only代码生成模型进行继续预训练，提升其表征能力。

### 1. 数据构建
- 从The Stack数据集中提取6种编程语言（Python、Java、Go、PHP、JavaScript、Ruby）
- 使用Tree-Sitter提取双模态数据，构建(query, code)对：query为函数文档字符串的第一行
- 应用CodeSearchNet的过滤规则提升数据质量
- 构建百万级训练样本用于继续预训练

### 2. 模型架构
探索了两种从decoder-only模型中提取代码表征的方法：
- **Last Token**：使用最后一层最后一个token的embedding作为代码表征（因单向注意力机制，仅最后一个token聚合了所有前文信息）
- **Average**：对最后一层所有token的embedding取平均

采用双编码器架构（dual-encoder），两个共享权重的Transformer decoder分别编码query和code。

### 3. 对比学习目标
- **批内负样本（In-batch Negatives）**：同一batch中其他样本的code作为负样本
- **硬负样本（Hard Negatives）**：利用UniXcoder从整个代码库中，为每个query选取表征空间中距离近但语义不同的代码片段
- 损失函数采用InfoNCE形式，温度参数τ=0.05，使用余弦相似度计算相关性

### 4. 表征提取策略
经过消融实验发现：
- **右填充(right padding) + 平均所有token embedding** 是最优的表征提取方式
- 左填充时last token效果更好，右填充时average效果更好

### 训练细节
- 优化器：AdamW，学习率2e-5
- 训练2个epoch，batch size 64
- 8块A100 (80G)，最长训练时间约3天（phi-1）

## 实验

### 实验设置
- **代码搜索**：CodeSearchNet（CSN，6种语言）和CoSQA数据集，指标MRR
- **克隆检测**：POJ-104数据集，指标MAP
- 对比模型：encoder-only（CodeBERT/GraphCodeBERT/UniXcoder/CodeSage）和decoder-only（CodeGPT/CodeGen/SantaCoder/phi-1/DeepSeek-Coder）

### 表1：微调后整体性能对比

| 方法 | 规模 | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|------|------|-----------|-------------|----------------|
| CodeBERT (Enc) | 125M | 70.18 | 65.7 | 83.79 |
| GraphCodeBERT (Enc) | 125M | 72.08 | 68.4 | 85.50 |
| UniXcoder (Enc) | 125M | 74.40 | 70.1 | 89.56 |
| CodeSage (Enc) | 1.3B | 75.80 | 68.0 | 87.70 |
| CodeGPT + CL4D (Dec) | 125M | 70.20 | 69.0 | 87.96 |
| CodeGen + CL4D (Dec) | 350M | 73.30 | 71.5 | 89.68 |
| SantaCoder + CL4D (Dec) | 1.1B | 74.98 | 72.2 | 83.98 |
| phi-1 + CL4D (Dec) | 1.3B | 75.18 | 72.8 | 92.72 |
| **DeepSeek-Coder + CL4D (Dec)** | **1.3B** | **77.57** | **71.9** | **89.71** |

**关键发现**：CL4D使decoder-only模型在大多数任务上超越同等规模encoder-only模型约2%；模型规模越大，理解性能越好。

### 表2：Zero-shot性能对比（无微调）

| 方法 | 规模 | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|------|------|-----------|-------------|----------------|
| CodeBERT (Enc) | 125M | 0.10 | 0.24 | 20.38 |
| UniXcoder (Enc) | 125M | 46.40 | 42.11 | 42.08 |
| CodeSage (Enc) | 1.3B | 71.24 | 47.53 | 73.07 |
| CodeGPT (Dec) | 125M | 0.12 | 0.04 | 9.41 |
| DeepSeek-Coder (Dec) | 1.3B | 0.12 | 0.63 | 16.51 |
| CodeGPT + CL4D (Dec) | 125M | 67.56 (↑67.44) | 53.49 (↑53.45) | 25.93 (↑16.52) |
| CodeGen + CL4D (Dec) | 350M | 71.97 (↑70.55) | 51.18 (↑50.73) | 45.84 (↑32.64) |
| SantaCoder + CL4D (Dec) | 1.1B | 74.18 (↑74.11) | 52.82 (↑52.71) | 71.14 (↑55.57) |
| **DeepSeek-Coder + CL4D (Dec)** | **1.3B** | **76.02 (↑75.90)** | **48.34 (↑47.71)** | **71.18 (↑54.67)** |

**关键发现**：CL4D在zero-shot场景下将decoder-only模型的性能提升40%-76%，使其在不微调的情况下就能超越encoder-only模型。

### 消融实验
- 去除硬负样本后性能约下降1.5%
- 去除批内负样本后性能急剧下降（CSN从72.00降至1.42），证实对比学习是方法的核心

## 亮点

1. **实用性强**：无需从头训练大规模encoder模型，直接复用已有decoder-only模型的代码知识，成本大幅降低
2. **统一架构潜力**：证明了decoder-only架构可以同时胜任代码生成和代码理解任务，为统一代码模型架构提供了方向
3. **Zero-shot提升显著**：CL4D在zero-shot场景下提升幅度可达75.90%，甚至无需微调即可匹配encoder-only SOTA
4. **Scaling效应明确**：实验清晰展示了更大decoder-only模型带来更好的代码理解性能
5. **系统性探索**：对表征提取策略（padding方向 × 聚合方式）进行了完整的消融分析

## 局限性

1. **评估任务有限**：仅评估了代码搜索和克隆检测两个任务，未涉及代码摘要、bug检测、类型推断等其他理解任务
2. **模型规模受限**：实验中最大模型仅1.3B，未验证在更大规模（如7B、13B）decoder-only模型上的效果
3. **硬负样本依赖外部模型**：构建硬负样本需要UniXcoder作为辅助ranker，引入了额外依赖
4. **生成能力评估缺失**：未分析CL4D继续预训练后对原模型代码生成能力的影响（是否存在灾难性遗忘）
5. **语言覆盖偏向主流**：仅涵盖6种主流编程语言，未验证在低资源语言上的泛化能力

## 相关工作

- **代码表征学习**：CodeBERT、GraphCodeBERT（利用数据流）、TreeBERT（利用AST）、UniXcoder、CodeSage、CoLSBERT等encoder-only模型通过MLM等目标学习代码表征
- **代码对比学习**：CoSQA（query改写构建正样本）、SynCoBERT/Code-MVP（跨模态正样本对）、UniXcoder（dropout构建正样本）、CodeRetriever/R2/CodeSage（硬负样本构建）
- **Decoder-only代码模型**：Codex、CodeGen、StarCoder、CodeLlama、DeepSeek-Coder等，规模持续增长但主要面向生成任务

## 评分

- **新颖性**: ⭐⭐⭐ — 思路直觉合理但技术新颖性有限，本质上是将SimCSE/对比学习方法应用到decoder-only代码模型
- **实用性**: ⭐⭐⭐⭐ — 方法简洁高效，训练成本低，可直接复用现有模型，工程落地门槛低
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多数据集、多设置（微调/zero-shot）的对比实验，消融和可视化分析完整
- **写作清晰度**: ⭐⭐⭐⭐ — 结构清楚，研究问题明确，实验组织有条理
- **综合评分**: ⭐⭐⭐⭐ (7.5/10) — 实用价值高，实验充分，但技术创新有限；核心贡献在于系统性验证了对比学习能有效弥合decoder-only模型在代码理解上的不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](../../ACL2025/code_intelligence/benchmarking_long-context_language_models_on_long_code_understanding.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](../../ACL2025/code_intelligence/galla_graph_aligned_large_language_models.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](../../ACL2026/code_intelligence/can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)

</div>

<!-- RELATED:END -->

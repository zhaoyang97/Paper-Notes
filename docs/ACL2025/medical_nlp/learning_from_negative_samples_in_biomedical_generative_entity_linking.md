---
title: >-
  [论文解读] ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking
description: >-
  [ACL 2025][医疗NLP][biomedical entity linking] 提出 ANGEL 框架，首次在生成式生物医学实体链接（BioEL）中引入负样本训练，通过两阶段策略（正样本训练 + 负样本感知的偏好优化）显著提升模型区分表面形式相似但语义不同的实体的能力，在五个基准数据集上平均 top-1 准确率提升 1.7%。
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "biomedical entity linking"
  - "generative model"
  - "negative sampling"
  - "preference optimization"
  - "DPO"
---

# ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking

**会议**: ACL 2025  
**arXiv**: [2408.16493](https://arxiv.org/abs/2408.16493)  
**代码**: 无  
**领域**: 医学NLP  
**关键词**: biomedical entity linking, generative model, negative sampling, preference optimization, DPO

## 一句话总结

提出 ANGEL 框架，首次在生成式生物医学实体链接（BioEL）中引入负样本训练，通过两阶段策略（正样本训练 + 负样本感知的偏好优化）显著提升模型区分表面形式相似但语义不同的实体的能力，在五个基准数据集上平均 top-1 准确率提升 1.7%。

## 研究背景与动机

### 问题定义

生物医学实体链接（BioEL）旨在将文本中的实体提及（mention）映射到标准化知识库（如 UMLS、MeSH）中的概念。该任务面临两大核心挑战：

**同义词多样性**：同一概念有多种表达方式，如 "ADHD" 的同义词包括 hyperkinetic disorder 和 attention deficit hyperactivity disorder

**表面形式歧义性**：不同概念可能有相似的名称，如 "ADA" 可指 adenosine deaminase 或 American Diabetes Association

### 现有方法局限

当前方法分为两大类：

- **相似度方法**（BioSYN、SapBERT 等）：将 mention 和实体编码到同一向量空间计算相似度。缺点是需要大量内存索引所有候选实体的嵌入向量，且 bi-encoder 的单向量表示可能限制表征质量
- **生成式方法**（GENRE、GenBioEL 等）：基于 encoder-decoder 结构直接生成最可能的实体名称，内存效率更高。但**仅使用正样本训练**，不显式学习负样本，导致模型可能过拟合表面特征，难以区分形式相似但语义不同的实体

### 核心动机

相似度方法已通过 synonym marginalization 和对比学习利用负样本，但这些策略不能直接迁移到生成式模型。ANGEL 旨在填补这一空白——让生成式模型也能从负样本中学习。

## 方法详解

### 整体框架

ANGEL 是一个两阶段训练框架，可同时应用于预训练和微调：

**阶段一：正样本训练（Positive-only Training）**

- 训练生成模型根据输入 mention 生成知识库中具有相同标识符的同义词
- 使用 TF-IDF（trigram）相似度选择与输入 mention 最相似的 top-k 同义词作为训练目标
- 输入格式：`[BOS] c- [ST] m [ET] c+ [EOS]`，解码器前缀提示为 `m is`

**阶段二：负样本感知训练（Negative-aware Training）**

1. **收集正负样本对**：对训练集每个 mention 获取模型 top-k 预测，构建三元组 (x, e_w, e_l)
    - e_w：正确（preferred）实体
    - e_l：错误（dispreferred）实体
2. **筛选策略**：仅保留模型将错误实体排在正确实体之前的样本对；若 top-1 预测已正确，则配对最高排名的错误实体
3. **偏好优化**：采用 DPO（Direct Preference Optimization）损失函数更新模型

DPO 损失函数中，评分函数定义为当前模型与参考模型的概率对数比，参考模型为第一阶段训练的模型。温度参数 beta 控制偏好强度。

### 预训练中的应用

- 利用 UMLS 知识库（3.09M 实体，199K 含定义）自动生成训练数据
- 使用子句模板构造上下文，如 "[ST] s [ET] is defined as d_y" 或 "[ST] s1 [ET] has synonyms such as s2"
- 正样本训练：为每个实体基于 TF-IDF 选择最相似同义词作为目标
- 负样本感知训练：从 TF-IDF 相似但标识符不同的实体中选择负样本（而非模型预测，以提高效率）
- 每 500 步保存 checkpoint，共训练 5 个 epoch，8 块 A100 GPU 训练 12 小时

## 实验关键数据

### 主要结果

在五个 BioEL 基准数据集上的 Top-1 准确率（%）：

| 模型 | NCBI | BC5CDR | COMETA | AAP | MM-ST21pv | 平均 |
|------|------|--------|--------|-----|-----------|------|
| SapBERT | 92.3 | 88.6 | 75.1 | 89.0 | 50.3 | 79.1 |
| Prompt-BioEL | 91.9 | 94.3 | 82.7 | 89.7 | 72.6 | 86.2 |
| GenBioEL（复现） | 91.0 | 93.1 | 80.9 | 89.3 | 70.7 | 85.0 |
| + ANGEL_FT | 92.5 (+1.5) | 94.4 (+1.3) | 82.4 (+1.5) | 89.9 (+0.6) | 71.9 (+1.2) | 86.2 (+1.2) |
| + ANGEL_PT+FT | **92.8** (+1.8) | **94.5** (+1.4) | **82.8** (+1.9) | **90.2** (+0.9) | **73.3** (+2.6) | **86.7** (+1.7) |
| BioBART + ANGEL_FT | 91.9 (+2.5) | 94.7 (+1.2) | 82.2 (+0.9) | 89.9 (+0.6) | 73.4 (+2.1) | 86.4 (+1.4) |

### 预训练效果分析

不同预训练策略下的准确率对比：

| 模型 | 微调 | BC5CDR | AAP |
|------|------|--------|-----|
| BART | 否 | 0.8 | 15.6 |
| GenBioEL | 否 | 33.1 | 50.6 |
| + ANGEL | 否 | **49.7** | **61.5** |
| BART | 是 | 93.0 | 88.7 |
| GenBioEL | 是 | 93.1 | 89.3 |
| + ANGEL | 是 | **94.5** | **90.2** |

关键发现：ANGEL 预训练在未微调时提升极为显著（BC5CDR +16.6%，AAP +10.9%），微调后仍保持优势。

### 消融实验：负样本对构建策略

| 变体 | NCBI | BC5CDR | COMETA | AAP | MM-ST21pv | 平均 |
|------|------|--------|--------|-----|-----------|------|
| ANGEL（完整） | 92.8 | 94.5 | 82.8 | 90.2 | 73.3 | 86.7 |
| 模型预测负样本->TF-IDF负样本 | 91.8 | 94.4 | 81.6 | 90.0 | 71.5 | 85.9 |
| 仅保留排序错误对->所有可能对 | 92.9 | 94.0 | 81.9 | 90.0 | 72.0 | 86.2 |
| Top-5->Top-10 | 92.5 | 94.0 | 82.1 | 89.6 | 72.6 | 86.2 |
| 无负样本训练（GenBioEL） | 91.0 | 93.1 | 80.9 | 89.3 | 70.7 | 85.0 |

核心结论：从模型自身预测中选择负样本比 TF-IDF 选择更有效（平均差 0.8%）。

## 亮点

1. **首创性**：ANGEL 是首个在生成式实体链接中引入负样本训练的框架，将 DPO 偏好优化引入 BioEL
2. **模型无关性**：框架适用于多种骨干模型（BART/BioBART/GenBioEL），均获一致提升（0.9%~1.7%）
3. **双阶段通用性**：在预训练和微调两个阶段均有效，且效果可叠加
4. **深入分析**：TF-IDF 相似度 bin 分析表明，负样本训练在处理高形态相似度的困难负样本时优势尤其明显
5. **超越重排序方法**：无需额外重排序模块即超越 Prompt-BioEL（平均 +0.5%）

## 局限性

1. **模型架构限制**：仅在 encoder-decoder 模型上验证，未测试 decoder-only 模型（如 BioGPT）或大型语言模型
2. **领域限制**：仅在生物医学领域评估，未验证在通用领域实体链接上的泛化能力
3. **低相似度场景困难**：当输入 mention 与金标准实体表面形式差异很大时（TF-IDF 相似度 0-0.2 区间），准确率仅 34.2%
4. **Top-5 提升有限**：虽然 top-1 准确率提升显著，但部分数据集 top-5 改善较小
5. **训练复杂度**：需先完成正样本训练，再收集预测构建负样本对，流程比标准方法更复杂

## 相关工作

- **相似度方法**：BioSYN（同义词边际化）、SapBERT（对比学习）、ResCNN、KRISSBERT（聚类）
- **生成式方法**：GENRE（首个生成式 EL）、GenBioEL（UMLS 预训练的 BART）、BioBART（生物医学持续预训练）
- **混合方法**：Prompt-BioEL（检索 + 重排序）
- **偏好优化**：DPO（Rafailov et al., 2024）、LambdaRank（Burges, 2010）

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | 4/5 | 首次将负样本训练和 DPO 引入生成式 BioEL，思路自然且有效 |
| 实验充分性 | 5/5 | 五个数据集、三种骨干模型、详细消融实验和深入分析 |
| 写作质量 | 4/5 | 结构清晰，动机阐述充分，案例分析直观 |
| 实用性 | 4/5 | 框架通用且易于集成，代码公开 |
| 综合评分 | 4/5 | 扎实的工作，将 RLHF/DPO 的思想成功迁移到 BioEL 任务 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_nlp/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[ACL 2025\] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_nlp/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] SECRET: Semi-supervised Clinical Trial Document Similarity Search
description: >-
  [ACL2025][医疗NLP][clinical trial] 提出 SECRET，一种半监督临床试验协议相似性搜索方法，通过将临床试验文档转换为 Q/A 对表示，并结合局部（Q/A 级）和全局（试验级）对比学习来生成嵌入，在完整试验搜索的 recall@1 上相对最佳基线提升 78%。 临床试验是评估新疗法安全性和有效性…
tags:
  - "ACL2025"
  - "医疗NLP"
  - "clinical trial"
  - "document similarity"
  - "对比学习"
  - "半监督学习"
  - "information retrieval"
---

# SECRET: Semi-supervised Clinical Trial Document Similarity Search

**会议**: ACL2025  
**arXiv**: [2505.10780](https://arxiv.org/abs/2505.10780)  
**作者**: Trisha Das, Afrah Shafquat, Beigi Mandis, Jacob Aptekar, Jimeng Sun
**机构**: University of Illinois Urbana-Champaign, Medidata Solutions
**代码**: 未开源  
**领域**: 医学NLP  
**关键词**: clinical trial, document similarity, contrastive learning, semi-supervised, information retrieval

## 一句话总结

提出 SECRET，一种半监督临床试验协议相似性搜索方法，通过将临床试验文档转换为 Q/A 对表示，并结合局部（Q/A 级）和全局（试验级）对比学习来生成嵌入，在完整试验搜索的 recall@1 上相对最佳基线提升 78%。

## 研究背景与动机

临床试验是评估新疗法安全性和有效性的关键环节，但设计过程复杂且容易出错。检索相似的历史试验可为试验设计提供参考（如目标人群、入排标准、剂量方案、不良事件预判等），但现有方法面临四个核心挑战：

**标注数据匮乏**：公开可用的试验相似性标注数据极少，监督方法（如 GTSLNet）依赖私有数据集

**长文档问题**：临床试验协议往往超过 1000 词，现有方法（如 Trial2Vec）对长段落仍需截断，导致关键信息丢失

**局部语义理解不足**：两段包含相同医学实体的文本可能语义完全不同（如"检测胰岛素水平以诊断糖尿病" vs "开处方胰岛素以管理糖尿病"），基于实体匹配的方法无法区分

**对比监督效率低**：SimCSE 用同一文档做正样本过于简单，Trial2Vec 通过删除段落生成正样本可能丢失关键信息

SECRET 提出半监督框架，同时利用少量标注数据和大量无标注数据，以 Q/A 对为表示单元解决上述问题。

## 方法详解

SECRET 包含三个核心组件：

### 1. Q/A 对生成

将每个临床试验协议转换为一组 Q/A 对：
- **长段落**（如入排标准）：使用 Llama-3.1-8B-Instruct 生成 Q/A 对，提取关键信息并显著压缩文档长度
- **短段落**（如标题、疾病、干预措施等）：使用人工预定义问题
- 核心假设：两个相似的试验将拥有相似的 Q/A 对集合

### 2. 局部对比学习（Q/A 级）

在 Q/A 对粒度上进行对比训练，以 BioBERT 为骨干编码器：
- **正样本选择**：对于锚点 Q/A 对，从同一段落的 Q/A 池中选择余弦相似度最高的对作为正样本
- **负样本**：batch 内其余所有 Q/A 对
- **损失函数**：InfoNCE loss，温度参数 tau = 0.1
- 这种设计确保即使含相同医学实体但语义不同的句子也能获得不同的嵌入表示

### 3. 全局对比学习（试验级）

在试验整体层面进行对比训练，结合标注与无标注数据：
- **无标注数据的正样本**：从含多个 Q/A 对的段落中随机丢弃一个 Q/A 对生成正样本
- **标注数据的正样本**：直接使用标注的相似试验
- **硬负样本**：同疾病类别但不相似的试验
- **损失函数**：结合配对损失和 batch 内损失

最终通过余弦相似度对试验嵌入进行排序检索。

## 实验关键数据

### Table 2: 完整试验相似性搜索（核心结果）

| 方法 | P@1 | R@1 | P@5 | R@5 | nDCG@5 | MAP |
|------|-----|-----|-----|-----|--------|-----|
| TF-IDF | 0.363 | 0.244 | 0.217 | 0.687 | 0.522 | 0.501 |
| Trial2Vec | 0.422 | 0.263 | 0.227 | 0.689 | 0.553 | 0.539 |
| **SECRET** | **0.647** | **0.467** | **0.297** | **0.924** | **0.796** | **0.754** |

SECRET 在所有指标上大幅领先，recall@1 相对 Trial2Vec 提升 78%，precision@1 提升 53%，MAP 提升 40%。

### Table 3: 部分试验搜索（仅用标题查询）

| 方法 | P@1 | R@1 | R@5 | nDCG@5 | MAP |
|------|-----|-----|-----|--------|-----|
| Trial2Vec | 0.456 | 0.322 | 0.717 | 0.592 | 0.579 |
| **SECRET** | **0.548** | **0.390** | **0.902** | **0.745** | **0.696** |

部分查询场景下 SECRET 仍然显著优于所有基线，recall@2 相对最佳基线提升 29%。

### Table 4: 零样本患者-试验匹配（TREC2021）

| 方法 | P@1 | R@1 | nDCG@5 | MAP |
|------|-----|-----|--------|-----|
| Trial2Vec | 0.608 | 0.129 | 0.618 | 0.695 |
| **SECRET** | **0.710** | **0.158** | **0.666** | **0.744** |

在未经患者-试验匹配训练的零样本设置下，SECRET 仍优于所有基线，precision@1 提升 17%，recall 提升 22%。

### 消融实验

- 仅局部对比学习效果最差；仅全局（Q/A 表示）优于仅全局（全文表示）；两者结合效果最佳
- Q/A 对数量实验：选择 top-10 个 Q/A 对效果最佳，过多会引入噪声，过少则丢失信息
- 训练数据量仅为 Trial2Vec 的 1/4（约 10K 标注 + 60K 无标注 vs Trial2Vec 的全量数据）

## 亮点

- **Q/A 对表示**是解决长文档问题的优雅方案，将冗长协议压缩为结构化、可比较的信息单元
- **双层对比学习**设计精巧：局部捕获细粒度语义差异，全局建模试验间整体相似性
- **半监督框架**有效平衡了标注成本与性能需求，用不到 1/4 的训练数据超越全量训练的基线
- 零样本迁移到患者-试验匹配任务仍优于所有基线，泛化能力出色
- 案例分析表明 SECRET 能更好捕获年龄、干预措施等关键属性的精确匹配

## 局限与展望

- 仅使用了标题、疾病、干预、关键词、结局和入排标准，**未纳入描述和研究设计等重要段落**（受 LLM 资源限制）
- 未包含知情同意书、不良事件报告等其他临床试验相关文档
- Q/A 生成依赖 LLM（Llama-3.1-8B），生成质量可能存在不一致性
- 评估数据集规模有限（测试集 1420 对），且仅为英文数据
- 未探索不同段落的重要性权重，所有段落等权对待

## 与相关工作的对比

| 维度 | Trial2Vec | GTSLNet | SECRET |
|------|-----------|---------|--------|
| 学习范式 | 自监督 | 监督 | 半监督 |
| 文档表示 | 分段编码+合并 | 全文 | Q/A 对集合 |
| 对比粒度 | 实体级 | - | Q/A级 + 试验级 |
| 训练数据需求 | 大量无标注 | 大量标注（私有） | 少量标注 + 无标注 |
| 长文档处理 | 截断 | 截断 | Q/A 压缩 |
| 开源数据 | 是 | 否 | 是 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — Q/A 对表示 + 双层对比学习的组合具有较好的原创性
- 实验充分度: ⭐⭐⭐⭐ — 三个任务、10 个基线、消融实验和案例分析较完整
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，四个挑战和对应方案逻辑连贯
- 价值: ⭐⭐⭐⭐ — 临床试验检索是重要的实际需求，方法具备直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](../../ACL2026/medical_nlp/eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)
- [\[ACL 2025\] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/medical_nlp/document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering
description: >-
  [ICCV 2025][多模态VLM][VQA] 提出 ReasonVQA 数据集，通过低成本可扩展框架将结构化百科知识（Wikidata）与图像自动融合，生成 1/2/3 跳的多跳推理问题，包含 598K 图像和 4.2M 问题，显著挑战了现有 VQA 模型。 视觉问答（VQA）领域近年取得显著进展…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VQA"
  - "多跳推理"
  - "知识图谱"
  - "Wikidata"
  - "基准数据集"
---

# ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering

**会议**: ICCV 2025  
**arXiv**: [2507.16403](https://arxiv.org/abs/2507.16403)  
**代码**: [ReasonVQA](https://duong-tr.github.io/ReasonVQA)  
**领域**: 多模态VLM  
**关键词**: VQA, 多跳推理, 知识图谱, Wikidata, 基准数据集  

## 一句话总结

提出 ReasonVQA 数据集，通过低成本可扩展框架将结构化百科知识（Wikidata）与图像自动融合，生成 1/2/3 跳的多跳推理问题，包含 598K 图像和 4.2M 问题，显著挑战了现有 VQA 模型。

## 研究背景与动机

视觉问答（VQA）领域近年取得显著进展，但现有数据集存在明显短板：

**缺乏外部知识整合**：标准 VQA 数据集（VQAv2、GQA）主要关注图像中显式可见的内容，例如物体识别和属性判断，而真实世界的问题往往需要超越图像本身的知识

**多跳推理缺失**：像"这座教堂所在国家的首都是什么？"这类问题需要多步推理——先识别教堂，再查询其所在国家，最后查询该国首都

**构建成本高昂**：现有需要外部知识的数据集（如 OK-VQA、KVQA）依赖大量人工标注，扩展性差

**规模受限**：Encyclopedic-VQA 约 1M 问题，INFOSEEK 约 1.35M，仍然有限

作者的核心动机是：**能否以低成本、自动化的方式构建一个大规模、高质量的多跳推理 VQA 基准？** 通过利用已有的 CV 数据集标注和结构化知识库 Wikidata，实现问题的自动生成。

## 方法详解

### 整体框架

ReasonVQA 的构建分为三个步骤：
1. **外部知识整合**（External Knowledge Integration）
2. **问题生成**（Question Generation）
3. **数据集构建**（Dataset Construction）

### 外部知识整合

选择 **Wikidata** 作为外部知识源，利用 **SPARQL** 进行语义查询。图像来源有两个：

- **Visual Genome (VG)**：108K+ 图像，含丰富的场景图（scene graph）标注。VG 物体标注通过 WordNet synset 规范化，可利用 NLTK 和 SPARQL 查询连接到 Wikidata 实体
- **Google Landmarks v2 (GLDv2)**：500万+图像，200K 地标，标注了 Wikimedia URL，可直接提取 Wikidata 知识

### 模板化问题生成

**一跳问题**：为 Wikidata 的每个属性（property）设计填空模板。例如属性"architect"对应模板"Who designed __?"，占位符由物体类名填入（如"skyscraper"），生成"Who designed this skyscraper?"

**多跳问题**：通过嵌套子句模板实现。例如属性"architect"的子句模板 "the architect of __"，可递归嵌套生成 2-hop 和 3-hop 问题。还利用 VG 的场景图标注构建子句（如"__ parked next to the sidewalk"），将视觉语义信息融入问题。

**领域标签**：预定义 20 个知识领域，根据属性自动分类。同一问题可属于多个领域。

### 错误选项生成

为多选题生成有挑战性的错误选项，按答案类型分四类：
- **fixed**：封闭集合（性别、大洲），随机选取
- **date**：在正确答案 $\pm 10$ 年范围内随机选取 $N$ 个日期
- **number**：在 $[\frac{i}{2}, \max(1.5i, \frac{i}{2}+2N)]$ 范围内随机选取
- **literal**：从同一属性的其他实体中检索 $N$ 个值

### 答案分布平衡

借鉴 GQA 的方法，按属性分组排序答案频率，通过迭代删除高频答案的方式平滑分布，使头部和尾部大小可比，减少模型通过猜测最常见答案取巧的可能。

### 评估指标

采用三种字符串匹配方法：
- **Exact Match**：完全匹配
- **Substring**：子串包含
- **Semantic Similarity**：使用 all-MiniLM-L6-v2 模型计算语义相似度

## 实验

### 数据集规模对比

| 数据集 | 图像数 | 问题数 | 知识库 |
|------|------|------|------|
| OK-VQA | 14K | 14K | ✓ |
| KVQA | 24K | 183K | ✓ |
| CRIC | 96K | 494K | ✓ |
| InfoSeek | 8.9K | 1.35M | ✓ |
| Encyclopedic VQA | 514K | 1M | ✓ |
| **ReasonVQA Full** | **598.5K** | **4.2M** | ✓ |

用户研究显示：96% 答案正确，83%+ 问题自然。

### 零样本模型评估（语义相似度得分）

| 模型 | ReasonVQA-U | ReasonVQA-B | OK-VQA |
|------|------|------|------|
| BLIP-2 | 46.4 | 46.1 | 45.9 |
| mPLUG-Owl2 | 22.1 | 22.2 | 57.7 |
| GPT-4o | **62.8** | **60.8** | 71.8 |
| Qwen2.5-VL | 59.3 | 58.1 | 84.9 |
| PaliGemma-2-Mix | 43.7 | 41.8 | **86.8** |

**关键发现**：所有模型在 ReasonVQA 上的表现显著低于在标准 VQA 数据集上的表现。PaliGemma-2-Mix 在 OK-VQA 上达到 86.8%，但在 ReasonVQA-B 上仅 41.8%，差距巨大。

### 多选题场景

| 模型 | ReasonVQA-U | ReasonVQA-B | OK-VQA |
|------|------|------|------|
| GPT-4o | 76.6 | 73.4 | 96.7 |
| mPLUG-Owl3 | 68.9 | 68.1 | 99.1 |
| Mantis-Idefics2 | 68.7 | 68.5 | 98.9 |

多选题虽然显著提升准确率，但 ReasonVQA 仍比其他数据集低 20-30 个百分点。

### 微调实验

| 模型 | 零样本 | 微调后 | 提升 |
|------|------|------|------|
| Qwen2-VL-7B | 59.0 | 65.0 | +10.1% |
| PaliGemma-2-3B-Mix | 40.1 | 66.8 | **+66.5%** |
| PaliGemma-2-10B-Mix | 65.6 | 74.5 | +13.5% |

**关键发现**：小模型（PaliGemma-2-3B）通过微调获得了最大提升（+66.5%），表明小模型在适配到特定数据集时有更多的提升空间。

### 按跳数和场景图的分析

- **3 跳问题**准确率显著低于 1 跳和 2 跳，确认了多跳推理的高复杂度
- 2 跳问题有时比 1 跳表现更好，因为更长的问题提供了更多上下文信息
- **含场景图信息的问题**得分较低，说明融入场景图确实增加了数据集复杂度

## 亮点与洞察

1. **极低成本的大规模构建**：利用现有 CV 标注 + Wikidata + 模板化方法，避免了大量人工标注，实现了 4.2M 问题级别的规模
2. **可扩展性设计**：框架可轻松扩展到新的图像源和知识源，用户可通过领域过滤定制数据集
3. **多跳推理的系统性评估**：通过 1/2/3 跳的分级设计，精确衡量模型的推理链条能力
4. **揭示模型短板**：即使最强的 GPT-4o 在 ReasonVQA 上也仅达到 62.8%（开放式），说明当前 VLM 在知识整合和多跳推理上仍有巨大提升空间

## 局限性

1. 模板化方法在语言多样性上仍有局限，约 14% 的问题被评为"不自然"
2. 主要依赖 Wikidata，知识覆盖受限于 Wikidata 的内容
3. 错误选项生成策略相对简单，可能存在通过排除法解题的捷径
4. 答案分布平衡过程可能丢弃部分有价值的样本

## 相关工作

- **知识增强 VQA**：OK-VQA (2019) 首次要求外部知识，KVQA (2019) 支持多跳推理但依赖人工标注
- **自动化构建**：CRIC (2022) 使用 ConceptNet 常识知识，LORA 限于食物场景
- **基础模型评估**：INFOSEEK 和 Encyclopedic-VQA 揭示了人工标注的成本瓶颈

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](../../ACL2025/multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[CVPR 2026\] StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering](../../CVPR2026/multimodal_vlm/star-kvqa_structured_reasoning_traces_for_implicit-knowledge_visual_question_ans.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)

</div>

<!-- RELATED:END -->

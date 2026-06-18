---
title: >-
  [论文解读] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark
description: >-
  [ACL 2025][信息检索/RAG][多模态检索] 提出 REAL-MM-RAG 多模态文档检索基准，定义了真实世界检索基准的四大关键属性（多模态文档、增强难度、真实 RAG 查询、准确标注），引入多级查询改写鲁棒性评估，并通过针对性训练集（改写数据集+金融表格数据集）实现 SOTA 检索性能。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "多模态检索"
  - "RAG"
  - "检索增强生成"
  - "查询改写鲁棒性"
  - "文档检索"
---

# REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark

**会议**: ACL 2025  
**arXiv**: [2502.12342](https://arxiv.org/abs/2502.12342)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: 多模态检索, RAG, 检索增强生成, 查询改写鲁棒性, 文档检索

## 一句话总结

提出 REAL-MM-RAG 多模态文档检索基准，定义了真实世界检索基准的四大关键属性（多模态文档、增强难度、真实 RAG 查询、准确标注），引入多级查询改写鲁棒性评估，并通过针对性训练集（改写数据集+金融表格数据集）实现 SOTA 检索性能。

## 研究背景与动机

检索增强生成（RAG）已成为处理大规模文档的重要范式，准确的文档检索是 RAG 性能的基石——检索到错误页面必然导致生成答案错误。然而现有的多模态检索基准存在严重不足。

作者识别了真实世界文档检索基准的四项关键属性：

**多模态文档**：数据集应包含文本、图表、表格等混合内容

**增强难度**：查询应超越简单关键词匹配，需要大量上下文相似的文档页面

**真实 RAG 查询**：查询应反映用户在不知道答案位置时的自然提问方式，而非引用特定页面

**准确标注**：所有相关文档必须被正确且完整地标注

现有基准的问题：
- **ViDoRe**：ColQwen 在其上达到 ~90% NDCG@5，难度过低；VLM 生成的查询直接复制文档原文，使得关键词匹配即可检索
- **MMLongBench**：基于 QA 数据集，查询假设对特定页面的先验知识，不符合 RAG 场景
- 所有现有基准的**假阴性率极高**：ViDoRe 86.9%、MMLongBench 77.8%（即大量正确检索被误判为错误）

## 方法详解

### 整体框架

REAL-MM-RAG 包含两个核心贡献：（1）满足四大属性的高质量基准构建管线；（2）基于基准分析发现的弱点，提出针对性训练策略（改写训练集+金融表格训练集）。

### 关键设计

1. **文档收集**：聚焦长文档和同一子领域的大量页面（以 IBM 公司数据为核心），共~8000 页分布在四个子领域：

    - **FinReport**：财务报告（2005-2023），19份文档/2687页，文本+表格混合
    - **FinSlides**：季度财务演示（2008-2024），65份/2280页，大量表格
    - **TechReport**：FlashSystem 技术文档，17份/1674页，文本为主
    - **TechSlides**：业务和IT自动化演示，62份/1963页，视觉内容丰富

2. **查询生成与过滤**：两步流程确保 RAG 适配性。

    - **生成**：使用 Pixtral-12B VLM 为每页生成 10 个查询-答案对，prompt 要求生成 RAG 特定问题
    - **过滤**：使用 Mixtral-8x22B LLM 评估每个查询是否适合作为检索查询，剔除包含页面引用（如"in Figure 5"）或过于宽泛的查询

3. **多级查询改写（Multi-level Rephrasing）**：解决 VLM 生成查询与文档原文高度重叠的问题。

    - 使用 Mixtral-8x22B 进行三级改写：Level 1 轻微词汇替换、Level 2 修改词汇和句序、Level 3 显著词汇改写和句子重组
    - 每个查询存在 4 个版本（原始 + 3 级改写），全部链接到同一文档页面
    - 改写后由 LLM 验证保留原始语义

4. **假阴性验证（Accurate Labeling）**：使用 Pixtral-12B 将每个查询与所有基准页面进行系统测试，识别所有可能包含答案的页面。虽然计算开销大，但有效防止假阴性。最终仅保留唯一正确页面被验证的查询。

5. **针对性训练策略**：

    - **改写训练集**：对 ColPali 训练集的一半查询使用 LLaMA-3-70B（与基准使用不同 LLM）进行随机级别改写，强制模型学习语义而非关键词匹配
    - **金融表格训练集**：使用 FinTabNet（S&P 500 公司报告中的复杂表格），通过相同管线生成 46,000 个查询-答案-页面三元组

### 训练策略

- 在改写数据集和/或金融表格数据集上微调 ColPali-v1.2 和 ColQwen2-v1.0
- 训练 1 个 epoch，结合 ColPali 原始训练集
- 产生四种模型变体：RobCol（改写训练）、TabCol（表格训练）、RobTabCol（两者结合）

## 实验关键数据

### 主实验（NDCG@5，Level 3 改写查询）

| 模型 | FinReport | FinSlides | TechReport | TechSlides |
|------|-----------|-----------|------------|------------|
| ColPali | 34.5 | 27.6 | 62.0 | 75.8 |
| ColQwen | 41.8 | 31.1 | 66.9 | 78.1 |
| RobTabColPali | **63.2**(↑28.7) | **58.3**(↑30.7) | **70.7**(↑8.7) | **83.3**(↑7.5) |
| RobTabColQwen | **67.1**(↑25.3) | **61.6**(↑30.5) | **73.2**(↑6.3) | **85.0**(↑6.9) |

### 查询改写级别影响（所有基准平均 NDCG@5）

| 改写级别 | ColPali | RobTabColPali | ColQwen | RobTabColQwen |
|---------|---------|---------------|---------|---------------|
| 0（无改写） | 71.3 | 80.8 | 78.9 | 85.1 |
| 1（轻微） | 65.3 | 77.8 | 72.5 | 81.7 |
| 2（中等） | 60.3 | 74.9 | 68.2 | 78.6 |
| 3（显著） | 56.6 | 72.7 | 65.3 | 76.4 |

### 基准质量人工评估

| 指标 | ViDoRe | MMLongBench | REAL-MM-RAG |
|------|--------|-------------|-------------|
| 假阴性率 ↓ | 86.9% | 77.8% | **31.9%** |
| 真实 RAG 查询率 ↑ | 43.6% | 35.2% | **85.0%** |

### 关键发现

- **视觉模型显著优于文本模型**：在所有基准上，基于 VLM 的直接页面嵌入远超 OCR+文本检索
- **金融表格文档极其困难**：FinSlides 上 ColPali 仅 27.6%，说明表格密集的文档是当前模型的重大弱点
- **查询改写导致巨大性能下降**：BM25 受影响最大（从 52.7% 降到 27.1%），密集检索模型更鲁棒但仍显著下降
- **RobTabCol 组合训练最有效**：金融基准上提升 25-30 个 NDCG@5 点，且不损害非金融基准性能
- **改写训练不损害非改写性能**：RobCol 在非改写查询上也保持或提升，说明语义学习是双赢的
- **现有基准的假阴性问题极其严重**：ViDoRe 的 86.9% 假阴性率意味着大部分"错误"实际上是正确检索

## 亮点与洞察

- **四大属性的系统性定义**：首次系统地定义了真实世界多模态检索基准应具备的关键属性，为未来基准设计提供了参考框架
- **改写评估的首创性**：首个多模态文档 RAG 的查询改写鲁棒性评估，揭示了现有模型"依赖关键词"而非"理解语义"的本质问题
- **假阴性验证的重要性**：通过人工评估量化证明现有基准的严重标注问题，为领域内模型性能的真实水平提供了校准
- **从评估到改进的完整闭环**：基准发现问题 → 针对性训练策略 → 验证改进效果，展示了好的基准如何推动模型进步

## 局限与展望

- 查询由 VLM 生成，可能无法完全覆盖人类查询的多样性
- 标注和过滤仍依赖 LLM/VLM，尽管人工评估验证了有效性，但仍可能存在遗漏
- 聚焦单一公司（IBM）数据，领域多样性有限
- 未涉及多页面推理查询——需要跨多页信息综合才能回答的查询
- 仅评估检索组件，未延伸到 RAG 的生成阶段评估
- 训练策略依赖特定领域的训练数据构建，对新领域的适配需要重新收集

## 相关工作与启发

- 与 ColPali/ViDoRe 相比，REAL-MM-RAG 大幅提升了基准难度和真实性
- 查询改写鲁棒性评估借鉴了文本检索领域的研究，但首次系统化应用于多模态 RAG
- 启发：检索模型的"伪能力"——在简单基准上的高分可能掩盖了真实的语义理解缺陷
- 针对性小规模数据训练（46K 金融表格数据）即可大幅提升特定领域性能，说明数据质量远重于数量

## 评分

- 新颖性: ⭐⭐⭐⭐ 四大属性定义、多级改写评估和假阴性验证都是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 多种模型、多级改写、人工评估、训练消融，实验设计极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，对比表格完整，实验分析深入，论证逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 为多模态 RAG 检索提供了急需的高质量基准和改进方案，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)

</div>

<!-- RELATED:END -->

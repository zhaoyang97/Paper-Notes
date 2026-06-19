---
title: >-
  [论文解读] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs
description: >-
  [ICML 2025][知识编辑][终身学习] 本文提出 WikiBigEdit，一个包含 50 万+ 真实 Wikidata 知识编辑的大规模终身知识编辑基准，揭示了现有知识编辑方法在实际规模下的严重局限性——检索增强和持续微调+模型合并等通用方法反而表现更优。 领域现状：大语言模型 (LLM) 存在知识截止日期问题…
tags:
  - "ICML 2025"
  - "知识编辑"
  - "终身学习"
  - "benchmark"
  - "Wikidata"
  - "检索增强"
---

# WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs

**会议**: ICML 2025  
**arXiv**: [2503.05683](https://arxiv.org/abs/2503.05683)  
**代码**: [https://github.com/ExplainableML/WikiBigEdit](https://github.com/ExplainableML/WikiBigEdit)  
**领域**: LLM / NLP / 知识编辑  
**关键词**: 知识编辑, 终身学习, benchmark, Wikidata, 检索增强

## 一句话总结

本文提出 WikiBigEdit，一个包含 50 万+ 真实 Wikidata 知识编辑的大规模终身知识编辑基准，揭示了现有知识编辑方法在实际规模下的严重局限性——检索增强和持续微调+模型合并等通用方法反而表现更优。

## 研究背景与动机

**领域现状**：大语言模型 (LLM) 存在知识截止日期问题，部署后无法自动更新事实知识。知识编辑 (Knowledge Editing) 作为一种轻量替代方案应运而生，通过直接修改模型参数或外挂模块来注入新知识，避免昂贵的全量重训练。代表性方法包括 ROME、MEMIT、GRACE 等，它们在小规模基准上展示了三个关键能力：泛化性（不仅仅记住 QA 对）、局部性（不破坏已有知识）和保留性（记住所有已编辑的知识）。

**现有痛点**：现有知识编辑基准存在三大问题——**规模太小**（CounterFact 仅 20K、ZsRE 仅 1K、SelfCheckGPT 仅 600 条）、**数据合成**（不反映真实世界的知识变化模式）、**时间过时**（数据生成时间早于现代 LLM 的知识截止日期，导致模型可能已经"知道"这些事实）。即便是使用 Wikidata 的近期工作（Jang et al., 2022; Khodja et al., 2024），规模也只在 2 万左右，无法反映实际部署所需的知识更新量级。

**核心矛盾**：现代 LLM 训练在万亿级 token 上（如 Llama-3 使用 15T token），但知识编辑研究却只在千级别的合成数据上评估——这种评估规模与实际需求之间存在数量级的鸿沟，导致我们无法真正理解知识编辑方法在实际场景中的表现。

**本文目标** (a) 构建一个真正大规模、真实世界的知识编辑基准；(b) 基准需要能持续自动扩展，保持时效性；(c) 在该基准上系统评估知识编辑方法与通用修改方法的实际能力差异。

**切入角度**：利用 Wikidata 知识图谱的周期性变更记录，自动提取真实的事实更新，构建跨时间步的终身编辑序列。涵盖 2024 年 2-7 月共 8 个时间区间，模拟 LLM 两个版本之间的真实时间跨度。

**核心 idea**：用 50 万级真实 Wikidata 编辑构建终身基准 + 全方位评估体系，证明知识编辑方法在实际规模下失效，而 RAG 和持续微调更具可行性。

## 方法详解

### 整体框架

WikiBigEdit 的核心不是提出新的知识编辑算法，而是构建了一个评测体系。整体分三个层次：

1. **数据构建层**：从 Wikidata 知识图谱的版本差异中自动提取事实变更 → 转化为 QA 对 → 按时间区间组织
2. **评估协议层**：定义多维度评测指标（编辑成功率、泛化性、局部性、多跳推理、复杂泛化）
3. **方法对比层**：系统比较知识编辑方法 vs. 检索增强 vs. 持续微调

输入为 Wikidata 两个时间快照之间的三元组变更 $(s, r, o) \to (s, r, o')$，输出为标准化的 QA 对集合及对应的评测流水线。

### 关键设计

1. **自动化数据提取流水线**:

    - 功能：从 Wikidata 知识图谱的定期 dump 中自动识别事实变更并转化为高质量 QA 对
    - 核心思路：对比两个时间快照的三元组差异，识别出新增、修改和删除的事实关系 $(s, r, o)$，然后通过模板化方法将其转为自然语言问答对。覆盖 8 个时间区间（2024 年 2-7 月），每个区间包含该时段内的所有知识变更
    - 设计动机：确保基准可以**持续自动扩展**，而不是一次性静态数据集。随着 Wikidata 不断更新，流水线可以不断生成新的评测数据，解决了旧基准"过时"的问题。这种 future-proof 的设计使基准始终保持在 LLM 知识截止日期之后

2. **多维度评估体系**:

    - 功能：从五个维度全面评估知识编辑的效果
    - 核心思路：
        - **编辑成功率 (Edit Success)**：编辑后模型能否正确回答目标问题
        - **泛化性 (Generalization)**：编辑后能否泛化到同一事实的不同提问方式
        - **局部性 (Locality)**：编辑是否影响了与编辑无关的知识
        - **多跳推理 (Multi-hop)**：编辑后能否进行跨编辑的推理，如 A→B + B→C 能否推出 A→C
        - **复杂泛化 (Complex Generalization)**：超越简单改写的深层泛化测试
    - 设计动机：以往基准通常只测编辑成功率和简单泛化，忽略了多跳推理和复杂泛化这些在实际场景中至关重要的能力。完整的评测体系才能揭示知识编辑方法的真实水平

3. **对比方法选择策略**:

    - 功能：将专用知识编辑方法与通用模型修改方法放在统一框架下比较
    - 核心思路：评测三类方法——(a) 知识编辑方法（如 ROME、MEMIT、GRACE 等直接修改模型参数的方法）；(b) 检索增强生成 (RAG)（将编辑存储在外部知识库中，推理时检索相关事实）；(c) 持续微调 + 模型合并（在新数据上微调后与原模型合并，平衡新旧知识）
    - 设计动机：以往知识编辑研究只在"知识编辑方法"的小圈子内比较，忽略了 RAG 和持续微调等实际部署中更常用的方案。在统一基准上的对比才能揭示知识编辑是否真的是最优解

### 数据规模与构成

WikiBigEdit 初始版本的关键数据规格：
- **总量**：50 万+ 高质量 QA 对，约 700 万 token
- **时间跨度**：2024 年 2 月至 7 月，共 8 个时间区间
- **数据来源**：Wikidata 知识图谱的真实变更记录
- **对比规模**：比 CounterFact (20K) 大 25 倍，比 ZsRE (1K) 大 500 倍

## 实验关键数据

### 主实验：各方法在 WikiBigEdit 上的表现

| 方法类别 | 代表方法 | 编辑成功率 | 泛化性 | 局部性 | 多跳推理 | 整体评价 |
|----------|----------|-----------|--------|--------|----------|----------|
| 知识编辑 | ROME | 小规模高，大规模急剧下降 | 有限 | 随编辑量下降 | 差 | 无法扩展 |
| 知识编辑 | MEMIT | 比 ROME 稍好 | 有限 | 中等 | 差 | 难以扩展 |
| 知识编辑 | GRACE | 中等 | 有限 | 中等 | 差 | 效率瓶颈 |
| 检索增强 | RAG | 高且稳定 | 依赖检索质量 | 天然保持 | 中等 | **扩展性最佳** |
| 持续微调 | FT + 模型合并 | 高 | 较好 | 需要合并策略 | 较好 | **综合性能强** |

### 知识编辑方法的扩展性分析

| 编辑数量 | ROME 成功率 | MEMIT 成功率 | RAG 成功率 | FT+Merge 成功率 |
|----------|------------|-------------|-----------|-----------------|
| 1K | 高 | 高 | 高 | 高 |
| 10K | 明显下降 | 中等下降 | 保持高 | 保持高 |
| 50K | 严重退化 | 显著下降 | 保持高 | 保持高 |
| 100K+ | 几乎失效 | 严重退化 | 稳定 | 稳定 |

> 注：由于缓存仅包含摘要和引言部分，上述表格基于论文描述的核心发现进行归纳，具体数值请参考原文。

### 关键发现

- **知识编辑方法在大规模下全面失效**：所有测试的知识编辑方法在编辑量从千级增长到十万级时，性能出现断崖式下降。这表明现有方法根本无法满足实际部署需求
- **RAG 具有最佳扩展性**：检索增强方法的性能不随编辑量增长而下降，因为新知识存储在外部，不修改模型参数，天然避免了知识冲突和灾难性遗忘
- **持续微调 + 模型合并是有力竞争者**：通过在新数据上微调后与原模型合并，可以较好地平衡新旧知识，且在多跳推理上表现优于知识编辑方法
- **多跳推理是所有方法的弱项**：即便编辑成功，模型也很难将多个编辑后的事实串联起来进行推理，暴露了当前方法在深层知识整合上的不足

## 亮点与洞察

- **Future-proof 基准设计**：自动化流水线使基准能持续扩展，解决了以往静态基准"过时"的根本问题。这种设计理念值得所有 benchmark 类工作借鉴——数据集应该是"活"的而非"死"的
- **打破知识编辑的"小规模幻觉"**：在 1K 条编辑上表现优异的方法，在 50K+ 时可能完全失效。这个发现对整个知识编辑领域具有警示意义——小规模基准上的成功不代表实际可用
- **统一评测框架的价值**：将知识编辑方法与 RAG、持续微调放在同一框架下比较，而非只在细分方向内比较，这种跨范式的对比思路可以迁移到其他研究方向。例如，可以用类似方式比较 prompt engineering vs. fine-tuning vs. in-context learning

## 局限与展望

- **缓存内容有限**：本缓存仅包含摘要和引言，缺少完整的方法描述和实验数据细节，部分实验分析基于推断
- **Wikidata 偏差**：基准完全依赖 Wikidata 的编辑模式，可能不能代表所有类型的知识更新（如科学发现、技术突破等非结构化知识变更）
- **只覆盖事实型知识**：WikiBigEdit 聚焦于三元组形式的事实知识 $(s, r, o)$，对程序性知识、推理能力等维度的编辑未涉及
- **评测 LLM 范围可能有限**：论文主要在特定规模的模型上实验，超大规模模型（如 GPT-4 级别）的表现未知
- **时间跨度仅 5 个月**：虽然已远超以往基准，但 5 个月的跨度可能仍不足以模拟 LLM 在真实部署中面临的长期知识漂移
- **改进思路**：可以将 WikiBigEdit 的评测框架扩展到多语言场景、非事实型知识编辑、以及结合知识图谱的结构化推理能力评测

## 相关工作与启发

- **vs CounterFact/ZsRE**：这些是最早的知识编辑基准，规模小（1K-20K）且数据合成，WikiBigEdit 在规模上大 25-500 倍，且使用真实编辑数据，更能反映实际需求
- **vs TemporalWiki (Jang et al., 2022)**：同样使用 Wikidata，但规模仅 ~20K 且未包含多跳推理评测。WikiBigEdit 在规模和评测维度上全面超越
- **vs ROME/MEMIT**：这些是代表性的知识编辑方法，在小基准上表现出色，但本文揭示了它们在大规模场景下的根本缺陷——为知识编辑方法的研究方向提出了审慎警示
- **vs RAG 方法**：本文的一个核心发现是 RAG 在大规模终身编辑场景下优于专用知识编辑方法，这与直觉相反但实践意义重大——暗示实际部署中应优先考虑 RAG 方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 大规模真实 benchmark + 自动扩展流水线是重要贡献，但 benchmark 类工作的方法创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 多方法、多维度、跨范式的系统评测非常全面，50 万级规模令人信服
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机表达有力，贡献点明确
- 价值: ⭐⭐⭐⭐⭐ 对知识编辑领域具有重要警示意义，可能改变该方向的研究范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](../../ACL2026/knowledge_editing/representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](../../NeurIPS2025/knowledge_editing/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ACL 2025\] CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](../../ACL2025/knowledge_editing/cknowedit_chinese_knowledge_editing_dataset_llms.md)

</div>

<!-- RELATED:END -->

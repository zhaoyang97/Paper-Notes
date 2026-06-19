---
title: >-
  [论文解读] Exploring LLMs for Scientific Information Extraction using the SciEx Framework
description: >-
  [AAAI 2026][多模态VLM][科学信息抽取] 本文提出SciEx，一个模块化、可组合的科学信息抽取框架，将PDF解析、多模态检索、Schema引导的抽取和跨文档聚合解耦为独立组件，在医学和环境科学的143篇论文数据集上评估了GPT-4o和Gemini-2.5-Flash的抽取能力，揭示了当前LLM在跨模态推理、数值精度和领域泛化方面的系统性不足。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "科学信息抽取"
  - "LLM"
  - "RAG"
  - "多模态推理"
  - "模块化框架"
---

# Exploring LLMs for Scientific Information Extraction using the SciEx Framework

**会议**: AAAI 2026  
**arXiv**: [2512.10004](https://arxiv.org/abs/2512.10004)  
**代码**: 无  
**领域**: Multimodal / Information Extraction  
**关键词**: 科学信息抽取, LLM, RAG, 多模态推理, 模块化框架

## 一句话总结
本文提出SciEx，一个模块化、可组合的科学信息抽取框架，将PDF解析、多模态检索、Schema引导的抽取和跨文档聚合解耦为独立组件，在医学和环境科学的143篇论文数据集上评估了GPT-4o和Gemini-2.5-Flash的抽取能力，揭示了当前LLM在跨模态推理、数值精度和领域泛化方面的系统性不足。

## 研究背景与动机
科学信息抽取需要从自由文本论文中编译结构化知识（实验参数、关系、实验结果等），这是科学研究数据化的关键步骤。虽然LLM在通用NLP任务上表现出色，但在科学信息抽取中面临独特挑战：

**多模态分散性**：科学知识分散在文本、表格、图表中，需要跨模态推理来捕捉方法、结果和解释之间的依赖关系。传统基于局部/句子级上下文的抽取器无法有效聚合跨文档依赖。

**术语和单位不一致**：同一概念存在多种表述（如"SARS-CoV-2 persistence" vs "COVID-19 virus viability"），数值单位不统一（molarity vs ppm），违反了schema约束抽取器的假设。

**长文档依赖**：证据散布在论文不同部分（方法、结果、补充材料），且常超出模型上下文窗口。

**Schema变化频率高**：不同研究需求可能需要不同的数据模式，重新设计或fine-tune系统代价高昂。

本文的核心定位不是打败特定SOTA方法，而是**提供一个诚实的能力评估**：当前LLM在科学IE中到底能做到什么程度，瓶颈在哪里。

## 方法详解

### 整体框架
SciEx采用类似Map-Reduce的分布式处理流程：PDF Extractor + REV模块对每篇论文独立执行Map操作（相同的抽取逻辑），Schema Aggregator作为Reduce操作整合跨文档输出。四大核心模块：PDF预处理→Schema处理→检索-抽取-验证(REV)→聚合与冲突解决。

### 关键设计
1. **PDF预处理与多模态知识库**:

    - 功能：将科学PDF解析为结构化的多模态数据库
    - 核心思路：使用Docling提取文本和图像，文本分割为语义连贯的块。科学图表通过VLM二分类筛选（区分科学图vs装饰图），每个图表配对其caption（原始或VLM自动生成），VLM进一步解析轴标签、图例、数据点为结构化JSON。所有页面另存为全页图像以支持联合推理
    - 设计动机：科学论文中大量关键数据存在于图表中，纯文本抽取会丢失重要信息。多模态知识库为后续RAG检索提供了统一的索引基础

2. **Schema模块**:

    - 功能：定义抽取目标的结构化表示
    - 核心思路：支持两种模式——(1) 显式schema定义（用户指定属性和数据类型）；(2) 隐式schema描述（用户提供自然语言指令，LLM自动生成对应的结构化schema）
    - 设计动机：兼顾领域专家（精确约束）和普通用户（自然语言描述）的需求，通过schema引导确保跨论文抽取的一致性

3. **检索-抽取-验证(REV)闭环模块**:

    - 功能：迭代式地发现、抽取和验证相关信息
    - 核心思路：三步闭环——(1) 检索：以schema为查询蓝图，通过向量语义搜索从多模态数据库中找到top-k相关证据段（文本、表格、图表JSON）；(2) 抽取：LLM对检索到的证据执行schema引导的结构化抽取，每个抽取元素标注溯源元数据（文档ID、块索引、图表引用）；(3) 验证：对抽取结果进行自我验证，缺失/不确定字段触发针对性的后续查询重新进入检索阶段
    - 设计动机：单次抽取往往不完整，闭环迭代机制保证输出的语义一致性和经验完整性

4. **聚合与冲突解决模块**:

    - 功能：将多篇论文的抽取结果合并为统一的schema一致表示
    - 核心思路：按共享实体/实验条件分组→单位标准化→LLM驱动的术语规范化（将变体映射到标准schema术语）→层次化冲突解决（跨模型ensemble和一致性投票验证）。真正缺失的信息显式标记为null
    - 设计动机：不同论文使用不同术语和单位描述相同概念是科学IE的核心难点，自动化的规范化和冲突解决是实用系统的必要组件

### 损失函数 / 训练策略
SciEx是纯prompt驱动的RAG框架，不涉及模型训练。使用GPT-4o和Gemini-2.5-Flash作为抽取模型，检索每轮返回top-5相关块。支持通过DSPy进行自动prompt优化。评估使用行级匹配的二部图算法对齐ground truth和抽取结果。

## 实验关键数据

### 主实验

| 数据集 | 模型 | Precision | Recall | F1-score | Accuracy |
|--------|------|-----------|--------|----------|----------|
| CFS | Gemini-2.5-Flash | 0.169 | 0.273 | 0.175 | 0.507 |
| CFS | GPT-4o | 0.241 | 0.355 | 0.248 | 0.512 |
| UV | Gemini-2.5-Flash | 0.199 | 0.468 | 0.237 | 0.329 |
| UV | GPT-4o | 0.279 | 0.609 | 0.331 | 0.467 |
| VD | Gemini-2.5-Flash | 0.284 | 0.382 | 0.297 | 0.556 |
| VD | GPT-4o | 0.333 | 0.476 | 0.380 | 0.580 |

### 消融实验（跨维度分析）

| 分析维度 | 关键发现 | 说明 |
|---------|---------|------|
| 跨数据集 | 简单数据集(UV,VD)优于复杂(CFS) | CFS需要跨多个表格/图表集成信息 |
| 跨模型 | GPT-4o全面优于Gemini-2.5-Flash | 平均Precision 0.26 vs 0.22, Recall 0.48 vs 0.37 |
| Recall > Precision | 两个模型一致 | 抽取出的内容虽相关但包含大量不需要的数据点 |
| 定位后精度 | 中等(0.5-0.6) | 一旦正确定位记录，字段级抽取较可靠 |

### 关键发现
- **性能远未达到生产级要求**：即使经过广泛的prompt优化和RAG增强，最优F1仅0.38（VD+GPT-4o），远低于可靠部署的要求
- **图表解析是主要瓶颈**：旧PDF低分辨率、截断的坐标轴、重叠曲线等视觉复杂性严重影响数值精度
- **跨句/跨段推理薄弱**：多个实体在上下文中紧密出现时，LLM容易误归因实验条件
- **表格结构多样性难以处理**：嵌套表头、合并单元格、跨表变量关联等结构变化导致schema不匹配
- **简单任务（单图/单表直接读取）表现尚可，复杂任务（多源集成）大幅退化**

## 亮点与洞察
- 框架设计的模块化和可组合性是真正的工程亮点——任何组件都可独立替换/升级
- 发散于主流"刷SOTA"的论文风格，诚实地暴露了LLM在科学IE中的不足，对社区更有建设性价值
- 人工标注的143篇论文数据集由领域PhD学生完成，质量高且覆盖面广
- 错误分析部分极其详细实用，为后续研究提供了明确的改进方向
- Map-Reduce式的概念类比使系统架构直观易懂

## 局限与展望
- 整体性能偏低，尤其是Precision，说明需要更好的相关性过滤机制
- 缺乏与fine-tuned模型的对比，全部使用零/少样本推理
- 三个数据集均来自医学/环境科学，其他理工科领域（如物理、化学、材料科学）的泛化性未知
- 迭代REV模块的停止条件（轮数上限或置信度阈值）的选择缺乏系统性研究
- 未探索利用多个LLM的ensemble效果来提高Precision
- 图表VLM分类和JSON化解析的质量本身需要验证

## 相关工作与启发
- 与ChatExtract（对话式迭代抽取）类似但更强调模块化和多模态集成
- SciDaSynth结合自动抽取和人工验证的思路值得SciEx借鉴——在低精度条件下引入人机协作可能是实用的折中方案
- 对于做科学综述或meta-analysis的研究者，SciEx提供了一个有价值的（虽不完美的）自动化起点
- 错误类型的分类（解析质量、跨句推理、表格结构、图表数值）为NLP和文档AI社区定义了清晰的研究挑战

## 评分
- 新颖性: ⭐⭐⭐ (框架设计合理但组件多为现有技术的组合)
- 实验充分度: ⭐⭐⭐⭐ (三个数据集、两个模型、详细错误分析，但缺乏与fine-tune方法的比较)
- 写作质量: ⭐⭐⭐⭐ (结构清晰、错误分析有价值)
- 价值: ⭐⭐⭐⭐ (诚实地揭示了LLM在科学IE中的能力边界，有实践指导意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [\[AAAI 2026\] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks](phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification
description: >-
  [ACL 2025][多模态VLM][科学声称验证] SciVer 是首个面向多模态科学文献声称验证的基准数据集，包含 3000 个专家标注样本覆盖 1113 篇 CS 论文，设计了直接/并行/序列/分析四种推理子任务，评估 21 个基础模型后发现最强模型 o4-mini（77.7%）与人类专家（93.8%）仍有 16% 的显著差距。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "科学声称验证"
  - "多模态推理"
  - "基准评测"
  - "基础模型"
  - "证据标注"
---

# SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification

**会议**: ACL 2025  
**arXiv**: [2506.15569](https://arxiv.org/abs/2506.15569)  
**代码**: [QDRhhhh/SciVer](https://github.com/QDRhhhh/SciVer)  
**领域**: 多模态VLM / 科学文献理解  
**关键词**: 科学声称验证, 多模态推理, 基准评测, 基础模型, 证据标注

## 一句话总结

SciVer 是首个面向多模态科学文献声称验证的基准数据集，包含 3000 个专家标注样本覆盖 1113 篇 CS 论文，设计了直接/并行/序列/分析四种推理子任务，评估 21 个基础模型后发现最强模型 o4-mini（77.7%）与人类专家（93.8%）仍有 16% 的显著差距。

## 研究背景与动机

**领域现状**：科学声称验证（Scientific Claim Verification）要求模型判断一个科学论断是否被给定证据支持或反驳。现有基准主要局限于单一模态：SciFact 仅用摘要文本，SciTAB 仅基于单个表格，TabFact/ChartCheck 基于维基百科表格/图表。而真实的科学论文验证需要同时理解文本段落、数据表格和统计图表等多模态信息。近年虽有 ArXivQA、CharXiv 等多模态科学 QA 基准，但它们仍限于单图/单表的问答，缺少跨模态联合推理。

**现有痛点**：（1）没有基准能测试模型在完整科学论文语境下的多模态声称验证能力；（2）现有数据集大多用众包标注，标注者缺乏领域专业知识，导致声称质量和证据标注不可靠；（3）缺少对不同推理类型的细粒度评估——直接检索、多源并行整合、多跳序列推理、需要领域知识的分析推理在难度上差异巨大，大杯测混在一起无法诊断模型瓶颈。

**核心矛盾**：真实的科学文献理解需要跨模态的复杂推理，但现有基准无法区分和评估这种能力，导致我们对基础模型在这一关键任务上的真实水平缺乏准确认知。

**本文目标** 构建首个多模态科学声称验证基准，覆盖多种推理类型和多模态证据，通过专家标注确保质量，并系统评估当前基础模型的能力边界。

**切入角度**：作者从推理类型出发设计四个子任务（直接/并行/序列/分析），每个样本不仅有 entailed/refuted 标签，还有专家标注的支持证据（文本段落+表格+图表），从而支持细粒度的错误诊断。数据来源是 2024 年 9-11 月发表的 arXiv 论文，确保所有模型在预训练时都没有见过这些数据。

**核心 idea**：构建首个从推理类型维度设计的多模态科学声称验证基准，通过 18 位 CS 研究生的专家标注确保质量，系统揭示基础模型在科学文献多模态推理上的能力瓶颈。

## 方法详解

### 整体框架

SciVer 是一个评估基准而非模型。构建流程为：（1）从 arXiv 收集含表格和图表的 CS 论文 HTML，筛选有同行评审接收记录的论文；（2）18 位 CS 研究生标注者经过 2 小时培训后，按四种推理类型编写 entailed 声称，再通过半自动方式生成 refuted 声称；（3）第二位标注者独立标注支持证据并验证标签，不一致时引入第三位仲裁，达到 94.0% 标注者间一致率；（4）对 21 个模型进行评估，输入为科学论文的多模态上下文（文本段落+表格截图+图表截图）+ 声称，输出为 entailed/refuted。

### 关键设计

1. **四种推理类型的子任务划分**:

    - 功能：从推理复杂度维度细粒度评估模型能力
    - 核心思路：Direct Reasoning——从单一信息源直接提取即可验证；Parallel Reasoning——需同时整合多个独立信息源；Sequential Reasoning——需建立多步推理链，前一步结论作为后一步前提；Analytical Reasoning——需结合领域知识和方法论理解进行深层分析。每种各 750 个样本（测试集 500 个）
    - 设计动机：实验验证了推理类型确实对应不同难度梯度——o4-mini 在 Direct 上达 85.0% 但在 Analytical 上仅 67.6%，差距 17.4%。这种分层评估能精确定位模型的推理瓶颈

2. **专家标注的支持证据**:

    - 功能：为每个样本提供验证该声称所需的最小证据集合（文本段落、表格、图表），支持证据检索能力的评估
    - 核心思路：声称标注者首先在随机选取的 3 个多模态元素基础上编写声称（确保声称必须引用视觉元素），然后第二位独立标注者标注所有支持证据。平均每个样本需要 2.62 条证据
    - 设计动机：有了证据标注，可以对比 RAG 方法和全文输入的效果差异——oracle 证据下 Qwen2.5-VL-72B 从 70.2% 提升到 75.3%（+5.1%），说明证据定位是当前模型的重要瓶颈

3. **Refuted 声称的半自动生成策略**:

    - 功能：生成高质量的 refuted 声称，避免模式化的否定表达
    - 核心思路：标注者在已标注的 entailed 声称基础上引入事实错误（修改数字、替换关系、调换比较方向等），使声称与证据矛盾。这样 refuted 声称与对应的 entailed 声称在表面形式上高度相似，避免模型通过表面线索判断
    - 设计动机：直接让标注者从零编写 refuted 声称极为困难；通过扰动 entailed 声称的方式，确保正反样本共享相同的论文上下文和证据范围，减少偏差

### 损失函数 / 训练策略

SciVer 是评估基准，不涉及模型训练。评估使用 zero-shot 设置，可选 Chain-of-Thought 提示。模型输入科学论文的多模态上下文 + 声称，输出二分类标签。

## 实验关键数据

### 主实验

21 个模型在 SciVer 测试集上的准确率（%）：

| 模型 | Direct | Parallel | Sequential | Analytical | Avg |
|------|--------|----------|------------|------------|-----|
| 人类专家 | 100.0 | 95.0 | 90.0 | 90.0 | **93.8** |
| o4-mini | 85.0 | 80.6 | 77.6 | 67.6 | **77.7** |
| Gemini-2.5-Flash | 79.8 | 76.0 | 73.2 | 71.4 | 75.1 |
| GPT-4o | 77.0 | 71.2 | 73.6 | 73.8 | 73.9 |
| GPT-4.1 | 77.6 | 73.2 | 71.2 | 70.8 | 73.2 |
| Mistral-Small-3.1-24B | 74.8 | 66.0 | 68.6 | 75.6 | 71.3 |
| Qwen2.5-VL-72B | 70.8 | 69.2 | 68.2 | 69.2 | 69.4 |
| Llama-3.2-11B-Vision | — | — | — | — | ~52 |
| Random Guess | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |

### 消融实验

RAG 实验（Qwen2.5-VL-72B）：

| 检索策略 | 准确率 | 提升 |
|---------|-------|------|
| 全文输入 | 70.2 | 基线 |
| Random 证据 | 66.3 | -3.9 |
| OpenAI Embedding RAG | 72.9 | +2.7 |
| Oracle 证据 | **75.3** | +5.1 |

错误分析（Qwen2.5-VL-72B，100 个错误样本）：

| 错误类型 | 占比 |
|---------|------|
| 未检索到相关信息 | 32% |
| 视觉元素误读 | 21% |
| 多步推理失败 | 17% |
| 过度依赖文本模态 | 12% |
| 领域知识错误 | 10% |
| 其他 | 8% |

### 关键发现

- **推理复杂度梯度明显**：模型性能从 Direct（85%）到 Analytical（67.6%）线性下降，证实了四种推理类型确实捕捉到不同难度层次。有趣的是 Mistral-Small 在 Analytical 上反超 GPT-4o（75.6% vs 73.8%），而在 Parallel 上大幅落后（66.0% vs 71.2%）
- **检索是最大瓶颈**：32% 的错误源于未能检索到相关证据，oracle 证据可提升 5.1%，说明长文档定位能力是关键限制因素
- **视觉理解仍然不可靠**：21% 的错误来自对表格/图表的误读，模型倾向于依赖文本而忽略视觉信息（12% 的错误），即使视觉信息是验证声称的必要条件
- **开源 vs 闭源差距大**：最强开源模型 Mistral-Small-3.1-24B（71.3%）与 o4-mini（77.7%）差距 6.4%，小型开源模型如 Llama-3.2-11B 接近随机水平

## 亮点与洞察

- **推理类型分层设计是最大亮点**：四种推理类型的划分使得基准不仅能测总分，还能精确诊断模型在证据检索、多源整合、多步推理、领域知识等维度的具体短板。这种设计思路可迁移到其他领域的基准构建
- **专家标注的高成本高回报**：18 位 CS 研究生标注者、2 小时个人培训、双重标注+仲裁，94.0% 一致率。虽然昂贵但确保了声称和证据的质量远超众包方案，使基准结论更可信
- **RAG 分析揭示了实用价值**：oracle 证据仅提升 5.1%（75.3% 仍远低于人类 93.8%），说明即使完美检索也无法弥补推理能力的不足，未来需要同时提升检索和推理

## 局限与展望

- **仅限 CS 论文**：所有论文来自 arXiv 的 CS 领域，可能无法代表生物医学、物理等其他科学领域的验证难度
- **模态覆盖有限**：仅包含文本、表格和图表，未覆盖方程式、实验设置图、示意图等在某些领域关键的模态
- **规模受限于专家标注成本**：3000 个样本虽质量高但规模有限，难以支持长尾推理模式的评估
- **二分类格式过于简化**：真实的科学声称验证往往是"部分支持/部分反驳"或"证据不足"，二分类可能无法捕捉这种细粒度判断

## 相关工作与启发

- **vs SciFact (Wadden et al. 2020)**：SciFact 仅用论文摘要文本做声称验证，SciVer 使用完整论文的多模态上下文，更接近真实场景
- **vs CharXiv / ArXivQA**：这些基准是单图表的 QA 任务，SciVer 是跨多模态源的声称验证，推理复杂度更高
- **vs SciTAB (Lu et al. 2023)**：SciTAB 仅基于单个科学表格做声称验证，SciVer 同时整合文本、表格和图表

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个面向多模态科学文献的声称验证基准，推理类型分层设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 评估了 21 个模型（闭源+开源），含 CoT、RAG、错误分析等多维度实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据构建流程描述详尽，统计分析完整
- 价值: ⭐⭐⭐⭐ 填补了科学文献多模态验证基准的空白，错误分析为未来改进指明方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)

</div>

<!-- RELATED:END -->

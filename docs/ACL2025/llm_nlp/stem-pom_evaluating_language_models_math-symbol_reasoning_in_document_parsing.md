---
title: >-
  [论文解读] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing
description: >-
  [ACL2025][LLM 其他][数学符号分类] 提出 STEM-PoM 基准数据集（2K+ 数学符号实例），将 Part-of-Math Tagging 与文档解析结合，系统评估 LLM 对数学符号上下文多义性的分类能力，并证明符号分类能力的提升可迁移增强下游数学推理表现。 1. 数学符号的上下文多义性：同一数学符号（如…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "数学符号分类"
  - "Part-of-Math Tagging"
  - "benchmark"
  - "文档解析"
  - "数学推理"
---

# STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing

**会议**: ACL2025  
**arXiv**: [2411.00387](https://arxiv.org/abs/2411.00387)  
**代码**: [jiaruzouu/STEM-PoM](https://github.com/jiaruzouu/STEM-PoM)  
**领域**: LLM/NLP  
**关键词**: 数学符号分类, Part-of-Math Tagging, benchmark, 文档解析, 数学推理

## 一句话总结
提出 STEM-PoM 基准数据集（2K+ 数学符号实例），将 Part-of-Math Tagging 与文档解析结合，系统评估 LLM 对数学符号上下文多义性的分类能力，并证明符号分类能力的提升可迁移增强下游数学推理表现。

## 背景与动机
1. **数学符号的上下文多义性**：同一数学符号（如 $y$）在不同公式和上下文中可能分别是变量、常量或算子，LLM 难以仅凭符号本身判断其语义角色。
2. **Part-of-Math Tagging 数据匮乏**：类比 Part-of-Speech Tagging，数学领域的符号标注任务长期缺乏大规模、多学科、多分类的基准数据集。
3. **现有数据集局限**：已有工作（如 DLMF 上的标注）仅来自单一文献，符号分类类型单一且自洽，无法反映真实文献中跨学科的符号多义问题。
4. **文档解析中的数学理解瓶颈**：传统方法（如 LaTeXML）和先进 LLM 在处理数学密集文档时，对抽象符号的模式匹配和语义理解均存在显著不足。
5. **数学推理的基础能力**：准确分类数学符号是 LLM 进行更复杂数学推理（如解题、证明）的前置能力，但这一环节尚未被系统评估。
6. **多学科覆盖需求**：真实 STEM 文献横跨数学、物理、化学、计算机等领域，需要一个覆盖广泛且标注层次丰富的评估基准。

## 方法详解

### 整体框架：STEM-PoM 基准数据集构建与评估
- **功能**：从 arXiv 数学密集文档中提取数学符号，构建包含 2,109 个标注实例的两级分层分类基准，并在此基准上系统评估多种 LLM。
- **为什么**：填补 Part-of-Math Tagging 在大规模多学科场景下的数据空白，为评估和提升 LLM 的数学符号理解能力提供标准化工具。
- **怎么做**：从 10,000 篇 arXiv 论文中随机采样，结合 MTDE 预过滤符号集，由 33 位领域专家使用自研 STEM-PoM Labeler 工具进行标注。最终从 453 篇文章中提取 2,109 个符号实例，平均每篇 4.7 个符号。

### 关键设计 1：两级分层属性分类体系
- **功能**：定义第一级4类主属性（Variable / Constant / Operator / Unit Descriptor）和第二级6类子属性（Scalar/Vector/Matrix 用于变量；Local/Global/Discipline-Specific 用于常量和算子）。
- **为什么**：数学符号的语义不仅取决于"是什么类型"，还取决于更细粒度的维度/作用域信息。两级分类可以更全面地评估模型的理解深度。
- **怎么做**：每个符号首先由专家标注主属性，再根据上下文标注子属性。标注过程经过一致性检查和互标注者一致性验证（Cohen's Kappa 平均 0.903）。

### 关键设计 2：多粒度上下文评估策略
- **功能**：为每个符号提供三种上下文长度（单句 / 十句 / 全文）分别评估模型分类准确率。
- **为什么**：探究上下文信息量对 LLM 数学符号理解的影响程度，以及不同规模模型对上下文的利用效率差异。
- **怎么做**：通过预定义窗口由领域专家精选与符号最相关的完整句子作为上下文，确保输入信息的准确性和相关性。

### 关键设计 3：下游数学推理迁移验证
- **功能**：在 STEM-PoM 上 LoRA 微调后，评估模型在 GSM8K、MATH、OlympiadBench 上的推理表现变化。
- **为什么**：验证"数学符号分类能力的提升能否迁移到数学推理任务"，从而证明 STEM-PoM 的实际价值。
- **怎么做**：对 Llama2-13B、Mixtral-8x7B、Llama3.1-70B、GPT-4o 先在 STEM-PoM 上 LoRA 微调，再用 3-shot CoT 在下游任务上评估 pass@1。

## 实验关键数据

### 实验 1：第一级分类准确率（不同上下文长度 & 模型）

| 模型 | 单句 | 十句 | 全文 |
|------|------|------|------|
| LSTM | 18.7% | 22.6% | - |
| Llama2-13B | 36.8% | 42.7% | 45.9% |
| Mistral-8x7B | 47.3% | 49.8% | 53.6% |
| Llama3.1-70B | 48.9% | 53.0% | 51.7% |
| Claude3.5-Sonnet | 63.7% | 65.9% | 66.7% |
| GPT-3.5-turbo | 56.8% | 58.7% | 60.6% |
| GPT-4o | 64.9% | 67.4% | 68.5% |

**发现**：SOTA 模型（GPT-4o）在全文上下文下仅达 68.5%，远未解决该任务。GPT-4o 在三种上下文长度下均稳定领先 Llama3.1-70B 约 16%，说明预训练知识量是决定性因素。小模型从长上下文中获益更大。

### 实验 2：STEM-PoM 微调对下游数学推理的迁移效果

| 模型 | GSM8K | MATH | OlympiadBench | 平均 |
|------|-------|------|---------------|------|
| Llama2-13B | 42.5% | 29.1% | 11.5% | 27.7% |
| + LoRA (STEM-PoM) | 44.6% (+2.1) | 31.3% (+2.2) | 13.4% (+1.9) | 29.8% (+2.1) |
| Mixtral-8x7B | 72.4% | 32.6% | 13.7% | 39.6% |
| + LoRA (STEM-PoM) | 74.1% (+1.7) | 34.1% (+1.5) | 16.4% (+2.7) | 41.5% (+1.9) |
| Llama3.1-70B | 91.6% | 47.1% | 26.4% | 55.0% |
| + LoRA (STEM-PoM) | 93.2% (+1.6) | 48.8% (+1.7) | 28.2% (+1.8) | 56.7% (+1.7) |
| GPT-4o | 94.3% | 88.7% | 39.6% | 74.2% |
| + LoRA (STEM-PoM) | 95.2% (+0.9) | 88.9% (+0.2) | 41.2% (+1.6) | 75.1% (+0.9) |

**发现**：所有模型在 STEM-PoM 微调后下游推理均有提升（平均 +0.9~+2.1），在高难度任务 OlympiadBench 上提升尤为明显（最高 +2.7），说明符号分类能力确实可正向迁移到数学推理。

## 亮点
- **新颖的任务定义**：将 Part-of-Math Tagging 系统化并与文档解析结合，提出两级分层分类体系，是该方向首个大规模多学科基准。
- **高质量标注**：33 位领域专家标注，Cohen's Kappa 平均 0.903，标注质量极高。
- **揭示 LLM 盲区**：即使 GPT-4o 也仅达约 68% 准确率，充分暴露了 SOTA 模型在数学符号理解上的显著不足。
- **实用迁移价值**：在 STEM-PoM 上微调可迁移提升下游推理，验证了符号理解作为数学推理基础能力的假说。

## 局限与展望
- **数据集规模有限**：2,109 个实例、453 篇来源论文在统计上仍偏小，可能无法充分覆盖所有学科和符号多义场景。
- **下游迁移提升幅度有限**：微调后推理提升约 1-2 个点，说明符号分类只是数学推理的一个因素，不应被过度解读。
- **仅限 arXiv 英文文献**：未覆盖非英文学术文献和教科书等其他数学密集文本类型。
- **第二级分类中 Matrix 和 Discipline-Specific 样本极少**（Matrix 仅 33 个），可能导致评估不稳定。

## 与相关工作的对比

### vs DLMF-based PoM Tagging (Shan & Youssef, 2021/2024)
DLMF 基于单一文献（Digital Library of Mathematical Functions），符号分类自洽且无多义性。STEM-PoM 来自 10,000 篇多学科 arXiv 论文，充分反映真实文献中的上下文多义性和跨学科变异，更贴近实际应用场景。

### vs MTDE (Hamel et al., 2022)
MTDE 聚焦符号定义提取（NER 式任务），不涉及符号属性的分层分类。STEM-PoM 复用 MTDE 的预过滤符号集，但在此基础上设计了两级分类体系，任务更深入——要求模型不仅识别符号，还要理解其在特定上下文中的数学角色。

### vs 数学推理 Benchmark（GSM8K / MATH / OlympiadBench）
这些 Benchmark 评估的是端到端解题能力，而 STEM-PoM 关注的是更基础的符号理解能力。两者互补：本文实验证明 STEM-PoM 上的能力提升可正向迁移到这些下游推理任务。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个大规模多学科 Part-of-Math Tagging 基准，任务定义新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 7 个模型、3 种上下文长度、微调 + 下游迁移，分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机和实验设计阐述合理
- 价值: ⭐⭐⭐⭐ — 揭示 LLM 数学符号理解短板，提供有价值的评估工具和可迁移的训练数据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)
- [\[ACL 2025\] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)

</div>

<!-- RELATED:END -->

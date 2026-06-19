---
title: >-
  [论文解读] AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions
description: >-
  [ACL 2025][LLM 其他][合规检测] 提出 AIMSCheck——一个端到端的企业现代奴隶制声明合规评估框架，将评估任务分解为句子级多标签分类、token 级 SHAP 解释和证据状态追踪三个层级，同时构建英国和加拿大两个新标注数据集，验证了在澳大利亚数据上微调的模型能有效跨司法管辖区泛化。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "合规检测"
  - "现代奴隶制"
  - "跨司法管辖泛化"
  - "多标签句子分类"
  - "SHAP可解释性"
  - "证据追踪"
---

# AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions

**会议**: ACL 2025  
**arXiv**: [2506.01671](https://arxiv.org/abs/2506.01671)  
**领域**: LLM/NLP, 法律合规  
**关键词**: 合规检测, 现代奴隶制, 跨司法管辖泛化, 多标签句子分类, SHAP可解释性, 证据追踪

## 一句话总结

提出 AIMSCheck——一个端到端的企业现代奴隶制声明合规评估框架，将评估任务分解为句子级多标签分类、token 级 SHAP 解释和证据状态追踪三个层级，同时构建英国和加拿大两个新标注数据集，验证了在澳大利亚数据上微调的模型能有效跨司法管辖区泛化。

## 研究背景与动机

**领域现状**：全球超过 5000 万人受现代奴隶制影响。英国（2015）、澳大利亚（2018）和加拿大（2024）先后颁布《现代奴隶制法案》（MSA），要求大型企业每年发布声明，披露其在运营和供应链中打击现代奴隶制的努力。三国预计每年分别有约 12000、3000 和 6000 份声明提交至政府注册处。

**现有痛点**：
1. **人工审查瓶颈**：全球约 8 万份累积声明，人工审查规模无法扩展——WikiRate 用 8 年才标注了 3500 份
2. **标注数据稀缺**：大多数研究仅人工审查 100-200 份声明，NLP 标注数据集极度匮乏
3. **跨司法管辖泛化未知**：此前唯一可用的数据集 AIMS.au 仅覆盖澳大利亚，是否能泛化到其他法律体系未经验证
4. **缺乏端到端系统**：现有方法缺乏可解释性和证据追踪能力，简单二分类不足以支持真实场景中的合规决策

**核心思路**：构建跨司法管辖区的标注数据集 + 设计多层次端到端评估框架，将合规评估拆解为句子分类→token 解释→证据状态追踪流程，支持人机协作式审查。

## 方法详解

### 整体框架

AIMSCheck 由三个层级串联而成：

1. **句子级预测**（Sentence-Level）：对声明中的每个句子进行多标签二分类，判断其是否与 9 项合规标准相关
2. **Token 级解释**（Token-Level）：使用 SHAP 值量化每个 token 对分类决策的贡献，提供可解释性支撑
3. **证据状态追踪**（Evidence Status）：对已被判定为相关的句子，进一步分类为"已实施"、"未来承诺"或"否认/缺失"，支持纵向合规监控

### 关键设计 1：跨司法管辖区数据集构建 + 标准映射

- 构建 **AIMS.uk**（50 份英国声明，2807 句）和 **AIMS.ca**（50 份加拿大声明，3658 句），遵循与 AIMS.au 相同的预处理和标注规范
- 由领域专家建立三国法律的合规标准映射，从各自法规中提取 **9 项共同标准**（如声明批准/签署、组织结构/运营/供应链描述、风险识别、风险缓解/补救措施、有效性评估等）
- Cohen's Kappa = 0.776，Jaccard Similarity = 0.813，标注一致性达到"显著一致"水平

### 关键设计 2：多模型对比的句子级分类

- 微调模型：BERT（全参数）和 Llama3.2 3B（LoRA），在 AIMS.au 训练集上训练
- 零样本/少样本模型：GPT-4o（零样本 + CoT + few-shot）、DeepSeek-R1（2.51bit 量化）
- 两种输入设定：无上下文 vs. 带 100 词上下文窗口（目标句前后各 50 词）
- 核心发现：微调模型一致优于零样本/少样本模型；100 词上下文是性能与效率的最佳平衡点

### 关键设计 3：证据状态追踪机制

- **未来行动检测**：基于 NLTK 的时态分类器 + 关键词匹配（如"plan to"、"aim to"）
- **否定证据检测**：使用零样本 BART-MNLI 模型，对每个句子构造两个假设（否认/回避 vs. 承认），选择概率更高者；将分类阈值从 0.5 降至 0.35 以提高灵敏度
- 该设计让审查者能区分"已有政策"、"计划实施"和"明确否认"三种状态，支持纵向合规趋势分析

## 实验结果

### 表1：各模型整体 F1 分数对比

| 模型 | 上下文词数 | AIMS.au | AIMS.ca | AIMS.uk |
|------|-----------|---------|---------|---------|
| **Llama3.2 3B** | 100 | **0.738** | **0.719** | **0.686** |
| Llama3.2 3B | 0 | 0.726 | 0.716 | 0.672 |
| BERT | 100 | 0.719 | 0.700 | 0.669 |
| BERT | 0 | 0.694 | 0.677 | 0.653 |
| GPT-4o CoT (few-shot) | 100 | 0.617 | 0.614 | 0.573 |
| GPT-4o | 100 | 0.601 | 0.582 | 0.542 |
| DeepSeek-R1 | 100 | 0.548 | 0.550 | 0.505 |

### 表2：最佳模型（Llama3.2 3B + 100词上下文）各标准 F1

| 合规标准 | AIMS.au | AIMS.ca | AIMS.uk |
|---------|---------|---------|---------|
| 声明批准 (Approval) | 0.864 | 0.947 | 0.783 |
| 供应链描述 | 0.805 | 0.656 | 0.704 |
| 签署 (Signature) | 0.790 | 0.816 | 0.686 |
| 运营描述 (Operations) | 0.769 | 0.803 | 0.789 |
| 组织结构 (Structure) | 0.749 | 0.741 | 0.773 |
| 风险描述 | 0.738 | 0.596 | 0.622 |
| 风险缓解 (Mitigation) | 0.669 | 0.674 | 0.646 |
| 补救措施 (Remediation) | 0.667 | 0.567 | 0.651 |
| 有效性评估 (Effectiveness) | 0.592 | 0.526 | 0.525 |

**关键发现**：
- 微调模型从澳大利亚迁移到英国/加拿大仅有轻微性能下降（<5%），跨司法管辖泛化效果良好
- 明确定义的标准（如 Approval）表现好，主观模糊的标准（如 Effectiveness）较难
- Jensen-Shannon 散度分析证实训练集与测试集词汇分布差异很小，解释了稳定的跨域性能

## 亮点与创新

- **首个跨司法管辖区合规检测框架**：同时覆盖澳/英/加三国法律，建立了标准映射方法论
- **三层分解设计实用性强**：句子分类→token 解释→证据追踪，形成"AI 辅助 + 人工终审"的协作流程
- **验证了跨域泛化可行性**：仅用澳大利亚数据训练即可有效服务英国和加拿大，大幅降低标注成本
- **校准分析表明微调模型概率可信**：预测概率与真实正确率高度吻合，可直接用作置信度指标
- **开源数据集和框架**：数据集发布在 HuggingFace，代码和模型权重公开

## 局限性

- **数据集规模有限**：英国和加拿大各仅 50 份声明，仅为总量的很小子集
- **单一标注者**：虽有迭代精炼，但主要由一位领域专家完成标注，存在主观偏差风险
- **仅英语**：三国法律体系恰好都是英语国家，未涉及多语言场景（如法国尽职调查法、德国供应链法）
- **预处理管道脆弱**：OCR 和句子分割错误会导致列表项和签名等格式问题，影响分类准确性
- **证据追踪方法较简单**：未来行动检测依赖基础时态/关键词匹配，否定证据检测在复杂企业话术下仍有不足
- **模型难以区分近似标准**：如"组织结构"与"运营"、"风险描述"与"风险缓解"之间混淆率较高

## 相关工作

- **法律NLP任务**：LEGAL-BERT (Chalkidis et al., 2020) 针对法律文本预训练；LegalBench (Guha et al., 2023) 提供法律推理基准；ClimateBERT 面向气候变化文本。但这些均未涉及现代奴隶制声明分析
- **现代奴隶制文本分析**：Nersessian & Pachamanova (2022) 使用无监督主题建模分析英国声明趋势；Bora (2019) 用增强智能技术分析声明；AIMS.au (Bora et al., 2025) 提供首个句子级标注数据集，但仅覆盖澳大利亚
- **微调与提示技术**：LoRA 高效微调 (Xu et al., 2024)、上下文感知建模 (Tian et al., 2017; Yang et al., 2021)、Chain-of-Thought 推理 (Wei et al., 2022)
- **可解释性方法**：SHAP (Lundberg & Lee, 2017) 基于博弈论计算 token 级贡献；BERT+SHAP 组合 (Kokalj et al., 2021) 用于 Transformer 分类器解释

## 评分

⭐⭐⭐

> 在社会影响力层面有明确的现实意义——将合规审查从纯人工扩展到 AI 辅助规模化处理。三层分解设计务实且具良好工程价值。但核心技术创新有限：句子分类用的是标准微调/提示方法，SHAP 解释和 BART-MNLI 零样本检测都是现有工具的直接应用，证据追踪的时态关键词方法过于简单。跨司法管辖泛化的验证结论有价值，但三个英语系国家的法律本身高度相似，对更广泛的泛化能力说服力不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)

</div>

<!-- RELATED:END -->

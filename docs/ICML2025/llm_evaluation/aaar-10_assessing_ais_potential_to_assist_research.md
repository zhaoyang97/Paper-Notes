---
title: >-
  [论文解读] AAAR-1.0: Assessing AI's Potential to Assist Research
description: >-
  [ICML 2025][LLM评测][LLM评测基准] 提出 AAAR-1.0 基准，通过公式推断、实验设计、论文弱点发现、审稿质量鉴别四个专家级任务，系统评估 LLM 辅助科研的真实能力，揭示当前模型在深度研究任务上仍有显著不足。 LLM 在日常任务（邮件、问答、创作）上已展现出色能力，但研究人员的核心工作——如头脑风暴研…
tags:
  - "ICML 2025"
  - "LLM评测"
  - "LLM评测基准"
  - "AI辅助科研"
  - "论文审稿"
  - "实验设计"
  - "研究能力评估"
---

# AAAR-1.0: Assessing AI's Potential to Assist Research

**会议**: ICML 2025  
**arXiv**: [2410.22394](https://arxiv.org/abs/2410.22394)  
**代码**: [https://renzelou.github.io/AAAR-1.0/](https://renzelou.github.io/AAAR-1.0/)  
**领域**: 人类理解  
**关键词**: LLM评测基准, AI辅助科研, 论文审稿, 实验设计, 研究能力评估

## 一句话总结

提出 AAAR-1.0 基准，通过公式推断、实验设计、论文弱点发现、审稿质量鉴别四个专家级任务，系统评估 LLM 辅助科研的真实能力，揭示当前模型在深度研究任务上仍有显著不足。

## 研究背景与动机

LLM 在日常任务（邮件、问答、创作）上已展现出色能力，但研究人员的核心工作——如头脑风暴研究想法、设计实验、撰写/审阅论文——是否也能被 LLM 有效辅助？现有工作（如 AI-Scientist、LLM 生成 idea）大多聚焦于高度主观的端到端任务链，评估成本高且难以复现。本文的核心动机有三：

**缺乏面向单步研究任务的系统基准**：已有基准多关注代码实现和模型训练（MLAgentBench 等），忽略了 idea 生成、实验规划、论文审稿等认知密集型环节。

**需要可自动化的评估指标**：生成 idea 等任务需要大量人工评估（Si et al., 2024），阻碍了大规模对比。

**单步任务透明度更高**：相比复杂任务链，单步任务有明确的输入输出预期，能更精确地定位模型能力边界。

## 方法详解

### 整体框架

AAAR-1.0 将科研人员的日常活动解构为四个独立且专家级的任务，每个任务均有清晰的输入/输出定义和配套的自动评估指标：

| 任务 | 缩写 | 输入 | 输出 | 能力考察 |
|------|------|------|------|----------|
| 公式推断 | EqInfer | 论文上下文 + 公式 | 正确/错误（二分类） | 局部上下文推理、符号理解 |
| 实验设计 | ExpDesign | 实验前论文内容（含图） | 实验列表 + 动机解释 | 高层实验规划、领域知识 |
| 论文弱点 | Weakness | 完整论文（含图表） | 弱点列表 | 批判性分析、深度审稿 |
| 审稿鉴别 | ReviewCritique | 论文 + 审稿意见 + rebuttal | 每段审稿是否有缺陷 | 元审稿能力、高级研究经验 |

### 关键设计

#### 任务一：EquationInference（公式推断）

**数据构建四阶段流水线：**

1. **数据爬取与清洗**：从 ACL Anthology（2019-2023）获取 1,762 篇已发表论文的 LaTeX 源码，用正则提取 3,877 个人类撰写的正例公式。选择 LaTeX 源码而非 PDF 解析，避免 PyMuPDF 等工具引入噪声。
2. **LLM 合成负例**：针对每个正例公式，用 GPT-4 基于论文上下文合成 3 个错误公式（高温度解码以保证多样性）。
3. **LLM 过滤**：用 GPT-4 识别"上下文不对齐"的负例（例如出现论文中未定义的符号），剔除全部 3 个负例都有快捷线索的样本，保留 1,449 个正例。
4. **专家审核**：5 位资深博士生使用 TeXlive 等工具编译验证，检查每对正负例是否满足：(a) 语法正确；(b) 编译后负例确实与正例不同。每对样本至少 2 人交叉审查，最终保留 1,049 个正例（淘汰 27.6%）。

**任务设计亮点**：设计为二分类而非多选，因实验表明二分类对 LLM 更具挑战性。

#### 任务二：ExperimentDesign（实验设计）

**高标准标注流程：**

1. **数据源**：从 arXiv 爬取 10k+ 论文（cs.AI/CL/CV，2018-2023），仅保留顶会论文。
2. **领域专家标注**：设严格门槛——资深博士、至少 1 篇顶会论文、4 年以上研究经验、频繁担任审稿人。10 位专家各标注 10 篇，每篇标注所有关键实验及动机解释。
3. **多轮同行讨论**：每篇标注结果由另一位专家 review，检查：实验是否遗漏、摘要是否覆盖关键信息、解释是否合理。迭代讨论直至双方达成一致。
4. **信息泄露处理**：用 GPT-4 删除输入中可能泄露实验的句子（约 9.8% 被删除）。

最终收集 100 个实例，输入为实验前论文上下文（含图片），输出为专家标注的实验计划和解释列表。

#### 任务三：PaperWeakness（论文弱点发现）

1. **数据源**：从 OpenReview 爬取 ICLR 2023 的 3,779 篇匿名投稿，均匀采样 1,000 篇（500 接收 + 500 拒稿），覆盖全部 13 个 track。
2. **弱点提取**：用 GPT-4 从审稿意见中提取弱点，保留原始审稿人文本不做修改。对同一审稿人重复提及的弱点保留不去重（体现重要性）。
3. **输入处理**：因 OpenReview 论文多无 LaTeX 源码，使用 VILA 解析 PDF 文本，PDFFigures-2.0 提取图表。最终 993 个实例。

#### 任务四：ReviewCritique（审稿鉴别）

复用 Du et al. (2024) 的数据集：100 篇论文、380 份审稿，每份审稿分解为句级段落（共 11,376 段），由 40+ 位 AI 研究专家标注每段是否存在缺陷并给出解释。这是四个任务中对高级研究经验要求最高的任务。

### 损失函数 / 训练策略

本文为评测基准而非训练方法，核心创新在于**自动评估指标**设计：

**EqInfer 评估**：标准 F1 分数（二分类），随机猜测基线约 40%。

**ExpDesign 评估——基于蕴含的 Precision/Recall**：

- **En-Precision**：对预测实验列表中每个条目，用 LLM（GPT-4o）判断是否被 ground-truth 列表蕴含
- **En-Recall**：对 ground-truth 中每个条目，用 LLM 判断是否被预测列表蕴含
- **S-Match**：用 SentenceBERT 计算预测解释与 ground-truth 解释的语义相似度

**Weakness 评估——多审稿人语义匹配**：

- **S-Precision**：预测弱点与每位审稿人的弱点列表取 max 语义相似度后对审稿人取均值
- **S-Recall**：每位审稿人的弱点与预测弱点列表取 max 后对审稿人取均值

这种指标设计保留了多审稿人视角的结构信息，避免简单合并丢失多样性。

**ReviewCritique 评估**：分类 F1 分数，评估模型对每个审稿段落的缺陷判断准确性。

## 实验关键数据

### 主实验

| 任务 | 指标 | 最佳模型 | 最佳得分 | 随机/基线 | 备注 |
|------|------|----------|----------|-----------|------|
| EqInfer | F1 | 顶级闭源LLM | ~46% | 40% | 仅略高于随机猜测 |
| ExpDesign | En-Precision | GPT-4 系列 | 中等 | — | 创新但不可行实验多 |
| Weakness | S-Precision | GPT-4 系列 | 中等 | — | 弱点缺乏深度和针对性 |
| ReviewCritique | F1 | 顶级闭源LLM | 较低 | — | 难以识别缺陷审稿 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 开源 vs 闭源 LLM | 各任务 F1/Precision/Recall | 闭源模型整体优于开源，但差距在缩小 |
| LaTeX 源码 vs PDF 解析文本 | EqInfer F1 | LaTeX 源码作为输入噪声更低、信息更丰富 |
| 有/无图片输入 | ExpDesign 指标 | 图片输入对实验设计任务有一定帮助 |
| 不同 prompt 策略 | 各任务指标 | 任务特定 prompt 优于通用 prompt |
| 负例过滤前后 | EqInfer 难度 | 过滤掉浅层快捷线索后任务难度显著提升 |
| 多审稿人 vs 单审稿人评估 | Weakness S-Precision | 多审稿人结构化评估更公平全面 |

### 关键发现

1. **EqInfer**：绝大多数 LLM 的 F1 仅略高于 40% 的随机基线（最高约 46%），说明即便仅需局部上下文推理，LLM 在公式理解方面仍极为薄弱。
2. **ExpDesign**：LLM 设计的实验比人类更多样和创新，但大量实验是平凡的（trivial）、缺乏可行性，且偏离原始研究目标。
3. **Weakness**：LLM 发现的论文弱点缺乏深度和针对性，倾向生成"万金油"式的模糊弱点，可套用于任何论文，缺乏领域特定知识。
4. **ReviewCritique**：LLM 难以有效识别有缺陷的审稿意见，表明其在辅助元审稿方面能力有限。

## 亮点与洞察

- **任务拆解思路优秀**：将科研流程解构为独立可测的单步任务，避免了端到端评估的不可控性，为后续研究提供了清晰的能力诊断框架。
- **数据质量把控严格**：四阶段流水线（爬取→合成→过滤→专家审核）确保每个任务的数据质量，EqInfer 的 27.6% 淘汰率体现了严格标准。
- **评估指标设计精巧**：ExpDesign 的蕴含式 Precision/Recall 和 Weakness 的多审稿人语义匹配巧妙解决了自由文本评估的难题，兼顾了效率和公平性。
- **伦理立场清晰**：强调 LLM 应辅助初级研究者获得不完美但有洞察的建议，而非取代人类主导整个研究过程。

## 局限与展望

1. **数据规模有限**：ExpDesign 仅 100 个实例、ReviewCritique 仅 100 篇论文，可能不足以支撑鲁棒的统计结论。
2. **领域覆盖偏窄**：主要聚焦 AI/NLP/CV 领域（ACL、ICLR 的论文），未覆盖其他学科。
3. **评估指标的局限**：依赖 GPT-4o 作为 ExpDesign 评估器引入了 LLM 自身偏差；SentenceBERT 的语义相似度可能无法捕捉细粒度学术差异。
4. **缺少多模态深度测试**：虽然 ExpDesign 和 Weakness 包含图片输入，但未充分探索视觉理解对科研任务的影响。
5. **时效性挑战**：基准数据来源截至 2023 年，随着 LLM 快速迭代，需要持续更新以保持评估效力。

## 相关工作与启发

- **Si et al. (2024)**：大规模人类研究表明 LLM 可生成新颖但缺乏可行性的 idea，本文的 ExpDesign 任务与此呼应。
- **AI-Scientist (Lu et al., 2024)**：自主科研 agent 包含完整流程，但本文指出评估单步中间输出同样重要。
- **Du et al. (2024)**：揭示 LLM 擅长总结论文优势但难以发现缺陷，直接启发了 Weakness 和 ReviewCritique 任务设计。
- **MLAgentBench (Huang et al., 2024)**：关注实验实现/执行，而本文强调实现之前的高层实验规划同样关键。
- **启发**：可将 AAAR 的单步任务评估方式迁移到其他学科领域，也可作为 LLM 科研 agent 的中间能力检测点。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首个系统化的单步科研任务基准，任务设计新颖 |
| 技术深度 | ⭐⭐⭐⭐ | 数据构建流程严谨，评估指标设计精巧 |
| 实验充分度 | ⭐⭐⭐ | 覆盖多种 LLM，但部分任务数据量偏少 |
| 实用价值 | ⭐⭐⭐⭐ | 为 AI 辅助科研提供了可量化的评估工具 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，动机阐述充分 |
| **总分** | **⭐⭐⭐⭐** | **优秀的基准论文，填补了科研任务评估的空白** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ICLR 2026\] AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation](../../ICLR2026/llm_evaluation/airqa_a_comprehensive_qa_dataset_for_ai_research_with_instance-level_evaluation.md)
- [\[NeurIPS 2025\] The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](../../NeurIPS2025/llm_evaluation/the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](../../ACL2025/llm_evaluation/towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)

</div>

<!-- RELATED:END -->

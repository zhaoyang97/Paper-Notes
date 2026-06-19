---
title: >-
  [论文解读] Automated Reproducibility Has a Problem Statement Problem
description: >-
  [AAAI 2026][可复现性] 提出基于科学方法的可复现性形式化问题定义，将经验性AI研究表示为假设-实验-解释的图结构，并用LLM自动从20篇论文中提取该结构，经原作者评审验证其有效性。 可复现性是科学方法的基石，但独立复现需要大量人力投入。近年多项工作尝试自动化该过程，但各自定义不同的评价方式…
tags:
  - "AAAI 2026"
  - "可复现性"
  - "科学方法"
  - "问题形式化"
  - "LLM自动化"
  - "经验研究"
---

# Automated Reproducibility Has a Problem Statement Problem

**会议**: AAAI 2026  
**arXiv**: [2601.04226](https://arxiv.org/abs/2601.04226)  
**代码**: [有](https://github.com/thijssnelleman/automated-reproducibility)  
**领域**: 其他  
**关键词**: 可复现性, 科学方法, 问题形式化, LLM自动化, 经验研究

## 一句话总结

提出基于科学方法的可复现性形式化问题定义，将经验性AI研究表示为假设-实验-解释的图结构，并用LLM自动从20篇论文中提取该结构，经原作者评审验证其有效性。

## 研究背景与动机

可复现性是科学方法的基石，但独立复现需要大量人力投入。近年多项工作尝试自动化该过程，但各自定义不同的评价方式，导致不同系统之间无法比较：

- **PaperBench**: 评估多个LLM复现能力，最佳模型仅43.4%平均复制分数，但rubric逐篇手工定制，缺乏通用性
- **REPRO-bench**: 社科领域单agent复现，最佳准确率36.6%，依赖代码/数据可用性，不具跨学科通用性
- **SciReplicate-Bench**: 双agent系统（paper agent + code agent），擅长算法总结但实现执行差
- **AutoReproduce**: 引入论文谱系算法但代码实现差距大，且引入独有评价指标无法与他人比较

**核心问题**: 所有工作都缺乏对可复现性的统一形式化定义。各自引入不同评价指标（rubric分数、SSRP指标、CodeBLEU等），导致自动化系统之间无法横向比较。本文旨在基于科学方法提出通用问题陈述框架。

## 方法详解

### 整体框架

将可复现性问题建模为**有向图结构**：任何经验性AI研究可分解为以下要素的图：

```
研究 → 假设(Hypotheses) → 实验(Experiments) → 结果(Outcomes) → 分析(Analysis) → 解释(Interpretations)
```

各要素定义：
1. **假设**: 研究核心主张，可显式陈述或从研究目的推导出的后验假设
2. **实验**: 包含输入数据集、方法/策略、产生的测量结果
3. **分析**: 简化为基于确定指标和统计方法的结果提取
4. **解释**: 基于多实验的分析结果支持或反驳假设

图结构的灵活性：每个实验可关联多个假设；结果可经多种分析处理；解释可基于跨实验的多种分析。**解释被视为相对静态**——在自动化场景中允许解释变化会引入不可控性。

### 关键设计

**1. 后验假设(Post-hoc Hypothesis)构建**

AI论文很少显式表述可检验假设，更多是研究问题和发现。因此LLM从论文中构建的是后验假设——从独立复现角度，实验预期结果就是得出与原作者相同的结论。这一适配使得框架能适用于不正式表述假设的论文。

**2. LLM自动提取流程**

- 模型: Google Gemini 2.5 Pro，温度t=0.0
- 策略: Few-shot prompting，提供示例包括信息可能出现的章节位置和信号关键词
- 迭代优化: 在3篇候选论文(dettmer2024weighted, Gundersen2025, snelleman2024edge)上多轮迭代改进prompt
- 注意: 作者反馈仅用于改进prompt，不是few-shot learning

**3. 评估流程**

- 对20篇论文(涵盖AI多个子领域)进行自动提取
- 每篇论文第一作者审查LLM输出
- 审查内容: 修正措辞错误、验证假设/实验/解释链接、核查实验细节
- 评分: 假设用7点Likert，实验/解释用5点Likert

### 损失函数 / 训练策略

本文不涉及模型训练。核心评估指标：
- **Likert量表评分**: 对假设、实验描述、实验细节、结果解释分别打分
- **Levenshtein编辑距离**: 衡量作者修正幅度（字符级差异百分比）
- **错误率统计**: 对图中各元素及其链接关系的错误比例

## 实验关键数据

### 主实验

**表1: 评估论文统计**（20篇论文，token数从1291到11095不等）

**表2: 方法错误率统计**

| 错误类型 | 错误数 | 比例 |
|---------|--------|------|
| 假设陈述需修改 | 19 | 65.52% |
| 假设编辑距离(平均) | 43字符 | 14.90% |
| 解释陈述需修改 | 9 | 24.32% |
| 解释编辑距离(平均) | 35字符 | 4.79% |
| 实验-假设链接 | 6 | 18.75% |
| 解释-假设链接 | 0 | 0.00% |
| 解释-实验链接 | 2 | 5.41% |
| 实验指标 | 15 | 46.88% |
| 实验统计方法 | 9 | 28.12% |
| 实验策略 | 10 | 31.25% |
| 实验结果 | 1103 | 69.63% |

**整体成功率**: 75%的研究中方法正确捕获了所有元素

### 消融实验

**假设提取质量**:
- 6个案例未完全捕获假设，但均至少部分正确
- 最复杂案例(BosEtAl25): 9个假设捕获7个
- 虽65.52%需修改，但平均仅改43字符(14.90%)，修改幅度小

**实验提取质量**:
- 2个案例完全遗漏某个实验
- 结果数值错误率最高(69.63%)，主因是视觉化结果(图表)难以准确提取
- LLM倾向从文本而非图像提取，非矢量化PDF图像尤其困难

### 关键发现

1. **论文长度影响**: 长论文(>10K tokens)更易遗漏，但非唯一因素——SkaEtAl25(11095 tokens)表现良好
2. **解释优于假设**: 解释修改率(24.32%)远低于假设(65.52%)，因解释更多引用原文
3. **结构化vs视觉化**: 表格数据提取容易，图表数据极不稳定
4. **链接提取优秀**: 解释-假设链接零错误，解释-实验链接仅5.41%

## 亮点与洞察

1. **统一框架的价值**: 首次为自动化可复现性提出基于科学方法的形式化问题定义，使不同系统可横向比较
2. **图结构的可量化性**: 部分复现也可被量化——统计图中多少节点/边被成功复现
3. **大规模作者评审**: 20篇论文原作者参与验证，规模和严谨性在该领域罕见
4. **可作为"前端"**: 先提取问题结构，再交由代码agent执行复现，实现任务分解

## 局限与展望

1. **视觉结果提取差**: 图表/箱线图等提取准确率低，需多模态处理改进
2. **数值精度不足**: 69.63%结果数据需修正，是实际自动复现的关键瓶颈
3. **Prompt简单**: 仅用few-shot prompting，更复杂策略或后训练可提升质量
4. **未闭环**: 仅完成"提取问题"步骤，未与代码生成/执行系统集成
5. **评估偏差**: 作者自评可能偏正面，缺少独立第三方验证
6. **规模有限**: 20篇论文的验证规模偏小，需更大规模验证通用性

## 相关工作与启发

- 可与PaperBench等系统互补：本文提取结构化问题→PaperBench风格agent执行复现
- 图结构可扩展: 为节点定义复现难度权重，生成更细粒度的评分
- 多agent分工: 图的不同子任务可并行分配给不同agent

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ★★★★☆ |
| 技术深度 | ★★★☆☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] The Publication Choice Problem](the_publication_choice_problem.md)
- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](../../NeurIPS2025/others/put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[AAAI 2026\] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring](judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)
- [\[AAAI 2026\] Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)
- [\[AAAI 2026\] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting](on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)

</div>

<!-- RELATED:END -->

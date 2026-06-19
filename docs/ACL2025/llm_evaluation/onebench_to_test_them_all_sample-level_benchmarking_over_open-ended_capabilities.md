---
title: >-
  [论文解读] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities
description: >-
  [ACL 2025][LLM评测][基准评测] ONEBench提出了一种新的基准评测范式：将多个评测数据集的样本合并为统一数据池，通过Plackett-Luce排名聚合算法在样本级别进行模型比较，支持异构指标聚合、不完整数据处理和个性化能力探测。 深度学习已进入"后数据集时代"——基础模型的零样本能力不断扩展…
tags:
  - "ACL 2025"
  - "LLM评测"
  - "基准评测"
  - "模型排名"
  - "Plackett-Luce"
  - "样本级评估"
  - "个性化评测"
---

# ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities

**会议**: ACL 2025  
**arXiv**: [2412.06745](https://arxiv.org/abs/2412.06745)  
**代码**: [GitHub](https://github.com/open-psi/ONEBench)  
**领域**: LLM评测  
**关键词**: 基准评测, 模型排名, Plackett-Luce, 样本级评估, 个性化评测

## 一句话总结

ONEBench提出了一种新的基准评测范式：将多个评测数据集的样本合并为统一数据池，通过Plackett-Luce排名聚合算法在样本级别进行模型比较，支持异构指标聚合、不完整数据处理和个性化能力探测。

## 研究背景与动机

深度学习已进入"后数据集时代"——基础模型的零样本能力不断扩展，传统的固定测试集评测方式越来越不适应需求。静态基准面临以下问题：

**能力覆盖不足**：单个数据集只能测试特定能力，无法全面评估模型的开放式能力。

**数据集偏差**：每个数据集都有自己的采集偏差，可能导致不公平的评估。

**过拟合风险**：模型可能针对特定基准优化，导致实际能力被夸大。

**评价民主化缺失**：传统基准由特定团队创建，标准单一，不同用户群体无法定义自己的评价维度。

核心挑战在于：如何构建一个**动态的、样本级的、支持异构指标和不完整数据的**统一评测框架？

## 方法详解

### 整体框架

ONEBench由四个核心组件构成：
- **数据池D**：来自多个基准的测试样本集合，每个样本包含输入、参考答案和元数据
- **模型集M**：包含一个基线模型和所有待评测模型
- **样本级排名S**：对每个样本，将评测模型按该样本上的表现排序
- **能力标签**：分为任务（如问答、摘要）和概念（如免疫学、地理），支持结构化和语义检索

工作流程：用户通过查询（如"抗体研究"）检索相关样本 → 聚合这些样本上的排名 → 得到针对特定能力的模型排名。

### 关键设计

1. **样本级排名转换**：将不同基准的异构指标（二元正误、数值BLEU分数、偏好排名等）统一转换为序数排名。这种信息损失是有意为之的——序数比较比绝对分数更具鲁棒性和外部效度。Recht et al. (2019) 发现模型排名在不同测试集间保持稳定，即使绝对准确率变化很大。

2. **Plackett-Luce排名聚合**：这是ONEBench的核心算法。假设每个模型mk有一个潜在效用参数γk，样本上的排名由这些效用参数按特定概率模型生成。通过最大似然估计（MLE）恢复效用参数，然后按效用排序得到全局排名。

   Plackett-Luce模型的关键优势：
    - **可辨识性（Identifiability）**：在比较图连通的条件下，效用分布可以唯一恢复（除常数偏移）
    - **样本高效收敛**：只需Ω(|M|log|M|)/k个样本即可准确恢复排名
    - **社会选择性质**：满足匿名性、中性性和无关选项独立性

3. **能力探测（Capability Probing）**：结合两种检索方式：

    - **语义搜索**：使用all-MiniLM-L6-v2（文本）或SigLIP-B16（视觉语言）的嵌入空间进行kNN检索
    - **元数据搜索**：基于结构化元数据（如题目类型、领域分类）进行过滤

4. **终身扩展**：数据池、模型集和排名数据以关系数据库形式存储，支持增量插入新样本、新模型和新排名。

### 损失函数 / 训练策略

Plackett-Luce模型通过最大化对数似然进行参数估计：

γ̂ = argmax_γ log p(s|γ)

似然函数是严格凹的，因此MLE有唯一解。实际中使用rank-breaking技术加速计算。基线模型的效用设为0以消除常数偏移的不确定性。

## 实验关键数据

### 主实验

在四个主流基准上比较Plackett-Luce与其他排名方法的Kendall τ相关系数：

| 数据集 | Elo | LMArena(BT) | ONEBench(PL) |
|--------|-----|-------------|-------------|
| HELM | 0.35±0.13 | 0.85±0.00 | **0.88±0.00** |
| Open LLM Leaderboard | 0.21±0.07 | 0.97±0.00 | **0.99±0.00** |
| VHELM | 0.63±0.02 | 0.69±0.00 | **0.80±0.00** |
| LMMs-Eval | 0.33±0.11 | 0.42±0.00 | **0.64±0.00** |

### 与社会选择理论方法的比较

| 数据集 | Borda Count | Dowdall | ONEBench(PL) |
|--------|-------------|---------|-------------|
| HELM | 0.81 | 0.83 | **0.88** |
| Leaderboard | 0.95 | **0.99** | **0.99** |
| VHELM | 0.35 | 0.21 | **0.79** |
| LMMs-Eval | 0.08 | 0.18 | **0.64** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 95%数据缺失 | 排名仍然稳定 | 评测成本降低20倍 |
| 95%模型测量缺失 | Kendall τ仍然较高 | 适用于不完整评测场景 |
| Top-10模型保持率 | PL方法最优 | 可靠恢复头部排名 |

### 关键发现

1. Plackett-Luce在所有数据集上显著优于Elo和Bradley-Terry，特别是在异构性强的基准（VHELM、LMMs-Eval）上优势更为明显。
2. 即使95%数据缺失，排名仍可保持准确——这意味着评测成本可以降低高达20倍。
3. 能力探测实验中，50个精选概念的检索准确率达到Cohen-κ=0.79(LLM)/0.91(LMM)，CMC@1=0.95/0.94。
4. Elo评分方差极大（依赖对战顺序），不适合大规模基准评测。

## 亮点与洞察

1. **范式转变**：从"一个基准一个分数"到"样本级动态评测"的转变，代表了评测方法论的重要进步。
2. **理论严谨**：不同于许多评测工作的经验导向，ONEBench有坚实的社会选择理论和随机效用模型基础，提供了可辨识性和收敛性的理论保证。
3. **实用性强**：支持个性化查询、终身扩展、不完整数据处理，符合实际评测需求。
4. **排名vs分数的深刻洞察**：从信息损失到外部效度的权衡分析很有说服力——排名比分数更鲁棒、更可迁移。

## 局限与展望

1. **Plackett-Luce假设的限制**：模型假设样本间独立、效用参数固定，不能捕捉"某模型在数学好但语言差"这种能力差异结构。
2. **违反可分性和成对多数一致性**：论文自身承认Plackett-Luce违反了这两个社会选择性质。
3. **语义检索质量**：概念级检索依赖嵌入模型质量，可能存在False Positive。
4. **Ground Truth的循环性**：用原始排行榜分数均值作为Ground Truth来评估聚合算法，存在一定的循环论证。
5. **缺少对评测博弈的讨论**：动态评测也可能被模型开发者博弈。

## 相关工作与启发

- **Chatbot Arena**（Chiang et al., 2024）：使用Bradley-Terry模型聚合人工偏好对。ONEBench将其泛化到自动评测场景。
- **Plackett-Luce模型**（Maystre and Grossglauser, 2015）：高效排名聚合算法，本文首次将其系统应用于LLM/LMM评测。
- **Recht et al. (2019)**：发现模型排名比绝对分数更具跨数据集鲁棒性，为ONEBench使用序数比较提供了理论支持。
- **Zhang and Hardt (2024)**：从经典投票理论角度分析排名聚合，提出不同公平性概念之间的权衡。

## 评分

- 新颖性: ⭐⭐⭐⭐ 评测范式有创新但基础方法来自已有的社会选择理论
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖LLM和LMM两个领域，消融实验全面，对比方法丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分内容较冗长，数学符号有时过度
- 价值: ⭐⭐⭐⭐⭐ 对AI评测实践有直接指导意义，开源框架可立即使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation](beyond_one-size-fits-all_tailored_benchmarks_for_efficient_evaluation.md)
- [\[ICLR 2026\] AutoLibra: Agent Metric Induction from Open-Ended Human Feedback](../../ICLR2026/llm_evaluation/autolibra_agent_metric_induction_from_open-ended_human_feedback.md)
- [\[ACL 2025\] WebWalker: Benchmarking LLMs in Web Traversal](webwalker_benchmarking_llms_in_web_traversal.md)
- [\[ACL 2025\] VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models](voxeval_benchmarking_the_knowledge_understanding_capabilities_of_end-to-end_spok.md)
- [\[ACL 2026\] Automated Creativity Evaluation of Language Models Across Open-Ended Tasks](../../ACL2026/llm_evaluation/automated_creativity_evaluation_of_language_models_across_open-ended_tasks.md)

</div>

<!-- RELATED:END -->

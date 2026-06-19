---
title: >-
  [论文解读] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values
description: >-
  [NeurIPS 2025][LLM安全][分配公平性] 本文系统评估多个 SOTA LLM（GPT-4o、Claude-3.5S、Llama3-70b、Gemini-1.5P）在非策略性资源分配任务中的分配公平性偏好，发现 LLM 与人类存在显著偏差：LLM 偏好效率和无嫉妒性 (EF) 而忽视人类更看重的公平性/平等性 (EQ)，但在选择题模式下 GPT-4o 和 Claude 能正确识别公平方案。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "分配公平性"
  - "LLM对齐"
  - "人类价值观"
  - "公平分配"
  - "资源分配"
---

# Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values

**会议**: NeurIPS 2025  
**arXiv**: [2502.00313](https://arxiv.org/abs/2502.00313)  
**代码**: [github.com/SamarthKhanna/Distributive-Fairness-LLMs](https://github.com/SamarthKhanna/Distributive-Fairness-LLMs)  
**领域**: AI安全 / LLM对齐  
**关键词**: 分配公平性, LLM对齐, 人类价值观, 公平分配, 资源分配

## 一句话总结
本文系统评估多个 SOTA LLM（GPT-4o、Claude-3.5S、Llama3-70b、Gemini-1.5P）在非策略性资源分配任务中的分配公平性偏好，发现 LLM 与人类存在显著偏差：LLM 偏好效率和无嫉妒性 (EF) 而忽视人类更看重的公平性/平等性 (EQ)，但在选择题模式下 GPT-4o 和 Claude 能正确识别公平方案。

## 研究背景与动机

**领域现状**：LLM 越来越多地被用于社会和经济决策场景，其作为社会规划者的潜力备受关注。分配公平性——如何在多个个体间公平分配资源——是社会科学和算法决策的核心问题。

**现有痛点**：已有研究主要关注 LLM 在博弈论场景（囚徒困境、最后通牒等）中的行为，但对非策略性资源分配（社会规划者角色）的公平性研究几乎空白。

**核心矛盾**：公平性本身没有统一定义——公平性 (EQ)、无嫉妒性 (EF)、Rawls 最大化最小值 (RMM) 可能相互冲突，LLM 的偏好层级是否与人类一致？

**本文目标** LLM 在资源分配中是否与人类价值观对齐？行为受哪些公平公理支配？错位根源是什么？

**切入角度**：采用 Herreiner & Puppe (2010) 的经典人类实验数据集，设计不可分物品（有/无金钱）的分配实例，构造公平-效率之间的 tradeoff 场景。

**核心 idea**：人类优先追求平等性 (EQ)，而 LLM 优先追求经济效率 (PO/USW) 和无嫉妒性 (EF)——但当 LLM 从预设选项中选择（而非自行生成）时，GPT-4o 和 Claude 能正确识别最公平方案。

## 方法详解

### 整体框架
设计一系列不可分物品分配实例（2-3 个个体，3-6 件物品，部分带金钱），每个实例让 LLM 和人类分别生成/选择"最公平"的分配方案，统计分析各公平/效率概念的满足频率。

### 关键设计

1. **实例设计与数据集**:

    - 功能：采用 10 个精心设计的实例 I_1 - I_10，每个构造特定公平概念之间的 tradeoff
    - 核心指标：个体 i 对物品 g 的估值为 v_{i,g}，效用函数为加法可分的 u_i(A_i, p_i) = v_i(A_i) + p_i
    - 涵盖场景：EQ vs EF、公平 vs 效率、带金钱缓解不平等、决策者偏见

2. **公平性指标体系**:

    - 公平性 (EQ)：最小化不平等差距 Delta(A,p) = max_{i,j}{u_i - u_j}，完全公平 EQ* 表示 Delta = 0
    - 无嫉妒性 (EF)：对所有 i,j，u_i(A_i, p_i) >= u_i(A_j, p_j)
    - Rawls 最大化最小值 (RMM)：max_{(A,p)} min_i u_i(A_i, p_i)
    - 效率：帕累托最优 (PO)、功利主义社会福利最大化 (USW) max sum_i u_i

3. **选择题模式实验 (Section 4.1)**:

    - 功能：不要求 LLM 生成方案，而是从 5 个预设选项中选择最公平的
    - 核心发现：GPT-4o 和 Claude-3.5S 在 >60% 和 >70% 的情况下选择 EQ* 方案——说明 LLM 知道什么是公平的，但在生成时无法做到
    - 设计动机：区分"计算能力不足"和"价值观不对齐"两种错位来源

4. **Persona / CoT / 意图实验 (Section 5)**:

    - 功能：给 LLM 赋予特定公平概念的 persona，或使用 Chain-of-Thought 提示
    - 核心发现：赋予 EQ persona 后 LLM 仍然难以生成公平方案（GPT-4o 在 EQ persona 下 EQ 满足率 <20%），说明问题不在于理解而在于计算
    - CoT 提示对 GPT-4o 和 Claude 在部分实例上有效，但不一致

### 评估策略
- 每个模型对每个实例查询 100 次，温度 1.0
- 使用 Fisher 精确检验验证人类与 LLM 分布的显著差异 (p < 0.05)
- 二阶段提示策略消除模板敏感性

## 实验关键数据

### 主实验：分配偏好聚合排名（所有实例平均）

| 排名 | 人类 | GPT-4o | Claude-3.5S | Llama3-70b | Gemini-1.5P |
|------|------|--------|-------------|------------|-------------|
| 1st | EQ* (12.4%) | PO (20.4%) | PO (14.9%) | USW (30.8%) | EF (19%) |
| 2nd | EF (9.9%) | USW (11.2%) | EF+PO (14.8%) | PO (26%) | PO (16.8%) |
| 3rd | EF+RMM+PO (9%) | EF+RMM+PO (9.9%) | EF (12.9%) | EF+RMM (7.2%) | USW (11.6%) |

### 选择题模式下的公平偏好

| 模型 | 选择 EQ* 的比例 | 选择 USW 的比例 |
|------|----------------|----------------|
| GPT-4o | >60% | <15% |
| Claude-3.5S | >70% | <10% |
| Llama3-70b | <1% | ~40% |
| Gemini-1.5P | <2% | ~50% |

### 关键发现
- **生成 vs 选择的鲜明对比**：GPT-4o 在生成模式下从不返回 EQ*，但在选择模式下 >60% 选择 EQ*，暗示 LLM 的公平理解存在但计算能力不足
- **金钱利用能力差异大**：GPT-4o 能用金钱缓解不平等（8% 返回 EQ* 方案），其他模型几乎完全不会用金钱实现公平
- **LLM 使用贪心算法**：分析发现 LLM 倾向于轮流分配或按最高估值分配，这类贪心策略天然导致 EF 或 USW 方案
- **自利偏见**：LLM 在作为参与者时表现不一致——有时自利，有时自我牺牲

## 亮点与洞察
- 生成/选择差异的发现非常有洞察力——LLM 不是不知道公平，而是在开放式生成中缺乏探索公平方案的搜索能力。这提示了 RL/SFT 改进方向
- 公平概念层级分析为 LLM 对齐提供了细粒度框架，比简单的"对齐/不对齐"二元判断更有价值
- 实验设计与经济学实证研究方法论紧密结合，每个实例精心设计了特定 tradeoff

## 局限与展望
- 人类数据来源于单一研究 (H&P 2010)，可能存在文化和情境依赖性，跨文化验证缺乏
- 仅考虑加法可分估值的不可分物品分配，未涉及组合估值或策略性环境
- 未尝试 SFT/RLHF 方法直接改善 LLM 的公平分配生成能力
- 不平等容忍度实验中构造的放大实例有限，未系统化探索极端场景

## 相关工作与启发
- **vs Fish et al. (2025, EconEvals)**: 他们评估 LLM 在效率-公平 tradeoff 中的表现，但使用同质化货币资源；本文使用异质估值的不可分物品，更贴近现实
- **vs Horton (2023)**: 他通过 persona 影响 LLM 在独裁者博弈中的行为；本文扩展到更复杂的多人资源分配
- **vs Scherrer et al. (2024, MoralChoice)**: 他们评估 LLM 的道德判断；本文聚焦分配公平这一更具操作性的维度
- 发现 LLM 使用贪心算法的倾向值得关注——可能反映了预训练数据中算法描述的偏向性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统评估 LLM 的分配公平性偏好，生成/选择差异的发现很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖4个模型、10+实例、选择/生成/persona/CoT 多种模式
- 写作质量: ⭐⭐⭐⭐ 结构清晰，与经济学文献衔接好
- 价值: ⭐⭐⭐⭐ 为 LLM 公平性对齐提供了重要的实证基础和改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Fairness of Large Language Models in Multi-document Summarization](../../ACL2025/llm_safety/improving_fairness_of_large_language_models_in_multi-document_summarization.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](../../ACL2026/llm_safety/can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[ACL 2025\] ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](../../ACL2025/llm_safety/dialect_fairness_robustness.md)
- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](../../ACL2025/llm_safety/tug_of_war_fairness_privacy.md)
- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](../../ACL2025/llm_safety/ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)

</div>

<!-- RELATED:END -->

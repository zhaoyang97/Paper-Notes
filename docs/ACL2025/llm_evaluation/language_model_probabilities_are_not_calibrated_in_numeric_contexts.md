---
title: >-
  [论文解读] Language Model Probabilities are Not Calibrated in Numeric Contexts
description: >-
  [ACL 2025][LLM评测][概率校准] 系统研究了语言模型在数值上下文中的概率校准问题，发现即使在简单场景（如从袋中取弹珠）下，包括 GPT-4o 在内的所有测试模型均严重校准不良，存在基于词序、词频和词标识的系统性偏差（如某些模型总选第一个选项，其他模型总选第二个），指令微调加剧了模式崩塌。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "概率校准"
  - "数字推理"
  - "语言模型偏差"
  - "模式崩塌"
  - "系统性偏差"
---

# Language Model Probabilities are Not Calibrated in Numeric Contexts

**会议**: ACL 2025  
**arXiv**: [2410.16007](https://arxiv.org/abs/2410.16007)  
**代码**: 未公开  
**作者**: Charles J. Lovering, Michael Krumdick, Viet Dac Lai, Varshini Reddy, Seth Ebner, Nilesh Kumar, Rik Koncel-Kedziorski, Chris Tanner  
**机构**: Kensho Technologies, Adobe, RIT, Apple  
**领域**: LLM评测  
**关键词**: 概率校准, 数字推理, 语言模型偏差, 模式崩塌, 系统性偏差

## 一句话总结

系统研究了语言模型在数值上下文中的概率校准问题，发现即使在简单场景（如从袋中取弹珠）下，包括 GPT-4o 在内的所有测试模型均严重校准不良，存在基于词序、词频和词标识的系统性偏差（如某些模型总选第一个选项，其他模型总选第二个），指令微调加剧了模式崩塌。

## 研究背景与动机

**问题定义**：某些文本有确定的唯一续写（如"艾菲尔铁塔在[巴黎]"），而另一些文本有自然的概率分布（如"抛硬币结果是[正面/反面]"）。理想情况下，语言模型的输出概率应当与上下文中隐含的数值信息相匹配。

**为何重要**：
   - 若袋中有 98 颗蓝色弹珠和 99 颗红色弹珠，模型应以约 50.2% 的概率输出"红色"
   - 不校准可能在单次交互中不产生影响，但在大量用户或重复使用中会造成系统性危害
   - 典型场景：推荐系统中模型可能因餐厅名字等无关因素总推荐同一家餐厅，医疗诊断中不校准可能导致错误建议

**预训练数据偏差**：预训练数据集中不同数字出现频率不同（以 5 结尾的数字比以 7 结尾的多），这可能导致模型对不同数字产生偏差

**与数学能力的关系**：模型数学推理能力差意味着对数字的表示和使用不佳，这是校准的前提条件

## 方法详解

### 整体框架：三个模板数据集

论文引入三个模板数据集来系统测试模型的数值上下文校准能力：

1. **colors 数据集**（16.5 万题）：从 N₁ 个某色弹珠和 N₂ 个某色弹珠中随机取一颗，模型需在两种颜色之间分配概率。使用 5 个模板、3 种数字范围（1-10, 10-100, 100-999）、110 种颜色排列
2. **wordproblems 数据集**（3.36 万题）：更自然的场景（如"林中有 17 棵云杉和 99 棵雪松，闪电击中的树种是什么"），10 个模板，4-10 对选项
3. **distributions 数据集**（0.45 万题）：从均匀分布中采样整数（如"从区间 [2,5) 中采样"），5 个模板，320 对定义区间的数字

### 关键设计：概率质量的度量方式

对于每个问题实例，上下文 C 的合理续写 token 集合 T = {t₁, t₂, ..., tₙ}，每个 token 对应理想概率 pᵢ。模型输出概率 πᵢ 通过对常见分词变体（大小写、空格等）的概率求和得到。

### 评估指标

1. **Probability Mass (PM)**：合理 token 上的总概率质量 PM(T) = Σπₜ。PM 高表示模型理解了任务，将概率分配在了合理的选项上
2. **Wasserstein Distance (WD)**：衡量模型概率分布与理想校准分布之间的距离，WD 越低校准越好
3. **Relative Entropy (RE)**：模型分布与理想分布的熵差 RE = H(Π) - H(P)。RE < 0 表示模型过度集中（模式崩塌），RE > 0 表示过度分散

### 参考行为分类

论文定义了 6 种参考行为，用于描述模型的系统性偏差模式：

- **Null**：PM 接近零
- **Calibrated**：理想情况 Π = P
- **Pick Higher**：总选数量大的那个选项
- **Pick Lower**：总选数量小的那个
- **Pick First**：总选 prompt 中第一个出现的选项
- **Pick Second**：总选第二个出现的选项

### 测试模型

- **开源模型**（Base + Chat 版本）：Mistral-7B-v0.1/v0.3, Mixtral-8x7B, Yi-1.5-9B/34B, Llama-3.1-8B, gemma-2-9b/27b
- **闭源模型**：gpt-3.5, gpt-4-turbo, gpt-4o-mini, gpt-4o

## 实验结果

### 主实验 1：Probability Mass

| 模型 | colors (Base→Chat) | wordproblems (Base→Chat) | distributions (Base→Chat) |
|------|:---:|:---:|:---:|
| Llama-3.1-8B | 0.38 → 0.80 | 0.54 → 0.86 | 0.82 → 0.78 |
| Mixtral-8x7B | 0.36 → 0.99 | 0.57 → 0.97 | 0.96 → 1.00 |
| gemma-2-27b | 0.54 → 1.00 | 0.59 → 1.00 | 0.96 → 1.00 |
| gpt-4o | - → 1.00 | - → 0.60 | - → 0.95 |

**关键发现**：指令微调模型的 PM 统计显著高于 Base 版本（概率集中在合理选项上），说明模型理解了任务。但 PM 高只是校准的前提，不等于校准好。

### 主实验 2：校准结果 (WD)

| 模型 | colors | wordproblems | distributions |
|------|:---:|:---:|:---:|
| Pick Higher 基线 | 0.47 | 0.44 | - |
| Pick Higher(p=0.7) 基线 | **0.15** | **0.17** | - |
| Llama-3.1-8B | 0.40 | 0.48 | 0.43 |
| gemma-2-27b | 0.40 | 0.48 | 0.59 |
| gpt-4o-mini | 0.40 | 0.57 | 0.57 |
| gpt-4o | 0.40 | 0.57 | 0.49 |

**核心结论**：**所有模型校准表现都很差**。简单的"Pick Higher(p=0.7)"基线（将 70% 的概率分配给较大数字对应的选项）就超越了所有模型，这说明模型虽能识别合理选项，但无法在选项间正确分配概率。

### 主实验 3：Relative Entropy

- 所有模型的 RE 在所有数据集上均**统计显著低于校准水平**
- 指令微调导致熵**大幅下降**，平均在三个数据集上分别降低 0.50/0.36/1.19 bits
- 这意味着指令微调后模型仅保留了理想校准结果 47%/42%/55% 的熵——**模式崩塌**
- 最好校准模型（gpt-*）同时也有最低的相对熵，说明没有模型接近良好校准

### 消融实验：选项身份与顺序的影响

论文对 colors 数据集进行了详细的选项对分析（以 gpt-4o-mini 为例）：

1. **对角线不对称**：将颜色 A 列为第一还是第二个选项，会显著改变模型行为。例如"白色"列为第一时模型倾向 Pick Higher，列为第二时倾向 Pick First
2. **词标识影响**：不同颜色词触发不同偏差模式。某些颜色对（如 purple-white）导致模型几乎 100% 选择第一个选项
3. **词频效应**：预训练语料中不同数字和颜色的频率差异影响校准表现

### 关键发现汇总

1. 模型可以识别合理选项（高 PM），但**无法在选项间正确分配概率**
2. 指令微调虽提高 PM，但导致**严重的模式崩塌**（过度集中于一个选项）
3. 不同模型有不同的系统偏差：gpt-4o-mini 倾向选第一个，Llama-3.1-8B 倾向选第二个
4. 选项的**词标识**（具体是什么颜色词）和**词序**（哪个颜色先出现）显著影响偏差方向
5. 即使在极其简单的数值推理场景下，模型也无法做到基本校准

## 亮点与洞察

1. **问题重要性被低估**：单次交互中校准偏差不明显，但在大规模应用中可能造成系统性不公平（如推荐系统、医疗诊断）
2. **简洁有力的实验设计**：用最简单的概率场景（弹珠抽取）暴露模型的根本缺陷，不需要复杂的数学推理
3. **揭示指令微调的副作用**：RLHF/SFT 让模型"更确定"但非"更正确"，是模式崩塌的根源
4. **系统性偏差的发现**：偏差不是随机的，而是可预测的、与词序和词标识相关的系统性模式
5. **Reference Behavior 分类法**：将模型行为抽象为 6 种参考行为，提供了清晰的分析工具

## 局限性

1. 仅测试了数字范围到 999，更大数字的行为未充分探索
2. 未测试 Chain-of-Thought 等提示策略是否能改善校准
3. 数据集仅涉及简单的比率和概率，未涵盖贝叶斯定理、条件独立等复杂概念
4. 概率质量累积方式（对分词变体求和）是不完美的近似
5. 未探索校准问题的缓解方案

## 相关工作

- **预测校准**: Guo et al. (2017) 研究置信度与准确率的匹配；Wei et al. (2024) 报告 GPT 有良好校准，但 Phan et al. (2025) 持相反结论
- **语言校准**: Yona et al. (2024) 发现模型难以在文本中表达内部不确定性；Kumar et al. (2024) 测量内部（logits）和外部（Likert 量表）的置信度一致性
- **随机性模拟**: Van Koevering & Kleinberg (2024) 发现 LLM 模拟抛硬币时偏向正面和 prompt 中先提到的选项
- **位置偏差**: Pezeshkpour & Hruschka (2024) 展示 LLM 在多选题中偏好特定位置的选项

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ 首次系统研究数值上下文下的概率校准问题，问题定义清晰
- **实验充分性**: ⭐⭐⭐⭐⭐ 三个数据集、16 个模型、多种评估指标、详细的偏差分析
- **实用价值**: ⭐⭐⭐⭐ 揭示了 LLM 在概率推理中的根本缺陷，对下游应用有重要警示
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，可视化出色，Reference Behavior 分类直观易懂

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)
- [\[ICML 2025\] Communicating Activations Between Language Model Agents](../../ICML2025/llm_evaluation/communicating_activations_between_language_model_agents.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](../../NeurIPS2025/llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[NeurIPS 2025\] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories](../../NeurIPS2025/llm_evaluation/not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)

</div>

<!-- RELATED:END -->

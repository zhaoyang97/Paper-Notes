---
title: >-
  [论文解读] TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation
description: >-
  [ACL2025][幻觉检测][不可回答数学题] 提出 TreeCut，一种基于树结构的合成数据集生成方法，通过在树路径上移除必要条件边来系统性生成无穷多的不可回答数学应用题，用以评估 LLM 在面对不可解问题时的幻觉行为。 1. LLM 数学推理能力被高估： GPT-4o 等模型在 GSM8K 上已达人类水平（90% 准确…
tags:
  - "ACL2025"
  - "幻觉检测"
  - "不可回答数学题"
  - "LLM幻觉"
  - "合成数据集"
  - "树结构"
  - "数学推理评估"
---

# TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation

**会议**: ACL2025  
**arXiv**: [2502.13442](https://arxiv.org/abs/2502.13442)  
**代码**: [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math)  
**领域**: 幻觉检测  
**关键词**: 不可回答数学题, LLM幻觉, 合成数据集, 树结构, 数学推理评估

## 一句话总结
提出 TreeCut，一种基于树结构的合成数据集生成方法，通过在树路径上移除必要条件边来系统性生成无穷多的不可回答数学应用题，用以评估 LLM 在面对不可解问题时的幻觉行为。

## 背景与动机

1. **LLM 数学推理能力被高估**: GPT-4o 等模型在 GSM8K 上已达人类水平（>90% 准确率），但这是否真正反映推理能力仍有争议，可能只是模式匹配。
2. **LLM 对不可回答问题产生幻觉**: 已有研究表明 LLM 面对不可回答的数学题时，往往自信地给出错误答案，而非识别出问题不可解。
3. **现有不可回答数据集依赖预设题库**: Li et al. (2024)、Zhou et al. (2024a) 等方法基于 GSM8K 等已有数据集做修改，面临训练数据污染风险，且规模有限。
4. **缺乏结构可控的生成机制**: 现有方法无法精确控制问题的复杂度（深度、变量数、剪切位置等），限制了对幻觉成因的细粒度分析。
5. **人工标注成本高**: Sun et al. (2024) 依赖人工标注创建不可回答问题，仅产出 2600 组，规模小且难以扩展。
6. **需要可回答/不可回答配对**: 理想的评估需要同一问题的可回答与不可回答版本进行对比分析，现有方法难以保证配对质量。

## 方法详解

### 树结构表示
TreeCut 将每道数学应用题建模为一棵树：

- **节点**：每个非根节点代表一个变量（如食物的价格），根节点是保留的特殊节点
- **边**：根节点到子节点的边赋予变量初始值；非根节点间的边表示两个变量之间的线性关系
- **求解路径**：对于任意变量，沿根到该节点的唯一路径即可通过基本算术运算逐步求解，无需解线性方程组

### 可控参数
生成过程通过四个关键参数精确控制问题结构：

| 参数 | 含义 |
|------|------|
| `numVars` | 总变量数（节点数） |
| `ansDepth` | 根节点到被询问变量的距离（路径深度） |
| `compositeName` | 是否使用复合名称（如"Bistro Nice 的汉堡" vs "汉堡"） |
| `cutDepth` | 不可回答版本中，剪切边到被询问变量的距离 |

### 不可回答问题生成（TreeCut 操作）
从一道可回答的数学题出发，在根到被询问变量的路径上移除一条边（cut），使得被询问变量变得不可确定。例如：已知 x₁ 的价格，且 x₂ 和 x₁ 存在线性关系，x₃ 和 x₂ 存在线性关系。若移除 x₁-x₂ 的边，则 x₂ 和 x₃ 均无法求解——两个未知数只有一个方程。

### 数值约束
- 每个食物单价限制为 5–15 的整数
- 线性方程系数取 -3 到 3 之间的非零整数
- 变量随机映射到食物名称，公式通过模板转译为自然语言

### 数据集规模
每组参数配置生成 500 道题，理论上可无限生成互不相同的题目。

## 实验关键数据

### 实验一：各 LLM 在不可回答问题上的幻觉率（零样本）

| ansDepth | Llama-8B | Llama-70B | Qwen-7B | Qwen-72B | GPT-4o | o3-mini |
|----------|----------|-----------|---------|----------|--------|---------|
| 2 | 80.2% | 24.6% | 84.6% | 59.8% | 12.0% | 44.0% |
| 4 | 86.2% | 40.2% | 90.4% | 82.8% | 18.0% | 25.2% |
| 6 | 86.0% | 63.4% | 95.6% | 88.4% | 47.4% | 19.2% |
| 8 | 84.2% | 65.0% | 93.4% | 85.2% | 64.0% | 25.6% |

**关键发现**：所有模型均无法令人满意地识别不可回答问题。Llama-8B 和 Qwen-7B/72B 几乎完全失败；GPT-4o 在深层问题（ansDepth=8）上幻觉率高达 64%。o3-mini 在深层问题上表现最好，但在最简单的 ansDepth=2 时反而幻觉率高达 44%，呈现反常的偏差模式。

### 实验二：可回答问题准确率对比（零样本）

| ansDepth | Llama-8B | GPT-4o | o3-mini |
|----------|----------|--------|---------|
| 2 | 68% (14%) | 99% (1%) | 100% (0%) |
| 8 | 5% (12%) | 84% (2%) | 100% (0%) |

括号内为将可回答问题错误判断为不可回答的比例。GPT-4o 在 ansDepth=8 时可正确解答 84% 的可回答题，却只能识别 36% 的不可回答题，说明**数学计算能力与不可解判断能力之间存在巨大鸿沟**。

### 实验三：GPT-4o 幻觉率的结构因素分析

| 因素 | 影响 |
|------|------|
| 更深的树结构（ansDepth↑） | 幻觉率单调递增 |
| 更复杂的树结构（numVars=ansDepth+2） | 在所有深度上幻觉率高于简单路径 |
| 复合物品名称 | 一致性地提高幻觉率 |
| cutDepth 在路径中段（3–6） | 幻觉率最高（>60%，cutDepth=5 时>70%） |
| cutDepth 在两端（1,2,7） | 幻觉率<50%，模型更容易识别 |

## 亮点

- **无穷可扩展**：基于参数化生成，理论上可产生无穷多不重复的题目，彻底避免数据污染
- **精确可控分析**：四个独立参数允许细粒度研究各因素对幻觉的影响，如首次揭示"剪切位置在路径中段最易引发幻觉"
- **可回答/不可回答配对**：同一棵树生成两种版本，确保对比公平
- **覆盖前沿模型**：评估了包括 o3-mini 在内的最新推理模型，揭示了其特殊的幻觉偏差模式
- **简洁优雅的形式化**：用树+剪边的数学框架统一了问题生成和不可解性验证

## 局限与展望

1. **仅覆盖线性算术应用题**：不涉及几何、概率、代数方程等更广泛的数学领域，生成的题目类型单一
2. **仅评估 zero-shot 和 few-shot CoT**：未探索 self-consistency、tool-augmented reasoning 等更高级的推理框架
3. **问题结构较为人工化**：基于模板生成的自然语言可能与真实数学题的语言复杂度存在差距
4. **缺乏对缓解策略的探索**：识别出问题但未提出如何减少 LLM 幻觉的改进方案
5. **未评估微调后的效果**：不清楚在 TreeCut 数据上微调是否能提升模型的不可解问题识别能力

## 与相关工作的对比

### vs GSM-Plus (Li et al., 2024)
GSM-Plus 基于 GSM8K 通过 GPT-4 提示修改生成不可回答问题，需人工审核，规模受限于原始数据集大小。TreeCut 完全合成生成，规模无上限，且提供精确的结构控制参数。但 GSM-Plus 的题目更贴近真实数学题风格。

### vs MathGAP (Opedal et al., 2024)
MathGAP 同样使用树结构生成合成数学题，但其树中节点代表逻辑命题、根代表答案，且**不涉及不可回答问题**。TreeCut 的树中节点代表变量、叶代表被询问变量，专注于通过剪边制造不可解性。两者虽都用树但本质不同。

### vs Sun et al. (2024)
Sun et al. 依赖人工标注将现有题目改为不可回答，仅产出 2600 对，成本高且难扩展。TreeCut 自动化生成，可产出任意数量的配对题目，并额外提供结构参数用于幻觉成因分析。

## 评分
- 新颖性: ⭐⭐⭐⭐ (树+剪边的形式化框架新颖直观，将不可回答问题生成系统化)
- 实验充分度: ⭐⭐⭐⭐ (覆盖6个模型×多参数配置，zero-shot/few-shot对比，结构因素分析全面)
- 写作质量: ⭐⭐⭐⭐ (形式化清晰，图示直观，但论文较短，部分分析不够深入)
- 价值: ⭐⭐⭐⭐ (为LLM幻觉评估提供了可扩展的系统化工具，对理解推理缺陷有重要意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](../../CVPR2025/hallucination/phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)
- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)

</div>

<!-- RELATED:END -->

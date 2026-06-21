---
title: >-
  [论文解读] Is In-Context Learning Learning?
description: >-
  [ICLR2026][Reasoning][in-context learning] 通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。
tags:
  - "ICLR2026"
  - "Reasoning"
  - "in-context learning"
  - "ICL"
  - "memorisation"
  - "distributional shift"
  - "generalization"
  - "autoregressive models"
---

# Is In-Context Learning Learning?

**会议**: ICLR2026  
**arXiv**: [2509.10414](https://arxiv.org/abs/2509.10414)  
**代码**: 未开源  
**领域**: LLM推理  
**关键词**: in-context learning, ICL, memorisation, distributional shift, generalization, autoregressive models  

## 一句话总结

通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。

## 背景与动机

**领域现状**：In-context learning (ICL) 使自回归语言模型能够通过 next-token prediction 在不更新参数的情况下解决下游任务，只需在 prompt 中提供少量示例（exemplars）。这种能力引发了大量关于 LLM 是否能从少量示例中"学习"未见任务的讨论。

**现有痛点**：

- 关于 ICL 的研究混淆了"推演"（deduction）与"学习"（learning）的概念边界——推演不等于学习
- ICL 并不显式编码给定的观测数据，而是依赖模型的先验知识和 prompt 中的示例
- 现有研究缺乏对记忆效应（memorisation）、预训练数据泄露、分布偏移等混淆因素的系统控制
- 难以判断 ICL 的优异表现究竟来自"真正从示例中学习"还是"先验知识的检索 + 模式匹配"

**核心矛盾**：ICL 在形式上类似于学习（从 prompt 中的示例推断任务规则），但其自回归编码机制是否提供了足够的归纳偏置来支撑稳健的泛化和真正的知识获取？这一问题的答案直接关系到如何正确看待和使用 LLM。

**本文方案**：从理论和实证两个层面回答"ICL 是否是学习"——先从数学上论证 ICL 满足学习的形式定义，再通过大规模控制变量实验（ablation study）评估 ICL 的实际学习能力边界，系统消除记忆效应、预训练泄露、分布偏移和提示风格等混淆因素。

## 方法详解

### 整体框架

本文用理论加实证的双轨方式回答"ICL 是不是学习"：先从数学上论证 ICL 在形式定义层面（类似 PAC 学习框架）确实满足"学习"的标准，再用一套覆盖多种模型架构、提示风格、任务类型和数据分布的大规模控制变量实验，去检验这种"形式上的学习"在实证中能走多远。核心思路是把 ICL 表现拆解为若干混淆来源——预训练记忆、示例分布、提示格式——逐个消除或扰动，看剩下的"真正从示例中学到的"部分还剩多少。

### 关键设计

**1. 把"先验检索"从"示例学习"里剥出来**

ICL 的高准确率有两种来源混在一起：一是预训练时就见过类似任务（记忆/泄露），二是真从 prompt 示例里学到了输入-输出映射；不拆开就会高估 ICL 的学习能力。本文用三条线把二者分离：先用多种污染检测量化预训练数据对测试任务的覆盖，再改用无污染（contamination-free）benchmark 重测；用 zero-shot 表现衡量纯先验水平，只把 few-shot 相对 zero-shot 的增量视为可能的"学习"，即学习增益

$$\Delta_{\text{learn}} = \mathbb{E}[\mathcal{L}(\hat{f}_{\text{zero-shot}})] - \mathbb{E}[\mathcal{L}(\hat{f}_{\text{few-shot}})]$$

其中 $\mathcal{L}$ 为任务损失；最后用随机标签和反事实（label 反转）去探测模型到底有没有真正利用示例里的映射。结果是：充分控制记忆后 $\Delta_{\text{learn}}$ 明显缩小，且随机/反事实标签只带来小幅下降，说明 ICL 表现里相当一部分来自先验知识检索而非从示例中学习。

**2. 看准确率随示例增多到底会不会持续涨**

真正的学习应当随着更多、更好的数据持续改进，所以本文系统扫描示例数量 $k$（从 $k=0$ 的 zero-shot 到 many-shot）、示例分布（类别平衡、样本难度、选择策略与呈现顺序）、提示风格（标准 few-shot、CoT、不同模板与措辞），以及多种规模和架构的模型，观察准确率的缩放模式。关键发现是 $k$ 增大时准确率趋向一个与配置几乎无关的极限

$$\lim_{k \to \infty} \text{Acc}(k; \mathcal{D}, \mathcal{M}, \mathcal{S}) \approx C_{\text{task}}$$

其中 $\mathcal{D}$ 为示例分布、$\mathcal{M}$ 为模型、$\mathcal{S}$ 为提示风格。准确率快速饱和（约 $k>16$ 后增益可忽略）而非随数据持续上升——这条饱和曲线正好和"真正学习"的预期相反，是本文否定性结论的核心证据。

**3. CoT 的提升来自结构规律而非更深的学习**

CoT 在部分任务上确实显著提升准确率，容易被解读为"更会推理"，但本文发现它对 prompt 的分布特征和格式异常敏感：标准 few-shot 表现相对稳定但天花板低，CoT 波动更大且高度依赖推理链的格式与结构，在形式相似、语义不同的任务上准确率差异明显。这种高方差说明 CoT 的收益并非来自对任务更深的学习，而是利用推理链的结构规律性更高效地做模式推演（deduction）——ICL 本质上是在从 prompt 的统计规律里提取模式，而不是编码新知识。

## 实验结果

### 主实验：ICL 学习能力的系统评估

| 控制变量 | 核心结论 | 关键证据 |
|:---------|:---------|:---------|
| 记忆效应 | 预训练记忆显著贡献 ICL 表现 | 使用 contamination-free benchmark 后性能显著下降 |
| 示例数量 | 准确率快速饱和 | $k > 16$ 后增益可忽略，呈对数饱和曲线 |
| 示例分布 | 极限下分布不敏感 | 不同类别分布/难度分布下收敛值相近 |
| 模型选择 | 不同模型在极限下差异缩小 | 多种架构和规模的模型对比 |
| 提示风格 | 标准 few-shot 不敏感，CoT 敏感 | 格式变化实验显示 CoT 波动更大 |
| 输入语言特征 | 表面特征对极限表现影响有限 | 同义改写/格式变化对准确率影响微弱 |

### 消融实验：ICL 信息利用机制分析

| 实验条件 | 准确率表现 | 核心含义 |
|:---------|:----------|:---------|
| 正确标签 | 基线水平（最高） | 标准 ICL 表现 |
| 随机标签 | 微小下降 | 模型不完全依赖示例中的映射关系 |
| 反事实标签 | 中等下降 | 模型部分利用标签信息但并非核心依据 |
| 无标签（仅输入格式） | 接近 few-shot | prompt 的结构性特征比标签内容更重要 |
| 打乱示例顺序 | 几乎不变 | 模型对示例呈现顺序不敏感 |

## 评价

**评分**: ⭐⭐⭐⭐

**优点**：

- 提出了关于 ICL 本质的重要问题，兼具理论深度和实证广度
- 实验设计极为严谨：系统消除记忆、分布偏移、风格等多个混淆因素
- 核心发现（示例增多后准确率饱和且对超参不敏感）具有重要的理论和实践意义
- 对 CoT 的分布敏感性分析提供了理解 CoT 机制的新视角

**不足**：

- 主要得出否定性结论（ICL 的学习能力有限），缺少建设性的改进方向和解决方案
- 部分实验可能受限于特定 benchmark 任务集的选择偏差，对更复杂的推理和生成任务的推广性需验证
- "学习"的定义本身存在多种理解方式（PAC 学习 vs. 贝叶斯学习 vs. 泛化学习），不同定义下结论的稳健性未充分讨论
- 缺少对 ICL 在不同模型规模下行为差异的系统 scaling law 分析

**与相关工作的关键区别**：

- 不同于从 Bayesian inference 角度解释 ICL 的工作（Xie et al., 2021），本文从经验学习论角度系统质疑了 ICL 作为稳健学习机制的假说
- 不同于单因素分析的工作，本文同时控制记忆、分布、模型、风格等多维变量，得出更全面和可靠的否定性结论
- 不同于纯理论分析的工作，本文将数学论证与大规模实验结合，增强了结论的说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning](perk_long-context_reasoning_as_parameter-efficient_test-time_learning.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)
- [\[ICLR 2026\] Divide and Abstract: Autoformalization via Decomposition and Abstraction Learning](divide_and_abstract_autoformalization_via_decomposition_and_abstraction_learning.md)
- [\[ICLR 2026\] Agentic Reinforcement Learning with Implicit Step Rewards](agentic_reinforcement_learning_with_implicit_step_rewards.md)

</div>

<!-- RELATED:END -->

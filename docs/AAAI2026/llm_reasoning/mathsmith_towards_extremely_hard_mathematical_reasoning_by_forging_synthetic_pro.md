---
title: >-
  [论文解读] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy
description: >-
  [AAAI 2026][LLM推理][数学推理] 提出 MathSmith 框架，通过从 PlanetMath 随机抽取数学概念对、采用9种预定义难度策略生成数学题目、并利用 GRPO 强化学习联合优化结构有效性/推理复杂度/答案一致性，生成的高难度合成问题在 AIME 和 OlympiadBench 上显著提升 LLM 数学推理能力。
tags:
  - "AAAI 2026"
  - "LLM推理"
  - "数学推理"
  - "合成数据"
  - "强化学习"
  - "大语言模型"
  - "难度控制"
---

# MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy

**会议**: AAAI 2026  
**arXiv**: [2508.05592](https://arxiv.org/abs/2508.05592)  
**代码**: [https://github.com/Jasaxion/MathSmith](https://github.com/Jasaxion/MathSmith)  
**领域**: 强化学习  
**关键词**: 数学推理, 合成数据, 强化学习, 大语言模型, 难度控制

## 一句话总结

提出 MathSmith 框架，通过从 PlanetMath 随机抽取数学概念对、采用9种预定义难度策略生成数学题目、并利用 GRPO 强化学习联合优化结构有效性/推理复杂度/答案一致性，生成的高难度合成问题在 AIME 和 OlympiadBench 上显著提升 LLM 数学推理能力。

## 研究背景与动机

大语言模型在数学推理上取得了显著进展，但其进步受限于以下关键瓶颈：

**高难度训练数据稀缺**：现有的高质量数学问题大多来自人工编写，数量有限且难度分布不均衡。模型缺乏足够的高难度训练数据来突破推理能力的上限。

**现有合成方法的局限性**：大多数数学问题合成方法依赖于从已有题目中提取模板/结构，然后进行改写（MetaMath）、增强（OpenMathInstruct）、反向翻译（MathGenie）或进化变换（WizardMath）。这些方法本质上受限于人工编写题目的分布和结构，缺乏生成自主性和精确的难度控制。

**数据污染风险**：基于现有题目变换的方式容易产生与测试集相似的问题，引发数据污染问题，使得性能提升的真实性存疑。

**"苦涩教训"的启示**：如 Sutton 所指出的，AI 的可持续进步应依赖通用的、计算密集型方法，而非手工知识。未来的推理智能体应能自主生成高质量、高挑战性的数学问题。

MathSmith 的核心理念类似于"数学铁匠"：从原材料（数学概念和解释对）出发，逐步锻造出复杂而连贯的数学问题，完全不依赖已有的人工编写题目。

## 方法详解

### 整体框架

MathSmith 包含三个核心阶段：
1. **概念-解释收集**：从 PlanetMath 收集具有挑战性的数学概念对
2. **监督微调阶段（SFT）**：用 GPT-4o 生成的种子数据训练基础生成能力
3. **强化学习阶段（RL）**：通过多目标奖励函数优化题目的格式、难度和正确性

此外还包含一个**弱点聚焦改进流水线**模块，用于针对性提升模型在特定概念上的表现。

### 关键设计

1. **概念-解释收集（Concept-Explanation Collection）**：从 PlanetMath（一个以高级数学和理论深度著称的数学百科全书）爬取数学相关页面，过滤非概念条目后，利用 GPT-4o 自动提取每页的核心概念，构建了包含 11,000 个数学概念及其解释的数据集。选择 PlanetMath 的原因在于其概念本身就具有高难度，这从源头保证了生成问题的挑战性。生成时**随机抽取5个概念及解释**作为输入，完全独立于任何已有数学题目，避免数据污染。

2. **九种预定义难度策略（Difficulty Strategies）**：通过分析高难度数学题目的结构和认知特征，设计了9种难度策略作为生成时的软约束：多步推理、跨主题融合、隐式或反向逻辑、干扰项构造、抽象建模、多解路径、高级操作、极端条件和非标准表示。每道生成的题目要求至少包含2种策略以确保足够复杂度。

   SFT 阶段：每个生成样本由两部分组成——**rationale 部分**（恰好5步推理步骤，描述题目构造过程）和 **problem 部分**（最终问题）。用 GPT-4o 生成约 8K 冷启动样本对 Qwen3-8B 进行微调，得到 MathSmith-SFT。

3. **多目标强化学习奖励函数**：核心创新在于设计了由三个分量组成的复合奖励：

   **(1) 结构奖励** $r_{structure}$：检查输出是否包含 rationale 和 problem 两个部分（$r_{format} \in \{0,1\}$），以及推理步数是否为5步（$r_{step}$，5步时达到最大值，偏离时衰减）。$r_{structure} = \alpha_{format} \cdot r_{format} + \alpha_{step} \cdot r_{step}$，其中 $\alpha_{format}=0.7$，$\alpha_{step}=0.3$。

   **(2) 推理复杂度奖励** $r_{complexity}$：利用教师模型 Qwen3-30B-A3B 对生成的题目进行求解，以其推理轨迹的 token 长度作为难度的间接估计：$r_{complexity} = \frac{1}{K \cdot T_{max}} \sum_{i=1}^{K} \ell_{cot}^{(i)}$。其动机是：更具挑战性的问题倾向于引发显著更长的推理轨迹，且长轨迹中包含低熵的中间 token，这些 token 在训练时提供更有信息量的监督信号。

   **(3) 答案一致性奖励** $r_{consistency}$：从教师模型采样 $K$ 个答案，如果存在多数答案（即某个答案出现次数 >K/2），奖励为1，否则为0。这鼓励生成"清晰、无歧义"的问题。

   最终奖励：$r_{total} = r_{structure} + \beta_{complexity} \cdot r_{complexity} + \beta_{consistency} \cdot r_{consistency}$，其中 $\beta_{complexity}=0.7$，$\beta_{consistency}=0.3$。

### 损失函数 / 训练策略

采用 **GRPO（Group Relative Policy Optimization）** 算法优化策略模型 $\pi_\theta$。对每组5个概念输入 $c$，生成 $G$ 道题，计算各自的复合奖励分数 $R_i$，然后归一化为优势估计：$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$，通过 PPO 式的裁剪目标函数（公式8-10）加上 KL 散度惩罚进行更新。

实现细节：
- 基础生成模型：Qwen3-8B，LoRA rank=16，SFT 训练 5 epochs（8×H100）
- RL 阶段使用 verl 库，20×H100 训练，在第100步收敛选取最终模型
- 教师模型采样：$K=5$
- 评估训练统一用 LlamaFactory，学习率 $1e{-5}$，5 epochs

两个模型变体：
- **MathSmith-HC**：使用完整的复杂度 + 一致性奖励（最终推荐版本）
- **MathSmith-Hard**：仅使用复杂度奖励，不含一致性项

## 实验关键数据

### 主实验

基准测试分两个难度层级：简单&中等（GSM8K, MATH-500）和困难（AIME2024, AIME2025, OlympiadBench）。所有方法使用相同数量（50K）训练数据和统一教师模型。

| 模型 | 方法 | GSM8K | MATH-500 | AIME2024 | AIME2025 | Olympiad | Hard Avg (Rel.Imp.) |
|------|------|-------|----------|----------|----------|----------|---------------------|
| Qwen2.5-7B (short-CoT) | baseline | 92.2 | 72.2 | 16.7 | 6.7 | 38.6 | 20.7 |
| Qwen2.5-7B (short-CoT) | PromptCOT | 87.6 | 73.2 | 23.3 | 6.7 | 35.9 | 21.9 (+6.2%) |
| Qwen2.5-7B (short-CoT) | **MathSmith-HC** | 91.2 | 75.2 | **23.3** | **10.0** | **39.9** | **24.4 (+18.1%)** |
| Qwen3-8B (short-CoT) | baseline | 93.4 | 82.8 | 30.0 | 16.7 | 51.0 | 32.6 |
| Qwen3-8B (short-CoT) | **MathSmith-HC** | 92.9 | 84.4 | **33.3** | **23.3** | **53.1** | **36.6 (+12.3%)** |
| DS-R1 (long-CoT) | baseline | 89.3 | 88.6 | 43.3 | 36.7 | 52.4 | 44.1 |
| DS-R1 (long-CoT) | **MathSmith-HC** | 89.2 | 91.6 | **53.3** | **43.3** | **56.5** | **51.0 (+15.6%)** |
| Qwen3-8B (long-CoT) | baseline | 94.8 | 94.4 | 66.7 | 63.3 | 66.2 | 65.4 |
| Qwen3-8B (long-CoT) | **MathSmith-HC** | 95.1 | **96.4** | **76.7** | **70.0** | **68.8** | **71.8 (+9.8%)** |

### 消融实验

| 训练阶段 | Easy&Med Avg | Hard Avg | Available Ratio | 说明 |
|----------|-------------|----------|-----------------|------|
| MathSmith-SFT | 87.7 | 30.3 | 71.50% | 仅 SFT |
| MathSmith-Hard | 89.25 | **36.6** | 84.92% | RL（仅复杂度奖励） |
| MathSmith-HC | 88.65 | **36.6** | **95.38%** | RL（复杂度+一致性）|

| 弱点聚焦方法 | Easy&Med Avg | Hard Avg | Practice Acc |
|-------------|-------------|----------|-------------|
| Original | 38.2 | 14.5 | 23.6 |
| WF Epoch 1 | 69.9 | 18.8 | 33.1 |
| WF Epoch 3 | 77.6 | 21.6 | 34.7 |
| Random（对照） | 69.4 | 15.6 | 30.0 |

### 关键发现

1. **难度越高，提升越大**：在 Hard 基准上改进幅度（9.8%-18.1%）远超 Easy&Medium 基准
2. **Long-CoT 场景优势更明显**：MathSmith 在 long-CoT 设置下的提升显著高于 short-CoT，表明生成的高难度问题能引发更深层推理
3. **可扩展性好**：从 50K 到 200K 数据，MathSmith-HC 保持领先且差距扩大
4. **模型越大受益越大**：在 Qwen3 系列（1.7B→30B）上，大模型从 MathSmith 数据中获益更多
5. **Available Ratio**：MathSmith-HC 的可用率（95.38%）远高于 MathSmith-Hard（84.92%），说明一致性奖励有效提高了题目质量
6. **推理轨迹最长**：MathSmith-HC/Hard 生成的题目在所有数据集中引发最长推理轨迹，验证了 RL 阶段进一步增强了难度

## 亮点与洞察

- **"从头锻造"范式的突破**：完全不依赖已有题目，从随机概念对出发生成题目，彻底避免数据污染——这是与 MetaMath、NuminaMath 等方法的根本区别
- **推理轨迹长度作为难度代理**：简单而有效的启发式——更难的问题引发更长的推理链。虽然长度不直接等于质量，但长链中包含更多低熵中间 token，提供更好的训练信号
- **弱点聚焦机制**：由于每道题可追溯到概念集，可以针对模型薄弱的概念定向生成变体题目，迭代提升。这种可追溯性是框架的独特优势
- **HC vs Hard 的权衡**：一致性奖励看似"降低难度"，实际上大幅提高了可用率（从85%到95%），使大规模合成更实用

## 局限与展望

1. 推理轨迹长度作为难度度量只是启发式的，并不必然等于"真正有助于提升推理能力的难度"
2. 在 GSM8K 等简单文字题上性能偶尔下降，说明过重的推理可能对简单任务产生负面影响
3. 概念集仅来自 PlanetMath，覆盖范围可能有限（偏高等数学，缺少初等、应用数学）
4. 当前难度策略是预定义的9种，未来可探索自适应策略发现
5. 教师模型的能力上限制约了生成问题的质量和难度天花板

## 相关工作与启发

- **PromptCOT**（Zhao et al. 2025a）：最相关工作——用概念驱动提示 + 多步规划生成 Olympiad 级题目，但仍依赖人工选择的概念且缺乏更深的推理控制
- **ScaleQuest**（Ding et al. 2024）：从零开始生成新问题，但缺乏难度控制
- **JiuZhang3.0**（Zhou et al. 2024）：按教育阶段分层控制提示难度
- **GRPO**（Shao et al. 2024）：MathSmith 直接使用的策略优化算法，来自 DeepSeek

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （从概念对出发合成题目 + 用推理长度做难度代理 + 多目标 RL 优化，很有创意）
- 实验充分度: ⭐⭐⭐⭐⭐ （5个基准，4个模型，short/long-CoT，数据/模型缩放实验，弱点聚焦）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式规范，但图表密度大）
- 价值: ⭐⭐⭐⭐⭐ （解决了数学推理数据合成的关键瓶颈，对整个 LLM 推理社区有重大意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](../../ACL2026/llm_reasoning/think_outside_the_policy_in-context_steered_policy_optimization.md)
- [\[ICML 2025\] FMC: Formalization of Natural Language Mathematical Competition Problems](../../ICML2025/llm_reasoning/fmc_formalization_of_natural_language_mathematical_competition_problems.md)
- [\[ICML 2026\] The Easy, the Hard, and the Learnable: Confidence and Difficulty-Adaptive Policy Optimization for LLM Reasoning](../../ICML2026/llm_reasoning/the_easy_the_hard_and_the_learnable_confidence_and_difficulty-adaptive_policy_op.md)
- [\[ACL 2026\] Reinforced Efficient Reasoning via Semantically Diverse Exploration](../../ACL2026/llm_reasoning/reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)
- [\[AAAI 2026\] SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)

</div>

<!-- RELATED:END -->

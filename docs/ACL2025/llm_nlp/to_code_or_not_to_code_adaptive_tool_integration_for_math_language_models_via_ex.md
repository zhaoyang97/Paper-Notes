---
title: >-
  [论文解读] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization
description: >-
  [ACL2025][LLM 其他][数学推理] 提出基于EM框架的AutoCode方法，让数学LLM自主决定何时使用代码工具辅助推理，通过E-step引导探索高潜力代码触发决策+M-step离线RL优化，7B模型在MATH500上提升11%+。 数学推理需要混合能力：数学问题求解同时需要抽象推理（CoT）和精确计算（代码执行…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "数学推理"
  - "工具集成"
  - "元认知"
  - "EM算法"
  - "强化学习"
  - "代码生成"
---

# To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization

**会议**: ACL2025  
**arXiv**: [2502.00691](https://arxiv.org/abs/2502.00691)  
**代码**: [HaozheH3/AutoCode](https://github.com/HaozheH3/AutoCode)  
**领域**: LLM/NLP  
**关键词**: 数学推理, 工具集成, 元认知, EM算法, 强化学习, 代码生成

## 一句话总结
提出基于EM框架的AutoCode方法，让数学LLM自主决定何时使用代码工具辅助推理，通过E-step引导探索高潜力代码触发决策+M-step离线RL优化，7B模型在MATH500上提升11%+。

## 研究背景与动机

**数学推理需要混合能力**：数学问题求解同时需要抽象推理（CoT）和精确计算（代码执行），两者互补但各有短板——CoT易传播数值错误，代码生成存在语义到符号的翻译鸿沟。

**现有混合框架缺乏元认知能力**：Mammoth、DeepSeek-Math、Qwen-2.5-Math等模型虽支持CoT-代码交替，但依赖外部指令或固定模板来决定何时使用代码，无法根据自身能力动态选择。

**SFT范式的根本局限**：监督微调让模型被动模仿：(1)依赖用户指令（"Let's write a Python program"），(2)复制固定代码模板，(3)模仿teacher-forced的工具使用轨迹——无法自主发展工具使用策略。

**标准RL在学习自主代码集成上效率低下**：RL倾向于利用局部策略邻域，在CoT-代码交织的巨大组合空间中探索不足，难以发现高回报的混合推理路径。

**实验证据：现有模型的AutoCode能力接近随机**：DeepSeek-Math-Instruct-7B在自主模式下比显式指令代码模式低11.54%，其代码选择准确率接近50%（随机水平）。

**需要类人的元认知学习机制**：人类学习者通过反复试错、观察结果、更新策略来学习工具使用时机——LLM需要类似的渐进式探索机制。

## 方法详解

### 整体框架：EM for AutoCode
将代码触发决策$c \in \{0, 1\}$建模为隐变量，在EM框架下交替执行：
- **E-step（引导探索）**：找到参考策略$s(c|x_q)$，指示每个问题应倾向使用代码还是纯推理
- **M-step（自我优化）**：在参考策略引导下进行离线RL优化，同时更新工具使用策略和推理能力

### 关键设计1：E-step参考策略估计
通过Monte Carlo rollout估计参考策略：
$$s^*(c|x_q) = \frac{\exp(\alpha \cdot \pi_\theta(c|x_q) Q(x_q, c; \theta))}{Z(x_q)}$$

- 对每个问题$x_q$，分别在$c=0$（纯推理）和$c=1$（代码集成）下生成$K=8$个rollout
- 计算$Q(x_q, c; \theta)$为各决策下的期望成功率
- 参考策略将更高概率分配给期望价值更高的决策，同时平衡模型当前的先验$\pi_\theta(c|x_q)$
- 使用prefix-guided generation引导代码集成解的生成（如"Let's first analyze the problem, then consider if python code could help"）

### 关键设计2：M-step离线RL
根据参考策略$s$从rollout数据中子采样训练集$\mathcal{D}_{\text{train}}$，然后优化：
$$\underset{(x_q, y_a)}{\mathbb{E}}\left[\text{clip}\left(\frac{\pi_\theta(y_a|x_q)}{\pi_{\text{ref}}(y_a|x_q)}, 1-\epsilon, 1+\epsilon\right) \cdot A\right] - \mathbb{E}_{(x_q, c)}\left[\log \pi_\theta(c|x_q)\right]$$

- 第一项：PPO-style clipped策略梯度，优化推理生成质量
- 第二项：交叉熵项，将模型的代码触发策略对齐到参考策略
- 双优化目标联合提升工具使用策略和推理能力

### 训练策略
- 基础模型：Qwen2-Math-Base-7B / DeepSeek-Math-7B / Qwen-2.5-Base-7B
- 训练数据：从Openmath、Math-Instruct、Metamath、MMOS收集119K公开query
- 每轮EM：对7K query生成$K=8$ rollout，子采样后离线RL训练
- 硬件：8×A100 (80GB)，约10小时完成3个epoch

## 实验关键数据

### 表1：AutoCode4Math主要结果（Pass@1 accuracy %）

| 模型 | 代码? | GSM8K | MATH500 | GaoKao | Olympiad | AIME24 | AMC23 |
|------|-------|-------|---------|--------|----------|--------|-------|
| GPT-4o | ✗ | 92.9 | 76.4 | 67.5 | 43.3 | 9.3 | 45.8 |
| NuminaMath-72B | ✓ | 91.4 | 59.2 | 49.4 | 36.7 | 6.5 | 40.6 |
| Qwen2.5-Base-7B | ✗ | 84.88 | 60.4 | 45.45 | 30.37 | 13.2 | 39.38 |
| **AutoCode-Qwen2.5** | **★** | **89.12** | **71.4** | **51.69** | **32.6** | **22.6** | **45.18** |
| Δ | | +4.24 | **+11.0** | +6.24 | +2.23 | **+9.4** | +5.8 |
| DeepSeek-Math-Inst-7B | ✓ | 84.46 | 51.0 | 44.68 | 20.44 | 1.6 | 17.4 |
| **AutoCode-DeepSeek** | **★** | **89.26** | **63.32** | **50.53** | **26.95** | **9.5** | **28.8** |
| Δ | | +4.8 | +12.32 | +5.85 | +6.51 | +7.9 | +11.4 |

### 表2：训练方法消融对比

| 方法 | 训练效率 | 代码调用率 | 收敛性能 | 核心问题 |
|------|---------|-----------|---------|---------|
| Base+RL (DeepSeek-R1式) | 低 | <5% | 持续提升但慢 | 从零学工具使用极低效 |
| SFT only | 高（初始） | 固定模板 | 早期收敛 | 无法超越示范数据 |
| SFT+RL | 中 | 趋向极端化 | 平台期 | 探索不足，陷入局部最优 |
| **EM (本文)** | **高** | **自适应~90%选择准确率** | **持续提升** | 引导探索避免局部最优 |

### 关键发现

1. **MATH500上最高提升11%**：Qwen2.5-Base-7B从60.4%→71.4%，AIME24从13.2%→22.6%（+9.4%）。
2. **标准RL在AutoCode上严重探索不足**：SFT+RL训练过程中代码调用率分布逐渐向两极集中（0%或100%），说明模型利用局部策略而非探索多样路径。
3. **AutoCode选择准确率达89.53%**：AutoCode4Math-Qwen2.5在MATH500上的CoT/Code选择准确率接近90%，远超baseline的~50%。
4. **"免费午餐"效应验证**：AutoCode性能高于单独显式指令CoT或Code的任一方，说明自主集成确实实现了协同增益。
5. **跨模型家族一致有效**：在Qwen2-Math、DeepSeek-Math、Qwen-2.5三个基础模型上均获得显著提升。

## 亮点与洞察

- **问题定义精准**：明确定义了"元认知工具使用"（metacognitive tool-use）这一gap——现有模型能用工具但不知道何时该用，与人类的元认知能力形成对比。
- **EM框架的优雅应用**：将代码触发决策建模为隐变量，自然地将"探索最优策略"（E-step）与"学习该策略"（M-step）解耦，解决RL探索不足的问题。
- **SFT+RL探索不足的可视化**：Figure 5展示了SFT+RL训练中代码调用率分布的极化趋势，直观证明了局部利用问题。
- **Prefix-guided generation的实用技巧**：简单的前缀引导即可诱导模型探索代码集成路径，无需复杂的reward shaping。
- **实现极为高效**：仅7K query、8×A100、10小时即完成训练，对资源受限的研究者友好。

## 局限与展望

1. **仅限数学领域**：未验证在科学推理、通用代码生成等其他需要工具集成的领域的泛化性。
2. **未与o1-like长CoT模型对比**：论文明确排除了依赖test-time scaling的模型（如MCTS、长CoT），与DeepSeek-R1等的直接对比缺失。
3. **代码触发仅二元决策**：当前框架将$c$简化为0/1，未考虑更细粒度的工具选择（如何时用计算器、何时用符号求解器、何时用搜索）。
4. **E-step的rollout成本**：每轮EM对每个query需生成16次rollout（$c=0$和$c=1$各8次），数据策展的计算成本随训练轮次线性增长。
5. **基模型依赖**：需要基模型已具备基本的代码生成能力，对纯语言模型可能不适用。
6. **中间步骤代码触发的涌现**：论文提到mid-reasoning代码触发在warm-up后自然涌现，但未深入分析这一涌现的条件和机制。

## 相关工作与启发

### vs DeepSeek-R1 (Guo et al., 2025)
DeepSeek-R1展示了纯RL可以提升推理能力，但本文实验表明纯RL在学习代码集成策略上效率极低（代码调用率<5%）。关键区别在于：R1的目标空间是推理链的质量，而AutoCode的目标空间是CoT-Code交织的组合空间，后者的探索难度远高于前者。EM框架通过E-step的引导探索弥补了这一gap。

### vs ToRA (Gou et al., 2023) / Mammoth (Yue et al., 2023)
ToRA和Mammoth是早期CoT-Code混合方法的代表，但都依赖SFT从curated数据学习固定的工具使用模式。本文Figure 1明确展示了这些模型的"刚性"问题——它们的AutoCode性能不如显式指令模式，因为它们的代码触发策略本质上是在模仿训练数据的固定模式，而非自主决策。

### vs Singh et al. (2023) / Ni et al. (2022) EM-style自训练
这些工作将EM应用于数学自训练（迭代生成正确解→重训练），但未涉及工具使用决策。本文的关键创新在于将代码触发决策作为EM的隐变量，使EM框架服务于元认知能力的学习，而非仅仅提升推理质量。

## 评分
- 新颖性: 8/10 — EM框架在工具使用决策建模上的应用很有新意，"元认知"角度的问题定义清晰独特
- 实验充分度: 7/10 — 消融充分但benchmark主要为数学，跨领域泛化和与o1类模型的对比缺失
- 写作质量: 8/10 — 动机清晰，数学推导完整，可视化（Figure 5/6）说服力强
- 价值: 8/10 — 为LLM自主工具使用提供了可行且高效的训练框架，实际提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)

</div>

<!-- RELATED:END -->

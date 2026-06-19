---
title: >-
  [论文解读] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game
description: >-
  [ICML 2025][LLM推理][自博弈] 本文提出 Critic-Discernment Game（CDG），通过自博弈语言游戏让 LLM 与"有帮助的批评者"和"误导性批评者"互动，用 ReST 强化学习联合优化三个角色，无需人类或更强模型的监督即可显著提升 LLM 对自身推理过程的理性理解，在数学推理、逐步错误检测、自我纠错和长链推理四个任务上均取得一致提升。
tags:
  - "ICML 2025"
  - "LLM推理"
  - "自博弈"
  - "推理理性"
  - "Critic-Discernment Game"
  - "强化学习"
  - "自我纠错"
---

# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game

**会议**: ICML 2025  
**arXiv**: [2506.22920](https://arxiv.org/abs/2506.22920)  
**代码**: 有（论文中提到已开源）  
**领域**: LLM推理  
**关键词**: 自博弈, 推理理性, Critic-Discernment Game, 强化学习, 自我纠错

## 一句话总结
本文提出 Critic-Discernment Game（CDG），通过自博弈语言游戏让 LLM 与"有帮助的批评者"和"误导性批评者"互动，用 ReST 强化学习联合优化三个角色，无需人类或更强模型的监督即可显著提升 LLM 对自身推理过程的理性理解，在数学推理、逐步错误检测、自我纠错和长链推理四个任务上均取得一致提升。

## 研究背景与动机
**领域现状**：LLM 在数学和代码等推理任务上展现出了强大的能力，但近期研究表明即使是最好的模型也缺乏对推理过程的真正理解，更多依赖于概率模式匹配。

**现有痛点**：LLM 推理过程不稳定，容易产生幻觉和错误，且难以自主检测和纠正这些问题。长链推理中，中间错误会不断累积，导致最终结果偏差越来越大。

**核心矛盾**：现有方法（如 PRM 或偏好数据对）依赖人工标注的步级监督数据，难以扩展；且无法显式定义细粒度步骤，仅能指出哪步更好而无法解释原因。

**本文目标**：如何在不依赖人类或更强模型监督的前提下，增强 LLM 对推理过程的理性理解能力。

**切入角度**：通过设计一个语言层面的自博弈游戏，让模型在与不同意图的批评者互动中学习辨别自己推理步骤的正确性。

**核心 idea**：让模型同时学会在面对误导性批评时坚守正确答案，以及在收到建设性反馈时修正错误答案，从而真正理解自身推理过程。

## 方法详解

### 整体框架
CDG 是一个三角色自博弈框架。Prover（证明者）首先对问题给出解答，然后接受来自 Critic 的批评。Critic 有两种角色：Helpful Critic（帮助者）在 Prover 给出错误解答时协助纠正，Misleading Critic（误导者）在 Prover 给出正确解答时试图诱导修改。三个角色通过 ReST 强化学习进行联合优化，经过多轮迭代自博弈提升游戏能力。

### 关键设计

1. **Prover（证明者）**:

    - 功能：接收问题后给出带有思维链的解答，随后接收意图未知的批评并决定是否修改答案
    - 核心思路：Prover 需要在不知批评者意图的情况下做出理性判断——面对误导性批评时保持正确答案不变，面对建设性反馈时修正错误
    - 设计动机：这种"辨别"能力正是 LLM 理解推理过程的核心体现。Prover 的胜利条件有两种：(1) 首次就正确且成功抵御误导，获得更高奖励（含额外奖励 η）；(2) 首次错误但在 Helpful Critic 帮助下修正成功

2. **Helpful Critic（建设性批评者）**:

    - 功能：接收问题和 Prover 的错误答案，指出推理中的错误但不直接给出正确答案，引导 Prover 自己修正
    - 核心思路：模拟真实学术讨论场景，批评者可自由选择批评的粒度，以自然语言形式呈现
    - 设计动机：与 Prover 形成合作关系——Helpful Critic 的奖励 $R_\mu$ 定义为成功引导 Prover 从错误修正到正确的概率

3. **Misleading Critic（误导性批评者）**:

    - 功能：接收问题和 Prover 的正确答案，捏造一个不存在的错误来误导 Prover 修改答案
    - 核心思路：通过对抗性训练迫使 Prover 深入理解自己的推理过程，不被假反馈动摇
    - 设计动机：与 Prover 形成对抗关系——Misleading Critic 的奖励 $R_\rho$ 定义为成功欺骗 Prover 改变正确答案的概率。随着训练推进，误导者越来越强，Prover 必须更深入理解推理才能胜出

### 损失函数 / 训练策略

**奖励函数设计**：
- Prover 总奖励包含两项：$R_\pi = \mathbb{E}[\mathbb{1}_{\text{correct}}(z,y)(\mathbb{1}_{\text{correct}}(z',y) + \eta) + (1 - \mathbb{1}_{\text{correct}}(z,y))\mathbb{1}_{\text{correct}}(z',y)]$
- 首次正确且抵御误导的奖励 > 首次错误但被帮助纠正的奖励（通过超参数 η 控制）

**训练方法（ReST）**：
- 采用 Reinforced Self-Training，通过阈值筛选高奖励样本进行语言建模损失训练
- 阈值设置：$\tau_\pi = 0.5$，$\tau_\rho = 0.75$（误导者要求更高成功率），$\tau_\mu = 0.5$
- 离线学习方案，每轮先收集自博弈数据，累积到历史数据集，从初始模型重新训练
- 数据均衡：三类样本（首次正确、抵御误导、修正错误）各保持 10000 条

**多轮迭代**：通常进行 2 轮自博弈训练。第 2 轮因误导者经 RL 训练后攻击更强，Prover 获得更大提升。

## 实验关键数据

### 主实验（数学推理）

| 数据集 | 指标 | LLaMA-3.1-8B-Instruct | CDG-2 | 提升 |
|--------|------|----------------------|-------|------|
| GSM8K | P@1 | 85.3 | 86.8 | +1.5 |
| GSM8K | M@32 | 93.0 | 93.1 | +0.1 |
| MATH500 | P@1 | 49.4 | 51.7 | +2.3 |
| MATH500 | M@32 | 63.4 | 66.0 | +2.6 |
| Qwen2.5-1.5B(MATH500) | P@1 | 55.4 | 57.6 | +2.2 |

### 逐步错误检测

| 数据集 | 指标 | 原始模型 | CDG | 提升 |
|--------|------|---------|-----|------|
| GSM8K | F1 / Acc | 74.0 / 64.4 | 76.9 / 69.3 | +2.9 / +4.9 |
| MATH500 | F1 / Acc | 64.4 / 55.4 | 71.4 / 67.5 | +7.0 / +12.1 |

### 消融实验

| 配置 | 数学(GSM8K) | 错误检测(MATH) | 自我纠错(MATH) | 长链推理 | 说明 |
|------|------------|---------------|---------------|---------|------|
| CDG（完整） | 86.8 | 71.4 | +1.4 | 29.7 | 最佳 |
| CDG w/o Helpful Critic | 86.2 | 69.7 | -3.0 | 28.3 | 自我纠错严重下降 |
| CDG w/o Misleading Critic | 84.9 | 68.9 | -0.5 | 29.4 | 推理能力下降 |
| Expert Iteration | 87.2 | 67.4 | +0.8 | 22.8 | 长链推理很差 |
| Step-DPO | 84.6 | 58.2 | -2.1 | 27.6 | 依赖 GPT-4o 标注 |

### RL 方法对比

| 方法 | GSM8K P@1 | MATH500 P@1 | MATH500 M@32 |
|------|-----------|-------------|--------------|
| CDG-ReST | 86.8 | 51.7 | 66.0 |
| CDG-DPO | 83.3 | 46.0 | 54.8 |
| CDG-PPO | 86.6 | 51.6 | 62.6 |

### 关键发现
- CDG 在更难的 MATH500 数据集上提升更大，说明方法对复杂问题改善更显著
- 第二轮训练提升大于第一轮，因为误导者经 RL 训练后攻击更强
- ReST 稳定性最好，DPO 甚至低于基线，PPO 对超参数敏感
- 长链推理中，CDG 训练后的模型蒸馏效果比原始模型高 3-5 个百分点
- 自我纠错实验：CDG 将 GSM8K 上错误修改正确答案的概率降低了一半以上

## 亮点与洞察
- 首次在完全对齐的指令模型上通过自博弈语言游戏提升推理能力，不需要更强模型做教师
- 奖励完全来自游戏规则（答案正确性），无需人类标注，具有天然可扩展性
- 批评者可自由选择批评粒度，实现了灵活的自然语言步级监督
- CDG 训练可作为"预处理"步骤提升后续蒸馏的效果，具有通用价值
- 自博弈训练中 Prover 和 Misleading Critic 的军备竞赛动态有趣且符合预期

## 局限与展望
- 目前仅在数学推理领域验证，代码、逻辑等其他推理任务有待推广
- 仅在 8B 和 1.5B 模型上实验，更大模型效果未知
- 第一轮自博弈提升有限，对初始模型的博弈能力有一定要求
- 预训练模型需要额外的模仿学习步骤建立基本游戏能力
- 可探索在线 RL（如 PPO）替代离线 ReST 以获得更好的训练效率

## 相关工作与启发
- 与 SPAG（Adversarial Taboo Game）最为相似，但 SPAG 在预训练模型上激活潜在能力，本文在对齐模型上进一步提升
- 启发于 AlphaGo Zero 的自博弈范式，将博弈从棋盘游戏扩展到自然语言推理
- 可视为 PRM/ORM 之外的第三种推理监督范式：基于博弈的自监督
- Prover-Verifier Games 也是类似思路，但本文的三角色设计（合作 + 对抗）更丰富

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration](soft_reasoning_navigating_solution_spaces_in_large_language_models_through_contr.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](../../ACL2025/llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ACL 2025\] FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](../../ACL2025/llm_reasoning/finereason_evaluating_and_improving_llms_deliberate_reasoning_through_reflective.md)
- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)

</div>

<!-- RELATED:END -->

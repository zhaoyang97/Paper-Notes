---
title: >-
  [论文解读] Incentivizing LLM Reasoning via Reinforcement Learning with Functional Monte Carlo Tree Search
description: >-
  [ICLR 2026][Reasoning][功能性 token] RFTT 把 `<analyze>` `<verify>` `<refine>` 等一组可学习的"功能性 token"直接塞进模型词表，先用功能性提示引导的 MCTS 自造带标注的 SFT 数据热身，再让模型在 RL 阶段直接采样功能性 token 做树搜索探索，使 7B/8B 小模型无需任何提示就学会人类式多步推理。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "功能性 token"
  - "蒙特卡洛树搜索"
  - "强化学习"
  - "自我提升"
  - "learn-to-reason"
---

# Incentivizing LLM Reasoning via Reinforcement Learning with Functional Monte Carlo Tree Search

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lHbhzxiVI9](https://openreview.net/forum?id=lHbhzxiVI9)  
**代码**: [https://github.com/sastpg/RFTT](https://github.com/sastpg/RFTT)  
**领域**: LLM 推理 / 强化微调  
**关键词**: 功能性 token、蒙特卡洛树搜索、强化学习、自我提升、learn-to-reason  

## 一句话总结
RFTT 把 `<analyze>` `<verify>` `<refine>` 等一组可学习的"功能性 token"直接塞进模型词表，先用功能性提示引导的 MCTS 自造带标注的 SFT 数据热身，再让模型在 RL 阶段直接采样功能性 token 做树搜索探索，使 7B/8B 小模型无需任何提示就学会人类式多步推理。

## 研究背景与动机
- **领域现状**：DeepSeek-R1 等工作证明 RLVR（可验证奖励的 RL）能自发激发 LLM 的复杂推理与"aha moment"，而树搜索类方法（ToT、GoT、MCTS、rStar）则通过结构化探索去寻找更优推理路径。
- **现有痛点**：一方面 learn-to-reason 高度依赖高质量 CoT 数据，靠更强模型或人工标注既贵又难扩展，对 7B/8B 小模型尤其吃力；另一方面有研究指出纯 RL 并不会引入全新推理能力，只是提高采样到模型已知正确答案的概率，且会**牺牲推理多样性、压缩探索空间**。
- **核心矛盾**：有效的学会推理同时需要"足够强的初始推理能力"和"持续的探索多样性"——初始能力不足则自我博弈拿不到有用反馈，探索多样性不足又会过早收敛到次优推理模式。
- **本文目标**：让小参数 LLM 在不依赖人工 CoT、不依赖外部提示约束的前提下，通过自我训练同时获得初始推理能力与探索多样性。
- **核心 idea**：**功能性 token 内化（functional token internalization）**——不像 rStar 那样把功能性提示当成推理期的外部约束，而是把它们当作新的特殊 token 嵌入词表，让模型在训练中学会"自己选择该用哪种推理行为"，把 prompt-guided 推理升级成 token-guided 推理。

## 方法详解

### 整体框架
RFTT（Reinforced Functional Token Tuning）把复杂推理建模为以问题为根、以"推理行为"为边、以子步骤为节点的树搜索，动作空间是 8 个人类式功能性 token（`<clarify>` `<analysis>` `<subquestion>` `<next_step>` `<direct_answer>` `<verify>` `<refine>` `<output>`）。整个流程分两阶段：先用功能性提示引导的 MCTS 自造带 token 标注的数据做 SFT 热身，再在 RL 阶段让模型直接采样功能性 token 做树搜索并强化高价值路径。

```mermaid
flowchart LR
    Q[问题 x] --> MCTS[功能性提示引导 MCTS<br/>采多条轨迹]
    MCTS --> CV[交叉验证<br/>生成纠错节点]
    CV --> BM[分支合并<br/>拼成含自验证/自纠正的路径]
    BM --> SFT[SFT 热身<br/>学会用功能性 token]
    SFT --> RL[RL 阶段<br/>采样 token 做树搜索]
    RL --> REINF[Reinforce++ 强化高价值路径]
```

### 关键设计

**1. 功能性 MCTS 数据生成：把对错轨迹缝成会自我纠错的样本。** SFT 数据不是简单留下正确链，而是刻意把一条正确轨迹和一条错误轨迹"对照"起来教模型反思。具体先用过程奖励模型挑出平均奖励最高的正确轨迹 $\tau_c = \arg\max_{\tau_i \in T_c} \bar{R}(\tau_i)$，再从与它有交集的错误轨迹里挑奖励最低的 $\tau_w = \arg\min_{\tau_i \in T_w} \bar{R}(\tau_i)$；把二者重叠节点记为 $\tau^+ = \tau_c \cap \tau_w$，然后用提示 "Compared to similar correct steps before, where does the current step go wrong?" 生成一个**交叉验证节点** $s_v$，专门指出错误链 $\tau_w^-$ 错在哪。最后做**分支合并** $\tau_f = \tau^+ \cup \tau_w^- \cup \{s_v\} \cup \tau_c^+$，并用功能性 token 把各步串接成可训练路径 $D_{\text{SFT}} = \otimes_{s_i \in \tau_f}(\langle a_i\rangle s_i \langle/a_i\rangle)$。这样合成的数据天然带有"先走错→自我验证→纠正回正确"的结构，让模型在 SFT 阶段就内化自验证与自纠正行为。

**2. 从 prompt-guided 到 token-guided 的转换：把功能性 token 当词表里的动作来采样。** SFT 把 8 个功能性 token 作为新的特殊 token 直接嵌入词表，训练后模型可以像生成普通 token 一样显式地选择某个功能性 token，再生成该 token 对应的具体推理步骤。这一步是 RFTT 与 rStar 的本质差别——rStar 的功能性提示只是推理期外部约束，而这里推理行为变成了模型内部可学习、可被 RL 优化的决策，从而把"靠提示拼接"的外部树搜索升级成"靠采样 token"的内生树搜索，显著提升探索效率。

**3. 功能性树搜索 RL：用 UCT 选 token、规则奖励 + KL 约束驱动 Reinforce++。** RL 阶段类似 AlphaZero：若存在未探索的功能性 token 就按对数似然 $\arg\max_{a\in U(s_t)}\pi_\theta(a\mid s_{0:t})$ 选取，否则按 UCT 分数 $\text{UCT}(s_t,a)=\frac{Q(s_t,a)}{N(s_t,a)} + c\cdot\sqrt{\frac{\ln N(s_t)}{N(s_t,a)}}$ 在利用与探索间权衡。奖励由规则化答案抽取器 $\text{ANS}(\cdot)$ 给出——答对得 1、答错但有答案得 0.1、无答案得过程奖励 $\sigma$（仅用结果奖励时 $\sigma=0$，否则由 PRM 给）——并减去对 SFT 参考模型的 KL 惩罚 $R_t = \text{RM}(\cdot) - \beta\cdot\text{KL}(t)$。策略用 Reinforce++ 的裁剪目标 $L_{\text{RL}}(\theta)=-\mathbb{E}_t[\min(r_t\hat{A}_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)]$ 优化。

**4. 为何更强：高熵探索 + 区分性优势归因。** 不同于 GRPO 的独立 rollout，RFTT 在每个节点按功能性 token 概率做树状分支——某节点 token 分布越集中（低熵）越抑制分支，越分散（高熵）越鼓励分支，从而**把探索资源自适应分配到熵更高、更不确定的推理区域**。同时树搜索产生大量"共享前缀、分叉后续"的轨迹：共享 token 拿到整组聚合的优势信号，分叉 token 拿到各自路径特有的优势，使模型能在关键推理步上精确学习不同推理行为的差异。

## 实验关键数据

### 主实验表格
训练只用 MATH 训练集（约 1k 条带功能性 token 的 SFT 数据 + RL），评测覆盖 5 个数学基准。Pass@1 准确率（部分）：

| 模型 | 方法 | MATH-500 | GSM8K | Olympiad | AMC | 平均 |
|------|------|----------|-------|----------|-----|------|
| Qwen-3-4B-Base | Zero-shot CoT | 58.6 | 80.6 | 31.4 | 50.0 | 59.90 |
| | GRPO | 75.6 | 92.7 | 42.9 | 67.5 | 73.42 |
| | TreeRL | 82.2 | 95.3 | 46.5 | 70.0 | 77.36 |
| | **RFTT** | **83.4** | **96.1** | **48.1** | **75.0** | **79.46** ↑19.56 |
| Qwen-2.5-7B-Instruct | Zero-shot CoT | 72.0 | 91.1 | 35.1 | 45.0 | 66.34 |
| | TreeRL | 78.4 | 94.8 | 39.6 | 62.5 | 73.52 |
| | **RFTT** | **79.8** | **95.2** | **40.3** | **70.0** | **75.66** ↑9.32 |
| LLaMA-3.1-8B-Instruct | Zero-shot CoT | 50.6 | 84.5 | 18.6 | 17.5 | 49.88 |
| | TreeRL | 57.8 | 92.6 | 26.7 | 47.5 | 62.14 |
| | **RFTT** | **60.2** | 91.9 | **29.8** | **55.0** | **64.88** ↑15.0 |

Qwen-2.5-7B-Instruct + RFTT 在 MATH-500 上 79.8% 已逼近 Qwen-2.5-14B-Instruct 的 80%；树搜索效率对比中，RFTT 在 MATH-500 取 72.0% 仅需 131s/题，而 rStar 61.0%/526s、LLaMA-Berry 69.4%/674s，**更准且快约 4–5 倍**。

### 消融实验表格
组件消融（Qwen-2.5-7B-Instruct，平均）：

| 设置 | MATH-500 | AMC | 平均 |
|------|----------|-----|------|
| RFTT w/o SFT Warmup | 74.8 | 52.5 | 69.40 |
| RFTT w/o MCTS（随机采样） | 75.2 | 62.5 | 72.34 |
| RFTT w/o PRM（仅结果奖励） | 77.2 | 67.5 | 73.92 |
| **RFTT（完整）** | **79.8** | **70.0** | **75.66** |

功能性 token 掩码消融显示 `<verify>`(a6)、`<refine>`(a7)、`<next_step>`(a4) 最关键，去掉后准确率从 79.8% 分别掉到 72.8%/72.6%/73.4%。

### 关键发现
- **测试时计算可扩展**：随 rollout 数增加准确率持续上升，MATH-500 上 8 个 rollout、AMC 上 20 个 rollout 即超过 o1-preview。
- **强泛化**：只在数学上训练，却在 MMLU-Pro、GPQA、CommonsenseQA、FOLIO、TableBench 等域外基准上同样提升；用 DeepSeek-R1 在 MMLU-Pro 上抽样，约 98.4% 的推理步可映射到这 8 个功能性 token，说明该动作空间紧凑且表达力足。

## 亮点与洞察
- 把"推理行为"从外部提示约束变成词表里可学习、可被 RL 直接采样优化的 token，是 prompt-guided → token-guided 的范式切换，思路干净且可复用。
- 用"正确链 × 错误链"交叉验证 + 分支合并合成自带自验证/自纠正结构的 SFT 数据，巧妙绕开了人工标注 CoT 的成本。
- 树搜索按 token 熵自适应分配探索预算 + 共享/分叉 token 的区分性优势归因，给"为什么比 GRPO 探索更有效"提供了清晰直觉。

## 局限与展望
- 训练只用 MATH 单一数据源、模型规模都在 10B 以下，更大模型与更多领域的可扩展性尚待验证。
- 数据生成阶段依赖 math-shepherd PRM 做过程奖励，且 8 个功能性 token 的设计与对应提示由人工经验设定，迁移到非数学/非 STEM 任务时是否需重新设计 token 集仍是开放问题。
- MCTS 采样虽比 rStar/LLaMA-Berry 快，但相比纯 GRPO 的独立 rollout 仍引入了树搜索与 PRM 打分的额外开销。

## 相关工作与启发
- **learn-to-reason**：长 CoT 合成 SFT、偏好对优化（DPO 系）、RLVR（DeepSeek-R1）——RFTT 用功能性树搜索缓解 RLVR 压缩多样性的问题。
- **树搜索推理**：ToT、GoT、MCTS、rStar、ResT-MCTS*、LLaMA-Berry、TreeRL；与 rStar 最相关但区别在"功能性提示只在推理期当外部约束 vs 功能性 token 在训练期被内化"。
- **启发**：把"控制性/结构性信号"作为可学习特殊 token 嵌入词表，再让 RL 去优化其使用策略，这条路或可推广到工具调用、检索决策、多智能体角色切换等需要"显式选择行为类型"的场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把功能性提示内化为词表可学习 token 并用功能性树搜索 RL 优化，是对 rStar/RLVR 的清晰且有说服力的范式推进。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 3 个 backbone、5 个数学 + 6 个域外基准、组件/token/效率多维消融，但训练数据单一、未上更大模型。
- **写作质量**: ⭐⭐⭐⭐ 公式与图表清晰，两阶段动机与"高熵探索/区分性优势归因"的分析到位。
- **价值**: ⭐⭐⭐⭐ 让 7B/8B 小模型无提示自我训练即逼近 14B/o1-preview，且代码数据开源，对低成本推理增强有实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RL of Thoughts: Navigating LLM Reasoning with Inference-Time Reinforcement Learning](rl_of_thoughts_navigating_llm_reasoning_with_inference-time_reinforcement_learni.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](../../AAAI2026/llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)
- [\[ICLR 2026\] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning](curriculum_reinforcement_learning_from_easy_to_hard_tasks_improves_llm_reasoning.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)
- [\[ICLR 2026\] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](emergent_hierarchical_reasoning_in_llms_through_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] FlowRL: Matching Reward Distributions for LLM Reasoning
description: >-
  [ICLR 2026][Reasoning][奖励分布匹配] FlowRL 把 LLM 推理 RL 从"最大化标量奖励"改成"匹配完整奖励分布"——用可学习配分函数把标量奖励归一化成目标分布，再借 GFlowNets 的轨迹平衡损失最小化策略与目标分布的反向 KL，从而保留多个有效推理模式、缓解模式坍塌，在数学上比 GRPO/PPO 平均高 10.0%/5.1%。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "奖励分布匹配"
  - "GFlowNets"
  - "轨迹平衡"
  - "模式坍塌"
  - "探索多样性"
  - "RLHF"
---

# FlowRL: Matching Reward Distributions for LLM Reasoning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lObnTKbm9U](https://openreview.net/forum?id=lObnTKbm9U)  
**代码**: [https://github.com/Xuekai-Zhu/FlowRL](https://github.com/Xuekai-Zhu/FlowRL)  
**领域**: LLM 推理 / 强化学习  
**关键词**: 奖励分布匹配, GFlowNets, 轨迹平衡, 模式坍塌, 探索多样性, RLHF  

## 一句话总结
FlowRL 把 LLM 推理 RL 从"最大化标量奖励"改成"匹配完整奖励分布"——用可学习配分函数把标量奖励归一化成目标分布，再借 GFlowNets 的轨迹平衡损失最小化策略与目标分布的反向 KL，从而保留多个有效推理模式、缓解模式坍塌，在数学上比 GRPO/PPO 平均高 10.0%/5.1%。

## 研究背景与动机
**领域现状**：从 REINFORCE → PPO → GRPO，主流 LLM 推理 RL 算法的目标始终是"最大化期望奖励"。GRPO 更是通过组内对比省掉了价值网络，成为 R1 类强推理模型的标配。

**现有痛点**：奖励最大化天然会**过拟合奖励分布的主导模式**（dominant mode）。在长链式思维（long CoT）推理里，一道题往往有多条合法解法路径，但 reward-maximizing 方法会把概率质量全部压到某一个高奖励峰上，导致**模式坍塌**——生成的推理路径高度同质、重复套用同一招式（如反复用 AM-GM 不等式陷入死循环），对低频但同样正确的解法泛化能力差。论文用 KL 散度量化了这一现象：reward-maximizing 方法相对目标分布的 KL 高达 8.68，而分布匹配只有 0.11。

**核心矛盾**：有效泛化需要**覆盖多个解法模式**（mode coverage），但标量奖励信号 + 最大化目标本质上是**收敛到单峰**的。现有缓解手段（调 clip ratio、entropy-based advantage shaping、高熵 token 选择性提升）都是在最大化框架内"打补丁"地隐式增加多样性，没有从目标函数层面解决问题。

**本文目标**：从根本上把优化目标从"奖励最大化"换成"奖励分布匹配"，让策略按奖励比例采样多样的高奖励轨迹。

**核心 idea**：**[分布匹配]** 引入可学习配分函数 $Z_\phi(x)$ 把标量奖励 $r(x,y)$ 归一化成目标分布 $\tilde\pi(y|x)=\exp(\beta r(x,y))/Z_\phi(x)$，**[流平衡]** 再证明"最小化策略与该目标的反向 KL"在期望梯度意义上等价于 GFlowNets 的轨迹平衡损失，从而把难解的 KL 优化转成稳定的平方损失。

## 方法详解

### 整体框架
FlowRL 的核心是把 RL 目标重写为反向 KL 散度并接到 GFlowNets 的轨迹平衡（Trajectory Balance）损失上：先用可学习配分函数把标量奖励变成归一化目标分布，再通过最小化平方形式的轨迹平衡损失逼近"按奖励比例采样"的策略；为适配长 CoT 训练，额外加上长度归一化（治梯度爆炸）和重要性采样（治采样错配），最终得到可直接插入 veRL 框架的 FlowRL 损失。

```mermaid
flowchart TD
    A[问题 x] --> B[策略 πθ 采样一组 G 条 CoT]
    B --> C[标量结果奖励 r + 组内归一化 r̂]
    A --> D[配分函数 Zφ: 3层MLP<br/>输入=隐状态均值, 输出=标量]
    C --> E[目标分布 exp βr·πref / Zφ]
    D --> E
    E --> F[反向KL = 轨迹平衡损失<br/>logZφ + logπθ − βr̂ − logπref]
    F --> G[长度归一化 1/|y|·logπ<br/>治梯度爆炸]
    F --> H[重要性采样 w=clip detach<br/>治off-policy错配]
    G --> I[FlowRL 损失]
    H --> I
    I --> B
```

### 关键设计

**1. 配分函数把标量奖励变成可匹配的目标分布：** 长 CoT 推理里能拿到的监督只有一个标量奖励，而枚举所有合法轨迹去还原真实奖励分布在计算上不可行。FlowRL 借鉴能量模型，引入一个可学习配分函数 $Z_\phi(x)$ 来归一化奖励，把优化写成最小化反向 KL：$\min_\theta D_{KL}\!\big(\pi_\theta(y|x)\,\|\,\tfrac{\exp(\beta r(x,y))}{Z_\phi(x)}\big)$，其最优解是 $\pi_\theta(y|x)\propto\exp(\beta r(x,y))$，即策略按奖励指数比例采样而非坍塌到单峰。这里用反向 KL 是因为只能从策略采样、无法直接从目标分布采样。$Z_\phi$ 用一个 3 层 MLP 参数化，输入是语言模型编码 $x$ 后隐状态的均值，输出一个标量。

**2. 轨迹平衡损失把难解的 KL 优化转成稳定平方损失：** 论文给出 Proposition 1——在期望梯度意义上，最小化 Eq.2 的 KL 目标**等价于**最小化 GFlowNets 的轨迹平衡损失 $\big(\log Z_\phi(x)+\log\pi_\theta(y|x)-\beta r(x,y)\big)^2$。这一等价是关键桥梁：它把生成式建模（GFlowNets）和策略优化连了起来，让原本要显式计算难解配分函数、又难以稳定优化的 KL，变成一个**平方损失**——$Z_\phi$ 当作可学习参数直接梯度下降，无需求积分。为引入参考模型作为先验约束，论文把奖励项从 $\exp(\beta r)$ 改成 $\exp(\beta r)\cdot\pi_{ref}(y|x)$，并对组内奖励做归一化 $\hat r_i=(r_i-\text{mean}(r))/\text{std}(r)$，代入后得到 $\min_\theta(\log Z_\phi(x)+\log\pi_\theta(y|x)-\beta\hat r_i-\log\pi_{ref}(y|x))^2$。

**3. 长度归一化治长 CoT 的梯度爆炸：** 轨迹平衡是序列级目标，而 $\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t|y_{<t},x)$ 是逐 token 求和——当 CoT 长达 8K token 时，这一项会随序列长度线性放大，导致梯度范数爆炸、更新不稳定（这是短轨迹的传统 GFlowNets 工作没遇到过的新问题）。FlowRL 的解法是把 log 概率项按序列长度归一化，用 $\tfrac{1}{|y|}\log\pi_\theta(y|x)$ 替代原始求和，从奖励塑形（reward shaping）角度平衡长短序列的贡献，稳定学习信号。

**4. 重要性采样治 off-policy 采样错配：** PPO/GRPO 为了数据效率会复用旧策略 $\pi_{\theta_{old}}$ 采的轨迹做 micro-batch 更新，但 KL-轨迹平衡目标假设完全 on-policy。FlowRL 引入 PPO 式重要性采样，用比率 $w=\pi_\theta(y|x)/\pi_{old}(y|x)$ 重加权陈旧轨迹；由于目标是优化轨迹平衡而非期望回报，它**对当前策略的梯度做 detach**（$w=\text{detach}[\pi_\theta]/\pi_{old}$）以防策略漂移过度，并加 PPO 式 clipping 把权重 bound 住。最终 FlowRL 损失为 $L=w\cdot\big(\log Z_\phi(x)+\tfrac{1}{|y|}\log\pi_\theta(y|x)-\beta\hat r(x,y)-\tfrac{1}{|y|}\log\pi_{ref}(y|x)\big)^2$，可直接接入 veRL 训练管线。

## 实验关键数据

### 主实验表格
数学推理（Avg@16，6 个 benchmark 平均），Qwen2.5-Base，max response 8K：

| 模型 | 方法 | AIME24 | AIME25 | MATH500 | Olympiad | 平均 |
|------|------|--------|--------|---------|----------|------|
| 32B | PPO | 26.87 | 20.41 | 69.17 | 37.90 | 43.25 |
| 32B | GRPO | 23.12 | 14.58 | 61.60 | 34.94 | 38.34 |
| 32B | **FlowRL** | 23.95 | **21.87** | **80.75** | **51.83** | **48.39** |
| 7B | GRPO | 13.54 | 9.79 | 57.05 | 26.88 | 32.48 |
| 7B | PPO | 9.38 | 7.29 | 57.98 | 27.25 | 31.98 |
| 7B | **FlowRL** | **15.41** | **10.83** | **66.96** | **34.61** | **35.63** |

32B 上 FlowRL 平均 48.39%，比 PPO 高 5.1%、比 GRPO 高 10.1%；在 MATH-500、Olympiad 等难题上提升尤为明显。

代码推理（DeepSeek-R1-Distill-Qwen-7B）：

| 方法 | LiveCodeBench Avg@16 | CodeForces Rating | Percentile | HumanEval+ |
|------|----------------------|-------------------|-----------|-----------|
| PPO | 35.10 | 1403.07 | 73.7% | 82.32 |
| GRPO | 32.75 | 1313.82 | 67.1% | 80.13 |
| **FlowRL** | **37.43** | **1549.47** | **83.3%** | **83.28** |

代码三个 benchmark 上 FlowRL 全面领先。

### 消融实验表格
Qwen2.5-7B，数学 6 benchmark 平均（Avg）：

| 方法 | 平均 |
|------|------|
| **FlowRL** | **35.63** |
| w/o 重要性采样 | 26.71 |
| Zhang et al.(2025a) 联合损失 | 33.67 |

β 超参消融：β=5→31.34，β=10→34.41，**β=15→35.63（最优）**，β=30→35.09。

### 关键发现
- **重要性采样是命门**：去掉后平均从 35.63% 掉到 26.71%（−8.92），印证修正 rollout 与策略间分布错配的关键作用；用轨迹级比率比 Zhang et al. 的 GFlowNets+PPO 联合损失更合适。
- **多样性近乎翻倍**：GPT-4o-mini 评判 AIME24/25 rollout 的多样性分数，FlowRL 达 2.28，远超 PPO（1.31）、GRPO（1.23）、R++（1.11），说明它生成的是"质上不同的解法"而非同一策略的微小变体。
- **Case study 直观对比**：同一道 AIME 题，GRPO 反复套 AM-GM 不等式陷入恒等式死循环、得出 a=b=c 的矛盾结论而失败；FlowRL 则做出 a=b 的对称假设、化为三次方程 $a^3-27a+46=0$、用有理根测试求解成功。

## 亮点与洞察
- **目标函数层面的范式转变**：不是在 reward-maximizing 框架里打补丁加多样性，而是直接把目标换成分布匹配，从源头解决模式坍塌——这比 entropy bonus、clip 调整等隐式手段更"治本"。
- **理论桥梁优雅**：Proposition 1 证明"反向 KL 最小化 ⟺ 轨迹平衡损失"，把 GFlowNets 的生成式建模和 RL 策略优化统一起来，并把难算的配分函数变成可学习参数 + 稳定平方损失，工程上可落地。
- **直面长 CoT 的两个真实工程坑**：长度归一化治梯度爆炸、重要性采样治 off-policy 错配——这两点正是把短轨迹 GFlowNets 搬到 8K token 长 CoT 时会踩的雷，消融也证明缺一不可。
- **多样性—泛化的因果链清晰**：多样性翻倍 → 难 benchmark（MATH-500/Olympiad）提升最大，把"探索多样性"和"泛化能力"用实验串了起来。

## 局限与展望
- **只用结果奖励（outcome reward）**：未探索过程奖励或更细粒度信号下分布匹配的表现，长 CoT 中间步骤的奖励塑形仍是开放问题。
- **配分函数 $Z_\phi$ 的设计较朴素**：用隐状态均值喂 3 层 MLP 输出标量，对不同难度/不同模式的题目是否需要更强的条件建模未深入讨论。
- **规模与任务范围**：实验止于 7B/32B、数学与代码两域、8K response，更大模型、更长上下文、开放域推理上的可扩展性待验证。
- **β 需调**：分布"温度" β 对结果敏感（β=15 最优），跨任务/跨规模是否需要重新搜参增加了使用成本。

## 相关工作与启发
- **GFlowNets**（Bengio et al. 2023）：本文方法论根基，把"按奖励比例采样组合对象"的流平衡思想引入 LLM RL；相比短轨迹的原始 GFlowNets，本文贡献在于解决长序列下的稳定性。
- **奖励最大化 RL**（PPO/GRPO/REINFORCE++）：本文的对照与批判对象，揭示其模式坍塌的本质缺陷。
- **熵正则化 / 高熵 token 提升 / DAPO 调 clip**：同样为缓解多样性退化，但都在最大化框架内做隐式调整，本文从目标函数层面替代之。
- **启发**：当任务存在"多条有效解法路径"且泛化依赖模式覆盖（多步推理、程序合成、分子/图生成）时，"匹配奖励分布"可能比"最大化奖励"更合适；轨迹平衡作为 KL 的稳定代理损失，是把生成式建模目标接入现有 RL 管线的一条通用路径。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 LLM 推理 RL 从奖励最大化重构为分布匹配，并用 Proposition 1 严格桥接 GFlowNets 轨迹平衡与反向 KL，是目标函数层面的原创贡献。
- 实验充分度: ⭐⭐⭐⭐ 数学+代码双域、7B/32B 双规模、三个强 baseline、多样性量化与 case study 齐全；但局限于结果奖励、8K 长度、单一模型族（Qwen/DeepSeek-Distill）。
- 写作质量: ⭐⭐⭐⭐ 动机—理论—工程改进—实验链条清晰，KL=0.11 vs 8.68、多样性翻倍等数字很有说服力，公式推导完整。
- 价值: ⭐⭐⭐⭐⭐ 直击长 CoT RL 的模式坍塌痛点，提供可直接接入 veRL 的开源方法与显著增益，对探索导向的推理 RL 有较强借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning](linking_process_to_outcome_conditional_reward_modeling_for_llm_reasoning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] PEAR: Phase Entropy Aware Reward for Efficient Reasoning](pear_phase_entropy_aware_reward_for_efficient_reasoning.md)
- [\[ICLR 2026\] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)
- [\[ICML 2026\] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](../../ICML2026/llm_reasoning/powerflow_unlocking_the_dual_nature_of_llms_via_principled_distribution_matching.md)

</div>

<!-- RELATED:END -->

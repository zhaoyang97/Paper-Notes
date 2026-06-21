---
title: >-
  [论文解读] GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning
description: >-
  [ICLR 2026][Reasoning][Policy Gradient] GPG（Group Policy Gradient）回归最朴素的策略梯度，直接优化 RL 原始目标——砍掉 critic、reference 模型、KL 约束和 surrogate loss，只保留组内均值归一化加一个梯度去偏修正，就在数学与多模态推理任务上稳定超过 GRPO。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Policy Gradient"
  - "GRPO"
  - "推理强化学习"
  - "梯度估计偏差"
  - "RLHF 简化"
---

# GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=inccdtfx8x](https://openreview.net/forum?id=inccdtfx8x)  
**代码**: [https://github.com/AMAP-ML/GPG](https://github.com/AMAP-ML/GPG)  
**领域**: LLM 推理 / 强化学习后训练  
**关键词**: Policy Gradient, GRPO, 推理强化学习, 梯度估计偏差, RLHF 简化  

## 一句话总结
GPG（Group Policy Gradient）回归最朴素的策略梯度，直接优化 RL 原始目标——砍掉 critic、reference 模型、KL 约束和 surrogate loss，只保留组内均值归一化加一个梯度去偏修正，就在数学与多模态推理任务上稳定超过 GRPO。

## 研究背景与动机
**领域现状**：RL 后训练（RFT）是 OpenAI o1、DeepSeek R1 等推理模型的核心引擎，主流做法是 PPO 和 GRPO。PPO 需要同时维护 critic 与 reference 模型，资源开销巨大；GRPO 去掉了 critic，用组内归一化奖励来估计优势，但仍保留了 reference 模型、KL 约束和裁剪后的 surrogate loss。

**现有痛点**：业界一直在做"PPO 减法"（ReMax 去 critic、GRPO 用组归一化、Dr. GRPO 研究奖励/loss 归一化细节），但这些方法要么仍背着冗余组件，要么虽然指出了优势函数里的奖励偏差却没真正带来明显提升（论文复现发现 Dr. GRPO 并未显著超过 GRPO）。

**核心矛盾**：PPO 当年是为 Atari 这类需要从零学视觉表征+控制策略的任务设计的；但在 LLM 时代，策略本身就是一个已经经过预训练和 SFT、表征能力很强的 LLM。继续套用为通用 RL 设计的复杂机制（surrogate loss、信赖域约束）在 scalability 上是负担。策略梯度的主要弱点是高方差，而这恰好能用"基线优势估计 + 多轨迹采样"缓解——这两件事 LLM 后训练本就在做。

**本文目标**：构建一个保留最小 RL 组件、直接优化原始目标的精简方法，在不依赖任何辅助技巧的情况下追平甚至超过 GRPO。

**核心 idea**：**回到策略梯度本源**——既然 LLM 后训练天然满足策略梯度降方差的两个条件，就不需要 surrogate loss 和约束，只要把优势估计的奖励偏差去掉、把梯度估计的偏差修正掉，朴素策略梯度就是最强 baseline。

## 方法详解

### 整体框架
GPG 在一个组（group）内对同一问题采样 $G$ 个回答，用组内奖励均值对每个回答做优势归一化，然后直接最小化 $-\log\pi_\theta \cdot \hat{A}$ 形式的策略梯度目标，全程不引入 critic、reference、KL 约束或裁剪。在此最简骨架上叠加两个关键修正：去掉优势函数中隐含的奖励偏差，以及对全对/全错样本造成的梯度估计偏差做缩放校正（AGE），再辅以一个阈值重采样机制控制方差。

```mermaid
flowchart LR
    A[问题 q] --> B[采样 G 个回答 o_i]
    B --> C[规则奖励 R_i<br/>对1.0/错0.0]
    C --> D[组内均值归一化<br/>Â = (r - mean R)/F_norm]
    D --> E[AGE 梯度去偏<br/>×α = B/(B-M)]
    E --> F{有效样本比例<br/>≥ β_th?}
    F -- 否 --> G[累积重采样]
    G --> F
    F -- 是 --> H[loss = -log π · Â · α]
```

### 关键设计

**1. Group Policy Gradient：直接优化原始目标，砍掉一切冗余组件。** GPG 的目标函数（式 5）就是把策略梯度定理里的优势加权对数似然直接写成 loss：$J_{GPG}(\theta)=\mathbb{E}\big[\frac{1}{\sum|o_i|}\sum_i\sum_t -\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\hat{A}_{i,t}\big]$。它不是 PPO/GRPO 那样优化一个带重要性采样比和裁剪的 surrogate，而是直接对原始 RL 目标做梯度上升。优势用组内均值归一化得到 $\hat{A}_{i,t}=\frac{r_i-\text{mean}(\{R_i\}_{i=1}^G)}{F_{norm}}$（式 6）。如表 2 所示，相比 PPO（value+reference+surrogate+约束全有）、GRPO（去 critic 但保留其余三项）、TRPO，GPG 把四项全部置空，是最简形态，既好实现又高效。

**2. 去除奖励偏差：$F_{norm}=1$ 而非组内标准差。** GRPO 取 $F_{norm}=\text{std}\{R(o)\}$，作者指出这本质上是式 2 里关于状态 $s_t$ 的一个函数，会显式引入奖励偏差，让优化目标偏离原始问题。GPG 主张直接解原问题、不加任何 surrogate 或偏置，故令 $F_{norm}=1$。但单纯去掉这一偏置项（43.9%）并不能明显超过 GRPO（43.7%），这与 Dr. GRPO 的观察相反——说明真正的提升来源不在这里，必须配合下一个设计。作者进一步分析发现，GRPO 的 std 归一化（组内 std 在 0.10~0.35 之间）其实暗含了一种类似梯度修正的效果，这正引出 AGE。

**3. AGE 精确梯度估计：对全对/全错样本做 $\alpha=\frac{B}{B-M}$ 缩放。** 在一个 batch 内，若某些组的 $G$ 个回答全对或全错，则组内优势全为 0、对梯度贡献为零，但标准反向传播仍以 $B$ 为分母平均，等于用无效样本稀释了梯度。设 batch 中有 $M$ 个这样的无效样本，真实梯度应为 $\hat{g}=\frac{\sum_{i=M+1}^B g_i}{B-M}=g\cdot\frac{B}{B-M}=\alpha g$（式 7）。于是修正后目标 $\hat{J}_{GPG}(\theta)=\alpha J_{GPG}(\theta)$（式 8）。$\alpha$ 不是常数，随 batch 难度动态变化（实测在 1.5~4.0 之间），自动放大有效信号。加上 AGE 后平均分从 43.9% 跳到 47.8%，这才是性能跃升的真正来源。论文还给出多卡训练下无需额外通信的等价形式（附录证明），避免跨 GPU 收集非零梯度的开销。

**4. 阈值重采样：保证最小有效样本比例以降方差。** AGE 给出无偏估计，但当有效样本比例过低时 $\alpha$ 很大、梯度方差爆炸。GPG 引入阈值 $\beta_{th}=\frac{1}{\alpha_{th}}$，当有效样本占比低于该值时，把有效样本累积进后续重采样 batch，直到占比超过阈值再更新。相比 DAPO 那种"一直重采样到 $M=0$（即 $\alpha=1$）"的做法，GPG 不会被最慢的 worker 拖住、训练更高效，还能根据 batch 表现自动调节 loss 强度。加上 $\beta_{th}=0.6$ 后平均分进一步提升到 48.3%。

## 实验关键数据

### 主实验：消融各组件（Qwen2.5-Math-7B，MATH-lighteval）

| 配置 | Average | AIME24 | MATH-500 | AMC23 |
|------|---------|--------|----------|-------|
| GRPO | 43.7 | 16.7 | 73.4 | 62.5 |
| Dr. GRPO† | 43.7 | 26.7 | 74.6 | 50.0 |
| GPG ($F_{norm}{=}1,\alpha{=}1$) | 43.9 | 23.3 | 76.3 | 52.5 |
| GPG ($F_{norm}{=}\text{std},\alpha{=}1$) | 45.3 | 23.3 | 73.6 | 60.0 |
| GPG ($F_{norm}{=}1,\alpha{=}\tfrac{B}{B-M}$) | 47.8 | 30.0 | 75.0 | 62.5 |
| **GPG (+$\beta_{th}{=}0.6$)** | **48.3** | **30.0** | **76.2** | 62.5 |

### 1.5B 蒸馏模型（五项数学基准 zero-shot pass@1）

| 模型 | Average | AIME24 | MATH-500 | AMC23 |
|------|---------|--------|----------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | 48.9 | 28.8 | 82.8 | 62.9 |
| Open-RS1† | 53.1 | 33.3 | 83.8 | 67.5 |
| **GPG-RS1** | **55.7** | 33.3 | **87.6** | 77.5 |
| **GPG-RS3** | 55.5 | 33.3 | 85.0 | **80.0** |

### 关键发现
- **去偏置不是关键，去梯度偏差才是**：单去掉 std 归一化几乎追平 GRPO，加上 AGE 才有 ~4% 跃升，证明性能来源是梯度估计修正而非奖励归一化。
- **GRPO 的 std 归一化暗含梯度修正**：std 在组内随难度变化（0.10~0.35），等效地起到了部分 $\alpha$ 缩放的作用，这解释了为何它比纯 $F_{norm}{=}1$ 略好。
- **多模态泛化**：在视觉推理（SAT/CV-Bench）、几何推理（GEOQA）、few-shot 分类与 grounding（Flower102/LISA 等）上同样稳超其他 RL 方法，验证方法不依赖任务模态。
- **更省算力**：去掉 reference 模型省一份前向、去掉重采样到底的策略避免被慢 worker 拖累，训练成本低于 GRPO。

## 亮点与洞察
- **"减法"做到极致**：表 2 直观展示 GPG 是唯一把 value/reference/surrogate/约束四项全部清零的方法，把"GRPO 还能更简单吗"这个问题给出了肯定答案。
- **诊断精准**：把 GRPO 的提升来源拆解为"奖励偏差去除"与"梯度偏差修正"两件事，并用实验证明前者无效、后者才是关键，纠正了 Dr. GRPO 的归因。
- **AGE 简单但有理**：$\alpha=\frac{B}{B-M}$ 一行公式，既符合无偏估计的直觉，又给出多卡无通信开销的等价实现，工程落地友好。
- **动态自适应**：$\alpha$ 随 batch 难度自动伸缩，相当于内建的难度自适应学习率，无需额外调参。

## 局限与展望
- **奖励设置偏简单**：实验主要在规则奖励（对 1.0 / 错 0.0）的数学/视觉任务上，论文也坦言中间步骤奖励难以设计而做了简化；在奖励模型连续、噪声大的开放任务上，去掉 std 归一化是否仍稳健有待验证。
- **方差控制依赖阈值**：$\beta_{th}$ 是新引入的超参，重采样累积也可能改变 batch 的数据分布，长程训练下的稳定性边界未充分讨论。
- **去 KL 约束的风险**：完全移除 reference 与 KL 约束在长时训练中可能带来分布漂移/reward hacking，论文未深入分析极端 scaling 下的行为。
- **理论 vs 经验**：AGE 是对全对/全错样本的直觉修正，但缺乏对收敛性、最优性更严格的理论保证。

## 相关工作与启发
- **PPO/GRPO 谱系**：GPG 沿着 PPO→TRPO→Policy Gradient 的脉络做"逆向简化"，主张在 LLM 时代回到策略梯度本源，与"PPO 是 TRPO 的简化、TRPO 又建立在 PG 之上"的历史一脉相承。
- **对 Dr. GRPO 的回应**：同期工作 Dr. GRPO 指出 GRPO 的奖励偏差并倾向生成更多 token，GPG 复现后认为其提升不显著，并把真正原因归到梯度估计偏差，提供了不同视角。
- **与 DAPO 的对比**：DAPO 类方法通过重采样到 $M=0$ 来消除无效样本，GPG 用阈值+AGE 缩放达成同样效果但更高效，是对"动态采样"思路的轻量化替代。
- **启发**：在已有强表征的基座上做 RL，"组件越少越好"可能是普适规律；把性能提升精确归因到单一机制（梯度去偏）比堆叠 trick 更有方法论价值。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是全新算法，但"回归朴素 PG + 精确归因 + AGE 去偏"的组合简洁有力，纠正了同期工作的误判。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 1.5B/7B 多模型、五项数学基准与多模态任务，消融清晰地隔离了每个组件的贡献。
- 写作质量: ⭐⭐⭐⭐ 动机链条（PPO 历史→减法趋势→本源回归）讲得清楚，表 2 的组件对比极具说服力。
- 价值: ⭐⭐⭐⭐ 作为"简单又强"的推理 RL baseline，工程落地成本低、可复现性强，对社区有直接参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models](a_simple_motivation_can_enhance_reinforcement_finetuning_of_large_reasoning_mode.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[ICLR 2026\] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](emergent_hierarchical_reasoning_in_llms_through_reinforcement_learning.md)
- [\[ICLR 2026\] Process-Verified Reinforcement Learning for Theorem Proving via Lean](process-verified_reinforcement_learning_for_theorem_proving_via_lean.md)

</div>

<!-- RELATED:END -->

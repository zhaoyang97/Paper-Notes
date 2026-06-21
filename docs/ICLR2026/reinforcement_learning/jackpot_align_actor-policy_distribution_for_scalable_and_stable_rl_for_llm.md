---
title: >-
  [论文解读] Jackpot: Align Actor-Policy Distribution for Scalable and Stable RL for LLM
description: >-
  [ICLR 2026][强化学习][off-policy RL] Jackpot 用「最优预算拒绝采样（OBRS）」直接把 actor（推理 rollout）分布拉近到 policy（训练）分布，配合 Top-K 概率估计与稳定化的 Jackpot-PPO 损失，让 LLM 的强化学习能在大批量、异步乃至「两个不同模型分别 rollout/训练」的极端 off-policy 设定下保持稳定收敛。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "off-policy RL"
  - "分布失配"
  - "拒绝采样"
  - "OBRS"
  - "重要性采样"
  - "PPO"
  - "大批量训练"
---

# Jackpot: Align Actor-Policy Distribution for Scalable and Stable RL for LLM

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5RATVAQGPx](https://openreview.net/forum?id=5RATVAQGPx)  
**代码**: [https://infini-ai-lab.github.io/jpt_website/](https://infini-ai-lab.github.io/jpt_website/)  
**领域**: 强化学习 / LLM 后训练  
**关键词**: off-policy RL, 分布失配, 拒绝采样, OBRS, 重要性采样, PPO, 大批量训练  

## 一句话总结
Jackpot 用「最优预算拒绝采样（OBRS）」直接把 actor（推理 rollout）分布拉近到 policy（训练）分布，配合 Top-K 概率估计与稳定化的 Jackpot-PPO 损失，让 LLM 的强化学习能在大批量、异步乃至「两个不同模型分别 rollout/训练」的极端 off-policy 设定下保持稳定收敛。

## 研究背景与动机
**领域现状**：RL 已成为 LLM 后训练（对齐、推理、代码）的关键范式，但训练极其昂贵——超过 70% 的时间花在 rollout（采样轨迹算 reward）上。如果允许 actor 与 policy 分布不同，就能解锁大批量训练、异步训练、甚至用量化/稀疏/蒸馏的轻量模型来做 rollout，大幅提升吞吐。

**现有痛点**：一旦 actor 和 policy 分布拉开，现有基于重要性采样（IS）的修正方法都吃亏。截断重要性采样（TIS）用 $\min\!\big(\frac{p_{\text{ref}}(x)}{p_{\text{inf}}(x)}, C\big)$ 来纠偏：截断阈值 $C$ 设小则保守、收敛远逊于 on-policy；设大则在 policy 收敛前就崩。更糟的是，当 actor 漂移过远，很多 actor 高概率采到的 token 在 policy 下概率极低（$p_{\text{inf}} > p_{\text{target}}$），TIS 等于在训练那些 policy 推理时根本不会选的 token，制造越来越大的训练-推理失配。

**核心矛盾**：IS 类方法只是「事后给样本加权纠偏」，无法改变 actor 实际采到的轨迹本身；要么稳定但学不动，要么激进但崩溃，二者不可兼得。

**本文目标**：能不能**直接修改 actor 的采样分布和采样轨迹**，从源头缩小它与 policy 的分布差距？

**核心 idea**：**用拒绝采样改造 actor 分布**。经典拒绝采样要求采样后分布与目标完全一致，在 10 万词表的 LLM 上接受率几乎为零、完全不可用。Jackpot 改用 **OBRS（Optimal Budgeted Rejection Sampling）**——给定一个目标接受率「预算」，求出在该预算下使 KL 散度最小的拒绝规则；它不强求分布完全相同，但**理论保证**调整后的 actor 分布严格比原始更接近 policy，从而在「样本效率」和「分布对齐」之间取得最优折中。

## 方法详解

### 整体框架
Jackpot 在标准 PPO/GRPO 训练循环上做轻量改造，由三块组成：**(1) OBRS 拒绝-重加权**，按 $\min(1, \frac{p_{\text{target}}}{\lambda p_{\text{inf}}})$ 概率对 token 做掩码，把保留 token 的有效分布拉向目标；**(2) Top-K 概率估计 + 批级偏差校正**，避免对 10 万词表归一化常数做全词表求和；**(3) 稳定化 Jackpot-PPO 损失**，把 OBRS 重要性比和 PPO 信赖域约束联合进损失。整条流程不需额外采样轨迹、不需额外前向、不改 vLLM。

```mermaid
flowchart TD
    A[Phase 1: 标准 rollout<br/>actor p_inf 采样轨迹] --> B[同一前向存 Top-K logprob]
    B --> C[Phase 2: PPO 更新]
    C --> D[OBRS 拒绝掩码<br/>min(1, p_target/λp_inf)]
    D --> E[Top-K 估 Z_approx<br/>批级 κ 校正去偏]
    E --> F[Jackpot 权重<br/>ρ = min(w_OBRS,c1)·min(p_ref/p_new,c2)]
    F --> G[L_final = SG(ρ)·L_PPO<br/>更新 policy]
```

### 关键设计

**1. OBRS 拒绝-重加权：从源头拉近分布，而非事后纠偏**。对 actor 采到的 token $x$，以概率 $\min(1, \frac{p_{\text{target}}(x)}{\lambda p_{\text{inf}}(x)})$ 接受，被拒的 token 直接掩掉、不参与 loss 和梯度。拒绝后保留 token 的有效分布变成 $p_{\text{OBRS}}(x) = \frac{\min(p_{\text{inf}}(x), p_{\text{target}}(x)/\lambda)}{\sum_{x'} \min(p_{\text{inf}}(x'), p_{\text{target}}(x')/\lambda)}$。缩放因子 $\lambda$（论文实验固定 $\lambda=1$，对应控制旋钮 $C$）越大越靠近目标分布但接受率越低。理论上 OBRS 在给定平均接受率下是唯一最小化到目标 KL 的规则，且保证 $p_{\text{kept}}$ 严格比原始 $p_{\text{inf}}$ 更接近目标——数值模拟里在初始 KL 很大时仍保持约 95% 接受率，却把 KL 降了近一个数量级。这是它区别于「会拒绝绝大多数 token」的朴素拒绝采样的关键。

**2. Jackpot-PPO 损失：把 OBRS 比值与信赖域联合稳定**。直接对 OBRS 后的分布做 PPO 会破坏信赖域。Jackpot 在 TIS 比值上叠加 OBRS 修正，把 $\min(\frac{p_{\text{ref}}}{p_{\text{inf}}}, C)$ 改写为 $\min\!\big(Z\cdot\max(\lambda, \frac{p_{\text{ref}}(x)}{p_{\text{inf}}(x)}), C\big)$，其中 $Z = \sum_{x'} \min(p_{\text{inf}}(x'), \frac{p_{\text{ref}}(x')}{\lambda})$ 是归一化常数。进一步，论文发现高 staleness（大批量/异步）下 reference policy 太旧，不如**逼近最新 policy $p_{\text{new}}$**：通过 $\mathbb{E}_{x\sim p_{\text{ref}}}[f(x)] = \mathbb{E}_{x\sim p_{\text{new}}}[\frac{p_{\text{ref}}}{p_{\text{new}}} f(x)]$ 换元，最终损失为 $L = \mathbb{E}_{x\sim p_{\text{inf}}}\big[\min(Z\cdot\max(\lambda, \frac{p_{\text{new}}(x)}{p_{\text{inf}}(x)}), C_1)\cdot \min(\frac{p_{\text{ref}}}{p_{\text{new}}}, C_2)\cdot f(x)\big]$，用 $C_1, C_2$ 两个截断同时罩住 OBRS 比值与原 PPO 信赖域，权重以 stop-gradient 形式乘到标准 PPO 目标上。用户可按需选择对齐 $p_{\text{ref}}$ 还是 $p_{\text{new}}$。

**3. Top-K 归一化 + 批级偏差校正：让归一化常数可算**。$Z$ 要对整个词表（$|V|>100\text{k}$）求和，存全 logit 向量（batch × seqlen × vocab）会爆显存。Jackpot 利用 LM 概率质量集中在少数 token 的特性，只在 $V_k = \text{top-}k(p_{\text{inf}}) \cup \text{top-}k(p_{\theta_{\text{new}}})$（取两个分布的 Top-K 并集，因为 $\min$ 函数让重叠区重要）上求和得 $Z_{\text{approx}}$，$k=20$ 时额外计算 <3%。但截断只会低估真值（$\mathbb{E}[Z_{\text{approx}}] \le Z$），系统性偏差会改变梯度尺度。校正办法很优雅：归一化常数 $Z$ 恰好等于期望接受率 $\bar{\alpha} = \sum_a \min(p_{\text{inf}}(a), \frac{p_{\text{target}}(a)}{\lambda})$，而 rollout 阶段可由「接受样本数/提议样本数」无偏估出 $\hat{\bar{\alpha}}$。于是用批级校正因子 $\kappa = \frac{\hat{\bar{\alpha}}}{\frac{1}{B}\sum_i Z_{\text{approx}}^{(i)}}$ 把低方差但有偏的 $Z_{\text{approx}}$ 缩放到无偏尺度，兼得低方差与无偏。

## 实验关键数据

### 主实验：大批量训练（Qwen3-Base + DeepScaleR）
rollout 大 batch、train 小 batch（放大 staleness）。Best 分数（部分基准）：

| 设置 | 方法 | MATH-500 | AMC22&23 | AIME24(Mean@16) | AIME25(Mean@16) |
|------|------|----------|----------|------|------|
| Qwen3-4B 64× | On Policy | 81.55 | 58.43 | 23.12 | 22.91 |
| | Off Policy | 71.15 | 39.15 | 13.96 | 11.04 |
| | TIS+Adjust | 79.50 | 57.22 | 18.75 | 17.71 |
| | **Jackpot** | **80.05** | 53.92 | **20.63** | **18.13** |
| Qwen3-4B 128× | Off Policy | 60.20 | 33.00 | 8.00 | 5.00 |
| | TIS+Adjust | 19.10 | 7.80 | 1.00 | 1.00（崩溃） |
| | **Jackpot** | **80.00** | **51.20** | **19.16** | **18.52** |
| Qwen3-8B 64× | On Policy | 93.99 | 28.95 | 28.95 | 22.50 |
| | Off Policy | 77.15 | 50.60 | 18.54 | 14.16 |
| | TIS+Adjust | 82.55 | 60.54 | 24.58 | 20.00 |
| | **Jackpot** | **83.05** | **63.55** | **26.87** | **20.41** |

128× 极端设定下 TIS+Adjustment 从一开始就崩（MATH-500 仅 19.1），而 Jackpot 仍逼近 on-policy；相比 off-policy 基线在 AMC 上 +20%、AIME 上 +8%。

### 极端 off-policy：两个不同模型分别 rollout/训练
trainer = Qwen2.5-3B-Base，rollout 用 Qwen2.5-1.5B-Instruct / Qwen2.5-Math-1.5B-Instruct，MATH-8K 训练。AMC22&23 各步 Mean@k：

| 方法 | rollout 模型 | step40 | step50 | step60 | step70 |
|------|------|------|------|------|------|
| Vanilla GRPO | Q2.5-1.5B-IT | 33.3 | 30.0 | 22.4 | 5.7 |
| TIS | Q2.5-1.5B-IT | 32.5 | 26.8 | 0（崩） | 0 |
| **Jackpot** | Q2.5-1.5B-IT | **33.9** | **31.9** | **28.8** | **13.4** |

在「rollout 和训练是完全不同模型」的极端失配下，Jackpot 仍持续提升，MATH-500 上较基线 +12%，而 TIS/GRPO 早早崩溃。

### 关键发现
- **延迟崩溃**：Jackpot 的核心价值是在大失配下显著推迟 training collapse 的发生时点，让 off-policy 收敛速度大幅接近 on-policy。
- **可单独生效**：在 KV FP8 量化导致的较温和失配下，仅用 OBRS 拒绝采样（不加任何截断技巧）就能让崩溃中的训练恢复。
- **几乎零额外开销**：$k=20$ 仅增 <3% 计算，却支持 64×+ 大批量，端到端吞吐相比小批量 on-policy 提速 4× 以上。

## 亮点与洞察
- **范式转变**：从「事后给样本加权」（IS/TIS）转向「事前改采样分布」（拒绝采样），从根本上而非补丁式地解决 actor-policy 失配。
- **理论与工程兼备**：OBRS 带 KL 最小化的最优性证明，又通过 Top-K + 偏差校正把它落到可在生产 RL 系统跑的形态。
- **「$Z = \bar{\alpha}$」这一恒等式很漂亮**：把难算的全词表归一化常数等价成易测的经验接受率，用 $\kappa$ 一步去偏，是全文最优雅的工程巧思。
- **正交可叠加**：Jackpot 改的是 $p_{\text{inf}}$，TIS 等仍可叠在其上做剩余纠偏；即插即用、不改 vLLM、不需额外前向。

## 局限与展望
- **接受率-对齐的折中仍存**：$\lambda/C$ 越大越对齐但接受率越低，极端设定下被掩 token 增多可能浪费部分 rollout 算力，论文未深入量化这一损耗。
- **Top-K 近似的边界**：$k=20$ 在概率集中的数学推理任务上够用，但在熵更高、分布更平的任务（开放生成、agentic）上是否仍低偏差，缺乏验证。
- **任务面较窄**：实验集中在 Qwen 系 + 数学推理（GSM8K/MATH/AMC/AIME），未覆盖代码、对齐、多轮 agent 等任务。
- **极端设定仍只是 early glimpse**：两模型分跑虽超基线，但 step70 后绝对分数（13.4）仍大幅下滑，距真正稳定还有距离。

## 相关工作与启发
- **失配纠偏的演化**：从 IMPALA 的 V-trace、TIS，到 FlashRL、AReal、LlamaRL 的 $p_{\text{ref}}/p_{\text{inf}}$ 截断重要性采样；Jackpot 与它们正交，主攻「改分布」而非「改权重」。
- **拒绝采样脉络**：OBRS（Verine et al. 2024）源自统计/生物/ML 的拒绝采样传统，本文把它和投机解码式的接受准则（Leviathan et al. 2023）嫁接到 RL，但只掩码不重采、不回溯重生轨迹。
- **系统侧互补**：FP32 LM head、确定性推理等解决的是 serving 数值问题，与 Jackpot 的分布对齐互补，可共同提升 off-policy RL 的可靠性。
- **启发**：「把 staleness 当作可建模的分布失配来主动校正」这一视角，可推广到异步 RL、推测式 rollout、轻量 actor 等更激进的高吞吐训练方案。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— 首次把 OBRS 拒绝采样引入 LLM RL，从「改分布」而非「改权重」的角度切入 actor-policy 失配，思路确实新。
- **实验充分度**: ⭐⭐⭐ —— 覆盖大批量、量化、两模型极端失配三类设定且对比 on/off-policy 与 TIS，但任务限于数学推理、模型限于 Qwen 系，部分消融放在附录。
- **写作质量**: ⭐⭐⭐⭐ —— 动机清晰、公式推导完整、$Z=\bar{\alpha}$ 的偏差校正讲得很透，配图直观。
- **价值**: ⭐⭐⭐⭐ —— 直击 RL 训练 70% 成本在 rollout 的痛点，即插即用、近零开销、可叠加，对高吞吐 off-policy RL 落地有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GEPO: Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning](gepo_group_expectation_policy_optimization_for_stable_heterogeneous_reinforcemen.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring](master_skill_learning_with_policy-grounded_synergy_of_llm-based_reward_shaping_a.md)
- [\[ICLR 2026\] Learning to Reason as Action Abstractions with Scalable Mid-Training RL](learning_to_reason_as_action_abstractions_with_scalable_mid-training_rl.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](../../NeurIPS2025/reinforcement_learning/scalable_policy-based_rl_algorithms_for_pomdps.md)

</div>

<!-- RELATED:END -->

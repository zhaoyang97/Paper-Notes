---
title: >-
  [论文解读] On Entropy Control in LLM-RL Algorithms
description: >-
  [ICLR 2026][机器人][熵控制] 从理论解释为什么传统熵正则化在LLM-RL中几乎无效（因极大动作空间+稀疏最优导致熵偏差压倒优化增益），提出AEnt方法用截断熵（在缩小的token空间上计算）+自适应系数来有效平衡偏差与收益，在数学推理上持续超越baseline。 领域现状：策略梯度方法（PPO/GRPO/DAP…
tags:
  - "ICLR 2026"
  - "机器人"
  - "熵控制"
  - "RLVR"
  - "LLM-RL"
  - "策略优化"
  - "探索-利用"
---

# On Entropy Control in LLM-RL Algorithms

**会议**: ICLR 2026  
**arXiv**: [2509.03493](https://arxiv.org/abs/2509.03493)  
**代码**: [antgroup/AEnt](https://github.com/antgroup/AEnt)  
**领域**: 机器人  
**关键词**: 熵控制, RLVR, LLM-RL, 策略优化, 探索-利用

## 一句话总结
从理论解释为什么传统熵正则化在LLM-RL中几乎无效（因极大动作空间+稀疏最优导致熵偏差压倒优化增益），提出AEnt方法用截断熵（在缩小的token空间上计算）+自适应系数来有效平衡偏差与收益，在数学推理上持续超越baseline。

## 研究背景与动机

**领域现状**：策略梯度方法（PPO/GRPO/DAPO）是LLM-RL的主流。传统RL中熵正则化（SAC/A3C/PPO）通过保持策略随机性防止过早收敛，效果显著。

**现有痛点**：实验发现熵正则化在LLM-RL中几乎无增益。Cui等人观察到不同熵系数对验证准确率影响微乎其微。这与机器人/游戏RL中的显著效果形成矛盾。

**核心矛盾**：理论上熵正则化有优化优势（改善收敛），但在LLM中引入的偏差 $O(H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*(s_0)|^{1/H}})$ 随动作空间 $|\mathcal{A}|$ 和最优稀疏度增大而剧增。LLM词汇表~10万+、最优token极其稀疏→偏差远大于优化增益。

**切入角度**：既然全词汇表上的熵偏差太大，就在更小的合理token空间上计算截断熵——只鼓励在"合理候选"中探索而非在整个词汇表中。

## 方法详解

### 整体框架

这篇论文先回答一个困惑、再给一套解法。困惑是：在传统 RL 里立竿见影的熵正则化，搬到 LLM-RL（PPO/GRPO/DAPO 这类策略梯度方法）上几乎不涨点。论文用两条性能界把原因量化出来，再据此提出 AEnt——把熵限制在一小撮"合理候选 token"上计算，从而保住探索收益、压掉偏差。

整套分析建立在两条命题上。**无熵控制时**（Proposition 1），策略熵是策略梯度范数的上界 $\|\nabla V^{\pi_\theta}\| \leq 2\mathcal{H}(\pi_\theta)$——熵一旦崩溃，梯度就趋零、学习随之停滞；此时性能差被界为 $V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon}{C^{\pi_\theta}(s_0)}$。**加上传统熵正则化后**（Proposition 2），性能界变成

$$V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon^2}{2\lambda C_\lambda} + \lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$$

第一项 $\epsilon^2/2\lambda$ 是熵带来的优化增益（收敛更好），第二项 $\lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$ 是它引入的偏差。关键观察是：LLM 词汇表 $|\mathcal{A}|$ 高达 10 万量级、最优 token 又极其稀疏，偏差项被这个巨大的 $\log|\mathcal{A}|$ 撑爆，彻底压过优化增益——这正是传统熵在 LLM 上失灵、在机器人/游戏（$|\mathcal{A}|$ 只有数十到数百）上有效的根本差异。合成 MDP 实验（$|\mathcal{A}|=10^5$）也印证了这点：最优动作有 10~15 个时传统熵还能涨点，一旦稀疏到少于 5 个就完全失效。AEnt 的全部设计都围绕把熵的作用范围从整个词表缩到一小撮合理候选 token、从而压低这个偏差项展开。

### 关键设计

**1. 截断熵（Clamped Entropy）：只在头部高概率 token 上算熵，把偏差项里的有效 $|\mathcal{A}|$ 压下来**

前面的偏差项之所以爆炸，是因为熵把策略往全词表均匀分布 $1/|\mathcal{A}|$ 拉，鼓励模型在整个 10 万词表上保持随机——可绝大多数 token 根本不该被探索，拉它们只白白制造偏差。截断熵的做法是：在每个状态下只保留概率最高的 top $(1-p)$ 比例 token（丢掉概率最低的 $p$ 那段长尾，实验里 $p=0.25\sim0.33$），构成一个随输入变化的子空间 $\mathcal{A}(s)$，在这个子空间上把策略重归一化

$$\tilde{\pi}_\theta(a|s) = \frac{\exp(\theta_{s,a})}{\sum_{a' \in \mathcal{A}(s)} \exp(\theta_{s,a'})}, \quad \mathcal{A}(s) = \{\text{top }(1-p)\text{ 比例 token}\}$$

再用 $\tilde{\pi}_\theta$ 计算熵。这样探索只发生在"合理候选"之间，偏差项里起作用的不再是全词表 $\log|\mathcal{A}|$ 而是远小得多的有效候选数，优化增益却基本保留——背后的直觉是：base 模型经过预训练/微调后，垫底的低概率 token 几乎不可能是最优的，把它们排除在外只减偏差、不伤探索。合成 MDP 上截断熵在传统熵失效（最优动作 $\le 5$）时仍能涨点，对稀疏度增大也更鲁棒。

**2. 自适应系数：按当前截断熵把 $\lambda$ 拉回目标区间，省掉手调又适配训练全程**

固定的熵系数 $\lambda$ 在机器人/游戏里够用，但 LLM 训练中熵会中途剧烈波动（论文观察到约 200 步后熵骤变、性能饱和），一个写死的 $\lambda$ 顾此失彼。AEnt 给截断熵设一个目标区间 $[\tilde{\mathcal{H}}_{\text{low}}, \tilde{\mathcal{H}}_{\text{high}}]$，每个全局步后按下式投影更新系数：

$$\lambda' \leftarrow \text{Proj}_{[\lambda_{\text{low}}, \lambda_{\text{high}}]}\big[\lambda - \beta\min(\tilde{\mathcal{H}}(\pi_\theta) - \tilde{\mathcal{H}}_{\text{low}},\, 0) + \beta\min(\tilde{\mathcal{H}}_{\text{high}} - \tilde{\mathcal{H}}(\pi_\theta),\, 0)\big]$$

直白说：截断熵掉到下界以下（探索不足）就调高 $\lambda$ 去借熵的好处；冲到上界以上（已经够随机）就调低 $\lambda$ 把权重让回奖励最大化、顺带消耗掉多余的熵；同时把 $\lambda$ 本身夹在 $[\lambda_{\text{low}}, \lambda_{\text{high}}]$ 内防止过冲。实验里这套自适应还能抑制熵和回答长度的爆炸，让推理更高效。

### 损失函数

把上面两件拼起来，AEnt 在每个全局步近似优化的目标就是在原策略优化损失上加一项截断熵正则：

$$\mathcal{L}_{\text{AEnt}}(\theta; \lambda) = \mathcal{L}_{\text{PO}}(\theta) + \lambda \tilde{\mathcal{H}}(\pi_\theta)$$

其中 $\mathcal{L}_{\text{PO}}$ 是底层策略优化目标（实验里取 GRPO），$\tilde{\mathcal{H}}$ 是截断熵，$\lambda$ 由上面的投影规则逐步自适应——和原始熵正则的唯一改动就是把全词表熵换成截断熵、把固定系数换成自适应系数。

## 实验关键数据

### 数学推理基准（取训练中平均分最高的 checkpoint 评测，每题 4 次取平均）

设置 (a) = Qwen2.5-Math-1.5B 在 MATH 上训练；设置 (b) = DeepSeek-R1-Distilled-Qwen-1.5B 在 OpenR1-Math 子集上训练。EntReg 为传统熵正则（GRPO + 原始熵 bonus）。

| 方法 | 设置 | MATH-Hard | MATH-500 | AIME24 | Minerva | Olympiad | AMC |
|------|------|-----------|----------|--------|---------|----------|-----|
| Base | (a) | 0.368 | 0.584 | 0.083 | 0.179 | 0.279 | 0.406 |
| GRPO | (a) | 0.524 | **0.756** | 0.192 | 0.311 | 0.364 | 0.550 |
| EntReg | (a) | 0.546 | 0.752 | 0.167 | 0.316 | 0.370 | 0.562 |
| **AEnt** | (a) | **0.552** | 0.750 | **0.217** | **0.330** | **0.377** | **0.581** |
| Base | (b) | 0.661 | 0.792 | 0.225 | 0.311 | 0.432 | 0.594 |
| GRPO | (b) | 0.773 | 0.865 | 0.367 | 0.347 | 0.576 | 0.769 |
| EntReg | (b) | 0.808 | 0.872 | 0.342 | 0.359 | 0.576 | 0.794 |
| **AEnt** | (b) | **0.813** | **0.882** | **0.392** | 0.359 | **0.591** | **0.825** |

### 关键发现
- **传统熵几乎无增益、有时还掉点**：EntReg 相对 GRPO 在多数基准上变化微乎其微，AIME24 反而下降（(a) 0.192→0.167、(b) 0.367→0.342），印证了 Cui 等人的观察。
- **AEnt 持续领先**：在两种设置平均下、6 个基准里 5 个取得最优（MATH-500 (a) 是唯一例外），说明截断熵确实把偏差问题解决了。
- **合成 MDP 定位病根**：在 $|\mathcal{A}|=10^5$ 的玩具 MDP 上，最优动作有 10~15 个时传统熵尚能涨点，稀疏到少于 5 个时传统熵失效、截断熵仍有效——直接验证了 Proposition 2 的偏差分析。
- **自适应系数更稳**：固定系数在熵中途剧烈波动（约 200 步后）时失控，自适应把熵和回答长度都压在合理区间，避免爆炸。

## 亮点与洞察
- **理论解释LLM-RL的长期困惑**：为什么传统熵在LLM中不work？因为 $O(H\log|\mathcal{A}|)$ 的偏差在$|\mathcal{A}|=10^5$时压倒了一切。这个解释简洁有力。
- **截断熵的直觉**：不应鼓励模型探索所有10万个token，只应在合理候选中保持多样性。从top-1000中随机选比从全词汇表中随机选合理得多。
- **偏差-增益权衡的量化**：Proposition 1和2给出了可操作的理论指导——当$\log|\mathcal{A}|$大且最优稀疏时，需要特殊处理。

## 局限与展望
- top-k的k需要手动设置——自适应k可能更好
- 理论分析基于softmax策略假设，实际LLM有更复杂的结构
- 仅在数学推理上验证，代码/通用推理效果未知
- 截断熵可能过度限制某些需要大范围探索的场景

## 相关工作与启发
- **vs DAPO**: DAPO通过clip/约束间接控制熵，AEnt直接在截断空间上做正则化
- **vs Cui等人**: 他们观察到熵bonus无效但未给出理论解释，本文提供了解释+解法
- **vs SAC**: SAC的熵正则化在机器人任务中成功因为 $|\mathcal{A}|$ 小（数十到数百），LLM的 $|\mathcal{A}|$ 差几个数量级

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论解释+截断熵方案都有洞察力
- 实验充分度: ⭐⭐⭐⭐ 多模型+多基准+合成MDP验证
- 写作质量: ⭐⭐⭐⭐ 理论与实践结合自然
- 价值: ⭐⭐⭐⭐⭐ 解决了LLM-RL训练中一个重要的实践问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Self-Improving Vision-Language-Action Models with Data Generation via Residual RL](self-improving_vision-language-action_models_with_data_generation_via_residual_r.md)
- [\[ICLR 2026\] Masked Generative Policy for Robotic Control](masked_generative_policy_for_robotic_control.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](../../ICML2026/robotics/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[ICLR 2026\] From Embedding to Control: Representations for Stochastic Multi-Object Systems](from_embedding_to_control_representations_for_stochastic_multi-object_systems.md)

</div>

<!-- RELATED:END -->

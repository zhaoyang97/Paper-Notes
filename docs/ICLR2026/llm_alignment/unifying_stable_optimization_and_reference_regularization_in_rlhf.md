---
title: >-
  [论文解读] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)
description: >-
  [ICLR 2026][LLM对齐][RLHF] 提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "RLHF"
  - "双KL正则化"
  - "优势回归"
  - "参考策略插值"
  - "奖励黑客"
---

# Unifying Stable Optimization and Reference Regularization in RLHF (DAR)

**会议**: ICLR 2026  
**arXiv**: [2602.11523](https://arxiv.org/abs/2602.11523)  
**代码**: [https://github.com/tmllab/2026_ICLR_DAR](https://github.com/tmllab/2026_ICLR_DAR)  
**领域**: 对齐RLHF  
**关键词**: RLHF, 双KL正则化, 优势回归, 参考策略插值, 奖励黑客

## 一句话总结
提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。

## 研究背景与动机

**领域现状**：在线RLHF（PPO/RLOO/GRPO）通过RL优化LLM策略。两个核心难题：reward hacking(策略过度优化代理奖励)和训练不稳定(策略剧烈偏移导致崩溃)。

**现有痛点**：
   - 防reward hacking用KL(π_θ||π_0)约束到初始模型
   - 防训练不稳定用clip/KL(π_t||π_θ)约束到当前策略
   - **关键发现**：这两个约束逐步冲突——策略必须同时接近π_0和π_t，但随着训练推进π_t远离π_0，两者交集缩小，高奖励策略被排除在外

**核心矛盾**：稳定性约束和参考正则化的冲突导致优化空间过度受限

**核心 idea**：用对数空间插值的动态参考策略 $\pi_0^\alpha \cdot \pi_t^{1-\alpha}$ 统一两个约束 + 回归变换消除策略比率不稳定

## 方法详解

### 整体框架
DAR 要解决的，是 RLHF 里两类正则化互相打架的问题：为防 reward hacking 要把策略约束到初始模型 $\pi_0$，为防训练崩溃又要把策略约束到当前策略 $\pi_t$，而随训练推进 $\pi_t$ 越来越远离 $\pi_0$，两个约束的可行交集越缩越小，高奖励策略被挤出去。DAR 的做法分两步：先把两个 KL 约束写成一个带可调权重 $\alpha$ 的双KL对齐目标，并证明它等价于对一个**动态插值参考** $\pi_0^\alpha\pi_t^{1-\alpha}$ 的单KL 约束，于是两个对抗的约束被合并成一个会随训练演化的参考；再对这个带 KL 约束的 RL 目标求闭式最优策略，把它解析地转写成一个**加权 SFT（优势回归）损失**，从根上绕开 PPO 那种靠策略比率估计带来的高方差不稳定。换句话说，从"双约束统一"到"插值参考"，再到"闭式最优策略"、最后落到"加权回归损失"，整条推导把一个不稳定的 RL 优化问题化成了一次稳定的监督式拟合。

### 关键设计

**1. 双KL对齐目标：把"防 hacking"和"防崩溃"两个对抗的约束合并成一个**

标准做法是分别加 $\text{KL}[\pi_\theta\|\pi_0]$（约束到初始模型、防过度优化奖励）和 $\text{KL}[\pi_\theta\|\pi_t]$（约束到当前策略、防剧烈偏移），但这两项随训练推进会互相排斥，把可行域压得太死。DAR 把它们用一个权重 $\alpha$ 线性组合进同一个目标：

$$\mathcal{J} = \max_{\pi_\theta} \mathbb{E}[A(x,y)] - \beta\big(\alpha\,\text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\,\text{KL}[\pi_\theta\|\pi_t]\big)$$

关键的一步是 Proposition 4.1：这个加权双KL在对数空间里等价于对单个插值参考的 KL 约束，$\alpha\,\text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\,\text{KL}[\pi_\theta\|\pi_t] = \text{KL}\big[\pi_\theta \,\big\|\, \tfrac{1}{C}\pi_0^\alpha \pi_t^{1-\alpha}\big]$，其中 $C$ 是归一化常数。也就是说真正的参考策略是 $\pi_{\text{ref}} \propto \pi_0^\alpha \pi_t^{1-\alpha}$——一个会随 $\pi_t$ 演化的动态目标，它自动跟踪当前的高奖励区域、提供更好的支持覆盖，而不是死锁在初始模型上。权重 $\alpha$ 就是这条 trade-off 的旋钮：$\alpha\to1$ 偏保守（贴近初始模型），$\alpha\to0$ 偏探索（贴近当前策略）。

**2. 回归变换（Advantage Regression）：把 RL 目标转成加权 SFT，消掉策略比率的方差**

有了单KL 形式的目标，就能写出它的闭式最优策略（Theorem 4.2）：$\pi^* \propto \pi_0^\alpha \pi_t^{1-\alpha} \exp(\tfrac{1}{\beta}A)$，即在插值参考上按优势做指数加权。DAR 不去用 PPO 那样估计并裁剪策略比率，而是直接把这个最优解拟合成一个加权对数似然（SFT）损失：

$$\mathcal{L} = -\,\mathbb{E}\big[(w_{\text{reg}} \cdot w_{\text{adv}}) \cdot \log\pi_\theta(y|x)\big]$$

两个权重各管一头：$w_{\text{reg}} = (\pi_0/\pi_t)^\alpha$ 是正则化权重，惩罚偏离参考的回答；$w_{\text{adv}} = \exp(\tfrac{1}{\beta}A)$ 是优势权重，奖励好回答。因为损失本质上是加权 SFT 而非 RL，它绕开了 PPO 里策略比率估计的高方差来源，梯度更平滑稳定。为防止指数权重把梯度顶爆，再对乘积做裁剪 $\min(w_{\text{reg}} \cdot w_{\text{adv}},\, w_{\text{clip}})$。

### 损失函数 / 训练策略
- Monte Carlo 采样估计优势，避免单独训练一个价值模型
- 批次内做优势归一化
- 超参：$w_{\text{clip}} = 20$，$\alpha = 0.1$，$\beta = 0.05$

## 实验关键数据

### 主实验：直接AI对齐（Qwen2-7B, GPT-4-Turbo评估）

| 方法 | TL;DR | Helpful | Harmless | 平均胜率 |
|------|-------|---------|----------|---------|
| DPO(offline) | 67.17% | 81.34% | 77.91% | 75.47 |
| Online DPO | 78.47% | 88.86% | 83.55% | 83.63 |
| GRPO | 83.03% | 86.93% | 85.50% | 85.15 |
| **DAR** | **98.27%** | **93.16%** | **85.84%** | **92.42** |

### 标准RLHF：Qwen2-7B-Instruct

| 方法 | MT-Bench(GPT-4) | LC% vs π₀ | 长度 |
|------|-----------------|-----------|------|
| GRPO | 8.425 | 50.50 | 1559 |
| RLOO | 8.409 | 52.25 | 1580 |
| **DAR** | **8.538** | **54.17** | **1358** |

### 消融：α的影响

| α | 效果 | 说明 |
|---|------|------|
| α=1.0 | 保守，低奖励 | 完全绑定初始模型 |
| α=0.1 | **最佳平衡** | 允许探索但有约束 |
| α=0.0 | 高奖励但reward hacking | 8% missing-EOS率 |

### 关键发现
- **DAR在TL;DR上达98.27%胜率**：几乎完美的偏好对齐
- **回归变换是关键**：直接RL的双KL(DAO)训练不稳定，双PPO高方差，只有DAR稳定优越
- **样本效率**：DAR用**一半的标注量**达到DAP方法同等效果
- **长度控制**：DAR生成长度(1358)接近原始模型(1340)，不会length hacking

## 亮点与洞察
- **两个约束冲突的深刻发现**：指出RLHF中两类正则化(防hacking vs 防崩溃)实际上在优化中逐步对抗。这个观察解释了为什么很多RLHF方法效果不如预期
- **对数空间插值的优雅解法**：将两个KL项统一为对插值参考的单KL，理论上等价且实践上释放了优化空间
- **回归变换消除RL不稳定性**：将RL问题转化为加权SFT，避免了策略比率估计的方差问题；权重裁剪提供了进一步的稳定性

## 局限与展望
- **需要在线采样**：每步需要从当前策略采样计算优势，开销比offline DPO大
- **α和β需要联合调优**：Pareto前沿依赖(α,β)的选择
- **改进思路**：可结合NSPO的零空间投影——在DAR的加权SFT中确保安全梯度不损害通用能力

## 相关工作与启发
- **vs PPO**：PPO独立处理两类约束→冲突；DAR统一→Pareto前沿提升
- **vs GRPO**：GRPO无价值模型+组相对优势，但仍用策略比率做RL；DAR用回归变换消除比率不稳定
- **vs DPO(offline)**：DPO在固定偏好数据上训练，DAR在线采样+动态参考→更强泛化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 两类约束冲突的发现+对数插值的解法都很深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 多设置(直接对齐+标准RLHF)×多模型×多评估器，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，问题动机清晰
- 价值: ⭐⭐⭐⭐⭐ 为RLHF训练稳定性提供了新理论视角和实用解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] Learning to Summarize User Information for Personalized RLHF（PLUS）](learning_to_summarize_user_information_for_personalized_reinforcement_learning_f.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->

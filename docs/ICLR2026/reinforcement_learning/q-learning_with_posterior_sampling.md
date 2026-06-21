---
title: >-
  [论文解读] Q-learning with Posterior Sampling
description: >-
  [ICLR 2026][强化学习][后验采样] 本文提出 PSQL——第一个用「对 Q 值维护高斯后验、采样后取 argmax」来做探索的 Q-learning 算法，并通过把目标值（target）的计算改成「乐观的多样本采样」，首次为这种最自然的后验采样式 Q-learning 证明了 $\tilde{O}(H^2\sqrt{SAT})$ 的近最优 regret 上界。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "后验采样"
  - "Q-learning"
  - "Thompson 采样"
  - "regret 上界"
  - "表格型 MDP"
---

# Q-learning with Posterior Sampling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=a4z7OlgSxC](https://openreview.net/forum?id=a4z7OlgSxC)  
**代码**: 待确认  
**领域**: 强化学习 / 探索理论  
**关键词**: 后验采样, Q-learning, Thompson 采样, regret 上界, 表格型 MDP

## 一句话总结
本文提出 PSQL——第一个用「对 Q 值维护高斯后验、采样后取 argmax」来做探索的 Q-learning 算法，并通过把目标值（target）的计算改成「乐观的多样本采样」，首次为这种最自然的后验采样式 Q-learning 证明了 $\tilde{O}(H^2\sqrt{SAT})$ 的近最优 regret 上界。

## 研究背景与动机
**领域现状**：在线表格型 episodic RL 里管理「探索-利用」权衡有两条主流路线。一条是 UCB（乐观置信上界），给 Q 值估计加一个显式的探索 bonus，Jin 等人 2018 的 UCBQL 是其代表，能拿到近最优 regret。另一条是后验采样（多臂老虎机里叫 Thompson 采样），靠对参数维护后验、采样来隐式地表达不确定性。大量经验研究表明后验采样在实践中往往优于 UCB。

**现有痛点**：后验采样虽然好用，但**理论分析极其困难**，尤其在 RL 这种带 bootstrap 的复杂场景里。对「最自然」的做法——直接对 Q 值放后验——此前没有可证明 regret 的 model-free 算法。已有的可证明结果要么绕开了「对 Q 值放后验」这条路：Tiapkin 等人 2023 的 Staged-RandQL 改成对转移概率放 Dirichlet 后验、再用学习率随机化来隐式采样；要么是 RLSVI 这种近似值迭代，需要把历史数据重新最小二乘拟合，计算/存储开销大且在表格设定下退化成 model-based，且 regret 对状态数 $S$ 的依赖偏松（$\tilde{O}(H^3 S^{1.5}\sqrt{AT})$）。

**核心矛盾**：把后验采样套进 Q-learning 的 TD 更新，最大的拦路虎是 **bootstrap 带来的误差累积**。Q-learning 的目标 $z = r_h + \hat{V}_{h+1}$ 是用「下一步的当前估计」拼出来的，下一步的误差会被层层传到当前步。更要命的是后验采样缺乏 UCB 那种「高概率乐观性」：UCB 估计天然是真值的高概率上界，而后验采样通常只能证明**常数概率**的乐观性，一旦在 $H$ 步递归里相乘，乐观概率会指数衰减到 $p^H$。

**本文目标**：设计一个真正「对 Q 值放后验、采样取 argmax」的简单 Q-learning 算法，并为它证明出近最优、且对 $S$ 的依赖最优的 regret。

**切入角度**：作者先用贝叶斯推断重新解释 Q-learning 的更新规则，发现它本就是一个带正则的 ELBO 优化的解；这给了「在哪里、怎么注入后验采样」一个有原则的设计依据。关键观察是：要打破乐观概率的递归指数衰减，**不必**让决策处的采样高概率乐观，只需保证**目标值里用到的下一步 value 估计**高概率乐观即可。

**核心 idea**：决策时用「单样本」后验采样（保持探索的自然性与高效），构造目标时用「多样本取最大 + 特制的 argmax 动作」来注入有限的乐观性，二者分工，恰好绕开 bootstrap 的指数误差。

## 方法详解

### 整体框架
PSQL 对每个 $(s,a,h)$ 维护一个关于真值 $Q_h(s,a)$ 的高斯后验 $\mathcal{N}(\hat{Q}_h(s,a),\sigma(N_h(s,a))^2)$，其中 $\hat{Q}$ 是后验均值，方差随访问次数 $n$ 衰减：

$$\sigma(n)^2 = \frac{64H^3}{n+1}\log(KH/\delta).$$

每个 episode 的每一步 $h$，算法做三件事：(1) **决策**——对当前状态 $s_h$ 的每个动作各采**一个**样本 $\tilde{Q}_h(s_h,a)\sim\mathcal{N}(\hat{Q}_h(s_h,a),\sigma(N_h(s_h,a))^2)$，玩 $a_h=\arg\max_a\tilde{Q}_h(s_h,a)$；(2) **构造目标**——观测到 $r_h,s_{h+1}$ 后，用一个**乐观的多样本**过程算出下一步 value 估计 $\overline{V}_{h+1}(s_{h+1})$，得到目标 $z=r_h+\overline{V}_{h+1}(s_{h+1})$；(3) **更新后验均值**——用带特制学习率的 Q-learning 规则

$$\hat{Q}_h(s_h,a_h)\leftarrow(1-\alpha_n)\hat{Q}_h(s_h,a_h)+\alpha_n z,\qquad \alpha_n=\frac{H+1}{H+n}.$$

整套方法的「不对称」是精髓：**决策处是 vanilla 的单样本后验采样**（自然、高效），**目标处是乐观的多样本后验采样**（为了拿到可分析的高概率乐观性）。三个关键设计分别对应「为什么是这个更新规则与学习率」「决策怎么采样」「目标怎么乐观地采样」。

### 关键设计

**1. 贝叶斯 ELBO 视角重新推导 Q-learning：解释更新规则与那个奇怪的学习率**

作者先回答一个根本问题：Q-learning 的更新规则到底「是什么」。把待推断参数 $\theta$ 取为 $Q_h(s,a)$，给定先验 $p$、对数似然 $\ell(\theta,z)$ 和样本 $z$，贝叶斯后验 $q(\theta)\propto p(\theta)\exp(\ell(\theta,z))$ 同时也是下式 ELBO 的最优解：$\max_q\, \mathbb{E}_{\theta\sim q}[\ell(\theta,z)]-\mathrm{KL}(q\|p)$。当先验取高斯 $\mathcal{N}(\hat{\mu},\frac{\sigma^2}{n-1})$、似然取高斯 $\mathcal{N}(\theta,\sigma^2)$ 时，后验恰为 $\mathcal{N}(\hat{\mu},\frac{\sigma^2}{n})$，均值更新就是学习率 $\alpha_n=\frac1n$ 的 Q-learning 规则。这说明标准 Q-learning 更新本质上是在做贝叶斯后验推断。

但这里有个 caveat：上式假设 $z$ 是无偏样本，而 Q-learning 的 $z$ 因 bootstrap 而**有偏**。Jin 等人 2018 为了拿到理论保证，把学习率手动改成 $\alpha_n=\frac{H+1}{H+n}$，当时只是「为了让 regret 分析跑通」。本文给了它一个有原则的解释：在 ELBO 上加一个熵正则项

$$\max_q\, \mathbb{E}_{\theta\sim q}[\ell(\theta,z)]-\mathrm{KL}(q\|p)+\lambda_n\mathcal{H}(q),$$

当 $\lambda_n=\frac{H}{n}$、似然方差取 $\frac{\sigma^2}{H+1}$ 时，最优后验恰好给出学习率 $\alpha_n=\frac{H+1}{H+n}$。熵正则的直觉很清楚：bootstrap 目标本身有额外偏差，所以要**人为多保留一点后验不确定性**；而随着样本数 $n$ 增大、目标偏差变小，正则权重 $\lambda_n=H/n$ 自然衰减。这个推导既是 PSQL 设计的基石，也顺手把 Jin 等人那个「靠分析硬凑出来的」学习率解释清楚了。

**2. 单样本后验采样做决策：把 Thompson 采样自然搬进 Q-learning**

针对「后验采样在实践中比 UCB 强、但没人给 Q 值版本做出理论」的痛点，PSQL 在决策处采用最朴素的 Thompson 式做法：对当前状态每个动作各采一个样本，玩采样 Q 值最大的动作。访问次数少的 $(s,a)$ 后验方差 $\sigma(n)^2\propto H^3/(n+1)$ 大，采样更可能偏离均值、从而被探索；访问多的方差小、采样贴近均值、趋于利用。这种探索不靠加 bonus、也不靠扰动，而是直接从后验里采，是后验采样区别于 UCB 的本质所在。

之所以决策处**只采一个样本**，是因为作者论证：若在决策处也用多样本取最大来强行获得乐观性，bootstrap 会把误差按 $\epsilon\sqrt{J^{H}\log(1/\delta)}$ 量级放大，即便 $J=2$ 也会随 $H$ 指数爆炸。所以决策必须保持 vanilla，乐观性要从别处获取。

**3. 乐观的多样本目标构造：只在 target 里注入有限乐观，绕开递归指数衰减**

这是全文的技术枢纽，解决「常数概率乐观在 $H$ 步递归里指数衰减」的核心难题。作者的洞察是：要打破递归相乘，**不需要**让决策处的估计乐观，只需保证**目标里用到的那个下一步 value $\overline{V}_{h+1}$ 高概率乐观**即可。为此，目标构造（Algorithm 2）做两件事：

其一，**对单个特制动作 $\hat{a}$ 多采样取最大**。先选 $\hat{a}=\arg\max_a\big(\hat{Q}_{h+1}(s',a)+\sigma(N_{h+1}(s',a))\big)$（后验均值加一个标准差偏移），再从 $\mathcal{N}(\hat{Q}_{h+1}(s',\hat{a}),\sigma(N_{h+1}(s',\hat{a}))^2)$ 采 $J$ 个样本取最大：

$$\overline{V}_{h+1}(s')=\max_{j\in[J]}\tilde{V}^j,\qquad J=\frac{\log(SAT/\delta)}{\log(4/(4-p_1))},\quad p_1=\frac{\Phi(-1)-\delta}{H-\delta}.$$

取 $J$ 个样本的最大值把「常数概率乐观」boost 成「高概率乐观」（Lemma 1：$\overline{V}_{k,h}(s_{k,h})\ge V^*_h(s_{k,h})$）。关键是这个乐观**只作用在目标计算上**，不污染 Line 6 的决策，因此误差不会沿决策递归相乘——这正是与 UCBQL 那种「决策处就乐观」的根本分野。

其二，**用 $\hat{a}$ 而非实际所玩动作 $a_{h+1}$**。难点在于 $\overline{V}_{h+1}$ 是在 $\hat{a}$ 处采样得到的，而实际决策玩的是 $a_{h+1}=\arg\max_a\tilde{Q}$。作者证明（Lemma C.2）$\hat{a}$ 是一个「特殊动作」：其标准差以常数概率满足 $\sigma(N(s,\hat{a}))^2<2\sigma(N(s,a_h))^2\log(1/\delta)$，即与所玩动作的不确定性同阶。于是 $\overline{V}_h(s_h)$ 与决策样本 $\tilde{Q}_h(s_h,a_h)$ 的差距能被 $\tilde{O}(\sigma(N_{k,h}(s_{k,h},a_{k,h})))$ 控制住（Lemma 2），而这个量随访问次数以 $1/\sqrt{N}$ 衰减。这一步是把「目标乐观」转化为「可控 regret」的桥梁。

### 损失函数 / 训练策略
没有传统意义的损失函数——PSQL 是在线 TD 更新。核心训练量是后验均值更新 $\hat{Q}_h\leftarrow(1-\alpha_n)\hat{Q}_h+\alpha_n z$，配合学习率 $\alpha_n=\frac{H+1}{H+n}$ 与方差表 $\sigma(n)^2=64H^3\log(KH/\delta)/(n+1)$。初始化 $\hat{Q}_h=\hat{V}_h=H$（乐观初值）、$\hat{Q}_{H+1}=0$。

## 实验关键数据

### 主结果：regret 上界对比
本文是理论工作，核心「实验」是与已有算法的 regret 量级对比（表格型 episodic RL，$T=KH$）。

| 算法 | Regret | 类型 |
|------|--------|------|
| UCBQL (Jin et al., 2018) | $\tilde{O}(H^{1.5}\sqrt{SAT})$ | Q-learning + UCB |
| Q-EarlySettled-Advantage (Li et al., 2021a) | $\tilde{O}(H\sqrt{SAT})$ | Q-learning + UCB |
| RLSVI (Russo, 2019) | $\tilde{O}(H^3 S^{1.5}\sqrt{AT})$ | 近似值迭代 |
| C-RLSVI (Agrawal et al., 2021) | $\tilde{O}(H^2 S\sqrt{AT})$ | 近似值迭代 |
| Staged-RandQL (Tiapkin et al., 2023) | $\tilde{O}(H^2\sqrt{SAT})$ | 学习率随机化 |
| **PSQL（本文）** | $\tilde{O}(H^2\sqrt{SAT})$ | **Q 值高斯后验** |
| 下界 (Jin et al., 2018) | $\Omega(H\sqrt{SAT})$ | — |

PSQL 的 $\tilde{O}(H^2\sqrt{SAT})$ 对 $S,A,T$ 的依赖已达近最优，与下界仅差一个 $H$ 因子；相比 RLSVI 把对 $S$ 的依赖从 $S^{1.5}$ 改进到 $\sqrt{S}$；与更复杂的 Staged-RandQL 持平，但算法更简单、随机采样步骤更少。

### 经验对比
在 chain MDP 环境（Figure 1）里，PSQL 的启发式变体 PSQL\* 与 UCBQL、Staged-RandQL、RLSVI 对比，验证了「后验采样优于 UCB」的经验结论也适用于本文的 Q-learning 路线。值得注意：用于实验的 PSQL\* 是**单样本 vanilla 版**（决策与目标都用单样本），经验表现比可分析的 Algorithm 1 更好，但理论上难分析。

### 关键发现
- **乐观注入的位置决定一切**：只在 target 处乐观（而非决策处）是绕开「乐观概率指数衰减」的关键；若在决策处也用多样本，误差会随 $H$ 指数爆炸。
- **特制动作 $\hat{a}$ 是分析的支点**：Lemma 2 把「在 $\hat{a}$ 采样的目标」和「实际所玩动作」的差距以 $\sigma(N)\propto1/\sqrt{N}$ 绑定，使估计误差可递归求和。
- **经验最优 ≠ 易分析**：vanilla 单样本版（PSQL\*）经验最强，但目前无法证明界，作者列为主要 future work。

## 亮点与洞察
- **把「靠分析凑出来的学习率」讲出了道理**：用熵正则 ELBO 解释 Jin 等人 2018 的 $\alpha_n=\frac{H+1}{H+n}$，把一个工程式的 trick 升格为有原则的贝叶斯推断结果——熵正则对应「bootstrap 目标有偏，要多留不确定性」，权重 $H/n$ 随样本增多自然衰减，直觉极顺。
- **「不对称采样」的设计哲学可迁移**：决策处保持自然（单样本），只在影响递归的关键量（目标里的下一步 value）注入乐观——这个「在递归链的哪一环注入高概率性质」的思路，对其他 bootstrap 型算法的分析有启发。
- **诊断难点的论证很清晰**：作者明确把「常数概率乐观 → $p^H$ 指数衰减」「多样本决策 → $\sqrt{J^H}$ 指数误差」两个失败模式摆出来，再说明自己的设计如何各个击破，读起来像一篇好的「为什么别的做法不行」的反面教材。

## 局限与展望
- **理论与最优实践有 gap**：能证明界的 Algorithm 1（多样本乐观目标）经验上**弱于**单样本的 PSQL\*，而后者目前无法分析。作者把「分析 vanilla 版」列为头号 future work。
- **$H$ 依赖仍非最优**：上界 $\tilde{O}(H^2\sqrt{SAT})$ 与下界 $\Omega(H\sqrt{SAT})$ 差一个 $H$。附录 F 草拟了用更复杂算法换 $\sqrt{H}$ 改进的思路，但会牺牲简洁性。
- **局限于表格型设定**：分析依赖表格 MDP 的有限 $S,A$ 与显式访问计数 $N$，尚未扩展到函数逼近/深度 RL；而后验采样真正诱人的正是深度 RL 场景，这一步留待后续。
- **常数偏大**：$\sigma(n)^2$ 里的常数 64、$H^3$ 量级方差等使有效探索强度偏保守，可能解释了为何可分析版经验上不如 vanilla。

## 相关工作与启发
- **vs UCBQL (Jin et al., 2018)**：两者都基于 Q-learning，但 UCBQL 在**决策处**加显式 bonus 实现乐观，PSQL 在**决策处只采单样本**、把乐观挪到**目标处**。本文还借用并重新诠释了 UCBQL 的学习率 $\frac{H+1}{H+n}$。PSQL 在 $H$ 依赖上略逊（$H^2$ vs UCBQL 后续改进到 $H$），但提供了后验采样路线的首个 Q 值版可证明结果。
- **vs Staged-RandQL (Tiapkin et al., 2023)**：同为 model-free、同样 $\tilde{O}(H^2\sqrt{SAT})$，但 Staged-RandQL 走的是「对转移概率放 Dirichlet 后验 + 学习率随机化」，本质更接近 model-based 思路；PSQL 直接对 Q 值放高斯后验，更自然、采样步骤更少、经验相当或更好。
- **vs RLSVI (Russo, 2019)**：RLSVI 靠给奖励注入随机噪声再重新最小二乘拟合 Q 函数，不 bootstrap 旧估计，计算/存储重，且在表格下退化为 model-based、对 $S$ 依赖偏松（$S^{1.5}$）。PSQL 用增量 TD 更新、对 $S$ 依赖改进到 $\sqrt{S}$。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个「对 Q 值放后验采样」的可证明 Q-learning，ELBO 重新诠释学习率是漂亮的独立贡献。
- 实验充分度: ⭐⭐⭐ 理论为主，经验只在 chain MDP 等小环境验证，且证的是次优变体而非经验最优的 PSQL\*。
- 写作质量: ⭐⭐⭐⭐⭐ 难点诊断（指数衰减/指数误差）与设计动机讲得极清楚，反面论证有说服力。
- 价值: ⭐⭐⭐⭐ 为「后验采样 + TD/DP」这一长期开放的分析难题提供了起点与可复用的技术洞察。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](../../ICML2025/reinforcement_learning/fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICLR 2026\] Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning](policy_likelihood-based_query_sampling_and_critic-exploited_reset_for_efficient_.md)
- [\[AAAI 2026\] Speculative Sampling with Reinforcement Learning](../../AAAI2026/reinforcement_learning/speculative_sampling_with_reinforcement_learning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Toward Efficient Exploration by Large Language Model Agents](toward_efficient_exploration_by_large_language_model_agents.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Predictive CVaR Q-Learning
description: >-
  [ICLR 2026][强化学习][CVaR] 本文提出 Predictive CVaR Q-learning（PCVaR-Q），通过引入一对"预测式尾部值/尾部概率函数"把原本只能在轨迹末端结算的 CVaR 目标改写成可逐步递推的贝尔曼形式，再配一个同时探索动作和风险预算的"双向探索"策略，显著提升了风险敏感 RL 的样本效率与训练稳定性，在决策树和随机网格世界上都逼近 CVaR 最优策略。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "CVaR"
  - "风险敏感强化学习"
  - "Q-learning"
  - "贝尔曼方程"
  - "样本效率"
---

# Predictive CVaR Q-Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=B4SCegRJOA](https://openreview.net/forum?id=B4SCegRJOA)  
**代码**: 无  
**领域**: 强化学习 / 风险敏感RL  
**关键词**: CVaR、风险敏感强化学习、Q-learning、贝尔曼方程、样本效率

## 一句话总结
本文提出 Predictive CVaR Q-learning（PCVaR-Q），通过引入一对"预测式尾部值/尾部概率函数"把原本只能在轨迹末端结算的 CVaR 目标改写成可逐步递推的贝尔曼形式，再配一个同时探索动作和风险预算的"双向探索"策略，显著提升了风险敏感 RL 的样本效率与训练稳定性，在决策树和随机网格世界上都逼近 CVaR 最优策略。

## 研究背景与动机
**领域现状**：在自动驾驶、机器人手术、金融这类高风险序贯决策里，罕见但灾难性的结果不能被忽视。标准 RL 优化期望回报、默认风险中性，并不合适。在各种风险度量中，条件风险价值（CVaR，即回报分布最差 $q$ 分位区间内的期望损失）因为数学上可处理、又直接盯住最坏情形，成为最常用的目标。优化 CVaR 的方法主要分两条线：策略梯度方法和基于值的方法。

**现有痛点**：CVaR RL 出了名地样本效率低，通常被归因于它只盯着一小撮最坏情形轨迹。但本文指出，低效其实来自两个更根本的问题。其一是**含噪的策略评估**：很多基于值的方法把 CVaR 当成只在回合末端兑现的、不可分解的单一奖励 $-(\eta-R_{1:T})_+$，整条轨迹的学习信号被压缩成一个延迟到末端的结果，智能体无法评估每一步动作的即时影响，导致评估极其嘈杂。其二是**无效的探索**：因为 CVaR 由最坏情形驱动，学习信号几乎只来自"失败"轨迹，落在低风险分位之外的高回报轨迹被直接忽略，智能体学不会在已经成功的行为上继续改进——这就是文献里的"对成功视而不见"（blindness to success），会让训练停滞在过度保守的次优策略上。

**核心矛盾**：CVaR 目标 $-(\eta-R_{1:T})_+$ 在时间上**不可分解**，把奖励的兑现拖到了整段时域末端；同时它的非零有效奖励只出现在约 $(1-q)$ 比例的尾部轨迹里，绝大多数轨迹的有效奖励为零、对学习毫无贡献。可分解性缺失和尾部稀疏，正是两难的根源。

**本文目标**：(1) 给 CVaR 目标找到一个能在每一步传播学习信号的递推结构，消除评估噪声；(2) 设计一种能跳出过度保守的探索机制。

**切入角度**：作者把视角从"末端结算的尾部期望"转向"逐步预测尾部"——既然 $-(\eta-R_{1:T})_+$ 难以分解，那就改去递推地预测"从当前往后剩余回报落入尾部的概率"以及"落入尾部时的回报值"，这两个量恰好可以满足风险中性式的贝尔曼递推。

**核心 idea**：用一对"预测式尾部值函数 $f^\chi$ + 预测式尾部概率函数 $g^\chi$"把 CVaR 目标改写成可时序分解的贝尔曼递推，再叠加对风险预算 $\eta$ 的随机化探索，把 CVaR 优化变成一个像普通 Q-learning 那样可训练的问题。

## 方法详解

### 整体框架
PCVaR-Q 建立在 CVaR 的**变分表示**和**状态增广**之上。给定风险水平 $q\in(0,1]$，CVaR 有变分形式

$$q\cdot \mathrm{CVaR}_q^\pi[R_{1:T}] = \max_{\eta\in\mathbb{R}}\Big\{q\eta + \mathbb{E}^\pi\big[-(\eta-R_{1:T})_+\big]\Big\},$$

这把 CVaR 最大化拆成对尾部预算 $\eta$（外层）和策略 $\pi$（内层）的两段优化。把"残余预算"作为附加状态 $Y_t^\eta := \eta - R_{1:t-1}$ 引入，就得到增广状态空间上的马尔可夫核 $\chi_t:\mathcal{S}\times\mathbb{R}\to\Delta_{|\mathcal{A}|}$，动作按 $A_t\sim\chi_t(\cdot|S_t,Y_t^\eta)$ 选取。已有工作（Pflug & Pichler 2016 等）在这个增广空间上对终端奖励 $-(\eta-R_{1:T})_+$ 做动态规划，但因为终端奖励不可时序分解、且大多数轨迹有效奖励为零，样本效率很差。

本文不沿用"末端结算"的值函数，而是定义一对**预测式函数**重写整个学习目标，整体可以概括为三件事：(1) 用 $f^\chi$（尾部值）和 $g^\chi$（尾部概率）替换原终端值函数 $u^\chi$，并证明它们满足风险中性式的贝尔曼递推，从而让学习信号能逐步传播；(2) 用两套 TD 损失分别拟合 $\hat f_\theta$、$\hat g_\phi$，并周期性更新风险预算 $\eta$，整体是一套广义策略迭代（GPI）；(3) 在采样轨迹时对动作（$\epsilon$-greedy）和初始风险预算（从 $\mathcal{N}(\eta,\sigma_k^2)$ 采样）同时随机化，做双向探索。下面三个关键设计依次对应这三件事。

### 关键设计

**1. 预测式尾部值/概率函数：把不可分解的 CVaR 目标改写成可递推的贝尔曼形式**

这是全文的理论基石，针对的是"末端结算导致评估含噪"这个痛点。作者定义两个函数（约定 $\eta=0$，并证明它们对 $\eta$ 不变）：预测式尾部概率函数

$$g_t^\chi(s,y,a) := \mathbb{P}^{\chi}\big(R_{t:T}\le y \mid S_t=s, Y_t=y, A_t=a\big),$$

刻画"从 $t$ 起剩余回报 $R_{t:T}$ 落到阈值 $y$ 以下（即进入 CVaR 尾部）的概率"；以及预测式尾部值函数

$$f_t^\chi(s,y,a) := \mathbb{E}^{\chi}\big[\mathbb{I}\{R_{t:T}\le y\}\,R_{t:T} \mid S_t=s, Y_t=y, A_t=a\big],$$

它是标准动作值函数（Q 函数）的风险敏感版本，捕捉"以保持在尾部的概率加权的剩余回报"，同时反映尾部结果的**幅度与发生概率**。关键在于 $R_{t:T}$ 本身可递归分解，因此在 Assumption 1（剩余回报分布无概率质点）下，作者证明了 $f^\chi$ 满足一个**带即时奖励项的贝尔曼方程**（Theorem 1）：

$$f_t^\chi(s,y,a) = \mathbb{E}\Big[f_{t+1}^\chi(S_{t+1}, y-R_t, A_{t+1}) + g_{t+1}^\chi(S_{t+1}, y-R_t, A_{t+1})\cdot R_t\Big].$$

与先前工作的贝尔曼方程（$u_t^\chi = \mathbb{E}[u_{t+1}^\chi]$，无即时奖励项）相比，这里多出的 $g_{t+1}^\chi\cdot R_t$ 项正对应标准贝尔曼方程里的即时奖励，意味着"当前奖励对最终目标的预期贡献"被显式地、按尾部概率加权地传回了每一步——于是学习信号在整条轨迹上密集传播，而不是只在末端兑现，评估噪声大幅下降。配套地，Proposition 1 给出时序分解式 $f_t^\chi = \mathbb{E}[\sum_{\tau\ge t} g_{\tau+1}^\chi\cdot R_\tau]$，$g_t^\chi=\mathbb{E}[g_{t+1}^\chi]$（鞅性质），并把原目标改写为 $\mathbb{E}[-(\eta-R_{1:T})_+] = \mathbb{E}_{A_1}[f_1^\chi(s_1,\eta,A_1) - g_1^\chi(s_1,\eta,A_1)\cdot\eta]$。在这套结构上，作者进一步建立了贝尔曼最优方程（Theorem 2，定义 $v_t^\chi:=\mathbb{E}[f_t^\chi - g_t^\chi\cdot y]$，并证明 $\sup_\pi q\cdot\mathrm{CVaR}_q = \max_\eta\{q\eta + v_1^*(s_1,\eta)\}$）和策略改进定理（Theorem 3：对 $(f^\chi,g^\chi)$ 取贪心核 $\chi'$ 必有 $v_t^\chi\le v_t^{\chi'}$，从而 CVaR 单调不降），把经典 Q-learning 理论完整推广到了 CVaR 设定。

**2. 分离估计尾部概率与尾部值：用解耦换取低方差、可稳定学习的目标**

把 CVaR 拆成 $g$（概率）和 $f$（值）两个量来分别估计，而不是直接回归一个尾部期望，是本文一个被刻意强调的优势，针对的是"尾部样本既少又高方差"这一困难场景。直接回归尾部期望需要拟合尾部回报的幅度，当非零有效奖励样本既稀少、方差又大时会非常不稳；而尾部概率 $g$ 可以相对稳定地估计（极端情况下就是数一数有多少比例的轨迹越过阈值），且 $g\in[0,1]$ 天然有界，可以套用 log-loss、KL 散度这类稳定的学习目标。本文借助前述时序分解，把概率估计（$g$）与值估计（$f$）显式解耦：概率分量学得更稳，值分量则借分解结构降方差。这条设计与设计 1 是一体两面——正是因为引入了 $f,g$ 这对函数，才有"分而治之"的空间。

**3. 双向随机化探索 + 周期性风险预算更新：在增广状态空间里探索风险偏好，破解"对成功视而不见"**

这针对的是探索痛点。常规 $\epsilon$-greedy 只在**动作层面**扰动；本文额外在**增广状态空间**里探索——每个回合的初始残余预算不再固定，而是从 $Y_1\sim\mathcal{N}(\eta,\sigma_k^2)$ 采样，其中 $\eta$ 是当前对最优风险预算的估计。围绕这个中心采样，智能体会经历不同风险敏感度下的轨迹（有时激进、有时保守），从而真正探索"风险偏好"这一维度，避免过早收敛到过度安全的次优策略；$\sigma_k$ 类比 $\epsilon$，随训练退火（实验里按阶段从 3→2→1→0）。整套训练是 GPI：参数 $\theta,\phi$ 用两套从 Theorem 1 / Proposition 1 导出的 TD 损失更新，

$$L_f(\theta) = \tfrac{1}{B|H|}\sum_{\eta'\in H}\sum_j\Big(\hat f_j^\theta - [\hat f_{j+1}^\theta + \hat g_{j+1}^\phi\cdot R_j]\Big)^2,\quad L_g(\phi) = \tfrac{1}{B|H|}\sum_{\eta'\in H}\sum_j\Big(\hat g_{j+1}^\phi - \hat g_j^\phi\Big)^2,$$

这里有一个实用技巧：每个样本的损失在一组离散候选预算 $H\subset\mathbb{R}$ 上同时计算，利用"预测式函数对初始预算 $\eta$ 不变"的性质，让函数逼近器在各风险水平间泛化。风险预算 $\eta$ 则每 $c$ 回合按变分外层优化更新一次：$\eta\leftarrow\arg\max_{\eta'\in H}\max_a\{\hat f_1^\theta(s_1,\eta',a) + \eta'(q-\hat g_1^\phi(s_1,\eta',a))\}$，更新后的 $\eta$ 又作为下一阶段风险探索的中心点。此外可选地用已有轨迹（如风险中性策略采的数据）预训练 $(\theta,\phi,\eta)$ 做 warm-start，进一步缓解早期的 blindness-to-success。

### 损失函数 / 训练策略
核心训练目标即上面的 $L_f(\theta)$（由 Theorem 1 贝尔曼方程导出）与 $L_g(\phi)$（由 Proposition 1 的鞅性质导出），均用 Adam 优化；动作按 $\epsilon$-greedy + 对 $\hat f_\theta - \hat g_\phi\cdot Y_t$ 取贪心选取。实验中两个环境都用表格型函数逼近器，学习率 $\alpha_\theta=0.01$、$\alpha_\phi=0.0001$，$\epsilon_t=0.1\cdot0.9^{\lfloor t/100\rfloor}$，批量为 8 条轨迹，风险预算每 500 回合更新一次，PCVaR-Q 与基线 CVaR-Q 用完全相同的超参以保证公平。

## 实验关键数据

实验在两个可控环境上对比三种策略：RN（风险中性最优）、CVaR-Q（基于 Pflug & Pichler 2016 贝尔曼算子的末端结算式 Q-learning 基线）、PCVaR-Q（本文）。

### 主实验：CVaR 性能（$q=0.1$）

| 环境 | 指标 | RN | PCVaR-Q（本文） | CVaR 最优 |
|------|------|------|------|------|
| 序贯决策树 | CVaR@0.1 | 1.96 | **2.45** | 2.50 |
| 随机网格世界 | CVaR@0.1 | −58.37 | **−55.84** | −53.34 |

在决策树里，RN 总是选高期望但高风险的 up 路径（均值回报 5.0、CVaR 仅 1.96），而 PCVaR-Q 学到了更谨慎的 CVaR 最优策略，CVaR 达到 2.45，几乎贴住理论最优 2.50。网格世界（8×10、转移成功率 0.7、撞障碍罚 $\mathcal{N}(-50,1)$、到达目标奖 $\mathcal{N}(50,1)$）里，PCVaR-Q 学到避障的更安全路径，把分布下尾从 RN 的 −58.37 改善到 −55.84，逼近最优 −53.34。

### 稳定性 / 样本效率分析

| 配置 | 学习曲线表现 | 说明 |
|------|------|------|
| PCVaR-Q | 平稳、快速、单调收敛到近最优 | 10 次独立试验均值，每 100 迭代评估一次（每点 10 万次采样估计） |
| CVaR-Q（基线） | 高方差、收敛到次优 | 同样超参下波动明显 |

### 关键发现
- **递推结构是稳定性的来源**：相同超参下，PCVaR-Q 学习曲线平稳收敛、CVaR-Q 高方差且收敛到次优——印证了把 CVaR 改写成逐步传播的贝尔曼递推确实降低了评估噪声，这是样本效率提升的直接原因。
- **逼近理论最优**：两个环境的 CVaR 都非常接近各自的理论最优值（2.45 vs 2.50；−55.84 vs −53.34），说明该框架不仅稳，还能真正找到 CVaR 最优策略。
- **理想条件下 CVaR-Q 也能收敛到最优**，本文的卖点是"同等条件下"的鲁棒性与稳定性，而非渐近最优性的差异。

## 亮点与洞察
- **把"末端结算"换成"逐步预测尾部"**：最巧妙之处是用 $f,g$ 一对预测函数，让一个本质上不可时序分解的 CVaR 目标重新获得了贝尔曼递推——多出来的 $g_{t+1}\cdot R_t$ 项正好扮演标准贝尔曼方程里的即时奖励，整个改写既自然又把经典 Q-learning 的全套理论（贝尔曼最优、策略改进）搬了过来。
- **概率/值解耦带来工程红利**：$g\in[0,1]$ 有界、可用 log-loss/KL 稳定训练，这一点把"尾部稀疏+高方差"的难题拆成一个好估的概率问题和一个降了方差的值问题，思路可迁移到其他分位/尾部敏感的估计任务。
- **在状态增广维度上探索**：把风险预算 $\eta$ 当成可随机化的"偏好旋钮"来探索，是对"blindness to success"一个直接而优雅的回应——它让智能体主动体验不同保守程度的轨迹，而不是被尾部信号锁死在过度保守里。

## 局限与展望
- **模型复杂度增加（作者承认）**：需要同时学两个函数逼近器 $\hat f_\theta$、$\hat g_\phi$，还要持续跟踪/更新残余阈值 $\eta$，比单一值函数的基线更重。
- **实验规模有限**：仅在两个小型、表格型环境（序贯决策树、8×10 网格世界）上验证，用的是表格函数逼近器，尚未在深度 RL / 高维连续控制上检验，泛化性存疑。
- **依赖 Assumption 1**（剩余回报分布无概率质点）：对带确定性奖励或离散回报的环境，该假设是否成立、不成立时理论是否退化，值得关注。
- **展望**：作者提出可扩展到深度 RL、与基于模型的风险敏感规划结合，以及自适应风险建模与安全攸关的真实场景应用。

## 相关工作与启发
- **vs CVaR-Q（Pflug & Pichler 2016 / Wang et al. 2023 式末端结算）**：基线把 CVaR 当成只在末端兑现的终端奖励 $-(\eta-R_{1:T})_+$，贝尔曼方程无即时奖励项、大多数轨迹有效奖励为零；本文用 $f,g$ 的递推把学习信号铺到每一步，样本效率与稳定性显著更好。
- **vs Predictive CVaR Policy Gradient（Kim & Min 2024）**：该工作在**策略梯度**设定下用类似的风险条件概率量来指导更新；本文把这一思想**适配并扩展到基于值的学习**——引入动作条件、嵌入贝尔曼式递推，从而能做直接最大化 CVaR 的动作选择，而非停留在轨迹级估计。
- **vs 分布式 RL 风险敏感方法（Lim & Malik 2022、Singh et al. 2020 等）**：它们学整个回报分布再算风险；本文只精准地预测"尾部概率+尾部值"两个标量函数，目标更聚焦、与 Q-learning 的对接更直接。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用预测式尾部值/概率函数把不可分解的 CVaR 目标改写成贝尔曼递推，并配双向探索，理论与算法都成体系。
- 实验充分度: ⭐⭐⭐ 论证清楚但仅限两个小型表格环境，缺深度 RL 与大规模验证。
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨、动机与方法衔接清晰，定义/定理组织得当。
- 价值: ⭐⭐⭐⭐ 为风险敏感 RL 提供了把经典 Q-learning 理论推广到 CVaR 的扎实基座，思路具可迁移性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Wavelet Predictive Representations for Non-Stationary Reinforcement Learning](wavelet_predictive_representations_for_non-stationary_reinforcement_learning.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](../../AAAI2026/reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[ICLR 2026\] Q-learning with Posterior Sampling](q-learning_with_posterior_sampling.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Models for Reinforcement Learning](from_observations_to_events_event-aware_world_models_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Minimax Optimal Adversarial Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][对抗转移核] 本文首次证明：在转移核被对手任意选择（fully adversarial）的 episodic MDP 中，仍可达到次线性 regret，提出 AD-FTRL 算法把 regret 压到 $\tilde{O}(\sqrt{(|S||A|)^K T})$，并构造匹配下界证明其 minimax 最优。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "对抗转移核"
  - "历史依赖策略"
  - "FTRL"
  - "轨迹占用测度"
  - "minimax 最优"
  - "次线性 regret"
---

# Minimax Optimal Adversarial Reinforcement Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QEcSLhfOoQ](https://openreview.net/forum?id=QEcSLhfOoQ)  
**代码**: 未提供（理论论文）  
**领域**: 强化学习 / 在线学习 / 对抗 MDP  
**关键词**: 对抗转移核, 历史依赖策略, FTRL, 轨迹占用测度, minimax 最优, 次线性 regret  

## 一句话总结
本文首次证明：在转移核被对手任意选择（fully adversarial）的 episodic MDP 中，仍可达到次线性 regret，提出 AD-FTRL 算法把 regret 压到 $\tilde{O}(\sqrt{(|S||A|)^K T})$，并构造匹配下界证明其 minimax 最优。

## 研究背景与动机

**领域现状**：对抗强化学习已有大量工作，但绝大多数只让损失函数 $\ell_t$ 对抗变化、转移核 $P$ 固定不变；此时可以把问题归约到"对抗平均损失 $\bar\ell$"，存在 Markov 最优策略，并能拿到 $\tilde{O}(\sqrt{|S||A|T})$ 的次线性 regret。

**现有痛点**：一旦转移核 $P_t$ 也由对手在每个 episode 任意选择，问题就难得多。已有工作（Jin et al. 2023; Wei et al. 2022）的做法是去估计一个"中心/平均"转移核 $\bar P$，再基于估计核更新策略，得到 $\tilde{O}(\sqrt{|S||A|T}+C_P)$ 的界，其中 $C_P$ 衡量转移核的对抗扰动量。

**核心矛盾**：在 fully adversarial 情形下 $C_P$ 可以大到 $O(T)$，于是这些界退化成**线性 regret**，等于没学到东西。更深的原因是：平均占用测度并不等于平均转移核下的占用测度（$\bar p_\pi \ne p_{\bar P,\pi}$），估计 $\bar P$ 这条路天生带来无法消除的 $C_P$ 惩罚，且 $\bar p_\pi$ 甚至可能不对应任何真实可实现的转移模型。

**本文目标**：回答一个基础问题——在转移核完全对抗、且只有 bandit 反馈（仅观测到访问过的状态-动作损失）的设定下，能否做到次线性 regret，并刻画其信息论极限。

**核心 idea**：① **放弃估计转移核**，转而用重要性采样直接无偏估计**轨迹级占用测度** $q_{P,\pi}$，从根上绕开 $C_P$ 惩罚；② 在**历史依赖策略类**上跑 FTRL（论文先证明对抗转移下最优策略必须 history-dependent，Markov 策略严格次优）；③ 构造硬 MDP 实例，用复合假设检验给出匹配下界，坐实 minimax 最优。

## 方法详解

### 整体框架
AD-FTRL 把对抗转移 MDP 转写成一个"轨迹空间上的在线线性优化"问题：每个 episode 用当前历史依赖策略 $\pi_t$ 采一条轨迹 $\tau_t$，通过重要性采样得到轨迹占用测度与损失的无偏估计，再用 FTRL（带精心设计的熵/KL 正则）更新到下一个历史依赖策略。整套流程**完全不触碰转移核估计**，因而摆脱 $C_P$ 项。

```mermaid
flowchart LR
    A[历史依赖策略 πt] --> B[采轨迹 τt<br/>bandit 反馈损失]
    B --> C[重要性采样<br/>无偏估计轨迹占用 q̂ 与损失 L̂t]
    C --> D[累加 Υt+1 = Υt + L̂t]
    D --> E[FTRL 更新<br/>⟨Ππ,Υ⟩ + 1/η·Φ(Ππ)]
    E --> A
```

### 关键设计

**1. 历史依赖策略类：对抗转移下 Markov 不再最优。** 固定转移核的标准 MDP 里最优策略可取 Markov，但论文用一个交替转移序列 $\{P_1,P_2,P_1,P_2,P_1\}$ 的反例（Figure 1）说明：当初始状态会"泄露"对手所选的转移核时，最优策略必须依赖历史。该反例中 $\bar V(\pi^*_{\text{Markov}})=\tfrac{2}{5}$ 而 $\bar V(\pi^*_{\text{his}})=0$，存在不消失的间隙。因此作者把搜索空间扩到历史空间 $H_k$（远大于状态空间），策略 $\pi^k_t: H_k \to \Delta(A)$ 把完整历史映射到动作分布——这是后续算法与下界都必须建立在其上的根本前提。

**2. 轨迹占用测度 + 重要性采样：绕开转移核估计。** 不再估计 state-action 占用 $p_{P,\pi}$（依赖已知转移），而是定义轨迹级占用 $q_{P,\pi}(\tau)=P(s_0)\prod_{k=0}^{K-1}\pi(a_k\mid h_k)\,P(s_{k+1}\mid s_k,a_k)$，并用行为策略 $\pi_t$ 采得的轨迹做重要性加权得到无偏估计（Lemma 3.1）：$\mathbb{E}_{\tau_t}\big[\frac{\prod_k \pi(a_k\mid h_k)}{\prod_k \pi_t(a_k\mid h_k)}\mathbb{I}_{\tau_t=\tau}\big]=q_{\pi,P_t}(\tau)$。由此估计的回报 $\hat V_t(\pi)=\langle \hat q_{\pi,P_t},\ell_t\rangle$ 满足 $\mathbb{E}[\hat V_t(\pi)]=V_t(\pi)$，完全不需要知道 $P_t$。这正是消除 $C_P$ 惩罚的关键——它直接估计"被访问的轨迹分布"，而非"平均转移核"这个有偏代理。论文还在 Remark 1 论证该问题无法简单归约成（上下文）bandit：动作会改变后续状态分布，违反上下文 bandit 的"上下文外生"假设。

**3. 策略-环境解耦的 FTRL 正则：避免未知量进正则项。** 直接把含未知转移的估计塞进正则会导致高方差、更新不稳。作者把平均轨迹占用分解为 $\bar q_{\pi,t}(\tau)=\Pi_\pi(\tau)\cdot\bar{\mathsf{F}}_t(\tau)$，其中 $\Pi_\pi(\tau)=\prod_{k}\pi(a_k\mid h_k)$ 只含策略、$\bar{\mathsf F}_t$ 只含（未知的）转移分布。正则**只用策略相关部分** $\Phi_t(\Pi_\pi)=\sum_\tau \Pi_\pi(\tau)\log\Pi_\pi(\tau)$（Shannon 熵 / KL 形式），从而把 FTRL 更新写成 $\pi_t=\arg\min_\pi \langle \Pi_\pi,\Upsilon_t\rangle+\frac{1}{\eta_t}\Phi_t(\Pi_\pi)$，其中 $\Upsilon_t=\sum_{\iota<t}L_\iota$、$L_\iota(\tau)=\mathsf F_\iota(\tau)\ell_\iota(\tau)$。损失估计用裁剪的重要性权重 $\hat L_t=\frac{\ell_{\tau_t}}{\gamma+\prod_k \pi_t(a_k\mid h_k)}[\mathbb{I}_{\tau=\tau_t}]$（$\gamma$ 做数值稳定）。更新后的策略被约束在 $\epsilon$-greedy 历史依赖类 $C_{\pi,\epsilon}$（典型取 $\epsilon=1/T$）以保证重要性采样无偏。

**4. 复合假设检验下界：坐实 minimax 最优。** 为证下界不可改进，作者构造一个硬 MDP 实例，把 regret 最小化归约为**复合假设检验**（而非标准 RL 常用的二元/多元假设检验）——这是对抗转移的内在要求。借助信息论工具得到 $\mathrm{Reg}_T\ge \Omega(\sqrt{(|S||A|)^K T})$，与上界同阶。相比 Tian et al. (2021) 在 Markov game 中 $\Omega(\sqrt{2^K T})$ 的下界，本文的界更紧且显式刻画了对状态/动作空间维度的依赖。

## 实验关键数据

本文为纯理论工作，无数值实验；下表汇总其核心定理与同设定下的对比。

### 主结果（regret 上下界）

| 结果 | 设定 | regret 界 |
|---|---|---|
| Theorem 5.1（上界） | 对抗 bandit 损失 + 未知对抗转移，取 $\eta=(\tfrac{|S|}{|A|})^{K/2}\tfrac{1}{\sqrt{T\log(|S||A|)}}$, $\gamma=\tfrac{1}{2(|S||A|)^{K/2}\sqrt{T}}$ | $\tilde{O}\big((|S||A|)^{K/2}\sqrt{T}\big)$ |
| Theorem 5.2（下界） | 任意算法，$T>4(|S||A|)^K\log T$ | $\Omega\big(\sqrt{(|S||A|)^K T}\big)$ |

### 与现有工作对比

| 工作 | 转移核设定 | regret |
|---|---|---|
| Jin et al. 2020a/2021 | 对抗损失 + **固定**转移 | $\tilde{O}(\sqrt{|S||A|T})$ |
| Jin et al. 2023 / Wei et al. 2022 | 对抗转移（估计 $\bar P$） | $\tilde{O}(\sqrt{|S||A|T}+C_P)$，fully adversarial 下 $\to O(T)$ 线性 |
| Tian et al. 2021（下界） | Markov game | $\Omega(\sqrt{2^K T})$，未含 $|S|,|A|$ 依赖 |
| **本文 AD-FTRL** | **fully adversarial** 未知转移 | $\tilde{O}(\sqrt{(|S||A|)^K T})$，**次线性且 minimax 最优** |

### 关键发现
- 在 fully adversarial（$C_P=O(T)$）情形下，旧方法退化为线性 regret，AD-FTRL 在 $T$ 足够大时仍是次线性——首次给出肯定回答。
- regret 关于状态-动作维度的指数项 $(|S||A|)^{K/2}$ 不可改进：上下界同阶，刻画了该问题的本质难度。
- 该 $K$ 次幂依赖是问题固有的（来自历史空间随 horizon 指数膨胀），并非算法松弛。

## 亮点与洞察
- **换坐标系而非换估计器**：把"估计对抗转移核"这条注定带 $C_P$ 惩罚的路彻底放弃，改在轨迹占用测度空间做无偏估计，是打破线性 regret 壁垒的核心一招。
- **策略-环境解耦的正则设计**很巧妙：通过 $\bar q=\Pi_\pi\cdot\bar{\mathsf F}$ 分解，让正则只依赖可计算的策略部分，未知转移自然被隔离在线性损失项里。
- **下界的复合假设检验框架**是独立的技术贡献，为"对抗/时变转移"这一大类问题提供了统一的下界分析模板。

## 局限与展望
- **regret 关于 horizon $K$ 指数增长**（$(|S||A|)^{K/2}$），作者也坦承这在长 horizon 下不理想；虽证明是 minimax 最优，但也意味着 fully adversarial 转移在长 horizon 下本质困难，实用性受限。
- **纯理论、无实验**：算法需要在 $|A|^K$ 量级的轨迹空间上做 FTRL 更新，计算/存储可行性未讨论，缺乏数值验证。
- 结果限于**表格型分层 MDP**；函数逼近、连续状态/动作、以及结构化对手（介于完全对抗与固定之间）下的最优界仍开放。
- $\epsilon$-greedy（$\epsilon=1/T$）与重要性权重裁剪 $\gamma$ 引入的常数/对数因子在有限样本下的实际影响未刻画。

## 相关工作与启发
- **对抗损失 + 固定转移**：Even-Dar et al. 2009、Zimin & Neu 2013、Jin et al. 2020a/2021 等，奠定了占用测度 + FTRL 的范式，本文把它从 state-action 占用升级到轨迹占用。
- **对抗/受扰转移**：Jin et al. 2023、Wei et al. 2022、Chen et al. 2021、Lykouris et al. 2021 走"估计中心核 + $C_P$"路线，本文指出其在 fully adversarial 下的根本局限并给出替代方案。
- **下界**：Domingues et al. 2021（标准 episodic RL 的 $\Omega(\sqrt{|S||A|T})$）、Tian et al. 2021（Markov game），本文用复合假设检验把下界推进到对抗转移并显式含维度依赖。
- **启发**：当某个中间量（如转移核）的估计天然带不可消除的偏差时，换一个能被无偏估计的等价表示（轨迹占用）往往比硬估计更优——这一思路对其他"难估计核"的对抗在线学习问题有借鉴价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次在 fully adversarial 转移下打破线性 regret，轨迹占用 + 历史依赖策略 + 复合假设检验下界三者结合是实打实的概念突破。
- **实验充分度**: ⭐⭐ 纯理论论文，定理完整且上下界匹配，但完全无数值实验、未讨论算法在指数级轨迹空间上的可计算性。
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰、challenge 与 solution 逐步铺陈，反例与分解推导讲得透；个别记号（如轨迹分布符号）稍重。
- **价值**: ⭐⭐⭐⭐ 解决了对抗 RL 一个长期开放的基础问题并给出 minimax 极限，理论价值高；实用价值受限于 $K$ 指数依赖与缺乏实验。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning](potentially_optimal_joint_actions_recognition_for_cooperative_multi-agent_reinfo.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)

</div>

<!-- RELATED:END -->

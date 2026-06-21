---
title: >-
  [论文解读] Q-Learning with Fine-Grained Gap-Dependent Regret
description: >-
  [ICLR 2026][强化学习][model-free RL] 针对 episodic tabular MDP 的免模型强化学习，本文提出一套「把最优与次优状态-动作对分开统计访问频次」的细粒度分析框架，首次为 UCB-Hoeffding 证明了带单个间隙 $\Delta_h(s,a)$ 的细粒度 gap-dependent regret 上界，并据此修复了此前唯一的非 UCB 算法 AMB 在截断和鞅差条件上的两处缺陷，给出 ULCB-Hoeffding 与 Refined AMB 两个改进版本。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "model-free RL"
  - "gap-dependent regret"
  - "UCB-Hoeffding"
  - "多步自举"
  - "tabular MDP"
---

# Q-Learning with Fine-Grained Gap-Dependent Regret

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fE0RJto3Na](https://openreview.net/forum?id=fE0RJto3Na)  
**代码**: 随附补充材料（synthetic 实验）  
**领域**: 强化学习 / 理论 / Regret 分析  
**关键词**: model-free RL、gap-dependent regret、UCB-Hoeffding、多步自举、tabular MDP

## 一句话总结
针对 episodic tabular MDP 的免模型强化学习，本文提出一套「把最优与次优状态-动作对分开统计访问频次」的细粒度分析框架，首次为 UCB-Hoeffding 证明了带单个间隙 $\Delta_h(s,a)$ 的细粒度 gap-dependent regret 上界，并据此修复了此前唯一的非 UCB 算法 AMB 在截断和鞅差条件上的两处缺陷，给出 ULCB-Hoeffding 与 Refined AMB 两个改进版本。

## 研究背景与动机
**领域现状**：在 $S$ 状态、$A$ 动作、每幕 $H$ 步的 episodic tabular MDP 上，免模型方法（直接学值函数、内存随状态数线性增长）是实践主力。$K$ 幕、$T=KH$ 步下的 minimax regret 下界是 $\Omega(\sqrt{H^2SAT})$，多数 UCB 类算法都能匹配这一 $\sqrt{T}$ 型最坏情况界。

**现有痛点**：实践中当存在正的次优间隙（每个状态的最优动作比其他动作好一个 margin）时，算法表现远好于最坏情况界，于是有了 gap-dependent regret。但已有的免模型 gap-dependent 结果都很「粗」：Yang et al. (2021) 给 UCB-Hoeffding 证的是 $\tilde{O}(H^6SA/\Delta_{\min})$，依赖的是全局最小间隙 $\Delta_{\min}$ 和一个粗糙的 $SA/\Delta_{\min}$ 项，而不是逐个间隙 $\Delta_h(s,a)$，因此并不紧。

**核心矛盾**：要得到细粒度的界，关键是控制 Q 估计的「累积加权估计误差」，而最优与次优状态-动作对的访问模式天差地别——次优对一般只被访问 $\hat{O}(\log T)$ 次，最优对却可能被访问无穷多次。已有分析对所有 $(s,a)$ 一视同仁，必然把界放松、把对 $1/\Delta_{\min}$ 的依赖估得过于保守。另一边，唯一拿到细粒度界的非 UCB 算法 AMB（靠 ULCB + 多步自举）其实证明有漏洞，能否在非 UCB 框架下严格证出细粒度界存疑。

**本文目标**：回答这个公开问题——能否为免模型 RL 建立带单个间隙 $\Delta_h(s,a)$、且对 $1/\Delta_{\min}$ 依赖更优的严格细粒度 gap-dependent regret 上界？

**切入角度**：作者从「最优对 vs 次优对访问频次极不均衡」这个观察出发，主张在分析里把两类对**分开处理**，而不是统一放缩；这恰好对上了 model-based 方向已经做到的形如 $\tilde{O}\big(\sum_{\Delta_h(s,a)>0}\frac{1}{\Delta_h(s,a)}+\frac{|Z_{\mathrm{opt}}|}{\Delta_{\min}}+SA\big)\mathrm{poly}(H)$ 的界。

**核心 idea**：用一套「逐 $(s,a)$、逐步 $h$ 分别控制累积加权估计误差」的归纳分析框架替代统一放缩，从而把 regret 精确归因到每个间隙上。

## 方法详解

### 整体框架
论文本质是一篇分析框架论文，整体逻辑是「先证一个通用框架 → 用框架打通一个老算法 → 再用框架修一个有 bug 的算法」。

第一步建立**细粒度分析框架**：先用一个对任意算法都成立的恒等式把期望 regret 写成加权访问次数之和，再逐个状态-动作对、逐步把「累积加权估计误差」递归地往后传，最后对步数做归纳，得到访问次数的细粒度上界。第二步把这套框架套到经典的 **UCB-Hoeffding**（Jin et al. 2018）上，得到它的首个细粒度 gap-dependent 界（Theorem 3.1）。第三步转向唯一的非 UCB 算法 **AMB**：先指出它在算法设计（不当截断）和理论分析（违反鞅差条件）上的两处硬伤，再给出两个修复版——去掉多步自举、只留置信区间淘汰的 **ULCB-Hoeffding**，以及保留多步自举但补全无偏性证明的 **Refined AMB**。两个改进版都拿到了严格的细粒度界，且经验上都优于原始 AMB。

整篇方法没有可串成数据流水线的多阶段 pipeline，核心是数学分析与算法改写，故不配框架图。

### 关键设计

**1. 细粒度分析框架：把最优对与次优对分开统计访问频次**

针对「统一放缩导致界过松」这一痛点，框架的起点是 Lemma 3.1 给出的一个对任意算法都成立的恒等式：
$$\mathbb{E}[\mathrm{Regret}(T)] = \mathbb{E}\Big[\sum_{h=1}^{H}\sum_{s,a}\Delta_h(s,a)\,N^{K+1}_h(s,a)\Big],$$
其中 $\Delta_h(s,a):=V^\star_h(s)-Q^\star_h(s,a)$ 是次优间隙，$N^{K+1}_h(s,a)$ 是访问次数。于是控制 regret 就等价于控制加权访问次数，而后者又能由累积加权估计误差 $\sum_k \omega^k_h(Q^k_h-Q^\star_h)(s^k_h,a^k_h)$ 来界定（用到 $(Q^k_h-Q^\star_h)(s^k_h,a^k_h)\ge\Delta_h(s^k_h,a^k_h)$ 这一乐观性推论）。

关键创新在于**不再把所有 $(s,a)$ 一锅炖**：作者为每个状态-动作对、在每一步 $h'$ 单独定义递归权重 $\omega_h(k',h')$ 并保留各自的贡献项 $\sqrt{H^3\|\omega_h(\cdot,h')\|_\infty\|\omega_h(\cdot,h',s,a)\|_1\,\iota}$（Lemma 3.2、3.3），而非像 Yang et al. (2021) 那样一次性对所有对用 Cauchy–Schwarz。在做归纳时（Lemma 3.4），把状态-动作对划分为最优集 $Z_{\mathrm{opt},h}$（$\Delta_h(s,a)=0$）和次优集 $Z_{\mathrm{sub},h}$（$\Delta_h(s,a)>0$），对最优对逐步、对次优对跨步分别应用 Cauchy–Schwarz。这正好利用了「次优对只被访问 $\hat{O}(\log T)$ 次」的事实，从而把 regret 拆成逐个间隙的倒数和，而不是被全局 $1/\Delta_{\min}$ 主导。

**2. 用框架打通 UCB-Hoeffding：首个细粒度上界**

UCB-Hoeffding 本身用的是标准 Bellman 更新（步长 $\eta_t=(H+1)/(H+t)$、Hoeffding bonus $b_t=2\sqrt{H^3\iota/t}$）加贪婪选动作，并不需要改算法，难点全在分析。把设计 1 的框架套上去，Theorem 3.1 给出期望 regret 上界
$$O\Big(\sum_{h=1}^{H}\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\sum_{h=1}^{H}\frac{H^3\big(\sum_{t=h+1}^{H}\sqrt{|Z_{\mathrm{opt},t}|}\big)^2\log(SAT)}{\Delta_{\min,h}}+SAH^3\Big),$$
其中 $Z_{\mathrm{opt},h}=\{(s,a):\Delta_h(s,a)=0\}$。它能进一步松弛成更简洁的 $\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\frac{H^5|Z_{\mathrm{opt}}|\log(SAT)}{\Delta_{\min}}+SAH^3$。这比 Yang et al. (2021) 的 $\tilde{O}(H^6SA/\Delta_{\min})$ 全面更优：在只有单个次优三元组（且 $h=H$）的理想情形里降到 $\tilde{O}(H^5/\Delta_{\min})$，在所有间隙都等于 $\Delta_{\min}$ 的最坏情形里优雅退化、恰好匹配旧界；其对 $|Z_{\mathrm{opt}}|/\Delta_{\min}$ 的依赖还匹配了 Simchowitz & Jamieson (2019) 和 Xu et al. (2021) 的下界（差 $H$ 的多项式因子）。

**3. ULCB-Hoeffding：去掉多步自举、只留置信区间淘汰的 UCB 改进**

AMB 之所以难分析，根子在多步自举。作者先做一个「减法版」改进：ULCB-Hoeffding 同时维护 $Q^\star_h$ 的上界 $\overline{Q}^k_h$ 与下界 $\underline{Q}^k_h$，都用**标准 Bellman 更新**（上界加 bonus、下界减 bonus），不碰多步自举。它的探索靠两件事：一是维护候选动作集 $A^k_h(s)$，凡满足 $\overline{Q}^{k+1}_h(s,a)<\underline{V}^{k+1}_h(s)$ 的动作被判定次优并淘汰（因为此时存在 $a'$ 使 $Q^\star_h(s,a)\le\overline{Q}^{k+1}_h(s,a)<\underline{V}^{k+1}_h(s)\le\underline{Q}^{k+1}_h(s,a')\le Q^\star_h(s,a')$）；二是在候选集里选**置信区间最宽**的动作 $\arg\max_{a\in A^k_h(s)}(\overline{Q}^k_h-\underline{Q}^k_h)(s,a)$，即挑不确定性最大的动作去探索。因为更新规则回到了标准 Bellman，设计 1 的框架可以原样套用，Theorem 4.2 证明它拿到与 UCB-Hoeffding **完全相同**的细粒度界，同时 Theorem 4.1 保证最坏情况下仍是 $O(\sqrt{H^4SAT\iota})$。这说明「只靠 ULCB 原则、不需要多步自举」也能达到细粒度保证，反过来印证了框架的通用性。

**4. Refined AMB：修正截断与鞅差，证明多步自举无偏性的非 UCB 改进**

这一步直面 AMB 的两处硬伤。其一是**算法层的不当截断**：AMB 在多步自举更新里对 Q 估计在 $H$ 和 $0$ 处做截断（其 Algorithm 1 第 13–14 行），破坏了把 Q 估计与历史 V 估计相连的递归结构（其 Eq. (A.5)），而这一结构正是证乐观/悲观性的命脉。其二是**分析层违反鞅差条件**：AMB 把 Q 函数拆成「沿已定最优动作累积的奖励」与「从首个未定最优动作处收集的奖励」两个估计量，用 Azuma–Hoeffding 控偏差时，因忽略自举步的随机性，把估计量错误地中心化到其无条件期望、而非条件期望上，违反了不等式所需的鞅差条件。

Refined AMB 同时保留 ULCB 与多步自举，并做四点修正：(a) 把截断从 Q 估计挪到对应的 V 估计上，保住递归结构；(b) 严格证明多步自举的两个估计量之和是 $Q^\star$ 的无偏估计——靠的是对条件期望做**逐状态分解**、再对步数归纳（Theorem 4.3 / Appendix G.3 的核心技术）；(c) 把估计量中心化到真实条件期望，恢复鞅差条件；(d) 联合分析两个估计量的集中性，把置信区间收紧、bonus 常数减半。最终 Refined AMB 在非 UCB 设定下证出 $O\big(\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\frac{H^5|Z_{\mathrm{mul}}|\log(SAT)}{\Delta_{\min}}\big)$，其中 $Z_{\mathrm{mul}}$ 是「最优动作不止一个」的最优三元组集合，且经验上稳定优于原始 AMB。

## 实验关键数据

实验为纯合成环境验证，奖励 $r_h(s,a)\sim\mathrm{Unif}[0,1]$、转移核从概率单纯形均匀采样，对比 AMB / Refined AMB / UCB-Hoeffding / ULCB-Hoeffding 四个算法，绘制 $\mathrm{Regret}(T)/\log(K+1)$ 随 $K$ 的曲线（每算法 10 条轨迹，取中位数 + 10–90 分位带）。

### 主实验（四种规模下的相对表现）

| 规模 $(H,S,A,K)$ | 表现排序（regret 由低到高） | 关键现象 |
|------------------|------------------------------|----------|
| $(2,3,3,10^5)$ | UCB-Hoeffding < ULCB-Hoeffding ≈ Refined AMB < AMB | 除 AMB 外曲线随 $K$ 变平 |
| $(5,5,5,6\times10^5)$ | 同上 | 体现对数增长，符合细粒度理论 |
| $(7,8,6,5\times10^6)$ | 同上 | 改进版稳定超原始 AMB |
| $(10,15,10,2\times10^7)$ | 同上 | UCB-Hoeffding 整体最优 |

### 关键发现对照

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| UCB-Hoeffding | 整体 regret 最低 | 标准 Bellman + 框架分析，经验最佳 |
| ULCB-Hoeffding / Refined AMB | 两者相当、均优于 AMB | 修复后的两个改进版表现接近 |
| AMB（原版） | regret 曲线不变平 | bonus 常数 $c=2$（两估计量各做一次集中），界更松 |

### 关键发现
- 除 AMB 外，所有算法的 $\mathrm{Regret}(T)/\log(K+1)$ 曲线都随 $K$ 增大趋于平坦，说明 regret 呈对数增长，与细粒度 gap-dependent 理论一致。
- Refined AMB 把 bonus 常数从 $c=2$ 降到 $c=1$（联合集中只用一次不等式而非对两个估计量各用一次），这是它经验上压过原始 AMB 的直接原因。
- UCB-Hoeffding 在所有规模都表现最好，提示：在这类 tabular 任务里，简单的标准 Bellman + 贪婪未必输给复杂的多步自举，反而因 bonus 更省而更优。

## 亮点与洞察
- **「分开统计访问频次」是全文的题眼**：抓住「次优对只被访问 $O(\log T)$ 次、最优对可被访问无穷次」这个不对称，把统一放缩换成逐 $(s,a)$ 逐步的递归归纳，是从粗粒度走到细粒度的关键技术杠杆。
- **先用恒等式把 regret 转成访问次数**（Lemma 3.1，对任意算法成立）再下手，把「证 regret 界」干净地降维成「证访问次数界」，框架因此能从 UCB-Hoeffding 平滑迁移到 ULCB-Hoeffding。
- **认真复核前人证明**：作者精确指出 AMB 的截断破坏递归结构、以及中心化到无条件期望违反鞅差条件两处，并对症修复——这种「找 bug + 给可用替代」的工作对理论社区价值很高。
- **多步自举无偏性的逐状态条件期望分解**（Theorem 4.3）是可复用的技术，凡涉及多步自举的算法分析都可能用得上。

## 局限与展望
- 仅限 tabular、有限状态 episodic MDP；如何把细粒度框架推广到函数逼近 / 连续状态尚未触及。
- regret 界对 $H$ 的多项式依赖（如 $H^5$）仍可能偏松，与下界之间留有 $H$ 的多项式 gap，未必紧。
- 实验只在小规模合成 MDP 上做，缺少更大状态空间或真实任务的验证；「UCB-Hoeffding 经验最优」的结论在更复杂环境是否成立存疑。
- Refined AMB 的完整算法与无偏性证明都压在附录，正文只给 informal 版本（Theorem 4.3），实现细节需查 Appendix G。

## 相关工作与启发
- **vs Yang et al. (2021)**：他们给 UCB-Hoeffding 的是粗粒度 $\tilde{O}(H^6SA/\Delta_{\min})$，依赖全局 $SA/\Delta_{\min}$；本文换成逐个间隙 $\sum 1/\Delta_h(s,a)$，在所有情形都不劣、理想情形显著更优。
- **vs Xu et al. (2021) 的 AMB**：AMB 首个声称拿到细粒度界的非 UCB 算法，但证明有截断与鞅差两处缺陷；本文指出问题并给出 ULCB-Hoeffding（去自举）和 Refined AMB（修自举）两个严格可证的替代。
- **vs model-based 细粒度工作（Simchowitz & Jamieson 2019；Dann et al. 2021；Chen et al. 2025）**：它们早已拿到形如 $\sum 1/\Delta_h(s,a)+|Z_{\mathrm{opt}}|/\Delta_{\min}+SA$ 的细粒度界；本文把同等粒度首次带进**免模型**设定，并匹配了 UCB 类算法的下界。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个免模型 UCB-Hoeffding 细粒度 gap-dependent 界，并修复了 AMB 长期未被发现的证明缺陷。
- 实验充分度: ⭐⭐⭐ 仅合成环境小规模验证，作为理论论文够用但不充分。
- 写作质量: ⭐⭐⭐⭐ 框架—算法—修复三段逻辑清晰，但大量细节压在附录。
- 价值: ⭐⭐⭐⭐ 给免模型 gap-dependent 分析提供了可迁移的通用框架与可复用的自举无偏性技术。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[ICML 2026\] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs](../../ICML2026/reinforcement_learning/data-_and_variance-dependent_regret_bounds_for_online_tabular_mdps.md)
- [\[ICLR 2026\] Bridging the Performance-Gap Between Target-Free and Target-Based Reinforcement Learning](bridging_the_performance-gap_between_target-free_and_target-based_reinforcement_.md)

</div>

<!-- RELATED:END -->

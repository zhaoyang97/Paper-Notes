---
title: >-
  [论文解读] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization
description: >-
  [ICLR 2026][学习理论][Q-learning] 本文为采用「线性衰减到零」（LD2Z，$\eta_{t,n}=\eta(1-t/n)$）及其幂律推广（PD2Z-$\nu$，$\eta_{t,n}=\eta(1-t/n)^\nu$）学习率的 Q-learning 给出了首套完整渐近理论——包括尖锐的非渐近误差界、尾部 Polyak-Ruppert 平均估计量的中心极限定理、以及偏序和过程的强不变原理（时间一致高斯逼近），从理论上解释了为何这种"两阶段"步长能兼得常数步长的快速忘记初值与多项式步长的渐近收敛保证。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "强化学习理论"
  - "Q-learning"
  - "学习率调度"
  - "LD2Z"
  - "非渐近误差界"
  - "强不变原理"
---

# Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WjEAMyLDoh](https://openreview.net/forum?id=WjEAMyLDoh)  
**代码**: 论文提及有 GitHub 仓库（未给出具体地址）  
**领域**: 学习理论 / 强化学习理论  
**关键词**: Q-learning, 学习率调度, LD2Z, 非渐近误差界, 强不变原理

## 一句话总结
本文为采用「线性衰减到零」（LD2Z，$\eta_{t,n}=\eta(1-t/n)$）及其幂律推广（PD2Z-$\nu$，$\eta_{t,n}=\eta(1-t/n)^\nu$）学习率的 Q-learning 给出了首套完整渐近理论——包括尖锐的非渐近误差界、尾部 Polyak-Ruppert 平均估计量的中心极限定理、以及偏序和过程的强不变原理（时间一致高斯逼近），从理论上解释了为何这种"两阶段"步长能兼得常数步长的快速忘记初值与多项式步长的渐近收敛保证。

## 研究背景与动机
**领域现状**：Q-learning 作为估计 MDP 最优策略的经典 model-free 方法，其统计性质（渐近/非渐近误差界、中心极限定理、泛函中心极限定理）已被大量研究。但这些理论工作几乎一律假设步长是**常数**（$\eta_t\equiv\eta$）或**多项式衰减**（$\eta_t=\eta t^{-\alpha}$）。

**现有痛点**：这两类步长各有硬伤。常数步长收敛快、能迅速摆脱初始值的影响，但 Q-learning 链最终会收敛到 $Q^\star$ 附近的一个**平稳分布**——即存在不可忽略的**渐近偏差**，必须靠 jackknife 之类的手段才能消除。多项式衰减步长理论上漂亮（Polyak-Ruppert 平均下能得到渐近正态），却收敛**慢得令人发指**：从初值出发的"忘记速度"只有 $\exp(-ct^{1-\alpha})$，离 $1$ 越远（即 $\alpha$ 越大）越慢。

**核心矛盾**：「快速忘记初值」与「无渐近偏差」之间存在 trade-off。常数步长占了前者、多项式步长占了后者，没有一种被理论充分分析的步长同时拥有两者。

**本文目标**：实证上，深度学习社区近年常用的 LD2Z 步长（BERT、LLaMA 等都在用，常见的 "knee schedule" 也是先 warm-up 再 LD2Z）表现极好，但它的渐近/统计性质几乎是空白，尤其在 Q-learning 场景下。本文要回答：（1）LD2Z 及其推广在 Q-learning 下的非渐近收敛速率是多少；（2）能否做合法的统计推断（CLT + bootstrap）。

**切入角度**：作者不直接分析 $\nu=1$ 的 LD2Z，而是先退一步考虑更一般的幂律族 PD2Z-$\nu$（$\eta_{t,n}=\eta(1-t/n)^\nu$，LD2Z 是 $\nu=1$ 的特例）。这一步使他们能顺带回答"为什么线性（$\nu=1$）就够了、要不要随迭代调 $\nu$"这一更深的问题。

**核心 idea**：把 Q-learning 当作以 Bellman 方程为目标的随机逼近（SA），对 PD2Z-$\nu$ 步长**逐级建立**理论——非渐近矩界 → 尾部 PR 平均 CLT → 强不变原理，证明这族步长具有"两全其美"（best-of-both-worlds）属性。

## 方法详解

### 整体框架
本文是一篇**纯理论**工作，没有可画 pipeline 的算法流程，"方法"即一条层层递进的证明链：在标准正则性假设（奖励 $p$ 阶矩有界 + 最优策略附近的局部吸引盆条件）下，针对同步 Q-learning 迭代

$$Q_{t,n}=(1-\eta_{t,n})Q_{t-1,n}+\eta_{t,n}\widehat{B}_tQ_{t-1,n},\qquad \widehat{B}_t\ \text{为经验 Bellman 算子}$$

配上 PD2Z-$\nu$ 步长，依次产出三层结果：(1) 任意 $p\ge 2$ 阶矩的**非渐近误差界**，从中读出"瞬态快衰 + 收敛期慢衰"的两段式速率；(2) 在此界基础上提出**尾部 Polyak-Ruppert 平均**估计量并证其**渐近正态**；(3) 给出偏序和过程的**强不变原理**（时间一致高斯逼近），为 bootstrap 推断铺路。三层结果环环相扣：后一层都建立在前一层的矩控制之上，术语与符号（$Q^\star$、$Z_t$ Bellman 噪声、$G=I-\gamma H_{\pi^\star}$）贯穿始终。

### 关键设计

**1. PD2Z-$\nu$ 的两段式非渐近误差界：把"先快后稳"写成可量化的速率**

这是全文的地基（Theorem 3.1）。它针对的痛点是：人们只在实验里看到 LD2Z"先猛降再稳住"，却没有解析刻画。作者证明，在 $\eta<\frac{2(1-\gamma)}{(1-\gamma)^2+2(p-1)\gamma^2}$ 时，第 $t$ 步迭代满足
$$\|Q_{t,n}-Q^\star\|_p\ \le\ \exp\!\big(-c_3\eta t(1-n^{-1})^\nu\big)\,|Q_0-Q^\star|\ +\ \begin{cases}\sqrt{C_1}\,\sqrt{\eta_{t,n}}, & t\le n-\tfrac{2}{(c_3\eta)^{1/(\nu+1)}}n^{\nu/(\nu+1)}\\[4pt]\sqrt{C_2}\,n^{-\frac{\nu}{2(\nu+1)}}, & t> n-\tfrac{2}{(c_3\eta)^{1/(\nu+1)}}n^{\nu/(\nu+1)}\end{cases}$$
关键在于**两个 regime、两种速率**。在**瞬态期**（$t\lesssim n^c$，$c<1$），$\eta_{t,n}\asymp 1$，步长行为像常数步长，误差随 $\sqrt{\eta_{t,n}}$ 走，偏差快速消散；越过阈值进入**收敛期**后，误差锁定在 $n^{-\nu/(2(\nu+1))}$。最尖锐的对比是初值忘记速度：PD2Z-$\nu$ 对所有 $t$ 都以 $\exp(-ct)$ 指数级忘记初值（Remark 3.2 给出 $t=n$ 时 $\|Q_{n,n}-Q^\star\|_p\lesssim\exp(-n/4)|Q_0-Q^\star|+n^{-1/4}$），而多项式步长（Theorem 3.3）只有 $\exp(-ct^{1-\alpha})$。两者收敛期主项相近（$n^{-\nu/(2(\nu+1))}$ vs $n^{-\alpha/2}$），但 LD2Z 把初值影响压得低得多——这就是它实证上更快的理论根源。Remark 3.1 进一步给出样本复杂度 $N(\epsilon)=O\!\big(\frac{1}{(1-\gamma)^2}\log\frac{|Q_0-Q^\star|}{\epsilon}+\frac{1}{(1-\gamma)^{4+2/\nu}\epsilon^{2(\nu+1)/\nu}}\big)$，大 $\nu$ 时逼近 Li et al. (2024a) 的速率，而本文用的是远更弱的"奖励 $p$ 阶矩有限"假设（对手假设奖励 $\in[0,1]$）。

**2. 最优 $\nu$ 的标度：解释为什么 $\nu=1$（纯线性）就够用**

退到 PD2Z-$\nu$ 一般族后，自然要问该不该精调 $\nu$。Corollary 3.2 把界整理成 $\exp(-c_3\eta(1-n^{-1})^\nu t)|Q_0-Q^\star|+O(\sqrt{\eta_{t,n}}\vee n^{-\nu/(2(\nu+1))})$，并指出在 $t=n$ 处右端关于 $\nu$ 的最小值取在 $\nu\asymp\log_2\log n$。其机制（Remark 3.3）是一个 trade-off：常数 $C_2(c_3,\nu,2)$ 随 $\nu$ 增大，而衰减项 $n^{-\nu/(2(\nu+1))}$ 随 $\nu$ 减小，二者平衡给出 $\log\log n$ 这个**增长极慢**的阈值。$\log_2\log n$ 在任何现实 $n$ 下都几乎是常数，这就从理论上**正当化了固定、与迭代无关的 $\nu$**——尤其是 $\nu=1$ 的 LD2Z。Corollary 3.4 进一步把 $\nu=1$ 代入给出干净的 LD2Z 专用界（瞬态 $O(\sqrt{\eta_{t,n}})$、收敛期 $O(n^{-1/4})$，阈值在 $n-\frac{2}{\sqrt{c\eta}}\sqrt{n}$）。

**3. 尾部 Polyak-Ruppert 平均 + 中心极限定理：绕开常数步长段的偏差**

普通的 PR 平均（对全部 $n$ 步取平均）在 LD2Z 下**会失效**。作者把全程平均拆成 $A_n+B_n+C_n$ 三段（前半 $t\le n/2$、中段、末段 $t\in[n-\sqrt{n},n]$）：由于 $t\le n/2$ 时 $\eta_{t,n}\ge\eta/2^\nu$ 仍是"常数级"，$A_n$ 根本不收敛到 $Q^\star$、更谈不上高斯性，除非 $B_n$ 的渐近分布恰好抵消 $A_n$（不可指望），全程平均的误差反而远大于 $Q^\star$（Figure 4 实证）。因此作者只对**最后一批最新迭代**做平均，定义尾部 PR 估计量
$$\bar{Q}_n=\frac{1}{\lfloor cn^{\nu/(\nu+1)}\rfloor}\sum_{t=n-\lfloor cn^{\nu/(\nu+1)}\rfloor+1}^{n}Q_{t,n},$$
并证明（Theorem 3.5）存在与 $n$ 无关的正定 $\Sigma$ 使 $n^{\nu/(2(\nu+1))}(\bar{Q}_n-Q^\star)\xrightarrow{w}N(0,\Sigma)$。据作者所知，这是 SA 文献里"尾部 PR 平均"的首个结果。代价是 $\Sigma$ 的解析表达极难处理，无法直接估计——这就把球传给了第 4 点的 bootstrap。

**4. 强不变原理：用协方差匹配的高斯过程做时间一致逼近，撑起 bootstrap**

为了绕过 $\Sigma$ 不可解析的困难、并得到比泛函 CLT 更尖锐的逼近，作者建立偏序和过程的**强不变原理**（强高斯逼近）。由于 $(Q_{t,n})$ 是**非平稳**序列，标准布朗运动逼近失效；他们改用一个专门匹配协方差结构的非平稳高斯过程：取 $\aleph_t$ 为 i.i.d. 中心高斯、$\mathrm{Cov}(\aleph_t)=\mathrm{Cov}(Z_t)$，递推
$$Y_t=(I-\eta_{t,n}G)Y_{t-1}+\eta_{t,n}\aleph_t,\qquad G=I-\gamma H_{\pi^\star}.$$
Theorem 4.1 证明在某概率空间上可构造 $Q^c_{t,n}\overset{D}{=}Q_{t,n}$ 使 $\max_{k_n\le t\le n}\big|\sum_{l=t}^n(Q^c_l-Q^\star-Y_l)\big|_\infty=o_P(n^{1/p+1/(\nu+1)})$，这是 PD2Z-$\nu$ 步长下偏序和过程的**首个**强高斯逼近（只在尾段 $[k_n,n]$ 上，与尾部 PR 平均呼应）。Theorem 4.2 对多项式步长 $\tilde\eta_t=\eta t^{-\beta}$ 也给出全程逼近 $o_P(n^{1/p}\log n)$，且作者论证这种**协方差匹配**逼近比泛函 CLT（只在渐近意义下匹配协方差）更尖锐：在 vanilla SGD 玩具例里，$Y^G_t$ 与真迭代的协方差结构**逐时刻完全相同**，而布朗运动逼近只能渐近匹配。最实用的是 (4.4)：尾段最大偏序和的分布可被 $Y_t$ 的对应量一致逼近，而 $\Gamma$ 与 $H_{\pi^\star}$ 可借 $\widehat{B}_t$、$BQ^\star=Q^\star$ 估出，于是只需**并行跑多条独立的 $Y_t$ 链**即可实现 Gaussian bootstrap，对 $\bar{Q}_n$ 做时间一致推断。

## 实验关键数据

实验为验证理论的数值仿真，统一在 $4\times4$ FrozenLake 滑动网格世界（$\gamma=0.1$，特殊状态奖励 $r(A)=10$、$r(B)=5$，越界 $-1$，意图方向以 0.9 概率执行）上做 Monte-Carlo 重复。

### 主实验（不同学习率对比，§5.2）

| 对比项 | 设置 | 现象 |
|--------|------|------|
| LD2Z vs 多项式衰减 vs 常数 | $\eta=0.05$，$n=5000$，$B=1000$ 链 | LD2Z 大幅优于多项式衰减（Figure 1）；常数步长收敛到带固定偏差的平稳分布，PD2Z-$\nu$ 全程优于常数 |
| 瞬态误差 $|Q_{t,n}-Q^\star|_\infty$ | $1000\le t\le n$，画均值±1 标准差 | 增大 $\nu$（$\nu=2,3$）在 $t<n$ 时**略微**降低误差 |
| 终点误差 $|Q_{n,n}-Q^\star|_\infty$ | $n\in\{500,\dots,2500\}$ | $\nu\in\{1,2,3\}$ 终点误差**几乎一致**，印证 PD2Z 族的稳定性，正当化 $\nu=1$ |

### 时间一致逼近与 CLT 验证（§5.3–5.4）

| 实验 | 配置 | 关键发现 |
|------|------|----------|
| 强不变原理 Q–Q 图（Figure 3 左） | LD2Z $\eta_t=0.05(1-t/n)$，$n=5000$，$B=500$ | $\max_{k_n\le t\le n}|\sum(Q^c_l-Q^\star)|_\infty$ 的分位数与高斯逼近 $\sum Y_l$ 高度吻合 |
| 多项式步长逼近（Figure 3 右） | $\eta_t=0.05t^{-0.65}$ | 协方差匹配逼近优于布朗运动（泛函 CLT）逼近，后者在 sup-norm 一致逼近上明显次优 |
| 尾部 PR vs 普通 PR（Figure 4） | $n\in\{1000,\dots,5000\}$，$B=1000$ | $\bar{Q}_n$（尾部）的 $L_\infty$ 误差显著低于 $\tilde{Q}_n$（全程），实证全程平均因常数步长段而失效 |

### 关键发现
- **两段式速率是真实存在的**：误差先以指数速率快速摆脱初值、再在收敛期稳定到 $n^{-\nu/(2(\nu+1))}$，与 Figure 1 的"先猛降后稳住"完全对应。
- **$\nu$ 几乎不影响终点精度**：理论上最优 $\nu\asymp\log_2\log n$ 增长极慢，实验上 $\nu=1,2,3$ 终点误差一致——线性衰减（LD2Z）就是实践中的甜点。
- **平均要平尾巴而非平全程**：常数步长段的不收敛会污染全程 PR 平均，尾部 PR 平均才能恢复渐近正态。

## 亮点与洞察
- **"先退一步到 PD2Z-$\nu$ 再回收 LD2Z"的证明策略很漂亮**：通过引入幂律参数 $\nu$，作者不仅证了 LD2Z，还顺手解释了"为什么不用调 $\nu$"（最优 $\nu\asymp\log\log n$ 几乎是常数），把一个工程经验上升为定理。
- **协方差匹配的非平稳高斯过程**是核心技术武器：用 $Y_t=(I-\eta_{t,n}G)Y_{t-1}+\eta_{t,n}\aleph_t$ 逐时刻匹配真迭代的协方差，比布朗运动/泛函 CLT 尖锐，且天然适配 LD2Z 的非平稳性。这一构造可迁移到其他非平稳 SA/SGD 的强逼近分析。
- **尾部 PR 平均**作为一个独立贡献：它揭示了"什么时候不能对全程做 PR 平均"，对所有用两阶段/衰减到零步长的 SA 推断都有警示意义。
- **弱假设**：只要奖励 $p$ 阶矩有限，而非主流文献的奖励有界 $[0,1]$，因此用的是 Burkholder 不等式（无界下尖锐）而非 Freedman 不等式。

## 局限与展望
- **作者承认**：LD2Z 需要**预先指定总步数 $n$**，因此主要适用于**离线 RL**（数据集预先收集、$n$ 已知）；不过他们证明在 $n-n_0\le\alpha\sqrt{n}$ 的误指定下结论仍成立。把 LD2Z/PD2Z-$\nu$ 推广到**在线 RL** 是公开方向。
- **$\Sigma$ 不可解析**：尾部 PR 平均的渐近协方差表达极难处理，只能靠 bootstrap 间接推断；bootstrap 算法本身的理论保证与 Berry-Esseen 界（量化 CLT 收敛速度）尚未给出，是明确的 future work。
- **自己发现的局限**：实验仅在 $\gamma=0.1$ 的小 $4\times4$ 网格上做，$\gamma$ 接近 1（长视界）时常数 $c_3$、样本复杂度中的 $(1-\gamma)^{-(4+2/\nu)}$ 会急剧恶化，理论在大 $\gamma$ 下的实际紧度未验证。强逼近仅覆盖尾段 $[k_n,n]$（LD2Z 时即 $[n-\sqrt{n},n]$），不是全程。

## 相关工作与启发
- **vs Goldreich et al. (2025)**：他们首次分析了 LD2Z 在强凸 SGD 下的性质，但只给末迭代 $Q_{n,n}$ 的 $L_2$ 控制，且**不适用于 Q-learning**；本文把分析推广到 Q-learning，并补齐了 CLT 与强不变原理。
- **vs Li et al. (2023b) / Chen et al. (2020b)（多项式步长）**：这些工作只对多项式衰减步长建立泛函 CLT；本文指出泛函 CLT 在 sup-norm 一致逼近上次优，改用协方差匹配的强逼近，且 LD2Z 忘记初值的速度 $\exp(-ct)$ 远快于多项式的 $\exp(-ct^{1-\alpha})$。
- **vs Li et al. (2024a)（样本复杂度）**：大 $\nu$ 时本文速率逼近其结果，但本文用更弱的 $p$ 阶矩假设（对方假设奖励有界），有限 $\nu$ 的 gap 正源于此。
- **vs Bonnerjee et al. (2024)（协方差匹配高斯逼近）**：本文沿用其协方差匹配思想，但推广到 Q-learning 迭代特有的非平稳结构。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Q-learning 下 LD2Z/PD2Z-$\nu$ 的首套完整渐近理论，尾部 PR 平均与协方差匹配强逼近均为新贡献。
- 实验充分度: ⭐⭐⭐⭐ 仿真扎实地验证了每条理论，但仅限单一小网格、单一 $\gamma$，缺大规模/大视界检验。
- 写作质量: ⭐⭐⭐⭐⭐ 逐级递进、Remark 与 Corollary 把直觉讲透，理论动机清晰。
- 价值: ⭐⭐⭐⭐⭐ 把工程上广泛使用却无理论支撑的 LD2Z 步长落地为可推断的统计理论，并给出 bootstrap 实操路径。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICML 2026\] Performative Learning Theory](../../ICML2026/learning_theory/performative_learning_theory.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Convergence of Regret Matching in Potential Games and Constrained Optimization
description: >-
  [ICLR2026][优化/理论][遗憾匹配] 本文首次证明了遗憾匹配 +（RM+）在「单纯形乘积上的约束优化」（含势博弈作为特例）中是一个**可靠且快速的一阶优化器**——$O_\epsilon(1/\epsilon^4)$ 步收敛到 $\epsilon$-KKT 点；与此同时反向证明原始遗憾匹配（RM）在二人同利益博弈里都可能要指数步才能收敛，给出了 RM 与 RM+ 之间首个最坏情况（且是指数级）的分离。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "遗憾匹配"
  - "势博弈"
  - "约束优化"
  - "收敛性分析"
  - "无遗憾学习"
---

# Convergence of Regret Matching in Potential Games and Constrained Optimization

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EOV1q1U23N](https://openreview.net/forum?id=EOV1q1U23N)  
**代码**: 待确认  
**领域**: 优化理论 / 在线学习 / 博弈论  
**关键词**: 遗憾匹配, 势博弈, 约束优化, 收敛性分析, 无遗憾学习

## 一句话总结
本文首次证明了遗憾匹配 +（RM+）在「单纯形乘积上的约束优化」（含势博弈作为特例）中是一个**可靠且快速的一阶优化器**——$O_\epsilon(1/\epsilon^4)$ 步收敛到 $\epsilon$-KKT 点；与此同时反向证明原始遗憾匹配（RM）在二人同利益博弈里都可能要指数步才能收敛，给出了 RM 与 RM+ 之间首个最坏情况（且是指数级）的分离。

## 研究背景与动机

**领域现状**：遗憾匹配（regret matching, RM）由 Hart & Mas-Colell (2000) 提出，思想上可追溯到 Blackwell (1956) 的可达性框架，是无遗憾在线学习里最基础的算法之一：每个动作以「它累积的非负遗憾」成正比的概率被选中，既无超参、又对尺度不变。它的改进版 **RM+**（Tammelin 2014）只多做一件事——每一步把遗憾向量里的负分量截断到 $0$——却在实践中显著更强，是扑克（Libratus / DeepStack）乃至暗棋超人 AI 的核心引擎。在二人零和博弈里，无遗憾 ⇒ 平均策略收敛到 minimax/Nash 均衡，这条线的理论已经相当成熟。

**现有痛点**：一旦走出二人零和的舒适区，几乎什么都不知道。「RM 在势博弈里是否收敛到 Nash 均衡」这个问题被提出来已经悬而未决二十年（Kleinberg et al. 2009; Marden et al. 2007）。更尴尬的是，Tewolde et al. (2025) 的实验发现 RM 家族（尤其 RM+）在一批约束优化 benchmark 上表现极好，明显打败梯度下降类算法——但没有任何理论能保证它哪怕渐近收敛到约束优化的 KKT 点。

**核心矛盾**：障碍在于「无遗憾」和「收敛到 KKT 点」在非凸问题里本质上是两件不相干的事。本文 Proposition 3.2 直接给了一个反例：存在一段迭代序列，它对观测到的梯度遗憾恰好为零，但每一个点的 KKT gap 都是 $\Omega(1)$。也就是说，单靠无遗憾性根本推不出收敛。雪上加霜的是，RM+ 不具备梯度下降那种「一步改进」（one-step improvement）性质——哪怕把它初始化在 KKT 点附近，它也可能一步就严重 overshoot；而且它无超参，没法像 GD 那样靠调学习率来兜底。

**本文目标**：在「单纯形乘积 $X=\Delta(A_1)\times\cdots\times\Delta(A_n)$ 上最大化一个 $L$-光滑函数 $u$」这个统一框架下（势博弈对应 $u$ 多线性的特例），刻画 RM 与 RM+ 收敛到 $\epsilon$-KKT 点 / $\epsilon$-Nash 均衡所需的迭代步数，并分清两者的本质差别。

**切入角度**：作者的关键观察是——尽管遗憾和 KKT gap 一般无关，但在「RM+ 自己生成的迭代轨迹」这个特定设定里，它们以一种出人意料的方式纠缠在一起。具体说，RM+ 的遗憾向量 $\ell_2$ 范数是**单调上升**的，且每步至少涨 $\text{KKTGap}^2$；而当这个范数足够大时，RM+ 又恰好恢复了一步改进性质。

**核心 idea**：把收敛速度参数化为「累积遗憾的增长率」的函数——遗憾长得越慢，收敛越快。由此把 RM+ 的无遗憾性（$\sqrt T$ 遗憾）直接翻译成 $O_\epsilon(1/\epsilon^4)$ 的收敛速度；而 RM 之所以可能指数级慢，正是因为它在「好动作的遗憾长期为负」时会被卡死。

## 方法详解

### 整体框架

全文是一篇纯理论论文，没有需要画 pipeline 的算法流程，核心是一条「遗憾 ↔ KKT gap」的分析主线。设定如下：把约束优化问题 $\max_{x\in X} u(x)$ 看作 $n$ 个「玩家」各自控制一个单纯形 $\Delta(A_i)$、各自观测对应那块梯度 $\nabla_{x_i} u(x)$ 的协同优化；要最小化的量是 **KKT gap**

$$\text{KKTGap}(x) := \max_{x'\in X}\langle x'-x,\nabla u(x)\rangle = \sum_{i=1}^n \text{BRGap}_i\big(x_i,\nabla_{x_i}u(x)\big),$$

即所有玩家最优响应间隙（best-response gap）之和；KKT gap 为零等价于一阶平稳点。当 $u$ 多线性时，这正好退化成势博弈里求 Nash 均衡。两个被分析的算法只差一个截断：每轮观测到瞬时遗憾向量后，RM 用 $\theta^{(t)}=[r^{(t-1)}]_+$ 归一化出策略、但累积遗憾 $r^{(t)}$ 保留负分量；**RM+** 则把整个累积遗憾截断到非负：$r^{(t)}=[r^{(t-1)}+u^{(t)}-\langle x^{(t)},u^{(t)}\rangle\mathbf 1]_+$。论文要回答的是：在同时更新（simultaneous）和交替更新（alternating，类似坐标下降）两种模式下，各需要多少步收敛、RM 和 RM+ 差在哪。

后面四个关键设计，前两个解决势博弈（多线性、好分析），第三个把分析推广到一般非凸约束优化（最难的一步），第四个反向给 RM 一个指数下界、坐实两者的分离。

### 关键设计

**1. RM+ 的一步改进引理：把效用提升下界成「最优响应间隙的平方 / 遗憾范数」**

在势博弈（效用对单个玩家是线性的）里，RM+ 有一个干净的单调改进性质（Lemma 3.3）：对任意非负遗憾 $r$、$x=r/\|r\|_1$，更新后 $x'=r'/\|r'\|_1$ 满足

$$\langle x'-x,u\rangle \ge \frac{1}{\|r'\|_1}\Big(\max_{a}u[a]-\langle x,u\rangle\Big)^2 = \frac{1}{\|r'\|_1}\text{BRGap}(x,u)^2.$$

直观含义：只要当前策略还不是对 $u$ 的最优响应，效用就严格上升；而且只要遗憾范数 $\|r'\|_1$ 不太大，这个提升就很可观（正比于最优响应间隙的平方）。这个引理对任意非负初始遗憾都成立——这是 RM+ 截断带来的不变性，RM 没有。它直接说明 RM+「不会在不改善的情况下空转」。

**2. 用遗憾增长率参数化收敛速度：把 Nash 收敛和无遗憾性绑在一起**

把上面的一步改进对势函数做望远镜求和，就得到 Theorem 3.4 / 1.2 的核心结论：如果每个单纯形上的遗憾增长至多为 $T^\alpha$（$\alpha\in[0,1/2]$），那么 $\epsilon$-lazy 交替 RM+ 在

$$1+\frac{(C(n,m)\Phi_{\text{range}})^{\beta}}{\epsilon^{2\beta}}\ \text{步内收敛到 }\epsilon\text{-Nash},\quad \beta=\tfrac{1}{1-\alpha}$$

（$\Phi_{\text{range}}$ 是势函数的值域）。这里关键的转折是：RM+ 总能保证遗憾 $\le\sqrt{mT}$，即 $\alpha=1/2$，代入立刻得到主定理 **Theorem 1.1 的 $O_\epsilon(1/\epsilon^4)$**（对应 $T^{-1/4}$ 速率）。这个结果之所以「surprising」，是因为它**第一次让遗憾去支配 Nash 均衡的收敛速度**：以往大家只知道遗憾驱动收敛到粗相关均衡（CCE），现在证明它也驱动收敛到 Nash。一个推论是——若 RM+ 没有无遗憾性、遗憾线性增长 $\Omega(t)$，由 $\sum_t 1/t=\Theta(\log T)$ 只能证出指数界；反之若遗憾只长成常数（$\alpha=0$），速率直接提升到 $T^{-1/2}$。

**3. 非凸 / 同时更新下的「条件一步改进 + 遗憾范数单调增」：保证永不卡死**

势博弈之外，$u$ 不再多线性，RM+ 的一步改进彻底失效——哪怕初始化在 KKT 点附近也可能 overshoot。作者的破局点是两条引理的耦合。其一（Lemma 3.7）：用二次光滑界 $u(x')\ge u(x)+\langle\nabla u(x),x'-x\rangle-\tfrac L2\|x-x'\|_2^2$，证明当遗憾范数**足够大**时（$\|r'\|_2\ge\max\{2m,9mL\}$）才有一步改进 $u(x')-u(x)\ge\tfrac{1}{2\|r'\|_1}\text{KKTGap}^2$——注意这里是「范数太小才是障碍」，和势博弈情形正好相反。其二（Lemma 3.8）是最关键的胶水：

$$\|r^{(t)}\|_2^2 \ge \|r^{(t-1)}\|_2^2 + \|[g^{(t)}]_+\|_2^2 \ge \|r^{(t-1)}\|_2^2 + \text{KKTGap}(x^{(t)})^2,$$

即遗憾向量的 $\ell_2$ 范数**单调不减，且每步至少涨 KKT gap 的平方**。两者一拼就排除了「RM+ 永远在小范数区域里打转」的可能：只要还没收敛（KKT gap 大），遗憾范数就会迅速涨大，一旦涨过阈值，Lemma 3.7 的一步改进就接管。由此得到一般 $L$-光滑函数上的 $O_\epsilon(1/\epsilon^4)$（Theorem 3.9），并进一步覆盖对称势博弈的同时更新（Corollary 3.10）、多单纯形（Corollary 3.11，需把遗憾初始化在阈值 $\max\{2\sqrt{m_i},9\sqrt{m_i}L\}\mathbf 1$）。如果坚持用最常见的「零初始化、无超参」版本，作者只能保证较差的 $O_\epsilon(1/\epsilon^8)$（Theorem 3.12）——这道出了「参数无关」的代价。

**4. RM 的指数下界：好动作的遗憾被压成「越来越负」而长期停滞**

与 RM+ 形成尖锐对比，作者构造了一类二人 $m\times m$ 同利益博弈（势博弈的特例），证明 RM（无论同时还是交替）需要 $m^{\Omega(m)}$ 步才能收敛到 $\tfrac{1}{2m}$-Nash 均衡（Theorem 4.4 / 1.3），且这个下界不仅对对抗性初始化成立，对最常用的「均匀随机初始化」也成立。病根在于 RM 的一步改进是**有条件**的（Lemma 3.6）：只有当某个最优响应动作的累积遗憾恰好非负时，效用才改进。而在构造的博弈里，唯一「足够好」的动作，其遗憾要等很久——指数级地久——才会重新浮回正值；在那之前 RM 完全停滞、毫无进展。与此同时 RM 仍以 $T^{-1/2}$ 收敛到 CCE（因为它始终无遗憾），于是得到一个反差强烈的推论（Corollary 1.4）：存在一类二人势博弈，RM 收敛到 $\epsilon$-CCE 只要 $O_\epsilon(1/\epsilon^2)$ 步，收敛到 $\epsilon$-Nash 却要 $\exp(\Omega(1/\epsilon))$ 步。这是 RM 与 RM+ 之间**首个最坏情况分离**，而且是指数级的——在零和博弈里两者速率都是 $T^{-1/2}$、并无可证的分离。

### 损失函数 / 训练策略

论文还给出一个加速手段：**带折扣的 RM+（DRM+）**。每轮把遗憾向量乘上折扣因子 $\alpha^{(t)}=1-\gamma$（几何折扣，$\gamma\in(0,1)$ 固定）。折扣让遗憾范数被钉在 $\sqrt{m/\gamma}$ 以内（对应关键设计 2 里 $\alpha=0$ 的「常数遗憾」情形），同时仍保有一步改进性质，于是 $\epsilon$-lazy 交替 DRM+ 把收敛速率从 $T^{-1/4}$ 提升到 $T^{-1/2}$（Corollary 3.5，约 $1+\tfrac{m\Phi_{\text{range}}}{\epsilon^2\sqrt\gamma}$ 步）。这是首次证明折扣相对非折扣 RM+ 有「可证的最坏情况改进」。

## 实验关键数据

这是纯理论论文，没有大规模实证实验，仅有一张示意图（Figure 1）说明在构造的二人同利益博弈里 RM 的 KKT gap 在十万步后仍居高不下、RM+ 几百步即收敛。下面把核心「数据」整理为各定理给出的收敛复杂度。

### 主结果：RM+ 收敛速度

| 设定 | 算法 / 条件 | 复杂度（到 $\epsilon$-KKT / $\epsilon$-Nash） | 出处 |
|------|------------|---------------------------------------------|------|
| 单纯形乘积约束优化 | RM+（一般情形） | $O_\epsilon(1/\epsilon^4)$ | Thm 1.1 |
| 同上 | RM+，遗憾增长 $\le T^\alpha$ | $O_\epsilon(1/\epsilon^{2/(1-\alpha)})$ | Thm 1.2 |
| 同上 | RM+，遗憾有界（$\alpha=0$） | $O_\epsilon(1/\epsilon^2)$ | Thm 1.2 |
| 势博弈 | $\epsilon$-lazy 交替 RM+ | $1+(m\Phi_{\text{range}})^2/\epsilon^4$ | Thm 3.4 |
| 势博弈 | $\epsilon$-lazy 交替 **DRM+**（折扣 $1-\gamma$） | $1+m\Phi_{\text{range}}/(\epsilon^2\sqrt\gamma)$ | Cor 3.5 |
| 多单纯形（零初始化、无超参） | RM+ | $O_\epsilon(1/\epsilon^8)$ | Thm 3.12 |

### 对照：RM vs RM+ 的分离

| 收敛目标 | RM | RM+ | 说明 |
|----------|-----|-----|------|
| $\epsilon$-CCE（平均分布） | $O_\epsilon(1/\epsilon^2)$ | $O_\epsilon(1/\epsilon^2)$ | 两者都无遗憾，CCE 都快 |
| $\epsilon$-Nash（个体迭代） | $\exp(\Omega(1/\epsilon))$ | $O_\epsilon(1/\epsilon^4)$ | 二人同利益博弈即可指数分离（Cor 1.4 / Thm 4.4） |

### 关键发现
- **遗憾范数的单调性是整篇证明的支点**：Lemma 3.8（范数每步至少涨 $\text{KKTGap}^2$）保证 RM+ 不会在小范数区域永久打转，是把「条件一步改进」激活的开关；去掉它，Theorem 3.9 就立不住。
- **截断负遗憾这一个动作决定了天壤之别**：RM 与 RM+ 唯一差别就是是否把负遗憾归零，却导致 Nash 收敛从指数变成多项式——病根是 RM 的好动作可能积累极负遗憾、要指数久才翻正。
- **「无超参」不是免费的**：要拿到最优的 $1/\epsilon^4$，多单纯形情形需要把遗憾初始化在一个与 $L$、$m$ 相关的阈值上；坚持零初始化的标准 RM+ 只能保证 $1/\epsilon^8$。
- **折扣带来可证加速**：DRM+ 把速率从 $T^{-1/4}$ 提到 $T^{-1/2}$，首次给「折扣在实践中更好」提供了最坏情况层面的理论解释。

## 亮点与洞察
- **把两个本应无关的量「KKT gap」与「累积遗憾」桥接起来**：一般情况下它们毫无关系（Prop 3.2 给了零遗憾却 $\Omega(1)$ KKT gap 的反例），本文却证明在 RM+ 的轨迹上 KKT gap 能被遗憾控制——非渐近速率竟是无遗憾性的直接产物，这是非常漂亮的洞察。
- **「条件一步改进 + 范数单调」这套组合拳可迁移**：对任何「缺乏全局一步改进、但能证某种势/范数单调」的无超参一阶方法，都可以照搬「先证范数会迅速变大、再在大范数区证改进」的两段式论证，绕开了调学习率的传统套路。
- **下界构造把「停滞机制」讲得很透**：通过一个递归定义的支付矩阵 $A_{m,k}$，让动作按 $k$ 逐个「轮到」、而每个动作翻正所需时间 $T_k\ge\frac{k-2}{2}T_{k-1}$ 阶乘式爆炸，精准坐实 RM 的指数停滞——这种「逐层延迟」的构造思路对分析其他延迟型动力学有借鉴价值。

## 局限与展望
- **RM+ 在势博弈里是否真有 $\Omega(\sqrt T)$ 遗憾仍未知**：作者坦言这一点在零和博弈已知、但势博弈未定；若能突破 $\sqrt T$ 遗憾壁垒，按 Theorem 1.2 会自动翻译成更快的 Nash 收敛——这是个直接的开放问题。
- **只覆盖全反馈（full feedback）**：所有结果都假设每轮能观测完整效用/梯度向量，随机反馈或 bandit 反馈下能否保持类似速率尚无定论。
- **RM 在交替更新下是否渐近收敛仍开放**：下界说明它可能指数慢，但「是否最终收敛」这个定性问题作者也没解决。
- **lazy 版本需预先知道精度 $\epsilon$**：$\epsilon$-lazy 交替 RM+ 不是 anytime 算法；非 lazy 版（Theorem C.7）虽也 $O_\epsilon(1/\epsilon^4)$，但要付出额外的迭代依赖。

## 相关工作与启发
- **vs 二人零和博弈里的 RM/RM+ 理论（Farina et al. 2023）**：那里 RM 和 RM+ 速率都是 $T^{-1/2}$、没有可证分离（尽管 RM+ 实践更好）；本文换到势博弈/约束优化这个根本不同的设定，首次给出指数级最坏情况分离，为「实践中该用 RM+ 而非 RM」补上了理论依据。
- **vs 梯度下降类一阶优化器**：GD 靠调小学习率拿一步改进，但 Tewolde et al. (2025) 实验显示 RM+ 在约束优化 benchmark 上更强；本文证明 RM+ 是「可靠且快速」的一阶优化器，把它正式纳入优化工具箱，解释了那批实验。
- **vs 势博弈里收敛到 CCE 的经典结论（Moulin & Vial 1978 等）**：以往遗憾只被用来保证收敛到更宽松的 CCE，本文 Theorem 1.2 首次证明遗憾也能支配到更强的 Nash 均衡的收敛速度，拉近了 CCE 与 Nash 两个解概念在 RM+ 下的距离。
- **vs Panageas et al. (2023) 对虚拟博弈（fictitious play）的下界**：本文的指数下界构造在其支付矩阵基础上改造而来，把「逐层延迟翻正」的思想从 fictitious play 迁移到 RM。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解决悬置二十年的开放问题，给出 RM/RM+ 首个（指数级）最坏情况分离。
- 实验充分度: ⭐⭐⭐⭐ 纯理论论文，仅一张示意图，但定理覆盖同时/交替/折扣/多单纯形等多种设定、自洽完整。
- 写作质量: ⭐⭐⭐⭐⭐ 主线清晰，把「遗憾↔KKT gap」这条非显然的桥讲得很透。
- 价值: ⭐⭐⭐⭐⭐ 为扑克/博弈 AI 核心算法 RM+ 提供了坚实理论基础，并指导「用 RM+ 不用 RM、加折扣」的实践选择。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High-dimensional Mean-Field Games by Particle-based Flow Matching](high-dimensional_mean-field_games_by_particle-based_flow_matching.md)
- [\[ICML 2026\] Minibatch Selection via Partition Matroid Constrained Gradient Matching](../../ICML2026/optimization/minibatch_selection_via_partition_matroid_constrained_gradient_matching.md)
- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)
- [\[ICLR 2026\] Online Black-Box Prompt Optimization with Regret Guarantees under Noisy Feedback](online_black-box_prompt_optimization_with_regret_guarantees_under_noisy_feedback.md)
- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)

</div>

<!-- RELATED:END -->

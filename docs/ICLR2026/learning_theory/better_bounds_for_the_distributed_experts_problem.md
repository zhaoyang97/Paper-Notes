---
title: >-
  [论文解读] Better Bounds for the Distributed Experts Problem
description: >-
  [ICLR2026][学习理论][分布式专家问题] 本文研究"专家分布在多台服务器上、损失按 $\ell_p$ 范数跨服务器聚合"的分布式在线预测问题，提出一套基于指数随机变量嵌入 + 几何均值方差缩减的协议，首次在 coordinator（消息传递）模型里处理一般 $\ell_p$ 损失，在目标 regret $R$ 下把通信量做到 $\left(\tfrac{n+s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\mathrm{polylog}(nsT)$ 比特，优于此前只能处理 $\ell_1$ 的工作。
tags:
  - "ICLR2026"
  - "学习理论"
  - "在线学习"
  - "通信复杂度"
  - "分布式专家问题"
  - "ℓp 损失"
  - "几何均值估计"
---

# Better Bounds for the Distributed Experts Problem

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=6oiqYdYFn8](https://openreview.net/forum?id=6oiqYdYFn8)  
**代码**: https://github.com/samsonzhou/distributed-experts  
**领域**: 学习理论 / 在线学习 / 通信复杂度  
**关键词**: 分布式专家问题, 在线学习, 通信复杂度, ℓp 损失, 几何均值估计

## 一句话总结
本文研究"专家分布在多台服务器上、损失按 $\ell_p$ 范数跨服务器聚合"的分布式在线预测问题，提出一套基于指数随机变量嵌入 + 几何均值方差缩减的协议，首次在 coordinator（消息传递）模型里处理一般 $\ell_p$ 损失，在目标 regret $R$ 下把通信量做到 $\left(\tfrac{n+s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\mathrm{polylog}(nsT)$ 比特，优于此前只能处理 $\ell_1$ 的工作。

## 研究背景与动机
**领域现状**：在线专家预测（online learning with experts）是一个经典框架——有 $n$ 个专家，每个时刻 $t\in[T]$ 算法选一个专家 $i_t$ 并观测全体损失向量 $(L_1(t),\dots,L_n(t))$，目标是最小化相对于"事后最优专家"的累积 regret $R=\frac1T\big(\sum_t L_{i_t}(t)-\sum_t L_{i^*}(t)\big)$。指数权重（EWA）/ 乘性权重更新（MWU）在全信息下能拿到最优 regret $O(\sqrt{\log n/T})$。但当专家数 $n$ 和轮数 $T$ 都很大时，把所有损失集中起来跑 MWU 在计算和通信上都很贵。

**现有痛点**：现实场景里专家的损失天然是"分散"的。本文关注的不是流式（内存受限）那条线，而是另一个互补设定：专家的代价被切分到 $s$ 台服务器上，一个中心 coordinator 想实现低 regret 算法，同时尽量少地与服务器通信。关键是，专家 $i$ 在时刻 $t$ 的真实损失并不显式给出，而是要跨服务器聚合：

$$L_i(t)=\Big(\sum_{j\in[s]}\ell_i(j,t)^p\Big)^{1/p}$$

即 $s$ 个服务器上的局部损失 $\ell_i(j,t)$ 的 $\ell_p$ 范数。coordinator 拿不到 $L_i(t)$，必须靠服务器执行一个分布式协议把它（近似）算出来。

**核心矛盾**：通信量和 regret 之间存在 trade-off——想要 regret 越小，就得越频繁、越精细地把服务器上的损失同步给 coordinator，通信代价越高。此前 Jia et al. (2025) 的工作有两个局限：(1) 它的一般 $\ell_p$ 协议只适用于 broadcast/blackboard 模型（服务器可以"公开广播"，消息对所有服务器可见），而在更现实的消息传递模型里，把同一条消息发给所有服务器要付 $s$ 倍开销，技术上根本不一样；(2) 它在消息传递模型里只解决了 $\ell_1$（SUM）损失，通信量带一个 $O(Ts)$ 项。$\ell_1$ 之所以好做，是因为它对服务器是**可加**的——总损失就是各服务器损失之和，于是可以"按损失大小成比例采样"。但 $\ell_p$（$p>1$）是次可加/超可加的，这套采样技巧直接失效。

**本文目标**：在 coordinator（消息传递）模型里，为一般 $\ell_p$ 损失（$p\ge 1$）设计协议，达到接近最优的 regret，同时把通信量压到可证明的低水平，并刻画出干净的"通信—regret"权衡曲线。

**切入角度**：既然 $\ell_p$ 损失难直接采样，那就借助指数随机变量的**最大稳定性**把 $\ell_p$ "嵌入"成一个 $\ell_\infty$（求最大值）问题——求最大值在分布式里只需要少数几个大值上报，天然省通信。

**核心 idea**：用指数随机变量缩放把 $\ell_p$ 损失变成"取最大值"，再用几何均值估计器压住指数变量带来的无界方差，最后把这个低方差的损失估计喂给 MWU。

## 方法详解

### 整体框架
全局目标是在每个时刻 $t$ 为每个专家 $i$ 估出一个近似损失 $\widehat{s_i}(t)\approx L_i(t)$，把它当作 MWU 的权重增量 $w_i(t)=w_i(t-1)+\widehat{s_i}(t)$，从而以正比于 $\exp(-\eta w_i(t))$ 的概率选专家。难点全在"怎么在不显式拿到 $L_i(t)$、又少通信的前提下，估出一个**无偏且方差可控**的 $\widehat{s_i}(t)$"。

整套方法分三层递进：先是一个 warm-up 协议（Algorithm 2，假设损失落在常数区间 $[a,b]$，拿到固定 regret）；再把它参数化得到通信—regret 权衡（Algorithm 3，用采样概率 $\varrho$ 调档）；最后去掉 $[a,b]$ 假设、允许服务器按更细的阈值更激进地上报，得到主结果（Algorithm 4）。三者共用同一套"指数缩放 + 阈值上报 + 几何均值 + MWU"骨架，逐步放宽假设、收紧通信界。

### 关键设计

**1. 指数随机变量缩放：把 $\ell_p$ 损失变成"取最大值"**

直接对 $\ell_p$ 损失做成比例采样行不通，因为 $\ell_p$ 不可加。本文用指数随机变量的**最大稳定性**（Lemma 2.3）绕过去：给每个服务器 $j$ 的局部损失 $\ell_i(j,t)$ 配一个独立的 rate-1 指数变量 $e_i(j,t)$，做缩放 $\ell_i(j,t)/e_i(j,t)^{1/p}$，则跨服务器取最大值满足

$$\max_{j\in[s]}\frac{\ell_i(j,t)}{e_i(j,t)^{1/p}}\ \sim\ \frac{\big(\sum_{j\in[s]}\ell_i(j,t)^p\big)^{1/p}}{e^{1/p}}=\frac{L_i(t)}{e^{1/p}}$$

其中 $e$ 又是一个 rate-1 指数变量。也就是说，"缩放后取最大"在分布上正好等于"真实 $\ell_p$ 损失"除以一个已知分布的因子。由于 $\mathbb{E}[1/e^{1/p}]$ 可以算出来，只要服务器能把那个**最大的缩放值**报给 coordinator，就能反推出 $L_i(t)$ 的无偏估计。这一步把"聚合 $\ell_p$"这个棘手操作转化成了"找最大值"——而找最大值在分布式里只需要少数大值发声，是省通信的关键。

**2. 阈值上报：让"找最大值"只花少量通信**

服务器并不知道全局最大值是谁，不能直接"发最大"。本文证明：以高概率，那个最大缩放值会超过一个约为 $s^{1/p}$ 量级的阈值，而且**超过该阈值的缩放值只有很少几个**。于是协议让每个服务器只在 $q_i^{(b)}(j,t)\ge \tfrac{s^{1/p}}{100\log(nsT)}$ 时才把 $q_i^{(b)}(j,t)$ 发给 coordinator（Algorithm 2 第 8–10 行）。这样既能高概率抓到最大值，又把每个时刻每个专家的上报数控制在 $\mathrm{polylog}$ 级，从而把总通信压到 $O(sT)+nT\cdot\mathrm{polylog}(nsT)$。代价是：当最大值低于阈值时它不会被上报，使估计略微有偏，但这种事件只以 $1/\mathrm{poly}(nT)$ 的概率发生，对 regret 只产生低阶项扰动。

**3. 几何均值估计器：压住指数变量的无界方差**

最大稳定性给的是无偏估计，但 $1/e^{1/p}$ 这种形式方差**无界**（指数变量的 PDF 让大值出现概率不够快地衰减），直接喂给 MWU 会炸。本文借鉴范数估计里的经典做法（Li, 2008），对 $B=\lceil 3/p\rceil$ 个独立缩放各自取最大，再取它们的**几何均值**：

$$\widehat{s_i}(t)=\prod_{b\in[B]}\Big(\max_{j\in[s]}q_i^{(b)}(j,t)\Big)^{1/B}$$

Lemma 3.2 证明：存在常数 $C_{3.2}\in(0,2B]$ 使 $\mathbb{E}[Z]=C_{3.2}$ 且 $\mathbb{E}[Z^2]\le 3^B$，即几何均值把二阶矩压到了一个有界的小量。配合 Lemma 3.3——只要损失估计 $\widehat{s_i}(t)$ 近似无偏且二阶矩 $\le\rho$，用它替代真实损失跑 MWU 就能拿到 regret $O(\sqrt{\rho\log n/T})$——方差受控直接转化成 regret 受控。作者明确指出，"用几何均值给指数变量缩放估计做方差缩减"是本文相对 Jia et al. (2025) 的核心算法与技术新意，且可能对其他用到指数变量的场景有独立价值。

**4. 概率采样 + 自适应阈值：刻画通信—regret 权衡并去掉 $[a,b]$ 假设**

要把"固定 regret"变成"目标 regret $R$ 可调"，Algorithm 3 在每个时刻让每台服务器以概率 $\varrho=\tfrac{1}{R^2T}$ 独立决定是否参与该轮协议（不参与就不发声），coordinator 借助公共随机性即可知道哪些服务器被采样、无需额外通信；被采样时估计要乘 $1/\varrho$ 保持无偏。Lemma 4.2 给出 $\mathbb{E}[\widehat{s_i}(t)]=L_i(t)+\tfrac{1}{\mathrm{poly}(nT)}$、二阶矩 $\le O(TR^2)\cdot 3^B\cdot L_i(t)^2$，于是 regret $O(Rs^{1/p}\sqrt{\log n})$、通信 $\left(\tfrac{n+s}{R^2}\right)\mathrm{polylog}(nsT)$（Theorem 1.2）。Algorithm 4 进一步去掉"损失落在常数区间"的假设：以概率 $\tfrac{\varrho}{2^a}$ 选档位 $a$，让服务器在被采样时按更小的阈值 $\tfrac{s^{1/p}}{100(2^a)^{1/p}\log(nsT)}$ 更激进地上报小值，并用 $a^*$（满足 $\widehat{s_i}(t)\ge (s/2^{a^*})^{1/p}$ 的最小整数）做自适应缩放修正权重（第 11 行 $w_i(t)\mathrel{+}=2^{a^*}R^2T\cdot\widehat{s_i}(t)$）。这换来主结果通信 $\left(\tfrac{n+s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\mathrm{polylog}(nsT)$（Theorem 1.3），其中 $\max(s^{1-2/p},1)$ 这个因子刻画了去掉区间假设后、不同 $p$ 下额外要付的通信代价。

### 损失函数 / 训练策略
底座是标准 MWU（Algorithm 1）：维护权重 $w_i$，每轮加上损失、以正比于 $\exp(-\eta w_i)$ 的概率抽专家，学习率 $\eta\approx 1/\sqrt T$。本文不改 MWU 本身，所有创新都在"如何在分布式、少通信下喂给它一个好的损失估计 $\widehat{s_i}(t)$"。理论保证 regret 下界 $R\ge 1/\sqrt T$ 是信息论硬限（Cover, 1966），不可能更小，所以追求 $R\ge 1/\sqrt T$ 是标准的。

## 实验关键数据

实验只作为概念验证（proof-of-concept），跑在 HPO-B 超参优化基准上：把基准里不同的模型类当作专家，不同数据集分配到不同服务器，每个搜索步当作一"天"，代价向量取模型在各数据集上的归一化负准确率。对比对象是离线 MWU 和 Jia et al. (2025) 的协议。

### 主结果对比

| 对比维度 | 本文协议（Algorithm 2/3） | 基线 | 观察 |
|----------|--------------------------|------|------|
| 通信 vs $p$ | 随阈值变化 | — | $p>1$ 时通信随阈值升高而增；$p<1$ 趋势相反 |
| Reward vs $p$ | 优于 MWU | 离线 MWU | 本文 reward 反而更好（作者归因于 MWU 学习率未充分调优） |
| 通信 vs regret（$p=1$） | 更省通信 | Jia et al. (2025) | $p=1$ 特例下通信优于 Jia et al.，并复现出理论预测的通信—regret 权衡曲线 |

### 通信复杂度对比（理论，Figure 1）

| 来源 | 损失函数 | 通信代价（比特） |
|------|----------|------------------|
| Jia et al. (2025) | $\ell_1$ | $\left(\tfrac{n}{R^2}+Ts\right)\cdot\mathrm{polylog}(nsT)$ |
| 本文 Theorem 1.2 | $\ell_p$ | $\left(\tfrac{n+s}{R^2}\right)\cdot\mathrm{polylog}(nsT)$ |

### 关键发现
- 即使在 $p=1$ 这个 Jia et al. 已覆盖的特例上，本文也把 $O(Ts)$ 依赖改善成对一般 regret $R$ 的参数化：比如 $R=O(1)$ 时 Jia 仍是 $O(Ts)$，本文是 $O(s)$，长时间跨度 $T$ 下差距显著。
- 几何均值估计器是把 regret 界做出来的关键——它把指数缩放的无界方差压成 $\mathbb{E}[Z^2]\le 3^B$，直接决定了喂给 MWU 的损失二阶矩 $\rho$。
- 实验中本文 reward 超过离线 MWU 略反直觉，作者诚实地把它归为 MWU 学习率没调好，而非协议本身更强。

## 亮点与洞察
- **把"聚合"问题转成"取最大值"问题**：指数随机变量的最大稳定性是这篇文章的灵魂——$\ell_p$ 难采样，但 $\max$ 在分布式里只需少数大值发声，这个转化同时解决了"难聚合"和"省通信"两件事，思路非常优雅，可迁移到其它需要在分布式/流式下估计 $\ell_p$ 量的任务。
- **几何均值做方差缩减**：指数缩放给的是无偏但无界方差的估计，几何均值（而非算术均值）恰好能把这类重尾估计的二阶矩压到 $3^B$ 量级，是连接"无偏估计"和"MWU 需要二阶矩有界"两端的桥梁，作者也点明它可能有独立价值。
- **三层递进的写法**：warm-up（固定 regret）→ 概率采样（通信—regret 权衡）→ 自适应阈值（去掉区间假设）的层层放宽，让读者能清楚看到每个假设各贡献了哪一块通信代价（尤其是 $\max(s^{1-2/p},1)$ 这个因子的来历）。

## 局限与展望
- 实验只是概念验证，规模小（单机 i7-3770、HPO-B 一个基准），并未在真实大规模分布式系统上验证通信节省。
- reward 超过离线 MWU 的现象未被严格解释，作者自己把它归因于基线调参，说明对照不够 controlled。
- $\max(s^{1-2/p},1)$ 这个因子意味着去掉 $[a,b]$ 假设、在 $p<2$ 时仍要付额外通信代价，是否最优、能否去掉，文中未给下界。
- 作者指出几何均值可被其它方差缩减技术替换，但没探索；同时把推广到 submodular 目标、$\ell_\infty$（max）损失列为未来方向。

## 相关工作与启发
- **vs Jia et al. (2025)（最直接对标）**：他们在 broadcast/blackboard 模型里做一般 $\ell_p$（靠遍历服务器找每个专家最大损失服务器），在消息传递模型里只做 $\ell_1$（SUM，靠成比例采样）。本文首次在更现实的 coordinator/消息传递模型里处理一般 $\ell_p$，技术路线根本不同（指数缩放 + 几何均值 vs 广播遍历/可加采样），并在 $p=1$ 特例上也改善了通信。
- **vs 流式在线学习（Srinivas 2022 / Woodruff 2023 / Braverman 2026 等）**：那条线在内存受限下顺序处理损失、只维护紧凑摘要；本文是互补设定——coordinator 可掌握全部专家的全信息，瓶颈不是内存而是通信。
- **vs 范数估计中的指数缩放 / 几何均值（Li 2008；Woodruff & Zhou 2021）**：这些技术此前用于流式范数估计，本文首次把它们搬到分布式在线学习场景，是一次跨子领域的方法迁移。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 coordinator 模型解决一般 $\ell_p$ 分布式专家问题，指数缩放 + 几何均值的组合是实打实的算法新意。
- 实验充分度: ⭐⭐⭐ 理论扎实但实验仅小规模概念验证，对照不够严格。
- 写作质量: ⭐⭐⭐⭐⭐ 三层递进结构清晰，每个假设贡献的通信代价交代得很干净。
- 价值: ⭐⭐⭐⭐ 给分布式在线学习的通信—regret 权衡补上了 $\ell_p$ 这一块，方法对其它分布式估计任务有借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](../../ICML2026/learning_theory/robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICLR 2026\] Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion](better_learning-augmented_spanning_tree_algorithms_via_metric_forest_completion.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)

</div>

<!-- RELATED:END -->

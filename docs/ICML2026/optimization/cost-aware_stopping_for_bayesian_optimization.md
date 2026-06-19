---
title: >-
  [论文解读] Cost-Aware Stopping for Bayesian Optimization
description: >-
  [ICML2026][优化/理论][Bayesian optimization] 作者把 Weitzman 的 Pandora's Box 停下原则推广到带相关性的贝叶斯优化场景，证明 PBGI/LogEIPC 这两个 cost-aware 采集函数在共同的"采集函数值越过当前最优"停下规则下，期望代价调整 simple regret 不会比"采一次就停"更差，从而给出首个对 cost-adjusted simple regret 有理论保证的自适应停下规则。
tags:
  - "ICML2026"
  - "优化/理论"
  - "Bayesian optimization"
  - "cost-aware stopping"
  - "Pandora's Box"
  - "Gittins index"
  - "expected improvement per cost"
---

# Cost-Aware Stopping for Bayesian Optimization

**会议**: ICML2026  
**arXiv**: [2507.12453](https://arxiv.org/abs/2507.12453)  
**代码**: https://github.com/QianJaneXie/CostAwareStoppingBayesOpt  
**领域**: 贝叶斯优化 / AutoML / 决策理论  
**关键词**: Bayesian optimization, cost-aware stopping, Pandora's Box, Gittins index, expected improvement per cost

## 一句话总结
作者把 Weitzman 的 Pandora's Box 停下原则推广到带相关性的贝叶斯优化场景，证明 PBGI/LogEIPC 这两个 cost-aware 采集函数在共同的"采集函数值越过当前最优"停下规则下，期望代价调整 simple regret 不会比"采一次就停"更差，从而给出首个对 cost-adjusted simple regret 有理论保证的自适应停下规则。

## 研究背景与动机

**领域现状**：贝叶斯优化 (BO) 是处理昂贵黑盒目标 $f:X\to\mathbb{R}$ 的主流框架——用 GP 拟合后验，再用采集函数 $\alpha_t$ 在 explore/exploit 之间权衡，挑下一个采点 $x_{t+1}$。常用采集函数包括 EI、LCB、KG、TS。当评估每个点的代价 $c(x)$ 显著异质（例如不同超参对应的训练时间差很多）时，社区给出了 cost-aware 版本，其中两个 SOTA 是 **PBGI**（Pandora's Box Gittins Index，Xie et al. 2024）与 **LogEIPC**（Ament et al. 2023 提出的 log EI per cost）。

**现有痛点**：BO 的"什么时候停"几乎被忽视。现有停下规则要么是启发式（达到固定迭代、最优值连续几轮不变），要么基于 simple regret 的收敛准则（UCB-LCB、PRB、SRGap-med 等），但**都不显式建模评估代价**。在 cost-aware 场景下，这些规则常常导致"小幅 regret 提升却付出巨额评估开销"，本质上让累计代价远高于真正的边际收益。

**核心矛盾**：用户实际想最小化的是 cost-adjusted simple regret $\mathcal{R}_c = \mathbb{E}[\min_{1\le t\le\tau} f(x_t)-\inf_{x\in X} f(x) + \sum_{t=1}^\tau c(x_t)]$，但现有停下规则只盯着前一项，对后一项视而不见；同时即使是 EI thresholding (Nguyen et al. 2017) 这种"当 EI 小于阈值就停"的规则，阈值是启发式调出的而非来自原理。

**本文目标**：设计一个**显式 cost-aware**、**理论可证**、**用户参数最少**的自适应停下规则；统一适用于均匀代价和异质代价场景，并能自然搭配 PBGI / LogEIPC 这两个 cost-aware 采集函数。

**切入角度**：作者回到 Weitzman (1979) 的 Pandora's Box 经典最优策略——其 Gittins-index 论证要求选点策略与停下时间**绑定使用**才贝叶斯最优。把这个 stopping condition 从独立离散 setting 推广到相关 GP setting，自然衍生出 PBGI 的停下规则；然后通过 EI 单调性把它**等价**改写成 LogEIPC 的停下规则，于是两种貌似不同来源的采集函数共享同一个停下规则。

**核心 idea**：停下条件 $\min_x \alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ 等价于 $\max_x \alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$ ——把"没有未评估点的 fair value 优于当前最优"这一直觉表达出来。

## 方法详解

### 整体框架
PBGI 把每个候选点视为 Pandora's Box 中的一个"盒子"，其 fair value $\alpha_t^{\mathrm{PBGI}}(x)$ 定义为使该点的期望改进恰好等于代价的阈值，即 $\mathrm{EI}_{f\mid x_{1:t}, y_{1:t}}(x; \alpha_t^{\mathrm{PBGI}}(x))=c(x)$。Weitzman 经典论证表明：在独立离散场景下，**选** $\arg\min_x \alpha^{\mathrm{PBGI}}(x)$ + **停**于 $\min_x \alpha^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ 是贝叶斯最优策略。作者在相关 GP setting 下保留这套结构：每轮先用后验更新 $\alpha_t^{\mathrm{PBGI}}$，再判断停下条件。

### 关键设计

**1. 基于 PBGI 的停下规则：用更新后的 $\alpha_t$ 而非 $\alpha_{t-1}$ 判断"还值不值得开盒子"**

现有停下规则要么是启发式、要么只盯着 simple regret 而无视评估代价，在 cost-aware 场景里常常"花大钱换微小提升"。本文的停下条件直接把这件事问出来：$\min_{x\in X\setminus\{x_1,\dots,x_t\}} \alpha_t^{\mathrm{PBGI}}(x) \ge y^*_{1:t}$，即"剩下所有未开的盒子，其公平价格都不优于当前最优"时就停。这里一个微妙却关键的选择是用**后验更新之后**的 $\alpha_t$，而不是先前理论工作（Gergatsouli & Tzamos 2023）用的 $\alpha_{t-1}$：$\alpha_t^{\mathrm{PBGI}}(x)$ 是当前全部信息下 $x$ 的"公平价格"，对已评估点退化成观测值，对未评估点同时反映 $f(x)$ 的不确定性和代价 $c(x)$，只有用最新的 $\alpha_t$ 才真实回答"此刻该不该继续投钱"。之所以非这么做不可，是因为 Weitzman 原版的"选 + 停"在离散独立场景下必须配套才贝叶斯最优，推广到相关 GP 时若沿用滞后的 $\alpha_{t-1}$，就会出现"该停不停、该继续反而停"的信息滞后；Section C.2 的实验确认了用 $\alpha_t$ 带来明显的 cost-adjusted regret 改进。

**2. 通过 EI 单调性导出等价的 LogEIPC 停下规则：一条规则同时适配两种采集函数**

PBGI 来自 Gittins index，LogEIPC 来自 cost-normalized EI，看起来是两套东西，用户换采集函数时似乎还得换停下规则。作者发现两者其实共享同一停下条件。关键是 $\mathrm{EI}_\psi(x;y)$ 关于 $y$ 严格单调递增，于是 $\alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ 当且仅当 $\mathrm{EI}_{f\mid x_{1:t},y_{1:t}}(x;y^*_{1:t}) \le c(x)$，取对数即得 $\max_{x\in X\setminus\{x_1,\dots,x_t\}}\alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$。当代价均匀 $c(x)\equiv c_0$ 时它进一步化简成 $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$——恰好回收了 Nguyen et al. (2017) / Zhou et al. (2024) 的 EI thresholding 规则，只不过那里靠手调的启发式阈值，被这里有原理的"每样本代价"替换掉了。这层等价让同一条规则有了两个解读视角：Pandora's Box 的决策论视角和 EI-per-cost 的经济学视角，LogEIPC 用户也无需另立停下条件就能直接套用同一套理论保证。

**3. 代价调整 regret 的"不差于立即停"保证 + 有限时间终止：给 cost-adjusted regret 上第一条非渐近界**

前面两点解决了"怎么停"，这一点回答"停得有没有保证"——这是现有规则普遍缺失的。证明分两步推进：Lemma 3.1 先说明在停下之前每一轮 $t<\tau$ 选出的 $x_{t+1}$ 都满足 $\alpha_t^{\mathrm{EI}}(x_{t+1})\ge c(x_{t+1})$，即"期望改进永远不低于代价"，于是每多采一步都是划算的；Theorem 3.2 顺着这个事实给出

$$\mathbb{E}\Big[y^*_{1:\tau}-\min_x f(x)+\sum_{t=1}^\tau c(x_t)\Big]\le \mathbb{E}\big[y_1-\min_x f(x)+c(x_1)\big]=U+C,$$

其中 $U=\mu(x_1)-\mathbb{E}[\min_x f(x)]$、$C=c(x_1)$。这句话的含义是：用这套规则的最坏结果也不会比"采一次立刻停"更差——一种在 cost-adjusted 意义下的"无悔"保证，用户最差等于没开 BO，不会被坑。Corollary 3.3 进一步把期望累计代价也压在 $U+C$ 之内，并在 $c(x)\ge c_0>0$ 时给出 $\frac{U+C}{\delta c_0}$ 步内以 $1-\delta$ 概率终止的有限时间保证；Corollary 3.5 再把结论迁到 budget-constrained 场景，导出 $\lambda=U/(B-C)$ 这种"按预算选 cost scaling"的原理化取法。Figure 2、3 印证了这一性质的实际意义：不少 baseline 反而比"立即停"更差。

### 损失函数 / 训练策略
不涉及训练；仅在 BO 主循环里增加一行判断。实际部署还配套两个工程细节：(i) **稳定期 + 滑动平均**：高维空间下 GP 超参与采集函数优化都不稳，作者用前 $W=20$ 轮作为 stabilization 期不允许触发停下，并对停下信号做 $W$ 轮 moving average；(ii) **未知代价处理**：把 $\ln c(x)$ 建模为 GP 后用 $\mathbb{E}[c(x)]=\exp(\mu_{\ln c}+\sigma_{\ln c}^2/2)$ 替换 $c(x)$，原理保证仍然成立。

## 实验关键数据

### 主实验
作者在三套场景里把 PBGI/LogEIPC 停下规则与 7 个 baseline 停下规则交叉评估：1 维 Bayesian regret（精确 grid search 排除优化误差）、8 维 Bayesian regret（含三种代价：uniform / linear / periodic）、LCBench (35 个数据集 HP tuning) 与 NATS-Bench (32k 神经架构) 两个 AutoML benchmark。

| 场景 | 维度 / 代价 | PBGI+本文停下 | LogEIPC+本文停下 | UCB-LCB | Convergence | Hindsight (oracle) |
|------|----------|--------------|--------------|---------|-------------|------------|
| Bayesian regret 1D | $\lambda=0.1$ | 接近 hindsight | 接近 hindsight | 偏高 | 偏高 | 下界 |
| Bayesian regret 8D | linear cost | 几乎等于 hindsight | 接近 hindsight | 显著差 | 显著差 | 下界 |
| LCBench (35 数据集) | $\lambda=10^{-3}$ | 约 75% 数据集 Top-3 | 约 75% 数据集 Top-3 | 偏高 | 中等 | 下界 |
| NATS-Bench | $\lambda=10^{-5}$ | 接近 hindsight，2 个任务例外 | 略弱于 PBGI | 多次到达 200 轮上限 | 中等 | 下界 |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|--------|------|
| 用 $\alpha_t$ vs $\alpha_{t-1}$ | $\alpha_t$ 显著优 | 用后验更新后的 fair value 才真实反映"现在还该不该继续" |
| stabilization + moving average ($W=20$) | 高维下停下更稳 | 抑制 spurious stops，否则会因 GP 超参震荡过早停 |
| 已知代价 vs 未知代价 (GP for $\ln c$) | 两者表现接近 | 验证用 $\mathbb{E}[c(x)]$ 替换 $c(x)$ 后保证仍然成立 |
| cost model 误设（proxy runtime vs actual runtime） | 退化平缓 | 即便代价模型有偏，本文规则依然 robust |

### 关键发现
- 多数 baseline 停下规则（SRGap-med、UCB-LCB）在 NATS-Bench 上**经常打不到 200 轮上限就死活不停**，本文规则总能在上限前结束，体现了 cost-aware 终止性。
- "搭档关系"重要：本文停下规则 + PBGI 在 LCBench 上比 + LogEIPC 略强，作者把这归因于 PBGI 在 GP 误设场景下更鲁棒（Figure 10 显示固定预算下 PBGI 的 simple regret 也更低）。
- 即使代价测量有偏（如把训练时间 proxy 成模型参数线性函数），停下规则的相对排名不变，证明工程化时不必精确测量代价。
- LCBench 上少数表现欠佳的数据集大多是 instance 数 $<10000$ 的极小数据集，问题更可能出在 benchmark 自身的 val/test 分布错配而非停下规则本身。

## 亮点与洞察
- **"选 + 停"必须配套**才贝叶斯最优——Weitzman 1979 的这一关键性质被绝大多数 BO 工作忽视，本文重新把它请回来，并优雅地从离散 Pandora's Box 推广到相关 GP setting。
- **两条不同起源的采集函数共享同一停下规则**——PBGI 来自 Gittins index，LogEIPC 来自 one-step lookahead 的 cost-normalized EI，但通过 EI 单调性发现停下条件**字面等价**，这种"殊途同归"的揭示让用户选采集函数时无需重新挑停下规则。
- **"不差于立即停"是个聪明的最坏情况保证**——比直接保证 regret bound 更弱，但对 cost-adjusted setting 来说恰好够用，因为它说明用户用这套规则永远不会被坑（最差等于不开 BO）。

## 局限与展望
- 理论保证局限在 PBGI / LogEIPC 这两个特定采集函数；KG、MES 这种来自 value-of-information / entropy search 原理的采集函数需要自己配套的停下规则，本文方法无法直接搬过去。
- 高维空间下需要 $W=20$ 的 stabilization + smoothing 窗口，这虽是工程必要但引入了一个超参；对小预算任务（总迭代数本身就 <50）这个窗口可能让规则永远不触发。
- "不差于立即停"的保证在最坏情况下是 tight 的（即当代价远大于可改进幅度时立即停就是最优），但对中间区间没有给出比这更紧的 regret 速率刻画。
- 假定代价 $c(x)$ 是确定函数或可由 GP 建模；当代价本身依赖于 $f$ 的观测值（如失败的训练会触发额外清理代价）时这套框架会失效。

## 相关工作与启发
- **vs Nguyen et al. (2017) / Zhou et al. (2024) EI thresholding**: 他们当 $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$ 时停下，但 $c_0$ 是启发式阈值；本文证明把 $c_0$ 替换为真正的"每样本代价"就是 LogEIPC 停下规则在均匀代价下的特例，从而把启发式阈值升级为有原理的常数。
- **vs UCB-LCB (Makarova et al. 2022)**: UCB-LCB 仅根据置信带宽度判停，与代价无关，在 cost-aware 场景容易过度评估；本文显式把代价 $c(x)$ 写进停下条件。
- **vs PRB (Wilson, 2024)**: PRB 给 simple regret 一个 $(1-\delta)$ 置信保证，但同样不管代价；与 Hindsight 对比常常显得"花钱过多"。
- **vs Chick & Frazier (2012) cost-aware stopping**: 他们在有限域独立采样里做了 cost-aware 停下，但局限于无相关性场景；本文把这条思路通过 PBGI 推广到了相关 GP，是真正适合 BO 的 cost-aware stopping。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 Weitzman 经典 stopping rule 完整迁到 cost-aware BO，并发现 PBGI/LogEIPC 停下规则字面等价。
- 实验充分度: ⭐⭐⭐⭐⭐ 1D/8D Bayesian regret + LCBench 35 数据集 + NATS-Bench，三层场景覆盖完整，consistent 排名 Top-3。
- 写作质量: ⭐⭐⭐⭐ 三段论清晰（直觉→等价改写→理论保证），符号自洽。
- 价值: ⭐⭐⭐⭐⭐ 给 AutoML 工程师一个"放心默认停下规则"，且首次给出 cost-adjusted regret 的非渐近界。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](../../CVPR2026/optimization/dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/optimization/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)

</div>

<!-- RELATED:END -->

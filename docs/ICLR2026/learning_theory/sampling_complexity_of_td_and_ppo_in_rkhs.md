---
title: >-
  [论文解读] Sampling Complexity of TD and PPO in RKHS
description: >-
  [ICLR 2026][学习理论][RKHS] 本文在再生核希尔伯特空间（RKHS）这一统一函数空间里，把策略评估（核化 TD critic）与策略改进（KL 正则的近端/自然梯度更新）解耦分析，给出依赖 RKHS 熵的非渐近、实例自适应的收敛界，并推导出保证 $O(k^{-1/2})$ 收敛所需的每轮采样规则，在 CartPole、Acrobot、HalfCheetah 上验证了理论预测的步长调度。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "强化学习理论"
  - "RKHS"
  - "时序差分"
  - "PPO/NPG"
  - "样本复杂度"
  - "非渐近界"
---

# Sampling Complexity of TD and PPO in RKHS

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5gUMhTUDi0](https://openreview.net/forum?id=5gUMhTUDi0)  
**代码**: https://github.com/conniemessi/TD-and-PPO-in-RKHS  
**领域**: 学习理论 / 强化学习理论  
**关键词**: RKHS、时序差分、PPO/NPG、样本复杂度、非渐近界

## 一句话总结
本文在再生核希尔伯特空间（RKHS）这一统一函数空间里，把策略评估（核化 TD critic）与策略改进（KL 正则的近端/自然梯度更新）解耦分析，给出依赖 RKHS 熵的非渐近、实例自适应的收敛界，并推导出保证 $O(k^{-1/2})$ 收敛所需的每轮采样规则，在 CartPole、Acrobot、HalfCheetah 上验证了理论预测的步长调度。

## 研究背景与动机

**领域现状**：策略梯度与信赖域方法（NPG、TRPO、PPO、actor-critic）配上 TD critic，是现代连续控制 RL 的主力工具，因为它们能兼容表达力强的函数逼近器并提供稳定的改进步。

**现有痛点**：尽管经验上很成功，这类算法在"富函数逼近"下的全局收敛理论却支离破碎。已有分析大多只在 tabular/linear 情形成立，或要依赖很强的 realizability / concentrability 条件；用非线性 critic 时，保证往往假设值函数/优势函数是"精确"估计出来的。更关键的是，许多策略改进界把策略期望项当成"没有采样噪声"来处理，因此**保证一次改进到底需要多少数据从来没被讲清楚**。

**核心矛盾**：策略评估的统计误差（critic 学得有多准）和策略改进的步长/采样量之间存在耦合——评估越不准、就越需要保守的步长或更多样本，但旧理论把两者割裂开，要么假设评估精确，要么不指明 per-iteration 的数据需求。同时，不同函数类（tabular、Sobolev、Gaussian 核、宽网络的 NTK）各有各的分析，缺一个统一框架。

**本文目标**：(i) 设计一个在富函数类下做策略评估、且统计误差可控的算法；(ii) 把这个评估步和一个可证明朝最优策略上升的改进步耦合起来，并**显式量化每轮所需样本数**。

**切入角度**：作者采用"函数空间"视角，直接在 RKHS 里优化策略。RKHS 的好处是它能用同一套熵/覆盖数语言统一表示 tabular、linear、Sobolev、Gaussian 核以及宽神经网络（通过其 NTK）；而核梯度下降的动力学恰好镜像了用梯度下降训练的宽网络的 NTK 动力学，于是 RKHS 上的结论可直接迁移到神经 critic/actor。

**核心 idea**：在 RKHS 里把"核化 TD 评估器"（用隐式预条件做几何收敛、避免矩阵求逆）和"KL 正则的近端策略更新"（对评估值取指数即得 NPG/PPO 风格的更新）拼起来，用 RKHS 熵刻画收敛率，并据此反推每轮采样量。

## 方法详解

### 整体框架

考虑 MDP $(S,A,P,r,\gamma)$，假设要学的 Q 函数 $Q^\pi$ 落在由对称正定核 $K$ 诱导的 RKHS $\mathcal{H}(S\times A)$ 中。算法是一个外层策略迭代（NPG in RKHS，Algorithm 1）：每个外层轮 $k$ 先用当前策略 $\pi_k$ 采集 $n^{(k)}$ 个一步转移四元组 $(s_0,a_0,s_1,a_1)$，再用**核化 TD** 在 RKHS 里迭代 $T^{(k)}$ 步得到 critic $f^{(k)}\approx Q^{(k)}$，最后用一步 **KL 正则近端更新**把策略推进为 $\pi_{k+1}\propto \pi_k\exp\{\Delta_k f^{(k)}\}$。整篇论文的贡献是给这两步分别建立非渐近界，并把它们串起来证明全局收敛率 $O(k^{-1/2})$，同时给出达到该率所需的采样规则。

这是一篇纯理论分析论文（评估误差分解 + 经验过程 + mirror-descent 风格的策略改进不等式），没有多模块协同的工程 pipeline，因此不画框架图——核心是几条界和它们之间的耦合关系。

### 关键设计

**1. 核化 TD critic：用 RKHS 泛函梯度做隐式预条件，避免立方阶矩阵求逆**

策略评估被写成一个 RKHS 上的不动点核岭回归（fixed-point KRR）：
$$\hat{Q}^\pi=\min_{f\in\mathcal{H}}\frac{1}{n}\sum_{i=1}^n\big(f(\omega_0^{(i)})-r(\omega_0^{(i)})-\gamma\hat{Q}^\pi(\omega_1^{(i)})\big)^2+\lambda\|f\|_{\mathcal{H}}^2,$$
其中 $\omega=(s,a)$。由表示定理它有闭式解 $\hat{Q}^\pi=K(\cdot,\omega_0)b^\pi$，$b^\pi=[K+\lambda nI-\gamma C]^{-1}r$，但直接求逆是立方阶。作者不去在 $\mathbb{R}^n$ 上解这个线性系统，而是把它当成 RKHS 上的泛函优化，用核梯度下降迭代：
$$f_{t+1}=(1-\alpha_t)f_t-\eta_t\sum_{i=1}^n\big(f_t(\omega_0^{(i)})-r(\omega_0^{(i)})-\gamma f_t(\omega_1^{(i)})\big)K(\omega_0^{(i)},\cdot).$$
关键在于：用 RKHS 内积 $\langle\cdot,\cdot\rangle_{\mathcal{H}}$ 而非 $\mathbb{R}^n$ 上的 $\ell_2$ 内积，会改变梯度与 Hessian，**等效于一个隐式预条件器**，使迭代沿好方向走。若用 $b_t=K^{-1}f_t(\omega_0)$ 表示，更新式 $b_{t+1}=(1-\alpha_t)b_t-\eta_t(f_t(\omega_0)-r-\gamma f_t(\omega_1))$ 形式上很像 semi-gradient TD(0)，区别仅在于这里是无穷维 RKHS 的泛函梯度。选好常数权重衰减 $\alpha$ 与步长 $\eta$ 后，误差满足 $b_{t+1}-b^\pi=[(1-\alpha)I-\eta K+\eta\gamma C]^{t+1}[b_0-b^\pi]$，只要矩阵谱半径小就**几何（指数）收敛**到 $\hat{Q}^\pi$，整个过程不需要任何矩阵求逆。这一构造同样支持 N-step TD：N 步等价于折扣因子改为 $\gamma^N$、转移核改为 $P(s_N|s_0,a_0)$、奖励改为 $\bar r=\sum_{t=0}^{N-1}\gamma^t r$ 的一步 TD，故收敛分析可直接套用。

**2. 依赖 RKHS 熵的非渐近 TD 误差界：一套界统一 tabular / Sobolev / NTK / Gaussian**

作者用经验过程刻画 critic 的统计误差，核心是 RKHS 复杂度由覆盖数/熵度量：单位球的熵满足 $H(\delta,\|\cdot\|_{L^\infty},\mathcal{B})\le C\delta^{-2\beta}|\log\delta|^{2\kappa}$（$\beta\in[0,1)$，$\kappa\ge 0$）。在此假设下，定理 10 给出训练集上的非渐近 TD 误差：
$$\sqrt{\frac{1}{n}\sum_{i=1}^n\big(f_t(\omega_0^{(i)})-Q^\pi(\omega_0^{(i)})\big)^2}\le O_p\Big((1-c\gamma)^{-\frac{2+\beta}{2+2\beta}}\,n^{-\frac{1}{2+2\beta}}\,|\log n|^{\frac{\kappa}{1+\beta}}\Big)\|Q^\pi\|_{\mathcal{H}}.$$
注意常数 $1-c\gamma$（来自 Assumption 6，刻画一步转移分布偏离初始分布的程度）出现在速率里：$c$ 越大、采样质量越差，收敛越慢，它直接反映采样复杂度。进一步在具体核下（Corollary 11），该界就退化为各非参数估计的**极小极大最优率**：tabular 为 $(1-c\gamma)^{-1}n^{-1/2}$；Sobolev（光滑度 $m$、内蕴维度 $d$）为 $(1-c\gamma)^{-\frac{2m+d/2}{2m+d}}n^{-\frac{m}{2m+d}}$；NTK 为 $(1-c\gamma)^{-\frac{3d+1}{4d}}n^{-\frac{d+1}{4d}}$；Gaussian 核为 $(1-c\gamma)^{-1}n^{-1/2}|\log n|^{(d+1)/2}$。一个熵假设串起四类函数空间，是本文"实例自适应"的体现。

**3. KL 正则近端更新：对评估值取指数得到 RKHS 版 NPG，并显式给出每轮采样规则**

策略改进用如下更新：
$$\pi_{k+1}=\arg\max_{\int_A\pi=1,\pi\ge0}\mathbb{E}_n\Big[\Delta_k\int_A f^{(k)}(s,a)\pi(a|s)da-\mathrm{KL}\big(\pi(\cdot|s)\,\|\,\pi_k(\cdot|s)\big)\Big].$$
注意这里 KL 的方向是 $\mathrm{KL}(\pi\|\pi_k)$（与原版 PPO 的 $\mathrm{KL}(\pi_k\|\pi)$ 相反），它有闭式解 $\pi_{k+1}\propto\pi_k\exp\{\Delta_k f^{(k)}\}\propto\exp\{\sum_{j=0}^k\Delta_j f^{(j)}\}$，等价于自然策略梯度，作者称之为 **explicit PPO (NPG)**；原版方向对应的隐式解则称 **implicit PPO**（附录分析）。这一步把 NPG 从 tabular/linear/两层网络推广到一般 RKHS 函数，且策略可以是 $(s,a)$ 的连续函数，并用经验期望 $\mathbb{E}_n$ 替换总体期望、对数据自适应。它最重要的贡献是**显式量化保证改进所需的样本量**——这正是旧理论把策略期望当精确处理时留下的空白。

**4. 评估—改进耦合：mirror-descent 不等式 + 采样表，锁定 $O(k^{-1/2})$ 全局收敛**

把性能用期望总回报 $R[\pi]=\mathbb{E}_{S\sim\nu^*}[V^\pi(S)]$ 度量，作者改造无穷维空间的 mirror descent 分析得到核心不等式（定理 13）：
$$\inf_k\big(R[\pi^*]-R[\pi_k]\big)\le\frac{\big(\sum_k 2\Delta_k\|f^{(k)}-Q^{(k)}\|_{L^\infty}\big)+\big(\sum_k\Delta_k^2(1-\gamma)^{-1}\|r\|_{L^\infty}\big)+\mathbb{E}_{S\sim\nu^*}\mathrm{KL}(\pi^*\|\pi_0)}{\sum_k\Delta_k}.$$
分子第一项是评估误差（$L^\infty$ 范数下 critic 与真 Q 的差），第二项是步长平方累加的"漂移代价"，第三项是初始策略到最优策略的 KL 距离。由此可见：要达到随机优化的最优 $O(k^{-1/2})$ 率，必须把评估误差在 $L^\infty$ 下压得足够小。取步长 $\Delta_k=1/\sqrt{k}$，并按 Table 1 给四类核分别设定每轮样本数 $n(k)$ 与正则 $\lambda(k)$（再据定理 10 定 $\alpha,\eta,T$），即得 Corollary 14：$\inf_{1\le k\le k^*}(R[\pi^*]-R[\pi_k])\le O_p(1/\sqrt{k^*})$。采样规则揭示了一个直觉：所需样本数随迭代步 $k$ 与策略复杂度（RKHS 范数 $\|\pi_k\|_{\mathcal{H}}$）增长——策略越接近最优、性能差越窄，就越需要精确评估、样本越多；策略若趋于 delta 分布则 RKHS 范数发散、采样需求随之上升。

### 损失函数 / 训练策略

理论算法是"评估 $T$ 步 + 改进一步"两阶段，但实践中 $T$ 很小（如 $T=4$）时 critic 学不充分，会让基于劣质 critic 的 actor 更新不稳定。因此实现上采用 PPO 常用的**联合优化**：把 critic 的 TD-error 损失（Algorithm 1 第 9 行）与 actor 的 NPG 目标（第 11 行）合成统一损失，在每批数据上一起最小化 $T$ 个 epoch。网络用两隐层 MLP，actor 与 critic 共享全部参数，policy 取 $\pi_\theta\propto\mathrm{Softmax}(f_\theta)$，端到端训练。

## 实验关键数据

实验意在**验证理论预测**而非刷 SOTA：用神经网络实现 $f^{(k)}$，在三个标准控制任务上跑 1000 episode、10 个随机种子取均值±标准差。

### 主实验：步长调度与采样效率

| 环境 | 验证点 | 关键现象 |
|------|--------|----------|
| CartPole-v1（4 维状态/2 离散动作） | 步长指数 $\alpha$ | $\Delta_k=k^{-0.5}$ 收敛，$k^{-0.2}$ 发散，$k^{-1.5}$ 停滞 |
| Acrobot-v1（6 维状态/3 离散动作） | 步长指数 $\alpha$ | 同上趋势；$k^{-0.5}$ 在若干 episode 后略有回落（网络变复杂、样本不足） |
| HalfCheetah-v5（17 维状态/6 维连续动作） | 高维可扩展性 | 维持合理运行时间与内存，$k^{-0.5}$ 仍最优 |

三个环境一致呈现 Corollary 14 预测的图景：唯有 $\Delta_k=1/\sqrt{k}$ 的调度同时收敛且稳定，过大或过小的指数分别导致发散或停滞，10 个种子的窄置信区间增强了结论的可靠性。

### 采样/计算效率对比（CartPole-v1，vs clip-PPO）

| 方法 | 数据需求 | 表现 | 吞吐（steps/s） |
|------|----------|------|------|
| clip-PPO（GAE） | 需完整轨迹段算优势，时间上非局部 | 收敛更慢、方差大，仅部分种子达最优 | 5406.0 |
| 本文 NPG（一步 TD） | 仅需 $(s_0,a_0,r_0,s_1,a_1)$ | 稳定解到最大回报 500 | 7210.1 |

本文 critic 用动作值 $Q(s,a)$、只靠单步 TD-error 学习，比 GAE 沿整段轨迹递归计算优势更轻量，吞吐高约 33%，同时样本效率更优。

### 关键发现
- **步长调度是 NPG+TD 稳定性与效率的决定性因素**：理论给出的 $k^{-1/2}$ 不是经验调参，而是被三套环境一致验证的"临界指数"，偏离即发散或停滞。
- **一步 TD critic 优于 GAE**：去掉对完整轨迹段的依赖，既提采样效率又提计算吞吐，说明把评估局部化（只用一步转移）在实践中是划算的。
- **采样需求随策略复杂度增长是真实约束**：当网络变复杂、可用样本不足以支撑训练时，$k^{-0.5}$ 曲线会回落，正对应理论里 $n(k)$ 随 $\|\pi_k\|_{\mathcal{H}}$ 上升的预测。
- 附录补充：implicit PPO 表现与本文 NPG 相近且迭代更稳（类比隐式 SGD），discrete SAC 找不到最优，DQN 高效但不稳。

## 亮点与洞察
- **用一个 RKHS 熵假设统一四类函数空间**：tabular、Sobolev、NTK、Gaussian 核在同一条 $\delta^{-2\beta}|\log\delta|^{2\kappa}$ 熵界下各自退化为极小极大最优率，把分散的 RL 函数逼近理论收进一个框架，可复用性强。
- **泛函梯度即隐式预条件**：把 KRR 不动点问题放到 RKHS 内积下做梯度下降，自动获得预条件并几何收敛，绕开立方阶矩阵求逆——这是"换内积"带来的免费午餐，思路可迁移到其他核方法的迭代求解。
- **把"每轮要多少样本"写成显式公式**：定理 13 的 mirror-descent 不等式清楚地把全局次优性拆成"评估误差 + 步长漂移 + 初始 KL"，从而能反推采样表，填补了旧 PPO/NPG 分析"假装期望精确"的洞。
- **NTK 桥接理论与实践**：因为核梯度下降镜像宽网络的 NTK 动力学，RKHS 结论可直接指导神经 critic/actor，实验里用 MLP 就能复现理论趋势。

## 局限与展望
- **收敛分析在 $L^2$ 度量下建立极小极大最优性，但 RL 更该用 $L^\infty$**：而不动点 KRR 形式与标准 KRR 差别很大，$L^\infty$ 下的分析作者留作未来工作；这意味着定理 13 分子里的 $L^\infty$ 评估误差与定理 10 给的 $L^2/$训练集误差之间还有 gap 需要桥接。
- **RKHS 范数 $\|\pi_k\|_{\mathcal{H}}$ 难以估计**：尤其 NTK 情形策略对应深网络，范数发散且不可直接测，只能用网络稳定性/结构当代理，采样表在实践中难以精确执行。
- **依赖采样质量假设（Assumption 1 & 6）**：要求初始分布有界、一步转移不远离初始分布（$c\gamma<1$），实际中需靠先验或 MCMC 选好 $\mu_0$，否则常数 $1-c\gamma$ 变小、采样需求骤增。
- **理论两阶段与实现联合优化不完全一致**：实践里把评估与改进合成统一损失联合训练，与"评估 $T$ 步再改进一步"的分析对象有偏差，理论保证是否严格覆盖该实现值得商榷。
- **作者展望**：可推广到更一般的 Mirror Descent 策略优化框架。

## 相关工作与启发
- **vs 神经 TD（Cai et al., 2019）**：他们首次给出过参数化网络下神经 TD 的有限样本分析，但只是亚线性率、且限于连续状态-离散动作；本文用 RKHS 泛函梯度得到几何收敛 + 极小极大率，并覆盖连续动作。
- **vs 神经 policy 的 mirror descent（Liu et al., 2019）**：他们在 NTK 区证两层过参数化神经策略的非渐近全局收敛（连续状态-离散动作）；本文把 NPG 推广到一般 RKHS（含连续动作），并显式给出每轮采样量。
- **vs 核 LSTD（Duan et al., 2024）**：他们提出 RKHS 里正则化核 LSTD 估计器、给非渐近界与匹配下界，但针对与动作无关的 Markov 链；本文用**梯度式**核 TD（非最小二乘 TD）处理含动作的策略评估，并把评估耦合进策略改进。
- **vs 原版 PPO/TRPO（Schulman et al., 2015/2017）**：原版把期望项当精确、不指明数据需求；本文区分 explicit PPO(NPG) 与 implicit PPO 两种 KL 方向，并对前者给出达到 $O(k^{-1/2})$ 所需的采样复杂度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 RKHS 熵统一四类函数空间、把评估误差与采样复杂度显式耦合，理论视角新颖。
- 实验充分度: ⭐⭐⭐ 三个标准控制任务足以验证步长调度与效率趋势，但作为理论文章实验偏轻、未刷大规模 benchmark。
- 写作质量: ⭐⭐⭐⭐ 定理-推论-采样表层层递进，主线清晰；但符号密集、$L^2$ 与 $L^\infty$ 度量的衔接需读附录才清楚。
- 价值: ⭐⭐⭐⭐ 为 PPO/NPG 在富函数逼近下的全局收敛提供更坚实的理论地基，并给出可操作的采样/步长规则。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] 为什么高秩神经网络也能泛化？：基于 RKHS 的代数框架](why_high-rank_neural_networks_generalize_an_algebraic_framework_with_rkhss.md)
- [\[ICLR 2026\] Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and Beyond](complexity_analysis_of_normalizing_constant_estimation_from_jarzynski_equality_t.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] Stable Coresets: Unleashing the Power of Uniform Sampling](stable_coresets_unleashing_the_power_of_uniform_sampling.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Policy Newton Algorithm in Reproducing Kernel Hilbert Space
description: >-
  [ICLR 2026][强化学习][RKHS] 本文提出 RKHS 中第一个二阶策略优化方法 Policy Newton in RKHS：通过三次正则化的辅助目标绕开无穷维 Hessian 求逆，再用 Representer 定理把无穷维优化等价转化为维度随轨迹数据量 $NT$ 增长的有限维问题，理论上证明收敛到局部最优且具备局部二次收敛率，实验上比一阶 RKHS 方法和参数化二阶方法收敛更快、回报更高。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "RKHS"
  - "二阶优化"
  - "Policy Newton"
  - "Representer 定理"
  - "二次收敛"
---

# Policy Newton Algorithm in Reproducing Kernel Hilbert Space

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PYnLd91wZY](https://openreview.net/forum?id=PYnLd91wZY)  
**代码**: 待确认（论文附匿名补充材料，含完整实现）  
**领域**: 强化学习 / 策略优化理论  
**关键词**: RKHS、二阶优化、Policy Newton、Representer 定理、二次收敛

## 一句话总结
本文提出 RKHS 中第一个二阶策略优化方法 Policy Newton in RKHS：通过三次正则化的辅助目标绕开无穷维 Hessian 求逆，再用 Representer 定理把无穷维优化等价转化为维度随轨迹数据量 $NT$ 增长的有限维问题，理论上证明收敛到局部最优且具备局部二次收敛率，实验上比一阶 RKHS 方法和参数化二阶方法收敛更快、回报更高。

## 研究背景与动机

**领域现状**：把强化学习策略表示在再生核希尔伯特空间（RKHS）里是一种强大的非参数方案——它定义在无穷维函数空间中，具备万能逼近能力，并且借助 Representer 定理，策略更新只发生在已观测数据点张成的有限维子空间里，模型规模能随任务复杂度自适应。这种表示在样本受限、对鲁棒性要求高的安全攸关场景中尤其有吸引力，已经在 meta-RL、分布式 RL 等方向取得成功。

**现有痛点**：但 RKHS 策略的优化方法至今几乎只停留在一阶。当前标准做法 RKHS Policy Gradient（Paternain et al., 2020）通过往 RKHS 里加入梯度导出的函数来更新策略，因此继承了所有一阶方法的通病——在曲率大、存在狭长峡谷的复杂优化地形上收敛很慢，而且梯度方向不是尺度不变的，搜索方向常常被严重地病态缩放。

**核心矛盾**：在参数化策略里，二阶（Newton）方法靠引入 Hessian 曲率信息能显著加速、给出尺度更合理的更新步，是治这个病的良方。但把 Newton 法直接搬到 RKHS 里却卡在一个根本困难上：RKHS 中 Hessian 的对应物是期望累积回报的二阶 Fréchet 导数，它是作用在函数空间上的**算子**，是无穷维的；标准 Newton 步需要显式求这个算子的逆，而在无穷维空间这通常不可行。更本质地说，这个 Hessian 算子在 RKHS 里是迹类（trace-class）因而紧致的，而无穷维空间中紧算子不存在有界逆，导致标准 Newton 步本身就是病态（ill-posed）的。

**本文目标**：在 RL 这种数据分布随策略漂移的设定下，造出第一个能落地、且有收敛保证的 RKHS 二阶策略优化算法。已有的 RKHS Newton 研究都集中在数据同分布的在线学习/regret 设定，无法直接迁移到 RL。

**切入角度 + 核心 idea**：不要硬碰硬地去求那个无穷维 Hessian 的逆，而是**优化一个三次正则化的辅助目标**来得到 Newton 步，再用 **Representer 定理**证明这个无穷维优化等价于一个有限维欧氏空间问题，维度只随轨迹数据量 $N\times T$ 增长——"用求解有限维三次正则子问题，代替无穷维 Hessian 求逆"。

## 方法详解

### 整体框架

整篇方法回答一个问题：在无法显式表示、更无法求逆的无穷维 RKHS Hessian 下，如何拿到一个二阶的策略更新步？整体思路分三步走通。

第一步，把策略直接建模成 RKHS 中的函数 $h$ 而非参数向量 $\theta$。对离散动作空间，策略写作 $\pi_h(a_t\mid s_t)=\frac{1}{Z}e^{T h(s_t,a_t)}$，其中 $Z=\sum_{a'}e^{T h(s_t,a')}$ 是归一化常数、$T$ 是温度。一阶 Fréchet 导数 $\nabla_h J(\pi_h)$ 仍是 RKHS 里的一个元素，可由核截面表达；而二阶 Fréchet 导数 $\nabla_h^2 J(\pi_h)$ 则升格为作用在函数空间上的算子，可视作张量积空间 $\mathcal H_K\otimes\mathcal H_K$ 中的元素。

第二步，承认 $\nabla_h^2 J$ 无法显式写出、其逆更不可求，于是改为**求解一个三次正则化辅助函数**来得到 RKHS Newton 步 $\Delta h$（见关键设计 1）。这一步把"求逆"换成了"优化"，但优化变量仍在无穷维空间 $\mathcal H_K$ 里。

第三步，用 **Representer 定理**把这个无穷维优化坍缩成有限维问题：最优解必形如数据核截面的线性组合 $\bar h_\alpha=\sum_{i,t}\alpha^i_t K((s^i_t,a^i_t),\cdot)$，于是只需求有限个系数 $\bar\alpha\in\mathbb R^{NT}$。代入辅助目标并利用再生性，二次项和三次项都能写成 $\bar\alpha$ 的显式代数形式（Theorem 3.3，见关键设计 2）。最后把这个带三次正则的有限维二次问题交给共轭梯度法求解（关键设计 3），解出 $\bar\alpha$ 后重建 $\Delta h$、按 $h_{m+1}=h_m+\eta\Delta h$ 更新策略，循环采样—估计—求解—更新，直到收敛。整套流程的收敛性与二次收敛率由第 4 节理论给出（关键设计 4）。

### 关键设计

**1. 三次正则辅助函数：用优化绕开无穷维 Hessian 求逆**

这一步直击核心矛盾——RKHS Hessian 算子是紧的、无有界逆，标准 Newton 步 $[\nabla_h^2\hat J]^{-1}\nabla_h\hat J$ 根本无法定义。本文借鉴有限维 Newton 中的立方正则思路（Nesterov–Polyak 系），不去求逆，而是把 Newton 步 $\Delta h$ 定义为如下辅助泛函的极小元：

$$\Delta h=\arg\min_{\bar h\in\mathcal H_K}\left\langle\nabla_h\hat J(h_k),\bar h\right\rangle+\frac12\left\langle\nabla_h^2\hat J(h_k)\circ\bar h,\bar h\right\rangle+\frac{\beta}{6}\|\bar h\|^3.$$

关键在于：虽然 Hessian 算子 $\nabla_h^2\hat J(h_k)=\frac1N\sum_{\omega}H_h(\omega;\pi_h)$ 本身不可显式计算，但二阶项 $\langle\nabla_h^2\hat J\circ\bar h,\bar h\rangle$ 这个**双线性型的取值**借助张量积 RKHS 的定义是容易算的——也就是说，我们永远不需要把算子写出来，只需要会算它作用在具体函数上的内积。三次项 $\frac\beta6\|\bar h\|^3$ 提供正则，保证子问题良定、解唯一且稳定，$\beta$ 是正则强度超参。这样"求逆"被替换成"求一个良定凸性辅助问题的极小"。

**2. Representer 定理降维：把无穷维优化等价为 $NT$ 维问题**

设计 1 给出的辅助问题优化变量仍在无穷维 $\mathcal H_K$ 里，没法直接算。Representer 定理（Schölkopf et al., 2001）保证：极小化"代价 + 范数正则"形式泛函的解一定能写成训练样本核截面的有限线性组合。代入 $\bar h_\alpha=\sum_{i=1}^N\sum_{t=1}^T\alpha^i_t K((s^i_t,a^i_t),\cdot)$，把对函数 $\bar h$ 的搜索转成对系数向量 $\bar\alpha\in\mathbb R^{NT}$ 的搜索。利用再生性 $\langle K(x,\cdot),K(y,\cdot)\rangle=K(x,y)$，Theorem 3.3 把辅助目标显式化为一个带三次正则的有限维二次优化：

$$\bar\alpha^*=\arg\min_{\bar\alpha\in\mathbb R^{NT}}\left\langle v,\bar\alpha\right\rangle+\frac12\left\langle H\bar\alpha,\bar\alpha\right\rangle+\frac{\beta}{6}\|\bar\alpha\|_2^3.$$

其中一阶系数向量 $v$ 是把泛函梯度 $\nabla_h\hat J$ 投影到数据核子空间上的结果；二阶系数矩阵分解为 $H=\frac{T^2}{N}bc^\top-\frac{T}{N}\sum_l\Psi_l(\omega)\Sigma^{(l)}$，第一项 $\frac{T^2}{N}bc^\top$ 对应一阶梯度的外积、捕捉曲率的秩一部分，第二项含 $\Sigma^{(l)}$ 是策略动作分布的协方差结构（$\Sigma^{(l)}_{ij}=\mathrm{Cov}_{a'\sim\pi(\cdot\mid s_l)}[K'_{il},K'_{jl}]$）。所有 $v$、$b$、$c$、$\Sigma^{(l)}$ 都只由核值 $K_{il}=K((s_i,a_i),(s_l,a_l))$ 及其对动作的期望构成，完全可计算。问题复杂度只依赖数据量 $N\times T$，这与 SVM、RBF 网络等核方法同一性质——模型规模随数据自适应。

**3. 共轭梯度求解子问题：避开解析解的训练不稳定**

有限维问题 7 有两条解法：(1) 直接对目标求导置零解临界点；(2) 用经典优化器（梯度下降 / Newton / 共轭梯度）迭代求解。本文明确选 (2) 中的**共轭梯度法**，理由是解析法 (1) 虽然概念上直接，但会给训练注入显著不稳定，在复杂环境中甚至导致误差指数级增长。共轭梯度在保持求解效率的同时数值更稳，是更实用的折中。整个算法（Algorithm 1）即：每轮用当前策略 $\pi_{h_m}$ 采 $N$ 条轨迹 → 按 Theorem 3.3 估计 $v$ 和 $H$ → 共轭梯度解出 $\bar\alpha$ → 用 $\bar\alpha$ 经 Representer 重建更新步 $\Delta h$ → 更新 $h_{m+1}=h_m+\eta\Delta h$。

**4. 收敛性与局部二次收敛率：保证降维没有损失二阶性质**

把无穷维问题降到有限维后，必须证明二阶优化的"快"没有在这一步丢掉。本文先在随机（采样估计）设定下证明**收敛到驻点**：依赖 Hessian Lipschitz 连续假设（常数 $L\le\beta$）导出泰勒上界 $J(h_2)\le J(h_1)+\langle\nabla_h J,h_2-h_1\rangle+\frac12\langle\nabla_h^2 J\circ(h_2-h_1),h_2-h_1\rangle+\frac L6\|h_2-h_1\|^3$，再分别给出更新步范数的上界与下界（蒙特卡洛估计以 $O(1/\sqrt N)$ 收敛），最终证明在随机抽取的迭代 $R$ 上期望梯度范数趋零：$\lim_{M,N\to\infty}\mathbb E[\|\nabla J(h_{R+1})\|]=0$。进一步在确定性理想设定下（假设可拿到真梯度与真 Hessian、且正则化 Hessian 的逆一致有界），证明**局部二次收敛**：存在常数 $C_q>0$ 使 $\|h_{k+1}-h^*\|\le C_q\|h_k-h^*\|^2$。这正是二阶方法相对一阶 RKHS Policy Gradient（一阶收敛率）的核心优势所在。

### 损失函数 / 训练策略

优化目标是把累积回报取负后的期望折扣代价 $J(\pi)=\mathbb E_{\omega\sim p(\omega;\pi)}\big[\sum_{t=0}^{T-1}\gamma^{t-1}r(s_t,a_t)\big]$（约定瞬时代价为负回报，从而把 RL 写成最小化问题）。每轮以 $N$ 条轨迹做蒙特卡洛估计 $v$、$H$，三次正则强度 $\beta$ 需满足 $\beta\ge L$（Hessian Lipschitz 常数），子问题用共轭梯度求解，更新学习率 $\eta$。核函数统一用高斯核。连续动作（如 Hopper）下用连续高斯策略，二阶 RKHS 步的推导见附录 I。

## 实验关键数据

实验分两块：(a) 简化版 Asset Allocation 玩具环境，用来验证二次收敛的理论性质；(b) Gymnasium 标准 RL 基准（离散 CartPole / Lunar Lander，连续 Inverted Pendulum / Hopper）验证实用性能。统一用高斯核，参数化基线用线性 + 多项式特征扩展以保证表示能力公平。

### 主实验（玩具环境：二次收敛验证）

| 方法 | 收敛行为 | 最终策略 |
|------|----------|----------|
| Policy Gradient（参数化） | 收敛慢 | 收敛但慢 |
| Policy Newton（参数化二阶） | 比一阶快，但轨迹不稳 | 落入次优局部最大 |
| Policy Gradient in RKHS | 快速收敛 | 接近最优 |
| **Policy Newton in RKHS（本文）** | **明显二次收敛** | **逼近全局最优** |

关键观察：玩具环境里四种方法表示能力相同（状态/动作空间有限，都直接优化每个 (s,a) 的概率值），都能精确表示最优策略，所以差异**纯粹来自优化几何而非表达力**。RKHS 更新张成数据相关的核基，提供更丰富的下降方向，能逃离困住参数化方法的次优吸引域。

### 训练性能（Gymnasium 基准，5 次独立运行均值 + 95% 置信区间）

| 环境 | 动作空间 | 本文表现 | 对比基线 |
|------|---------|---------|---------|
| CartPole | 离散 | 快速收敛、样本效率高 | 显著优于一阶 RKHS 与参数化 Newton |
| Lunar Lander | 离散 | 快速收敛、样本效率高 | 同上 |
| Inverted Pendulum | 连续 | 快速稳定 | 优于基线 |
| Hopper | 连续 | 用更少迭代达到高回报 | 样本效率明显领先 |

### 关键发现
- **二次收敛是真的**：玩具环境中 Policy Newton in RKHS 在逼近最优策略时呈现清晰的二次收敛曲线，而参数化 Policy Newton 虽快于一阶但轨迹震荡且陷入次优——说明把二阶信息放进 RKHS 表示比放进参数化表示更稳。
- **优势源自曲率 + 表达力双重作用**：作者把性能归因于两点——Hessian 算子提供的曲率信息帮助穿越复杂地形，以及 RKHS 灵活的表示能力提供更丰富的下降方向。
- **离散/连续都成立**：方法不仅在离散动作（直接用第 3 节框架）上有效，经附录 I 的高斯策略扩展后在连续高维控制（Hopper）上同样领先，验证了通用性。

## 亮点与洞察
- **"不求逆，改求解"的范式迁移很巧妙**：把有限维立方正则 Newton 的思路搬到无穷维算子上，规避了紧算子无有界逆这一根本障碍——核心 trick 是二阶项只以双线性内积形式出现，永远不需要把算子物化。
- **Representer 定理在这里发挥了"降维不失真"的作用**：无穷维优化坍缩到 $NT$ 维，且 Theorem 3.3 还证明降维后二次收敛率被完整保留——这是把核方法和二阶优化真正缝合起来的关键一步。
- **诊断式实验设计值得借鉴**：玩具环境特意让四种方法表示能力等同，从而把"优化几何"和"表达力"两个混杂因素解耦，干净地证明 RKHS 提供的是更好的下降方向而非更强的表示——这种把变量控住再下结论的实验思路很有说服力。
- $H=\frac{T^2}{N}bc^\top-\frac TN\sum_l\Psi_l\Sigma^{(l)}$ 的"秩一外积 + 协方差"分解给曲率矩阵了清晰物理意义，可迁移到其他核化二阶方法。

## 局限与展望
- **二次收敛率只在确定性理想设定下证明**：需要假设能拿到真梯度与真 Hessian、且正则化 Hessian 逆一致有界；更现实的随机假设下的完整二次收敛分析被留作未来工作。
- **复杂度随数据量增长**：问题维度是 $N\times T$，与 SVM 等核方法同病——在大数据量/长 horizon 下核矩阵规模会成为计算瓶颈，论文未给出稀疏化/近似方案。
- **未上更复杂环境**：作者自承在 Humanoid 这类高复杂任务上可能暴露鲁棒性与稳定性问题；提出的改进方向是把神经网络与 Policy Newton in RKHS 框架结合。
- **自己发现的局限**：实验环境（CartPole / Lunar Lander / Hopper）相对偏小，缺少与现代深度 RL（PPO/SAC 等）的对比；玩具环境的"全局最优可显式表示"是为验证理论而专门简化的，普适性需谨慎看待。

## 相关工作与启发
- **vs RKHS Policy Gradient（Paternain et al., 2020；Lever & Stafford, 2015）**：他们是 RKHS 一阶方法，靠加梯度函数更新、一阶收敛率；本文是首个 RKHS 二阶方法，引入曲率信息、局部二次收敛，在病态地形上更快且更易逃离次优。
- **vs 参数化 Policy Newton（Li et al., 2023；Maniyar et al., 2024）**：他们在有限维参数空间求 Hessian 及其立方正则步；本文把二阶优化提升到无穷维 RKHS 表示，实验显示在相同表示能力下 RKHS 版本优化几何更好、不易陷入次优局部最大。
- **vs RKHS 在线 Newton（Calandriello et al., 2017a,b；Lu et al., 2016）**：他们在同分布在线学习设定下迭代近似 Hessian 逆；本文针对 RL 中分布随策略漂移的设定，且给出收敛保证，填补了非参数策略表示与二阶优化之间的空白。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 RKHS 二阶策略优化框架，"立方正则 + Representer 定理"组合解决无穷维 Hessian 求逆，理论切口清晰。
- 实验充分度: ⭐⭐⭐ 玩具 + 4 个 Gym 基准验证了收敛性与样本效率，但环境偏小、缺与现代深度 RL 基线的对比。
- 写作质量: ⭐⭐⭐⭐ 理论推导层次分明，从动机到降维到收敛性一气呵成，物理直觉解释到位。
- 价值: ⭐⭐⭐⭐ 把核方法与二阶优化缝合，为样本受限/安全攸关场景的非参数 RL 提供了更快收敛的理论工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[AAAI 2026\] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](../../AAAI2026/reinforcement_learning/qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICLR 2026\] Grouping Nodes with Known Value Differences: A Lossless UCT-based Abstraction Algorithm](grouping_nodes_with_known_value_differences_a_lossless_uct-based_abstraction_alg.md)
- [\[ICLR 2026\] Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)

</div>

<!-- RELATED:END -->

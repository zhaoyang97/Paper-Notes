---
title: >-
  [论文解读] Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling
description: >-
  [ICLR 2026][学习理论][Pass@k 推理] 本文把 LLM 的 Pass@k 推理（采 $N$ 个候选、提交至多 $k$ 个、只看最好那个）形式化为一个 regret 最小化问题，证明多数投票和 Best-of-N 在这个设定下都不是最优的，提出"先按出现频率预筛、再按奖励取 top-$k$"的 Best-of-Majority（BoM）策略，给出 $\tilde O(\epsilon_{\mathrm{opt}}+\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$ 的 regret 上界并配上匹配的下界，从而首次给出 Pass@k 推理的极小极大最优算法。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "测试时计算"
  - "Pass@k 推理"
  - "推理缩放"
  - "极小极大最优"
  - "Best-of-N"
  - "多数投票"
---

# Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=a6CVQpjbXq](https://openreview.net/forum?id=a6CVQpjbXq)  
**代码**: 待确认  
**领域**: 学习理论 / 测试时计算  
**关键词**: Pass@k 推理, 推理缩放, 极小极大最优, Best-of-N, 多数投票

## 一句话总结
本文把 LLM 的 Pass@k 推理（采 $N$ 个候选、提交至多 $k$ 个、只看最好那个）形式化为一个 regret 最小化问题，证明多数投票和 Best-of-N 在这个设定下都不是最优的，提出"先按出现频率预筛、再按奖励取 top-$k$"的 Best-of-Majority（BoM）策略，给出 $\tilde O(\epsilon_{\mathrm{opt}}+\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$ 的 regret 上界并配上匹配的下界，从而首次给出 Pass@k 推理的极小极大最优算法。

## 研究背景与动机
**领域现状**：训练时的 scaling law 已经被研究得很透，但训练昂贵，于是大家转向"推理时缩放"——固定模型，靠多采样来提升效果。最常见的两种聚合策略是**多数投票**（majority voting，选出现频率最高的答案）和 **Best-of-N**（BoN，按奖励模型打分选最高的）。

**现有痛点**：现实评测里普遍用的是 Pass@k 指标——允许模型提交最多 $k$ 个答案，只要有一个对就算解出。但已有的推理缩放理论几乎都只分析"输出单个答案"（Pass@1）的情形：Wu et al. (2024b) 给了多数投票的收敛率，Huang et al. (2025a) 给了 BoN 在 $k=1$ 时的上界。**没有人刻画 regret 对 $k$ 的依赖**，也没人回答"在 Pass@k 下到底什么策略最优"。

**核心矛盾**：多数投票只利用参考策略 $\pi_{\mathrm{ref}}$ 的分布信息（频率），完全不看奖励；BoN 只信奖励模型 $\hat r$，完全不看分布。前者在 $\pi_{\mathrm{ref}}$ 给错误答案分配高概率时会被带偏，后者在奖励模型不准时会 reward overoptimization。两种信息源都没被充分利用。

**本文目标**：拆成两个根本问题——(Q1) Pass@k 推理 regret 的最优缩放（关于 $k$、$N$）是什么？(Q2) 哪种策略既能达到这个最优界、又是 scaling-monotonic（随 $N$ 增大性能不退化）？

**切入角度**：作者用强化学习里的**悲观主义（pessimism）**原则——$\pi_{\mathrm{ref}}$ 很少采到的答案，奖励模型在那片区域监督稀疏、误差大，所以不该轻信奖励。

**核心 idea**：把多数投票的"频率筛选"和 BoN 的"奖励排序"缝合起来——**先用频率把奖励不可信的低概率候选剔掉，再在可信子集上按奖励取 top-$k$**，从而同时吃到两种信息的红利。

## 方法详解

### 整体框架
这篇论文先建立一个 **Pass@k 推理缩放问题**的理论框架，再在框架里证明现有策略次优、提出新策略 BoM 并证其最优。

设定如下：给定 prompt $x$，参考策略 $\pi_{\mathrm{ref}}(\cdot\mid x)$ 独立采样 $N$ 个候选答案 $\hat Y=\{\hat y_1,\dots,\hat y_N\}$，算法从中挑出至多 $k$ 个提交，目标是**让这 $k$ 个里真实奖励最高的那个尽量接近最优**。形式化的 regret 为

$$\mathrm{Regret}(x)=\mathbb{E}_{\pi^*}\big[r^*(x,\cdot)\big]-\mathbb{E}_{y_1,\dots,y_k}\Big[\max_{1\le i\le k} r^*(x,y_i)\Big],$$

其中 $r^*:\mathcal X\times\mathcal Y\to[0,1]$ 是真实奖励，$\pi^*$ 是 $r^*$ 的最大化者。算法手上只有一个**不完美的奖励模型** $\hat r$，其精度由两个量刻画：整体平方误差 $\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}[(r^*-\hat r)^2]\le\epsilon_{\mathrm{RM}}^2$，以及最优答案处的误差 $\epsilon_{\mathrm{opt}}=|r^*(x,y^*)-\hat r(x,y^*)|$。在数学题这类有唯一可验证答案的任务里，$r^*$ 退化成 $\{0,1\}$ 验证器，最小化 regret 等价于最大化"$k$ 个里至少一个对"的概率，正好就是 Pass@k。

BoM 的执行流程很短：**采 $N$ 个候选 → 算每个答案的经验频率 $\hat\pi(y)$ → 丢掉频率低于阈值 $\alpha$ 的候选得到 $\hat Y_\alpha$ → 在 $\hat Y_\alpha$ 上查奖励、取 top-$k$**。和多数投票的区别是后面接了奖励排序，和 BoN 的区别是前面加了频率门槛——这道门槛正是全篇理论的胜负手。

### 关键设计

**1. Pass@k 推理的 regret 形式化与覆盖系数：把"分布质量"压成一个标量**

已有推理理论只盯 Pass@1，本文第一步是把"提交 $k$ 个、取其最好"这件事写成上面的 regret 度量，并显式追踪它对 $k$ 的依赖（Remark 3.3 强调这是相比 sample-and-evaluate 框架的新焦点）。刻画参考策略好不好用的是 **L1 覆盖系数** $C^*(x):=\mathbb{E}_{y\sim\pi^*}[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]$。因为 Assumption 3.2 假设最优答案 $y^*$ 唯一、$\pi^*$ 是 one-hot，覆盖系数直接坍缩成一个干净的量：$C^*(x)=1/\pi_{\mathrm{ref}}(y^*\mid x)$——也就是参考策略采到正确答案的概率的倒数。$C^*$ 越大说明正确答案越罕见、问题越难，它将作为后面所有界里的"难度因子"。另一个关键性质 **scaling-monotonicity**（Definition 3.4）：若奖励模型足够准、采样数 $N$ 足够大，regret 能被压到任意小，且性能不会随 $N$ 增大而退化——这是实践中最想要的性质，因为难题只要无脑加大 $N$ 就行，不用精调。

**2. 高频预筛 + top-$k$ 奖励：用悲观主义把候选限制在"可信区"**

这是 BoM 的算法核心（Algorithm 3）。痛点是奖励模型 $\hat r$ 在 $\pi_{\mathrm{ref}}$ 罕见的答案上误差不可控，BoN 直接对全体 $N$ 个候选按 $\hat r$ 排序，等于把信任无差别地铺给所有人，容易被一个"被高估的坏答案"带偏。BoM 的做法是先算经验频率 $\hat\pi(y)=\frac1N\sum_{i=1}^N\mathbb 1(\hat y_i=y)$（它是 $\pi_{\mathrm{ref}}$ 的估计），只保留 $\hat Y_\alpha=\{y\in\hat Y:\hat\pi(y)\ge\alpha\}$，阈值取 $\alpha=3/(4C^*(x))$，再在这个"高频可信子集"上查奖励、选 $\hat y_1,\dots,\hat y_k=\text{Top-}k\{y\in\hat Y_\alpha:\hat r(y)\}$。为什么有效：高频意味着 $\pi_{\mathrm{ref}}(y\mid x)\ge\alpha$，这给了奖励误差一个杠杆——证明里关键的不等式（5.1）

$$\min_{i\in[k]}\Delta_i\le\sqrt{\textstyle\sum_{i=1}^k\Delta_i^2/k}\le\sqrt{\textstyle\sum_{i=1}^k\pi_{\mathrm{ref}}(\hat y_i\mid x)\Delta_i^2/(\alpha k)}\le\sqrt{\epsilon_{\mathrm{RM}}^2/(\alpha k)}$$

把每个候选的奖励误差 $\Delta_i=|\hat r-r^*|$ 用频率下界 $\alpha$ 兑换成整体误差 $\epsilon_{\mathrm{RM}}$，从而拿到 $1/\sqrt k$ 的衰减。一句话：**频率门槛把"不可信的奖励"挡在门外，剩下的奖励才敢用来排序**。附带好处是只对筛后子集查奖励，reward 调用次数也省了。

**3. 匹配的上下界：BoM 是极小极大最优**

光有算法不够，作者要证它"已经到顶了"。上界（Theorem 5.1）：取 $\alpha=3/(4C^*)$，则

$$\mathrm{Regret}(x)\le\epsilon_{\mathrm{opt}}(x)+O\!\Big(\sqrt{C^*(x)\epsilon_{\mathrm{RM}}^2(x)/k}\Big)+O\!\big(C^*(x)e^{-N/(32C^*(x))}\big),$$

当采样预算 $N\ge 16C^*\log(kC^*/\epsilon_{\mathrm{RM}}^2)$（即 $N=\tilde\Theta(C^*)$）时第三项指数级消失，得到 $\mathrm{Regret}\le\epsilon_{\mathrm{opt}}+\tilde O(\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$。下界（Theorem 6.1）：对**任意** Pass@k 算法，都存在一个困难实例使 $\mathrm{Regret}=\Omega(\epsilon_{\mathrm{opt}}+\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$。两者形式完全一致，说明 BoM 已经是极小极大最优，没有算法能在最坏情况下做得更好。这里有两个值得记的结构：$\epsilon_{\mathrm{opt}}$ 项不随 $k$ 衰减（最优答案处的奖励误差是绕不过的硬墙），而 $\epsilon_{\mathrm{RM}}$ 项以 $1/\sqrt k$ 衰减（多提交几个候选确实让问题变易）。证明上界时还有个技巧：经验频率 $\hat\pi$ 只是 $\pi_{\mathrm{ref}}$ 的近似，需要一个"夹逼"事件 $\mathcal E:Y_{1/C^*}\subset\hat Y_{3/(4C^*)}\subset Y_{1/(4C^*)}$ 来保证筛选集既不漏掉 $y^*$ 又不放进太低频的答案，用 Chernoff + 把无界集合装进有限个"桶"再做 union bound。

**4. 现有策略为何不是 scaling-monotonic：多数投票常数 regret、BoN 随 $N$ 退化**

为了凸显 BoM 的优势，作者反过来证两个 baseline 的下界。多数投票（Theorem 4.1）：即便奖励模型完美（$\epsilon_{\mathrm{RM}}=\epsilon_{\mathrm{opt}}=0$），只要构造多个比 $y^*$ 概率更高的"坏答案"，当 $N\ge 9C^*\log(2k+2)$ 时就会卡在 $\mathrm{Regret}=\Omega(1)$——加大 $N$ 或 $k$ 都救不回来，因为它根本不看奖励。BoN（Theorem 4.2）：若 $N\le C^*$ 是常数 regret；否则 $\mathrm{Regret}=\Omega(\min\{1,\sqrt{N\epsilon_{\mathrm{RM}}^2/k}\})$——注意这个界**随 $N$ 增大而变差**，因为采得越多，越容易撞上一个被奖励模型高估的坏答案（reward overoptimization）。两者都不是 scaling-monotonic。相比之下 BoM（Corollary 5.2）是 scaling-monotonic：$N\to\infty$、$\epsilon_{\mathrm{RM}}\to0$ 时 regret 收敛到 0，所以放心加大采样预算就行。作者还在 Remark 4.3 留了个开放问题：BoN 在 $k>1$ 时是否存在带最优 $1/\sqrt k$ 的上界，他们猜测**根本不可能**。

## 实验关键数据

### 主实验
参考策略用 Qwen3-4B-Instruct-2507，奖励模型用 AceMath-7B-RM，在 GSM8K（$N=2000$）、MATH-500、AIME24（$N=500$）上对比 BoM、多数投票、BoN，以及 SBoN / SBoN(C)（softmax 采样变体）。

| 数据集 | 设定 | BoM | 多数投票 | BoN | 结论 |
|--------|------|-----|----------|-----|------|
| MATH-500 | $k\in\{1,2,3,5,10\}$ | 全程领先 | 较低 | 居中 | BoM 对所有 $k$ 一致最优 |
| GSM8K | 小 $k$ | 领先 | 较低 | 接近 | 小 $k$ 时 BoM 明显优 |
| AIME24 | 小 $k$ | 领先 | 最低 | 接近 | 大幅超多数投票 |

### 随 $N$ 变化（验证 scaling-monotonicity）
在 AIME24 上固定 $k\in\{1,3,5\}$、$N$ 从 100 扫到 2000：

| 方法 | $N$ 增大时的行为 | 对应理论 |
|------|------------------|----------|
| 多数投票 | 始终低位、不受益于更大 $N$ | Theorem 4.1（常数 regret） |
| BoN | 随 $N$ 增大趋于下降 | Theorem 4.2（非单调） |
| BoM | $N\ge200$ 后稳定领先、不随 $N$ 显著下滑 | Corollary 5.2（单调） |

### 消融实验（阈值 $\alpha$，MATH-500 / Qwen3-4B）

| Pass@$k$ | 1 | 2 | 3 | 5 | 10 |
|----------|-----|-----|-----|-----|-----|
| $\alpha=0$（=BoN） | 83.6 | 87.2 | 88.8 | 89.2 | 90.0 |
| $\alpha=0.005$ | 86.8 | 89.2 | 89.6 | 89.8 | 90.2 |
| $\alpha=0.015$ | 87.4 | 88.8 | 89.4 | 89.4 | 89.8 |
| 多数投票 | 86.4 | 88.0 | 88.4 | 88.8 | 90.0 |

### 关键发现
- $\alpha=0$ 时 BoM 退化成 BoN，证实了频率门槛就是 BoM 区别于 BoN 的唯一关键开关。
- 增大 $\alpha$ 在小 $k$（尤其 $k=1$，83.6→87.4）时提升明显，但会略微损害大 $k$ 的表现——门槛越严越保守，候选集变小，在需要多样性的大 $k$ 下反而吃亏。
- 三种方法随 $N$ 的行为与理论一一对上：多数投票平、BoN 跌、BoM 稳，这是全文"理论预言→实验印证"最干净的一处。

## 亮点与洞察
- **频率当作 $\pi_{\mathrm{ref}}$ 的免费估计器**：BoM 不需要真去访问 $\pi_{\mathrm{ref}}$ 的概率，直接用经验频率 $\hat\pi$ 做门槛，再用一个夹逼事件把"频率筛选集 $\approx$ 概率筛选集"严格化，工程上零额外成本就拿到了悲观主义的好处。
- **把覆盖系数 $C^*$ 坍缩成 $1/\pi_{\mathrm{ref}}(y^*)$**：因为假设最优答案唯一，原本抽象的 L1 覆盖系数变成"参考模型采对的概率的倒数"，整个难度刻画一下子有了直觉——这条假设虽强，却让整套界异常干净。
- **上下界匹配 + scaling-monotonic 的双重卖点**：不仅给出最优算法，还顺手把两个最常用 baseline 的失败原因（多数投票不看奖励、BoN 被高估答案带偏）用困难实例钉死，理论叙事非常完整。
- **可迁移的设计哲学**："先用分布信息划定可信区、再用价值信息排序"这套悲观过滤，可以迁移到任何"生成器 + 不完美打分器"的选择问题（如 RLHF 的 best-of-n 采样、检索重排、agent 动作选择）。

## 局限与展望
- **唯一最优答案假设（Assumption 3.2）很强**：现实里很多任务的"对"不止一种写法或一个答案，一旦 $\pi^*$ 不是 one-hot，$C^*=1/\pi_{\mathrm{ref}}(y^*)$ 这个干净刻画就不成立，覆盖系数会依赖最优响应的选取（作者在结论里也点了这一点）。
- **极小极大（最坏情况）视角**：所有下界都是构造困难实例得到的，反映的是 worst-case，对"平均/典型实例"可能过于悲观；作者把 instance-dependent 下界列为未来工作。
- **阈值 $\alpha$ 与 $C^*$ 耦合**：理论取 $\alpha=3/(4C^*)$，但 $C^*$ 实践中未知，实验里 $\alpha$ 是手调（0.003~0.015 量级），$N=100$ 时还得单独调到 0.015，说明落地仍有调参负担。
- **依赖一个满足假设的预训练奖励模型**：奖励模型的训练被排除在分析之外，若 $\epsilon_{\mathrm{RM}}$ 实际不满足平方误差假设，界不一定成立。

## 相关工作与启发
- **vs 多数投票（Wang et al. 2022 等）**：多数投票只用频率、不看奖励，本文证明它在最坏情况下常数 regret、不 scaling-monotonic；BoM 在它前面保留频率筛选、后面接奖励排序，既继承"高频更可信"的直觉又补上价值信息。
- **vs Best-of-N（Lightman et al. 2023；Huang et al. 2025a）**：BoN 只信奖励、随 $N$ 增大反而退化（reward overoptimization）。本文在 $k=1$ 时恢复 Huang et al. (2025a) 的 $\Omega(\min\{1,\sqrt{N\epsilon_{\mathrm{RM}}^2}\})$，并在 $k>1$ 揭示出额外的 $1/\sqrt k$ 因子；BoM 通过频率门槛拿到 $1/\sqrt k$ 最优缩放且单调。
- **vs SBoN（Aminian et al. 2025；Verdun et al. 2025）**：SBoN 用 softmax 温度 $\beta$ 软化 BoN 来缓解过优化，本文作为额外 baseline 对比，BoM 用硬频率门槛而非软温度，且带极小极大最优保证。
- **vs Pass@k 训练（Tang et al. 2025；Chen et al. 2025b）**：这些工作把 Pass@k 当训练目标，本文是第一个把 Pass@k 当**推理**问题做理论刻画的，二者互补——一个对齐训练、一个分析推理选择。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 Pass@k 推理形式化并给出极小极大最优算法，揭示了 $1/\sqrt k$ 这一被忽略的因子。
- 实验充分度: ⭐⭐⭐⭐ 三个数学数据集 + $k$/$N$/$\alpha$ 三组消融，理论预言与实验对得很齐，但都集中在数学验证类任务。
- 写作质量: ⭐⭐⭐⭐⭐ 用一张对比表把多数投票/BoN/BoM/下界的 regret 和单调性讲清楚，理论叙事干净。
- 价值: ⭐⭐⭐⭐ 为"生成器+不完美奖励"的候选选择提供了原则性算法和理论基线，悲观过滤思路可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Minimax-Optimal Aggregation for Density Ratio Estimation](minimax-optimal_aggregation_for_density_ratio_estimation.md)
- [\[ICLR 2026\] Transformers as Measure-Theoretic Associative Memory: A Statistical Perspective and Minimax Optimality](transformers_as_measure-theoretic_associative_memory_a_statistical_perspective_a.md)
- [\[ICLR 2026\] Learning Shrinks the Hard Tail: Training-Dependent Inference Scaling in a Solvable Linear Model](learning_shrinks_the_hard_tail_trainingdependent_inference_scaling_in_a_solvable.md)
- [\[ICLR 2026\] A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits](a_near-optimal_best-of-both-worlds_algorithm_for_federated_bandits.md)
- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)

</div>

<!-- RELATED:END -->

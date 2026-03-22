# Extreme Value Monte Carlo Tree Search for Classical Planning

**会议**: AAAI 2026  
**arXiv**: [2405.18248](https://arxiv.org/abs/2405.18248)  
**代码**: https://github.com/guicho271828/pyperplan-mcts  
**领域**: 经典规划 / 搜索算法  
**关键词**: MCTS, 极值理论, UCB1-Uniform, Generalized Pareto, 经典规划, 启发式搜索, Full Bellman Backup

## 一句话总结

利用 Peaks-Over-Threshold 极值理论（POT EVT）为经典规划中 MCTS 的 Full Bellman Backup 提供统计理论基础，提出 UCB1-Uniform bandit 算法，用均匀分布（Generalized Pareto 的特例）的 MLE 估计指导动作选择，在 Pyperplan 上以 $10^4$ 节点预算超越 GBFS 67.8 个实例、超越 Softmin-Type(h) 33.2 个实例。

## 研究背景与动机

1. **领域现状**：MCTS + MAB 在博弈和强化学习中成功，但在经典规划中长期表现不佳。前序工作（Wissow & Asai 2024）发现 UCB1 的有限支撑假设 $[0, c]$ 与规划启发式值不匹配，提出高斯 bandit（UCB1-Normal2）改善了性能。
2. **现有痛点**：UCB1-Normal2 虽然不再违反支撑假设，但高斯分布的支撑 $(-\infty, +\infty)$ 是对启发式值的**欠规范**（under-specification），因为启发式值实际上是半有界的：$h \in [h^+, \infty)$（如 $h^{FF}$）或 $h \in [0, h^+]$（如 $h^{max}$），支撑范围本应更窄。
3. **核心矛盾**：Full Bellman Backup（回传子节点中的最小/最大值）在实践中有效，但在 UCB1/UCB1-Normal2 的理论框架中缺乏统计正当性——这些 bandit 的遗憾界证明围绕**均值**而非**极值**展开，形成理论与实践的脱节。
4. **本文要解决什么**：(a) 为启发式值找到理论上更准确的分布建模（既不过度规范也不欠规范）；(b) 为 Full Bellman Backup 提供严格的统计理论基础；(c) 解决死端节点 $h = \infty$ 对均值估计的破坏性影响。
5. **切入角度**：从极值统计学（EVT）出发，利用 POT 极限定理——样本超过高阈值的部分收敛到 Generalized Pareto (GP) 分布——统一解释规划中对极值（最小 $h$）的关注，以及为何可以丢弃死端样本。
6. **核心idea**：将 GP 分布特化为均匀分布 $U(l, u)$（对应 $\xi = -1$ 的短尾 GP），其 MLE 恰好是样本的 min/max，与 Full Bellman Backup 天然对应；基于此设计 UCB1-Uniform bandit，具有多项式~常数遗憾界。

## 方法详解

### 整体框架

在 MCTS 框架（Trial-Based Heuristic Tree Search）中，用 UCB1-Uniform 替代 UCB1/UCB1-Normal2 作为动作选择的 MAB 算法。回传策略使用 Full Bellman Backup（传播子树叶节点的 min/max），这与 Uniform 分布的 MLE 估计在理论上一致。搜索聚焦于 agile 设定（不关心解质量，只关心找到解的速度）。

### 关键设计1：POT EVT 统计框架

- **做什么**：用 Peaks-Over-Threshold 极值理论建模启发式值分布，将样本超过阈值 $\theta$ 的部分视为服从 GP 分布 $\text{GP}(\theta, \sigma, \xi)$。
- **核心思路**：POT 极限定理类比于中心极限定理——CLT 说样本均值趋向高斯，POT 说样本极值趋向 GP，两者都不假设原始分布的具体形状。短尾 GP（$\xi < 0$）有上界 $\theta - \sigma/\xi$，可估计 $-h^+$；且 GP 以 $x > \theta$ 为条件，自然地排除死端。
- **设计动机**：搜索天然聚焦于低 $h$ 值区域（实验验证仅 3.3% 节点的 $h(s) > h(I)$），因此无需显式设阈值，搜索本身已提供隐式过滤。

### 关键设计2：UCB1-Uniform Bandit

- **做什么**：提出新的 MAB 算法，假设每个臂的回报服从未知支撑的均匀分布 $U(l_i, u_i)$，选择使 LCB1-Uniform 最小的臂。
- **核心思路**：$\text{U/LCB1-Uniform}_i = \frac{\hat{u}_i + \hat{l}_i}{2} \pm (\hat{u}_i - \hat{l}_i)\sqrt{6 t_i \log T}$，其中 $\hat{l}_i = \min_j r_{ij}$，$\hat{u}_i = \max_j r_{ij}$。均值估计 $(\hat{u}_i + \hat{l}_i)/2$ 由 Uniform 的期望导出，exploration 项与支撑范围 $\hat{u}_i - \hat{l}_i$ 成正比。
- **设计动机**：(a) 均匀分布是 GP 的 $\xi = -1$ 特例，避免 GP 三参数估计的数值困难；(b) MLE 是 min/max，与 Full Bellman Backup 完美匹配，消除理论-实践脱节；(c) 支撑范围为零的 plateau（$\hat{u}_i = \hat{l}_i$）exploration 项也为零→深度优先穿越 plateau。

### 关键设计3：Spread-Aware 探索与 Plateau 逃逸

- **做什么**：当两个子树同样"有信息"（相同 $u, l$）但搜索量不同时，UCB1-Uniform 倾向于继续深搜已探索更多的子树。
- **核心思路**：exploration 项含 $\sqrt{t_i}$（而非 $\sqrt{1/t_i}$），拉的次数越多 exploration 项越大→倾向于继续，类似深度优先的 plateau 穿越策略。
- **设计动机**：在启发式值平坦的区域（plateau），均匀分摊搜索资源会导致两个方向都无法突破；集中力量于一个方向可更快找到出口。

### 关键设计4：死端处理的理论化

- **做什么**：将死端节点（$h = \infty$）从样本中排除。
- **核心思路**：GP 分布以 $x > \theta$ 为条件定义，低于阈值的值（包括 $-\infty$）自然被排除。
- **设计动机**：此前 Schulte & Keller (2014) 凭直觉排除死端缺乏统计正当性，POT 框架为其提供了理论依据。

## 实验关键数据

### 表1：Pyperplan 实验（10,000节点预算, $h^{FF}$, 5种子平均）

| 算法 | $h^{FF}$ | $h^{add}$ | $h^{max}$ | $h^{GC}$ | $h^{FF}$+PO |
|------|---------|----------|----------|---------|------------|
| GBFS | 538 | 518 | 224 | 354 | — |
| Softmin-Type(h) | 576 | 542.6 | 297.2 | 357.6 | 575.8 |
| GUCT-Normal2 | 582.95 | 538 | 316.6 | 380.6 | 623.2 |
| **GUCT-Uniform** | **606.4** | **563.4** | **455.6** | **492.2** | **635.6** |
| GUCT*-Normal2 | 567.2 | 533.8 | 263 | 341.2 | 619.8 |
| MaxSearch | 253.75 | 243.4 | 260 | 255.2 | 368.6 |

GUCT-Uniform 在所有启发式上均最优，尤其在弱启发式 $h^{max}$ 上优势巨大（+139 vs GUCT-Normal2）。

### 表2：Fast Downward IPC2018 实验（$h^{FF}$, 5min/8GB, 3种子平均）

| 算法 | 解题数 | IPC分数 |
|------|--------|---------|
| GBFS | 67.0 | 39.5 |
| Softmin-Type(h) | 79.6 | 47.0 |
| GUCT-Normal2 | 81.6 | 48.6 |
| **GUCT-Uniform** | **80.0** | **53.2** |

实例解题数接近（C++实现差异影响），但 IPC 分数（衡量速度）GUCT-Uniform 显著领先。

## 关键发现

1. **Full Bellman Backup 与 Uniform 分布理论一致**：min/max 回传恰好是 $U(l, u)$ 的 MLE，而非 UCB1 的 $[0, c]$ 或 Normal2 的 $\mathcal{N}(\mu, \sigma)$，消除了以往理论-实践脱节。
2. **非渐近最优 bandit 可优于渐近最优**：UCB1-Uniform（多项式遗憾界）显著优于 CHK-Uniform（渐近最优），类似 UCB1-Normal2 优于 UCB1-Normal 的现象。
3. **GUCT*-Normal2 劣于 GUCT-Normal2**：Full Bellman Backup 在高斯 bandit 上反而有害，因为 min/max 回传与均值估计矛盾。
4. **Max-$k$ bandit 范式不适用于规划**：三种 Max-$k$ 算法表现远差于 GUCT-Uniform（>300 实例差距），因为它们针对长尾而非短尾分布。
5. **弱启发式上优势更大**：$h^{max}$ 和 $h^{GC}$ 上 GUCT-Uniform 优势以百计，说明更好的探索策略在信息不足时尤为关键。

## 亮点与洞察

- **统计理论驱动的算法设计**：从"什么分布能正确建模启发式值"出发，用 EVT 推导出 bandit 算法，体现了理论先行的方法论——不是先设计算法再证性质，而是先定义分布再导出算法。
- **简洁优雅的特例选择**：将 GP 特化为 Uniform（$\xi = -1$）牺牲一个自由度换取参数估计的简单性，且 MLE 正好是 min/max，与已有的 Full Bellman Backup 无缝对接。
- **Plateau 逃逸机制**：UCB1-Uniform 的 exploration 项与 $\sqrt{t_i}$ 成正比（而非反比），在平坦区域自然形成深度优先策略，是对"explore = 分散"直觉的反转。
- **统一了三个独立问题**：支撑范围建模、Full Bellman Backup 正当性、死端处理，用同一个 POT 框架一次性解决。

## 局限性 / 可改进方向

1. **仅限 agile 设定**：未评估 satisficing/optimal 规划场景，UCB1-Uniform 在需要解质量时的表现未知。
2. **Uniform 分布是 GP 的粗近似**：$\xi = -1$ 固定了尾部行为，估计 $\xi$ 可能更精确但数值困难——未来可探索稳健的 GP 参数估计方法。
3. **C++ 实现差距**：Pyperplan 上的大量优势在 Fast Downward 上缩小（80 vs 81.6 解题数），底层实现效率对比较有影响。
4. **未与 novelty/BFWS 组合**：当前是单启发式 MCTS，与 Bilevel MCTS + novelty 的 Nεbula 组合是明确的未来方向。
5. **隐式阈值选择**：依赖搜索本身过滤高 $h$ 值状态，缺乏自适应阈值机制。

## 相关工作与启发

- **Wissow & Asai (2024)**：高斯 bandit UCB1-Normal2 的前序工作，本文在其基础上进一步精化分布假设。
- **Schulte & Keller (2014, THTS)**：经典规划 MCTS 框架，提出 Full Bellman Backup 但未给出 MAB 统计正当性。
- **Softmin-Type(h) (Kuroiwa & Beck 2022)**：多样化搜索的 SOTA，基于队列的竞争方法。
- **Max-$k$ Bandit (Cicirello & Smith 2004)**：虽同用 EVT 但针对长尾分布的极大值优化，与规划任务的短尾最小化目标相反。
- **启发**：MAB 算法设计必须与回报分布的统计性质匹配——错误的分布假设比错误的算法结构危害更大；EVT 在非机器学习场景（规划、组合优化）中有广阔应用空间。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将 POT 极值理论系统引入 MCTS for planning，统一解决三个理论缺陷
- **实验充分度**: ⭐⭐⭐⭐ 多启发式、多 bandit 对比全面，但 C++ 实验仅限 IPC2018
- **写作质量**: ⭐⭐⭐⭐⭐ 从 CLT→EVT 的类比讲解堪称教科书级别，理论推导与直觉解释兼备
- **价值**: ⭐⭐⭐⭐ 深化了 MCTS in planning 的理论基础，UCB1-Uniform 有实用价值但需与更多技术组合

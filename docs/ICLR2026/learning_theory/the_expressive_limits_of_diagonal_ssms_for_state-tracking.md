---
title: >-
  [论文解读] The Expressive Limits of Diagonal SSMs for State-Tracking
description: >-
  [ICLR 2026][序列建模][状态空间模型] 建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。
tags:
  - "ICLR 2026"
  - "序列建模"
  - "理论"
  - "状态空间模型"
  - "表达能力"
  - "群状态追踪"
  - "对角SSM"
  - "可解群"
  - "Mamba"
---

# The Expressive Limits of Diagonal SSMs for State-Tracking

**会议**: ICLR 2026  
**arXiv**: [2603.01959](https://arxiv.org/abs/2603.01959)  
**领域**: 序列建模 / 理论  
**关键词**: 状态空间模型, 表达能力, 群状态追踪, 对角SSM, 可解群, Mamba

## 一句话总结

建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。

## 研究背景与动机

**领域现状**：状态空间模型（SSM），如 Mamba、S4D 等，已作为 Transformer 的高效替代品在长序列建模任务上取得了强大的经验性能。它们通过线性递归和对角化状态转移矩阵实现了 $O(\log T)$ 的并行化。然而，SSM 的理论表达能力理解仍然有限，特别是在状态追踪（state-tracking）这类任务上。

**现有痛点**：
1. SSM最初被期望在状态追踪上优于Transformer（因为有显式状态表示），但实验表明SSM同样失败
2. Merrill et al. 证明SSM属于 $\mathsf{TC}^0$ 复杂性类，推测不能追踪非可解群（如 $S_5$），但这依赖于复杂性理论中的未解猜想
3. Sarrof et al. 证明 Mamba（非负对角矩阵）不能解决奇偶问题（最简单的非平凡群 $C_2$），但对角 SSM 的精确能力边界仍然未知
4. 单层 vs 多层的表达能力差异在理论上不清楚——深度是否带来严格益处？

**核心矛盾**：对角化是 SSM 高效并行化的基础，但也限制了状态转移矩阵的表达能力。如何在保持效率的同时理解其表达极限？

**本文方案**：从代数群论的视角出发，精确刻画 $k$ 层 DCD SSM 能追踪哪些群——给出不依赖任何猜想的充要条件。

## 方法详解

### 整体框架

本文不提新模型，而是把"输入依赖复数对角（DCD）SSM 到底能追踪哪些群"这个问题彻底算清楚。研究对象的状态递归为 $h_t = A(x_t) h_{t-1} + b(x_t)$，其中 $A(x_t) \in \mathbb{C}^{d \times d}$ 是对角矩阵、$b(x_t) \in \mathbb{C}^d$ 是输入向量，输出经解码器 $y_t = \text{dec}(h_t, x_t)$ 给出，并假设 $A$、$b$、$\text{dec}$ 都是通用函数逼近器；任务是群状态追踪——给定群 $G$ 与序列 $x_1,\ldots,x_T \in G$，逐步输出累积乘积 $y_t = x_1 x_2 \cdots x_t$。全文围绕两个层次展开：先卡死单层的能力上界，再用子正规链把层数和可追踪群一一对应，最后把结论对照到 Mamba 等现有变体上。

### 关键设计

**1. 单层不可能性：对角的交换性把单层锁死在阿贝尔群**

第一个核心结论（Theorem 1）是，单层 DCD SSM 在有限精度下能追踪 $G$ 当且仅当 $G$ 是阿贝尔群——非阿贝尔群单层根本无望。证明的关键在于对角矩阵彼此可交换，而群乘积的非交换性无法被可交换的状态更新还原，三个引理依次把"逃逸通道"堵死。Lemma 1 先消除无用坐标：若某坐标在某输入下收缩（$|\lambda(x)_j| < 1$）、扩张（$|\lambda(x)_j| > 1$）或漂移（$|\lambda(x)_j| = 1$ 且 $b(x)_j \neq 0$），都可以构造等价 SSM 把它钉成定值而不损失追踪力，于是只剩纯旋转坐标值得分析。Lemma 2 指出旋转中心若不一致会出事：中性旋转 $h \mapsto \lambda(h - c_1) + c_1$ 与共轭旋转 $h \mapsto \lambda^*(h - c_2) + c_2$ 在 $c_1 \neq c_2$ 时复合出非零平移，状态随之发散、有限精度下追踪失败。Lemma 3 因此要求所有坐标共享同一旋转中心，再叠加对角阵的交换性，状态更新必然可交换，群只能是阿贝尔群。

**2. 多层充要条件：层数 = 子正规链长度**

第二个核心结论（Theorem 2）把深度量化了：$k$ 层 DCD SSM 能追踪 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_k = G$$

且每个商因子 $G_{i+1}/G_i$ 都是阿贝尔群。这正是可解群的定义，但被精细化成"需要**恰好** $k$ 层"——非可解群（如 $A_5$）任意深度都追不到，可解群则有明确的最小层数。充分性是构造性的：以 $S_3 = C_3 \rtimes C_2$ 为例，第一层先追踪 $C_2$ 奇偶位（$\Lambda((0,\beta)) = 0$、$\Lambda((1,\beta)) = \pi$），第二层把第一层的奇偶状态当条件，让 $C_3$ 的模 3 旋转方向随奇偶翻转（$\Lambda^{(2)}((1,\alpha,\beta)) = \tfrac{2\pi}{3}\beta$、$\Lambda^{(2)}((-1,\alpha,\beta)) = -\tfrac{2\pi}{3}\beta$），两层叠起来正好复原非交换乘积。每一层处理一个阿贝尔商，链有多长就需要多少层，深度的收益由此有了严格的群论刻度。

**3. 对照现有变体：负/复特征值是解锁追踪力的开关**

把上述刻画套回主流 SSM，就能一眼看出谁卡在哪。差别全在转移矩阵的特征值能落到哪：Mamba 的 $A(x) = \exp(\Delta(x) \odot \Lambda)$ 取值在 $(0,1]$，连负数都没有，最简单的 $C_2$（奇偶）都解不了；Negative Mamba 用 $2\exp(\Delta(x) \odot \Lambda) - I$ 把范围拓到 $(-1,1]$，单层只够 $C_2$；而 AUSSM 用 $\exp(i\Delta(x) \odot \Lambda(x))$ 让特征值跑遍单位圆 $|z|=1$，单层即可覆盖所有阿贝尔群。这条对照直接给出设计指引：限制特征值非负是 Mamba 表达力的根本瓶颈，放开到复单位圆才谈得上追踪非平凡群。

| 模型 | 转移矩阵 $A(x)$ | 特征值范围 | 能追踪的群 |
|------|-----------------|-----------|-----------|
| S4/S4D | $\Lambda$（固定对角） | $\mathbb{C}$ | 无（时间不变） |
| Mamba | $\exp(\Delta(x) \odot \Lambda)$ | $(0, 1]$（非负实数） | 无群（不含负值） |
| Negative Mamba | $2\exp(\Delta(x) \odot \Lambda) - I$ | $(-1, 1]$ | 仅 $C_2$（单层） |
| AUSSM | $\exp(i\Delta(x) \odot \Lambda(x))$ | 单位圆 $|z|=1$ | 所有阿贝尔群（单层） |

## 实验关键数据

### 主实验：状态追踪任务（最长泛化序列长度，训练长度≤60）

| 群       | 阿贝尔 | 可解 | Mamba | Neg Mamba | Simple AUSSM | AUSSM | RNN  |
|----------|--------|------|-------|-----------|-------------|-------|------|
| $C_2$    | ✓      | ✓    | ✘     | 1000      | 160         | 1000  | 1000 |
| $C_6$    | ✓      | ✓    | ✘     | ✘         | 240         | 940   | 1000 |
| $C_{24}$ | ✓      | ✓    | ✘     | ✘         | 240         | 260   | 1000 |
| $C_{60}$ | ✓      | ✓    | ✘     | ✘         | 300         | 240   | ✘    |
| $S_3$    | ✗      | ✓    | ✘     | ✘         | ✘           | ✘     | 1000 |
| $A_4$    | ✗      | ✓    | ✘     | ✘         | ✘           | ✘     | 1000 |
| $A_5$    | ✗      | ✗    | ✘     | ✘         | ✘           | ✘     | ✘    |

> ✘ 表示模型无法泛化到长度≥100的序列。报告的是准确率>90%的最长序列长度。

### 消融实验：单层 vs 双层

| 群              | 模型           | 单层  | 双层  | 理论预期 |
|-----------------|---------------|-------|-------|----------|
| $C_2$           | AUSSM         | 1000  | 200   | 均可     |
| $C_2 \times C_4$| Neg Mamba     | ✘     | 360   | 双层可   |
| $S_3$           | AUSSM         | ✘     | ✘     | 双层可（gap!） |
| $A_4$           | AUSSM         | ✘     | ✘     | 双层可（gap!） |

### 核心发现

- **阿贝尔群**：AUSSM 和 Simple AUSSM 的单层理论掌控所有阿贝尔群，实验中梯度优化成功找到泛化解
- **非阿贝尔群**：双层 AUSSM 理论上可追踪 $S_3$ 和 $A_4$，但实验中标准训练**从未成功**——表达能力 ≠ 可学习性
- **初始化敏感性**：将 AUSSM 初始化到 $S_3$ 解析解附近后，训练成功并泛化到4倍长度——解存在但 loss landscape 中难以找到
- **RNN 无理论限制**：RNN 可追踪所有可解群，但在大群（$C_{60}$）上也遇到优化困难

## 亮点与洞察

- **精确的代数刻画**：给出充要条件而非"大概能/不能"——$k$ 层严格对应子正规链长度为 $k$ 的群
- **深度的严格益处**：$k$ 层严格 > $(k-1)$ 层——非阿贝尔群需要深度，且需要的层数有精确的群论含义
- **表达 ≠ 学习的重要区分**：理论可表达但 SGD 学不到——提醒社区不能仅看理论表达力，优化瓶颈同样关键
- **对 Mamba 设计的直接指导**：Mamba 的非负特征值约束是根本限制；允许负值/复值特征值是解锁更强追踪能力的必要条件

## 局限与展望

- 结果限于对角 SSM，分块对角（如 $2 \times 2$）可能将表达力扩展到 $\mathsf{NC}^1$
- 可学习性 gap 的根本原因未解——loss landscape 的几何结构（平坦区域/鞍点/孤立极小值）需要进一步研究
- 目前仅讨论群状态追踪，实际序列任务更复杂——理论结论对实际任务的预测力有待验证
- 未探索混合架构（如 AUSSM+Mamba 交替层）在非阿贝尔群上的可学习性
- 扩展到半群和幺半群的讨论仅为初步，完整理论有待建立

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Expressiveness of State Space Models via Temporal Logics](on_the_expressiveness_of_state_space_models_via_temporal_logics.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] Characterizing Pattern Matching and Its Limits on Compositional Task Structures](characterizing_pattern_matching_and_its_limits_on_compositional_task_structures.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Beyond Self-Repellent Kernels: History-Driven Target Towards Efficient Nonlinear MCMC on General Graphs
description: >-
  [ICML 2025 (Oral)][优化/理论][MCMC] 提出 History-Driven Target (HDT) 框架，通过修改目标分布（而非转移核）将自排斥机制嵌入任意 MCMC 采样器，在保持 O(1/α) 方差缩减的同时解决了 SRRW 的计算开销大、仅限可逆链、内存占用高三大问题。
tags:
  - "ICML 2025 (Oral)"
  - "优化/理论"
  - "MCMC"
  - "图采样"
  - "非线性马尔可夫链"
  - "随机逼近"
  - "自排斥随机游走"
---

# Beyond Self-Repellent Kernels: History-Driven Target Towards Efficient Nonlinear MCMC on General Graphs

**会议**: ICML 2025 (Oral)  
**arXiv**: [2505.18300](https://arxiv.org/abs/2505.18300)  
**代码**: 无  
**领域**: 优化  
**关键词**: MCMC, 图采样, 非线性马尔可夫链, 随机逼近, 自排斥随机游走

## 一句话总结

提出 History-Driven Target (HDT) 框架，通过修改目标分布（而非转移核）将自排斥机制嵌入任意 MCMC 采样器，在保持 O(1/α) 方差缩减的同时解决了 SRRW 的计算开销大、仅限可逆链、内存占用高三大问题。

## 研究背景与动机

图上的随机游走是物理学、统计学、机器学习、社会科学等领域的基础工具，典型应用包括社交网络采样、网页爬取、机器人探索和移动网络。Metropolis-Hastings (MH) 随机游走是最经典的图采样方法，但存在混合速度慢和局部陷阱的问题。

近年来，Self-Repellent Random Walk (SRRW) 作为一种突破性方法被提出：它构造非线性马尔可夫链，根据历史访问频率动态调整转移概率，使采样器倾向于访问欠采样的状态，从而实现近零方差。SRRW 的核心公式为：

$$K_{ij}[\mathbf{x}] = Z_i^{-1} P_{ij} (x_j / \mu_j)^{-\alpha}$$

其中 $\alpha \geq 0$ 控制自排斥强度，$Z_i$ 是归一化常数。

然而 SRRW 存在三个关键局限：

**计算开销大**：每一步需预计算当前节点所有邻居的转移概率来求归一化常数 $Z_i$，包括自转移概率 $P_{ii}$（需遍历所有邻居的接受率），这完全破坏了 MH 算法"按需计算"的轻量本质。邻域越大（如稠密图），开销越严重。

**仅支持可逆链**：SRRW 的理论保证严格依赖时间可逆马尔可夫链，无法与加速性能更好的非可逆 MCMC 方法（如 lifted chain、2-cycle MCMC、MHDA）结合。

**内存约束**：经验测度 $\mathbf{x}$ 的维度等于状态空间大小，在大图或指数级状态空间中内存不可行。

## 方法详解

### 整体框架

HDT 的核心思想是将自排斥机制从转移核移至目标分布。不修改 MCMC 的转移核 $\mathbf{P}$，而是用一个 history-driven target $\boldsymbol{\pi}[\mathbf{x}]$ 替换原始目标 $\boldsymbol{\mu}$，然后将 $\boldsymbol{\pi}[\mathbf{x}]$ 作为任意 MCMC 采样器的目标分布。这样，任何 MCMC 方法（可逆或非可逆）都可以作为"基础采样器"直接插入框架，只需用 $\boldsymbol{\pi}[\mathbf{x}]$ 替换 $\boldsymbol{\mu}$。

### 关键设计

#### HDT 分布的设计原则

作者提出四个条件来约束 $\boldsymbol{\pi}[\mathbf{x}]$ 的形式：

- **C1 (尺度不变性)**：$\pi_i[\mathbf{x}, \boldsymbol{\mu}] = \pi_i[\tilde{\mathbf{x}}, \tilde{\boldsymbol{\mu}}]$，意味着只需非归一化量即可工作
- **C2 (局部依赖)**：非归一化项 $\tilde{\pi}_i$ 仅依赖 $\mu_i$ 和 $x_i$，无需邻居信息
- **C3 (不动点)**：$\boldsymbol{\pi}[\boldsymbol{\mu}] = \boldsymbol{\mu}$，收敛后目标分布不变
- **C4 (历史依赖)**：对欠采样状态赋予更高概率，对过采样状态赋予更低概率

**引理 3.1** 证明满足 C1-C4 的分布必须且仅有如下形式：

$$\pi_i[\mathbf{x}] \propto \mu_i (x_i / \mu_i)^{-\alpha}, \quad \alpha > 0$$

这个形式非常简洁优美。当 $\alpha = 0$ 时退化为原始目标 $\boldsymbol{\mu}$。关键优势在于 C2：非归一化项 $\tilde{\pi}_i = \tilde{\mu}_i (\tilde{x}_i / \tilde{\mu}_i)^{-\alpha}$ 仅涉及当前状态和候选状态的局部信息，完全消除了对邻居信息的依赖。

#### Algorithm 1: HDT-MCMC 框架

输入图 $\mathcal{G}$、参数 $\alpha$、非归一化目标 $\tilde{\boldsymbol{\mu}}$、迭代次数 $T$、基础采样器。每步执行：

1. 用基础采样器（可逆或非可逆）以 $\boldsymbol{\pi}[\mathbf{x}]$ 为目标采样 $X_{n+1}$
2. 更新访问计数 $\tilde{x}(X_{n+1}) \leftarrow \tilde{x}(X_{n+1}) + 1$

以 MH 为基础采样器为例，接受率变为：

$$A_{ij}[\mathbf{x}] = \min\left\{1, \frac{\tilde{\mu}_j (\tilde{x}_j / \tilde{\mu}_j)^{-\alpha} Q_{ji}}{\tilde{\mu}_i (\tilde{x}_i / \tilde{\mu}_i)^{-\alpha} Q_{ij}}\right\}$$

只需 $\tilde{\mu}_i, \tilde{\mu}_j, \tilde{x}_i, \tilde{x}_j$ 四个局部量，计算代价与标准 MH 完全相同。且该接受率天然嵌入自排斥效果：若状态 $j$ 相对于 $i$ 欠采样（$\tilde{x}_j / \tilde{\mu}_j < \tilde{x}_i / \tilde{\mu}_i$），则 $A_{ij}[\mathbf{x}] \geq A_{ij}$，即更容易被接受。

#### 与 SRRW 的关键对比

SRRW 的稳态分布包含邻居求和项 $\sum_{j \in \bar{\mathcal{N}}(i)} P_{ij}(x_j/\mu_j)^{-\alpha}$，违反 C2，导致高计算开销。HDT 的设计解耦了邻居依赖，这是核心创新。

#### 兼容多种采样器

HDT 框架可无缝集成：可逆方法（MH、MTM、MHDR）和非可逆方法（MHDA、2-cycle Markov chains、非可逆 MH）。

### 损失函数 / 训练策略

#### 理论保证

**定理 3.3 (遍历性与 CLT)**：HDT-MCMC 满足 (a) $\mathbf{x}_n \to \boldsymbol{\mu}$ a.s. 和 (b) $\sqrt{n}(\mathbf{x}_n - \boldsymbol{\mu}) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{V}_{\text{HDT}}(\alpha))$，其中：

$$\mathbf{V}_{\text{HDT}}(\alpha) = \frac{1}{2\alpha + 1} \mathbf{V}_{\text{base}}$$

方差以 $O(1/\alpha)$ 速率缩减。**推论 3.4** 保证已知的采样器性能排序在 HDT 下不变。

**引理 3.6 (Cost-Based 比较)**：$C_{\text{HDT}} \mathbf{V}_{\text{HDT}}(\alpha) \preceq \frac{2}{\mathbb{E}[|\bar{\mathcal{N}}(i)|]} \cdot C_{\text{SRRW}} \mathbf{V}_{\text{SRRW}}(\alpha)$，在稠密图中优势更大。

#### LRU 缓存策略

为解决大图内存问题，只追踪最近访问的状态。对不在缓存中的邻居 $j$，用局部邻居平均值近似：

$$\hat{x}_j = \tilde{\mu}_j \cdot |\bar{\mathcal{N}}(i) \cap \mathcal{C}|^{-1} \sum_{k \in \bar{\mathcal{N}}(i) \cap \mathcal{C}} \tilde{x}_k / \tilde{\mu}_k$$

## 实验关键数据

### 主实验

在 Facebook (4039 节点, 平均度 43.7) 和 p2p-Gnutella04 (10876 节点, 平均度 7.4) 上评估，均匀目标，1000 次独立运行。

| 方法 | Facebook TVD | Facebook NRMSE | p2p TVD | 类型 |
|------|-------------|---------------|---------|------|
| MHRW | 0.520 | 0.079 | 0.545 | 可逆基线 |
| HDT-MHRW | 0.371 | 0.028 | 0.403 | HDT 可逆 |
| MTM | 0.487 | 0.056 | 0.514 | 可逆基线 |
| HDT-MTM | 0.285 | 0.062 | 0.328 | HDT 可逆 |
| MHDA | 0.513 | 0.068 | 0.522 | 非可逆基线 |
| HDT-MHDA | 0.365 | 0.027 | 0.388 | HDT 非可逆 |

### 消融实验

| 配置 | TVD (Facebook) | 说明 |
|------|---------------|------|
| α=0 (MHRW) | 0.520 | 无自排斥效果 |
| α=1 | ~0.45 | 小幅改进 |
| α=5 | 0.371 | 显著改进 |
| α=10 | ~0.36 | 更大 α 持续改善 |
| LRU r=0.1 (10% 内存) | < MHRW | 90% 内存缩减仍优于基线 |
| LRU r=0.01 (100K 图) | ≈ r=0.1 | 极低内存仍有效 |

### 关键发现

1. **计算预算公平比较**：相同预算下 HDT-MHRW 一致优于 SRRW，在 Facebook 图（平均度 43.6）上差距远大于 p2p-Gnutella04（平均度 7.4），验证了引理 3.6。
2. **初始化鲁棒性**：不同初始节点和伪访问计数对 HDT-MCMC 影响极小。
3. **大规模图**：在 ego-Gplus（~100K 节点）上 LRU 缓存 r=0.01 和 r=0.1 性能可比，均优于 MHRW。

## 亮点与洞察

1. **范式转换**：将自排斥从转移核移至目标分布，一个看似简单的视角变换解决了三大核心问题。
2. **唯一性证明**：引理 3.1 通过 Cauchy 函数方程证明满足四条件的分布形式是唯一的，将设计空间约束到单参数族。
3. **通用即插即用**："Bring Your Own MCMC"——对基础采样器完全黑盒。
4. **Lyapunov 分析**：用 $V(\mathbf{x}) = \sum_i \mu_i (x_i/\mu_i)^{-\alpha}$ 结合 LaSalle 不变性原理证明全局稳定性，Cauchy-Schwarz 不等式的使用极为简洁。

## 局限与展望

1. **指数级状态空间未覆盖**：高维问题（如 Ising 模型）上的应用留作未来工作。
2. **LRU 缺理论保证**：缓存方案是启发式的，缺乏收敛性理论分析。
3. **α 的自适应选择**：如何根据图结构自动选择最优 α 未深入讨论。
4. **仅限离散空间**：连续空间的推广未涉及。

## 相关工作与启发

- **SRRW (Doshi et al., 2023, ICML)**：直接修改转移核实现自排斥，本文的直接前驱
- **MTM with locally balanced weights (Chang et al., 2022, NeurIPS)**：多候选 MH 变体
- **MHDA (Lee et al., 2012)**：非可逆非回溯 MH
- **随机逼近理论 (Delyon 2000, Fort 2015)**：收敛性分析框架

核心启发：当自适应机制与算法某组件深度耦合导致局限时，可考虑将该机制迁移到另一组件，可能以更低代价获得等效甚至更好的效果。

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ | 范式转换，从核修改到目标修改 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 唯一性、收敛性、CLT、cost-based CLT 完备 |
| 实验充分性 | ⭐⭐⭐⭐ | 多图多采样器多指标，缺少高维实验 |
| 实用性 | ⭐⭐⭐⭐ | 即插即用，大规模场景待验证 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 动机清晰、对比到位、图示直观 |

**总评**: 4.5/5 — ICML Oral 当之无愧。以极简设计解决 SRRW 核心痛点，理论完备且实验验证充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](../../NeurIPS2025/optimization/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](../../ICLR2026/optimization/test-time_meta-adaptation_with_self-synthesis.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)
- [\[NeurIPS 2025\] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](../../NeurIPS2025/optimization/escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)

</div>

<!-- RELATED:END -->

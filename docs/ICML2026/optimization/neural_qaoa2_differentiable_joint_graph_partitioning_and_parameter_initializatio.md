---
title: >-
  [论文解读] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization
description: >-
  [ICML 2026][优化/理论][QAOA] 用一个生成-评估神经网络（GEN）一次性地把 QAOA² 的"图分割 + 量子电路参数初始化"两件事联合可微化：评估器学一个高保真的 quantum performance surrogate，生成器在它的梯度引导下吐出离散分区 + 参数初值，配合直通估计器 + 正交补头让端到端可训练；在 183 个 QUBO/Ising/MaxCut 实例（21-1000 变量）上超越启发式 baseline，101 个实例排第一。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "QAOA"
  - "divide-and-conquer"
  - "可微图分割"
  - "参数热启动"
  - "零样本泛化"
---

# Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization

**会议**: ICML 2026  
**arXiv**: [2605.13072](https://arxiv.org/abs/2605.13072)  
**代码**: https://github.com/0SliverBullet/Neural-QAOA-Squared (有)  
**领域**: 量子优化 / 可微编程 / 图分割  
**关键词**: QAOA、divide-and-conquer、可微图分割、参数热启动、零样本泛化

## 一句话总结
用一个生成-评估神经网络（GEN）一次性地把 QAOA² 的"图分割 + 量子电路参数初始化"两件事联合可微化：评估器学一个高保真的 quantum performance surrogate，生成器在它的梯度引导下吐出离散分区 + 参数初值，配合直通估计器 + 正交补头让端到端可训练；在 183 个 QUBO/Ising/MaxCut 实例（21-1000 变量）上超越启发式 baseline，101 个实例排第一。

## 研究背景与动机

**领域现状**：QAOA 是 NISQ 时代解 QUBO/MaxCut 的旗舰算法，但真实问题动辄上千变量，而量子硬件只有百量级 qubits。divide-and-conquer 范式（代表是 QAOA²）通过把大图切成可装进硬件的子图、分别用 QAOA 求解、再把局部解用 ℤ₂ 对称性合并，把可扩展性问题处理掉。

**现有痛点**：现有 D&C 框架有两个解耦缺陷。第一，分图启发式（modularity、boundary、KL）是为"图论指标好看"设计的，和最终量子求解质量没有直接关系——作者在 g05_100.1 上跑出 modularity 与 performance ratio 的 Pearson 相关只有 0.2859，几乎随机。第二，子图上的 QAOA 参数 $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ 用随机初始化，完全不看子图拓扑，导致 cold-start——即使把优化步数翻倍 ($T=40$) 也追不上一个 topology-aware 的 warm-start ($T=20$)。

**核心矛盾**：两件事——分区和参数初始化——其实都是"从图拓扑映射到量子性能"的子任务，但被分别用启发式或随机处理，没人让二者协同。要让它们端到端可学，必须解决"离散分区怎么传梯度"和"分区受 qubit 容量硬约束"两个工程难题。

**本文目标**：构造一个能同时输出分区 + 初值的可微生成器，并且让它的训练信号来自"最终量子性能"，而非中间代理指标。

**切入角度**：把 QAOA² 性能预测建模为一个 differentiable surrogate（quantum evaluator），让生成器在它的梯度上做梯度上升；用直通估计器（STE）+ 贪心容量离散化（GCD）把硬约束的离散分区"夹"进可微链路；最后用正交补头（OCH）给 cluster center 一个几何归纳偏置防止 GNN over-smoothing。

**核心 idea**：用一个 evaluator + generator 的双网络结构，把"分什么图、给什么初值"做成一个可微的联合策略，由 evaluator 提供 quantum-aware 梯度，实现真正"为量子求解结果而优化"的 D&C。

## 方法详解

### 整体框架
GEN（Generative Evaluative Network）由两部分组成。其一是 **Quantum evaluator** $f_\phi(G, \mathbf{S}, \mathbf{P}) \to \hat{\rho}$，多视图 GNN，把图 $G$、分区 $\mathbf{S}$、参数 $\mathbf{P}$ 编码到统一隐空间，预测 performance ratio $\rho \in [0.5, 1]$（公式 $\rho = (\text{Cut} - \text{Neg}) / (\text{OPT} - \text{Neg})$ 保证有界）。先用 supervised MSE 在标注数据集 $\mathcal{D}_{\text{offline}} = \{(G_i, \mathbf{S}_i, \mathbf{P}_i, \rho_i)\}$ 上训到收敛。其二是 **Joint generator** $g_\theta(G) \to (\mathbf{S}, \mathbf{P})$，按 $P(\mathbf{S}, \mathbf{P} | G) = P(\mathbf{S} | G) P(\mathbf{P} | \mathbf{S}, G)$ 先分区后参数。冻结 $f_\phi$，用 $\max_\theta \mathbb{E}_G [f_\phi(G, g_\theta(G))]$ 做无监督梯度上升。

推理时先一次前向 $(\mathbf{S}_0, \mathbf{P}_0) = g_\theta(G_{\text{new}})$ 拿初值，再做 test-time adaptation——在该单个实例上 fine-tune 生成器参数 $\theta$ 几步梯度上升，得到 $\theta^*$，输出 $(\mathbf{S}^*, \mathbf{P}^*) = g_{\theta^*}(G_{\text{new}})$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    G["输入图 G"] --> GEN
    subgraph GEN["联合生成器 gθ（先分区后参数）"]
        direction TB
        OCH["正交补头 OCH<br/>拓扑编码 + cluster center 正交约束<br/>→ soft partition S̃"]
        OCH --> GCD["贪心容量离散化 GCD + STE<br/>前向离散满足 qubit 容量 / 反向直通回传梯度<br/>→ 离散分区 S"]
        GCD --> PG["参数生成器<br/>sg(A_sub) + arctan → 参数初值 P"]
    end
    GEN -->|输出 (S, P)| EVAL["多视图量子评估器 fφ<br/>topology / partition / param 三路 GNN<br/>→ performance ratio ρ̂"]
    EVAL -->|梯度上升引导（fφ 冻结）| GEN
    EVAL --> OUT["推理：前向取初值 + test-time adaptation 微调 θ"]
```

### 关键设计

**1. 多视图量子评估器 $f_\phi$：学一个可微 proxy，把"跑一遍量子模拟"换成"过一次 GNN"**

整个 D&C 过去的死穴是分区/参数选择只能靠启发式或随机，因为真正的反馈信号——量子求解性能——评估代价太高、又不可微。GEN 先解决这点：训一个高保真 surrogate $f_\phi$ 来预测 performance ratio，让"算梯度"的代价从 O(量子模拟) 降到 O(GNN 前传)。它用三个并行 encoder 处理异质输入——topology encoder 吃全图邻接 $\mathbf{A}$；partition encoder 吃屏蔽了跨分区边的子图邻接 $\mathbf{A}_{\text{sub}}=\mathbf{A}\odot(\mathbf{S}\mathbf{S}^T)$；param encoder 把参数通过 $\mathbf{X}_{\text{param}}=\mathbf{S}\mathbf{P}^T$ 广播到节点级，再用 $\tilde{\mathbf{X}}_{\text{param}}=[\sin(\mathbf{X}_{\text{param}}),\cos(\mathbf{X}_{\text{param}})]$ 嵌入以尊重 $2\pi$ 周期性。三路 global mean pool 后拼接过 MLP，输出 $\hat{\rho}=0.5(\text{sigmoid}(\text{MLP}(\mathbf{H}))+1)$ 强制落在理论区间 $[0.5,1]$。多视图设计的用意是让每种输入信号都有专属 encoder，不被混合稀释，从而 surrogate 足够保真、梯度才可信。

**2. 正交补头 OCH：给 cluster center 一个不动的几何锚点，挡住 GNN over-smoothing**

生成器要把节点嵌入投到 $k$ 个 cluster center 上得到 soft partition $\tilde{\mathbf{S}}\in[0,1]^{N\times k}$，但标准 GNN + softmax 有个老毛病：节点嵌入趋同（over-smoothing），partition 概率退化成近乎均匀，训练梯度被稀释、分区几乎随机。OCH 的对策是给 cluster center 矩阵 $\mathbf{C}\in\mathbb{R}^{k\times h}$ 强加两个正交约束 $\mathbf{C}\boldsymbol{g}=\mathbf{0}$ 且 $\mathbf{C}\mathbf{C}^T=\mathbf{I}$，其中 $\boldsymbol{g}=\text{GMP}(\mathbf{H}_{\text{topology}})$ 是全图嵌入，$\mathbf{C}$ 通过对随机矩阵相对 $\boldsymbol{g}$ 做 QR 分解动态生成，最后 $\tilde{\mathbf{S}}=\text{softmax}(\mathbf{H}_{\text{topology}}\mathbf{C}^T)$。把 center 钉在全图嵌入的正交补里，等价于"用全图上下文做减法"，强制 inter-cluster separability 最大化——这比把 center 当可学参数（容易和 encoder 一起 collapse）稳得多，消融里去掉这个约束后 partition 就退化成随机。

**3. 贪心容量离散化 GCD + 直通估计器 STE：硬约束严格可行，梯度还能回传**

qubit 容量是硬件物理上限，$\sum_i\mathbf{S}_{ij}\le\text{max\_nodes}$ 一步都不能违反，所以 Gumbel-Softmax 那种连续松弛在这里不可用。GCD 的做法是按概率从高到低贪心地把节点塞进 cluster、满了就跳过，保证容量约束 100% 满足。可离散化会断梯度，于是前向用离散 $\mathbf{S}$ 进 evaluator 算精确得分、反向用直通估计器 $\nabla_{\tilde{\mathbf{S}}}f\approx\nabla_{\mathbf{S}}f$ 跨过离散算子；参数生成时再加一个 stop-gradient $\text{sg}(\mathbf{A}_{\text{sub}})$ 防止参数优化反过来扰动分区。本质是牺牲一点梯度精度换严格可行性——在 NISQ + 离散决策这种硬约束场景里，GCD + STE 几乎是唯一可行的可微化路径。

### 损失函数 / 训练策略
两阶段：(1) Evaluator 阶段最小化 MSE $\mathbb{E}_{(G, \mathbf{S}, \mathbf{P}, \rho)} [(f_\phi - \rho)^2]$，数据来自启发式分区 + 均匀采样参数 + QAOA² 模拟得到的真值；(2) Generator 阶段冻结 $f_\phi$，最大化 $\mathbb{E}_G [f_\phi(G, g_\theta(G))]$。生成器只在 $p=1$ 上训，更深电路 ($p=2, 3$) 用 Zhou 2020 的参数扩展策略而非重训。

## 实验关键数据

### 主实验
在 50 个 held-out 测试实例（B/BE/W 三个数据集 20% 留出，问题规模与训练分布一致）上：

| 数据集 | Random | Modularity | Boundary | KL | **Neural QAOA²** |
|--------|--------|------------|----------|------|-----------------|
| B (8 个 QUBO) | 0.8047 (rank 4.75) | 0.8351 (2.38) | 0.8246 (2.63) | 0.8092 (3.75) | **0.8417 (1.50, 5/3 wins)** |
| BE (16 个 QUBO) | 0.8626 (4.81) | 0.8692 (3.13) | 0.8722 (2.31) | 0.8672 (3.69) | **0.8824 (1.06, 15/1 wins)** |
| W (26 个 MaxCut) | 0.8962 (3.23) | 0.9137 (2.23) | 0.9114 (2.96) | 0.8934 (4.27) | **0.9153 (2.23, 8/18 wins)** |
| **Overall (50)** | 0.8708 (3.98) | 0.8869 (2.54) | 0.8850 (2.70) | 0.8716 (4.00) | **0.8930 (1.74, 28/22 wins)** |

BE 数据集上 Neural QAOA² 几乎横扫 (15/16)，原因是 QUBO 一般缺乏显式社区结构，modularity 这种 graph-theoretic 启发式直接失效；W 是 MaxCut 自带社区结构，所以 modularity 跟 Neural QAOA² 打平 (都是 rank 2.23)。

### 消融实验
93 个 OOD 实例（GKA + L，分布外，规模相当）：

| 配置 | GKA (45 个 QUBO) | L (48 个 Ising) | Overall (93) |
|------|------------------|-----------------|--------------|
| Random | 0.8478 (4.16) | 0.6984 (4.65) | rank 4.41 |
| Modularity | 0.8659 (2.40) | 0.7391 (3.06) | rank 2.73 |
| Boundary | 0.8601 (2.89) | 0.8205 (1.60) | rank 2.24 |
| KL | 0.8503 (4.04) | 0.7022 (4.27) | rank 4.16 |
| **Neural QAOA² (Ours)** | **0.8762 (1.51, 32/13)** | **0.8160 (1.42, 28/20)** | **rank 1.46, 60/33 wins** |

零样本迁移到分布外拓扑（Ising 训练集里根本没有）也是 SOTA，说明 GEN 学到的不是某种特定数据集特征，而是 partition-quantum-performance 的通用映射。

### 关键发现
- 启发式分区与最终性能的相关性低（Pearson 0.2859）的实证证据，是支撑整个 paper 动机的关键观测——之前没人这么明确地把"指标失配"量化出来。
- 即使 random 初始化 + $T=40$ 步优化，也打不过 topology-aware 初始化 + $T=20$ 步，说明 cold-start 损失远不是"多迭代几步"能补偿的。
- 在 $p=1$ 上训练的模型迁到 $p=2, 3$ 仍打过 TQA/INTERP/FOURIER/QIBPI 等高级初始化 baseline，说明学到的拓扑映射是"参数 schedule 无关"的。
- OCH 的设计非常关键：消融里去掉正交补约束后 GNN 输出退化成几乎均匀概率分布，partition 决策几乎随机。

## 亮点与洞察
- 把"启发式 D&C → 端到端可微 D&C"做成功是个工程精彩活：作者同时解决了 (a) 离散决策的梯度回传、(b) 硬容量约束、(c) GNN over-smoothing 三个独立难题，每个对应一个干净的组件（STE/GCD/OCH），可拼可拆。
- "用 evaluator 做 differentiable surrogate 提供梯度信号"这个思路其实在 neural architecture search 里早有先例，但搬到量子组合优化是新的——它把"昂贵的 oracle 评估"和"可微优化"之间的鸿沟做了系统性桥接。
- OCH 用 QR 分解动态生成 cluster center 这一招很巧妙：传统聚类把 center 当可学参数，容易和 GNN encoder 一起 collapse；把 center 钉在"全图上下文的正交补"里相当于给它一个不动的几何锚点。
- Test-time adaptation 的设计承认了"训练分布 ≠ 推理分布"这一现实，让模型从 distribution prior 过渡到 instance-specific 配置，对工业部署很友好。

## 局限与展望
- $\rho$ 的上界 1.0 是相对 best-known cut 的，而 best-known cut 本身可能不是真正最优——大问题上是经典启发式给的，存在 ground-truth bias。
- 训练集和评估都依赖 QAOA² 模拟，没在真实硬件（噪声、读出错误、连通性约束）上验证。
- 生成器只在 $p=1$ 训，扩展到深层电路靠经验性 schedule，没给理论保证。
- max_nodes=10 这个硬上限对应不到目前最大 QPU（百量级），结果在更高 qubit 数下的扩展性未验证。
- 训练集是 80% B/BE/W，OOD 评估虽然在 GKA/L 上做了，但仍属于 benchmark library 同一来源，真正"野生"工业图未测。

## 相关工作与启发
- **vs DC-QAOA / 原 QAOA²**: 都属于 D&C 范式，但都用启发式分区 + 随机初值；Neural QAOA² 把这两个手工组件替换成端到端可学网络。
- **vs INTERP / FOURIER / TQA / QIBPI 参数初始化**: 这些方法只解决"参数怎么初始化"，分区还是启发式给的；本文把两件事打通联合优化，从消融看联合训练的增益远大于单独优化参数。
- **vs Sampled MuZero / 神经组合优化中的 GNN policy**: 都是用 GNN + RL/可微优化做组合决策；本文的特殊性是 evaluator-generator 双网络 + 量子 surrogate，反映了 quantum performance 评估比一般 reward signal 更昂贵这一约束。
- 启发：evaluator-generator 这套架构能不能搬到其他"昂贵 oracle + 离散决策"问题——比如芯片布局、编译器调优、超参搜索？只要能学出一个高保真的可微 proxy，原理就通用。

## 评分
- 新颖性: ⭐⭐⭐⭐ D&C 的可微化是组合创新而非颠覆式新机制，但 OCH + GCD + STE 的具体组合属于扎实贡献
- 实验充分度: ⭐⭐⭐⭐ 183 实例 + 50 IID + 93 OOD + 多种启发式 baseline + 不同 $p$ 深度，比较系统
- 写作质量: ⭐⭐⭐⭐⭐ 动机图（Figure 1）一目了然，pipeline 描述清晰，关键设计都有专门 section
- 价值: ⭐⭐⭐⭐ 对 NISQ 时代的量子组合优化部署有直接价值，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICML 2026\] A Fully First-Order Layer for Differentiable Optimization](a_fully_first-order_layer_for_differentiable_optimization.md)
- [\[ICLR 2026\] Gradient-Based Diversity Optimization with Differentiable Top-$k$ Objective](../../ICLR2026/optimization/gradient-based_diversity_optimization_with_differentiable_top-k_objective.md)
- [\[ICLR 2026\] Conditioned Initialization for Attention](../../ICLR2026/optimization/conditioned_initialization_for_attention.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](../../ICML2025/optimization/quantum_optimization_via_gradient-based_hamiltonian_descent.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers
description: >-
  [ICLR2026][优化/理论][旅行商问题] 这是一篇"打假"性质的评估论文：作者系统拆解"Heatmap + MCTS"求解 TSP 这条主流范式，用大量实验证明大家拼命卷的"热力图复杂度"其实没那么关键——被长期忽视的 MCTS 搜索超参才是性能主导因素，一个零学习、零参数的 k-近邻先验热力图（GT-Prior）配上调好的 MCTS 就能追平甚至超过 DIFUSCO 这类复杂学习模型。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "旅行商问题"
  - "Heatmap+MCTS"
  - "蒙特卡洛树搜索"
  - "超参调优"
  - "公平评估"
---

# Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=H6PLJnnK6e](https://openreview.net/forum?id=H6PLJnnK6e)  
**代码**: https://github.com/LOGO-CUHKSZ/beyond-heatmap-mcts-tsp  
**领域**: 神经组合优化 / TSP 求解 / 评估方法学  
**关键词**: 旅行商问题, Heatmap+MCTS, 蒙特卡洛树搜索, 超参调优, 公平评估

## 一句话总结
这是一篇"打假"性质的评估论文：作者系统拆解"Heatmap + MCTS"求解 TSP 这条主流范式，用大量实验证明大家拼命卷的"热力图复杂度"其实没那么关键——被长期忽视的 MCTS 搜索超参才是性能主导因素，一个零学习、零参数的 k-近邻先验热力图（GT-Prior）配上调好的 MCTS 就能追平甚至超过 DIFUSCO 这类复杂学习模型。

## 研究背景与动机

**领域现状**：旅行商问题（TSP）是组合优化里的经典 NP-hard 问题。近年用机器学习解大规模 TSP 的主流范式是 Fu et al. (2021) 提出的"Heatmap + Monte Carlo Tree Search（MCTS）"：先用一个神经网络给每条边 $(i,j)$ 预测一张热力图 $P^N \in [0,1]^{N\times N}$，其中 $P^N_{ij}$ 表示边 $(i,j)$ 出现在最优环路中的概率；再把这张热力图当作先验，喂给一个基于 k-opt 移动的 MCTS 去搜出高质量解。沿着这条线，社区不断把热力图生成模型做得更花哨：从有监督 GCN（Att-GCN）到元学习 GNN（DIMES）、扩散模型（DIFUSCO）、无监督学习（UTSP）。

**现有痛点**：几乎所有后续工作的注意力都砸在"怎么把热力图学得更准"上，而范式里的另一半——MCTS 搜索组件——却被当成一个固定的、调好的黑盒，大家普遍沿用默认超参、极少调优，也很少报告稀疏化、额外监督这些辅助步骤的真实影响。这就埋了一个评估上的大坑：如果不同方法的 MCTS 配置没对齐，那么"热力图 A 比热力图 B 好"这个结论，到底是热力图本身的功劳，还是 MCTS 碰巧调得更好？

**核心矛盾**：MCTS 配置是一个被系统性忽略的混淆变量（confounder）。一个本来很强的热力图，配上没调好的 MCTS 可能表现稀烂；一个很弱的热力图，配上调到极致的 MCTS 反而可能逆袭。在这种情况下，固定 MCTS 设置的横向比较根本不公平，可能严重误导整个研究方向。

**本文目标**：作者不打算再提一个新 SOTA 求解器，而是要回答三个问题——Q1：MCTS 配置到底在多大程度上左右最终解质量？Q2：一个简单、无参数的热力图配上调优的 MCTS，能不能匹敌甚至超过复杂学习热力图？Q3：哪些 MCTS 超参最关键，它们的影响随热力图类型和问题规模怎么变？

**切入角度**：作者的两个核心论断是——(1) MCTS 的策略性校准对解质量影响巨大，必须认真对待；(2) 他们提出的 GT-Prior（一个基于 TSP 内在 k-近邻结构的无参数热力图）能与复杂学习热力图掰手腕，还有更强的泛化性。

**核心 idea**：与其继续卷热力图复杂度，不如老老实实把 MCTS 调公平、再加一个强而简单的 baseline，重新审视"复杂热力图是性能主要来源"这个被默认接受却从没严格验证的假设。

## 方法详解

### 整体框架
这篇论文的"方法"不是一个新模型，而是一套**评估方法学（evaluation methodology）**，目标是把"Heatmap + MCTS"范式里热力图和搜索两个组件的贡献拆开、各自量化。整体逻辑是：先把 MCTS 这个被忽视的变量"标准化、可调、可解释"——为每种热力图独立做一遍 MCTS 超参网格搜索（保证公平），再用 SHAP 分析哪些超参重要（回答 Q1/Q3）；然后引入一个零参数的 GT-Prior 热力图作为"地板基线"，在调优 MCTS 的前提下和一堆复杂学习热力图同台竞技（回答 Q2）。

先交代清楚被评估的 MCTS 本体。MCTS 把 TSP 建模成 MDP：每个状态是一条合法环路，动作是修改当前解的 k-opt 移动。搜索从一条按概率 $\propto e^{P^N_{ij}}$ 采样得到的初始环路出发，边权矩阵初始化为 $W_{ij} = 100 \cdot P^N_{ij}$。每步模拟用一个势函数挑选要改的边：

$$Z_{ij} = \frac{W_{ij}}{\Omega_i} + \alpha \sqrt{\frac{\ln(M+1)}{Q_{ij}+1}}$$

其中 $\Omega_i = \sum_{j\neq i} W_{ij}$ 是节点 $i$ 的边权归一化项，$M$ 是全局移动计数，$Q_{ij}$ 是边的访问频次，$\alpha$ 是探索系数——前一项是利用（边权越高越想选），后一项是 UCB 式探索。若一个 k-opt 移动让环路变短（$\Delta L < 0$）就接受，否则从新采样的初始环路重启。每次移动后按改善幅度更新边权：

$$W_{ij} \leftarrow W_{ij} + \beta\left(\exp\left(\frac{L(\pi)-L(\pi')}{L(\pi)}\right)-1\right)$$

$\beta$ 是学习率。整个评估框架就是围着这套 MCTS 的几个关键旋钮（$\alpha$、$\beta$、最大 k-opt 深度等）展开的。

### 关键设计

**1. MCTS 超参调优流水线：把"搜索"从黑盒变成对齐的公平基准**

针对的痛点是：以前大家固定 MCTS 默认配置做横向比较，导致热力图的"真实价值"被搜索配置污染。作者的做法是强制为**每种热力图、每个问题规模（TSP-500/1000/10000）独立**做一遍 MCTS 超参网格搜索：在一个专门的合成调优集上（TSP-500/1000 各 128 个实例、TSP-10000 用 16 个）评估各种配置，选平均 optimality gap 最低的那一组用于测试。被调的关键超参有六个：探索系数 Alpha（$\alpha$）、边权更新激进度 Beta（$\beta$）、k-opt 最大深度 Max\_Depth、每节点候选边集大小 Max\_Candidate\_Num、每步移动的模拟次数 Param\_H，以及一个布尔开关 Use\_Heatmap（初始候选集构造到底用不用热力图）。这一步之所以有效，是因为它把"搜索强弱"这个混淆变量对齐成了"对每种热力图都尽力调到最好"，从而让后续比较真正反映热力图的内在质量；而且作者强调这一次性网格搜索是离线预计算，成本和训练那些学习热力图相当，不影响推理时间。

**2. 基于 SHAP 的超参重要性归因：量化每个旋钮到底有多大用**

光调优还不够，作者想知道"为什么"以及"哪个旋钮最关键"，于是用博弈论里的 SHAP（SHapley Additive exPlanations）把解长度的变化归因到每个 MCTS 超参上——正 SHAP 值代表解变长（更差），负值代表变短（更好）。这个分析是模型无关的，能直接量化每个超参的边际贡献和非线性交互。结论很具体：Max\_Candidate\_Num 一致地有强（常为正）的影响，说明把候选集从超大的默认值缩小，既能提速又能提质；Max\_Depth 普遍呈正 SHAP，意味着 k-opt 探得太深反而拖累快速找好解；Alpha 和 Use\_Heatmap 是混合效应，其最优值依赖具体热力图（强非线性交互）；Beta 在 SoftDist 上有明显正影响，暗示其默认更新策略不优；而 Param\_H 在测试范围内影响普遍很小。这一设计直接回答了 Q3，并把"该优先调哪些旋钮"这件事从经验玄学变成了可量化的结论。

**3. GT-Prior：一个零学习、零参数的 k-近邻先验热力图，专门用来当"照妖镜"**

针对的痛点是：缺一个足够强的简单 baseline，导致复杂热力图的"附加值"无从衡量。作者的观察是——最优 TSP 环路的边压倒性地连向城市的最近邻：实测中选到前 5 近邻的概率超过 94%，前 10 近邻超过 99%，而且这个分布在不同规模、不同分布的实例上高度一致。这个"k-近邻先验"以前只被隐式地用来构造稀疏图输入，从没被直接当作主热力图。GT-Prior 就把它显式化：先在 (近) 最优解上统计第 $k$ 近邻被选中的经验分布

$$\hat{P}_N(k) = \frac{1}{|\mathcal{I}|}\sum_{I\in\mathcal{I}} P^I_N(k), \quad P^I_N(k) = \frac{n^I_k}{N}$$

其中 $n^I_k$ 是一个实例最优解里"连到第 $k$ 近邻"出现的次数。然后直接令 $P^N_{ij} = \hat{P}_N(k_{ij})$，$k_{ij}$ 是城市 $j$ 在 $i$ 的邻居里的接近度排名。这张热力图**无参数、与规模无关、不需任何训练或推理生成时间**。它有效的原因是：它直接编码了 TSP 解的本质结构性先验（局部性），既然这个先验本身就极强，再叠一层昂贵的神经网络带来的边际收益自然有限——GT-Prior 正是用来量化这层边际收益到底值不值的标尺。

## 实验关键数据

实验覆盖三种规模（TSP-500/1000/10000）、多种合成分布（uniform、cluster、explosion、implosion）和真实 TSPLIB benchmark；ground-truth 用 Concorde（500/1000）或 LKH-3（10000）求得，指标是 optimality gap（相对最优解的相对差距，越低越好）。

### 主实验：GT-Prior 追平甚至反超复杂学习热力图

所有方法都用本文流水线调优后的 MCTS。GT-Prior 在零热力图生成时间下做到了和 DIFUSCO 这类重模型同档，TSP-10000 上甚至最好：

| 热力图 | 类型 | TSP-500 Gap | TSP-1000 Gap | TSP-10000 Gap |
|--------|------|------|------|------|
| Zero（纯调优 MCTS） | MCTS | 0.66% | 1.16% | 3.80% |
| Att-GCN | SL+MCTS | 0.69% | 1.09% | 3.03% |
| DIMES | RL+MCTS | 0.43% | 1.11% | 3.06% |
| UTSP | UL+MCTS | 0.90% | 1.53% | — |
| SoftDist | dist+MCTS | 0.43% | 0.80% | 2.95% |
| DIFUSCO | SL+MCTS | 0.33% | 0.53% | 2.37% |
| Fast-T2T | SL+MCTS | 0.12% | 0.65% | 4.22% |
| **GT-Prior** | prior+MCTS | **0.50%** | **0.85%** | **2.14%** |

GT-Prior 在 TSP-10000 上的 2.14% 优于所有学习方法（DIFUSCO 2.37%、Fast-T2T 4.22%），且热力图生成耗时为 0。即便是完全不给边引导的 Zero 热力图，光靠调优 MCTS（此时 Use\_Heatmap 设为 False、靠距离选候选）也能拿到 TSP-500 上 0.66% 的体面成绩——直接说明搜索组件本身就贡献了大量解质量。

### MCTS 配置的影响幅度

| 现象 | 数据 | 说明 |
|------|------|------|
| 同一热力图、不同 MCTS 配置的极差 | DIMES@TSP-10000：4.86% → 91.31% | 仅靠 MCTS 设置就能从可用变成灾难 |
| 默认 MCTS vs 最优 MCTS | 默认配置常远离最优 | 不调 MCTS 的横向比较站不住脚 |

这组对比直接回答 Q1：MCTS 配置是主导性能因素，调好它对所有热力图都能大幅提质。

### 泛化实验：在 TSP-500 上确定的配置迁移到其他规模

| 方法 | TSP-1000 退化 | TSP-10000 退化 |
|------|------|------|
| DIFUSCO | +0.33% | +2.91% |
| Fast-T2T | +0.75% | -0.06% |
| **GT-Prior** | **+0.03%** | **-0.01%** |

GT-Prior 的退化几乎为零（甚至轻微变好），泛化性显著优于 DIFUSCO 这类学习模型——因为它编码的是与规模无关的结构先验，天生不挑规模。

### 关键发现
- **搜索 > 热力图**：MCTS 调优带来的收益常常匹敌甚至超过升级热力图模型；默认 MCTS 配置往往离最优很远，是过去比较不公平的根源。
- **最该调的旋钮**：Max\_Candidate\_Num（缩小候选集既提速又提质）和 Max\_Depth（别探太深）影响最大；Param\_H（模拟次数）反而几乎没用，颠覆"模拟越多越好"的直觉。
- **复杂度的边际收益存疑**：在搜索已经调好的前提下，零参数 GT-Prior 就能逼近复杂模型，说明很多"SOTA 提升"其实可能来自调得更好的搜索，而非更聪明的热力图。

## 亮点与洞察
- **把"混淆变量"摆上台面**：这篇论文最大的价值是方法学层面的——它明确指出 MCTS 配置是一个被整个社区系统性忽略的混淆变量，并用 SHAP + 网格搜索把它量化清楚。这种"先把评估做干净再下结论"的思路，可以直接迁移到任何"学习模块 + 搜索/解码模块"的范式（如 LLM 推理里的"模型 + 解码策略"、检索增强里的"检索器 + reranker"）。
- **强简单 baseline 的威力**：GT-Prior 用一个统计上极强的结构先验（前 5 近邻 >94%）就把一堆扩散模型、元学习模型逼到墙角，提醒大家在刷点之前先问"我的复杂模型相对一个零参数先验到底强多少"。
- **泛化几乎免费**：GT-Prior 与规模无关，迁移退化近零，这对实际部署很有吸引力——不用为每个规模重训一个热力图模型。
- **可复现的调优流水线**：作者把标准化 MCTS 调优流水线开源，等于给后续研究提供了一把"公平比较"的尺子，本身就是一个可复用的工程贡献。

## 局限与展望
- **结论限定在 Heatmap + MCTS 这条线**：所有论断都基于这一特定范式和这套特定 MCTS 实现，不代表所有神经组合优化方法里"学习组件不重要"——换一个搜索后端（如 LKH、其他局部搜索）结论未必成立。
- **GT-Prior 依赖统计先验的稳定性**：k-近邻先验在 uniform 及几种合成分布上很稳，但作者自己也承认要靠 (近) 最优解统计得到分布；在结构极端、最近邻假设失效的分布上是否仍然成立，是开放问题。
- **网格搜索的成本**：虽然是离线一次性的，但为每种热力图、每个规模独立网格搜索仍有不小开销；作者提到可用 SMAC3 等更高效的超参优化替代，但论文主体仍是朴素网格搜索。
- **没有给出"新更优解法"**：这是篇评估/打假论文而非求解器论文，落地时它给的是"怎么评、用什么 baseline"的方法论，而非一个可以直接刷榜的新 SOTA。

## 相关工作与启发
- **vs Att-GCN / DIMES / DIFUSCO / UTSP（卷热力图复杂度路线）**：这些工作不断升级热力图生成模型（监督 GCN → 元学习 → 扩散 → 无监督），默认"热力图越复杂越好"。本文反其道而行，固定并对齐 MCTS 后证明这些复杂模型相对零参数 GT-Prior 的边际优势很有限，TSP-10000 上甚至被反超。
- **vs SoftDist（Xia et al., 2024）**：SoftDist 已经表现出对模型复杂度的怀疑，用一个基于距离的简化热力图。本文与之同向但更彻底——GT-Prior 完全无参数、无学习，并配上系统的 MCTS 调优与 SHAP 分析，把"简单也能行"从一个观察升级为一套可复现的评估方法学。
- **vs 原始 Fu et al. (2021) MCTS 框架**：本文复用其 MCTS 本体，但把它从"固定黑盒"重新定位为"必须为每种热力图单独调优的关键变量"，相当于回头把这条线最被忽视的一半补完整。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新模型而是新视角，但"打假主流假设 + 零参数 baseline 反超"足够有冲击力
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 3 种规模、4 种分布、TSPLIB、7+ 热力图、SHAP 归因、泛化分析，非常扎实
- 写作质量: ⭐⭐⭐⭐⭐ 问题驱动（Q1/Q2/Q3）、论证清晰、结论有节制
- 价值: ⭐⭐⭐⭐⭐ 给整条 Heatmap+MCTS 研究线提供了公平评估的标尺，方法论可迁移到其他"学习+搜索"范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](../../ICML2026/optimization/lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICLR 2026\] Beyond Short Steps in Frank-Wolfe Algorithms](beyond_short_steps_in_frank-wolfe_algorithms.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)

</div>

<!-- RELATED:END -->

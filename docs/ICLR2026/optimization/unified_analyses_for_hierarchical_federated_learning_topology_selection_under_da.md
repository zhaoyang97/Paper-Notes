---
title: >-
  [论文解读] Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity
description: >-
  [ICLR 2026][优化/理论][层次联邦学习] 本文为层次联邦学习（HFL）的四种两层拓扑（Star-Star / Star-Ring / Ring-Star / Ring-Ring）建立了**首个统一的非凸收敛框架**，用同一套假设和"有效学习率"把它们的收敛界放进同一张表里直接比较，进而推导出三条可落地的拓扑选择原则，并在 CIFAR-10/CINIC-10/Fashion-MNIST/SST-2 上验证。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "层次联邦学习"
  - "收敛性分析"
  - "拓扑选择"
  - "数据异质性"
  - "环形聚合"
---

# Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ojdYVjLX7S](https://openreview.net/forum?id=ojdYVjLX7S)  
**领域**: 联邦学习 / 分布式优化理论  
**关键词**: 层次联邦学习, 收敛性分析, 拓扑选择, 数据异质性, 环形聚合

## 一句话总结
本文为层次联邦学习（HFL）的四种两层拓扑（Star-Star / Star-Ring / Ring-Star / Ring-Ring）建立了**首个统一的非凸收敛框架**，用同一套假设和"有效学习率"把它们的收敛界放进同一张表里直接比较，进而推导出三条可落地的拓扑选择原则，并在 CIFAR-10/CINIC-10/Fashion-MNIST/SST-2 上验证。

## 研究背景与动机

**领域现状**：联邦学习（FL）让设备在不交换原始数据的前提下协同训练模型，但单层 FL 在大规模部署时会遇到通信瓶颈、同步延迟和单点故障。层次联邦学习（HFL）通过在客户端和全局服务器之间插入一层中间聚合节点（边缘服务器 / 簇头），把协调压力分摊到两层结构上，从而支撑海量设备。

**现有痛点**：HFL 的每一层聚合都可以是**并行更新（星型 Star）**或**串行更新（环型 Ring）**，于是组合出四种拓扑——Star-Star、Star-Ring、Ring-Star、Ring-Ring。但已有理论分析几乎只覆盖最简单的 Star-Star，且各家工作用的假设互不相同（有的还用了"梯度一致有界"这种强假设），导致**根本没法横向比较四种拓扑谁更好**。Ring-Ring 甚至从未被分析过。

**核心矛盾**：实践者真正想知道的是"在给定的数据异质性、分组结构、网络条件下，到底该选哪种拓扑"，但缺一个统一框架就只能拍脑袋。难点在于 HFL 有**级联异质性**（组间分布 δ 与组内分布 ζ 非平凡地交织）、**跨层依赖**（一层的更新直接影响另一层的误差传播，使有效学习率依赖拓扑）、以及星型/环型本身就是两种统计性质截然不同的更新（星型是无偏高方差的并行平均，环型是有偏低方差的串行传播），这些性质会逐层叠加。

**本文目标**：把四种拓扑放进同一套非凸假设下，给出可比较的收敛界，并据此回答"什么场景选什么拓扑"。

**切入角度**：作者注意到全局数据异质性可以用**全方差定律**精确拆成"组间 + 组内"两部分，而四种拓扑的差异本质上是这两部分误差在两层间如何被放大/抑制。只要用一个**拓扑相关的有效学习率 $\tilde\eta$** 把架构参数（组数 $G$、组轮数 $P$、组内客户端数 $M$、本地步数 $K$）吸收进去，四个界就能写成同一种形态、逐项对照。

**核心 idea**：用"统一非凸假设 + 有效学习率 + 组间/组内异质性分解"三件套，把四种 HFL 拓扑的收敛界统一推导出来并直接比较，从误差项里读出拓扑选择规律。

## 方法详解

### 整体框架

本文不是一个新算法，而是一套**统一收敛分析框架**。先把两层 HFL 的优化目标形式化：全局目标是最小化
$$F(x) = \frac{1}{G}\sum_{g=1}^{G} F_g(x) = \frac{1}{G}\sum_{g=1}^{G}\frac{1}{M}\sum_{m=1}^{M} F_{g,m}(x),$$
其中 $F_g$ 是第 $g$ 组内所有客户端目标的平均，$F_{g,m}$ 是第 $g$ 组第 $m$ 个客户端的本地目标。四种拓扑的区别只在于**两层各自用星型还是环型聚合**：星型让组内/组间的成员从同一个起点并行更新、最后取平均；环型让成员按顺序从上一个成员的结果接着更新、像接力一样串行传播。

框架的核心动作有三步：① 用全方差定律把全局异质性拆成组间 $\delta$ 与组内 $\zeta$ 两个有界量（假设 3、4）；② 为每种拓扑定义一个吸收了架构参数的有效学习率 $\tilde\eta$（如 Star-Star 取 $\tilde\eta=PK\eta$，Ring-Ring 取 $\tilde\eta=GPMK\eta$）；③ 在 $L$-光滑、方差有界、组间/组内异质性有界这套**对四种拓扑完全一致**的假设下，推出统一形态的收敛界（定理 1 全参与、定理 2 部分参与），再用最优 $\tilde\eta$ 化简成可比较的收敛率（推论 1、2）。框架还扩展到了随机执行顺序（random-shuffle）和部分参与（采样 $S_1$ 个组、每组 $S_2$ 个客户端）两种现实设定。

### 关键设计

**1. 异质性二分解：把"组间 δ vs 组内 ζ"拆开，才能定位真正的瓶颈**

HFL 的麻烦在于数据异质性是分层的——既有组与组之间分布的差异，又有组内客户端之间的差异，二者交织影响收敛。本文用**全方差定律**给出一个恒等式：
$$\frac{1}{G}\sum_{g}\frac{1}{M}\sum_{m}\|\nabla F_{g,m}(x)-\nabla F(x)\|^2 = \underbrace{\frac{1}{G}\sum_{g}\|\nabla F_g(x)-\nabla F(x)\|^2}_{\text{组间 }\delta^2} + \underbrace{\frac{1}{G}\sum_{g}\frac{1}{M}\sum_{m}\|\nabla F_{g,m}(x)-\nabla F_g(x)\|^2}_{\text{组内 }\zeta^2},$$
即全局异质性恰好是组间 $\delta^2$（假设 3）与组内 $\zeta^2$（假设 4）的**正交划分**。这个分解不是为了好看：它让收敛界里 $\delta$ 和 $\zeta$ 各自成项，从而能直接读出"哪一类异质性衰减得慢、是真正的瓶颈"。这正是后面"组间异质性主导"结论的数学源头。

**2. 拓扑相关的有效学习率 $\tilde\eta$：把四种拓扑写成同一形态再对照**

四种拓扑的更新机制差异很大，直接比较收敛界会陷入参数迷宫。本文的关键技巧是为每种拓扑定义一个**有效学习率 $\tilde\eta$**，把架构参数全部吸收进去：Star-Star 用 $\tilde\eta=PK\eta$、Star-Ring 用 $PMK\eta$、Ring-Star 用 $GPK\eta$、Ring-Ring 用 $GPMK\eta$。代入后，所有界都化成"优化项 $\frac{A}{\tilde\eta R}$ + 随机噪声项 + 异质性项"的统一结构，例如 Ring-Ring 的全参与界为
$$\mathbb{E}\|\nabla F(\bar x^{(R)})\|^2 \lesssim \frac{A}{\tilde\eta R} + \frac{L\tilde\eta\sigma^2}{GPMK} + \frac{L^2\tilde\eta^2\sigma^2}{GPMK} + \frac{L^2\tilde\eta^2\zeta^2}{G^2P^2} + L^2\tilde\eta^2\delta^2.$$
$\tilde\eta$ 越大优化越快、但误差项被放大，存在权衡；取最优 $\tilde\eta$ 后（推论 1）四种拓扑分享同一渐近率 $O(1/\sqrt{GPMKR})$，差异全部落在低阶的拓扑相关项 $T$ 上——于是逐项对照 $T$ 就能比出优劣。这也解释了实验里观察到的原始学习率规律 $\eta_{\text{Star-Star}} > \eta_{\text{Star-Ring}} \approx \eta_{\text{Ring-Star}} > \eta_{\text{Ring-Ring}}$：环型聚合自带更大的"有效放大"，所以需要更小的原始 $\eta$。

**3. 从误差项读出三条拓扑选择原则：顶层主导、组间瓶颈、结构匹配**

把统一收敛界的各项摆在一起（表 1 / 推论 1），三条规律自然浮现。其一**顶层主导原则**：把顶层从星型换成环型（Ring-Star vs Star-Star、Ring-Ring vs Star-Ring）会在 SGD 方差项和组内异质性项的分母里多出一个因子 $G$（部分参与下是 $S_1$），例如组内项从 $\frac{\zeta^2}{P^2}$ 变成 $\frac{\zeta^2}{G^2P^2}$，意味着环型顶层把误差额外压了 $G^2$ 倍，因此对噪声和异质性更鲁棒；顶层拓扑的差异比底层拓扑的差异更显著。其二**组间异质性瓶颈**：全参与下所有拓扑里组间项都以 $O\big(\frac{(L^2A^2\delta^2)^{1/3}}{R^{2/3}}\big)$ 慢速衰减、且分母不含 $G/P$，而组内项在环型顶层下能享受 $\frac{1}{G^{2/3}P^{2/3}}$ 的额外加速——所以 $\delta$ 才是真正卡住收敛的主因，系统设计应优先让分组逼近 IID（组间分布接近全局分布）。其三**拓扑-结构匹配**：Ring-Star 的收敛率随 $G$ 增大而显著改善，适合"许多小组"（IoT、零售网点）；Star-Ring 的组内环型聚合能在全局同步前做更深的本地精修，适合"少数大组"（医院网络）；Star-Star 因两层都做平均、双重平均严重抑制有效学习率，在所有场景下都最差。

### 损失函数 / 训练策略

本文不引入新损失，沿用标准 FL 的本地经验风险最小化。理论建立在四条假设上：$L$-光滑（假设 1）、随机梯度无偏且方差被 $\sigma^2$ 界住（假设 2）、组间异质性被 $\delta^2$ 界住（假设 3）、组内异质性被 $\zeta^2$ 界住（假设 4）。这套假设的关键是**对四种拓扑完全一致且足够一般**（不依赖"梯度一致有界"这种强假设），从而支持公平的横向比较；这也是它相比前人工作（Lee et al. 2020、Yan et al. 2025 给出的是 $O(1/\sqrt{R})$）能为 Star-Ring/Ring-Star 推出更紧的 $O(1/\sqrt{GPMKR})$ 界的原因。

## 实验关键数据

实验设置：$N=100$ 个客户端分成 $G=10$ 组，四种数据划分（组间/组内各自 IID 或 Non-IID，用 Dirichlet 分布生成 Non-IID），每种拓扑单独调学习率以保证公平。

### 主实验：四种拓扑的最终测试精度

| 数据集/模型 | 异质性(组间/组内) | Star-Star | Star-Ring | Ring-Star | Ring-Ring |
|------------|------------------|-----------|-----------|-----------|-----------|
| CIFAR-10 / ResNet-18 | IID / IID | 88.48 | 90.30 | 90.40 | **91.53** |
| CIFAR-10 / ResNet-18 | Non-IID / Non-IID | 86.78 | 87.40 | 90.01 | **90.33** |
| CINIC-10 / ResNet-18 | Non-IID / Non-IID | 73.63 | 74.21 | **77.11** | 76.78 |
| Fashion-MNIST / ResNet-10 | Non-IID / Non-IID | 88.04 | 92.18 | 92.27 | **93.33** |
| SST-2 / MLP | Non-IID / IID | 68.12 | 73.85 | 79.13 | **81.08** |

**环型顶层（Ring-Star / Ring-Ring）几乎在所有设置下都优于星型顶层**，Star-Star 一致最差（等价于标准 HFedAvg）。SST-2 上 Ring 系比 Star-Star 高出约 11~13 个百分点，差距最夸张。

### 组间 vs 组内异质性的影响

| 场景（CIFAR-10/ResNet-18, Star-Ring） | 精度 | 相对 IID-IID 掉点 |
|------|------|------|
| 组间 IID + 组内 IID | 90.30 | — |
| 组间 IID + 组内 Non-IID | 89.55 | −0.75 |
| 组间 Non-IID + 组内 IID | 88.22 | **−2.08** |

CINIC-10 上同样设置：组间 Non-IID 掉 6.06%，组内 Non-IID 只掉 2.50%；Fashion-MNIST 上组间 Non-IID 掉 3.18%、组内 Non-IID 仅掉 0.26%。

### 关键发现
- **组间异质性远比组内异质性致命**：把组间从 IID 变成 Non-IID 造成的掉点是组内变化的 2~12 倍，强力佐证了"$\delta$ 是主瓶颈"的理论结论——分组策略应优先让组间分布接近全局分布（如按地理/语义相似度聚类），而非纠结组内同质化。
- **拓扑选择随组数 $G$ 反转**：固定 $N=100$、变 $G\in\{1,5,10,20,100\}$，Star-Ring 在小 $G$（少而大的组）下最好（更长的组内更新链带来更深本地精修），Ring-Star 在大 $G$（多而小的组）下最好（小组减少了星型平均对梯度的稀释，串行组间更新带来细粒度全局对齐）。极端退化（Ring-Star 取 $G=N$、Star-Ring 取 $G=1$，都退化成纯环）结论仍成立，且因步长随 $G$ 缩放保持有效学习率恒定，不会出现"灾难性遗忘"。
- **环型更激进也更抖**：环型顶层让更新串行传播、能更快穿越损失地形，但对数据偏斜更敏感，训练曲线波动更大，需要更仔细的学习率校准。

## 亮点与洞察
- **统一框架的价值在于"可比较"**：以前每篇论文用不同假设分析不同拓扑，根本没法说谁好谁坏；本文用同一套一般假设 + 有效学习率，把四种拓扑写进同一张表逐项对照，这才是"统一分析"的真正意义。
- **全方差定律的优雅运用**：把全局异质性正交拆成组间+组内，不仅是记号方便，而是让"组间 vs 组内谁是瓶颈"变成可从收敛界分母直接读出的事实——理论与可操作建议之间的桥。
- **反直觉结论**：HFL 主要价值是"可扩展性"而非"加速收敛"，精心配置的单层 FL 甚至可能比两层 HFL 收敛更快；这提醒实践者别把 HFL 当万能加速器。
- **可迁移的分析范式**：用"架构参数吸收进有效学习率，再逐项对照低阶误差项"的思路，可推广到其它多层/去中心化优化的拓扑比较。

## 局限与展望
- 仅限两层 HFL，多层（≥3 层）层次结构的统一分析尚未覆盖。
- 收敛界依赖标准的 $L$-光滑 + 方差有界 + 异质性有界假设，对自适应优化器、压缩通信、客户端漂移更复杂的真实场景未必直接适用。
- 实验固定 $N=100$、$G=10$（变 $G$ 实验也在此规模内），更大规模、更真实的网络/带宽异质性下的 wall-clock 表现只在附录做了分析性讨论，缺乏大规模实测。
- 环型拓扑的稳定性敏感问题（对数据偏斜抖动大）只给了经验观察，缺乏理论刻画与自动调参方案。

## 相关工作与启发
- **vs Star-Star 专属分析（Wang et al. 2022, Jiang & Zhu 2024）**：他们只分析最简单的 Star-Star，本文把同一套假设推广到全部四种拓扑并给出 Ring-Ring 的首个收敛分析，能横向比较。
- **vs Star-Ring/Ring-Star 早期分析（Chen et al. 2020, Lee et al. 2020, Yan et al. 2025）**：他们用了"梯度一致有界"的强假设、且只给出 $O(1/\sqrt{R})$ 的松界；本文用更一般的假设推出更紧的 $O(1/\sqrt{GPMKR})$ 界。
- **vs 单层 FL 理论（Li & Lyu 2023 的 SFL、Koloskova et al. 2020 的 PFL）**：本文在 $P=M=1,\zeta=0$ 时退化为标准 FedAvg 的已知率，把单层结果作为特例统一进来。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个覆盖全部四种 HFL 拓扑（含从未分析过的 Ring-Ring）的统一收敛框架，假设一致、界更紧。
- 实验充分度: ⭐⭐⭐⭐ 四数据集四模型四种异质性 + 变组数实验，覆盖面好，但规模偏小、缺大规模 wall-clock 实测。
- 写作质量: ⭐⭐⭐⭐⭐ 从分析挑战到三条原则层层递进，理论与可操作建议衔接清晰。
- 价值: ⭐⭐⭐⭐⭐ 给出"什么场景选什么拓扑"的理论指导，对 HFL 系统设计有直接落地意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[ICLR 2026\] Federated Learning with Profile Mapping under Distribution Shifts and Drifts](federated_learning_with_profile_mapping_under_distribution_shifts_and_drifts.md)

</div>

<!-- RELATED:END -->

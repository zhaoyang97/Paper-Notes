---
title: >-
  [论文解读] Perturbed Dynamic Time Warping: A Probabilistic Framework and Generalized Variants
description: >-
  [ICLR 2026][时间序列][动态时间规整] 本文用"给对齐代价加随机噪声再取期望最小"的扰动优化视角重新解释了 soft-DTW（证明它恰好是 Gumbel 噪声下的特例），并把噪声推广到广义极值（GEV）分布，得到带可调偏斜度的 nested-soft-DTW（ns-DTW），在时间序列重心计算、聚类和分类上稳定优于 soft-DTW。
tags:
  - "ICLR 2026"
  - "时间序列"
  - "动态时间规整"
  - "soft-DTW"
  - "扰动优化"
  - "广义极值分布"
  - "随机效用理论"
---

# Perturbed Dynamic Time Warping: A Probabilistic Framework and Generalized Variants

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=gxCOTacltM](https://openreview.net/forum?id=gxCOTacltM)  
**代码**: 随补充材料提供（UCR 数据集公开）  
**领域**: 时间序列 / 可微动态规划  
**关键词**: 动态时间规整, soft-DTW, 扰动优化, 广义极值分布, 随机效用理论

## 一句话总结
本文用"给对齐代价加随机噪声再取期望最小"的扰动优化视角重新解释了 soft-DTW（证明它恰好是 Gumbel 噪声下的特例），并把噪声推广到广义极值（GEV）分布，得到带可调偏斜度的 nested-soft-DTW（ns-DTW），在时间序列重心计算、聚类和分类上稳定优于 soft-DTW。

## 研究背景与动机
**领域现状**：动态时间规整（DTW）是衡量两条时间序列相似度的经典方法，它通过寻找代价最小的对齐路径来容忍时间轴上的拉伸/压缩，比欧氏距离更适合长度不等、节奏不一致的序列，广泛用于语音、动作识别、轨迹聚类等。但 DTW 的核心是一个 $\min$ 操作，不可微，没法塞进端到端的梯度训练里。

**现有痛点**：Cuturi & Blondel（2017）提出的 soft-DTW 把硬 $\min$ 换成光滑的 soft-min（$\min_\gamma x = -\gamma\log\sum_i\exp(-x_i/\gamma)$），从而可微、可递归求梯度，成了事实标准。但 soft-DTW 的 soft-min 本质上是一个**启发式的光滑算子**——为什么用 log-sum-exp、它对应什么概率结构、能不能换别的形式来获得更灵活的对齐，这些问题 soft-DTW 自己回答不了。

**核心矛盾**：可微性是被"硬塞"进来的（用一个特定的解析光滑函数），而不是从某个统一原理推导出来的。这导致 soft-DTW 形态单一：它给出的期望对齐分布是固定的，无法刻画"偏向竖直方向 / 偏向水平方向"这类有结构偏好的对齐。

**切入角度**：作者借用扰动优化器（perturbed optimizer，Berthet et al. 2020）和随机效用理论（random utility theory）的视角——把"选哪条对齐路径"看成一次离散选择，每条路径的效用 $-\langle A,C\rangle$ 被一个随机冲击 $\gamma\varepsilon$ 扰动，最终的可微量是"扰动后最小代价的期望"。这个视角天然可微（期望把不连续的 $\arg\min$ 抹平），而且换不同的噪声分布就能得到不同的 DTW 变体。

**核心 idea**：用"对齐代价加随机扰动 + 取期望最小"替换 soft-DTW 的启发式 soft-min；证明 Gumbel 噪声下它精确等于 soft-DTW（给出概率解释），再把噪声换成 GEV 分布得到可调偏斜度的 ns-DTW。

## 方法详解

### 整体框架
整篇方法是一条"从一般原理推到具体算法"的逻辑链，而非数据流 pipeline。出发点是把 DTW 的不可微 $\min$ 用一个**扰动最小算子**替换：给每条合法对齐路径的代价 $\langle A,C\rangle$ 减去一个随机噪声 $\gamma\varepsilon$，先在所有路径上取最小，再对噪声分布求期望，得到 perturbed-DTW（定义 1）。这个量天生可微：最优对齐不再是单一的 0/1 矩阵，而变成路径上的一个概率分布，其对代价矩阵 $C$ 的梯度恰好是"期望对齐矩阵" $E$。

接着作者代入两种具体噪声。第一种是 i.i.d. Gumbel 噪声，利用 Gumbel 变量最小值的解析性质（引理 1），证明 perturbed-DTW 精确退化为 soft-DTW（命题 1）——这把 soft-DTW 解释成"Gumbel 扰动下 DTW 的期望"。第二种是更一般的广义极值（GEV）分布，它能在路径分组之间引入相关性，得到新变体 ns-DTW（定理 1），其对齐分布可以有**可调的偏斜度**。最后把这套框架落到动态规划上：用扰动最小算子替换 Bellman 递推里的硬 $\min$，得到可在 $O(mn)$ 内计算的递推公式，并同步算出每步三个方向的转移概率张量 $G$，从而反向递推出期望对齐矩阵 $E$（算法 1）。

整条链子是：扰动最小算子（一般框架）→ Gumbel 特例（= soft-DTW，给出概率解释）→ GEV 推广（= ns-DTW，可调偏斜）→ DP 实现（可计算）。

### 关键设计

**1. 扰动-DTW：用"加噪声取期望最小"替换不可微的硬 min**

DTW 不可微的根源是 $\min_{A\in\mathcal{A}_{m,n}}\langle A,C\rangle$ 这个硬最小。作者不去找一个光滑函数硬套，而是给每条对齐路径的代价注入一个随机扰动，再取期望：

$$\text{perturbed-DTW}_\gamma(C) := \mathbb{E}_{\varepsilon\sim P}\Big[\min_{A\in\mathcal{A}_{m,n}}\{\langle A,C\rangle - \gamma\varepsilon\}\Big]$$

其中 $\varepsilon$ 是维度为 $|\mathcal{A}_{m,n}|$（所有合法对齐数）的扰动向量，$\gamma>0$ 是温度。直觉上每次抽一组噪声就解出一条硬 DTW 最优路径，对噪声求期望就把这些"随机实现下的最优路径"聚合成一个分布。期望算子把不连续的 $\arg\min$ 抹平，所以这个量对 $C$ 可微，且梯度就是期望对齐矩阵 $E=\sum_A A\cdot P(A;C)$（命题 3）。这与随机效用理论里"选择由被随机冲击扰动的效用决定"完全对应：对齐矩阵 $A$ 是一个选择，$-\langle A,C\rangle$ 是它的效用，$\gamma\varepsilon$ 是随机效用冲击。和 soft-DTW 靠"启发式 soft-min"不同，这里的可微性来自**随机化**这一统一原理，因此换噪声就能换出新变体。

**2. Gumbel 扰动恰好还原 soft-DTW：给 soft-DTW 一个概率解释**

把噪声取成 i.i.d. Gumbel$(-c,1)$（$c\approx0.5772$ 为欧拉常数）时，利用"Gumbel 变量最小值等于其取负后最大值的负数"这一性质（引理 1）：

$$\mathbb{E}\big[\min\{x_1-\gamma\varepsilon_1,\dots,x_n-\gamma\varepsilon_n\}\big] = -\gamma\log\sum_{i=1}^n\exp(-x_i/\gamma)$$

右边正是 soft-min。代入 perturbed-DTW 即得 $-\gamma\log\sum_{A}\exp(-\langle A,C\rangle/\gamma)$，与 soft-DTW 完全相同；其最优对齐矩阵也精确是 soft-DTW 的 Gibbs 分布 $P(A;C)=\exp(-\langle A,C\rangle/\gamma)/\sum_{A'}\exp(-\langle A',C\rangle/\gamma)$（命题 1）。更进一步（命题 4），若定义随机扰动代价矩阵 $[\tilde C_\gamma]_{i,j}=C_{i,j}-\gamma\varepsilon_{i,j}$，则 $\text{soft-DTW}_\gamma(C)=\mathbb{E}[\text{DTW}(\tilde C_\gamma)]$——soft-DTW 就是"局部代价被 Gumbel 扰动后 DTW 的期望"。这个等价不只是漂亮，它说明 soft-DTW 的 log-sum-exp 并非随意选取，而是 Gumbel 噪声这一特定先验的必然结果，从而把"换噪声 = 换 DTW 变体"这条路打通。

**3. GEV 扰动得到 ns-DTW：用分组相关噪声引入可调偏斜的对齐**

Gumbel 噪声是各路径独立的，对齐分布的形态被锁死。作者把噪声换成广义极值（GEV）分布——它可以看成 Gumbel 的相关多元推广。把所有对齐 $\mathcal{A}_{m,n}$ 划分成 $J$ 组，组内用相似度参数 $\tau_\ell\in(0,1]$ 刻画相关性，其联合 CDF 为 $F=\exp\big(-\sum_j(\sum_k\exp(-\varepsilon_{jk}/\tau_j))^{\tau_j}\big)$。在零均值化后，perturbed-DTW 在 GEV 噪声下变成 nested-soft-DTW（定理 1）：

$$\text{ns-DTW}_\gamma(C) = -\gamma\log\sum_{\ell=1}^{J}\Big(\sum_{A\in\ell}\exp\big(-\tfrac{\langle A,C\rangle}{\gamma\tau_\ell}\big)\Big)^{\tau_\ell}$$

当所有 $\tau_\ell=1$ 时它退化回 soft-DTW；当 $0<\tau_\ell<1$ 时组内路径之间产生相关性，对齐分布可以**偏斜**——这正是它比 soft-DTW 多出来的灵活性。这个嵌套结构对应随机效用理论里的 nested logit 模型，其正则项也从 Shannon 熵变成"嵌套 Shannon 熵"（命题 2、式 13）。$\tau$ 越小模型越"挑剔"：高累积代价的转移权重趋于消失，低代价转移被强调，对齐路径被结构性地推向低代价方向（如竖直 $\downarrow$）。

**4. 扰动最小算子的 DP 递推：在 O(mn) 内算出 ns-DTW 与期望对齐**

直接按定义枚举 $\mathcal{A}_{m,n}$ 是指数级的，没法算。作者把 DTW 的 Bellman 递推里的硬 $\min$ 换成扰动最小算子，定义扰动累积代价矩阵 $V$：每步只在三个方向 $\{\rightarrow,\downarrow,\searrow\}$ 上对噪声求期望。Gumbel 下递推化简为 $V_{i,j}=-\gamma\log\big(\exp(-V_{i,j-1}/\gamma)+\exp(-V_{i-1,j}/\gamma)+\exp(-V_{i-1,j-1}/\gamma)\big)+C_{i,j}$（式 14），与 soft-DTW 的递推一致。GEV 下则按分组方案（如 $J_1=\{\downarrow,\rightarrow\}, J_2=\{\searrow\}$）写成嵌套形式（式 16），不同分组 $g_1,g_2,g_3$ 让对齐偏向不同方向。同时每步算出转移概率张量 $G\in(0,1]^{m\times n\times 3}$（式 17，是 $V$ 的梯度），再反向递推 $E_{i,j}=G_{i,j+1,1}E_{i,j+1}+G_{i+1,j,2}E_{i+1,j}+G_{i+1,j+1,3}E_{i+1,j+1}$ 得到期望对齐矩阵，全程 $O(mn)$。作者诚实指出：局部 DP（对每步三方向各自施加低维 GEV 扰动）与定理 1 里"全局单个 $|\mathcal{A}_{m,n}|$ 维 GEV 扰动"并不严格等价——因为 GEV 算子嵌套后不像 log-sum-exp 那样保持分布族封闭，所以式 16 的 $V_{m,n}$ 是 ns-DTW 的一个**可计算近似实现**而非精确值；但全文仍把它称作 ns-DTW。⚠️ 公式细节以原文为准。

### 损失函数 / 训练策略
ns-DTW 本身作为可微的序列差异度量使用，没有引入额外训练损失。在重心计算任务里，目标是最小化 $\min_x\sum_i \text{ns-DTW}(C(x,y_i))$（Fréchet 均值）；分类用最近质心（NCC）和 1NN；聚类用 k-means，类内距离与质心都基于 ns-DTW。关键超参是温度 $\gamma\in\{0.1,0.01,0.001,0.0001\}$、相似度参数 $\tau\in\{0.80,0.85,0.90,0.95\}$ 和方向分组方案 $\{g_1,g_2,g_3\}$，均通过交叉验证选取。

## 实验关键数据

实验在 UCR 时间序列分类档案的 47 个数据集子集上做重心、分类、聚类三类任务，baseline 含子梯度法（Subgradient）、DBA、soft-DTW。

### 主实验：重心计算（barycenter，比谁的 DTW 损失更低）

下表为 ns-DTW（选最优分组 $g_i$ 和 $\tau$）在多少比例数据集上取得比对手更低的 DTW 损失：

| $\gamma$ | vs Subgradient | vs DBA | vs soft-DTW |
|----------|----------------|--------|-------------|
| 0.1 | 68.09% | 46.81% | 36.17% |
| 0.01 | 80.85% | 72.34% | 59.57% |
| 0.001 | 95.74% | 87.23% | 80.85% |
| 0.0001 | 100.00% | 91.49% | 91.49% |

在双方都调到最优 $\gamma$ 的公平比较下，ns-DTW 相对 Subgradient 在 100%、DBA 在 97.87%、最强 baseline soft-DTW 在 74.47% 的数据集上更优。

### 分类与聚类（ns-DTW 取得 ≥ baseline 准确率的数据集比例）

| 任务 | vs Subgradient | vs DBA | vs soft-DTW |
|------|----------------|--------|-------------|
| NCC 分类 | 93.02% | 88.37% | 86.05% |
| 1NN 分类 | — | 88.37% | 86.05% |
| k-means 聚类 | — | 100.00% | 76.60% |

### 关键发现
- **$\gamma$ 越小，ns-DTW 优势越明显**：从 $\gamma=0.1$ 到 $0.0001$，对 soft-DTW 的胜率从 36% 升到 91%。小 $\gamma$ 时度量更接近硬 DTW，ns-DTW 的偏斜机制能更精准地贴近真实最优路径。
- **偏斜度（由 $\tau$ 和分组控制）是核心增益来源**：$\tau\to0$ 时模型剪掉高代价偏离、把期望路径推向低代价方向（如竖直 $\downarrow$），这种结构偏好正是 soft-DTW 缺失的，也是 ns-DTW 在重心质量上更优的根本原因。
- **三种分组 $g_1,g_2,g_3$ 给出方向不同的对齐**（图 3、图 4），让方法能适配不同的对齐结构，而非一刀切。

## 亮点与洞察
- **把 soft-DTW 从"启发式光滑"升级为"概率模型的特例"**：证明 soft-DTW $=\mathbb{E}[\text{DTW}(\tilde C_\gamma)]$（Gumbel 扰动局部代价后的期望），这是最让人"啊哈"的地方——它解释了 log-sum-exp 为何长这样，并立刻打开了"换噪声换变体"的设计空间。
- **借随机效用理论搭桥**：把对齐选择映射成离散选择问题，Gumbel→logit、GEV→nested logit 的对应关系直接搬过来，使得"如何引入相关性/偏斜"有现成的经济学工具可用，思路可迁移到其他可微组合优化（如可微排序、子集选择）。
- **理论与可计算性分得清楚**：作者明确承认 DP 实现（局部低维 GEV）与全局定义（单个高维 GEV）不严格等价，这种诚实在偏理论的论文里难得，也提示了后续"如何让 DP 真正逼近全局 GEV"的研究空间。

## 局限与展望
- **ns-DTW 的 DP 是近似而非精确**：嵌套 GEV 算子不保持分布族封闭，$V_{m,n}$ 只是理论 ns-DTW 的可计算实现，二者差距没有定量刻画。
- **超参搜索空间大**：分组方案 $\times\,\tau\,\times\,\gamma$ 需要交叉验证联合调，且实验里报告的是"选最优组合后"的胜率，实际部署时调参成本不低。
- **实验局限在经典任务**：只在 UCR 上做重心/分类/聚类，没有验证嵌进深度网络做端到端训练（而这正是 soft-DTW 最常用的场景），可微性带来的下游收益尚未直接展示。
- **作者展望**：把扰动框架推广到基于散度（divergence）的形式、以及更广的扰动动态规划算子族。

## 相关工作与启发
- **vs soft-DTW（Cuturi & Blondel, 2017）**：soft-DTW 用启发式 soft-min 获得可微；本文证明它只是 Gumbel 扰动下的特例，并用 GEV 推广出可调偏斜的 ns-DTW，灵活性更高，代价是引入额外超参 $\tau$ 和分组。
- **vs 扰动优化器（Berthet et al., 2020）**：他们提出"给线性目标加噪声取期望"使一般组合优化可微；本文把这套框架专门落到 DTW 的结构化对齐上，并具体给出 DP 递推与转移概率张量的计算。
- **vs 可微动态规划（Mensch & Blondel, 2018）**：他们把 soft-DTW 重写成熵正则 DP；本文把正则项从 Shannon 熵推广到嵌套 Shannon 熵，对应 nested logit 的相关结构。
- **vs DBA / 子梯度重心法（Petitjean 2011 / Schultz & Jain 2018）**：传统重心方法不可微或依赖子梯度；ns-DTW 提供光滑可微的重心目标且实验上质量更优。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用扰动优化+随机效用理论统一解释 soft-DTW 并推广出 GEV 变体，视角新且漂亮
- 实验充分度: ⭐⭐⭐⭐ UCR 上三类任务全面对比，但缺端到端深度训练场景验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，对 DP 近似与全局定义的差异交代诚实
- 价值: ⭐⭐⭐⭐ 为 soft-DTW 家族提供了原理性框架，思路可迁移到其他可微组合优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] End-to-End Probabilistic Framework for Learning with Hard Constraints](end-to-end_probabilistic_framework_for_learning_with_hard_constraints.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](../../NeurIPS2025/time_series/timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)

</div>

<!-- RELATED:END -->

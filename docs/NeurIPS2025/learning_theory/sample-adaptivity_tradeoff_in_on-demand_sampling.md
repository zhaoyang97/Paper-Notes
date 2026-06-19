---
title: >-
  [论文解读] Sample-Adaptivity Tradeoff in On-Demand Sampling
description: >-
  [NeurIPS 2025][多分布学习] 系统研究了按需采样（on-demand sampling）中样本复杂度与自适应轮次之间的权衡关系，在可实现设定下证明 $r$ 轮算法的最优样本复杂度为 $dk^{\Theta(1/r)}/\varepsilon$，在不可知设定下提出仅需 $\widetilde{O}(\sqrt{k})$ 轮即可达近最优样本复杂度的LazyHedge算法，并引入OODS抽象框架建立了近紧的轮次复杂度下界。
tags:
  - "NeurIPS 2025"
  - "多分布学习"
  - "样本复杂度"
  - "轮次复杂度"
  - "自适应采样"
  - "在线优化"
  - "Boosting"
---

# Sample-Adaptivity Tradeoff in On-Demand Sampling

**会议**: NeurIPS 2025  
**arXiv**: [2511.15507](https://arxiv.org/abs/2511.15507)  
**作者**: Nika Haghtalab (UC Berkeley), Omar Montasser (Yale), Mingda Qiao (UMass Amherst)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: 多分布学习, 样本复杂度, 轮次复杂度, 自适应采样, 在线优化, Boosting  

## 一句话总结

系统研究了按需采样（on-demand sampling）中样本复杂度与自适应轮次之间的权衡关系，在可实现设定下证明 $r$ 轮算法的最优样本复杂度为 $dk^{\Theta(1/r)}/\varepsilon$，在不可知设定下提出仅需 $\widetilde{O}(\sqrt{k})$ 轮即可达近最优样本复杂度的LazyHedge算法，并引入OODS抽象框架建立了近紧的轮次复杂度下界。

## 研究背景与动机

### 问题背景
现代机器学习中，自适应数据采集（根据中间学习信号动态调整采样策略）已成为重要范式。多分布学习（Multi-Distribution Learning, MDL）框架允许算法从 $k$ 个分布中自适应选择采样，以最小化最坏情况误差。然而自适应性要求顺序收集数据，限制了并行化和可扩展性。这引出了核心问题：**自适应采样在多大程度上是必要的？轮次数与样本复杂度之间存在怎样的定量权衡？**

### 已有工作的不足
- **可实现设定**：已知全自适应（$O(\log k)$ 轮）可达最优样本复杂度 $\widetilde{O}((d+k)/\varepsilon)$，完全非自适应（1轮）需 $\Omega(dk/\varepsilon)$，但两者之间的中间地带完全未知
- **不可知设定**：最优样本复杂度 $\widetilde{O}((d+k)/\varepsilon^2)$ 已通过HJZ (2022)和ZZC+ (2024)等工作建立，但这些算法的轮次数至少为 $\text{poly}(1/\varepsilon)$，有的甚至与样本复杂度一样大（每轮仅采一个样本）
- 是否存在常数轮次、或仅依赖 $k$ 而独立于 $\varepsilon$ 的轮次就能恢复最优样本复杂度的算法，此前完全未知

### 核心动机
填补全自适应与完全非自适应之间的理论空白，给出样本-轮次权衡的精确刻画。同时发展新的算法框架（OODS）来理解这一权衡的本质困难性。

## 方法详解

### 问题建模：多分布学习（MDL）
- 给定 $k$ 个未知分布 $D_1, \ldots, D_k$，假设类 $\mathcal{H}$（VC维为 $d$）
- 目标：学习预测器 $\hat{h}$ 使得 $\max_{i \in [k]} \text{err}(\hat{h}, D_i) \leq \text{OPT} + \varepsilon$
- $r$-轮算法在 $r$ 轮中自适应采样，轮次间可根据已有样本调整策略
- 可实现设定：$\text{OPT}=0$（存在完美假设）；不可知设定：$\text{OPT}$ 任意

### 可实现设定：基于AdaBoost的分布级Boosting（Algorithm 1）

**核心思想**：在分布层面运行AdaBoost变体，利用边缘（margin）放大性质。

算法流程：
1. 初始化均匀权重 $q_1(j) = 1/k$，设边缘参数 $\theta = r/(2\log k)$
2. 设弱学习器误差率 $p = \frac{1}{2}(4k^{2/r})^{-1/(1-\theta)}$
3. 每轮 $t = 1, \ldots, r$：
    - 从混合分布 $q_t = \sum_j q_t(j) D_j$ 采 $m = O(d/(\varepsilon p))$ 个样本
    - 调用PAC学习器 $\mathbb{A}$ 得到假设 $h_t$
    - 对每个 $D_j$ 采少量样本估计 $\text{err}(h_t, D_j)$
    - 根据阈值化损失 $\mathbb{1}[\text{err}(h_t, D_j) > \tau/2]$ 按AdaBoost规则更新权重
4. 输出多数投票 $F(x) = \mathbb{1}[\frac{1}{r}\sum_{t=1}^r h_t(x) \geq 1/2]$

**关键性质**：通过 $r$ 轮boosting，对每个分布 $D_j$，至少 $1/2 + \theta/2$ 比例的弱学习器误差 $\leq \tau$，由此保证多数投票在所有分布上误差 $\leq \varepsilon$。

**样本复杂度**（Theorem 1）：

$$O\left(k^{2/r} \log k \cdot \frac{d}{\varepsilon} + \frac{k \log(k) \log(k/\delta)}{\varepsilon}\right)$$

特殊情况：$r = \log k$ 恢复最优 $\widetilde{O}((d+k)/\varepsilon)$；$r = 3$ 或 $4$ 即达 $\widetilde{O}((d\sqrt{k}+k)/\varepsilon)$。

### 可实现设定下界（Theorem 2）

$$\Omega\left(\frac{dk^{1/r}}{r\log^2 k}\right)$$

**证明构造**（$r=2$ 即 $\Omega(d\sqrt{k})$ 为例）：
- 实例空间 $\mathcal{X} = \mathbb{F}_2^d$，假设类为线性函数，$h^\star$ 随机选取
- 构造 $k$ 个分布，每个是随机子空间 $V_i$ 上的均匀分布
- 分布难度设为3种级别（$\Theta(d)$、$\Theta(d/\sqrt{k})$、$\Theta(d/k)$）的随机排列
- 子空间线性独立 $\Rightarrow$ 不同分布的样本不提供交叉信息
- 2轮算法必须"跳过"一个识别步骤，导致 $\Omega(d\sqrt{k})$ 开销

### 不可知设定：LazyHedge算法（Algorithm 2）

**出发点**：ZZC+ (2024) 的Hedge算法需 $T = \Theta(\log k / \varepsilon^2)$ 轮，通过懒惰更新减少实际采样轮次。

**核心创新——懒惰上界更新**：
1. 维护权重 $w^{(t)}$ 和上界向量 $\bar{w}^{(t)}$（代理各分布的数据集大小）
2. 每迭代开始检查 $w^{(t)} \in \mathcal{O}(\bar{w}^{(t-1)})$ 是否成立
3. 若**成立**：直接用已有数据执行ERM和Hedge更新，无需新采样
4. 若**不成立**：更新上界 $\bar{w}_i^{(t)} = C \cdot \max\{w_i^{(1)}, \ldots, w_i^{(t)}\}$，补充采样

**两种可观察区域**：
- Box版本：$\mathcal{O}(\bar{w}) = \{w \in \Delta^{k-1}: w_i \leq \bar{w}_i, \forall i\}$
    - 轮次 $O(k\log k)$：每坐标最多 $O(\log_C k)$ 次成为"culprit"
- Ellipsoid版本：$\mathcal{O}(\bar{w}) = \{w \in \Delta^{k-1}: \sum_i w_i^2/\bar{w}_i \leq 1\}$
    - 轮次 $\widetilde{O}(\sqrt{k})$：利用 $\sum_i \max_t w_i^{(t)} \leq \widetilde{O}(1)$（Lemma 3），将更新分为Type I（坐标超 $1/\sqrt{k}$）和Type II两类

**最终结果**（Proposition 1）：$\min\{\widetilde{O}(\sqrt{k}), O(k\log k)\}$ 轮，样本复杂度 $\widetilde{O}((d+k)/\varepsilon^2)$。

### OODS框架：按需采样优化

**抽象化**：将MDL的样本-轮次权衡建模为一般凸优化问题。
- 最大化 $\Delta^{k-1}$ 上的未知凹函数 $f$
- 每轮选择上界 $\bar{w}^{(t)}$，可在 $\mathcal{O}(\bar{w}^{(t)})$ 内查询一阶oracle
- 上界的累计和 $\sum_i \bar{w}_i^{(r)}$ 对应样本开销

**上界**（Theorem 4）：LazyHedge在样本开销 $\widetilde{O}(s)$ 下，Box设定 $\widetilde{O}(k/s)$ 轮，Ellipsoid设定 $\widetilde{O}(\sqrt{k/s})$ 轮。

**下界**（Theorem 9）：构造 $f(w) = \min_{j \in [m]}\{w_{i_j^\star} + j/m^2\}$，利用随机关键索引证明：
- $\varepsilon \leq O(1/k)$ 时，Box $\geq \Omega(\sqrt{k/s})$，Ellipsoid $\geq \Omega((k/s)^{1/4})$
- $\varepsilon \leq e^{-\Omega(k)}$ 时，Box $\geq \Omega(k/s)$，Ellipsoid $\geq \Omega(\sqrt{k/s})$

## 实验关键数据

本文为纯理论工作，无实验部分。核心贡献是建立精确的复杂度刻画。

### 表1：可实现设定的样本-轮次权衡

| 轮次 $r$ | 样本复杂度上界（Theorem 1） | 样本复杂度下界（Theorem 2） | 说明 |
|----------|--------------------------|--------------------------|------|
| $1$ | $O(dk/\varepsilon)$ | $\Omega(dk/\varepsilon)$ | 非自适应，已知紧 |
| $3$ | $\widetilde{O}((d\sqrt{k}+k)/\varepsilon)$ | $\Omega(dk^{1/3}/\varepsilon)$ | 3轮已显著优于1轮 |
| $r$ | $O(k^{2/r}\log k \cdot d/\varepsilon)$ | $\Omega(dk^{1/r}/(r\log^2 k))$ | $k^{\Theta(1/r)}$ 量级匹配 |
| $\log k$ | $\widetilde{O}((d+k)/\varepsilon)$ | $\Omega((d+k)/\varepsilon)$ | 恢复已知最优 |

### 表2：不可知设定——与已有工作对比

| 来源 | 样本复杂度 | 轮次复杂度 |
|------|-----------|-----------|
| HJZ (2022) | $\widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$ | $\widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$ |
| AHZ (2023) | $\widetilde{O}(d/\varepsilon^4 + k/\varepsilon^2)$ | $O(\log(k)/\varepsilon^2)$ |
| ZZC+ (2024) | $\widetilde{O}((d+k)/\varepsilon^2)$ | $O(\log(k)/\varepsilon^2)$ |
| Peng (2024) | $\widetilde{O}((d+k)/\varepsilon^2) \cdot (\log k)^{O(\log(1/\varepsilon))}$ | $(\log k)^{O(\log(1/\varepsilon))}$ |
| **本文** | $\widetilde{O}((d+k)/\varepsilon^2)$ | $\min\{\widetilde{O}(\sqrt{k}), O(k\log k)\}$ |

本文首次实现样本复杂度近最优、且轮次独立于精度 $\varepsilon$ 和VC维 $d$。

## 亮点

- **首次刻画样本-轮次权衡**：在可实现设定给出 $dk^{\Theta(1/r)}/\varepsilon$ 的近紧刻画，揭示常数轮（$r=3$）即可将样本复杂度从 $dk$ 降至 $d\sqrt{k}$ 量级
- **LazyHedge算法**：通过懒惰上界更新策略，将不可知设定轮次从 $\text{poly}(1/\varepsilon)$ 降至 $\widetilde{O}(\sqrt{k})$，首次实现轮次独立于精度
- **OODS抽象框架**：引入按需采样优化问题，统一捕获现有MDL算法的核心结构，建立独立于MDL的轮次下界，揭示了进一步降低轮次的本质障碍
- **AdaBoost的新应用**：巧妙利用AdaBoost的边缘放大性质在分布级别进行boosting，而非传统的样本级别
- **下界构造的精巧性**：可实现下界利用有限域上随机线性函数和多级难度分层；OODS下界通过分层关键索引的凹函数构造

## 局限与展望

- **可实现设定上下界gap**：上界 $k^{2/r}$ vs 下界 $k^{1/r}$，指数差2倍未完全闭合
- **不可知设定缺乏MDL层面的轮次下界**：OODS下界不直接蕴含MDL下界，Open Question 1（是否 $\text{polylog}(k/\varepsilon)$ 轮即可）仍未解决
- **纯理论工作**：没有实验验证算法在实际场景（联邦学习、领域适应等）中的表现
- **仅考虑二分类**：标签空间 $\mathcal{Y} = \{0,1\}$，未扩展到多分类或回归
- **VC类限制**：分析依赖VC维理论，未涉及深度学习中的无穷假设类或函数逼近设定
- **Ellipsoid版本OODS的gap**：下界 $\Omega((k/s)^{1/4})$ vs 上界 $\widetilde{O}(\sqrt{k/s})$，差距仍然显著

## 与相关工作的对比

- **Blum et al. (2017), Chen et al. (2018)**：可实现MDL的 $O(\log k)$ 轮最优算法，本文揭示 $r < \log k$ 时的精确退化行为 $k^{\Theta(1/r)}$
- **Haghtalab et al. (2022)**：首个不可知MDL近最优样本复杂度算法，但轮次 $= \widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$，可能极大
- **Zhu et al. (2024)**：不可知MDL的 $\widetilde{O}((d+k)/\varepsilon^2)$ 样本复杂度，轮次 $O(\log(k)/\varepsilon^2)$ 仍依赖 $\varepsilon$
- **Peng (2024)**：轮次 $(\log k)^{O(\log(1/\varepsilon))}$ 但样本复杂度有同量级额外因子，本文在两个指标上同时更优
- **Agarwal et al. (2017)**：自适应性权衡的先驱工作（多臂赌博机等），本文聚焦MDL给出更精细的刻画
- **Jun et al. (2016), Gao et al. (2019)**：批量多臂赌博机中的自适应性权衡，与本文的OODS框架在结构上有对应关系

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统刻画MDL中样本-轮次权衡，OODS框架是全新抽象
- 实验充分度: ⭐⭐ — 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，从简单到复杂层层递进，直觉解释出色
- 价值: ⭐⭐⭐⭐⭐ — 为理解自适应数据采集的本质困难提供了坚实理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](../../ICML2026/learning_theory/when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](../../ICML2026/learning_theory/on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)
- [\[ICLR 2026\] Alternating Diffusion for Proximal Sampling with Zeroth Order Queries](../../ICLR2026/learning_theory/alternating_diffusion_for_proximal_sampling_with_zeroth_order_queries.md)
- [\[ICML 2026\] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](../../ICML2026/learning_theory/multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)
- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)

</div>

<!-- RELATED:END -->

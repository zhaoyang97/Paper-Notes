---
title: >-
  [论文解读] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions
description: >-
  [AAAI2026][优化/理论][全局优化] 提出ECPv2算法，通过三项创新（自适应下界、Worst-$m$ memory、固定随机投影），将Lipschitz函数全局优化的运行时从$\Omega(n^2 d)$降至$\Omega(n(m+d)\log n)$，同时保持与minimax下界匹配的$O(n^{-1/d})$ regret收敛速率。
tags:
  - "AAAI2026"
  - "优化/理论"
  - "全局优化"
  - "Lipschitz函数"
  - "黑盒优化"
  - "随机投影"
  - "no-regret"
---

# ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions

**会议**: AAAI2026  
**arXiv**: [2511.16575](https://arxiv.org/abs/2511.16575)  
**作者**: Fares Fourati (KAUST), Mohamed-Slim Alouini (KAUST), Vaneet Aggarwal (Purdue)  
**代码**: [GitHub](https://github.com/fouratifares/ECP)  
**领域**: 优化  
**关键词**: 全局优化, Lipschitz函数, 黑盒优化, 随机投影, no-regret  

## 一句话总结

提出ECPv2算法，通过三项创新（自适应下界、Worst-$m$ memory、固定随机投影），将Lipschitz函数全局优化的运行时从$\Omega(n^2 d)$降至$\Omega(n(m+d)\log n)$，同时保持与minimax下界匹配的$O(n^{-1/d})$ regret收敛速率。

## 背景与动机

### 全局优化的核心挑战
全局优化是优化领域的经典难题：目标函数可能非凸、非光滑，且只能通过黑盒评估访问。在机器人控制、机器学习超参数调优、黑盒LLM查询优化等场景下，每次函数评估代价高昂，因此需要在有限预算内尽可能逼近全局最优。

### ECP的优势与不足
前驱工作ECP (Every Call is Precious) 提出了保守的接受准则：仅当候选点在Lipschitz假设下可能是全局最大值时才进行评估，从而保证每次函数调用都具有信息价值。ECP在理论上达到了no-regret保证，并在多种基准上优于Bayesian optimization、CMA-ES等方法。然而ECP存在两大瓶颈：

1. **计算瓶颈**：接受条件需要对所有历史点计算距离，总复杂度为$\Omega(n^2 d)$
2. **保守性过强**：当参数$\varepsilon_t$较小时接受区域可能为空，导致大量无效拒绝

## 方法详解

### 创新一：自适应下界 $\varepsilon_t^{\oslash}$

ECPv2引入理论推导的自适应下界以避免空的接受区域：

$$\varepsilon_t^{\oslash} = \frac{\max_i f(x_i) - \min_j f(x_j)}{\text{diam}(\mathcal{X})}$$

**Lemma 1**证明：若存在任意点$x$满足ECP接受条件，则$\varepsilon_t$必须不小于该下界。实际操作中，每步更新$\varepsilon_t = \max(\tau_{n,d} \cdot \varepsilon_{t-1}, \varepsilon_t^{\oslash})$，额外维护代价仅为$O(1)$（增量跟踪最大最小值）。

**Lemma 2**进一步证明：使用下界后的接受区域是ECP原始接受区域的超集，即$\mathcal{A}_{\text{ECP}}(\varepsilon_t, t) \subseteq \mathcal{A}_t(\varepsilon_t, t)$。

### 创新二：Worst-$m$ Memory机制

不再对所有$t$个历史点检查接受条件，而是仅对函数值最差的$m$个点进行比较。定义最差$m$点的索引集：

$$\mathcal{I}_t^m = \arg\min_{S \subseteq \{1,\ldots,t\}, |S|=m} \sum_{i \in S} f(x_i)$$

修改后的接受条件变为：

$$\min_{i \in \mathcal{I}_t^m} \left(f(x_i) + \max\{\varepsilon_t, \varepsilon_t^{\oslash}\} \cdot \|x - x_i\|_2\right) \geq \max_{j \in [t]} f(x_j)$$

直觉上，差值最大的点对接受条件的约束最强，而高值点的排除反而释放了探索空间。当$m = n$时退化为原始ECP。

### 创新三：固定随机投影加速

利用Johnson-Lindenstrauss引理，将距离计算从$\mathbb{R}^d$投影至$\mathbb{R}^{d'}$空间，其中$d' = 8\log(\beta n) / (\delta^2 - \delta^3)$。投影矩阵$\mathbf{P}$使用i.i.d. $\mathcal{N}(0,1)$随机矩阵，以概率至少$1 - 1/\beta^2$保证所有pairs的距离畸变不超过$\delta$：

$$(1-\delta)\|x_i - x_j\|_2^2 \leq \|\mathbf{P}x_i - \mathbf{P}x_j\|_2^2 \leq (1+\delta)\|x_i - x_j\|_2^2$$

投影后接受条件中的参数调整为$\tilde{\varepsilon}_t = \max\{\varepsilon_t, \varepsilon_t^{\oslash}\} / \sqrt{1-\delta}$，补偿距离的下界畸变。

### 理论保证

**Corollary 1**：三项创新的组合保证$\mathcal{A}_{\text{ECP}}(\varepsilon_t, t) \subseteq \mathcal{A}_{\text{ECPv2}}(\varepsilon_t, t, m, \mathbf{P})$以高概率成立。

**Theorem 2（有限时间regret bound）**：对任意$f \in \text{Lip}(k)$，以概率至少$1 - 1/\beta^2 - \xi$，

$$\mathcal{R}_{\text{ECPv2},f}(n) \leq k \cdot \text{diam}(\mathcal{X}) \cdot \log_{\tau_{n,d}}\!\left(\frac{k}{\varepsilon_1}\right)^{1/d} \cdot \left(\frac{\ln(1/\xi)}{n}\right)^{1/d}$$

该bound与minimax下界$\Omega(k \cdot n^{-1/d})$匹配，保持最优收敛率。

## 实验关键数据

### 复杂度对比

| 方法 | 内存 | 运行时 | Regret上界 |
|------|------|--------|-----------|
| AdaLIPO | $O(nd)$ | $\Omega(n^2 d)$ | $O(n^{-1/d})$ |
| ECP | $O(nd)$ | $\Omega(n^2 d)$ | $O(n^{-1/d})$ |
| **ECPv2** | $O((m+d)\log n)$ | $\Omega(n(m+d)\log n)$ | $O(n^{-1/d})$ |

### 高维基准实验
- Rosenbrock函数 $d \in \{3, 100, 200, 300, 500\}$：ECPv2（$\beta=5, \delta=2/3, m=8$）在$n=200$次评估后性能与ECP持平或更优，wall-clock时间显著减少
- Rosenbrock $d=500$ 和 Powell $d=1000$：ECPv2收敛速度约为ECP的2倍，且优化分数更高
- Worst-$m$消融：$m \in \{8, 16, 32, 64, 128\}$时接受测试代价降低4-5倍，小$m$值有时反而优于完整方法
- 自适应下界单独启用：在低评估预算时运行时约快2倍

### 超参数选择
- $\beta = 5$：保证96%的距离保持成功率
- $\delta = 2/3$：解析推导的最优投影维度最小化值

## 亮点

- **巨大的效率提升**：运行时从$O(n^2 d)$降至$O(n(m+d)\log n)$，内存从$O(nd)$降至$O((m+d)\log n)$，对高维大预算场景意义重大
- **理论保证完整**：regret bound匹配minimax下界，接受区域严格包含ECP，三项创新各有独立的理论支撑
- **模块化设计**：三项创新可独立使用或组合，灵活适配不同场景
- **原则性超参数**：$\beta=5, \delta=2/3$有解析推导支撑，非启发式调参

## 局限与展望

- **仅限Lipschitz连续假设**：对非Lipschitz函数无理论保证
- **regret中的$n^{-1/d}$维度诅咒**：这是Lipschitz优化的固有限制，非算法问题
- **Worst-$m$的$m$选择**：虽然实验表明小$m$即可工作，但缺乏自适应选择$m$的理论指导
- **随机投影引入概率失败**：以概率$1/\beta^2$失败，虽然可通过增大$\beta$控制，但增加投影维度

## 与相关工作的对比

- **AdaLIPO / AdaLIPO+**：依赖均匀随机探索估计Lipschitz常数，样本效率低；ECPv2通过保守接受准则避免浪费评估
- **DIRECT**：确定性空间划分，高维计算量大且过于保守；ECPv2随机采样+投影加速更适合高维
- **Bayesian Optimization (BO)**：依赖高斯过程模型假设，对超参敏感；ECPv2仅需Lipschitz假设，更为通用
- **CMA-ES / Dual Annealing**：缺乏no-regret保证，需大量评估和调参

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三项创新各有理论支撑，但核心思想是对ECP的改进而非全新范式
- 实验充分度: ⭐⭐⭐⭐ — 涵盖多维度基准和详尽消融，但缺乏真实应用场景验证
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，理论推导严谨，定义-引理-定理体系完整
- 价值: ⭐⭐⭐⭐ — 显著推进了Lipschitz优化的可扩展性，解决了ECP的实际部署瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](../../ICML2026/optimization/learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)

</div>

<!-- RELATED:END -->

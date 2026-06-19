---
title: >-
  [论文解读] Optimism Without Regularization: Constant Regret in Zero-Sum Games
description: >-
  [NeurIPS 2025][Fictitious Play] 首次证明无正则化的Optimistic Fictitious Play在2×2零和博弈中获得O(1)常数遗憾，匹配了正则化Optimistic FTRL的最优率，同时证明Alternating Fictitious Play的遗憾下界为Ω(√T)，分离了乐观和交替在无正则化情况下的能力。
tags:
  - "NeurIPS 2025"
  - "Fictitious Play"
  - "乐观学习"
  - "零和博弈"
  - "遗憾界"
  - "无正则化"
---

# Optimism Without Regularization: Constant Regret in Zero-Sum Games

**会议**: NeurIPS 2025  
**arXiv**: [2506.16736](https://arxiv.org/abs/2506.16736)  
**代码**: 暂无  
**领域**: 其他  
**关键词**: Fictitious Play, 乐观学习, 零和博弈, 遗憾界, 无正则化

## 一句话总结

首次证明无正则化的Optimistic Fictitious Play在2×2零和博弈中获得O(1)常数遗憾，匹配了正则化Optimistic FTRL的最优率，同时证明Alternating Fictitious Play的遗憾下界为Ω(√T)，分离了乐观和交替在无正则化情况下的能力。

## 研究背景与动机

在线学习算法在博弈中的遗憾（regret）界是博弈论和机器学习的核心问题。核心背景如下：

**正则化是必要的吗？**
- 在一般对抗设置中，正则化对于无遗憾学习至关重要（如FTRL需要有界步长 $\eta$）
- 但在零和博弈这一特殊结构中，**无正则化算法也能获得次线性遗憾**

**Fictitious Play (FP) 的历史**：
- FP是最经典的无正则化算法（Brown, 1951），等价于步长 $\eta \to \infty$ 的Follow-the-Leader
- FP在一般设置下遗憾可达 $\Omega(T)$（对振荡极度敏感）
- 但在零和博弈中，Robinson (1951) 证明FP的遗憾是次线性的，虽然上界很慢：$O(T^{1-1/n})$（$n \times n$ 博弈）
- 近期工作不断改进FP的遗憾界，如对角矩阵和广义石头剪刀布矩阵上的 $O(\sqrt{T})$

**正则化的乐观算法**：
- Optimistic FTRL（包括Optimistic MWU和Optimistic GD）在零和博弈中获得 $O(1)$ 常数遗憾
- 但标准的RVU bound证明技术**关键依赖**有限的步长上界（即非零正则化）

这引出了核心问题：**零和博弈中，是否可以在完全无正则化（$\eta \to \infty$）的情况下获得 $O(1)$ 遗憾？FP的变体能否达到 $O(1)$？**

这不仅有理论意义，在组合博弈的均衡计算和多智能体强化学习的自博弈训练中有直接应用。

## 方法详解

### 整体框架

研究Optimistic Fictitious Play (OFP)——FP的乐观变体，其更新规则为：

$$x_1^{t+1} = \arg\max_{x \in \{e_i\}_m} \langle x, \sum_{k=0}^t Ax_2^k + Ax_2^t \rangle$$

与标准FP相比，OFP额外加入了对最近反馈的偏置（$+Ax_2^t$ 项），等价于无正则化的Optimistic FTRL。

### 关键设计

1. **对偶空间的几何视角**：定义能量函数 $\Psi(y) = \max_{x \in \Delta_m \times \Delta_n} \langle x, y \rangle$（$\Delta_m \times \Delta_n$ 的支撑函数），其中 $y^t$ 是累积收益向量。关键关系：

$$\text{Reg}(T) = \Psi(y^{T+1})$$

即遗憾恰好等于对偶迭代的能量值（Proposition 3.4）。因此证明常数遗憾等价于证明能量有界。

2. **斜梯度下降统一视角（Proposition 3.6）**：FP和OFP都可以写成关于能量 $\Psi$ 的斜子梯度下降，差别仅在于预测对偶向量：

$$y^{t+1} = y^t + J \partial\Psi(\tilde{y}^{t+1}), \quad J = \begin{pmatrix} 0 & A \\ -A^\top & 0 \end{pmatrix}$$

其中 $\tilde{y}^{t+1} = y^t$（FP）或 $\tilde{y}^{t+1} = 2y^t - y^{t-1}$（OFP）。

    - **标准FP**：由于 $J$ 的反对称性，单步能量变化 $\Delta\Psi \geq 0$（能量不减），在 $\sqrt{T}$ 步中至少增长常数
    - **乐观FP**：当预测对偶向量和真实对偶向量映射到同一原始顶点时（$\partial\Psi(y^{t+1}) = \partial\Psi(\tilde{y}^{t+1})$），$\Delta\Psi \leq 0$（能量不增）

3. **2×2博弈的结构利用**：利用Assumption 1（WLOG可将任意2×2零和博弈变换为 $\det A = 0$, $a,d > \max\{0,b,c\}$ 的形式），使得对偶向量 $y^t$ 全部落在同一二维子空间中，从而可用二维平面几何分析OFP的轨迹行为。

### 主定理证明核心

**Theorem 3.5**（能量有界）：在2×2零和博弈中，OFP的对偶能量满足：

$$\Psi(y^{T+1}) \leq 8a_{\max}\left(1 + 2\frac{a_{\max}}{a_{\text{gap}}}\right)^2$$

其中 $a_{\max} = \|A\|_\infty$，$a_{\text{gap}} = \min_{(i,j),(k,\ell)} |A_{ij} - A_{k\ell}|$。

证明的核心思路：在对偶空间中，$y^t$ 的轨迹被限制在一个有界区域内，因为每当能量增长到某个阈值时，乐观预测会使算法"回弹"到低能量区域。

### Alternating FP的下界

**Theorem 3.2**：在Matching Pennies博弈上，对几乎所有初始化，Alternating FP的遗憾为 $\Omega(\sqrt{T})$。这分离了乐观和交替在无正则化设置下的能力：乐观可以获得 $O(1)$，但交替单独不能改善超过 $O(\sqrt{T})$。

## 实验关键数据

### 主实验：经验遗憾比较

| 博弈 | FP遗憾增长 | OFP遗憾增长 | AFP遗憾增长 |
|---|---|---|---|
| Matching Pennies (2×2) | $\sim\sqrt{T}$ | **常数** | $\sim\sqrt{T}$ |
| 15×15 单位矩阵 | $\sim\sqrt{T}$ | **常数** | $\sim\sqrt{T}$ |
| 15×15 广义石头剪刀布 | $\sim\sqrt{T}$ | **常数** | $\sim\sqrt{T}$ |

### 消融实验：2×2环境中的遗憾界总结

| 算法变体 | $\eta$ 有界 (FTRL) | $\eta \to \infty$ (FP) |
|---|---|---|
| 标准 | $O(\sqrt{T})$ | $O(\sqrt{T})$ |
| 乐观 (Optimistic) | $O(1)$ | **$O(1)$ ⭐本文** |
| 交替 (Alternating) | $O(T^{1/5})$ | **$\Omega(\sqrt{T})$ ⭐本文** |

### 关键发现

- **OFP的常数遗憾在高维博弈中经验成立**：虽然理论只证明了2×2，但在15×15博弈上OFP仍展现常数遗憾，强烈暗示结论可推广
- **交替不能替代乐观**：在无正则化的情况下，两者被严格分离（$O(1)$ vs $\Omega(\sqrt{T})$），但在有正则化时交替可以获得 $O(T^{1/5})$
- **能量函数提供了直观的几何图景**：OFP的对偶轨迹在有界区域内"振荡"但不发散，而FP和AFP的能量严格增长

## 亮点与洞察

- **首次在无正则化范围内匹配了正则化的最优率**，挑战了"正则化是O(1)遗憾的必要条件"的隐含假设
- **对偶空间的能量分析**提供了一种全新的证明技术，完全不同于标准的RVU bound方法
- **乐观vs交替的分离结果**具有理论优美性：同一框架（FP）下，两种改进策略在大步长极限下有本质不同
- 对均衡计算算法（如组合博弈中的MCCFR变体）和自博弈RL有潜在影响

## 局限与展望

- **仅限2×2零和博弈**：定理的证明深度依赖于2×2的结构（对偶向量落在二维子空间），推广到一般 $n \times n$ 是主要的未解决问题
- **依赖唯一内点Nash均衡假设**：退化博弈（如纯策略NE）不在讨论范围内
- **常数依赖于博弈参数**：$8a_{\max}(1+2a_{\max}/a_{\text{gap}})^2$ 可能在 $a_{\text{gap}}$ 很小（接近退化）时非常大
- **未讨论last-iterate收敛**：仅证明了time-average收敛，OFP的last-iterate行为仍未知
- **Alternating FP下界仅对特定初始化成立**：虽然覆盖了几乎所有无理数初始化

## 相关工作与启发

- 与连续时间Hamiltonian流的能量守恒有深刻联系：离散化导致标准FP能量增长，但乐观修正"恢复"了近似守恒
- 可能启发在extensive-form games中无正则化算法的设计
- 对理解自博弈训练中的收敛行为（如AlphaStar, OpenAI Five）有理论参考价值
- 与Frank-Wolfe优化方法中无正则化加速的工作有方法论联系

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次证明FP变体可获得O(1)遗憾，结果出人意料
- 实验充分度: ⭐⭐⭐ 实验主要是验证性的（T=10000），缺乏大规模或应用性实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导优美，直觉解释到位，表1的景观总结一目了然
- 价值: ⭐⭐⭐⭐ 对在线学习和博弈论理论有重要推进，但限于2×2限制了即时应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)
- [\[NeurIPS 2025\] Reliably Detecting Model Failures in Deployment Without Labels](reliably_detecting_model_failures_in_deployment_without_labels.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](../../ICML2026/learning_theory/on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](../../ICLR2026/learning_theory/a_faster_parameter-free_regret_matching_algorithm.md)
- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](../../ICLR2026/learning_theory/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)

</div>

<!-- RELATED:END -->

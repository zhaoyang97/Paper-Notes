---
title: >-
  [论文解读] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time
description: >-
  [ICML 2026][优化/理论][重尾噪声] 本文提出 Lévy Mirror Flow（LMF）——一种由 Lévy 噪声驱动的随机镜像下降连续时间 SDE 模型，证明即使在无穷方差的重尾梯度噪声下，SMD 仍保持收敛保证（凸情形 $O(\varepsilon^{-p/(p-1)})$，强凸情形 $\tilde{O}(\varepsilon^{-1/(p-1)})$），并将连续时间结果无缝传递到离散时间算法。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "重尾噪声"
  - "随机镜像下降"
  - "Lévy过程"
  - "收敛率"
  - "凸优化"
---

# Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time

**会议**: ICML 2026  
**arXiv**: [2606.03769](https://arxiv.org/abs/2606.03769)  
**代码**: 无  
**领域**: 优化  
**关键词**: 重尾噪声, 随机镜像下降, Lévy过程, 收敛率, 凸优化  

## 一句话总结

本文提出 Lévy Mirror Flow（LMF）——一种由 Lévy 噪声驱动的随机镜像下降连续时间 SDE 模型，证明即使在无穷方差的重尾梯度噪声下，SMD 仍保持收敛保证（凸情形 $O(\varepsilon^{-p/(p-1)})$，强凸情形 $\tilde{O}(\varepsilon^{-1/(p-1)})$），并将连续时间结果无缝传递到离散时间算法。

## 研究背景与动机

**领域现状**：随机镜像下降（SMD）及其变体是凸随机优化中最经典的一阶方法之一，核心思想是用非欧几何的 Bregman 散度替代欧氏投影，从而在约束优化中获得接近维度无关的收敛保证。现有理论分析几乎全部建立在梯度噪声轻尾（有限方差）的假设之上。

**现有痛点**：大量实证表明，深度神经网络训练中的梯度噪声呈重尾分布（$\alpha$-稳定分布），从 CNN 到 LLM 再到强化学习均有报道。当梯度噪声方差为无穷大时，标准 SGD 甚至在一维二次函数上就可能发散。现有连续时间分析（随机镜像流 SMF）仅处理布朗噪声驱动的扩散 SDE，其轨迹连续且增量高斯，完全无法刻画重尾场景下的"大跳跃"行为。

**核心矛盾**：重尾噪声导致的无穷方差使得经典 Itô 公式（依赖二阶矩有限）失效，且 Fenchel 耦合函数仅 Lipschitz 光滑而非 $C^2$，标准随机分析工具链断裂。

**本文目标**：建立一个从连续时间到离散时间的统一理论框架，严格证明 SMD 在重尾噪声下的收敛性、集中性与首达时间保证。

**切入角度**：将 SMD 的噪声源从布朗运动替换为中心化 Lévy 过程（$p$ 阶矩有限，$1 < p \le 2$），由此产生的 SDE 自然允许无穷方差和任意大小的跳跃不连续，更忠实地描述重尾训练动态。

**核心 idea**：用 Lévy 噪声驱动的镜像流（LMF）作为重尾 SMD 的连续时间代理模型，开发新的弱 Itô 公式处理非 $C^2$ 凸函数，建立透明的收敛率表征后向离散时间传递。

## 方法详解

### 整体框架

输入为凸优化问题 $\min_{x \in X} f(x)$，其中 $X$ 为紧凸集。优化器通过黑箱梯度预言机获取随机梯度 $g_t = \nabla f(x_t) + U_t$，噪声 $U_t$ 仅有 $p$ 阶矩有限（$1 < p \le 2$），方差可为无穷。方法分两层展开：(1) 连续时间层定义 LMF 并建立收敛/集中/首达时间理论；(2) 离散时间层分析 SDA、LMD、SMD 三种 SMD 变体，证明离散界可分解为"连续时间项 + 离散化项"。

### 关键设计

**1. Lévy Mirror Flow（LMF）：用 Lévy 噪声驱动的镜像流当重尾 SMD 的连续时间代理**

布朗驱动的随机镜像流 SMF 轨迹连续、增量高斯，根本刻画不了重尾噪声的"大跳跃"。LMF 的做法是把 SMF 的布朗噪声换成 Lévy 过程 $L(t)$，定义对偶空间 SDE $dY(t) = -\nabla f(X(t))dt + dL(t)$，原始迭代 $X(t) = Q(\eta(t)Y(t))$。Lévy 过程经 Lévy-Itô 分解为三部分——扩散分量 $M(t)$（布朗）、短跳分量 $S(t)$（有界跳跃）、无界跳分量 $U(t)$（$p$ 阶矩有限但方差可无穷）；相应地把噪声强度拆成 tame 部分 $\sigma^2_{\text{tame}} = \sigma^2_0 + \sigma^2_{\text{short}}$ 与 heavy 部分 $\sigma^p_{\text{heavy}} = \sigma^p_{\text{long}}$。这样做的好处是收敛率里能清晰看到轻尾与重尾各自的独立贡献，而且 LMF 恰好作为 SMD 在重尾噪声下的缩放极限自然出现，能同时容纳无穷方差和任意大跳跃。

**2. 弱 Itô 公式：为只 Lipschitz 光滑的凸函数补上 Lévy 版链式法则**

整套分析的技术核心是经典 Itô 引理在这里用不了——它要求函数二阶连续可微，而本文要分析的 Fenchel 耦合 $F(q,y) = h(q) + h^*(y) - \langle y, q \rangle$ 只是 Lipschitz 光滑、并非 $C^2$。作者的处理是先对 $F$ 做磨光（mollification），再推导一条只以不等式形式成立的"弱 Itô 公式"；处理 Lévy 跳跃带来的二阶跳跃项时，靠的是噪声的 $p$ 阶矩有限性而非二阶矩去控制无界跳跃。没有这件工具，Lévy 跳跃项就无法在能量函数 $E(t) = F(q, \eta(t)Y(t))/\eta(t)$ 的演化中被控住。据作者所知，这个结果在随机分析文献里是全新的，也是它能服务于任何后续 Lévy 驱动优化分析的原因。

**3. 连续—离散统一分析框架：把连续时间保证"加法分解"地传到离散算法**

光有连续时间理论不够，还得落到真正在跑的离散算法上。本文对 SDA（Stochastic Dual Averaging）、LMD（Lazy Mirror Descent）和 SMD 三种变体，在相对光滑性假设 $f(x') \le f(x) + \langle \nabla f(x), x'-x\rangle + LD(x',x)$ 下建立收敛率，并证明所有离散界都能分解成"连续时间项 + $[f(x_1) - \min f]$ 离散化项"，后者随迭代自然消失。之所以用相对光滑性而非标准 Lipschitz 光滑，是因为它与 Bregman 几何天然兼容，能处理梯度在约束集边界发散的情形（如 Poisson 逆问题、熵正则最优传输）——这正是标准 Lipschitz 光滑性失效的地方。

### 训练策略

本文是纯理论工作，不引入新的训练目标或算法实现。分析建立在两条核心假设上：约束集 $X$ 紧凸；梯度预言机的噪声满足鞅差条件且 $p$ 阶矩有限（$1 < p \le 2$，方差可无穷）。步长按 $\eta(t) \propto 1/t^{1/p}$（凸）或常数（强凸）选取，与重尾指数 $p$ 直接挂钩。

## 实验关键数据

### 主要理论结果对比

| 设定 | 算法 | 收敛率 | 备注 |
|------|------|--------|------|
| 凸 + 连续时间 | LMF, $\eta(t) = 1/t^{1/p}$ | $O(1/t^{(p-1)/p})$ | $p=2$ 退化为经典 $O(1/\sqrt{t})$ |
| 凸 + 离散时间 | SDA, $\eta_t = \beta/t^{1/p}$ | $O(1/T^{(p-1)/p})$ | 与连续时间速率匹配 |
| 强凸 + 连续时间 | LMF, 常数 $\eta$ | 几何收敛至 $O(\delta^2_\eta)$ 球 | $\delta^2_\eta$ 随 $\eta$ 和噪声强度缩放 |
| 强凸 + 离散时间 | SDA, 常数 $\eta$ | $\tilde{O}(\varepsilon^{-1/(p-1)})$ 达 $\varepsilon$-最优 | 比遍历率 $O(\varepsilon^{-p/(p-1)})$ 更优 |
| 强凸相对 + 离散 | LMD, $\gamma_t = \beta/t$ | $O(1/t^{p-1})$（$p < 1+\beta\mu$） | 分三段依 $p$ 与 $\beta\mu$ 关系而变 |

### 理论保证类型对比

| 保证类型 | 连续时间定理 | 离散时间定理 | 关键量 |
|----------|------------|------------|--------|
| 遍历收敛 | Theorem 1 | Theorem 5 | 时间平均 $\bar{X}(t)$ |
| 集中不等式 | Theorem 2 | Theorem 7 | 停留时间占比 $\mu_T(B_\delta)$ |
| 首达时间 | Theorem 3 | Theorem 8 | $\tau_\delta = \inf\{t: \|X(t)-x^*\| \le \delta\}$ |
| 末迭代收敛 | Theorem 4 | Theorem 6/9 | $E[\|x_t - x^*\|^2]$ |

### 关键发现

- 重尾噪声使收敛率从 $O(1/\sqrt{t})$ 退化为 $O(1/t^{(p-1)/p})$，退化程度随 $p$ 平滑变化，且完全由长跳项 $\sigma^p_{\text{heavy}}$ 控制
- 尽管 LMF 轨迹有任意大的跳跃不连续和无穷方差，SMD 仍保持收敛——Bregman 结构的约束机制有效"吸收"了长跳
- 离散时间界与连续时间界的定量匹配验证了 LMF 作为重尾 SMD 代理模型的忠实性
- 数值实验在简单二维强凸函数上验证了 $f(\bar{x}_T)$ 按幂律衰减，重尾（$\alpha = p = 3/2$）比轻尾（$\alpha = p = 2$）收敛更慢但仍收敛

## 亮点与洞察

- **弱 Itô 公式的通用性**：该工具不仅服务于本文，对任何需要分析 Lévy 驱动优化的后续工作都有直接价值——它将 Itô 随机微积分的链式法则从 $C^2$ 函数 + 布朗运动推广到 Lipschitz 凸函数 + Lévy 过程
- **噪声解耦设计**：将 Lévy 噪声拆分为 $\sigma_{\text{tame}}$（轻尾）和 $\sigma_{\text{heavy}}$（重尾）两部分，使收敛率中能独立追踪各噪声源的贡献，这种"诊断式"分析思路可迁移到自适应优化器设计中（如根据噪声尾部指数 $p$ 自动调节学习率 $\eta \propto 1/t^{1/p}$）
- **连续到离散的"加法分解"**：所有离散界 = 连续时间项 + 离散化项，提供了一种系统化的分析范式——先在连续时间中建立干净的收敛保证，再处理离散化误差

## 局限与展望

- 强凸情形的离散时间率 $\tilde{O}(\varepsilon^{-1/(p-1)})$ 未匹配 Zhang et al. 的下界 $\Omega(\varepsilon^{-p/[2(p-1)]})$，作者猜测通过更换能量函数可以改进
- 理论假设要求约束集 $X$ 紧凸，且需要梯度预言机的噪声满足鞅差条件和 $p$ 阶矩有限——对无约束或开集优化不直接适用
- 数值实验仅在简单二维函数上验证，缺少深度学习实际训练的大规模实证
- 未探索 LMF 在采样问题（如约束空间上的 Langevin 动力学）中的应用潜力

## 相关工作与启发

- Nemirovski & Yudin 的经典镜像下降理论及其最优下界 $\Omega(t^{-(p-1)/p})$
- Zhang et al. (2020) 证明 SGD 在重尾噪声下固定步长可能发散，梯度裁剪可恢复收敛
- Şimşekli (2017) 提出分数阶 Langevin Monte Carlo，用 $\alpha$-稳定 Lévy 过程建模 SGD
- Liu (2024) 对 SGD/Dual Averaging 在无界方差下建立了类似收敛率
- 启发：Lévy 过程的噪声分解思路（tame + heavy）可用于设计自适应梯度裁剪阈值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)

</div>

<!-- RELATED:END -->

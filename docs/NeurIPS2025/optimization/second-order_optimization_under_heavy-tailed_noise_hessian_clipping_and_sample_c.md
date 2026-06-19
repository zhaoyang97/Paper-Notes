---
title: >-
  [论文解读] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity
description: >-
  [NeurIPS 2025][优化/理论][二阶优化] 首次系统研究重尾噪声条件下二阶随机优化的理论基础，建立了紧的样本复杂度下界，提出了基于梯度和 Hessian 裁剪的归一化SGD算法（Clip NSGDHess），并证明其近似达到信息论极限。 重尾噪声在现代机器学习中普遍存在（数据异质性、离群值、非平稳环境）…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "二阶优化"
  - "重尾噪声"
  - "Hessian裁剪"
  - "样本复杂度"
  - "高概率收敛"
---

# Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity

**会议**: NeurIPS 2025  
**arXiv**: [2510.10690](https://arxiv.org/abs/2510.10690)  
**代码**: 无  
**领域**: 优化  
**关键词**: 二阶优化, 重尾噪声, Hessian裁剪, 样本复杂度, 高概率收敛

## 一句话总结

首次系统研究重尾噪声条件下二阶随机优化的理论基础，建立了紧的样本复杂度下界，提出了基于梯度和 Hessian 裁剪的归一化SGD算法（Clip NSGDHess），并证明其近似达到信息论极限。

## 研究背景与动机

重尾噪声在现代机器学习中普遍存在（数据异质性、离群值、非平稳环境），现有的理论和算法框架面临根本性不足：

**一阶方法已成熟**：SGD 及其变体在 $p$-BCM（有界 $p$ 阶中心矩）噪声模型下已有最优收敛保证和完备的复杂度理论，梯度裁剪（gradient clipping）已成为标准工具。

**二阶方法严重滞后**：(1) 现有二阶方法（随机立方正则化 SCN、信赖域方法等）要求有界噪声或有界方差假设，在重尾情况下缺乏保证；(2) 没有任何现有二阶方法能在无界方差噪声下可靠运行；(3) Hessian 估计比梯度更敏感于噪声，但对应的鲁棒性工具（如 Hessian 裁剪）此前从未被研究。

**核心问题**：能否设计出在梯度和 Hessian 同时受到重尾噪声污染时，仍具有高概率收敛保证的二阶算法？

## 方法详解

### 整体框架

本文工作围绕三个层次展开：(1) 建立二阶方法在 $p$-BCM 噪声下的极小极大样本复杂度下界；(2) 设计匹配下界的最优算法 NSGDHess；(3) 通过 Hessian 裁剪实现高概率收敛保证。

### 关键设计

1. **极小极大下界 (Theorem 1)**：对于 $q$ 阶零尊重算法，在 $p$-BCM 噪声（$p \in (1,2]$）下，找到 $\varepsilon$-稳定点所需的最小 oracle 查询次数为：

    $\Omega\left(\frac{\Delta \sigma_h}{\varepsilon^2} \left(\frac{\sigma}{\varepsilon}\right)^{\frac{1}{p-1}}\right)$
   
   这比一阶方法的最优复杂度优了 $1/\varepsilon$ 倍（均匀地对所有 $p \in (1,2]$ 成立），证明即便在重尾情况下二阶信息仍能提供可证明的加速。关键技巧是利用"零链"性质的最坏情况函数构造，将 $p$-BCM 噪声模型推广到高阶导数。

2. **归一化 SGD with Hessian 修正 (NSGDHess, Algorithm 1)**：核心更新规则：

    - 在连续迭代间均匀随机插值：$\hat{x}_t = q_t x_t + (1-q_t) x_{t-1}$
    - 构建 Hessian 修正的递归动量：$g_t = (1-\alpha)(g_{t-1} + \nabla^2 f(\hat{x}_t, \hat{\xi}_t)(x_t - x_{t-1})) + \alpha \nabla f(x_t, \xi_t)$
    - 归一化步长：$x_{t+1} = x_t - \gamma \frac{g_t}{\|g_t\|}$
   
   设计动机：随机插值点使得 $\mathbb{E}[\nabla^2 f(\hat{x}_t)(x_t - x_{t-1})] = \nabla F(x_t) - \nabla F(x_{t-1})$，即精确的梯度差，这是 Hessian 修正动量有效的理论基础。归一化步长保证每步更新量有界（$\|x_{t+1} - x_t\| = \gamma$），对重尾噪声至关重要。

3. **Hessian 裁剪 (Clip NSGDHess, Algorithm 2)**：在 Algorithm 1 基础上引入梯度和 Hessian-向量积的双重裁剪：
   
    $g_t = (1-\alpha)(g_{t-1} + \gamma \cdot \text{clip}(\gamma^{-1}\nabla^2 f \cdot (x_t - x_{t-1}), \bar{\lambda}_h)) + \alpha \cdot \text{clip}(\nabla f, \lambda)$
   
   其中 $\text{clip}(v, \lambda) = \min\{1, \lambda/\|v\|\} \cdot v$。**关键洞察**：直接裁剪 Hessian 矩阵需要 $O(d^2)$ 计算量（需估算算子范数），而裁剪 Hessian-向量积只需 $O(d)$ 操作，且保持了反向传播兼容性。

### 损失函数 / 训练策略

- **收敛目标**：找到 $\varepsilon$-一阶稳定点，即 $\|\nabla F(x)\| \leq \varepsilon$
- **NSGDHess 样本复杂度 (Theorem 2)**：$O\left(\frac{\Delta(L+\sigma_h)}{\varepsilon^2}\left(\frac{\sigma}{\varepsilon}\right)^{\frac{1}{p-1}}\right)$，在 $L \leq \sigma_h$ 时精确匹配下界
- **Clip NSGDHess 高概率保证 (Theorem 3)**：以概率 $\geq 1-\delta$ 达到近最优复杂度，仅有 $\text{polylog}(T/\delta)$ 的额外开销

## 实验关键数据

### 主实验

合成实验：$F(x) = 0.5\|x\|^2$, $d=10$，使用双边 Pareto 分布（尾指数 $p=1.1$）生成重尾噪声。

| 算法 | 噪声鲁棒性 | 收敛性 | 是否需要裁剪 |
|------|-----------|--------|------------|
| NSGDM (一阶归一化动量) | 差 | 振荡严重 | 否 |
| NSGDHess (无裁剪) | 中等 | 有改善但不稳定 | 否 |
| **Clip NSGDHess** | **强** | **稳定且快速** | **是** |

### 消融实验

| 配置 | 复杂度（关于 $\varepsilon$） | 高概率？ | 备注 |
|------|--------------------------|---------|------|
| 一阶 FOSO（最优） | $O(\varepsilon^{-3} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | 是 | 已知下界 |
| 二阶 NSGDHess（期望） | $O(\varepsilon^{-2} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | 否 | 本文 Thm 2 |
| **二阶 Clip NSGDHess** | $O(\varepsilon^{-2} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | **是** | **本文 Thm 3** |
| SCN (Tripuraneni 2018) | $O(\varepsilon^{-7/2})$ | 否 | 需有界噪声 |
| SGDHess (Tran 2021) | $O(\varepsilon^{-3})$ | 否 | 需有界方差 |

### 关键发现

- 二阶方法在重尾噪声下仍能比一阶方法节省 $1/\varepsilon$ 因子的样本量
- 无裁剪版本在合成实验中受噪声影响显著，验证了 Hessian 裁剪的必要性
- 该复杂度结果不依赖二阶光滑常数 $L_h$（可以为无穷），这是实际中重要的特性
- $p=2$（有界方差）特殊情况下结果也是新的，统一了此前分散的结论

## 亮点与洞察

- **首个完整的二阶优化重尾噪声理论**：从下界到匹配上界，提供了完整的极小极大刻画
- **Hessian 裁剪是全新概念**：将梯度裁剪的成功经验推广到 Hessian，且通过向量积裁剪保持了计算可行性
- **理论上证明了二阶方法在重尾噪声下的优势不会消失**：$1/\varepsilon$ 的改进是本质性的
- Hessian-向量积裁剪的公式 $\gamma \cdot \text{clip}(\gamma^{-1} \nabla^2 f \cdot v, \bar{\lambda}_h)$ 设计巧妙，使裁剪阈值 $\lambda$ 和 $\bar{\lambda}_h$ 在同一量级，便于实际调参

## 局限与展望

- 实验仅在合成问题上验证，缺乏真实机器学习任务的实证
- 算法需要多个超参数（步长 $\gamma$、动量 $\alpha$、两个裁剪阈值），实用性有待进一步简化
- 目标仅限于一阶稳定点，扩展到二阶稳定点（鞍点逃逸）是重要的开放问题
- 归一化步长在裁剪版本中是否必要、能否用纯裁剪替代归一化，值得探索

## 相关工作与启发

- **与 gradient clipping 的关系**：本文是 clipping 技术从一阶到二阶的自然推广，延续了 Gorbunov (2020), Sadiev (2023) 等工作的思路
- **与强化学习的联系**：NSGDHess 灵感来自 RL 中的 SHARP/HAR-PG 算法（Fatkhullin et al., 2023），说明优化理论与 RL 之间的交叉越来越深入
- **启发**：Hessian 裁剪思想可能对分布式优化、联邦学习中的鲁棒性设计有价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首个完整的重尾二阶优化理论，Hessian 裁剪全新）
- 实验充分度: ⭐⭐⭐ （理论贡献突出但仅有合成实验）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，层次分明，图表信息量大）
- 价值: ⭐⭐⭐⭐ （为重尾噪声下的优化提供了坚实理论基础）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](../../ICML2025/optimization/sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](../../ICML2026/optimization/can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)

</div>

<!-- RELATED:END -->

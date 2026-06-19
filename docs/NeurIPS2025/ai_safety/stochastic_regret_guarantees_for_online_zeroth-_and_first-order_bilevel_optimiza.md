---
title: >-
  [论文解读] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization
description: >-
  [NeurIPS 2025][AI安全][双层优化] 提出了一种新的搜索方向并证明基于该方向的一阶和零阶在线双层优化算法能够在不需要窗口平滑的条件下实现次线性随机双层遗憾保证，同时通过降低 oracle 依赖、并行更新和零阶 Hessian/Jacobian 估计来提升效率。 双层优化（Bilevel Optimizatio…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "双层优化"
  - "在线学习"
  - "零阶优化"
  - "随机遗憾"
  - "超梯度估计"
---

# Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2511.01126](https://arxiv.org/abs/2511.01126)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 双层优化, 在线学习, 零阶优化, 随机遗憾, 超梯度估计

## 一句话总结

提出了一种新的搜索方向并证明基于该方向的一阶和零阶在线双层优化算法能够在不需要窗口平滑的条件下实现次线性随机双层遗憾保证，同时通过降低 oracle 依赖、并行更新和零阶 Hessian/Jacobian 估计来提升效率。

## 研究背景与动机

双层优化（Bilevel Optimization）是机器学习中的重要框架，广泛应用于超参数优化、元学习、对抗训练等场景。其形式为：

$$\min_{x} f(x, y^*(x)) \quad \text{s.t.} \quad y^*(x) = \arg\min_{y} g(x, y)$$

在线双层优化（OBO）进一步要求在目标函数随时间变化的情况下进行优化，这对算法的自适应性提出了更高要求。

当前的 OBO 方法依赖于**窗口平滑遗憾**（window-smoothed regret）来衡量性能。窗口平滑遗憾通过对一个时间窗口内的函数值取平均来"平滑"目标的快速变化，但这种度量可能无法准确反映函数快速变化时的真实系统性能。此外，现有方法通常需要大量的梯度 oracle 调用，在零阶（无梯度信息）设置下效率低下。

## 方法详解

### 整体框架

本文提出了统一的在线双层优化框架，包含以下关键组件：

1. **新的搜索方向**: 设计了一种更高效的超梯度（hypergradient）近似方向
2. **一阶版本 (SOBO-FO)**: 使用一阶梯度信息的在线双层优化算法
3. **零阶版本 (SOBO-ZO)**: 仅使用函数值查询的在线双层优化算法
4. **随机遗憾分析**: 证明了不依赖窗口平滑的次线性遗憾

### 关键设计

**新的搜索方向**: 在双层优化中，外层目标的隐式梯度（或超梯度）需要计算内层问题的 Hessian 和 Jacobian。作者设计了一种近似搜索方向 $d_t$ 满足：

$$d_t \approx \nabla_x f_t - \nabla_{xy}^2 g_t \cdot [\nabla_{yy}^2 g_t]^{-1} \cdot \nabla_y f_t$$

关键改进在于将线性系统 $[\nabla_{yy}^2 g_t] v = \nabla_y f_t$ 的求解与内外层变量的更新**并行进行**，而非像先前方法那样需要先完全求解内层后再更新外层。

**降低 oracle 复杂度**: 通过以下方式减少梯度 oracle 的调用次数：
- 在每次迭代中同时更新内层变量 $y_t$、外层变量 $x_t$ 和线性系统辅助变量 $v_t$
- 避免在每步完全求解内层优化问题

**零阶（ZO）方法**: 当无法获取梯度信息时（如黑箱攻击场景），使用随机有限差分来估计：
- 梯度: $\hat{\nabla} f(x) \approx \frac{d}{\delta} [f(x + \delta u) - f(x)] u$，其中 $u$ 是随机方向
- Hessian-向量积和 Jacobian-向量积同样通过有限差分估计

### 损失函数 / 训练策略

本文是优化理论工作，不涉及特定损失函数设计。算法的遗憾度量为随机双层遗憾：

$$\text{Regret}_T = \sum_{t=1}^{T} \mathbb{E}[f_t(x_t, y^*(x_t))] - \min_{x \in \mathcal{X}} \sum_{t=1}^{T} f_t(x, y^*(x))$$

**理论结果**:
- 一阶算法: $\text{Regret}_T = O(\sqrt{T})$
- 零阶算法: $\text{Regret}_T = O(d^{3/2}\sqrt{T})$，其中 $d$ 是问题维度

## 实验关键数据

### 主实验

**实验一: 在线参数化损失调优（Online Parametric Loss Tuning）**

在 MNIST 和 CIFAR-10 上进行在线超参数调优，外层优化验证损失，内层优化训练损失。

| 方法 | MNIST 遗憾 (T=500) | MNIST 遗憾 (T=1000) | CIFAR-10 遗憾 (T=500) | CIFAR-10 遗憾 (T=1000) |
|------|-------------------|--------------------|-----------------------|----------------------|
| SOBO-FO (本文) | **12.3** | **18.7** | **25.6** | **38.2** |
| OBO-FO (窗口平滑) | 15.8 | 25.4 | 31.2 | 49.5 |
| 在线 SGD | 22.1 | 38.6 | 42.8 | 68.3 |
| SOBO-ZO (本文) | 16.5 | 24.1 | 30.8 | 45.7 |

**实验二: 黑箱对抗攻击（Black-box Adversarial Attacks）**

在 CIFAR-10 分类器上使用零阶方法生成对抗样本。

| 方法 | 攻击成功率 ↑ | 查询次数 ↓ | $L_\infty$ 扰动 |
|------|------------|-----------|----------------|
| SOBO-ZO (本文) | **92.3%** | **850** | 8/255 |
| ZO-BiAdam | 89.7% | 1200 | 8/255 |
| SimBA | 85.4% | 1500 | 8/255 |
| Sign-OPT | 87.1% | 1100 | 8/255 |

### 消融实验

**Oracle 调用次数对比（每次迭代）**:

| 方法 | 梯度 oracle | Hessian oracle | Jacobian oracle | 总计 |
|------|------------|---------------|----------------|------|
| SOBO-FO (本文) | 3 | 1 | 1 | **5** |
| 先前 FO 方法 | 3 | K (内层迭代数) | K | **2K+3** |
| SOBO-ZO (本文) | O(d) | O(d) | O(d) | **O(d)** |
| 先前 ZO 方法 | O(d) | O(Kd) | O(Kd) | **O(Kd)** |

本文方法将内层求解与外层更新并行化，显著减少了 oracle 依赖。

### 关键发现

1. **窗口平滑非必需**: 首次证明 OBO 可以在不使用窗口平滑的情况下获得次线性随机遗憾
2. **效率大幅提升**: Oracle 调用从 $O(K)$ 降低到 $O(1)$（一阶）或从 $O(Kd)$ 降低到 $O(d)$（零阶）
3. **零阶方法实用**: SOBO-ZO 在黑箱设置下的表现接近一阶方法
4. **理论与实验一致**: 遗憾的收敛趋势与理论预测的 $O(\sqrt{T})$ 吻合

## 亮点与洞察

- **理论贡献扎实**: 88 页的论文包含详尽的理论分析，从新的搜索方向设计到完整的遗憾界证明
- **实用的零阶版本**: 零阶方法使该框架可应用于黑箱优化场景（如对抗攻击）
- **并行化思想**: 将线性系统求解与变量更新并行化是提升效率的关键
- **统一框架**: 一阶和零阶算法在同一理论框架下分析

## 局限与展望

1. **强凸性假设**: 对内层问题要求强凸性，限制了应用范围（如神经网络训练通常非凸）
2. **实验规模较小**: 仅在 MNIST/CIFAR-10 上验证，缺乏大规模场景的测试
3. **常数因子**: $O(\sqrt{T})$ 的常数项可能较大，实际收敛速度取决于问题条件数
4. **与实际超参数优化工具的对比**: 缺乏与 Optuna、Ray Tune 等实际工具的对比
5. **动态环境的特化**: 当函数变化模式已知（如周期性）时，是否可以利用这些信息

## 相关工作与启发

- **在线双层优化**: 本文的直接前驱工作，特别是窗口平滑遗憾方法
- **双层优化**: Franceschi et al. (2018), Ghadimi & Wang (2018) 等经典方法
- **零阶优化**: Nesterov & Spokoiny (2017), 随机有限差分方法
- **在线学习**: Hazan (2016), 在线凸优化理论
- **超参数优化**: 元学习中的双层优化应用

## 评分

- **创新性**: 4/5 — 新的搜索方向和去除窗口平滑的遗憾保证
- **技术质量**: 5/5 — 理论分析严谨完整
- **表达质量**: 3/5 — 88 页论文，信息密度高但篇幅过长
- **实用性**: 3/5 — 理论意义大于实际应用
- **综合评分**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](fairness-regularized_online_optimization_with_switching_costs.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](../../ICML2026/ai_safety/privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[NeurIPS 2025\] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization](dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)
- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)

</div>

<!-- RELATED:END -->

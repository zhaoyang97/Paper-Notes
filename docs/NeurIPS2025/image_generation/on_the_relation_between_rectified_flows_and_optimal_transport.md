---
title: >-
  [论文解读] On the Relation between Rectified Flows and Optimal Transport
description: >-
  [NeurIPS 2025][图像生成][Rectified Flow] 本文深入研究了 rectified flow（流匹配）与最优传输之间的理论关系，通过构造多个反例证明了此前文献中关于"梯度约束的 rectified flow 可以渐近收敛到最优传输"的等价性声明并不成立，需要比已知条件更强的假设才能保证两者的等价关系。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "Rectified Flow"
  - "最优传输"
  - "Flow Matching"
  - "反例构造"
  - "Wasserstein距离"
---

# On the Relation between Rectified Flows and Optimal Transport

**会议**: NeurIPS 2025  
**arXiv**: [2505.19712](https://arxiv.org/abs/2505.19712)  
**代码**: 暂无  
**领域**: 扩散模型 / 流匹配 / 最优传输  
**关键词**: Rectified Flow, 最优传输, Flow Matching, 反例构造, Wasserstein距离

## 一句话总结

本文深入研究了 rectified flow（流匹配）与最优传输之间的理论关系，通过构造多个反例证明了此前文献中关于"梯度约束的 rectified flow 可以渐近收敛到最优传输"的等价性声明并不成立，需要比已知条件更强的假设才能保证两者的等价关系。

## 研究背景与动机

最优传输（Optimal Transport, OT）是概率分布之间的经典度量和变换工具，在机器学习的聚类、域自适应、生成建模中有广泛应用。Benamou-Brenier 的动态公式将最优传输表述为寻找最小 $L^2$ 范数的速度场，使得源分布沿连续性方程传输到目标分布。

Flow matching / rectified flow 是近年来兴起的生成建模方法，其核心思想是从任意的耦合出发，对 $X_0$ 和 $X_1$ 的线性插值进行条件期望投影来学习速度场。迭代 rectification 可以拉直传输路径，降低传输代价。Liu (2022) 声称，如果对速度场施加梯度约束（即 $v_t = \nabla \varphi_t$），那么 rectified flow 的不动点就等价于最优传输耦合。

**核心矛盾**：这一等价性声明在实际中被广泛引用以证明 rectified flow 是计算最优传输的可靠途径，但本文作者发现该声明缺少关键假设，在多种情况下并不成立。

**本文切入角度**：通过严格的数学构造，给出了两类反例——断开支撑集和不可rectify的耦合，证明了梯度约束下的 rectified flow 不动点并不一定是最优传输映射，从而澄清了理论边界。

## 方法详解

### 整体框架

本文从理论分析角度出发，研究 rectified flow 的三个层面：(1) 无约束 rectification 的仿射不变性和高斯情形解析解；(2) 梯度约束下最优速度场的存在性；(3) 梯度约束 rectification 不动点与最优传输之间关系的反例。

### 关键设计

1. **仿射不变性分析（Theorem 2）**：证明了 rectification 步骤 $\mathcal{R}$ 在仿射变换下的等变性。具体而言：

    - 若 $(Z_0, Z_1) = \mathcal{R}(X_0, X_1)$，则 $\mathcal{R}(AX_0+b, AX_1+b) = (AZ_0+b, AZ_1+b)$
    - 对目标分布的平移和缩放也满足类似的等变性
    - **设计动机**：这些性质与最优传输的不变性相似，但仿射变换部分(i)在使用梯度约束后不再成立，暗示了两种方法的根本差异

2. **高斯情形解析解（Theorem 3）**：当 $(X_0, X_1)$ 服从联合高斯分布时，最优速度场有解析形式：
    $v_t(x) = \frac{1}{1-t}\left(((1-t)\Sigma_{01} + t\Sigma_1)\Sigma_t^{-1} - \text{Id}\right)x$
   其中 $\Sigma_t = (1-t)^2\Sigma_0 + (1-t)t(\Sigma_{01}+\Sigma_{10}) + t^2\Sigma_1$。特别地，当 $\Sigma_0$ 和 $\Sigma_1$ 可联合对角化时，一步 rectification 就能得到最优耦合。对高斯混合模型也给出了显式速度场（Theorem 5）。

3. **梯度约束速度场的弱解存在性（Proposition 8）**：证明了约束问题 $\min_{w_t = \nabla\varphi_t} \mathcal{L}(w_t | X_0, X_1)$ 的解总是以弱形式存在的。关键结论：

    - 梯度约束下的最优速度场 $v_t^p$ 是无约束解 $v_t$ 在空间 $T_{\mu_t}$ 上的正交投影
    - $v_t^p$ 是连续性方程的最小范数解
    - 当 $X_0 \sim \mathcal{N}(0, I_d)$ 且独立耦合时，无约束解本身就是梯度场（Corollary 9），两个问题的解一致

4. **反例一：断开支撑集（Section 4.1, Proposition 10）**：构造了一个 $\mu_0, \mu_1$ 的例子，其中支撑集由两个分离区域组成。在每个子域上分别最优但全局不最优的耦合 $(\tilde{X}_0, \tilde{X}_1)$ 是 $\mathcal{R}_p$ 的不动点且损失为零，但其传输代价 $\mathbb{E}[\|\tilde{X}_1 - \tilde{X}_0\|^2] > \mathbb{E}[\|X_1 - X_0\|^2]$，证明了原始等价性声明 (8) 不成立。

5. **反例二：不可 rectify 的耦合（Section 4.2, Proposition 13）**：设 $X_0 = -X_1 \sim \mathcal{N}(0, I_d)$，则速度场 $v_t(x) = -\frac{2}{1-2t}x$，ODE 在 $t=1/2$ 处奇异，解不唯一。该耦合达到零损失但并非最优。进一步证明了（Corollary 17）即使损失任意小且 $W_2$ 距离接近最优，传输代价仍可保持远离最优。

### 修正条件与噪声注入

作者明确了保证等价性的修正条件（Theorem 11）：需额外假设 $\text{supp}(X_t) = \mathbb{R}^d$（完全支撑）。同时提出了**平滑 rectification**（Section 5）：在每一步 rectification 中注入少量高斯噪声 $X_0^{(i)} = \sqrt{1-c_i} Z_0^{(i)} + \sqrt{c_i} W^{(i)}$，确保耦合始终可 rectify（Theorem 14），同时保持损失函数的单调递减性质。

## 实验关键数据

### 主实验：反例数值验证

| 配置 | 传输代价 $\mathbb{E}[\|X_1-X_0\|^2]$ | 损失 $\mathcal{L}$ | 是否最优 |
|------|------|------|------|
| 最优耦合 $(X_0,X_1)$（竖直传输） | 4 | 0 | ✓ |
| 非最优不动点 $(\tilde{X}_0,\tilde{X}_1)$（水平传输） | 16 | 0 | ✗ |
| 不可 rectify 耦合 $X_1=-X_0$ | 4 | 0 | ✗ |

### 高斯情形解析验证

| 设置 | 一步 rectification 后是否最优 | 条件 |
|------|------|------|
| $\Sigma_0, \Sigma_1$ 可联合对角化 | ✓ | Theorem 3(ii) |
| $\Sigma_0, \Sigma_1$ 不可联合对角化 | ✗ | 仿射不变性(i)失效 |
| 一维情形 | ✓（任意可rectify耦合） | Proposition 4 |

### 关键发现

- 断开支撑集在实际数据集中非常常见，极大限制了 rectified flow 计算最优传输的适用性
- 在 $t=1/2$ 时插值分布 $\mu_{1/2}$ 可能退化（如反例二中所有路径交叉于原点），导致速度场不唯一
- 熵正则化（DSBM）虽然保证收敛但当正则化参数趋于零时收敛速率可能任意慢

## 亮点与洞察

- 这是一篇纯理论工作，构造精巧。反例一利用支撑集断开使得局部最优不等于全局最优；反例二利用所有路径穿越同一点造成ODE不唯一
- Proposition 8 建立了梯度约束解与连续性方程最小范数解的等价关系，比直接讨论梯度场存在性更深刻
- 噪声注入平滑化是一个实用的修复方案，保持了理论保证同时只引入可控误差

## 局限与展望

- 反例主要在低维空间（$\mathbb{R}^2$）中构造，高维情况下是否存在更隐蔽的陷阱尚不清楚
- 对于实际训练中使用神经网络近似速度场的情况，泛化误差可能自然提供某种正则化，使问题不那么严重
- 平滑 rectification 的噪声量 $c_i$ 如何选择缺乏实用指导

## 相关工作与启发

- 与 mini-batch OT 初始化 rectified flow 的方法（PBDALC2023, TFMH2024）互补，后者也不保证全局最优但在实践中有效
- 与 Schrödinger bridge 方法（DSBM）的关系：DSBM 对应熵正则化 OT，收敛有保证但正则化趋零时可能收敛慢
- 本文的修正定理（Theorem 11）可指导未来 rectified flow 训练中监控支撑集连通性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 通过精心构造的反例推翻了已发表的等价性定理，理论贡献突出
- **实验充分度**: ⭐⭐⭐ 以理论证明为主，数值验证在附录中提供，缺乏大规模实验
- **写作质量**: ⭐⭐⭐⭐⭐ 数学严谨且表述清晰，反例构造直观易懂
- **价值**: ⭐⭐⭐⭐ 纠正了文献中的重要错误，对 rectified flow 理论有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](../../ICML2026/image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)

</div>

<!-- RELATED:END -->

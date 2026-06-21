---
title: >-
  [论文解读] The Price of Robustness: Stable Classifiers Need Overparameterization
description: >-
  [ICLR 2026][学习理论][过参数化] 建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。 - 过参数化的悖论：经典学习理论认为参数越多过拟合越严重…
tags:
  - "ICLR 2026"
  - "学习理论"
  - "泛化理论"
  - "过参数化"
  - "鲁棒性"
  - "稳定性"
  - "分类器"
  - "泛化界"
  - "margin"
---

# The Price of Robustness: Stable Classifiers Need Overparameterization

**会议**: ICLR 2026  
**arXiv**: [2603.02806](https://arxiv.org/abs/2603.02806)  
**代码**: 无  
**领域**: 学习理论 / 泛化理论  
**关键词**: 过参数化, 鲁棒性, 稳定性, 分类器, 泛化界, margin

## 一句话总结

建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。

## 研究背景与动机

- **过参数化的悖论**：经典学习理论认为参数越多过拟合越严重，但现代神经网络在过参数化时反而泛化更好（双下降现象）
- **Bubeck & Sellke 2021 的鲁棒性定律**：证明了回归场景中 Lipschitz 连续函数的平滑性-过参数化权衡。但该结果依赖 Lipschitz 假设，不适用于分类器（离散输出，天然不连续）
- **经验发现**：大规模研究（Jiang et al. 2019）显示 40+ 复杂度度量中，与泛化最一致相关的是 **margin**（到决策边界的距离），而非范数类度量
- **核心问题**：如何为不连续分类器建立稳定性驱动的泛化理论？

## 方法详解

### 整体框架

这是一篇纯理论论文，整体围绕一条逻辑推导链展开：先用"期望 margin"把分类器的鲁棒性量化成一个可分析的标量——类稳定性 $S(f)$；再借助数据分布的等周（isoperimetry）假设，把这个量塞进 Rademacher 复杂度的上界，证明稳定性会压低模型类的有效复杂度，并反向读出"鲁棒性代价定律"：要让一个能插值训练集的分类器同时保持高稳定性，参数量必须从 $p \approx n$ 抬到 $p \approx nd$ 量级；最后再把这套从有限函数类得到的结论，借助得分函数的 Lipschitz 连续性推广到神经网络这类无限函数类。

> 框架图：跳过。本文是纯定理/证明分析，"框架"是一条数学推导链（定义稳定性度量 → 等周假设 → 证 Rademacher 界 → 反推代价定律 → 推广到无限类），不存在多模块/多阶段的数据流 pipeline，画 flowchart 反而把不等式推导硬塞进方框、失真。下面的关键设计已按推导顺序逐步交代，与整体框架同序同名。

### 关键设计

**1. 类稳定性：用期望 margin 把"不连续"变成可分析的标量**

分类器输出离散标签、天然不连续，没法直接套用回归里的 Lipschitz 平滑性框架，于是作者改用样本到决策边界的距离来刻画鲁棒性。对分类器 $f: \mathcal{X} \to \{-1, 1\}$，先定义无符号 margin 为样本到最近异类点的欧氏距离 $h_f(x) := |d_f(x)| = \inf\{\|x - z\|_2 : f(z) \neq f(x), z \in \mathcal{X}\}$（定义 1），它度量了把 $x$ 推过决策边界所需的最小扰动；再取它在数据分布下的期望，得到类稳定性 $S(f) := \mathbb{E}[h_f]$。$S(f)$ 越大，说明平均而言要更大的扰动才能翻转预测，正好对应"分类器对输入扰动的平均鲁棒性"。和 Sokolic 等人用最小鲁棒半径不同，这里取的是**期望**（平均距离）而非最小值或经验分位数，它是一个连续标量，可以干净地进入后续的复杂度分析。

**2. 等周假设：把高维集中现象注入泛化界**

要让 margin 真正约束泛化，必须刻画数据在高维空间里的几何，否则稳定的函数仍可能任意拟合随机标签。作者假设分布 $\mu$ 满足 $c$-等周性：对任意有界 $L$-Lipschitz 函数 $f$ 和 $t \geq 0$，有 $\mathbb{P}(|f(x) - \mathbb{E}[f]| \geq t) \leq 2 e^{-\frac{dt^2}{2cL^2}}$（定义 3）。高斯分布、正曲率流形上的均匀测度（如球面）等都满足这一条件，它表达的是高维测度集中——函数值以指数速度集中在均值附近，而集中速率由维度 $d$ 控制。根据流形假设，这里的 $d$ 可解释为数据的内在流形维度而非环境维度，这也是后面 $nd$ 量级里那个 $d$ 的来源。

**3. 稳定性压低复杂度，反推出鲁棒性代价定律**

这是整套理论的技术核心，分两步走。第一步证明稳定性能直接削减复杂度：在 $\min_{f \in \mathcal{F}} S(f) > S > 0$ 且 $\log|\mathcal{F}| \geq n$ 的条件下，定理 4 通过"Lipschitz 代理 + 等周"给出 $\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_1 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S} \cdot \frac{\log|\mathcal{F}|}{n\sqrt{d}}\right\}$，正则条件下进一步改进为 $\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_2 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S}\sqrt{\frac{\log|\mathcal{F}|}{nd}}, 2\exp\left(-\frac{dS^2}{8c}\right)\right\}$。关键在于 $1/S$ 出现在 $\sqrt{\log|\mathcal{F}|}$ 前面——稳定性越高，复杂度项被压得越低，于是一个表面上很大的模型类，只要其中的函数都足够稳定，有效复杂度就显著下降（注意离散化下 $\log|\mathcal{F}| \in \mathcal{O}(p)$，因此 $\log|\mathcal{F}|$ 就代表参数量 $p$）。第二步把这个界**反向使用**逼出参数量下界：令 $p := \log|\mathcal{F}| \geq n$，推论 6 给出 $\hat{R}_{\text{0-1}}(f) \leq R^* - \varepsilon \implies S(f) < \max\left\{\frac{3K}{\varepsilon}\sqrt{\frac{c\log|\mathcal{F}|}{nd}}, \sqrt{\frac{8c}{d}\log\frac{6K}{\varepsilon}}\right\}$。这个蕴含式直接读出代价：当 $p \approx n$ 时右端的稳定性上界被压得很小，任何能插值训练集的分类器都注定不稳定；要同时拿到低训练误差和高稳定性，唯一出路是把参数量抬到 $p \approx nd$ 量级——这正是标题"鲁棒性需要过参数化"的精确表述。

**4. 推广到无限函数类：用归一化 co-stability 接住神经网络**

上面的结论建立在有限函数类（$\log|\mathcal{F}|$ 有限）之上，而神经网络是连续参数化的无限类，需要一座桥；而且仅有类稳定性还不够——例子 9 说明高 $S(f)$ 也阻止不了得分函数本身的不连续。作者于是把分类器写成 $f = \text{sgn} \circ g_w$，转而在**输出空间（codomain）的 margin** 上做文章：对得分函数 $g_w$ 的输出 margin 做归一化，得到 co-margin 及其期望——**归一化 co-stability**（定义 10），它是比类稳定性更强的鲁棒性度量；再结合 $g_w$ 在参数 $w$ 和输入 $x$ 上的 Lipschitz 连续性控制住无限类的复杂度，从而推出对应的泛化界（定理 13）和鲁棒性定律（推论 15）。值得注意的是，这里只要求得分函数 $g_w$ 连续，而最终分类器 $\text{sgn} \circ g_w$ 仍可不连续，这让框架能覆盖量化网络、脉冲网络乃至 self-attention 这类不满足整体 Lipschitz 性的模型。

## 实验结果

### MNIST 实验

| 网络宽度 | 测试准确率 | 类稳定性 $S(f)$ | 谱范数 |
|---------|-----------|----------------|--------|
| 小 | 较低 | 低 | 无规律 |
| 中 | 中等 | 中 | 无规律 |
| 大 | 高 | **高** | 无规律 |

### CIFAR-10 实验

| 网络宽度 | 测试准确率 | 归一化 co-stability | 谱范数 |
|---------|-----------|-------------------|--------|
| 窄 | ~70% | 低 | 变化不一致 |
| 宽 | ~85% | **高** | 变化不一致 |
| 更宽 | ~90% | **更高** | 变化不一致 |

### 关键发现

- 稳定性和归一化 co-stability 随网络宽度**单调增加**
- 稳定性与测试性能呈**正相关**
- 传统范数度量（谱范数等）与泛化**无系统性关联**
- 验证了理论预测：过参数化→高稳定性→好泛化

## 亮点与洞察

1. **扩展到不连续函数**：首次将鲁棒性定律从 Lipschitz 回归推广到不连续分类器
2. **0-1 损失的直接分析**：不需要 Lipschitz 损失假设
3. **解释过参数化**：过参数化不是过拟合的源头，而是实现鲁棒性的必要条件
4. **适用广泛**：覆盖量化神经网络、脉冲神经网络等天然不连续模型
5. **Transformer 的特殊意义**：self-attention 通常不是 Lipschitz 连续的，本框架比 Lipschitz 框架更适用

## 局限性

- 等周假设对某些数据分布可能不成立
- 有限函数类到无限类的推广需要额外的参数 Lipschitz 假设
- 泛化界可能仍然是 vacuous 的（与 Nagarajan & Kolter 2021 的批评一致）
- 理论维度 $d$ 的实际值难以精确估计（外在维度 vs 内在维度）
- 未建立与优化动力学（如 implicit bias）的直接联系

## 相关工作

- **鲁棒性定律**：Bubeck & Sellke 2021（Lipschitz 回归）
- **Margin 泛化界**：Bartlett et al. 2017（谱归一化 margin）
- **算法稳定性**：Bousquet & Elisseeff 2002
- **双下降**：Belkin et al. 2019

## 评分

- **创新性**: ⭐⭐⭐⭐ — 将鲁棒性定律推广到分类的自然且重要的扩展
- **技术深度**: ⭐⭐⭐⭐⭐ — 证明技巧精巧，从 Lipschitz 代理到 signed distance 表示
- **实验充分性**: ⭐⭐⭐ — MNIST 和 CIFAR-10 验证，定性为主
- **实用价值**: ⭐⭐⭐⭐ — 为理解过参数化泛化提供理论支撑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Stable Coresets: Unleashing the Power of Uniform Sampling](stable_coresets_unleashing_the_power_of_uniform_sampling.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Price of Quality: Sufficient Conditions for Sparse Recovery using Mixed-Quality Data](price_of_quality_sufficient_conditions_for_sparse_recovery_using_mixed-quality_d.md)
- [\[ICLR 2026\] Revenue Maximization under Sequential Price Competition via the Estimation of s-Concave Demand Functions](revenue_maximization_under_sequential_price_competition_via_the_estimation_of_s-.md)

</div>

<!-- RELATED:END -->

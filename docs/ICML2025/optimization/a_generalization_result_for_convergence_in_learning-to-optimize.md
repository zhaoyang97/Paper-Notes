---
title: >-
  [论文解读] A Generalization Result for Convergence in Learning-to-Optimize
description: >-
  [ICML 2025 Spotlight][优化/理论][Learning-to-Optimize] 提出一个概率框架，将 PAC-Bayesian 泛化理论与变分分析中的 Kurdyka-Łojasiewicz (KL) 收敛定理相结合，首次在不限制学习算法设计的前提下，以高概率证明了学习型优化算法收敛到临界点。
tags:
  - "ICML 2025 Spotlight"
  - "优化/理论"
  - "Learning-to-Optimize"
  - "PAC-Bayesian"
  - "收敛性保证"
  - "Kurdyka-Łojasiewicz"
  - "非光滑非凸优化"
---

# A Generalization Result for Convergence in Learning-to-Optimize

**会议**: ICML 2025 Spotlight  
**arXiv**: [2410.07704](https://arxiv.org/abs/2410.07704)  
**代码**: [GitHub](https://github.com/MichiSucker/COLA_2024)  
**领域**: 优化  
**关键词**: Learning-to-Optimize, PAC-Bayesian, 收敛性保证, Kurdyka-Łojasiewicz, 非光滑非凸优化

## 一句话总结

提出一个概率框架，将 PAC-Bayesian 泛化理论与变分分析中的 Kurdyka-Łojasiewicz (KL) 收敛定理相结合，首次在不限制学习算法设计的前提下，以高概率证明了学习型优化算法收敛到临界点。

## 研究背景与动机

Learning-to-Optimize (L2O) 利用机器学习来加速优化算法，在实践中展现了远超经典优化算法的性能提升。然而，一个长期存在的核心问题是：**学习到的优化算法缺乏理论收敛性保证**。

传统优化中的收敛性证明依赖于几何论证（如充分下降条件、相对误差条件），这些论证对手工设计的算法有效，但无法直接应用于学习到的算法。原因在于：

**问题实例是函数，无法全局观测**：训练期间探索的区域受初始化和最大迭代次数的限制

**极限和数学归纳法受阻**：由于只能观测有限步迭代，无法直接使用极限概念来证明收敛性

**收敛是尾事件**：收敛不依赖于任何有限数量的迭代步，因此无法直接在训练中观测

现有的解决方案主要依赖"安全保护机制"(safeguards)——限制更新步骤使其满足特定几何性质（如下降方向约束），从而可以像分析手工算法一样分析学习算法。但这种做法**严重限制了学习算法的设计空间和性能**，使其退化为接近手工算法的表现。

本文转换视角：**利用训练期间可以观测算法行为这一独特优势**，通过泛化理论来推导收敛性，而非直接迁移传统收敛结果。

## 方法详解

### 整体框架

核心思路可以概括为一个简洁的逻辑链：

1. **确定可观测的充分条件**：找到一组在训练中可检验的性质 $a$ 和 $b$，它们蕴含目标性质 $c$（收敛到临界点），即 $a \wedge b \Rightarrow c$
2. **概率化处理**：在概率空间中，将上述蕴含关系转化为集合包含 $\mathsf{A} \cap \mathsf{B} \subset \mathsf{C}$，从而得到 $\mu\{\mathsf{A} \cap \mathsf{B}\} \leq \mu\{\mathsf{C}\}$
3. **泛化到未见问题**：通过 PAC-Bayesian 定理，确保训练集上对 $\mu\{\mathsf{A} \cap \mathsf{B}\}$ 的估计可以泛化到新问题

具体而言，本文结合了两个核心理论工具：

- **Attouch et al. (2013) 的抽象收敛定理 (Theorem 6.5)**：在满足充分下降条件、相对误差条件和有界性条件时，对 KL 函数，迭代序列收敛到临界点
- **Sucker & Ochs (2024) 的 PAC-Bayesian 泛化定理 (Theorem 6.3)**：轨迹的可测性质可以在 PAC-Bayesian 框架下泛化

### 关键设计

#### 问题设定

考虑参数化损失函数 $\ell: \mathscr{Z} \times \mathscr{P} \to [0, \infty]$，其中 $\mathscr{Z} = \mathbb{R}^d$ 为优化空间，$\mathscr{P} = \mathbb{R}^q$ 为参数空间。算法更新规则为：

$$z^{(t+1)} = \mathcal{A}(h, p, z^{(t)}, r^{(t+1)})$$

其中 $h \in \mathscr{H}$ 是超参数（学习得到），$p \in \mathscr{P}$ 是问题参数，$r^{(t+1)}$ 是算法内部随机性。

**关键观察**：当 $h$ 和 $p$ 固定时，上述迭代可以被视为 Markov 过程的函数方程，从而在轨迹空间 $\mathscr{Z}^{\mathbb{N}_0}$ 上定义了一个概率分布。

#### 三个可测集的构造

本文的技术核心是将收敛定理所需的三个条件构造为可测集：

**1. 充分下降集 $\mathsf{A}_{\mathrm{desc}}$**：序列满足 $\ell(z^{(t+1)}, p) + a\|z^{(t+1)} - z^{(t)}\|^2 \leq \ell(z^{(t)}, p)$，即每一步迭代都严格减少损失函数值，减少量至少与步长的平方成正比。

**2. 相对误差集 $\mathsf{A}_{\mathrm{err}}$**：存在次梯度选择 $v(z,p) \in \partial_1 \ell(z,p)$ 满足 $\|v(z^{(t+1)}, p)\| \leq b\|z^{(t+1)} - z^{(t)}\|$，即次梯度的大小被步长控制。

**3. 有界集 $\mathsf{A}_{\mathrm{bound}}$**：序列 $(z^{(t)})$ 保持有界，$\|z^{(t)}\| \leq c$。

证明这三个集合的可测性是高度非平凡的（特别是 $\mathsf{A}_{\mathrm{conv}}$ 和 $\mathsf{A}_{\mathrm{desc}}$），因为需要处理无穷序列空间上的乘积 $\sigma$-代数。核心技巧是利用 Polish 空间的可分性和完备性，将这些集合表示为可测集的可数交/并。

#### 核心推论 (Corollary 7.5)

在 KL 函数假设下：

$$\mathsf{A}_{\mathrm{desc}} \cap \mathsf{A}_{\mathrm{err}} \cap \mathsf{A}_{\mathrm{bound}} \subset \mathsf{A}_{\mathrm{conv}}$$

即，满足充分下降、相对误差和有界性的序列必然收敛到临界点。

#### 主定理 (Theorem 7.6)

结合 PAC-Bayesian 泛化定理，记 $\mathsf{A} = \mathsf{A}_{\mathrm{desc}} \cap \mathsf{A}_{\mathrm{err}} \cap \mathsf{A}_{\mathrm{bound}}$，则以概率至少 $1 - \varepsilon$：

$$\rho[\mathbb{P}_{(P,\xi)|H}\{\mathsf{A}_{\mathrm{conv}}\}] \geq 1 - \Phi_{\lambda/N}^{-1}\left(\frac{1}{N}\sum_{n=1}^N \rho[\mathbb{P}_{(P_n,\xi_n)|H,P_n}\{\mathsf{A}^c\}] + \frac{D_{\mathrm{KL}}(\rho \| \mathbb{P}_H) + \log(1/\varepsilon)}{\lambda}\right)$$

其中 $\Phi_a^{-1}(p) = \frac{1 - \exp(-ap)}{1 - \exp(-a)}$。

直觉解读：当训练集 $N$ 足够大时，bound 的第二项被 $\lambda$ 压制，bound 约等于训练集上满足条件 $\mathsf{A}$ 的经验概率加上一个正则项。

### 损失函数 / 训练策略

算法 $\mathcal{A}$ 在 i.i.d. 问题参数数据集 $S = (P_1, \ldots, P_N)$ 上训练。训练目标是优化超参数 $h$ 使得算法在给定问题族上表现优异。

**实践中的检验策略**：

- 充分下降条件在有限迭代步 $t_{\mathrm{train}}$ 内逐步检验
- 相对误差条件通过计算 $b = \max_t \|v(z^{(t)}, p)\| / \min_t \|z^{(t)} - z^{(t-1)}\|$ 直接满足
- 有界性条件在强凸问题中自动满足，在非凸问题中需要额外检验

值得强调的是：**学习算法的更新步骤不受任何约束**，这是本文方法相比 safeguard 方法的核心优势。

## 实验关键数据

### 主实验

#### 实验一：二次问题 (Quadratic Problems)

| 配置 | 指标 | 学习算法 | HBF (基线) | 说明 |
|------|------|---------|-----------|------|
| $d=200$, $N=250$ | 收敛速度 | ~100 步达 $10^{-16}$ | ~500 步仍未达 | 学习算法显著更快 |
| 250 个测试集 | PAC bound 对 $\mathbb{P}\{\mathsf{A}\}$ | ~75% | - | 保证 75% 问题收敛 |
| 250 个测试集 | $\mathbb{P}\{\mathsf{A}_{\mathrm{conv}}\}$ 估计 | ~95% | - | 实际收敛概率更高 |

- 优化变量 $z \in \mathbb{R}^{200}$，损失为 $\frac{1}{2}\|Az - b\|^2$
- 强凸常数和光滑常数在区间 $[m_-, m_+]$, $[L_-, L_+]$ 中随机采样
- 训练迭代次数 $t_{\mathrm{train}} = 500$

#### 实验二：神经网络训练

| 配置 | 指标 | 学习算法 | Adam (基线) | 说明 |
|------|------|---------|-----------|------|
| $d=351$, $K=50$ | 损失下降速度 | 更快 | 较慢 | 学习算法优于 Adam |
| 250 个测试集 | PAC bound | ~92% | - | 保证 92% 问题收敛 |
| 250 个测试集 | $\mathbb{P}\{\mathsf{A}\}$ 估计 | ~94% | - | bound 非常紧 |

- 训练一个两层全连接 ReLU 网络做回归
- 使用 MSE 损失（对参数 $\beta$ 非光滑非凸）
- Adam 步长通过 100 个值的网格搜索调优（$\kappa = 0.008$）
- 训练迭代次数 $t_{\mathrm{train}} = 250$

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅充分下降条件 | 无法保证收敛 | 论文附录 B 提供反例 |
| 无有界性假设 | 序列可能发散 | 如 $z^{(t)} = \sum_{k=1}^t 1/k$，满足充分下降但无界 |
| $\mathsf{A}$ vs $\mathsf{A}_{\mathrm{conv}}$ 的差距 | 二次问题 ~20% | 说明 bound 虽紧但非最优 |
| $\mathsf{A}$ vs $\mathsf{A}_{\mathrm{conv}}$ 的差距 | 神经网络 ~2% | 非凸场景 bound 更紧 |

### 关键发现

1. **PAC bound 在两个实验中都相当紧**：训练集上的经验估计与泛化 bound 之间差距不大
2. **学习算法在不受约束的情况下仍然高概率收敛**：验证了理论预测
3. **$\mathsf{A}_{\mathrm{conv}} \setminus \mathsf{A}$ 的差距在非凸场景更小**：说明所选充分条件在非凸设定下更接近充要条件
4. **收敛概率随训练集增大而提高**：符合 PAC-Bayesian 理论的预期

## 亮点与洞察

1. **视角转换的深刻性**：传统方法试图证明"对所有问题收敛"（worst-case），本文转为"以高概率收敛"（average-case），这更符合 L2O 的实际使用场景
2. **不限制算法设计**：核心突破在于无需 safeguard，学习算法的更新步骤完全自由，性能不受约束
3. **方法的通用性**：该框架不限于优化，可推广到任何具有 Markov 结构的序列预测模型
4. **可测性证明的技术深度**：证明 $\mathsf{A}_{\mathrm{conv}}$ 的可测性是高度非平凡的，需要利用 Polish 空间的结构性质
5. **理论与实验的一致性**：实验完美验证了理论预测的不等式链 $1 - \Phi^{-1}(\ldots) \leq \mathbb{P}\{\mathsf{A}\} \leq \mathbb{P}\{\mathsf{A}_{\mathrm{conv}}\}$

## 局限与展望

1. **有限迭代的近似**：实践中无法验证无限步迭代的条件，只能在 $t_{\mathrm{train}}$ 步内检验，这是对理论结果的近似
2. **不适用于随机优化**：充分下降条件要求每步损失严格下降，这在 mini-batch SGD 等随机场景中不成立
3. **训练困难**：使学习算法在大多数问题上同时满足三个条件（特别是充分下降）的训练过程较为困难和耗时
4. **bound 的松紧程度**：$\mathbb{P}\{\mathsf{A}_{\mathrm{conv}} \setminus \mathsf{A}\}$ 的差距是未知的，因此 bound 对收敛概率的估计可能偏保守
5. **先验分布选择**：PAC-Bayesian bound 的质量强烈依赖于先验 $\mathbb{P}_H$ 的选择

## 相关工作与启发

- **Safeguard 方法** (Möller et al. 2019; Prémont-Schwarz et al. 2022)：通过约束更新步保证收敛，但牺牲性能
- **展开方法** (Gregor & LeCun 2010)：固定迭代次数，通过 Rademacher 复杂度或稳定性分析给出泛化保证
- **PAC-Bayesian 学习** (Sucker & Ochs 2024)：本文直接建立在该工作基础上，首次将轨迹级泛化应用于收敛性分析
- **KL 不等式** (Attouch et al. 2013)：提供了非光滑非凸场景下的抽象收敛条件
- **算法设计原则** (Castera & Ochs 2024)：从经典算法中提取几何性质指导 L2O 设计

**对未来研究的启发**：将该框架扩展到随机优化场景（如 SGD），或将收敛速率（而非仅收敛性）纳入泛化分析。

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ | 首次在不限制算法设计的前提下证明 L2O 收敛性 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 融合 PAC-Bayesian、变分分析、随机过程三个领域 |
| 实验充分度 | ⭐⭐⭐ | 仅两个实验，规模较小，缺少大规模实际应用验证 |
| 实用性 | ⭐⭐⭐ | 理论突破显著，但训练难度大，尚难直接应用 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，但技术门槛较高 |
| 综合 | ⭐⭐⭐⭐ | 解决了 L2O 领域一个长期开放的理论问题 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](../../ICML2026/optimization/interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)

</div>

<!-- RELATED:END -->

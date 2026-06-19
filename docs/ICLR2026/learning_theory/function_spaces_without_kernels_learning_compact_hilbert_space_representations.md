---
title: >-
  [论文解读] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations
description: >-
  [ICLR 2026][学习理论][函数编码器] 证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。 机器学习方法长期面临计算效率与理论保证之间的权衡： - 神经网络：灵活、可扩展…
tags:
  - "ICLR 2026"
  - "学习理论"
  - "函数编码器"
  - "核方法"
  - "Hilbert空间"
  - "泛化界"
  - "PCA基选择"
---

# Function Spaces Without Kernels: Learning Compact Hilbert Space Representations

**会议**: ICLR 2026  
**arXiv**: [2509.20605](https://arxiv.org/abs/2509.20605)  
**代码**: [有](https://github.com/suann124/function_encoder_kernels)  
**领域**: 学习理论  
**关键词**: 函数编码器, 核方法, Hilbert空间, 泛化界, PCA基选择

## 一句话总结

证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。

## 研究背景与动机

机器学习方法长期面临**计算效率与理论保证之间的权衡**：

- **神经网络**：灵活、可扩展，但理论保证有限（现有的界往往空洞或依赖限制性假设）
- **核方法**：理论完善（RKHS理论、精确的统计保证），但可扩展性差——推理成本与训练集大小 m 成正比（对偶表示中需要计算 m×m 的Gram矩阵）

函数编码器（Function Encoders）是一种近期提出的迁移/表示学习技术：学习一组神经网络基函数 $\{\psi_j\}_{j=1}^n$ 作为显式特征映射 $\phi(x) = [\psi_1(x), \ldots, \psi_n(x)]^\top$，推理成本仅依赖于基函数个数 n 而非数据集大小 m。但其理论基础尚不完善：

1. 缺乏与Hilbert空间技术的形式化联系
2. 没有选择基函数个数 n 的原则性方法
3. 缺乏有限样本的泛化保证

本文旨在填补这三个理论缺口。

## 方法详解

### 整体框架

本文把函数编码器同时放到原始空间（primal）和对偶空间（dual）两个视角下看：原始空间里它就是一个显式特征映射 $\phi(x)=[\psi_1(x),\ldots,\psi_n(x)]^\top$，做线性预测的代价只有 $\mathcal{O}(n)$；对偶空间里它的内积 $\langle\phi(x),\phi(x')\rangle$ 恰好是一次核评估，于是核方法那一整套理论工具都能拿来用——这层“神经特征即核”的等价是全文的支点。整个方法分两段：离线阶段在函数集合 $\{D_1,\ldots,D_N\}$ 上联合训练出一组基函数，在线阶段把基函数冻住、只用正则化最小二乘求解系数 $c$ 来拟合新函数。中间最关键的取舍是“要几个基、留哪几个”，本文给出两条可互换的 PCA 式路线——渐进式逐个长出、或先过参数化再剪枝，两者都收敛到同一个内在维度；而对偶空间的核视角又让它能导出随基数 $n$ 与样本量 $m$ 变化的有限样本泛化界，反过来为“基数怎么选”提供理论依据。下面四个设计就沿“核等价 → 渐进式选基 → 剪枝式选基 → 泛化界”这条线展开。

### 关键设计

**1. 函数编码器即可学习核：把神经特征接到 RKHS 理论上**

这套方法最关键的一步，是证明学到的基函数自带一个合法的核。论文的 Proposition 1 指出，基函数定义的内积 $k(x,x')=\langle\phi(x),\phi(x')\rangle=\sum_{j=1}^n \psi_j(x)\psi_j(x')$ 必然是对称正半定的——因为对任意点和系数都有 $\|\sum_i \alpha_i \phi(x_i)\|^2 \geq 0$，所以它是一个合法的再生核。这个等式看似简单，意义却很大：它说明函数编码器不只是在原始空间提供 $\mathcal{O}(n)$ 的高效预测，在对偶空间还实例化了一个有效核，于是 RKHS 的统计保证、泛化分析这些工具就能直接套到这个神经特征上，原本"神经网络灵活但理论空洞、核方法理论完善但推理代价 $\mathcal{O}(m)$"的割裂局面被接到了一起。

**2. 渐进式训练：像 PCA 加主成分一样逐个长出有序的基**

确定要多少个基、每个基负责什么，本文给的第一个方案是模仿 PCA 逐步添加主成分的过程，一个一个地训练基函数。第一步只训 $\psi_1$ 然后冻结，第二步加进 $\psi_2$ 且只让它去拟合前面留下的残差，依此类推；每加完一个就算一次系数协方差矩阵 $\Sigma_b$ 的特征值 $\lambda_1\geq\ldots\geq\lambda_b$，并用累积解释方差 $\text{CEV}_r=\sum_{i=1}^r\lambda_i/\sum_{j=1}^b\lambda_j\geq\tau$（如 99%）作为停止准则。这样长出来的基天然是有序、可解释的，停在哪里也有清晰的数值判据；代价是整个过程本质串行，没法在 GPU 上并行训练。

**3. 训练后剪枝：先过参数化再砍，换取并行效率**

第二个方案反过来——先用远多于所需的 $B$ 个基（$B\gg r$）一次性联合训练，再对系数协方差矩阵做特征分解，按累积方差定出有效秩 $r=\min\{n:\sum_{i=1}^n\lambda_i/\sum_{j=1}^B\lambda_j\geq\tau\}$。接着给每个基打分 $s_p=\sum_{i=1}^r\lambda_i U_{pi}^2$ 衡量其对前 $r$ 个主方向的贡献，保留 top-$r$、剪掉其余，最后短期微调把性能找回来。和渐进式相比，它牺牲了基的逐步可解释性，但训练完全可并行，在计算上更划算，两者最终都收敛到同一个内在维度 $r$。

**4. 泛化界：用核视角给出非空洞的推理时保证**

有了"函数编码器即核"这层桥，论文就能给出有限样本的泛化界，这也是全文的理论核心。Rademacher 复杂度路线给出 $L(f_{\hat{c}_\lambda})\lesssim \hat{L}_m(f_{\hat{c}_\lambda})+\tilde{\mathcal{O}}\!\left(Y^2 R^2 \frac{n}{\lambda\sqrt{m}}\right)$，PAC-Bayes 路线给出更保守的 $L(f_{\hat{c}_\lambda})\lesssim \hat{L}_m(f_{\hat{c}_\lambda})+\tilde{\mathcal{O}}\!\left(Y^2 R^2 \frac{n^{3/2}}{\lambda\sqrt{m}}\right)$。两个界的 scaling 一致地说明了同一件事：复杂度项随基函数数 $n$ 上升、随样本量 $m$ 和正则化强度 $\lambda$ 下降，所以多加基会提升表达力但也加重过拟合风险——这正好为前两个设计里"选多少基"的取舍提供了理论依据。推导 PAC-Bayes 界时本文用截断高斯分布来处理无界损失，从而把分析扩展到多变量输出和"特征映射本身也是学出来的"这一更难的场景，这个技巧本身具有独立价值。

### 损失函数 / 训练策略

离线训练用的是多任务正则化最小二乘，在 $N$ 个函数上同时拟合并对系数加正则：$\frac{1}{Nm}\sum_{j=1}^N\sum_{i=1}^m \|f_j(x_i)-\hat{f}_j(x_i)\|^2 + \lambda\|\hat{f}_j\|^2$；在线推理时基函数固定，只对新函数的系数再解一次同形式的正则化最小二乘即可。

## 实验关键数据

### 主实验

**多项式空间基准**（已知内在维度 d+1）：

| 多项式阶数 | 内在维度 | Progressive恢复 | Train-Prune恢复 |
|-----------|---------|----------------|----------------|
| d=3 | 4 | 4 ✓ | 4 ✓ |
| d=4 | 5 | 5 ✓ | 5 ✓ |
| d=5 | 6 | 6 ✓ | 6 ✓ |

两种算法均精确恢复已知的内在维度，剪枝后模型精度与过参数化原始模型一致。

**Van der Pol振荡器**：

| 方法 | 基函数数量 | 预测精度 |
|------|-----------|---------|
| 先前工作 (Ingebrand et al.) | 100 | 基准精度 |
| Progressive/Prune | **2** | **相同精度** |

从100个基降至2个，减少**50倍**，精度不变。

**二体轨道问题**：

| 方法 | 基函数数量 | 解释方差>99% |
|------|-----------|-------------|
| 过参数化 | 大量 | — |
| Progressive/Prune | **5-6** | ✓ |

5-6个基捕获>99%方差，与轨道参数的5维空间一致。

### 消融实验

**与深度核学习的对比**（degree-3多项式）：
- 训练时间：函数编码器比深度核（RBF）快约**10倍**（m=20时）
- 原因：深度核需要 $\mathcal{O}(m^3)$ 的Gram矩阵求逆
- Gram矩阵几何结构几乎一致，验证函数编码器可视为原始空间的自适应核

### 关键发现

1. 两种PCA引导算法均能精确识别函数空间的内在维度
2. 紧凑基表示可大幅减少基函数数量（50倍），无精度损失
3. 函数编码器在训练效率上显著优于深度核方法
4. 理论界提供了非空洞的推理时保证
5. 学习到的基函数在动力学系统中展现出可解释的物理结构

## 亮点与洞察

1. **统一了两个世界**：将神经网络的可扩展性与核方法的理论保证桥接，函数编码器恰好处于交汇点
2. **PCA与基选择的类比精妙**：将函数空间的维度选择问题转化为系数空间的PCA截断问题
3. **理论贡献扎实**：截断高斯PAC-Bayes技术本身具有独立价值，可用于其他无界损失场景
4. **实验设计有层次**：从已知维度的多项式验证到实际动力学系统，逐步展示方法的有效性

## 局限与展望

1. 停止阈值 τ 仍是**启发式**的，缺乏非启发式的基选择准则
2. 渐进式训练本质上是串行的，在大规模问题上效率受限
3. 实验仅在低维动力学系统上验证，高维复杂系统（PDE、视觉任务）的表现未知
4. 泛化界中的常数可能过大，虽然非空洞但实际松紧度有待检验
5. 未探索与具体应用（如机器人控制、轨道预测）的端到端集成

## 相关工作与启发

- **核方法（GP/KRR/SVM）**：理论完善但扩展性差，函数编码器在保留理论性质的同时解决这一问题
- **核近似（Nyström/随机Fourier特征）**：缓解计算成本但固定核无法适应数据
- **深度核学习**：参数化核提升表达力但保留 $\mathcal{O}(m)$ 推理成本
- **字典学习/Koopman/SINDy**：也学习基函数但目标不同（稀疏恢复/动力学线性化）
- 启发点：函数编码器的核视角可能为其他迁移学习方法提供理论分析工具

## 评分

- **新颖性**: ★★★★☆ — 统一视角有理论新意，但函数编码器本身并非全新
- **技术深度**: ★★★★★ — Rademacher+PAC-Bayes双重理论分析，截断高斯技术有独立价值
- **实验说服力**: ★★★★☆ — 多项式验证充分，动力学系统展示有力，但缺乏大规模实验
- **实用价值**: ★★★☆☆ — 理论贡献大于实际应用，紧凑基在嵌入式系统中有潜力
- **表达清晰度**: ★★★★★ — 理论推导层次分明，实验设计循序渐进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](../../ICML2026/learning_theory/catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)
- [\[ICML 2026\] Finite-Width Neural Tangent Kernels from Feynman Diagrams](../../ICML2026/learning_theory/finite-width_neural_tangent_kernels_from_feynman_diagrams.md)
- [\[ICML 2026\] CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](../../ICML2026/learning_theory/core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)
- [\[ICML 2026\] Task-Restricted Symmetries in Recurrent Weight Space](../../ICML2026/learning_theory/task-restricted_symmetries_in_recurrent_weight_space.md)
- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](../../ICML2026/learning_theory/on_the_robustness_of_langevin_dynamics_to_score_function_error.md)

</div>

<!-- RELATED:END -->

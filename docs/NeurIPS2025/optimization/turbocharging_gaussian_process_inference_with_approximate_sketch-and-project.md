---
title: >-
  [论文解读] Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project
description: >-
  [NeurIPS 2025][优化/理论][Gaussian Process] 提出 ADASAP 算法，通过近似子空间预条件、分布式计算和 Nesterov 加速，将 sketch-and-project 方法扩展到大规模 GP 推断，首次将精确 GP 推断扩展到 $>3\times10^8$ 样本规模，同时在理论上证明了 SAP 方法的 condition number-free 收敛性。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Gaussian Process"
  - "Sketch-and-Project"
  - "大规模推断"
  - "分布式计算"
  - "Nyström近似"
---

# Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project

**会议**: NeurIPS 2025  
**arXiv**: [2505.13723](https://arxiv.org/abs/2505.13723)  
**代码**: [GitHub](https://github.com/pratikrathore8/scalable_gp_inference)  
**领域**: 高斯过程 / 大规模推断  
**关键词**: Gaussian Process, Sketch-and-Project, 大规模推断, 分布式计算, Nyström近似

## 一句话总结

提出 ADASAP 算法，通过近似子空间预条件、分布式计算和 Nesterov 加速，将 sketch-and-project 方法扩展到大规模 GP 推断，首次将精确 GP 推断扩展到 $>3\times10^8$ 样本规模，同时在理论上证明了 SAP 方法的 condition number-free 收敛性。

## 研究背景与动机

高斯过程 (GP) 是贝叶斯优化、科学计算等领域的核心工具，能提供概率预测和不确定性量化。然而 GP 推断需要求解 $n \times n$ 的稠密线性系统 $(K + \lambda I)w = y$，直接方法如 Cholesky 分解需要 $\mathcal{O}(n^3)$ 计算，限制在 $n \sim 10^4$。

现有两类大规模方法各有缺陷：
- **PCG（预条件共轭梯度）**：对病态问题鲁棒，但当 $n \sim 10^6$ 时速度慢，每步 $\mathcal{O}(n^2)$，且需线性内存存储预条件器
- **SDD（随机对偶下降）**：可扩展到 $n \gg 10^6$，但缺乏理论保证，对步长敏感（SDD-100 在所有数据集上发散），面对病态矩阵收敛变慢

核心矛盾在于：没有一种方法能同时兼顾推断质量、病态鲁棒性、超参数易调和可扩展性。ADASAP 的切入点是 sketch-and-project 框架——通过子空间预条件引入二阶信息，同时保持与 SDD 相当的每步成本。

## 方法详解

### 整体框架

ADASAP 基于 Gower & Richtárik (2015) 的 sketch-and-project 框架，结合三大关键设计：近似预条件、分布式矩阵乘法和 Nesterov 加速。整体目标是求解 $K_\lambda W = Y$，其中 $K_\lambda = K + \lambda I$。

### 关键设计

1. **SAP 基础算法**：每步随机采样大小为 $b$ 的行索引 $\mathcal{B}$，计算子空间梯度 $K_{\mathcal{B}n}W_t + \lambda I_\mathcal{B}W_t - Y_\mathcal{B}$，然后用子空间 Hessian $(K_{\mathcal{B}\mathcal{B}} + \lambda I)^{-1}$ 进行预条件。与 SDD 不同，SAP 包含二阶信息。每步代价 $\mathcal{O}(bn + b^3)$，远低于 PCG 的 $\mathcal{O}(n^2)$。设计动机是在 SDD 的可扩展性和 PCG 的病态鲁棒性之间取得平衡。

2. **近似子空间预条件**：SAP 的瓶颈在于 $b^3$ 的精确分解成本限制了 block size。ADASAP 将精确的 $K_{\mathcal{B}\mathcal{B}}$ 替换为秩 $r$ 的 Nyström 近似 $\hat{K}_{\mathcal{B}\mathcal{B}}$，使得预条件器的应用成本从 $\mathcal{O}(b^3)$ 降到 $\mathcal{O}(br)$。这利用了核矩阵的近似低秩结构。在 taxi 数据集上 block size 可达 $b = 1.65 \times 10^5$。

3. **分布式矩阵乘法**：两个瓶颈操作 $K_{\mathcal{B}n}W_t$（$\mathcal{O}(bn)$）和 $K_{\mathcal{B}\mathcal{B}}\Omega$（$\mathcal{O}(b^2r)$）通过 ColDistMatMat 和 RowDistMatMat 分布到多 GPU 上。在 taxi 数据集上使用 4 GPU 实现 $3.4\times$ 加速，接近完美并行。

4. **Nesterov 加速**：引入动量参数 $\mu, \nu$（默认为 $\lambda$ 和 $n/b$），通过三序列更新 $(W, V, Z)$ 加速收敛。每步自动计算步长，无需像 SDD 那样手动调参。

### 理论贡献

利用行列式点过程 (DPP) 理论证明了 SAP 沿 top-$\ell$ 谱基函数的两阶段收敛：

$$\mathbb{E}\|\text{proj}_\ell(\hat{m}_t) - \text{proj}_\ell(m_n)\|^2_\mathcal{H} \leq \min\left\{\frac{8\phi(b,\ell)}{t}, \left(1-\frac{1}{2\phi(b,n)}\right)^{t/2}\right\}\|y\|^2_{K_\lambda^{-1}}$$

其中 $\phi(b,\ell)$ 是平滑条件数。第一项是 condition number-free 的亚线性速率（与 SGD 类似但更优），第二项是渐近线性速率。当核矩阵有多项式谱衰减时，$\phi(b,\ell)=\mathcal{O}(1)$，即初始收敛完全不依赖条件数。

## 实验关键数据

### 主实验：大规模 GP 推断

| 数据集 | n | d | ADASAP RMSE | PCG RMSE | SDD-10 RMSE | 最优方法 |
|--------|---|---|------------|----------|-------------|--------|
| yolanda | 3.6×10⁵ | 100 | **0.795** | **0.795** | 0.801 | 并列 |
| song | 4.6×10⁵ | 90 | **0.752** | **0.752** | 0.767 | 并列 |
| benzene | 5.7×10⁵ | 66 | **0.012** | 0.141 | 0.112 | ADASAP |
| malonaldehyde | 8.9×10⁵ | 36 | **0.015** | 0.273 | 发散 | ADASAP |
| acsincome | 1.5×10⁶ | 9 | **0.789** | 0.875 | **0.792** | ADASAP |
| houseelec | 1.8×10⁶ | 9 | **0.027** | 1.278 | 0.119 | ADASAP |

### 超大规模实验

| 数据集 | n | ADASAP RMSE | SDD-10 RMSE | PCG |
|--------|---|-------------|-------------|-----|
| taxi | 3.31×10⁸ | **0.50** | 0.52 | OOM |

ADASAP 比 SDD-10 快 45%（节省 14 小时），PCG 因内存不足无法运行。

### 消融实验（贝叶斯优化）

| 方法 | 长度尺度=2.0 提升(%) | 长度尺度=3.0 提升(%) |
|------|---------------------|---------------------|
| ADASAP | **10.42** | **13.86** |
| ADASAP-I (无预条件) | 7.04 | 11.27 |
| SDD-1 | 6.50 | 11.17 |
| PCG | 0.13 | 5.54 |

### 关键发现

- SDD 对步长极其敏感：SDD-100 在所有数据集上发散，SDD-1 和 SDD-10 性能差异大
- ADASAP 默认超参数开箱即用，无需手动调参
- 近似预条件（ADASAP vs ADASAP-I）在病态问题上提升显著（benzene: 0.012 vs 0.168）
- 首次将精确 GP 推断扩展到 $>3\times10^8$ 样本

## 亮点与洞察

- 将 DPP 理论与 sketch-and-project 结合，得到首个 condition number-free 的后验均值估计算法
- "平滑条件数" $\phi(b, \ell)$ 的定义精妙：block size $b$ 和子空间维度 $\ell$ 共同控制收敛，实现了"二阶信息的低成本注入"
- 默认超参数设计优雅：加速参数 $\mu=\lambda$, $\nu=n/b$ 直接从问题结构推导

## 局限与展望

- 理论分析尚未覆盖 ADASAP 的近似预条件、加速和均匀采样（目前只证了 SAP）
- 需要 GPU，对硬件有要求
- 未探索半精度（float16）的潜力
- Tail averaging 在实验中反而不如不用，理论与实践存在 gap

## 相关工作与启发

- Nyström 近似在核方法中已广泛使用，本文将其与 sketch-and-project 结合是新颖点
- SDD 的步长敏感性问题在实践中是真实痛点，ADASAP 的自动步长是重要优势
- 可启发其他需要大规模核矩阵求解的场景（如核岭回归、核 SVM）

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论上 DPP+SAP 的结合和 condition number-free 保证是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 从小到超大规模全覆盖，taxi 数据集令人印象深刻
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但理论与算法实际差距可更明确讨论
- 价值: ⭐⭐⭐⭐⭐ 直接解决 GP 推断的可扩展性瓶颈，有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs](learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)

</div>

<!-- RELATED:END -->

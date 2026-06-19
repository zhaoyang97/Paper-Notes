---
title: >-
  [论文解读] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes
description: >-
  [NeurIPS 2025][优化/理论][神经正切核] 为梯度下降训练的浅层神经网络提供了在任意正训练时间 $t \geq 0$ 下向高斯过程收敛的显式定量上界，证明了二次Wasserstein距离以 $O(\log n_1 / n_1)$ 的速率多项式衰减。 深度学习的理论理解一直是重要研究方向。在过参数化体系中…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "神经正切核"
  - "高斯过程"
  - "Wasserstein距离"
  - "有限宽度"
  - "无限宽极限"
---

# Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes

**会议**: NeurIPS 2025  
**arXiv**: [2509.24544](https://arxiv.org/abs/2509.24544)  
**代码**: 无  
**领域**: 优化理论 / 神经网络理论  
**关键词**: 神经正切核, 高斯过程, Wasserstein距离, 有限宽度, 无限宽极限

## 一句话总结

为梯度下降训练的浅层神经网络提供了在任意正训练时间 $t \geq 0$ 下向高斯过程收敛的显式定量上界，证明了二次Wasserstein距离以 $O(\log n_1 / n_1)$ 的速率多项式衰减。

## 研究背景与动机

深度学习的理论理解一直是重要研究方向。在过参数化体系中，当网络参数从高斯分布初始化时，Neal (1996)和de G. Matthews et al. (2018)证明了网络输出在宽度趋于无穷时收敛到高斯过程。Jacot et al. (2018)引入的**神经正切核（NTK）**框架进一步刻画了无限宽网络在梯度下降下的训练动态：网络围绕初始化近似线性演化，训练可以理解为用固定核的核回归。

**核心问题**：NTK分析的实际相关性取决于其在有限宽度下的近似精度。尽管Lee et al. (2020)建立了NTK体系下的定性收敛，但**严格的定量结果——提供显式的有限宽度误差界——仍然极其稀缺**。

这一空白带来两个实际问题：(1) 无法量化有限宽网络预测与无限宽NTK对应体之间的差异；(2) 无法揭示网络宽度、深度、初始化和训练超参数如何影响线性近似的有效性。

已有工作如Basteri & Trevisan (2024)和Favaro et al. (2025)只在**初始化时**给出了定量收敛率，本文的核心贡献是将这些结果**扩展到任意正训练时间**。

## 方法详解

### 整体框架

考虑单隐层（浅层）全连接神经网络：
$$f(x; \theta) = \frac{1}{\sqrt{n_1}} \Phi\left(\frac{1}{\sqrt{n_0}} x \theta^{(0)}\right) \theta^{(1)}$$

其中 $n_0$ 为输入维度，$n_1$ 为隐层宽度，$\theta^{(0)} \in \mathbb{R}^{n_0 \times n_1}$，$\theta^{(1)} \in \mathbb{R}^{n_1}$。

对关联高斯过程 $G_t(x)$，其均值和协方差由解析NTK $k_\infty$ 确定：
$$\mu_t(x) = k_\infty(x, \mathcal{X}) I_t(k_\infty) y$$
$$\Sigma_t(x, x') = \mathcal{K}(x,x') - \text{(训练修正项)}$$

目标是量化 $\mathcal{W}_2^2(f(x; \theta_t), G_t(x))$ 随 $n_1$ 的衰减行为。

### 关键设计

1. **三角不等式分解**：将总Wasserstein距离分为两项：
    $\mathcal{W}_2(f(x;\theta_t), G_t(x)) \leq \underbrace{\mathcal{W}_2(f, f^{\text{lin}})}_{\text{非线性误差}} + \underbrace{\mathcal{W}_2(f^{\text{lin}}, G_t)}_{\text{CLT误差}}$
    - 第一项：控制真实网络与其线性化版本之间的偏差
    - 第二项：控制线性化网络（CLT）向高斯过程的收敛

2. **线性化网络分析**：定义线性化网络 $f^{\text{lin}}(x;\theta_t) = f(x;\theta_0) + \nabla_\theta f(x;\theta_0)|_{\theta_0} \omega_t$，其梯度流方程有解析解：
    $f^{\text{lin}}(x;\bar{\theta}_t) = f(x;\theta_0) - k_{x\mathcal{X}} I_t(k_{\mathcal{X}\mathcal{X}})(f(x;\theta_0) - y)$
   其中 $I_t(B) = (\mathbb{1}_n - e^{-Bt})B^{-1}$ 是一个辅助算子。

3. **好事件分区**：将参数空间划分为"好事件" $S$（满足NTK条件数等关键性质）和其补集 $S^C$。在 $S$ 上利用Proposition B.9控制 $f$ 与 $f^{\text{lin}}$ 的 $L^2$ 距离；在 $S^C$ 上利用Lemma B.12控制尾事件。$t^8$ 项来自尾事件的控制。

### 主定理

**Theorem 3.4**：在Assumptions 1-4下，对每个测试点 $x \in \mathbb{R}^{n_0}$，存在不依赖于 $n_0, n_1, t$ 的正常数 $a_1, a_2$：

$$\mathcal{W}_2^2(f(x;\theta_t), G_t(x)) \leq r\left(\frac{a_1 \log n_1}{(\lambda_{\min}^\infty)^3 n_1 n_0} + \frac{a_2 n_0}{(\lambda_{\min}^\infty)^r n_1^{r/4}}(1 + t^8)\right)$$

关键特性：
- 对固定 $t$，主项为 $O(\log n_1 / n_1)$，即多项式衰减
- 只要 $t$ 以 $n_1$ 的多项式增长，选取足够大的 $r$ 即可使右侧趋于零
- $\lambda_{\min}^\infty$ 是解析NTK的最小特征值，要求正定（温和假设）

### 假设条件

- Assumption 1: 高斯初始化
- Assumption 2: 解析NTK正定（当训练数据在一般位置且 $\Phi$ 非多项式时成立）
- Assumption 3: $\Phi$ 和 $\Phi'$ 为Lipschitz连续有界（sigmoid、tanh、Gaussian等满足，ReLU不满足但可期望结论仍成立）
- Assumption 4: 充分过参数化条件（左端在 $\min\{n_0, n_1\}$ 增大时趋于零）

## 实验关键数据

### 数值验证

| 网络宽度 $n_1$ | $\mathcal{W}_2^2$ (实验) | 理论上界趋势 | 说明 |
|-------------|--------------|-----------|------|
| 小 → 大 | 单调递减 | $O(\log n_1/n_1)$ | 收敛率与理论预测吻合 |

### 验证实验设计

| 实验设置 | 描述 | 验证结论 |
|---------|------|---------|
| 不同 $n_1$ | 固定 $n_0$，变化隐层宽度 | $\mathcal{W}_2^2$ 随 $n_1$ 多项式衰减 |
| 不同训练时间 $t$ | 固定宽度，变化训练步数 | 误差在多项式时间内保持有界 |
| 不同激活函数 | sigmoid、tanh等 | 满足Assumption 3的激活均验证通过 |

### 关键发现

- 定量收敛率在训练全程成立，不仅限于初始化
- 收敛率对网络宽度 $n_1$ 几乎最优（接近已知的 $n^{-1}$ 最优率，多一个 $\log$ 因子）
- 误差界可在 $t$ 多项式增长的时间尺度上维持，涵盖实际训练时长

## 亮点与洞察

- 首次为训练中的浅层网络提供了显式的Wasserstein-2定量收敛率，弥合了初始化与训练期间的理论空白
- 揭示了宽度、输入维度、NTK最小特征值和训练时间之间的精确相互作用
- 通过好事件/坏事件分区和辅助算子 $I_t(B)$ 的巧妙定义，优雅地处理了随机矩阵的可逆性问题

## 局限与展望

- 仅适用于浅层（单隐层）网络，扩展到深层网络是重要但困难的方向
- 不适用于ReLU激活（需要Lipschitz连续性），虽然作者预期结论仍成立
- $t^8$ 项来自尾事件的粗糙控制，细化尾部估计可能改善时间依赖
- 实际深度网络中的NTK不保持恒定（lazy training假设的局限性）

## 相关工作与启发

- **vs Basteri & Trevisan (2024)**: 后者仅在初始化时给出定量率，本文扩展到任意正训练时间
- **vs Lee et al. (2020)**: 后者建立了NTK体系的定性收敛，本文提供了定量的有限宽度误差界
- **vs Bordino et al. (2025)**: 后者用二阶Poincaré不等式得到次优收敛率，本文在浅层设置下给出更优的率
- **vs de G. Matthews et al. (2018)**: 后者处理更一般的深层网络但收敛度量较弱（$\rho_F$ 度量），本文在浅层下给出更强的 $\mathcal{W}_2$ 度量

## 评分

- 新颖性: ⭐⭐⭐⭐ 将初始化时的定量结果扩展到训练期间是非平凡的贡献，但主要是对已有技术的精细化
- 实验充分度: ⭐⭐⭐ 作为理论工作，数值验证存在但规模有限，未在大规模实际网络上验证
- 写作质量: ⭐⭐⭐⭐ 理论陈述严谨，符号统一，证明路径清晰
- 价值: ⭐⭐⭐⭐ 为NTK理论增加了重要的定量工具，对理解有限宽网络的理论属性有意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization](../../ICML2025/optimization/global_convergence_and_rich_feature_learning_in_l-layer_infinite-width_neural_ne.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[NeurIPS 2025\] Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project](turbocharging_gaussian_process_inference_with_approximate_sketch-and-project.md)

</div>

<!-- RELATED:END -->

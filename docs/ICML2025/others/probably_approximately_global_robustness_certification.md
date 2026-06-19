---
title: >-
  [论文解读] Probably Approximately Global Robustness Certification
description: >-
  [ICML2025][全局鲁棒性] 提出基于 ε-net 采样的概率近似全局鲁棒性（PAG）认证框架，所需样本量与输入维度、类别数和模型架构无关，可高效认证大规模神经网络的全局鲁棒性。 核心问题：如何高效地为大规模神经网络提供全局鲁棒性保证？ 现有方法面临两难困境： - 形式验证方法：（Marabou、α-CROWN 等）：…
tags:
  - "ICML2025"
  - "全局鲁棒性"
  - "ε-net"
  - "概率认证"
  - "对抗鲁棒性"
  - "VC维"
  - "采样认证"
---

# Probably Approximately Global Robustness Certification

**会议**: ICML2025  
**arXiv**: [2511.06495](https://arxiv.org/abs/2511.06495)  
**代码**: 待确认  
**领域**: 鲁棒性认证 (Robustness Certification)  
**关键词**: 全局鲁棒性, ε-net, 概率认证, 对抗鲁棒性, VC维, 采样认证

## 一句话总结

提出基于 ε-net 采样的概率近似全局鲁棒性（PAG）认证框架，所需样本量与输入维度、类别数和模型架构无关，可高效认证大规模神经网络的全局鲁棒性。

## 研究背景与动机

**核心问题**：如何高效地为大规模神经网络提供全局鲁棒性保证？

现有方法面临两难困境：

- **形式验证方法**（Marabou、α-CROWN 等）：可提供可证明的局部鲁棒性保证，但计算开销极大，仅适用于数百参数的小网络。近期扩展到全局鲁棒性认证的工作也仅限于极小规模模型。
- **对抗攻击方法**（FGSM、PGD、C&W 等）：可扩展到大网络，但只能经验性地评估鲁棒性，无法提供理论保证。

**关键洞察**：能否用采样的方式获得关于全局鲁棒性的**概率保证**？即以高概率保证分类器在大部分输入上都是鲁棒的。通过引入学习理论中的 ε-net 工具，可以仅从数据分布中 iid 采样，结合局部鲁棒性预言机，得到与输入维度无关的全局鲁棒性认证。

## 方法详解

### 1. 问题形式化

给定分类器 $f: \mathcal{X} \to \mathbb{R}^n$，定义：

- **局部 ρ-鲁棒性**：若鲁棒性预言机 $\mathbf{rob}_f(\mathbf{x}) \geq \rho$，则 $f$ 在 $\mathbf{x}$ 周围是局部 ρ-鲁棒的
- **全局 ρ-κ-鲁棒性**：对所有预测置信度 $\mathbf{conf}_f(\mathbf{x}) \geq \kappa$ 的点，$f$ 都是局部 ρ-鲁棒的

$$\forall \mathbf{x} \in \mathcal{X}: \mathbf{conf}_f(\mathbf{x}) \geq \kappa \implies \mathbf{rob}_f(\mathbf{x}) \geq \rho$$

由于严格的全局鲁棒性认证计算上不可行，论文提出**概率松弛**。

### 2. PAG 鲁棒性定义

**近似全局鲁棒性（Definition 4.1）**：给定数据分布 $\mathcal{D}$，分类器 $f$ 的 PAG 鲁棒性要求：

$$\Pr\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \epsilon$$

即在高置信度预测中，不鲁棒的点的概率小于 ε。

### 3. 质量空间与 ε-net

**质量空间映射**：将每个输入 $\mathbf{x}$ 映射到二维"质量空间"：

$$q(\mathbf{x}) \mapsto (\mathbf{rob}_f(\mathbf{x}), \mathbf{conf}_f(\mathbf{x}))$$

在质量空间中，反例区域定义为：

$$R(\rho, \kappa) = \{(\rho', \kappa') \in \mathbb{R}^2 : \rho' < \rho \wedge \kappa' \geq \kappa\}$$

这是两个轴对齐半空间的交集，对应的 range space 的 **VC 维度 d=2**。

**ε-net 采样定理**：从数据分布 $\mathcal{D}$ 中 iid 采样 $s$ 个点，当样本量满足：

$$s \geq \frac{2}{\ln(2)\epsilon}\left(\ln\frac{1}{\delta} + d\ln(2s) - \ln(1 - e^{-s\epsilon/8})\right)$$

时，以至少 $1-\delta$ 的概率构成 ε-net。**由于 d=2，该样本量与输入维度、类别数、模型架构完全无关**。

### 4. PAG 鲁棒性认证（Theorem 4.3）

核心定理：取 $|N| \geq s(\epsilon, \delta/2, 2)$ 个 iid 样本，若样本中不存在反例（即 $q(N) \cap R(\rho, \kappa) = \emptyset$），则以至少 $1-\delta$ 的概率：

$$\Pr\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \frac{\epsilon}{p_{\min}}$$

证明思路：(1) 用 ε-net 约束反例的联合概率上界；(2) 用 Chernoff 界约束置信度分位数的下界；(3) 联合界组合两个保证。

### 5. 鲁棒性下界映射 M(κ)

从样本 $N$ 构造映射 $M(\kappa)$：对每个置信度 κ，返回样本中所有置信度 ≥κ 的点的最小鲁棒性值：

$$M(\kappa) = \min_{\mathbf{x} \in N} \mathbf{rob}_f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{conf}_f(\mathbf{x}) \geq \kappa$$

该映射可在 $O(|N|\log|N|)$ 时间内构造，为新数据点提供鲁棒性下界。

### 6. 分布偏移下的保证

当只能从近似分布 $\mathcal{D}'$ 采样时，若两分布的全变差距离为 $\Lambda = \mathrm{TV}(\mathcal{D}, \mathcal{D}')$，则保证退化为：

$$\Pr_{\mathcal{D}}\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \frac{\epsilon + \Lambda}{p_{\min} - \Lambda}$$

## 实验关键数据

### 实验设置

| 数据集 | 架构 | 参数量 | 鲁棒性预言机 |
|--------|------|--------|-------------|
| MNIST | FeedForward | 39K | PGD, LiRPA |
| MNIST | ConvBig | 1,663K | LiRPA |
| CIFAR-10 | ResNet20 | 272K | PGD |
| CIFAR-10 | VGG11_BN | 9,491K | PGD |

每个架构训练 5 个标准训练实例 + 5 个 TRADES 对抗训练实例，交叉验证。

### 参数设置

- PGD 预言机：$\epsilon=10^{-4}$, $p_{\min}=0.01$, $\delta=0.01$，采样量 $s = 989,534$
- LiRPA 预言机：$\epsilon=2.5 \times 10^{-3}$, $p_{\min}=0.05$, $\delta=0.01$，采样量 $s = 31,635$

### 主要结果

| 实验 | 网络 | $\hat{p} \times 10^3$ | $\epsilon/p_{\min} \times 10^3$ | $n_c$ | 合格率 | 运行时间(min) |
|------|------|----------|------------|-------|--------|------------|
| MNIST_ST,L | FeedForward | **0.00** | 50.00 | **0** | 25/25 | 8.3 |
| MNIST_ST,L | ConvBig | **0.00** | 50.00 | **0** | 15/15 | 245 |
| MNIST_AT,L | FeedForward | **1.30** | 50.00 | **7** | 25/25 | 8.3 |
| MNIST_AT,L | ConvBig | **3.39** | 50.00 | 44 | 13/15 | 246 |
| MNIST_ST,P | FeedForward | **0.18** | 10.00 | **2** | 25/25 | 0.2 |
| MNIST_AT,P | FeedForward | 16.50 | 10.00 | **4** | 19/25 | 0.3 |
| CIFAR_ST,P | ResNet20 | **1.35** | 10.00 | 8 | 17/25 | 4.5 |
| CIFAR_ST,P | VGG11_BN | **0.00** | 10.00 | **0** | 25/25 | 38.8 |
| CIFAR_AT,P | ResNet20 | **1.51** | 10.00 | **4** | 25/25 | 133.4 |
| CIFAR_AT,P | VGG11_BN | 17.00 | 10.00 | **9** | 22/25 | 308 |

**关键发现**：230 次实验中有 211 次（91.7%）保证在未见测试数据上成立。违反案例中多数仅轻微偏离，且可归因于高斯噪声采样与真实分布的微小偏移。

## 亮点与洞察

1. **维度无关性**：样本复杂度仅取决于 ε、δ 和 VC 维（d=2），与输入维度、类别数、模型规模完全无关。这使得方法可直接应用于千万参数级网络。
2. **预言机无关性**：框架对局部鲁棒性预言机完全透明，可无缝集成 PGD、LiRPA、Marabou 等任意方法。
3. **一次采样多重保证**：单次采样即可同时获得所有 $(\rho, \kappa)$ 组合的鲁棒性保证，无需对每个参数重新采样。
4. **分布偏移鲁棒性**：理论上刻画了在近似分布下采样时保证的退化程度，增强了实际可用性。
5. **理论优雅**：巧妙地将高维鲁棒性认证问题压缩到二维质量空间，利用 ε-net 的经典理论实现了计算与统计效率的统一。

## 局限与展望

1. **依赖 iid 采样假设**：实际中难以从数据真实分布精确 iid 采样，论文用高斯噪声近似可能引入偏差。
2. **保证强度有限**：$\epsilon/p_{\min}$ 的比值可能较大（如 PGD 实验中为 1%），对安全攸关场景可能不够严格。
3. **置信度依赖 softmax**：当模型校准较差时，softmax 置信度可能无法真实反映预测可靠性。
4. **预言机质量上界**：当使用 PGD 等启发式预言机时，鲁棒性估计仅为上界，可能导致保证偏乐观。
5. **样本量仍较大**：PGD 设置下需近百万样本，对于推理代价高的模型仍有运行时压力。
6. **未考虑预测类别信息**：当前方法对所有类别统一处理，未利用不同类别可能有不同鲁棒性特征的先验。

## 相关工作与启发

- **形式验证**：Marabou、α-CROWN 等提供精确局部鲁棒性验证，但仅适用于小网络
- **对抗攻击**：FGSM、PGD、C&W 用于经验性鲁棒性评估
- **认证训练**：TRADES 等方法训练可认证鲁棒的网络
- **全局鲁棒性**：Athavale et al. 2024 首次将形式验证扩展到全局鲁棒性，但仅限数百参数网络
- **统计鲁棒性**：随机平滑（Cohen et al. 2019）提供概率鲁棒性保证，但仅针对单点

本文的思路——将高维问题通过质量空间映射降维后利用 ε-net 理论——可能启发其他需要高维概率保证的验证任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 质量空间 + ε-net 的组合思路新颖，维度无关性结论令人印象深刻
- 实验充分度: ⭐⭐⭐⭐ — 230次实验覆盖4种架构×2种训练×2种预言机，统计充分
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰严谨，定义-命题-定理层层递进
- 价值: ⭐⭐⭐⭐ — 填补了大规模网络全局鲁棒性认证的空白，实际可用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ACL 2025\] Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability](../../ACL2025/others/neuron_empirical_gradient_discovering_and_quantifying_neurons_global_linear_cont.md)
- [\[CVPR 2026\] Global Information Thresholding for Sufficient and Necessary Circuits](../../CVPR2026/others/global_information_thresholding_for_sufficient_and_necessary_circuits.md)

</div>

<!-- RELATED:END -->

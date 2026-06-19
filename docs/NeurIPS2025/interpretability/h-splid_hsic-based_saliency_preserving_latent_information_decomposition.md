---
title: >-
  [论文解读] H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition
description: >-
  [NeurIPS 2025][可解释性][显著性特征学习] 提出 H-SPLID，通过将隐空间显式分解为**显著（任务相关）**和**非显著（任务无关）**两个子空间，结合 HSIC 正则化实现信息压缩，证明预测偏差上界受显著子空间维度和 HSIC 控制，在无对抗训练条件下显著提升对非显著区域扰动的鲁棒性。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "显著性特征学习"
  - "潜空间分解"
  - "HSIC"
  - "对抗鲁棒性"
  - "维度压缩"
---

# H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition

**会议**: NeurIPS 2025  
**arXiv**: [2510.20627](https://arxiv.org/abs/2510.20627)  
**代码**: [GitHub](https://github.com/neu-spiral/H-SPLID)  
**领域**: 表示学习 / 鲁棒性  
**关键词**: 显著性特征学习, 潜空间分解, HSIC, 对抗鲁棒性, 维度压缩

## 一句话总结

提出 H-SPLID，通过将隐空间显式分解为**显著（任务相关）**和**非显著（任务无关）**两个子空间，结合 HSIC 正则化实现信息压缩，证明预测偏差上界受显著子空间维度和 HSIC 控制，在无对抗训练条件下显著提升对非显著区域扰动的鲁棒性。

## 研究背景与动机

神经网络虽然在分类任务上达到高准确率，但常常依赖**任务无关的冗余特征**（如图像背景），导致：
- 对非显著区域的对抗攻击异常脆弱：在双数字诊断测试中，标准 CNN 对右侧数字的 PGD 攻击（$\epsilon=1.0$）使准确率从 96.86% 暴跌至 31.76%
- 冗余维度扩大对抗攻击的可利用空间

已有方法的局限：
- 对抗训练计算昂贵且针对特定攻击
- 信息瓶颈方法（HBaR 等）在单一隐空间中压缩，无法显式分离显著/非显著成分
- 对比分析方法需要额外的背景数据集

## 方法详解

### 显著性感知隐空间分解

编码器 $f_\psi: \mathcal{X} \to \mathbb{R}^m$ 将输入映射到隐表示 $\mathbf{z}$，通过可学习对角掩码 $\mathbf{M}_s = \text{diag}\{\boldsymbol{\beta}\}$（$\boldsymbol{\beta} \in \{0,1\}^m$）分解为：

$$\mathbf{z}_s = \boldsymbol{\beta} \odot \mathbf{z}, \quad \mathbf{z}_n = (\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}$$

分类仅使用显著部分：$\hat{\mathbf{y}} = \mathbf{W}^\top \mathbf{M}_s f_\psi(\mathbf{x})$。

### 正则化设计

**聚类损失**促进结构化分离：
$$\mathcal{L}_s = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{M}_s(\mathbf{z}_i - \boldsymbol{\mu}_k)\|^2, \quad \mathcal{L}_n = \sum_{i=1}^n \|\mathbf{M}_n(\mathbf{z}_i - \boldsymbol{\mu})\|^2$$

$\mathcal{L}_s$ 在显著子空间聚类同类、$\mathcal{L}_n$ 在非显著子空间拉齐所有样本。

**HSIC 正则化**：
$$\rho_s \widehat{\text{HSIC}}(\mathbf{X}, \mathbf{Z}_s) + \rho_n \widehat{\text{HSIC}}(\mathbf{Y}, \mathbf{Z}_n)$$

- 第一项：压缩显著子空间与输入的统计依赖（去冗余）
- 第二项：消除非显著子空间与标签的依赖（去标签信息泄漏）

**总目标**：
$$\mathcal{L} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_s \mathcal{L}_s + \lambda_n \mathcal{L}_n + \rho_s \text{HSIC}(\mathbf{X}, \mathbf{Z}_s) + \rho_n \text{HSIC}(\mathbf{Y}, \mathbf{Z}_n)$$

### 交替优化

- **固定掩码优化网络**：标准 SGD
- **固定网络优化掩码**：闭式解

$$\beta_i^* = \frac{\lambda_n \sum_{\mathbf{z}}(\mathbf{z}_i - \mu_i)^2}{\lambda_s \sum_k \sum_{\mathbf{z} \in C_k}(\mathbf{z}_i - (\mu_k)_i)^2 + \lambda_n \sum_{\mathbf{z}}(\mathbf{z}_i - \mu_i)^2}$$

直觉：类内方差小的维度被分配为显著维度（$\beta \to 1$）。

### 理论保证

**鲁棒性上界**（Theorem 3.2）：对有界扰动 $\|\delta(\mathbf{x})\|_2 \leq r$：

$$\mathbb{E}_\mathbf{x}\left[\|h_\theta(\mathbf{x}+\delta(\mathbf{x})) - h_\theta(\mathbf{x})\|_2\right] \leq \frac{rRB\sqrt{ks}(LR+\|f_\psi(0)\|_2)}{\sigma^2 K_\mathcal{F} K_\mathcal{G}} \cdot \text{HSIC}(\mathbf{x}, \mathbf{z}_s) + o(r)$$

其中 $s = \|\mathbf{M}_s\|_0$ 为显著维度数。**同时减少 HSIC 和降低 $s$ 可收紧上界**。

## 实验关键数据

### COCO 背景攻击（ResNet-18）

| 方法 | 无攻击 | Block PGD $\frac{25}{255}$ | 背景 PGD $\frac{2}{255}$ | 全局 PGD $\frac{2}{255}$ |
|------|--------|---------------------------|-------------------------|------------------------|
| Vanilla | 98.1 | 56.3 | 56.6 | 34.2 |
| WD | 94.3 | 43.9 | 59.9 | 40.7 |
| GLA | 97.1 | 60.4 | 57.4 | 37.3 |
| HBaR | — | — | — | — |
| **H-SPLID** | **97.8** | **82.5** | **70.7** | **47.6** |

H-SPLID 在背景块攻击下比 Vanilla 高 **26 个百分点**。

### C-MNIST 诊断测试

| 方法 | 正常准确率 | PGD 攻击右数字 ($\epsilon=1.0$) |
|------|-----------|-------------------------------|
| Vanilla (CE) | 96.86 | 31.76 |
| **H-SPLID** | 97.14 | **87.46** |

### ImageNet-9 迁移学习（ResNet-50）

| 方法 | Original | MixedRand | Only-FG |
|------|----------|-----------|---------|
| Vanilla | 94.92 | 73.93 | 89.70 |
| HBaR | 95.03 | 74.12 | 89.76 |
| **H-SPLID** | **95.24** | **75.63** | **90.39** |

### ISIC-2017 医学影像

在亮度、散焦模糊、雪/遮挡等真实世界腐蚀下，H-SPLID 一致优于所有基线。

## 亮点与洞察

- **无需对抗训练和显著性标注**：纯粹通过隐空间分解+统计正则化实现鲁棒性
- **闭式掩码更新**：避免二值优化的指数复杂度
- **理论联系清晰**：将 HSIC + 维度压缩与鲁棒性上界建立严格关联
- **多类扩展**：将 Wang et al. 的二分类理论扩展到任意 $k$ 类

## 局限与展望

1. HSIC 经验估计器的计算复杂度为 $O(n^2)$（kernel 矩阵），大 batch 时可能成为瓶颈
2. 显著/非显著维度的阈值固定（0.5），未探索自适应阈值
3. 全局攻击下鲁棒性提升有限（显著区域本身也被扰动），这是方法本质局限
4. 实验主要在 ResNet 系列上，未验证 ViT 等现代架构
5. 非显著子空间的重构能力未被利用（可考虑自编码器扩展）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 隐空间显式分解+HSIC 信息压缩的组合思路新颖
- **技术深度**: ⭐⭐⭐⭐⭐ — 理论上界严谨，多类推广和体积界均有贡献
- **实验充分度**: ⭐⭐⭐⭐ — 5 个数据集覆盖合成/自然/医学场景，攻击类型多样
- **实用性**: ⭐⭐⭐⭐ — 即插即用于分类网络，但 HSIC 计算开销限制大规模应用
- **总体**: ⭐⭐⭐⭐

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)
- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](deep_modularity_networks_with_diversity-preserving_regularization.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)

</div>

<!-- RELATED:END -->

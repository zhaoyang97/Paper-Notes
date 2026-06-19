---
title: >-
  [论文解读] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification
description: >-
  [NeurIPS 2025][优化/理论][深度ReLU网络] 证明了深度ReLU网络上梯度下降的泛化速率达到 $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$，首次在深度ReLU网络上同时实现：(1) 对样本量 $n$ 的最优 $1/n$ 依赖，(2) 对深度 $L$ 仅多项式依赖。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "深度ReLU网络"
  - "泛化界"
  - "Rademacher复杂度"
  - "NTK可分"
  - "梯度下降"
---

# Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification

**会议**: NeurIPS 2025  
**arXiv**: [2510.02779](https://arxiv.org/abs/2510.02779)  
**代码**: 无  
**领域**: 优化  
**关键词**: 深度ReLU网络, 泛化界, Rademacher复杂度, NTK可分, 梯度下降

## 一句话总结

证明了深度ReLU网络上梯度下降的泛化速率达到 $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$，首次在深度ReLU网络上同时实现：(1) 对样本量 $n$ 的最优 $1/n$ 依赖，(2) 对深度 $L$ 仅多项式依赖。

## 研究背景与动机

理解深度神经网络的泛化性能是理论研究的核心问题。已有工作在分析GD训练的深度网络泛化时面临两大困难：

**次优的样本复杂度**: 基于Rademacher复杂度或NTK框架的方法通常只能得到 $\widetilde{O}(1/\sqrt{n})$ 的泛化界，或者只适用于浅层网络

**对深度的指数依赖**: 基于算法稳定性的方法可以达到 $\widetilde{O}(1/n)$ 的最优速率，但需要光滑激活函数且对深度 $L$ 有 $e^{O(L)}$ 的指数依赖

本文的核心问题是：能否在深度ReLU（非光滑）网络上同时获得最优的 $1/n$ 速率和多项式的 $\text{poly}(L)$ 依赖？

## 方法详解

### 整体框架

采用"优化误差+泛化误差"的经典分解 $\mathcal{L}(\mathbf{W}) = (\mathcal{L}(\mathbf{W}) - \mathcal{L}_S(\mathbf{W})) + \mathcal{L}_S(\mathbf{W})$。对优化误差使用参考模型分析，对泛化误差使用改进的Rademacher复杂度技术。关键创新在于控制ReLU激活模式的变化。

### 关键设计

1. **参考模型优化分析（Theorem 1）**: 定义关于参考模型 $\bar{\mathbf{W}}$ 的量 $F_S(\bar{\mathbf{W}}) = 3\eta T \mathcal{L}_S(\bar{\mathbf{W}}) + \|\mathbf{W}(0) - \bar{\mathbf{W}}\|_F^2$，证明所有GD迭代满足 $\|\mathbf{W}(t) - \bar{\mathbf{W}}\|_F^2 \leq F_S(\bar{\mathbf{W}})$ 且累积训练损失有界。核心不等式为 $\|\mathbf{W}(t+1) - \bar{\mathbf{W}}\|_F^2 \leq \|\mathbf{W}(t) - \bar{\mathbf{W}}\|_F^2 - \eta \mathcal{L}_S(\mathbf{W}(t)) + 3\eta \mathcal{L}_S(\bar{\mathbf{W}})$。相比已有工作，过参数化要求降低了 $L^6$ 倍。

2. **改进的Rademacher复杂度（核心贡献）**: 将网络输出差异表示为 $f_{\mathbf{W}}(\mathbf{x}_i) - f_{\mathbf{W}(0)}(\mathbf{x}_i) = \mathbf{a}^\top \sum_{l=1}^L \hat{\mathbf{G}}_{L,0}^l(\mathbf{x}_i)(\mathbf{W}^l - \mathbf{W}^l(0)) h_0^{l-1}(\mathbf{x}_i)$，其中 $\hat{\mathbf{G}}_{L,0}^l$ 是稀疏矩阵的乘积。关键在于证明 $\hat{\mathbf{G}}_{L,0}^l = \widetilde{O}(L/\sqrt{m})$，从而得到 $\mathfrak{R}_{S_1,n}(\mathcal{F}) = \widetilde{O}(L^2 \sqrt{F(\bar{\mathbf{W}})/n})$。已有方法得到的是 $\widetilde{O}(4^L L \sqrt{mF/n})$，含指数深度依赖和 $\sqrt{m}$ 宽度因子。

3. **覆盖数策略（消除指数深度依赖）**: 对Lipschitz常数的分析，已有递推方法导致 $\|h^L(\mathbf{x}) - h_0^L(\mathbf{x})\|_2 \leq C^L R/\sqrt{m}$ 的指数界。本文创新性地取输入空间的 $1/(C^L\sqrt{m})$-覆盖 $D$，先在覆盖点上证明 $\|h^l(\mathbf{x}^j) - h_0^l(\mathbf{x}^j)\|_2 = \widetilde{O}(L^2 R/\sqrt{m})$，再通过最近覆盖点近似扩展到所有输入，最终得到 $\sup_\mathbf{x} \|h^l(\mathbf{x}) - h_0^l(\mathbf{x})\|_2 = \widetilde{O}(L^2 R/\sqrt{m})$。覆盖数虽可能指数级，但只需其对数，因此保留多项式深度依赖。

### 损失函数 / 训练策略

使用logistic损失 $\ell(z) = \log(1+\exp(-z))$ 进行分类。GD更新 $\mathbf{W}^l(k+1) = \mathbf{W}^l(k) - \eta \partial \mathcal{L}_S / \partial \mathbf{W}^l(k)$。使用对称初始化保证 $f_{\mathbf{W}(0)}(\mathbf{x}) = 0$。步长 $\eta \leq 4/(5L)$，迭代次数 $T$ 满足 $\eta T \asymp n$。

## 实验关键数据

### 主实验

本文为纯理论贡献。主要结果汇总如下：

| 工作 | 激活函数 | 宽度要求 | 泛化误差 | 网络深度 |
|------|---------|---------|---------|---------|
| Ji et al. 2020 | ReLU | $\widetilde{\Omega}(1/\gamma^8)$ | $\widetilde{O}(1/(\gamma^2\sqrt{n}))$ | 浅层 |
| Lei 2024 | ReLU | $\widetilde{\Omega}(1/\gamma^8)$ | $\widetilde{O}(1/(\gamma^2 n))$ | 浅层 |
| Chen et al. 2021 | ReLU | $\widetilde{\Omega}(L^{22}/\gamma^8)$ | $\widetilde{O}(e^{O(L)}\sqrt{m/n}/\gamma)$ | 深层 |
| Taheri 2025 | 光滑 | $\widetilde{\Omega}(1/\gamma^{6L+4})$ | $\widetilde{O}(e^{O(L)}/(n\gamma^2))$ | 深层 |
| **本文** | **ReLU** | $\widetilde{\Omega}(L^{16}/\gamma^8)$ | $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ | **深层** |

### 消融实验

| 技术贡献 | 改进幅度 | 说明 |
|---------|---------|------|
| 稀疏矩阵Rademacher界 | $4^L \to L^2$ | 消除深度指数依赖 |
| 消除宽度因子 | $\sqrt{m} \to \log$ | 使泛化界不显著依赖网络宽度 |
| 覆盖数Lipschitz分析 | $C^L \to L^2$ | 网络近初始化Lipschitz常数从指数降到多项式 |
| 过参数化要求 | $L^{22} \to L^{16}$ | 降低 $L^6$ 倍 |

### 关键发现

- 对NTK可分数据（margin $\gamma$），泛化速率 $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ 在忽略深度因子后匹配SVM最优速率 $\widetilde{O}(1/(n\gamma^2))$
- 关键突破在于将ReLU激活模式的变化编码为稀疏矩阵 $\hat{\mathbf{G}}$，并精确控制其范数

## 亮点与洞察

- 首次在深度ReLU网络上同时实现最优 $1/n$ 速率和多项式深度依赖，解决了一个长期开放问题
- 覆盖数策略是一个精巧的技术创新：虽然覆盖数本身可能是指数级的，但分析中只用到其对数
- 稀疏矩阵表示 $\hat{\mathbf{G}}_{L,0}^l$ 捕获了ReLU激活模式的变化，是比简单递推估计更精细的工具
- 参考模型方法比NTK-Gram矩阵分析更灵活，不需要研究核或对应的Gram矩阵

## 局限与展望

- 过参数化要求 $m = \widetilde{\Omega}(L^{16}/\gamma^8)$ 仍然较大，与实际训练中使用的网络宽度有较大差距
- 仅分析了分类任务（logistic损失），回归任务（如MSE损失）的最优泛化速率分析是自然延伸
- 对称初始化假设虽非本质限制但增加了理论复杂性，去掉后是否影响结论需验证
- 深度因子 $L^4(1+\gamma L^2)$ 能否进一步紧化尚不确定，猜测最优依赖可能是 $\text{poly}(L)$
- 仅分析GD，SGD的随机性引入额外方差需要新的技术处理
- NTK可分数据假设在实际数据上是否成立仍有争议

## 相关工作与启发

- NTK框架将深度网络与核方法联系，本文在NTK可分数据假设下获得最优速率
- 算法稳定性方法可以得到 $1/n$ 速率但受限于光滑激活函数，本文的Rademacher方法突破了这一限制
- 覆盖数+稀疏矩阵范数控制的技术组合可能推广到其他分段线性激活函数
- 与Taheri 2025的对比最具启发性：两者都追求 $1/n$ 速率，但稳定性方法天然需要光滑性
- 参考模型方法的优势：不需要NTK Gram矩阵正定（$\lambda_0 > 0$），后者在大样本下趋向零
- He初始化 $\mathbf{w}_r^l \sim \mathcal{N}(0, 2/m)$ 与本文的 $\sqrt{2/m}$ 缩放因子理论上等价
- Lazy training regime的实际相关性：本文证明所有迭代停留在初始化附近，但不依赖核或Gram矩阵分析
- 未来方向：将类似分析推广到CNN、ResNet等实际架构，以及SGD设定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 解决了深度ReLU网络泛化理论中的一个重要开放问题
- 实验充分度: ⭐⭐ 纯理论工作
- 写作质量: ⭐⭐⭐⭐ 对比表清晰，proof sketch帮助理解核心思想
- 价值: ⭐⭐⭐⭐⭐ 深度学习泛化理论的重要进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](chiqpm_calibrated_hierarchical_interpretable_image_classification.md)

</div>

<!-- RELATED:END -->

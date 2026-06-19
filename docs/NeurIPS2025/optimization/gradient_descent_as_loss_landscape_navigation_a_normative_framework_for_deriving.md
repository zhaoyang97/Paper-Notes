---
title: >-
  [论文解读] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules
description: >-
  [NeurIPS 2025][优化/理论][学习规则] 提出将学习规则视为在（部分可观测的）损失景观中的最优导航策略，通过变分法求解连续时间最优控制问题，在统一框架下推导出梯度下降、动量、自然梯度、Adam及持续学习策略。 - 学习规则（如梯度下降、Adam）通常是假设给定的，缺乏从第一性原理推导的统一理论基础 - 不同优化…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "学习规则"
  - "最优控制"
  - "损失景观导航"
  - "动量"
  - "自适应优化器"
---

# Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules

**会议**: NeurIPS 2025  
**arXiv**: [2510.26997](https://arxiv.org/abs/2510.26997)  
**代码**: 无  
**领域**: 优化理论 / 学习规则  
**关键词**: 学习规则, 最优控制, 损失景观导航, 动量, 自适应优化器

## 一句话总结

提出将学习规则视为在（部分可观测的）损失景观中的最优导航策略，通过变分法求解连续时间最优控制问题，在统一框架下推导出梯度下降、动量、自然梯度、Adam及持续学习策略。

## 研究背景与动机

- 学习规则（如梯度下降、Adam）通常是假设给定的，缺乏从第一性原理推导的统一理论基础
- 不同优化器（动量、自适应学习率、自然梯度等）各自独立发展，彼此关系不清
- 核心问题：为何某些学习规则优于另一些？在什么假设下某个规则是"最优"的？
- 本文提出规范化框架，将这些问题转化为最优控制问题，通过改变假设条件自然推导出不同的优化器

## 方法详解

### 整体框架

将学习过程建模为在损失景观中的连续时间最优导航问题。定义一个泛函目标：

$$J(\{\boldsymbol{\theta}_t\}) = \mathbb{E}\left\{\int_0^\infty \left(\frac{1}{2\eta}[\dot{\boldsymbol{\theta}}_t - \boldsymbol{f}(\boldsymbol{\theta}_t)]^T \boldsymbol{G}(\boldsymbol{\theta}_t)[\dot{\boldsymbol{\theta}}_t - \boldsymbol{f}(\boldsymbol{\theta}_t)] + k\hat{\mathcal{L}}_t(\boldsymbol{\theta}_t)\right) e^{-\gamma t} dt\right\}$$

其中包含三个核心要素：参数变化的代价（动能项）、损失函数（势能项）、以及时间折扣因子 $\gamma$。三个关键洞察构成框架基石：

1. **多步优化**：优化不仅看下一步，而是规划整个未来参数轨迹
2. **参数空间几何**：通过度量 $\boldsymbol{G}$ 编码参数空间的非欧几何
3. **部分可观测性**：损失景观只是部分可观测的，需要通过贝叶斯推断来估计

### 关键设计

1. **动量从多步优化自然涌现**:
    - 功能：从最简单的多步目标推导 Euler-Lagrange 方程
    - 核心思路：最优轨迹满足 $\dot{\boldsymbol{\theta}}_t = \boldsymbol{p}_t$，$\dot{\boldsymbol{p}}_t = \gamma \boldsymbol{p}_t + \eta k \nabla \mathcal{L}$，其中 $\boldsymbol{p}_t$ 即为动量
    - 设计动机：动量不需要额外假设，仅从"考虑多步"就自然产生；时间折扣 $\gamma$ 控制从动量到标准梯度下降的插值

2. **自然梯度来自参数空间几何**:
    - 功能：在目标中引入非欧度量 $\boldsymbol{G}(\boldsymbol{\theta})$（如 Fisher 信息矩阵）
    - 核心思路：度量 $\boldsymbol{G}$ 和 Hessian $\boldsymbol{H}$ 扮演根本不同的角色——$\boldsymbol{G}$ 定义环境几何，$\boldsymbol{H}$ 描述损失曲率
    - 设计动机：澄清了自然梯度并非"伪装的二阶方法"这一长久争论

3. **Adam 源于损失景观形状的贝叶斯推断**:
    - 功能：假设学习者维护对局部损失景观梯度 $\boldsymbol{m}_t$ 和曲率 $\boldsymbol{V}_t$ 的贝叶斯信念
    - 核心思路：通过 Ornstein-Uhlenbeck 先验建模景观信念的时间演化，推导出的最优更新形式为 $\Delta\boldsymbol{\theta} \propto \boldsymbol{V}_t^{-1/2} \boldsymbol{m}_t$
    - 设计动机：解释了 Adam 中平方根归一化的理论根据——它是弹道限（ballistic regime，即长期规划）的最优结果

### 损失函数 / 训练策略

- 通过 Euler-Lagrange 方程求解最优轨迹，不同假设对应不同的 EL 方程
- 三种极限情况对应三种学习规则：大 $\Delta t$ → Newton 法；小 $\Delta t$ + 大 $\gamma$ → 梯度下降；小 $\Delta t$ + $\gamma=0$ → 弹道规则
- 对持续学习：通过权重的分布估计（均值 $\mu_i$ + 方差 $v_i$）推导出方差敏感的学习动力学，解释了权重重置策略的合理性

## 实验关键数据

### 主实验（表格）

| 方法 | MNIST (MLP) 准确率 | CIFAR-10 (CNN) 准确率 |
|------|-------------------|---------------------|
| SGD | ~97% | ~68% |
| Adam | ~98% | ~73% |
| Ballistic (本文) | ~97.5% | ~71% |

### 消融实验

- 时间折扣率 $\gamma$ 的影响：$\gamma \to 0$（弹道极限）收敛更快但方案更长；$\gamma \gg 1$（过阻尼极限）退化为标准梯度下降
- 各向异性损失下不同 $\gamma$ 的收敛速率比：梯度下降中曲率差 4 倍则收敛速度差 4 倍，弹道规则中只差 $\sqrt{4}=2$ 倍
- 双井损失景观中：多步优化可以找到全局最小值，而梯度下降陷入局部最小值

### 关键发现

- 弹道学习规则（$\gamma \approx 0$）通常优于 SGD，但不一定优于 Adam
- 物理类比：学习过程 = 质量 $1/\eta$ 的粒子在势场 $k\mathcal{L}$ 中运动，$\gamma$ 为摩擦系数
- 漂移项 $\boldsymbol{f}$ 可导致非梯度学习规则成为最优——这为生物神经网络中的非梯度突触可塑性提供了理论支持

## 亮点与洞察

- **大统一视角**：首次在单一目标函数下统一推导出梯度下降、动量、自然梯度、Adam 和持续学习策略
- **物理-AI 桥接**：学习动力学与经典力学的精确对应，可利用 Noether 定理等物理工具分析学习算法
- Adam 中的平方根不是 bug 而是 feature——它是弹道（长期规划）假设的最优结果
- 自然梯度与二阶方法的本质区别得到澄清

## 局限与展望

- 框架中的"最优"未考虑内存需求、计算效率等实际约束
- 最优规则可能需要昂贵的矩阵运算（如 Hessian），实际可用性受限
- 仅在小规模数据集（MNIST/CIFAR-10）上验证弹道规则，大规模实验缺失
- 理论框架与具体经验情境的连接需要进一步探索

## 相关工作与启发

- 与 Wibisono et al. 的加速优化变分框架有类似精神，但关键区别在于目标函数的符号（本文是加和而非相减），使得最优轨迹是最小值而非驻点
- Khan & Rue 的自然梯度统一框架未纳入多步规划和部分可观测性
- 对生物神经网络中的 Hebbian/STDP 等非梯度规则提供了规范性解释

## 评分

- 理论创新：⭐⭐⭐⭐⭐
- 实验验证：⭐⭐⭐
- 实用价值：⭐⭐⭐
- 写作质量：⭐⭐⭐⭐⭐
- 综合评分：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[CVPR 2026\] Globscope: Toward a Global View of the Loss Landscape](../../CVPR2026/optimization/globscope_toward_a_global_view_of_the_loss_landscape.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

</div>

<!-- RELATED:END -->

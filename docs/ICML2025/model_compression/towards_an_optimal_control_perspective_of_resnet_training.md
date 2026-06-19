---
title: >-
  [论文解读] Towards an Optimal Control Perspective of ResNet Training
description: >-
  [ICML 2025][模型压缩][ResNet] 将 ResNet 训练形式化为最优控制问题，通过在中间层添加阶段成本 (stage cost) 损失实现自正则化，证明多余的深层权重渐近趋零，为理论驱动的层剪枝奠定基础。 - ResNet 的残差连接可视为连续时间神经 ODE 的欧拉前向离散化 - 从最优控制视角：数据前向…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "ResNet"
  - "optimal control"
  - "stage cost"
  - "剪枝"
  - "self-regularization"
---

# Towards an Optimal Control Perspective of ResNet Training

**会议**: ICML 2025  
**arXiv**: [2506.21453](https://arxiv.org/abs/2506.21453)  
**代码**: -  
**领域**: Theory / Neural Network Training  
**关键词**: ResNet, optimal control, stage cost, layer pruning, self-regularization  

## 一句话总结

将 ResNet 训练形式化为最优控制问题，通过在中间层添加阶段成本 (stage cost) 损失实现自正则化，证明多余的深层权重渐近趋零，为理论驱动的层剪枝奠定基础。

## 研究背景与动机

- ResNet 的残差连接可视为连续时间神经 ODE 的欧拉前向离散化
- 从最优控制视角：数据前向传播 = 动态系统状态轨迹，可训练参数 = 控制信号
- 现有最优控制训练方法仅适用于固定隐藏维度的架构 + 对损失函数有限制性假设
- **目标**：将阶段成本公式推广到标准 ResNet（含非平凡 skip connection）和通用损失函数

## 方法详解

### 1. ResNet 作为动态系统

标准 ResNet-N 的前向传播：

$$x_{k+1}^{(i)} = f_k(x_k^{(i)}, \mathbf{w}_k) = \begin{cases} x_k^{(i)} + \mathcal{F}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{F},k}), & k \notin \mathcal{K}_\mathcal{S} \\ \mathcal{S}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{S},k}) + \mathcal{F}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{F},k}), & k \in \mathcal{K}_\mathcal{S} \end{cases}$$

其中 $\mathcal{K}_\mathcal{S}$ 为具有非平凡 skip connection（1×1 卷积）的层索引集。

### 2. 中间输出头设计

**关键创新**：利用后续层的 skip connection 和输出头构造中间输出：

$$\hat{y}_k^{(i)} = \mathcal{H}(\mathcal{S}_{N-1}(\cdots \mathcal{S}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{S},k}), \cdots, \mathbf{w}_{\mathcal{S},N-1}), \mathbf{w}_\mathcal{H})$$

这意味着中间预测**复用主干网络既有参数**（skip connection 权重 + 输出头），无需额外参数。

### 3. 阶段成本训练目标

$$\min_\mathbf{w} J_N(\mathbf{w}) = \sum_{k=0}^{N-1} \gamma \mathcal{L}(\hat{\mathbf{y}}_k) + \mathcal{L}(\hat{\mathbf{y}})$$

当 $\gamma = 0$ 退化为标准训练。

### 4. 理论结果：渐近损失界

**定理 3.1**：设 ResNet-N 使用阶段成本 + 权重衰减训练，ResNet-M ($M<N$) 为其 SubResNet，则：

$$J_N(\mathbf{w}^N) \leq \sum_{k=0}^{M-1}\left[\gamma\mathcal{L}(\hat{\mathbf{y}}_k^M) + \frac{\lambda}{2}\|\mathbf{w}_{\mathcal{F},k}^M\|^2\right] + (1+\gamma(N-M))\bar{\mathcal{L}}$$

**含义**：当较浅的 SubResNet 已能完成学习任务时（$\bar{\mathcal{L}}$ 小），深层残差块的权重将趋近于零——网络自动发现所需的最优深度。

无权重衰减时的简化结果：

$$\mathcal{L}_\text{avg} = \frac{1}{N+1}\sum_{k=0}^N \mathcal{L}(\hat{\mathbf{y}}_k^N) \leq \bar{\mathcal{L}} + \frac{C}{N+1}$$

## 实验结果

### CIFAR-10 损失轨迹

- 阶段成本 ResNet 在 **12 个残差块后即达到良好拟合**（78.74% 测试精度）
- 标准训练仅在最终输出处有良好拟合
- 性能在同一阶段（相同滤波器数）内趋于平稳，在新阶段（增加滤波器）起始处跳升

### 层剪枝对比

| 模型 | MNIST | CIFAR-10 | CIFAR-100 |
|------|-------|----------|-----------|
| ResNet-54 标准训练 | 99.64 | 93.05 | 71.94 |
| SubResNet-12 标准训练 | 99.63 | 91.43 | 68.77 |
| ResNet-54 阶段成本 | 99.62 | 91.59 | 69.97 |
| SubResNet-12 剪枝 | 99.54 | 91.02 | 66.55 |

- 齐次模型中，剪枝后的 SubResNet-12 与标准训练的 SubResNet-12 差距仅 **≤3.5%**
- 标准训练的 ResNet 难以进行无损层剪枝

### 理论界的紧密性

实验验证定理 3.1 提供的界**相对紧密**。

## 亮点

- 将最优控制的阶段成本概念优雅地映射到标准 ResNet 架构
- 不需要额外参数——中间输出头复用 skip connection 和输出层
- 理论证明深层权重渐近趋零，为**理论驱动的层剪枝**奠定基础
- 训练动态的深度收敛分析提供了关于 ResNet 最优深度的新洞察
- 适用于通用损失函数（包括标准交叉熵）

## 局限性

- 实验仅限于 MNIST、CIFAR-10/100 等小规模数据集
- 标准 ResNet（非齐次维度）的层剪枝尚不直接
- 阶段成本权重 $\gamma$ 的选择缺乏理论指导
- 与 early exit 方法的联系讨论不够深入
- 未在大规模模型（如 ImageNet + ResNet-152）上验证

## 评分

⭐⭐⭐⭐ — 理论优美，将控制论视角系统化地应用于标准 ResNet，但实验规模限制了实际影响力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[ICML 2025\] Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination](best_subset_selection_optimal_pursuit_for_feature_selection_and_elimination.md)
- [\[ICML 2025\] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](rethinking_the_stability-plasticity_trade-off_in_continual_learning_from_an_arch.md)
- [\[ICML 2025\] Training a Generally Curious Agent (Paprika)](training_a_generally_curious_agent.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)

</div>

<!-- RELATED:END -->

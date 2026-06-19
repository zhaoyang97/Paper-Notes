---
title: >-
  [论文解读] Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection
description: >-
  [AAAI 2026][模型压缩][权重剪枝] 首次系统探索权重剪枝与核心集选择之间的交互关系，提出SWaST机制交替执行两者以建立协同效应，并设计状态保持机制解决"双重损失"问题，在10%–90% FLOPs削减下实现最高17.83%的精度提升。 问题背景 现代深度神经网络依赖大量模型参数和训练样本，计算开销巨大…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "权重剪枝"
  - "核心集选择"
  - "协同效应"
  - "训练加速"
  - "状态保持机制"
---

# Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection

**会议**: AAAI 2026  
**arXiv**: [2511.09901](https://arxiv.org/abs/2511.09901)  
**作者**: Weilin Wan, Fan Yi, Weizhong Zhang, Quan Zhou, Cheng Jin (Fudan University)
**代码**: 未公开  
**领域**: 模型压缩  
**关键词**: 权重剪枝, 核心集选择, 协同效应, 训练加速, 状态保持机制

## 一句话总结

首次系统探索权重剪枝与核心集选择之间的交互关系，提出SWaST机制交替执行两者以建立协同效应，并设计状态保持机制解决"双重损失"问题，在10%–90% FLOPs削减下实现最高17.83%的精度提升。

## 研究背景与动机

### 问题背景
现代深度神经网络依赖大量模型参数和训练样本，计算开销巨大。权重剪枝（weight pruning）和核心集选择（coreset selection）是两种主流的训练效率提升范式：前者移除冗余权重，后者筛选最具代表性的训练样本。

### 已有工作的不足
- 权重剪枝和核心集选择**始终被独立研究**，两者的交互关系被忽视
- 在经典机器学习（如SVM）中，特征筛选与样本筛选的协同效应已被充分研究，理论工具（KKT条件）可保证安全移除的正确性
- 深度学习中由于高度非凸的训练问题，缺乏类似的理论保证，导致两种范式被分开开发
- 同时进行剪枝和样本选择时可能出现**"双重损失"**现象——关键权重和其支持样本被同时误删，导致几乎不可逆的性能退化

### 核心动机
在深度学习中探索并利用权重剪枝与核心集选择之间的协同效应：冗余样本（尤其是噪声样本）使权重被过度调优、增加剪枝难度；冗余权重倾向于过拟合噪声数据、破坏核心集选择的有效性。这种双向干扰暗示了将两者联合优化的潜力。

## 方法详解

### 关键设计1：交替优化机制 SWaST

SWaST（Simultaneous Weight and Sample Tailoring）包含预热阶段和交替优化阶段：

1. **预热阶段**：在完整数据集上训练 $\mathcal{K}$ 个epoch，建立良好初始化
2. **交替优化阶段**：每 $\mathcal{R}$ 个epoch执行一次核心集选择，识别最具代表性的样本；随后在选出的核心集上进行训练并执行在线剪枝

两个变体根据剪枝策略的激进程度而区分：

- **SWaST-trim**：仅剪枝全连接层，保留大部分参数以确保训练稳定性，效率提升主要来自核心集选择
- **SWaST-cut**：对整个网络进行全局剪枝，效率增益更大，但可能导致训练不稳定（"双重损失"问题）

核心集选择难度可通过以下指标衡量：

$$\mathcal{I}(\mathcal{D}, \hat{\mathcal{D}}) = \sup_{\boldsymbol{\theta}} \frac{|\mathcal{L}(\boldsymbol{\theta}) - \hat{\mathcal{L}}(\boldsymbol{\theta})|}{\mathcal{L}(\boldsymbol{\theta})}$$

实验表明 $\mathcal{I}(\mathcal{D}, \hat{\mathcal{D}})$ 随多项式阶数（即模型参数维度）指数级增长，说明剪枝减少模型规模可显著降低核心集选择难度。

### 关键设计2：状态保持机制（State Preservation）

针对"双重损失"问题，设计了两阶段的状态保持机制：

**阶段1 — 状态记录**（每 $\mathcal{R}$ 个epoch执行）：核心集更新后，通过前向传播捕获模型状态：

$$\tilde{\mathcal{D}} = \{(\mathbf{x}_i, \mathbf{z}_i) : \mathbf{z}_i = f_{\boldsymbol{\theta}_{\text{pre}}}(\mathbf{x}_i), \mathbf{x}_i \in \mathcal{X}\}$$

**阶段2 — 状态约束训练**：通过复合损失函数强制状态一致性：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{(\mathbf{x}_i, \mathbf{z}_i) \in \tilde{\mathcal{D}}} \text{KL}(\sigma(\mathbf{z}_i) \| \sigma(f_{\boldsymbol{\theta}}(\mathbf{x}_i)))$$

其中 $\lambda=0.1$ 平衡主学习目标与状态一致性，$\sigma$ 为softmax函数。KL散度项可检测因错误剪枝导致的分布偏移，并在后续步骤中纠正，从而稳定联合优化过程。

### 关键设计3：协同效应的理论分析

通过多项式插值任务进行透明分析，揭示两个关键观察：

1. **冗余样本阻碍剪枝**：过多样本（尤其是噪声样本）使多项式系数被过度调优以拟合所有样本，基于权重幅值的剪枝方法难以识别不相关权重
2. **冗余权重阻碍核心集选择**：在标准度量下，核心集选择难度随冗余模型权重数量指数级增长，因为 $\mathcal{L}(\boldsymbol{\theta})$ 由于对噪声数据的过拟合而快速趋向零

## 实验关键数据

### 实验1：SWaST-trim 不同剪枝率与核心集大小（ResNet-18, CIFAR-10/100）

| 数据集 | 核心集大小 | 剪枝率90% | 剪枝率50% | 剪枝率10% | 仅核心集 |
|--------|-----------|----------|----------|----------|---------|
| CIFAR-10 | 10% | 92.58 (+0.38) | 92.40 (+0.20) | 92.29 (+0.09) | 92.20 |
| CIFAR-10 | 5% | 90.16 (+0.49) | 89.92 (+0.25) | 89.77 (+0.10) | 89.67 |
| CIFAR-10 | 1% | 78.60 (+2.97) | 77.14 (+1.61) | 76.01 (+0.48) | 75.53 |
| CIFAR-100 | 10% | 71.94 (+0.97) | 71.44 (+0.47) | 71.20 (+0.23) | 70.97 |
| CIFAR-100 | 5% | 66.19 (+1.40) | 65.56 (+0.77) | 65.09 (+0.30) | 64.79 |
| CIFAR-100 | 1% | 38.38 (+1.96) | 37.58 (+1.16) | 36.88 (+0.46) | 36.42 |

**规律**：核心集越小，SWaST的增益越显著（1%核心集下提升最大），验证了剪枝对核心集选择的增强效果。

### 实验2：SWaST-cut 全局剪枝（ResNet-101, 多数据集）

| 数据集 | 核心集大小 | 剪枝率90% | 剪枝率50% | 剪枝率30% | 仅核心集 |
|--------|-----------|----------|----------|----------|---------|
| CIFAR-10 | 1% | 64.57 (+14.85) | 66.92 (+17.20) | **67.55 (+17.83)** | 49.72 |
| CIFAR-100 | 5% | 59.27 (+5.39) | 62.87 (+8.99) | 62.97 (+9.09) | 53.88 |
| TinyImageNet | 5% | 38.65 (+3.55) | 42.93 (+7.83) | 41.94 (+6.84) | 35.10 |
| ImageNet-1K | 10% | 37.55 (+5.83) | 38.92 (+7.20) | 39.19 (+7.47) | 31.72 |
| ImageNet-1K | 5% | 31.63 (+1.35) | 32.71 (+2.43) | 34.34 (+4.06) | 30.28 |

**关键发现**：ResNet-101 + CIFAR-10 + 1%核心集下，SWaST-cut在30%剪枝率实现了**17.83%**的最大精度提升；中等剪枝率（30%–50%）通常最优，过度剪枝（90%）有时反而降低性能。

### 实验3：核心集选择改善剪枝效果

在ResNet-18/CIFAR-10上，SWaST-cut在90%剪枝率下达到93.15%精度，比仅剪枝基线提升3.33%。在相同FLOPs预算下，SWaST精度领先最多4.43%（相对FLOPs=0.01时）。

### 实验4：噪声抵抗与核心集质量

- SWaST-cut使最终核心集中的噪声比例降低了**10.62%**
- 剪枝率越高，过拟合（测试损失-验证损失之差）减少越明显
- 剪枝后模型在噪声样本上的损失更高，表明剪枝有效阻止了对错误模式的记忆

## 关键发现

1. **协同效应确实存在**：权重剪枝和核心集选择在深度学习中具有显著的双向增益——剪枝降低核心集选择难度，核心集选择提升剪枝效果
2. **小核心集获益最大**：核心集越小，剪枝带来的精度提升越显著，因为小核心集面临更大的选择挑战和过拟合风险
3. **双重损失现象**：同时激进地移除权重和样本时，关键权重及其支持样本可能被同时误删，导致不可逆退化——这是深度学习特有的问题
4. **状态保持机制有效**：KL散度约束可检测错误剪枝并在后续步骤中纠正，显著稳定联合优化

## 亮点

- **首创性交互分析**：首次在深度学习中系统探索权重剪枝与核心集选择的交互关系，从经典ML的联合筛选理论出发
- **透明的理论洞察**：通过多项式插值任务提供可解释的分析，揭示冗余样本与权重的双向干扰机制
- **双重损失问题的发现与解决**：识别出深度学习特有的联合优化陷阱，并用状态保持机制有效缓解
- **通用框架设计**：SWaST可搭配任意在线剪枝算法和核心集选择方法，具有良好的灵活性
- **显著的实验增益**：最高17.83%精度提升，同时实现10%–90%的FLOPs削减

## 局限与展望

- **仅覆盖非结构化剪枝**：当前仅使用RigL等非结构化方法，未验证结构化剪枝（如通道/滤波器剪枝）的协同效应
- **核心集选择方法有限**：主要使用GradMatch、Moderate和EL2N三种方法，未覆盖更新型的方法
- **状态保持开销**：需要额外的前向传播来记录logits和计算KL散度，增加了训练成本
- **缺乏理论保证**：协同效应仅通过实验验证，未建立深度学习中的理论分析框架
- **超参数敏感性**：$\lambda$、$\mathcal{R}$、$\mathcal{K}$等超参数的选择缺乏系统指导
- **大规模验证不足**：ImageNet-1K实验规模有限，未在更大模型（如ViT、GPT）上验证

## 与相关工作的对比

- **经典ML联合筛选** (Shibagaki et al. 2016; Zhang et al. 2017)：在SVM等凸模型中利用KKT条件安全移除特征和样本，本文将此思想扩展到非凸的深度学习
- **GradMatch** (Killamsetty et al. 2021)：基于梯度匹配的核心集选择，本文将其作为CIFAR实验的默认选择方法
- **RigL** (Evci et al. 2020)：动态稀疏训练方法，在训练过程中更新稀疏拓扑，本文作为默认剪枝算法
- **EL2N** (Paul et al. 2021)：基于error L2-norm的样本重要性度量，本文用于ImageNet-1K和噪声实验
- **Moderate** (Xia et al. 2022)：基于样本间距离的核心集选择，本文用于TinyImageNet实验
- **知识蒸馏相关**：状态保持机制与自蒸馏有相似之处，但目标不同——前者保持联合优化稳定性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统化联合探索剪枝与核心集选择的协同效应，"双重损失"问题的发现有启发性
- 实验充分度: ⭐⭐⭐⭐ — 覆盖CIFAR-10/100、TinyImageNet、ImageNet-1K，多架构多设定，消融全面
- 写作质量: ⭐⭐⭐⭐ — 从经典ML到深度学习的叙事线清晰，多项式实验提供了直观理解
- 价值: ⭐⭐⭐⭐ — 揭示了训练效率优化中被忽视的联合优化方向，实用性强但缺乏大模型验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection](../../CVPR2026/model_compression/fast_topology-aware_frequency-domain_distribution_matching_for_coreset_selection.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](../../ICML2026/model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ACL 2025\] Mitigating Selection Bias with Node Pruning and Auxiliary Options](../../ACL2025/model_compression/selection_bias_node_pruning.md)

</div>

<!-- RELATED:END -->

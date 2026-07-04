---
title: >-
  [论文解读] Conformal Online Learning of Deep Koopman Linear Embeddings
description: >-
  [NEURIPS2025][LLM评测][Koopman operator] 提出 COLoKe 框架，将 conformal prediction 重新解读为模型一致性诊断工具，仅在 Koopman 模型的预测误差超过动态校准阈值时才触发参数更新，从而实现对非线性动力系统的高效在线 Koopman 线性嵌入学习。
tags:
  - "NEURIPS2025"
  - "LLM评测"
  - "Koopman operator"
  - "在线学习"
  - "共形预测"
  - "dynamical systems"
  - "深度学习"
---

# Conformal Online Learning of Deep Koopman Linear Embeddings

**会议**: NEURIPS2025  
**arXiv**: [2511.12760](https://arxiv.org/abs/2511.12760)  
**代码**: [ben2022lo/COLoKe](https://github.com/ben2022lo/COLoKe)  
**领域**: LLM评测  
**关键词**: Koopman operator, 在线学习, 共形预测, dynamical systems, 深度学习

## 一句话总结
提出 COLoKe 框架，将 conformal prediction 重新解读为模型一致性诊断工具，仅在 Koopman 模型的预测误差超过动态校准阈值时才触发参数更新，从而实现对非线性动力系统的高效在线 Koopman 线性嵌入学习。

## 背景与动机
Koopman 算子理论提供了一种将非线性动力系统提升到无穷维函数空间中进行线性分析的强大框架。其核心思想是：虽然状态空间中的动力学是非线性的，但 observable 函数沿轨迹的演化可以被一个线性算子（即 Koopman 算子）精确描述。在有限维近似中，这等价于寻找一个特征映射 $\Phi$ 和一个矩阵 $K$，使得 $\Phi(x_{t+1}) = K \Phi(x_t)$。

现有方法的主要局限：

- **离线方法**（DMD、EDMD、深度 Koopman 自编码器等）假设可以一次性访问全部数据，不适用于流式数据场景
- **在线方法**（Online DMD、Online EDMD）依赖线性 observable 或固定字典，表达能力有限
- **基于神经网络的在线方法**（OnlineAE、DKLT 等）缺乏原则性的更新策略，通常每步执行固定次数的梯度更新，不论是否真正需要，容易导致过拟合或计算浪费

实际应用中（机器人控制、实时预测、在线监测等），数据是流式到达的，且系统可能随时间发生分布漂移。因此需要一种既能自适应更新、又能避免不必要计算的在线学习策略。

## 核心问题
如何在流式数据场景下，自适应地决定 **何时更新** 以及 **更新多少** Koopman 嵌入模型的参数，在保持长期预测精度的同时避免过拟合和计算浪费？

## 方法详解

### 1. 深度 Koopman 嵌入架构
特征映射设计为保留原始状态的结构：
$$\Phi_{\theta}(x) = [x, \tilde{\Phi}_{\theta}(x)]^\top$$
其中 $\tilde{\Phi}_{\theta}$ 是可学习的神经网络部分。这种设计使提升后的表示既保留了原始状态信息，又通过学习到的非线性嵌入增强了表达能力，同时消除了对显式解码器的需求。

### 2. 多步预测损失
在每个时间步 $t$，维护一个大小为 $w$ 的滑动窗口 $\mathcal{D}_t = \{x_{t-w}, \ldots, x_t\}$。训练损失为窗口内所有有效多步预测对的累积误差：
$$\mathcal{L}_t(\theta, K) = \sum_{(s,\tau) \in \mathcal{I}_t} \sum_{j=1}^{\tau} \|\Phi_\theta(x_{s+\tau}) - K^j \Phi_\theta(x_{s+\tau-j})\|^2$$
多步预测有助于识别持久的谱模式和近似 Koopman 本征函数，从而提升长期预测能力。

### 3. Conformal 更新机制（核心创新）
传统 conformal prediction 用于构建预测区间来量化不确定性。本文将其**重新解读**为模型一致性诊断工具：

**预测一致性评分**：定义当前模型在新观测 $x_t$ 上的一致性评分为：
$$s_t = \ell_{t-w,w}(\theta_t, K_t) = \sum_{\tau=1}^{w} \|\Phi_{\theta_t}(x_t) - K_t^\tau \Phi_{\theta_t}(x_{t-\tau})\|^2$$

**自适应阈值**：采用 Conformal PI 控制动态调整阈值 $q_t$：
$$q_{t+1} = q_t + \gamma(e_t - \alpha) + r_t\left(\sum_{i=1}^{t}(e_i - \alpha)\right)$$
其中 $e_t = \mathbf{1}\{s_t > q_t\}$ 是二值误差信号，$\gamma$ 是学习率，$r_t$ 是积分校正项。

**更新决策**：
- 若 $s_t \leq q_t$：模型仍然与数据一致，**不更新**
- 若 $s_t > q_t$：模型不再一致，**触发梯度更新**，持续迭代直到 $s_t \leq q_t$

关键视角转换：不是评估新数据点是否符合模型（传统 conformal prediction），而是评估当前模型参数是否仍与新数据一致。

### 4. 理论保证
在标准的在线学习假设（损失函数光滑、动态 oracle 路径有界变分）下，证明了动态遗憾界：
$$\sum_{t=1}^{T}[\mathcal{L}_t(\theta_t, K_t) - \mathcal{L}_t(\theta_t^*, K_t^*)] \leq \mathcal{O}(\alpha h(T) + V_T + S_T)$$
其中 $h(T)$ 是次线性函数，$V_T$ 和 $S_T$ 分别是 oracle 路径的总变分和平方变分。

## 实验关键数据

### 数据集
- **合成数据**：Single Attractor、Duffing oscillator、Van der Pol oscillator、Lorenz system
- **真实数据**：Electricity Transformer (ETD)、EEG Motor Movement、Turbulence (CASES-99)

### 主要结果（Table 2，泛化误差）

| 数据集 | ODMD | OnlineAE | OLoKe (无conformal) | **COLoKe** |
|--------|------|----------|---------------------|------------|
| Single Attractor | 1.1e-3 | 1.0e-2 | 2.1e-6 | **2.4e-7** |
| Duffing | 2.5e-4 | 8.7e-3 | 5.5e-5 | **3.1e-6** |
| VdP | 2.1e-3 | 1.7e-2 | 6.6e-4 | **3.8e-4** |
| Lorenz | 2.7e-1 | 5.9e-1 | 7.6e-3 | **6.5e-3** |

### 关键发现
- COLoKe 在所有合成和真实数据集上均取得最优或接近最优性能
- 在混沌 Lorenz 系统上，(C)OLoKe 相比基线方法提升近 **两个数量级**
- 相比固定步数更新的 OLoKe，conformal 触发机制系统性地带来改善
- 谱分析验证：在解析可求解系统上，COLoKe 恢复了准确的 Koopman 本征值（例如真实值 $\{-1, -0.05, -0.1\}$，估计为 $\{-1.0091, -0.04996, -0.1001\}$）

### 计算效率
在高维 EEG 数据集上的 Pareto 前沿分析表明，COLoKe 在不需要手动调优的情况下，同时实现了更低的在线误差和更短的执行时间，优于所有固定迭代次数的 OLoKe 变体（1/5/10/50/100/150 步）。

## 亮点
1. **视角创新**：将 conformal prediction 从"评估新数据是否符合模型"重新解读为"评估当前模型是否仍与数据一致"，这一视角转换简洁而深刻
2. **自适应更新**：自动决定何时更新和更新多少，无需手动调优迭代次数，同时避免过拟合
3. **无解码器设计**：通过在提升表示中嵌入原始状态，将重建约束隐式融入一致性损失，消除了对解码器的需求
4. **理论支撑**：提供了动态遗憾界的理论分析
5. **方法通用**：conformal 在线学习框架不限于 Koopman 场景，可推广到更一般的在线非凸学习

## 局限与展望
1. **假设 (A3) 未从第一原理导出**：累积阈值次线性增长的条件仅有实验支持，尚未严格证明，作者承认这是开放问题
2. **仅处理自治系统**：当前方法假设动力学由时不变映射 $T$ 驱动，未覆盖非自治系统（时变外力驱动的系统）
3. **依赖窗口大小 $w$ 的选择**：窗口大小是预设超参数，对不同系统可能需要不同设置
4. **线性 Koopman 假设的内在局限**：对于高度非线性/混沌系统，有限维线性近似的精度天花板仍然存在
5. **缺少与离线深度 Koopman 方法的全面对比**：仅在一个系统上与离线方法做了简单比较

## 与相关工作的对比

| 方法 | 无需历史 | 在线更新 | 自适应嵌入 | 内置重建 |
|------|---------|---------|-----------|---------|
| Online DMD | ✓ | ✓ | ✗ | ✓ |
| R-EDMD | ✗ | ✓ | ✗ | ✗ |
| OnlineAE | ✓ | ✓ | ✓ | ✗ |
| DKLT | ✓ | 仅批量 | ✓ | ✗ |
| **COLoKe** | **✓** | **✓** | **✓** | **✓** |

COLoKe 是唯一同时满足四个特性的方法。相比 OnlineAE（固定步数更新策略），COLoKe 通过 conformal 机制实现了原则性的自适应更新。

## 启发与关联
- **Conformal prediction 的新应用范式**：传统用于不确定性量化和预测集构建，本文将其用作模型诊断/触发条件，这一思路可迁移到其他在线学习场景（如在线微调 LLM、自适应推理等）
- **"何时更新"问题**：在持续学习、在线适应等领域，这是一个普遍但常被忽视的问题，conformal score 提供了一种优雅的自动判断机制
- **Koopman 与深度学习结合**：将算子理论的结构化先验（线性演化）与深度学习的灵活性结合，是一个有前景的技术路线

## 评分
- 新颖性: ⭐⭐⭐⭐ — conformal prediction 作为模型更新触发条件的视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖合成和真实数据，包含谱分析验证和效率对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数学表述严谨
- 价值: ⭐⭐⭐⭐ — 框架通用性强，对在线学习社区有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](../../ICLR2026/learning_theory/deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Avoiding Catastrophe in Online Learning by Asking for Help](../../ICML2025/learning_theory/avoiding_catastrophe_in_online_learning_by_asking_for_help.md)

</div>

<!-- RELATED:END -->

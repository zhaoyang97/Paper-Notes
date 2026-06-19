---
title: >-
  [论文解读] Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena
description: >-
  [ICML2025][layerwise linear models] 提出**动态反馈原则 (Dynamical Feedback Principle)**，论证逐层线性模型（layerwise linear models）足以统一解释 neural collapse、emergence、lazy/rich regime 和 grokking 四大深度学习动力学现象，呼吁优先研究逐层结构而非非线性激活。
tags:
  - "ICML2025"
  - "layerwise linear models"
  - "dynamical feedback principle"
  - "neural collapse"
  - "emergence"
  - "lazy/rich regime"
  - "grokking"
---

# Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena

**会议**: ICML2025  
**arXiv**: [2502.21009](https://arxiv.org/abs/2502.21009)  
**代码**: 无（理论/立场论文）  
**领域**: 理论（深度学习理论、优化动力学）  
**关键词**: layerwise linear models, dynamical feedback principle, neural collapse, emergence, lazy/rich regime, grokking

## 一句话总结

提出**动态反馈原则 (Dynamical Feedback Principle)**，论证逐层线性模型（layerwise linear models）足以统一解释 neural collapse、emergence、lazy/rich regime 和 grokking 四大深度学习动力学现象，呼吁优先研究逐层结构而非非线性激活。

## 研究背景与动机

深度神经网络（DNN）是复杂的非线性动力系统，直接分析极为困难。物理学中常将复杂系统简化为可解的最小模型（如将牛建模为球体、线性化摆运动）。类比地，**逐层线性模型**（如线性神经网络）虽然去掉了非线性激活，但其梯度流动力学本身是非线性的。

近年来，DNN 中涌现了多种难以解释的动力学现象：

- **Neural Collapse**：分类网络最后一层特征坍缩为低秩单纯形结构
- **Emergence**：大语言模型随规模突然获得新能力
- **Lazy/Rich Regime**：网络在线性动力学和非线性动力学之间切换
- **Grokking**：训练精度早已完美但泛化严重延迟

这些现象看似毫无关联，但本文论证它们都源于**逐层参数乘积结构**产生的动态反馈。

## 方法详解

### 核心：动态反馈原则

以对角线性网络 $f(x) = \sum_i x_i a_i b_i$ 为例，梯度流方程为：

$$\frac{da_i}{dt} = -b_i \mathbf{E}[x_i^2](a_ib_i - S_i), \quad \frac{db_i}{dt} = -a_i \mathbf{E}[x_i^2](a_ib_i - S_i)$$

关键观察：$a_i$ 的大小控制 $b_i$ 的变化速率，反之亦然——形成**动态反馈**。对比无隐层线性模型 $\frac{d\theta_i}{dt} = -\mathbf{E}[x_i^2](\theta_i - S_i)$，后者参数独立演化，无反馈效应。

### 守恒量

由梯度方程的对称性可得守恒量：$a_i^2 - b_i^2 = \mathcal{C}_i$，在训练过程中保持恒定。推广到矩阵形式：$W_2 W_2^T - W_1^T W_1$ 守恒。

### 现象一：Emergence ← 放大动力学 + Sigmoid 饱和

小初始化下 ($a_i(0) = b_i(0) \ll 1$)，每个模态遵循 sigmoid 饱和曲线：

$$a_i(t)b_i(t)/S_i = \frac{1}{1 + \left(\frac{S_i}{a_i(0)b_i(0)} - 1\right) e^{-2S_i \mathbf{E}[x_i^2] t}}$$

与线性模型的指数饱和 $\theta_i(t)/S_i = 1 - e^{-\mathbf{E}[x_i^2]t}$ 不同，sigmoid 动力学导致**阶段式训练**——不同模态按相关性大小依次学习，产生突变式能力涌现。

### 现象二：Neural Collapse ← 贪婪低秩动力学

线性神经网络 $f = x^T W_1 W_2$ 的动力学解耦为 $c$ 个独立模态（$c$ 为输出维度），每个模态同样遵循 sigmoid 饱和。网络优先学习与目标最相关的特征（最大奇异值），产生**最小秩偏好**。最终层特征矩阵 $XW_1$ 的秩坍缩到 $c$，形成 simplex ETF 结构。

### 现象三：Lazy/Rich Regime ← 层间不平衡

引入 $\lambda$-balanced 条件 $W_2 W_2^T - W_1^T W_1 = \lambda I$：

- $|\lambda| \approx 0$（平衡层）→ 非线性贪婪动力学 → **Rich regime**
- $|\lambda| \gg 0$（不平衡层）→ 仅轻层训练，线性动力学 → **Lazy regime**

### 现象四：Grokking ← 权重-目标比

定义权重-目标比 $\Sigma_0 / S$，其中 $\Sigma_0 = \sum_i \frac{a_i(0)^2 + b_i(0)^2}{2Z}$ 为初始权重尺度。关键量：

$$\gamma_+ = \frac{S + \sqrt{\Sigma_0^2 - \mathcal{S}_0^2 + S^2}}{\Sigma_0 + \mathcal{S}_0}$$

- $\gamma_+ \gg 1$ → 特征间差异大 → Rich regime → 快速泛化
- $\gamma_+ \approx 1$ → 特征几乎不变 → Lazy regime → Grokking（延迟泛化）

降低 $\Sigma_0/S$ 的方法（权重缩小、目标放大、输入缩小、输出缩小）都能消除 grokking。

## 实验关键数据

| 现象 | 简化模型 | 核心机制 | 实际验证 |
|------|---------|---------|---------|
| Emergence | 对角线性网络 + 预建技能函数 | Sigmoid 饱和 + 阶段式训练 | 2层 ReLU 网络，多任务稀疏奇偶问题 |
| Neural Collapse | 线性神经网络 (UFM) | 贪婪低秩动力学 | ResNet18 on CIFAR10，特征坍缩为 9-simplex ETF |
| Lazy/Rich | $\lambda$-balanced 线性网络 | 层间不平衡控制 | CNN 上游初始化改善特征学习和可解释性 |
| Grokking | 标量输入输出线性网络 | 权重-目标比 $\Sigma_0/S$ | 4层 tanh MLP on MNIST，Transformer on 模运算 |

消除 grokking 的实验结果（4层 tanh MLP, 1000 MNIST 样本）：

| 方法 | 修改 | 效果 |
|------|------|------|
| 默认设置 | 大权重初始化 | 出现明显 grokking |
| 权重缩小 | 降低 $\Sigma_0$ | 消除泛化延迟 |
| 目标放大 | 增大 $S$ | 消除泛化延迟 |
| 输入缩小 | 等效增大 $S/\Sigma_0$ | 消除泛化延迟 |
| 输出缩小 | 增加 $Z$ 降低 $\Sigma_0$ | 消除泛化延迟 |

## 亮点与洞察

1. **统一性极强**：用一个「动态反馈原则」串联四个看似无关的现象，物理直觉优美
2. **可解性**：逐层线性模型在适当假设下可得精确解析解，避免近似带来的误导
3. **实用指导**：理论直接给出消除 grokking 的操作方案（缩小权重、放大目标等）
4. **Scaling Laws 预测**：通过 sigmoid 动力学 + 幂律技能分布成功预测 2 层网络的 scaling laws
5. **深度效应**：更深网络使 sigmoid 曲线趋近阶跃函数，加剧贪婪动力学，解释 Lottery Ticket Hypothesis

## 局限与展望

1. **表达力差距**：逐层线性模型无法拟合实际数据，其动力学结论向非线性 DNN 的迁移缺乏严格保证
2. **假设较强**：多数结论依赖小初始化、白化输入、特定守恒结构等理想化条件
3. **Position Paper 属性**：缺少大规模实验（如 LLM 训练）的直接验证，更多是类比论证
4. **非线性效应被低估**：某些现象（如 double descent、feature selection in ReLU 网络）可能本质需要非线性
5. **multi-layer 推广困难**：多数精确解限于 2 层，更深网络的解析处理仍然开放

## 相关工作与启发

- **Saxe et al. (2014)**：线性网络精确动力学解的奠基工作
- **Papyan et al. (2020)**：Neural Collapse 的实证发现
- **Nam et al. (2024)**：用对角线性网络 + 预建技能解释 emergence 和 scaling laws
- **Kunin et al. (2024)**：层间不平衡控制 grokking；更深网络偏向 $L_1$ 范数解
- **Dominé et al. (2025)**：$\lambda$-balanced 模型的 lazy/rich regime 分析
- **Mixon et al. (2020) & Fang et al. (2021)**：Unconstrained Feature Model 分析 neural collapse

## 评分

- 新颖性: ⭐⭐⭐⭐ — 动态反馈原则作为统一框架是新颖贡献，但构建块多来自已有工作
- 实验充分度: ⭐⭐⭐ — Position paper 性质，实验以演示为主，缺少大规模验证
- 写作质量: ⭐⭐⭐⭐⭐ — 物理直觉清晰，从简单模型到复杂现象的叙述逻辑优美
- 价值: ⭐⭐⭐⭐ — 为理解 DNN 动力学提供统一视角，对理论研究有较好指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](../../NeurIPS2025/others/modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](../../ACL2025/others/using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](../../ICLR2026/others/a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[ICML 2025\] GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs](gpu-friendly_and_linearly_convergent_first-order_methods_for_certifying_optimal_.md)
- [\[ICML 2025\] Access Controls Will Solve the Dual-Use Dilemma](access_controls_will_solve_the_dual-use_dilemma.md)

</div>

<!-- RELATED:END -->

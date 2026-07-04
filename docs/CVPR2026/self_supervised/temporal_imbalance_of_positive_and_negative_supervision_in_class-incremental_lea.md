---
title: >-
  [论文解读] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning
description: >-
  [CVPR2026][自监督学习][Class-Incremental Learning] 提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。
tags:
  - "CVPR2026"
  - "自监督学习"
  - "Class-Incremental Learning"
  - "catastrophic forgetting"
  - "Temporal Imbalance"
  - "Loss Reweighting"
  - "continual learning"
---

# Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning

**会议**: CVPR2026  
**arXiv**: [2603.02280](https://arxiv.org/abs/2603.02280)  
**代码**: 待确认  
**领域**: 自监督  
**关键词**: Class-Incremental Learning, catastrophic forgetting, Temporal Imbalance, Loss Reweighting, continual learning

## 一句话总结

提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

## 研究背景与动机

**类增量学习（CIL）的核心挑战**：模型需顺序学习新类，但旧类数据不可再访问，导致灾难性遗忘——模型对新类预测偏差严重。

**现有方法局限于类别不平衡视角**：主流方法（Balanced Fine-tuning、Prototype-based Classifier、Output Layer Calibration）将预测偏差归因于新旧类的类别不平衡，仅在分类头层面做修正。

**忽视了时序不平衡**：即使旧类之间样本数相同，因正样本到达时间不同，较早的类在训练后期累积了更多负监督，导致 precision 高但 recall 低的不对称现象。

**时序偏差影响全局参数**：CIL 中训练数据的时序顺序引入系统性偏差，不仅局限于分类头，而是影响整个模型（含 backbone 特征空间）。

**图1 的关键示例**：Task 2 中类 A 和类 B 样本数相同，但类 A 的正样本集中在 Task 0，类 B 在 Task 1——类 A 遭受更严重的遗忘，证明类别平衡不能解释所有偏差。

**缺乏损失层面的时序建模**：尽管时序衰减在时间序列预测、强化学习、在线学习等领域广泛使用，CIL 中尚无在损失函数层面显式建模正负监督时序不平衡的工作。

## 方法详解

### 整体框架

这篇工作要解决的是类增量学习里一个被忽视的偏差：旧类不是因为样本变少才被遗忘，而是因为它们"出现得早"，在后续每个任务里都被当成负样本反复压制，最终落入高 precision、低 recall 的尴尬。TAL 的整条思路就围着这一点转：先给每个类装一个能在线追踪"近期被当正样本程度"的标量 $Q_k[N]$，用指数衰减记忆核让久远的监督逐渐淡出；再把这个标量塞进交叉熵的分母，按它的大小动态给负监督降权——旧类的负监督被松绑，新类的负监督照旧。整个过程不动模型架构，只换损失函数。

### 关键设计

**1. 时序正监督强度 $Q_k[N]$：用指数衰减记忆核追踪每个类近期"被当正样本"的程度**

要量化"旧类越早出现、后期累积的负监督越多"这件事，作者为每个类 $k$ 维护一个标量状态 $Q_k[N]$。做法是把每一步的监督极性写成序列 $a_k[n]\in\{+1,-1\}$（这一步的样本属于类 $k$ 记 +1，否则记 -1），再用指数衰减核 $f[n]=\lambda^{n+1}$（$0<\lambda<1$，$\lambda$ 是记忆参数）对历史做加权卷积：$Q_k[N]=\sum_{n=0}^{N-1} f[N-1-n]\,a_k[n]$。越近的监督权重越大、越久远的越被遗忘。它的妙处是有递推形式 $Q_k[N+1]=\lambda\big(Q_k[N]+a_k[N]\big)$，每步只需 $\mathcal{O}(1)$ 的时间和空间就能在线更新，取值上界为 $Q_{\max}=\lambda/(1-\lambda)$。于是一个长期不再出现正样本的旧类，$Q$ 会衰减到接近 0；而正在学习的新类 $Q$ 稳定在 $Q_{\max}$ 附近——一个标量就把"时序新鲜度"压成了可直接使用的信号。

**2. 时序不平衡定理（Theorem 1）：样本一样多，来得早的类注定吃更多负监督**

有了 $Q$ 这把尺子，作者把直觉证成了定理：两个样本总数相同的类，正样本平均出现得更早的那个，在训练结束时 $Q$ 值更小。$Q$ 小意味着它近期几乎只在被当负样本压制，于是呈现"高 precision、低 recall"的不对称——模型不敢把样本判给它，判了大多对，却漏判很多。这正好解释了图1 的反常：类 A 和类 B 样本数相同，仅因 A 的正样本集中在更早的 Task 0、B 在 Task 1，A 就遭受更严重的遗忘。换句话说，仅看类别是否平衡解释不了全部偏差，监督在时间轴上的不均才是另一个真正的源头。

**3. Temporal-Adjusted Loss：按 $Q$ 给负监督动态降权，把旧类从过度抑制里捞回来**

既然问题出在旧类被负监督压得太狠，TAL 就在交叉熵的分母里给每个负类乘上一个随 $Q$ 变化的权重：

$$\ell_{\text{TAL}}(y, z, Q[N]) = -\log \frac{e^{z_y}}{e^{z_y} + \alpha \sum_{k \neq y} w(Q_k[N]) \cdot e^{z_k}}$$

权重函数取 $w(Q_k[N]) = (Q_k[N] / Q_{\max})^r$：当 $Q_k$ 很小（旧类近期缺正监督）时 $w\to 0$，这个类在分母里的负监督几乎被抹掉，不再被新类样本反复压制；当 $Q_k$ 接近 $Q_{\max}$（新类正监督充足）时 $w\to 1$，保留完整负监督。指数 $r>0$ 控制这条曲线的陡峭程度，$r$ 越大对低 $Q$ 旧类的保护越激进。和只在分类头做校准的旧方法不同，TAL 改的是训练时梯度的来源，这层保护会一路传到 backbone 的特征空间，而不止停在最后一层的 logits 上。

**4. 频率对齐参数 $\alpha$：保证没有时序偏差时 TAL 自动退回标准交叉熵**

给负监督降权会改变损失的整体尺度，若不校正，即便在时序均匀、类别平衡（每类先验 $p=1/C$）的理想情形下 TAL 也会和 CE 不一致。$\alpha$ 就是为消除这个尺度漂移而设的对齐项：作者要求在该理想情形下每个负类的权重系数恢复成 1，即 $\alpha\cdot w(Q^{*})=1$，其中 $Q^{*}$ 是平衡条件下的稳态 $Q$ 值。记 $x^{*}=Q^{*}/Q_{\max}$，它是方程 $(1-\tfrac1C)(x^{*})^r + x^{*} - \tfrac1C = 0$ 在 $(0,1)$ 上的唯一解，于是 $\alpha = 1/(x^{*})^r$——可见 $\alpha$ 完全由类别数 $C$ 和指数 $r$ 决定，不引入新的可调超参。它把 TAL 锚定成 CE 的一个"无偏推广"——只有当时序确实不平衡时才偏离 CE 去保护旧类，否则严丝合缝地退化回原始交叉熵。

### 损失函数 / 训练策略

TAL 全程只引入两个真正需要调的超参——衰减速率 $\lambda$ 和权重陡峭度 $r$，$\alpha$ 自动确定不算调参。为保持 $Q$ 度量的一致性，负监督在更新 $Q$ 时也要乘上同样的 $w(Q_k)$。整套方法不碰模型架构，等价于把现有 CIL 框架里的 CE loss 直接替换掉，因此能即插即用地集成进 iCaRL、DER、TagFex 等任意基线。

## 实验

### 主实验：多数据集多基线一致提升

| 方法 | CIFAR-100 10-task $A_{\text{Mean}}$ | CIFAR-100 10-task $A_{\text{Last}}$ | ImageNet-100 10-task $A_{\text{Mean}}$ | ImageNet-100 10-task $A_{\text{Last}}$ |
|------|------|------|------|------|
| iCaRL | 58.76 | 45.39 | 43.71 | 24.38 |
| iCaRL + TAL | **60.82** | **47.36** | **52.19** | **32.78** |
| DER | 63.53 | 50.75 | 52.25 | 40.28 |
| DER + TAL | **66.33** | **53.82** | **54.57** | **42.62** |
| TagFex | 65.97 | 55.99 | 54.73 | 41.70 |
| TagFex + TAL | **68.68** | **57.91** | **57.05** | **43.01** |

在 CIFAR-100、ImageNet-100、Food101 三个数据集的 10-task 和 20-task 设置下，TAL 对所有五个基线（iCaRL、FOSTER、MEMO、DER、TagFex）均带来一致且显著的提升。

### 消融实验

| $r$ \ $\lambda$ | 0.99 $A_{\text{Mean}}$ | 0.995 $A_{\text{Mean}}$ | 0.999 $A_{\text{Mean}}$ |
|------|------|------|------|
| 0.5 | 62.12 | 61.27 | 61.72 |
| 1.0 | 62.60 | **63.36** | 62.46 |
| 2.0 | 62.51 | 60.24 | 62.85 |
| CE baseline | 59.96 | - | - |

- 最优超参组合为 $\lambda = 0.995, r = 1.0$
- 在宽泛的超参范围内 TAL 均优于 CE，表明对超参选择具有鲁棒性
- $r$ 过大（如 5.0）会过度抑制新类导致性能下降

### 关键发现

1. **Precision-Recall 不对称是普遍现象**：在 iCaRL、DER、MEMO、TagFex 等方法中，早期类均呈现高 precision 低 recall 的模式（图2c）
2. **TAL 影响特征空间**：UMAP 可视化显示 TAL 缓解了旧类特征被新类侵占的问题，说明其效果不局限于分类头（图5）
3. **TAL 非均匀保护旧类**：不同旧类获得不同程度的保护，较新的旧类甚至可能被轻微抑制（图7）
4. **计算开销极小**：额外开销仅约 0.8%，得益于 $Q$ 的 $\mathcal{O}(1)$ 递推更新

## 亮点

- **新颖视角**：首次从时序不平衡角度形式化分析 CIL 中的预测偏差，区别于传统的类别不平衡解释
- **理论扎实**：建立了时序监督模型，证明了时序不平衡定理和 TAL 在平衡条件下退化为 CE 的性质
- **即插即用**：仅修改损失函数，不改架构，可无缝集成到任意 CIL 方法中
- **全局效果**：不仅修正分类头，还改善 backbone 特征空间的类间分布
- **广泛验证**：5 个基线 × 3 个数据集 × 2 种 task 设置，一致有效

## 局限性

- 衰减核固定为指数形式，实际中 $\lambda$ 可能随任务阶段变化
- 指数衰减随时间趋于零，可能无法完全捕捉已学表示的持续影响
- 未探索非参数化或自适应的时序建模形式
- 实验仅在中等规模数据集（100 类）上验证，大规模场景有待测试

## 相关工作

- **CIL 预测偏差修正**：Balanced Fine-tuning [49]、iCaRL 原型分类器 [29]、Weight Alignment [50]——均局限于分类头修正
- **时序建模**：指数平滑（Holt-Winters）、强化学习中的 Eligibility Traces / TD(λ) [37]、在线学习中的 ADWIN [4]——共享"近期数据权重更大"的直觉，但未用于 CIL 损失设计
- **动态架构方法**：DER [50]、MEMO [55]、TagFex [52]——扩展网络容量适应新任务，TAL 可作为正交补充

## 评分

- 新颖性: ⭐⭐⭐⭐ — 时序不平衡视角新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ — 多基线多数据集全面验证，消融设计合理
- 写作质量: ⭐⭐⭐⭐ — 从问题定义到方法推导逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 即插即用特性使其实用性强，新视角可启发后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](../../ECCV2024/self_supervised/stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)
- [\[CVPR 2026\] Beyond Myopic Alignment: Lookahead Optimization for Online Class-Incremental Learning](beyond_myopic_alignment_lookahead_optimization_for_online_class-incremental_lear.md)
- [\[CVPR 2026\] Geometry-driven OOD Detectors Are Class-Incremental Learners](geometry-driven_ood_detectors_are_class-incremental_learners.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets
description: >-
  [ICLR 2026][Lipschitz连续性] 系统研究了三种常用集合聚合函数（sum、mean、max）及注意力机制在三种多重集距离函数下的 Lipschitz 连续性，并推导了集合神经网络的 Lipschitz 常数上界，进而分析了模型的扰动稳定性和分布偏移下的泛化性能。 神经网络的 Lipschitz 常数与其鲁棒…
tags:
  - "ICLR 2026"
  - "Lipschitz连续性"
  - "集合聚合函数"
  - "多重集"
  - "鲁棒性"
  - "泛化"
---

# On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets

**会议**: ICLR 2026  
**arXiv**: [2505.24403](https://arxiv.org/abs/2505.24403)  
**代码**: 无  
**领域**: 其他  
**关键词**: Lipschitz连续性, 集合聚合函数, 多重集, 鲁棒性, 泛化

## 一句话总结

系统研究了三种常用集合聚合函数（sum、mean、max）及注意力机制在三种多重集距离函数下的 Lipschitz 连续性，并推导了集合神经网络的 Lipschitz 常数上界，进而分析了模型的扰动稳定性和分布偏移下的泛化性能。

## 研究背景与动机

神经网络的 Lipschitz 常数与其鲁棒性和泛化能力密切相关。先前的工作主要估计 MLP 和 CNN 的 Lipschitz 常数，但在许多实际场景（如点云处理、自然语言处理）中，输入数据是向量的集合或多重集。这类模型（如 DeepSets、PointNet）通常采用置换不变的聚合函数（sum、mean、max）来处理多重集输入。然而，关于这些聚合函数的 Lipschitz 连续性和稳定性的研究尚不充分。

**核心问题**：在哪些多重集距离函数下，常用的聚合函数是 Lipschitz 连续的？其 Lipschitz 常数是多少？基于这些聚合函数的神经网络模型是否也保持 Lipschitz 连续性？

## 方法详解

### 整体框架

这篇论文要回答一个被忽略的基础问题：当输入是一个**多重集**（点云、文档里的词向量集合等）而非单个向量时，那些把集合压成一个向量的置换不变聚合函数，到底稳不稳定？衡量稳定性的工具是 Lipschitz 连续性——若 $\text{Lip}(f)$ 小，则输入多重集的微小变动只能引起输出的小幅变化。难点在于"输入变动有多大"取决于用什么距离来量多重集之间的差异，而多重集距离本身就有多种定义。

因此全文沿着"距离 × 聚合函数"的二维网格展开：先固定三种多重集距离（Earth Mover's Distance、Hausdorff 距离、Matching 距离），逐一判定三种标准聚合函数（sum/mean/max）和注意力聚合在每种距离下是否 Lipschitz 连续、常数是多少；再把单个聚合函数的结论沿"MLP₁ → 聚合 → MLP₂"的复合结构推广到完整的集合神经网络，最后落到扰动稳定性与分布偏移泛化这两个下游推论。

### 关键设计

**1. 聚合函数与距离的一一配对：每个聚合函数只对一种距离"天生" Lipschitz**

核心结论是三种聚合函数和三种距离之间存在干净的对角对应：在任意大小的多重集上，mean 只对 EMD 是 Lipschitz 连续的且常数 $L=1$，sum 只对 Matching 距离连续且 $L=1$，max 只对 Hausdorff 距离连续且 $L=\sqrt{d}$（$d$ 为元素维度）。也就是说，离开各自配对的那种距离，聚合函数的 Lipschitz 常数就会变得无界。直觉上，EMD 度量的是把一个多重集"搬运"成另一个的平均代价，恰好和 mean 的平均行为同构；Matching 距离累加逐元素差异，与 sum 的累加行为对齐；Hausdorff 距离只看最坏的那个元素，正好对应 max 的取极值行为。这个配对表是后续一切结论的基石——它告诉你想让模型对哪种扰动鲁棒，就该选哪个聚合函数搭配哪种距离来评估。

**2. 等势多重集下的扩展：大小固定时配对会放宽**

实际任务里多重集大小往往是固定的（例如点云统一采样到 $M$ 个点）。当所有多重集势相同（都等于 $M$）时，Matching 距离与 EMD 之间出现确定关系 $d_M = M \cdot d_{\text{EMD}}$，原本只对单一距离连续的聚合函数因此对更多距离也变得 Lipschitz 连续，只是常数会乘上 $M$ 相关的因子。最有用的特例是 max：在等势设定下它对三种距离全部 Lipschitz 连续。这条结果把第 1 点的"严格对角"在实践常见的固定大小场景里松绑，给点云这类应用提供了额外的稳定性保证。

**3. 注意力聚合的负面结论：在三种距离下都不 Lipschitz**

与标准聚合不同，注意力聚合函数对所考虑的任何一种距离都**不是** Lipschitz 连续的。证明方式是构造反例：总能找到一对仅差一个元素的多重集，使注意力输出的变化相对于它们的距离任意放大，从而 Lipschitz 常数无界；即便把 softmax 注意力换成 ℓ₂ 形式的注意力，这一结论依旧成立。这解释了注意力机制为何在对抗扰动下相对脆弱，也提示若要用注意力聚合则需要额外的稳定化手段。

**4. 集合神经网络的 Lipschitz 上界与泛化推论：把单算子结论复合成整网保证**

实际模型是"逐元素 MLP₁ → 聚合 → MLP₂"的复合，利用 Lipschitz 常数沿函数复合相乘的性质即可得到整网上界：

$$\text{Lip}(\text{NN}) \le \text{Lip}(\text{MLP}_2)\cdot \text{Lip}(\text{agg})\cdot \text{Lip}(\text{MLP}_1)$$

据此对 NN_mean 和 NN_max 都能给出明确的 Lipschitz 上界；而 NN_sum 在一般情况下（尤其含非零偏置时）可能不再 Lipschitz 连续——去掉偏置可以恢复连续性。这个上界不只是鲁棒性指标：把它代入 Shen et al. (2018) 的域适应理论，就得到分布偏移下目标误差的上界

$$\varepsilon_T(h) \le \varepsilon_S(h) + 2L\cdot W_1(\mu_S, \mu_T) + \lambda$$

其中 $L$ 即上面的整网 Lipschitz 常数，$W_1$ 是源域分布 $\mu_S$ 与目标域分布 $\mu_T$ 之间的 Wasserstein 距离，并以各模型配对的距离（NN_mean 用 EMD、NN_max 用 Hausdorff）作为底层度量。于是"聚合函数 → Lipschitz 常数 → 泛化界"被串成一条完整链条。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 扰动类型 | 准确率下降 | 说明 |
|--------|------|----------|-----------|------|
| ModelNet40 | NN_mean | 元素添加 | 2.0% (±1.3) | 对单元素添加鲁棒 |
| ModelNet40 | NN_max | 元素添加 | 20.1% (±1.8) | 对单元素添加敏感 |
| Polarity | NN_mean | 随机噪声 | 13.6% (±7.1) | 对分布噪声较敏感 |
| Polarity | NN_max | 随机噪声 | 4.8% (±3.7) | 对分布噪声鲁棒 |

### 消融实验

| 配置 | 相关系数 (小→大/大→小) | 说明 |
|------|----------------------|------|
| NN_mean + EMD | r=0.92 / r=0.94 | Wasserstein 距离与泛化误差高度相关 |
| NN_max + Hausdorff | r=0.90 / r=0.90 | 同样存在强相关性 |

### 关键发现

- 每种聚合函数与恰好一种距离函数存在自然对应关系
- mean 聚合对"局部"扰动（单元素添加）更鲁棒；max 聚合对"全局"噪声（所有元素微扰）更鲁棒
- 分布漂移下的泛化误差与 Wasserstein 距离高度相关（r > 0.90）
- 注意力机制在所有三种距离下都不是 Lipschitz 连续的

## 亮点与洞察

- 首次系统建立了集合聚合函数与多重集距离函数之间的 Lipschitz 连续性对应关系
- 理论结果表明 sum 聚合虽然表达力最强，但可能导致模型不具有 Lipschitz 连续性（非零偏置情况下）
- 每种聚合函数对不同类型的扰动展现出互补的鲁棒性特性，这为模型选择提供了理论依据

## 局限与展望

- 仅考虑了置换不变的聚合函数，未涉及置换等变的消息传递架构
- Lipschitz 上界可能较松（尤其 max 函数），可探索更紧的界
- 实验仅使用简单的三层网络，未验证在更复杂架构上的适用性
- 注意力机制的分析结论较为消极，未提出替代方案

## 相关工作与启发

- 与 Chuang & Jegelka (2022) 的工作互补，后者研究了 GNN 的 Lipschitz 连续性
- 启发了集合学习中"选择什么聚合函数"这一设计选择可以根据期望的稳定性保证来做出

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究集合聚合函数的 Lipschitz 性质，但技术手段较为标准
- 实验充分度: ⭐⭐⭐⭐ 理论和实验相互验证，但数据集规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 条理清晰，定理陈述严谨
- 价值: ⭐⭐⭐⭐ 为集合学习领域的模型鲁棒性分析提供了重要理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICLR 2026\] Beyond Uniformity: Regularizing Implicit Neural Representations through a Lipschitz Lens](beyond_uniformity_regularizing_implicit_neural_representations_through_a_lipschi.md)
- [\[ICLR 2026\] A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks](a_brain-inspired_gating_mechanism_unlocks_robust_computation_in_spiking_neural_n.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICLR 2026\] Online Pseudo-Zeroth-Order Training of Neuromorphic Spiking Neural Networks](online_pseudo-zeroth-order_training_of_neuromorphic_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->

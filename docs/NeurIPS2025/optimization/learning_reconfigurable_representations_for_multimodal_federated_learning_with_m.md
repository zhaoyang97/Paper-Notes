---
title: >-
  [论文解读] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 提出 PEPSY 框架，通过学习客户端侧的嵌入控制来编码数据缺失模式，将全局聚合表示重新配置为适应各客户端本地上下文的数据完整特征，在多模态联邦学习中处理模态缺失和特征缺失两类问题。 领域现状：多模态联邦学习（MMFL）中，多个客户端观测不同子集的模态并协作训练共同模型…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "多模态学习"
  - "缺失数据"
  - "可重配置表示"
  - "嵌入控制"
---

# Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data

**会议**: NeurIPS 2025  
**arXiv**: [2510.22880](https://arxiv.org/abs/2510.22880)  
**代码**: [GitHub](https://github.com/nmduonggg/PEPSY)  
**领域**: 优化  
**关键词**: 联邦学习, 多模态学习, 缺失数据, 可重配置表示, 嵌入控制

## 一句话总结

提出 PEPSY 框架，通过学习客户端侧的嵌入控制来编码数据缺失模式，将全局聚合表示重新配置为适应各客户端本地上下文的数据完整特征，在多模态联邦学习中处理模态缺失和特征缺失两类问题。

## 研究背景与动机

**领域现状**：多模态联邦学习（MMFL）中，多个客户端观测不同子集的模态并协作训练共同模型。近年来涌现了 FedMSplit、MIFL、FedInMM、FedMAC 等方法。

**现有痛点**：现实中存在两类数据缺失事件——(1) 客户端仅拥有部分模态（如一台设备采集音频、另一台采集生理信号）；(2) 每个模态内部特征部分缺失（如传感器故障）。现有方法仅单独处理其中一类，无法同时应对两类缺失。

**核心矛盾**：当本地模型在不同特征子集上优化时，产生不兼容的表示空间。不加对齐地聚合会导致信息坍塌或退化。而服务器无法观测训练数据，客户端也无法完全解释全局聚合表示。

**本文目标**：设计一种机制来捕获和传递每个客户端本地数据的缺失模式特征，使共享模型能够适应各客户端的特定缺失情况。

**切入角度**：将缺失模式特征编码为一组可学习的嵌入控制，作为重配置信号来对齐全局表示与本地上下文。

**核心 idea**：学习数据缺失画像（data-missing profile），包含多个嵌入控制，用于将偏差表示重新配置为数据完整的特征——相似缺失模式的客户端可共享聚合的嵌入控制。

## 方法详解

### 整体框架

PEPSY 在客户端-服务器之间进行多轮通信。**客户端侧**：提取模态特定和数据特定特征，查询数据缺失画像选择相关嵌入控制，构建数据完整表示。**服务器侧**：使用 FedAvg 聚合神经网络参数，使用非参数聚类同步数据缺失画像。

### 关键设计

1. **数据缺失表示（Data-Missing Representations）**：将多模态实例信息分解为三个组件：

    - **模态特定特征** $\mathbf{w}_{di}^{\text{mod}}$：可学习嵌入 $W^{\text{mod}} = \{\mathbf{w}_i^{\text{mod}}\}_{i=1}^{|\mathcal{M}|}$，跨数据不变，编码模态身份
    - **数据特定特征** $\mathbf{w}_{di}^{\text{ins}}$：将每个观测模态映射为表示 $\mathbf{h}_{di}$，缺失模态用其他可用模态特征的均值替代：$\mathbf{w}_{di}^{\text{ins}} = \mathbf{I}(i \notin \mathcal{S}_d)\mathbf{h}_{di} + \mathbf{I}(i \in \mathcal{S}_d) \frac{1}{|\mathcal{M}|-|\mathcal{S}_d|}\sum_{j \notin \mathcal{S}_d}\mathbf{h}_{dj}$
    - **数据特定对比损失** $\mathcal{L}_{ds}$：使同一实例不同模态的特征更接近，不同实例的特征更远

2. **嵌入控制选择（Embedding Controls Selection）**：通过查询-键匹配机制，将数据缺失特征与嵌入控制进行交互。相关性定义为：
    $\gamma(\mathbf{x}_{di}, \boldsymbol{\psi}_p) = e(\mathbf{q}(\mathbf{x}_{di}), \mathbf{k}(\boldsymbol{\psi}_p))$
   每个实例仅选取 $\kappa$ 个最相关的嵌入控制（$\kappa \ll |\Psi|$），并通过正则化项 $\mathcal{R}$ 鼓励稀疏选择。最终缺失模式表示 $\mathbf{w}_{di}^{\text{mis}}$ 为所选嵌入的均值。

3. **重配置正则化（Reconfiguration Regularization）**：对比损失 $\mathcal{L}_{rc}$ 确保拼接了缺失模式信息的最终表示 $\mathbf{w}_{di} = [\mathbf{w}_{di}^{\text{mod}} \circ \mathbf{w}_{di}^{\text{ins}} \circ \mathbf{w}_{di}^{\text{mis}}]$ 忠实反映完整模态信息。

4. **模态融合（Modality Fusion）**：利用高层表示 $\hat{\mathbf{w}}_{di}$ 之间的相似度作为注意力权重融合跨模态信息，并通过自适应门控 $\boldsymbol{\alpha}_{di}$ 结合跨模态表示和原始表示得到最终表示 $\mathbf{c}_{di}$。

5. **服务器聚合（Server Aggregation）**：数据缺失画像因客户端学习顺序不同而无法直接合并。采用非参数聚类方法 PFPT 动态分组相似嵌入，自适应调整聚类数量以反映系统整体缺失复杂度。

### 损失函数 / 训练策略

总训练目标：

$$\mathcal{L} = \mathcal{L}_{task} + \lambda(\mathcal{L}_{ds} + \mathcal{L}_{rc}) - \eta\mathcal{R}$$

其中 $\mathcal{L}_{task}$ 是任务特定损失，$\mathcal{L}_{ds}$ 和 $\mathcal{L}_{rc}$ 分别是数据特定对比损失和重配置对比损失，$\mathcal{R}$ 是嵌入相关性正则化项。

## 实验关键数据

### 主实验

**PTBXL 数据集（12 模态，IID，$p_m=0.2$）准确率 (%)：**

| 方法 | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|------|-----------|-----------|-----------|-----------|-----------|
| FedProx | 73.43 | 73.64 | 71.42 | 71.37 | 69.93 |
| FedMAC | 78.56 | 77.30 | 76.25 | 75.49 | 74.70 |
| FedMSplit | 54.84 | 53.63 | 52.12 | 52.50 | 55.84 |
| **PEPSY** | **78.81** | **77.43** | **76.75** | **76.13** | **75.41** |

**PTBXL Non-IID 设置（$p_m=0.2$）：**

| 方法 | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|------|-----------|-----------|-----------|-----------|-----------|
| FedProx | 54.01 | 51.15 | 50.06 | 54.89 | 44.17 |
| FedMAC | 58.26 | 58.55 | 54.98 | 50.94 | 48.38 |
| **PEPSY** | **71.45** | **69.70** | **66.92** | **68.26** | **66.75** |

**EDF 数据集（5 模态，Non-IID，$p_m=0.8$）：**

| 方法 | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|------|-----------|-----------|-----------|-----------|-----------|
| FedMAC | 46.01 | 45.73 | 45.66 | 46.22 | 34.21 |
| **PEPSY** | **48.95** | **51.52** | **50.97** | **50.96** | **46.07** |

### 消融实验

PEPSY 在严重数据不完整情况下性能提升最为显著：

| 场景 | 最大提升 |
|------|----------|
| PTBXL Non-IID ($p_m=0.2, p_s=1.0$) | +18.37% (vs FedMAC) |
| PTBXL Non-IID ($p_m=0.8, p_s=0.6$) | +32.24% (vs FedMAC) |
| EDF Non-IID ($p_m=0.8, p_s=0.4$) | +5.79% (vs FedMAC) |

### 关键发现

- PEPSY 在 Non-IID 设置下优势最大，Non-IID + 高缺失率（$p_m=0.8$）场景下可实现高达 36.45% 的性能提升
- IID 条件下 PEPSY 与 FedMAC 性能接近，说明数据缺失画像在数据分布一致时作用较小
- 现有方法（FedMSplit、FedInMM）仅解决单一缺失类型，在同时出现两种缺失时性能显著下降
- 理论分析表明缺失模态导致的预测偏差受 $\mathcal{L}_{ds}$ 直接控制，验证了损失设计的有效性

## 亮点与洞察

- **问题形式化出色**：将多模态 FL 中的缺失模式分解为模态特定、数据特定和缺失模式三个正交组件，设计清晰
- **嵌入控制机制新颖**：将缺失模式编码为可学习嵌入并通过查询-键匹配选择，是一个优雅的解决方案
- **非参数聚类聚合**：巧妙解决了本地缺失画像在不同客户端间的对齐问题
- 理论界（Theorem 3.1）直接连接了训练损失和缺失模态下的预测稳定性，理论与实验一致

## 局限与展望

- 服务器聚合中的非参数聚类（PFPT）增加了通信和计算开销，扩展性需要验证
- 实验仅在医疗/睡眠两类数据集上验证，其他多模态场景（如自动驾驶、环境监测）待测试
- 嵌入控制数量 $\tau$ 和选择数量 $\kappa$ 的设置对性能影响未充分分析
- 数据特定特征中缺失模态用均值替代较为简单，可能限制了特征质量

## 相关工作与启发

- FedMAC 是最接近的基线，处理特征缺失但不处理模态缺失
- FedMSplit 处理模态缺失但不处理特征缺失
- PEPSY 通过嵌入控制统一处理两种缺失，提供了更全面的解决方案
- 数据缺失画像的思想可以推广到其他需要处理异构客户端的 FL 场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 嵌入控制+重配置的核心机制新颖，统一处理两类缺失问题
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多种缺失配置的系统对比，但缺少更多领域验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，理论分析扎实，符号体系一致
- 价值: ⭐⭐⭐⭐ 解决了MMFL中一个重要且实际的问题，36.45%的提升说明了方法的有效性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->

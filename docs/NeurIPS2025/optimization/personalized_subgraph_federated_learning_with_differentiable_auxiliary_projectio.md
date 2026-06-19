---
title: >-
  [论文解读] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 提出FedAux框架，通过可微分的辅助投影向量（APV）将节点嵌入映射到一维空间并用高斯核进行软排序聚合，APV既作为局部子图的紧凑隐私保护摘要用于服务器端相似度计算，又参与客户端的联合优化，实现了个性化的子图联邦学习。 子图联邦学习（Subgraph FL）中…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "图神经网络"
  - "个性化聚合"
  - "子图异质性"
  - "辅助投影向量"
---

# Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections

**会议**: NeurIPS 2025  
**arXiv**: [2505.23864](https://arxiv.org/abs/2505.23864)  
**代码**: [GitHub](https://github.com/JhuoW/FedAux)  
**领域**: 优化  
**关键词**: 联邦学习, 图神经网络, 个性化聚合, 子图异质性, 辅助投影向量

## 一句话总结

提出FedAux框架，通过可微分的辅助投影向量（APV）将节点嵌入映射到一维空间并用高斯核进行软排序聚合，APV既作为局部子图的紧凑隐私保护摘要用于服务器端相似度计算，又参与客户端的联合优化，实现了个性化的子图联邦学习。

## 研究背景与动机

子图联邦学习（Subgraph FL）中，每个客户端持有一个全局图的子图，子图之间存在严重的non-IID问题。例如多个区域社交平台，每个区域用户的交互模式和兴趣各不相同，直接用FedAvg聚合GNN模型效果很差。

个性化FL的关键挑战是**如何在不共享数据的前提下衡量客户端相似度**：

**直接比较参数矩阵**：高维参数空间中距离度量不可靠（维度灾难）

**比较梯度**：信息有限且偏向启发式

**共享embedding**：违反隐私约束

**锚图方法**：在服务器生成公共图作为测试台，但不能显式建模子图异质性

本文的洞察是：一个**紧凑的低维代理**可以从模型参数直接导出，忠实总结局部子图特征而不泄露隐私。这个代理应该足够紧凑以避免高维距离度量的陷阱，又足够表达以反映有意义的客户端差异。

## 方法详解

### 整体框架

FedAux的工作流程：
1. 服务器维护全局GNN参数 $\theta$ 和辅助投影向量APV $\mathbf{a}$
2. 每轮通信：广播 $(\theta, \mathbf{a})$ → 客户端本地训练 → 上传 $(\theta_k, \mathbf{a}_k)$ → 服务器个性化聚合

### 关键设计

1. **辅助投影向量（APV）与一维空间映射**: 每个客户端将GNN产出的节点嵌入 $h_{k,i}$ 投影到APV方向上，得到标量相似度分数 $s_{k,i} = \langle \hat{h}_{k,i}, \mathbf{a}_k \rangle$。这将每个节点映射到一维 $\mathbf{a}_k$-空间上。APV是可学习的，客户端通过训练自适应调整这个空间以捕获节点间的关系。

2. **可微分核聚合替代硬排序**: 早期方法用硬排序+1D卷积来聚合邻近节点信息，但排序操作不可微，导致APV无法通过反向传播优化。本文提出用高斯核实现软排序：

    $z_{k,i} = \frac{1}{M_i} \sum_{j=1}^{N_k} \kappa(s_{k,i}, s_{k,j}) h_{k,j}, \quad \kappa(s_i, s_j) = \exp\left(-\frac{(s_i - s_j)^2}{\sigma^2}\right)$

   这个连续聚合器对APV完全可微——APV的变化平滑地调整每个 $s_{k,i}$，进而调整核权重。

3. **APV的理论保证（Theorem 3.1）**: 证明了在高斯核聚合下，损失对APV的梯度在 $\sigma \to 0$ 极限下趋向于 $-\frac{2}{\sigma^2} \mathbf{C}\mathbf{a}$（$\mathbf{C}$ 是嵌入协方差矩阵）。通过单位范数重归一化，更新规则退化为**Oja学习规则**，其全局吸引子是 $\mathbf{C}$ 的主特征向量。这意味着APV不是任意的可训练参数，而是**统计最优的、方差最大化的局部嵌入摘要**。

4. **服务器端个性化聚合**: 计算客户端APV之间的余弦相似度，通过softmax温控得到聚合权重：
   
    $w_{k,l} = \frac{\exp(\alpha \text{Sim}(\mathbf{a}_k, \mathbf{a}_l))}{\sum_{r=1}^K \exp(\alpha \text{Sim}(\mathbf{a}_k, \mathbf{a}_r))}$
   
   然后为每个客户端生成个性化参数 $\theta_k = \sum_l w_{k,l} \theta_l$。这强调了相似客户端的贡献，减少了不相似客户端的干扰。

### 损失函数 / 训练策略

- 客户端损失：交叉熵 $\mathcal{L}_k = \frac{1}{N_k} CE(\text{CLF}(\Gamma_k), Y_k)$，其中 $\Gamma_k = [h_{k,i} \| z_{k,i}]$
- 联合优化GNN参数 $\theta_k$、APV $\mathbf{a}_k$ 和分类器 $\Phi_k$
- 核带宽 $\sigma = 1$ 对所有数据集有效
- 全局线性收敛保证（Theorem 3.3）：$E[\mathcal{L}(\Psi^{(T)}) - \mathcal{L}^\star] \leq (1-\eta\mu)^{QT}(\mathcal{L}(\Psi^{(0)}) - \mathcal{L}^\star) + \frac{\eta\mathscr{L}\zeta^2}{2\mu} + \frac{2\eta\mathscr{L}\rho^2}{\mu(1-\rho)^2}$

## 实验关键数据

### 主实验（联邦节点分类）

| 数据集 | 客户端数 | FedAux | FED-PUB (SOTA) | FedAvg | Local |
|---|---|---|---|---|---|
| Cora | 5 | **84.57±0.39** | 83.72±0.18 | 74.45±5.64 | 81.30±0.21 |
| Cora | 10 | **82.05±0.71** | 81.45±0.12 | 69.19±0.67 | 79.94±0.24 |
| Cora | 20 | **81.60±0.64** | 81.10±0.64 | 69.50±3.58 | 80.30±0.25 |
| CiteSeer | 5 | **72.99±0.82** | 72.40±0.26 | 71.06±0.60 | 69.02±0.05 |
| CiteSeer | 10 | **73.16±0.29** | 71.83±0.61 | 63.61±3.59 | 67.82±0.13 |
| CiteSeer | 20 | **68.10±0.35** | 66.89±0.14 | 64.68±1.83 | 65.98±0.17 |
| Pubmed | 5 | **88.10±0.16** | 86.81±0.12 | 79.40±0.11 | 84.04±0.18 |
| Pubmed | 10 | **86.43±0.20** | 86.09±0.17 | 82.71±0.29 | 82.81±0.39 |
| Pubmed | 20 | **84.87±0.42** | 84.66±0.54 | 80.97±0.26 | 82.65±0.03 |

### 更多数据集

| 数据集 | 客户端数 | FedAux | GCFL | FedPer | FedAvg |
|---|---|---|---|---|---|
| Amazon-Computer | 10 | 90.50+ | 90.03 | 89.73 | 79.54 |
| Amazon-Photo | 10 | 92.50+ | 92.06 | 91.76 | 83.15 |
| ogbn-arxiv | 5 | 67.00+ | 66.80 | 66.87 | 65.54 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| 硬排序 vs 核聚合 | 核聚合更优 | 可微性使APV能被充分优化 |
| 有APV vs 无APV聚合 | 有APV显著更优 | APV作为子图摘要的有效性 |
| $\sigma=1$ vs 其他值 | $\sigma=1$ 最优 | 作为正则化防止过拟合 |
| Theorem 3.2验证 | $\sigma \to 0$ 时收敛到硬排序 | 理论与实验一致 |

### 关键发现

- FedAux在全部6个数据集、3种客户端规模下**一致超越**所有baseline
- 在CiteSeer 10 clients上提升最明显（73.16 vs 71.83，绝对提升1.33%）
- APV仅是一个低维向量，隐私泄露风险极低
- Pubmed上FedAux比Local训练提升4%，说明跨客户端知识迁移有效

## 亮点与洞察

- APV的设计巧妙：一石三鸟——既参与本地训练提升模型、又作为客户端签名计算相似度、还保护隐私
- Theorem 3.1揭示APV是嵌入协方差矩阵的主成分，为设计选择提供了理论支撑
- 用高斯核代替硬排序是关键——解决了梯度阻断问题，使端到端训练成为可能
- 复杂度分析清晰：客户端 $O(|E_k|d' + N_k^2 d')$，服务器 $O(K^2 d')$

## 局限与展望

- 核聚合的复杂度为 $O(N_k^2)$，对大规模子图可能成为瓶颈
- APV作为主成分可能在某些数据分布下无法捕获关键的子图结构差异
- 仅在节点分类任务上验证，图级别和链接预测任务未涉及
- 温度参数 $\alpha$ 对结果的敏感性分析不够充分

## 相关工作与启发

- 与FED-PUB的子图mask方法形成对比：FedAux用连续投影代替离散mask
- Oja学习规则的联系为联邦学习中的表示对齐提供了新思路
- 启发：其他需要客户端签名/指纹的FL场景（如异质性检测）也可采用类似的可微投影方法

## 评分

- 新颖性: ⭐⭐⭐⭐ APV概念新颖，可微核聚合设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、3种规模、丰富的baseline对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，理论分析严谨
- 价值: ⭐⭐⭐⭐ 为子图FL提供了简洁高效的个性化方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[CVPR 2026\] Few-for-Many Personalized Federated Learning](../../CVPR2026/optimization/few-for-many_personalized_federated_learning.md)

</div>

<!-- RELATED:END -->

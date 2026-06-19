---
title: >-
  [论文解读] Training Robust Graph Neural Networks by Modeling Noise Dependencies
description: >-
  [NeurIPS 2025][优化/理论][图神经网络鲁棒性] 提出依赖感知图噪声(DANG)和DA-GNN框架，通过建模节点特征噪声→图结构噪声→标签噪声的因果依赖链，利用变分推断推导ELBO来训练对多源协同噪声鲁棒的GNN。 GNN在实际应用中面临的核心挑战之一是数据噪声。现有鲁棒GNN方法分别处理特征噪声（AirGNN…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "图神经网络鲁棒性"
  - "噪声依赖"
  - "因果建模"
  - "变分推断"
  - "数据生成过程"
---

# Training Robust Graph Neural Networks by Modeling Noise Dependencies

**会议**: NeurIPS 2025  
**arXiv**: [2502.19670](https://arxiv.org/abs/2502.19670)  
**代码**: [GitHub](https://github.com/yeonjun-in/torch-DA-GNN)  
**领域**: 优化  
**关键词**: 图神经网络鲁棒性, 噪声依赖, 因果建模, 变分推断, 数据生成过程

## 一句话总结

提出依赖感知图噪声(DANG)和DA-GNN框架，通过建模节点特征噪声→图结构噪声→标签噪声的因果依赖链，利用变分推断推导ELBO来训练对多源协同噪声鲁棒的GNN。

## 研究背景与动机

GNN在实际应用中面临的核心挑战之一是数据噪声。现有鲁棒GNN方法分别处理特征噪声（AirGNN）、结构噪声（RSGNN、STABLE）或标签噪声（NRGNN、RTGNN），但它们都假设噪声来源之间是独立的，即所谓的独立节点特征噪声(IFN)假设。

**核心痛点**：IFN假设与现实严重脱节。在社交网络中，Bob创建虚假个人资料（特征噪声）→ Alice和Tom基于虚假资料与Bob建立连接（结构噪声）→ Alice和Tom的社区标签因此被改变（标签噪声）。这种噪声的级联依赖关系在社交、电商、生物等图应用中普遍存在。

**已有方法的根本局限**：每种鲁棒GNN方法都假设至少一种数据源（特征/结构/标签）是干净的。例如AirGNN假设结构无噪声，RTGNN依赖结构信息来缓解标签噪声（但结构本身可能有噪声），SG-GSR假设标签无噪声。当所有数据源都含噪时，这些方法的假设被违反，性能显著下降。

**核心idea**：正式定义依赖感知图噪声(DANG)及其数据生成过程(DGP)，引入三个隐变量（噪声变量 $\epsilon$、潜在干净图结构 $Z_A$、潜在干净标签 $Z_Y$），并建立它们与观测变量（$X$, $A$, $Y$）之间的因果关系。然后设计深度生成模型DA-GNN直接捕获这些因果关系。

## 方法详解

### 整体框架

DA-GNN是一个基于变分推断的编码器-解码器框架：
- **推断编码器**（$\phi_1, \phi_2, \phi_3$）：从含噪数据推断隐变量 $Z_A, \epsilon, Z_Y$
- **生成解码器**（$\theta_1, \theta_2, \theta_3$）：建模从隐变量生成观测数据的过程
- 通过最小化负ELBO来训练

### 关键设计

1. **DANG的数据生成过程(DGP)**：定义了六个因果关系——

    - $X \leftarrow (\epsilon, Z_Y)$：噪声变量和真实标签共同产生（可能含噪的）特征
    - $A \leftarrow (Z_A, X)$：潜在干净结构和特征共同产生（可能含噪的）边
    - $A \leftarrow \epsilon$：噪声变量也可直接导致结构噪声
    - $Y \leftarrow (Z_Y, X, A)$：真实标签、特征和结构共同产生（可能含噪的）观测标签

   这个DGP的关键特点是：图中不存在任何完全干净的数据源。

2. **推断编码器的设计**：

    - **$q_{\phi_1}(Z_A|X,A)$（推断干净图结构）**：使用GCN编码器计算节点表示 $\mathbf{Z} = \text{GCN}_{\phi_1}(\mathbf{X}, \mathbf{A})$，然后通过余弦相似性得到潜在图 $\hat{p}_{ij} = \rho(s(\mathbf{Z}_i, \mathbf{Z}_j))$。用 $\gamma$-hop子图相似性先验进行正则化（对应KL散度项）。为避免 $O(N^2)$ 的全图计算，预定义代理图来限制计算范围。

    - **$q_{\phi_3}(Z_Y|X,A)$（推断干净标签）**：GCN分类器，在推断的干净图 $\hat{\mathbf{A}}$ 上操作。用同质性正则化鼓励连接节点有相似预测：

    $\mathcal{L}_{\text{hom}} = \sum_{i \in \mathcal{V}} \frac{\sum_{j \in \mathcal{N}_i} \hat{p}_{ij} \cdot kl(\hat{\mathbf{Y}}_j || \hat{\mathbf{Y}}_i)}{\sum_{j \in \mathcal{N}_i} \hat{p}_{ij}}$

    - **$q_{\phi_2}(\epsilon|X,A,Z_Y)$（推断噪声变量）**：分解为结构噪声 $\epsilon_A$（通过early-learning阶段的小损失方法估计边清洁度）和特征噪声 $\epsilon_X$（通过MLP从 $X$ 和 $Z_Y$ 推断，正则化为标准正态分布）。用EMA技巧缓解单点估计的不确定性：$\hat{p}_{ij}^{el} \leftarrow \xi \hat{p}_{ij}^{el} + (1-\xi)\hat{p}_{ij}^c$，$\xi=0.9$。

3. **生成解码器的设计**：

    - **$p_{\theta_1}(A|X,\epsilon,Z_A)$（重建含噪边）**：边重建损失，对预测和标签都进行正则化处理含噪监督。预测正则化 $\hat{p}_{ij}^{reg} = \theta_1 \hat{p}_{ij} + (1-\theta_1)s(\mathbf{X}_i, \mathbf{X}_j)$，当特征相似性高时惩罚边预测（因为可能是特征噪声导致的虚假连接）。标签平滑到 $[0.9, 1]$ 区间。
    - **$p_{\theta_2}(X|\epsilon,Z_Y)$（重建含噪特征）**：MLP解码器，输入为 $\epsilon_X$ 和 $Z_Y$，使用重参数化技巧采样。
    - **$p_{\theta_3}(Y|X,A,Z_Y)$（重建含噪标签）**：GCN分类器建模从干净标签到噪声标签的转移关系。

### 损失函数 / 训练策略

总损失为ELBO各项的加权和：

$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{cls-enc}} + \lambda_1 \mathcal{L}_{\text{rec-edge}} + \lambda_2 \mathcal{L}_{\text{hom}} + \lambda_3(\mathcal{L}_{\text{rec-feat}} + \mathcal{L}_{\text{cls-dec}} + \mathcal{L}_{\text{p}})$$

其中 $\lambda_3=0.001$ 固定（这三项对性能影响较小），$\lambda_1, \lambda_2$ 需调参，$k$（代理图参数）是最关键的超参数。

## 实验关键数据

### 主实验（DANG下的节点分类准确率%）

| 数据集 | 噪声级别 | WSGNN | GraphGLOW | AirGNN | STABLE | RTGNN | SG-GSR | DA-GNN |
|--------|---------|-------|-----------|--------|--------|-------|--------|--------|
| Cora | Clean | 86.2 | 85.2 | 85.0 | 86.1 | 86.1 | 85.7 | **86.2** |
| Cora | DANG-10% | 80.7 | 79.7 | 79.7 | 82.2 | 81.8 | 82.7 | **82.9** |
| Cora | DANG-30% | 70.0 | 71.6 | 71.5 | 74.3 | 72.6 | 76.1 | **78.2** |
| Cora | DANG-50% | 55.9 | 59.6 | 56.2 | 62.8 | 60.9 | 64.3 | **69.7** |
| Photo | DANG-50% | 31.9 | 85.4 | 57.8 | 80.2 | 79.2 | 84.1 | **87.6** |
| Comp | DANG-50% | 39.6 | 80.1 | 44.1 | 68.8 | 69.4 | 78.6 | **82.2** |

### 消融实验（DGP因果关系逐步移除）

| 配置 | Cora Clean | Cora DANG-30% | Cora DANG-50% | 说明 |
|------|-----------|--------------|--------------|------|
| Case 1 (IFN, 移除所有依赖) | 84.6 | 68.3 | 55.2 | 退化为独立噪声假设 |
| Case 2 (移除 $A \leftarrow X$) | 84.8 | 68.5 | 56.1 | 忽略特征→结构的噪声传播 |
| Case 3 (仅移除 $Y \leftarrow (X,A)$) | 86.2 | 77.3 | 68.7 | 忽略标签转移关系 |
| **Proposed (完整DANG)** | **86.2** | **78.2** | **69.7** | 完整因果建模 |

### 真实世界DANG数据集

| 任务 | 数据集 | 设置 | SG-GSR (次优) | DA-GNN |
|------|--------|------|-------------|--------|
| 节点分类 | Auto + DANG | 准确率 | 62.0±1.1 | **61.4±0.4** |
| 节点分类 | Garden + DANG | 准确率 | 80.2±0.4 | **80.2±0.8** |
| 链接预测 | Auto + DANG | ROC-AUC | 65.6±7.4 | **73.6±0.6** |
| 链接预测 | Garden + DANG | ROC-AUC | 86.0±7.2 | **92.4±0.4** |

### 关键发现

- **噪声越高，DA-GNN的优势越大**：在Cora DANG-50%上，DA-GNN (69.7%) 超过次优SG-GSR (64.3%) 5.4个百分点
- **DGP中每条因果关系都有贡献**：逐步移除因果边导致性能单调下降（69.7→68.7→56.1→55.2）
- **广泛适用性**：DA-GNN在DANG、纯特征噪声、纯结构噪声、纯标签噪声和极端噪声（三种同时出现）五种场景下均表现最优或有竞争力
- **在大图Arxiv上**，多数基线因OOM无法运行，DA-GNN在DANG-50%上达44.0%准确率

## 亮点与洞察

- **DANG的定义填补了重要空白**：现有鲁棒GNN研究的独立噪声假设是一个被广泛接受但不合理的简化，DANG提供了更贴近现实的噪声模型
- **因果建模的优雅性**：通过有向图模型的DGP，清晰定义了六个因果关系，使得ELBO的推导自然且有理论依据
- **实用的新数据集**：基于Amazon评论数据构建的Auto和Garden数据集模拟了电商场景中的DANG，为后续研究提供了实用benchmark
- **消融设计精巧**：通过逐步移除DGP中的因果边，清晰展示了每条依赖关系的贡献

## 局限与展望

- DANG未考虑 $X \leftarrow A$（图结构噪声反过来影响节点特征）的反向依赖，这在某些场景中也很自然
- 训练复杂度较高（需要同时推断三个隐变量），在大规模图上可能受限
- 代理图的预计算引入了 $O(N^2)$ 的初始化开销虽已优化，但仍是瓶颈
- 超参数 $k$（代理图参数）需要调参，虽然搜索空间小但增加了实践门槛
- 在真实世界DANG数据集上的优势相对合成数据集更小，说明合成数据可能高估了方法优势

## 相关工作与启发

- 与实例依赖标签噪声(IDN)的生成方法有理论联系，但扩展到图域引入了额外隐变量和复杂因果关系
- WSGNN和GraphGLOW虽然也使用变分推断和图结构推断，但假设图无噪声，适用范围更窄
- 启发：在其他图学习任务（推荐系统、知识图谱）中，噪声间的依赖关系同样广泛存在，DANG的建模思路可推广

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ DANG的定义和基于DGP的因果建模方法论有重要的概念贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖合成/真实DANG、单一噪声、极端噪声等多种场景，消融详尽
- 写作质量: ⭐⭐⭐⭐ 因果建模部分清晰，但模型实例化细节较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ 对鲁棒GNN研究范式的推进——从"假设某数据源干净"到"所有数据源都可能含噪"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](../../ICLR2026/optimization/rapid_training_of_hamiltonian_graph_networks_using_random_features.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)

</div>

<!-- RELATED:END -->

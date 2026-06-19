---
title: >-
  [论文解读] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement
description: >-
  [ICCV 2025][AI安全][联邦学习] 本文提出LoRA-FAIR方法，通过在服务器端引入残差校正项 $\Delta\mathbf{B}$ 来同时解决联邦学习+LoRA微调中的服务器端聚合偏差和客户端初始化滞后两大挑战，在ViT和MLP-Mixer模型上一致超越现有联邦微调方法，且不增加通信开销。
tags:
  - "ICCV 2025"
  - "AI安全"
  - "联邦学习"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "aggregation bias"
  - "foundation models"
---

# LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement

**会议**: ICCV 2025  
**arXiv**: [2411.14961](https://arxiv.org/abs/2411.14961)  
**代码**: 无  
**领域**: AI安全 / 联邦学习  
**关键词**: federated learning, LoRA, parameter-efficient fine-tuning, aggregation bias, foundation models

## 一句话总结
本文提出LoRA-FAIR方法，通过在服务器端引入残差校正项 $\Delta\mathbf{B}$ 来同时解决联邦学习+LoRA微调中的服务器端聚合偏差和客户端初始化滞后两大挑战，在ViT和MLP-Mixer模型上一致超越现有联邦微调方法，且不增加通信开销。

## 研究背景与动机
- **领域现状**：大型基础模型（如ViT）的全参数微调计算代价过高，LoRA通过低秩分解大幅减少可训练参数；联邦学习（FL）通过隐私保护的协作学习解决数据不足问题
- **现有痛点**：将LoRA与FL直接结合（FedIT）面临两个根本性挑战：
    - **挑战1：服务器端聚合偏差**：独立平均 $\bar{\mathbf{A}}$ 和 $\bar{\mathbf{B}}$ 后其乘积 $\bar{\mathbf{B}}\bar{\mathbf{A}}$ 不等于理想全局更新 $\sum p_k \mathbf{B}_k \mathbf{A}_k$（矩阵乘法对求和不可分配）
    - **挑战2：客户端初始化滞后**：FLoRA等方法每轮重新初始化LoRA模块（$\mathbf{A}$随机、$\mathbf{B}$置零），导致梯度初期无信息（$\partial L/\partial \mathbf{A} \to 0$），在有限本地训练轮次下学习效率低
- **核心矛盾**：现有方法（FFA-LoRA、FLoRA、FlexLoRA）只能解决其中一个挑战，无法兼顾
- **本文要解决的问题**：设计一个同时解决聚合偏差和初始化滞后，且不增加通信和计算开销的联邦LoRA微调方法
- **切入角度**：在服务器端保持 $\bar{\mathbf{A}}$ 不变，引入残差 $\Delta\mathbf{B}$ 修正 $\bar{\mathbf{B}}$，使 $(\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}} \approx \Delta\mathbf{W}$
- **核心idea**：残差 $\Delta\mathbf{B}$ 通过最小化与理想全局更新的差异+正则化项求解，兼顾精确聚合与稳定初始化

## 方法详解

### 整体框架
每轮训练流程：各客户端本地训练LoRA模块 → 上传 $\mathbf{A}_k, \mathbf{B}_k$ 到服务器 → 服务器计算加权平均 $\bar{\mathbf{A}}, \bar{\mathbf{B}}$ → 计算理想全局更新 $\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ → 优化残差 $\Delta\mathbf{B}$ 使 $(\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}} \approx \Delta\mathbf{W}$ → 下发 $\bar{\mathbf{A}}$ 和 $\bar{\mathbf{B}}' = \bar{\mathbf{B}} + \Delta\mathbf{B}$ 给客户端 → 客户端用 $\bar{\mathbf{A}}, \bar{\mathbf{B}}'$ 初始化下轮训练。

### 关键设计
1. **残差校正优化（Residual Correction）**:

    - 功能：在服务器端求解残差 $\Delta\mathbf{B}$，修正聚合偏差
    - 核心思路：$\arg\min_{\Delta\mathbf{B}} \underbrace{\mathcal{S}(\Delta\mathbf{W}, (\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}})}_{\text{校正项}} + \underbrace{\lambda \|\Delta\mathbf{B}\|}_{\text{正则化项}}$，其中 $\mathcal{S}$ 为余弦相似度，$\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ 为理想更新
    - 设计动机：校正项解决挑战1（使聚合更新逼近理想值），正则化项解决挑战2（约束 $\bar{\mathbf{B}}' \approx \bar{\mathbf{B}}$ 保持平均信息，提供稳定初始化）

2. **Avg-Initial客户端初始化策略**:

    - 功能：客户端直接使用服务器下发的平均LoRA模块作为下轮初始化
    - 核心思路：$\mathbf{A}_k \leftarrow \bar{\mathbf{A}}$，$\mathbf{B}_k \leftarrow \bar{\mathbf{B}} + \Delta\mathbf{B}$，预训练权重 $\mathbf{W}_0$ 不变
    - 设计动机：对比Re-Initial（重新随机初始化）和Local-Initial（用某客户端的本地模块），Avg-Initial兼顾训练连续性和全局信息融合

3. **残差位置选择**:

    - 功能：确定将 $\Delta$ 应用到 $\mathbf{A}$ 还是 $\mathbf{B}$
    - 核心思路：消融实验表明修正 $\mathbf{B}$ 优于修正 $\mathbf{A}$
    - 设计动机：$\mathbf{A}$ 主要捕获通用信息，保持稳定的平均更新更有利；$\mathbf{B}$ 更适合承载校正信号

### 损失函数 / 训练策略
- 客户端：SGD优化器，学习率0.01，batch size 128
- 服务器端残差优化：SGD求解Eq. 8
- 正则化权重 $\lambda = 0.01$（小正值即可）
- LoRA秩默认设为16

## 实验关键数据

### 主实验

| 数据集 | 模型 | LoRA-FAIR | FedIT | FLoRA | FlexLoRA | Centralized |
|--------|------|-----------|-------|-------|----------|-------------|
| DomainNet (feature non-IID) | ViT | **77.07** | 75.75 | 75.53 | 76.02 | 77.77 |
| DomainNet (feature non-IID) | MLP-Mixer | **65.87** | 64.37 | 64.38 | 64.79 | 66.64 |
| NICO++ (feature non-IID) | ViT | **91.24** | 90.58 | 90.93 | 90.60 | 91.51 |
| NICO++ (feature non-IID) | MLP-Mixer | **83.56** | 82.51 | 82.29 | 83.08 | 84.50 |
| DomainNet (feat.+label non-IID) | ViT | **74.99** | 73.89 | 74.26 | 74.25 | 77.77 |
| NICO++ (feat.+label non-IID) | ViT | **90.04** | 89.48 | 89.60 | 89.65 | 91.51 |

### 消融实验

| 配置 | DomainNet平均准确率 | 说明 |
|------|-------------------|------|
| $\Delta\mathbf{B}$（默认） | 77.07 | 修正B矩阵 |
| $\Delta\mathbf{A}$ | 76.42 | 修正A矩阵，略低 |
| $\Delta\mathbf{A}, \Delta\mathbf{B}$ | 75.55 | 同时修正，过度拟合 |
| $\lambda = 0$（无正则化） | 73.22 | 聚合偏差消除但初始化不稳定 |
| $\lambda = 0.01$（默认） | 77.07 | 平衡两个挑战 |

### 关键发现
- LoRA-FAIR在所有设置下一致超越所有基线方法，接近集中式训练上界
- FLoRA虽解决了聚合偏差但因重新初始化导致性能甚至不如简单的FedIT
- 正则化权重 $\lambda$ 至关重要：$\lambda=0$ 时虽然 $(\bar{\mathbf{B}}+\Delta\mathbf{B})\bar{\mathbf{A}}$ 与 $\Delta\mathbf{W}$ 的相似度最高（0.9998），但 $\bar{\mathbf{B}}'$ 与 $\bar{\mathbf{B}}$ 的相似度降至0.9715，证实了初始化稳定性的重要性
- 通信开销与FedIT/FlexLoRA相同，远低于FLoRA

## 亮点与洞察
- 问题分析精辟：将联邦LoRA的困难分解为聚合偏差和初始化滞后两个独立但需同时解决的挑战
- 解决方案极简优雅：仅引入一个残差矩阵+正则化，无额外通信/客户端计算
- 初始化策略的系统对比（Avg-Initial vs Re-Initial vs Local-Initial）为FL社区提供了实用指导
- 正则化的双重作用设计巧妙：同一个 $\lambda\|\Delta\mathbf{B}\|$ 既限制校正幅度确保初始化稳定，又自然保留了平均信息

## 局限与展望
- 理想全局更新 $\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ 需要在服务器端计算所有客户端的矩阵乘积，当客户端数量大时可能成为瓶颈
- 只在视觉模型（ViT、MLP-Mixer）上验证，未涉及LLM等更大规模基础模型
- 残差优化依赖SGD迭代求解，其收敛速度和精度对超参数敏感
- 假设所有客户端使用相同LoRA秩，异构秩场景的扩展留作未来工作
- 数据集限于分类任务，未涉及生成、检测等更复杂下游任务

## 相关工作与启发
- **FedIT**：LoRA+FedAvg的最早尝试，简单直接但忽略两大挑战
- **FFA-LoRA**：冻结A只训练B来避免聚合偏差，代价是可训练参数减半、性能受限
- **FLoRA**：通过堆叠所有客户端的LoRA模块解决聚合偏差，但通信开销大且初始化重置
- **FlexLoRA**：SVD分解重构全局更新，性能可靠但计算量更大
- **启发**：残差校正的思想可推广到其他需要在聚合阶段处理非线性关系的联邦学习场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题分解清晰，残差校正+正则化的双重设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 两个模型×两个数据集×两种non-IID设置，消融全面，但缺少NLP/LLM实验
- 写作质量: ⭐⭐⭐⭐ 问题motivation论述严谨，图表直观，逻辑链条完整
- 价值: ⭐⭐⭐⭐ 对联邦LoRA微调的两大挑战提供了首个统一解决方案，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](../../ICML2026/ai_safety/fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] FRET: Feature Redundancy Elimination for Test Time Adaptation
description: >-
  [ICCV 2025][AI安全][测试时自适应] 本文提出特征冗余消除（FRET）作为测试时自适应（TTA）的新视角，发现分布偏移时嵌入特征冗余度显著增加，并设计了S-FRET（直接最小化冗余分数）和G-FRET（基于GCN的注意力-冗余分解+双层优化）两种方法，G-FRET在多种架构和数据集上达到SOTA性能。
tags:
  - "ICCV 2025"
  - "AI安全"
  - "测试时自适应"
  - "特征冗余消除"
  - "分布偏移"
  - "图卷积网络"
  - "对比学习"
---

# FRET: Feature Redundancy Elimination for Test Time Adaptation

**会议**: ICCV 2025  
**arXiv**: [2505.10641](https://arxiv.org/abs/2505.10641)  
**代码**: [GitHub](https://anonymous.4open.science/r/fret-21BD)  
**领域**: AI安全  
**关键词**: 测试时自适应, 特征冗余消除, 分布偏移, 图卷积网络, 对比学习

## 一句话总结
本文提出特征冗余消除（FRET）作为测试时自适应（TTA）的新视角，发现分布偏移时嵌入特征冗余度显著增加，并设计了S-FRET（直接最小化冗余分数）和G-FRET（基于GCN的注意力-冗余分解+双层优化）两种方法，G-FRET在多种架构和数据集上达到SOTA性能。

## 研究背景与动机
深度神经网络在训练与测试数据独立同分布（i.i.d.）假设下表现良好，但在实际场景中常面临分布偏移（distribution shift）问题。测试时自适应（TTA）仅需访问预训练模型和未标注的测试数据，特别适合隐私敏感场景。

**现有方法分类**：
- **BN校准方法**：用目标域统计量替换训练域BN统计量
- **伪标签方法**：通过阈值或熵筛选可靠伪标签
- **一致性训练方法**：保持输入扰动下的预测稳定性
- **聚类方法**：利用聚类减少目标预测的不确定性

**核心观察**：作者在ResNet-18的CIFAR10-C上发现，随着分布偏移加剧，嵌入特征的二阶关系图（协方差矩阵）冗余度显著增加——协方差矩阵的热力图越红说明特征间相关性越高。定量分析显示，冗余分数 $R_e = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$ 与腐蚀程度呈正相关。

**核心矛盾**：现有TTA方法均未关注分布偏移导致的特征冗余增加问题，而冗余特征恰恰削弱了模型对新数据的适应能力。

**切入角度**：直接在测试时消除嵌入特征冗余，从全新的冗余消除视角解决TTA问题。

## 方法详解

### 整体框架
FRET框架分为两个层次：
1. **S-FRET**：直接将冗余分数 $R_e$ 作为优化目标，简单高效
2. **G-FRET**：引入GCN将特征关系分解为注意力部分和冗余部分，在表示层和预测层同时消除冗余并增强判别性

### 关键设计
1. **特征冗余度量（Feature Redundancy Score）**:

    - 功能：量化嵌入特征的冗余程度
    - 核心思路：对嵌入矩阵 $Z$ 按列归一化得到 $\tilde{Z}$，计算冗余分数 $R_e = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$。理想的非冗余特征应使协方差矩阵接近单位阵
    - 设计动机：协方差矩阵的非对角元素表示特征间的线性相关性，最小化这些元素可消除冗余

2. **注意力-冗余分解（Attention-Redundancy Decomposition）**:

    - 功能：将特征关系图分解为有用的注意力关系和需要消除的冗余关系
    - 核心思路：构建二阶特征关系图 $G_F = Z^TZ$，通过掩码矩阵 $M_M = I_d$ 分解为注意力图 $G_A = G_F \odot I_d$（仅保留对角线）和冗余图 $G_R = G_F - G_A$。然后通过GCN生成注意力表示和冗余表示：
    $R_A = Z D_A^{-1/2} G_A D_A^{-1/2}, \quad P_A = R_A \theta^h$
    $R_R = Z D_R^{-1/2} G_R D_R^{-1/2}, \quad P_R = R_R \theta^h$
    - 设计动机：直接最小化冗余分数的S-FRET无法感知标签分布，不能处理label shift；通过GCN将数据信息与特征关系信息融合，可以同时处理covariate shift和label shift

3. **表示层冗余消除（Representation-Layer Redundancy Elimination）**:

    - 功能：通过对比学习使注意力表示具有类别判别性，同时远离冗余表示
    - 核心思路：定义对比损失 $\mathcal{L}_R$，正样本为注意力表示 $R_{A_i}$ 与其对应的类中心 $c_o$，负样本包括其他类中心 $\{c_j\}$ 和冗余表示 $R_{R_i}$：
    $\mathcal{L}_R = -\sum_{i=1}^{n_t} \log \frac{\exp(\text{sim}(R_{A_i}, c_o))}{\sum_{j=1}^{C} \exp(\text{sim}(R_{A_i}, c_j)) + \exp(\text{sim}(R_{A_i}, R_{R_i}))}$
   类中心通过伪标签聚类计算
    - 设计动机：单纯消除冗余不够，还需增强有用特征的判别性以应对label shift

4. **预测层冗余消除（Prediction-Layer Redundancy Elimination）**:

    - 功能：在预测层增强注意力预测的置信度，同时抑制冗余预测
    - 核心思路：结合熵最小化和负学习：
    $\mathcal{L}_P = -\sum_{i=1}^{N} \sigma(P_{A_i}) \log \sigma(P_{A_i}) - \sum_{i=1}^{N} \sigma(P_{R_i}) \log \sigma(1 - P_{A_i})$
   第一项最小化注意力预测的熵（使预测更锐利），第二项通过负学习惩罚冗余预测
    - 设计动机：双层（表示层+预测层）优化比仅在单层操作更有效

### 损失函数 / 训练策略
- **S-FRET损失**：$\mathcal{L}_{SFRET} = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$
- **G-FRET总损失**：$\mathcal{L}_{GFRET} = \mathcal{L}_R + \lambda \mathcal{L}_P$
- 在线自适应：接收测试数据后，用上一步参数的模型生成预测，然后用单步梯度下降更新
- 仅更新BN层参数，保持其他参数冻结

## 实验关键数据

### 主实验（域泛化TTA - PACS + OfficeHome）

| 方法 | 骨干 | PACS Avg | OfficeHome Avg |
|------|------|----------|---------------|
| Source | ResNet-18 | 81.84 | 62.01 |
| BN | ResNet-18 | 82.66 | 62.03 |
| TENT | ResNet-18 | 85.60 | 63.24 |
| TSD | ResNet-18 | 88.13 | 62.55 |
| TEA | ResNet-18 | 87.98 | 63.06 |
| TIPI | ResNet-18 | 87.23 | 63.29 |
| **G-FRET** | **ResNet-18** | **88.51** | **63.81** |
| TSD | ResNet-50 | 89.97 | 68.74 |
| TEA | ResNet-50 | 88.72 | 68.95 |
| **G-FRET** | **ResNet-50** | **91.28** | **69.96** |

### 消融实验

| 配置 | PACS Avg | 说明 |
|------|---------|------|
| S-FRET（仅冗余最小化） | 86.20 | 简单有效，但对label shift脆弱 |
| G-FRET w/o $\mathcal{L}_R$ | 87.53 | 缺少表示层对比学习 |
| G-FRET w/o $\mathcal{L}_P$ | 87.89 | 缺少预测层负学习 |
| G-FRET (full) | 88.51 | 双层优化效果最佳 |
| $\lambda = 0.1$ | 87.92 | 平衡参数偏小 |
| $\lambda = 1.0$ | 88.51 | 最佳平衡 |
| $\lambda = 10$ | 88.05 | 预测层损失权重过大 |

### 关键发现
- 特征冗余度与分布偏移程度呈正相关，这一观察在多种架构和数据集上成立
- S-FRET虽简单，但在covariate shift场景下已经很有效
- G-FRET通过注意力-冗余分解和双层优化，在label shift场景下大幅超越S-FRET
- G-FRET生成的特征可视化显示冗余显著降低，判别性增强

## 亮点与洞察
- **全新视角**：首次将特征冗余消除引入TTA，提供了与BN校准、伪标签、一致性训练等正交的新思路
- **从简到繁的方法设计**：S-FRET简洁优雅（一行公式），G-FRET在此基础上逐步添加GCN、对比学习、负学习，逻辑清晰
- **GCN的巧妙使用**：将GCN的图传播用于建模特征间关系，使得注意力和冗余关系可以在特征层级显式分离
- **label shift的处理**：通过引入类中心感知的对比学习，G-FRET弥补了纯冗余最小化方法的不足

## 局限与展望
- G-FRET引入了GCN和对比学习，增加了测试时的计算开销（每个batch需要构建图和传播）
- 掩码矩阵 $M_M$ 固定为单位阵，可能不是所有场景的最优选择
- 对极端分布偏移（如corruption level 5）的性能提升有限
- 类中心计算依赖伪标签质量，在noise较大时可能不稳定
- 未探索与其他TTA方法的组合使用

## 相关工作与启发
- 与特征选择领域的SOFT方法有理论联系（共享掩码矩阵的思想）
- 冗余消除思路可能扩展到持续学习、域自适应等其他经典设定
- 对比学习+负学习的双层框架可能适用于其他需要特征去冗余的任务
- 为TTA方法的设计提供了新的诊断工具（冗余分数曲线）

## 评分
- 新颖性: ⭐⭐⭐⭐ 特征冗余视角是TTA领域的新贡献，但技术组件（GCN、对比学习）本身不新
- 实验充分度: ⭐⭐⭐⭐⭐ 多架构（ResNet-18/50/ViT）、多数据集（PACS/OfficeHome/CIFAR-C）、详尽消融
- 写作质量: ⭐⭐⭐⭐ 动机可视化直观，方法描述清晰，但公式排版较密集
- 价值: ⭐⭐⭐⭐ 提供了实用的TTA新方法和对特征冗余的新理解，对后续研究有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/ai_safety/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[CVPR 2026\] TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models](../../CVPR2026/ai_safety/ttp_test-time_padding_for_adversarial_detection_and_robust_adaptation_on_vision-.md)
- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)

</div>

<!-- RELATED:END -->

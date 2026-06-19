---
title: >-
  [论文解读] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights
description: >-
  [CVPR 2025][神经架构搜索] VKDNW提出了一种基于Fisher信息矩阵（FIM）特征值谱熵的训练无关NAS代理，首次成功地将Fisher信息理论应用于大规模深度网络架构搜索，无需任何训练即可评估网络分类精度潜力，并提出了更适合NAS任务的nDCG评估指标。 神经架构搜索（NAS）旨在系统地寻找最优网络架构…
tags:
  - "CVPR 2025"
  - "神经架构搜索"
  - "训练无关代理"
  - "Fisher信息矩阵"
  - "Cramér-Rao界"
  - "评估指标"
---

# VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights

**会议**: CVPR 2025  
**arXiv**: [2502.04975](https://arxiv.org/abs/2502.04975)  
**代码**: [GitHub](https://www.github.com/ondratybl/VKDNW)  
**领域**: 其他（神经架构搜索）  
**关键词**: 神经架构搜索, 训练无关代理, Fisher信息矩阵, Cramér-Rao界, 评估指标

## 一句话总结

VKDNW提出了一种基于Fisher信息矩阵（FIM）特征值谱熵的训练无关NAS代理，首次成功地将Fisher信息理论应用于大规模深度网络架构搜索，无需任何训练即可评估网络分类精度潜力，并提出了更适合NAS任务的nDCG评估指标。

## 研究背景与动机

神经架构搜索（NAS）旨在系统地寻找最优网络架构，但其核心瓶颈是天文级的计算成本——基本设置下需要训练数千个不同架构。

训练无关NAS（TF-NAS）的关键挑战：
- **现有代理的理论基础薄弱**：GradNorm、NASWOT、ZenNAS等方法多基于经验启发，缺乏严格的统计理论支撑
- **Fisher信息在大网络上的应用失败**：尽管FIM是分析网络的"显而易见的选择"，但在大规模过参数化模型上面临数值不稳定和计算不可行的问题
- **评估指标不适合NAS目标**：Kendall's $\tau$和Spearman's $\rho$对所有网络一视同仁，而NAS实际只关心能否识别出最好的网络

核心洞察：网络架构的质量可以通过权重估计过程的难度来衡量——FIM特征值越均匀，权重估计的不确定性在各方向上越一致，训练越容易，架构越好。

## 方法详解

### 整体框架

VKDNW分三步：(1) 在随机初始化的网络上使用随机输入计算经验Fisher信息矩阵（无需真实标签）；(2) 提取FIM特征值谱的十分位数表示；(3) 计算特征值分布的熵作为架构质量代理，与可训练层数结合形成最终排序。

### 关键设计

**设计一：稳定高效的FIM特征值谱估计**

- **功能**：在大规模过参数化深度网络上可靠地计算Fisher信息矩阵的特征值
- **核心思路**：将经验FIM分解为$\hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^N A_n^T A_n$的形式，利用softmax函数的分析公式处理数值不稳定的矩阵分解。仅从每个可训练层抽取少量代表性参数构建低维FIM。利用SVD替代特征值分解处理半正定矩阵的病态谱
- **设计动机**：直接计算$p \times p$（$p$为参数量）的FIM不可行。初始化时网络输出严重不平衡，必须正确处理$\text{diag}(\sigma) - \sigma\sigma^T$项以避免上溢/下溢。少量代表性参数即可捕获架构的本质结构特性

**设计二：VKDNW — 基于FIM特征值谱熵的架构代理**

- **功能**：量化网络架构的训练可行性，不依赖网络大小
- **核心思路**：定义$\text{VKDNW}(f) = -\sum_{k=1}^9 \tilde{\lambda}_k \log \tilde{\lambda}_k$，其中$\tilde{\lambda}_k = \lambda_k / \sum_j \lambda_j$是FIM特征值谱的归一化十分位数。VKDNW在所有特征值相等时取最大值，特征值越不均匀值越低。与可训练层数$\aleph(f)$结合：$\text{VKDNW}_\text{single}(f) = \aleph(f) + \text{VKDNW}(f)$
- **设计动机**：基于Cramér-Rao界，FIM特征值的均匀性反映了权重估计在各方向上的不确定性平衡。特征值越均匀意味着训练难度在各参数方向上越一致，预测对权重扰动的敏感性也越平衡。归一化使代理独立于网络大小，与层数组合同时捕获架构结构质量和容量

**设计三：nDCG评估指标 — 关注顶部排序质量**

- **功能**：更准确地评估TF-NAS代理对最佳网络的识别能力
- **核心思路**：借鉴信息检索中的NDCG指标，定义$\text{nDCG}_P = \frac{1}{Z} \sum_{j=1}^P \frac{2^{\text{acc}_{k_j}} - 1}{\log_2(1+j)}$，其中$P$为考虑的顶部网络数量。高精度网络出现在排名前列时获得更高权重
- **设计动机**：Kendall's $\tau$和Spearman's $\rho$均等对待所有排序对，无法区分"顶部排序差但底部排序好"和"顶部排序好但底部排序差"的两种代理。NAS只关心找到最佳网络，nDCG正是衡量这一能力的标准指标

### 损失函数

VKDNW是一种无需训练的评分方法，不涉及损失函数。经验FIM使用网络初始化权重$\theta_\text{init}$和随机输入计算，不需要真实数据标签。

## 实验关键数据

### 主实验：NAS-Bench-201搜索空间

| 方法 | CIFAR-10 KT | CIFAR-100 KT | ImageNet16 KT | CIFAR-10 nDCG | ImageNet16 nDCG |
|------|------------|-------------|--------------|--------------|----------------|
| FLOPs | 0.623 | 0.586 | 0.545 | 0.745 | 0.403 |
| GradNorm | 0.328 | 0.341 | 0.310 | 0.509 | 0.265 |
| NASWOT | 中 | 中 | 中 | 中 | 中 |
| **VKDNW** | **最高** | **最高** | **最高** | **最高** | **最高** |

### 关键发现

- VKDNW在三个数据集和两个搜索空间上均达到SOTA，在KT、SPR和nDCG三个指标上全面领先
- 有趣发现：简单的"可训练层数"$\aleph$作为代理，在多个指标上显著优于FLOPs
- VKDNW的信息与模型大小正交，二者组合效果优于单独使用
- 经验FIM不依赖真实标签（使用随机输入即可），说明架构的内在结构特性可以在初始化时被捕获
- nDCG指标揭示了之前方法间被传统指标掩盖的显著差异

## 亮点与洞察

1. **理论深度**：首次成功将Fisher信息理论应用于大规模NAS，克服了长期存在的数值和计算障碍
2. **优雅的代理设计**：FIM特征值谱的熵捕获了"训练难度均衡性"这一直觉，既有统计理论支撑又简洁实用
3. **评估指标的贡献**：nDCG的引入填补了TF-NAS评估的空白，toy example的展示极具说服力

## 局限与展望

- 基于Cramér-Rao界的动机假设FIM在正确权重处评估，而实际在随机初始化处，理论保证有gap
- 十分位数的选择和9个代表性特征值的设定较为经验性
- 主要在分类任务上验证，对检测/分割等任务的泛化性有待验证
- 未来可探索FIM在训练全过程中的动态变化

## 相关工作与启发

- **NASWOT/ZenNAS**：经验性TF-NAS代理，缺乏理论支撑
- **GradNorm/GraSP**：基于梯度的代理，VKDNW提供了互补信息
- **Fisher信息在ML中的应用**：EWC（持续学习）、自然梯度优化，本文将其扩展到架构搜索
- 启发：统计学中经典的理论工具（如FIM、Cramér-Rao界）在深度学习中仍有巨大的应用潜力，关键在于解决计算可行性问题

## 评分

⭐⭐⭐⭐ — 理论贡献扎实，首次成功将FIM应用于NAS令人印象深刻。nDCG评估指标的提出具有独立价值。实验全面且结果一致。对TF-NAS领域提供了新的理论视角。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[ECCV 2024\] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](../../ECCV2024/others/auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)

</div>

<!-- RELATED:END -->

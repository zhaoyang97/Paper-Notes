---
title: >-
  [论文解读] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging
description: >-
  [ICML2025][优化/理论][联邦学习] 针对高数据异质性下 FedSAM 泛化失败的问题，提出 FedSWA（周期学习率 + EMA 聚合）和 FedMoSWA（动量方差缩减控制变量），在理论和实验上均证明优于 FedSAM 及其变体，在 CIFAR-100 Dirichlet-0.1 上比 FedSAM 高出 21.8% 准确率。
tags:
  - "ICML2025"
  - "优化/理论"
  - "联邦学习"
  - "泛化"
  - "随机权重平均"
  - "数据异质性"
  - "方差缩减"
  - "损失景观平坦性"
---

# FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging

**会议**: ICML2025  
**arXiv**: [2507.20016](https://arxiv.org/abs/2507.20016)  
**代码**: [junkangLiu0/FedSWA](https://github.com/junkangLiu0/FedSWA)  
**领域**: 优化 / 联邦学习  
**关键词**: 联邦学习, 泛化, 随机权重平均, 数据异质性, 方差缩减, 损失景观平坦性

## 一句话总结

针对高数据异质性下 FedSAM 泛化失败的问题，提出 FedSWA（周期学习率 + EMA 聚合）和 FedMoSWA（动量方差缩减控制变量），在理论和实验上均证明优于 FedSAM 及其变体，在 CIFAR-100 Dirichlet-0.1 上比 FedSAM 高出 21.8% 准确率。

## 研究背景与动机

- **联邦学习泛化挑战**：FL 中数据异质性（Non-IID）使全局模型倾向收敛到尖锐局部极小值，泛化性能差
- **FedSAM 的失败**：SAM 在局部客户端寻找的是**局部平坦极小值**而非**全局平坦极小值**，在高异质性场景（Dirichlet-0.1）下反而比 FedAvg 更差（CIFAR-100：FedSAM 40.1% vs FedAvg 45.8%）
- **SAM 的额外计算开销**：SAM 需要额外的前向和反向传播计算扰动，效率低于 SWA
- **核心观察**：SWA 通过平均训练后期的不同权重，天然能找到损失景观中平坦区域的中心点，更适合联邦场景

## 方法详解

### 整体框架

提出两个递进算法：FedSWA 改进服务器聚合策略，FedMoSWA 进一步通过控制变量对齐局部与全局模型。

### FedSWA：周期学习率 + EMA 聚合

**局部学习率衰减策略**：每轮通信内，学习率从 $\eta_l$ 线性衰减到 $\rho \eta_l$：

$$\eta_k^t = \eta_l \left(1 - \frac{k}{K}\right) + \frac{k}{K} \rho \eta_l$$

局部更新：$\theta_{i,k+1}^{(t)} = \theta_{i,k}^{(t)} - \eta_k^t \cdot g_i(\theta_{i,k}^{(t)})$

**服务器 EMA 聚合**（受 LookAhead 启发）：

$$v_t = \frac{1}{s} \sum_{i=1}^{s} \theta_{i,K}^t, \quad \theta_t = \theta_{t-1} + \alpha (v_t - \theta_{t-1})$$

- 每轮通信结束后恢复初始大学习率（learning rate restart），帮助跳出较差的局部极小值
- 与 FedAvg 的区别：FedAvg 使用恒定学习率 + 简单平均聚合，FedSWA 使用周期学习率 + EMA 聚合

### FedMoSWA：动量方差缩减控制变量

在 FedSWA 基础上引入动量方差缩减机制，局部更新变为：

$$\theta_{i,k+1}^{(t)} = \theta_{i,k}^{(t)} - \eta_k^t \left( g_i(\theta_{i,k}^{(t)}) - c_i + m \right)$$

其中 $c_i$ 是客户端控制变量，$m$ 是服务器控制变量。

**客户端控制变量更新**（两种选项，实验中用选项 II）：

$$c_i^+ \leftarrow c_i - m + \frac{1}{\sum_k \eta_k^t} (\theta_{t-1} - \theta_{i,K}^{(t)})$$

**服务器控制变量更新**（动量机制）：

$$m \leftarrow m + \gamma \frac{1}{s} \sum_{i \in \mathcal{S}} (c_i^+ - m)$$

**与 SCAFFOLD 的关键区别**：SCAFFOLD 的全局变量 $c$ 对新旧 $c_i$ 赋予相同权重，存在更新延迟问题。FedMoSWA 的动量更新给最新上传的 $c_i$ 更高权重，有效缓解客户端参与率低时的延迟问题。

### 泛化误差理论分析

基于一致稳定性（uniform stability）建立 FL 泛化分析框架：

- **FedSWA 泛化界**：$\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\tilde{c}L + \tilde{c}\sigma_g + \tilde{c}\sigma)\right)$
- **FedMoSWA 泛化界**：$\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\tilde{c}L + \sigma_g + \tilde{c}\sigma)\right)$
- **FedSAM 泛化界**：$\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\bar{c}L + \bar{c}\sigma_g + \bar{c}\sigma)\right)$

其中 $\tilde{c} = 1 + (2+1/KT)^{K-1}/T \gg 1$，$\bar{c} > \tilde{c}$。FedMoSWA 中数据异质性项 $\sigma_g$ 前的系数从 $\tilde{c}$ 降为 1，显著减小异质性对泛化的影响。

**FedMoSWA 优化误差（非凸）**：$\mathcal{O}\left(\frac{\sigma\sqrt{F}}{\sqrt{TKs}} \sqrt{1+s/\alpha^2} + \frac{\beta F}{T}(m/s)^{2/3}\right)$，不受数据异质性参数 $\sigma_g$ 影响。

## 实验关键数据

### 主实验（Table 2, Dirichlet-0.6）

| 方法 | CIFAR-10 ResNet-18 | CIFAR-100 ResNet-18 |
|------|-------------------|-------------------|
| FedAvg | 86.0% | 54.2% |
| FedSAM | 83.6% | 51.9% |
| SCAFFOLD | 85.9% | 54.1% |
| MoFedSAM | 87.0% | 60.1% |
| FedSWA | 89.5% | 59.8% |
| **FedMoSWA** | **91.2%** | **67.9%** |

### 不同异质性程度（Table 3, CIFAR-100 ResNet-18）

| 方法 | Dir-0.1 | Dir-0.3 | Dir-0.6 |
|------|---------|---------|---------|
| FedSAM | 40.1% | 49.0% | 51.9% |
| MoFedSAM | 51.5% | 57.5% | 60.1% |
| FedSWA | 50.3% | 55.5% | 59.8% |
| **FedMoSWA** | **61.9%** | **66.2%** | **67.9%** |

### Tiny ImageNet (ViT-Base)

| 方法 | Dir-0.1 | Dir-0.3 | Dir-0.6 |
|------|---------|---------|---------|
| FedAvg | 70.9% | 71.8% | 72.8% |
| SCAFFOLD | 71.6% | 72.5% | 73.1% |
| FedSWA | 71.9% | 72.6% | 73.2% |
| **FedMoSWA** | **73.8%** | **74.4%** | **74.7%** |

### 消融实验

| 消融项 | 发现 |
|--------|------|
| 学习率衰减 $\rho$ | $\rho=0.1$ 最优；$\rho=1, \alpha=1$ 时 FedSWA 退化为 FedAvg，验证 SWA 的有效性 |
| EMA 系数 $\alpha$ | $\alpha=1.5$ 最优，过大性能下降 |
| 动量参数 $\gamma$ | $\gamma=0.2$ 最优，$\gamma=0.05$ 收敛极慢 |
| 控制变量消融 | $\rho=1, \alpha=1$ 时 FedMoSWA 退化为带方差缩减的 FedAvg（65.9%），优于 SCAFFOLD（52.3%） |

### 关键发现

1. **异质性越高优势越大**：Dir-0.1 下 FedMoSWA 比 MoFedSAM 高 10.4%，Dir-0.6 下高 6.2%
2. **通信效率更高**：FedMoSWA 以更少通信轮数达到目标精度（CIFAR-100 Dir-0.6：330轮 vs MoFedSAM 603轮）
3. **损失景观更平坦**：FedMoSWA 的全局训练损失曲面明显平坦于所有基线

## 亮点与洞察

1. **精准定位 FedSAM 失败机制**：明确指出 FedSAM 在高异质性下找到的是局部平坦而非全局平坦极小值，是一个有价值的实证发现
2. **SWA 替代 SAM 思路新颖**：将集中式学习中的 SWA 成功迁移到 FL，且计算效率更高（无需额外前向/反向传播）
3. **理论完整**：同时给出泛化界和优化界，并证明 FedMoSWA 在两个指标上均严格优于 FedSAM
4. **动量更新克服 SCAFFOLD 延迟问题**：对 SCAFFOLD 全局变量更新延迟的分析和改进具有实际意义
5. **实验覆盖全面**：3个数据集 × 4种网络 × 3种异质性程度 × 10种基线，共55+组实验

## 局限与展望

1. **仅测试图像分类**：未在 NLP、推荐系统等其他 FL 应用场景验证
2. **客户端参与率固定为 10%**：未探讨极低参与率（如 1%）下的表现
3. **通信开销**：FedMoSWA 需要额外传输控制变量 $c_i^+ - m$，通信量约为 FedAvg 的两倍（与 SCAFFOLD 相同）
4. **强凸假设**：理论泛化界的部分结果依赖强凸假设，与深度学习非凸实际有差距
5. **超参数敏感**：$\alpha, \gamma, \rho$ 三个额外超参数需要调优，增加部署难度

## 相关工作与启发

- **FedSAM / MoFedSAM**（Qu et al., 2022）：SAM 用于 FL 局部训练的先驱工作
- **SCAFFOLD**（Karimireddy et al., 2020）：方差缩减控制变量思想的来源，FedMoSWA 用动量改进其全局变量更新
- **SWA**（Izmailov et al., 2018）：随机权重平均的原始工作，FedSWA 将其适配到 FL 场景
- **FedACG**（Kim et al., 2024）：前瞻梯度一致性方法，在部分配置下是次优基线
- **LookAhead**（Zhang et al., 2019）：EMA 聚合策略的灵感来源

## 评分

- 新颖性: ⭐⭐⭐⭐ — SWA+动量方差缩减的组合在 FL 中首次提出
- 实验充分度: ⭐⭐⭐⭐⭐ — 多数据集、多网络、多异质性、完整消融
- 写作质量: ⭐⭐⭐⭐ — 理论和实验结构清晰，loss surface 可视化直观
- 价值: ⭐⭐⭐⭐ — 高异质性 FL 泛化改进显著，计算效率优于 SAM 方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)

</div>

<!-- RELATED:END -->

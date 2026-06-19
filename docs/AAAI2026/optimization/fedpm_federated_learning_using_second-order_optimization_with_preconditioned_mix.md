---
title: >-
  [论文解读] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters
description: >-
  [AAAI 2026][优化/理论][联邦学习] 提出 FedPM（Federated Preconditioned Mixing），一种新型联邦学习方法，通过在服务器端用"预条件混合"替代传统的简单参数平均，解决了现有二阶联邦优化方法中局部预条件器漂移问题，在理论上证明了强凸目标的超线性收敛速率，并在异质数据场景中显著超越现有方法。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "联邦学习"
  - "二阶优化"
  - "预条件混合"
  - "数据异质性"
  - "收敛分析"
---

# FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters

**会议**: AAAI 2026  
**arXiv**: [2511.09100](https://arxiv.org/abs/2511.09100)  
**代码**: [https://github.com/rioyokotalab/fedpm](https://github.com/rioyokotalab/fedpm)  
**领域**: 优化  
**关键词**: 联邦学习, 二阶优化, 预条件混合, 数据异质性, 收敛分析

## 一句话总结

提出 FedPM（Federated Preconditioned Mixing），一种新型联邦学习方法，通过在服务器端用"预条件混合"替代传统的简单参数平均，解决了现有二阶联邦优化方法中局部预条件器漂移问题，在理论上证明了强凸目标的超线性收敛速率，并在异质数据场景中显著超越现有方法。

## 研究背景与动机

联邦学习（FL）方法可根据两个维度分为四类：

| 类别 | 优化方式 | 服务器处理 | 代表方法 |
|------|---------|-----------|---------|
| FOGM | 一阶 | 梯度混合 | PSGD |
| FOPM | 一阶 | 参数混合 | FedAvg, FedProx, SCAFFOLD |
| SOGM | 二阶 | 梯度混合 | FedNL, FedNew, FedNS |
| SOPM | 二阶 | 参数混合 | LocalNewton, LTDA, FedSophia |

现有 SOPM 方法存在一个**根本性局限**：服务器端对局部参数的简单混合（简单平均）**并不对应真正的全局二阶优化**。

具体分析：在单步局部更新（K=1）时，SOGM 的全局更新为：

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \underbrace{\left(\frac{1}{N}\sum_{i=1}^N \nabla^2 f_i(\boldsymbol{\theta}^{(t)})\right)^{-1}}_{\text{全局预条件器}} \underbrace{\frac{1}{N}\sum_{i=1}^N \nabla f_i(\boldsymbol{\theta}^{(t)})}_{\text{全局梯度}}$$

而同样 K=1 的 SOPM 简化为：

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \frac{\eta}{N}\sum_{i=1}^N \underbrace{(\nabla^2 f_i(\boldsymbol{\theta}^{(t)}))^{-1}}_{\text{局部预条件器}} \underbrace{\nabla f_i(\boldsymbol{\theta}^{(t)})}_{\text{局部梯度}}$$

关键区别：SOPM 使用的是**局部预条件梯度之和**而非**全局预条件的全局梯度**。在异质数据设置下，局部曲率（每个客户端的 Hessian）是全局曲率的糟糕近似，导致**局部预条件器漂移**，严重阻碍收敛。

## 方法详解

### 整体框架

FedPM 的核心思想是将理想的全局二阶优化更新（公式 6）分解为客户端的局部参数更新和服务器端的参数混合。这种分解自然地导出了"预条件混合"——一种感知曲率的参数混合方式，替代传统简单平均。

### 关键设计

#### 1. **预条件混合的推导**: 从全局二阶优化出发的精确分解

**核心推导过程**（公式 8）：从理想全局二阶更新出发

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{g}^{(t)}$$

通过插入 $(\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}^{(t)}$ 恒等变换和 $\frac{1}{N}\sum_{i=1}^N \boldsymbol{P}_i^{(t)}(\boldsymbol{P}_i^{(t)})^{-1}$ 展开，经过代数推导得到：

$$\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)} \underbrace{(\boldsymbol{\theta}^{(t)} - \eta(\boldsymbol{P}_i^{(t)})^{-1}\boldsymbol{g}_i^{(t)})}_{\text{客户端局部二阶更新}}$$

其中 $\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t)}$（全局预条件器），$\boldsymbol{P}_i^{(t)} = \nabla^2 f_i(\boldsymbol{\theta}^{(t)})$（局部预条件器）。

**FedPM 单步更新规则**：

- **客户端**：$\boldsymbol{\theta}_i^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta(\boldsymbol{P}_i^{(t)})^{-1}\nabla f_i(\boldsymbol{\theta}^{(t)})$
- **服务器**：$\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t)}$
  
  $\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N \underbrace{(\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)}\boldsymbol{\theta}_i^{(t+1)}}_{\text{预条件混合}}$

**与简单混合的区别**：简单混合使用 $\frac{1}{N}\sum_i \boldsymbol{\theta}_i$，而预条件混合使用 $\frac{1}{N}\sum_i (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)}\boldsymbol{\theta}_i$。后者通过全局预条件器的逆与局部预条件器的乘积来重新加权各客户端的参数，从而捕获全局曲率信息。

**设计动机**：这不是一个启发式设计，而是从理想全局优化公式的精确数学分解自然得出的。在 K=1 时，FedPM 的全局更新等价于理想二阶优化。

#### 2. **多步局部更新扩展**: 自然推广至 K>1 的通信高效场景

将单步分解自然扩展到多步局部更新：

- **客户端**：进行 K 步局部二阶更新，每步计算 $\boldsymbol{P}_i^{(t,k)} = \nabla^2 f_i(\boldsymbol{\theta}_i^{(t,k)})$ 并更新参数
- **服务器**：$\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t,K-1)}$（使用最后一步的预条件器）

  $\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t,K-1)}\boldsymbol{\theta}_i^{(t,K)}$

**额外通信开销**：与传统 SOPM 不同，FedPM 需要传输局部参数和局部预条件器。但在异质数据场景中的性能优势足以弥补这一开销。

#### 3. **FOOF 预条件器近似**: 大规模 DNN 的高效实现

直接计算完整 Hessian 矩阵在 DNN 中不可行（$\mathcal{O}(d^2)$ 存储）。采用 FOOF（Fisher Information Matrix 近似）：

- 将 FIM 近似为块对角矩阵，每块对应 DNN 的一层
- 每块进一步近似为该层输入的无中心协方差矩阵 $\boldsymbol{A}_{i,l}^{(t,k)}$
- 更新规则：$\boldsymbol{\theta}_{i,l}^{(t,k+1)} = \boldsymbol{\theta}_{i,l}^{(t,k)} - \eta \text{vec}((\boldsymbol{A}_{i,l}^{(t,k)})^{-1}\text{vec}^{-1}(\boldsymbol{g}_{i,l}^{(t,k)}))$

| 操作 | FedPM (完整 Hessian) | FedPM + FOOF |
|------|---------------------|--------------|
| 构建 | $\mathcal{O}(Md^2)$ | $\mathcal{O}(Md)$ |
| 求逆 | $\mathcal{O}(d^3)$ | $\mathcal{O}(d\sqrt{d/L})$ |
| 通信 | $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ |

### 理论收敛分析

**Theorem 1**（非形式化）：在强凸性和 Hessian 光滑性假设下，当初始参数充分接近最优解时，FedPM 在单步局部更新（K=1）下实现**超线性收敛速率**。

分析基于 FedNL 的策略，但有一个微妙区别：FedPM 使用**最新的 Hessian**进行预条件，这比 FedNL 原始证明中使用的陈旧 Hessian 更直接地应用了牛顿法。

## 实验关键数据

### Test 1: 强凸模型（验证理论）

使用带 L2 正则化的逻辑回归，w8a (d=300, N=142 clients) 和 a9a (d=123, N=80 clients) 数据集。

实验结果：
- FedPM 和 FedNL 显著优于所有其他方法
- 参数距离 $\|\boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*\|$ 的快速加速下降与超线性收敛理论一致
- FedPM (K=1) 与 FedNL 的全局更新等价，结果几乎一致（微小差异源于数值精度）

### 主实验 - Test 2: 非凸 DNN 模型

| 方法 | CIFAR10 α=1.0 | CIFAR10 α=0.1 | CIFAR100 α=1.0 | CIFAR100 α=0.1 |
|------|--------------|--------------|----------------|----------------|
| FedAvg | 70.1±1.3 | 63.1±2.1 | 59.8±0.7 | 52.4±0.9 |
| FedAvgM | 72.6±1.5 | 64.8±3.0 | 61.2±0.2 | 54.9±0.6 |
| FedProx | 70.1±1.1 | 62.8±1.3 | 56.6±0.7 | 50.7±0.9 |
| SCAFFOLD | 71.4±0.5 | 65.3±2.7 | 64.7±1.1 | 58.2±0.6 |
| FedAdam | 70.4±1.6 | 64.6±2.8 | 58.3±0.3 | 51.9±0.4 |
| LocalNewton | 73.4±1.4 | 62.4±2.3 | 70.2±0.3 | 61.9±0.6 |
| **FedPM** | **74.8±0.5** | **68.6±2.0** | **68.4±2.2** | **63.1±0.8** |

设置：CIFAR10 用简单 CNN，CIFAR100 用 ResNet18（BatchNorm 替换为 GroupNorm），N=10 客户端，5 个 local epochs，100 轮通信。$\alpha$ 为 Dirichlet 分布参数，越小异质性越强。

### 消融实验

**不同 local epoch 数量的影响**（CIFAR10, α=0.1, 总 local epochs 固定 500）：

| 配置 | 通信轮数 | Local Epoch | FedAvg | SCAFFOLD | LocalNewton | FedPM |
|-----|---------|------------|--------|----------|-------------|-------|
| 低通信频率 | 50 | 10 | ~60% | ~62% | ~60% | ~66% |
| 中等通信频率 | 100 | 5 | ~63% | ~65% | ~62% | ~69% |
| 高通信频率 | 500 | 1 | ~67% | ~68% | ~69% | ~72% |

FedPM 在所有设置下一致性地优于其他方法。

### 关键发现

1. **FedPM 在高异质性场景提升最大**：α=0.1 时，FedPM 在 CIFAR10 上比 LocalNewton 高 +6.2%，而 α=1.0 时仅高 +1.4%。这证实了预条件混合对数据异质性的鲁棒性
2. **LocalNewton 在异质数据下退化严重**：α=0.1 时 LocalNewton 性能甚至低于一些一阶方法，这正是局部预条件器漂移问题的体现
3. **FedPM 在通信轮数和运行时间上都更快**：得益于高效的预条件器近似和减少的通信轮数需求
4. **FOOF 近似有效**：在 FOOF 近似下，FedPM 仍显著优于使用完整 Hessian 的 LocalNewton
5. **超线性收敛在强凸情况下得到验证**：参数距离的加速下降模式与理论预测一致

## 亮点与洞察

1. **问题分析精准**：清晰地识别了现有 SOPM 方法的核心缺陷——简单混合导致"局部预条件梯度之和≠全局预条件的全局梯度"
2. **推导优雅**：从一个简单的恒等变换出发，精确分解全局二阶更新为客户端-服务器更新规则
3. **理论与实践统一**：超线性收敛理论在实验中得到验证，FOOF 近似使大规模实现可行
4. **分类学贡献**：将 FL 方法系统化为 FOGM/FOPM/SOGM/SOPM 四类，并展示 FedPM 是唯一能在 SOPM 框架下实现全局二阶优化的方法

## 局限与展望

1. **额外通信开销**：需要传输局部预条件器（或 FOOF 矩阵），增加了通信量
2. **理论分析受限**：收敛证明仅限于强凸+单步更新，K>1 的理论保证缺失
3. **实验模型规模有限**：仅在简单 CNN 和 ResNet18 上验证，未在大型模型（如 Transformer）上测试
4. **隐私考量**：传输预条件器可能泄露更多关于局部数据的信息
5. **客户端采样**：主实验未使用客户端采样（全参与），实际场景中部分参与的效果有待验证

## 相关工作与启发

- 与 FedNL/FedNew/FedNS（SOGM 类）互补：FedPM 在 SOPM 框架下实现了等价的全局二阶优化，同时保持通信效率
- 预条件混合的思路可能推广到其他分布式优化场景，如去中心化学习
- FOOF 近似的成功使用为在 FL 中部署大规模二阶方法开辟了道路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 精准识别问题并给出优雅解决方案，预条件混合的推导简洁而深刻
- 实验充分度: ⭐⭐⭐⭐ — 强凸验证+DNN 实验+消融充分，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 问题动机、分类学、推导、实验组织得非常清晰
- 价值: ⭐⭐⭐⭐ — 为联邦学习中的二阶优化提供了重要理论和实践贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](../../ICML2025/optimization/sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)

</div>

<!-- RELATED:END -->

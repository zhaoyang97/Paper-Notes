---
title: >-
  [论文解读] PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture
description: >-
  [NeurIPS 2025][AI安全][纵向联邦学习] 本文提出PubSub-VFL，一种基于发布/订阅架构的高效两方纵向联邦学习框架，通过分层异步机制和基于系统画像的超参数优化，在保证隐私和模型精度的前提下实现2~7倍的训练加速和高达91%的计算资源利用率。 纵向联邦学习（VFL）允许持有不同特征的多方在不暴露原始数据的…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "纵向联邦学习"
  - "分裂学习"
  - "Pub/Sub架构"
  - "异步训练"
  - "资源异构"
---

# PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture

**会议**: NeurIPS 2025  
**arXiv**: [2510.12494](https://arxiv.org/abs/2510.12494)  
**代码**: 暂无  
**领域**: AI安全  
**关键词**: 纵向联邦学习, 分裂学习, Pub/Sub架构, 异步训练, 资源异构

## 一句话总结

本文提出PubSub-VFL，一种基于发布/订阅架构的高效两方纵向联邦学习框架，通过分层异步机制和基于系统画像的超参数优化，在保证隐私和模型精度的前提下实现2~7倍的训练加速和高达91%的计算资源利用率。

## 研究背景与动机

纵向联邦学习（VFL）允许持有不同特征的多方在不暴露原始数据的情况下协作训练模型，是数据协作的重要隐私保护方案。在VFL中，各方训练底部模型到切分层，将embedding安全传输给持有标签的主动方，再由主动方完成顶部模型训练。然而，现有VFL架构面临两大效率瓶颈：

**系统耦合问题**: 即使引入参数服务器（PS）架构提升了数据并行度，VFL的同步训练依赖和ID对齐步骤导致不同方的worker之间存在等待瓶颈。直接引入异步机制又受限于VFL特有的ID对齐约束。

**资源和数据异构问题**: 各方的计算资源和数据特征维度差异显著，现有方法通常只关注单方效率提升，忽略全局负载均衡，导致整体资源利用不足。

这两个问题的核心在于：缺乏将数据对齐与训练任务解耦的系统设计，以及缺乏考虑隐私约束下的全局资源优化策略。

## 方法详解

### 整体框架

PubSub-VFL通过三层设计解决上述问题：(1) 使用Pub/Sub架构实现**党间异步**，将ID对齐与训练解耦；(2) 在PS内部实现**党内半异步**，自适应调节同步间隔；(3) 基于系统画像的**动态规划**确定最优超参数。

### 关键设计

1. **发布/订阅架构与通道设计**: 引入嵌入通道和梯度通道两类通信通道。每个训练batch分配唯一的batch ID用于标记通道，使worker可以独立地向对应通道发布/订阅中间结果，无需等待对方。对于 $n$ 个样本和batch大小 $B$，系统维护 $\lceil n/B \rceil$ 个通道。为防止通道拥塞，设计了**缓冲机制**（FIFO淘汰过期数据，容量为 $p$ 个embedding/$q$ 个gradient）和**等待截止时间机制**（超时 $T_{ddl}$ 未收到数据则丢弃当前batch并重新分配）。

2. **党内半异步机制**: 在PS框架内进一步引入自适应同步间隔，定义为：
    $\Delta T_t = \left\lceil \frac{\Delta T_0}{2} \cdot \tanh\left(\frac{2t}{\Delta T_0} - 2\right) + \frac{\Delta T_0}{2} \right\rceil$
   其中 $\Delta T_0$ 是初始间隔，$t$ 是当前训练轮次。训练初期间隔短以保证稳定学习，随精度提升逐渐增大间隔减少同步频率。与Pub/Sub的党间异步共同构成分层异步机制。

3. **系统画像与动态规划优化**: 对双方的计算和通信延迟建模：前向传播时间 $T_f^{(a)}(B) = \frac{\lambda_a B^{\gamma_a} w_a}{C_a}$，后向传播类似。优化目标为最小化双方的最大迭代时间：
    $\min \mathcal{O}(w_A, w_P, B) = \min_{w_a, w_p, B \leq B_{max}} \left\{ \max(T_A, T_P) \right\}$
   受内存约束 $B_{max} = \min\left\{\left(\frac{\bar{M}_A - M_{A0}}{\rho_A}\right)^{1/\chi}, \left(\frac{\bar{M}_P - M_{P0}}{\rho_P}\right)^{1/\chi}\right\}$。使用动态规划在离散空间 $(w_a, w_p, B)$ 中搜索最优配置。

### 隐私保护

采用高斯差分隐私（GDP）协议对被动方发送的embedding添加扰动。作者证明了PubSub-VFL在集成GDP后仍能稳定收敛。

## 实验关键数据

### 主实验——精度对比

| 数据集 | 指标 | VFL | VFL-PS | AVFL | AVFL-PS | PubSub-VFL |
|--------|------|-----|--------|------|---------|------------|
| Energy | RMSE↓ | 84.58 | 84.44 | 85.41 | 85.39 | **85.64** |
| Blog | RMSE↓ | 23.20 | 23.12 | 23.38 | 23.45 | **22.34** |
| Bank | AUC↑ | 94.54 | 94.13 | 94.12 | 94.16 | **96.54** |
| Credit | AUC↑ | 81.90 | 81.34 | 80.83 | 80.34 | **82.34** |
| Synthetic | AUC↑ | 91.27 | 91.31 | 90.97 | 91.21 | **92.87** |

### 效率对比

| 方法 | 运行时间 | CPU利用率 | 等待时间/epoch | 通信成本 |
|------|---------|----------|--------------|---------|
| VFL | 基准 | ~50% | 较高 | 较高 |
| AVFL-PS | 次优 | ~55% | 中等 | 中等 |
| PubSub-VFL | **7×加速** | **+35% CPU** | **最低** | **最低** |

### 消融实验

| 配置 | Energy | Bank | Credit | Synthetic | 说明 |
|------|--------|------|--------|-----------|------|
| PubSub-VFL (完整) | 83.94 | **96.97** | **86.07** | **94.17** | 最佳 |
| 去掉等待截止时间 | 84.35 | 95.26 | 85.74 | 92.86 | AUC下降 |
| 去掉动态规划 | 84.07 | 96.33 | 85.79 | 93.82 | 轻微下降 |
| 去掉半异步机制 | 85.68 | 95.01 | 84.45 | 92.07 | 下降明显 |
| 去掉PubSub架构 | 83.98 | 95.17 | 85.93 | 93.52 | 效率损失 |
| 纯VFL | 84.24 | 94.97 | 83.42 | 92.74 | 基准 |

### 超参数敏感性

| Worker数量 | 精度(%) | 时间(s) | CPU(%) | 等待(s) |
|-----------|---------|---------|--------|---------|
| 4 | 92.13 | 712.78 | 67.52 | 1.47 |
| 8* | 92.06 | **668.11** | **88.04** | 1.53 |
| 20 | 92.00 | 1420.32 | 42.77 | 8.09 |

| Batch大小 | 精度(%) | 时间(s) | CPU(%) |
|-----------|---------|---------|--------|
| 32 | 92.06 | 668.11 | 88.04 |
| 256* | 92.67 | **92.54** | **91.07** |
| 1024 | 92.21 | 865.74 | 52.67 |

### 关键发现

- PubSub-VFL在不牺牲精度的前提下实现了2~7倍的训练加速
- 在资源严重异构场景（CPU比50:14）下，PubSub-VFL仍保持87.42%的CPU利用率，而AVFL-PS仅42.12%
- 分层异步机制和等待截止机制是性能的关键保障，去掉后Synthetic数据集AUC下降2.10%
- 差分隐私协议对精度和CPU利用率影响极小，但增加了通信成本

## 亮点与洞察

- **解耦设计理念**: 将ID对齐从训练任务中分离出来是一个优雅的系统设计思路，Pub/Sub架构天然适合这种松耦合场景
- **理论保证完备**: 提供了收敛性证明和差分隐私兼容性证明
- **端到端的系统优化**: 从架构设计到超参数自动选择形成完整的解决方案

## 局限与展望

- 仅支持两方学习场景，未扩展到多方VFL
- 动态规划的系统画像参数需要通过经验实验确定，自动化程度有限
- 实验使用CPU而非GPU，在GPU异构场景下的表现未知
- 通信模型采用简单的加法公式，实际管线化可以进一步减少延迟

## 相关工作与启发

- 相比FATE和PaddleFL等工业框架的PS架构，PubSub-VFL通过增加Pub/Sub层实现了更好的解耦
- 与AVFL类异步方法相比，分层异步设计更好地平衡了收敛稳定性和效率
- 启发：消息队列/事件驱动架构在分布式ML系统中的应用值得进一步探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ Pub/Sub架构引入VFL场景有新意，但核心技术（异步训练、参数服务器）都是已有技术的组合
- **实验充分度**: ⭐⭐⭐⭐ 五个数据集、多种异构场景、完整消融实验，但缺少GPU实验
- **写作质量**: ⭐⭐⭐⭐ 系统设计描述清晰，数学建模规范
- **价值**: ⭐⭐⭐⭐ 对VFL系统效率提升有实际意义，但两方限制降低了通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](../../ECCV2024/ai_safety/fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

</div>

<!-- RELATED:END -->

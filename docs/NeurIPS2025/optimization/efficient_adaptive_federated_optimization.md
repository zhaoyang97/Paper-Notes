---
title: >-
  [论文解读] Efficient Adaptive Federated Optimization
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] FedAda2/FedAda2++ 提出在联邦学习中实现高效的服务器-客户端联合自适应优化：客户端本地预条件器从零初始化（无需服务器传输），并可选地用 SM3 等内存高效优化器压缩本地统计量，在理论上保持与完整联合自适应相同的 $O(T^{-1/2})$ 收敛率，实测通信成本与 FedAvg 一致。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "自适应优化"
  - "通信效率"
  - "内存效率"
  - "联合自适应性"
---

# Efficient Adaptive Federated Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2410.18117](https://arxiv.org/abs/2410.18117)  
**代码**: 无（未提及公开代码）  
**领域**: 联邦学习 / 分布式优化  
**关键词**: 联邦学习, 自适应优化, 通信效率, 内存效率, 联合自适应性

## 一句话总结
FedAda2/FedAda2++ 提出在联邦学习中实现高效的服务器-客户端联合自适应优化：客户端本地预条件器从零初始化（无需服务器传输），并可选地用 SM3 等内存高效优化器压缩本地统计量，在理论上保持与完整联合自适应相同的 $O(T^{-1/2})$ 收敛率，实测通信成本与 FedAvg 一致。

## 研究背景与动机

**领域现状**：联邦学习中，自适应优化器（Adam、AdaGrad）可以加速收敛和提升精度。自适应性可用于服务器端（如 FedAdam）、客户端或两端联合使用。联合自适应性（joint adaptivity）已被证明效果最好。

**现有痛点**：联合自适应的朴素实现需要服务器在每轮通信中将全局预条件器（如二阶矩统计量）传输给客户端，这带来 (a) 通信开销倍增—— 每轮传输量从 $2d$ 增至 $3d$（$d$ 为模型维度）；(b) 客户端内存开销增加——需维护本地预条件器。

**核心矛盾**：联合自适应性带来性能提升，但其通信和内存成本与跨设备联邦学习的资源受限环境相矛盾。

**本文目标** 如何在保持联合自适应性优势的同时消除额外通信和内存开销。

**切入角度**：一个简单但未被正式验证的 idea——客户端每轮从零初始化本地预条件器，不需要服务器传输全局预条件器。本文首次为此策略提供收敛性理论保证。

**核心 idea**：客户端本地预条件器从零开始初始化 + 可选 SM3 内存压缩，即可实现与昂贵联合自适应相同的收敛和性能。

## 方法详解

### 整体框架
FedAda2 的每轮流程：(1) 服务器将当前模型 $x_t$ 发送给采样的客户端子集；(2) 每个客户端初始化本地预条件器为零，用自适应优化器做 $K$ 步本地更新；(3) 客户端仅回传模型更新 $\Delta_i^t$（不含预条件器）；(4) 服务器用自适应优化器聚合更新。FedAda2++ 进一步在客户端用 SM3 压缩预条件器以节省内存。

### 关键设计

1. **零本地预条件器初始化（Zero Local Preconditioner Initialization）**:

    - 功能：客户端每轮重新从零初始化本地梯度统计量
    - 核心思路：在跨设备联邦学习中客户端是无状态的，两次参与间隔可能很长导致旧状态陈旧。从零初始化消除了服务器→客户端的预条件器传输，通信量保持 $2d$（与 FedAvg 一致）
    - 设计动机：实测发现从零初始化的性能不劣于传输全局预条件器的方案，同时通信量减半
    - 额外好处：服务器和客户端可以使用不同的优化器（如服务器 Adam + 客户端 AdaGrad）

2. **SM3 内存高效客户端优化（FedAda2++）**:

    - 功能：用 SM3 算法压缩客户端本地预条件器
    - 核心思路：SM3 将二阶矩统计量按参数组维度分解为低秩近似，对 ViT 模型预条件器仅需 0.48% 的额外内存（vs 完整预条件器的 100%），99% 内存节省
    - 设计动机：资源受限设备（手机、IoT）无法维护完整的 Adam 二阶矩

3. **延迟预条件器更新**:

    - 功能：允许客户端每 $z$ 步才更新一次本地统计量
    - 核心思路：减少本地计算开销，理论分析表明收敛率对 $z$ 鲁棒
    - 设计动机：进一步减轻客户端计算负担

4. **混合优化框架（Blended Optimization）**:

    - 功能：允许每个客户端在每轮使用不同的优化器策略
    - 核心思路：框架统一了同构和异构优化器配置，优化器选择可基于设备资源动态调整
    - 设计动机：实际部署中不同设备资源差异大

### 理论保证

**Theorem 4.1**：在 $L$-平滑、梯度有界假设下，FedAda2/FedAda2++ 满足：
$$\min_{t \in [T]} \|\nabla f(x_{t-1})\|^2 \leq \frac{\Psi_1 + \Psi_2 + \Psi_3 + \Psi_4 + \Psi_5}{\Psi_6}$$

**Corollary 4.3**：收敛率为 $O(T^{-1/2})$，与完整联合自适应方法（如 FedAdam + 传输预条件器）相同。

## 实验关键数据

### 主实验（ViT 微调）

| 数据集 | 方法 | 最终测试精度 | 通信 | 客户端内存 |
|--------|------|------------|------|-----------|
| CIFAR-100 | FedAvg | 最低 | 2d | d |
| CIFAR-100 | FedAdam (server-only) | 中等 | 2d | d |
| CIFAR-100 | Costly Joint Adap. | 高 | 3d | 2d |
| CIFAR-100 | **FedAda2** | **高** | **2d** | 2d |
| CIFAR-100 | **FedAda2++** | **高** | **2d** | **~d** |

三个数据集（CIFAR-100、GLD-23K、FEMNIST）上一致趋势：FedAda2/FedAda2++ 达到昂贵联合自适应的性能，但通信成本与 FedAvg 一致。FEMNIST 上联合自适应优势最为显著（vs FedAvg 和 server-only 均有大幅提升）。

### 通信效率对比

| 方法 | 联合自适应 | 通信 | 计算(梯度调用) | 客户端内存 |
|------|----------|------|-------------|-----------|
| FedAvg | 否 | 2d | 1 | d |
| FedAdam | 否 | 2d | 1 | d |
| MIME | 否 | 5d | 3 | 4d |
| Costly Joint Adap. | 是 | 3d | 1 | 2d |
| **FedAda2** | **是** | **2d** | 1 | 2d |
| **FedAda2++** | **是** | **2d** | 1 | **~d** |

FedAda2 是唯一通信量=FedAvg 且具有联合自适应性的方法。

### 差分隐私场景

| 方法 | StackOverflow 精度 |
|------|-------------------|
| FedAvg | 最低 |
| FedAdam | 中等 |
| **FedAda2/FedAda2++** | **显著最高** |

在 $(ε,δ)=(13.1, 0.0025)$ 隐私预算下，FedAda2 优势更大。

### 关键发现
- 零初始化本地预条件器在所有场景下不劣于传输全局预条件器，有时甚至略优
- 跨优化器配置（如 server Adam + client AdaGrad）稳健，性能无显著下降
- SM3 压缩预条件器对超参数鲁棒，推测是投影的去噪效应
- 本地 epoch 数少时 FedAda2++ 优势最大；本地 epoch 充足时优势缩小但仍稳定
- 隐私约束下联合自适应的优势更加明显

## 亮点与洞察
- **极其简单但有效的核心思想**：从零初始化本地预条件器——这个 idea 之前就有人用但缺乏理论保证，本文首次给出严格收敛证明
- **通信节省是"免费"的**：将联合自适应的通信量从 $3d$ 降到 $2d$，不损失性能，且理论收敛率完全一致
- **SM3 的 99% 内存节省**：预条件器从 $d$ 维压缩到 $0.0048d$，极大拓展了自适应优化在移动设备的可行性
- **混合优化框架**：允许异构设备使用不同优化器，是一个面向实际部署的实用特性

## 局限与展望
- 理论分析基于全批次梯度假设，向随机梯度扩展是明确的 future work
- 仅在 ViT 和 logistic regression 模型上实验，未覆盖 LLM 级别的规模
- 零初始化在本地 epoch 数很大时优势缩小，可能需要探索更智能的初始化策略
- 混合优化的最优策略选择缺乏自动化方法

## 相关工作与启发
- **vs FedAdam/FedAdaGrad**: 仅服务器端自适应，遗漏了客户端自适应带来的加速；FedAda2 补齐了客户端端
- **vs MIME/MIMELite**: MIME 传输额外优化器状态模拟集中式自适应，通信 5d/4d 且计算 3×/2×；FedAda2 通信 2d 计算 1×
- **vs Local AdaAlter/Local AMSGrad**: 仅客户端自适应，需传输客户端预条件器给服务器聚合；FedAda2 双端自适应且无额外传输

## 评分
- 新颖性: ⭐⭐⭐ 核心 idea（零初始化）简单且已被非正式使用，贡献在于理论证明和系统化研究
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多优化器配置、隐私场景、超参数敏感性分析、20 次重复运行
- 写作质量: ⭐⭐⭐⭐ 理论部分严谨，实验细节充分，但正文部分过于技术化
- 价值: ⭐⭐⭐⭐ 对联邦学习实际部署有直接指导意义，理论保证填补了空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->

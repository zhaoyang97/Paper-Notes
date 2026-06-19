---
title: >-
  [论文解读] Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching
description: >-
  [AAAI 2026][优化/理论][联邦学习] 提出 pFed1BS 框架，通过单比特随机草图实现联邦学习中上下行双向极致通信压缩（降低 99%+），同时引入基于符号的正则化器实现客户端模型个性化，在非 IID 数据场景下同时解决通信瓶颈和数据异质性两大难题。 联邦学习（FL）面临两大根本挑战：(1) 客户端数据非 IID…
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "联邦学习"
  - "通信效率"
  - "个性化"
  - "one-bit compression"
  - "random sketching"
---

# Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching

**会议**: AAAI 2026  
**arXiv**: [2511.13144](https://arxiv.org/abs/2511.13144)  
**代码**: 待确认  
**领域**: 优化 / 联邦学习  
**关键词**: 联邦学习, 通信效率, 个性化, one-bit compression, random sketching

## 一句话总结

提出 pFed1BS 框架，通过单比特随机草图实现联邦学习中上下行双向极致通信压缩（降低 99%+），同时引入基于符号的正则化器实现客户端模型个性化，在非 IID 数据场景下同时解决通信瓶颈和数据异质性两大难题。

## 研究背景与动机

联邦学习（FL）面临两大根本挑战：(1) 客户端数据非 IID 导致单一全局模型性能下降；(2) 高维模型参数在服务器与客户端之间反复传输造成巨大通信开销。这两个问题在 IoT、V2X、遥感等带宽极度受限的场景中尤为突出。

现有工作分为两条独立路线：
- **个性化联邦学习（PFL）**：如 pFedMe、Ditto、FedRep 等通过正则化或架构分层实现个性化，但仍需传输全精度高维参数，通信开销大
- **通信高效联邦学习（CEFL）**：如 OBDA（双向单比特量化）、OBCSAA（单比特压缩感知上行）、zSignFed（符号压缩）等实现了极致压缩，但只训练单一全局模型，无法处理数据异质性

作者通过 Table 1 的系统对比揭示了一个明确的研究空白：**没有现有框架同时支持上下行双向极致压缩和原生个性化能力**。pFed1BS 正是为填补这一空白而设计。

## 方法详解

### 整体框架

pFed1BS 将联邦学习问题建模为双层优化：
- **客户端层**：每个客户端 $k$ 学习个性化模型 $\bm{w}_k$，最小化包含本地损失、符号正则项和 $\ell_2$ 惩罚的目标函数
- **服务器层**：聚合所有客户端的单比特草图，生成全局共识向量 $\bm{v}$

### 核心组件

**1. 单比特随机草图通信**

每个客户端不传输高维模型参数 $\bm{w}_k \in \mathbb{R}^n$，而是传输低维单比特草图 $\text{sign}(\mathbf{\Phi}\bm{w}_k)$，其中 $\mathbf{\Phi} \in \mathbb{R}^{m \times n}$（$m \ll n$）为随机投影矩阵。服务器也仅广播单比特共识向量 $\bm{v} \in \{\pm 1\}^m$，实现上下行双向极致压缩。

**2. 基于符号的正则化器**

引入正则项 $g(\bm{v}, \mathbf{\Phi}\bm{w}_k) = \frac{1}{2}(\|\mathbf{\Phi}\bm{w}_k\|_1 - \langle\bm{v}, \mathbf{\Phi}\bm{w}_k\rangle)$，鼓励本地模型投影的符号与全局共识对齐，同时保留本地数据特征。由于 $\ell_1$ 范数不光滑，采用 $h_\gamma(\bm{z}) = \frac{1}{\gamma}\sum_i \log(\cosh(\gamma z_i))$ 进行可微近似，使梯度为 $\nabla\tilde{g} = \mathbf{\Phi}^\top(\tanh(\gamma\mathbf{\Phi}\bm{w}_k) - \bm{v})$。当 $\gamma \to \infty$ 时，$\tanh$ 逼近 $\text{sign}$ 函数。

**3. 快速 Hadamard 变换加速投影**

朴素稠密随机矩阵投影 $\mathbf{\Phi}\bm{w}$ 的复杂度为 $\mathcal{O}(mn)$，对大模型不可行。采用子采样随机 Hadamard 变换（SRHT）：$\mathbf{\Phi}\bm{w} = \bm{S}'\bm{H}\bm{D}\tilde{\bm{w}}$，其中 $\bm{D}$ 为随机符号翻转矩阵，$\bm{H}$ 为 Walsh-Hadamard 矩阵，$\bm{S}'$ 为子采样矩阵。将计算复杂度从 $\mathcal{O}(mn)$ 降至 $\mathcal{O}(n\log n)$，且无需存储稠密矩阵。

**4. 服务器聚合**

服务器收到客户端草图后，求解离散优化 $\min_{\bm{v}\in\{\pm 1\}^m} \sum_k p_k \tilde{g}(\bm{v}, \bm{z}_k)$，其闭合形式解为加权多数投票：$\bm{v}^* = \text{sign}(\sum_k p_k \bm{z}_k)$，保证聚合步骤是最优的而非启发式。

### 算法流程

每轮通信：客户端接收全局共识 $\bm{v}^t$ → 执行 $R$ 步本地 SGD（含符号正则项）→ 计算单比特草图上传 → 服务器加权多数投票生成新共识 $\bm{v}^{t+1}$ → 广播。

### 理论保证

在标准假设（$L$-光滑、有界方差等）下，证明 pFed1BS 以 $\mathcal{O}(1/(RT))$ 速率收敛到全局势函数的稳定邻域。收敛误差由三部分控制：随机噪声 $\mathcal{O}(\eta L_F \sigma^2)$、通信误差 $\mathcal{O}(\Delta_{\max}/(\eta R))$ 和客户端采样误差 $\mathcal{O}(\lambda E_S/(\eta R))$。正则化系数需满足 $\lambda = \mathcal{O}(1/n)$。

## 实验

### 实验设置

- **数据集**：MNIST、FMNIST、CIFAR-10、CIFAR-100、SVHN
- **模型**：MNIST/FMNIST 用两层 MLP，其余用 VGG 架构
- **非 IID 划分**：20 个客户端，按标签分配数据
- **基线**：FedAvg、OBDA、OBCSAA、zSignFed、EDEN、FedBAT
- **关键超参数**：$\lambda=0.0005$，$\mu=0.00001$，$\gamma=10000$，压缩比 $m/n=0.1$
- **硬件**：NVIDIA RTX 3090 Ti，10 次独立运行取平均

### 主实验结果（Table 2）

| 方法 | MNIST Acc(%) | CIFAR-10 Acc(%) | CIFAR-100 Acc(%) | 通信降幅 |
|------|-------------|-----------------|------------------|---------|
| FedAvg | 97.21 | 87.78 | 59.60 | - |
| OBDA | 92.54 | 73.26 | 42.47 | ↓96.88% |
| OBCSAA | 92.20 | 83.57 | 48.99 | ↓49.84% |
| zSignFed | 94.83 | 67.60 | 40.17 | ↓48.45% |
| EDEN | 96.50 | 84.91 | 47.55 | ↓60.88% |
| FedBAT | 96.42 | 81.20 | 46.89 | ↓61.75% |
| **pFed1BS** | **97.83** | **85.21** | **52.88** | **↓99.69%** |

pFed1BS 在所有数据集上实现最高或接近最高精度，同时通信成本降低超过 99%。在 CIFAR-10 上，pFed1BS 仅用 0.13 MB/轮即达到 85.21% 精度，而 OBDA 需 1.34 MB 却只有 73.26%。

### 收敛性分析（Figure 3-4）

在 MNIST 非 IID 场景下，pFed1BS 的收敛曲线显示：
- 相比所有基线收敛速度更快
- 最终精度更高（97.83%）
- 训练损失下降更快且更稳定

在 CIFAR-100 等更具挑战性的数据集上，其他单比特方法出现性能崩溃（如 zSignFed 仅 40.17%），而 pFed1BS 保持 52.88%，证明个性化机制对极致压缩的可行性至关重要。

## 亮点与创新

- **首次统一框架**：第一个将单比特双向通信和个性化联邦学习统一为严格的联合优化问题
- **极致压缩比**：通信成本降低 99%+ 的同时保持甚至超过全精度方法的精度
- **计算高效**：通过 Fast Hadamard Transform 将投影复杂度从 $\mathcal{O}(mn)$ 降至 $\mathcal{O}(n\log n)$
- **理论完备**：提供了考虑个性化、本地更新、单比特草图误差和客户端采样交互的完整收敛分析
- **闭式最优聚合**：服务器端聚合不是启发式方法，而是有严格证明的最优解

## 局限性

1. **模型规模有限**：实验仅在 MLP 和 VGG 上验证，未在 ResNet、Transformer 等大规模现代架构上测试
2. **压缩比固定**：$m/n=0.1$ 的压缩比为固定值，缺少自适应调整机制
3. **非 IID 类型单一**：仅测试了基于标签的非 IID 划分，未涉及特征分布偏移、数量不平衡等更复杂场景
4. **超参数敏感性未充分探讨**：$\lambda$、$\gamma$、$\mu$ 三个超参数的交互影响及鲁棒性分析不足
5. **缺少与更多 PFL 方法的对比**：未与 pFedMe、Ditto、Per-FedAvg 等经典 PFL 方法直接比较

## 相关工作

- **个性化联邦学习**：pFedMe（正则化）、Ditto（双模型）、FedRep（表示共享+个性化头）、DisPFL（个性化稀疏掩码）
- **通信高效联邦学习**：Top-k 稀疏化、混合精度量化（Chen & Vikalo 2024）、压缩感知（Li et al. 2021）、OBDA/OBCSAA/zSignFed（单比特方法）
- **随机投影**：子采样随机 Hadamard 变换（SRHT）用于高效降维

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |

**综合评分：⭐⭐⭐⭐**（4/5）

填补了通信压缩与个性化联邦学习之间的明确空白，理论扎实，但实验规模偏小且缺少与经典 PFL 方法的直接对比。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](../../NeurIPS2025/optimization/multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[CVPR 2026\] Few-for-Many Personalized Federated Learning](../../CVPR2026/optimization/few-for-many_personalized_federated_learning.md)
- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](../../ICML2025/optimization/understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](../../NeurIPS2025/optimization/personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->

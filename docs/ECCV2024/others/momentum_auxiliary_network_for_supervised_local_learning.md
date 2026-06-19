---
title: >-
  [论文解读] Momentum Auxiliary Network for Supervised Local Learning
description: >-
  [ECCV 2024 (Oral)][局部学习] 本文提出动量辅助网络（MAN），通过指数移动平均（EMA）将相邻局部块的参数信息传递到当前块的辅助网络，并引入可学习偏置弥补跨块特征差异，解决了监督局部学习中块间信息交换缺失导致的"短视"问题，在 ImageNet 上以不到 E2E 训练一半的 GPU 显存实现更高性能。
tags:
  - "ECCV 2024 (Oral)"
  - "局部学习"
  - "动量辅助网络"
  - "指数移动平均"
  - "梯度隔离"
  - "GPU显存优化"
---

# Momentum Auxiliary Network for Supervised Local Learning

**会议**: ECCV 2024 (Oral)  
**arXiv**: [2407.05623](https://arxiv.org/abs/2407.05623)  
**代码**: [https://github.com/JunhaoSu0/MAN](https://github.com/JunhaoSu0/MAN)  
**领域**: 其他 / 深度学习训练策略  
**关键词**: 局部学习, 动量辅助网络, 指数移动平均, 梯度隔离, GPU显存优化

## 一句话总结

本文提出动量辅助网络（MAN），通过指数移动平均（EMA）将相邻局部块的参数信息传递到当前块的辅助网络，并引入可学习偏置弥补跨块特征差异，解决了监督局部学习中块间信息交换缺失导致的"短视"问题，在 ImageNet 上以不到 E2E 训练一半的 GPU 显存实现更高性能。

## 研究背景与动机

**领域现状**：深度神经网络传统上使用端到端（E2E）反向传播进行训练——损失信号从最后一层逐层传回更新所有参数。这种方式在生物可信性上存在问题（生物突触是局部信号处理），更重要的是它产生了"锁定困境"——必须等待完整的前向和反向传播完成才能更新参数，导致GPU显存使用量巨大（需存储所有层的激活值和梯度）。

**现有痛点**：(1) 监督局部学习将网络分割为多个梯度隔离的局部块，每个块由独立的辅助网络提供局部监督。这种方式节省了 GPU 显存（梯度只在块内传播），但精度远不如 E2E 训练——因为梯度只在局部块内传播，块与块之间缺乏信息交流。(2) 现有改进主要集中在设计更好的辅助网络结构（如 DGL）或局部损失函数（如 InfoPro、PredSim），但仍未从根本上解决块间信息交流的问题。(3) 当网络被分成更多块时，性能差距更加显著。

**核心矛盾**：局部学习的并行性和显存优势来自梯度隔离，但梯度隔离本身切断了块间信息流。每个块只关注局部目标，可能丢弃对全局有益的信息——这就是"短视"问题。需要一种既不破坏梯度隔离、又能实现块间信息传递的机制。

**切入角度**：受动量对比学习（MoCo）中 EMA 更新的启发，作者提出让每个辅助网络不仅接受当前块的输入，还通过 EMA 吸收下一个块的参数信息。EMA 提供了一种"平滑"的信息传递方式——不需要梯度流过块边界，但可以让当前块感知到后续块的状态。

**核心 idea**：用 EMA 将相邻块的参数信息渗透到当前块的辅助网络，配合可学习偏置弥补特征差异，让局部学习获得近似全局的视野。

## 方法详解

### 整体框架

将深度网络分为 $K$ 个梯度隔离的局部块。前 $K-1$ 个块各配一个辅助网络用于局部监督。MAN 的核心改进在于辅助网络的参数更新方式：先用局部梯度更新辅助网络参数 $\gamma_j$，然后用 EMA 将下一个局部块的参数 $\theta_{j+1}$ 融入 $\gamma_j$。同时为每个辅助块引入可学习偏置 $b_j$，增强适应不同块间特征差异的能力。第 $K$ 个块直接连接输出分类器，不使用辅助网络。

### 关键设计

1. **EMA 动量参数传递（EMA-based Information Transfer）**:

    - 功能：在不打破梯度隔离的前提下实现块间信息交流
    - 核心思路：第 $j$ 个辅助网络的参数更新分两步——先用局部梯度更新 $\gamma_j \leftarrow \gamma_j - \eta_a \nabla_{\gamma_j} \mathcal{L}(\hat{y}_j, y)$，再用 EMA 融合下一块第一层的参数 $\gamma_j \leftarrow \text{EMA}(\gamma_j, \theta_{j+1})$。EMA 公式为 $\gamma_j = m \cdot \gamma_j + (1-m) \cdot \theta_{j+1}$，其中动量系数 $m = 0.995$。只使用下一个块的第一层参数（而非全部参数），以保持 GPU 显存使用的平衡
    - 设计动机：EMA 提供了渐进式的信息交流——不像直接复制参数那样粗暴，而是平滑地让辅助网络逐渐感知到后续块的学习状态。消融实验表明仅 EMA 就能将 DGL 在 CIFAR-10 ResNet-32(K=16) 上的测试误差从 14.08 降到 11.07

2. **可学习偏置（Learnable Bias）**:

    - 功能：弥补不同梯度隔离块之间的特征差异
    - 核心思路：为每个辅助网络引入额外的可学习偏置参数 $b_j$，在 EMA 更新后与 $\gamma_j$ 一起通过局部梯度优化：$(\gamma_j, b_j) \leftarrow (\gamma_j, b_j) - \eta_a \nabla_{(\gamma_j, b_j)} \mathcal{L}(\hat{y}_j, y)$。偏置参数是轻量的，仅增加约 1% 的 GPU 显存
    - 设计动机：由于不同局部块的特征分布存在差异（它们学习不同层次的特征），直接用 EMA 传递参数可能因为特征不匹配而效果有限。可学习偏置提供了适应性调整能力，特征可视化表明 EMA 和偏置的贡献是互补的

3. **通用即插即用架构**:

    - 功能：可无缝集成到任何监督局部学习方法中
    - 核心思路：MAN 不修改局部学习方法的核心架构和损失函数设计，仅改变辅助网络的参数更新方式。实验中成功与 PredSim、DGL、InfoPro 三种方法集成，在所有配置下均获得显著提升
    - 设计动机：好的方法应该是通用的。MAN 的 EMA + 偏置机制与具体的辅助网络结构和损失函数无关

### 损失函数 / 训练策略

局部损失函数沿用基线方法的设计（如 DGL 的交叉熵损失、InfoPro 的信息熵损失+重建损失）。训练策略方面：SGD 优化器 + Nesterov 动量 0.9，余弦退火学习率调度。CIFAR-10 训练 400 epochs，ImageNet 训练 90 epochs。EMA 动量系数固定为 0.995。

## 实验关键数据

### 主实验

CIFAR-10 上不同网络深度和块数的测试误差（%）：

| 方法 | ResNet-32 K=8 | ResNet-32 K=16 | ResNet-110 K=32 | ResNet-110 K=55 |
|------|-------------|--------------|----------------|----------------|
| E2E | 6.37 | 6.37 | 5.42 | 5.42 |
| DGL | 11.63 | 14.08 | 12.51 | 14.45 |
| DGL + MAN | **8.42** (↓3.21) | **9.11** (↓4.97) | **9.65** (↓2.86) | **9.73** (↓4.72) |
| InfoPro | 11.51 | 12.93 | 12.26 | 13.22 |
| InfoPro + MAN | **9.32** (↓2.19) | **9.65** (↓3.28) | **9.06** (↓3.20) | **9.77** (↓3.45) |

ImageNet 上的结果：

| 网络 | 方法 | Top1-Error | Top5-Error | GPU显存(GB) | 显存节省 |
|------|------|-----------|-----------|-----------|---------|
| ResNet-101 | E2E | 22.03 | 5.93 | 19.71 | - |
| ResNet-101 | InfoPro(K=4) | 22.81 | 6.54 | 10.37 | 47.3% |
| ResNet-101 | InfoPro*(K=4) | **21.73** (↓1.08) | **5.81** (↓0.73) | ~10.47 | ~46.9% |
| ResNet-152 | E2E | 21.60 | 5.92 | 26.29 | - |
| ResNet-152 | InfoPro(K=4) | 22.21 | 6.26 | 13.48 | 48.7% |
| ResNet-152 | InfoPro*(K=4) | **20.78** (↓1.43) | **5.56** (↓0.70) | ~13.61 | ~48.2% |

### 消融实验

在 CIFAR-10 ResNet-32(K=16) 上使用 DGL 基线：

| 配置 | EMA | 可学习偏置 | 测试误差(%) |
|------|-----|---------|-----------|
| 原始 DGL | ✗ | ✗ | 14.08 |
| + 仅 EMA | ✓ | ✗ | 11.07 |
| + 仅偏置 | ✗ | ✓ | — |
| + EMA + 偏置（完整 MAN）| ✓ | ✓ | **9.11** |

直接使用下一块参数 vs EMA 使用（CIFAR-10 DGL）：

| 方式 | 测试误差(%) |
|------|-----------|
| 不使用下一块参数 | 14.08 |
| 直接使用（无 EMA） | 有限提升 |
| EMA 使用 | **11.07** |

### 关键发现

- MAN 在所有方法、所有数据集、所有网络深度和块数配置下均带来显著提升
- 在 ImageNet 上，InfoPro + MAN 以不到 E2E 一半的 GPU 显存超越了 E2E 的性能
- EMA 和可学习偏置的贡献互补——EMA 提供全局信息，偏置弥补特征差异
- 线性可分性分析表明：加入 MAN 后，前级块学到更通用的特征（局部精度略降），后级块显著提升——这正是全局信息流动的证据
- CKA 分析证实 MAN 使局部学习的特征表示更接近 E2E 训练

## 亮点与洞察

- **简洁而有效的信息传递机制**：EMA 是一种优雅的方式在不打破梯度隔离的前提下传递信息，计算开销极小
- **ImageNet 上超过 E2E**：以 ~47% 的显存节省获得更好的性能，这是局部学习的里程碑式成果
- **即插即用普适性**：对三种不同的局部学习方法均有效，证明了方法的通用性
- **丰富的分析**：线性可分性分析和 CKA 分析提供了 MAN 工作机理的深刻洞察——前级块学通用特征，后级块受益于全局信息

## 局限与展望

- 在常规数据集上（CIFAR-10 等），局部学习 + MAN 仍不如 E2E 精确，尤其是块数很多时
- 目前仅使用下一块的第一层参数做 EMA，探索更深层的参数传递可能进一步提升
- 仅在图像分类任务上验证，未探索在检测、分割等密集预测任务上的效果
- 可探索与自监督局部学习方法的结合

## 相关工作与启发

- **DGL**：解耦贪婪学习，MAN 的主要基线之一
- **InfoPro**：利用信息论原理保留中间层信息的局部学习方法
- **PredSim**：使用层级损失函数的局部学习先驱
- **MoCo**：EMA 在对比学习中的成功应用，启发了 MAN 的设计
- 启发：EMA 是一种通用的跨边界信息传递工具，可能在其他梯度隔离场景（如联邦学习、模型并行训练）中也有价值

## 评分

- 新颖性: ⭐⭐⭐⭐（EMA 用于局部学习的块间信息传递思路新颖且有效）
- 实验充分度: ⭐⭐⭐⭐⭐（4个数据集、3种基线方法、多种网络架构、详尽消融和分析）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐（ECCV Oral，对局部学习领域有重要推动）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->

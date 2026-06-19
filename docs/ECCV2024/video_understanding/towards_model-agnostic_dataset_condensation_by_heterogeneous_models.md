---
title: >-
  [论文解读] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models
description: >-
  [ECCV 2024][视频理解][数据集蒸馏] 提出异构模型数据集压缩（HMDC）方法，通过同时使用两个结构不同的模型（如 ConvNet 和 ViT）进行数据集压缩，并设计梯度平衡模块和互蒸馏机制，生成对各种模型普遍适用的压缩图像，解决传统方法过度适配单一模型的问题。 数据集蒸馏（Dataset Condensation…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "数据集蒸馏"
  - "模型无关"
  - "异构模型"
  - "梯度平衡"
  - "知识蒸馏"
---

# Towards Model-Agnostic Dataset Condensation by Heterogeneous Models

**会议**: ECCV 2024  
**arXiv**: [2409.14538](https://arxiv.org/abs/2409.14538)  
**代码**: [有](https://github.com/KHU-AGI/HMDC)  
**领域**: 视频理解  
**关键词**: 数据集蒸馏, 模型无关, 异构模型, 梯度平衡, 知识蒸馏

## 一句话总结

提出异构模型数据集压缩（HMDC）方法，通过同时使用两个结构不同的模型（如 ConvNet 和 ViT）进行数据集压缩，并设计梯度平衡模块和互蒸馏机制，生成对各种模型普遍适用的压缩图像，解决传统方法过度适配单一模型的问题。

## 研究背景与动机

数据集蒸馏（Dataset Condensation, DC）旨在从大规模训练数据中生成极少量的合成图像，使模型在这些图像上训练后能达到接近全数据集的性能。然而，**现有方法存在严重的模型依赖性问题**：

- 传统方法几乎都使用 3 层 ConvNet 进行压缩和评估（ConvNet-to-ConvNet）
- 压缩出的图像在 ConvNet 上表现优异，但迁移到 ResNet、ViT 等广泛使用的模型时，性能急剧下降
- 这意味着压缩图像被**过度压缩（over-condensed）**到特定模型上

如图所示，IDC、DREAM 等方法在 ConvNet 上远超随机采样，但在 ResNet 和 ViT 上甚至不如随机采样。这严重限制了数据集蒸馏的实用价值——引入新模型就需要重新生成压缩数据，违背了减少存储和计算的初衷。

## 方法详解

### 整体框架

HMDC 同时使用两个异构模型 $f_{\theta_1}$（如 ConvNet）和 $f_{\theta_2}$（如 ViT-tiny）进行梯度匹配式数据集压缩。核心思路是：通过两个结构完全不同的模型的交叉约束，提取更通用的特征，避免对单一模型的过拟合。

框架面临两个核心挑战：

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| 梯度幅度差异 | 不同结构/深度的模型对合成图像产生的梯度大小差异显著 | 梯度平衡模块（GBM） |
| 语义距离 | 两个模型学习到的特征语义随训练逐渐分化 | 互蒸馏 + 空间-语义分解（MD + SSD） |

### 关键设计

**梯度平衡模块（GBM）**：为每个优化目标维护一个梯度幅度累加器 $\mathcal{A}$，记录各损失梯度的最大值历史。通过累加器的归一化倒数来缩放各损失，确保不同模型对合成图像的影响力相当：

$$\mathcal{L}_{\text{target}} = \textbf{L} \cdot \min(\mathcal{A}) \mathcal{A}^{\mathrm{R}}$$

其中 $\mathcal{A}^{\mathrm{R}}$ 是累加器的逐元素倒数。实践中每 10 步采样一次以降低计算开销。

**空间-语义分解（SSD）**：不同模型的特征在维度、空间大小、层数方面都不同，需要统一表示才能进行蒸馏。SSD 将特征分解为：
- **语义特征**：代表整幅图像的全局信息（ViT 的 CLS token / CNN 的全局平均池化结果）
- **空间特征**：每个空间位置的局部信息（ViT 的 image tokens / CNN 的特征图）

通过双线性插值对齐空间尺寸，通过可学习仿射变换对齐特征维度，通过 softmax 层匹配矩阵对齐层数。

**互蒸馏（MD）**：在训练过程中，对两个模型经过 SSD 对齐后的特征施加 MSE 约束，缩小语义距离：

$$\mathcal{L}_{\text{MD}}(\mathbf{x}) = \text{MSE}(\text{SSD}(f_{\theta_1}(\mathbf{x}), f_{\theta_2}(\mathbf{x})))$$

互蒸馏损失同时用于模型训练和合成图像优化（通过其梯度的匹配）。

### 损失函数 / 训练策略

总损失包含 3 个优化目标，通过 GBM 自动平衡：

1. $\mathcal{L}^1$：模型 1 的梯度匹配损失（真实图像 vs 合成图像在 $f_{\theta_1}$ 上的梯度差异）
2. $\mathcal{L}^2$：模型 2 的梯度匹配损失
3. $\text{MSE}(\nabla \mathcal{L}_{\text{MD}}(\mathbf{x}^t), \nabla \mathcal{L}_{\text{MD}}(\mathbf{x}^s))$：互蒸馏梯度匹配

训练细节：
- 异构模型对：ConvNet + ViT-tiny
- 迭代次数：100 次外循环 × 100 次内循环
- 模型学习率 0.001，仿射层/层匹配矩阵学习率 0.01
- SGD 优化器，batch size 128

## 实验关键数据

### 主实验（表格）

**CIFAR-10 上不同模型的测试准确率（IPC=10）**：

| 方法 | ConvNet | ResNet18 | ResNet50 | ResNet101 | ViT-tiny | ViT-small | ViT-base | 平均 |
|------|---------|----------|----------|-----------|----------|-----------|----------|------|
| Random | 36.45 | 56.59 | 69.55 | 82.03 | 59.41 | 90.11 | 81.26 | 67.91 |
| IDC | 46.67 | 56.35 | 67.53 | 60.94 | 50.23 | 68.67 | 64.32 | 59.24 |
| DREAM | 49.23 | 57.97 | 67.85 | 64.12 | 55.40 | 71.77 | 68.88 | 62.18 |
| **HMDC** | **47.54** | **69.75** | **77.99** | **82.25** | **73.60** | **89.02** | **85.58** | **75.10** |

### 消融实验（表格）

**各组件贡献（CIFAR-10, IPC=10）**：

| GBM | MD | ConvNet | ResNet18 | ResNet50 | ViT-tiny | ViT-small | 平均 |
|-----|-------|---------|----------|----------|----------|-----------|------|
| ✗ | ✗ | 49.23 | 57.97 | 67.85 | 55.40 | 71.77 | 62.18 |
| ✗ | ✗（两模型无模块） | 46.42 | 72.11 | 76.46 | 74.09 | 90.01 | 74.09 |
| ✓ | ✗ | 47.43 | 71.62 | 76.92 | 73.67 | 86.62 | 73.21 |
| ✗ | ✓ | 47.30 | 72.01 | 78.05 | 70.96 | 86.25 | 73.40 |
| ✓ | ✓ | 47.54 | 69.75 | 77.99 | 73.60 | 89.02 | 75.10 |

### 关键发现

1. **现有方法的模型依赖性惊人**：IDC、DREAM 等在 ResNet/ViT 上甚至不如随机采样，说明合成图像严重过拟合 ConvNet
2. **HMDC 在 IPC=1 时优势最大**：在极少样本场景中，通用特征的价值更高
3. **GBM 和 MD 有协同效应**：单独使用时各有偏向（GBM 偏小模型，MD 偏大模型），组合使用达到平衡
4. **HMDC 仅需 100 迭代**：而其他方法通常需要 1,200–20,000 迭代，效率大幅提升
5. **ConvNet 上有轻微性能牺牲**：为换取跨模型通用性，HMDC 在 ConvNet 上略低于专门优化的方法

## 亮点与洞察

- **揭示了数据集蒸馏领域的重要盲区**：现有 ConvNet-to-ConvNet 评估范式掩盖了模型依赖问题
- **异构模型互补的思想**：两个结构完全不同的模型相互约束，提取的"共识"特征更具通用性
- **无需超参搜索的梯度平衡**：GBM 通过历史梯度自适应调节，避免了手动设置损失权重
- **SSD 的通用性**：空间-语义分解可适配任意带有空间特征的模型架构（不限于 CNN 和 ViT）

## 局限与展望

1. 仅在 CIFAR-10 上进行了充分实验，需要在更大规模数据集（ImageNet 等）上验证
2. 目前仅使用两个模型，可以探索三个或更多异构模型的组合
3. 预训练模型（ImageNet-1K）的使用使得大模型评估不完全公平
4. 合成图像的可视化分析和解释性不足
5. 可以探索将 HMDC 与轨迹匹配等其他蒸馏范式结合

## 相关工作与启发

- 数据集蒸馏的核心范式包括梯度匹配（DC, IDC, DREAM）和分布匹配（CAFE, IDM），本文基于梯度匹配增加异构模型约束
- 知识蒸馏从"大模型教小模型"被重新用于"缩小两个模型间的语义距离"
- 多任务学习中的梯度平衡方法启发了 GBM 的设计

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次提出模型无关的数据集蒸馏，异构模型策略新颖 |
| 技术深度 | 4 | GBM + SSD + MD 的组合设计合理，解决了实际问题 |
| 实验充分性 | 3.5 | 仅 CIFAR-10 实验，虽模型覆盖广但数据集单一 |
| 实用性 | 4 | 解决了数据集蒸馏的核心实用性问题 |
| 总体 | 4 | 对数据集蒸馏领域的重要贡献，值得关注 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)
- [\[ECCV 2024\] SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild](semtrack_a_large-scale_dataset_for_semantic_tracking_in_the_wild.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](../../CVPR2025/video_understanding/heterogeneous_skeleton-based_action_representation_learning.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->

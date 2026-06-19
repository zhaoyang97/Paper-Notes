---
title: >-
  [论文解读] End-to-End Implicit Neural Representations for Classification
description: >-
  [CVPR 2025][3D视觉][隐式神经表示] 提出 Meta Weight Transformer (MWT)，通过端到端元学习 SIREN 初始化参数和学习率调度，让 INR 的权重结构同时优化重建质量和分类性能，使用简单标准 Transformer 在 SIREN 权重上分类即可超越所有等变架构方法，首次在高分辨率 ImageNet-1K 上实现 INR 分类。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "隐式神经表示"
  - "SIREN分类"
  - "元学习"
  - "权重空间"
  - "Transformer"
---

# End-to-End Implicit Neural Representations for Classification

**会议**: CVPR 2025  
**arXiv**: [2503.18123](https://arxiv.org/abs/2503.18123)  
**代码**: [https://github.com/SanderGielisse/MWT](https://github.com/SanderGielisse/MWT)  
**领域**: 3D视觉  
**关键词**: 隐式神经表示, SIREN分类, 元学习, 权重空间, Transformer

## 一句话总结
提出 Meta Weight Transformer (MWT)，通过端到端元学习 SIREN 初始化参数和学习率调度，让 INR 的权重结构同时优化重建质量和分类性能，使用简单标准 Transformer 在 SIREN 权重上分类即可超越所有等变架构方法，首次在高分辨率 ImageNet-1K 上实现 INR 分类。

## 研究背景与动机

**领域现状**：隐式神经表示（INR）如 SIREN 将图像编码为 MLP 权重参数 $\theta$，在信号重建上效果出色。但在分类等下游任务上，需要在权重空间 $\theta$ 上设计分类器 $g(\theta)$，由于权重存在置换对称性和缩放对称性（不同排列/缩放的权重可对应同一函数），直接分类非常困难。

**现有痛点**：当前主流方法（DWS-Net、NFN、ScaleGMN 等）致力于设计对权重对称性等变的架构来处理这些对称性。但即使采用了复杂的等变设计，INR 分类性能仍然远低于基于像素的 CNN 方法。更关键的是，这些方法采用两步式流程：先给每张图单独拟合 INR（不考虑分类反馈），再在 INR 权重上训练分类器——分类器无法影响 INR 的权重结构。

**核心矛盾**：INR 权重缺乏足够的"结构性"，使下游模型难以识别有用的图像特征。研究表明共享初始化并减少更新步数能提升分类，但更少步数会降低重建质量——重建和分类之间存在 trade-off。

**本文目标** 如何让 INR 的权重结构既保留良好重建质量，又具有下游分类器可解读的结构？如何让分类目标反过来影响 INR 的拟合过程？

**切入角度**：通过元学习将 INR 拟合过程嵌入分类器训练循环中，让分类损失通过反向传播影响 SIREN 的共享初始化和每步学习率，实现端到端优化。由于只需少量更新步（k=4~6），计算效率高，可扩展到高分辨率图像。

**核心 idea**：通过端到端元学习让分类损失指导 SIREN 初始化和学习率调度的优化，使拟合后的权重结构天然适合分类，无需设计复杂的等变架构。

## 方法详解

### 整体框架
系统包含三个可学习组件：(1) 共享 SIREN 初始化参数 $\theta$；(2) 每步每参数的学习率调度 $\alpha \in \mathbb{R}^{k \times |\theta|}$；(3) 标准 Transformer 分类器 $h_\psi$。训练流程：对每张训练图像，从共享 $\theta$ 出发用 MSE 重建损失做 $k$ 步内循环更新得到图像特定参数 $\phi$，然后用 Transformer 分类 $\phi$。外循环同时反向传播重建损失和分类损失来更新 $\theta$、$\alpha$（元学习）以及分类器 $\psi$。推理时对新图像也是先做 $k$ 步更新再分类。

### 关键设计

1. **端到端元学习 SIREN（Meta-Learned SIREN）**:

    - 功能：学习一个共享初始化和学习率调度，使得 $k$ 步更新后的权重同时具有好的重建质量和分类友好的结构
    - 核心思路：内循环用 MSE 重建损失做 $k$ 步 SGD 更新：$\phi \leftarrow \phi - \alpha_i \nabla_\phi \mathcal{L}_{rec}$。外循环计算两个梯度——重建梯度 $g^{rec}_{\theta,\alpha}$ 和分类梯度 $g^{cls}_{\theta,\alpha}$，用 $w_{cls}$ 加权组合后更新 $\theta$ 和 $\alpha$。内循环不用分类损失是因为测试时没有标签。
    - 设计动机：共享初始化使所有图像的权重在同一参考系下变化，减少了对称性问题。分类损失的梯度反传让初始化"适配"分类器需求。$w_{cls}=0.01$ 时最平衡。与 fit-a-nef 的手动调参不同，这里用元学习自动寻找平衡点。

2. **权重差分+缩放的 Transformer 输入处理**:

    - 功能：将 SIREN 权重转换为 Transformer 可高效处理的 token 序列
    - 核心思路：将每个隐藏层的每个输出神经元作为一个 token，特征维度为 $c_{in}+1$（输入权重+bias）。关键处理：不直接输入 $\phi$，而是输入差分 $\phi_{scaled} = \lambda(\phi - \theta + \beta)$，其中 $\lambda=500$ 是缩放因子，$\beta$ 是可学习位置偏置。对 4 层 128 维 SIREN，共有 $128 \times 4 = 512$ 个 token。
    - 设计动机：$\theta$ 和 $\phi$ 通常非常接近，差值很小，直接输入容易被 Transformer 的低频偏好忽略。取差分+大缩放让信号更明显。$\beta$ 提供位置信息，帮助 Transformer 区分不同神经元。

3. **随机像素子采样策略**:

    - 功能：降低高分辨率图像上元学习的计算成本
    - 核心思路：在每步内循环中，不用全部像素计算重建损失，而是随机采样比例 $s$ 的像素子集。当 $s=1/k$ 时平均每个像素被看到一次。实验发现即使 $s=0.05$（5% 像素）也几乎不影响分类和重建质量。
    - 设计动机：元学习需要存储 $k$ 步的计算图，对高分辨率图像（如 ImageNet-1K 的 224×224）内存消耗巨大。子采样大幅减少内存需求（从 24.1 GiB 降到 13.5 GiB），同时暗示 SIREN 可能学到了隐式的图像先验，能从部分像素推断整体。

### 损失函数 / 训练策略
- 内循环：MSE 重建损失 $\mathcal{L}_{rec}$，plain SGD 优化
- 外循环：$g_{\theta,\alpha} = g^{rec}_{\theta,\alpha} + w_{cls} \cdot g^{cls}_{\theta,\alpha}$，$w_{cls}=0.01$
- 分类器：用 $\mathcal{L}_{cls}$ 单独更新 $\psi$
- 优化器：AdamW，lr=1e-4，学习率 $\alpha$ 的 lr=1e-2
- 支持空间增强（旋转、翻转、缩放等），因为只需重新拟合 SIREN 几步

## 实验关键数据

### 主实验

| 数据集 | 指标 | MWT-L | 之前SOTA (ScaleGMN-B) | 提升 |
|--------|------|-------|----------------------|------|
| MNIST | 准确率 | **98.80%** | 96.59% | +2.2% |
| Fashion-MNIST | 准确率 | **90.43%** | 80.78% | +9.7% |
| CIFAR-10 (无增强) | 准确率 | **59.57%** | 38.82% | +20.7% |
| CIFAR-10 (有增强) | 准确率 | **64.7%** | 63.4% (inr2array) | +1.3% |
| Imagenette | 准确率 | **60.8%** | - (首次) | - |
| ImageNet-1K | 准确率 | **23.6%** | - (首次) | - |

### 消融实验

| 配置 | CIFAR-10 准确率 | PSNR | 说明 |
|------|----------------|------|------|
| WT ($w_{cls}=0$) | 43.78% | 较高 | 无分类梯度反馈 |
| MWT ($w_{cls}=0.01$) | 56.90% | 适中 | 分类梯度指导元学习 |
| MWT-L (宽=256) | **59.57%** | 适中 | 更大 SIREN + 更大 Transformer |
| $w_{cls}=0.1$ (过高) | 下降 | 下降 | 分类过度干预损害重建 |
| $k=4$ steps | 略降 | 略降 | 少步数仍能很好工作 |
| $k=6$ steps | **最优** | 较好 | 平衡计算和性能 |

### 关键发现
- 分类梯度反馈是核心贡献：MWT vs WT 在 CIFAR-10 上提升 13.1%（56.90% vs 43.78%），证明让分类损失影响 INR 结构至关重要
- WT（无分类反馈）就已媲美或超越大部分等变方法，说明共享初始化+少步更新本身就很有效
- 像素子采样几乎无损：Imagenette 上 $s=0.05$ 和 $s=0.25$ 的分类准确率差异 <1%，但训练内存减半
- $w_{cls}$ 存在明显的 sweet spot：过高会同时损害重建和分类，0.01 最优
- 首次证明 INR 分类可扩展到 ImageNet-1K 规模（23.6% top-1），尽管仍远低于像素方法

## 亮点与洞察
- **"结构"比"等变"更重要**：与其设计复杂的等变架构来处理权重对称性，不如直接通过元学习强制权重具有结构性。这是一个范式转换——从"适应对称性"到"消除对称性"。
- **端到端元学习框架的通用性**：内循环用任务 A 的损失更新参数，外循环用任务 B 的损失优化初始化——这个框架可迁移到任何"先 fit 表示、再做下游任务"的场景。
- **计算效率使高分辨率成为可能**：少步更新（k=4~6）+像素子采样使得在 ImageNet 规模上训练可行，是首个在高分辨率数据集上做 INR 分类的工作。

## 局限与展望
- INR 分类与像素分类仍有巨大差距（ImageNet 上 23.6% vs CNN 的 76%+），实用价值有限
- 元学习需要存储 $k$ 步计算图进行二阶梯度计算，训练内存开销较大
- 仅验证了 SIREN 作为 INR，未探索其他 INR 架构（如 hash-based、hybrid）
- 分类器使用标准 Transformer 未做特别优化，可能还有提升空间
- 未探索 INR 表示在其他下游任务（如检测、分割）上的迁移能力

## 相关工作与启发
- **vs ScaleGMN**: ScaleGMN 设计了同时对置换和缩放对称性等变的图网络，MWT 不考虑任何等变性但性能远超，挑战了"必须设计等变架构"的范式。
- **vs fit-a-nef**: fit-a-nef 发现共享初始化和少步更新帮助分类，但手动调参；MWT 进一步用元学习自动优化并引入分类反馈，性能大幅提升。
- **vs inr2array (NFT)**: inr2array 在 CIFAR-10 有增强时达到 63.4%，MWT-L 达到 64.7%，但 MWT 的增强计算代价更低（只需重拟合几步）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 端到端元学习让分类指导 INR 表示结构，范式创新
- 实验充分度: ⭐⭐⭐⭐⭐ 从 MNIST 到 ImageNet-1K 全覆盖，大量消融，首次设立高分辨率基线
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融分析到位
- 价值: ⭐⭐⭐⭐ 在 INR 分类方向有重要推动，但与像素方法差距仍大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] SiNR: Sparsity Driven Compressed Implicit Neural Representations](sinr_sparsity_driven_compressed_implicit_neural_representations.md)
- [\[AAAI 2026\] FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM](../../AAAI2026/3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)
- [\[ICML 2026\] EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](../../ICML2026/3d_vision/eps3d_end-to-end_feed-forward_3d_panoptic_segmentation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?
description: >-
  [AAAI 2026][模型压缩][二值神经网络] 本文提出 BD-Net，通过引入 1.58-bit 卷积和 pre-BN 残差连接，首次成功将深度可分离卷积（depth-wise convolution）应用于二值神经网络（BNN），在 ImageNet 上以 33M OPs 的极低计算量实现了 BNN 领域的新 SOTA，多个数据集上精度提升最高达 9.3 个百分点。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "二值神经网络"
  - "深度可分离卷积"
  - "低比特量化"
  - "轻量化网络"
---

# BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?

**会议**: AAAI 2026  
**arXiv**: [2511.17633](https://arxiv.org/abs/2511.17633)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 二值神经网络, 深度可分离卷积, 模型压缩, 低比特量化, 轻量化网络

## 一句话总结

本文提出 BD-Net，通过引入 1.58-bit 卷积和 pre-BN 残差连接，首次成功将深度可分离卷积（depth-wise convolution）应用于二值神经网络（BNN），在 ImageNet 上以 33M OPs 的极低计算量实现了 BNN 领域的新 SOTA，多个数据集上精度提升最高达 9.3 个百分点。

## 研究背景与动机

**领域现状**：模型压缩是深度学习部署到边缘设备的关键技术之一。二值神经网络（BNN）将权重和激活都量化为 $\{-1, +1\}$，使得卷积运算可以用高效的 XNOR 和 popcount 操作替代，理论上可实现极致的计算和存储压缩。近年来 BNN 研究取得了显著进展，如 ReActNet、IR-Net 等方法逐步缩小了与全精度网络的精度差距。

**现有痛点**：深度可分离卷积（depth-wise convolution，DWConv）是轻量化网络（如 MobileNet）的核心组件，能够大幅降低参数量和计算量。然而，在 BNN 中直接应用 DWConv 会导致严重的性能退化。原因在于：（1）DWConv 每个通道只有一个卷积核，信息容量本身就很有限；（2）在二值化后，每个权重只有 $\{-1, +1\}$ 两种取值，表达能力进一步被极大压缩；（3）二值化引入的量化误差在 DWConv 的逐通道运算中无法通过通道间的信息交互来缓解。

**核心矛盾**：BNN 追求极致压缩率，而 DWConv 的逐通道分离结构进一步限制了信息流通，两者叠加导致表征能力严重不足。此外，二值化带来的梯度近似（STE）在 DWConv 场景下会导致训练不稳定，Hessian 矩阵条件数恶化，优化困难。

**本文目标**：（1）提升 BNN 中 DWConv 的表征能力，使其能有效编码特征；（2）稳定 BNN 训练过程，改善优化景观；（3）实现首个在 BNN 中成功使用 DWConv 的网络架构。

**切入角度**：作者观察到，将权重从严格二值 $\{-1, +1\}$ 微幅扩展到三值 $\{-1, 0, +1\}$（即 1.58-bit），可以在几乎不增加计算成本的前提下显著提升表达能力。同时，通过在残差连接中引入 pre-BN 结构，可以有效改善 Hessian 条件数，稳定训练。

**核心 idea**：用 1.58-bit 卷积替代严格二值卷积来增强 DWConv 的表达力，并用 pre-BN 残差连接来稳定优化，从而首次成功在 BNN 中使用深度可分离卷积。

## 方法详解

### 整体框架

BD-Net 基于 MobileNet V1 架构构建。输入为标准 RGB 图像，经过一系列二值化的深度可分离卷积层进行特征提取，最终输出分类结果。整体 pipeline 包含：（1）1.58-bit DWConv 模块替代传统二值 DWConv；（2）pre-BN 残差连接结构替代传统残差连接；（3）标准的分类头。

### 关键设计

1. **1.58-bit 卷积（Ternary Weight Convolution）**:

    - 功能：提升二值 DWConv 的表征能力，使每个通道的卷积核能编码更丰富的特征模式。
    - 核心思路：在传统 BNN 中，权重被约束为 $\{-1, +1\}$，信息量为 1 bit/权重。BD-Net 将权重空间扩展为 $\{-1, 0, +1\}$，即 $\log_2(3) \approx 1.58$ bit/权重。零值的引入使得网络可以"选择性忽略"某些输入，起到类似注意力门控的作用。具体实现时，使用两个二值掩码的组合来表示三值权重，使得计算仍然可以高效执行。
    - 设计动机：对于 DWConv 这种逐通道操作，每个核本身参数极少（通常 $3 \times 3 = 9$ 个参数），二值化后只有 $2^9 = 512$ 种可能的核模式，远远不够编码丰富的视觉特征。引入零值后，可能的模式增加到 $3^9 = 19683$ 种，表达能力提升约 38 倍。

2. **Pre-BN 残差连接（Pre-BatchNorm Residual Connection）**:

    - 功能：稳定 BNN 的训练过程，改善损失景观，加速收敛。
    - 核心思路：传统 BNN 的残差连接格式为 $y = \text{BN}(\text{Conv}(x)) + x$，BD-Net 改为 $y = \text{Conv}(\text{BN}(x)) + x$ 的 pre-BN 格式。这一改变使得 BN 层的归一化操作在二值化之前执行，保证了输入到 sign 函数的分布更加稳定和对称。
    - 设计动机：作者通过理论分析表明，pre-BN 结构可以显著改善 Hessian 矩阵的条件数。在 BNN 训练中，STE（Straight-Through Estimator）近似带来的梯度噪声会被不良的 Hessian 条件数放大，导致训练振荡和收敛困难。Pre-BN 通过规范化 sign 函数的输入分布，减少了梯度估计的方差，使优化更加平坦稳定。

3. **架构级适配设计**:

    - 功能：将 1.58-bit DWConv 和 pre-BN 残差整合到 MobileNet V1 的二值化版本中。
    - 核心思路：在 MobileNet V1 的每个深度可分离卷积块中，DWConv 部分采用 1.58-bit 量化，PWConv（pointwise convolution）部分保持标准二值化。通道扩展因子和网络宽度根据计算预算进行调整。
    - 设计动机：DWConv 是信息瓶颈所在，因此优先提升其表征能力；PWConv 参数量大但有丰富的通道间交互，二值化后仍有足够表达力。这种差异化量化策略实现了精度-效率的最优平衡。

### 损失函数 / 训练策略

采用标准的交叉熵损失函数进行训练。训练策略包括：（1）使用知识蒸馏从全精度 MobileNet 教师模型中迁移知识；（2）采用两阶段训练——先训练全精度模型预热，再进行二值化微调；（3）学习率采用 cosine annealing 调度。

## 实验关键数据

### 主实验

| 数据集 | 指标 | BD-Net | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| ImageNet | Top-1 Acc | -- | -- | 新SOTA（33M OPs） |
| CIFAR-10 | Top-1 Acc | -- | -- | +9.3pp |
| CIFAR-100 | Top-1 Acc | -- | -- | 显著提升 |
| STL-10 | Top-1 Acc | -- | -- | 显著提升 |
| Tiny ImageNet | Top-1 Acc | -- | -- | 显著提升 |
| Oxford Flowers 102 | Top-1 Acc | -- | -- | 显著提升 |

BD-Net 在 ImageNet 上以仅 33M OPs 的计算量建立了 BNN 新 SOTA，在其他五个数据集上全面超越现有方法，最大精度提升达 9.3 个百分点。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full BD-Net | 最佳 | 完整模型，1.58-bit DWConv + pre-BN 残差 |
| 标准二值 DWConv | 显著下降 | 去掉1.58-bit，用标准BNN的DWConv |
| w/o pre-BN | 下降 | 去掉 pre-BN 残差，用标准残差连接 |
| 标准 BNN (无DWConv) | 计算量更高 | 不用DWConv的传统BNN，OPs更多但精度不一定更好 |

### 关键发现

- 1.58-bit 卷积是使 DWConv 在 BNN 中可行的关键因素，去掉后精度大幅下降，说明严格二值化确实无法支撑 DWConv 的逐通道操作。
- Pre-BN 残差连接对训练稳定性有重要贡献，其效果通过 Hessian 条件数的理论分析和实验验证得到了支持。
- BD-Net 在计算量远低于现有 BNN 方法的情况下仍能取得更高精度，证明了 DWConv 在 BNN 中的潜力。
- 在小数据集（如 CIFAR-10）和大数据集（如 ImageNet）上均有效，泛化性良好。

## 亮点与洞察

- **首次成功将 DWConv 应用于 BNN** 是一个重要的里程碑。DWConv 是现代轻量化网络的基石，打通这一技术路径意味着 BNN 可以从 MobileNet、EfficientNet 等成熟架构中受益，大大拓宽了 BNN 的架构设计空间。
- **1.58-bit 的精妙设计**：从 1-bit 微调到 1.58-bit，在几乎不增加硬件成本的前提下（三值运算仍可高效实现）获得了约 38 倍的核模式扩展，这是"最小代价获取最大收益"的典范。
- **Pre-BN 的理论驱动设计**：不是盲目尝试各种BN位置，而是通过 Hessian 条件数分析明确了 pre-BN 的优势，展示了理论指导实践的范式。

## 局限与展望

- 论文仅基于 MobileNet V1 验证，未探索更先进的架构（如 MobileNet V2/V3、EfficientNet）中的效果。
- 1.58-bit 虽然在理论上近乎免费，但在实际硬件（如 FPGA、ASIC）上的部署效率需要进一步验证。
- 未与最新的混合精度量化方法进行详细对比，在精度-效率 Pareto 前沿上的位置有待进一步明确。
- 可以探索将 1.58-bit 的思路推广到其他低比特量化场景，如 2-bit、4-bit 网络中的关键层。

## 相关工作与启发

- **vs ReActNet**: ReActNet 通过 learnable shift 和 reshape 操作增强二值激活函数的表达力，但仍使用标准卷积结构。BD-Net 则从网络架构层面（DWConv）突破，两者的思路可以互补。
- **vs IR-Net**: IR-Net 关注信息保留的二值化方法，通过最小化量化前后的信息损失来提升精度。BD-Net 的 1.58-bit 可视为一种信息保留的替代方案，代价更低但效果也很出色。
- **vs BinaryConnect/XNOR-Net**: 这些经典 BNN 方法奠定了领域基础，但均未涉及 DWConv 的二值化，BD-Net 填补了这一空白。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次成功在BNN中应用DWConv是开创性工作，1.58-bit设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 在6个数据集上验证，但缺乏详细的消融数据和可视化分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，理论分析扎实
- 价值: ⭐⭐⭐⭐ 打通了BNN使用DWConv的技术路径，对BNN架构设计有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](../../ICML2026/model_compression/surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[ICLR 2026\] BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training](../../ICLR2026/model_compression/bep_a_binary_error_propagation_algorithm_for_binary_neural_networks_training.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](../../ICML2025/model_compression/an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->

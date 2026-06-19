---
title: >-
  [论文解读] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging
description: >-
  [CVPR 2025][图像恢复][单像素成像] 提出 ProxUnroll 方法，通过设计近端轨迹（PT）损失函数训练 HQS/ADMM 展开网络，使其中的深度图像修复器（DIR）逼近理想正则化的近端算子，从而让展开网络同时具备 PnP 算法的灵活性（一个模型处理任意压缩比）和展开网络的高精度高速度。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "单像素成像"
  - "算法展开"
  - "近端算子"
  - "Plug-and-Play"
  - "压缩感知重建"
---

# Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging

**会议**: CVPR 2025  
**arXiv**: [2505.23180](https://arxiv.org/abs/2505.23180)  
**代码**: [https://github.com/pwangcs/ProxUnroll](https://github.com/pwangcs/ProxUnroll)  
**领域**: 图像复原 / 压缩感知  
**关键词**: 单像素成像, 算法展开, 近端算子, Plug-and-Play, 压缩感知重建

## 一句话总结

提出 ProxUnroll 方法，通过设计近端轨迹（PT）损失函数训练 HQS/ADMM 展开网络，使其中的深度图像修复器（DIR）逼近理想正则化的近端算子，从而让展开网络同时具备 PnP 算法的灵活性（一个模型处理任意压缩比）和展开网络的高精度高速度。

## 研究背景与动机

**领域现状**：单像素成像（SPI）是一种在亚奈奎斯特采样率下用单个光敏探测器恢复图像的压缩感知技术，在太赫兹成像和非可见光3D成像中具有独特优势。目前主流求解器有两类：Plug-and-Play（PnP）算法和深度展开（Unrolling）网络。

**现有痛点**：PnP 算法通过迭代地交替执行数据保真近端算子和预训练去噪器来重建图像，灵活性高（一个模型适用于不同压缩比 CR），但精度一般、推理速度慢且需要繁琐的参数调节。展开网络将截断的迭代优化过程转化为端到端可训练网络，精度高、推理快，但必须为特定 CR 训练，CR 变化时需重新训练。

**核心矛盾**：灵活性与精度/速度之间的 trade-off。PnP 的灵活性来自预训练去噪器可近似正则化的近端算子而与退化程度无关；展开网络的高精度来自端到端优化，但训练后的子网络学到了什么缺乏可解释性，也不保证中间结果逐步改善。

**本文目标**：如何融合两者优势——让展开网络具有 PnP 的灵活性和可解释性，同时保持甚至超越展开网络的精度和速度？

**切入角度**：作者观察到，PnP 的灵活性和可解释性本质上源于深度去噪器逼近了正则化函数的近端算子。如果能让展开网络中的神经网络也逼近一个理想正则化的近端算子，就能同时获得灵活性和高精度。

**核心 idea**：定义一个以退化图像与清晰图像距离为正则化函数的理想近端算子（有解析解），用它生成"近端优化轨迹"来监督展开网络的训练，使网络中的 DIR 模块逼近该理想近端算子。

## 方法详解

### 整体框架

输入为压缩测量值 $\mathbf{Y}$ 和测量矩阵 $(\mathbf{H}, \mathbf{W})$，初始化 $\mathbf{X}^0 = \mathbf{H}^\top \mathbf{Y} \mathbf{W}$。网络由 $K=6$ 次迭代组成，每次迭代执行：(1) 显式近端算子 $\text{Prox}_f$（有封闭解的数据保真步）；(2) 深度图像修复器 DIR $\mathcal{R}_\theta$（隐式正则化步）。最终输出重建图像 $\mathbf{X}^K$。训练时用提出的近端轨迹（PT）损失同时监督每次迭代的中间输出。

### 关键设计

1. **近端轨迹（PT）损失函数**:

    - 功能：让展开网络中的 DIR 逼近理想正则化的近端算子，从而获得灵活性和可解释性
    - 核心思路：定义一个理想的显式正则化函数 $\bar{g}(\mathbf{X}') = \frac{1}{2}\|\mathbf{X}' - \mathbf{X}\|_F^2$，其近端算子有闭式解 $\text{Prox}_{\bar{g}}(\mathbf{Q}) = \frac{\mu\mathbf{Q} + \lambda\mathbf{X}}{\mu + \lambda}$，即在输入和清晰图像之间做加权平均。在训练时，利用每对 $(\mathbf{X}^0, \mathbf{X})$ 通过理想近端算法迭代产生"完美轨迹" $\mathbf{X}^0 \to \bar{\mathbf{X}}^1 \to \cdots \to \mathbf{X}$，然后让展开网络每步的输出 $\mathbf{X}^{k+1}$ 逼近轨迹上对应的 $\bar{\mathbf{X}}^{k+1}$，损失为 $\text{PL} = \sum_{k=0}^{K-1} \alpha_k \|\mathbf{X}^{k+1} - \bar{\mathbf{X}}^{k+1}\|_F^2$
    - 设计动机：端到端训练只保证最终输出接近 GT，不保证中间结果逐步改善，也不保证学到的子网络有可解释性。PT 损失通过显式轨迹监督每一步，使 DIR 逼近近端算子，从而同时获得收敛保证、可解释性和灵活性

2. **深度图像修复器（DIR）架构**:

    - 功能：作为展开网络中每次迭代共享的图像修复模块
    - 核心思路：4 层非对称 encoder-decoder 架构，采用 CNN-Transformer 混合块（CTB）。CTB 将特征均分两半，分别用 Swin Transformer（SwinSA 捕获低频全局信息）和门控动态 CNN（GD-CNN 利用 AdaConv 生成输入依赖的卷积核捕获高频细节）并行处理后融合。AdaConv 通过从输入预测系数对一组学习到的静态卷积核做线性组合来实现动态卷积
    - 设计动机：结合 CNN 的高频建模能力和 Transformer 的低频/全局建模能力，同时通过 AdaConv 保持对输入的高依赖性

3. **记忆块（Memory Block）**:

    - 功能：在迭代间传播有用的特征表示，避免信息丢失
    - 核心思路：在 encoder 的每个层级使用通道交叉注意力（ChanCA），以当前迭代的特征为 query、上一迭代的特征为 key/value，自适应地聚合先前有用表示。第一次迭代时 key/value 为零，MB 自动失效
    - 设计动机：端到端训练的展开网络中间结果不保证逐步改善，部分原因在于有用表示无法跨迭代传播。MB 通过侧路径传播特征，避免中间伪影恶化

### 损失函数 / 训练策略

总损失 = 标准像素损失 + PT 损失。训练时 CR 在 [0.01, 0.50] 范围内变化，分辨率在 256×256 到 512×512 之间变化，以覆盖尽可能广的退化程度。使用 BSD500 中 400 张图像生成 20000 训练样本，初始学习率 $1\times10^{-3}$ 降至 $1\times10^{-4}$。测量矩阵设为行正交、可训练的浮点矩阵。

## 实验关键数据

### 主实验

在 Set11 和 CBSD68 数据集上评估，CR 从 0.01 到 0.50：

| 方法 | 灵活性 | CR=0.01 | CR=0.04 | CR=0.10 | CR=0.25 | CR=0.50 | 平均PSNR |
|------|--------|---------|---------|---------|---------|---------|----------|
| SAUNet | 单CR | 22.43 | 27.80 | 32.15 | 37.11 | 41.91 | 32.28 |
| HATNet | 单CR | 22.54 | 27.98 | 32.26 | 37.24 | 42.05 | 32.41 |
| PnP-DRUNet | 多CR | 21.75 | 26.81 | 30.16 | 35.00 | 40.54 | 30.85 |
| **ADMM-ProxUnroll** | **任意CR** | **22.76** | **28.30** | **32.55** | **37.35** | **41.97** | **32.59** |

### 消融实验

| 配置 | 平均PSNR | 说明 |
|------|---------|------|
| Unrolling (SAUNet, 单CR) | 32.28 | 需为每个CR单独训练 |
| PnP-DRUNet (多CR) | 30.85 | 灵活但精度差 |
| ADMM-ProxUnroll (w/o PT loss) | ~31.5 | 去掉PT损失，灵活性下降 |
| ADMM-ProxUnroll (full) | 32.59 | 灵活性+SOTA精度 |

### 关键发现

- ProxUnroll 用单个模型在所有 CR 上都超越了需要为每个 CR 专门训练的展开网络 SAUNet 和 HATNet
- PT 损失确保了中间结果的逐步改善（收敛轨迹可视化验证），而传统展开网络的中间结果质量波动
- 记忆块对跨迭代信息传播至关重要，移除后性能明显下降
- 在真实 SPI 数据上同样表现优异，证明了实际应用潜力

## 亮点与洞察

- **PT 损失的巧妙之处**：利用理想正则化近端算子的解析解生成监督轨迹，不需要额外数据或预训练，是一种"免费"的监督信号。这个想法可迁移到任何展开网络框架
- **打破 PnP vs Unrolling 的二元对立**：证明了灵活性和精度并非不可调和，关键在于让网络模块具有近端算子的数学性质
- **动态卷积 AdaConv 的简洁设计**：通过对静态卷积核做输入依赖的线性组合，以极低开销实现了类似注意力机制的自适应能力

## 局限与展望

- 当前假设测量矩阵行正交，这在实际 SPI 相机中可能不完全成立
- PT 损失要求训练时能访问 GT 图像来计算理想轨迹，半监督或无监督场景下如何设计类似机制值得探索
- DIR 架构不够轻量，在极端实时场景下仍有优化空间
- 目前只验证了 SPI 重建，将 PT 损失推广到其他逆问题（如 CT 重建、MRI 重建）是很自然的方向

## 相关工作与启发

- **vs PnP-DRUNet**: PnP 方法用预训练去噪器隐式近似近端算子，灵活但精度受限。ProxUnroll 通过 PT 损失显式训练 DIR 逼近近端算子，精度大幅超越
- **vs SAUNet/HATNet**: 传统展开网络精度高但每个 CR 需单独训练。ProxUnroll 用单模型就超越了它们的每个 CR 的专用模型
- **vs 梯度去噪器（GS-DRUNet等）**: 过去的近端学习方法要求网络满足非扩张性或输入凸性等限制，牺牲表达力。PT 损失无此约束，通过轨迹监督间接实现近端算子性质

## 评分

- 新颖性: ⭐⭐⭐⭐ PT 损失的思路很精巧，但整体框架仍是展开网络+设计更好的子网络的范式
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实数据，多个 CR，多个 baseline，消融完善
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐ 对 SPI 领域贡献显著，PT 损失的思路有望推广到更多逆问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Bit is All You Need! Efficient Video Capture via Single Bit Imaging](../../CVPR2026/image_restoration/a_bit_is_all_you_need_efficient_video_capture_via_single_bit_imaging.md)
- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](../../ECCV2024/image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[ECCV 2024\] Intrinsic Single-Image HDR Reconstruction](../../ECCV2024/image_restoration/intrinsic_single-image_hdr_reconstruction.md)
- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)

</div>

<!-- RELATED:END -->

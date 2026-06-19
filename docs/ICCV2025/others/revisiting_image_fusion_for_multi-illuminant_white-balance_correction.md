---
title: >-
  [论文解读] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction
description: >-
  [ICCV 2025][白平衡] 针对多光源场景白平衡校正问题，提出一种基于 Transformer 的高效融合模型来替代传统线性融合，并构建了包含 16,000+ 张图像的大规模多光源白平衡数据集，在新数据集上实现比现有方法提升 100% 的校正质量。 领域现状：白平衡（White Balance, WB）是相机 ISP…
tags:
  - "ICCV 2025"
  - "白平衡"
  - "多光源"
  - "图像融合"
  - "Transformer"
  - "数据集"
---

# Revisiting Image Fusion for Multi-Illuminant White-Balance Correction

**会议**: ICCV 2025  
**arXiv**: [2503.14774](https://arxiv.org/abs/2503.14774)  
**代码**: 无  
**领域**: 图像处理  
**关键词**: 白平衡, 多光源, 图像融合, Transformer, 数据集

## 一句话总结

针对多光源场景白平衡校正问题，提出一种基于 Transformer 的高效融合模型来替代传统线性融合，并构建了包含 16,000+ 张图像的大规模多光源白平衡数据集，在新数据集上实现比现有方法提升 100% 的校正质量。

## 研究背景与动机

**领域现状**：白平衡（White Balance, WB）是相机 ISP 管线中的核心步骤，目标是消除光源色温对图像颜色的影响。大多数 WB 方法假设场景中只有一个主光源，在此假设下通过估计全局色温并应用校正矩阵即可完成。近年来出现了基于融合的 WB 方法——将同一张图像用多个预定义 WB 预设（如日光、荧光灯、阴天等）分别处理，然后通过神经网络学习像素级的融合权重。

**现有痛点**：(1) 现有融合方法仅做线性加权融合——对于每个像素，输出是各 WB 预设版本的凸组合。虽然这对单光源场景足够，但在多光源场景中严重受限，因为多光源下不同区域需要完全不同的校正策略，线性融合难以捕捉这种空间异质性。(2) 现有 WB 数据集（如 WB-sRGB、Rendered WB）缺少专门的多光源图像，导致模型在多光源场景的训练和评测都不充分。

**核心矛盾**：多光源场景要求空间自适应的非线性校正，但现有方法的线性融合假设和单光源数据集无法满足这一需求。

**本文目标**：(1) 设计一种能捕捉空间依赖关系的非线性融合模型；(2) 构建专门的大规模多光源 WB 数据集。

**切入角度**：观察到多光源场景的 WB 校正本质上是一个空间依赖的融合问题——图像不同区域受不同光源影响，需要全局信息来判断每个区域应该用哪种 WB 策略。Transformer 的长程依赖建模能力天然适合这个任务。

**核心 idea**：用高效 Transformer 替代线性融合来融合多个 WB 预设版本的图像，使模型能够利用全局上下文做出更准确的空间自适应白平衡决策。

## 方法详解

### 整体框架

输入为一张 sRGB 图像的 5 个 WB 预设版本（日光 D65、阴天、荧光灯 A、荧光灯 CWF、白炽灯），这些预设版本通过已知的 WB 预设矩阵从原始图像渲染得到。模型的目标是输出一张 WB 校正后的图像。整体采用编码器-解码器结构：编码器将 5 个版本的图像特征提取并融合，Transformer 模块建模跨预设和跨空间的依赖关系，解码器生成每个像素对应的融合系数图。

### 关键设计

1. **跨预设 Transformer 融合（Cross-Preset Transformer Fusion）**:

    - 功能：建模不同 WB 预设版本之间及其空间位置之间的依赖关系
    - 核心思路：将 5 个 WB 预设版本的特征图沿通道维度拼接后送入高效 Transformer 块。使用窗口注意力（Window Attention）+ 移位窗口（Shifted Window）机制来控制计算复杂度（类似 Swin Transformer），使自注意力在局部窗口内计算再通过移位实现跨窗口信息流通。注意力操作同时作用于空间和预设两个维度——对于每个像素位置，模型可以关注同一区域的所有预设版本以及其他区域的相同预设。
    - 设计动机：线性融合的根本局限在于各 WB 版本的融合权重独立计算，无法利用全局上下文。Transformer 的注意力机制允许模型根据图像的全局光源分布（远处区域的颜色线索）来推断局部区域的最优融合策略。

2. **轻量级多尺度编码器（Lightweight Multi-Scale Encoder）**:

    - 功能：高效提取多尺度特征
    - 核心思路：使用共享权重的轻量 CNN（基于 MobileNetV2 的变体）对 5 个 WB 版本分别提取特征，在 3 个尺度上输出特征图（1/4、1/8、1/16 分辨率）。共享权重使得参数量仅为非共享的 1/5，且保证各版本在相同特征空间中表示。多尺度特征通过 skip connection 传递给解码器，保留低级颜色细节的同时捕获高级语义。
    - 设计动机：多光源场景中既需要细粒度的像素级颜色信息（区分阴影/高光区域的光源），又需要粗粒度的场景理解（判断整体光源分布），多尺度设计同时满足两者。

3. **大规模多光源数据集（Multi-Illuminant WB Dataset）**:

    - 功能：提供专门的多光源 WB 训练和评测数据
    - 核心思路：基于物理渲染管线构建：(a) 收集多光源场景的 RAW 图像（包含已知的多光源信息）；(b) 对每张 RAW 图像用 5 种标准 WB 预设分别渲染为 sRGB；(c) 同时生成 ground truth——对每个像素根据其所受主要光源应用正确的 WB 矩阵。最终数据集包含 16,000+ 张图像对（5 个预设 + 1 个 GT），涵盖多种多光源场景——室内混合光源（窗户自然光 + 人工照明）、彩色照明、日光与阴影混合等。
    - 设计动机：现有数据集如 Set1/Set2 仅有数百张多光源图像且缺少对应的多预设渲染——新数据集填补了这一空白，使社区可以首次系统性地研究多光源 WB。

### 损失函数 / 训练策略

主损失为 L1 颜色重建损失 $\mathcal{L}_{color} = \|I_{pred} - I_{gt}\|_1$，加上一个角度损失 $\mathcal{L}_{ang}$ 用于约束色度方向的准确性（计算每个像素 RGB 向量与 GT 之间的角度误差）。总损失 $\mathcal{L} = \mathcal{L}_{color} + \alpha \mathcal{L}_{ang}$，$\alpha=0.5$。训练使用 AdamW，余弦退火学习率调度，patch-based 训练，patch 大小 256×256。

## 实验关键数据

### 主实验

在本文提出的多光源数据集上对比：

| 方法 | 类型 | MAE↓ | $\Delta E$↓ | PSNR↑ | SSIM↑ |
|------|------|------|------------|-------|-------|
| Deep WB (Afifi 2020) | 非融合 | 5.82 | 7.34 | 23.1 | 0.841 |
| Mixed-Ill WB | 非融合 | 4.95 | 6.21 | 24.3 | 0.867 |
| WB-sRGB Fusion | 线性融合 | 4.12 | 5.38 | 25.7 | 0.889 |
| CLCC | 线性融合 | 3.87 | 4.91 | 26.2 | 0.895 |
| 本文方法 | Transformer融合 | 1.93 | 2.46 | 31.4 | 0.952 |

### 消融实验

| 配置 | MAE↓ | PSNR↑ | 说明 |
|------|------|-------|------|
| Full model | 1.93 | 31.4 | 完整 Transformer 融合 |
| 用线性融合替代 Transformer | 3.72 | 26.5 | 退化到传统方案，掉点 48% |
| w/o 角度损失 | 2.31 | 30.1 | 色度方向约束重要 |
| w/o 多尺度 skip connection | 2.48 | 29.6 | 细节恢复受损 |
| 3 个预设（去掉 2 个） | 2.67 | 28.9 | 预设数量重要但边际递减 |
| 不共享编码器权重 | 1.98 | 31.2 | 无共享略好但参数量 5 倍 |

### 关键发现

- **线性融合是主要瓶颈**：用 Transformer 替代线性融合后 MAE 从 3.72 降至 1.93，说明空间依赖建模是关键
- **数据集影响显著**：同样的方法在单光源数据集上训练后应用于多光源场景性能远低于在本文数据集上训练的结果
- 5 个 WB 预设的效果显著优于 3 个，但增加到 7 个的提升有限
- 角度损失对色彩保真度的贡献大于 L1——L1 优化会倾向低亮度区域，角度损失则平等对待所有亮度

## 亮点与洞察

- **揭示线性融合的根本局限**：通过理论分析和实验证明，线性融合在多光源场景下存在原理性的不足——不同光源区域的校正不是原始预设的简单凸组合。这个发现推动了整个融合式 WB 研究范式的升级。
- **数据集贡献突出**：16,000+ 多光源图像的数据集是该领域首个专门针对多光源场景的大规模基准，填补了重要的数据空白。
- **方法设计高效实用**：基于 Swin Transformer 的窗口注意力保持了线性计算复杂度，使得方法可在实际 ISP 管线或后处理中应用。

## 局限与展望

- 方法依赖 5 个预定义 WB 预设，如果真实光源的色温不在这些预设覆盖范围内，性能可能下降
- 数据集通过物理渲染生成，与真实相机 ISP 管线的渲染可能存在 domain gap
- 未考虑视频场景——时序一致的多光源 WB 校正是更有挑战的问题
- 可以探索自适应预设选择——动态决定使用哪些/多少个 WB 预设版本
- 与 RAW 域 WB 方法的结合——先在 RAW 域做初始校正，再在 sRGB 域用融合方法做精细修正

## 相关工作与启发

- **vs Deep WB (Afifi et al.)**: Deep WB 直接从单张 sRGB 图像预测 WB 校正参数，不使用多预设融合策略。在多光源场景下因为全局估计单一色温而失败，本文的局部融合方案更适合。
- **vs CLCC**: CLCC 是最强的线性融合基线，通过色度约束改进融合权重。但其仍受线性融合限制，本文 Transformer 方案在 MAE 上提升 50%。
- **vs AWB 传统方法（Gray World/White Patch）**: 传统方法完全基于统计假设，在多光源下假设失效。本文方法通过数据驱动学习绕开了这些假设限制。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 Transformer 引入 WB 融合并构建专门数据集，组合创新扎实
- 实验充分度: ⭐⭐⭐⭐ 消融全面、对比充分，但缺少在其他已有数据集上的交叉验证
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，数据集构建过程详细
- 价值: ⭐⭐⭐⭐ 数据集对社区有长期价值，方法有实际应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion](../../CVPR2026/others/customized_fusion_a_closed-loop_dynamic_network_for_adaptive_multi-task-aware_in.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)

</div>

<!-- RELATED:END -->

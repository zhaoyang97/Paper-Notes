---
title: >-
  [论文解读] Unsupervised Multi-modal Medical Image Registration via Invertible Translation
description: >-
  [ECCV 2024][医学图像][多模态配准] 本文提出 INNReg，通过可逆神经网络将多模态医学图像翻译为单模态，再利用单模态图像进行配准，结合基于归一化互信息的 barrier 损失函数，在 MRI T1/T2 和 MRI/CT 数据集上取得了优于现有方法的配准精度。 领域现状：多模态医学图像配准在临床诊断和图像引导…
tags:
  - "ECCV 2024"
  - "医学图像"
  - "多模态配准"
  - "可逆神经网络"
  - "图像翻译"
  - "无监督学习"
  - "互信息"
---

# Unsupervised Multi-modal Medical Image Registration via Invertible Translation

**会议**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04467.pdf)
**代码**: [https://github.com/MeggieGuo/INNReg](https://github.com/MeggieGuo/INNReg)  
**领域**: 医学图像  
**关键词**: 多模态配准, 可逆神经网络, 图像翻译, 无监督学习, 互信息

## 一句话总结

本文提出 INNReg，通过可逆神经网络将多模态医学图像翻译为单模态，再利用单模态图像进行配准，结合基于归一化互信息的 barrier 损失函数，在 MRI T1/T2 和 MRI/CT 数据集上取得了优于现有方法的配准精度。

## 研究背景与动机

**领域现状**：多模态医学图像配准在临床诊断和图像引导治疗中至关重要，能为医生提供互补的解剖/功能信息。现有方法主要分为两类：基于传统相似性度量（如互信息、归一化互相关）的直接配准方法，以及基于图像翻译的间接配准方法。

**现有痛点**：直接配准方法面临多模态图像间复杂且未知的空间关系，难以设计有效的相似性度量。基于翻译的方法（如 CycleGAN、RegGAN）虽然将问题转化为单模态配准，但翻译过程中容易破坏图像的几何一致性——翻译后的图像可能在结构上与原图不一致，导致配准结果失真。

**核心矛盾**：图像翻译需要足够的表达能力来捕获跨模态的外观变换，但同时必须严格保持几何一致性。现有翻译网络（如 U-Net、ResNet）的单向映射难以同时满足这两个需求，且训练不稳定。

**本文目标** (1) 如何在图像翻译过程中严格保持几何一致性？(2) 如何设计更有效的配准损失来提升多模态配准的精度？

**切入角度**：作者观察到可逆神经网络（INN）天然具有双射特性——正向和逆向映射严格互逆，这确保了翻译前后的几何结构完全一致。同时，INN 的信息无损特性使其能保留翻译过程中的全部结构信息。

**核心 idea**：用可逆神经网络作为图像翻译器保证几何一致性，配合基于归一化互信息的 barrier 损失函数约束配准网络，实现无监督多模态医学图像配准。

## 方法详解

### 整体框架

INNReg 由两个子网络组成：(1) 基于 INN 的图像翻译网络，将不同模态的图像（如 MRI T1 和 T2）翻译到同一个模态空间；(2) 配准网络，接受翻译后的单模态图像对，预测变形场来对齐原始多模态图像。输入是一对多模态图像（浮动图像和固定图像），输出是对齐后的配准图像和对应的变形场。

### 关键设计

1. **可逆神经网络图像翻译器（INN Translator）**:

    - 功能：将来自不同模态的医学图像翻译到统一的模态空间，同时严格保持几何结构
    - 核心思路：基于仿射耦合层（affine coupling layer）构建可逆翻译网络。输入特征被分成两部分 $x_1, x_2$，通过交叉仿射变换实现可逆映射：$y_1 = x_1 \odot \exp(s_2(x_2)) + t_2(x_2)$，$y_2 = x_2 \odot \exp(s_1(y_1)) + t_1(y_1)$。其中 $s, t$ 是任意函数。逆变换只需反向计算即可精确还原输入，确保几何结构零损失
    - 设计动机：与 CycleGAN 等方法的 cycle consistency 约束不同，INN 的可逆性是数学保证的，不依赖额外的重构损失来近似几何一致性

2. **动态深度可分离卷积局部注意力机制（DDC-Local Attention）**:

    - 功能：增强 INN 翻译器中仿射函数 $s, t$ 的局部特征提取能力
    - 核心思路：在仿射耦合层的子网络中引入动态深度可分离卷积，根据输入内容动态生成卷积核权重，同时结合局部注意力机制捕获空间邻域内的模态特异性特征。这使得翻译网络能自适应地关注不同区域的模态差异
    - 设计动机：标准的仿射耦合层使用简单的 MLP 或 CNN 作为子网络，表达能力有限，难以处理医学图像中复杂的局部模态差异（如 MRI T1/T2 中灰白质的对比度反转）

3. **基于归一化互信息的 Barrier 损失（NMI-Barrier Loss）**:

    - 功能：约束配准网络的优化方向，避免局部最优解
    - 核心思路：将归一化互信息（NMI）转化为 barrier 形式的损失函数 $L_{barrier} = -\log(\text{NMI}(I_{fixed}, I_{warped}) - \tau)$，其中 $\tau$ 是一个阈值。当 NMI 接近 $\tau$ 时，损失急剧增加，形成一个"屏障"，迫使优化过程远离低 NMI 的区域
    - 设计动机：传统 NMI 损失在优化过程中梯度平坦，容易陷入局部最优。barrier 形式通过对数函数放大了 NMI 在目标阈值附近的梯度，加速收敛并提升配准精度

### 损失函数 / 训练策略

总损失为翻译损失和配准损失的加权和：$L = L_{trans} + \lambda L_{reg}$。翻译损失包括对抗损失和 L1 重建损失；配准损失包括 NMI-barrier 损失和变形场的正则化项（鼓励光滑变形场）。训练采用端到端方式，翻译网络和配准网络联合优化。

## 实验关键数据

### 主实验

| 数据集 | 指标 | INNReg | RegGAN | CycleGAN+VoxelMorph | 提升 |
|--------|------|--------|--------|---------------------|------|
| MRI T1/T2 | Dice ↑ | **0.812** | 0.782 | 0.769 | +3.8% |
| MRI T1/T2 | HD95 ↓ | **2.34** | 2.71 | 2.89 | -13.7% |
| MRI/CT | Dice ↑ | **0.776** | 0.741 | 0.728 | +4.7% |
| MRI/CT | HD95 ↓ | **3.12** | 3.58 | 3.79 | -12.8% |

### 消融实验

| 配置 | Dice ↑ | HD95 ↓ | 说明 |
|------|--------|--------|------|
| Full INNReg | **0.812** | **2.34** | 完整模型 |
| w/o INN (用 ResNet) | 0.783 | 2.67 | INN 贡献约 3.6% |
| w/o DDC-Attention | 0.795 | 2.52 | 动态注意力贡献约 2.1% |
| w/o Barrier Loss (用标准 NMI) | 0.798 | 2.48 | Barrier 损失贡献约 1.7% |
| w/o 翻译网络 (直接多模态配准) | 0.752 | 3.11 | 翻译策略至关重要 |

### 关键发现
- INN 翻译器是最关键的组件，替换为普通 ResNet 后 Dice 下降最多，证明了几何一致性保持的重要性
- Barrier 损失相比标准 NMI 损失在收敛速度上快约 30%，最终精度也更高
- 在 MRI/CT 这种模态差异更大的场景中，INNReg 的优势更加明显

## 亮点与洞察
- **INN 保证几何一致性**是本文最巧妙的设计：利用数学上的可逆性替代 cycle consistency 的近似约束，从根本上解决了翻译过程中几何失真的问题。这个思路可以迁移到任何需要保持结构一致性的图像翻译任务中
- **Barrier 损失函数**的设计灵感来自优化理论中的 barrier method，将 NMI 从一个评估指标转变为一个具有强梯度信号的优化目标，值得在其他需要互信息优化的场景中借鉴
- 端到端联合训练翻译和配准网络，避免了两阶段方法中误差梯累积的问题

## 局限与展望
- 仅在 2D 切片上进行实验，未扩展到 3D 体积配准，而临床应用通常需要 3D 配准
- 数据集规模较小（BraTS + Harvard），泛化性有待在更大规模数据集上验证
- INN 的内存消耗较大（需要存储中间状态用于逆向计算），限制了处理高分辨率图像的能力
- 未考虑大形变场景下的配准，可引入级联或 diffeomorphic 约束提升大形变处理能力

## 相关工作与启发
- **vs RegGAN**: RegGAN 使用普通 GAN 做图像翻译，依赖 cycle consistency 约束几何一致性，但这是一个软约束，无法完全避免结构失真。INNReg 用 INN 的数学可逆性取代了这个软约束
- **vs VoxelMorph**: VoxelMorph 是经典的无监督配准方法，但只能处理单模态。INNReg 通过翻译将多模态问题转化为单模态，扩展了 VoxelMorph 的适用范围
- **vs SYMNet**: SYMNet 使用对称配准损失，但不涉及跨模态翻译。两者可以结合——用 INN 翻译配合对称配准

## 评分
- 新颖性: ⭐⭐⭐⭐ INN 用于图像翻译保持几何一致性的思路新颖，但 INN 本身不是新技术
- 实验充分度: ⭐⭐⭐ 仅两个数据集，无 3D 实验，消融实验不够详细
- 写作质量: ⭐⭐⭐⭐ 动机链清晰，方法描述准确
- 价值: ⭐⭐⭐⭐ 为多模态医学图像配准提供了一个有效且有理论保证的框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](../../CVPR2025/medical_imaging/multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)

</div>

<!-- RELATED:END -->

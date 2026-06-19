---
title: >-
  [论文解读] CARL: A Framework for Equivariant Image Registration
description: >-
  [CVPR 2025][医学图像][图像配准] 提出 CARL（Coordinate Attention with Refinement Layers）——通过坐标注意力机制实现对平移和旋转的 [W,U] 等变性的深度配准框架，在多步配准架构中仅替换第一步即获得全局 [W,U] 等变性，在腹部、肺部和脑部三个医学配准基准上匹配或超越 SOTA，尤其在具有不同视野的腹部配准任务上大幅领先。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "图像配准"
  - "等变性"
  - "坐标注意力"
  - "多步配准"
  - "微分同胚"
---

# CARL: A Framework for Equivariant Image Registration

**会议**: CVPR 2025  
**arXiv**: [2405.16738](https://arxiv.org/abs/2405.16738)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 图像配准, 等变性, 坐标注意力, 多步配准, 微分同胚

## 一句话总结

提出 CARL（Coordinate Attention with Refinement Layers）——通过坐标注意力机制实现对平移和旋转的 [W,U] 等变性的深度配准框架，在多步配准架构中仅替换第一步即获得全局 [W,U] 等变性，在腹部、肺部和脑部三个医学配准基准上匹配或超越 SOTA，尤其在具有不同视野的腹部配准任务上大幅领先。

## 研究背景与动机

**领域现状**：深度学习图像配准方法（VoxelMorph、GradICON 等）通过神经网络预测变形场来估算图像对之间的空间对应关系，已在多个基准上取得强劲表现。这些方法通常是无监督的，通过最小化图像相似度损失和正则化损失来训练。

**现有痛点**：现有基于位移场预测的深度配准网络（如 VoxelMorph、GradICON）只具有 [U,U] 等变性——即当两张图像被同一个变换作用时，配准结果能保持一致。但当两张输入图像被不同的变换作用时（[W,U] 等变性），这些方法会失效。这在临床场景中是关键问题：不同 CT 扫描往往有不同的视野（field of view）、不同的患者定位和不同的裁剪范围，相当于对两张图像施加了不同的平移/旋转。

**核心矛盾**：卷积网络天然对平移具有 [U,U] 等变性（相同平移），但预测位移场的卷积网络无法实现 [W,U] 等变性（不同平移）。这是因为卷积的平移等变性作用在函数空间上（输出值跟随平移），而配准需要的等变性要求输出变换同时被两个不同的变换前后作用。

**本文目标**：设计一个深度配准框架，使其对输入图像的独立平移（和可选的旋转）具有 [W,U] 等变性，同时保持精确处理复杂局部变形的能力。

**切入角度**：注意力机制对排列等变且注意力权重和为 1——这意味着值向量经过仿射变换后，注意力输出也经历相同的仿射变换。如果将值向量设为坐标，就能计算"注意力掩码的质心"——这正是坐标注意力的核心。

**核心 idea**：用坐标注意力（将值设为坐标的标准注意力）替换多步配准网络第一步中的位移预测网络，即可获得全局 [W,U] 等变性——因为第一步 [W,U] 等变 + 后续步骤 [U,U] 等变 = 整体 [W,U] 等变。

## 方法详解

### 整体框架

CARL 采用多步多分辨率配准架构，整体结构为 $\text{CARL} = \text{TwoStep}\{\text{TwoStep}\{\text{Down}\{\text{TwoStep}\{\text{Down}\{\Xi_\theta\}\}, \Psi_1\}\}, \Psi_2\}, \Psi_3\}$。其中 $\Xi_\theta$ 是本文提出的坐标注意力网络（[W,U] 等变），$\Psi_i$ 是标准的 U-Net 位移预测网络（[U,U] 等变），Down 是降采样算子。输入为一对 3D 医学图像，输出为（近似微分同胚的）变形场。

### 关键设计

1. **坐标注意力网络 $\Xi_\theta$**:

    - 功能：实现粗配准阶段的 [W,U] 等变变形场估计
    - 核心思路：对给定的 moving/fixed 图像对，先用共享的卷积编码器（含膨胀卷积，操作在同一分辨率上）分别提取特征，然后用标准注意力操作——以 fixed 图像特征为 Query、moving 图像特征为 Key、而 **moving 图像的体素坐标** 为 Value。这样注意力输出就是"每个 fixed 体素在 moving 图像上对应位置的加权质心"，即直接输出坐标映射而非位移。编码器前后加 padding/cropping 以处理边界效应，使用 flash attention 在 $43^3$ 体积上高效计算
    - 设计动机：标准注意力对排列等变（Key/Value同排列时输出不变），且对 Value 的仿射变换等变（因为权重和为 1）。平移是排列（在体素空间）也是仿射（在坐标空间），所以坐标注意力兼具两者，自然实现 [W,U] 等变性

2. **TwoStep 多步配准的等变性传递**:

    - 功能：保证整体网络的 [W,U] 等变性
    - 核心思路：TwoStep 算子定义为 $\text{TwoStep}\{\Phi, \Psi\}[I_M, I_F] = \Phi[I_M, I_F] \circ \Psi[I_M \circ \Phi[I_M, I_F], I_F]$，即先用 $\Phi$ 粗配准，再将 warped 后的 moving 图像和 fixed 图像送入 $\Psi$ 细化。论文严格证明：若 $\Phi$ 是 [W,U] 等变、$\Psi$ 是 [U,U] 等变，则 TwoStep 整体为 [W,U] 等变。这一性质可递归应用于任意步数
    - 设计动机：单独的坐标注意力网络精度有限（在低分辨率 $43^3$ 上操作），需要后续精细化步骤。TwoStep 理论保证了"第一步 [W,U] + 后续 [U,U] = 整体 [W,U]"，因此只需替换第一步就能升级整个 GradICON 架构

3. **旋转等变扩展 CARL{ROT}**:

    - 功能：将等变性从平移扩展到任意旋转
    - 核心思路：三个修改——（1）编码器对 4 种轴对齐 $\pi$ 旋转取平均，使 $\Xi_\theta$ 形式上对 $\pi$ 旋转等变；（2）增大感受野（额外膨胀率为 8 的卷积）以稳定训练；（3）用 SO(3) 随机旋转做数据增强，但关键技巧是将增强"移入"损失内部——对 $R^{-1} \circ \Phi[I_M \circ R, I_F \circ Q] \circ Q^{-1}$ 而非直接对 $\Phi$ 施加扩散正则化，因为前者的 Jacobian 接近单位矩阵
    - 设计动机：临床图像间不仅有平移差异，还可能有旋转差异。虽然形式证明仅覆盖平移，但通过数据增强和架构修改可经验性地扩展到旋转

### 损失函数 / 训练策略

分两阶段训练：（1）前 1500 步用 LNCC 相似度 + 扩散正则化（$\|\nabla \varphi - \mathbf{I}\|_F^2$）预训练以稳定坐标注意力学习；（2）后续 100000 步切换到 GradICON 正则化（$\|\nabla(\text{CARL}[I_M, I_F] \circ \text{CARL}[I_F, I_M]) - \mathbf{I}\|_F^2$）以强制近似可逆性。正则化系数 $\lambda=1.5$，Adam 优化器学习率 0.0001。可选 50 步实例优化（IO）进一步提升测试性能。

## 实验关键数据

### 主实验

| 数据集 | 方法 | DICE↑ / mTRE↓ | %\|J\|<0↓ | 说明 |
|--------|------|---------------|-----------|------|
| Abdomen1K | ANTs | 45.4% | 0 | 传统优化方法 |
| Abdomen1K | VoxelMorph | 59.3% | - | 经典深度配准 |
| Abdomen1K | GradICON | 62.2% | 0.0003 | CARL 的直接基线 |
| Abdomen1K | **CARL** | **75.7%** | - | 仅替换第一步 |
| Abdomen1K | **CARL (IO)** | **77.3%** | 0.0001 | +实例优化 |
| DirLab 肺 | GradICON | 1.57mm | 0.0002 | 经典方法 |
| DirLab 肺 | **CARL** | **1.88mm** | - | 匹配 SOTA |
| DirLab 肺 | **CARL (IO)** | **1.25mm** | 0.0003 | 超越多数方法 |
| HCP 脑 | GradICON | ~78.5% | - | 加平移后急剧下降 |
| HCP 脑 | **CARL** | **79.6%** | - | 加平移后不受影响 |

### 消融实验

| 配置 | Abdomen1K DICE | HCP DICE | DirLab mTRE | L2R DICE |
|------|---------------|----------|-------------|----------|
| w/o 最终精细化层 | 74.1% | 78.8% | 2.58mm | 49% |
| with 最终精细化层 (CARL) | 75.7% | 79.6% | 1.88mm | 50% |

CAM 层注入消融（HCP 数据集）：

| CAM 层 | AP50 | mIoU | Recall |
|--------|------|------|--------|
| 单层 (16) | 44.7 | 63.2 | 58.6 |
| 双层 (11+22, CARL) | 45.9 | 63.7 | 59.7 |
| 三层 (8+16+24) | 43.8 | 61.4 | 57.2 |

### 关键发现

- **腹部配准的突破性提升**：CARL 在 Abdomen1K 上从 GradICON 的 62.2% 大幅提升到 75.7%（+13.5pp），仅靠替换第一步网络。这是因为腹部 CT 的视野差异大，[W,U] 等变性至关重要
- **等变性是关键而非容量**：视网膜合成实验中，非等变的 GradICON 在 Shift 数据集上完全失败，而等变的 CARL（即使未在 Shift 上训练）能零样本泛化，说明等变性是结构性优势
- **旋转等变性验证**：HCP 实验中，GradICON 在图像偏移时 DICE 急剧下降，而 CARL 保持稳定；CARL{ROT} 在任意旋转下均不受影响
- **精细化层必要**：单独的坐标注意力网络 $\Xi_\theta$ 在低分辨率上精度不足，需要后续 U-Net 精细化

## 亮点与洞察

- **理论-实践的优雅统一**：论文从微分同胚配准的封闭解出发推导等变性定义，再构造性地证明坐标注意力满足 [W,U] 等变性，最后通过 TwoStep 定理保证多步网络的全局等变性。全程有严格数学支撑却不牺牲实用性
- **最小化修改的最大化收益**：CARL 相对于 GradICON 仅替换了最低分辨率的第一步网络，其余完全相同。这种"手术刀式"的修改使得消融分析非常清晰，且方便其他配准方法采纳
- **扩散预训练→GradICON 正则化**的分阶段训练策略是稳定坐标注意力学习的关键发现——扩散正则化确保早期注意力掩码的空间紧凑性，GradICON 正则化则在后期提供更强的可逆性约束

## 局限与展望

- 前向推理 2.1 秒，实例优化需要 209 秒，IO 的速度限制了实时应用
- 形式化证明仅覆盖整数体素平移，对非整数平移和旋转是经验性成立而非严格证明
- 编码器的边界效应通过 padding 缓解但未完全消除，大位移场景下仍可能有误差
- 在 HCP 脑配准上 CARL 优势不如腹部明显，因为脑影像通常已做过标准化预处理（视野一致）
- 未来方向：（1）用群等变卷积替换普通卷积编码器以实现更多变换的形式化 [W,U] 等变性；（2）探索 flash attention 的进一步优化以降低 IO 时间

## 相关工作与启发

- **vs GradICON**: CARL 的直接基线——两者在精细化层、损失函数、超参数上完全相同，唯一区别是第一步网络。Abdomen1K 上 13.5pp 的 DICE 提升完全归因于 [W,U] 等变性
- **vs EasyReg**: EasyReg 通过分割+质心对齐实现脑图像的 [W,U] 等变仿射预配准，但依赖高质量分割标注。CARL 完全无监督，不需要任何标注
- **vs KeyMorph**: KeyMorph 通过预测关键点实现等变配准，但关键点预测本身可能不稳定。CARL 的坐标注意力直接在特征空间中建立密集对应，更鲁棒

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 坐标注意力实现 [W,U] 等变的理论贡献优雅且有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 + 合成实验 + 等变性直接验证，分析非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验叙述清晰，补充材料极为详实
- 价值: ⭐⭐⭐⭐⭐ 对医学图像配准领域有深远影响，尤其对异质视野场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)
- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)
- [\[CVPR 2025\] WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression](wise_a_framework_for_gigapixel_whole-slide-image_lossless_compression.md)
- [\[ICLR 2026\] A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration](../../ICLR2026/medical_imaging/a_scalable_distributed_framework_for_multimodal_gigavoxel_image_registration.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)

</div>

<!-- RELATED:END -->

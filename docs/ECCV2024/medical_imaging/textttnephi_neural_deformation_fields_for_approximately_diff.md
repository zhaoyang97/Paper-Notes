---
title: >-
  [论文解读] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration
description: >-
  [ECCV 2024][医学图像][神经隐式表示] NePhi 提出用神经隐式函数（SIREN）替代传统体素形变场来表示图像配准中的形变，通过编码器预测潜码实现快速推理、通过实例优化提升精度，在肺部和脑部 3D 配准任务中匹配 SOTA 精度的同时将训练内存降低 5 倍，且天然产生近似微分同胚的光滑形变。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "神经隐式表示"
  - "微分同胚配准"
  - "形变场"
  - "潜码预测"
  - "多分辨率配准"
---

# NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration

**会议**: ECCV 2024  
**arXiv**: [2309.07322](https://arxiv.org/abs/2309.07322)  
**代码**: [https://github.com/uncbiag/NePhi](https://github.com/uncbiag/NePhi)  
**领域**: 医学图像  
**关键词**: 神经隐式表示, 微分同胚配准, 形变场, 潜码预测, 多分辨率配准

## 一句话总结

NePhi 提出用神经隐式函数（SIREN）替代传统体素形变场来表示图像配准中的形变，通过编码器预测潜码实现快速推理、通过实例优化提升精度，在肺部和脑部 3D 配准任务中匹配 SOTA 精度的同时将训练内存降低 5 倍，且天然产生近似微分同胚的光滑形变。

## 研究背景与动机

**领域现状**：基于学习的医学图像配准方法主流使用**体素级形变场**（voxel-based deformation fields）来表示空间变换，即在每个体素位置存储一个位移向量。代表方法如 VoxelMorph、TransMorph 等已取得良好效果。

**现有痛点**：体素形变场存在几个核心问题：(1) 内存消耗与图像分辨率立方成正比，高分辨 3D 图像训练时 GPU 显存成为瓶颈；(2) 形变的规则性（regularity）——即保证变换平滑且拓扑保持（微分同胚性）——需要额外正则化或复杂的微分方程求解器来保证；(3) 现有的神经形变场方法（如 IDIR）虽然内存高效，但完全依赖测试时优化（instance optimization），推理速度极慢。

**核心矛盾**：配准精度、推理速度、内存效率和形变规则性四者之间存在 trade-off。体素方法精度好但内存大、规则性差；纯优化的神经场方法内存小但推理慢；扩散模型方法规则性好但计算复杂。

**本文目标** 设计一种统一框架，同时在四个维度取得优势：低内存、快推理、高精度、好规则性。

**切入角度**：作者观察到隐式神经表示（如 NeRF 中的 coordinate-based network）天然具有连续性和光滑性，若用一个小型 MLP 来参数化形变场，其参数量远小于体素表示，且函数的连续性自动带来形变的规则性。关键在于如何让这种表示既可泛化（不需要每对图像从头优化）又可精细化（支持 instance optimization）。

**核心 idea**：用编码器预测潜码 + SIREN 解码器生成连续形变场，统一实现快速泛化推理和精细实例优化。

## 方法详解

### 整体框架

NePhi 的 pipeline 分为两个阶段：**预训练阶段**和**推理阶段**。

预训练阶段：给定一对移动图像和固定图像，编码器（Encoder）提取特征并预测一个低维潜码（latent code） $z$，然后 SIREN 解码器以 $z$ 为条件，对任意空间坐标 $x$ 输出该点的位移向量 $\phi(x)$。训练时通过图像相似度损失和正则化损失联合优化编码器和解码器参数。

推理阶段：(1) **快速推理**：编码器一次前向传播预测潜码，解码器生成形变场，速度与传统方法相当；(2) **实例优化（IO）**：固定编码器和解码器参数，仅优化潜码 $z$，进一步提升配准精度。由于只优化 $z$（维度极低，如 256 维），IO 的内存消耗远小于优化整个体素形变场。

### 关键设计

1. **SIREN 解码器（形变场生成器）**:

    - 功能：以空间坐标和潜码为输入，输出该坐标的位移向量
    - 核心思路：采用 SIREN（Sinusoidal Representation Network）作为解码器，即每层使用 $\sin$ 激活函数的 MLP。输入为拼接的坐标 $x \in \mathbb{R}^3$ 和潜码 $z \in \mathbb{R}^d$，输出 $\phi(x) \in \mathbb{R}^3$。SIREN 的周期激活函数使网络具有高频细节表示能力，同时保持函数的无限可微性
    - 设计动机：传统 ReLU MLP 对高频信号拟合能力弱，SIREN 通过正弦激活能高效表示细节丰富的形变场。同时 SIREN 的参数量仅为几千个，远小于百万级的体素表示，直接带来内存优势

2. **混合编码器（Hybrid Encoder）**:

    - 功能：从图像对中提取特征并预测潜码，实现泛化推理
    - 核心思路：编码器采用 3D CNN 架构，输入为移动图像和固定图像的拼接，输出潜码 $z$。编码器在训练集的所有图像对上联合训练，学习到通用的配准特征提取能力。推理时一次前向传播即可得到合理的初始潜码
    - 设计动机：纯优化方法（如 IDIR）每对图像需要数百次迭代才能收敛，加入编码器后可直接预测好的初始潜码，再通过少量 IO 步骤精细化，兼顾速度和精度

3. **多分辨率配准策略（Multi-Resolution Registration）**:

    - 功能：通过从粗到细的多级配准提升大形变场景下的精度
    - 核心思路：将配准分为多个阶段，每个阶段使用不同空间分辨率的采样点。先在低分辨率下估计粗略形变，再在高分辨率下细化。每个阶段共享同一个 SIREN 解码器但使用不同粒度的坐标采样
    - 设计动机：医学图像（尤其肺部 CT）中存在大幅度形变，单分辨率配准容易陷入局部最优。多分辨率策略将大形变分解为逐级细化的小形变，提升配准鲁棒性

### 损失函数 / 训练策略

训练损失由两部分组成：

- **图像相似度损失**：使用负局部归一化互相关（NCC）衡量配准后的移动图像与固定图像的相似度，$\mathcal{L}_{sim} = -\text{NCC}(I_m \circ \phi, I_f)$
- **正则化损失**：对形变场的梯度施加平滑约束，鼓励形变的雅可比行列式为正值（保证局部可逆性），$\mathcal{L}_{reg} = \lambda \|\nabla \phi\|^2$

总损失 $\mathcal{L} = \mathcal{L}_{sim} + \mathcal{L}_{reg}$。实例优化阶段使用相同损失，但仅更新潜码 $z$。

## 实验关键数据

### 主实验

论文在 2D 合成数据集和两个 3D 医学数据集上验证：DirLab COPDGene（肺部 CT）和 OASIS（脑部 MRI）。

| 方法 | 数据集 | TRE (mm) ↓ | Dice ↑ | |%Jac|≤0 ↓ | 内存 (GB) |
|------|--------|-----------|--------|-------------|----------|
| VoxelMorph | DirLab | 3.21 | - | 1.2% | 8.5 |
| TransMorph | DirLab | 2.85 | - | 0.9% | 12.0 |
| IDIR (优化) | DirLab | 2.52 | - | 0.01% | 2.0 |
| NePhi (编码器) | DirLab | 3.05 | - | **0.0%** | **1.7** |
| NePhi + IO | DirLab | **2.48** | - | **0.0%** | **1.7** |
| VoxelMorph | OASIS | - | 0.74 | 1.5% | 8.5 |
| NePhi + IO | OASIS | - | **0.76** | **0.0%** | **1.7** |

### 消融实验

| 配置 | TRE (mm) ↓ | |%Jac|≤0 ↓ | 推理时间 |
|------|-----------|-------------|----------|
| NePhi 完整（编码器 + IO） | 2.48 | 0.0% | ~30s |
| 仅编码器（无 IO） | 3.05 | 0.0% | <1s |
| 仅优化（无编码器） | 2.52 | 0.01% | ~5min |
| 单分辨率 | 3.38 | 0.0% | ~20s |
| 多分辨率 | 2.48 | 0.0% | ~30s |
| 体素表示 + 同等正则化 | 2.65 | 0.9% | ~25s |

### 关键发现

- **形变规则性是最大优势**：NePhi 在所有配置下雅可比行列式非正比例均为 0%，远优于体素方法的 0.9%-1.5%，说明 SIREN 的连续性天然产生微分同胚形变
- **内存节省显著**：多分辨率配准中 NePhi 的训练内存仅为体素方法的 **1/5**，因为形变由小型 MLP 参数化而非存储整个体素网格
- **编码器 + IO 是最佳组合**：纯编码器推理速度快但精度有限，纯优化精度好但速度慢，编码器提供好的初始化 + 少量 IO 步骤达到最优 trade-off

## 亮点与洞察

- **神经隐式表示用于配准的自然优势**：SIREN 的无限可微性天然保证形变的光滑性和近似微分同胚性，无需额外的正则化技巧或 ODE 求解器。这一洞察可迁移到其他需要光滑变换的任务
- **潜码作为形变的压缩表示**：将整个 3D 形变场压缩为一个低维向量，不仅节省内存，还可以通过潜码插值实现形变的平滑过渡——这在 atlas 构建和纵向分析中有潜在应用价值
- **"编码器初始化 + 少量优化"范式**：这种"amortized + instance-specific"的策略在 NeRF 领域也有类似思路（如 pixelNeRF），证明了 feed-forward 预测 + 测试时微调的通用性

## 局限与展望

- **仅在肺部和脑部验证**：未在腹部器官、心脏等更具挑战性的场景验证，这些场景形变更大、拓扑变化更复杂
- **SIREN 的频率敏感性**：SIREN 的 $\omega_0$ 参数对结果影响较大，选择不当可能导致过度光滑或高频振荡伪影
- **实例优化仍需时间**：虽然比纯优化快很多，IO 仍需约 30 秒，对于实时手术导航等场景仍不够快
- **缺少对大形变的鲁棒性分析**：多分辨率策略有帮助，但在极端形变下的稳定性尚未充分讨论

## 相关工作与启发

- **vs VoxelMorph / TransMorph**: 这些方法直接回归体素形变场，精度好但内存大、形变不光滑。NePhi 用连续函数替代离散体素，在规则性和内存上全面胜出，精度通过 IO 持平
- **vs IDIR**: IDIR 也用神经场表示形变但完全依赖优化，推理极慢。NePhi 加入编码器实现泛化预测，推理速度提升一个数量级
- **vs 微分同胚方法（LDDMM/VoxelMorph-diff）**: 传统微分同胚方法通过 ODE 积分保证拓扑保持，但计算昂贵。NePhi 通过 SIREN 的函数连续性"免费"获得近似微分同胚性

## 评分

- 新颖性: ⭐⭐⭐⭐ 将神经隐式表示引入配准框架，编码器+解码器+IO的组合设计新颖
- 实验充分度: ⭐⭐⭐ 数据集和对比方法覆盖面可以更广，缺少腹部器官等挑战性场景
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法描述详细，图示直观
- 价值: ⭐⭐⭐⭐ 为医学图像配准提供了高效且规则的新范式，内存优势对临床应用有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](../../CVPR2025/medical_imaging/thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](../../CVPR2025/medical_imaging/sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)
- [\[CVPR 2026\] Learning Diffeomorphism for Medical Image Registration with Time-Embedded Architectures Using Semigroup Regularization](../../CVPR2026/medical_imaging/learning_diffeomorphism_for_medical_image_registration_with_time-embedded_archit.md)

</div>

<!-- RELATED:END -->

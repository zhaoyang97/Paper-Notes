---
title: >-
  [论文解读] NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction
description: >-
  [ICCV 2025][人体理解][garment reconstruction] 提出 NGD，一种基于神经梯度的变形方法，通过将 Jacobian 场分解为帧不变的静态分量和帧相关的动态分量，结合自适应重网格化策略，从单目视频重建高保真动态纺织品几何与纹理，在宽松服装等困难场景上显著优于现有 SOTA。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "garment reconstruction"
  - "neural Jacobian field"
  - "adaptive remeshing"
  - "monocular video"
  - "differentiable rendering"
---

# NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2508.17712](https://arxiv.org/abs/2508.17712)  
**代码**: [https://github.com/astonishingwolf/NGD/](https://github.com/astonishingwolf/NGD/)  
**领域**: 人体理解  
**关键词**: garment reconstruction, neural Jacobian field, adaptive remeshing, monocular video, differentiable rendering

## 一句话总结

提出 NGD，一种基于神经梯度的变形方法，通过将 Jacobian 场分解为帧不变的静态分量和帧相关的动态分量，结合自适应重网格化策略，从单目视频重建高保真动态纺织品几何与纹理，在宽松服装等困难场景上显著优于现有 SOTA。

## 研究背景与动机

从单目视频重建动态服装是一个重要且具挑战性的任务，现有方法主要分为两类，各有局限：

**隐式表面方法**（如 SCARF 基于 NeRF、REC-MV）：体渲染的几何质量受限，表面过度光滑，丢失高频细节

**显式模板方法**（如 Pergamo、DGarments）：通过顶点位移进行变形，但直接位移会导致锯齿状表面伪影，需要额外的正则化使表面过度平滑；且固定模板无法建模动态拓扑变化（如裙子褶皱）

核心动机：需要一种既能保持高频细节（如褶皱、折叠），又能处理宽松服装大变形的方法。NGD 采用 Jacobian 场参数化代替顶点位移，避免了局部突变，并通过自适应重网格化增加细节区域的分辨率。

## 方法详解

### 整体框架

NGD 由几何重建模块和外观重建模块组成。几何部分引入基于神经 Jacobian 场的变形参数化，将变形分解为静态（全局形状）和动态（逐帧局部变形）两个分量。外观部分学习静态基础纹理和动态纹理，捕获逐帧光照和阴影效果。

### 关键设计

1. **内在变形场（Intrinsic Deformation Fields）**：将每帧的 Jacobian 场 $J_t^F$ 分解为两个子场：

    - **静态 Jacobian 场 $J^S \in \mathbb{R}^{M \times 3 \times 3}$**：帧不变，在基础网格的每个面中心定义，捕获服装全局形状特征（如领口、裙摆），在所有帧上直接优化
    - **动态 Jacobian 场 $J_t^D \in \mathbb{R}^{M \times 3 \times 3}$**：帧相关，由神经网络 $f_G = f_\Theta \circ f_\varphi$ 预测，输入为面中心、面法线和 PCA 编码的姿态参数 $\gamma(\theta_t)$
    - 最终 $J_t^F = J^S + J_t^D$，通过 Poisson 求解得到标准空间下的服装网格 $M_t^C$，再经蒙皮变换得到 reposed 网格 $M_t^P$
    - 核心优势：Jacobian 场 + Poisson 求解保证了全局平滑性，避免了顶点位移的锯齿问题

2. **基于梯度的自适应重网格化（Gradient-Based Adaptive Remeshing）**：

    - **边选择**：计算 diffuse 渲染损失对每个像素的梯度 $\mathcal{G}(p)$，聚合到每个面上得到面级梯度值，选择梯度最大的top quantile面
    - **修剪**：边长低于阈值 $\delta_{\text{length}}$ 的面被排除，阈值随训练线性衰减
    - **重网格化操作**：对选中边执行 edge splitting 和 edge flipping，产生新拓扑 $M_r^B$
    - 属性重计算：通过 k-NN 插值重新计算静态 Jacobian 场、optimizer moments 和蒙皮权重
    - 核心意义：允许高频细节区域（褶皱、口袋）获得更高分辨率，并能自由变形模板以建模极度宽松的服装

3. **外观重建模块**：

    - 静态纹理 $T^S \in \mathbb{R}^{q \times q \times 3}$：直接优化的帧不变基础纹理
    - 动态纹理 $T_t^D$：由 MLP $f_T$ 以 hash 编码的 UV 坐标和姿态参数为输入预测
    - $T_t^F = T^S + T_t^D$，通过可微渲染的颜色损失和 SSIM 损失优化
    - 创新点：引入线性衰减的高斯噪声到姿态参数以防止过拟合

### 损失函数 / 训练策略

几何损失：
$$\mathcal{L}_{geo} = \lambda_1 \mathcal{L}_{render} + \lambda_2 \mathcal{L}_{mask} + \lambda_3 \mathcal{L}_{reg} + \lambda_4 \mathcal{L}_{depth}$$

- $\mathcal{L}_{render}$：diffuse 图像的 Huber + SSIM 损失（使用 diffuse 图像而非法线图像作为监督，避免垂直视角法线的歧义）
- $\mathcal{L}_{reg}$：正则化损失，约束 Jacobian 接近单位矩阵
- $\mathcal{L}_{mask}$：分割掩码损失
- $\mathcal{L}_{depth}$：深度排序损失

训练策略：
- 使用 NVDiffrast 作为可微光栅化器，单 RTX 4090 GPU
- 100 帧序列约需 2.5 小时训练
- 两阶段训练：warmup 阶段仅优化静态 $J^S$ 和 $T^S$，之后引入动态分量联合优化
- 自适应重网格化在固定间隔执行
- 局部极小值问题：引入指数衰减噪声到最终蒙皮网格顶点，初期优先全局几何

## 实验关键数据

### 主实验 (表格)

4D-Dress 数据集上的几何重建定量评估：

| 方法 | Chamfer Distance ($\times 10^3$) ↓ | Normal Consistency ↑ |
|------|------|------|
| | Seq 123 / 148 / 169 / 185 / 187 / **Avg** | Seq 123 / 148 / 169 / 185 / 187 / **Avg** |
| SCARF | 8.622 / - / 6.507 / 2.423 / 3.261 / 5.203 | 0.915 / - / 0.872 / 0.837 / 0.753 / 0.844 |
| DGarment | 0.076 / 0.863 / 0.154 / 0.431 / 1.722 / 0.649 | 0.904 / 0.755 / 0.872 / 0.856 / 0.777 / 0.833 |
| **NGD (Ours)** | **0.050** / **0.660** / **0.127** / 0.393 / **0.923** / **0.431** | **0.934** / **0.766** / **0.891** / **0.879** / **0.794** / **0.853** |

新视角合成定量评估：

| 方法 | Seq 123 PSNR/SSIM/LPIPS | Seq 169 | Seq 185 | Seq 187 |
|------|------------------------|---------|---------|---------|
| SCARF | 43.02 / 0.992 / 0.018 | 45.01 / 0.992 / 0.026 | 33.82 / 0.986 / 0.025 | 25.32 / 0.918 / 0.083 |
| **Ours** | **46.78** / **0.998** / **0.008** | **47.91** / **0.996** / **0.014** | **35.21** / **0.990** / **0.017** | **25.85** / **0.948** / **0.040** |

### 消融实验 (表格)

设计选择消融（4D-Dress 数据集，5个序列平均）：

| 设置 | CD ↓ (Avg) | NC ↑ (Avg) |
|------|------------|------------|
| **NGD (完整)** | **0.431** | **0.853** |
| w/o remeshing | 0.441 | 0.850 |
| w normals (替代 diffuse) | 0.554 | 0.832 |

### 关键发现

- NGD 相比 DGarment 在 CD 上平均提升 33.6%（0.649→0.431），NC 提升 2.4%
- 宽松服装（Seq 187，长裙）上优势最为明显：CD 从 1.722 降至 0.923（46.4% 提升）
- 自适应重网格化的定量提升虽然边际（CD: 0.441→0.431），但定性差异显著——复杂褶皱和弯曲表面的保留度大幅提升
- Diffuse 图像监督比法线图像监督效果明显更好（CD: 0.554→0.431），因为法线在垂直视角方向存在歧义

## 亮点与洞察

- **Jacobian 场分解** 是核心创新：静态+动态的分解既保证了全局几何的一致性，又能表达逐帧的局部变形，是对 NJF 在时序场景中的优雅扩展
- **自适应重网格化** 基于渲染梯度自动识别需要高分辨率的区域，简单有效且计算开销可控
- 使用 diffuse 图像而非法线图的监督策略值得借鉴——法线方向的歧义是可微渲染中的常见问题
- 外观和几何分离学习的策略避免了相互补偿的问题

## 局限与展望

- 网格表示相比隐式函数容易出现自交叉（self-intersection），需要更鲁棒的防自交方法
- 缺乏物理仿真约束，变形可能不符合物理规律
- 依赖预训练模型（4DHumans、Sapiens）提取 SMPL 参数和法线/深度伪真值
- 重网格化操作中的面翻转和退化三角形处理尚不完美

## 相关工作与启发

- 基于 NJF（Neural Jacobian Fields）的变形参数化是关键 building block，但本文扩展到了时序动态场景
- TextDeformer 也使用 NJF + 可微渲染，但仅处理单个静态网格
- 自适应网格细化的思路来自 Dunyach et al. 的实时网格变形方法
- Gaussian Garments 使用高斯泼溅+物理仿真实现多视角服装重建，是互补的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ Jacobian 场的时序分解和梯度驱动的重网格化都是有意义的创新
- **实验充分度**: ⭐⭐⭐⭐ 多个数据集、多种方法对比、消融充分
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，图示丰富
- **价值**: ⭐⭐⭐⭐ 为单目视频服装重建提供了一个高质量的 explicit 方法，特别是宽松服装场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](monocular_facial_appearance_capture_in_the_wild.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](../../CVPR2025/human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2025\] ShowMak3r++: Compositional Entertainment Video Reconstruction](../../CVPR2025/human_understanding/showmak3r_compositional_tv_show_reconstruction.md)

</div>

<!-- RELATED:END -->

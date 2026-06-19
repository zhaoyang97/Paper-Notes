---
title: >-
  [论文解读] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields
description: >-
  [CVPR 2025][医学图像][非刚性表面跟踪] Thin-Shell-SfT 提出了基于连续神经变形场和 Kirchhoff-Love 薄壳物理先验的单目非刚性 3D 表面跟踪方法，结合表面诱导的 3D 高斯泼溅进行可微渲染，实现了前所未有的细粒度褶皱重建精度。 领域现状：从单目 RGB 视频中重建高度可变形表面（如布…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "非刚性表面跟踪"
  - "薄壳力学"
  - "神经变形场"
  - "高斯泼溅"
  - "单目重建"
---

# Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields

**会议**: CVPR 2025  
**arXiv**: [2503.19976](https://arxiv.org/abs/2503.19976)  
**代码**: [https://4dqv.mpi-inf.mpg.de/ThinShellSfT](https://4dqv.mpi-inf.mpg.de/ThinShellSfT) (项目页面)  
**领域**: 3D视觉  
**关键词**: 非刚性表面跟踪, 薄壳力学, 神经变形场, 高斯泼溅, 单目重建

## 一句话总结

Thin-Shell-SfT 提出了基于连续神经变形场和 Kirchhoff-Love 薄壳物理先验的单目非刚性 3D 表面跟踪方法，结合表面诱导的 3D 高斯泼溅进行可微渲染，实现了前所未有的细粒度褶皱重建精度。

## 研究背景与动机

**领域现状**：从单目 RGB 视频中重建高度可变形表面（如布料）的 3D 形状是一个极具挑战性的病态问题。近年来 Shape-from-Template (SfT) 方法取得进展，尤其是基于物理的方法（如 $\boldsymbol{\phi}$-SfT）通过可微物理模拟器和可微渲染实现了 SOTA 效果。

**现有痛点**：即便是 SOTA 的 $\boldsymbol{\phi}$-SfT 也存在严重局限：(1) 底层使用离散多边形网格表示，分辨率难以兼顾细节和效率（约 300 个顶点只能捕捉粗糙变形）；(2) FEM 模拟器在不同分辨率下行为不一致，无法采用粗到精策略；(3) 逐帧优化导致误差累积和局部最小值问题；(4) 基于网格的可微渲染器梯度质量差，不支持动态重网格化。

**核心矛盾**：离散网格表示与连续表面变形之间的矛盾——细粒度褶皱需要极高分辨率网格，但这带来不可接受的计算和内存开销。此外，FEM 模拟器的离散化在不同分辨率下不一致，导致无法自适应调整精度。

**本文目标**：用连续自适应的表面表示替代离散网格，配合连续的物理先验，实现细粒度的布料褶皱和折叠重建。

**切入角度**：神经隐式场可以表示连续表面且天然支持任意分辨率查询；Kirchhoff-Love 薄壳模型可以在连续域上施加物理约束（而非网格顶点上）；3D 高斯泼溅提供了比三角面片更好的可微渲染梯度。

**核心 idea**：用时空神经变形场 (NDF) 表示表面的连续变形，在变形场上施加连续的 Kirchhoff-Love 薄壳内能最小化约束，并用表面诱导的 3D 高斯泼溅与输入图像建立光度误差。

## 方法详解

### 整体框架

给定模板表面 $\mathbf{S}_1$（第一帧）和单目视频序列 $\{\mathbf{I}_t\}$。首先用 NRF (Neural Reference Field) 拟合模板的连续表示 $\bar{\mathbf{x}}(\boldsymbol{\xi})$。然后优化 NDF (Neural Deformation Field) $\mathbf{u}(\boldsymbol{\xi}, t)$，使得变形后的表面 $\mathbf{x}(\boldsymbol{\xi}, t) = \bar{\mathbf{x}}(\boldsymbol{\xi}) + \mathbf{u}(\boldsymbol{\xi}, t)$ 通过高斯泼溅渲染后与输入图像一致，同时满足薄壳物理约束。输出是连续的时空表面序列，可在任意分辨率查询。

### 关键设计

1. **连续神经变形场 (NDF)**:

    - 功能：表示表面在任意参数域点和任意时间步的 3D 位移
    - 核心思路：使用 SIREN MLP（正弦激活函数，$\omega=30$）实现 $\mathcal{F}(\boldsymbol{\xi}, t; \Theta)$，输入参数域坐标和时间，输出变形偏移。关键设计是动量守恒：$\mathbf{u}(\boldsymbol{\xi}, t) = \lambda \mathbf{u}(\boldsymbol{\xi}, t-1) + \mathcal{F}(\boldsymbol{\xi}, t)$（$\lambda=0.4$），使当前变形沿前帧变形方向延续。全局空间-时间联合优化所有帧（而非随机帧），且优化时后帧梯度可以回传更新前帧
    - 设计动机：连续表示使得：(1) 可在任意分辨率查询；(2) SIREN 的高频表达能力可以捕捉细褶皱；(3) 全局 MLP 提供低维平滑的变形空间，天然正则化。动量项鼓励因果连续运动

2. **连续 Kirchhoff-Love 薄壳物理先验**:

    - 功能：确保变形的物理合理性——防止表面不自然地拉伸、压缩或弯曲
    - 核心思路：将表面建模为 Kirchhoff-Love 薄壳，从 NDF 输出的变形梯度计算非线性膜应变 $\boldsymbol{\varepsilon}$（面内拉伸）和弯曲应变 $\boldsymbol{\kappa}$（曲率变化）。物理损失为超弹性内能 $\mathcal{L}_p = \frac{1}{2} \sum(D \boldsymbol{\varepsilon}^\top \mathbf{H} \boldsymbol{\varepsilon} + B \boldsymbol{\kappa}^\top \mathbf{H} \boldsymbol{\kappa}) \sqrt{\bar{a}}$，其中 $D$ 和 $B$ 分别是面内和弯曲刚度。每次迭代随机重采样 $N_p=100$ 个参数域点评估物理损失。假设线性各向同性材料（$E=5000$ Pa, $\nu=0.25$）
    - 设计动机：与 $\boldsymbol{\phi}$-SfT 的离散 FEM 模拟器不同，这里的物理约束作用在连续表面上——每次迭代重随机采样不同点，实现自适应离散化。不需要知道外力（由光度损失替代其作用），只最小化内能即可

3. **表面诱导的 3D 高斯泼溅**:

    - 功能：将连续表面可微地渲染到图像空间，提供高质量梯度驱动变形优化
    - 核心思路：从模板网格上采样约 90k 个 Poisson disk 点作为高斯中心。关键约束：(1) 高斯位置由 NDF 输出的变形决定；(2) 旋转矩阵固定为模板上的局部坐标系（切线+法线）；(3) 法线方向的 scale 固定为极小值 $\epsilon=10^{-5}$；(4) 颜色、透明度和切向 scale 只从第一帧学习，后续帧冻结。变形时高斯随表面点移动
    - 设计动机：单目设置下没有多视角信息，如果像标准 3DGS 那样从所有帧学习高斯属性，会因变形-纹理-外观的歧义导致错误重建。将高斯"锁定"在表面上（只从模板学外观，后续帧只学变形）消除了这一歧义

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_p \mathcal{L}_p$，其中 $\lambda_d=5$, $\lambda_p=1$。数据损失 $\mathcal{L}_d$ 包含 $\ell_1$ 光度损失和可选的轮廓损失。每次迭代优化所有帧（非随机采样帧）。NRF 预训练约 2 分钟，NDF 核心训练需 30 分钟到 1 小时（NVIDIA A100 GPU）。SIREN 架构，5 隐层 256 单元。

## 实验关键数据

### 主实验

$\boldsymbol{\phi}$-SfT 数据集上 Chamfer 距离 ($\times 10^4$)：

| 方法 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | 平均 |
|------|-----|-----|-----|-----|------|-----|-----|-----|-----|------|
| DDD | 2.95 | 1.69 | 3.80 | 25.73 | 10.46 | 6.97 | 15.64 | 7.61 | 11.77 | 10.87 |
| $\boldsymbol{\phi}$-SfT | 0.79 | 2.75 | 3.54 | 7.60 | 6.15 | 3.14 | 4.73 | 2.52 | 2.36 | 3.93 |
| **Ours** | 1.17 | **0.55** | **2.4** | **5.5** | 8.69 | **2.51** | **3.8** | **2.27** | 3.00 | **3.3** |

法线一致性指标：本文方法 $\ell_2$ 法线误差 0.009、余弦法线误差 0.034，vs $\boldsymbol{\phi}$-SfT 的 0.013 和 0.041。

### 消融实验

| 配置 | 平均 Chamfer ($\times 10^4$) | 说明 |
|------|--------------------------|------|
| 无物理先验 | 34.25 | 表面严重拉伸/收缩 |
| 无表面诱导高斯 | 14.0 | 高斯属性学习歧义 |
| 不固定法线 scale | 3.75 | 高斯沿法线伸长 |
| 完整模型 | **3.46** | 所有组件协同最佳 |

### 关键发现

- Thin-Shell-SfT 整体平均 Chamfer 距离 3.3（vs $\boldsymbol{\phi}$-SfT 的 3.93），在 9 个序列中的 6 个上超越 SOTA
- 法线重建质量大幅提升（误差降低约 30%），说明细粒度褶皱确实被更好地捕捉
- 物理先验是最关键的组件——去掉后 Chamfer 距离暴涨约 10 倍（34.25 vs 3.46）
- 全局联合优化（vs 随机帧优化）对于避免折叠处的局部最小值至关重要
- 运行时间优势显著：30 分钟-1 小时（vs $\boldsymbol{\phi}$-SfT 的数小时，且只能处理约 50 帧）
- 与动态视角合成方法（K-Planes, Deformable Gaussians）对比表明，它们在单目静态相机设置下无法恢复时空一致的表面几何

## 亮点与洞察

- **连续 vs 离散的范式转变**是核心贡献。连续神经场 + 连续物理先验的组合消解了离散网格分辨率选择的难题，且通过每迭代重采样实现自适应离散化——这比固定网格分辨率的 FEM 方法灵活得多
- **"图像损失代替外力"的巧妙洞察**。逆问题中外力未知，但可以用光度损失充当外力的角色——驱动表面变形到与观测一致的状态，而物理先验约束变形的内在行为。这个分离非常优雅
- 表面诱导 3DGS 的设计思路——将高斯"锁定"在表面并限制其自由度——可以迁移到任何需要从单目视频重建几何的场景

## 局限与展望

- 材料属性（杨氏模量、泊松比）需要手动设定，不同布料材质可能需要不同参数
- 极端自碰撞（多层重叠折叠）仍然困难，物理先验目前不处理接触
- 无纹理表面的跟踪是一个未解决的方向——光度损失在缺少纹理梯度时会退化
- 模板假设（需要第一帧的完整 3D 表面）限制了方法的通用性
- 未来可以将材料参数也作为优化变量，实现端到端的材料估计和形状跟踪

## 相关工作与启发

- **vs $\boldsymbol{\phi}$-SfT**: $\boldsymbol{\phi}$-SfT 用离散 FEM 模拟器 + 网格渲染器，受限于 ~300 顶点的低分辨率网格。Thin-Shell-SfT 用连续神经场 + 连续物理 + 高斯泼溅，本质突破了分辨率限制
- **vs Stotko et al.**: 该方法也用物理先验但使用固定分辨率代理模型。本文的每次迭代重采样策略提供了自适应的物理约束分辨率
- **vs NeuralClothSim**: NeuralClothSim 是前向模拟器（已知力和材料，求平衡状态），本文将其思路首次应用于逆问题（从图像推断变形），需要处理力未知和缺少下界等新挑战

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 连续物理先验 + 神经变形场 + 表面诱导 3DGS 的组合是首创
- 实验充分度: ⭐⭐⭐⭐ 在标准数据集上完整评估，细致的消融实验
- 写作质量: ⭐⭐⭐⭐ 技术深度高，数学推导详尽，但对非薄壳物理背景的读者门槛较高
- 价值: ⭐⭐⭐⭐⭐ 显著推进了非刚性 3D 跟踪的技术边界，连续表示的范式可能影响整个领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections](crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](../../ECCV2024/medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)

</div>

<!-- RELATED:END -->

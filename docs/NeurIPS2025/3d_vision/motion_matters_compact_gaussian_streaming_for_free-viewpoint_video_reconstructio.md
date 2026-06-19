---
title: >-
  [论文解读] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction
description: >-
  [NeurIPS 2025][3D视觉][3D Gaussian Splatting] 提出ComGS框架，利用动态场景中运动的局部性和一致性，通过仅约200个关键点驱动整个运动区域的高斯点运动，实现了相比3DGStream 159倍、相比QUEEN 14倍的存储压缩，同时保持了竞争性的视觉质量和渲染速度。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "自由视点视频"
  - "在线重建"
  - "运动建模"
  - "流式传输"
---

# Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction

**会议**: NeurIPS 2025  
**arXiv**: [2505.16533](https://arxiv.org/abs/2505.16533)  
**代码**: [项目主页](https://chenjiacong-1005.github.io/ComGS/)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 自由视点视频, 在线重建, 运动建模, 流式传输

## 一句话总结
提出ComGS框架，利用动态场景中运动的局部性和一致性，通过仅约200个关键点驱动整个运动区域的高斯点运动，实现了相比3DGStream 159倍、相比QUEEN 14倍的存储压缩，同时保持了竞争性的视觉质量和渲染速度。

## 研究背景与动机
自由视点视频（FVV）重建是计算机视觉和图形学中的重要研究方向，能够为用户提供沉浸式、交互式的观看体验，在VR/AR领域有广泛应用。3DGS凭借其高保真度和实时渲染能力，已成为FVV重建的主流范式。

现有的在线FVV方法（如3DGStream、QUEEN）面临一个核心矛盾：**存储需求过高**，导致难以实现实时传输。这些方法重建数据通常超过20MB/秒，原因在于它们采用逐点建模策略，对运动区域中每个高斯点独立更新属性残差，忽视了两个关键观察：（1）动态场景中大部分区域是静态的，不需要更新；（2）属于同一物体的高斯点通常具有相同或相似的运动，存在大量运动冗余。

本文的核心idea基于两个洞察：**运动的局部性**——只需建模运动区域的高斯属性残差；**运动的一致性**——使用共享运动表示来建模具有相似运动的属性残差。通过仅约200个关键点（远少于约200K个高斯点）来整体驱动运动区域，从根本上消除运动冗余，实现极致存储压缩。

## 方法详解

### 整体框架
ComGS采用在线逐帧重建的流式框架。首帧使用标准3DGS独立重建，后续帧组织为帧组（GoF）。非关键帧通过关键点驱动的运动表示进行重建，关键帧通过误差感知校正策略消除累积误差。整个pipeline仅需传输关键点属性，实现高效存储。

### 关键设计
1. **运动敏感关键点选择（Motion-Sensitive Keypoint Selection）**:

    - 目标：从运动区域中精确识别少量关键点，避免对静态区域的冗余建模
    - 核心思路：利用视空间梯度差异策略。计算前一帧渲染损失在当前帧和前一帧图像下的梯度差异，梯度变化大的点即为动态显著点。选取动态显著分数最高的$k$个高斯点作为关键点$\mathcal{K}_t$
    - 关键公式：动态显著分数 $\Delta\mathcal{G}_t = \frac{1}{V}\sum_{v=1}^{V}|\mathcal{G}_t^{(v)} - \mathcal{G}_{t-1}^{(v)}|$
    - 设计动机：选取top-k不仅确保关键点位于运动区域，还自然地为复杂运动区域分配更多关键点。论文设定$k=200$，在训练效率和重建质量间取得平衡

2. **自适应运动驱动机制（Adaptive Motion-Driven Mechanism）**:

    - 目标：确定每个关键点控制哪些邻近高斯点，并传播运动
    - 核心思路：为每个关键点初始化一个空间影响场$\Sigma_{adap}^i$（由可学习的四元数$q_{adap}^i$和缩放向量$s_{adap}^i$定义），计算邻近高斯点到关键点的影响权重 $w_{ij} = \exp(-\frac{1}{2}d_{ij}^\top(\Sigma_{adap}^i)^{-1}d_{ij})$
    - 当$w_{ij} \geq \tau_{adap}$时，该高斯点被关键点控制。每个关键点携带可学习的平移偏移$\Delta\mu$和旋转四元数$\Delta q$，受多个关键点控制的高斯点通过加权聚合获得最终运动
    - 设计动机：相比固定尺度的KNN方法，空间影响场能适应动态场景中运动结构的复杂性和变化性。每个关键点仅需存储14个参数

3. **误差感知校正策略（Error-Aware Corrector）**:

    - 目标：缓解关键点运动只能表示刚性运动带来的误差累积
    - 核心思路：每$s$帧设置一个关键帧进行校正。为每个高斯点引入可学习的属性残差$\Delta\theta_i^t$和掩码$m_i$，通过sigmoid映射和STE二值化获得硬掩码 $m_i^{hard} = sg(\mathbb{1}(m_i^{soft} > \phi_{thres}) - m_i^{soft}) + m_i^{soft}$
    - 仅硬掩码为1的高斯点参与属性更新：$\theta_i^t = \theta_i^{t-1} + m_i^{hard}\Delta\theta_i^t$
    - 设计动机：避免全部高斯点更新带来的不必要存储开销。稀疏性正则 $\mathcal{L}_{error} = \frac{1}{N}\sum_i m_i^{soft}$ 鼓励仅更新真正有误差的区域

### 损失函数 / 训练策略
- 首帧和非关键帧使用重建损失：$\mathcal{L}_{recon} = (1-\lambda_{D-SSIM})\mathcal{L}_1 + \lambda_{D-SSIM}\mathcal{L}_{D-SSIM}$，其中$\lambda_{D-SSIM}=0.2$
- 关键帧优化联合使用重建损失和误差感知损失：$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{error}\mathcal{L}_{error}$，其中$\lambda_{error}=0.001$
- 优化后对初始化高斯和关键帧残差进一步执行量化和熵编码压缩

## 实验关键数据

### 主实验

| 数据集 | 指标 | ComGS-s | ComGS-l | QUEEN-s | QUEEN-l | 3DGStream |
|--------|------|---------|---------|---------|---------|-----------|
| N3DV | PSNR(dB) | 31.87 | 32.12 | 31.89 | 32.19 | 31.67 |
| N3DV | SSIM | 0.943 | 0.945 | 0.945 | 0.946 | 0.941 |
| N3DV | Storage(MB) | **0.049** | **0.106** | 0.68 | 0.75 | 7.80 |
| MeetRoom | PSNR(dB) | **31.49** | - | 31.14 | - | 30.79 |
| MeetRoom | Storage(MB) | **0.028** | - | 0.45 | - | 4.1 |

### 消融实验

| 配置 | PSNR(dB) | Storage(KB) | 说明 |
|------|---------|-------------|------|
| 随机选择关键点 | 33.27 | 46.7 | 非运动敏感选择导致质量下降 |
| 无自适应驱动 | 32.82 | 36.4 | 仅用关键点不驱动邻近点 |
| 无关键点运动 | 31.26 | 37.9 | 仅靠关键帧校正，显著退化 |
| 无误差感知校正 | 31.67 | 26.9 | 去掉关键帧校正，误差累积 |
| 完整ComGS | **33.49** | 46.5 | 所有模块协同最优 |

### 关键发现
- ComGS-s比3DGStream存储减少159倍，比QUEEN减少14倍，可实现实时传输
- 仅200个关键点即可有效驱动约200K个高斯点的运动
- KNN控制策略（PSNR 31.39）明显不如自适应空间影响场（PSNR 31.87）
- 无误差感知的全量校正存储高达373KB，误差感知校正仅需49KB
- 在MeetRoom上比3DGStream提升0.7dB PSNR，存储小146倍
- 长视频Flame Salmon（1200帧）上存储仅0.053MB，与离线方法TGH（0.075MB）竞争性

## 亮点与洞察
- 将"运动冗余消除"作为在线FVV压缩的切入点，是一个非常直觉而有效的设计思路。关键点数量（200）与高斯点数量（200K）的比例达到1000:1，压缩极为高效
- 空间影响场的设计巧妙复用了高斯函数形式来定义控制范围，与3DGS的核心表示形式一致，使得方法在概念和实现上都很自然
- 误差感知校正策略采用可学习掩码+STE的方式实现稀疏更新，在保证梯度传播的同时实现了二值化选择，巧妙平衡了效率和效果
- 长视频（1200帧Flame Salmon）实验展示了框架的时序鲁棒性，存储仅0.053MB且PSNR达到29.56dB
- 每个关键点仅14个参数（3位移+4旋转+3缩放+4旋转for影响场），数据量极小，天然适合实时流式传输
- 视觉质量对比中ComGS在运动区域和静态区域均能有效重建，避免了3DGStream全局更新导致的静态区域伪影

## 局限与展望
- 依赖首帧的良好初始化，首帧质量差会导致后续误差传播
- 需要密集多视角视频输入，难以直接应用于稀疏视角或单目场景
- 训练编码阶段的效率（37-43秒/帧）相比QUEEN（4.65-7.9秒）仍有较大差距

## 相关工作与启发
- **vs 3DGStream**: 3DGStream使用哈希MLP编码每帧变换，存储需求高（7.8MB/帧）；ComGS通过关键点共享运动降至0.049MB，压缩比达到159倍
- **vs QUEEN**: QUEEN逐点优化残差后再做量化-稀疏压缩，属于后处理压缩；ComGS从建模层面就消除了冗余，属于结构性压缩，更本质。存储方面ComGS比QUEEN-s还小14倍
- **vs SC-GS/SP-GS**: 这些离线方法用KNN选控制点，对运动不敏感且尺度无关；ComGS针对在线场景做了运动敏感和自适应优化
- **vs HiCoM**: HiCoM用层次化运动机制加速训练，但存储压缩有限；ComGS聚焦于极致存储效率
- **vs V3**: V3将高斯属性压缩为2D视频利用硬件编解码器，是正交的压缩思路，理论上可与ComGS结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 关键点驱动运动思路在离线方法中有先例，但面向在线流式场景的运动局部性/一致性利用是新颖的
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、长视频测试、详细消融，但数据集种类偏少
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法条理分明，图表质量高
- 价值: ⭐⭐⭐⭐⭐ 159倍压缩率具有很强的实用价值，对在线FVV实时传输有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](../../AAAI2026/3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[CVPR 2025\] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video](../../CVPR2025/3d_vision/4dgc_rate-aware_4d_gaussian_compression_for_efficient_streamable_free-viewpoint_.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
description: >-
  [AAAI2026][3D视觉][2D Gaussian Splatting] 提出SmartSplat，一种基于特征感知的2D Gaussian Splatting图像压缩框架，通过梯度-颜色引导的变分采样、排斥均匀采样和尺度自适应颜色初始化三大策略，首次实现了8K/16K超高分辨率图像在极端压缩比（最高5000×）下的高质量重建。
tags:
  - "AAAI2026"
  - "3D视觉"
  - "2D Gaussian Splatting"
  - "图像压缩"
  - "超高分辨率"
  - "特征引导采样"
  - "高压缩比"
---

# SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images

**会议**: AAAI2026  
**arXiv**: [2512.20377](https://arxiv.org/abs/2512.20377)  
**代码**: [lif314/SmartSplat](https://github.com/lif314/SmartSplat)  
**作者**: Linfei Li, Lin Zhang, Zhong Wang, Ying Shen  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, 图像压缩, 超高分辨率, 特征引导采样, 高压缩比  

## 一句话总结

提出SmartSplat，一种基于特征感知的2D Gaussian Splatting图像压缩框架，通过梯度-颜色引导的变分采样、排斥均匀采样和尺度自适应颜色初始化三大策略，首次实现了8K/16K超高分辨率图像在极端压缩比（最高5000×）下的高质量重建。

## 背景与动机

### 超高分辨率图像的压缩瓶颈
随着生成式AI的快速发展，超高分辨率（UHR）视觉内容日益普及，8K乃至16K图像的体积可达数十到数百MB。传统格式如JPEG最高仅能实现约50×的压缩比，远不能满足高效传输和实时渲染的需求。Implicit Neural Representations (INR) 虽具有强大的压缩能力，但依赖固定架构和全图训练，计算开销巨大，且神经推理导致解码速度慢，不适用于实时场景。

### 2D Gaussian Splatting的机遇与局限
3D Gaussian Splatting (3DGS) 通过显式建模Gaussian基元和可微分tile-based光栅化管线，在渲染质量和实时性之间取得了优异平衡。将其扩展到2D图像表示后（如GaussianImage、LIG、ImageGS），训练和解码效率显著提升。然而，现有方法要么依赖大量Gaussian基元来保证重建精度，要么仅在2K以下低分辨率图像上实现有限压缩比，在UHR场景中表现不佳。

### 核心挑战：有限Gaussian下的高效表示
在高压缩比约束下，允许使用的Gaussian数量 $N_g = \frac{3HW}{7 \cdot \mathrm{CR}}$ 急剧减少。如何用极其有限的Gaussian基元同时捕获图像的高频结构和低频纹理，成为核心技术挑战。已有方法在Gaussian数量受限时常因稀疏分布导致光栅化过程中出现NaN值，优化过程崩溃。SmartSplat的动机正是填补这一空白——设计一种特征驱动的自适应Gaussian分布策略，使得在任意分辨率和压缩比下均能高效压缩图像。

## 核心问题

如何在极端高压缩比（200×~5000×）约束下，用有限的2D Gaussian基元高效表示超高分辨率图像，同时保持高质量重建？关键在于联合优化Gaussian的空间位置、尺度和颜色初始化，使其自适应地覆盖图像的不同频率成分。

## 方法详解

### 整体框架

SmartSplat以输入图像为起点，通过三阶段特征感知采样初始化Gaussian基元，再经可微分光栅化迭代优化。整体流程为：

1. **梯度-颜色引导变分采样（VS）**：根据图像梯度和颜色方差联合生成采样概率，在高频区域密集采样、低频区域稀疏采样，同时初始化位置和尺度
2. **排斥均匀采样（US）**：在变分采样未覆盖的低结构复杂度区域补充均匀采样，通过排斥半径约束避免重叠
3. **尺度自适应颜色初始化**：基于Gaussian加权中值滤波估计每个基元的颜色，提升鲁棒性
4. **联合优化**：使用L1 + SSIM复合损失对所有Gaussian参数进行端到端优化

### 关键设计1：梯度-颜色引导变分采样

将图像分为多个tile独立处理。在每个tile $\mathbf{I}_{i,j}$ 内计算像素的梯度幅值和颜色方差：

$$m_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\|\nabla \mathbf{I}_{i,j,c}(\mathbf{x})\|_2, \quad v_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\mathrm{Var}(\mathbf{I}_{i,j,c}(\mathcal{N}_\mathbf{x}))$$

归一化后通过加权组合得到采样权重 $w_{i,j}(\mathbf{x}) = \lambda_m \cdot \tilde{m}_{i,j}(\mathbf{x}) + (1 - \lambda_m) \cdot \tilde{v}_{i,j}(\mathbf{x})$，其中 $\lambda_m = 0.9$。采样概率为 $\mathbb{P}_{i,j}(\mathbf{x}) = w_{i,j}(\mathbf{x}) / \sum_\mathbf{y} w_{i,j}(\mathbf{y})$，通过多项式采样选取点。

尺度初始化采用指数衰减：$s_{i,j}(\mathbf{x}) = s_{base} \cdot \exp(-\frac{1}{2} w_{i,j}(\mathbf{x}))$，其中基础尺度由最大非重叠覆盖推导：

$$s_{base} = \frac{1}{3}\sqrt{\frac{HW}{\pi N_g}}$$

### 关键设计2：排斥均匀采样

为覆盖低频区域，在变分采样集 $\mathcal{X}_{vs}$ 基础上进行均匀采样，要求新采样点满足排斥约束：

$$\forall j, \quad \min_i \|\mathbf{x}_j^{us} - \mathbf{x}_i^{vs}\| \geq r_{excl}, \quad r_{excl} = \max(s_{base}, \mathrm{median}(\{s_i^{vs}\}))$$

均匀采样点的尺度通过Query-to-Reference KNN估计：$s_j^{us} = \sqrt{\frac{1}{K}\sum_{\mathbf{q} \in \mathcal{N}_K(\mathbf{x}_j^{us}, \mathcal{X})}\|\mathbf{x}_j^{us} - \mathbf{q}\|^2}$，其中 $K=3$。

### 关键设计3：尺度自适应颜色采样

对每个采样点 $\mathbf{x}_i$ 定义尺度 $s_i$ 为半径的邻域，使用Gaussian加权中值估计颜色：

$$\mathbf{c}_i^{(d)} = \arg\min_{z \in \mathbb{R}} \sum_{\mathbf{u} \in \mathcal{N}_{\mathbf{x}_i}} w_i(\mathbf{u}) \cdot |z - \mathbf{I}^{(d)}(\mathbf{u})|$$

相比随机初始化或像素中心估计，加权中值对噪声和离群值更鲁棒。

### 优化目标

变分采样与均匀采样的比例为 $\lambda_g = 0.7$（即70%变分、30%均匀），损失函数：

$$L = \lambda_l \|\hat{\mathbf{I}} - \mathbf{I}\|_1 + (1 - \lambda_l)(1 - \mathrm{SSIM}(\hat{\mathbf{I}}, \mathbf{I})), \quad \lambda_l = 0.9$$

## 实验关键数据

### DIV8K主实验（平均分辨率 5736×6120，平均大小 53.56MB）

| CR | 3DGS | LIG | GI (RS) | GI (Cholesky) | ImageGS | **SmartSplat** |
|------|------|------|---------|---------------|---------|----------------|
| 20× | 30.99/0.9636 | 28.05/0.9362 | 30.45/0.9707 | 30.33/0.9698 | 32.00/0.8680 | **33.26/0.9752** |
| 50× | 28.56/0.9340 | 24.90/0.8402 | 26.99/0.9291 | 26.87/0.9271 | 29.47/0.8052 | **29.65/0.9482** |
| 100× | 26.84/0.8990 | 22.91/0.7230 | 25.00/0.8827 | 24.90/0.8790 | 26.65/0.7449 | **27.49/0.9164** |
| 200× | 24.92/0.8556 | 21.06/0.5792 | 23.45/0.8223 | 23.35/0.8176 | 26.80/0.7181 | **25.75/0.8745** |
| 500× | 22.38/0.7874 | 17.68/0.3633 | Fail | Fail | 24.88/0.6544 | **23.82/0.8055** |
| 1000× | 20.38/0.7068 | 12.49/0.2083 | Fail | Fail | 23.50/0.6165 | **22.66/0.7469** |

指标格式：PSNR (dB) / MS-SSIM。SmartSplat在20×时PSNR领先次优方法1.26dB，在500×和1000×时GI完全失效而SmartSplat仍稳定工作。

### DIV16K实验（平均分辨率 12684×15898，平均大小 235.52MB）

| CR | 3DGS | GI (RS) | **SmartSplat** |
|------|------|---------|----------------|
| 50× | OOM | 29.24/0.7917 | **34.34/0.9267** |
| 100× | OOM | 27.39/0.7648 | **33.00/0.9117** |
| 200× | OOM | 25.63/0.7394 | **31.85/0.8897** |
| 500× | 28.61/0.8117 | Fail | **29.40/0.8524** |
| 1000× | 27.06/0.7854 | Fail | **27.49/0.8226** |
| 2000× | 25.54/0.7642 | Fail | **25.70/0.7966** |
| 3000× | Fail | Fail | **24.72/0.7844** |

SmartSplat是唯一能在3000×压缩比下完成训练的方法，在16K上平均PSNR领先GI约5.64dB。

### 效率对比（10848×16320图像，CR=200）

| 方法 | 迭代速度 | 训练时间(s) | 显存(GB) | FPS | PSNR |
|------|---------|-----------|---------|-----|------|
| 3DGS (10K) | 1.32 it/s | 7841.80 | 50.19 | 10.98 | 24.42 |
| GI (10K) | 7.44 it/s | 1334.73 | 16.29 | 62.33 | 19.86 |
| SmartSplat (10K) | 5.01 it/s | 2237.52 | 19.59 | 32.35 | **31.87** |
| SmartSplat (1K) | 5.03 it/s | 336.12 | 19.38 | 33.12 | 30.52 |

SmartSplat仅需1K迭代即达30.52dB，超过3DGS和GI在10K迭代时的结果。显存仅为3DGS的39%。

### 消融实验（4416×6720图像，CR=200，10K迭代）

| 配置 | PSNR (dB) | MS-SSIM |
|------|----------|---------|
| Full Random | 22.34 | 0.8435 |
| +VS/US位置初始化 | 22.18 | 0.8270 |
| +VS/US尺度初始化 | 23.12 | 0.8647 |
| +尺度自适应颜色（完整SmartSplat） | **24.38** | **0.8972** |

尺度初始化贡献最大（+0.94dB），颜色初始化进一步提升1.26dB，三者缺一不可。

## 亮点

- **首个UHR GS压缩框架**：首次将GS-based图像压缩推进到8K/16K级别，支持最高5000×的极端压缩比
- **无超参数尺度初始化**：$s_{base}$ 完全由图像分辨率和压缩比推导，无需人工调参或启发式clamp
- **三阶段联合初始化**：位置、尺度、颜色的协同初始化策略，使SmartSplat仅需1K迭代即可超越baseline的10K迭代结果
- **鲁棒的颜色估计**：Gaussian加权中值滤波比随机初始化和像素中心方法在高频纹理区域表现更优
- **极强的可扩展性**：在其他方法因OOM或NaN失效时，SmartSplat仍能稳定工作

## 局限与展望

- **仅优化空间分布**：当前框架聚焦于Gaussian的空间分布优化，未涉及Gaussian属性（颜色、透明度）的进一步压缩（如量化、编码），这是提升压缩效率的重要方向
- **DIV16K数据集的构建方式**：通过超分辨率工具从DIV2K放大获得，与真实16K捕获的图像在纹理细节上可能存在分布差异
- **评估规模有限**：DIV8K仅选取16张图片、DIV16K仅8张图片进行评估，统计显著性存疑
- **解码速度中等**：FPS为32左右，虽远优于INR方法，但不如GI的62 FPS，实时应用仍有优化空间
- **与神经编解码器未比较**：缺乏与端到端学习的图像编解码器（如Hyperprior、ELIC等）的对比

## 与相关工作的对比

- **GaussianImage (GI)**：采用两阶段优化+向量量化，在高压缩比时易因Gaussian不足导致NaN失效；SmartSplat通过自适应初始化规避此问题，在相同CR下PSNR领先2.57dB
- **LIG**：层级Gaussian策略优先保证拟合精度而非压缩性能，需大量Gaussian组件；在高CR下性能骤降
- **ImageGS**：内容感知初始化+渐进训练，在极端压缩场景下不稳定；SmartSplat在DIV16K上ImageGS因OOM完全无法运行
- **3DGS**：直接扩展到2D有一定效果，但因身份矩阵映射导致训练慢、显存高（3DGS占50GB vs SmartSplat占20GB）

## 启发与关联

- **特征引导初始化的通用范式**：利用图像梯度和颜色方差引导离散基元的空间分布，这一思路可推广到其他基于基元表示的任务（如点云压缩、NeRF初始化）
- **压缩比与基元数量的显式关系**：$N_g = 3HW / (7 \cdot \mathrm{CR})$ 这一公式清晰建立了压缩比与表示能力的联系，为后续工作提供了分析框架
- **排斥采样的思路**：通过排斥半径约束避免采样点重叠，类似于Poisson Disk Sampling在图形学中的应用，可借鉴到其他需要均匀覆盖的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将GS压缩推进到UHR级别，三阶段初始化策略设计合理，但核心思想（特征引导采样）并非全新
- 实验充分度: ⭐⭐⭐ — 实验设计完整（主实验+消融+效率对比），但评估图片数量少（16+8张），缺乏与神经编解码器的对比
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，方法描述详细，图表丰富；但部分notation较冗长
- 价值: ⭐⭐⭐⭐ — 在UHR图像压缩这一实际需求强烈的方向上取得显著进展，代码开源提升了复现性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LoG3D: Ultra-High-Resolution 3D Shape Modeling via Local-to-Global Partitioning](../../CVPR2026/3d_vision/log3d_ultra-high-resolution_3d_shape_modeling_via_local-to-global_partitioning.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[CVPR 2026\] Scalable Feature Matching via State Space Modeling and Sparse Correlation](../../CVPR2026/3d_vision/scalable_feature_matching_via_state_space_modeling_and_sparse_correlation.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[NeurIPS 2025\] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](../../NeurIPS2025/3d_vision/zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting
description: >-
  [ICCV 2025][3D视觉][4D重建] 本文提出Vivid4D，将单目视频的多视角增广任务转化为视频修复（inpainting）问题——先用单目深度先验将视频warp到新视角，再用视频扩散模型修复遮挡区域，通过迭代视角扩展策略和鲁棒重建损失显著改善了单目4D动态场景的重建质量。 从随意拍摄的单目视频重建4D动态场景是…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "4D重建"
  - "单目视频"
  - "视频修复"
  - "扩散模型"
  - "视角增广"
---

# Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting

**会议**: ICCV 2025  
**arXiv**: [2504.11092](https://arxiv.org/abs/2504.11092)  
**代码**: [https://xdimlab.github.io/Vivid4D/](https://xdimlab.github.io/Vivid4D/)  
**领域**: 3D视觉  
**关键词**: 4D重建, 单目视频, 视频修复, 扩散模型, 视角增广

## 一句话总结
本文提出Vivid4D，将单目视频的多视角增广任务转化为视频修复（inpainting）问题——先用单目深度先验将视频warp到新视角，再用视频扩散模型修复遮挡区域，通过迭代视角扩展策略和鲁棒重建损失显著改善了单目4D动态场景的重建质量。

## 研究背景与动机

从随意拍摄的单目视频重建4D动态场景是计算机视觉与图形学的核心挑战。每个时间戳仅有一个视角的观测，这使得问题严重欠定。

现有方法的两条路线及其局限性：

**几何先验路线**（光流、深度估计、tracking等）：这些辅助监督信号本身可能不可靠，且与渲染性能不一定线性相关（例如小的深度误差可能导致大的颜色偏移）。更关键的是，这些先验仅来自输入视角，无法为输入中被遮挡/未观测的区域提供指导

**生成先验路线**（视频扩散模型）：能为未知视角生成合理的RGB图像，但现有方法要么仅限于静态场景，要么需要大量带有相机位姿的数据进行训练（获取动态场景相机位姿非常困难）

核心矛盾：**几何先验无法生成新内容，生成先验缺乏精确的几何约束，如何将两者优势结合？**

本文的核心idea：**将视角增广重新定义为视频修复任务——利用深度先验的几何信息warp现有视角到新视点（保留已知区域），然后用视频扩散模型修复因遮挡产生的缺失区域（生成新内容），最重要的是训练数据只需无位姿的网络视频**。

## 方法详解

### 整体框架

Vivid4D包含三个核心阶段：(1) 训练视频修复扩散模型；(2) 迭代式视角增广生成多视角监督；(3) 使用原始+增广视频进行4D重建。

具体流程：给定单目视频，先用COLMAP获取相机参数和稀疏点云，用单目深度估计获取深度图并对齐到度量尺度，然后迭代地将视频warp到新视点，用修复模型填充遮挡区域，最终用所有视频监督基于运动场的3DGS进行4D重建。

### 关键设计

1. **锚点条件视频修复扩散模型**：

    - 功能：在预训练视频扩散模型基础上微调，同时接收mask视频、二值mask和锚点视频作为输入
    - 核心思路：扩展标准视频扩散模型的输入通道，将遮罩视频的VAE编码 $\mathbf{z}_m$（4通道）、下采样二值mask $\mathcal{M}'$（1通道）、噪声latent $\mathbf{z}_t$（4通道）和锚点视频的VAE编码 $\mathbf{z}_a$（4通道）沿通道维度拼接。训练目标：
    $\mathcal{L} = \mathbb{E}_{\mathbf{x},t,\epsilon \sim \mathcal{N}(0,1)} ||\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_m, \mathcal{M}', \mathbf{z}_a)||_2^2$
    - 设计动机：锚点视频（原始未warp的视频）提供完整的时空上下文，帮助模型在填充遮挡区域时保持与原始场景的一致性

2. **基于2D Tracking的训练数据生成**：

    - 功能：从无位姿的网络视频中自动生成训练数据对
    - 核心思路：使用预训练的2D tracking模型从视频第一帧采样N个点并跟踪到后续帧。每帧中未被任何跟踪点覆盖的像素区域即为mask区域 $\mathcal{M}_t$，代表因物体/相机运动而新暴露的区域
    - 设计动机：这些自然产生的遮挡区域完美模拟了warp后产生的空洞，无需已知相机位姿即可获取训练数据，极大扩展了可用训练数据范围

3. **迭代式视角增广策略**：

    - 功能：渐进地将warp角度从小到大扩展，分N次迭代生成augmented views
    - 核心思路：维护一个数据缓冲区 $\mathcal{D}_j = \mathcal{D}_{j-1} \cup (\hat{\mathcal{V}}^j, D^j, \mathbf{T}^j)$，每次迭代从缓冲区中选择warp角度最小的帧进行warp，生成的视频送入修复模型后加入缓冲区。同时引入监督mask $S_t^j$ 避免跨迭代重复修复导致的不一致
    - 设计动机：深度先验的不准确性（如bleeding edges）在大角度warp时会引入明显伪影，递进式warp从小角度开始，每次利用已有的高质量视图来生成更偏的视角，最小化失真

4. **Invariant Vicinity (IV) RGB损失**：

    - 功能：对augmented views使用鲁棒的像素级损失
    - 核心思路：计算每个渲染像素与对应监督图像3×3邻域中最近像素的L1损失，仅对最小误差像素反传梯度
    - 设计动机：深度估计误差和扩散模型伪影会导致augmented views与ground truth有轻微错位，IV损失降低了对这种错位的敏感性

### 损失函数 / 训练策略

4D重建损失分两部分：
- 原始帧：标准 L1 + SSIM + LPIPS 全图监督
- 增广帧：IV RGB loss + SSIM + LPIPS，仅在监督mask $S_t^j = 1$ 的区域计算
$$\mathcal{L} = \sum_{j=1}^N \sum_{t=1}^T (\lambda_r \mathcal{L}_\text{IV}^{t,j} + \lambda_s \mathcal{L}_\text{ssim}^{t,j} + \lambda_l \mathcal{L}_\text{lpips}^{t,j})$$

4D表示：采用Shape of Motion的运动场3DGS表示
深度估计：学习方法初始化 + COLMAP稀疏点云对齐到metric scale
训练数据：OpenVid-1M中处理5K视频用于训练修复模型

## 实验关键数据

### 主实验（iPhone数据集 + HyperNeRF数据集）

| 方法 | iPhone mPSNR↑ | iPhone mSSIM↑ | iPhone mLPIPS↓ | HyperNeRF mPSNR↑ | HyperNeRF mLPIPS↓ |
|------|-------------|-------------|--------------|----------------|----------------|
| 4D GS | 14.01 | 0.3877 | 0.5939 | 18.24 | 0.4450 |
| Shape of Motion | 14.56 | 0.4570 | 0.5292 | 18.82 | 0.4589 |
| CoCoCo | 14.99 | 0.4701 | 0.5280 | 19.00 | 0.4692 |
| StereoCrafter | 14.85 | 0.4945 | 0.5676 | 18.86 | 0.5181 |
| ViewCrafter | 14.94 | 0.4888 | 0.5772 | 18.91 | 0.4888 |
| **Vivid4D (Ours)** | **15.20** | **0.5004** | **0.4930** | **19.45** | **0.4449** |

### 消融实验

| 配置 | mPSNR↑ | mSSIM↑ | mLPIPS↓ | 说明 |
|------|--------|--------|---------|------|
| (a) 无warp, 无inpaint, 无深度 | 16.04 | - | - | 基线（仅原始帧） |
| (b) 无warp, 无inpaint, 有深度监督 | 轻微提升 | - | - | 直接深度监督效果有限 |
| (c) 有warp, 无inpaint | 较大提升 | - | - | 几何warp本身就比深度监督更有效 |
| (d) 有warp, 有inpaint (ours) | **最佳** | - | - | 修复遮挡区域进一步提升 |
| 无锚点视频 | 25.34 PSNR | 0.8053 SSIM | 0.1056 LPIPS | 修复5K视频评测 |
| **有锚点视频** | **27.22** | **0.8223** | **0.0801** | 锚点提供时空上下文 |
| 直接warp (N=1) | 次优 | - | - | 大角度warp引入伪影 |
| **迭代warp (N=2)** | **最优** | - | - | 递进扩展减少失真 |

### 关键发现

- 将深度先验用于warp（间接利用），比直接作为监督信号（直接利用）效果更好——验证了几何先验与渲染性能的非线性关系
- 锚点视频条件显著提升修复质量（PSNR从25.34提升到27.22），有效减少伪影
- 迭代扩展策略比一步直接warp更好，因为小角度warp的深度误差影响更小
- Vivid4D在所有指标上超越所有基线，包括专门的视频修复方法（CoCoCo）和3D感知修复方法（ViewCrafter）
- 该方法能有效填充输入视频中不可见的区域（4D GS和SoM中的黑洞/白色区域），用修复内容替代

## 亮点与洞察

1. **视角增广 = warp + inpaint 的巧妙重定义**：将几何先验（warp提供已知内容的布局）与生成先验（扩散模型填充未知区域）无缝结合，两者互补而非替代
2. **训练数据获取的创新**：利用2D tracking在无位姿网络视频中自动生成"模拟warp遮挡"的训练数据，避免了获取动态场景相机位姿的困难
3. **锚点视频机制**：简单的通道拼接实现了对原始视频时空信息的有效利用，3D U-Net自然学习到跨视角的时空对应关系
4. **IV RGB损失的实用性**：用3×3邻域最小值替代严格的逐像素L1，优雅地处理了深度误差引起的亚像素错位

## 局限与展望

- 依赖COLMAP获取初始相机位姿，对于纯动态场景（如手持晃动+物体运动）可能失败
- 修复模型的训练数据量（5K视频）相对有限，可能限制修复质量的上限
- 迭代warp增加了预处理时间，每次迭代需要运行扩散模型+深度估计
- 修复内容的几何一致性依赖于扩散模型的能力，可能在极端视角变化下失败
- 仅在少量场景（5个iPhone + 3个HyperNeRF）上评测，规模偏小

## 相关工作与启发

- Shape of Motion [Wang et al.] 是本文采用的4D表示基础，Vivid4D在其之上通过视角增广获得显著提升
- StereoCrafter [Zhao et al.] 和 ViewCrafter [Yu et al.] 也利用扩散模型进行warp后修复，但分别需要视频warp或已知位姿的训练数据
- 启示：对于视角受限的重建任务，**间接利用几何先验**（通过warp转化为修复问题）比直接将其作为损失函数更有效，因为前者的"错误"会被修复模型部分吸收

## 评分
- 新颖性: ⭐⭐⭐⭐ 视角增广=warp+inpaint的思路自然且有效，2D tracking生成训练数据很聪明
- 实验充分度: ⭐⭐⭐⭐ 消融实验设计合理，但测试场景数量偏少（仅8个场景）
- 写作质量: ⭐⭐⭐⭐⭐ 图示清晰（尤其是直接warp vs 迭代warp的对比），逻辑流畅
- 价值: ⭐⭐⭐⭐ 为单目4D重建提供了一个实用的增强方案，与多种4D方法兼容

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Predict-Optimize-Distill: A Self-Improving Cycle for 4D Object Understanding](predict-optimize-distill_a_self-improving_cycle_for_4d_object_understanding.md)
- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2025/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)

</div>

<!-- RELATED:END -->

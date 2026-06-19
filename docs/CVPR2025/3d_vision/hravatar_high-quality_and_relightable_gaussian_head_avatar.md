---
title: >-
  [论文解读] HRAvatar: High-Quality and Relightable Gaussian Head Avatar
description: >-
  [CVPR 2025][3D视觉][头部重建] HRAvatar 提出了一种基于3DGS的单目视频头部重建方法，通过可学习blendshapes和LBS实现灵活变形，结合端到端表情编码器减少追踪误差，并引入物理渲染模型实现高质量实时重光照。 领域现状：从单目视频重建可驱动的3D头部avatar是电影、游戏、AR/VR等领域的…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "头部重建"
  - "3D高斯溅射"
  - "重光照"
  - "可动画化"
  - "单目视频"
---

# HRAvatar: High-Quality and Relightable Gaussian Head Avatar

**会议**: CVPR 2025  
**arXiv**: [2503.08224](https://arxiv.org/abs/2503.08224)  
**代码**: 有（项目页面提供）  
**领域**: 3D视觉  
**关键词**: 头部重建, 3D高斯溅射, 重光照, 可动画化, 单目视频

## 一句话总结
HRAvatar 提出了一种基于3DGS的单目视频头部重建方法，通过可学习blendshapes和LBS实现灵活变形，结合端到端表情编码器减少追踪误差，并引入物理渲染模型实现高质量实时重光照。

## 研究背景与动机

**领域现状**：从单目视频重建可驱动的3D头部avatar是电影、游戏、AR/VR等领域的重要需求。近年来基于3DGS的方法（如Splatting-avatar、Flash-avatar）通过将高斯点绑定到参数化头部模型（如FLAME）上实现了实时渲染，但重建质量仍有限。

**现有痛点**：现有方法存在三个核心问题：(1) 变形灵活性不足——将高斯点刚性绑定到通用参数模型的mesh面上，无法捕捉个人化的面部变形特征；(2) 表情追踪不准确——预追踪的FLAME参数通过拟合伪2D关键点获得，误差会传播到重建质量；(3) 无法重光照——直接拟合颜色使得人物固有外观与环境光照耦合在一起。

**核心矛盾**：参数化头部模型的通用性与个人化变形需求之间的矛盾，以及未知光照下外观分解的不确定性。

**本文目标** (1) 如何让高斯点的变形更灵活更个人化；(2) 如何减少表情追踪误差对重建的影响；(3) 如何在未知光照的单目视频下实现真实的重光照。

**切入角度**：作者观察到每个人的面部形状和变形模式都是独特的，通用参数模型无法精确表达，因此为每个高斯点独立学习blendshapes基函数和混合权重。同时利用端到端训练的表情编码器替代独立的预追踪步骤。

**核心 idea**：用可学习的逐点blendshapes和LBS替代刚性FLAME绑定，结合端到端表情编码器和物理渲染模型，实现高质量可重光照的头部avatar重建。

## 方法详解

### 整体框架
输入为一段未知光照的单目头部视频（M帧），输出为可动画化、可重光照的3D头部avatar。整体流程分为三个阶段：(1) 首先通过迭代优化预追踪固定的形状参数$\beta$和姿态参数$\{\theta_j\}$；(2) 通过一个表情编码器端到端估计表情参数$\psi$和下巴姿态$\theta^{jaw}$；(3) 利用可学习的线性blendshapes和LBS将高斯点从canonical空间变换到pose空间，然后渲染albedo、roughness、reflectance和normal图，最后通过物理渲染计算像素颜色。

### 关键设计

1. **端到端精确表情追踪（Expression Encoder）**:

    - 功能：精确估计面部表情参数，减少预追踪误差对重建质量的影响
    - 核心思路：使用预训练的SMIRK编码器$\mathcal{E}$，输入当前帧图像$I$，输出表情参数$\psi$和下巴姿态$\theta^{jaw}$。关键在于该编码器在训练过程中通过光度损失进行端到端优化，而非依赖伪2D关键点的预追踪结果。同时引入下巴姿态正则化损失$\mathcal{L}_{jaw}$约束推断值与预追踪值的距离，防止偏移过大。
    - 设计动机：传统fitting方法用伪标签优化参数误差大，而端到端训练利用真实图像进行监督，既提高了精度又保持了泛化性。之前的PointAvatar直接优化参数会引入训练/测试不一致。

2. **可学习线性Blendshapes与LBS（Learnable Deformation）**:

    - 功能：实现灵活的个人化几何变形，将高斯点从canonical空间映射到pose空间
    - 核心思路：为每个高斯点引入三组可学习的blendshapes基：形状基$S$、表情基$E$和姿态基$P$，以及可学习的混合权重$\mathcal{W}$。变形过程分两步：先通过线性blendshapes计算形状/表情/姿态偏移量（$X_e = X_c + \mathcal{BS}(\psi, E) + \mathcal{BS}(\mathcal{R}(\theta^*) - \mathcal{R}(\theta^0), P)$），再通过LBS进行关节驱动的刚性变换（$X_p = R_{lbs}X_e + T_{lbs}$）。初始化时利用FLAME mesh面的线性插值提供先验。
    - 设计动机：与GBS等方法使用共享MLP不同，逐点独立学习基函数和权重能更好捕捉个人化变形，尤其是头发、饰品等非标准区域。实验证明这种策略优于共享MLP方案。

3. **物理渲染外观建模（Physically-Based Shading）**:

    - 功能：将头部外观分解为多个物理属性，实现真实的重光照效果
    - 核心思路：为每个高斯点定义albedo $a$、roughness $o$和Fresnel基反射率$f_0$三个属性。渲染时先通过光栅化得到albedo图、roughness图、reflectance图和normal图，然后利用SplitSum近似和BRDF模型计算漫反射$I_{diffuse} = \mathbf{A} \cdot I_{irr}(\mathbf{N})$和镜面反射$I_{specular}$。训练时优化两个cube map（环境辐照图$I_{irr}$和预滤波环境图$I_{env}$）。此外引入albedo伪先验损失$\mathcal{L}_{albedo}$和normal一致性损失$\mathcal{L}_{normal}$确保材质分解合理。
    - 设计动机：直接拟合颜色的SH表示无法支持重光照。使用物理渲染模型虽然参数灵活性略低，但能在保持可比重建质量的同时支持实时重光照和材质编辑。albedo先验防止局部光照效果被错误耦合到albedo中。

### 损失函数 / 训练策略
总损失为：$\mathcal{L}_{total} = \mathcal{L}_{rgb} + 0.1\mathcal{L}_{jaw} + 10^{-5}\mathcal{L}_{normal} + 0.25\mathcal{L}_{albedo} + 0.02\mathcal{L}_{tv}(\mathbf{O})$。其中$\mathcal{L}_{rgb}$结合了MAE（权重0.8）和D-SSIM（权重0.2），$\mathcal{L}_{tv}$为roughness图的全变差损失保证平滑性。保留3DGS原有的点密集化和裁剪策略。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HRAvatar | GBS | Flash-avatar | 提升 |
|--------|------|----------|-----|-------------|------|
| INSTA (10人) | PSNR↑ | **30.36** | 29.64 | 29.13 | +0.72 |
| INSTA | LPIPS↓ | **0.0569** | 0.0823 | 0.0719 | -30.9% |
| HDTF (8人) | PSNR↑ | **28.55** | 27.81 | 27.58 | +0.74 |
| HDTF | LPIPS↓ | **0.0825** | 0.1297 | 0.1095 | -36.4% |
| Self-captured (5人) | PSNR↑ | **28.97** | 28.59 | 27.46 | +0.38 |
| Self-captured | LPIPS↓ | **0.1059** | 0.1560 | 0.1456 | -32.1% |

渲染速度约 **155 FPS**，支持实时动画和重光照。

### 消融实验

| 配置 | PSNR↑ | MAE*↓ | SSIM↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| Full model | **30.36** | **0.845** | **0.9482** | **0.0569** |
| Rigged to FLAME | 29.79 | 0.937 | 0.9431 | 0.0695 |
| MLP deform | 29.67 | 0.966 | 0.941 | 0.0706 |
| w/o exp. encoder | 29.70 | 0.933 | 0.9438 | 0.0667 |
| w/o learnable deform | 29.83 | 0.923 | 0.9440 | 0.0684 |
| w/o PBS | 30.34 | 0.850 | 0.9480 | 0.0563 |

### 关键发现
- 可学习变形模型贡献最大：rigged to FLAME掉了0.57 PSNR，MLP deform掉了0.69 PSNR，说明逐点独立学习的策略最优
- 表情编码器有效：去掉后PSNR掉0.66，且视觉上嘴型和眨眼等精细表情明显恶化
- PBS模式在重建指标上与标准3DGS几乎持平（仅差0.02 PSNR），但获得了重光照能力
- 去掉$\mathcal{L}_{albedo}$后albedo与高光耦合，导致重光照不真实；去掉$\mathcal{L}_{normal}$后法线图混乱，重光照出现blocky artifacts

## 亮点与洞察
- **逐点独立学习blendshapes**：不同于GBS的全局Gaussian basis或PointAvatar的共享MLP，为每个高斯点独立学习变形基函数，实现了最灵活的个人化变形建模。这种"以空间换精度"的思路可迁移到身体重建等任务。
- **表情追踪与重建联合优化的巧妙设计**：编码器端到端训练但加了下巴正则化，在精度提升和稳定性之间取得了好平衡。这一思路可用于任何需要参数估计+下游任务联合的场景。
- **PBS近似不掉点**：物理渲染分支在保持与纯拟合方法几乎相同PSNR的同时获得了重光照能力，说明近似物理模型对头部场景足够有效。

## 局限与展望
- 训练数据不足时仍受FLAME先验约束，头发、饰品等非标准元素的控制有限
- 部分阴影或皱纹可能被错误耦合到albedo或reflectance中，镜面反射和阴影的重光照效果仍有瑕疵
- 无法处理未知相机位姿的全头重建——当偏航角接近90度时面部关键点不可靠
- 改进方向：引入更好的albedo估计模型（如扩散先验）、拓展到全身重建、支持更复杂的光照模型（如子表面散射）

## 相关工作与启发
- **vs Flash-avatar**: Flash-avatar将高斯点刚性绑定到FLAME mesh面，变形灵活性低；本文独立学习逐点变形，PSNR提升1.2+，LPIPS大幅改善
- **vs GBS (3D Gaussian Blendshapes)**: GBS学习全局Gaussian basis处理表情但姿态变化差；本文同时处理表情和姿态，且引入了LBS和重光照
- **vs FLARE**: FLARE用mesh+BRDF重光照但重建质量有限且法线噪声大；本文用3DGS获得更高质量和更平滑法线
- **vs PointAvatar**: PointAvatar用共享MLP预测变形且直接优化参数，测试时需post-optimization；本文用编码器保持泛化性

## 评分
- 新颖性: ⭐⭐⭐⭐ 逐点可学习blendshapes+LBS+端到端表情编码器+物理渲染的组合很完整，但各技术单独看并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集23个subject，全面的消融实验，与5个baseline比较，包含重光照和cross-reenactment
- 写作质量: ⭐⭐⭐⭐ 条理清晰，动机阐述充分，但部分公式符号较密
- 价值: ⭐⭐⭐⭐ 实现了高质量+实时+可重光照的单目头部重建，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[CVPR 2025\] Synthetic Prior for Few-Shot Drivable Head Avatar Inversion](synthetic_prior_for_few-shot_drivable_head_avatar_inversion.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](../../ICCV2025/3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)

</div>

<!-- RELATED:END -->

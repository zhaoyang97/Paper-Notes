---
title: >-
  [论文解读] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion
description: >-
  [CVPR 2025][视频生成][头部虚拟形象] 提出 Zero-1-to-A，通过共生生成（SymGEN）和渐进式学习策略，从单张图片利用预训练视频扩散模型生成高保真可动画化 4D 头部虚拟形象，有效解决了视频扩散的时空不一致性问题。 - 可动画化头部虚拟形象生成通常需要大量真实或合成人类数据，收集困难 - 基于分数蒸馏…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "头部虚拟形象"
  - "视频扩散"
  - "3D高斯"
  - "渐进式学习"
  - "零样本生成"
---

# Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion

**会议**: CVPR 2025  
**arXiv**: [2503.15851](https://arxiv.org/abs/2503.15851)  
**代码**: [GitHub](https://github.com/ZhenglinZhou/Zero-1-to-A)  
**领域**: 视频生成  
**关键词**: 头部虚拟形象, 视频扩散, 3D高斯, 渐进式学习, 零样本生成

## 一句话总结

提出 Zero-1-to-A，通过共生生成（SymGEN）和渐进式学习策略，从单张图片利用预训练视频扩散模型生成高保真可动画化 4D 头部虚拟形象，有效解决了视频扩散的时空不一致性问题。

## 研究背景与动机

- 可动画化头部虚拟形象生成通常需要大量真实或合成人类数据，收集困难
- 基于分数蒸馏（SDS）的方法利用预训练扩散模型实现零样本生成，但直接从视频扩散蒸馏 4D 虚拟形象会因时空不一致导致过度平滑
- 视频扩散模型在生成肖像视频时存在空间不一致（不同视角的外观变化）和时间不一致（表情序列的不连贯）
- SDS 损失直接将虚拟形象与扩散模型的伪真值对齐，但伪真值本身不稳定，导致结果质量下降
- 图像条件的 4D 虚拟形象生成相比文本条件更具挑战性，但对实际应用更重要
- 需要一种稳健的方法来从不一致的视频扩散输出中合成一致的数据集用于虚拟形象重建

## 方法详解

### 整体框架

Zero-1-to-A 基于可动画化高斯头部（FLAME + 3DGS）作为 4D 表示。核心包含两个组件：(1) SymGEN 共生生成——建立数据集构建与虚拟形象重建之间的互惠关系，通过可更新数据集缓存视频扩散结果并迭代优化；(2) 渐进式学习策略——将视频扩散生成解耦为空间一致性学习（固定表情，从正面到侧面）和时间一致性学习（固定视角，从松弛到夸张表情），从简单到复杂逐步提升质量。

### 关键设计

**1. 共生生成 (SymGEN)**
- **功能**: 建立数据集与虚拟形象之间的互惠提升循环，迭代提高两者质量
- **核心思路**: (1) 虚拟形象驱动数据集增强：渲染当前虚拟形象视频→提取 Mediapipe 面部关键点→VAE编码+DDIM反转获取噪声→以关键点为几何引导去噪→生成增强视频替换数据集；(2) 数据集精炼虚拟形象重建：在更新后的数据集上使用 $\mathcal{L}_1$+LPIPS+位置损失+缩放损失训练虚拟形象；(3) 每30次迭代更新一次数据集
- **设计动机**: 一次性生成数据集质量差（空间和时间不一致），通过迭代互惠提升可逐步消除不一致。虚拟形象质量越高→渲染引导越准确→视频扩散输出越一致→反过来提升虚拟形象

**2. 空间一致性学习 (Spatial Consistency Learning)**
- **功能**: 固定表情，从正面到侧面渐进学习多视角
- **核心思路**: 创建 $n_s=20$ 个空间样本，每个包含固定的 ARKit 基础表情和一条从正面到随机侧面的相机轨迹。通过 $p_i = \hat{p}_{\min(i,j)}$、$j = \min(\lfloor k/d_s \rfloor + 1, n_f)$ 渐进引入更大角度的侧面视角
- **设计动机**: 视频扩散在简单相机姿态和松弛表情下产生更一致的结果。先从正面开始建立良好初始化，再逐步添加侧面视角

**3. 时间一致性学习 (Temporal Consistency Learning)**
- **功能**: 固定近正面相机，从合成松弛表情到真实夸张表情渐进学习
- **核心思路**: 使用固定近正面相机姿态。前 $k_s=5000$ 次仅学空间，5000-8000 次添加 $n_{syn}=10$ 个合成松弛表情样本，8000 次后添加 $n_{real}=10$ 个来自脱口秀视频的真实夸张表情。类似于测试时训练（TTT）策略
- **设计动机**: 真实表情序列比合成表情更夸张更多变，直接学习易导致不一致。先在简单合成表情上建立鲁棒基础，再逐步引入挑战性数据

### 损失函数

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{lpips} \mathcal{L}_{LPIPS} + \lambda_{pos} \mathcal{L}_{pos} + \lambda_s \mathcal{L}_s$$

其中 $\lambda_1=10$, $\lambda_{lpips}=10$, $\lambda_{pos}=0.1$, $\lambda_s=10$。$\mathcal{L}_{pos}$ 和 $\mathcal{L}_s$ 分别约束 3D 高斯点与 FLAME 网格的位置和缩放对齐。

## 实验关键数据

### 主实验：定量评估

| 方法 | CLIP-Score (ViT-L/14)↑ | 渲染速度(FPS) |
|------|----------------------|-------------|
| DreamHead | ~0.68 | ~2 |
| HeadStudio | ~0.72 | ~5 |
| Portrait4D-v2 | ~0.70 | ~15 |
| **Zero-1-to-A** | **~0.76** | **~90** |

*Zero-1-to-A 在保真度和渲染速度上均大幅领先*

### 消融实验

| 变体 | CLIP-Score↑ | 视觉质量 |
|------|-----------|---------|
| SDS Loss (baseline) | ~0.62 | 过度平滑 |
| One-time Dataset | ~0.66 | 质量较差 |
| SymGEN w/o Progressive | ~0.72 | 部分伪影 |
| **Full Zero-1-to-A** | **~0.76** | **清晰细节** |

### 关键发现

- 直接 SDS 蒸馏产生过度平滑的结果，SymGEN 的迭代数据集更新显著提升质量
- 渐进式学习打破了"先有鸡还是先有蛋"的困境——初始低质量虚拟形象无法提供好的引导
- 空间和时间解耦学习各自发挥作用，组合效果最优
- 基于 3DGS 的表示实现了约 90 FPS 的实时渲染

## 亮点与洞察

1. **共生生成范式**: 将数据集构建与模型训练的互惠关系系统化，不依赖外部数据即可渐进提升质量
2. **从简到繁的课程学习**: 将4D生成解耦为空间→时间两阶段，每阶段内由易到难，避免了一步到位的不稳定性
3. **零样本+实时渲染**: 仅需单张图片+5小时优化即可生成可动画化的高质量虚拟形象

## 局限与展望

- 优化过程仍需约5小时（单张 A6000 GPU），不适合实时应用
- 对极端侧面视角和极端表情的处理仍有改进空间
- 依赖视频扩散模型的质量，不同基础模型可能导致不同效果
- 未来可探索更高效的数据集更新策略以减少优化时间

## 相关工作与启发

- 与 DreamHead、HeadStudio 等基于 SDS 的方法相比，用重建替代蒸馏避免了过度平滑
- 共生生成的思路可推广到其他需要从不一致生成结果中学习的场景
- 渐进式学习策略为所有视频扩散蒸馏任务提供了有价值的参考

## 评分

⭐⭐⭐⭐ — 巧妙地解决了视频扩散蒸馏中的时空不一致性问题，共生生成和渐进式学习两个创新互相配合。实验结果在保真度和渲染速度上均令人信服。但5小时的优化时间和对基础模型的依赖是实际部署的限制。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2026\] Are Image-to-Video Models Good Zero-Shot Image Editors?](../../CVPR2026/video_generation/are_image-to-video_models_good_zero-shot_image_editors.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[ICML 2026\] WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](../../ICML2026/video_generation/wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)

</div>

<!-- RELATED:END -->

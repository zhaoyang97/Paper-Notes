---
title: >-
  [论文解读] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][3DGS风格迁移] 本文提出FantasyStyle，首个完全基于扩散模型蒸馏的3DGS风格迁移框架，通过多视图频率一致性（MVFC）机制抑制低频分量减少视角间冲突，并设计可控风格化蒸馏（CSD）引入负引导消除风格图像的内容泄漏，在风格化质量和内容保持上均超越现有VGG和扩散方法。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3DGS风格迁移"
  - "扩散模型蒸馏"
  - "多视图一致性"
  - "频率分析"
  - "负引导"
---

# FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2508.08136](https://arxiv.org/abs/2508.08136)  
**代码**: [https://github.com/yangyt46/FantasyStyle](https://github.com/yangyt46/FantasyStyle)  
**领域**: 3D Vision / Style Transfer  
**关键词**: 3DGS风格迁移, 扩散模型蒸馏, 多视图一致性, 频率分析, 负引导

## 一句话总结
本文提出FantasyStyle，首个完全基于扩散模型蒸馏的3DGS风格迁移框架，通过多视图频率一致性（MVFC）机制抑制低频分量减少视角间冲突，并设计可控风格化蒸馏（CSD）引入负引导消除风格图像的内容泄漏，在风格化质量和内容保持上均超越现有VGG和扩散方法。

## 研究背景与动机
随着VR/AR对艺术化3D内容需求增长，3D风格迁移成为研究热点。3D Gaussian Splatting (3DGS)以其快速渲染和高视觉质量成为新兴的3D表示方法，但现有3DGS风格迁移仍面临两大核心挑战：

**多视图不一致性**：使用2D扩散先验指导3D风格化时，不同视角的风格化结果常常出现风格冲突（颜色、笔触不一致），导致优化时的相互抵消，最终产生模糊和几何扭曲。

**内容泄漏与过度风格化**：主流方法依赖VGG特征提取，但VGG难以有效解耦风格和内容。这导致风格图像的内容信息（如特定物体形状）被错误迁移到目标场景，同时低层纹理的过度匹配产生过度风格化，遮盖结构细节。

本文是**首个完全基于扩散模型蒸馏**（不使用任何VGG特征）的3DGS风格迁移框架，核心切入角度是从频率域和引导机制两个维度解决上述问题。

## 方法详解

### 整体框架
FantasyStyle基于DDS（Delta Denoising Score）的双路径架构：Source Image路径和Rendered Image路径。对渲染图像路径引入MVFC增强多视图一致性，通过IP-Adapter注入风格特征获取2D风格化先验，使用负引导抑制内容泄漏，最终通过CSD优化3D Gaussians的颜色参数（保持几何不变）。

### 关键设计
1. **多视图频率一致性 (MVFC)**:

    - 功能：在DDIM加噪后的多视图latent上施加3D频率域滤波，提高视角间一致性
    - 核心思路：通过3D FFT将多视图噪声latent分解为低频和高频分量。关键观察是——低频分量主要反映视角相关的局部细节，跨视角一致性差；高频分量更稳定地捕捉纹理特征，跨视角一致性好。因此保留所有高频分量，选择性地衰减低频分量（用系数$\gamma$控制），并引入跨视图共享的低频高斯噪声以显式增强一致性
    - 设计动机：灵感来自FreeU和FreeInit对频率成分在图像/视频生成中关键作用的发现。通过频率域操作可以在不破坏纹理的前提下有效减少视角间的风格冲突

2. **可控风格化蒸馏 (CSD)**:

    - 功能：设计新的蒸馏损失函数，利用2D风格化先验优化3D场景
    - 核心思路：首先分析SDS和DDS失败的原因——它们的重建项$\delta_{z_t}^{recon}$导致输出过度平滑，丢失关键笔触细节。CSD的做法是直接去除重建项，仅保留CFG引导项。同时将标准CFG中的空文本条件替换为风格图像的内容特征作为负引导，使生成的风格化先验不含内容信息
    - 设计动机：在风格迁移任务中只需修改颜色参数而不涉及几何/身份保持，重建项反而成为限制因素；负引导能主动排除风格图像中不需要迁移的内容信息

3. **IP-Adapter + ControlNet集成**:

    - 功能：注入风格信息的同时保持结构一致
    - 核心思路：使用IP-Adapter-Instruct从风格图像中分别提取风格特征$\text{IP}(I_r)^s$和内容特征$\text{IP}(I_r)^c$；风格特征用于正向引导，内容特征用于负引导。ControlNet引导结构信息，弥补生成2D先验时的几何信息损失
    - 设计动机：单纯的文本prompt不足以精确描述风格图像的视觉特征，IP-Adapter提供了更直接有效的风格注入方式

### 损失函数 / 训练策略
CSD梯度公式：

$$\nabla_\theta \mathcal{L}_{CSD} = \mathbb{E}_{t,\epsilon}[\omega(t)(\Phi^{tgt} - \Phi^{src})\frac{\partial z_t^{tgt}}{\partial \theta}]$$

其中 $\Phi^{tgt} = \beta(\epsilon_\phi(z_t^{tgt}, t, [\mathcal{P}, \text{IP}(I_r)^s]) - \epsilon_\phi(z_t^{tgt}, t, \text{IP}(I_r)^c))$

$\Phi^{src} = \beta(\epsilon_\phi(z_t^{src}, t, \mathcal{P}) - \epsilon_\phi(z_t^{src}, t, \varnothing))$

使用SDXL作为扩散模型主干，CFG scale $\beta=7.5$，MVFC参数$\gamma=0.9$。从离散时间步集合中随机采样（模拟DDIM去噪过程）。所有实验在2×NVIDIA L20 (48GB) GPU上进行。

## 实验关键数据

### 主实验

| 方法 | ArtFID↓ | FID_style↓ | FID_content↓ | Short LPIPS↓ | Long LPIPS↓ |
|------|---------|-----------|-------------|-------------|-------------|
| StyleGaussian | 45.31 | 398.17 | 331.53 | 0.290 | 0.542 |
| SGSST | 44.70 | 370.03 | 314.09 | 0.295 | 0.569 |
| **FantasyStyle** | **43.52** | **347.61** | **261.71** | **0.285** | **0.529** |

FantasyStyle在所有关键指标上均取得最优或次优，FID_content相比次优方法降低约50。

### 消融实验

| 消融项 | Short LPIPS↓ | Long LPIPS↓ |
|--------|-------------|-------------|
| w/o MVFC | 0.253 | 0.587 |
| **完整方法** | **0.250** | **0.574** |

| 优化策略 | 视觉效果 |
|---------|---------|
| SDS | 颜色迁移成功但笔触纹理丢失，过度平滑 |
| DDS | 类似SDS，丢失笔触细节 |
| **CSD** | **保留笔触特征，风格化质量最优** |

MVFC的提升在长时一致性上更显著（专门针对多视图一致性设计）。去除重建项后CFG scale变得不敏感，降低了超参调整复杂度。

### 关键发现
- **低频分量是视角不一致的元凶**：适度衰减低频仅略微减少局部细节但大幅提升多视图一致性，而去除高频分量则严重破坏纹理
- **SDS/DDS的重建项在风格迁移中有害**：因为风格迁移只修改颜色不涉及身份保持，重建项导致过度平滑并减慢优化
- **VGG方法的根本局限**：VGG过度关注风格图像的外观而非提取可迁移的抽象风格表示，导致内容泄漏。扩散模型能提取更高层次的风格语义
- **方法可扩展性**：FantasyStyle可以灵活集成其他2D风格迁移方法（如图7所示），2D风格化质量的提升直接转化为3D视觉质量的提升

## 亮点与洞察
- 首个完全基于扩散模型蒸馏的3DGS风格迁移框架，填补了2D到3D扩散风格迁移的空白
- 频率域分析揭示了多视图不一致性的本质原因，MVFC设计简洁优雅
- CSD去除重建项+负引导的组合巧妙解决了内容泄漏和过度平滑两个问题
- 架构设计具有良好的可扩展性，可作为将2D风格迁移扩展到3D场景的通用桥梁

## 局限与展望
- 基于SDXL的优化过程耗时较长，但可通过更小的模型、更低分辨率或调整学习率缓解
- 定量比较的baseline方法仅两种（StyleGaussian、SGSST），可增加更多对比
- 未在动态场景（如4D Gaussian）上验证
- MVFC中$\gamma$参数的选择缺乏自适应机制
- 风格化质量在很大程度上依赖IP-Adapter的特征提取能力

## 相关工作与启发
- 与VGG方法形成鲜明对比：VGG在2D风格迁移中已被扩散方法取代，但3DGS领域仍在大量使用VGG，本文推动了3DGS风格迁移向扩散范式的转型
- CSD的设计（去除重建项+负引导）可推广到其他3D编辑任务
- 频率域控制多视图一致性的思路可能对3D生成、NeRF/3DGS编辑等其他任务也有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首个纯扩散蒸馏的3DGS风格迁移+频率域一致性控制)
- 实验充分度: ⭐⭐⭐⭐ (消融充分但baseline数量有限)
- 写作质量: ⭐⭐⭐⭐⭐ (公式推导清晰，motivation分析到位)
- 价值: ⭐⭐⭐⭐ (在3DGS风格迁移领域具有开创意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](../../CVPR2025/3d_vision/dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2026\] EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images](../../CVPR2026/3d_vision/ecosplat_efficiency-controllable_feed-forward_3d_gaussian_splatting_from_multi-v.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] PoseMaster: A Unified 3D Native Framework for Stylized Pose Generation](../../CVPR2026/3d_vision/posemaster_a_unified_3d_native_framework_for_stylized_pose_generation.md)

</div>

<!-- RELATED:END -->

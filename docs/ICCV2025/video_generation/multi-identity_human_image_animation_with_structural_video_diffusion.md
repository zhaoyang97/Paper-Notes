---
title: >-
  [论文解读] Multi-identity Human Image Animation with Structural Video Diffusion
description: >-
  [ICCV 2025][视频生成][多身份人体动画] 本文提出Structural Video Diffusion框架，通过基于掩码引导的身份特定嵌入保持多人外观一致性，联合学习RGB/深度/法线三模态几何结构信息建模人物-物体交互，配合25K多人交互视频数据集Multi-HumanVid，实现多身份人体视频生成。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "多身份人体动画"
  - "视频扩散模型"
  - "身份嵌入"
  - "深度法线联合学习"
  - "人物交互"
---

# Multi-identity Human Image Animation with Structural Video Diffusion

**会议**: ICCV 2025  
**arXiv**: [2504.04126](https://arxiv.org/abs/2504.04126)  
**代码**: [GitHub](https://github.com/AvatarAnything/StructuralVideoDiffusion)  
**领域**: 视频生成  
**关键词**: 多身份人体动画, 视频扩散模型, 身份嵌入, 深度法线联合学习, 人物交互

## 一句话总结
本文提出Structural Video Diffusion框架，通过基于掩码引导的身份特定嵌入保持多人外观一致性，联合学习RGB/深度/法线三模态几何结构信息建模人物-物体交互，配合25K多人交互视频数据集Multi-HumanVid，实现多身份人体视频生成。

## 研究背景与动机

1. **领域现状**: 人体图像动画领域以Animate Anyone, MagicAnimate, CamAnimate等为代表，使用姿态引导的扩散模型从单张图像生成高保真人体视频。
2. **现有痛点**: 现有方法专注于单人动画，在多身份场景（多人交互、人物-物体交互）中表现差——无法关联正确的外观-姿态对，也缺乏对3D空间关系的建模。双人跳舞时身份混淆，手持物品时物体模糊浮空或消失。
3. **核心矛盾**: (1) 多人场景需要可追踪的身份特定特征来保持外观一致性，但现有框架没有身份区分机制；(2) 仅靠姿态引导无法建模人物-物体交互的3D空间关系。
4. **本文目标**: 如何在多人场景中保持每个人的外观一致性？如何建模复杂的人物-物体3D交互？
5. **切入角度**: 借鉴DETR的可学习查询思路设计身份嵌入；联合预测RGB+深度+法线作为隐式3D结构监督。
6. **核心 idea**: 通过掩码引导的身份嵌入关联人物外观和姿态，联合学习RGB/深度/法线三模态使模型理解3D空间交互。

## 方法详解

### 整体框架
输入为包含$N$个人的参考图像$C$、各帧的身份掩码$M^f$、2D骨架$P^f$和相机参数$R^f$。基于Stable Diffusion 1.5和AnimateDiff构建视频扩散模型，包含Denoising UNet和Reference UNet。

### 关键设计

1. **身份特定嵌入学习（ID-Specific Embedding）**:
    - 功能: 为场景中每个人维持跨帧的外观一致性
    - 核心思路: 引入$N$个可学习ID嵌入$E_{query} \in \mathbb{R}^{N \times C}$。对每帧$f$，将掩码$M^f$转换为空间ID嵌入图$E^f \in \mathbb{R}^{H \times W \times C}$——将第$n$个嵌入复制到$M^f(h,w)=n$的所有空间位置。最终$E \in \mathbb{R}^{F \times H \times W \times C}$通过零初始化卷积（ControlNet方式）与噪声潜变量相加: $\widetilde{x}_t = x_t + \text{zero\_conv}(E)$。
    - 设计动机: 零初始化确保多人训练初始化等效于单人模型。SAM2生成的掩码提供可靠的身份追踪。即使人物交换位置，嵌入仍正确关联各身份。灵活支持最多$N$个身份，实际出现更少时忽略未用嵌入。

2. **潜空间结构视频扩散（Latent Structural Video Diffusion）**:
    - 功能: 通过联合预测RGB、深度、法线三模态捕获3D结构信息
    - 核心思路: 不把深度/法线作为输入（实际中逐帧无法获取），而是作为**输出模态**与RGB联合预测。Denoising UNet和Reference UNet都复制conv_in/conv_out/首尾DownBlock/UpBlock为三模态独立路径，中间层共享。训练时用DepthCrafter预测深度、Sapiens预测法线作为监督。三模态用同一时间步$t$、独立噪声、联合损失: $\mathcal{L} = \|v_{rgb} - \hat{v}_{rgb}\|^2 + \|v_{depth} - \hat{v}_{depth}\|^2 + \|v_{normal} - \hat{v}_{normal}\|^2$。
    - 设计动机: 深度提供遮挡和相对距离线索，法线保持物体/衣物形状。联合去噪让模型学习外观-几何耦合动力学，改善人物-物体交互质量。无需显式物体级条件，通过深度/法线推断物体与人的空间关系。

3. **Multi-HumanVid数据集**:
    - 功能: 提供大规模多人交互训练数据
    - 核心思路: 使用Pexels API以交互关键词（如party）查询，通过2D姿态检测筛选（上半身置信度>0.5，主体占比>0.07，人数≤5），收集25K新视频扩展至总计45K。标注流程: Grounding-DINO定位人物→SAM2跟踪掩码→TRAM估计相机→DepthCrafter和Sapiens提供深度/法线。
    - 设计动机: 现有HumanVid仅20K无交互视频，多人交互场景的训练需要专门数据。标注pipeline全自动化可扩展。

### 损失函数 / 训练策略
两阶段训练: Stage1训练全部网络参数（Denoising UNet + ReferenceNet + Pose Guider），batch取决于模态数量。Stage2冻结前者仅训练相机编码器和运动模块。40K+20K迭代，8张A100。

## 实验关键数据

### 主实验

| 方法 | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | FID↓ |
|--------|------|------|----------|------|------|
| MimicMotion | 0.628 | 19.878 | 0.258 | 1042.6 | 59.11 |
| CamAnimate | 0.649 | 19.552 | 0.265 | 982.1 | 54.09 |
| **本文** | **0.691** | **20.685** | **0.233** | **878.2** | **30.57** |

用户研究: 本文方法获91.25%偏好（vs CamAnimate）。

### 消融实验

| 配置 | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | FID↓ |
|------|------|------|------|------|------|
| Baseline (CamAnimate) | 0.649 | 19.552 | 0.265 | 982.1 | 54.09 |
| + ID-embedding | 0.686 | 20.374 | 0.237 | 873.5 | 33.75 |
| + Multi-modality | 0.668 | 20.139 | 0.240 | 907.8 | 47.67 |
| + Both | **0.691** | **20.685** | **0.233** | **878.2** | **30.57** |

模态消融: 仅加深度效果最好(+Depth: FID 30.57)，加法线反而降低(+Normal: FID 60.58)。

### 关键发现
- ID嵌入和多模态结构学习分别有效，组合进一步提升——两个组件互补
- 深度贡献远大于法线：DepthCrafter（从视频模型微调）的时序一致性远优于Sapiens的逐帧法线
- 法线仅在人体区域有效，不完整的监督限制了其贡献
- 交叉身份运动迁移：可在保持外观一致性的同时将一个视频的动作模板迁移到编辑后的角色

## 亮点与洞察
- ID嵌入设计简洁高效：零初始化确保向后兼容，掩码引导自然关联身份与空间位置
- "预测而非输入"几何信息的策略：避免对帧级深度/法线的不切实际输入需求
- 多模态分支共享骨干的设计：在最小参数增量下实现跨模态相关性学习
- Multi-HumanVid数据集的全自动标注流程具有很好的可扩展性

## 局限与展望
- 基于SD 1.5的底层模型限制了视觉质量和运动稳定性
- 法线估计质量不足反而引入噪声，需更好的时序一致法线估计器
- 未在大规模DiT模型（HunyuanVideo, CogVideoX）上实现
- 不包含物体级显式条件，复杂物理交互仍有限

## 相关工作与启发
- **vs Animate Anyone/MagicAnimate**: 仅支持单人，多人场景外观混乱
- **vs CamAnimate**: 增加了相机控制但无多身份机制
- **vs Champ**: 用SMPL + 深度/法线渲染图作为输入条件，但仅单人且需显式几何输入

## 评分
- 新颖性: ⭐⭐⭐⭐ 多身份嵌入+三模态联合学习的组合首次解决多人交互动画
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+用户研究+全面消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 开创多身份人体视频生成方向，数据集有持续价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](../../CVPR2026/video_generation/vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[CVPR 2026\] PLACID: Identity-Preserving Multi-Object Compositing via Video Diffusion with Synthetic Trajectories](../../CVPR2026/video_generation/placid_identity-preserving_multi-object_compositing_via_video_diffusion_with_syn.md)

</div>

<!-- RELATED:END -->

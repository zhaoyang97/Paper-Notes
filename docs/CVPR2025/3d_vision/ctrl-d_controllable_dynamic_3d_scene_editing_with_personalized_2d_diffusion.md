---
title: >-
  [论文解读] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion
description: >-
  [CVPR 2025][3D视觉][动态3D编辑] 通过单张编辑参考图像微调 InstructPix2Pix 模型以"学习"编辑能力，结合两阶段可变形3D高斯优化，实现可控、一致的动态3D场景编辑。 - 动态3D场景编辑是VR/AR、数据增强和内容创作的关键需求，但现有方法面临编辑不一致和可控性差的问题 - Instruct…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "动态3D编辑"
  - "可变形高斯"
  - "InstructPix2Pix"
  - "个性化扩散"
  - "场景编辑"
---

# Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion

**会议**: CVPR 2025  
**arXiv**: [2412.01792](https://arxiv.org/abs/2412.01792)  
**代码**: [项目页面](https://IHe-KaiI.github.io/CTRL-D/)  
**领域**: 3D视觉 / 动态场景编辑  
**关键词**: 动态3D编辑, 可变形高斯, InstructPix2Pix, 个性化扩散, 场景编辑

## 一句话总结

通过单张编辑参考图像微调 InstructPix2Pix 模型以"学习"编辑能力，结合两阶段可变形3D高斯优化，实现可控、一致的动态3D场景编辑。

## 研究背景与动机

- 动态3D场景编辑是VR/AR、数据增强和内容创作的关键需求，但现有方法面临编辑不一致和可控性差的问题
- Instruct 4D-to-4D 等先前工作依赖预训练扩散模型（如原始 IP2P），受限于其编辑骨干的能力，无法精确进行局部编辑
- 动态场景中追踪编辑区域比静态场景困难得多，传统方法依赖噪声差异确定编辑区域不够稳定
- 核心洞察：将复杂的动态场景编辑任务简化为简单的2D图像编辑问题 —— 用户只需编辑一张图像，即可将编辑效果传播到整个动态场景

## 方法详解

### 整体框架

三阶段 pipeline：(1) 用任意2D编辑工具编辑单张参考图像；(2) 使用编辑前后的图像对微调 IP2P 模型获得个性化编辑器；(3) 两阶段优化可变形3D高斯场景。

### 关键设计

1. **个性化 InstructPix2Pix 微调**:
    - 功能：让 IP2P 从单张编辑参考图像"学习"特定编辑能力
    - 核心思路：使用 GPT-4V 生成文本指令 $C_T^{\star}$，引入特殊 token `<V>` 增强特异性；加入 Prior Preservation Loss（受 DreamBooth 启发）保持模型泛化能力
    - 设计动机：原始 IP2P 的编辑能力受限于训练数据分布，个性化微调可让模型直接从参考图像学习编辑区域和风格，无需显式追踪编辑区域
    - 微调损失：$\mathcal{L}_{\text{finetune}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, I, C_T)\|_2^2] + \lambda \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t^{\star}, t, I_d, C_T^{\star})\|_2^2]$
    - 数据增强：通过仿射变换（旋转、平移、剪切）对源图和编辑图进行增强，防止单图微调的过拟合

2. **两阶段动态高斯优化**:
    - 功能：渐进式编辑已训练好的动态3D高斯场景
    - 核心思路：Stage 1 仅优化 canonical space 并进行高斯 densification（冻结变形场）；Stage 2 同时优化变形场和3D高斯，使用编辑图像缓冲区加速收敛
    - 设计动机：分阶段优化可以先建立编辑区域的粗略几何，再通过全局优化实现时间一致性

3. **编辑图像缓冲区 (Edited Image Buffer)**:
    - 功能：加速编辑过程并增强时间一致性
    - 核心思路：每次迭代随机选未编辑的帧，用个性化 IP2P 生成编辑图并加入缓冲区；仅用缓冲区内图像训练3D高斯和变形场（warm-up 阶段）
    - 设计动机：避免每次都从原始帧开始编辑，利用已有编辑结果加速收敛

### 损失函数 / 训练策略

- 场景优化总损失：$\mathcal{L} = (1-\lambda_d)\mathcal{L}_1 + \lambda_d \mathcal{L}_{\text{D-SSIM}} + \lambda_t \mathcal{L}_{\text{temp}}$
- 参数设置：$\lambda_d = 0.2$，$\lambda_t = 0.001$
- 单目场景使用 [Yang et al.] 建模，Stage 1 为前 300 次迭代；多相机场景使用 [Wu et al.]，Stage 1 为前 100 次迭代
- 每 50 次迭代编辑一张图像

## 实验关键数据

### 主实验

| 场景 | 方法 | CLIP Score↑ | Consistency↑ | 时间↓ |
|------|------|-------------|-------------|-------|
| Portrait | Ctrl-D | **27.75** | **0.953** | **60 min** |
| Portrait | IN4D | 27.38 | 0.933 | 2 hours |
| Cat | Ctrl-D | **31.81** | **0.968** | **60 min** |
| Cat | IN4D | 31.72 | 0.964 | 2 hours |
| Steak | Ctrl-D | **28.52** | **0.988** | **40 min** |
| Steak | IN4D | 28.23 | 0.983 | 2 hours |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| w/o 数据增强 | 模糊、帧间不一致 | IP2P 微调过拟合导致编辑不稳定 |
| w/ 数据增强 | 高质量、一致性好 | 仿射变换有效防止过拟合 |
| w/o 编辑缓冲区 | 1000步后仍接近原始场景 | 随机选全帧训练效率低 |
| w/ 编辑缓冲区 | 1000步后成功编辑 | 聚焦已编辑帧加速收敛 |

### 关键发现

- 总时间（微调+优化）不到 IN4D 的一半
- 编辑能力可跨域泛化：用猫图像微调的 IP2P 可应用于人像和全身场景
- 支持文本驱动、图像驱动和风格迁移等多种2D编辑方式

## 亮点与洞察

- 将复杂的4D编辑问题简化为2D编辑问题，极大降低了动态场景编辑门槛
- 个性化 IP2P 可以直接从参考图学习编辑区域，避免了动态场景中困难的区域追踪
- 编辑能力的跨域泛化性证明个性化微调学到了通用的编辑"技能"而非过拟合单个场景

## 局限与展望

- 当动态3D高斯渲染质量差（如运动模糊的手部），编辑结果也会模糊
- 在空白区域添加复杂内容（如给狗添加包）时，多视角一致性仍有问题
- 未来可使用更强的重建骨干和更强大的基础扩散模型

## 相关工作与启发

- InstructPix2Pix → 基础编辑能力；DreamBooth → Prior Preservation 思路
- Deformable 3DGS → 动态场景表示
- Instruct-NeRF2NeRF → Iterative Dataset Update 策略的灵感来源

## 评分

- 新颖性: ⭐⭐⭐⭐ 将动态编辑简化为2D编辑的思路新颖，个性化微调策略实用
- 实验充分度: ⭐⭐⭐⭐ 定性定量全面，消融充分，跨域泛化实验有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pipeline 描述详细
- 价值: ⭐⭐⭐⭐ 实用性强，降低了动态场景编辑门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)
- [\[CVPR 2025\] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture](towards_high-fidelity_3d_talking_avatar_with_personalized_dynamic_texture.md)
- [\[ICLR 2026\] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](../../ICLR2026/3d_vision/color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

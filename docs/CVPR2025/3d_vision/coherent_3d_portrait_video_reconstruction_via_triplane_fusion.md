---
title: >-
  [论文解读] Coherent 3D Portrait Video Reconstruction via Triplane Fusion
description: >-
  [CVPR 2025][3D视觉][3D人像重建] 提出一种基于三平面融合（Triplane Fusion）的方法，将个人化3D先验与逐帧观测融合，在单目RGB视频中同时实现时间一致性和动态外观的忠实重建，用于3D远程呈现。 3D远程呈现（telepresence）是将远处的人以3D形式面对面呈现的核心技术…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D人像重建"
  - "三平面融合"
  - "时间一致性"
  - "远程呈现"
  - "单目视频"
---

# Coherent 3D Portrait Video Reconstruction via Triplane Fusion

**会议**: CVPR 2025  
**arXiv**: [2405.00794](https://arxiv.org/abs/2405.00794)  
**代码**: [https://research.nvidia.com/labs/amri/projects/stable3d](https://research.nvidia.com/labs/amri/projects/stable3d)  
**领域**: 3D视觉  
**关键词**: 3D人像重建, 三平面融合, 时间一致性, 远程呈现, 单目视频

## 一句话总结

提出一种基于三平面融合（Triplane Fusion）的方法，将个人化3D先验与逐帧观测融合，在单目RGB视频中同时实现时间一致性和动态外观的忠实重建，用于3D远程呈现。

## 研究背景与动机

3D远程呈现（telepresence）是将远处的人以3D形式面对面呈现的核心技术。现有方法面临两难困境：

- **逐帧3D重建**（如LP3D）：能忠实捕捉每帧的动态外观（表情、光照），但帧间不一致，侧面输入时出现严重伪影和身份扭曲
- **自驱动重演方法**（如GPAvatar）：从参考图构建规范帧再驱动，时间一致但无法忠实重建实时动态外观（如特定表情、光照变化、舌头等参考帧中不存在的细节）

本文的核心洞察是：**需要同时维持时间一致性和逐帧动态外观的忠实重建**。解决方案是融合式（fusion-based）方法——利用个人化三平面先验的稳定性，同时保留逐帧观测的动态信息。

## 方法详解

### 整体框架

输入单目RGB视频和一张（近）正面参考图。使用预训练冻结的LP3D分别将参考图和每帧输入编码为三平面（triplane）。然后通过两个核心模块：**Triplane Undistorter**去除原始三平面的视角依赖畸变，**Triplane Fuser**将去畸变后的三平面与个人先验三平面融合，生成最终时间一致且保留动态外观的三平面。整个系统仅用3D GAN（Next3D）生成的合成数据训练。

### 关键设计

1. **Triplane Undistorter（去畸变模块）**:
    - 功能：校正侧面输入导致的原始三平面几何畸变
    - 核心思路：基于SPyNet光流架构，以原始三平面$T_{raw}$为源、先验三平面$T_{prior}$为条件，预测去畸变流场$T_{flow}$，通过warp操作得到$T_{undist} = Warp(T_{raw}, T_{flow})$。注意这不是光流对齐，而是以先验为条件的校正warp
    - 设计动机：LP3D在侧面输入时三平面出现方向性畸变和异常激活（如拍摄左侧时三平面左侧出现过强激活），直接通过warp校正比从头生成更有效

2. **Triplane Fuser（融合模块）**:
    - 功能：将去畸变三平面与个人先验融合，恢复遮挡区域并稳定身份
    - 核心思路：基于RVRT（循环视频恢复Transformer）架构，以$T_{undist}$和$T_{prior}$及其各自的可见性三平面$T_{vis}$为输入。显式预测3D可见性图，Fuser在可见区域保留逐帧动态信息、在遮挡区域引入先验三平面的个人化细节（如胎记、纹身）。替换了RVRT的summation skip connection为卷积skip connection，因为三平面畸变的尺度远大于图像去噪
    - 设计动机：不同部位在不同帧被遮挡，正面参考图通常包含完整的双侧面部信息，可有效补偿遮挡

3. **合成动态多视角数据生成**:
    - 功能：生成训练数据，绕过真实3D人像数据稀缺问题
    - 核心思路：利用Next3D（表情可控3D GAN）生成带不同表情的合成3D人像对，并设计肩部旋转增强（通过在体渲染时warp相机射线模拟肩部运动）和颜色空间增强（模拟光照变化）。用冻结LP3D从正面渲染生成伪真值三平面$T_{frontalGT}$作为监督信号
    - 设计动机：Next3D无法控制肩部旋转，通过射线warp在不修改三平面的情况下实现2D渲染中的肩部姿态多样性

### 损失函数 / 训练策略

总损失为四项加权和：

$$L = w_{undist}L_{undist} + w_{vis}L_{vis} + w_{fusion}L_{fusion} + w_{render}L_{render}$$

- $L_{undist}$：去畸变三平面与伪真值三平面的L1损失
- $L_{vis}$：预测可见性三平面与真值可见性的L1损失
- $L_{fusion}$：融合三平面与伪真值的L1损失，遮挡区域加权更高
- $L_{render}$：渲染新视角图像与真值的LPIPS感知损失

Undistorter和Fuser对三个平面（xy/xz/yz）使用3个独立但相同的网络，避免坍缩为2D。

## 实验关键数据

### 主实验

| 方法 | 类型 | Expr↓ | ID↓ | Overall PSNR↑ | Overall LPIPS↓ | NVS PSNR↑ | NVS LPIPS↓ |
|------|------|-------|-----|--------------|----------------|-----------|------------|
| Li et al. | reenact | 0.2657 | 0.2410 | 18.57 | 0.2546 | 18.20 | 0.2624 |
| GPAvatar | reenact | 0.2041 | 0.2074 | 21.95 | 0.2334 | 21.95 | 0.2334 |
| VIVE3D | invert | 0.2900 | 0.3951 | 18.58 | 0.2593 | 18.14 | 0.2710 |
| LP3D | recon | 0.1676 | 0.2154 | 22.33 | 0.2232 | 21.52 | 0.2374 |
| **Ours** | **recon** | **0.1584** | **0.1865** | **22.77** | **0.2189** | **22.44** | **0.2240** |

### 消融实验

| 配置 | Overall PSNR↑ | Overall LPIPS↓ | 输入视角变化↓ | 新视角变化↓ |
|------|--------------|----------------|-------------|-----------|
| LP3D (baseline) | 22.33 | 0.2232 | 高 | 高 |
| Only Fuser | 略低 | 略低 | 中 | 中 |
| Only Undistorter | 中 | 中 | 低 | 低 |
| **U + F (完整)** | **22.77** | **0.2189** | **最低** | **最低** |

### 关键发现

- LP3D严重过拟合输入视角，Overall和NVS质量差距大（PSNR 22.33 vs 21.52），本文方法差距最小（22.77 vs 22.44），证明了时间一致性的提升
- 重演方法（GPAvatar）无法捕捉动态外观细节（如伸舌、特定皱纹），表情误差远高于本方法
- 单独使用Fuser而不先Undistort效果不佳，说明先校正几何畸变再融合的两阶段设计是必要的

## 亮点与洞察

- **问题定义精准**：首次明确提出3D远程呈现中时间一致性和动态外观重建需要同时解决
- **多视角评估协议**设计合理：N×N评分矩阵覆盖所有输入-评估视角组合，避免了单视角评估的过拟合假象
- **纯合成数据训练**即可泛化到真实世界，归功于精心设计的数据增强（肩部旋转、光照变化）

## 局限与展望

- 依赖LP3D作为前端，其性能上限制约了整体质量
- 需要一张近正面参考图，当无法获取高质量正面图时效果可能退化
- 实时性未详细讨论，Undistorter + Fuser的推理速度可能影响实时远程呈现
- 肩部增强通过射线warp实现，可能无法覆盖复杂的身体动作

## 相关工作与启发

- 与LP3D的"逐帧重建"和GPAvatar的"参考驱动"形成互补：fusion思路取两者之长
- Triplane Undistorter使用光流架构做三平面去畸变是巧妙的跨领域迁移
- 可见性三平面（visibility triplane）的显式预测为融合过程提供了空间引导，值得在其他三平面融合任务中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ 融合个人先验与逐帧重建的思路新颖，三平面去畸变+融合的两阶段设计合理
- 实验充分度: ⭐⭐⭐⭐ 提出了新的多视角评估协议，在NeRSemble上与多类方法对比全面；但缺少真实场景定量评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法动机和设计决策论述充分，术语定义严谨
- 价值: ⭐⭐⭐⭐ 对3D远程呈现的实际部署有重要推动，评估协议也有独立贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PERSE: Personalized 3D Generative Avatars from A Single Portrait](perse_personalized_3d_generative_avatars_from_a_single_portrait.md)
- [\[CVPR 2025\] FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video](fluidnexus_3d_fluid_reconstruction_and_prediction_from_a_single_video.md)
- [\[CVPR 2025\] TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features](tritex_learning_texture_from_a_single_mesh_via_triplane_semantic_features.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2026\] EfficientMonoHair: Fast Strand-Level Reconstruction from Monocular Video via Multi-View Direction Fusion](../../CVPR2026/3d_vision/efficientmonohair_fast_strand-level_reconstruction_from_monocular_video_via_mult.md)

</div>

<!-- RELATED:END -->

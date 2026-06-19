---
title: >-
  [论文解读] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos
description: >-
  [ICCV 2025][视频理解][360° video generation] 提出 Argus 模型，首次实现从普通透视视频生成完整 360° 全景视频，通过相机运动模拟、视角对齐帧校准和混合解码三大几何-运动感知技术，在基于扩散模型的框架上让生成的全景视频具备空间一致性和时序连贯性。 360° 视频相比标准视频能提供无…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "360° video generation"
  - "video outpainting"
  - "扩散模型"
  - "panoramic video"
  - "view synthesis"
---

# Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos

**会议**: ICCV 2025  
**arXiv**: [2504.07940](https://arxiv.org/abs/2504.07940)  
**代码**: 无（有项目页面）  
**领域**: 视频理解 / 视频生成  
**关键词**: 360° video generation, video outpainting, diffusion model, panoramic video, view synthesis

## 一句话总结

提出 Argus 模型，首次实现从普通透视视频生成完整 360° 全景视频，通过相机运动模拟、视角对齐帧校准和混合解码三大几何-运动感知技术，在基于扩散模型的框架上让生成的全景视频具备空间一致性和时序连贯性。

## 研究背景与动机

360° 视频相比标准视频能提供无边界的视场角，更完整地呈现动态视觉世界。但现有视频模型只能生成窄视角的标准视频，无法生成全景视频。Video-to-360° 任务面临三大挑战：

**视场角极大扩展**：输入视频仅提供有限视角，需要理解场景的空间布局和物体动态来外推整个场景

**现有 outpainting 方法局限**：训练在窄视角视频上，离输入视角越远生成质量越差

**等距矩形投影的非线性畸变**：物体和空间布局在投影中被扭曲，边界不连续

作者的关键洞察是：互联网上大量 360° 视频是一个相对未被充分利用的数据源，可以从中学习全景先验知识。

## 方法详解

### 整体框架

Argus 基于 Stable Video Diffusion 架构，将问题建模为动态遮罩下的视频外扩问题。给定透视视频，先估计相机位姿将其投影到等距矩形坐标系中，然后以此为条件进行扩散生成。核心组件包括：视频条件化 360° 扩散、相机运动模拟、视角帧对齐和混合解码。

### 关键设计

1. **视频条件化 360° 扩散（Video-Conditioned 360° Diffusion）**:

    - 输入透视视频 $X_{\text{pers}} \in \mathbb{R}^{T \times 3 \times H \times W}$，输出 360° 全景视频 $Y_{\text{equi}} \in \mathbb{R}^{T \times 3 \times H' \times W'}$
    - 将透视视频转换为等距矩形格式 $X_{\text{equi}}$，未映射区域设为黑色
    - 编码器 $\mathcal{E}$ 将两者编码为潜在表示，对 $\mathbf{y}_{\text{equi}}$ 加噪后与 $\mathbf{x}_{\text{equi}}$ 拼接输入去噪网络 $f_\theta$
    - 使用 CLIP 提取图像特征序列通过交叉注意力进行条件化
    - 训练损失引入高度相关的重加权函数：$\lambda(h) = (\frac{1}{2} - |\frac{1}{2} - h|)^2 + \delta$，赤道附近区域权重更高（因为极点区域在等距投影中不成比例地放大）

2. **相机运动模拟（Camera Movement Simulation）**:

    - 从 360° 视频中采样真实视角的训练数据
    - 用线性漂移、振荡和噪声项模拟自然人类运动：
        - $\phi_{\text{yaw}}(k) = \mathcal{N}(0, \eta_y) + a_y \sin(\omega k + \tau_y) + d_y k + \phi_0$
    - 水平和垂直视场角在 $[30°, 120°]$ 间随机选取
    - 数据增强：随机圆周平移（水平旋转保持 360° 属性）

3. **视角帧对齐（View-Based Frame Alignment）**:

    - 核心问题：如何将透视视频投影到等距矩形格式？
    - 若天真地将每帧居中到等距矩形地图中心，模型需隐式学习相机运动，天空等内容在不同帧中出现位置不同，增加学习难度
    - 解决方案：用 SLAM（MegaSaM）估计相对相机位姿，计算相对首帧的欧拉角，将帧投影到共享坐标系
    - 效果：确保等距矩形地图的每个部分跨帧对应大致相同的场景区域（天空始终在顶部，道路始终在底部）

4. **混合解码（Blended Decoding）**:

    - 问题：等距矩形图像左右边缘在图像空间距离远但在场景空间相邻，边界处产生伪影
    - 解码原始潜在表示和 180° 旋转版本，生成两个内容相同但伪影位置不同的输出
    - 基于距离的加权平均融合：$Y_{k,i,j} = h_W(i) \cdot Y_{k,i,j} + (1 - h_W(i)) \cdot Y'_{k,i,j}$
    - 其中 $h_W(x) = 1 - 2|\frac{x}{W} - \frac{1}{2}|$，远离边界的像素权重更高

### 损失函数 / 训练策略

- 基于 EDM 扩散框架，使用 score matching 目标函数
- 从 Stable Video Diffusion-I2V-XL 初始化
- 两阶段训练：先以 384×768 分辨率训练 100K 迭代，再以 512×1024 高质量子集微调 20K 迭代，batch size 16
- 长视频生成：上下文感知训练，交替标准输入和 context-aware 输入
- 数据：从 360-1M 数据集过滤得到约 283,863 视频片段

## 实验关键数据

### 主实验（Video-to-360° 生成）

| 方法 | PSNR↑ | LPIPS↓ | FVD↓ | Motion↑ | Line cons.↑ |
|------|-------|--------|------|---------|-------------|
| PanoDiffusion | 16.44 | 0.4138 | 2649.0 | 0.9426 | 0.6504 |
| **Argus** | **21.83** | **0.2409** | **1228.6** | **0.9802** | **0.8506** |

与视频外扩方法对比（FoV=60°）：

| 方法 | Imaging↑ | Aesthetic↑ | Motion↑ |
|------|----------|------------|---------|
| Be-Your-Outpainter | 0.4014 | 0.3461 | 0.9683 |
| Follow-Your-Canvas | 0.4268 | 0.4750 | 0.9704 |
| **Argus** | **0.4760** | **0.4722** | **0.9816** |

### 消融实验

| 变体 | PSNR↑ | LPIPS↓ | FVD↓ | Imaging↑ |
|------|-------|--------|------|----------|
| 无帧对齐 | 20.42 | 0.3194 | 1349.6 | 0.3816 |
| 无混合解码 | 22.09 | 0.2675 | 1226.3 | 0.4574 |
| **完整模型** | **21.83** | **0.2409** | **1228.6** | **0.4939** |
| VAE 重建上界 | 24.54 | 0.1663 | 121.8 | 0.5272 |

### 关键发现

- 视角帧对齐对整体性能提升至关重要（移除后 Imaging 从 0.4939 降至 0.3816）
- 混合解码显著改善边界一致性，虽然对定量指标影响较小
- Argus 能理解输入视频中的动态信息（如车辆运动）并合理外推，预测轨迹与真实轨迹高度吻合
- 3D 重建验证：生成 panoramic 视频的旋转角平均偏差仅 $(0.22°, 0.30°, 0.34°)$
- 视频外扩方法离输入视角越远质量越差，而 Argus 在全视角范围内保持一致质量

## 亮点与洞察

- **360° 视频作为先验数据源**：打破了只用窄视角视频训练的思路，利用互联网上丰富的 360° 视频学习全景先验
- **几何感知设计**：高度重加权损失、视角对齐、混合解码等设计都体现了对等距矩形投影几何特性的深入理解
- **丰富的下游应用**：视频稳定（不损失视角）、自由视角控制、动态环境映射、交互式 VQA，展现了 360° 视频生成的广阔应用前景

## 局限与展望

- 输出分辨率 512×1024 低于典型 4K 全景视频，解包回透视视图时分辨率进一步下降
- 仍存在物体形状不一致和物理伪影，与 SVD 等基础模型共享的限制
- 推理时需要相机位姿估计，增加了 pipeline 复杂度
- 计算资源限制了更高分辨率和更长视频的生成

## 相关工作与启发

- 与 VidPanos 等视频全景合成方法不同，Argus 能外推超越输入视角的内容
- 混合解码的思路（像素空间而非潜在空间混合）可能适用于其他存在边界不连续的生成任务
- 相机运动模拟策略为从 360° 数据训练提供了系统的数据构造方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统解决 video-to-360° 任务，技术设计有丰富的几何洞察
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+消融+多种下游应用展示
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 开拓了全景视频生成的新方向，下游应用潜力巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[CVPR 2026\] Seeing Beyond 8bits: Subjective and Objective Quality Assessment of HDR-UGC Videos](../../CVPR2026/video_understanding/seeing_beyond_8bits_subjective_and_objective_quality_assessment_of_hdr-ugc_video.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)

</div>

<!-- RELATED:END -->

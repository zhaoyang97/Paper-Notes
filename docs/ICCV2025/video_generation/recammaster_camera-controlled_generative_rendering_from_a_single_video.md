---
title: >-
  [论文解读] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video
description: >-
  [ICCV 2025][视频生成][相机控制] 提出 ReCamMaster，通过帧维度拼接的视频条件注入机制和 UE5 合成的多相机同步数据集，实现从单视频输入以新相机轨迹重新生成视频，显著超越现有方法。 相机运动是影视制作的基础元素，深刻影响观众体验和叙事意图。然而，业余摄影师受限于硬件和技术，难以实现专业级相机运动…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "相机控制"
  - "视频重渲染"
  - "扩散模型"
  - "多相机数据集"
---

# ReCamMaster: Camera-Controlled Generative Rendering from A Single Video

**会议**: ICCV 2025  
**arXiv**: [2503.11647](https://arxiv.org/abs/2503.11647)  
**代码**: [https://github.com/KwaiVGI/ReCamMaster](https://github.com/KwaiVGI/ReCamMaster)  
**领域**: 视频生成  
**关键词**: 相机控制, 视频重渲染, 视频生成, 扩散模型, 多相机数据集

## 一句话总结

提出 ReCamMaster，通过帧维度拼接的视频条件注入机制和 UE5 合成的多相机同步数据集，实现从单视频输入以新相机轨迹重新生成视频，显著超越现有方法。

## 研究背景与动机

相机运动是影视制作的基础元素，深刻影响观众体验和叙事意图。然而，业余摄影师受限于硬件和技术，难以实现专业级相机运动。研究者希望在后期修改视频的相机轨迹，以更好的视角展示动态场景。

现有工作的不足：

**GCD** 开创了相机控制的视频到视频生成，但受限于 Kubric 合成数据的域差距，在真实视频上效果有限

**ReCapture** 需要逐视频优化（LoRA 微调），实用性受限
3. 基于 4D 重建的方法依赖单视频重建精度，质量受限

核心动机：利用预训练文本到视频（T2V）模型的生成能力，通过简洁而强大的视频条件注入机制，实现开放域的相机轨迹修改。

## 方法详解

### 整体框架

ReCamMaster 基于预训练的 T2V 扩散模型（DiT 架构 + 3D VAE + Rectified Flow），以源视频 $V_s$、目标相机轨迹 $\text{cam}_t$ 和文本提示 $p_t$ 为条件，生成目标视频 $V_t$。框架包含三个关键设计：帧维度条件注入、相机位姿编码、以及多任务训练策略。

### 关键设计

1. **帧维度条件注入（Frame Dimension Conditioning）**：将源视频 token 和目标视频 token 沿帧维度拼接，$x_i = [x_s, x_t]_{\text{frame-dim}} \in \mathbb{R}^{b \times 2f \times s \times d}$。无需额外注意力层，源/目标视频在 3D 时空注意力中自然交互。相比通道维度拼接（GCD 等方法）和视图维度拼接，帧维度拼接能在所有 Transformer 块中实现源/目标的全时空交互，保留内容一致性和动态同步。

2. **相机位姿编码**：仅条件化目标相机的外参 $\text{cam}_t \in \mathbb{R}^{f \times 3 \times 4}$（旋转+平移矩阵），不使用源相机参数（因推理时难以精确估计）。通过可学习的相机编码器 $\mathcal{E}_c$（全连接层，12→d）将相机参数注入到每个 Transformer 块的空间注意力输出上：$F_i = F_o + \mathcal{E}_c(\text{cam}_t)$。

3. **多相机同步数据集（Multi-Cam Video）**：使用 Unreal Engine 5 构建 136K 视频、13.6K 动态场景、40 个高质量 3D 环境、122K 不同相机轨迹。精心模拟真实拍摄特征，缩小合成-真实域差距。

### 损失函数 / 训练策略

- 基础损失：Conditional Flow Matching 的速度回归损失 $\mathcal{L}_{LCM}$
- **增强泛化能力**：仅微调相机编码器和 3D 注意力层，冻结其他参数；对条件视频 latent 添加中等噪声（200-500 步），缓解合成域差距
- **多任务统一训练**：以 20% 概率替换所有帧为噪声（T2V 任务），以 20% 概率替换 $f-1$ 帧（I2V 任务），增强内容生成能力

## 实验关键数据

### 主实验

在 WebVid 1000 视频 + 10 种相机轨迹上评估：

| 方法 | FID↓ | FVD↓ | CLIP-T↑ | RotErr↓ | TransErr↓ | Mat.Pix.(K)↑ | FVD-V↓ | CLIP-V↑ |
|------|------|------|---------|---------|-----------|--------------|--------|---------|
| GCD | 72.83 | 367.32 | 32.86 | 2.27 | 5.51 | 639.39 | 365.75 | 85.92 |
| Traj-Attn | 69.21 | 276.06 | 33.43 | 2.18 | 5.32 | 619.13 | 256.30 | 88.65 |
| DaS | 63.25 | 159.60 | 33.05 | 1.45 | 5.59 | 633.53 | 154.25 | 87.33 |
| **ReCamMaster** | **57.10** | **122.74** | **34.53** | **1.22** | **4.85** | **906.03** | **90.38** | **90.36** |

VBench 质量评估：

| 方法 | Aesthetic↑ | Imaging Quality↑ | Temporal Flickering↑ | Subject Consistency↑ | Background Consistency↑ |
|------|-----------|------------------|---------------------|---------------------|------------------------|
| GCD | 38.21 | 41.56 | 95.81 | 88.94 | 92.00 |
| **ReCamMaster** | **42.70** | **53.97** | **97.36** | **92.05** | **93.83** |

### 消融实验

| 视频条件方式 | FID↓ | FVD↓ | Mat.Pix.(K)↑ | FVD-V↓ | CLIP-V↑ |
|------------|------|------|-------------|--------|---------|
| 通道维度拼接 | 74.09 | 187.94 | 521.10 | 148.51 | 84.62 |
| 视图维度拼接 | 80.51 | 194.47 | 573.92 | 177.68 | 83.40 |
| **帧维度拼接** | **57.10** | **122.74** | **906.03** | **90.38** | **90.36** |

训练策略消融：

| 策略 | FID↓ | FVD↓ | Aesthetic↑ | Imaging Quality↑ |
|------|------|------|-----------|------------------|
| Baseline | 66.67 | 171.80 | 40.02 | 51.93 |
| + 添加噪声 | 65.17 | 164.04 | 40.36 | 52.22 |
| + 3D-Attn 微调 | 59.47 | 132.58 | 43.08 | 52.80 |
| + 全部策略 | **57.10** | **122.74** | **42.70** | **53.97** |

### 关键发现

- 帧维度拼接显著优于其他条件注入方式，匹配像素数提升 73%（906K vs 521K）
- 模型还可拓展到视频稳定、超分辨率和视频外拓等应用

## 亮点与洞察

- 帧维度拼接机制简洁优雅，无需额外模块即可利用预训练模型的全部时空注意力实现条件生成
- UE5 数据集的真实拍摄模拟设计是泛化到真实视频的关键
- 多任务训练（T2V/I2V/V2V 统一）是提升内容生成能力的有效策略

## 局限与展望

- 帧维度拼接使输入 token 翻倍，增加计算开销
- 继承预训练 T2V 模型的缺陷（如手部生成效果不佳）
- 仅使用外参条件化，未利用内参信息

## 相关工作与启发

- 条件注入策略对条件生成质量影响巨大，帧维度拼接可作为通用方案推广到其他视频条件生成任务
- 高质量合成数据 + 域适应训练策略可有效弥补真实数据不足

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SpaceTimePilot: Generative Rendering of Dynamic Scenes Across Space and Time](../../CVPR2026/video_generation/spacetimepilot_generative_rendering_of_dynamic_scenes_across_space_and_time.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](../../CVPR2026/video_generation/unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[CVPR 2026\] LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models](../../CVPR2026/video_generation/lavr_scene_latent_conditioned_generative_video_trajectory_re-rendering_using_lar.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)

</div>

<!-- RELATED:END -->

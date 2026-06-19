---
title: >-
  [论文解读] MikuDance: Animating Character Art with Mixed Motion Dynamics
description: >-
  [ICCV 2025][视频理解][Character Animation] 提出 MikuDance，一种基于扩散模型的角色艺术动画系统，通过 Mixed Motion Modeling（将角色运动和 3D 相机运动统一到像素空间表示）和 Mixed-Control Diffusion（在 Reference UNet 中隐式对齐角色形状/尺度与运动引导），实现了复杂角色画作的高动态动画生成。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Character Animation"
  - "扩散模型"
  - "Camera Control"
  - "Mixed Motion"
  - "Image-to-Video"
---

# MikuDance: Animating Character Art with Mixed Motion Dynamics

**会议**: ICCV 2025  
**arXiv**: [2411.08656](https://arxiv.org/abs/2411.08656)  
**代码**: [https://kebii.github.io/MikuDance](https://kebii.github.io/MikuDance)  
**领域**: 视频理解  
**关键词**: Character Animation, diffusion model, Camera Control, Mixed Motion, Image-to-Video

## 一句话总结

提出 MikuDance，一种基于扩散模型的角色艺术动画系统，通过 Mixed Motion Modeling（将角色运动和 3D 相机运动统一到像素空间表示）和 Mixed-Control Diffusion（在 Reference UNet 中隐式对齐角色形状/尺度与运动引导），实现了复杂角色画作的高动态动画生成。

## 研究背景与动机

将静态角色艺术画作动画化在影视、游戏、数字设计领域需求巨大。传统软件（MMD、Live2D）需要专业技能，现有图像到视频方法（Animate Anyone、DISCO）主要针对真实人类，无法直接用于角色画作，面临两大挑战：

**高动态运动引导**：角色画作动画需要同时处理复杂前景角色运动和背景的大尺度相机运动。现有方法仅支持静态背景 + 人体运动，无法建模整个场景的动态

**参考-引导不对齐**：动漫角色有独特的头身比、夸张姿势和多样艺术风格，与运动引导（通常来自真人视频）之间存在严重的尺度和体型差异。显式对齐不可行（角色形态各异），需要隐式对齐

## 方法详解

### 整体框架

MikuDance 基于 SD-1.5，包含：
- **Mixed Motion Modeling**：从驱动视频提取角色 pose（Xpose）和相机位姿（DROID-SLAM），通过 Scene Motion Tracking 将相机运动转化为像素级场景运动表示
- **Mixed-Control Diffusion**：将参考图、参考 pose、所有角色 pose 通过 VAE 编码后拼接作为 Reference UNet 的输入，通过 Motion-Adaptive Normalization 注入场景运动
- **两阶段混合源训练**：先用风格化帧对训练 Reference UNet，再加入 MAN + 时序模块训练

### 关键设计

1. **Scene Motion Tracking (SMT)**：

    - 给定参考图的深度图构建场景点云 $\phi^l \in \mathbb{R}^{N \times 3}$
    - 通过相机-世界坐标变换矩阵 $\mathcal{T}^l$ 和 $\mathcal{Y}^{l+1}$，将点云从第 $l$ 帧投影到第 $l+1$ 帧
    - 计算两帧投影图像间的像素对应关系得到场景运动 $\mathbf{m}^s \in \mathbb{R}^{N \times 2}$
    - 公式：$(z^l - z^{l+1})[\mathbf{m}^s; \mathbf{1}] = \mathcal{K}^l[\phi^l; \mathbf{1}] - \mathcal{K}^{l+1}\mathcal{Y}^{l+1}\mathcal{T}^l[\phi^l; \mathbf{1}]$

   与光流的关键区别：(1) SMT 与驱动视频内容无关，光流是内容相关的；(2) SMT 跟踪 3D 点云，光流跟踪 2D 像素。因此 SMT 提供解耦的相机动态信息。

   设计动机：将 3D 相机运动转化为与角色 pose 同域的 2D 像素表示，统一运动引导。

2. **Mixed-Control Diffusion**：

    - **去掉独立编码器**：与 Animate Anyone 等方法不同，不使用独立的 pose 编码器或 ControlNet
    - 将参考图 + 参考 pose + 所有驱动 pose 全部 VAE 编码后沿通道维拼接，作为 Reference UNet 输入
    - 扩展 Reference UNet 输入卷积层的通道，新增参数用 zero convolution 初始化
    - 参考图还通过 CLIP 图像编码器嵌入，作为两个 UNet 的交叉注意力 key

   设计动机：混合隐式对齐优于显式对齐 + 独立编码器的复杂设计。实验验证此简洁架构效果最好。

3. **Motion-Adaptive Normalization (MAN)**：

    - 受 SPADE 启发，对 Reference UNet 每个下采样块的特征做实例归一化后，用场景运动 $\mathbf{m}^s$ 生成空间自适应的缩放参数 $\gamma^i$ 和偏移参数 $\beta^i$
    - $f^{i'} = \gamma^i_{C,H,W}(\mathbf{m}^s) \frac{f^i_{C,H,W} - \mu^i_C}{\sigma^i_C} + \beta^i_{C,H,W}(\mathbf{m}^s)$
    - $\gamma^i$ 和 $\beta^i$ 具有空间维度，实现像素级场景运动引导

   设计动机：场景运动对动画帧产生全局影响，自适应归一化能有效注入全局运动同时保持局部一致性。

### 损失函数 / 训练策略

**两阶段混合源训练**：

- **第一阶段**：成对视频帧训练，不含 MAN 和时序模块
    - 随机混合风格化帧对：将初始帧沿空间拼接，用 SDXL-Neta 做风格迁移
    - 参考帧随机选择不在目标序列中的帧（模拟推理时参考与驱动不相关）
    - 768×768 分辨率，batch 128，训练 120k 步

- **第二阶段**：加入 MAN + 时序模块（冻结其他参数）
    - 混合 MMD 视频和无角色的相机运动视频
    - 24帧序列，batch 16，训练 60k 步
    - 16×A800 GPU

两个阶段都随机 dropout pose 和场景运动引导（比例 0.2）以增强鲁棒性。推理用 DDIM 20 步。

标准扩散训练损失：$\mathcal{L}_{simple} = \mathbb{E}_{\epsilon, t, c}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2]$

## 实验关键数据

### 主实验 (表格)

**定量对比（100 个 MMD 测试视频）：**

| 方法 | FID ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | L1 ↓ | FID-VID ↓ | FVD ↓ |
|------|-------|--------|--------|---------|------|-----------|-------|
| AniAny | 43.945 | 0.488 | 12.530 | 0.548 | 7.31E-5 | 38.179 | 846.414 |
| AniAny* | 28.833 | 0.526 | 13.610 | 0.517 | 6.23E-5 | 26.764 | 575.304 |
| DISCO | 59.221 | 0.313 | 10.732 | 0.615 | 9.25E-5 | 46.852 | 923.921 |
| MagicPose | 44.258 | 0.424 | 12.357 | 0.554 | 7.77E-5 | 41.347 | 886.691 |
| UniAnimate | 47.328 | 0.417 | 12.074 | 0.571 | 7.93E-5 | 40.924 | 882.245 |
| **MikuDance** | **24.597** | **0.576** | **14.592** | **0.493** | **5.73E-5** | **22.868** | **502.380** |

### 消融实验 (表格)

**关键设计消融：**

| 设置 | FID ↓ | SSIM ↑ | PSNR ↑ | FID-VID ↓ | FVD ↓ |
|------|-------|--------|--------|-----------|-------|
| w/o MIX (独立 Ref UNet + 2 ControlNets) | 27.315 | 0.523 | 14.004 | 24.124 | 541.453 |
| w/o MAN (场景运动直接拼接) | 24.985 | 0.542 | 14.501 | 23.366 | 509.342 |
| w/o SMT (无场景运动) | 25.472 | 0.534 | 14.312 | 23.362 | 517.673 |
| w/ Plücker (Plücker 坐标替代) | 25.918 | 0.538 | 14.261 | 23.471 | 521.853 |
| w/ Flow (光流替代) | 26.141 | 0.516 | 14.088 | 23.079 | 505.533 |
| **MikuDance (完整)** | **24.597** | **0.576** | **14.592** | **22.868** | **502.380** |

**用户研究**：邀请 50 名志愿者对 20 个视频的 4 种方法匿名结果排名，MikuDance 在整体质量、帧质量、时序质量三个维度均大幅领先，>97% 用户更偏好 MikuDance。

### 关键发现

1. **混合控制优于独立编码器**：w/o MIX（2个ControlNet） FID 27.3 vs 完整 24.6，独立处理无法解决角色-引导不对齐
2. **MAN 比直接拼接更优**：MAN 通过空间自适应归一化注入全局运动，FVD 502 vs 509（直接拼接）
3. **SMT 优于 Plücker 和光流**：像素级场景运动表示与角色 pose 域一致，Plücker 坐标有域差距，光流内容相关不可泛化
4. **即使微调到动漫域（AniAny*），现有方法仍然远逊**：AniAny* FVD 575 vs MikuDance 502
5. **高动态场景处理能力独特**：大幅舞蹈动作 + 快速相机移动的组合中，MikuDance 保持高保真动画

## 亮点与洞察

- **统一像素空间运动表示的思路精妙**：将异质的角色 pose（关键点）和相机运动（3D位姿）统一到 2D 像素运动空间
- **"简单就是好"的混合控制设计**：去掉 ControlNet 等复杂架构，直接在 Reference UNet 中混合所有引导信号，反而效果最好
- **实用的两阶段训练策略**：用风格化帧对模拟各种角色风格 + 用相机运动视频学习背景动态
- **3600 个 MMD 动画作为训练数据**：构建了 120k 片段、1020 万帧的 MMD 数据集
- **用户研究压倒性胜出**：97% 偏好率

## 局限与展望

1. 部分生成动画存在背景扭曲和伪影——图像动画中的 3D 不确定性是本质难题
2. SMT 假设场景静态，实际场景中背景物体也可能运动
3. 深度图质量影响场景点云精度，进而影响 SMT 质量
4. 依赖 DROID-SLAM 提取相机位姿，极端运动下可能不准确
5. 长视频生成依赖时序聚合方法拼接，可能产生过渡伪影
6. 训练成本较高（16×A800，总计 180k 步）

## 相关工作与启发

- **Animate Anyone 系列**：建立了 Reference UNet + Denoising UNet 的基础架构，本文在此基础上解决了角色画作的特殊挑战
- **相机控制的视频生成**：Human4DiT、HumanVid 等使用 Plücker 坐标，本文证明像素级表示更优
- **SPADE → MAN**：将语义空间自适应归一化的思想迁移到运动自适应归一化
- 对动漫/游戏角色生成的产业应用价值极大

## 评分

- **新颖性**: ⭐⭐⭐⭐ SMT策略和混合控制设计有新意，统一像素空间运动表示思路巧妙
- **实验充分度**: ⭐⭐⭐⭐ 定量对比全面、消融充分、用户研究有说服力；但缺少其他角色动画特定基准
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示丰富，两大挑战两大方案对应明确
- **价值**: ⭐⭐⭐⭐ 首次在角色画作动画中引入高动态相机运动控制，对动漫/游戏产业有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dynamics: Language-Based Representation for Inferring Rigid-Body Dynamics From Videos](../../CVPR2026/video_understanding/dynamics_language-based_representation_for_inferring_rigid-body_dynamics_from_vi.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](../../CVPR2026/video_understanding/gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)

</div>

<!-- RELATED:END -->

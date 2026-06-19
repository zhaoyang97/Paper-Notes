---
title: >-
  [论文解读] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation
description: >-
  [NeurIPS 2025][3D视觉][embodied AI] EnerVerse 是一个生成式机器人基础模型，通过 chunk-wise 自回归视频扩散 + 稀疏上下文记忆 + 多视角生成先验构建 4D 具身空间，结合 4DGS 数据飞轮缩小 Sim2Real 差距，最终通过策略头将 4D 世界表示转化为物理动作，在 LIBERO 基准上达到 SOTA。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "embodied AI"
  - "扩散模型"
  - "multi-view generation"
  - "robotic manipulation"
  - "4D Gaussian Splatting"
---

# EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation

**会议**: NeurIPS 2025  
**arXiv**: [2501.01895](https://arxiv.org/abs/2501.01895)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: embodied AI, video diffusion, multi-view generation, robotic manipulation, 4D Gaussian Splatting

## 一句话总结

EnerVerse 是一个生成式机器人基础模型，通过 chunk-wise 自回归视频扩散 + 稀疏上下文记忆 + 多视角生成先验构建 4D 具身空间，结合 4DGS 数据飞轮缩小 Sim2Real 差距，最终通过策略头将 4D 世界表示转化为物理动作，在 LIBERO 基准上达到 SOTA。

## 背景与动机

- 视频生成模型在时空想象力方面取得重大进展，自然可联想到将其用于机器人动作规划
- 已有方法简单地将通用视频生成模型适配到机器人任务，忽略了 2D 视频表示空间与 3D 机器人环境之间的巨大鸿沟
- 多视角观测对机器人操作至关重要（解决遮挡和运动歧义），但多相机标定采集成本高昂
- Sim2Real 差距仍是模拟数据大规模应用的核心瓶颈

## 核心问题

如何构建一个既能生成高质量 4D 具身空间又能直接转化为物理动作的统一框架，并解决多视角数据稀缺和 Sim2Real 差距问题？

## 方法详解

### 1. Chunk-wise 自回归视频扩散

定义未来空间最小单元为 chunk。模型反复预测下一个 chunk 扩展空间。训练时优化去噪损失：

$$\min_{\theta} \mathbb{E}_{t, \mathbf{z}, \boldsymbol{\epsilon}} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{z}_t^{1:M}, \mathbf{o}_t^{1:K}, t)\|_2^2$$

推理时新生成的去噪帧成为下一迭代的干净输入，检测到 EOS 帧时终止。采用 v-prediction。

### 2. 稀疏记忆机制

训练时用稀疏采样帧（丢弃约 80%）作为上下文，而非连续帧。好处：
- 减少冗余，促进模型学习更深层的 chunk 预测能力
- 增强对分布外（OOD）场景的鲁棒性
- 推理时通过滑动窗口平滑过渡，节省 GPU 显存

消融实验：无稀疏记忆时 LIBERO-Long 仅 30.8 分 vs 有稀疏记忆 73 分。

### 3. 多视角扩散生成

将单视角扩展为多视角视频生成：
- 用射线方向图（ray direction map）编码相机内外参
- 跨视角注意力保证几何一致性
- 时间注意力捕捉场景动态

预训练在多视角数据上建立 3D 先验，推理时单相机+深度 warp 即可生成辅助视角。

### 4. EnerVerse-D 数据飞轮

结合生成模型与 4D Gaussian Splatting：
1. 从稀疏真实观测+生成模型补齐多视角视频
2. 用 4DGS 重建 4D 场景，渲染高精度图像
3. 渲染图像反馈给生成模型继续优化，形成迭代循环

### 5. EnerVerse-A 策略头

从 UNet 中间层第一次去噪步提取视觉特征 E，缓存后接 DiT 动作头。预测 action chunk（tau 步 x 7 维 delta pose）。单 RTX 4090 约 280ms 推理 8 步动作。

## 实验关键数据

### LIBERO 基准

| 模型 | 视觉输入 | Spatial | Object | Goal | Long | Avg |
|------|---------|---------|--------|------|------|-----|
| Diffusion Policy | S-RGB | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA | S-RGB | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| MAIL | S-RGB x2 | 76.0 | 90.0 | 82.0 | 78.0 | 81.5 |
| **EnerVerse** | S-RGB | **92.1** | 93.2 | 78.1 | 73.0 | 84.1 |
| **EnerVerse** | RGB+2Render | 91.2 | **97.7** | **85.0** | **80.0** | **88.5** |

### CALVIN (ABC -> D)

| 方法 | 输入 | 1 | 2 | 3 | 4 | 5 | Avg Len |
|------|------|---|---|---|---|---|---------|
| RoboFlamingo | S-RGB, G-RGB | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| GR-1 | S-RGB, G-RGB, P | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| **EnerVerse** | S-RGB | 90.8 | 73.0 | 57.3 | 43.7 | 35.6 | 3.00 |

### 训练策略消融（LIBERO-Spatial）

| 策略 | 成功率 |
|------|--------|
| 从头训练 | Failed |
| 加载通用预训练 | 79 |
| 单阶段联合训练 | 86.3 |
| **两阶段微调** | **92.1** |

## 亮点

- Chunk-wise 自回归+稀疏记忆的组合实现了理论上无限长序列生成
- 多视角扩散先验使单相机部署即可受益于 3D 理解能力
- 4DGS 数据飞轮优雅地解决了 Sim2Real 差距问题
- 统一框架：同一 backbone 同时支持视频生成和动作预测

## 局限与展望

- 视频生成不可避免产生 artifact，高动态机器人场景尤为明显
- 渲染视角目前靠启发式设定，未集成 Next-Best-View 方法
- 视频生成质量与控制成功率之间的关系理解不够深入
- 数据飞轮需要离线运行，未实现在线自适应

## 与相关工作的对比

- vs **AVID**: 简单适配 DynamicCrafter，缺少 3D 先验；EnerVerse 多视角预训练提供空间理解
- vs **Diffusion Policy**: 直接动作学习，无视频生成先验；EnerVerse 利用视频想象力增强策略
- vs **OpenVLA**: 7B VLA 模型，EnerVerse 以更小模型超越其性能
- vs **GR-2**: 同为视频预训练，但 GR-2 停留在 2D；EnerVerse 扩展到 4D

## 启发与关联

- 视频生成作为机器人策略学习的预训练任务，是一个有前景的范式
- 4DGS 数据飞轮的思路可推广到其他需要跨域数据增强的领域
- 单相机+深度 warp 生成多视角是一种实用的部署策略

## 评分

- ⭐ 新颖性: 4/5 — 4D 具身空间生成框架设计完整，数据飞轮有创意
- ⭐ 实验充分度: 4.5/5 — LIBERO/CALVIN/真实世界全面验证，消融详尽
- ⭐ 写作质量: 3.5/5 — 内容丰富但略显冗长，核心贡献可更突出
- ⭐ 价值: 4/5 — 为具身智能提供了视频生成+策略的统一范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[ICML 2025\] SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](../../ICML2025/3d_vision/se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)
- [\[NeurIPS 2025\] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](../../ICCV2025/3d_vision/robopearls_editable_video_simulation_for_robot_manipulation.md)

</div>

<!-- RELATED:END -->

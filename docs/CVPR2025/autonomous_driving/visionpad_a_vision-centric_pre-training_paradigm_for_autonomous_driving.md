---
title: >-
  [论文解读] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving
description: >-
  [CVPR 2025][自动驾驶][自监督预训练] 本文提出 VisionPAD，一种纯视觉自监督预训练框架，用基于锚点的 3D 高斯溅射替代体积渲染重建多视角图像，并引入自监督体素速度估计和多帧光度一致性约束来学习运动线索和 3D 几何信息，完全不依赖 LiDAR 深度监督，在 3D 检测、占用预测和地图分割三个下游任务上显著超越现有预训练方法。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "自监督预训练"
  - "3D高斯溅射"
  - "体素速度估计"
  - "光度一致性"
  - "自动驾驶感知"
---

# VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving

**会议**: CVPR 2025  
**arXiv**: [2411.14716](https://arxiv.org/abs/2411.14716)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 自监督预训练, 3D高斯溅射, 体素速度估计, 光度一致性, 自动驾驶感知

## 一句话总结

本文提出 VisionPAD，一种纯视觉自监督预训练框架，用基于锚点的 3D 高斯溅射替代体积渲染重建多视角图像，并引入自监督体素速度估计和多帧光度一致性约束来学习运动线索和 3D 几何信息，完全不依赖 LiDAR 深度监督，在 3D 检测、占用预测和地图分割三个下游任务上显著超越现有预训练方法。

## 研究背景与动机

**领域现状**：视觉中心的自动驾驶感知方法从多视角图像提取 BEV 和占用特征，已在各类下游任务中表现出色。预训练是扩展下游应用的关键策略，但高质量 3D 标注（占用、3D 框等）的采集成本极高。

**现有痛点**：最近的渲染式预训练方法（如 UniPAD）使用体积渲染重建多视角深度和图像，但严重依赖 LiDAR 投影的显式深度监督来学习 3D 几何。仅用图像监督时 UniPAD 效果不佳（NDS 甚至下降 0.2），限制了其在纯相机系统中的应用。此外，体积渲染每次迭代只能采样有限数量的光线，难以重建高分辨率图像中的细粒度细节。

**核心矛盾**：纯视觉预训练需要同时学习外观、3D 几何和运动信息，但缺少显式深度监督时从 RGB 图像中推断这些信息极为困难。体积渲染的计算成本对分辨率敏感，限制了监督信号的丰富度。

**本文目标**：设计一种仅依赖多帧多视角图像即可有效学习 3D 几何和运动表示的预训练框架。

**切入角度**：3D 高斯溅射基于 splat 光栅化，计算成本对分辨率不敏感，能渲染更高分辨率图像提供更丰富的监督信号。体素速度可以通过时间一致性自监督学习，光度一致性可以通过渲染深度和相对位姿进行跨帧约束。

**核心 idea**：用 3D 高斯溅射替代体积渲染进行图像重建预训练，通过体素速度预测 + warp 实现自监督运动学习，通过光度一致性损失实现自监督几何学习，三者协同构成完全无需深度标注的预训练方案。

## 方法详解

### 整体框架

VisionPAD 以通用视觉感知网络为 backbone，输入多帧多视角图像，通过 2D 特征提取和视角变换生成体素特征 $\mathbf{V} \in \mathbb{R}^{X \times Y \times Z \times C}$。预训练包含四个模块：(1) 体素构建模块提取 3D 体积表示；(2) 3D-GS 解码器将体素特征转换为高斯原语并渲染当前帧多视角图像和深度；(3) 体素速度估计模块预测逐体素速度并 warp 到邻近帧，用邻近帧图像自监督；(4) 光度一致性模块利用渲染深度和相对位姿进行跨帧投影约束。

### 关键设计

1. **基于锚点的 3D-GS 解码器**:

    - 功能：从体素特征高效渲染高分辨率多视角图像和深度图
    - 核心思路：每个体素中心作为锚点，通过 MLP 预测多个高斯原语的属性（偏移量、球谐系数、不透明度、尺度、旋转）。渲染公式 $\mathbf{C}(p) = \sum_{i \in K} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_i)$，深度图类似地通过 alpha-blending 距离值获得 $\mathbf{D}(p) = \sum_{i \in K} d_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$。实现高斯过滤：用 tanh 预测不透明度，丢弃 $<0$ 的低置信高斯以减低计算量
    - 设计动机：体积渲染需要沿光线采样、分辨率敏感；3D-GS 基于光栅化并行投影，分辨率影响小，同等计算预算下能渲染更高分辨率图像。消融实验证明仅替换为 3D-GS 就带来 NDS 提升，高斯过滤进一步提升 0.6 NDS

2. **自监督体素速度估计**:

    - 功能：在不使用任何运动标注的情况下学习逐体素运动信息，在物体表示中编码动态/静态区别
    - 核心思路：在体素特征后附加辅助速度头预测世界坐标系下的逐体素速度向量，利用帧间时间间隔近似体素流（速度 × 时间差），通过 GridSample 将当前帧体素 warp 到相邻帧位置。再用 3D-GS 解码器渲染邻帧多视角图像，以对应邻帧真实图像为监督。关键是反向传播时仅更新速度头参数，引导网络专注学习判别性运动特征
    - 设计动机：传统方法依赖 LiDAR 获取运动信息。该设计利用"如果速度预测正确，warp 后的体素应能重建邻帧图像"这一自洽性质进行自监督。消融显示加入速度估计后 mAP 提升 1.2 点

3. **多帧光度一致性**:

    - 功能：利用渲染深度图和已知相机位姿实现自监督 3D 几何学习
    - 核心思路：借鉴自监督深度估计的思想，用当前帧渲染的深度图 $\mathbf{D}_t = \text{3DGS}(\mathbf{V}_t, \mathbf{K}_t, \mathbf{T}_t)$ 将邻帧图像 $\mathbf{I}_{t'}$ 重投影到当前帧视角 $\mathbf{I}_{t' \to t} = \mathbf{I}_{t'}\langle \text{proj}(\mathbf{D}_t, \mathbf{T}_{t \to t'}, \mathbf{K}) \rangle$。光度一致性损失结合 SSIM 和 L1：$\mathcal{L}_{pc} = \alpha(1 - \text{SSIM}(\mathbf{I}_t, \mathbf{I}_{t' \to t})) + (1-\alpha)\|\mathbf{I}_t - \mathbf{I}_{t' \to t}\|_1$
    - 设计动机：深度图正确才能使重投影图像与目标帧一致，这一约束迫使模型学习精确的 3D 几何。消融证明这是最大贡献组件（NDS +2.4, mAP +4.4）

### 损失函数 / 训练策略

总预训练损失 $\mathcal{L} = \omega_1 \mathcal{L}_{img} + \omega_2 \mathcal{L}_{vel} + \omega_3 \mathcal{L}_{pc}$，其中 $\omega_1=0.5, \omega_2=1, \omega_3=1$。$\mathcal{L}_{img}$ 和 $\mathcal{L}_{vel}$ 均为 L1 重建损失。预训练 12 epoch，AdamW 优化器，学习率 $2 \times 10^{-4}$，batch size 4。微调阶段使用官方下游模型配置不做修改。数据增强包含随机缩放/旋转和部分输入遮罩。

## 实验关键数据

### 主实验

3D 目标检测（nuScenes val）：

| 方法 | 预训练模态 | NDS↑ | mAP↑ |
|------|-----------|------|------|
| UVTR (baseline) | - | 48.8 | 39.2 |
| UVTR + UniPAD (C only) | C | 48.6 (-0.2) | 40.5 (+0.7) |
| UVTR + UniPAD (C+L) | C+L | 50.2 (+1.4) | 42.8 (+3.6) |
| **UVTR + VisionPAD** | **C only** | **49.7 (+0.9)** | **41.2 (+2.0)** |
| **UVTR + VisionPAD** | **C+L** | **50.4 (+1.6)** | **43.1 (+3.9)** |

语义占用预测（Occ3D val）/ 地图分割：

| 方法 | 占用 mIoU↑ | 地图 Lane IoU↑ |
|------|-----------|-------------|
| UVTR (baseline) | 30.1 | 15.0 |
| UVTR + UniPAD | 31.0 (+0.9) | 16.3 (+1.3) |
| **UVTR + VisionPAD** | **35.4 (+5.4)** | **20.4 (+5.4)** |

### 消融实验

| 配置 | NDS | mAP | 说明 |
|------|-----|-----|------|
| UVTR baseline | 22.8 | 19.4 | - |
| + UniPAD (Vol. Rend.) | 22.3 (-0.5) | 18.3 (-1.1) | 体积渲染纯图像监督反而有害 |
| + 3DGS 解码器 | 22.8 (+0.0) | 18.2 (-1.2) | 替换为 3DGS |
| + 高斯过滤 | 23.4 (+0.6) | 18.9 (-0.5) | 过滤低不透明度高斯 |
| + 速度估计 | 23.6 (+0.8) | 20.1 (+0.7) | 运动线索 |
| + 光度一致性 | 26.0 (+3.2) | 24.5 (+5.1) | 最大贡献组件 |
| **Full VisionPAD** | **27.3 (+4.5)** | **26.5 (+7.1)** | 所有组件协同 |

### 关键发现

- 光度一致性是最重要的组件（单独贡献 NDS +2.4, mAP +4.4），证明跨帧几何约束对纯视觉预训练至关重要
- 纯图像监督下 UniPAD 的体积渲染预训练反而导致性能下降（NDS -0.5），而 VisionPAD 实现了 +4.5 NDS 的显著提升
- 数据效率实验表明，在仅使用 25% 微调数据时 VisionPAD 的优势更加明显（+6 mAP），证明预训练在数据稀缺场景下价值更大
- 每体素 1 个锚点即可，增加到 2/3/4 个锚点反而性能下降

## 亮点与洞察

- 首次将 3D-GS 应用于自动驾驶预训练，利用其分辨率不敏感的特性突破了体积渲染的分辨率瓶颈
- 体素速度估计的自监督设计非常巧妙——"如果速度正确则 warp 后应能重建邻帧"的自洽性提供了免费的运动监督信号，且只更新速度头参数避免干扰主干表示学习
- 光度一致性从自监督深度估计领域迁移到预训练中是自然且有效的结合——渲染深度图本身就是预训练的副产品

## 局限与展望

- 实验仅在 nuScenes 数据集上验证，缺乏跨数据集泛化性验证
- 光度一致性假设场景静态，动态物体区域可能引入错误梯度（虽然速度估计可部分缓解）
- 3D-GS 解码器引入额外的 MLP 和参数用于 Gaussian primitive 预测，增加了预训练阶段的内存开销
- 未来方向：(1) 结合时序预测（类似 ViDAR）进一步利用运动信息；(2) 扩展到更大规模无标注驾驶视频数据进行预训练；(3) 探索处理动态物体的光度一致性变体

## 相关工作与启发

- **vs UniPAD**：UniPAD 依赖 LiDAR 深度监督的体积渲染，纯图像模式效果差。VisionPAD 用 3D-GS + 自监督几何约束完全消除了对 LiDAR 的依赖，同等设置下 mAP 优 +2.5
- **vs ViDAR**：ViDAR 使用 Transformer 预测未来帧并渲染深度（仍需 LiDAR 监督）。VisionPAD 的体素速度估计从不同角度引入时序信息，且完全自监督
- **vs 自监督深度估计**：本文将 Monodepth 风格的光度一致性从独立深度模型推广到感知预训练框架中，是一种优雅的跨领域技术迁移

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 3D-GS 用于预训练、自监督速度估计和光度一致性的组合是新颖的，但各个组件有已知先例
- **实验充分度**: ⭐⭐⭐⭐ — 三个下游任务、详细消融、数据效率分析全面，但仅一个数据集
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，消融逻辑递进，与 UniPAD 的对比贯穿全文
- **价值**: ⭐⭐⭐⭐⭐ — 消除自动驾驶预训练对 LiDAR 的依赖有重要实际意义，方法即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](../../CVPR2026/autonomous_driving/dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[CVPR 2025\] UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)
- [\[CVPR 2025\] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](spatiotemporal_decoupling_for_efficient_vision-based_occupancy_forecasting.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)

</div>

<!-- RELATED:END -->

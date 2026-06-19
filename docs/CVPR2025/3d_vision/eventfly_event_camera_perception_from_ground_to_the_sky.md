---
title: >-
  [论文解读] EventFly: Event Camera Perception from Ground to the Sky
description: >-
  [CVPR 2025][3D视觉][事件相机] EventFly 提出了首个事件相机跨平台域适应框架，通过事件激活先验（EAP）识别高激活区域、EventBlend 混合源/目标域事件数据、EventMatch 双判别器对齐特征分布，在车辆→无人机→四足机器人三个平台间的语义分割任务上，相比 source-only 训练平均提升准确率 23.8%、mIoU 77.1%。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "事件相机"
  - "跨平台域适应"
  - "语义分割"
  - "数据混合"
  - "对抗训练"
---

# EventFly: Event Camera Perception from Ground to the Sky

**会议**: CVPR 2025  
**arXiv**: [2503.19916](https://arxiv.org/abs/2503.19916)  
**代码**: 有（[https://event-fly.github.io](https://event-fly.github.io)）  
**领域**: 3D视觉  
**关键词**: 事件相机, 跨平台域适应, 语义分割, 数据混合, 对抗训练

## 一句话总结
EventFly 提出了首个事件相机跨平台域适应框架，通过事件激活先验（EAP）识别高激活区域、EventBlend 混合源/目标域事件数据、EventMatch 双判别器对齐特征分布，在车辆→无人机→四足机器人三个平台间的语义分割任务上，相比 source-only 训练平均提升准确率 23.8%、mIoU 77.1%。

## 研究背景与动机

1. **领域现状**：事件相机凭借异步工作、高时间分辨率和高动态范围的特性，在自动驾驶、空中导航和机器人感知等高速动态环境中展现了巨大优势。然而，现有事件相机感知数据集和方法几乎全部集中在车载平台（15 个公共数据集中有 12 个来自地面车辆），无人机和四足机器人等平台的数据极其有限。

2. **现有痛点**：不同机器人平台有着截然不同的运动模式、视角和环境交互：车辆捕获道路和障碍物，无人机俯视地面特征，四足机器人视角不稳定且贴近地面。这些差异导致事件数据的空间-时间激活模式完全不同。传统帧图像的域适应方法（AdaptSegNet、DACS、MIC 等）无法处理事件数据独特的时空特性。

3. **核心矛盾**：事件相机的感知模型通常在单一平台（主要是车辆）上训练，但要部署到无人机或四足机器人等新平台时，数据分布差异导致性能严重下降。同时，为每个新平台标注大量数据成本过高。

4. **本文目标** 设计一个专门针对事件相机数据特性的跨平台域适应框架，使得在一个平台训练的感知模型可以有效迁移到其他平台。

5. **切入角度**：不同平台的事件数据虽然整体分布差异大，但在特定空间区域存在激活模式的重叠——利用这些共享的高激活区域作为域适应的桥梁。

6. **核心 idea**：通过"事件激活先验"识别跨平台共享的高激活区域，用基于激活相似度的数据混合构建中间域，再用双判别器对齐源域-中间域-目标域特征，实现渐进式跨平台适应。

## 方法详解

### 整体框架
EventFly 的输入是有标签的源域事件体素栅格（如车辆）和无标签的目标域事件体素栅格（如无人机）。整体流程：（1）EAP 计算目标域的聚合密度图，识别高激活区域；（2）EventBlend 基于源样本和目标聚合密度图的相似度，生成混合事件体素栅格和对应标签；（3）源、目标、混合三路数据通过共享特征提取器，EventMatch 的两个判别器分别对齐源-混合和混合-目标特征。

### 关键设计

1. **事件激活先验 EAP（Event Activation Prior）**:

    - 功能：识别目标域的高激活区域并引导模型在这些区域产生高置信度预测。
    - 核心思路：对目标域的每个子区域 $\mathbf{S}$，EAP 鼓励最小化条件熵 $H(y_\mathbf{S} | \mathbf{V}_\mathbf{S}, \mathbf{S})$。通过最大熵原理将约束 $\mathbb{E}_\theta[H(\mathbf{V_S}, y_\mathbf{S} | \mathbf{S})] \leq c$ 转化为参数先验 $P(\theta) \propto \exp(-\lambda H(y_\mathbf{S} | \mathbf{V_S}, \mathbf{S}))$，融入 MAP 训练目标 $C(\theta) = \mathcal{L}(\theta) - \lambda H_{\text{emp}}(y|\mathbf{V}, \mathbf{S})$。直觉是：不同平台的事件相机在特定区域有持续的高激活模式（车辆的下方=道路，无人机的上方=天空），这些区域的语义模式一致性高，适合低熵正则化。
    - 设计动机：传统熵最小化对所有区域一视同仁，但事件数据中大量区域无激活（无信息），在这些区域强制低熵反而引入噪声。EAP 聚焦于有事件活动的信息丰富区域，使正则化更有针对性。

2. **EventBlend 跨平台数据混合**:

    - 功能：构建融合源域和目标域特征的中间域数据。
    - 核心思路：第一步，计算每个源样本的密度图 $\mathbf{D}_i^\mathbf{v}(\mu,\nu) = \sum_{t=1}^T |\mathbf{V}_i^\mathbf{v}(t,\mu,\nu)|$ 和目标域的聚合密度图 $\tilde{\mathbf{D}}^\mathbf{d}$；第二步，计算逐像素相似度 $\mathbf{SIM}_i(\mu,\nu) = 1 - |\mathbf{D}_i^\mathbf{v}(\mu,\nu) - \tilde{\mathbf{D}}^\mathbf{d}(\mu,\nu)|$，只在至少一方有激活的位置计算；第三步，用阈值 $\tau$ 生成二值掩码 $\mathcal{M}_i$，相似度高的区域保留源域（有标签），相似度低的替换为目标域（需要适应）；第四步，标签通过源域真值+目标域伪标签（mean teacher）混合。
    - 设计动机：与 CutMix/ClassMix 等通用混合策略不同，EventBlend 的混合由事件激活的物理特性驱动——在两个平台事件模式相似的区域保留可靠的源域标注，在差异大的区域引入目标域数据促进适应。

3. **EventMatch 双判别器对齐**:

    - 功能：在特征层面对齐源域、目标域和混合域的分布。
    - 核心思路：使用两个全卷积判别器 $\sigma_1$ 和 $\sigma_2$：$\sigma_1$ 负责对齐源域和混合域特征（保持源域的可靠性），$\sigma_2$ 负责将混合域特征软对齐到目标域（增强目标域适应性，聚焦于 EAP 识别的高激活区域）。这种分层设计让混合域特征成为源域和目标域之间的中间桥梁——既保留源域的监督信号可靠性，又逐步向目标域分布靠拢。
    - 设计动机：单判别器直接对齐源-目标域在事件数据上效果不佳，因为两个平台的激活模式差异太大。通过混合域作为中间跳板，分两步渐进对齐，每步的分布差距更小，对抗训练更稳定。

### 损失函数 / 训练策略
总损失包括：源域有监督的交叉熵损失 $\mathcal{L}(\theta)$；EAP 熵正则化 $-\lambda H_{\text{emp}}$（仅在高激活区域）；混合域的交叉熵损失（使用混合标签）；$\sigma_1$ 的源-混合对抗损失；$\sigma_2$ 的混合-目标对抗损失。目标域伪标签通过 mean teacher 在线生成或离线预计算。

## 实验关键数据

### 主实验（车辆→无人机适应）

| 方法 | Acc | mAcc | mIoU | fIoU |
|------|-----|------|------|------|
| Source-Only | 43.69 | 33.81 | 15.04 | 11.81 |
| AdaptSegNet | 49.14 | 35.38 | 21.16 | 12.15 |
| DACS | 59.81 | 42.01 | 27.07 | 16.14 |
| MIC | 63.11 | 45.60 | 28.87 | 17.46 |
| PLSR | 64.61 | 45.93 | 29.69 | 17.99 |
| **EventFly** | **69.17** | **48.20** | **32.67** | **20.01** |
| Target (上界) | 79.57 | 52.25 | 42.90 | — |

### 消融（基于报告的增益分析）

| 配置 | Acc 提升 | mIoU 提升 | 说明 |
|------|---------|----------|------|
| Source-Only → EventFly | +25.48 | +17.63 | 完整框架 vs 无适应 |
| 相比最佳基线 PLSR | +4.56 | +2.98 | EventFly 的额外收益 |
| 相比 MIC | +6.06 | +3.80 | 超越上下文引导方法 |

### 关键发现
- EventFly 相比 Source-Only 提升极为显著（Acc +25.5%, mIoU +17.6），证明跨平台域差异对事件相机感知的影响非常严重。
- 相比现有帧图像域适应方法的最佳结果 PLSR，EventFly 仍有 4.56% Acc 和 2.98% mIoU 的提升，说明事件数据特有的适应策略是必要的。
- fence、person、sign 等小目标类别的适应最难（mIoU 仍然很低），这些类别在无人机俯视角下的外观与车辆前视差异最大。
- Target 上界（79.57% Acc, 42.90% mIoU）与 EventFly 之间仍有较大差距（~10% mIoU），说明跨平台事件域适应仍是开放挑战。

## 亮点与洞察
- **EAP（事件激活先验）**是最有洞察力的设计——它利用了事件相机数据独有的"稀疏激活"特性，只在有事件活动的区域做正则化，比全图熵最小化更合理。这个思路可以迁移到其他稀疏数据模态（如 LiDAR、雷达）的域适应中。
- **EventBlend 的物理驱动数据混合**优于随机混合——混合掩码由两域事件激活模式的物理相似度决定，而非随机裁剪。这种"让数据特性指导数据增强"的哲学可以迁移到其他域适应任务。
- **EXPo benchmark** 是首个覆盖车辆/无人机/四足机器人三平台的大规模事件感知数据集（~90K 样本），填补了重要的评测空白。
- 首次系统性地定义和研究了"事件相机跨平台适应"问题，建立了完整的问题形式化和评测体系。

## 局限与展望
- 各平台的事件相机型号可能不同（分辨率、阈值 C 等），本文未明确讨论传感器差异的影响。
- EventBlend 的阈值 $\tau$ 需要调优，不同平台对的最优值可能不同。
- 目前只做了语义分割，检测和深度估计等其他密集预测任务的跨平台适应有待探索。
- 三个平台的两两组合给了 6 个方向，但难度差异很大（车→无人机 vs 无人机→四足），可能需要不对称的适应策略。
- 可以考虑利用事件数据的时间维度信息（如运动模式差异）进一步增强跨平台适应。

## 相关工作与启发
- **vs Ev-Transfer/ESS**: 这些方法做的是 RGB→事件的跨模态域适应，而 EventFly 做的是事件→事件的跨平台域适应，关注的域差异来源完全不同（模态差异 vs 平台差异）。
- **vs DACS/MIC**: 这些通用域适应方法在帧图像上设计，未考虑事件数据的稀疏激活特性。EventFly 的 EAP 和 EventBlend 专门利用了事件数据的激活模式，性能更好。
- **vs HPL-ESS**: 同样做事件域适应但限于帧→事件迁移，EventFly 首次考虑跨机器人平台的事件→事件适应。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义跨平台事件域适应问题，EAP 和 EventBlend 的设计利用了事件数据特性
- 实验充分度: ⭐⭐⭐⭐ EXPo 大规模 benchmark 全面，6 个适应方向对比多种基线
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述系统完整，但公式符号较多
- 价值: ⭐⭐⭐⭐ 开创了事件相机跨平台适应这一新方向，benchmark 对后续研究有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [\[CVPR 2026\] Event Structural Valley: A Unified Theoretical and Practical Framework for Event Camera Autofocus](../../CVPR2026/3d_vision/event_structural_valley_a_unified_theoretical_and_practical_framework_for_event_.md)
- [\[CVPR 2026\] Unsupervised 3D Motion Estimation Using Event Camera](../../CVPR2026/3d_vision/unsupervised_3d_motion_estimation_using_event_camera.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)

</div>

<!-- RELATED:END -->

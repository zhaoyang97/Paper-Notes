---
title: >-
  [论文解读] Seurat: From Moving Points to Depth
description: >-
  [CVPR 2025][3D视觉][点轨迹深度估计] 本文提出 Seurat，一种基于 2D 点轨迹的单目视频深度估计方法，通过空间和时序 Transformer 分析跟踪点的运动模式来推断深度随时间的变化，仅在合成数据上训练即可实现零样本泛化到真实场景。 领域现状：单目深度估计（MDE）近年来借助大规模训练取得了显著进展…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "点轨迹深度估计"
  - "单目视频深度"
  - "Transformer"
  - "零样本泛化"
  - "时序深度比"
---

# Seurat: From Moving Points to Depth

**会议**: CVPR 2025  
**arXiv**: [2504.14687](https://arxiv.org/abs/2504.14687)  
**代码**: [https://seurat-cvpr.github.io](https://seurat-cvpr.github.io)  
**领域**: 3D视觉 / 深度估计  
**关键词**: 点轨迹深度估计, 单目视频深度, Transformer, 零样本泛化, 时序深度比

## 一句话总结

本文提出 Seurat，一种基于 2D 点轨迹的单目视频深度估计方法，通过空间和时序 Transformer 分析跟踪点的运动模式来推断深度随时间的变化，仅在合成数据上训练即可实现零样本泛化到真实场景。

## 研究背景与动机

**领域现状**：单目深度估计（MDE）近年来借助大规模训练取得了显著进展。MiDaS、DPT、DepthPro 等方法在单帧深度估计上表现优异，但将其应用于视频时缺乏时序一致性，导致深度闪烁。近期的视频深度估计方法（如 DepthCrafter、ChronoDepth）通过时序建模改善了这一问题，但仍依赖大规模标注数据和强特征骨干网络。

**现有痛点**：(1) 单帧 MDE 方法关注的是帧内空间相对深度，忽略了时序深度变化信息。(2) 现有方法严重依赖强预训练特征骨干（如 DINOv2、Stable Diffusion），且需要大量标注数据。(3) 对动态物体的深度估计仍具挑战性，因为多数方法假设静态场景。

**核心矛盾**：单目视频中蕴含丰富的时序深度线索（物体远离时投影点会收敛），但现有方法未能有效挖掘这些纯运动几何信息。

**本文目标**：设计一个仅依赖 2D 点轨迹就能推断深度变化的轻量方法，无需图像特征骨干、无需多视角或立体设置、无需真实标注数据。

**切入角度**：作者从结构光扫描（Structured Light 3D Scanning）获得灵感——结构光通过投影图案的变形来推断3D结构。类似地，视频中被跟踪的点其轨迹模式的变化也编码了深度信息。例如，一个远离相机的物体，其表面跟踪点会在图像平面上收敛聚拢。

**核心 idea**：将深度估计问题从"看图像猜深度"转化为"分析点轨迹运动模式推断深度比"，用 Transformer 学习点密度变化与深度变化之间的映射关系。

## 方法详解

### 整体框架

给定单目视频，首先使用现成的点跟踪模型（LocoTrack 或 CoTracker）提取 2D 轨迹及遮挡状态。然后将轨迹分为两个分支处理：支持轨迹分支（均匀网格采样点）捕获全局场景运动；查询轨迹分支处理用户关心的查询点，并通过交叉注意力从支持分支获取全局运动上下文。两个分支各自输出深度比预测。推理时使用滑动窗口，最终结合单帧度量深度模型输出完整的度量深度。

### 关键设计

1. **支持/查询轨迹的解耦设计**:

    - 功能：将全局场景运动建模与查询点深度预测解耦，避免查询点分布偏差影响深度估计
    - 核心思路：支持轨迹来自图像上的均匀网格采样，通过时序+空间交替的 Transformer 层处理，捕获全局运动动态。查询轨迹独立处理，通过交叉注意力机制从支持分支获取全局运动信息。两个分支各有独立的回归头输出深度比。每个查询点的轨迹独立处理，防止查询点分布对预测的影响
    - 设计动机：用户定义或数据集提供的查询点分布通常不均匀（偏向显著物体），如果直接参与全局运动建模，会引入分布偏差。解耦设计使得全局运动理解不受查询分布影响

2. **滑动窗口预测与 log-ratio 深度损失**:

    - 功能：将长视频的复杂深度预测分解为短时窗口内的简单深度变化预测
    - 核心思路：将视频切分为重叠的窗口（长度 $W$，步长 $S$），在每个窗口内预测相对于窗口起始帧的 log 深度比 $\ell_{i,t}^w = \log(d_{i,t}^w / d_{i,0}^w)$。关键设计是查询轨迹跨窗口持续，而支持轨迹每个窗口重新初始化（支持点更可能留在画面内）。推理时通过累积 log 深度比得到全局深度比：$\hat{r}_{i,t} = \exp(\hat{\ell}_{i,t_k}^k + \sum_{w=0}^{k-1}\hat{\ell}_{i,S}^w)$。训练使用 L1 损失
    - 设计动机：直接处理整段视频会因长距离运动的复杂性导致不稳定的深度预测。短窗口内深度变化更一致可管理，log 比形式使模型对绝对深度尺度不变

3. **深度比与度量深度的融合**:

    - 功能：将预测的时序深度比转换为最终的度量深度值
    - 核心思路：对每条查询轨迹的每段可见子序列，计算尺度因子 $s_{i,t} = \frac{\text{median}_{t' \in \mathcal{S}_{i,t}}(d_{\text{MDE}}(p_{i,t'}))}{\text{median}_{t' \in \mathcal{S}_{i,t}}(\hat{r}_{i,t'})}$，然后 $\hat{d}_{i,t} = s_{i,t} \cdot \hat{r}_{i,t}$。通过中位数匹配而非逐帧匹配，提升鲁棒性。可配合不同的 MDE 模型（ZoeDepth、DepthPro 等）使用
    - 设计动机：本方法预测的是深度变化率（时序相对深度），需要一个"锚点"来获得度量尺度。利用现有 MDE 模型在空间深度估计上的优势互补——本方法提供时序一致性，MDE 提供空间尺度

### 损失函数 / 训练策略

使用 L1 损失监督窗口内的 log 深度比预测，同时对查询轨迹和支持轨迹都施加损失。训练仅在合成数据集 PointOdyssey 上进行，不使用任何预训练特征骨干。采用迭代预测机制（借鉴 RAFT 等工作），将当前预测反馈到模型进行多轮精炼。

## 实验关键数据

### 主实验

TAPVid-3D 基准上的定量结果（per-trajectory depth scaling，使用 CoTracker）：

| 深度估计方法 | Aria 3D-AJ↑ | DriveTrack 3D-AJ↑ | PStudio 3D-AJ↑ | 平均 3D-AJ↑ | 平均 TC↓ |
|------------|------------|-------------------|----------------|------------|---------|
| ZoeDepth (per-frame) | 16.5 | 9.5 | 11.8 | 12.6 | 0.48 |
| DepthPro (per-frame) | 11.3 | 5.4 | 6.7 | 7.8 | 1.40 |
| DepthCrafter (video) | 15.1 | 8.4 | 11.1 | 11.5 | 0.35 |
| ChronoDepth (video) | 11.0 | 6.1 | 3.0 | 6.7 | 3.39 |
| **Seurat (Ours)** | **25.1** | **11.6** | **17.3** | **18.0** | **0.05** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅用理论公式计算 | 需要精确旋转朝向信息，实际场景不可行（Table 4 验证） |
| 不解耦支持/查询分支 | 查询点分布偏差影响全局运动建模，性能下降 |
| 全序列一起处理 | 长视频中运动复杂导致深度预测不稳定 |
| 滑动窗口预测 | 短窗口内深度变化更可管理，性能显著提升 |

### 关键发现

- Seurat 在平均 3D-AJ 指标上大幅领先所有基线（18.0 vs 次好的 12.6），且时序一致性（TC）指标极优（0.05 vs 0.35）
- 相比强大的逐帧 MDE 方法（DepthPro），Seurat 3D-AJ 高出 2.3 倍，说明纯运动线索的深度估计具有巨大潜力
- 仅在合成数据训练就能泛化到驾驶场景、第一人称视角、变形物体等多种真实场景
- 不依赖任何预训练特征骨干是一个显著优势，说明点轨迹运动模式中的深度信息非常丰富

## 亮点与洞察

- **极简主义的深度线索**：不需要图像纹理、语义特征、立体视差，仅凭 2D 点的运动轨迹就能推断深度变化。这揭示了运动中蕴含的几何信息被严重低估
- **结构光类比的巧妙性**：将视频中被跟踪的点类比为结构光投射的图案，这一洞察既直觉又有理论支撑（Section 3.2 的投影面积分析）
- **零样本泛化能力**：仅在合成数据上训练，却在真实多样场景中表现强劲，说明点轨迹运动模式的深度线索具有强域不变性
- **互补性设计**：时序深度比 + 空间度量深度的融合思路可推广到其他时序视觉任务

## 局限与展望

- 依赖点跟踪模型的质量，跟踪失败时深度预测也会失败
- 局部刚体假设在剧烈形变场景下可能不成立
- 滑动窗口累积可能引入长程漂移误差
- 当前仅预测稀疏点上的深度，未扩展到稠密深度图
- 未与 SpatialTracker 等 3D 跟踪方法对比在 2D 跟踪精度上的差异

## 相关工作与启发

- **vs DepthCrafter/ChronoDepth**: 这些视频深度方法依赖大规模标注和强特征骨干，本文完全不需要，且在 3D-AJ 上远超它们
- **vs SpatialTracker**: SpatialTracker 用 MDE 辅助 2D 跟踪（2D→3D→精细化→2D），本文反其道而行，从 2D 轨迹中挖掘 3D 几何信息
- **vs TrackTo4D**: 同样从轨迹重建3D，但引入了额外的动态约束和低维基假设，本文无此限制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从纯运动轨迹推断深度的视角非常新颖，结构光类比极具启发性
- 实验充分度: ⭐⭐⭐⭐ 在 TAPVid-3D 上有全面对比，但消融实验细节较少
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机论述从直觉到数学无缝衔接
- 价值: ⭐⭐⭐⭐⭐ 开辟了深度估计的新范式，零样本泛化能力使其极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2026\] Moving Border Ownership for Event-based Motion Segmentation](../../CVPR2026/3d_vision/moving_border_ownership_for_event-based_motion_segmentation.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](sharpdepth_sharpening_metric_depth_predictions_using_diffusion_distillation.md)

</div>

<!-- RELATED:END -->

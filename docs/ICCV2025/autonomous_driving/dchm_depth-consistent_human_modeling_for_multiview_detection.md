---
title: >-
  [论文解读] DCHM: Depth-Consistent Human Modeling for Multiview Detection
description: >-
  [ICCV 2025][自动驾驶][多视角行人检测] 提出 DCHM，一种无需 3D 标注的深度一致性人体建模框架，通过超像素级高斯溅射生成伪深度标签来微调单目深度估计网络，结合多视角标签匹配实现稀疏视角、遮挡严重场景下的高精度行人检测，在 Wildtrack 上 MODA 达 84.2%，MODP 较 UMPD 提升 31.2%。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "多视角行人检测"
  - "深度一致性"
  - "高斯溅射"
  - "单目深度估计"
  - "无标签方法"
---

# DCHM: Depth-Consistent Human Modeling for Multiview Detection

**会议**: ICCV 2025  
**arXiv**: [2507.14505](https://arxiv.org/abs/2507.14505)  
**代码**: [项目页面](https://github.com/)  
**领域**: 自动驾驶  
**关键词**: 多视角行人检测, 深度一致性, 高斯溅射, 单目深度估计, 无标签方法

## 一句话总结
提出 DCHM，一种无需 3D 标注的深度一致性人体建模框架，通过超像素级高斯溅射生成伪深度标签来微调单目深度估计网络，结合多视角标签匹配实现稀疏视角、遮挡严重场景下的高精度行人检测，在 Wildtrack 上 MODA 达 84.2%，MODP 较 UMPD 提升 31.2%。

## 研究背景与动机
多视角行人检测旨在利用多个摄像头图像检测行人，在严重遮挡场景中尤为有益。当前方法遵循"人体建模 + 行人定位"策略，面临以下挑战：

**特征投影不准确**：现有方法将图像特征或检测结果投影到地面平面，但缺乏行人高度信息导致地面以上点的对齐误差，需依赖 3D 标注训练来弥补

**标签依赖与泛化性**：有标签方法需大量 3D 标注，且难以泛化到新场景

**无标签方法局限**：通过单应矩阵投影的方法对像素级误差敏感；基于体积渲染的方法（UMPD）在稀疏视角和拥挤场景下效果差

**深度不一致**：现有单目深度估计方法虽能提供详细深度，但各视角独立预测的深度图缺乏跨视角一致性，反投影到 3D 空间后点云错位严重（如同一人被不同相机估计为多个目标）

**核心 idea**：利用高斯溅射从稀疏视角图像中自学习多视角一致的伪深度标签 → 微调单目深度网络获得一致深度 → 3D 点云人体建模 → 多视角标签匹配和聚类检测。

## 方法详解

### 整体框架
框架分为训练和推理两个阶段。训练阶段包含迭代的三步循环：(1) 超像素级 GS 优化生成伪深度标签 → (2) 单目深度模型微调 → (3) 多视角检测补偿。推理阶段用优化后的深度估计器生成 3D 点云，通过多视角行人匹配实现分割和定位。

### 关键设计
1. **超像素级高斯溅射初始化与优化**:

    - 功能：在稀疏视角下实现可靠的 3D 重建，生成伪深度标签
    - 核心思路：传统 SfM 在宽基线稀疏视角下失败。采用"均匀采样 + 过滤"策略：对每个超像素中心发射射线，沿射线均匀采样点初始化高斯。每个高斯的尺度基于超像素面积和距离 $t$ 计算：
    $s = \frac{tfr}{\|\mathbf{c} - \mathbf{o}\|_2 \cdot \sqrt{(\sqrt{\|\mathbf{c} - \mathbf{o}\|_2^2 - f^2} - r)^2 + f^2}}$
      使用超像素级光度损失 $\mathcal{L}_{sp}$（而非像素级）提升稀疏视角的一致性。还包含 mask 损失 $\mathcal{L}_m$、深度约束损失 $\mathcal{L}_d$（鼓励同一行人 mask 内深度方差小）和不透明度损失 $\mathcal{L}_o$
    - 设计动机：像素级监督在稀疏视角重叠区域因光照和相机差异导致不一致；超像素聚合局部特征增强一致性，同时加速优化

2. **伪深度过滤与单目深度微调**:

    - 功能：筛选可靠的 GS 输出深度用于微调深度估计网络
    - 核心思路：两步过滤策略：
        - **跨视角前景过滤**：将源视角前景像素重投影到参考视角，若落入背景则丢弃
        - **跨视角深度一致性过滤**：比较重投影深度与 GS 渲染深度，保留差异 < 阈值 $\tau$ 的像素
      过滤后的深度作为伪标签微调 Depth Anything v2 等深度估计网络
    - 设计动机：GS 仅在多视角可见的行人区域产生可靠深度，需过滤掉遮挡区域的错误深度

3. **多视角检测补偿 + 多视角行人标签匹配**:

    - 功能：恢复单视角漏检的行人；确保同一行人在不同视角被一致标记
    - 核心思路：
        - **检测补偿**：将源视角的行人 mask 通过预测深度投影到参考视角，生成 box prompt 和 point prompt 输入 SAM 得到补偿分割结果
        - **标签匹配**：从第一个相机开始，将高斯投影到图像平面，根据 blending weight 和行人 mask 的重叠关系分配 ID。逐视角传播（已有 ID 的高斯落入新 mask → 传递 ID；无 ID 的高斯 → 分配新 ID），确保跨视角一致的行人标识
    - 设计动机：遮挡导致的漏检会影响 GS 优化和最终检测；独立的单视角分割无法保证同一行人在不同视角的一致性

### 损失函数 / 训练策略
GS 优化损失：
$$\mathcal{L} = \lambda_{sp}\mathcal{L}_{sp} + \lambda_m\mathcal{L}_m + \lambda_d\mathcal{L}_d + \lambda_o\mathcal{L}_o$$

迭代训练循环：GS 优化 → 伪深度过滤 → 深度网络微调 → 检测补偿 → 新一轮 GS 优化。3 轮循环后精度提升趋于饱和。

## 实验关键数据

### 主实验（无标签方法对比）

| 方法 | Wildtrack MODA↑ | Wildtrack MODP↑ | Terrace MODA↑ | MultiviewX MODA↑ |
|------|----------------|----------------|---------------|-----------------|
| RCNN & clustering | 11.3 | 18.4 | -11 | 18.7 |
| BP & BB + CC | 56.9 | 67.3 | - | - |
| UMPD | 76.6 | 61.2 | 73.8 | 67.5 |
| **DCHM (Ours)** | **84.2** | **80.3** | **80.1** | **78.4** |

Wildtrack MODP 超 UMPD **31.2%**（80.3 vs 61.2），MultiviewX MODA 超 UMPD **16.1%**（78.4 vs 67.5）。

### 消融实验

| 配置 | MODA | 说明 |
|------|------|------|
| 像素级优化输入 | 71.7 | 稀疏视角不一致 |
| **超像素级优化输入** | **84.2** | 提升 12.5 点 |
| SIS + GVD+VBR (UMPD) | 76.6 | UMPD 原始配置 |
| SIS + Our Recon | 82.5 | 重建提升 5.9 |
| YOLOv11 + Our Recon | **84.2** | 更好的分割进一步提升 |
| 深度估计方法对比（同 Our Loc） | Depth Pro: 72.8, Our Recon: **84.2** | 深度一致性是关键 |

### 关键发现
- 深度一致性是多视角行人检测的关键瓶颈：即使是最好的现成深度估计方法（Depth Pro），在无多视角一致性约束时也远不如本方法
- 超像素级 GS 监督相比像素级监督在稀疏视角下提升巨大（MODA 提升 12.5 点）
- 迭代训练有效：有效伪深度区域逐轮增加，但 3 轮后收益递减
- 检测补偿机制有效恢复了遮挡导致的漏检（YOLOv9/v11 均有提升）
- 推理速度 1.2 FPS，在实时范围内

## 亮点与洞察
- **首次在稀疏视角、大规模、拥挤场景中实现行人重建和多视角分割**
- **自学习伪深度标签**的思路巧妙：用 GS 的跨视角一致性"教"单目深度网络
- **超像素级 GS**解决了稀疏视角下像素级光度损失不可靠的问题
- 多视角标签匹配通过 3D 高斯的 blending weight 在 3D 空间中完成，比 2D tracking 更鲁棒
- 将 DCHM 的人体建模与有标签定位方法结合，甚至超越了有标签方法的 SOTA

## 局限与展望
- 推理速度 1.2 FPS 限制了实时应用
- 训练时间较长（3 轮迭代，每轮需 GS 优化每帧的伪深度）
- 地面重建基于预定义深度范围，可能在复杂地形下失效
- 对分割网络质量敏感（YOLOv11 vs SIS 差异显著）
- 暂未处理动态场景（行人移动时 GS 优化假设静态场景）

## 相关工作与启发
- 超像素级 GS 优化的思路可推广到其他稀疏视角重建任务
- 伪深度过滤策略是自监督深度学习的有效实践
- 多视角标签匹配方法可用于其他多相机目标追踪/分割任务
- 检测补偿机制的 SAM prompt 生成策略有普适价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 超像素级 GS + 自学习伪深度 + 检测补偿的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多方法对比、详细消融、深度估计方法横评
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，逻辑连贯，技术细节充分
- 价值: ⭐⭐⭐⭐ 无标签多视角检测具有重要实用价值（部署成本低），但应用场景相对窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Bezier Degradation Modeling for LiDAR-based Human Motion Capture](../../CVPR2026/autonomous_driving/bezier_degradation_modeling_for_lidar-based_human_motion_capture.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[CVPR 2026\] Neuro-Cognitive Reward Modeling for Human-Centered Autonomous Vehicle Control](../../CVPR2026/autonomous_driving/neuro-cognitive_reward_modeling_for_human-centered_autonomous_vehicle_control.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](../../CVPR2025/autonomous_driving/prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)

</div>

<!-- RELATED:END -->

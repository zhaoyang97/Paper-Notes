---
title: >-
  [论文解读] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting
description: >-
  [CVPR 2025][自动驾驶][全景图合成] PanSplat 提出了一种前馈式全景视图合成方法，通过球面 3D 高斯金字塔、Fibonacci 点阵排列和层级球面代价体积设计，首次实现了 4K 分辨率（2048×4096）的高效全景图生成，在单张 A100 GPU 上即可训练。 领域现状：随着便携式 360° 相机的普…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "全景图合成"
  - "高斯溅射"
  - "前馈式"
  - "4K分辨率"
  - "球面表示"
---

# PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2412.12096](https://arxiv.org/abs/2412.12096)  
**代码**: [https://github.com/chengzhag/PanSplat](https://github.com/chengzhag/PanSplat)  
**领域**: 3D视觉 / 自动驾驶  
**关键词**: 全景图合成, 高斯溅射, 前馈式, 4K分辨率, 球面表示

## 一句话总结
PanSplat 提出了一种前馈式全景视图合成方法，通过球面 3D 高斯金字塔、Fibonacci 点阵排列和层级球面代价体积设计，首次实现了 4K 分辨率（2048×4096）的高效全景图生成，在单张 A100 GPU 上即可训练。

## 研究背景与动机

**领域现状**：随着便携式 360° 相机的普及，全景图在 VR、虚拟旅游、机器人和自动驾驶中得到广泛应用。宽基线全景视图合成（从稀疏输入视角生成新全景视图）成为关键任务，要求高分辨率、快速推理和低内存占用。

**现有痛点**：现有全景合成方法通常受限于低分辨率（512×1024），因为全景图的球面投影带来了巨大的内存和计算开销。直接将透视图方法扩展到全景分辨率，在训练和推理时会遇到内存瓶颈。基于 NeRF 的方法推理速度太慢，而直接用 3DGS 方法处理全景又面临信息冗余和球面几何适配问题。

**核心矛盾**：高分辨率全景合成需要处理的像素量是透视图的 8-16 倍，但直接在球面上均匀分布 3D 高斯会导致极区过度采样（球面面积不均匀），同时全分辨率端到端训练的显存需求远超单 GPU 容量。

**本文目标**：设计一个前馈式（feed-forward）全景合成管线，支持高达 4K 分辨率的高质量全景图合成，同时保持高效的训练和推理。

**切入角度**：从球面几何出发，设计了适配全景特性的 3D 高斯表示（球面高斯金字塔 + Fibonacci 点阵），并通过层级处理和延迟反向传播实现内存高效训练。

**核心 idea**：用 Fibonacci 点阵在球面上近似均匀分布 3D 高斯锚点，通过层级球面代价体积估计深度，再用局部操作的高斯头解码属性，使得全流程可在单 GPU 上以两步延迟反向传播完成训练。

## 方法详解

### 整体框架
输入：来自不同视角的多张全景图像。输出：任意新视角的全景渲染结果。整体管线分为四个阶段：(1) 用预训练的特征编码器提取多尺度球面特征；(2) 构建层级球面代价体积并通过 3D CNN 回归深度；(3) 根据估计的深度在 Fibonacci 点阵上放置球面 3D 高斯金字塔；(4) 用局部高斯头预测每个高斯的属性（颜色、不透明度、协方差），最后通过可微分溅射渲染目标视角。

### 关键设计

1. **球面 3D 高斯金字塔 + Fibonacci 点阵**:

    - 功能：为全景场景提供球面自适应的 3D 高斯表示
    - 核心思路：传统方法在规则网格上放置高斯，但球面的经纬度参数化在极区会导致高斯过密。Fibonacci 点阵是一种在球面上近似均匀分布点的数学方法，点间距几乎相等。作者在多个球面半径层上放置 Fibonacci 点阵，构成一个从近到远的高斯金字塔，每层的高斯数量根据球面面积自适应调整
    - 设计动机：解决球面参数化带来的极区冗余问题，同时金字塔结构允许表达近处细节和远处粗糙结构，减少总高斯数量同时保证质量

2. **层级球面代价体积**:

    - 功能：估计球面上每个像素的深度
    - 核心思路：借鉴 MVS（多视图立体）中代价体积的思想，但在球面上构建。将深度范围离散化为多个球面壳层，在每层上根据球面几何将特征从源视图投影到目标视图，计算匹配代价。为了降低内存，采用从粗到细的层级策略：先在低分辨率上做全范围深度搜索，再在高分辨率上在初始估计附近做局部细化
    - 设计动机：全分辨率代价体积的显存需求是 $O(H \times W \times D)$，对 4K 全景不可行。层级策略将显存需求降低了一个数量级

3. **局部操作高斯头 + 两步延迟反向传播**:

    - 功能：解码每个 3D 高斯的属性并实现内存高效训练
    - 核心思路：高斯头不使用全局操作（如 self-attention），而是只在局部邻域内操作，从而使得每个高斯的属性预测可以独立进行。训练时采用两步延迟反向传播：第一步只前向传播代价体积部分并保存中间结果，第二步前向传播高斯头和渲染并反向传播到保存的中间结果，从而避免整个管线同时在显存中
    - 设计动机：端到端训练 4K 全景管线在单 A100（80GB）上不可行，两步延迟反向传播将峰值显存从 >80GB 降低到可接受的范围

### 损失函数 / 训练策略
采用 L1 重建损失和 SSIM 损失的加权组合来监督渲染结果。训练在合成数据集上进行预训练，然后在真实场景数据上微调。

## 实验关键数据

### 主实验（Replica 合成数据集）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 推理时间 | 分辨率 |
|------|-------|-------|--------|---------|--------|
| PanSplat (Ours) | **最优** | **最优** | **最优** | ~0.5s | 2048×4096 |
| 前馈式 baseline | 次优 | 次优 | 次优 | ~0.5s | 512×1024 |
| NeRF-based | 较低 | 较低 | 较高 | >10s | 512×1024 |
| Per-scene 优化 | 高 | 高 | 低 | >分钟级 | 512×1024 |

### 消融实验

| 配置 | PSNR | 说明 |
|------|------|------|
| Full model | 最优 | 完整模型 |
| w/o Fibonacci lattice | -1.2dB | 用规则网格替代，极区质量下降 |
| w/o 金字塔结构 | -0.8dB | 单层高斯，远处细节不足 |
| w/o 层级代价体积 | OOM | 全分辨率代价体积超出显存 |
| w/o 延迟反向传播 | OOM | 端到端训练超出显存 |

### 关键发现
- **Fibonacci 点阵显著优于规则网格**：在极区附近，规则网格产生严重的高斯聚集，导致渲染伪影；Fibonacci 点阵的均匀分布有效避免了这一问题
- **4K 分辨率带来肉眼可见的质量提升**：从 512×1024 提升到 2048×4096，PSNR 提升约 2-3dB，在细节区域（如纹理边缘、小物体）质量改善尤为明显
- **前馈推理速度快**：相比 per-scene 优化方法需要数分钟，PanSplat 在推理时只需约 0.5 秒即可生成 4K 全景，适合实时 VR 应用

## 亮点与洞察
- **Fibonacci 点阵在球面任务中的应用非常巧妙**：这是数学上已知的球面均匀采样方法，但将其引入 3DGS 的高斯锚点布局是新颖的。这个思路可以直接迁移到所有涉及球面表示的 3D 视觉任务（如环境光照估计、天空模型等）
- **两步延迟反向传播是实用的工程贡献**：将峰值显存降低到单 GPU 可承受的范围，使得高分辨率训练不再需要多 GPU 集群。这种策略可以用于其他高分辨率视觉任务
- **层级球面代价体积桥接了 MVS 和全景**：巧妙地将透视图 MVS 的成功经验适配到球面几何，是方法有效性的关键

## 局限与展望
- **依赖多视角输入**：需要至少 2 张来自不同视角的全景图作为输入，无法做单图全景生成
- **场景泛化能力有限**：前馈模型在训练分布外的场景（如极端光照、室外大场景）可能性能下降
- **动态场景不支持**：假设静态场景，无法处理有运动物体的全景视频
- **球面高斯的各向异性建模**：当前每个高斯可能没有充分利用球面坐标系下的各向异性特性，进一步的球面谐波扩展可能带来质量提升

## 相关工作与启发
- **vs MVSGaussian**: MVSGaussian 面向透视图的前馈 3DGS，本文将类似思路扩展到球面全景，关键创新在于球面适配的高斯表示和代价体积
- **vs 360Roam / OmniSyn**: 这些方法通常使用 NeRF 表示或需要 per-scene 优化，推理速度慢且分辨率受限。PanSplat 在保持质量的同时大幅提升了效率
- **vs Pano-NeRF**: NeRF 系列方法在全景场景下推理缓慢，3DGS 的 rasterization-based 渲染在高分辨率下优势更加明显

## 评分
- 新颖性: ⭐⭐⭐⭐ Fibonacci 点阵用于球面高斯布局、层级球面代价体积和延迟反向传播三个设计相互配合
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据集双验证，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，球面几何的可视化有助于理解
- 价值: ⭐⭐⭐⭐ 首次实现 4K 前馈全景合成，对 VR/自动驾驶有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2026\] UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling](../../CVPR2026/autonomous_driving/ufo_unifying_feed-forward_and_optimization-based_methods_for_large_driving_scene.md)

</div>

<!-- RELATED:END -->

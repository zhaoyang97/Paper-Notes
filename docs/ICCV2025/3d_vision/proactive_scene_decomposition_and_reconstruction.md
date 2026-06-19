---
title: >-
  [论文解读] Proactive Scene Decomposition and Reconstruction
description: >-
  [ICCV 2025][3D视觉][动态SLAM] 提出基于主动人-物交互的在线场景分解与重建任务,通过观察自我中心视角下的交互行为来定义分解粒度,实现渐进式对象解耦和高质量全局重建。 传统的对象级场景重建方法面临一个根本性困境:分解粒度的歧义性：。例如:抽屉应该从柜子中分离吗？柜子里的物品算柜子的一部分吗？静态场景下这些问…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "动态SLAM"
  - "场景分解"
  - "人-物交互"
  - "Gaussian Splatting"
  - "在线重建"
---

# Proactive Scene Decomposition and Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2510.16272](https://arxiv.org/abs/2510.16272)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 动态SLAM, 场景分解, 人-物交互, Gaussian Splatting, 在线重建

## 一句话总结

提出基于主动人-物交互的在线场景分解与重建任务,通过观察自我中心视角下的交互行为来定义分解粒度,实现渐进式对象解耦和高质量全局重建。

## 研究背景与动机

传统的对象级场景重建方法面临一个根本性困境:**分解粒度的歧义性**。例如:抽屉应该从柜子中分离吗？柜子里的物品算柜子的一部分吗？静态场景下这些问题天然是病态的(ill-posed)。

现有方法的局限:

**对象级NeRF/3DGS方法**（如Gaussian Grouping）依赖预定义的分割粒度,无法适应上下文

**4D重建/动态SLAM**方法目标是处理任意动态,但无法充分利用交互线索

**静态分解方法**在交叉区域观测不完整,需要inpainting补全

本文的核心洞察:**人的行为本身就定义了最自然的分解粒度** — 被手抓取移动的部分就是一个独立单元。通过观察人-物交互,可以渐进式地消除分解歧义并获得完整观测。

## 方法详解

### 整体框架

系统由四个模块组成:提示分割、相机/物体位姿估计、掩码细化、分解式场景重建。

### Gaussian基元参数化

使用各向同性Gaussian基元:颜色 $\mathbf{c} \in \mathbb{R}^3$、位置 $\mu \in \mathbb{R}^3$、各向同性方差 $r \in \mathbb{R}$、不透明度 $o \in \mathbb{R}$。

### 联合优化目标

$$L = \lambda_p L_p + \lambda_d L_d + \lambda_{ID} L_{ID}$$

其中 $L_p, L_d, L_{ID}$ 分别是颜色、深度和实例分割的L1损失。

### 提示分割（Prompted Segmentation）

利用3D场景地图的渲染结果与当前观测的不一致性检测动态区域:

$$\frac{\sum_{(u,v) \in S_{grid}} \mathbb{1}(\hat{D}[u,v] - D[u,v] > t_d)}{|S_{grid}|} > t_p$$

检测到不一致后提取质心作为SAM2的提示,结合YOLO手部检测验证是否为真实的手-物交互。

### 掩码细化

针对SAM2的三类常见失败模式设计修复策略:
1. **不完整观测** — 设计灵活长度记忆库确保保留完整观测
2. **帧间不一致** — 基于渲染掩码与预测掩码的比较添加正/负提示
3. **物体不在视野** — 基于3D位置判断物体状态,直接设零掩码

### 渐进式分解

$$\frac{\sum_{f \in \mathcal{F}_{valid}} \mathbb{1}(P(\tilde{g}) \in M_r)}{|\mathcal{F}_{valid}|} > t_{3d}$$

在多个关键帧中频繁出现在掩码内的Gaussian被识别为物体的一部分并解耦到独立集合。

## 实验

### 掩码质量对比（HOI4D数据集）

| 方法 | Seq 1 | Seq 2 | Seq 3 | Seq 4 |
|------|-------|-------|-------|-------|
| SAM2 | 0.913 | 0.884 | 0.318 | 0.941 |
| **Ours** | **0.925** | **0.920** | **0.835** | **0.947** |

Seq 3（复杂剪刀）改进最显著,mIoU从0.318提升到0.835。

### SLAM性能对比

| 方法 | HOI4D ATE | PSNR(静态) | PSNR(动态) | MHOI ATE |
|------|-----------|-----------|-----------|----------|
| Co-SLAM | 0.172 | 17.35 | – | 0.221 |
| SplaTaM | 0.156 | 18.61 | – | 0.293 |
| NeuDySLAM | 0.094 | 25.15 | – | 0.189 |
| **Ours** | **0.076** | **29.12** | **27.58** | **0.093** |

### 关键发现

1. 本方法不仅重建静态背景,还同时重建运动中的交互物体,实现更全面的环境理解
2. 利用交互信息的掩码细化显著优于纯粹依赖SAM2的结果
3. 渐进式分解避免了静态分解方法的粒度歧义问题

## 亮点与洞察

1. **任务定义的创新性** — "主动场景分解"是一个原创性的问题定义,将人的意图行为作为分解的天然驱动力
2. **交互先验的高效利用** — 手-物交互提供了稳定可控的分解粒度定义
3. **在线系统设计** — 支持实时反馈,为增量式地图更新奠定基础
4. **统一框架** — 在SLAM系统中同时优化场景辐射、相机运动、物体位姿和实例分割

## 局限性

- 假设所有交互物体的运动近似刚体运动
- 依赖RGB-D输入,纯RGB设置未覆盖
- 仅处理人-物交互导致的动态,不适用于任意场景动态

## 相关工作

- **对象分解辐射场**: ObjectSDF, Gaussian Grouping, Panoptic Lifting, EgoGaussian
- **Agent-in-the-loop场景理解**: Roboexp, Autoscanning, iLabel
- **动态SLAM**: DynaSLAM, DRG-SLAM, CFP-SLAM

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (问题定义新颖,将交互作为分解驱动力)
- 技术深度: ⭐⭐⭐⭐ (系统集成复杂但合理)
- 实验充分度: ⭐⭐⭐⭐ (多角度验证,新建MHOI数据集)
- 实用价值: ⭐⭐⭐⭐ (机器人操作等下游任务有直接价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SuperDec: 3D Scene Decomposition with Superquadric Primitives](superdec_3d_scene_decomposition_with_superquadrics_primitives.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)
- [\[CVPR 2026\] Semantic Foam: Unifying Spatial and Semantic Scene Decomposition](../../CVPR2026/3d_vision/semantic_foam_unifying_spatial_and_semantic_scene_decomposition.md)

</div>

<!-- RELATED:END -->

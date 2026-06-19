---
title: >-
  [论文解读] Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes
description: >-
  [CVPR 2025][遥感][hyperspectral 3D imaging] 提出 Dense Dispersed Structured Light（DDSL）方法，利用廉价衍射光栅薄膜（<\$20）+ 立体 RGB 相机 + RGB 投影仪，设计光谱复用 DDSL 图案大幅减少所需投影帧数，实现 6.6fps 实时高光谱 3D 成像，光谱分辨率 15.5nm FWHM，深度误差 4mm。
tags:
  - "CVPR 2025"
  - "遥感"
  - "hyperspectral 3D imaging"
  - "structured light"
  - "diffraction grating"
  - "stereo cameras"
  - "dynamic scenes"
---

# Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes

**会议**: CVPR 2025  
**arXiv**: [2412.01140](https://arxiv.org/abs/2412.01140)  
**代码**: —  
**领域**: 遥感 / 计算成像  
**关键词**: hyperspectral 3D imaging, structured light, diffraction grating, stereo cameras, dynamic scenes

## 一句话总结
提出 Dense Dispersed Structured Light（DDSL）方法，利用廉价衍射光栅薄膜（<\$20）+ 立体 RGB 相机 + RGB 投影仪，设计光谱复用 DDSL 图案大幅减少所需投影帧数，实现 6.6fps 实时高光谱 3D 成像，光谱分辨率 15.5nm FWHM，深度误差 4mm。

## 研究背景与动机

**领域现状**：高光谱 3D 成像同时捕获深度图和高光谱图像，可用于全面的几何和材质分析。近年方法在光谱和深度精度上取得进展。

**现有痛点**：
   - **采集时间长**：现有方法通常需要数分钟，仅能拍摄静态场景
   - **系统昂贵庞大**：依赖专业高光谱相机或复杂光学系统
   - **无法处理动态场景**：慢速采集导致运动模糊和帧间不一致

**核心矛盾**：高光谱维度本身信息量大，传统方法需要大量不同波长的投影/采集；而动态场景要求极快的采集速度。

**切入角度**：利用衍射光栅将 RGB 投影仪的宽带光分散为光谱成分，通过精心设计的复用图案用少量投影帧编码多个光谱通道。

**核心 idea**：衍射光栅 + 光谱复用图案 + 图像形成模型 = 低成本实时高光谱 3D。

## 方法详解

### 硬件配置
- **RGB 投影仪**：标准 DLP 投影仪，投射设计好的 DDSL 图案
- **衍射光栅薄膜**：粘贴在投影仪镜头前，成本 <\$20
    - 作用：将投射的宽带结构光分散为不同波长分量
    - 每个投影像素在场景上形成一条"彩虹"条纹
- **立体 RGB 相机**：两台标准 RGB 相机，分别捕获场景

### 关键设计

1. **光谱复用 DDSL 图案设计**

    - 功能：大幅减少所需投影帧数
    - 核心思路：
        - 传统方法每个波长需独立投影，高光谱需要 20-30 帧
        - DDSL 利用衍射光栅的空间分散特性，在单帧中编码多个光谱通道
        - 设计正交或准正交的空间图案，使不同波长的投影在场景上互不干扰
    - 关键约束：确保不同波长的分散光不会空间重叠（使用偏移量编码）
    - 效果：仅需 ~3-5 帧投影即可覆盖全部光谱通道

2. **图像形成模型（Image Formation Model）**

    - 功能：建立投影图案、衍射光栅、场景反射率和相机观测之间的物理模型
    - 核心公式：
        - 投影仪发出的光经过衍射光栅后在空间上按波长分散
        - 分散角度由光栅方程决定：$d \sin\theta = m\lambda$
        - 场景每点反射的光包含其光谱反射率信息
        - RGB 相机观测是光谱反射率与相机响应函数的积分
    - 作用：将观测的 RGB 图像反演为高光谱图像

3. **高光谱和深度重建算法**

    - 功能：从立体 RGB 观测恢复高光谱图像和深度图
    - 深度估计：利用立体 RGB 相机的视差进行传统立体匹配
    - 光谱反射率恢复：
        - 基于图像形成模型建立线性方程组
        - 使用正则化优化求解光谱反射率
        - 利用多帧投影提供的冗余信息提高稳定性
    - 联合优化：深度和光谱反射率交替优化直到收敛

4. **快速采集流程**

    - 投影仪以高帧率（~20fps）投射设计好的 DDSL 图案序列
    - 两台 RGB 相机同步采集
    - 每 3-5 帧构成一个完整高光谱采样周期
    - 有效帧率：6.6 fps

### 光谱复用原理（几何直觉）
- 衍射光栅将白光分散为彩虹：红光偏转角大，蓝光偏转角小
- 投影特定空间位置的像素，经光栅后不同波长落在场景不同位置
- 通过精确设计投影图案的空间分布，控制每个波长的照明位置
- 相机端通过已知的几何关系解码光谱信息

## 实验关键数据

### 系统性能指标

| 指标 | 数值 |
|------|------|
| 光谱分辨率 (FWHM) | 15.5 nm |
| 深度误差 | 4 mm |
| 帧率 | 6.6 fps |
| 光栅成本 | <\$20 |
| 光谱范围 | 可见光 (~400-700nm) |

### 与现有方法对比

| 方法 | 帧率 | 采集时间 | 成本 | 动态场景 |
|------|------|---------|------|---------|
| 传统高光谱+结构光 | 极低 | 数分钟 | 高 | ✗ |
| 滤光轮方法 | 低 | 数十秒 | 中 | ✗ |
| Coded aperture | 中 | 数秒 | 高 | 有限 |
| **DDSL (Ours)** | **6.6 fps** | **实时** | **低** | **✓** |

### 光谱重建精度

| 评估 | 指标 |
|------|------|
| RMSE（光谱反射率） | 低 |
| SAM（光谱角映射） | 低 |
| 与参考光谱计的一致性 | 高 |

### 关键发现
- 衍射光栅薄膜引入的光效率损失可接受
- 光谱复用图案使投影帧数降低约 5-6 倍
- 深度精度与传统立体匹配方法相当（4mm）
- 15.5nm 光谱分辨率足以区分常见材质

## 亮点与洞察
- **极低成本**：仅需 <\$20 的衍射光栅薄膜 + 标准 RGB 设备
- **实时动态**：首个实用的动态场景高光谱 3D 成像方法
- **物理模型驱动**：基于衍射光学的精确成像模型，非纯数据驱动
- **设备简单**：现成投影仪和相机即可搭建

## 局限与展望
- 光谱分辨率（15.5nm）低于专业高光谱仪器（<5nm）
- 工作范围限于可见光，近红外需要不同光栅
- 投影距离和环境光影响系统性能
- 深度精度受立体基线和分辨率限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 衍射光栅+结构光的组合极具创意
- 实验充分度: ⭐⭐⭐⭐ 真实系统搭建+多场景测试
- 写作质量: ⭐⭐⭐⭐ 物理模型推导清楚
- 价值: ⭐⭐⭐⭐⭐ 低成本实时高光谱3D有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)
- [\[CVPR 2025\] SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion](sgformer_satellite-ground_fusion_for_3d_semantic_scene_completion.md)
- [\[CVPR 2026\] Semantic-Adaptive Diffusion for Dynamic Spatiotemporal Fusion](../../CVPR2026/remote_sensing/semantic-adaptive_diffusion_for_dynamic_spatiotemporal_fusion.md)
- [\[CVPR 2026\] UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes](../../CVPR2026/remote_sensing/unigeoseg_towards_unified_open-world_segmentation_for_geospatial_scenes.md)
- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](../../NeurIPS2025/remote_sensing/greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)

</div>

<!-- RELATED:END -->

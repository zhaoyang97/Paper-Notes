---
title: >-
  [论文解读] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis
description: >-
  [ICCV 2025][3D视觉][人体重打光] 提出HumanOLAT——首个公开可用的大规模全身人体多视角OLAT(One-Light-at-a-Time)数据集，包含21个被试×3个姿态×40视角×344种光照≈850K帧，为人体重打光和新视角合成提供了高质量基准。 现有痛点 现有痛点：全身人体同时重打光和新视角渲染是…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "人体重打光"
  - "OLAT数据集"
  - "Light Stage"
  - "新视角合成"
  - "全身捕获"
---

# HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis

**会议**: ICCV 2025  
**arXiv**: [2508.09137](https://arxiv.org/abs/2508.09137)  
**代码**: [项目页](https://vcai.mpi-inf.mpg.de/projects/HumanOLAT/)  
**领域**: 3D视觉  
**关键词**: 人体重打光, OLAT数据集, Light Stage, 新视角合成, 全身捕获

## 一句话总结

提出HumanOLAT——首个公开可用的大规模全身人体多视角OLAT(One-Light-at-a-Time)数据集，包含21个被试×3个姿态×40视角×344种光照≈850K帧，为人体重打光和新视角合成提供了高质量基准。

## 研究背景与动机

### 现有痛点

**现有痛点**：全身人体同时重打光和新视角渲染是计算机视觉的核心挑战，但进展受限于缺乏公开数据集：

**Light Stage设备稀缺且昂贵**：需要精确控制光照和多视角相机

**全身OLAT采集困难**：需要更大捕获空间和更长时间，期间被试即使轻微移动也会产生明显伪影

**现有数据集局限**：

### 领域现状

**领域现状**：ReNe/OpenIllumination：仅物体级

### 核心矛盾

**核心矛盾**：Dynamic OLAT/Goliath-4：仅面部/手部

### 解决思路

**解决思路**：Ultrastage：仅提供白光和色彩梯度，无OLAT

## 方法详解

### 数据采集设置

球形穹顶配备：
- 40台RED Komodo 6K相机
- 331个独立可控LED（RGBAW）
- 360°环绕被试
- 30 FPS同步捕获，5K图像分辨率

### 数据集内容

每个被试约40K帧，总计约850K帧：
- **1种白光照明**：用于校准、网格重建、分割
- **2种色彩梯度照明**：估计逐像素光度法线
- **10种环境光照**：直接加载到Light Stage
- **331种OLAT照明**：基于图像的重打光

### 运动补偿

捕获约11秒期间被试无法完全静止。使用Wenger方法：
- 每21帧OLAT插入一帧白光跟踪帧
- 使用Co-Tracker3追踪约12k稀疏网格点
- 线性插值到密集光流，将OLAT帧warp到目标帧

### 光度法线估计

从色彩梯度照明 $g^+$, $g^-$ 计算法线：

$$\mathbf{n} = \frac{\mathbf{d}}{|\mathbf{d}|}, \quad \mathbf{d} = \frac{g^+ - g^-}{g^+ + g^-}$$

### 基于图像的重打光

利用光传输线性性，给定目标环境光 $E_{\text{target}}$：

$$I_{\text{target}} = \sum_{i=0}^{N_{\text{OLAT}}} \mathbf{c}_i I_i$$

## 实验

### 逆渲染方法对比 (OLAT照明)


### 主实验

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ |
|------|-------|--------|-------|
| PRT-Gaussian | 24.06 | 0.212 | 0.810 |
| RNG | 27.38 | 0.139 | 0.905 |
| BiGS | 26.72 | 0.201 | **0.936** |
| **GS³** | **30.04** | **0.152** | 0.892 |

GS³在总体上表现最佳，但所有方法在手和面部区域仍存在明显伪影。

### 关键发现

- PRT-Gaussian难以从OLAT帧准确重建几何，出现鬼影
- GS³、RNG和BiGS表现更好但渲染仍偏模糊
- 即使最优方法也无法有效捕获镜面高光和锐利阴影
- IC-Light在保持面部细节（眼睛、嘴巴形状）方面持续失败

## 亮点与洞察

1. **填补关键数据空白**：首个公开的全身OLAT数据集，为社区提供了不可替代的资源
2. **完整的评估框架**：覆盖OLAT重打光、环境光重打光、光照和谐化等多个任务
3. **揭示现有方法不足**：即使最好的3DGS重打光方法也无法处理人体复杂光传输
4. **高质量数据处理**：运动补偿、精确校准(0.819像素误差)等保证数据质量

## 局限与展望

- 仅21个被试，多样性有限
- 仅静态姿态，不支持动态序列
- 约20个LED安装在活动舱口上，位置不完全准确
- 5K分辨率虽高但比Ultrastage(8K)低

## 相关工作

- Ultrastage: 100被试全身但仅白光+色彩梯度
- Dynamic OLAT, Goliath-4: 面部/手部OLAT
- IC-Light: 光照和谐化方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首个公开全身OLAT数据集)
- 技术深度: ⭐⭐⭐⭐ (完整的数据处理流水线)
- 实验充分度: ⭐⭐⭐⭐ (多方法多任务评估)
- 实用价值: ⭐⭐⭐⭐⭐ (对重打光研究极其重要)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[CVPR 2026\] Charge: A Comprehensive Novel View Synthesis Benchmark and Dataset to Bind Them All](../../CVPR2026/3d_vision/charge_a_comprehensive_novel_view_synthesis_benchmark_and_dataset_to_bind_them_a.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->

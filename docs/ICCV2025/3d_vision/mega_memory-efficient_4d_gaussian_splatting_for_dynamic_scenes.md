---
title: >-
  [论文解读] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
description: >-
  [ICCV 2025][3D视觉][4D高斯溅射] 提出 MEGA，一个面向4D Gaussian Splatting的内存高效框架，通过DC-AC颜色分解消除冗余球谐系数（8×压缩），结合熵约束Gaussian形变技术扩大每个Gaussian的作用范围并减少数量，最终在Technicolor和Neural 3D Video数据集上分别实现约190×和125×存储压缩，同时保持可比的渲染质量和实时速度。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "4D高斯溅射"
  - "内存高效"
  - "动态场景"
  - "颜色压缩"
  - "熵约束形变"
---

# MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes

**会议**: ICCV 2025  
**arXiv**: [2410.13613](https://arxiv.org/abs/2410.13613)  
**代码**: [Xinjie-Q/MEGA](https://github.com/Xinjie-Q/MEGA)  
**领域**: 3D视觉  
**关键词**: 4D高斯溅射, 内存高效, 动态场景, 颜色压缩, 熵约束形变

## 一句话总结

提出 MEGA，一个面向4D Gaussian Splatting的内存高效框架，通过DC-AC颜色分解消除冗余球谐系数（8×压缩），结合熵约束Gaussian形变技术扩大每个Gaussian的作用范围并减少数量，最终在Technicolor和Neural 3D Video数据集上分别实现约190×和125×存储压缩，同时保持可比的渲染质量和实时速度。

## 研究背景与动机

**4D Gaussian Splatting (4DGS)** 将3DGS扩展到动态场景，用4D时空Gaussian超柱体表示场景运动，通过时间切片获取每帧的3D Gaussian进行实时渲染。然而4DGS面临 **严重的存储瓶颈**：

**球谐系数的巨大冗余**：每个4D Gaussian的161个参数中，144个是4D球谐（SH）系数，占比89%。这些SH系数用于编码视点和时间相关的颜色变化，但存在大量冗余。

**Gaussian数量爆炸**：渲染Birthday场景需要高达1300万个Gaussian，存储开销约7.79GB。这主要源于两个原因：
   - 4DGS假设每个切片后的Gaussian仅线性运动、协方差不变，复杂非线性运动需要多个Gaussian叠加表示
   - 时间衰减不透明度 $\sigma(t) = e^{-\frac{(t-\mu_t)^2}{2\mathbf{W}}}$ 使每个Gaussian仅在其时间中心附近可见，**任何给定时间只有约6%的Gaussian参与渲染**

**现有3DGS压缩方法不适用**：3DGS的剪枝、SH蒸馏、矢量量化等技术面向静态场景设计，未考虑4DGS的时间和多视点因素。

**核心思路**：从两个维度压缩——(1)减少每个Gaussian的参数量（消除SH系数），(2)减少Gaussian总数（扩大每个Gaussian的时空作用范围）。

## 方法详解

### 整体框架

MEGA包含三个核心组件：

1. **内存高效4D Gaussian表示**：用DC-AC颜色替代SH系数
2. **熵约束Gaussian形变**：形变预测器+不透明度熵损失
3. **存储压缩**：FP16精度 + zip delta压缩

渲染流程四步：Per-Gaussian变换 → 时间切片 → 投影 → 可微光栅化。

### 关键设计一：DC-AC 颜色 (DAC) 表示

受电气工程中直流（DC）和交流（AC）概念启发，将颜色属性解耦为：

- **DC分量**：每Gaussian的直流颜色 $\mathbf{c}_{dc} \in \mathbb{R}^3$，仅3个参数，编码场景中固有的稳态颜色信息
- **AC预测器**：共享的轻量MLP $\mathcal{F}_\phi$，根据时间和视点预测颜色变化

最终颜色通过残差连接计算：
$$\mathbf{c}_{t,v} = \text{sigmoid}(\mathbf{c}_{dc} + \mathcal{F}_\phi(\text{sg}(\boldsymbol{\mu}_{3D}), \text{sg}(\mathbf{d}_v), t, \mathbf{c}_{dc}))$$

其中 $\text{sg}(\cdot)$ 是停止梯度操作，$\mathbf{d}_v$ 是归一化视线方向。AC预测器使用三层线性层，输入包含3D位置、视线方向、时间和DC颜色。

**效果**：每个Gaussian只需存储3个颜色参数（vs原始的144个SH系数），实现约 **8×参数压缩**。关键在于DC分量保留了核心颜色信息，AC预测器补充了缺失的时空变化信息。

### 关键设计二：熵约束Gaussian形变

**形变预测器**：为每个4D Gaussian预测时间-视点相关的几何形变。将4D中心 $\boldsymbol{\mu}_{4D}$、视线方向 $\mathbf{d}_v$ 和时间 $t$ 通过频率位置编码 $\gamma$ 映射到高维空间，再由轻量MLP $\mathcal{F}_\theta$ 预测形变增量：

$$(m_{\mu_{4D}}^{t,v}, m_{s_{4D}}^{t,v}, m_{q_l}^{t,v}, m_{q_r}^{t,v}) = \mathcal{F}_\theta(\gamma(\text{sg}(\boldsymbol{\mu}_{4D})), \gamma(\text{sg}(\mathbf{d}_v)), \gamma(t))$$

形变通过**乘法**作用于原始参数（而非加法），使Gaussian可以表示非线性运动和形状变化：

$$\boldsymbol{\mu}_{4D}^{t,v} = \boldsymbol{\mu}_{4D} \times m_{\mu_{4D}}^{t,v}, \quad \mathbf{s}_{4D}^{t,v} = \mathbf{s}_{4D} \times m_{s_{4D}}^{t,v}$$

**不透明度熵损失**：强制空间不透明度 $o$ 趋向二值（0或1），便于识别和剪枝无用Gaussian：

$$\mathcal{L}_{opa} = \frac{1}{N} \sum_{j=1}^N (-o_j \log(o_j))$$

每 $K$ 迭代剪枝近零不透明度Gaussian。配合形变预测器，Gaussian参与渲染的比例从不到50%提升至约75%。

### 损失函数

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{ssim} + \kappa \mathcal{L}_{opa}$$

其中 $\lambda = 0.2$, $\kappa = 0.0005$。

### 存储压缩

训练使用半精度（FP16），存储所有可学习参数为FP16格式后应用zip delta压缩，额外减少约10%存储。

## 实验

### 主实验一：Technicolor数据集

| 方法 | PSNR↑ | DSSIM1↓ | LPIPS↓ | FPS↑ | 存储↓ |
|------|-------|---------|--------|------|-------|
| DyNeRF | 31.80 | - | 0.1400 | 0.02 | 30.00MB |
| HyperReel | 32.70 | 0.0470 | 0.1090 | 4.00 | 60.00MB |
| Deformable 3DGS | 30.95 | 0.0696 | 0.1553 | 76.09 | 61.36MB |
| STG | 33.35 | 0.0404 | 0.0846 | 141.73 | 51.35MB |
| 4DGS | 32.07 | 0.0535 | 0.1189 | 55.26 | **6107.07MB** |
| **MEGA** | **33.57** | **0.0442** | **0.1014** | 83.14 | **32.45MB** |

MEGA相比4DGS实现 **190×存储压缩**（6107→32MB），PSNR提升1.5dB，FPS提升50%。相比SOTA方法STG，PSNR高0.22dB且存储更小。

### 主实验二：Neural 3D Video数据集

| 方法 | PSNR↑ | DSSIM2↓ | FPS↑ | 存储↓ |
|------|-------|---------|------|-------|
| MixVoxels-X | 31.73 | 0.0150 | 4.60 | 500.00MB |
| Dynamic 3DGS | 30.46 | 0.0190 | 460.00 | 2772.00MB |
| STG | 32.04 | 0.0145 | 273.47 | 175.35MB |
| 4DGS | 31.57 | 0.0164 | 96.69 | 3128.00MB |
| **MEGA** | **31.49** | **0.0165** | 77.42 | **25.05MB** |

在Neu3DV上实现 **125×存储压缩**（3128→25MB），视觉质量与4DGS相当。

### 消融实验：各组件贡献（Birthday / Fabien / Flame Steak / Sear Steak）

| 变体 | PSNR (Birthday) | Gaussian数 | 参数量 |
|------|-----------------|-----------|--------|
| 4DGS基线 | 31.00 | 13.00M | 2094M |
| w/ grid替代SH | 30.49 | 16.33M | 293M |
| w/ DAC | 31.60 | 15.43M | 309M |
| w/ DAC + 形变 | 31.35 | 15.75M | 315M |
| w/ DAC + $\mathcal{L}_{opa}$ | 31.46 | 9.15M | 183M |
| **w/ DAC + 形变 + $\mathcal{L}_{opa}$** | **32.02** | **0.91M** | **18M** |

- grid方法（直接替代SH）导致明显性能下降（-0.51dB），而DAC保持甚至提升质量（+0.60dB）
- 单独使用形变→Gaussian反而增多；单独使用 $\mathcal{L}_{opa}$ →限制了Gaussian的表达能力
- 二者结合：Gaussian从13M降至0.91M（**14倍减少**），PSNR反而提升1.02dB

## 亮点与洞察

1. **DC-AC分解的直觉优美**：将颜色类比为电信号——DC保留主体色、AC编码变化——既有效又优雅地替代了144维SH
2. **形变+熵约束的协同效应**：形变扩大作用范围使每个Gaussian更有价值，熵损失移除冗余Gaussian，二者结合（而非单独使用）才能同时减少数量和保持质量
3. **惊人的压缩比**：190×（Technicolor）和125×（Neu3DV）的存储压缩，且几乎无质量损失
4. 对AR/VR端侧部署有重要实际意义——将GB级模型压缩到几十MB

## 局限性

1. AC预测器和形变预测器都是共享MLP，对极端复杂场景可能成为容量瓶颈
2. 在Neu3DV上PSNR略低于4DGS（31.49 vs 31.57），压缩存在微小的质量损失
3. 仅在两个数据集上验证，缺少户外动态场景和更多样化的测试
4. 形变使用乘法操作，可能限制了某些形变模式的表达能力

## 相关工作

- **静态3D表示压缩**：Compact-3DGS（冗余剪枝+SH蒸馏）、LightGaussian（矢量量化）
- **动态场景NeRF方法**：DyNeRF、HyperReel、HexPlane、K-Planes
- **动态场景Gaussian方法**：Deformable 3DGS（正则+形变场）、4DGS（4D时空Gaussian）、STG（时空Gaussian）、E-D3DGS

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 5 |
| 总评 | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](4d_gaussian_splatting_slam.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](../../CVPR2026/3d_vision/motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](../../CVPR2025/3d_vision/efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[CVPR 2026\] Layered 4D-Rotor Gaussian Splatting: A Compressed Representation for Long Dynamic Scenes](../../CVPR2026/3d_vision/layered_4d-rotor_gaussian_splatting_a_compressed_representation_for_long_dynamic.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

</div>

<!-- RELATED:END -->

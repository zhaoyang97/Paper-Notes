---
title: >-
  [论文解读] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh
description: >-
  [CVPR 2025][3D视觉][3D Gaussian Splatting] Mani-GS 提出了一种基于三角网格操控 3D Gaussian Splatting 的方法——通过在每个三角形上定义局部坐标系来绑定高斯，使得网格变形时高斯的位置、旋转和缩放能自适应调整，从而实现大变形、局部编辑和软体仿真等多种操控类型，同时保持高质量渲染且对网格精度有高容忍度。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "网格操控"
  - "三角形局部坐标系"
  - "可编辑渲染"
  - "软体仿真"
---

# Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh

**会议**: CVPR 2025  
**arXiv**: [2405.17811](https://arxiv.org/abs/2405.17811)  
**代码**: 暂未公开（参见项目页面）  
**领域**: 3D视觉 / 3DGS编辑  
**关键词**: 3D Gaussian Splatting, 网格操控, 三角形局部坐标系, 可编辑渲染, 软体仿真

## 一句话总结
Mani-GS 提出了一种基于三角网格操控 3D Gaussian Splatting 的方法——通过在每个三角形上定义局部坐标系来绑定高斯，使得网格变形时高斯的位置、旋转和缩放能自适应调整，从而实现大变形、局部编辑和软体仿真等多种操控类型，同时保持高质量渲染且对网格精度有高容忍度。

## 研究背景与动机

1. **领域现状**：3DGS 凭借高保真实时渲染能力迅速成为 3D 表示领域的明星。然而作为显式点云表示，目前缺乏有效的操控方法——用户想要拉伸、弯曲、局部编辑 3DGS 场景时缺少直观工具。
2. **现有痛点**：(1) NeRF-based 编辑方法（NeRF-Editing、NeuMesh）训练推理慢且渲染质量有限; (2) SuGaR 将高斯严格平贴在三角面上（flat binding），强烈依赖网格精度——网格不准确处会产生缺失或多余的几何; (3) 简单给高斯加位置 offset 虽然能改善静态渲染，但 offset 是针对特定视角优化的，变形后会产生噪声和失真; (4) 不同类型的操控（大变形/局部/仿真）需要设计不同算法，使用麻烦
3. **核心矛盾**：如何让高斯在自由浮动（高渲染质量）和与网格严格绑定（可操控）之间找到平衡？
4. **本文目标**：设计一种通用的 Mesh-Gaussian 绑定策略，使高斯能偏离三角面（允许补偿网格不精确），同时在网格变形后自适应调整（保持相对空间关系）。
5. **切入角度**：在每个三角形上建立局部坐标系（Local Triangle Space），在该局部空间中优化高斯属性，变形时只需更新三角形的全局坐标系变换，局部属性保持不变。
6. **核心 idea**：将高斯绑定到三角形的局部坐标系中优化，变形时通过坐标系变换自适应调整高斯的全局位置、旋转和缩放，实现高质量可操控渲染。

## 方法详解

### 整体框架
三阶段流程：(1) 网格提取——从 3DGS（Marching Cube / Screened Poisson）或 NeuS 提取三角网格; (2) 高斯绑定与优化——在每个三角形上绑定 N=3 个高斯，在局部三角空间中优化属性; (3) 操控——通过 Blender 等工具编辑网格，3DGS 自动适应。

### 关键设计

1. **Local Triangle Space（三角形局部坐标系）**:
    - 功能：为每个三角形建立一个随其变形而变换的局部坐标系，在该空间中定义高斯属性
    - 核心思路：给定三角形三个顶点 $(v_1, v_2, v_3)$，定义三轴——第一轴沿第一条边 $r_1^t = (v_2 - v_1)/\|v_2 - v_1\|$，第二轴为法线方向 $r_2^t = n^t$，第三轴为前两者叉积 $r_3^t = r_1^t \times n^t$。组成旋转矩阵 $R^t$。高斯在局部空间中的位置 $\mu^l$ 和旋转 $R^l$ 被优化，全局属性通过变换得到：$R = R^t R^l$, $\mu = R^t \mu^l + \mu^t$
    - 设计动机：普通 offset 方案的 offset 是在全局空间中定义的，变形后不能适配；而局部坐标系中的属性在变形时保持不变，只需更新全局变换即可自适应

2. **Triangle Shape Aware Adaption（三角形形状感知自适应）**:
    - 功能：当三角形形状因变形而改变时（拉伸/压缩），自动调整高斯的缩放和位置
    - 核心思路：定义自适应向量 $e = [e_1, e_2, e_3]$，其中 $e_1$ 为第一条边长度 $l_1$，$e_3$ 为第二三条边平均长度 $0.5(l_2+l_3)$，$e_2 = 0.5(e_1+e_3)$。变形后的全局属性为 $s = \beta \cdot e \cdot s^l$, $\mu = e \cdot R^t \mu^l + \mu^t$（$\beta$ 为超参数）。当三角形放大时，高斯的缩放和距离等比放大
    - 设计动机：如果不做形状感知自适应，三角形放大后原来紧密覆盖表面的高斯会出现间隙，反之压缩后会过度重叠。等比缩放确保高斯始终与三角面积匹配

3. **多种网格提取方式的兼容（高网格容忍度）**:
    - 功能：即使使用低精度网格也能获得高质量渲染
    - 核心思路：支持三种网格来源——(1) 3DGS Marching Cube（质量最低，边界膨胀）; (2) Screened Poisson（中等，可能有断连区域）; (3) NeuS（最高质量但慢）。由于高斯可以在局部空间中偏离三角面，即使网格不准确也能通过自由浮动的高斯补偿几何缺陷
    - 设计动机：SuGaR 的 flat binding 完全继承了网格的缺陷——网格缺失=渲染缺失、网格多余=渲染多余。Mani-GS 允许高斯"飘出"三角面，在不精确网格上也能实现高保真渲染

### 损失函数 / 训练策略
两阶段训练：第一阶段 30K iterations 提取网格（或使用已有网格），第二阶段 20K iterations 在网格上绑定高斯并用多视角渲染损失（L1 + SSIM）优化。每个三角形初始化 N=3 个高斯，均匀分布在面上。网格通过 decimation 减少到约 300K 三角形。单 A100 GPU 训练。

## 实验关键数据

### 主实验

**NeRF Synthetic 静态渲染**:

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NeRF-Editing | — | — | — |
| SuGaR | 32.06 | 0.958 | 0.039 |
| Ours (SuGaR mesh) | 33.52 | 0.966 | 0.030 |
| Ours (NeuS mesh) | 33.65 | 0.966 | 0.031 |

**DTU 真实数据**:

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NeuMesh | 28.29 | 0.921 | 0.117 |
| SuGaR | 30.06 | 0.921 | 0.136 |
| Ours | 31.50 | 0.943 | 0.088 |

使用相同 SuGaR mesh 时仍超越 SuGaR 约 1.5 PSNR。

### 消融实验

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3DGS On Mesh (flat binding) | 30.87 | 0.9521 | 0.0447 |
| Mesh + Offset | 32.48 | 0.9625 | 0.0341 |
| Ours + Marching Cube Mesh | 32.11 | 0.9602 | 0.035 |
| Ours + Poisson Mesh | 33.42 | 0.9638 | 0.0324 |
| Ours + NeuS Mesh | 33.45 | 0.9646 | 0.0309 |

### 关键发现
- **Local Triangle Space 是核心**：flat binding（30.87 PSNR）vs 本方法（33.45 PSNR）差 2.6 PSNR，证明让高斯偏离三角面自由优化至关重要
- **Offset 方案静态好但变形差**：Mesh+Offset 在静态渲染中 PSNR=32.48 尚可，但变形后产生严重噪声和失真，因为 offset 不具有变形不变性
- **高网格容忍度**：使用最差的 Marching Cube mesh（32.11）仍比 flat binding 用最好网格（30.87）好 1.2 PSNR
- **NeuS mesh vs Poisson mesh 差异很小**（33.45 vs 33.42），说明方法对网格来源不敏感
- 支持大变形（拉伸/弯曲）、局部编辑（调味酱混合、钹重定位）、软体仿真全部通过网格传递实现

## 亮点与洞察
- **局部坐标系设计优雅且通用**：在每个三角形上定义局部空间，核心公式 $R = R^t R^l$, $\mu = e \cdot R^t \mu^l + \mu^t$ 简洁清晰，实现了绑定与自由浮动的最佳平衡。这种思路可迁移到任何需要将点云/粒子绑定到变形网格上的场景（如布料模拟、角色动画）
- **统一的操控框架**：不需要为大变形、局部编辑、物理仿真单独设计算法——只要能操控三角网格，3DGS 就自动跟随。变形后零额外计算（只需矩阵乘法更新全局属性）
- **对网格精度的高容忍度**：现实中高精度网格提取仍然困难，本方法允许使用粗糙网格，极大降低了应用门槛

## 局限与展望
- 作者承认：当局部区域发生高度非刚性变形时仍会出现渲染失真，因为局部刚性假设被破坏
- 物理仿真在 >35K 三角形的网格上可能需要数小时
- 目前不支持拓扑变化（如切割、合并）
- 每个三角形只绑定 3 个高斯——对于细节丰富的区域可能不够，自适应增减高斯数量可能更优
- 训练需要先提取网格再优化高斯，两阶段流程不够端到端
- 论文未与 PhysGaussian（基于物理的 3DGS 仿真）做定量对比

## 相关工作与启发
- **vs SuGaR**: SuGaR 的 flat binding 将高斯完全限制在三角面上，继承了网格所有缺陷; Mani-GS 在局部空间中允许自由偏移，同 mesh 下 PSNR 高 1.5+
- **vs GaMeS**: GaMeS 也约束高斯为 flat ellipse 且在三角面内; Mani-GS 允许高斯偏离表面且保持自适应旋转
- **vs NeRF-Editing**: NeRF-Editing 需要构建四面体网格做逆向映射，训练推理都慢; Mani-GS 基于显式 3DGS 操控即时完成且渲染更优
- **vs NeuMesh**: NeuMesh 在 mesh 顶点上定义辐射场并显式变形渲染，细节模糊; Mani-GS 利用 3DGS 的高斯 splatting 保持丰富细节

## 评分
- 新颖性: ⭐⭐⭐⭐ 三角形局部坐标系 + 形状感知自适应设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 合成数据+真实数据完整消融，多种操控类型展示充分，但缺少定量的操控后渲染对比
- 写作质量: ⭐⭐⭐⭐ 方法公式清晰，图示直观，但部分细节需看附录
- 价值: ⭐⭐⭐⭐ 为 3DGS 编辑提供了实用通用的解决方案，对 3D 内容创作有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[CVPR 2025\] Ref-GS: Directional Factorization for 2D Gaussian Splatting](ref-gs_directional_factorization_for_2d_gaussian_splatting.md)
- [\[CVPR 2025\] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

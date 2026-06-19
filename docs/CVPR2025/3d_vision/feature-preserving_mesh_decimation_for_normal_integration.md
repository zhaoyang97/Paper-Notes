---
title: >-
  [论文解读] Feature-Preserving Mesh Decimation for Normal Integration
description: >-
  [CVPR 2025][3D视觉][mesh decimation] 将经典的 quadric error metric（QEM）推导到屏幕空间并以法线贴图为输入，结合最优 Delaunay 三角化实现各向异性网格简化，在 90%+ 压缩率下仍保持亚毫米级精度，将高分辨率法线积分从小时级加速到分钟级。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "mesh decimation"
  - "normal integration"
  - "quadric error metric"
  - "anisotropic meshing"
  - "Delaunay triangulation"
---

# Feature-Preserving Mesh Decimation for Normal Integration

**会议**: CVPR 2025  
**arXiv**: [2504.00867](https://arxiv.org/abs/2504.00867)  
**代码**: [https://moritzheep.github.io/anisotropic-screen-meshing](https://moritzheep.github.io/anisotropic-screen-meshing)  
**领域**: 3D视觉  
**关键词**: mesh decimation, normal integration, quadric error metric, anisotropic meshing, Delaunay triangulation

## 一句话总结

将经典的 quadric error metric（QEM）推导到屏幕空间并以法线贴图为输入，结合最优 Delaunay 三角化实现各向异性网格简化，在 90%+ 压缩率下仍保持亚毫米级精度，将高分辨率法线积分从小时级加速到分钟级。

## 研究背景与动机

**领域现状**: 法线积分通过估计的法线贴图（如光度立体法获取）重建 3D 表面，通常在像素级稠密网格上求解 Poisson 方程。法线贴图能捕获像素级表面细节，但高分辨率下计算代价随像素数二次增长。

**现有痛点**:
- 像素级稠密网格在高分辨率下（如 16MP）导致法线积分运行时间达数小时
- 分辨率翻倍需要同时翻倍图像宽高，变量数呈二次增长
- 已有的自适应三角网格方法 [Heep et al.] 只能做各向同性简化（等边三角形），无法在脊线和沟槽处精确对齐
- 根本问题：规则网格中细节区需要高分辨率，但这会强制平坦区也保持同样高的分辨率

**核心矛盾**: 在像素网格中，局部的细节精度需求迫使全局分辨率升高；而三角网格虽允许局部自适应，但各向同性简化在几何特征处精度不足。

**本文目标**: 设计一种各向异性网格简化算法，使三角网格的顶点和边沿着表面脊线/沟槽对齐，在极高压缩率下仍保持几何精度。

**切入角度**: 将 3D 网格简化中的 QEM 推导到 2D 屏幕空间（用法线贴图代替 3D 几何），并利用广义 Delaunay 三角化实现边的各向异性对齐。

**核心 idea**: 在法线积分之前，用法线贴图驱动的屏幕空间 QEM + 广义 Delaunay 实现各向异性网格预处理，以极少顶点忠实保留几何特征。

## 方法详解

### 整体框架

1. 将前景像素网格初始化为稠密三角网格（每个像素两个三角形）
2. 迭代执行三步：边折叠（去除冗余顶点）→ 边翻转（对齐脊线）→ 顶点重定位（精确定位）
3. 在简化后的稀疏三角网格上进行法线积分（求解线性系统得到深度图）

目标函数: $E = E_{Geo} + \lambda \cdot E_{ODT}$，$E_{Geo}$ 估计网格与底层表面的偏差（QEM），$E_{ODT}$ 保证平坦区域的均匀分布（Delaunay 正则化）。

### 关键设计

**1. 屏幕空间 Quadric Error Metric**
- **功能**: 从法线贴图直接推导 QEM，而非依赖已知的 3D 几何。通过法线计算表面切线（Jacobian $J_f$），将 3D 表面积分转化为屏幕坐标积分。
- **核心思路**: 对每个顶点 $v$，计算其 quadric $Q_v(\delta\vec{x}) = \sum_{f \in \mathcal{F}_v} \frac{A_f^{(3)}}{|\mathcal{P}_f|} \sum_{p \in \mathcal{P}_f} \|J_f \delta\vec{u}_{vp} + \delta\vec{x}\|_{M_p}^2$，其中 $M_p = \vec{n}_p \cdot \vec{n}_p^t + \lambda \cdot \mathbb{1}$ 统一了几何误差和 Delaunay 正则化。
- **设计动机**: 运行时优势依赖于在积分之前简化网格，因此不能等到 3D 几何已知后再简化。法线→切线→Jacobian 的关系链让整个计算在屏幕空间即可完成。

**2. 广义 Delaunay 边翻转（Edge Alignment）**
- **功能**: 利用 $\|\cdot\|_M$ 范数扩展标准 Delaunay 三角化到各向异性场景，通过局部四面体判据决定是否翻转边。
- **核心思路**: 对每条非边界边 $e$，将相邻四个顶点投影到切平面得到扁平近似，用 $\|\cdot\|_{M_e}$ 将其"提升"成四面体。若当前边在四面体中不是最低边（即另一条对角线更"几何对齐"），则翻转。
- **设计动机**: 标准 Delaunay 是各向同性的，无法让边沿着脊线/沟槽对齐。广义范数 $M_e$ 中的法线外积项 $\vec{n}\cdot\vec{n}^t$ 引入了各向异性——在法线变化大（特征处）的方向上拉伸，使边自然对齐几何特征。

**3. QEM 驱动的边折叠（Decimation）**
- **功能**: 用 quadric 代价 $C_{vw} = \min_{\vec{u}_{vw}} \tilde{Q}_v(\vec{u} - \vec{u}_v) + \tilde{Q}_w(\vec{u} - \vec{u}_w)$ 衡量每次折叠对表面形状的影响，优先折叠代价最小的边。
- **核心思路**: 将所有可折叠边放入优先队列（按代价升序），折叠后将两端点 quadric 相加作为新顶点 quadric。
- **设计动机**: 沿用 Garland & Heckbert 的经典设计，但 quadric 是从法线贴图推导的屏幕空间版本。

### 损失函数 / 训练策略

- 非学习方法，无需训练
- 超参数 $\lambda = 10^{-5}$（几何误差与 Delaunay 正则化的权衡），$\alpha = 0.5$（顶点重定位步长）
- 5 轮迭代：每轮先边折叠再做边翻转和顶点对齐
- 支持两种分辨率控制：目标顶点数或折叠代价阈值
- C++ 实现，用 nvdiffrast 做三角形-像素映射，Eigen 求解线性系统

## 实验关键数据

### 主实验（DiLiGenT-MV 数据集，20 视图平均 RMSE，mm）

| 物体 | 像素基准 [2] | 各向同性 low | 各向同性 high | **Ours low** | **Ours high** |
|---|---|---|---|---|---|
| Bear | 2.97 | 3.95 | 3.37 | **3.84** | **3.04** |
| Buddha | 6.74 | 7.74 | 7.33 | **6.86** | **6.61** |
| Cow | 2.45 | 3.42 | 2.96 | **3.07** | **2.74** |
| Pot2 | 5.15 | 5.89 | 5.65 | **5.63** | **5.29** |
| Reading | 6.34 | 7.08 | 6.83 | **6.82** | **6.50** |

### 消融实验

| 配置 | 效果 |
|---|---|
| 仅边折叠 | 保留粗糙形状，高频细节丢失 |
| 折叠 + 顶点对齐 | RMSE 改善，顶点定位到特征处 |
| 折叠 + 边对齐 | MAE 改善，边沿脊线/沟槽对齐 |
| **完整算法** | RMSE 和 MAE 均最优 |

### 关键发现

1. **90%+ 压缩仍亚毫米精度**: 即使压缩超 95%（顶点数仅为像素数的 5%），精细表面细节仍清晰可见。
2. **各向异性优于各向同性**: 在相同顶点数下，本方法在 RMSE 和 MAE 上始终优于各向同性方法，且视觉上保留了更多细节（如熊的鼻子、佛像面部）。
3. **"少即是多"**: 各向同性方法在脊线处堆积更多顶点但精度反而更低，因为缺乏边对齐；本方法用更少顶点但更好的布局实现更高精度。
4. **16MP 法线贴图可处理**: 在相同时间内，像素方法只能处理 4MP，本方法可三角化+简化+积分 16MP。

## 亮点与洞察

- 将经典 QEM 从 3D 几何推导到屏幕空间法线贴图的理论贡献干净优雅
- 发现 QEM 和广义 Delaunay 在统一范数 $\|\cdot\|_M$ 下的内在联系，将几何保持和边对齐统一到同一框架
- 仅 3 种局部操作（折叠、翻转、重定位）的简洁算法设计
- 反直觉发现：在特征处用更少但更精确定位的顶点优于密集但各向同性的顶点

## 局限与展望

- 弱透视投影假设（假设恒定的相机-物体距离）可能在大深度变化场景中引入误差
- 法线贴图的质量直接决定简化质量，对噪声法线的鲁棒性未充分讨论
- C++ 实现效率尚可但未利用 GPU 并行
- 仅针对法线积分场景，未扩展到其他网格简化应用
- 低分辨率数据集 [DiLiGenT-MV, LUCES] 或缺乏 GT 几何 [PS, RGBN] 限制了评估的全面性

## 相关工作与启发

- Garland & Heckbert 的 QEM 是网格简化的经典方法；本文首次将其推导到无 3D 几何的屏幕空间
- CVT/ODT 的各向异性扩展通常需要将顶点和法线"提升"到 6D 空间；本方法更直接地在 2D 屏幕空间实现
- 启发：法线作为一种丰富的几何先验，可以驱动更多传统几何处理算法的"前置化"

## 评分

⭐⭐⭐⭐ — 理论推导优雅，实验效果显著，但应用场景相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[CVPR 2026\] Variational Graph-based Normal Integration](../../CVPR2026/3d_vision/variational_graph-based_normal_integration.md)
- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [\[CVPR 2025\] Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)
- [\[CVPR 2025\] Identity-preserving Distillation Sampling by Fixed-Point Iterator](identity-preserving_distillation_sampling_by_fixed-point_iterator.md)

</div>

<!-- RELATED:END -->

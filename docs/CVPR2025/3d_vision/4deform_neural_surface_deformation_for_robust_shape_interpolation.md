---
title: >-
  [论文解读] 4Deform: Neural Surface Deformation for Robust Shape Interpolation
description: >-
  [CVPR 2025][3D视觉][neural implicit representation] 提出 4Deform 框架，基于神经隐式表示和连续速度场学习实现鲁棒形状插值，通过修改的 level-set 方程链接隐式场与速度场，首次在噪声、部分、拓扑变化和非等距变形场景中均取得 SOTA，并支持真实世界 Kinect 点云序列的时间超分辨率。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "neural implicit representation"
  - "shape interpolation"
  - "velocity field"
  - "level-set equation"
  - "deformation"
---

# 4Deform: Neural Surface Deformation for Robust Shape Interpolation

**会议**: CVPR 2025  
**arXiv**: [2502.20208](https://arxiv.org/abs/2502.20208)  
**代码**: 无  
**领域**: 其他  
**关键词**: neural implicit representation, shape interpolation, velocity field, level-set equation, deformation

## 一句话总结
提出 4Deform 框架，基于神经隐式表示和连续速度场学习实现鲁棒形状插值，通过修改的 level-set 方程链接隐式场与速度场，首次在噪声、部分、拓扑变化和非等距变形场景中均取得 SOTA，并支持真实世界 Kinect 点云序列的时间超分辨率。

## 研究背景与动机

### 领域现状
**领域现状**：形状插值（Shape Interpolation）是计算机视觉和图形学中的基础任务，需要从稀疏的离散观测中恢复连续的 3D 运动。现有方法主要分为基于网格的方法（SMS、Neuromorph、LIMP 等，依赖预定义拓扑和密集精确对应关系）和基于神经隐式场的方法（NISE、LipMLP 等，通过隐式表示实现灵活的拓扑变化）。

### 现有痛点与挑战
**现有痛点**：(1) **网格方法的拓扑局限**——要求固定顶点连接关系，无法处理拓扑变化（如人与物体交互）、噪声或部分观测；输出分辨率受限于输入网格（通常降采样到 2500-5000 顶点）；(2) **隐式场方法的物理合理性不足**——潜空间插值方法（NISE、LipMLP）不考虑物理约束，产生不合理的中间形状；(3) **对应关系依赖**——多数方法要求精确的点对点对应，实际中难以获取；(4) **单对训练的扩展性差**——优化方法需为每对形状单独训练，无法处理序列数据。

**核心矛盾**：需要同时处理噪声/部分/拓扑变化/非等距变形的通用形状插值方法，且仅需粗糙不完整的对应关系。

### 研究目标
**本文目标**：解决上述核心问题，提出新的方法在关键指标上取得显著提升。

**核心 idea**：提出 4Deform 框架，基于神经隐式表示和连续速度场学习实现鲁棒形状插值，通过修改的 level-set 方程链接隐式场与速度场，首次在噪声、部分、拓扑变化

## 方法详解

### 整体框架
4Deform 采用 AutoDecoder 架构：(1) 通过匹配模块估计稀疏对应关系；(2) 用神经隐式场表示形状的时变符号距离函数 $\phi(x,t)$；(3) 用神经速度场 $\mathcal{V}(x,t)$ 建模变形；(4) 通过修改的 level-set 方程连接隐式场和速度场。训练时联合优化潜向量和网络参数，推理时可通过插值潜向量生成新序列。

### 关键设计

1. **修改的 Level-Set 方程**：

    - 功能：在隐式表示中直接驱动形状沿速度场变形
    - 核心思路：标准 level-set 方程 $\partial_t\phi + \mathcal{V}^\top \nabla\phi = 0$ 描述零水平集如何沿速度场移动。为保持符号距离函数性质，添加 Eikonal 正则化得 $\partial_t\phi + \mathcal{V}^\top \nabla\phi = -\lambda_l \phi \mathcal{R}(x,t)$，其中 $\mathcal{R}$ 为基于速度场梯度的再初始化项
    - 设计动机：直接将隐式表示与速度场关联，无需显式的表面点操作，天然支持拓扑变化

2. **物理与几何约束损失**：

    - 功能：确保生成的中间形状物理合理
    - 核心思路：引入两类新损失——(a) **空间平滑正则** $\mathcal{L}_s = \int \|(-\alpha\Delta + \gamma I)\mathcal{V}\|^2 dx$，防止速度场剧烈变化；(b) **体积保持约束** $\mathcal{L}_v = \int |\nabla \cdot \mathcal{V}| dx$，通过散度最小化防止形变过程中的体积变化
    - 设计动机：物理约束使插值结果在缺乏中间形状监督时仍然合理——仅用起止形状即可生成可信的中间帧

3. **基于全局描述向量的 AutoDecoder 架构**：

    - 功能：支持序列表示和外推
    - 核心思路：为每个点云分配一个可优化的潜向量 $z$，将相邻帧的潜向量拼接后作为网络的条件输入。训练时联合优化所有潜向量和网络权重。推理时通过线性插值潜向量实现新时间步的形状生成
    - 设计动机：AutoDecoder 不需要编码器前向传播，轻量且适合小数据集训练

## 实验关键数据

### 主实验：DFAUST 数据集形状插值

| 方法 | Chamfer-L1 ↓ | 法向量一致性 ↑ | 对应误差 ↓ | 支持拓扑变化 |
|------|-------------|--------------|-----------|-------------|
| Neuromorph | 0.82 | 0.89 | 3.2 | ✗ |
| LIMP | 1.15 | 0.84 | 4.1 | ✗ |
| NISE | 1.45 | 0.81 | - | ✓ |
| LipMLP | 1.23 | 0.83 | - | ✓ |
| **4Deform** | **0.65** | **0.93** | **2.1** | **✓** |

### 消融实验

| 配置 | Chamfer-L1 | 说明 |
|------|-----------|------|
| Full 4Deform | 0.65 | 完整方法 |
| w/o 物理约束 | 1.12 | +72% 退化 |
| w/o Eikonal 正则 | 0.89 | +37% 退化 |
| w/o 潜向量 | 0.95 | +46% 退化 |

### 泛化性验证

| 场景 | 效果 | 说明 |
|------|------|------|
| 噪声点云 | ✓ 鲁棒 | 隐式表示天然去噪 |
| 部分观测 | ✓ 补全 | 隐式场可外推 |
| 拓扑变化 | ✓ 处理 | Level-set 不依赖固定拓扑 |
| 非等距变形 | ✓ 支持 | 速度场无刚性假设 |
| Kinect 真实数据 | ✓ 首次 | 超分辨率到密集网格 |

### 关键发现
- 物理约束是最关键的组件——移除后 Chamfer-L1 退化 72%
- 首次在真实 Kinect 点云上实现形状插值——从噪声稀疏点云到密集网格
- 仅需稀疏粗糙对应关系即可工作——对匹配质量鲁棒

## 亮点与洞察
- 方法设计巧妙，核心思路清晰，解决了领域中的关键痛点
- 实验全面覆盖多个数据集和场景，验证了方法的有效性和鲁棒性
- 消融实验清晰展示了各模块的独立贡献

## 局限与展望
- 方法在更大规模数据和更复杂场景中的泛化性有待进一步验证
- 计算效率可进一步优化以支持实时应用
- 与其他相关方法的深入对比和互补性分析值得探索

## 相关工作与启发
- 本文方法相对于同领域代表性方法有明显的改进和创新
- 技术路线对后续相关工作有重要参考价值
- 核心模块设计可推广到更广泛的应用场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 方法设计有独特贡献
- 实验充分度: ⭐⭐⭐⭐ 多数据集全面验证
- 写作质量: ⭐⭐⭐⭐ 条理清晰
- 价值: ⭐⭐⭐⭐ 对领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](../../ICCV2025/3d_vision/neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[CVPR 2025\] Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[CVPR 2025\] Toward Robust Neural Reconstruction from Sparse Point Sets](toward_robust_neural_reconstruction_from_sparse_point_sets.md)
- [\[CVPR 2025\] MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)

</div>

<!-- RELATED:END -->

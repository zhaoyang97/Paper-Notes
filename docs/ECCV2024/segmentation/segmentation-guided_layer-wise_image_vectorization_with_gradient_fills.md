---
title: >-
  [论文解读] Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills
description: >-
  [ECCV 2024][语义分割][图像矢量化] 提出分割引导的矢量化框架，通过梯度感知分割子程序引导 Bézier 路径的初始化和优化，首次在保持分层拓扑的逐层矢量化方法中支持径向渐变填充，使矢量图形在更少路径数下达到更高的视觉质量。 矢量图形（SVG）在数字设计中广受欢迎（任意缩放、易编辑）。光栅化转矢量化（Raster…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "图像矢量化"
  - "分割引导"
  - "渐变填充"
  - "可微渲染"
  - "分层拓扑"
---

# Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills

**会议**: ECCV 2024  
**arXiv**: [2408.15741](https://arxiv.org/abs/2408.15741)  
**代码**: 无（基于 DiffVG 和 PyTorch 实现）  
**领域**: 分割 (图像矢量化)  
**关键词**: 图像矢量化, 分割引导, 渐变填充, 可微渲染, 分层拓扑

## 一句话总结

提出分割引导的矢量化框架，通过梯度感知分割子程序引导 Bézier 路径的初始化和优化，首次在保持分层拓扑的逐层矢量化方法中支持径向渐变填充，使矢量图形在更少路径数下达到更高的视觉质量。

## 研究背景与动机

矢量图形（SVG）在数字设计中广受欢迎（任意缩放、易编辑）。光栅化转矢量化（Rasterization-to-Vectorization）是生成矢量图形的重要途径。

现有矢量化方法的分类与局限：

| 类别 | 代表方法 | 优点 | 局限 |
|------|---------|------|------|
| 网格方法 | 三角形/矩形网格 | 近照片级真实感 | 基元复杂，丧失层级结构 |
| 曲线方法 | Diffusion curves | 高保真度 | 不直观，难编辑 |
| 学习方法 | SketchRNN, Im2Vec | 保留拓扑 | 依赖训练数据，泛化性差 |
| **LIVE** | DiffVG 优化 | 逐层拓扑，无需模型 | **不支持渐变填充** |

**LIVE** 是最近最相关的工作——基于可微渲染器 DiffVG 的逐层矢量化框架，但仅支持纯色填充。对于包含渐变效果的图像，LIVE 需要添加大量冗余路径来近似渐变，而且**简单地将纯色参数替换为渐变参数并不可行**——需要有效的方法确定哪些像素对路径的渐变填充有贡献。

## 方法详解

### 整体框架

渐进式矢量化：每个 epoch 添加一个或多个 Bézier 路径并优化。

在每个 epoch $i$：
1. 计算输入图像与当前输出 $I_{i-1}$ 的差异
2. 使用梯度感知分割确定 $n_i$ 个区域
3. 初始化新路径（Bézier 曲线 + 径向渐变）
4. 优化所有路径的几何参数和渐变参数

### 关键设计

**梯度感知分割（Gradient-aware Segmentation）**：

核心洞察：同一渐变填充内部颜色平滑变化，边界处颜色突变。因此用**二阶空间梯度**（Laplacian 滤波器）检测渐变边界：

$$S_0 = \text{correlate}((I - \hat{I})\mathbf{1}_{\|\hat{I}-I\|_2 > \epsilon}, L)$$

其中 $L$ 为离散 Laplacian 滤波器 $\begin{bmatrix}1&1&1\\1&-8&1\\1&1&1\end{bmatrix}$。

后续步骤：
- RGB 通道取绝对值求和 → 灰度图
- Otsu 自适应阈值二值化
- 形态学闭运算 + 分水岭算法 → 最终分割

**关键优势**：随着路径逐步添加，整体差异递减，Otsu 自适应阈值也相应下降——大的渐变区域先被拟合，小的细节（如脸红效果）后续自动被分割出来。

**分割引导初始化**：

选择**累积平方误差最大**的分割区域初始化新路径：

$$w_i = \sum_{p \in \tilde{S}[i]} \|I_p - \hat{I}_p\|^2$$

优先选大区域（鼓励层级化），避免选已拟合好的区域。

路径初始化为4段三次 Bézier 曲线组成的圆形，填充径向渐变：
- 圆心：区域的质心
- 直径：区域包围盒宽高的几何均值（裁剪到 [0.2, 1.0]）
- 两个色停（0% 和 100%）初始化为输入图像在质心处的颜色

### 损失函数 / 训练策略

**分割引导损失（SG Loss）**：

LIVE 使用 UDF（无符号距离场）权重聚焦路径轮廓像素。但对渐变来说，轮廓颜色正确不意味内部颜色正确。本文扩展 UDF 权重到路径覆盖的所有未被遮挡像素：

$$w_{\text{SG}}(i) = \begin{cases}\max(d_i', \alpha_s), & i \in F \\ d_i'(1-\alpha_s), & \text{otherwise}\end{cases}$$

其中 $F$ 为所有"聚焦像素"的集合（路径覆盖区域与分割区域的交集），$\alpha_s = 0.6$ 平衡 UDF 权重和分割权重。

$$\mathcal{L}_{\text{SG}} = \frac{1}{3}\sum_{i=1}^{w \times h} w_{\text{SG}}(i)\sum_{c=1}^{3}(I_{c,i} - \hat{I}_{c,i})^2$$

**Xing 损失**：防止 Bézier 曲线自交叉，惩罚控制点 $\vec{AB}$ 与 $\vec{CD}$ 的夹角超过 180°。

**总损失**：

$$\mathcal{L} = \mathcal{L}_{\text{SG}} + \lambda\mathcal{L}_{\text{Xing}}, \quad \lambda = 0.05$$

优化器：Adam，学习率 $10^{-2}$（渐变参数）和 1（路径控制点）。

## 实验关键数据

### 主实验（表格）

用户研究结果（用户偏好百分比）：

| 数据集 | 路径数 | LIVE 偏好 | 本文偏好 |
|--------|--------|----------|---------|
| Noto Emoji | 整体 | 40.4% | **59.6%** |
| Fluent Emoji | 整体 | 34.7% | **65.3%** |
| Iconfont | 整体 | 42.1% | **57.9%** |

渐变丰富的 Fluent Emoji 上优势最大（65.3% vs. 34.7%），路径数越少优势越明显。

### 消融实验（表格）

在 Noto Emoji 上的 PSNR 比较(数值越高越好)：

| 配置 | 8路径 | 16路径 | 32路径 | 64路径 |
|------|-------|--------|--------|--------|
| LIVE (无渐变, 无分割引导) | 基线 | 基线 | 基线 | 基线 |
| + 渐变 (无引导) | 下降 | 下降 | 接近 | 接近 |
| + 渐变 + 分割引导 | **最高** | **最高** | **最高** | 接近 |

少路径时本文方法 PSNR 显著更高；路径数充足时两方法趋于一致。

### 关键发现

1. **少路径时优势巨大**：仅 8 条路径即可矢量化 emoji 的主要元素（眼睛、嘴巴等），LIVE 需要更多路径
2. **分割引导至关重要**：没有分割引导直接优化渐变参数会导致性能退化
3. **自适应阈值的优势**：Otsu 动态阈值使框架无需假设输入类型，避免了超参数
4. 3个数据集（Noto Emoji、Fluent Emoji、Iconfont）上用户普遍偏好本文方法
5. 框架是 model-free 的，不依赖任何训练数据

## 亮点与洞察

1. **渐变检测 ≈ 分割问题**：将"确定哪些像素对渐变有贡献"建模为分割任务，是连接两个看似无关领域的巧妙洞察
2. **Laplacian 检测渐变边界**：渐变内部二阶导近零，边界处二阶导突变——简单而有效的物理直觉
3. **Model-free 设计**：整个流水线基于经典图像处理（Otsu、形态学操作、分水岭）+ 可微渲染优化，无需训练数据
4. **层级分解的实用性**：逐步添加的有序路径天然形成可编辑的图层结构，支持下游的颜色替换等编辑操作

## 局限与展望

1. 仅支持径向渐变（radial gradient），未覆盖线性渐变和锥形渐变
2. 对于极其复杂的真实照片，Bézier 路径数可能需要很多才能达到满意效果
3. 分割方法基于经典算法，在某些复杂输入上可能不够精确
4. 优化过程为迭代式，效率低于一次性前向推理的学习方法
5. 可探索融合深度学习分割方法（如 SAM）替代经典分割算法

## 相关工作与启发

- **LIVE**：最直接的前身工作，本文在其基础上增加渐变支持
- **DiffVG**：可微渲染器，是本文和 LIVE 的技术基础
- **Im2Vec**：学习方法矢量化，支持 DiffVG 反向传播但依赖训练数据
- 启发：在可微渲染优化框架中引入传统图像处理方法作为"引导信号"，可有效弥补纯优化方法的不足

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 技术深度 | 3 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| **综合** | **3.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [\[CVPR 2026\] FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching](../../CVPR2026/segmentation/flowdis_language-guided_dichotomous_image_segmentation_with_flow_matching.md)
- [\[ICCV 2025\] LayerAnimate: Layer-level Control for Animation](../../ICCV2025/segmentation/layeranimate_layer-level_control_for_animation.md)
- [\[ECCV 2024\] ReMamber: Referring Image Segmentation with Mamba Twister](remamber_referring_image_segmentation_with_mamba_twister.md)

</div>

<!-- RELATED:END -->

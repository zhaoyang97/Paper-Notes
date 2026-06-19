---
title: >-
  [论文解读] DragAnything: Motion Control for Anything using Entity Representation
description: >-
  [ECCV 2024][视频生成][可控视频生成] 提出DragAnything，利用扩散模型的隐空间特征作为实体表征（Entity Representation）来实现实体级运动控制，解决了现有轨迹驱动方法仅拖拽像素而无法精确控制目标对象运动的问题，在VIPSeg上实现SOTA的FVD/FID指标，用户研究中运动控制投票超出DragNUWA 26%。
tags:
  - "ECCV 2024"
  - "视频生成"
  - "可控视频生成"
  - "运动控制"
  - "实体表征"
  - "扩散模型"
  - "轨迹引导"
---

# DragAnything: Motion Control for Anything using Entity Representation

**会议**: ECCV 2024  
**arXiv**: [2403.07420](https://arxiv.org/abs/2403.07420)  
**代码**: [https://github.com/showlab/DragAnything](https://github.com/showlab/DragAnything)  
**领域**: 视频生成  
**关键词**: 可控视频生成, 运动控制, 实体表征, 扩散模型, 轨迹引导

## 一句话总结
提出DragAnything，利用扩散模型的隐空间特征作为实体表征（Entity Representation）来实现实体级运动控制，解决了现有轨迹驱动方法仅拖拽像素而无法精确控制目标对象运动的问题，在VIPSeg上实现SOTA的FVD/FID指标，用户研究中运动控制投票超出DragNUWA 26%。

## 研究背景与动机

**领域现状**：视频生成领域取得了显著进展（Imagen Video、SVD、SORA等），但可控视频生成的进展相对缓慢。在各种控制信号中，基于轨迹的运动控制因交互友好（用户只需画线）而成为主流方向。代表性工作如DragNUWA将稀疏轨迹编码为密集光流空间，MotionCtrl将轨迹坐标编码为向量图。

**现有痛点**：一个被忽视的核心问题——**单个像素点能否真正代表待控制的目标实体？**答案是否定的。现有方法存在两个关键insight：

**Insight 1：轨迹点不能代表实体**。如图所示，拖拽星空中一颗星的像素点时，模型无法分辨是要控制这颗星还是整个星空；拖拽云朵上的点时，实际效果是相机移动而非云朵移动。这说明单点无法承载实体的语义身份信息。

**Insight 2：越靠近拖拽点的像素运动越大**。在DragNUWA生成的视频中，像素运动量与到拖拽点的距离反相关。但我们期望的是整个对象按轨迹整体运动，而非像素级的不均匀形变。

**核心矛盾**：现有轨迹方法本质上是在做"像素拖拽"而非"实体控制"——它们没有建立"拖拽的点"与"要控制的实体"之间的语义关联。需要解决两个问题：(1) 如何表示实体？ (2) 如何只拖动选定实体？

**切入角度**：利用扩散模型自身的隐空间特征来表示实体。已有工作（AnyDoor用DINOv2特征、VideoSwap和DIFT用扩散模型特征）证明了隐特征表示对象的有效性。

**核心idea**：从扩散模型中提取目标实体对应区域的隐特征作为实体表征，通过操控该表征的空间位置实现实体级运动控制。

## 方法详解

### 整体框架
DragAnything基于Stable Video Diffusion (SVD)架构，核心组成：
- **去噪扩散模型**：3D U-Net完成时空去噪
- **实体语义表征提取**：从扩散特征中提取实体嵌入
- **2D高斯表征**：提供空间位置和中心聚焦的引导
- **控制编码器**：借鉴ControlNet，将两种表征编码后注入去噪U-Net的decoder

### 关键设计

#### 1. 实体表征提取（Entity Representation Extraction）

**功能**：从首帧图像中提取每个实体的语义嵌入，用于代替简单的坐标点来表示目标对象。

**核心思路**：利用扩散前向过程和一步去噪U-Net推理，提取与实体mask对应的隐特征，作为该实体的开放域嵌入。

**具体步骤**：
1. 给定首帧图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ 和实体mask $\mathbf{M}$
2. 通过扩散前向过程（加噪）得到噪声隐变量 $\boldsymbol{x}_t$
3. 用去噪U-Net做一步推理提取扩散特征：$\mathcal{F} = \epsilon_\theta(\boldsymbol{x}_t, t)$，$\mathcal{F} \in \mathbb{R}^{H \times W \times C}$
4. 用实体mask坐标从 $\mathcal{F}$ 中索引对应区域特征
5. 对提取的特征做平均池化，得到实体嵌入 $\{e_1, e_2, ..., e_k\}$

**轨迹关联**：
1. 初始化零矩阵 $\mathbf{E} \in \mathbb{R}^{H \times W \times C}$
2. 训练时用实体mask提取中心坐标，用Co-Tracker追踪得到轨迹序列
3. 在每帧中，将实体嵌入插入到轨迹点对应位置，得到实体表征 $\{\hat{\mathbf{E}}_i\}_{i=1}^L$

**设计动机**：扩散模型的中间特征天然包含丰富的语义信息。一步推理既快速又能获得高质量表征，比多步推理效果更好。实体表征是开放域的，能表示任何对象（包括背景）。

#### 2. 2D高斯表征（2D Gaussian Representation）

**功能**：提供空间位置引导，并使模型更关注实体中心区域。

**核心思路**：以实体中心坐标为中心、内切圆半径为参数生成2D高斯分布图，中心像素权重大、边缘递减。

**具体实现**：
1. 从实体mask计算内切圆的中心 $(x, y)$ 和半径 $r$
2. 在每帧轨迹点位置生成2D高斯分布图 $\{\mathbf{G}_i\}_{i=1}^L$
3. 通过编码器处理后与实体表征合并

**设计动机**：靠近实体中心的像素通常更重要（信息量大、遮挡少），2D高斯通过空间权重分配自然实现中心聚焦，弥补实体表征的均匀性。

#### 3. 控制信号编码与注入

**功能**：将两种表征编码到与SVD兼容的隐空间中。

**编码器结构**：4个卷积块（每块2层卷积 + SiLU激活），逐级2倍下采样到1/8分辨率。

**信号注入**：
$$\{\mathbf{R}_i\}_{i=1}^L = \mathcal{E}(\{\hat{\mathbf{E}}_i\}_{i=1}^L) + \mathcal{E}(\{\mathbf{G}_i\}_{i=1}^L) + \{\mathbf{Z}_i\}_{i=1}^L$$
其中 $\mathbf{Z}_i$ 是视频帧的噪声隐变量。编码后特征经3D U-Net编码器得到四个不同分辨率的特征，添加到去噪U-Net的对应层。

### 损失函数 / 训练策略

**损失函数**：带mask约束的MSE损失：
$$\mathcal{L}_\theta = \sum_{i=1}^L \mathbf{M} \| \epsilon - \epsilon_\theta(\boldsymbol{x}_{t,i}, \mathcal{E}_\theta(\hat{\mathbf{E}}_i), \mathcal{E}_\theta(\mathbf{G}_i)) \|_2^2$$

**Mask约束的动机**：优化目标仅关注实体运动控制，对背景和其他对象不施加约束，避免影响生成质量。

**训练数据**：使用VIPSeg视频分割数据集（提供实体级标注），Co-Tracker追踪中心坐标得到轨迹。

**训练设置**：基于SVD权重，AdamW优化器，学习率1e-5，100k步，Tesla A100 GPU，生成25帧320×576分辨率视频。

**推理交互**：用户通过SAM点击选择控制区域 → 在区域内拖拽形成轨迹 → 生成视频。

## 实验关键数据

### 主实验

**VIPSeg验证集 (256×256)**：

| 方法 | 基础架构 | ObjMC↓ | FVD↓ | FID↓ |
|------|---------|---------|--------|--------|
| DragNUWA | SVD | 324.6 | 519.3 | 39.8 |
| **DragAnything** | SVD | **305.7** | **494.8** | **33.5** |
| 提升 | - | **-18.9** | **-24.5** | **-6.3** |

**用户研究**：

| 评估维度 | DragAnything胜率 | DragNUWA胜率 |
|---------|----------------|--------------|
| 运动控制 | **63%** | 37% (**+26%**) |
| 视频质量 | **56%** | 44% (**+12%**) |

### 消融实验

**实体表征和2D高斯表征贡献**：

| Entity Rep. | Gaussian Rep. | ObjMC↓ | FVD↓ | FID↓ |
|:-----------:|:------------:|---------|--------|--------|
| ✗ | ✗ | 410.7 | 496.3 | 34.2 |
| ✓ | ✗ | 318.4 | 494.5 | 34.1 |
| ✗ | ✓ | 339.3 | 495.3 | 34.0 |
| **✓** | **✓** | **305.7** | **494.8** | **33.5** |

**关键观察**：
- Entity Rep.对ObjMC提升最大：-92.3（410.7→318.4）
- Gaussian Rep.提升：-71.4（410.7→339.3）
- 两者结合有互补效果：-105.0（410.7→305.7）

**Loss Mask消融**：

| Loss Mask | ObjMC↓ | FVD↓ | FID↓ |
|:---------:|---------|--------|--------|
| ✗ | 311.1 | 500.2 | 34.3 |
| ✓ | **305.7** | **494.8** | **33.5** |

### 关键发现

1. **实体表征是运动控制的核心**：带来92.3的ObjMC提升，证明扩散特征确实编码了足够的实体语义信息
2. **两种表征互补**：Entity Rep.提供语义身份，Gaussian Rep.提供空间位置和中心关注，结合效果最佳
3. **Loss Mask对精确控制的重要性**：约束损失只在目标区域回传，避免干扰其他区域的生成质量
4. **DragAnything在所有指标上全面超越DragNUWA**：FVD（时序一致性）、FID（视觉质量）、ObjMC（运动精度）
5. **支持多样化运动控制**：前景、背景、前景+背景同时控制、相机运动控制

## 亮点与洞察

1. **洞察深刻**：通过Co-Tracker追踪像素运动的toy experiment，直观揭示了"像素拖拽≠实体控制"的本质问题
2. **实体表征的通用性**：开放域嵌入可以表示任何对象——不仅限实例级物体，还能表示背景（云层、星空）和相机运动
3. **利用扩散模型自身特征**：不引入额外编码器（如CLIP、DINOv2），而是利用扩散U-Net已有的语义能力，计算高效且语义对齐
4. **对比 paradigm 分析到位**：系统对比了点表征、轨迹图、2D高斯、框表示和实体表征五种范式的优劣
5. **交互极其友好**：推理时用户只需SAM点选+画轨迹，整个流程无需专业知识

## 局限与展望

1. **大幅度运动控制较弱**：如图10所示的bad case，控制较大运动时可能生成失真
2. **依赖首帧实体mask**：需要SAM或手动标注的mask来提取实体表征，完全无mask的场景需要额外处理
3. **轨迹来源依赖Co-Tracker**：训练数据的轨迹质量受限于追踪器精度
4. **分辨率限制**：当前评估在256×256，高分辨率下的效果和效率需要验证
5. **多实体交互场景**：当多个实体重叠或物理交互时（如碰撞），运动控制的物理合理性可能不足
6. **可探索去mask化**：结合语言指令的实体选择（如"move the dog to the right"）可提升用户体验

## 相关工作与启发

- **DragNUWA / MotionCtrl**：轨迹控制video generation的先驱，但都是像素级操控
- **SVD (Stable Video Diffusion)**：DragAnything的基础模型
- **ControlNet**：控制信号注入去噪U-Net的经典架构，DragAnything的编码器设计借鉴
- **DIFT / VideoSwap**：证明扩散特征可以作为有效的对象表征
- **SAM**：提供交互式分割支持实体选择
- **Co-Tracker**：点追踪器，提供训练所需的运动轨迹标注
- 启发：扩散模型的中间表征本身就是极好的语义表示，充分利用现有模型的内在能力比额外引入模块更高效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从"像素拖拽"到"实体控制"的paradigm shift，洞察深刻，实体表征的设计思路优雅
- 实验充分度: ⭐⭐⭐⭐ 自动指标+用户研究+消融+多种控制模式可视化，较为全面；但仅与DragNUWA对比（MotionCtrl未开源SVD版本）
- 写作质量: ⭐⭐⭐⭐ toy experiment的动机展示直观有力，方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 在可控视频生成领域提出了关键的范式改进，实体级控制的思路对后续工作有重要影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SMRABooth: Subject and Motion Representation Alignment for Customized Video Generation](../../CVPR2026/video_generation/smrabooth_subject_and_motion_representation_alignment_for_customized_video_gener.md)
- [\[ICCV 2025\] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis](../../ICCV2025/video_generation/causal-entity_reflected_egocentric_traffic_accident_video_synthesis.md)
- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](../../CVPR2025/video_generation/tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2026\] 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation](../../CVPR2026/video_generation/3d-aware_implicit_motion_control_for_view-adaptive_human_video_generation.md)

</div>

<!-- RELATED:END -->

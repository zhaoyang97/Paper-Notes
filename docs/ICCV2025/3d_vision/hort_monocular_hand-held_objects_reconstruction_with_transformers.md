---
title: >-
  [论文解读] HORT: Monocular Hand-held Objects Reconstruction with Transformers
description: >-
  [3D视觉] 提出 HORT，基于 Transformer 的粗到细框架，从单目图像高效重建手持物体的稠密3D点云，通过整合图像特征和3D手部几何信息联合预测物体点云及其相对手部的位姿，在准确率和推理速度上均达到 SOTA。 从单目图像重建手持物体的3D形状在动作识别、人机交互和机器人操作中有广泛应用…
tags:
  - "3D视觉"
---

# HORT: Monocular Hand-held Objects Reconstruction with Transformers

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.21313](https://arxiv.org/abs/2503.21313)
- **代码**: [https://zerchen.github.io/projects/hort.html](https://zerchen.github.io/projects/hort.html)
- **领域**: 3D视觉 / 手持物体重建
- **关键词**: Hand-Object Reconstruction, Point Cloud, Transformer, Coarse-to-Fine, Monocular 3D

## 一句话总结
提出 HORT，基于 Transformer 的粗到细框架，从单目图像高效重建手持物体的稠密3D点云，通过整合图像特征和3D手部几何信息联合预测物体点云及其相对手部的位姿，在准确率和推理速度上均达到 SOTA。

## 研究背景与动机

从单目图像重建手持物体的3D形状在动作识别、人机交互和机器人操作中有广泛应用。现有方法面临关键瓶颈：

**隐式表示方法**（SDF等）：
- 生成的3D表面过于光滑，丢失几何细节
- 需要 Marching Cubes 后处理才能获得显式网格，推理速度慢（约2秒）
- 无法灵活用于下游应用

**显式表示方法**：
- HO 使用顶点表示但分辨率有限
- D-SCO 使用扩散模型重建高分辨率点云，但多步去噪导致推理极慢（>13秒）

核心矛盾在于：**高质量重建与高效推理之间的平衡**。此外，手部几何信息蕴含了关于物体几何和位置的隐式线索，但现有方法未能充分利用。

## 方法详解

### 整体框架

HORT 采用粗到细的两阶段策略，包含四个关键模块：

1. **图像编码器**：使用 DINOv2-Large 提取256+1个视觉特征 token
2. **手部编码器**：将 MANO 手部几何编码为丰富的3D特征
3. **稀疏点云解码器**：联合预测稀疏点云和手相对位姿
4. **稠密点云解码器**：利用像素对齐特征上采样为高分辨率点云

### 细粒度手部特征编码

关键创新在于如何编码3D手部几何：

1. 使用 MANO 模型重建手部网格 $v_h \in \mathbb{R}^{778\times3}$，由关节 $j_h \in \mathbb{R}^{16\times3}$ 驱动
2. 将手部顶点变换到22个局部坐标系（16个关节 + 5个指尖 + 1个掌心）
3. 拼接变换后的坐标和绝对顶点索引，得到 $e_h \in \mathbb{R}^{778\times67}$
4. 通过 PointNet 编码为手部特征 $f_h \in \mathbb{R}^{1024}$

这种多坐标系表示方式捕获了手部的姿态和形状相关几何信息，为物体重建提供强有力的结构先验。

### 稀疏点云解码器

将重建任务分解为两个子任务：
- **正则物体点云生成**：在手掌坐标系下生成
- **手相对物体位姿估计**：仅预测相对手掌的3D平移 $t_o \in \mathbb{R}^3$（回避因对称性导致的旋转预测ill-posed问题）

解码器使用统一的多层 Transformer：
- 定义 $1 + N_p^s$ 个可学习 token（1个位姿 + $N_p^s$ 个点云）
- 先自注意力，后分别对图像特征 $f_v$ 和手部特征 $f_h$ 做交叉注意力
- 共享 backbone 让位姿和点云预测互相促进

### 稠密点云解码器

从稀疏到稠密的上采样策略：

1. **像素对齐特征提取**：利用预测的相机参数和位姿，将稀疏点云投影到图像平面
$$f_o = F(\pi(p_o^s + t_p + t_o, K_{cam}), f_v^r)$$
2. **局部自注意力**：在 KNN 邻域内（k=16）进行自注意力，聚合空间和视觉上下文
3. **两级上采样**：分别以2倍和4倍上采样，最终得到 $N_p^d = 16384$ 个点

### 损失函数

端到端训练，总损失为：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{pose} + \lambda_2 \mathcal{L}_{cd}^s + \mathcal{L}_{cd}^d$$

- $\mathcal{L}_{pose}$：ℓ1损失监督物体相对手掌的3D平移
- $\mathcal{L}_{cd}^s$：稀疏点云的 Chamfer Distance
- $\mathcal{L}_{cd}^d$：稠密点云的 Chamfer Distance
- 两个超参数均设为2

## 实验

### 主实验：ObMan 数据集对比

| 方法 | FS@5 ↑ | FS@10 ↑ | CD ↓ |
|:---|:---:|:---:|:---:|
| HO | 0.23 | 0.56 | 6.4 |
| AlignSDF | 0.40 | 0.64 | 9.2 |
| gSDF | 0.44 | 0.66 | 8.8 |
| DDF-HO | 0.55 | 0.67 | 1.4 |
| D-SCO | 0.61 | 0.81 | 1.1 |
| **HORT (Ours)** | **0.66** | **0.88** | **1.0** |

HORT 在所有指标上超越 D-SCO，FS@5 提升 +0.05，FS@10 提升 +0.07。

### HO3D 和 DexYCB 真实数据集

| 方法 | HO3D FS@5 ↑ | HO3D CD ↓ | DexYCB FS@5 ↑ | DexYCB CD ↓ |
|:---|:---:|:---:|:---:|:---:|
| D-SCO | 0.38 | 3.2 | 0.48 | 2.9 |
| **HORT** | **0.41** | **2.5** | **0.52** | **2.5** |

在真实数据集上同样保持 SOTA 性能。

### 消融实验：编码器设计

| 配置 | 手掌 | 关节 | 图像编码器 | FS@5 ↑ | CD ↓ |
|:---|:---:|:---:|:---|:---:|:---:|
| R1 | × | × | Fine-tune | 0.45 | 3.1 |
| R2 | ✓ | × | Fine-tune | 0.53 | 2.4 |
| R3 | ✓ | ✓ | Fine-tune | 0.60 | 1.8 |
| R4 | ✓ | ✓ | Scratch | 0.51 | 2.6 |
| R5 | ✓ | ✓ | Frozen | 0.48 | 2.9 |

关键发现：(1) 手部多坐标系编码至关重要（R1→R3：FS@5 提升 0.15）；(2) 图像编码器微调策略影响显著。

### 推理速度对比

HORT 推理一张图像约 **0.08秒**（含网格化），而 D-SCO 需要 **13秒**，隐式方法需约 **2秒**。推理速度提升 **162倍**。

## 亮点与洞察
1. **速度-质量最优平衡**：在推理速度上比 D-SCO 快两个数量级的同时，重建质量更优
2. **巧妙的手部先验利用**：多坐标系变换 + PointNet 编码的手部几何特征提供了强大的物体形状约束
3. **端到端训练优势**：相比 D-SCO 分模块训练，端到端优化让各组件协同更好
4. **可作为优化方法的初始化**：前馈预测可加速后续基于优化的精细化方法

## 局限性
- 不预测物体旋转（因对称性问题），对非对称物体的位姿估计不完整
- 依赖手部姿态估计质量，手部重建失败会影响物体重建
- 点云表示无法直接用于需要网格拓扑的应用（如物理仿真）

## 相关工作
- 手持物体重建：IHOI, AlignSDF, gSDF, D-SCO
- 隐式3D表示：SDF, DDF, NeRF
- 点云生成：PoinTr, SnowflakeNet, Michelangelo
- 手部重建：MANO, HaMeR, WiLoR

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 粗到细 Transformer 框架结合手部几何的设计新颖实用
- **技术深度**: ⭐⭐⭐⭐ — 多坐标系手部编码和联合解码设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个数据集、全面消融、推理速度对比
- **实用价值**: ⭐⭐⭐⭐⭐ — 0.08秒推理使其具备实时应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

</div>

<!-- RELATED:END -->

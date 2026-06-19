---
title: >-
  [论文解读] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
description: >-
  [AAAI 2026][3D视觉][3D Gaussian Splatting] 将每个3D Gaussian视为拉格朗日物质点，引入时变材料场预测粒子速度和本构应力张量，通过Cauchy动量残差作为物理约束 + 拉格朗日粒子流匹配作为数据拟合项，在单目动态视图合成中实现了物理一致性和跨场景泛化能力，在自建物理驱动数据集和HyperNeRF真实数据集上均达到SOTA。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "动态场景重建"
  - "物理信息神经网络"
  - "连续介质力学"
  - "光流监督"
---

# Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field

**会议**: AAAI 2026  
**arXiv**: [2511.06299](https://arxiv.org/abs/2511.06299)  
**代码**: [https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 动态场景重建, 物理信息神经网络, 连续介质力学, 光流监督

## 一句话总结
将每个3D Gaussian视为拉格朗日物质点，引入时变材料场预测粒子速度和本构应力张量，通过Cauchy动量残差作为物理约束 + 拉格朗日粒子流匹配作为数据拟合项，在单目动态视图合成中实现了物理一致性和跨场景泛化能力，在自建物理驱动数据集和HyperNeRF真实数据集上均达到SOTA。

## 研究背景与动机
**领域现状**：3DGS因显式表示和实时渲染成为动态新视角合成的主流方法。动态建模方式包括增量法（DynamicGS）、变形场法（D-3DGS、Grid4D）和低秩分解法（SC-GS），配合4D分解哈希编码可高效表示时空信息。

**现有痛点**：现有方法将运动简化为刚体变换，忽略了不同材料的运动本构规律（流体、弹性体、布料等运动模式截然不同）。同时仅依赖2D视觉监督（RGB loss），无法统一约束3D粒子的物理状态，导致Gaussian粒子偏离真实运动模式。

**核心矛盾**：纯数据驱动的变形场缺乏物理归纳偏置，无法区分不同材料（流体 vs 弹性体 vs 刚体）的运动规律。已有物理嵌入方法（PhysGaussian、PINNs-based）依赖严格边界条件、固定材料属性或RGB-D/多视角输入，无法泛化到单目动态场景。

**本文目标** (1) 如何在没有粒子运动先验的情况下建模Gaussian粒子的位置和时变形变？(2) 什么样的边界条件或替代监督可以实现物理一致且可泛化的动态材料建模？

**切入角度**：从拉格朗日力学出发，将Cauchy动量方程作为统一的本构定律，让每个粒子的速度和应力通过时变材料场独立预测。同时利用光流分解提供运动流作为伪真值来指导速度场收敛。

**核心 idea**：将连续介质力学的Cauchy动量方程嵌入3DGS框架，每个Gaussian粒子作为拉格朗日物质点在时变材料场中演化，以物理残差+光流对齐双重监督实现跨材料泛化

## 方法详解

### 整体框架
PIDG由三个核心模块组成：(1) 正则哈希空间中的动态建模——用4D分解哈希编码高效表示时空变形，静动态区域解耦；(2) 物理信息Gaussian表示——将每个Gaussian视为拉格朗日粒子，通过时变材料场预测速度和应力张量，Cauchy动量残差作为物理约束；(3) 拉格朗日粒子流匹配——将光流分解为相机流和运动流，用运动流作为伪真值监督粒子的Gaussian流和速度流。整套pipeline全可微，从单目视频输入端到端训练。

### 关键设计

1. **4D分解哈希编码 + 静动态解耦**:

    - 功能：将4D时空坐标高效编码为特征，解耦静态和动态区域
    - 核心思路：将 $(x,y,z,t)$ 映射到四个独立的3D哈希网格 $G_{xyz}, G_{xyt}, G_{yzt}, G_{xzt}$，内存从 $\mathcal{O}(n^4)$ 降至 $\mathcal{O}(n^3)$。空间MLP提取方向性注意力权重 $a = 2\sigma(f_s(G_{xyz})) - 1$，调制时间MLP的输出特征 $h = a \odot f_t(G_{xyt}, G_{yzt}, G_{xzt})$。多头MLP解码变形参数 $D(h) = \{R_x, T_x, \Delta r, \Delta s\}$
    - 设计动机：相比4D MLP或低秩平面分解，哈希编码在保持精度的同时大幅降低内存。两阶段优化策略先联合优化几何和运动，再用动态掩码冻结静态区域，使物理建模专注于动态部分

2. **时变材料场 (Time-Evolving Material Field)**:

    - 功能：为每个Gaussian粒子预测随时间变化的速度和本构应力张量
    - 核心思路：将归一化4D坐标嵌入6个可学习空间/时间平面张量 $\mathbf{F}_{\text{Hash}}$，拼接傅里叶时间编码 $T(t)$ 和可学习粒子索引嵌入 $\mathbf{e}_i$，得到特征向量 $\mathbf{F} = [\mathbf{F}_{\text{Hash}}, T(t), \mathbf{e}_i]$。多头MLP $f_\theta$ 联合预测速度 $\bm{v} \in \mathbb{R}^3$ 和应力张量6个独立分量 $\bm{\sigma} \in \mathbb{R}^6$。Cauchy动量残差为 $\mathbf{r}(x,t) = \rho(\frac{\partial \bm{v}}{\partial t} + (\bm{v} \cdot \nabla)\bm{v}) - \nabla \cdot \bm{\sigma}$，惩罚其L2范数得到物理损失 $\mathcal{L}_{\text{CMR}}$
    - 设计动机：将速度和应力作为独立内禀属性建模，使每个Gaussian粒子不仅编码在变形场中，还能随时间持续演化。通过改变本构应力张量的形式，Cauchy动量方程可统一描述流体、弹性体和刚体动力学

3. **拉格朗日粒子流匹配 (Lagrangian Particle Flow Matching)**:

    - 功能：用光流分解提供运动监督，指导速度场和应力场收敛到物理可信解
    - 核心思路：**反向光流分解**——从 $I_{t+1}$ 反向计算运动流并变换到 $I_t$ 坐标系，避免正向策略中因双线性插值导致的条纹伪影。Gaussian流 $flow_g$ 由跟踪top-K Gaussian粒子的2D位移加权求和获得；速度流 $flow_v$ 用预测速度平流Gaussian粒子得到。两者分别与运动流真值对齐：$\mathcal{L}_{\text{LPFM}} = \lambda_g \|flow_g - flow_{gt}\|_1 + \lambda_v \|flow_v - flow_{gt}\|_1$
    - 设计动机：仅靠Cauchy动量残差作为物理约束，速度和应力预测难以收敛到物理可信解（方程欠定）。引入光流作为数据拟合项相当于PINN框架中的边界条件替代，锚定粒子轨迹使优化有明确收敛方向

### 损失函数 / 训练策略
总损失包含：渲染损失 $\mathcal{L}_{\text{renders}} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c\mathcal{L}_{\text{D-SSIM}}$（$\lambda_c=0.2$）、Cauchy动量残差 $\mathcal{L}_{\text{CMR}}$（$\lambda_{\text{CMR}}=0.1$）、拉格朗日粒子流匹配 $\mathcal{L}_{\text{LPFM}}$（$\lambda_{\text{LPFM}}=0.01$）。训练50K迭代（合成）/40K迭代（真实）。采用分块采样Cauchy动量残差策略避免显存爆炸——将粒子分块计算物理残差并释放计算图后聚合。缩放阈值从0.1调至0.015以过滤大尺度噪声Gaussian。

## 实验关键数据

### 主实验（PIDG物理驱动数据集 + HyperNeRF）

| 方法 | PIDG-PSNR↑ | PIDG-SSIM↑ | PIDG-LPIPS↓ | HyperNeRF-PSNR↑ | HyperNeRF-MS-SSIM↑ |
|------|-----------|-----------|------------|---------|-----------|
| D-NeRF | 23.45 | 0.866 | 0.124 | 25.7 | 0.726 |
| D-3DGS | 29.54 | 0.951 | 0.066 | - | - |
| GaussianPredict | 30.17 | 0.957 | 0.062 | 26.6 | 0.884 |
| Grid4D | 30.32 | 0.956 | 0.061 | 27.3 | 0.899 |
| D-2DGS | 29.23 | 0.944 | 0.061 | 17.7 | 0.509 |
| **PIDG (Ours)** | **30.96** | **0.967** | **0.058** | **27.8** | **0.906** |

### 消融实验

| 配置 | PIDG-PSNR↑ | PIDG-SSIM↑ | D-NeRF-PSNR↑ | 说明 |
|------|-----------|-----------|-------------|------|
| w/o ($\mathcal{L}_{\text{LPFM}} + \mathcal{L}_{\text{CMR}}$) | 30.46 | 0.956 | 42.00 | 仅静动态解耦+哈希编码 |
| w/o $\mathcal{L}_{\text{LPFM}}$ | 30.78 | 0.957 | 42.14 | 仅Cauchy动量残差约束 |
| **Full model** | **30.96** | **0.967** | - | 完整模型（D-NeRF无连续视角无法用流匹配） |
| Grid4D + $\mathcal{L}_{\text{CMR}}$ | - | - | 42.10 | 即插即用提升+0.10 PSNR |
| SC-GS + $\mathcal{L}_{\text{CMR}}$ | - | - | 41.85 | 即插即用提升+0.20 PSNR |

### 关键发现
- 完整流匹配在流体烟雾（Dry Ice: 25.34→26.12 PSNR）和弹性碰撞（Balls: 32.79→33.31）等复杂物理场景提升最显著，说明光流监督对物理复杂运动特别有效
- $\mathcal{L}_{\text{CMR}}$ 作为即插即用模块应用到GaussianPredict/SC-GS/Grid4D上均有一致提升，验证物理约束的通用性
- 去掉应力张量后 $\mathcal{L}_{\text{CMR}}$ 退化为 $\nabla \cdot \bm{v} = 0$（连续性约束），t-SNE可视化显示动态粒子特征区分度明显下降
- 训练效率优秀：72分钟/~85K Gaussian/250 FPS，内存6.2GB，优于大多数基线

## 亮点与洞察
- **统一本构定律框架**：通过有效场论(EFT)推导了Cauchy动量方程到刚体/弹性固体/流体的统一退化过程。网络可自适应学习材料属性而无需人工指定，极具理论优雅性。附录A的推导展示了从标量Goldstone场到Lamé参数再到Navier-Stokes的完整链路
- **反向光流分解**：将运动掩码应用到像素级motion flow而非Gaussian flow，保持真实位移不被条纹伪影破坏。这是对MotionGS正向策略的关键改进，在HyperNeRF真实数据上视觉效果差异明显
- **拉格朗日粒子身份继承**：densification时子粒子继承父粒子索引嵌入，避免了昂贵的最近邻搜索来恢复身份，简洁高效
- **CMR即插即用**：物理残差模块可零成本插入多种现有动态3DGS方法中作为正则化器，实用性强

## 局限与展望
- **计算开销**：训练仍需数小时（A800上72分钟）且消耗大量GPU显存（6.2GB），距离实时重建差距较大。作者计划开发轻量级前馈网络架构避免昂贵的优化循环
- **材料模型有限**：线性本构假设无法捕捉非线性弹塑性或粘弹性等复杂行为。可结合有限元/粒子混合方法提升材料建模丰富度
- **光流依赖**：需要预训练光流/深度/分割模型（UniMatch、Distill Any Depth、SAMv2），不连续视角场景无法使用流匹配。可探索自监督运动先验替代
- **2D评估局限**：作者呼吁社区建立包含几何/时序/物理指标的综合评估体系，例如引入速度场一致性或应力场合理性指标

## 相关工作与启发
- **vs Grid4D**: Grid4D用4D哈希编码高效建模动态但缺乏物理约束，PIDG在其基础上加入材料场后PSNR提高0.64（PIDG数据集）。PIDG的Gaussian数量更少（~85K vs ~100K）且FPS更高（250 vs 240），说明物理约束还带来了隐式的正则化效果
- **vs PhysGaussian/PhysDreamer**: 基于MPM的方法依赖网格离散化和固定材料属性，且需要RGB-D或多视角输入。PIDG通过PINN框架绕开网格离散限制，材料参数完全可学习，且只需单目视频输入
- **vs MotionGS**: 同用光流监督但MotionGS用正向分解策略有条纹伪影。PIDG的反向分解更鲁棒，在HyperNeRF上PSNR领先2.6且流可视化对齐更一致
- **vs GaussianFlow**: PIDG复用了GaussianFlow的CUDA光栅化器但做了效率优化（合并梯度计算、移除冗余反向传播），作为Lagrangian粒子流计算的基础

## 评分
- 新颖性: ⭐⭐⭐⭐ 将连续介质力学Cauchy动量方程统一嵌入3DGS框架，理论视角新颖但PINN本身已广泛应用
- 实验充分度: ⭐⭐⭐⭐⭐ 自建5场景物理数据集+D-NeRF+HyperNeRF三大平台，消融非常详尽，包含即插即用实验、未来预测、t-SNE分析等
- 写作质量: ⭐⭐⭐⭐ 理论推导完整（附录包含EFT统一本构定律推导），但主文公式密度略高
- 价值: ⭐⭐⭐⭐ 为动态3DGS提供有价值的物理归纳偏置范式，$\mathcal{L}_{\text{CMR}}$ 即插即用模块实用性强，自建PIDG数据集也有社区价值

<!-- 笔记写于 2026-04-10，基于 paper_cache/AAAI2026/2511.06299.txt 全文缓存 -->

<!-- 附注：PIDG数据集包含5个场景(Balls Reaction/Mechanics Cloth/Motion Kuro/Rubber Duck/Dry Ice)，
     每个场景150帧、1600×900分辨率，由Blender物理求解器生成 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](../../ICLR2026/3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](../../CVPR2026/3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] Gaussian Mapping for Evolving Scenes](../../CVPR2026/3d_vision/gaussian_mapping_for_evolving_scenes.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](../../ICML2026/3d_vision/physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)
- [\[CVPR 2026\] Seele: A Unified Acceleration Framework for Real-Time Gaussian Splatting on Mobile Devices](../../CVPR2026/3d_vision/seele_a_unified_acceleration_framework_for_real-time_gaussian_splatting_on_mobil.md)

</div>

<!-- RELATED:END -->

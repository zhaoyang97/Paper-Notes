---
title: >-
  [论文解读] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints
description: >-
  [ICCV 2025][3D视觉][铰接物体生成] 提出 PhysNAP，通过点云对齐损失和基于SDF的物理合理性约束（部件穿透+关节移动）引导预训练扩散模型 NAP 的逆扩散过程，实现类别感知的铰接物体生成，在对齐精度和物理合理性上显著优于无引导基线。 铰接物体（如抽屉、电器、笔记本电脑）在日常环境中极为常见…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "铰接物体生成"
  - "扩散模型"
  - "点云对齐"
  - "物理约束"
  - "SDF"
---

# Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints

**会议**: ICCV 2025  
**arXiv**: [2508.00558](https://arxiv.org/abs/2508.00558)  
**代码**: 未公开  
**领域**: 3D视觉  
**关键词**: 铰接物体生成, 扩散模型, 点云对齐, 物理约束, SDF

## 一句话总结

提出 PhysNAP，通过点云对齐损失和基于SDF的物理合理性约束（部件穿透+关节移动）引导预训练扩散模型 NAP 的逆扩散过程，实现类别感知的铰接物体生成，在对齐精度和物理合理性上显著优于无引导基线。

## 研究背景与动机

铰接物体（如抽屉、电器、笔记本电脑）在日常环境中极为常见，它们的数字孪生生成和重建对虚拟现实和机器人应用非常重要。

**现有方法的局限**：

**NAP**（基线方法）：使用DDPM生成铰接图，可表示部件形状（SDF）、关节参数和图结构，但是**无条件无引导**的生成，无法与观测数据对齐

**CAGE/SINGAPO**：需要铰接图结构（节点数、连接关系）作为输入先验，限制了通用性

**MIDGaRD**：分阶段生成图结构和形状，阻止了损失通过形状反向传播到图结构

**PhysPart**：最接近的工作，但假设已知完整底座点云且仅考虑底座-单个部件间碰撞

**PhysNAP 的创新定位**：
- **不假设已知铰接图结构**（部件数量和连接未知）
- **使用部分点云**（而非完整点云）作为训练自由引导
- **评估所有部件对之间的穿透**（而非仅底座-单部件）
- **首个同时具备类别感知、点云对齐和物理合理性引导的铰接物体扩散模型**

## 方法详解

### 整体框架

PhysNAP 基于 NAP 扩展，核心包含三部分：
1. 类别条件化的铰接图扩散模型（训练阶段）
2. 损失引导的逆扩散采样（推理阶段）
3. 三种引导损失设计（点云对齐 + 穿透 + 移动性）

### 铰接图表示

每个铰接物体表示为最多 $K=8$ 个节点的图：
- **节点属性**：存在指示 $o_i$、位姿 $\boldsymbol{T}_{gi} \in SE(3)$、包围盒 $\boldsymbol{b}_i \in \mathbb{R}^3$、形状隐码 $\boldsymbol{s}_i \in \mathbb{R}^{128}$
- **边属性**：存在指示 $c_{i,j} \in \{-1,0,1\}$、Plücker 坐标描述铰接轴、关节范围

形状通过预训练的神经SDF解码器表示，可通过 Marching Cubes 提取网格。

### 类别感知生成

在 NAP 的 AGNN（注意力图神经网络）中添加可学习的类别嵌入，加到节点和边嵌入上，然后用带类别标注的数据重新训练。

### 损失引导扩散

利用 loss-guided diffusion 框架，在逆扩散过程的最后 $n_g=500$ 步添加引导梯度：

$$\ell_{\boldsymbol{P}}(\hat{\boldsymbol{x}}_0) = w_{\text{pc}}\ell_{\text{pc}} + w_{\text{pen}}\ell_{\text{pen}} + w_{\text{mob}}\ell_{\text{mob}}$$

#### 点云对齐损失 $\ell_{\text{pc}}$

给定部分点云 $\boldsymbol{P} \in \mathbb{R}^{n_p \times 3}$，通过预测的SDF评估每个点到最近部件的距离。由于点-部件关联未知，使用软对应（类似EM）：

$$\alpha_{i,j} = \frac{\exp(-\tau \, d(\boldsymbol{P}_j, i)^2 / (\tilde{o}_i + \epsilon))}{\sum_k \exp(-\tau \, d(\boldsymbol{P}_j, k)^2 / (\tilde{o}_k + \epsilon))}$$

最终 $\ell_{\text{pc}} = \sum_j \sum_i \alpha_{i,j} d(\boldsymbol{P}_j, i)^2$，温度参数 $\tau=1000$。

#### 穿透损失 $\ell_{\text{pen}}$

计算所有部件对之间的包围盒交集，在交集区域采样3D网格点（目标 $N^*=1000$），评估SDF穿透误差：

$$\psi(\boldsymbol{q}_j, i, i') = \frac{1}{2}\min(0, -(d(\boldsymbol{q}_j, i) + d(\boldsymbol{q}_j, i')))^2$$

#### 移动性损失 $\ell_{\text{mob}}$

在预测的关节极限内随机采样铰接状态，通过螺旋变换计算部件间相对位姿，评估运动状态下的穿透误差。仅评估相邻部件之间的碰撞。

## 实验

### 实验设置

- **数据集**：PartNet-Mobility，包含多类别铰接物体
- **评估指标**：点云对齐($E_{\text{pc}}$, $D_{\text{pc}}$)、穿透($E_{\text{pen}}$)、移动性($E_{\text{mob}}$)、生成质量(MMD, 1-NNA)
- **测试集**：30个随机物体模型，每个1000个采样点

### 主实验结果

| 类别感知 | 引导变体 | $E_{\text{pc}}$ | $D_{\text{pc}}$ | $E_{\text{pen}}$ | $E_{\text{mob}}$ | MMD | 1-NNA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | pc+pen+mob | 0.0024 | 0.0705 | **0.0000** | **0.0003** | 0.1435 | 0.9547 |
| ✗ | pc | **0.0006** | **0.0435** | 0.0018 | 0.0031 | 0.1540 | 0.9666 |
| ✗ | uncond (NAP) | 0.0564 | 0.2063 | 0.0035 | 0.0033 | **0.0915** | **0.9440** |
| ✓ | pc+pen+mob | 0.0012 | 0.0483 | **0.0000** | **0.0003** | 0.1970 | 0.9774 |
| ✓ | pc | **0.0004** | **0.0301** | 0.0079 | 0.0059 | 0.2162 | 0.9817 |
| ✓ | uncond | 0.0076 | 0.0974 | 0.0019 | 0.0027 | 0.1687 | 0.9709 |

### 消融实验

| 消融设置 | 关键发现 |
|:---:|:---|
| 引导步数 $n_g$ | 500步与1000步效果相似，500步计算效率更高 |
| 权重 $w_{\text{pen}}$ | 权重增大 → 穿透减少但生成质量下降，存在trade-off |
| 仅pc vs pc+pen+mob | 加入物理约束后对齐略降但物理合理性大幅提升 |
| 类别感知 vs 无感知 | 类别信息提升对齐精度但降低多样性(1-NNA升高) |

### 关键发现

1. **引导有效性**：所有引导变体在对应指标上均优于无引导基线（$E_{\text{pc}}$ 从0.0564降至0.0006）
2. **Trade-off存在**：引导目标之间存在权衡——pc引导损害物理合理性，pen+mob引导损害点云对齐
3. **类别条件化**：提供类别信息改善对齐但牺牲生成多样性
4. **运行时间**：完整引导下每个样本约2分钟（Nvidia A40）

## 亮点与洞察

1. **Training-free guidance 的优雅应用**：无需修改预训练模型，仅在采样时添加梯度引导，实现灵活的条件生成
2. **SDF的双重利用**：SDF既用于形状表示又用于物理约束计算（穿透检测），设计紧凑
3. **软对应机制**：通过可微的软分配解决未知点-部件关联，使损失对节点存在性也可微

## 局限性

1. 引导过程增加推理时间（~2分钟/样本）
2. 点云对齐与物理合理性之间的trade-off难以完全消除
3. 类别感知模型降低了生成多样性
4. 仅在PartNet-Mobility数据集上验证，泛化能力未知

## 相关工作

- **铰接物体生成**：NAP, CAGE, SINGAPO, MIDGaRD, PhysPart
- **损失引导扩散**：DPS, Loss-Guided Diffusion
- **SDF表示**：DeepSDF, 铰接SDF (A-SDF)

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次将物理合理性约束整合到铰接物体扩散生成中
- 技术深度：⭐⭐⭐⭐ — SDF穿透检测和软对应设计精巧
- 实验完整性：⭐⭐⭐ — 与消融比较充分，但缺乏与同类方法的直接对比
- 实用价值：⭐⭐⭐ — 应用场景明确（机器人、VR/AR），但推理速度待优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](../../CVPR2025/3d_vision/guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)

</div>

<!-- RELATED:END -->

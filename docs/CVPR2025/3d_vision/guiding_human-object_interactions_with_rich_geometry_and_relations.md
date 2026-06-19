---
title: >-
  [论文解读] Guiding Human-Object Interactions with Rich Geometry and Relations
description: >-
  [CVPR 2025][3D视觉][人物交互] 本文提出ROG框架，通过在物体网格上采样富含几何信息的关键点构建交互距离场（IDF），并利用基于扩散的关系模型在推理时引导运动生成模型产生关系感知且语义对齐的人物-物体交互动作，在FullBodyManipulation数据集上显著超越SOTA。 领域现状：人物-物体交互（HO…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "人物交互"
  - "交互距离场"
  - "扩散模型"
  - "几何表示"
  - "运动生成"
---

# Guiding Human-Object Interactions with Rich Geometry and Relations

**会议**: CVPR 2025  
**arXiv**: [2503.20172](https://arxiv.org/abs/2503.20172)  
**代码**: [https://lalalfhdh.github.io/rog_page/](https://lalalfhdh.github.io/rog_page/)  
**领域**: 3D视觉 / 动作生成  
**关键词**: 人物交互, 交互距离场, 扩散模型, 几何表示, 运动生成

## 一句话总结

本文提出ROG框架，通过在物体网格上采样富含几何信息的关键点构建交互距离场（IDF），并利用基于扩散的关系模型在推理时引导运动生成模型产生关系感知且语义对齐的人物-物体交互动作，在FullBodyManipulation数据集上显著超越SOTA。

## 研究背景与动机

**领域现状**：人物-物体交互（HOI）合成是虚拟现实、动画和机器人等领域的核心技术。近年来扩散模型的成功推动了大量工作将其应用于HOI生成，包括引入文本描述、物理力、手部关节位置、接触映射等先验信息，或通过后优化策略来改善交互质量。

**现有痛点**：现有方法在表示物体几何和建模人物-物体空间关系时存在严重的简化问题。(1) 物体表示过于简单——许多方法仅用物体质心或到人体最近的单个点来代表物体，忽略了物体整体的几何复杂度。(2) 关系建模不充分——基于接触的方法使用固定接触点，在动态多阶段交互中表现差；基于距离的方法只考虑人体关节与物体最近顶点之间的单一视角关系，未能全面捕捉互相的空间关系。

**核心矛盾**：高保真HOI生成需要精确的几何表示和复杂的时空关系建模，但高维动态交互的复杂性使得直接使用所有表面点不可行，而过度简化又损失了关键几何信息。

**本文目标** (1) 如何高效表示物体几何而不丢失关键细节？(2) 如何有效建模和利用人物-物体之间复杂的时空关系来引导运动生成？

**切入角度**：作者从物体采样和空间距离场的角度出发，用24个精心选择的关键点高效表示物体几何，构建人体关节-物体关键点之间的完整距离矩阵（IDF），然后训练一个专门的关系模型来学习这个距离场的分布，最终在推理时用关系模型引导运动生成。

**核心 idea**：用边界点+泊松盘采样的24个关键点表示物体几何，构建交互距离场IDF编码时空关系，再用扩散关系模型引导运动生成。

## 方法详解

### 整体框架

ROG是一个两阶段扩散框架。给定物体网格、人体骨架和文本提示：(1) 运动生成模型基于MDM产生人和物体的初始运动序列；(2) 关系模型接收从初始运动计算的IDF矩阵，输出精炼后的IDF；(3) 精炼的IDF通过梯度引导优化初始运动，使最终输出的交互更真实、更符合语义。两个模型分别训练，推理时通过引导机制协作。

### 关键设计

1. **高效物体几何表示（Object Key Points Sampling）**:

    - 功能：用24个关键点紧凑而全面地表示物体3D几何形状
    - 核心思路：首先计算物体的轴对齐包围盒（AABB），找到物体表面上离包围盒8个顶点最近的8个边界点，这些点捕捉物体的整体轮廓和极端位置。然后使用泊松盘采样（Poisson Disk Sampling, PDS）在物体表面均匀采样额外16个点，PDS通过最小距离约束确保采样点均匀分布，捕捉细节形状变化。共24个关键点 $\mathbf{P} = \{\mathbf{p}_1, ..., \mathbf{p}_{24}\}$，与人体骨架的24个关节点一一对应
    - 设计动机：直接使用全部表面顶点计算开销太大且冗余，用质心等极简表示又丢失几何信息。边界点+PDS点的组合既覆盖物体轮廓（粗粒度），又保留表面细节（细粒度），实现了精度和效率的平衡

2. **交互距离场（Interactive Distance Field, IDF）**:

    - 功能：以矩阵形式全面编码人物-物体在整个交互过程中的时空距离关系
    - 核心思路：对于N帧的HOI序列，构建3D距离矩阵 $\mathbf{D} \in \mathbb{R}^{24 \times 24 \times N}$，其中第$(i,j,n)$个元素 $\mathbf{D}_{i,j,n} = \|\mathbf{q}_{i,n} - \mathbf{p}_{j,n}\|_2^2$ 表示第$n$帧中第$i$个人体关节与第$j$个物体关键点之间的欧氏距离平方。训练时引入IDF Loss $\mathcal{L}_{IDF} = \|\mathbf{D}_{pr} - \mathbf{D}_{gt}\|_2^2$ 直接监督运动生成模型学习正确的空间关系，权重$\lambda_{IDF}=5.0$
    - 设计动机：现有方法要么只看接触/不接触的二值关系，要么只基于质心计算单一距离。IDF矩阵提供了人体所有关节与物体所有关键点之间完整的距离映射，能精确描述交互过程中谁靠近谁、何时接触、何时分离的动态变化

3. **扩散关系模型与引导机制（Relation Model & Guidance）**:

    - 功能：学习真实的IDF分布先验，在推理时引导运动生成模型产生更真实的交互
    - 核心思路：关系模型是一个基于Video Diffusion Transformer的扩散模型，输入带噪的IDF矩阵，目标是去噪恢复ground truth IDF。模型内部集成了空间自注意力（在$4 \times 4$的降维空间网格上捕捉不同身体部位与物体部位的依赖关系）和时间自注意力（沿时间维度N捕捉交互的动态演变）。推理时，运动生成模型先产生初始运动 $\tilde{\mathbf{m}}_0$，从中计算IDF矩阵 $\mathbf{D}$，输入关系模型得到精炼的 $\tilde{\mathbf{D}}$，然后通过引导损失 $L_{guidance} = \|\mathbf{D} - \tilde{\mathbf{D}}\|_2^2$ 用L-BFGS优化器反向传播梯度到初始运动上。引导仅在去噪过程的最后10个时间步执行
    - 设计动机：单独的运动生成模型可能产生不真实的接触和动态行为。通过单独训练一个关系模型来学习IDF的真实分布，然后用它在推理时"纠正"运动生成模型的输出

### 损失函数 / 训练策略

- **运动生成模型**：$\mathcal{L}_m = \mathcal{L}_{rec} + \lambda_{IDF}\mathcal{L}_{IDF}$，重建损失+IDF损失，$\lambda_{IDF}=5.0$
- **关系模型**：$\mathcal{L}_D = \|\mathbf{D}_0 - \tilde{\mathbf{D}}_0\|_2^2$，标准扩散去噪目标
- 两个模型分别独立训练，运动生成用8层Transformer + batch_size=64，关系模型用VDT配置 + batch_size=8
- 优化器均为AdamW，lr=$1 \times 10^{-4}$，1000步扩散，DDPM采样

## 实验关键数据

### 主实验

| 方法 | R-Precision Top-1↑ | R-Precision Top-3↑ | FID↓ | Contact%↑ | Collision% | MDev↓ |
|------|-------|-------|------|-----------|------------|-------|
| Real motions | 0.651 | 0.917 | 0.001 | 0.623 | 0.157 | 4.846 |
| InterGen | 0.490 | 0.685 | 19.038 | 0.179 | 0.156 | 39.795 |
| MDM | 0.495 | 0.681 | 9.775 | 0.349 | 0.210 | 9.549 |
| HOI-Diff | 0.534 | 0.722 | 11.875 | 0.372 | 0.175 | 58.728 |
| CHOIS | 0.630 | 0.844 | 5.227 | 0.444 | 0.208 | 13.408 |
| **ROG (Ours)** | **0.706** | **0.902** | **5.119** | **0.466** | **0.200** | **5.815** |

### 消融实验

| 组件 | obj-kp | IDF loss | Guidance | Top-1↑ | FID↓ | Contact% | MDev↓ |
|------|--------|----------|----------|--------|------|----------|-------|
| Baseline(MDM) | ✗ | ✗ | ✗ | 0.495 | 9.775 | 0.349 | 9.549 |
| +obj-kp | ✓ | ✗ | ✗ | 0.547 | 7.514 | 0.374 | 9.227 |
| +IDF loss | ✓ | ✓ | ✗ | 0.666 | 5.726 | 0.424 | 7.020 |
| +Guidance(C) | ✓ | ✓ | 质心 | 0.668 | 5.902 | 0.364 | 9.936 |
| +Guidance(D) | ✓ | ✓ | 完整IDF | **0.706** | **5.119** | **0.466** | **5.815** |

### 关键发现

- 引入物体关键点表示（obj-kp）使FID从9.775降至7.514，几何信息直接改善了运动真实度
- IDF Loss是最大的性能提升来源，R-Precision Top-1从0.547大幅提升到0.666
- 使用完整距离矩阵（D）的引导效果远优于仅用质心距离（C）的引导

## 亮点与洞察

- **几何表示设计精妙**：边界点+PDS采样的组合既保证了物体轮廓的完整覆盖，又确保了表面细节的均匀采样
- **IDF概念有启发性**：将离散的接触/距离关系统一为连续的3D距离场，为建模人物-物体交互提供了优雅的数学框架
- **两阶段分工清晰**：运动生成模型负责"做什么动作"，关系模型负责"动作是否合理"

## 局限与展望

- 数据集限制：排除了铰接物体，手指运动也未建模
- L-BFGS优化增加了推理时间开销
- 未来可扩展到多人-多物体交互场景

## 相关工作与启发

- **MDM**：运动生成模型的基础架构
- **CHOIS**：最强baseline，但需要额外控制信号
- 启发：IDF的概念可以推广到其他需要建模空间关系的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐（IDF概念新颖，关系模型引导机制设计巧妙）
- 实验充分度: ⭐⭐⭐⭐（定量、定性、消融实验全面）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐（对HOI生成领域有显著推动）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](../../ICCV2025/3d_vision/scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](../../ICCV2025/3d_vision/guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)

</div>

<!-- RELATED:END -->

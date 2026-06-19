---
title: >-
  [论文解读] Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling
description: >-
  [CVPR 2025][射频传播建模] 提出 RFScape 框架，通过为每个物体学习对象级的神经电磁属性表示，结合传统射线追踪的可组合性，在稀疏训练样本下实现高精度 RF 传播建模，比传统光线追踪提升 13 dB、比 SOTA 神经基线提升 5 dB。 领域现状：射频（RF）传播建模对无线通信网络规划至关重要…
tags:
  - "CVPR 2025"
  - "射频传播建模"
  - "神经对象表示"
  - "光线追踪"
  - "电磁仿真"
  - "毫米波"
---

# Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling

**会议**: CVPR 2025  
**arXiv**: [2411.18635](https://arxiv.org/abs/2411.18635)  
**代码**: 无  
**领域**: 信号通信 / 3D视觉  
**关键词**: 射频传播建模, 神经对象表示, 光线追踪, 电磁仿真, 毫米波

## 一句话总结

提出 RFScape 框架，通过为每个物体学习对象级的神经电磁属性表示，结合传统射线追踪的可组合性，在稀疏训练样本下实现高精度 RF 传播建模，比传统光线追踪提升 13 dB、比 SOTA 神经基线提升 5 dB。

## 研究背景与动机

**领域现状**：射频（RF）传播建模对无线通信网络规划至关重要。传统方法使用射线追踪（ray tracing）模拟电磁波传播路径，但依赖粗糙的几何近似和简化的材料模型。近年来，神经辐射场等方法在可见光渲染中取得成功，启发了将神经表示应用于 RF 建模的研究。

**现有痛点**：(1) 传统射线追踪对物体的电磁属性（反射、透射、散射系数）使用固定参数，无法捕捉复杂的 RF-物体交互；(2) 现有神经方法（如 NeRF2）需要对整个场景密集采样，场景组合和泛化能力差；(3) RF 信号的波长（毫米到厘米级）与可见光差异巨大，需要不同的物理建模。

**核心矛盾**：传统方法可组合但不精确，神经方法精确但不可组合——需要一个兼具两者优势的方案。

**本文目标**：学习对象级的 RF 电磁属性神经表示，可灵活组合到任意场景中并支持精确的 RF 传播建模。

**切入角度**：不同于端到端预测 RF 场，RFScape 将 RF 交互分解到单个物体上——每个物体的反射/透射/散射行为独立学习，场景预测通过组合各物体的交互来实现。

**核心 idea**：对象中心的神经 RF 属性表示 + 传统射线追踪的路径搜索 = 可组合、高精度的 RF 传播建模。

## 方法详解

### 整体框架

RFScape 分两阶段：(1) 为每个物体训练一个神经网络，学习其 RF 反射/透射/散射特性（输入为入射方向和位置，输出为改变后的信号幅度和相位）；(2) 推理时，用几何射线追踪确定传播路径，在每个碰撞点调用对应物体的神经模型计算 RF 交互，沿路径累积得到总信号。

### 关键设计

1. **对象级神经电磁表示**:

    - 功能：为单个物体学习 RF 电磁交互模型
    - 核心思路：每个物体用一个小型 MLP 表示，输入包括入射方向、碰撞位置和频率，输出反射/透射的幅度衰减和相位变化。训练数据来自该物体在孤立环境中的 RF 测量。关键约束是保持物理一致性——能量守恒、互易性等。
    - 设计动机：对象级别的粒度使得模型可以跨场景复用——同一把椅子在不同房间中的 RF 行为一致。

2. **可组合的场景推理**:

    - 功能：将独立学习的物体模型组合到新场景中
    - 核心思路：给定场景的几何布局，用射线追踪确定 RF 传播路径（直射、反射、绕射等），在每个路径节点调用对应物体的神经模型计算信号变化，最终将所有路径的贡献叠加（考虑相位干涉）。
    - 设计动机：传统射线追踪的可组合性是其在工程实践中广泛使用的核心优势，RFScape 保留这一特性的同时提升精度。

3. **稀疏训练策略**:

    - 功能：仅需少量测量数据即可训练有效的物体模型
    - 核心思路：利用 RF 交互的物理约束（如 Fresnel 方程作为先验）正则化神经网络，减少对训练数据的需求。
    - 设计动机：在实际场景中密集部署 RF 传感器成本很高，稀疏训练是实际部署的前提。

### 损失函数 / 训练策略

基于实测 RF 信号幅度（RSS）和相位的回归损失训练，结合物理约束正则化。

## 实验关键数据

### 主实验

| 方法 | 建模误差(dB)↓ |
|------|--------------|
| 传统射线追踪 | 基线 |
| NeRF2 | 基线-8dB |
| **RFScape** | **基线-13dB** |

*RFScape 比传统射线追踪提升 13 dB，比 SOTA 神经方法提升 5 dB*

### 关键发现
- 对象级表示在新场景组合中泛化良好
- 稀疏训练样本（几十个测量点）即可获得高质量模型
- 在 60 GHz 毫米波实测系统上验证了方法的有效性
- 不同材质物体的神经模型学到了物理上合理的反射/透射特性

## 亮点与洞察
- **物理引导的神经表示**：将传统 NeRF 的"体渲染"范式适配到 RF 域的"射线追踪"范式，保持了物理可解释性
- **可组合性是核心优势**：对象级粒度使模型具有乐高积木式的灵活性，是端到端方法无法比拟的
- **跨领域迁移**：将计算机视觉中的神经场方法迁移到无线通信领域

## 局限与展望
- 需要每个物体单独测量和训练，新物体需额外数据
- 当前仅在室内小场景验证，大规模室外场景的适用性待测试
- 绕射等复杂传播现象的建模可能需要更复杂的网络结构
- 未考虑动态场景（人员移动等）

## 相关工作与启发
- **vs NeRF2**: NeRF2 端到端建模整个场景的 RF 场，不可组合；RFScape 对象级建模，可组合
- **vs WiNeRT**: WiNeRT 用可微射线追踪训练，RFScape 用预训练的物体模型组合
- 方法论可迁移到声学传播建模（室内声学仿真）

## 评分
- 新颖性: ⭐⭐⭐⭐ 对象级 RF 神经表示 + 射线追踪组合的思路新颖
- 实验充分度: ⭐⭐⭐ 真实硬件验证但场景规模有限
- 写作质量: ⭐⭐⭐⭐ 跨域背景阐述清晰
- 价值: ⭐⭐⭐⭐ 对 RF 建模领域有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](../../ICML2025/others/improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](../../NeurIPS2025/others/modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[CVPR 2025\] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](../../ICML2025/others/curvature_enhanced_data_augmentation_for_regression.md)

</div>

<!-- RELATED:END -->

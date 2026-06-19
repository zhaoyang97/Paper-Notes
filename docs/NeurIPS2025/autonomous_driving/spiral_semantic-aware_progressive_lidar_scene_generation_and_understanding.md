---
title: >-
  [论文解读] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding
description: >-
  [NeurIPS 2025][自动驾驶][LiDAR生成] Spiral 提出了一种语义感知的 range-view LiDAR 扩散模型，同时生成深度、反射率图像和语义分割图，通过渐进式语义预测和闭环推理机制增强跨模态一致性，以最小参数量（61M）取得 SOTA 效果。 LiDAR 数据的大规模采集和标注成本极高…
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "LiDAR生成"
  - "扩散模型"
  - "语义分割"
  - "Range-View"
  - "闭环推理"
---

# SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2505.22643](https://arxiv.org/abs/2505.22643)  
**代码**: [GitHub](https://github.com/worldbench/SPIRAL)  
**领域**: 自动驾驶 / LiDAR生成  
**关键词**: LiDAR生成, 扩散模型, 语义分割, Range-View, 闭环推理

## 一句话总结

Spiral 提出了一种语义感知的 range-view LiDAR 扩散模型，同时生成深度、反射率图像和语义分割图，通过渐进式语义预测和闭环推理机制增强跨模态一致性，以最小参数量（61M）取得 SOTA 效果。

## 研究背景与动机

LiDAR 数据的大规模采集和标注成本极高，利用扩散模型生成合成 LiDAR 场景是缓解数据瓶颈的重要方向。现有生成方法分为体素方法和 range-view 方法两大类。体素方法（如 XCube、DynamicCity）能同时生成几何结构和语义标签，但内存消耗和计算开销大；range-view 方法（如 LiDARGen、R2DM）计算效率高，但只能生成无标注的深度和反射率图像。

**核心痛点**：现有 range-view 方法如果需要语义标签，只能采用两步管线——先生成无标注场景，再用预训练分割模型（如 RangeNet++）预测语义图。这种做法有两个关键问题：
1. 生成模型和分割模型独立训练，无法共享表征，训练效率低
2. 语义图是事后预测的，无法在生成过程中反向指导深度和反射率的生成，导致跨模态一致性差

**切入角度**：扩散模型本身具有强大的特征学习能力，可以在去噪过程中同步预测语义标签，并通过闭环机制让语义预测反向引导几何生成。

## 方法详解

### 整体框架

Spiral 采用 4 层 Efficient U-Net 作为骨干网络，基于连续时间 DDPM 框架。输入为加噪的深度和反射率图像 $x_t$ 以及语义图 $y$（编码为 RGB 图像），输出通过两个独立分支分别预测扩散残差 $\hat{\epsilon}_t$ 和语义标签 $\hat{y}_t$。模型在"无条件步"和"条件步"之间交替，通过两个互斥开关 $\mathcal{A}$ 和 $\mathcal{B}$ 控制切换。

### 关键设计

1. **完整语义感知（Complete Semantic Awareness）**:

    - 无条件步：模型同时预测语义图 $\hat{y}_t$ 和噪声 $\hat{\epsilon}_t$，损失为 MSE + 交叉熵
    - 条件步：以给定语义图 $y$ 为条件，仅预测去噪残差 $\hat{\epsilon}_t$，损失为 MSE
    - 训练时以 50% 概率随机切换两种步骤，统一损失函数为 $\mathcal{L} = \mathcal{L}_c \cdot \mathbb{I}(\psi \leq 0.5) + \mathcal{L}_u \cdot \mathbb{I}(\psi > 0.5)$

2. **渐进式语义预测（Progressive Semantic Predictions）**:

    - 推理时每步无条件去噪都输出一个中间语义图 $\hat{y}_t$
    - 使用指数移动平均（EMA）平滑预测结果：$\bar{y}_t = \alpha \cdot \hat{y}_t + (1-\alpha) \cdot \bar{y}_{t+1}$
    - 抑制扩散过程的随机波动，输出稳定的逐像素置信度分数
    - 最终 $\bar{y}_0$ 作为语义输出

3. **闭环推理（Closed-Loop Inference）**:

    - 推理从开环模式开始，执行无条件步
    - 当 $\bar{y}_t$ 中超过 $\delta$ 比例的像素置信度超过阈值 $\delta$（默认 0.8），切换到闭环模式
    - 闭环模式下交替执行无条件步和条件步：无条件步预测语义+噪声，条件步以当前语义图引导深度/反射率生成
    - 实现语义与几何的联合优化，增强跨模态一致性

### 语义感知评估指标

论文还提出了新的语义感知评估体系：
- **学习特征**：用 RangeNet++ 编码器和 LiDM 语义编码器分别提取特征并拼接，计算 S-FRD、S-FPD、S-MMD
- **规则特征**：对每个语义类别分别计算 BEV 2D 直方图并聚合为 $h^s \in \mathbb{R}^{C \times B \times B}$，计算 S-JSD、S-MMD

## 实验关键数据

### 主实验（SemanticKITTI）

| 方法 | 参数量 | S-FRD↓ | S-FPD↓ | S-JSD↓ |
|------|--------|--------|--------|--------|
| LiDARGen + RangeNet++ | 80M | 1216.61 | 710.79 | 28.65 |
| LiDM + RangeNet++ | 325M | — | 458.33 | 16.69 |
| R2DM + RangeNet++ | 81M | 559.26 | 363.16 | 18.13 |
| R2DM + SPVCNN++ | 128M | 555.09 | 351.73 | 18.67 |
| **Spiral (Ours)** | **61M** | **382.87** | **153.61** | **9.16** |

### 消融实验

| 配置 | S-FRD↓ | S-FPD↓ | 说明 |
|------|--------|--------|------|
| 无闭环推理 | 较高 | 较高 | 闭环机制显著提升跨模态一致性 |
| 无 EMA 平滑 | 较高 | 较高 | EMA 抑制去噪过程的随机性 |
| 阈值 δ=0.8 | 最优 | 最优 | 过低导致噪声污染，过高则闭环启动过晚 |

### 关键发现
- Spiral 以最小参数量（61M）超越所有两步方法（80-372M），S-FRD 提升 31%，S-FPD 提升 56%，S-JSD 提升 50%
- 更大的分割模型 SPVCNN++ 在生成数据上反而不如 RangeNet++，大模型对噪声更敏感
- Spiral 生成数据可有效用于下游分割训练的数据增强，减少标注成本
- nuScenes 数据集上同样取得最优表现，泛化性好

## 亮点与洞察

- **闭环推理机制非常巧妙**：将扩散模型的中间预测反馈为条件输入，实现语义和几何的相互增强，这种思路可推广到其他多模态生成任务
- **EMA 渐进式语义预测**：利用去噪过程的迭代特性，天然适合渐进式预测和置信度积累
- **统一训练而非两阶段**：避免了生成模型和分割模型的训练割裂，大幅减少参数量
- **提出了语义感知评估指标**：填补了带标签 LiDAR 场景生成质量评估的空白

## 局限与展望

- range-view 表示在高分辨率下可能丢失远距离物体的细节
- 闭环推理增加了推理步数和时间（需要交替执行两种步骤）
- 语义预测依赖扩散模型的特征学习能力，对稀有类别的分割可能不够精确
- 未探索文本条件生成、4D 动态场景生成等更复杂的设置

## 相关工作与启发

- **vs R2DM**: R2DM 只生成深度和反射率，需要额外分割模型；Spiral 统一生成三种模态
- **vs LiDM**: LiDM 支持语义条件生成，但需要预先提供语义图；Spiral 能自主预测语义
- **vs 体素方法（XCube、DynamicCity）**: 体素方法参数量大、计算开销高；Spiral 基于 range-view 表示更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 闭环推理和渐进语义预测是有新意的贡献，但核心扩散框架是标准的
- 实验充分度: ⭐⭐⭐⭐⭐ 两个标准数据集 + 新评估指标 + 丰富消融 + 下游应用验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表专业，动机讲解连贯
- 价值: ⭐⭐⭐⭐ 对自动驾驶数据生成有实际价值，但影响范围主要局限在 LiDAR 领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](../../ICCV2025/autonomous_driving/hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[CVPR 2026\] DrivePTS: A Progressive Learning Framework with Textual and Structural Enhancement for Driving Scene Generation](../../CVPR2026/autonomous_driving/drivepts_a_progressive_learning_framework_with_textual_and_structural_enhancemen.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](../../CVPR2025/autonomous_driving/exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] ProReflow: Progressive Reflow with Decomposed Velocity
description: >-
  [CVPR 2025][图像生成][Flow Matching] 提出渐进式 Reflow（逐步从多窗口到少窗口拉直扩散轨迹）和对齐 v-prediction（在速度匹配中优先匹配方向而非幅度），使 SDv1.5 在 4 步采样下达到接近 32 步 DDIM 的生成质量。 Flow matching 通过将预训练扩散模型的采…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "Flow Matching"
  - "渐进式Reflow"
  - "速度分解"
  - "少步生成"
  - "扩散加速"
---

# ProReflow: Progressive Reflow with Decomposed Velocity

**会议**: CVPR 2025  
**arXiv**: [2503.04824](https://arxiv.org/abs/2503.04824)  
**代码**: [GitHub](https://github.com/)  
**领域**: 图像生成/扩散模型加速  
**关键词**: Flow Matching, 渐进式Reflow, 速度分解, 少步生成, 扩散加速

## 一句话总结

提出渐进式 Reflow（逐步从多窗口到少窗口拉直扩散轨迹）和对齐 v-prediction（在速度匹配中优先匹配方向而非幅度），使 SDv1.5 在 4 步采样下达到接近 32 步 DDIM 的生成质量。

## 研究背景与动机

Flow matching 通过将预训练扩散模型的采样轨迹拉直为直线，实现少步甚至单步生成。标准 Reflow 的训练流程是直接训练模型在所有时间步上预测一致的速度。然而这种策略并未充分发挥 rectified flow 的潜力：

- **速度差异过大**：预训练扩散模型在不同时间步的速度差异显著（通过 L2 距离和余弦相似度验证），直接消除所有差异在优化上很困难
- **方向比幅度更重要**：速度可以分解为方向和幅度，对于轨迹"直线性"来说，方向匹配比幅度匹配更关键，但现有方法的 MSE 损失同等对待两者
- PeRFlow 虽然提出了分段拉直，但在单阶段内就试图将复杂轨迹近似为分段线性，优化目标仍然困难

关键观察：预训练模型在相邻时间步的速度相似度很高，这为先在局部窗口内拉直、再逐步合并窗口的课程学习策略提供了基础。

## 方法详解

### 整体框架

ProReflow 基于 PeRFlow 的分段 Reflow 框架，引入两个改进：渐进式 Reflow（多阶段从多窗口到少窗口）和对齐 v-prediction（在 MSE 损失中加入方向匹配损失）。

### 关键设计一：渐进式 Reflow（Progressive Reflow）

- **功能**：通过多阶段课程学习降低 Reflow 的优化难度
- **核心思路**：首先将扩散过程分为 $N$ 个局部时间窗口进行拉直，然后通过 Cross-windows Reflow 将相邻两个窗口合并为一个更大的窗口，逐步从 $N \to N/2 \to N/4 \to \ldots$ 直到目标窗口数。例如先用 8 窗口近似原始轨迹（容易），再合并为 4 窗口（中等），最终实现少步生成
- **设计动机**：直接从复杂轨迹到 4 段线性近似的"学习率" $\beta$ 较小（困难）。引入中间表示（8 窗口）后，每阶段的跨度更小、优化更稳定，类似于知识蒸馏中引入中间模型

Cross-windows Reflow 的核心是将相邻窗口 $[t_{k-1}, t_k]$ 和 $[t_k, t_{k+1}]$ 的两段轨迹引导为从 $z_{t_{k-1}}$ 直接到 $z_{t_{k+1}}$ 的单段直线。

### 关键设计二：对齐 v-prediction（Aligned V-Prediction）

- **功能**：在速度匹配中增加方向匹配的权重
- **核心思路**：在标准 MSE 损失之外，引入余弦相似度损失来约束速度方向一致性。总损失为 $\mathcal{L} = \mathcal{L}_{MSE} + \alpha \cdot (1 - \cos(v_{pred}, v_{target}))$，其中 $\alpha$ 控制方向匹配的强度
- **设计动机**：实验验证了方向噪声比幅度噪声对 FID 的影响更大——相同扰动幅度下，方向扰动导致的 FID 退化始终大于幅度扰动。MSE 损失对方向和幅度同等对待是次优的

### 关键设计三：多阶段训练策略

- **功能**：实现从教师模型到学生模型的渐进知识传递
- **核心思路**：ProReflow-I 进行一级合并（8窗口→4窗口），ProReflow-II 进行两级合并（8→4→2窗口）。每阶段使用上一阶段的模型生成 (noise, image) 配对数据用于下一阶段训练
- **设计动机**：基于特权信息蒸馏理论，中间表示可以更有效地传递知识

### 损失函数

$\mathcal{L} = \mathcal{L}_{MSE} + \alpha \cdot \mathcal{L}_{cos}$，其中 $\mathcal{L}_{MSE} = \|v_{target} - v_\theta(z_t, t)\|^2$，$\mathcal{L}_{cos} = 1 - \frac{v_{target} \cdot v_\theta}{\|v_{target}\| \|v_\theta\|}$。

## 实验关键数据

### 主实验：SDv1.5 在 MSCOCO-2014 val

| 方法 | 步数 | FID ↓ |
|------|------|-------|
| Teacher (DDIM) | 32 | 10.05 |
| InstaFlow (1-ReFlow) | 4 | 23.40 |
| 2-ReFlow | 4 | 21.64 |
| PeRFlow | 4 | 11.90 |
| **ProReflow-I** | 4 | **10.70** |

### MSCOCO-2017 对比

| 方法 | 4步 FID | 2步 FID |
|------|---------|---------|
| 2-ReFlow | 23.13 | 46.32 |
| PeRFlow | 14.15 | 28.92 |
| **ProReflow-I** | **12.19** | - |
| **ProReflow-II** | - | **24.59** |

### 消融实验

| 配置 | FID (4步) |
|------|----------|
| PeRFlow baseline | 14.15 |
| + Progressive Reflow only | 12.95 |
| + Aligned v-prediction only | 13.10 |
| + 两者结合 | **12.19** |

### 关键发现

- ProReflow-I 4 步 FID=10.70，接近 32 步 DDIM 教师模型 (10.05)
- 相比 2-ReFlow，4 步减少 10.94 FID，2 步减少 21.73 FID
- 渐进式 Reflow 和对齐 v-prediction 各自独立有效，组合后效果更优
- 方向扰动实验明确证明了速度方向对生成质量的关键性

## 亮点与洞察

1. **课程学习在生成模型中的巧妙应用**：从简单的局部拉直到困难的全局拉直，符合学习的自然规律
2. **方向 vs 幅度的观察有说服力**：通过对比实验清楚展示了方向匹配的重要性
3. **保持采样器通用性**：不改变网络结构，训练后的模型可用标准采样器

## 局限与展望

- 多阶段训练增加了训练成本（需多轮生成配对数据+训练）
- 仅在 SDv1.5 和 SDXL 上验证，对更新架构（如 DiT）的适用性未知
- 1 步生成质量仍有提升空间
- 对齐 v-prediction 的系数 $\alpha$ 需要手动调节

## 相关工作与启发

- **InstaFlow**：首次将 ReFlow 扩展到大规模文生图模型
- **PeRFlow**：提出 piecewise reflow 分段拉直
- **Progressive Distillation**：渐进式蒸馏的经典工作
- 渐进式思想可推广到其他需要轨迹拉直的任务

## 评分

⭐⭐⭐⭐ — 渐进式 Reflow 设计简洁有效，方向匹配的观察有实验支撑。4 步接近 32 步教师模型的结果实用价值大。但多阶段训练开销是需要考虑的成本。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Progressive Tempering Sampler with Diffusion](../../ICML2025/image_generation/progressive_tempering_sampler_with_diffusion.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](../../ICML2026/image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[CVPR 2026\] From Sketch to Fresco: Efficient Diffusion Transformer with Progressive Resolution](../../CVPR2026/image_generation/from_sketch_to_fresco_efficient_diffusion_transformer_with_progressive_resolutio.md)
- [\[CVPR 2026\] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation](../../CVPR2026/image_generation/vde_training-free_accelerating_rectified_flow_model_via_velocity_decomposition_a.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)

</div>

<!-- RELATED:END -->

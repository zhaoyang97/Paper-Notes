---
title: >-
  [论文解读] Entropy Rectifying Guidance for Diffusion and Flow Models
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出 Entropy Rectifying Guidance (ERG)，通过操控注意力层的 Hopfield 能量景观（温度缩放、步长调整）来获取弱预测信号，替代传统 CFG 中的无条件预测，在文本到图像、类条件和无条件生成中同时提升质量、多样性和一致性。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "guidance mechanism"
  - "注意力机制"
  - "classifier-free guidance"
  - "flow matching"
---

# Entropy Rectifying Guidance for Diffusion and Flow Models

**会议**: NeurIPS 2025  
**arXiv**: [2504.13987](https://arxiv.org/abs/2504.13987)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: diffusion models, guidance mechanism, attention energy, classifier-free guidance, flow matching

## 一句话总结

提出 Entropy Rectifying Guidance (ERG)，通过操控注意力层的 Hopfield 能量景观（温度缩放、步长调整）来获取弱预测信号，替代传统 CFG 中的无条件预测，在文本到图像、类条件和无条件生成中同时提升质量、多样性和一致性。

## 研究背景与动机

扩散模型和 flow matching 模型是当前图像生成的 SOTA 方法。Classifier-Free Guidance (CFG) 是最广泛使用的引导技术，通过结合条件和无条件预测来提升生成质量和一致性。然而 CFG 存在固有的 quality-diversity-consistency 三方权衡问题：

- **多样性下降**：较高的引导尺度导致生成样本趋于单一
- **过饱和**：过高的引导强度导致图像色彩过饱和
- **需要无条件训练**：需花费训练资源在无条件生成上
- **不适用于无条件采样**：CFG 依赖条件/无条件对比，无法用于纯无条件生成

现有改进方案如 AutoGuidance 需要额外弱模型（增加内存占用），SEG/SAG 为 U-Net 设计难以迁移到 DiT 架构。本文希望在单一模型内、无需额外训练的前提下，同时改善三个维度的性能。

## 方法详解

### 整体框架

ERG 的核心思想是：利用注意力层的 Hopfield 能量解释，通过修改注意力机制获得一个"较弱"的预测信号，然后将其与正常预测对比来实现引导。整个方法分为两个部分：

1. **I-ERG (Image ERG)**：修改去噪模型的注意力层能量景观
2. **C-ERG (Condition ERG)**：修改文本编码器的注意力层能量景观

最终的引导更新公式将正常去噪模型预测 D 与修改后的去噪模型预测 D_xi 进行加权组合，其中 w 为引导强度。D_xi 使用修改后注意力层的去噪模型，phi_c_tau 是通过修改文本编码器注意力获得的弱条件嵌入。

### 关键设计

**Hopfield 能量视角的注意力操控**：标准注意力可被视为 Hopfield 能量函数的 CCCP 更新。ERG 在该能量函数中引入推理时超参数，修改能量景观：

- **温度参数 tau**：控制 softmax 注意力的锐度。tau < 1 使注意力更均匀（平滑），tau > 1 使注意力更集中（锐利）
- **模式匹配权重 alpha**：控制状态模式匹配项相对于状态模式范数的重要性
- **步长 gamma**：梯度下降步长，控制能量优化的幅度
- **迭代次数 K**：能量景观优化的梯度下降步数

**多步梯度下降更新**（Algorithm 2）：在每个注意力层中执行 K 步梯度下降，更新 query：Q = Q - gamma * (Q - alpha * softmax(tau * beta * Q * K_T) * V)。当 alpha = gamma = tau = K = 1 时退化为标准注意力。

**文本编码器操控（C-ERG）**：对文本编码器（如 Llama3-8B、Flan-T5-XL）的每层自注意力引入温度缩放，降低 key-query 匹配的确定性，得到"模糊"的条件嵌入。C-ERG 在整个去噪过程中应用。

**去噪模型操控（I-ERG）**：仅在特定层和特定时间步（kickoff threshold kappa 之后）应用修改，避免采样早期过度惩罚负分量。

**与其他方法的组合**：ERG 可无缝与 CADS（条件退火扩散采样器）和 APG（自适应投影引导）组合，进一步提升性能。

### 损失函数 / 训练策略

ERG 是纯推理时方法，不需要任何训练修改或额外模型训练。所有超参数仅在推理时设定。模型使用 rectified flow-matching 训练，训练过程中两个文本编码器各自以 sqrt(0.1) 概率被禁用（约 10% 概率两者同时禁用）。

## 实验关键数据

### 主实验

**文本到图像生成**（COCO'14, 512 分辨率, 1.9B 参数模型）：

| 方法 | FID | Density | Coverage | CLIPScore | VQAScore | NFE |
|------|-----|---------|----------|-----------|----------|-----|
| CFG | 12.81 | 98.24 | 71.12 | 26.45 | 70.15 | 2 |
| APG | 11.88 | 104.07 | 73.06 | 26.54 | 72.47 | 2 |
| SAG* | 11.68 | 103.58 | 72.74 | 26.81 | 72.16 | 2 |
| **ERG** | 13.62 | **120.25** | **73.21** | **26.86** | **73.96** | 2 |
| ERG+APG | **11.37** | 115.08 | **80.50** | 26.74 | 73.55 | 2 |
| ERG+CADS | 12.87 | **128.54** | 76.23 | 26.75 | 73.45 | 2 |

**无条件生成**（T2I 模型，空 prompt）：

| 方法 | FID | Density | Coverage |
|------|-----|---------|----------|
| No guidance | 101.50 | 8.99 | 3.63 |
| SEG* | 37.75 | 55.56 | 34.79 |
| **ERG** | **36.25** | **55.84** | **51.59** |

**类条件生成**（ImageNet, DiT-XL/2, 512 分辨率）：

| 方法 | FID | Density | Coverage |
|------|-----|---------|----------|
| CFG | 5.65 | 146.97 | **86.70** |
| **ERG** | **4.56** | **163.63** | 86.13 |

### 消融实验

**各组件贡献分析**：

| C-ERG | I-ERG | gamma | FID | Density | Coverage | CLIP | VQA |
|-------|-------|-------|-----|---------|----------|------|-----|
| x | x | x | 12.81 | 98.24 | 71.12 | 26.45 | 70.15 |
| o | x | x | 13.06 | 109.52 | 72.06 | 26.73 | 73.10 |
| o | o | x | 13.62 | 120.25 | 73.21 | 26.86 | 73.96 |
| o | o | o | 13.62 | 123.65 | 74.07 | 26.81 | 74.67 |

- C-ERG 主要提升 CLIPScore 和 VQAScore（一致性）
- I-ERG 主要提升 Density 和 Coverage（质量和多样性）
- 多步梯度下降（K>1）未带来显著增益，默认 K=gamma=1

### 关键发现

1. ERG 相比 CFG，Density 提升 +22 点，VQAScore 提升 +3.8 点，同时 Coverage 也提升
2. ERG + APG 实现所有三个维度的 Pareto 前沿改进
3. 在无条件生成中，Coverage 从 34.79（SEG*）跃升至 51.59，提升约 48%
4. tau_c < 1 提升 CLIPScore，tau_c > 1 提升 Coverage，温度参数可灵活控制多样性-一致性权衡

## 亮点与洞察

- **理论优雅**：将注意力与 Hopfield 能量联系，为引导提供了能量景观视角的理论基础
- **通用性强**：适用于条件/无条件/类条件生成，不受架构限制（U-Net 或 DiT 均可）
- **零额外训练**：纯推理时方法，无需训练额外弱模型，内存开销与 CFG 相同
- **可组合性**：与 APG、CADS 等方法正交可组合，叠加使用效果更优
- **NFE 不增加**：每步仅需 2 次函数评估，与 CFG 相同

## 局限与展望

- FID 指标上 ERG 单独使用略逊于 SAG*，需配合 APG 才能达到最优
- 超参数空间较大（alpha, gamma, tau, K + kickoff threshold kappa），需要网格搜索
- 未与非恒定权重调度（如 time-dependent CFG schedules）结合
- 实验仅在流匹配架构上验证，未测试传统 DDPM/DDIM 采样器
- 对哪些层应用 I-ERG 的选择需要实验确定

## 相关工作与启发

- **SEG (Hong, 2024)**：通过高斯平滑注意力实现能量平滑引导，但限于 U-Net 且需 3 次 NFE
- **AutoGuidance (Karras et al., 2024)**：使用弱模型对比，但需额外模型和内存
- **APG (Sadat et al., 2025)**：投影引导解决过饱和问题，与 ERG 互补
- **CADS (Sadat et al., 2024)**：通过条件噪声增加多样性，可与 ERG 叠加
- 启发：注意力机制的能量解释可能为更多推理时干预方法提供理论工具

## 评分

- **创新性**：4/5 - Hopfield 能量视角下的注意力操控引导，理论新颖
- **实用性**：5/5 - 纯推理时，无训练开销，即插即用
- **实验充分度**：4/5 - 多任务多消融，但仅限自有模型
- **写作质量**：4/5 - 结构清晰，理论推导完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Neural Entropy](neural_entropy.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] EVODiff: Entropy-aware Variance Optimized Diffusion Inference](evodiff_entropy-aware_variance_optimized_diffusion_inference.md)
- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)

</div>

<!-- RELATED:END -->

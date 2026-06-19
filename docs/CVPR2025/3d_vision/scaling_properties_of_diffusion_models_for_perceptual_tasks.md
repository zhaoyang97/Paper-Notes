---
title: >-
  [论文解读] Scaling Properties of Diffusion Models for Perceptual Tasks
description: >-
  [CVPR 2025][3D视觉][扩散模型缩放] 本文系统研究了扩散模型在深度估计、光流预测和 amodal 分割等感知任务上的 scaling 特性，建立了训练和推理的 power law 缩放规律，并证明通过增加测试时计算（更多去噪步数和多预测集成）可以显著提升性能，在使用远少于 SOTA 的数据和计算量的情况下达到了竞争力性能。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "扩散模型缩放"
  - "深度估计"
  - "光流预测"
  - "感知任务"
  - "测试时计算"
---

# Scaling Properties of Diffusion Models for Perceptual Tasks

**会议**: CVPR 2025  
**arXiv**: [2411.08034](https://arxiv.org/abs/2411.08034)  
**代码**: [https://scaling-diffusion-perception.github.io](https://scaling-diffusion-perception.github.io)  
**领域**: 扩散模型 / 视觉感知  
**关键词**: 扩散模型缩放, 深度估计, 光流预测, 感知任务, 测试时计算

## 一句话总结

本文系统研究了扩散模型在深度估计、光流预测和 amodal 分割等感知任务上的 scaling 特性，建立了训练和推理的 power law 缩放规律，并证明通过增加测试时计算（更多去噪步数和多预测集成）可以显著提升性能，在使用远少于 SOTA 的数据和计算量的情况下达到了竞争力性能。

## 研究背景与动机

**领域现状**：扩散模型在图像/视频生成中展现了卓越的缩放特性，但在视觉感知（判别式）任务中的缩放行为研究不足。Marigold 等工作已经证明了将图像扩散模型用于深度估计的可行性，FlowDiffuser 用于光流，pix2gestalt 用于 amodal 分割，但这些工作独立进行，缺少统一框架和系统的缩放分析。

**现有痛点**：当前将扩散模型用于感知任务的方法主要依赖大规模预训练（如 Stable Diffusion 使用互联网规模数据），缺少对"如何高效缩放计算"的系统研究。实践中常面临计算预算有限的问题，但不清楚是应该增大模型、增加训练数据、提高分辨率还是增加推理计算。

**核心矛盾**：扩散模型的迭代去噪特性使其天然支持测试时计算缩放（增加步数/集成多预测），但缺乏系统的 scaling law 来指导最优的训练和推理配置，尤其是训练计算和测试时计算之间的 trade-off 不明确。

**本文目标**：统一多种视觉感知任务为图像到图像翻译框架，系统建立扩散模型在这些任务上的训练/推理缩放规律，提供 compute-optimal 方案。

**切入角度**：类比 LLM 领域 OpenAI o1 的测试时计算缩放——"让模型在推理时多思考 20 秒，效果等同于模型放大 10 万倍"。扩散模型的迭代去噪天然适合这种思路。

**核心 idea**：将深度估计、光流、amodal 分割统一为条件去噪扩散，在模型大小/预训练计算/分辨率/MoE upcycling/去噪步数/集成次数/噪声调度等多维度建立 scaling power law。

## 方法详解

### 整体框架

所有感知任务被统一为条件图像到图像翻译：给定 RGB 输入图像 $I$ 和可选条件图像，通过 Stable Diffusion VAE 编码到潜空间，RGB 潜码 $i_0$ 与随机加噪的 ground truth 潜码 $d_t$ 在通道维度拼接，送入 DiT 模型条件去噪。推理时从纯噪声出发，通过 DDIM 迭代去噪生成感知预测。预训练在 ImageNet-1K 上进行类条件图像生成，再微调到具体感知任务。

### 关键设计

1. **训练阶段缩放分析**:

    - 功能：找到模型大小、预训练计算、分辨率和 MoE 对下游性能的 power law 关系
    - 核心思路：(a) **模型大小**：训练 6 个 Dense DiT（14.8M 到 1.9B），发现预训练损失与计算量呈幂律 $L(C) = 0.23 \times C^{-0.0098}$。(b) **预训练计算**：固定 a4 (458M)，不同步数（60K-120K），更多预训练持续提升微调性能。(c) **分辨率**：256→512，token 数增 4×，深度估计呈幂律提升。(d) **MoE Upcycling**：将微调后 Dense 模型转 Sparse MoE 继续训练，AbsRel 平均提升 5.3%
    - 设计动机：建立 scaling law 使研究者能在给定预算下选择最优配置

2. **测试时计算缩放策略**:

    - 功能：利用扩散模型迭代和随机特性，推理时增加计算提升精度
    - 核心思路：三种互补策略。(a) **增加去噪步数**：$T \in \{1,2,5,10,20,50,100\}$，性能呈 power law 提升。(b) **测试时集成**：$N$ 次独立预测（$N \in \{1,2,5,10,15,20\}$），用逐像素 median 或 Marigold median compilation 合并，也呈 power law。(c) **噪声调度**：cosine schedule 将更多计算分配到早期去噪步（全局结构），比 linear 更有效
    - 设计动机：类比 LLM test-time scaling，扩散模型每步去噪的粗到细特性提供天然的"推理时思考"机制

3. **统一多任务模型**:

    - 功能：一个 DiT-XL 同时完成深度、光流和 amodal 分割
    - 核心思路：PatchEmbedRouter 根据任务类型路由到不同卷积层。混合数据集微调后用 upcycling 转 MoE 继续训练
    - 设计动机：验证缩放策略的跨任务泛化性

### 损失函数 / 训练策略

标准 MSE 去噪损失。微调用指数衰减学习率 $1.2 \times 10^{-4}$ → $1.2 \times 10^{-6}$。DiT 第一个卷积层通道翻倍适配 RGB+noise 拼接，权重减半初始化。推理用 DDIM + cosine beta schedule。最优推理配置：200 步去噪 + 5 次集成。

## 实验关键数据

### 主实验

**深度估计**：

| 方法 | Hypersim AbsRel↓ | ETH3D AbsRel↓ | NYUv2 AbsRel↓ | 预训练数据 |
|------|-----------------|---------------|--------------|----------|
| DPT | - | 7.8 | 9.8 | 大规模 |
| Marigold | 13.5 | 6.5 | 5.5 | 互联网规模 |
| **Ours** | **13.6** | **4.8** | 6.8 | ImageNet-1K |

**光流** (FlyingChairs)：Ours w/ ensemble 3.08 EPE vs DeepFlow 3.53

**Amodal 分割**：Ours 在 MP3D 上 63.9 mIOU vs pix2gestalt 61.5

### 消融实验

| 缩放维度 | 观察到的 power law | 提升幅度 |
|---------|-----------------|---------|
| 模型大小 (14.8M→1.9B) | $L(C) \propto C^{-0.0098}$ | 持续 |
| 预训练步数 (60K→120K) | 明确幂律 | 持续 |
| 分辨率 (256→512) | 4× tokens → 幂律提升 | 显著 |
| MoE Upcycling | 等效/超越更大 Dense 模型 | AbsRel -5.3% |
| 去噪步数 (1→100) | 明确幂律 | 显著 |
| 集成次数 (1→20) | 明确幂律 | 中等 |
| Cosine vs Linear schedule | Cosine 显著更优 | 显著 |

### 关键发现

- 仅用 ImageNet-1K 预训练，在 ETH3D 上超越 Marigold（4.8 vs 6.5）后者用互联网规模数据——说明 scaling 策略比数据规模更重要
- 测试时计算的"性价比"极高——无需额外训练，增加步数和集成就能大幅提升
- Cosine schedule 通过分配更多计算给全局结构重建（早期步），比 linear 更有效
- MoE upcycling 是"免费午餐"——将已微调模型廉价增大容量，可达到甚至超越更大 Dense 模型

## 亮点与洞察

- 首次为扩散模型在感知任务上建立系统 scaling power law，提供 compute-optimal 指导
- "测试时计算缩放"在视觉感知中的验证是重要贡献——暗示扩散模型不仅是生成工具，更是通用"迭代计算"范式
- 用远少于 SOTA 的数据达到竞争性能，证明 scaling 策略的重要性
- 训练 vs 推理计算的 trade-off 分析有很强的实践指导意义

## 局限与展望

- 感知性能仍有提升空间，特别是光流和 amodal 分割与专用方法有差距
- 推理速度是部署瓶颈——100 步 DDIM + 多次集成计算量大
- 当前 scaling law 基于 ImageNet-1K 建立，更大规模数据上的迁移性有待验证
- 未来可探索 consistency distillation 减少推理步数，以及更多感知任务的验证

## 相关工作与启发

- 与 Marigold 的关系：本文是 Marigold 的"缩放和泛化版"——统一多任务并系统研究缩放
- 与 DiT 的关系：利用 DiT 的标准架构和缩放方法论，从生成推广到感知
- 启发：扩散模型的"测试时思考"与 LLM chain-of-thought 类似——更多步骤 = 更深推理

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 各组件不新，但系统 scaling analysis 和测试时缩放视角有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多维度缩放实验量极大，three tasks 验证全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，power law 拟合结果展示直观
- **价值**: ⭐⭐⭐⭐ — 为扩散模型感知应用提供重要实践指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](../../ICCV2025/3d_vision/a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] Olympus: A Universal Task Router for Computer Vision Tasks](olympus_a_universal_task_router_for_computer_vision_tasks.md)

</div>

<!-- RELATED:END -->

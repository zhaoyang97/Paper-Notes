---
title: >-
  [论文解读] ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering
description: >-
  [ICML 2025][图像生成][层缓存] 将扩散模型中的中间层缓存技术（DeepCache）扩展到实时渲染 pipeline 中的 U-Net/U-Net++ 网络，通过帧差自适应缓存策略实现平均 1.4× 推理加速，且画质损失微乎其微。 领域现状：实时渲染（如 DLSS 4.0）大量依赖 U-Net 风格神经网络进行去…
tags:
  - "ICML 2025"
  - "图像生成"
  - "层缓存"
  - "实时渲染"
  - "U-Net"
  - "帧间一致性"
  - "推理加速"
---

# ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering

**会议**: ICML 2025  
**arXiv**: [2506.13814](https://arxiv.org/abs/2506.13814)  
**代码**: [https://ubc-aamodt-group.github.io/reframe-layer-caching/](https://ubc-aamodt-group.github.io/reframe-layer-caching/)  
**领域**: 图像生成  
**关键词**: 层缓存, 实时渲染, U-Net, 帧间一致性, 推理加速

## 一句话总结
将扩散模型中的中间层缓存技术（DeepCache）扩展到实时渲染 pipeline 中的 U-Net/U-Net++ 网络，通过帧差自适应缓存策略实现平均 1.4× 推理加速，且画质损失微乎其微。

## 研究背景与动机
**领域现状**：实时渲染（如 DLSS 4.0）大量依赖 U-Net 风格神经网络进行去噪、超分、帧外推等任务，网络推理已占渲染 pipeline 的显著比例。

**现有痛点**：(a) 渲染帧之间有极高的时间相关性，但每帧都做完整推理；(b) DeltaCNN 等利用帧间差异的方法因稀疏计算在现有 GPU 上难以加速；(c) DeepCache 仅针对扩散模型的多步推理设计。

**核心矛盾**：渲染中每一帧都必须单独产出高质量输出（不像扩散模型可以容忍中间步的近似），但帧间特征变化缓慢。

**本文目标** 如何在实时渲染的严格质量要求下利用帧间冗余？

**切入角度**：缓存 U-Net 深层的中间特征，跳过编码器-解码器的大部分计算，仅重新计算浅层（对新输入变化敏感的部分）。

**核心 idea**：实时渲染中帧间特征缓慢变化 → 缓存深层特征 → 自适应刷新策略 → 免训练加速。

## 方法详解

### 整体框架
在 U-Net 中，完整推理时缓存深层特征 $C_t$；后续帧仅计算首层 $X^0$ 和末层 $X^n$，用 $C_t$ 替代中间层。对 U-Net++ 则缓存所有除首层以外的 skip connection 分支。

### 关键设计

1. **层缓存机制**:

    - 功能：跳过编码器-解码器的深层计算
    - 核心思路：缓存 $C_t = X^{n-1}(\text{concat}(\ldots))$，后续帧 $O = X^n(\text{concat}(C_t, X^0(I)))$
    - 设计动机：深层特征捕获高层语义，帧间变化最慢；浅层捕获低层细节，对新输入最敏感

2. **帧差自适应策略 (Frame Deltas)**:

    - 功能：根据输入变化程度决定是否刷新缓存
    - 核心思路：计算当前输入与缓存帧的 SMAPE，超过阈值 $\tau$ 则刷新。分高灵敏度(Delta_H)和低灵敏度(Delta_L)两档
    - 设计动机：固定 Every-N 策略无法适应渲染中不可预测的场景变化（快速镜头移动 vs 静止）

3. **运动向量阈值策略**:

    - 功能：利用渲染 pipeline 已有的运动向量判断是否刷新
    - 核心思路：当平均运动超过阈值 $\tau$ 时刷新缓存
    - 设计动机：无额外存储开销（运动向量已是渲染 pipeline 的副产物）

### 训练策略
- **完全免训练**：不修改网络权重，只在推理时增加缓存逻辑
- 可与量化、剪枝等正交技术组合使用

## 实验关键数据

### 主实验

| 任务 | 场景 | 策略 | 加速比 ↑ | FLIP ↓ | SSIM ↑ |
|------|------|------|---------|--------|--------|
| 帧外推 | Sun Temple | Delta_H | 1.42× | 0.017 | 0.994 |
| 帧外推 | Sun Temple | Delta_L | 1.72× | 0.033 | 0.984 |
| 超分辨率 | Sun Temple | Delta_H | 1.30× | 0.049 | 0.970 |
| 超分辨率 | Sun Temple | Delta_L | 1.85× | 0.118 | 0.930 |
| 图像合成 | Garden Chair | Delta_H | 1.05× | 0.001 | 1.000 |

### 消融实验

| 策略 | 平均跳帧率 | 平均加速 | FLIP | 说明 |
|------|-----------|---------|------|------|
| Every-2 | 50% | ~1.4× | 中等 | 固定间隔 |
| Every-4 | 75% | ~1.7× | 较高 | 运动快时质量骤降 |
| Delta_H | 30-50% | 1.1-1.4× | **最低** | 自适应保质量 |
| Delta_L | 60-80% | 1.5-1.9× | 低 | 自适应平衡 |

### 关键发现
- FLIP < 0.05 在渲染领域被认为是可接受的质量损失（参考值 0.05-0.28）
- 帧外推任务受益最大（连续帧变化最平滑），超分辨率次之
- 自适应策略避免了固定策略在快速镜头移动时的画质骤降

## 亮点与洞察
- **免训练 + 通用性**：无需重新训练网络，可应用于任何含 skip connection 的编码器-解码器网络
- **对 U-Net++ 的扩展**：首次将缓存技术从 U-Net 扩展到 U-Net++
- **省下的计算可回馈渲染**：节省的推理时间可用于提高光追采样率，总体质量反而可能提升

## 相关工作与启发
- **vs DeepCache**: DeepCache 针对扩散模型的固定步数推理，ReFrame 针对渲染的单帧输出需求，自适应刷新是关键区别
- **vs DeltaCNN**: DeltaCNN 利用像素级差异稀疏化计算，但现有 GPU 硬件难以加速稀疏操作；ReFrame 的层级缓存完全兼容现有硬件
- **vs DLSS**: DLSS 本身就使用 U-Net，ReFrame 可作为其进一步加速的组件
- 缓存策略与 DeltaCNN 理论上可以组合：缓存帧只计算浅层，浅层内部再用 delta 稀疏化

## 局限与展望
- 在 RTX 2080 Ti 上测试，未验证在最新 GPU（RTX 4090/5090）上的收益
- 自适应策略的阈值需要任务特定调优，缺乏自动化方法
- 仅测试了 3 个渲染任务和 5 个场景，覆盖面有限
- 对快速场景切换（如游戏中的传送）的处理未讨论
- 缓存一致性在多玩家同步渲染中的挑战未考虑

## 评分
- 新颖性: ⭐⭐⭐ DeepCache 到渲染的迁移较直接，自适应策略是增量贡献
- 实验充分度: ⭐⭐⭐ 任务和场景数量偏少
- 写作质量: ⭐⭐⭐⭐ 清晰系统，图示丰富
- 价值: ⭐⭐⭐⭐ 对渲染 pipeline 优化有实用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](../../NeurIPS2025/image_generation/real-time_execution_of_action_chunking_flow_policies.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2026\] SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching](../../CVPR2026/image_generation/sencache_accelerating_diffusion_model_inference_via_sensitivity-aware_caching.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](../../CVPR2025/image_generation/semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)

</div>

<!-- RELATED:END -->

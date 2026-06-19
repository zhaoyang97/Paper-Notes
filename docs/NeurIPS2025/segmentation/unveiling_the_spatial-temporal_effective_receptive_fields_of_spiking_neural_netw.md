---
title: >-
  [论文解读] Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks
description: >-
  [NeurIPS 2025][语义分割][spiking neural networks] 提出时空有效感受野（ST-ERF）分析框架来诊断 Transformer-based SNN 在视觉长序列建模中的瓶颈——缺乏全局感受野，并据此设计 MLPixer 和 SRB 两种通道混合器来增强 SNN 的全局建模能力。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "spiking neural networks"
  - "effective receptive field"
  - "Transformer"
  - "channel mixer"
  - "visual long-sequence modeling"
---

# Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2510.21403](https://arxiv.org/abs/2510.21403)  
**代码**: [https://github.com/EricZhang1412/Spatial-temporal-ERF](https://github.com/EricZhang1412/Spatial-temporal-ERF)  
**领域**: 图像分割  
**关键词**: spiking neural networks, effective receptive field, Transformer, channel mixer, visual long-sequence modeling

## 一句话总结

提出时空有效感受野（ST-ERF）分析框架来诊断 Transformer-based SNN 在视觉长序列建模中的瓶颈——缺乏全局感受野，并据此设计 MLPixer 和 SRB 两种通道混合器来增强 SNN 的全局建模能力。

## 研究背景与动机

脉冲神经网络（SNN）因事件驱动的特性具有高能效潜力，已在图像分类等任务取得进展。然而在**视觉长序列建模任务**（目标检测、语义分割等）上，SNN 仍远落后于 ANN。

**现有痛点**：

**SNN 在密集预测任务上表现差**：这些任务需要对整幅图像做空间密集输出，要求模型具备建模长距离空间依赖的能力

**Transformer-based SNN 未充分发挥全局建模潜力**：虽然引入了 Transformer 的自注意力机制，但现有设计（如 Spike-driven Transformer）仍大量使用卷积操作作为通道混合器，引入了局部性偏置

**缺乏分析工具**：传统 ERF 框架仅考虑空间维度，无法表征 SNN 固有的时空动态特性

**核心矛盾**：Transformer-based SNN 理论上应该有全局感受野，但实际上由于卷积通道混合器的局部性偏置，导致早期阶段无法建立有效的全局感受野，限制了长序列建模性能。

**切入角度**：(1) 扩展 ERF 到时间维度提出 ST-ERF 分析框架，量化诊断问题所在；(2) 基于诊断结果，用 MLP 替换卷积来设计新的通道混合器，消除局部性偏置。

## 方法详解

### 整体框架

1. **分析阶段**：用 ST-ERF 框架分析现有 Transformer-based SNN（Spikformer、SDT-V1、Meta-SDT 等）的感受野行为
2. **设计阶段**：提出 MLPixer 和 SRB 两种通道混合器，替换 Meta-SDT 前两个阶段的卷积通道混合器
3. **验证阶段**：在 COCO 2017 目标检测和 ADE20K 语义分割上验证

### 关键设计

1. **ST-ERF 理论框架**：

    - **功能**：量化 SNN 中各输入特征在不同时空位置对输出的贡献
    - **核心定义**：$\text{ERF}^{(\mathcal{S},\mathcal{T})}_{(i,j)}[y_{(m,n)}[t], \tau; \mathbf{x}] = \frac{\partial y_{(m,n)}[t]}{\partial x_{(i,j)}[t-\tau]}$
    - 空间 ERF 是 ST-ERF 在所有时间步上的加权平均；时间 ERF 是 ST-ERF 在空间维度上的积分
    - **Loss-Derived 计算方法**：利用 PyTorch 自动微分，通过设置特定的梯度刺激（gradient stimuli）高效计算。具体地，将中心位置所有通道和时间步的梯度设为 1，反向传播即可得到空间 ERF
    - **设计动机**：传统 ERF 接无法处理 SNN 的时间动态，ST-ERF 将时间维度纳入分析

2. **MLPixer（MLP-based Mixer）**：

    - **功能**：完全用 MLP 替换卷积通道混合器
    - **核心设计**：$\text{MLPixer}(\mathbf{X}) = \text{BN}(\text{MLP}(\mathbb{SN}\{\text{BN}(\text{MLP}\{\mathbb{SN}(\mathbf{X})\})\}))$
    - 两层 MLP + 批归一化 + 脉冲神经元的堆叠
    - **设计动机**：MLP 是逐像素操作，不引入空间局部性偏置，使通道混合时不会破坏全局空间特征。ST-ERF 可视化显示 MLPixer 确实实现了更广泛的全局感受野

3. **SRB（Splash-and-Reconstruct Block）**：

    - **功能**：折中方案——保留第一层卷积用于局部特征提取，第二层用 MLP
    - **核心设计**：$\text{SRB}(\mathbf{X}) = \text{BN}(\text{MLP}(\mathbb{SN}\{\text{BN}(\text{Conv}\{\mathbb{SN}(\mathbf{X})\})\}))$
    - 第一层用 1×1 卷积，第二层用 MLP
    - **设计动机**：在减少参数量的同时保持性能。SRB 在准确率和模型大小之间达到最优平衡

### 损失函数 / 训练策略

- 集成到 Meta-SDT 架构中，替换前两个阶段的通道混合器，后两个阶段保持 Transformer-SNN 块不变
- 采用膜快捷残差连接机制维持网络的脉冲驱动特性
- 目标检测使用 Mask R-CNN + 1× 训练schedule，语义分割使用 Semantic FPN + 160k iterations
- 所有骨干网络在 ImageNet-1K 上预训练

## 实验关键数据

### 主实验

**COCO 2017 目标检测（Mask R-CNN, 1× schedule）**：

| 架构 | 参数量 | AP^b | AP^b_50 | AP^m | AP^m_50 |
|------|--------|------|---------|------|---------|
| SDTv3-T | 25M | 15.2 | 35.5 | 15.2 | 33.0 |
| SDTv3-T + SRB(ε4) | 25M | 18.2 | 39.2 | 17.5 | 34.8 |
| SDTv3-B | 39M | 21.7 | 46.9 | 20.1 | 41.8 |
| SDTv3-B + SRB(ε4) | 37M | **25.8** | **48.9** | **22.5** | **43.9** |

SRB 变体在 Base 上 AP^b_50 提升 4.26%（46.9→48.9），同时参数量还减少了 2M。Tiny 上提升更大（10.42%）。

**ADE20K 语义分割（Semantic FPN, 160k iter）**：

| 架构 | 通道混合器 | 参数量 | mIoU |
|------|-----------|--------|------|
| SDTv3-T | Conv(ε4) | 6.5M | 34.9 |
| SDTv3-T + SRB(ε4) | SRB | 6.2M (↓0.3) | **38.2** (↑3.3) |
| SDTv3-B | Conv(ε4) | 20.4M | 41.1 |
| SDTv3-B + SRB(ε4) | SRB | 19.2M (↓1.2) | **43.7** (↑2.6) |

SRB 在减少参数的同时大幅提升 mIoU。

### 消融实验

| 混合器类型 | 参数变化 | mIoU 变化 | 说明 |
|-----------|---------|----------|------|
| MLPixer(ε4) | ↓0.6M | +0.0 | 参数最少但 Tiny 上提升有限 |
| MLPixer(ε6) | +0.1M | +1.0 | 增大扩展比有效 |
| SRB(ε4) | ↓0.3M | **+3.3** | 最佳准确率-参数平衡 |

**事件追踪（FE108 & VisEvent）**：

| 架构 | FE108 AUC | VisEvent AUC |
|------|-----------|-------------|
| SD-Track (Tiny) | 56.7% | 35.4% |
| + MLPixer(ε6) | **57.9%** | 34.5% |
| + SRB(ε4) | 58.2% | 33.8% |

在事件追踪任务上也有提升，但 VisEvent 上略有下降。

### 关键发现

1. **SDT-V1 的 ERF 问题**：SPS 模块中的多层卷积使感受野过度集中于中心，限制了空间范围
2. **Meta-SDT 的 ERF 问题**：RepConv 的引入增强了局部提取但约束了远距离聚合
3. **MLPixer 在 Stage 1 就建立了全局 ERF**，随网络加深收缩到特定区域
4. **SRB 在 Stage 2 开始形成全局 ERF**，行为略有不同
5. 减少卷积使用确实能使 SNN 获得更全局的感受野

## 亮点与洞察

1. **ST-ERF 分析框架**：填补了 SNN 感受野分析的理论空白，为 SNN 架构优化提供了量化工具
2. **问题诊断驱动设计**：先用 ST-ERF 发现问题（缺乏全局感受野），再针对性设计解决方案
3. **简洁有效的改进**：仅替换前两阶段的通道混合器就能显著提升性能，修改量小
4. **减参提性能**：SRB 在参数减少的情况下反而提升性能，说明卷积在这些位置是冗余的

## 局限与展望

- MLPixer 虽然全局 ERF 更强，但在某些任务上不如 SRB，说明完全去除局部特征提取未必最优
- 仅在 Meta-SDT 架构上验证，对其他 SNN 架构的适用性有待检验
- 事件追踪的 VisEvent 数据集上效果下降，说明全局感受野未必在所有任务上有利
- ST-ERF 分析目前主要用于可视化和定性分析，缺乏量化指标来指导架构搜索

## 相关工作与启发

- **ERF 理论 (Luo et al., 2016)**：本文将其扩展到时间维度，方法论贡献明确
- **MLP-Mixer (Tolstikhin et al., 2021)**：启发了用 MLP 替换卷积的思路
- **Meta-SDT (Yao et al., 2025)**：本文的基线架构，通过分析其瓶颈来改进
- **启发**：感受野分析对神经架构设计有很强的指导作用，可推广到更多 SNN 任务

## 评分

- 新颖性: ⭐⭐⭐⭐ ST-ERF 框架是有价值的理论贡献，设计改进相对直接
- 实验充分度: ⭐⭐⭐⭐ 覆盖检测、分割、事件追踪，可视化分析丰富
- 写作质量: ⭐⭐⭐⭐ 从分析到设计逻辑链清晰
- 价值: ⭐⭐⭐⭐ 对 SNN 架构设计有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking](step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](../../ICCV2025/segmentation/partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](../../ECCV2024/segmentation/long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)

</div>

<!-- RELATED:END -->

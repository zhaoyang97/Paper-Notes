---
title: >-
  [论文解读] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection
description: >-
  [AAAI 2026][语义分割][SAM2] 提出 SAM-DAQ，通过深度引导并行适配器（DPA）和查询驱动时序记忆（QTM）模块将 SAM2 适配到 RGB-D 视频显著性检测任务，解决了手动提示依赖、高显存消耗和计算负担三大挑战。 视频显著性目标检测（VSOD）旨在识别视频中最具吸引力的物体。结合深度信息的 RGB-…
tags:
  - "AAAI 2026"
  - "语义分割"
  - "SAM2"
  - "RGB-D显著性检测"
  - "视频理解"
  - "深度引导"
  - "查询驱动记忆"
---

# SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection

**会议**: AAAI 2026  
**arXiv**: [2511.09870](https://arxiv.org/abs/2511.09870)  
**代码**: [https://github.com/LinJ0866/SAM-DAQ](https://github.com/LinJ0866/SAM-DAQ)  
**领域**: 分割  
**关键词**: SAM2, RGB-D显著性检测, 视频理解, 深度引导, 查询驱动记忆

## 一句话总结

提出 SAM-DAQ，通过深度引导并行适配器（DPA）和查询驱动时序记忆（QTM）模块将 SAM2 适配到 RGB-D 视频显著性检测任务，解决了手动提示依赖、高显存消耗和计算负担三大挑战。

## 研究背景与动机

视频显著性目标检测（VSOD）旨在识别视频中最具吸引力的物体。结合深度信息的 RGB-D VSOD 可以利用空间结构信息有效缓解杂乱背景、遮挡和低光照等挑战。然而，将 SAM2 直接应用于 RGB-D VSOD 面临**三大关键问题**：

**手动提示依赖**：SAM2 需要点、框或掩码等人工提示来引导分割，但 RGB-D VSOD 推理时无法提供此类信息。现有去提示方案（如生成伪提示或仅用编码器提取特征）要么效果有限，要么未充分利用 SAM2 的架构优势。

**串行适配器的高显存消耗**：现有的参数高效微调方法（如在 Transformer 块之间插入串行 adapter 或 LoRA）由于反向传播梯度需要穿过整个编码器，导致训练时 GPU 显存消耗极高（可达 91-95 GB）。

**记忆注意力的计算负担**：SAM2 的记忆机制通过记忆库捕获帧间依赖，但当前帧特征与大型记忆库之间的关联计算代价高昂。

核心思路：（1）用并行跳跃连接式适配器替代串行适配器，大幅降低显存；（2）用可学习查询替代记忆库和提示embedding，统一时序建模和提示生成。

## 方法详解

### 整体框架

SAM-DAQ 基于 SAM2-Large 构建，包含三个核心组件：

1. **并行适配器多模态图像编码器（PAMIE）**：以跳跃连接方式嵌入深度引导并行适配器，在冻结编码器下实现无提示微调和 RGB-D 特征融合
2. **查询驱动时序记忆模块（QTM）**：用帧级查询和视频级查询替代记忆库和提示嵌入，选择性提取时序一致性特征
3. **掩码解码器**：沿用 SAM2 原始解码器

### 关键设计

#### 1. **并行适配器多模态图像编码器（PAMIE）**

**深度适配器**：在每个 Hiera 块的输入和输出之间以跳跃连接方式插入 adapter：

$$\tilde{\mathbf{F}}_D^{i-1} = \text{Adapter}(\mathbf{F}_D^{i-1})$$
$$\mathbf{F}_D^i = \text{Hiera}^i(\mathbf{F}_D^{i-1}) + \text{DS}(\tilde{\mathbf{F}}_D^{i-1})$$

其中 Adapter 由下投影线性层 + 激活函数 + 上投影线性层组成，DS 为双线性下采样。

**深度引导并行适配器（DPA）**：将 RGB 特征和深度特征拼接后输入适配器：

$$\tilde{\mathbf{F}}_{RGB}^{i-1} = \text{Adapter}(\text{Cat}(\mathbf{F}_{RGB}^{i-1}, \mathbf{F}_D^{i-1}))$$
$$\mathbf{F}_{RGB}^i = \text{Hiera}^i(\mathbf{F}_{RGB}^{i-1}) + \text{DS}(\tilde{\mathbf{F}}_{RGB}^{i-1})$$

**设计动机**：并行跳跃连接允许梯度绕过沉重的 Transformer 计算直接反传，显存消耗从串行 adapter 的 91.9 GB 降至 21.0 GB。同时通过 Cat 操作在 adapter 中实现 RGB-D 特征融合。

经过 FPN 后生成三级图像嵌入 $\mathbf{E}_I = \{\mathbf{E}_I^i\}_{i=2}^{4}$。此外引入**自推理方案**：对每级图像嵌入用轻量卷积+sigmoid 生成中间预测，仅在最高层施加监督。

#### 2. **查询驱动时序记忆模块（QTM）**

引入两组可学习查询：

- **帧级查询** $\mathbf{Q}_f \in \mathbb{R}^{N_f \times c}$（$N_f = 30$）：静态查询，与每帧的最高层图像嵌入交互，提取显著性相关的帧特征
- **视频级查询** $\mathbf{Q}_v \in \mathbb{R}^{N_v \times c}$（$N_v = 8$）：动态查询，跨帧迭代更新，捕获时序依赖

交互过程：

$$\mathbf{E}_f = \text{Linear}(\mathbf{Q}_f' \cdot \mathbf{E}_I^4)$$
$$\tilde{\mathbf{Q}}_v = \text{CA}(\mathbf{Q}_v', \mathbf{E}_f) + \mathbf{Q}_v'$$

视频级查询 $\tilde{\mathbf{Q}}_v$ 与 $\mathbf{E}_I^4$ 逐元素相乘生成可学习嵌入 $\mathbf{E}_L$，替代 SAM 的稀疏提示嵌入。

**时序更新机制**：使用 SAM2 的记忆编码器处理当前帧的图像嵌入和预测结果：

$$\mathbf{F}_m = \text{Linear}(\text{ME}(\mathbf{E}_{I,t}, \mathbf{P}_t))$$
$$\mathbf{Q}_{v,t+1} = \mathbf{Q}_{v,t} + \text{FFN}(\text{SA}(\text{CA}(\mathbf{Q}_{v,t}, \mathbf{F}_m)))$$

**设计动机**：（1）帧级查询通过 token 级注意力选择性关注视觉吸引区域，而非像素级密集特征匹配；（2）视频级查询通过迭代更新建立时序依赖，替代大型记忆库的高计算开销；（3）仅用稀疏嵌入（而非密集嵌入或两者组合）效果最优，因为查询的 token 级交互与 SAM 预训练中的稀疏嵌入结构一致。

#### 3. **掩码解码器**

沿用 SAM2 原始掩码解码器，接收可学习嵌入 $\mathbf{E}_L$ 和多级图像嵌入 $\mathbf{E}_I$ 生成最终分割结果。

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \alpha \cdot \mathcal{L}_{inter}$$

- $\mathcal{L}_{pred}$：最终预测结果的 BCE 损失
- $\mathcal{L}_{inter}$：中间预测结果的 BCE 损失
- $\alpha$：中间损失权重

训练设置：SAM 编码器完全冻结，仅训练适配器和 QTM 模块（19.2M 可训练参数），输入分辨率 1024×1024，每个视频随机采样 10 帧，AdamW 优化器（lr=0.0001，weight decay=0.05），batch size=1，训练 2000 迭代。**在单张 RTX-3090（24GB）上仅需 3 小时即可训练完成**。

## 实验关键数据

### 主实验

**三数据集定量对比**：

| 方法 | 来源 | RDVS $E_\xi$↑ | RDVS $S_\alpha$↑ | ViDSOD-100 $F_\beta$↑ | ViDSOD-100 MAE↓ | DViSal $F_\beta$↑ | DViSal MAE↓ |
|------|------|-----------|-----------|------------|---------|-----------|---------|
| DCTNet+ | TIP'24 | 0.909 | 0.876 | 0.809 | 0.030 | 0.689 | 0.095 |
| MDSAM | MM'24 | 0.813 | 0.791 | 0.815 | 0.026 | 0.715 | 0.071 |
| SAM2-UNet | arXiv'24 | 0.888 | 0.843 | 0.829 | 0.025 | 0.747 | 0.064 |
| KAN-SAM | ICME'25 | 0.888 | 0.854 | 0.846 | 0.025 | 0.783 | 0.052 |
| **SAM-DAQ** | **本文** | **0.913** | **0.879** | **0.868** | **0.020** | **0.818** | **0.046** |

相比 KAN-SAM，SAM-DAQ 在 E-measure、S-measure、F-measure 上分别平均提升 1.5%、1.0%、2.4%。

### 消融实验

**PAMIE 消融（RDVS 数据集）**：

| 配置 | 可训练/总参数(M) | 显存(GB) | $E_\xi$ | $S_\alpha$ | $F_\beta$ | MAE |
|------|------------|---------|---------|---------|---------|-----|
| w/o 深度投影器 | - | 20.3 | 0.899 | 0.870 | 0.808 | 0.023 |
| 串行 adapter | 17.4/236.0 | **91.9** | 0.860 | 0.830 | 0.778 | 0.028 |
| LoRA | 56.0/274.6 | **95.0** | 0.889 | 0.877 | 0.824 | 0.027 |
| w/o 多模态融合 | - | 17.9 | 0.876 | 0.853 | 0.782 | 0.029 |
| **DPA（本文）** | **19.2/237.9** | **21.0** | **0.913** | **0.879** | **0.827** | **0.026** |

**QTM 嵌入策略消融**：

| 策略 | $E_\xi$ | $S_\alpha$ | $F_\beta$ | MAE | 说明 |
|------|---------|---------|---------|-----|------|
| 稀疏嵌入（本文） | **0.913** | **0.879** | **0.827** | **0.026** | 最优 |
| 密集嵌入 | 0.875 | 0.856 | 0.783 | 0.032 | 密集不适配 |
| 两者结合 | 0.862 | 0.839 | 0.763 | 0.033 | 冲突反而下降 |

**更新机制消融**：

| 策略 | $E_\xi$ | $F_\beta$ | 说明 |
|------|---------|---------|------|
| 无更新 | 0.883 | 0.788 | 时序信息缺失 |
| SAM2 记忆库 | 0.853 | 0.796 | 传统方案 |
| 乘法更新 | 0.895 | 0.804 | 次优 |
| **加法更新（本文）** | **0.913** | **0.827** | 最优 |

### 关键发现

- **并行 vs 串行适配器**：并行跳跃连接将显存从 91.9GB 降至 21.0GB（降低 77%），同时性能更优（$E_\xi$ 0.913 vs 0.860）
- **仅稀疏嵌入最优**：仅用稀疏嵌入优于密集或两者组合，因为 QTM 的 token 级交互与 SAM 预训练的稀疏嵌入在结构上一致
- **查询数量敏感**：视频级查询 8 个、帧级查询 30 个为最优配置；减少视频级查询（5个）或增加（10个）都会降低性能
- **隐藏维度 64 最优**：高于或低于都导致性能下降
- **中间监督仅在最高层有效**：在低层额外添加监督反而降低整体性能
- **加法更新优于乘法更新和 SAM2 原始记忆库**

## 亮点与洞察

1. **并行跳跃连接适配器**是解决 SAM 微调显存问题的优雅方案：梯度不需要穿过冻结的 Transformer，显存降低 4 倍以上
2. **查询驱动的设计统一了提示生成和时序建模**两个看似不同的问题：可学习查询既充当 SAM 的提示嵌入，又承担跨帧记忆的角色
3. **极高的训练效率**：单卡 RTX-3090 仅需 3 小时，这对实际部署非常友好
4. **仅用稀疏嵌入优于密集嵌入**的发现有深度：SAM 预训练时的稀疏提示范式决定了微调时也应保持结构一致性

## 局限与展望

- 仅在 RGB-D VSOD 上验证，未扩展到多目标视频分割或其他多模态任务
- 视频级查询数量固定，无法自适应视频长度或场景复杂度
- 深度图的质量变化对性能的影响未深入分析
- 未考虑光流等运动信息，可能在快速运动场景中受限
- 帧级查询是静态的，可探索动态帧级查询的可能性

## 相关工作与启发

- 并行适配器的低显存优势可推广到所有基于 SAM/SAM2 的微调场景
- 查询驱动记忆可启发其他视频基础模型的时序建模方式
- 深度引导融合的 Cat-Adapter 设计简洁但有效，可作为多模态融合的轻量方案

## 评分

- 新颖性: ⭐⭐⭐⭐ （并行适配器+查询驱动记忆组合新颖，解决了实际工程痛点）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个数据集、全面消融、参数敏感性分析、显存对比）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，问题定义明确）
- 价值: ⭐⭐⭐⭐ （显存效率提升具有重要实用价值，但任务范围较窄）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](../../CVPR2025/segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2026\] Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection](../../CVPR2026/segmentation/uncertainty-aware_modality_fusion_for_unaligned_rgb-t_salient_object_detection.md)

</div>

<!-- RELATED:END -->

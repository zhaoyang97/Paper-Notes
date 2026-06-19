---
title: >-
  [论文解读] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting
description: >-
  [ICCV 2025][模型压缩][向量量化] 提出 Sign-Splitting Vector Quantization (SSVQ)，将权重的符号位与码本解耦，引入可学习符号位和增强的迭代冻结策略，使 VQ 微调时每个量化权重可以沿各自梯度方向独立更新，在极端压缩率下显著优于传统 VQ 和标量量化。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "向量量化"
  - "符号分离"
  - "可学习符号位"
  - "迭代冻结"
  - "硬件加速"
---

# SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting

**会议**: ICCV 2025  
**arXiv**: [2503.08668](https://arxiv.org/abs/2503.08668)  
**代码**: [https://github.com/list0830/SSVQ](https://github.com/list0830/SSVQ)  
**领域**: 模型压缩 / 向量量化  
**关键词**: 向量量化, 符号分离, 可学习符号位, 迭代冻结, 硬件加速

## 一句话总结

提出 Sign-Splitting Vector Quantization (SSVQ)，将权重的符号位与码本解耦，引入可学习符号位和增强的迭代冻结策略，使 VQ 微调时每个量化权重可以沿各自梯度方向独立更新，在极端压缩率下显著优于传统 VQ 和标量量化。

## 研究背景与动机

向量量化 (VQ) 在极端压缩场景下展现出比均匀量化 (UQ) 更低的量化误差，是一种前景广阔的权重压缩技术。然而，VQ 在微调阶段存在根本性限制：同一码字对应的所有权重向量只能沿相同方向更新（因为只能微调码本），这导致"梯度主导"现象——少数高幅值梯度支配整个簇的更新方向，迫使大部分权重偏离其局部最优。

作者通过实验验证了该问题：在 MobileNet-V2 上，前 5% 梯度与码字梯度的余弦相似度高达 0.66，而后 60% 梯度仅为 0.26，证实大部分权重被少数梯度"绑架"。此外，即使在全参数可训练的情况下（BN+FC+Codebook），VQ 微调后精度仅为 55.27%，远低于理想水平。

## 方法详解

### 整体框架

SSVQ 的核心思想是将权重 $W$ 分解为符号矩阵 $S$ 和绝对值 $|W|$，仅对绝对值执行 k-means 聚类，然后引入可学习的符号参数 $L_s$ 使每个量化权重可以独立决定更新方向。量化权重重构为：

$$W_q = \mathcal{C}[A] \circ \text{sign}(L_s)$$

### 关键设计

1. **符号分离 (Sign-Splitting)**：将权重的符号位提取为 1-bit 掩码，对全正值权重执行聚类。这显著降低了所需码字数量（全正值聚类更简单），从而可以使用更高维度和更小码本来抵消 1-bit 额外存储开销。在相同压缩率下，SSVQ 与传统 VQ 具有相似的聚类误差。

2. **可学习符号位 (Learnable Sign)**：引入连续隐变量 $L_s$，初始化为 $L_s \leftarrow \alpha \cdot W$。前向传播使用 $\text{sign}(L_s)$ 作为符号位，反向传播通过 STE (Straight-Through Estimator) 近似梯度。符号位的梯度为 $g_{L_s} \approx \frac{\nabla L}{\nabla W_q} \circ c$，其中包含码字幅值 $c$，这在一定程度上有助于稳定训练。

3. **增强的迭代冻结策略 (Enhanced Iterative Freezing)**：由于符号翻转导致量化值变化 $2|c|$（幅度很大），可学习符号容易振荡。作者分析发现两个关键现象：(a) 频率 EMA 初期剧烈波动，过早冻结会捕获不稳定结果；(b) 基于 sign-EMA 的冻结结果本身也频繁振荡，简单的多数投票更稳定。因此提出：每隔 $F_i$ 次迭代检查振荡频率 EMA，超过余弦衰减阈值的符号按正负计数的多数投票结果永久冻结。

### 损失函数 / 训练策略

- 使用 AdamW 优化器 + 余弦退火学习率调度
- 所有分类和检测任务统一训练 10 个 epoch，不使用蒸馏
- 通过学习率或初始化参数 $\alpha$ 来调控符号变化率
- SSVQ 需要存储码本、赋值索引和符号掩码三部分

## 实验关键数据

### 主实验

| 模型 | 方法 | 压缩率 | 指标 |
|------|------|--------|------|
| DeiT-Tiny (ImageNet) | VQ | 21× | ~47% Top-1 |
| DeiT-Tiny (ImageNet) | SSVQ | 21× | ~59% Top-1 (+12%) |
| MobileNet-V2 | MVQ (16×) | 16× | 65.1% |
| MobileNet-V2 | SSVQ (16×) | 16× | 65.9% |
| EfficientNet-lite | MVQ (16×) | 16× | 68.2% |
| EfficientNet-lite | SSVQ (18×) | 18× | 69.5% |
| YOLOV9-S (COCO) | VQ (20×) | 20× | 30.0 AP |
| YOLOV9-S (COCO) | SSVQ (20×) | 20× | 35.2 AP (+5.2) |
| GELAN-C (COCO) | VQ (21×) | 21× | 43.5 bbox AP |
| GELAN-C (COCO) | SSVQ (21×) | 21× | 47.6 bbox AP (+4.1) |
| SD v1-4 (COCO) | VQ 2-bit | 2-bit | 24.5 FID-to-FP |
| SD v1-4 (COCO) | SSVQ 2-bit | 2-bit | 13.6 FID-to-FP |
| Llama3.2-1B (Wiki2) | VQ 2-bit | 2-bit | 63.1 PPL |
| Llama3.2-1B (Wiki2) | SSVQ 2-bit | 2-bit | 13.9 PPL |

### 消融实验

| 配置 (DeiT-T, k=16, d=8) | Top-1 Acc |
|--------------------------|-----------|
| 固定符号 (baseline) | ~47% (21×) |
| 可学习符号（无冻结） | ~54% |
| SSVQ（可学习符号+迭代冻结） | ~62% (+14.8%) |
| 冻结间隔 100, MSV | 60.9% |
| 冻结间隔 200, MSV | 61.47% |
| 冻结间隔 500, MSV | **61.90%** |
| 冻结间隔 1000, MSV | 61.55% |
| 冻结间隔 500, EMA | 60.8% |

### 关键发现

- 可学习符号位在所有压缩率下均带来 3%-9% 的一致精度提升，压缩率越高提升越明显
- 多数投票比 EMA 更适合作为冻结标准（61.90% vs 60.8%）
- 冻结间隔过小（100）会过早捕获不稳定状态，过大（1000）影响训练稳定性，500 为最优
- SSVQ 在 NLP 任务（Llama3.2-1B）上同样显著优于 VQ：2-bit 下 PPL 从 63.1 降至 13.9
- 硬件仿真表明 SSVQ 在 DeiT-Tiny 上实现 3× 推理加速（减少主存访问）

## 亮点与洞察

1. **洞察深刻**：对 VQ 微调限制的分析（梯度主导现象）非常到位，通过余弦相似度实验清晰揭示了问题本质
2. **设计优雅**：符号分离将更新方向的自由度从码本级恢复到逐权重级，且 1-bit 额外开销可通过全正值聚类的简化被抵消
3. **广泛验证**：覆盖分类、检测、分割、生成、NLP 五类任务，CNN/ViT/UNet/DiT/LLM 多种架构
4. **硬件落地**：构建了支持 SSVQ 的硬件仿真器，验证了 3× 加速，不是"纸上谈兵"

## 局限与展望

- 1-bit 符号掩码在极端低比特下的存储占比较高，可探索更高效的符号编码
- 迭代冻结的超参（冻结间隔、阈值衰减）仍需手动调节
- 仅验证了视觉和语言模型，多模态大模型（如 VLM）和音频模型未覆盖
- 硬件仿真仅在单一加速器上验证，实际 ASIC/FPGA 部署效果待确认

## 相关工作与启发

- SSVQ 可与 MVQ（带剪枝的码本）、DKM（可微 k-means）等方法结合
- 符号分离的思路可推广到其他压缩方法（如积量化 PQ），值得探索
- 迭代冻结策略与 Oscillation-free QAT (OsQAT) 类似，但针对 VQ 场景做了关键改进

## 评分

- **新颖性**: ⭐⭐⭐⭐ 符号分离+可学习符号位是一个简洁而有效的新范式
- **实验充分度**: ⭐⭐⭐⭐⭐ 五类任务、多种架构、消融充分、硬件验证
- **写作质量**: ⭐⭐⭐⭐ 动机分析清晰，图表丰富
- **价值**: ⭐⭐⭐⭐ 对边缘部署有实际意义，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[AAAI 2026\] SIGN: Schema-Induced Games for Naming](../../AAAI2026/model_compression/sign_schema-induced_games_for_naming.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[ACL 2025\] Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](../../ACL2025/model_compression/direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)
- [\[ICML 2026\] UniSVQ: 2-bit Unified Scalar-Vector Quantization](../../ICML2026/model_compression/unisvq_2-bit_unified_scalar-vector_quantization.md)

</div>

<!-- RELATED:END -->

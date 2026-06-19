---
title: >-
  [论文解读] SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity
description: >-
  [ICML2025][语义分割][Spiking Neural Network] 提出 SpikeVideoFormer，首个面向视频任务的脉冲驱动 Transformer，通过 Hamming 注意力替代点积注意力实现 spike 特征相似性的准确度量，结合联合时空注意力保持 $\mathcal{O}(T)$ 线性时间复杂度，在三个视频任务上达到 SNN SOTA，同时效率比 ANN 高 5-16 倍。
tags:
  - "ICML2025"
  - "语义分割"
  - "Spiking Neural Network"
  - "Transformer"
  - "注意力机制"
  - "线性时间复杂度"
  - "视频分类"
  - "人体姿态追踪"
  - "视频语义分割"
---

# SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity

**会议**: ICML2025  
**arXiv**: [2505.10352](https://arxiv.org/abs/2505.10352)  
**代码**: [JimmyZou/SpikeVideoFormer](https://github.com/JimmyZou/SpikeVideoFormer)  
**领域**: SNN视频 / 脉冲神经网络 / 视频Transformer  
**关键词**: Spiking Neural Network, Video Transformer, Hamming Attention, 线性时间复杂度, 视频分类, 人体姿态追踪, 视频语义分割

## 一句话总结

提出 SpikeVideoFormer，首个面向视频任务的脉冲驱动 Transformer，通过 Hamming 注意力替代点积注意力实现 spike 特征相似性的准确度量，结合联合时空注意力保持 $\mathcal{O}(T)$ 线性时间复杂度，在三个视频任务上达到 SNN SOTA，同时效率比 ANN 高 5-16 倍。

## 研究背景与动机

**现有 SNN 的局限**：当前 SNN Transformer（SpikFormer、Meta-SpikeFormer）主要聚焦单图像任务的空间特征建模，未充分利用 SNN 的神经元级时序编码能力处理视频任务。

**点积注意力的缺陷**：现有 spike-driven 注意力直接沿用 ANN 的点积操作作为注意力分数，但在二值 spike 特征上效果不佳。当 spike query 中存在 0 元素时，点积会忽略 key 中对应位置的信息，导致对差异很大的 spike key 向量计算出相同分数（特征混淆）。

**时空注意力设计未知**：ANN 中的时空注意力设计（joint/hierarchical/factorized）不能直接套用到 SNN，需要探索适合 spike-driven 场景的最优方案。

## 方法详解

### 整体架构

Conv+ViT 混合架构，输入视频 $T \times H \times W \times 3$：

1. **时序脉冲化**：输入经 LIF 神经元时序脉冲编码
2. **2个 Spike-driven CNN 块**：可分离卷积 + 通道卷积，下采样到 $T \times \frac{H}{8} \times \frac{W}{8} \times 4C$
3. **2个 Spike-driven 时空 Transformer**：SDHA 联合注意力 + Channel MLP，输出 $T \times \frac{H}{16} \times \frac{W}{16} \times 10C$
4. **任务头**：分类/回归/分割头

### 核心：Spike-Driven Hamming Attention (SDHA)

**理论基础** — JL 引理的二值嵌入扩展（Proposition 3.1）：

归一化 Hamming 相似度 $f_{\mathcal{H}}$ 与余弦相似度 $f_{\mathcal{C}}$ 之间存在近似关系：

$$P\big(|f_{\mathcal{H}}(q_s, k_s) - g(f_{\mathcal{C}}(q, k))| \leq \delta\big) \geq 1 - 2e^{-\delta^2 D}$$

其中 $g(x) = 1 - \frac{1}{\pi}\arccos(x)$ 是单调连续函数。当通道维度 $D$ 足够大时，Hamming 相似度可高概率逼近传统注意力的余弦相似度（保持排序不变）。

**Hamming 相似度的高效实现**：

$$f_{\mathcal{H}}(q_s, k_s) = \frac{1}{2} + \frac{1}{2D}(2q_s - \mathbf{1})^\top(2k_s - \mathbf{1})$$

- $(2q_s - 1)$ 将 $\{0,1\}$ 映射为 $\{-1,1\}$，通过位移完成，无需乘法
- 缩放因子 $\frac{1}{2D}$ 合并到 LIF 神经元阈值中，推理时无额外乘法
- 重排计算顺序后保持 $\mathcal{O}(ND^2)$ 线性复杂度

**最终 SDHA 公式**：

$$\text{SDHA} = \mathcal{SN}_{2D}\Big((2Q_s - \mathbf{1})\big[(2K_s - \mathbf{1})^\top V_s\big]\Big)$$

### 时空注意力设计

对比三种方案（输入 $B \times T \times N \times D$）：

| 设计 | ANN 复杂度 | SNN 复杂度 | 参数量 |
|------|-----------|-----------|--------|
| Joint（联合） | $\mathcal{O}(T^2N^2D)$ | $\mathcal{O}(TND^2)$ | $4D^2$ |
| Hierarchical（分层） | $\mathcal{O}(TN(T+N)D)$ | $\mathcal{O}(TND^2)$ | $8D^2$ |
| Factorized（分解） | $\mathcal{O}(TN(T+N)D)$ | $\mathcal{O}(TND^2)$ | $7D^2$ |

关键发现：**Joint 联合注意力在 SNN 中既最优又最高效**（参数最少、性能最好），这与 ANN 中为降低二次复杂度而需要分解注意力的策略相反。SNN 的线性注意力天然消除了联合注意力的复杂度瓶颈。

## 实验关键数据

### 视频分类（Kinetics-400）

| 方法 | Spike | 参数(M) | 功耗(mJ) | Top-1(%) | Top-5(%) |
|------|-------|---------|----------|----------|----------|
| ViViT (ANN) | ✗ | 310.8 | 6651.6 | 80.6 | 94.7 |
| Swin-B (ANN) | ✗ | 88.1 | 1297.2 | 80.6 | 94.6 |
| Meta-SpikeFormer | ✓ | 55.9 | 396.4 | 75.5 | 90.1 |
| **SpikeVideoFormer** | ✓ | 55.9 | 412.1 | **79.8** | **94.0** |

比 Meta-SpikeFormer 高 **+4.3%** Top-1；接近 Swin-B 仅差 0.8%，功耗低 **3×**。

### 人体姿态追踪（MMHPSD, T=32, Video）

| 方法 | 参数(M) | 功耗(mJ) | PA-MPJPE↓ |
|------|---------|----------|-----------|
| GLoT (ANN) | 40.5 | 4046.1 | 46.5 |
| Meta-SpikeFormer* | 55.8 | 387.2 | 54.5 |
| **SpikeVideoFormer** | 55.8 | 391.2 | **47.5** |

比 SNN SOTA 降低 **7.0mm**，逼近 ANN 最优仅差 1.0mm，功耗低 **10×**。

### 视频语义分割（CityScapes, Integer-LIF=4）

| 方法 | 参数(M) | 功耗(mJ) | mIoU(%) |
|------|---------|----------|---------|
| SegFormer (ANN) | 13.8 | 270.2 | 74.1 |
| Meta-SpikeFormer | 17.8 | 63.5 | 65.9 |
| **SpikeVideoFormer** | 17.8 | 65.3 | **73.1** |

比 SNN SOTA 高 **+7.2%** mIoU，逼近 SegFormer 仅差 1%，功耗低 **4×**。

### 消融实验关键发现

| 变体 | Pose PA-MPJPE | VSS mIoU |
|------|--------------|----------|
| 完整模型 | 39.8 | 73.1 |
| Hamming → Dot-product | 45.7 (+5.9) | 65.9 (-7.2) |
| Joint → Spatial-Only | 54.2 (+14.4) | 62.1 (-11.0) |
| Pre-train → Random | 53.8 (+14.0) | 61.3 (-11.8) |

Hamming 注意力贡献最显著之一，去掉时序建模性能断崖式下降。

## 亮点与洞察

1. **理论驱动设计**：基于 JL 引理严格推导 Hamming 相似度替代点积的合理性，而非凭经验拼凑
2. **反直觉发现**：Joint 注意力在 SNN 中是最优选择（ANN 中反而需分解降低复杂度），因为 SNN 线性注意力天然解决了复杂度问题
3. **跨任务泛化**：同一模型在分类、回归、密集预测三类不同任务上均 SOTA，证明通用性
4. **效率优势随序列增长更明显**：T=8→32 时功耗仅增 4.1×，而 ANN 增 8.3×，体现 $\mathcal{O}(T)$ 复杂度优势
5. **阈值缩放的理论指导**：$s = 1/2D$ 是理论推导得到的最优值，优于以往经验设定的固定值

## 局限与展望

1. **VSS 性能仍有差距**：在 VSPW 大规模数据集上与 CFFM 差距较大（37.9 vs 49.3 mIoU），主要因采用简单分割头
2. **GPU 上推理速度优势不明显**：SNN 的加法优势体现在神经形态芯片上，当前 GPU 测试中推理时间改善有限
3. **预训练依赖**：去掉 ImageNet 预训练后性能大幅下降，说明模型从头训练能力有待提升
4. **仅验证视觉任务**：未扩展到视频理解（Q&A）、视频生成等更复杂任务
5. **Integer-LIF 的额外开销**：使用多位脉冲（{0,1,2,3}）提升性能但偏离严格二值约束

## 相关工作与启发

- **SpikFormer / Meta-SpikeFormer**：SNN Transformer 先驱，用点积注意力，本文证明其在 spike 特征上理论缺陷
- **ViViT / Video Swin**：ANN 视频 Transformer 的联合/分解注意力设计，启发了本文的 SNN 时空注意力探索
- **JL 引理 (Jacques et al., 2013)**：二值嵌入距离保持的理论工具，为 Hamming 注意力提供数学基础

## 评分

- 新颖性: ⭐⭐⭐⭐ — Hamming 注意力有严谨理论支撑，首次系统性探索 SNN 视频 Transformer
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个不同视频任务 + 两种输入模态 + 详尽消融
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，图表丰富
- 价值: ⭐⭐⭐⭐ — 为 SNN 在视频领域的应用树立新基线，效率优势明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation](../../CVPR2025/segmentation/samwise_infusing_wisdom_in_sam2_for_text-driven_video_segmentation.md)
- [\[ICCV 2025\] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation](../../ICCV2025/segmentation/inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](../../CVPR2026/segmentation/mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[CVPR 2025\] Rethinking Query-Based Transformer for Continual Image Segmentation](../../CVPR2025/segmentation/rethinking_query-based_transformer_for_continual_image_segmentation.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](../../CVPR2025/segmentation/mambavision_a_hybrid_mamba-transformer_vision_backbone.md)

</div>

<!-- RELATED:END -->

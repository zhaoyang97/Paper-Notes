---
title: >-
  [论文解读] Towards Practical Real-Time Neural Video Compression
description: >-
  [CVPR 2025][模型压缩][神经视频编解码] 提出DCVC-RT，首个在消费级硬件上实现1080p实时编解码且压缩率超越H.266/VTM的神经视频编解码器，核心发现是操作复杂度（而非计算复杂度）才是速度瓶颈，据此设计隐式时序建模和单尺度低分辨率潜表示，在A100上达到125/113 fps编解码速度，同时节省21%码率。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "神经视频编解码"
  - "实时编码"
  - "操作复杂度"
  - "隐式时序建模"
  - "模型整型化"
---

# Towards Practical Real-Time Neural Video Compression

**会议**: CVPR 2025  
**arXiv**: [2502.20762](https://arxiv.org/abs/2502.20762)  
**代码**: [https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)  
**领域**: 模型压缩/视频压缩  
**关键词**: 神经视频编解码, 实时编码, 操作复杂度, 隐式时序建模, 模型整型化

## 一句话总结

提出DCVC-RT，首个在消费级硬件上实现1080p实时编解码且压缩率超越H.266/VTM的神经视频编解码器，核心发现是操作复杂度（而非计算复杂度）才是速度瓶颈，据此设计隐式时序建模和单尺度低分辨率潜表示，在A100上达到125/113 fps编解码速度，同时节省21%码率。

## 研究背景与动机

神经视频编解码器（NVC）在压缩率上已超越传统编解码器（H.265/HM, H.266/VTM），但实时编码仍是实际部署的最大障碍：

1. **现有加速不够**：MobileNVC能实时解码但压缩率不如x264；C3高效解码但编码需耗时优化；DHVC-2.0需要4 GPU流水线做实时解码，单GPU上不可行
2. **传统认知误导**：大家专注于减少计算复杂度（MACs），但作者关键发现——减少通道数带来的速度提升是线性而非预期的二次方，说明计算复杂度不是瓶颈
3. **操作复杂度被忽视**：内存I/O开销（受潜表示大小 $P_{size}$ 影响）和函数调用开销（受模块数量 $P_{num}$ 影响）才是真正的速度瓶颈

这一发现开辟了全新的加速思路：保持计算能力不变，专注于降低操作复杂度。

## 方法详解

### 整体框架

DCVC-RT采用条件编码范式：当前帧通过patch embedding直接变换到1/8分辨率潜空间，与前帧的时序上下文拼接后由编码器-解码器联合处理。去除了所有显式运动估计/补偿模块，大幅简化流程。

### 关键设计

**设计一：单一低分辨率潜表示学习**

- **功能**：消除渐进下采样带来的大潜表示尺寸的高内存I/O开销
- **核心思路**：使用patch embedding将输入帧直接变换到1/8分辨率的单一尺度，所有关键模块（编码器、解码器、特征提取器、重建网络）都在这一尺度上操作
- **设计动机**：传统NVC逐层半分辨率下采样、双通道数的设计中，高分辨率层的潜表示尺寸 $P_{size}$ 很大。在1/8单尺度下，$C=256$ 对应的潜表示大小为 $4 \cdot H \cdot W$，足以保持表达能力，且感受野比渐进下采样更大（有利于时序建模）。编码速度较渐进下采样快3.6倍，BD-Rate仅退化0.3%

**设计二：隐式时序建模**

- **功能**：消除复杂运动估计/补偿模块带来的高模块数量 $P_{num}$
- **核心思路**：用单一简单的特征提取器从前帧重建潜表示中提取时序上下文，与当前帧潜表示沿通道维度拼接，由编码器-解码器联合处理时序冗余。完全移除运动编码分支
- **设计动机**：运动编码分支虽然计算量低（仅为条件编码的1/13），但模块层数高达123层（占条件编码225层的一半以上），函数调用次数多是速度瓶颈。隐式方法将计算能力重新分配给帧编码模块，编码速度提升3.4倍

在不同运动内容上的对比显示：小运动场景BD-Rate反而改善0.4%，大运动场景退化3.2%，场景切换改善4.7%。

**设计三：模块库率控 + 模型整型化**

- **功能**：支持灵活的码率控制和跨设备一致性
- **核心思路**：(a) 率控：引入模块库（module bank），为不同量化参数（qp）学习不同的超先验模块，精确估计hyper信息 $z$ 的分布，节省约3%码率。(b) 整型化：将浮点模型转为int16确定性计算（$v_i = \text{round}(512 \cdot v_f)$），用预计算查找表处理非线性Sigmoid函数
- **设计动机**：DCVC-RT中超信息 $z$ 占比超过10%（因无运动码流），单一因子化先验不够精确。16-bit整型化确保不同平台的编解码输出完全一致

### 损失函数

采用YUV和RGB双色彩空间联合失真损失，配合层级质量（hierarchical quality）的 $\lambda$ 插值设置。qp在0-63间随机采样实现单模型变码率。

## 实验关键数据

### 主实验：BD-Rate对比（YUV420，锚: VTM-17.0）

| 方法 | 平均BD-Rate | 编码fps | 解码fps |
|------|------------|--------|--------|
| VTM-17.0 (H.266) | 0.0% | 0.01 | 23.6 |
| HM-16.25 (H.265) | +42.4% | 0.05 | 39.6 |
| ECM-11.0 | -20.5% | 0.002 | 3.4 |
| DCVC-FM | -22.1% | 3.4 | 4.2 |
| **DCVC-RT (fp16)** | **-21.0%** | **125.2** | **112.8** |

### 速度对比（1080p, 不同设备）

| 设备 | DCVC-FM编码/解码 | DCVC-RT编码/解码 |
|------|----------------|----------------|
| A100 | 5.0 / 5.9 fps | 125.2 / 112.8 fps |
| RTX 4090 | 4.2 / 4.1 fps | 98.2 / 96.5 fps |
| RTX 2080Ti | 2.3 / 2.4 fps | 40.3 / 34.3 fps |

### 消融实验

| 组件 | BD-Rate | 编码时间(ms) |
|------|---------|------------|
| 渐进下采样 + 显式运动 (DCVC-FM) | -22.1% | ~200 |
| 1/8单尺度 + 显式运动 | -21.8% | ~55 |
| 1/8单尺度 + 隐式时序 | -21.0% | ~8 |

### 关键发现

1. DCVC-RT比DCVC-FM快18倍以上，BD-Rate仅退化1.1%，实现了极佳的率失真-复杂度权衡
2. 在消费级GPU（RTX 2080Ti）上首次实现40 fps编码/34 fps解码的1080p实时编码
3. 操作复杂度分析揭示：减少一半计算量（通道数减半）仅带来~1.5倍加速，而非理论上的4倍，证明操作复杂度是瓶颈
4. MACs从2642G降至385G（减少85%），是速度提升的计算基础

## 亮点与洞察

1. **操作复杂度 vs 计算复杂度的深刻洞察**：这一发现改变了NVC加速的思路，不再一味减少通道数/层数，而是减少模块数量和潜表示尺寸
2. **隐式时序建模的实用价值**：虽然在大运动场景略有退化，但3.4倍速度提升在实际应用中远比3.2%的BD-Rate退化重要
3. **端到端实用性**：同时解决了实时编码、率控、跨设备一致性三大实际部署难题

## 局限与展望

1. 隐式时序建模在大运动场景有3.2%的BD-Rate退化，可能不适合运动剧烈的视频
2. 模型整型化（int16→int32累加器）导致2.7%的BD-Rate退化，精度损失可以进一步优化
3. 单帧内编码（intra-period=-1）设置下的评估，实际应用中需要考虑随机访问等功能
4. 可以探索将隐式时序建模与轻量级运动hints结合，在速度和大运动性能间取得更好平衡

## 相关工作与启发

- **DCVC系列（DC→FM→RT）**：微软的NVC演进路线，从率失真性能转向实际部署
- **MobileNVC**：首个消费级实时NVC，但压缩率不足，DCVC-RT解决了这一矛盾
- **Neural Image Codecs**：图像编码的实时化经验（如ELIC）为视频编码提供了参考
- 启发：操作复杂度的分析框架可以推广到其他深度学习推理加速场景，不仅限于视频编码

## 评分

⭐⭐⭐⭐⭐ — 里程碑式工作。首次在消费级硬件上实现实时NVC且压缩率超越H.266，解决了NVC实际部署的核心阻碍。操作复杂度的洞察具有广泛影响力。来自微软亚洲研究院的高质量工程+研究结合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Real-Time Neural Video Compression with Unified Intra and Inter Coding](../../CVPR2026/model_compression/real-time_neural_video_compression_with_unified_intra_and_inter_coding.md)
- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2026\] Ultra-Fast Neural Video Compression](../../CVPR2026/model_compression/ultra-fast_neural_video_compression.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)

</div>

<!-- RELATED:END -->

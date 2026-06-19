---
title: >-
  [论文解读] Outlier-Aware Post-Training Quantization for Image Super-Resolution
description: >-
  [ICCV 2025][图像恢复][后训练量化] 提出一种面向图像超分辨率的离群值感知后训练量化方法，通过双区域分段线性量化平衡离群值保留与正常激活精度，并引入敏感度感知微调策略使模型关注量化敏感层，在 W4A4 设置下大幅超越现有 PTQ 方法并接近 QAT 性能。 图像 SR 模型的量化面临独特挑战…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "后训练量化"
  - "图像超分辨率"
  - "激活离群值"
  - "分段线性量化"
  - "敏感度感知"
---

# Outlier-Aware Post-Training Quantization for Image Super-Resolution

**会议**: ICCV 2025  
**arXiv**: [2511.00682](https://arxiv.org/abs/2511.00682)  
**代码**: 无  
**领域**: 图像超分辨率 / 模型量化  
**关键词**: 后训练量化, 图像超分辨率, 激活离群值, 分段线性量化, 敏感度感知

## 一句话总结

提出一种面向图像超分辨率的离群值感知后训练量化方法，通过双区域分段线性量化平衡离群值保留与正常激活精度，并引入敏感度感知微调策略使模型关注量化敏感层，在 W4A4 设置下大幅超越现有 PTQ 方法并接近 QAT 性能。

## 研究背景与动机

图像 SR 模型的量化面临独特挑战。现有 PTQ 方法在 SR 上效果不佳，核心原因是忽略了激活中的离群值问题。作者的经验分析揭示了两个关键观察：

**观察1：离群值与颜色信息强相关**。SR 网络激活分布中普遍存在离群值，大部分激活集中在浅范围（如 [-50, 50]），但会出现远超此范围的极端值。裁剪仅 1% 的离群值就会导致明显的颜色失真（如花朵褪色），说明离群值编码了关键的颜色信息。

**观察2：不同层对量化的敏感度差异悬殊**。在 SRResNet 中，head.0 层量化后 PSNR 从 32.06 dB 暴跌至 18.26 dB，而 body.4.conv1 层量化后仍保持 31.20 dB。这种异质性要求量化策略应差异化对待各层。

**核心矛盾**：保留离群值会占用大量 bit 位宽，压缩正常激活的表示空间；裁剪离群值又会导致严重的性能退化。

## 方法详解

### 整体框架

两阶段流程：(1) 校准阶段——使用 100 张 DIV2K 低分辨率图像统计量化参数；(2) 微调阶段——通过敏感度感知损失优化量化参数，无需标注数据。

### 关键设计

1. **分段线性量化器 (Piecewise Linear Quantizer, PLQ)**:

    - 核心思路：将激活分布划分为两个不重叠区域，分别进行均匀量化
    - 引入可学习断点 $bp$，将激活范围 $[l_a, u_a]$ 分为：
        - 稠密区域 $R_1 = [-bp, bp]$：包含大部分正常激活值
        - 离群区域 $R_2 = [l_a, -bp) \cup (bp, u_a]$：包含极端值
    - 对 $b$ bit 量化，稠密区域分配 $2^{b-1}-1$ 个量化点，离群区域的正负各分配 $2^{b-2}-1$ 个量化点
    - 初始化：$l_a$ = 最小激活值，$u_a$ = 最大激活值，$bp$ = 第 99 百分位值
    - 后续 batch 使用指数移动平均 (EMA, $\beta=0.9$) 更新参数

2. **敏感度感知微调 (Sensitivity-Aware Finetuning, SAFT)**:

    - 量化敏感度计算：$s_k = \frac{\exp(\frac{1}{N}\sum_{x \in D_{cal}} \sigma(x_k))}{\sum_{j=1}^K \exp(\frac{1}{N}\sum_{x \in D_{cal}} \sigma(x_j))}$
    - 其中 $\sigma(x_k)$ 为第 $k$ 层特征图的标准差，通过 softmax 归一化
    - 高方差层被赋予更高敏感度权重，量化时获得更多关注

3. **分阶段参数优化**:

    - epoch mod 3 = 1：更新权重上界 $u_w$
    - epoch mod 3 = 2：更新激活上下界 $l_a, u_a$
    - epoch mod 3 = 0：更新断点 $bp$
    - 循环迭代逐步精化量化参数

### 损失函数 / 训练策略

总损失 $L_{all} = \mathcal{L}_{sen} + \lambda \mathcal{L}_{rec}$（$\lambda = 5$）：
- **重建损失**：$\mathcal{L}_{rec} = \frac{1}{N}\sum_{i=1}^N \|\mathcal{K}(I_{lr}^i) - \mathcal{Q}(I_{lr}^i)\|_1$，全精度与量化网络输出的 L1 距离
- **敏感度感知损失**：$\mathcal{L}_{sen} = \frac{s_k}{K}\sum_{k=1}^K \|\frac{F_\mathcal{K}^k}{\|F_\mathcal{K}^k\|_2} - \frac{F_\mathcal{Q}^k}{\|F_\mathcal{Q}^k\|_2}\|_2$

关键优势：**仅需低分辨率图像即可训练**，无需高分辨率 ground truth。

## 实验关键数据

### 主实验

EDSR 和 RDN 在 W4A4 设置下的 PSNR 比较：

| 方法 | 微调 | Set5 | Set14 | BSD100 | Urban100 |
|------|------|------|-------|--------|----------|
| EDSR FP32 | - | 32.10 | 28.58 | 27.56 | 26.04 |
| EDSR-MinMax | ✗ | 26.83 | 25.04 | 24.57 | 23.12 |
| EDSR-PTQ4SR | ✓ | 30.51 | 27.62 | 26.88 | 24.92 |
| EDSR-AdaBM | ✓ | 31.02 | 27.87 | 26.91 | 25.11 |
| **EDSR-Ours** | **✓** | **31.54** | **28.26** | **27.36** | **25.61** |
| RDN FP32 | - | 32.24 | 28.67 | 27.63 | 26.29 |
| RDN-PTQ4SR | ✓ | 28.32 | 26.11 | 25.82 | 23.31 |
| RDN-AdaBM | ✓ | 28.71 | 26.30 | 26.10 | 23.38 |
| **RDN-Ours** | **✓** | **31.80** | **28.39** | **27.47** | **25.93** |

RDN W4A4 Urban100 上比次优方法提升 **2.55 dB**。

与 QAT 方法对比（EDSR W4A4）：

| 方法 | 需GT | 处理时间 | Set5 | Set14 | BSD100 | Urban100 |
|------|------|---------|------|-------|--------|----------|
| PAMS (QAT) | ✓ | 75× | 31.59 | 28.20 | 27.32 | 25.32 |
| ODM (QAT) | ✓ | 120× | 32.00 | 28.47 | 27.51 | 25.80 |
| **Ours (PTQ)** | **✗** | **1×** | **31.79** | **28.40** | **27.45** | **25.75** |

达到 QAT 可比性能，**速度提升 75 倍以上**。

### 消融实验

EDSR W4A4 组件消融：

| PLQ | SAFT | VFT | Set5 PSNR | Set14 PSNR | BSD100 PSNR | Urban100 PSNR |
|-----|------|-----|-----------|------------|-------------|---------------|
| ✗ | ✗ | ✗ | 26.83 | 25.04 | 24.57 | 23.12 |
| ✓ | ✗ | ✗ | 30.50 | 27.71 | 27.03 | 25.12 |
| ✗ | - | ✓ | 29.45 | 26.95 | 26.27 | 24.40 |
| ✗ | ✓ | - | 29.87 | 27.24 | 26.55 | 24.57 |
| ✓ | ✓ | - | **31.54** | **28.26** | **27.36** | **25.61** |

PLQ 单独贡献最大（Set5 +3.67 dB），SAFT 优于普通微调（VFT）。

### 关键发现

- **离群值保护至关重要**：MinMax（保留离群值但挤压正常值）和 Percentile（裁剪离群值）都不理想，双区域策略是最优选择
- **PLQ 是主要贡献模块**：单独使用 PLQ 即可在 Set5 上从 26.83 提升至 30.50 dB
- **SAFT 优于普通微调**：敏感度加权比均等对待所有层更有效（Set5 差距 0.42 dB）
- **效率极高**：整个 PTQ 过程仅需 73 秒，推理延迟与基线持平

## 亮点与洞察

- **离群值与颜色信息关联**的发现很有洞察力，为量化中的离群值处理提供了新视角
- **分段量化思想**简单有效，把一个困难的单区间量化问题分解为两个简单的子问题
- **无需 GT 数据**：PTQ 仅需低分辨率校准图像，极大降低了使用门槛
- **75× 加速**对比 QAT 方法极具实用价值

## 局限与展望

- 断点 $bp$ 的位置选择（99 百分位）可能不是最优，可尝试自适应确定
- 仅在 EDSR、RDN、SRResNet 等较为经典的 SR 网络上验证，缺少对 Transformer-based SR（如 SwinIR）的实验
- 双区域划分是固定的，可以探索更细粒度的多区域策略
- 实际硬件部署时分段量化的兼容性需要进一步验证

## 相关工作与启发

- 双区域量化思路可借鉴到其他含离群值问题的量化任务中（如 LLM 量化的 kv-cache）
- 敏感度感知策略可推广到混合精度量化的 bit 分配决策中
- 离群值-颜色关联的发现可能对其他图像生成/恢复任务的量化也有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 离群值-颜色关联的发现和双区域策略有新意
- **实验充分度**: ⭐⭐⭐⭐ 多模型多数据集验证，PTQ/QAT 对比全面，含消融和视觉分析
- **写作质量**: ⭐⭐⭐⭐ 观察→动机→方法的叙事流畅，图表清晰
- **实用价值**: ⭐⭐⭐⭐⭐ 高效 PTQ 方案对 SR 模型部署极具价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](../../ECCV2024/image_restoration/rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)

</div>

<!-- RELATED:END -->

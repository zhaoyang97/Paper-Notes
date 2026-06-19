---
title: >-
  [论文解读] OuroMamba: A Data-Free Quantization Framework for Vision Mamba
description: >-
  [ICCV 2025][模型压缩][量化] 首个面向 Vision Mamba 模型（VMM）的无数据后训练量化框架，通过增强隐式注意力生成高质量合成数据，并结合动态异常值检测的混合精度量化方案，在 W4A4 设置下显著超越现有数据驱动 PTQ 方法。 Vision Mamba 模型（VMM）因其亚二次计算复杂度成为 ViT…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "量化"
  - "Vision Mamba"
  - "SSM"
  - "Mixed-Precision"
---

# OuroMamba: A Data-Free Quantization Framework for Vision Mamba

**会议**: ICCV 2025  
**arXiv**: [2503.10959](https://arxiv.org/abs/2503.10959)  
**代码**: [GitHub](https://github.com/georgia-tech-synergy-lab/ICCV-OuroMamba)  
**领域**: 模型压缩 / 量化  
**关键词**: Data-Free Quantization, Vision Mamba, SSM, Mixed-Precision, Post-Training Quantization

## 一句话总结

首个面向 Vision Mamba 模型（VMM）的无数据后训练量化框架，通过增强隐式注意力生成高质量合成数据，并结合动态异常值检测的混合精度量化方案，在 W4A4 设置下显著超越现有数据驱动 PTQ 方法。

## 研究背景与动机

Vision Mamba 模型（VMM）因其亚二次计算复杂度成为 ViT 的有力替代，但大模型部署面临内存和延迟瓶颈。后训练量化（PTQ）是有效的压缩手段，但通常需要原始训练数据进行校准，在隐私敏感场景下受限。无数据量化（DFQ）通过从高斯噪声生成合成数据来替代校准数据，但现有 DFQ 方法均针对 ViT 设计，直接迁移到 VMM 面临两大挑战：

**合成数据质量差**：VMM 的循环状态转换缺乏显式自注意力机制，其隐式注意力难以区分前景和背景，导致基于 ViT 注意力的合成数据生成方法失效。不同扫描方向的隐式注意力存在方向偏差且不一致。

**激活动态异常值**：与 ViT 中静态异常值模式不同，VMM 的 S6 层激活（如 $\bar{A}$、$\bar{B}$）在不同时间步表现出动态变化的异常值通道位置，使得静态 PTQ 技术失效。

## 方法详解

### 整体框架

OuroMamba 包含两个阶段：OuroMamba-Gen（合成数据生成）和 OuroMamba-Quant（混合精度量化）。

### 关键设计

1. **Patched Hidden State $h_p(t)$**：针对 VMM 隐式注意力无法有效捕获长距离交互的问题，对每个时间步的隐藏状态 $h(t)$ 定义一个 $p \times p$ 的空间邻域 $\mathcal{N}(t)$，通过加权求和聚合邻域状态：$h_p(\tau) = \sum_{k \in \mathcal{N}(\tau)} w_k h(k)$。权重因子利用离散化张量 $\Delta(t)$ 的通道均值，因为 $\Delta(t)$ 在信息丰富区域（如前景）有更大的响应。由此得到的增强隐式注意力 $\alpha_p$ 能有效分离前景与背景。

2. **数据生成损失 $\mathcal{L}^{gen}$**：在增强隐式注意力 $\alpha_p$ 上应用 patch 级对比学习损失 $\mathcal{L}^C$（基于 infoNCE），利用余弦相似度识别正负 patch。结合任务特定的输出损失 $\mathcal{L}^O$（MAE），最终损失为 $L^{gen} = \mathcal{L}^C + \mathcal{L}^O$。

3. **动态异常值检测与混合精度量化**：

    - 离线校准阶段确定 per-time-step 内联缩放因子 $S^I(t)$ 和阈值 $\theta$
    - 推理时每个时间步动态计算 $S^D(t)$，若超过 $S^I(t)$ 则逐通道检测异常值通道并加入 $O_{\text{list}}$
    - 异常值通道以 $b_a^O$-bit（8-bit）per-channel 量化，内联通道以 $b_a^I$-bit（4-bit）group 量化
    - 每 $n_{\text{refresh}}=10$ 步刷新一次 $O_{\text{list}}$ 防止累积过时异常值

### 损失函数 / 训练策略

- 数据生成：$L^{gen} = \mathcal{L}^C + \mathcal{L}^O$，迭代 1000 次更新高斯噪声为合成数据
- 权重量化：per-channel 对称 group 量化（4-bit）
- 激活量化：每时间步动态混合精度（内联4-bit/异常值8-bit）

## 实验关键数据

### 主实验（ImageNet 分类）

| 模型 | 方法 | 数据 | W/A | Top-1 |
|------|------|------|-----|-------|
| Vim-S | FP Baseline | - | 32/32 | 81.60 |
| Vim-S | PTQ4VM | Real 256 | 4/8 | 74.37 |
| Vim-S | QMamba | Real 1024 | 4/8 | 74.12 |
| **Vim-S** | **OuroMamba** | **Syn 128** | **4/8** | **79.81** |
| Vim-S | PTQ4VM | Real 256 | 4/4 | 69.60 |
| Vim-S | QMamba | Real 1024 | 4/4 | 33.64 |
| **Vim-S** | **OuroMamba** | **Syn 128** | **4/4** | **75.93** |
| Vim-B | FP Baseline | - | 32/32 | 81.90 |
| **Vim-B** | **OuroMamba** | **Syn 128** | **4/4** | **77.34** |
| VMamba-B | FP Baseline | - | 32/32 | 83.90 |
| **VMamba-B** | **OuroMamba** | **Syn 128** | **4/4** | **78.91** |
| MambaVision-B | FP Baseline | - | 32/32 | 84.20 |
| **MambaVision-B** | **OuroMamba** | **Syn 128** | **4/4** | **79.24** |

### 消融实验

| $\mathcal{L}^{PSE}$ | $\mathcal{L}^C$ | $\mathcal{L}^O$ | W/A | Top-1 (%) |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | 4/8 | 71.68 |
| ✗ | ✗ | ✓ | 4/8 | 21.65 |
| ✓ | ✗ | ✓ | 4/8 | 73.45 |
| ✗ | ✓ | ✗ | 4/8 | 75.52 |
| ✗ | ✓ | ✓ | 4/8 | **79.81** |

**$n_{\text{refresh}}$ 消融**：刷新周期为 10 时延迟加速最优；过小导致频繁重置，过大导致异常值累积反而变慢，不刷新时甚至比 FP16 更慢。

### 关键发现

- OuroMamba 在 W4A4 下平均比 PTQ4VM 提升 **7.84%**，比 QMamba 提升 **19.40%**
- 仅用 128 张合成图像（无需真实数据）即超越使用 256-1024 张真实数据的方法
- 目标检测任务中 box AP 最多提升 **21.1**，分割任务 mIoU 最多提升 **6.6**
- 扩散模型 Zigma 的 W4A4 量化中 FID 仅从 37.8 → 39.2（FacesHQ），远优于 PTQ4VM 的 89.6
- 自定义 CUDA 核实现最高 **2.36×** 端到端加速（Vim-B）

## 亮点与洞察

- **首次揭示** VMM 激活中动态异常值问题，这与 ViT 的静态异常值模式形成鲜明对比
- 利用 $\Delta(t)$ 加权的 patched state 增强隐式注意力是一个巧妙的设计——利用 Mamba 自身的信息选择机制改善其固有缺陷
- 无数据场景下反而优于数据驱动方法，说明针对模型特性设计的合成数据比通用真实数据更有效
- 混合精度 GEMM kernel 的 INT4+INT8 融合流水线实现了真正的落地加速

## 局限与展望

- 动态异常值检测假设内联分布在校准后保持稳定，未来模型若内联值波动大需额外研究
- 目前仅实现了特定精度组合（W4A4O8）的 kernel，更多精度组合的 kernel 支持有待完善
- 未探讨与知识蒸馏等其他压缩技术的结合
- 邻域大小 $p$ 和刷新周期 $n_{\text{refresh}}$ 为超参数，不同架构可能需要调整

## 相关工作与启发

- **CLAMP-ViT**：本文的 patch 级对比学习数据生成范式之源，但仅适用于有显式自注意力的 ViT
- **QMamba**：同时期工作识别了 VMM 的动态激活变化，但采用静态分组量化未能解决
- **PTQ4VM**：将激活异常值迁移到权重的 SmoothQuant 策略，但未量化 SSM 激活

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个 VMM DFQ 框架，两阶段设计均有深入的问题分析和原创方案
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖分类/检测/分割/生成四大任务，多模型多精度，含 kernel 加速评估
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，分析透彻，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 对 Mamba 模型部署具有重要实用价值，合成数据+动态量化思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](../../CVPR2025/model_compression/efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[ICML 2026\] Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers](../../ICML2026/model_compression/selective_coupling_of_decoupled_informative_regions_masked_attention_alignment_f.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](msq_memory-efficient_bit_sparsification_quantization.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](../../CVPR2025/model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)

</div>

<!-- RELATED:END -->

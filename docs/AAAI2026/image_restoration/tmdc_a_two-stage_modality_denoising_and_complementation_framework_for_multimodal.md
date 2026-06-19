---
title: >-
  [论文解读] TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis
description: >-
  [AAAI 2026][图像恢复][多模态] 提出 TMDC 两阶段框架，第一阶段在完整数据上学习去噪的 modality-specific 和 modality-common 表示，第二阶段利用可用模态的去噪表示补全缺失模态，首次同时处理 MSA 中的噪声和缺失问题。 现有痛点 现有痛点：多模态情感分析（MSA）融合文本、…
tags:
  - "AAAI 2026"
  - "图像恢复"
  - "多模态"
  - "missing modality"
  - "noisy modality"
  - "variational information bottleneck"
  - "去噪"
---

# TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis

**会议**: AAAI 2026  
**arXiv**: [2511.10325](https://arxiv.org/abs/2511.10325)  
**代码**: 未公开  
**领域**: Multimodal Sentiment Analysis  
**关键词**: multimodal sentiment analysis, missing modality, noisy modality, variational information bottleneck, denoising

## 一句话总结

提出 TMDC 两阶段框架，第一阶段在完整数据上学习去噪的 modality-specific 和 modality-common 表示，第二阶段利用可用模态的去噪表示补全缺失模态，首次同时处理 MSA 中的噪声和缺失问题。

## 背景与动机

### 现有痛点

**现有痛点**：多模态情感分析（MSA）融合文本、音频、视频预测情感，但真实场景面临两大挑战：**模态缺失**（隐私/采集不完整）和**噪声输入**（传感器噪声）

### 领域现状

**领域现状**：现有方法将这两个问题**分开处理**：去噪方法假设数据完整，缺失补全方法假设数据干净

### 核心矛盾

**核心矛盾**：当噪声和缺失同时出现时，现有方法（如 IMDer、DiCMoR、MoMKE）性能显著下降

### 解决思路

**解决思路**：噪声输入 → 错误重建 → 误差在训练和推理中层层累积

### 解决思路

**本文目标**：如何联合解决多模态情感分析中的噪声干扰和模态缺失问题，避免错误传播？

## 方法详解

### 整体框架

两阶段训练：Intra-Modality Denoising (IMD) → Inter-Modality Complementation (IMC)

### 阶段一：Intra-Modality Denoising (IMD)

在**完整数据**上训练，包含两个去噪模块：

**1. Modality-Specific Denoising (MSD)**: 每个模态独立的 VIB + Attention 网络
- VIB 优化目标：$\mathcal{L}^m = \mathcal{L}_{TASK}(y^m, y) + \beta \text{KL}(p(e_s^m|e^m) \| \mathcal{N}(0, \mathbf{I}))$
- 重参数化：$X_s^m = \mu_s^m + \epsilon \sigma_s^m$，其中 $\mu_s^m = W_1^m e^m + b_1^m$
- 去噪后经 MHA self-attention + 残差 FC 提取 $\hat{X}_{Spe}^m$

**2. Modality-Common Denoising (MCD)**: 所有模态**共享参数**的 VIB + Attention 网络
- 结构与 MSD 相同，但 Conv1D、VIB、Attention 参数跨模态共享
- 提取 modality-invariant 表示 $\hat{X}_{Com}^m$

### 阶段二：Inter-Modality Complementation (IMC)

训练时随机将某些模态置零模拟缺失，利用可用模态补全：

**1. 单模态增强**: 将 specific 和 common 表示通过第一阶段学到的 attention 融合：
$$X_{All}^{m1} = \text{MHA}^{m1}(X_s^{m1}, X_c^{m1})$$

**2. 跨模态补偿**: 可用模态间双向 attention，交换 query/key 获取互补特征 $X_{T2V}$ 和 $X_{V2T}$

**3. 最终融合**: $X = [X_{Compensate}, \hat{X}_{All}^T, \hat{X}_{All}^V]$，经 FC 预测

**总训练目标**:
$$\mathcal{L}_{IMD} = \sum_{m} \left(\sum_{b \in \{Spe,Com\}} \mathcal{L}_b^m + \sum_{k \in \{s,c\}} \mathcal{L}_k^m \right), \quad \mathcal{L}_{IMC} = \mathcal{L}_{TASK}(y_{All}, y)$$

## 实验关键数据


### 主实验

| 方法 | MOSI (Avg ACC/F1) | MOSEI (Avg ACC/F1) | IEMOCAP (Avg WA/UA) |
|------|------------------|-------------------|---------------------|
| MoMKE | 77.05/76.46 | 80.44/79.98 | 73.35/72.78 |
| **TMDC** | **77.64/77.35** | **81.22/80.76** | **73.77/73.64** |

- MOSI 上提升：+0.59 ACC / +0.89 F1；MOSEI 上：+0.78/+0.78；IEMOCAP 上：+0.42/+0.86
- **噪声鲁棒性**（Gaussian noise ε=10）：TMDC vs MoMKE — MOSI 60.8 vs 53.9，MOSEI 71.2 vs 61.2，IEMOCAP 51.0 vs 34.4（平均领先约 **10 个点**）
- Ablation：去掉 IMC 阶段影响最大（MOSI ACC 从 77.64 降至 74.17），去掉 MSD 比去掉 MCD 影响更大

## 亮点与洞察

- **首次联合处理** MSA 中的噪声和缺失模态问题，填补了研究空白
- **两阶段设计合理**: 先去噪再补全，避免噪声通过重建传播
- **VIB 双路去噪**: specific 保留独有信息，common 捕获共享信息，分工明确
- **噪声鲁棒性显著**: 噪声强度 10 时仍比 MoMKE 平均高 10 个百分点
- 在 7 种缺失模态组合下全面评估，覆盖极端情况

## 局限与展望

- 仅在 MOSI/MOSEI/IEMOCAP 三个较小的情感数据集上验证
- 共享表示存在一定冗余（作者在结论中承认），可进一步压缩
- VIB 的 $\beta$ 等超参需要手动调节
- 仅考虑高斯噪声，实际噪声类型更多样（如模态错位、标注噪声）
- 未利用预训练多模态大模型（如 LLM/VLM），特征提取仍基于固定 backbone

## 相关工作与启发

与 MoMKE（先前 SOTA）：MoMKE 训练多个 modality-specific MoE experts 但不做去噪，噪声场景下性能急剧下降。TMDC 通过 VIB 显式去噪 + 两阶段训练策略，在噪声环境下优势明显。与 IMDer/DiCMoR：它们用 diffusion/flow 做模态重建但假设输入无噪声，噪声+缺失同时出现时失效。

## 启发

- "先去噪再补全" 的两阶段范式可推广到其他不完整多模态学习场景（如医学影像多模态融合）
- VIB 作为通用去噪工具在多模态学习中值得更多关注，其信息压缩特性天然适合过滤噪声
- Modality-specific vs modality-common 的解耦是处理模态缺失的有效策略

## 评分

⭐⭐⭐⭐ — 问题定义精准（噪声+缺失的联合处理），方法设计合理，噪声鲁棒性验证充分，但数据集规模偏小且未引入大模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Restore Text First, Enhance Image Later: Two-Stage Scene Text Image Super-Resolution with Glyph Structure Guidance](../../CVPR2026/image_restoration/restore_text_first_enhance_image_later_two-stage_scene_text_image_super-resoluti.md)
- [\[CVPR 2026\] MMDIR: Multimodal Instruction-Driven Framework for Mixed-Degradation Document Image Restoration](../../CVPR2026/image_restoration/mmdir_multimodal_instruction-driven_framework_for_mixed-degradation_document_ima.md)
- [\[CVPR 2026\] Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios](../../CVPR2026/image_restoration/degradation-robust_fusion_an_efficient_degradation-aware_diffusion_framework_for.md)
- [\[CVPR 2026\] Convexity-Aware Noise Calibration: A Self-Supervised Framework for Noise-Level-Unknown Image Denoising](../../CVPR2026/image_restoration/convexity-aware_noise_calibration_a_self-supervised_framework_for_noise-level-un.md)
- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](../../CVPR2026/image_restoration/unicac_universal_computational_aberration_correction_benchmark.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Implementing Adaptations for Vision AutoRegressive Model
description: >-
  [ICML 2025][医学图像][Vision AutoRegressive] 本文首次系统实现并评测了Vision AutoRegressive（VAR）模型的各种适配方法（FFT/LoRA/LNTuning）及差分隐私适配，发现VAR在非DP场景下显著超越扩散模型适配（DiffFit），收敛速度更快、计算效率更高，但DP适配性能仍然不佳，揭示了隐私保护图像生成领域的重要研究空白。
tags:
  - "ICML 2025"
  - "医学图像"
  - "Vision AutoRegressive"
  - "VAR微调"
  - "LoRA"
  - "差分隐私"
  - "参数高效微调"
  - "图像生成"
  - "DiffFit"
---

# Implementing Adaptations for Vision AutoRegressive Model

**会议**: ICML 2025  
**arXiv**: [2507.11441](https://arxiv.org/abs/2507.11441)  
**代码**: [https://github.com/sprintml/finetuning_var_dp](https://github.com/sprintml/finetuning_var_dp)  
**领域**: 医学图像  
**关键词**: Vision AutoRegressive, VAR微调, LoRA, 差分隐私, 参数高效微调, 图像生成, DiffFit

## 一句话总结

本文首次系统实现并评测了Vision AutoRegressive（VAR）模型的各种适配方法（FFT/LoRA/LNTuning）及差分隐私适配，发现VAR在非DP场景下显著超越扩散模型适配（DiffFit），收敛速度更快、计算效率更高，但DP适配性能仍然不佳，揭示了隐私保护图像生成领域的重要研究空白。

## 研究背景与动机

Vision AutoRegressive（VAR）模型近期被提出作为扩散模型（DM）在图像生成领域的强力替代。VAR将"下一个token预测"转变为"下一个尺度预测"，从低分辨率到高分辨率逐步生成2D token网格，速度更快。然而：

**适配方法缺失**：扩散模型已有丰富的微调技术（DiffFit、DreamBooth、Textual Inversion），但VAR的适配方法几乎未被探索。

**差分隐私适配空白**：当微调数据为敏感数据（如医学影像）时，需要隐私保护。DP适配已在DM上广泛研究，VAR尚无此类方案。

**实现层面的挑战**：VAR原始代码库的注意力算子和前向函数需要打补丁才能引入LoRA和DP-SGD。

核心动机：弥合图像自回归模型与扩散模型在模型适配和隐私保护方面的差距。

## 方法详解

### 整体框架

围绕预训练VAR模型（ImageNet-1K class-conditional, 256×256），系统实现三种适配策略：

1. **Full Fine-Tuning (FFT)**：更新模型全部参数
2. **LoRA**：在自注意力的Q/K/V和投影层插入低秩矩阵 $\Delta W = BA$，$r=16$, $\alpha=2r$
3. **LayerNorm Tuning (LNTuning)**：仅更新Adaptive LayerNorm模块的参数

### 关键设计

#### 1. VAR适配

**LoRA实现**：
- 目标模块：自注意力的query、key、value矩阵及投影层
- 同时微调Adaptive LayerNorm模块
- 低秩分解：$\Delta W \in \mathbb{R}^{d \times k}$，$\Delta W = BA$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$

**LNTuning实现**：
- 仅更新Adaptive LayerNorm模块中新引入的可训练参数
- 所有其他权重冻结

实现挑战：VAR原始注意力算子需要打补丁才能引入LoRA适配器（详见附录F）。

#### 2. 差分隐私适配

使用DP-SGD算法：

$\theta_{i+1} = \theta_i - \eta \left(\frac{1}{L}\sum_{k=1}^{L} \text{clip}(g(x_k)) + \mathcal{N}(0, \sigma^2 C^2 I)\right)$

其中 $\text{clip}(g(x_k)) = g(x_k) / \max(1, \|g(x_k)\|_2 / C)$

**增强多重性**（Augmentation Multiplicity）：对每个样本生成 $k$ 个增强视图，平均其梯度以提高信噪比。

实现挑战：需要解决VAR代码中模型特定buffer和非标准前向函数的问题。

### 评估指标

- **FID**（Fréchet Inception Distance）：量化生成质量
- **PFLOPs**：量化计算成本

## 实验关键数据

### 主实验：VAR vs DiffFit（FID↓）

| 模型 | 方法 | Food-101 | CUB-200 | Oxford Flowers | Stanford Cars | 可训练参数 |
|------|------|----------|---------|----------------|---------------|-----------|
| DiT-XL-2 | DiffFit | 6.96 | 5.48 | 20.18 | 9.90 | 0.83M (0.12%) |
| **VAR d16** | **FFT** | **6.11** | 5.74 | **12.08** | **7.42** | 309.6M |
| VAR d16 | LoRA | 6.94 | 7.84 | 13.18 | 8.87 | 6.02M (1.91%) |
| **VAR d20** | **FFT** | **5.38** | **5.58** | **11.65** | **6.31** | 599.7M |
| VAR d20 | LoRA | 6.97 | 6.29 | 11.16 | 9.42 | 9.42M (1.54%) |

关键发现：
- VAR FFT在所有数据集上全面超越DiffFit
- VAR LoRA在多数数据集上也可超越或匹配DiffFit
- **VAR收敛极快**：仅需数千步即达到最终FID，而扩散模型需要长时间训练

### 差分隐私适配（Oxford Flowers, LoRA）

| 模型 | $k=1$ | $k=128$ |
|------|-------|---------|
| VAR-d16 | 69.92 | 63.24 |
| VAR-d20 | 68.92 | 59.29 |

不同$\epsilon$下的DP-LoRA（$k=32$）：

| 模型 | $\epsilon=1$ | $\epsilon=10$ | $\epsilon=100$ | $\epsilon=1000$ |
|------|-------------|--------------|----------------|-----------------|
| VAR-d16 | 196.52 | 60.24 | 41.63 | 35.36 |
| VAR-d20 | 160.33 | 63.38 | 43.35 | 35.06 |

关键发现：
- DP微调下模型难以收敛，需要极高$\epsilon$值才能获得可接受的生成质量
- 增强多重性（$k=128$）仅带来适度改善，但计算成本增加128倍
- LoRA在DP场景下优于LNTuning，可能因为可训练参数更少

### 计算成本

- FFT的计算成本最高（Food-101上约为PEFT的4.5倍）
- LNTuning计算成本最低
- LoRA在性能和成本间取得最佳平衡

## 亮点与洞察

1. **首个VAR适配系统基准**：弥补了图像自回归模型在模型适配方面的评估空白
2. **收敛速度优势**：VAR在少量更新步后即收敛，对比DM需要大量扩展训练——这源于VAR确定性预测目标（无输入噪声随机性）
3. **DP适配的挑战揭示**：DP-SGD的梯度裁剪+噪声注入对VAR的影响比DM更严重，开辟了新的研究方向
4. **开源代码的实用价值**：公开了所有适配方法的实现和补丁，降低了后续研究的门槛

## 局限性

1. 仅在class-conditional VAR上评估，未涉及text-conditional或unconditional设置
2. DP适配性能不佳，尚未找到有效解决方案
3. 仅评估256×256分辨率，高分辨率场景未探索
4. 增强多重性改善有限且计算开销巨大

## 相关工作

- **图像自回归模型**：VAR、Infinity、iGPT
- **扩散模型适配**：DiffFit、DreamBooth、Textual Inversion
- **参数高效微调**：LoRA、LNTuning
- **差分隐私生成模型**：DPDM、DP-LDM

## 评分

⭐⭐⭐ — 工作扎实且开源代码有实用价值，但主要贡献在于"实现和基准测试"而非方法创新。DP适配性能不佳的问题被指出但未解决。作为首个VAR适配基准，具有奠基性意义，但深度和新颖性有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GeneVAR: Causal MeanFlow for Autoregressive Gene-to-WSI Tile Synthesis](../../CVPR2026/medical_imaging/genevar_causal_meanflow_for_autoregressive_gene-to-wsi_tile_synthesis.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](../../NeurIPS2025/medical_imaging/toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](../../CVPR2026/medical_imaging/tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[ICML 2025\] SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures](sgd_jittering_a_training_strategy_for_robust_and_accurate_model-based_architectu.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation
description: >-
  [ICCV 2025][语义分割][领域泛化] 提出 PDAF（概率扩散对齐框架），通过概率扩散建模显式估计潜在域先验（LDP），为现有分割网络提供域偏移补偿，在不需要目标域配对样本的情况下实现跨域泛化的 SOTA 性能。 域泛化语义分割（DGSS）面临的核心挑战是训练域与未见目标域之间的分布偏移。现有方法主要分两类： 数据…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "领域泛化"
  - "扩散模型"
  - "latent domain prior"
  - "probabilistic modeling"
---

# Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.21367](https://arxiv.org/abs/2507.21367)  
**代码**: [https://pdaf-iccv.github.io](https://pdaf-iccv.github.io)  
**领域**: 图像分割  
**关键词**: 领域泛化, 语义分割, 扩散模型, latent domain prior, probabilistic modeling

## 一句话总结

提出 PDAF（概率扩散对齐框架），通过概率扩散建模显式估计潜在域先验（LDP），为现有分割网络提供域偏移补偿，在不需要目标域配对样本的情况下实现跨域泛化的 SOTA 性能。

## 研究背景与动机

域泛化语义分割（DGSS）面临的核心挑战是训练域与未见目标域之间的分布偏移。现有方法主要分两类：

**数据增强法**：通过合成变化增加训练数据多样性，但严重依赖辅助域或生成模型

**域不变表示学习法**：提取跨域一致特征，但风格与内容的纠缠导致关键语义信息丢失

近期方法（如 SPC、DPCL、BlindNet）通过将特征投影到受限特征空间来实现对齐，但忽略了潜在域先验的内在属性。本文认为应当显式建模域先验本身，而非仅做特征对齐。

## 方法详解

### 整体框架

PDAF 将 DGSS 形式化为概率学习问题，引入潜在域先验（LDP）变量 z 来捕获不可直接观测的域偏移。预测函数为：p_{θ,ϕ}(y_t|x_t) = ∫ p_θ(y_t|x_t,z) p_ϕ(z|x_t) dz。PDAF 集成到预训练分割模型中，利用源域和伪目标域的配对图像模拟域偏移，进行 LDP 建模。三个核心模块分别对应 ELBO 中的变分后验、预测模型和先验估计。

### 关键设计

1. **潜在先验提取器 (LPE)**：实现变分后验 q_φ(z|x_t, x_s)。将冻结编码器提取的源域特征 h_{ϑ,s} 和伪目标域特征 h_{ϑ,t'} 拼接，通过残差块建模跨域特征关系，再经两个投影层得到均值 μ 和方差 σ，通过重参数化技巧采样得到最优 LDP z̃ ∈ R^{c'×h×w}（c'=4）。约束 z̃ 服从标准正态分布以增强正则化。

2. **域补偿模块 (DCM)**：实现预测模型 p_θ(y_{t'}|x_{t'}, z')。受空间特征变换（SFT）启发，将 LDP 通过投影层转换为仿射变换参数 γ̃ 和 β̃ ∈ R^{c×h×w}，对特征表示进行缩放和偏移：h̃_{θ,t'} = γ̃ ⊙ h_{θ,t'} ⊕ β̃，然后分割头 D_θ 生成分割图。这种设计在补偿域偏移的同时保留任务相关信息。

3. **扩散先验估计器 (DPE)**：实现先验 p_ϕ(z'|x_t)。关键创新——使用概率扩散建模在无需配对样本的情况下估计 LDP。前向扩散过程对最优 LDP z̃ 加噪声得到 z_T，反向过程以目标特征 h_{θ,t'} 为条件，从 z_T 去噪得到估计 LDP ẑ_0。采用加速扩散优化，仅需 T=4 步，可与分割头联合训练。推理时仅需目标域图像，从高斯噪声出发经 DPE 估计 LDP，再由 DCM 补偿特征。

### 损失函数 / 训练策略

总损失：L_total = λ_task · L_task + λ_sc · L_sc + λ_prior · L_prior

- **L_task**：任务损失，DeepLabV3+ 用加权交叉熵，Mask2Former 用 focal loss
- **L_sc**：语义一致性损失，衡量源域和伪目标域预测之间的差异，加速收敛
- **L_prior**：先验约束损失 = ||ẑ_0 - z̃||_2，对齐 DPE 和 LPE 的输出
- 损失系数：(λ_task, λ_sc, λ_prior) = (0.5, 0.5, 1.0)
- 训练：Adam 优化器，lr=1e-5，batch size=4，100 epochs
- 扩散参数：T=4 步，β 从 0.1 线性增至 0.99
- 硬件：单张 RTX 4090

## 实验关键数据

### 主实验

Cityscapes 训练 → 跨域测试（DeepLabV3+ / ResNet-50）:

| 方法 | BDD(B) | Mapillary(M) | GTAV(G) | SYNTHIA(S) | Avg |
|------|--------|-------------|---------|-----------|-----|
| DeepLabV3Plus | 44.96 | 51.68 | 42.55 | 23.29 | 40.62 |
| BlindNet | 51.84 | 60.18 | 47.97 | 28.51 | 47.13 |
| **PDAF** | **53.50** | **62.93** | **50.54** | **30.68** | **49.41** |

Mask2Former (Swin-L) 上的表现：

| 方法 | B | M | G | S | Avg |
|------|---|---|---|---|-----|
| CMFormer | 62.60 | 73.60 | 60.70 | 43.00 | 59.98 |
| **PDAF** | **63.00** | **74.10** | **63.20** | **44.00** | **61.08** |

ACDC 恶劣天气场景（Mask2Former Swin-L）：

| 方法 | Foggy | Night | Rain | Snow | Avg |
|------|-------|-------|------|------|-----|
| HGFormer | 69.90 | 52.70 | 72.00 | 68.60 | 65.80 |
| **PDAF** | **80.72** | **55.12** | **73.13** | **71.43** | **70.10** |

### 消融实验

各模块贡献（Cityscapes → 其他域, DeepLabV3+ ResNet-50）:

| 设置 | B | M | G | S | Avg |
|------|---|---|---|---|-----|
| Baseline | 44.96 | 51.68 | 42.55 | 23.29 | 40.62 |
| PDAF w/o LPE | 49.59 | 57.63 | 46.34 | 27.19 | 45.19 |
| PDAF w/o DCM | 51.51 | 60.79 | 49.42 | 29.55 | 47.82 |
| PDAF w/o DPE | 51.98 | 60.33 | 49.38 | 28.86 | 47.64 |
| **PDAF (完整)** | **53.50** | **62.93** | **50.54** | **30.68** | **49.41** |

### 关键发现

- LPE 贡献最大（+4.57 Avg），表明潜在域先验建模是核心
- 三个模块缺一不可，移除任何一个都导致性能下降 1.5-4.2 mIoU
- PDAF 在 CNN (DeepLabV3+) 和 Transformer (Mask2Former) 架构上均有效
- 在 ACDC 恶劣天气上提升尤为显著（平均 +4.3 mIoU vs HGFormer），说明 LDP 能有效补偿退化特征
- GTAV 源域训练时同样有效，证明从合成到真实域的迁移能力

## 亮点与洞察

- **理论优雅**：将 DGSS 形式化为变分推断问题，ELBO 推导完整，三个模块分别对应 ELBO 的不同项
- **即插即用**：PDAF 可集成到任意现有分割模型（编码器冻结 + 微调目标网络），无需改变原始架构
- **推理时不需配对样本**：DPE 学会从纯噪声估计 LDP，推理时无需源域数据
- 扩散过程仅 T=4 步，引入开销极小（单张 4090 可训练），实际增加的参数和计算量很小
- 在恶劣天气场景的大幅提升（ACDC 上 +4.3 mIoU）展示了极强的实用价值

## 局限与展望

- 伪目标域图像仅通过光度增强生成，增强策略的多样性受限
- LDP 通道数固定为 c'=4，不同域偏移的复杂度可能需要自适应维度
- 目前仅在城市场景验证，其他场景（如室内、医疗）的泛化性待验证
- 未与基于视觉基础模型（VFM）的最新 DGSS 方法进行全面比较

## 相关工作与启发

- 与 DiffIR、CDFormer 等利用扩散先验的方法相比，PDAF 将扩散用于域先验估计而非图像重建，更轻量
- 与 DatasetDM、DGInStyle 等使用外部扩散模型的方法相比，PDAF 内置轻量扩散模块，避免了大量计算开销
- LDP 的概率建模思路可推广到其他域适应任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将概率扩散建模引入域泛化，理论框架完整且实现轻量
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多backbone、恶劣天气评估，消融全面
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，模块关系清晰，图示质量高
- **价值**: ⭐⭐⭐⭐ 即插即用的域泛化增强方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[CVPR 2026\] Beyond Text: Visual Description Assembly by Probabilistic Model for CLIP-based Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/beyond_text_visual_description_assembly_by_probabilistic_model_for_clip-based_we.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->

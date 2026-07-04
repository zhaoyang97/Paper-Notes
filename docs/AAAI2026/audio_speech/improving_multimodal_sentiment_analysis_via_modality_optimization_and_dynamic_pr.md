---
title: >-
  [论文解读] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
description: >-
  [AAAI 2026][音频/语音][多模态] 提出 MODS 框架，通过图卷积动态序列压缩（GDC）消除非语言模态冗余，并设计样本级动态主模态选择器（MSelector）和主模态中心交叉注意力（PCCA），实现 MSA 中按样本自适应选择主导模态。 - MSA 中不同模态对情感预测贡献不均，language 通常信息密度最…
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "多模态"
  - "dynamic modality selection"
  - "graph convolutional network"
  - "capsule network"
  - "注意力机制"
  - "sequence compression"
---

# Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection

**会议**: AAAI 2026  
**arXiv**: [2511.06328](https://arxiv.org/abs/2511.06328)  
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: multimodal sentiment analysis, dynamic modality selection, graph convolutional network, capsule network, cross-modal attention, sequence compression  

## 一句话总结

提出 MODS 框架，通过图卷积动态序列压缩（GDC）消除非语言模态冗余，并设计样本级动态主模态选择器（MSelector）和主模态中心交叉注意力（PCCA），实现 MSA 中按样本自适应选择主导模态。

## 背景与动机

- MSA 中不同模态对情感预测贡献不均，language 通常信息密度最高，是默认主模态
- 已有方法固定以 language 为主模态（如 TCSP、ALMT），无法适应个别样本中非语言模态更具情感区分力的情况
- HCT-DMG 虽提出动态选择，但仅支持 batch-level 选择（因异步序列限制），且忽略非语言模态的**序列冗余**
- Audio/visual 序列信息密度远低于 text，直接作为主模态会引入噪声干扰

## 核心问题

如何在样本级别动态选择最强模态作为主模态，同时解决非语言模态序列冗余导致的特征质量问题？

## 方法详解

### 整体框架

MODS = **GDC**（图压缩模块）+ **MSelector**（主模态选择器）+ **PCCA**（主模态中心交叉注意力）。

### 关键设计 1：Graph-based Dynamic Compression (GDC)

用 Capsule Network 将 audio/visual 长序列压缩为与 text 等长的图节点：

$$\text{Caps}_m^{i,j} = W_m^{ij} H_m^i$$

通过动态路由迭代更新路由系数 $r_m^{i,j}$，含噪/冗余 capsule 自动获得低权重，生成高质量节点 $N_m^j = \sum_i \text{Caps}_m^{i,j} \times r_m^{i,j}$。

之后用 self-attention 构建边权重，再通过 GCN 学习图表示：

$$H_m^l = \text{ReLU}(D_m^{-1/2} E_m D_m^{-1/2} H_m^{l-1} W_m^l + b_m^l)$$

压缩后 $H_a, H_v \in \mathbb{R}^{T_l \times d}$，与 language 序列长度对齐。

### 关键设计 2：Primary Modality Selector (MSelector)

对每个模态做 attention-based aggregation 得到向量 $h_m$，拼接后通过 MLP + softmax 输出三个权重：

$$w = \text{softmax}(\text{MLP}(\text{concat}(h_a, h_l, h_v))), \quad p = \arg\max(w_a, w_t, w_v)$$

权重最高的模态被选为主模态 $p$，各模态特征乘以对应权重后送入后续模块。实现**样本级**动态选择。

### 关键设计 3：Primary-modality-Centric Cross-Attention (PCCA)

多层迭代增强，每层包含：
1. 两个 cross-attention $CA_{a \to p}$：辅助模态信息流向主模态
2. 一个 self-attention $SA_p$：主模态自增强
3. 融合：$H_p^{[i+1]} = H_{p_{update}}^{[i]} + \sum_{a} H_{a \to p}^{[i]}$
4. 反向 cross-attention $CA_{p \to a}$：增强后的主模态信息回传辅助模态

最终层仅保留 $CA_{a \to p}$，输出 $H_p$ 用于情感回归。

### 训练损失

$$\mathcal{L}_{task} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{NCE}$$

InfoNCE 损失从融合特征反向预测各单模态特征，稳定主模态选择。

## 实验关键数据

| 方法 | MOSI MAE↓ | MOSI Acc-7↑ | MOSI Acc-2↑ | MOSEI Acc-2↑ | SIMS Acc-5↑ |
|------|----------|------------|------------|-------------|------------|
| Self-MM | 0.708 | 46.67 | 83.44/85.46 | 83.76/85.15 | 41.53 |
| MMIM | 0.718 | 46.64 | 83.38/85.82 | 82.08/85.14 | - |
| DTN | 0.716 | 47.5 | -/85.1 | -/85.5 | 44.26 |
| **MODS** | **0.688** | **49.27** | **83.53/85.83** | **84.52/85.88** | **45.51** |

- 在 4 个数据集（MOSI、MOSEI、SIMS、SIMSv2）上全面 SOTA
- SIMS Acc-5 45.51%（vs DTN 44.26%），SIMSv2 Acc-5 55.51%（vs DTN 53.71%）
- 消融：去掉 GDC 后 MOSI Acc-7 从 49.27 降到 45.34（-3.93）；固定任一模态为主模态均降 3-4 个点
- Case study 展示了语言为正/音视觉为负时选 language，语言中性/音视觉为正时选非语言模态

## 亮点

- 首个实现**样本级**动态主模态选择的 MSA 方法（而非 batch 级）
- GDC 用 capsule network 构建图节点的设计巧妙：动态路由自动过滤冗余/噪声
- PCCA 以主模态为桥梁进行信息流动，避免辅助模态间直接交互产生干扰
- 在 SIMS/SIMSv2 等模态平衡数据集上也显著优于固定主模态方法，验证了动态选择的价值

## 局限与展望

- MSelector 的 argmax 操作不可微，训练时依赖 softmax 权重做近似，可能导致选择不够锐利
- 仅 3 模态场景验证，扩展到更多模态时 MSelector 设计需重新考虑
- GDC 将 audio/visual 压缩到与 text 等长，长度选择较刚性，可能不是所有样本的最优压缩比
- 未探索预训练多模态 backbone（如 CLIP、Whisper），仅用传统特征提取器

## 对比

| 维度 | MODS | HCT-DMG | PaSE | ALMT |
|------|------|---------|------|------|
| 主模态选择 | 样本级动态 | Batch级动态 | 无（均等） | 固定language |
| 序列压缩 | GDC (Capsule+GCN) | 无 | 无 | 无 |
| 融合方式 | PCCA（主模态中心） | 层级式 | Prototype门控 | Text中心注意力 |
| 核心问题 | 模态选择+冗余 | 模态选择 | 模态竞争 | 模态交互 |

## 启发

- Capsule network 的动态路由用于序列压缩是一个值得关注的范式，相比 pooling 更能保留关键信息
- "主模态中心"的融合范式可避免弱模态间的噪声交叉传播，在信息质量不均的场景中特别有效
- 动态主模态选择可推广到多模态 LLM 中处理不同质量输入模态的场景

## 评分

⭐⭐⭐⭐ — 样本级动态选择 + 图压缩的组合设计合理且有效，但核心模块的可微性和扩展性有待加强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](../../ACL2025/audio_speech/improving_language_and_modality_transfer_in.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)

</div>

<!-- RELATED:END -->

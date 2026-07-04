---
title: >-
  [论文解读] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis
description: >-
  [AAAI 2026][音频/语音][多模态] 提出 PaSE 框架，通过原型引导校准对齐（Entropic Optimal Transport）与 Shapley 值梯度调制的双阶段优化策略，显式解决多模态情感分析中的模态竞争问题。 - 多模态情感分析（MSA）融合 text、audio、visual 三模态…
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "多模态"
  - "modality competition"
  - "prototype alignment"
  - "Shapley value"
  - "optimal transport"
  - "gradient modulation"
---

# PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis

**会议**: AAAI 2026  
**arXiv**: [2511.17585](https://arxiv.org/abs/2511.17585)  
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: multimodal sentiment analysis, modality competition, prototype alignment, Shapley value, optimal transport, gradient modulation

## 一句话总结

提出 PaSE 框架，通过原型引导校准对齐（Entropic Optimal Transport）与 Shapley 值梯度调制的双阶段优化策略，显式解决多模态情感分析中的模态竞争问题。

## 背景与动机

- 多模态情感分析（MSA）融合 text、audio、visual 三模态，但实践中常出现**模态竞争**：强势模态压制弱势模态，导致融合效果不如预期
- 如 CMU-MOSI 上，在 text-only 基线上加入 audio/visual 带来的增益有限甚至负收益
- 已有方法多假设模态间天然互补，缺乏对模态竞争动态的显式建模
- 现有梯度调制方法（如 OGM-GE）依赖梯度范数等间接信号，缺乏对每个模态贡献的原则性量化

## 核心问题

如何在多模态情感分析中显式量化并平衡各模态贡献，缓解强势模态（通常是 text）对弱势模态的抑制？

## 方法详解

### 整体框架

PaSE 包含三个模块：**PCL**（模态内原型校准）→ **CAL**（跨模态对齐）→ **双阶段优化**（原型门控融合 + Shapley 梯度调制）。

### 关键设计 1：Prototype-guided Calibration Learning (PCL)

为每个模态的每个类别维护原型向量（动量更新，$\gamma=0.98$）：

$$c_k^m \leftarrow \gamma c_k^m + (1-\gamma) \frac{1}{|B_k|}\sum_{i \in B_k} h_i^m$$

通过对比学习损失拉近同类样本、推远异类：

$$\mathcal{L}_{\text{intra}}^m = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{\phi(h_i^m, c_{y_i}^m)/\tau}}{\sum_{k=1}^K e^{\phi(h_i^m, c_k^m)/\tau}}$$

### 关键设计 2：Cross-modal Alignment via Entropic Optimal Transport (CAL)

将各模态类别原型视为离散分布，用 Entropic OT 求解跨模态传输计划，最小化 Wasserstein 距离。引入双向对称匹配损失和一致性正则：

$$\mathcal{L}_{\text{match}} = \frac{1}{2}\left(\langle \mathbf{Q}^{(m \to n)}, \mathbf{C} \rangle_F + \langle \mathbf{Q}^{(n \to m)}, \mathbf{C}^\top \rangle_F\right)$$

$$\mathcal{L}_{\text{reg}} = \|\mathbf{Q}^{(m \to n)} - (\mathbf{Q}^{(n \to m)})^\top\|_F^2$$

### 关键设计 3：Shapley-based Gradient Modulation (SGM)

用 Shapley 值量化每个模态的边际贡献：

$$\psi_m(u) = \sum_{S \subseteq \mathcal{M} \setminus \{m\}} \frac{|S|!(k-|S|-1)!}{k!}[u(S \cup \{m\}) - u(S)]$$

归一化后计算调制因子 $\varphi_m = \exp(\tilde{\psi}_{\min}/\tilde{\psi}_m - 1)$，弱势模态获得更大学习率，强势模态被抑制。

### 双阶段训练

- **阶段 1**（warm-up）：用 entropy-guided 权重和 Prototype-Gated Fusion 正常训练，允许强模态引导
- **阶段 2**：验证集 entropy 稳定后自动切入 SGM，Shapley 梯度调制平衡模态贡献

## 实验关键数据

| 方法 | MOSI Acc-2↑ | MOSI Acc-7↑ | MOSEI Acc-2↑ | MOSEI Corr↑ |
|------|------------|------------|-------------|------------|
| MSAmba (AAAI'25) | 85.99/87.43 | 49.67 | 85.78/86.86 | 0.796 |
| Semi-IIN (AAAI'25) | 85.28/87.04 | 46.50 | 84.98/87.70 | 0.804 |
| **PaSE** | **86.40/88.32** | **50.92** | **86.07/88.10** | **0.831** |

- IEMOCAP 四类情感 F1：Happy 91.5, Sad 88.6, Angry 89.4, Neutral 73.2，全面最优
- 消融：去掉 SGM 后 MOSI Acc-2 下降 2.85%，影响最大；去掉 CAL 下降 1.52%
- 全模态 vs 最优双模态：平均提升 4.02%，有效解决"加模态反降"问题
- 对比 GPT-4o-mini：PaSE (BERT-base) 在 MOSI 上 88.32 vs 86.54，轻量模型仍占优

## 亮点

- 首次将 **Shapley 值**引入多模态情感分析的梯度调制，提供理论上有原则的模态贡献量化
- Entropic OT 的双向对称对齐 + 结构保持正则，比简单 contrastive loss 更严格
- 双阶段训练策略合理：先让强模态建立表征结构，再用 SGM 平衡，避免过早调制导致不稳定
- t-SNE 可视化表明 PaSE 的融合表征类别分离度远优于 SelfMM/EUAR

## 局限与展望

- Shapley 值计算需遍历所有模态子集（3 模态时 $2^3$ 种），模态数增多时计算开销指数增长
- 仅在 MOSI/MOSEI/IEMOCAP 上验证，缺少更多 challenging 场景（如讽刺检测、多语言）
- 特征提取器较旧（Facet、COVAREP），未与更强的视觉/音频 backbone 结合验证
- 原型更新策略较简单（动量 EMA），未探索更精细的原型维护机制

## 对比

| 维度 | PaSE | OGM-GE | PMR | ConFEDE |
|------|------|--------|-----|---------|
| 模态贡献量化 | Shapley value | 梯度范数 | 渐进强化 | 对比分解 |
| 对齐方式 | Entropic OT | 无 | 无 | 对比学习 |
| 融合策略 | Prototype-Gated | 简单融合 | 三向注意力 | 共享-私有 |
| 理论保证 | 博弈论 | 启发式 | 无 | 无 |

## 启发

- Shapley 值虽计算成本高，但在模态数有限（3-5）时完全可行，可推广到其他多模态任务
- "先让模型自由学习，再引入调制平衡"的双阶段范式比全程调制更稳定
- Entropic OT 做原型对齐的思路可用于 VLM 中不同模态 embedding space 的对齐

## 评分

⭐⭐⭐⭐ — 理论动机清晰，Shapley 梯度调制思路新颖且有效，但实验验证场景和 backbone 较保守

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding
description: >-
  [NeurIPS 2025][医学图像][fMRI解码] 提出 Zebra，首个零样本脑视觉解码框架，通过对抗训练与残差分解将 fMRI 表征解耦为主体不变和语义特定成分，无需对新被试做微调即可实现跨被试的视觉重建泛化。 fMRI-to-Image 重建是计算神经科学与计算机视觉的前沿方向，旨在将大脑视觉皮层的 BOLD 信…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "fMRI解码"
  - "零样本泛化"
  - "对抗训练"
  - "表征解耦"
  - "脑视觉解码"
---

# Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2510.27128](https://arxiv.org/abs/2510.27128)  
**代码**: [GitHub](https://github.com/xmed-lab/ZEBRA)  
**领域**: 其他  
**关键词**: fMRI解码, 零样本泛化, 对抗训练, 表征解耦, 脑视觉解码

## 一句话总结

提出 Zebra，首个零样本脑视觉解码框架，通过对抗训练与残差分解将 fMRI 表征解耦为主体不变和语义特定成分，无需对新被试做微调即可实现跨被试的视觉重建泛化。

## 研究背景与动机

fMRI-to-Image 重建是计算神经科学与计算机视觉的前沿方向，旨在将大脑视觉皮层的 BOLD 信号逆向工程为图像。然而，现有方法面临一个关键挑战：**无法跨个体泛化**。

当前方法（MindEye2、MindTuner 等）普遍采用两阶段方案：先在多被试数据上预训练统一模型，再对特定被试微调。这种范式存在严重限制：（1）每个新患者都需要 AI 专家进行微调；（2）微调过程耗时（约一天），阻碍脑机接口的实时应用；（3）没有能跨人类被试学习神经表征的通用特征空间。

**核心论点**：尽管个体间大脑活动存在差异，人类皮层以一种跨被试一致的、拓扑组织的方式编码语义信息（神经科学证据）。因此，可以通过显式分离主体不变成分和语义特定成分来实现零样本泛化。

现有方法在零样本设置下均失败：MindTuner 的主体特定设计在新被试上直接失效；NeuroPictor 虽能将不同被试 fMRI 变换为统一形状，但对主体噪声敏感，无法学到不变表征。

## 方法详解

### 整体框架

Zebra 基于一个基线框架（fMRI-PTE 编码器 + unCLIP 扩散先验 + SDXL 解码器），在此基础上增加两个核心模块：Subject-Invariant Feature Extraction (SIFE) 和 Semantic-Specific Feature Extraction (SSFE)。训练仅在训练集被试上进行一次，测试时直接对未见被试推理。

### 关键设计

1. **主体不变特征提取（SIFE）**：通过残差分解和对抗训练分离主体不变特征。自注意力模块 $\mathcal{F}_i$ 提取不变特征 $\bm{E}_i = \mathcal{F}_i(\bm{E})$，残差得到主体特定特征 $\bm{E}_s = \bm{E} - \bm{E}_i$。

   对抗训练确保 $\bm{E}_i$ 不含主体信息——主体判别器 $\mathcal{D}_{dis}$ 尝试从 $\bm{E}_i$ 识别被试身份，不变提取器 $\mathcal{F}_i$ 则阻止识别：

$$\min_{\theta_{\mathcal{E}}, \theta_{\mathcal{F}}} \max_{\theta_{\mathcal{D}_{dis}}} \left\{ \mathcal{L}_{dis}^{\bm{E}} := -\mathbb{E}_{x,s} [s \log \mathcal{D}_{dis}(\mathcal{E}(\mathcal{F}_i(\bm{E})))] \right\}$$

   同时训练分类器 $\mathcal{D}_{cls}$ 使 $\bm{E}_s$ 保留主体信息（$\mathcal{L}_{cls}^{\bm{E}}$），形成互补约束。

2. **表征保持锚（Representation Preservation Anchor）**：对抗训练可能扭曲原始特征空间。通过辅助 fMRI 重建任务保持特征空间的信息完整性：

$$\mathcal{L}_{rec} = \mathbb{E}_{(x, \hat{x})} [|\hat{x} - x|]$$

   使用两层反卷积 + 线性预测头重建输入信号，确保 $\bm{E}$ 在对抗训练下仍保留生物学保真度和语义连贯性。

3. **语义特定特征提取（SSFE）**：对 $\bm{E}_i$ 进一步注入语义信息。通过 vision projector 将脑特征投影到 CLIP 视觉空间，得到语义特定特征 $\bm{F}_s = \mathcal{P}_s(\bm{E}_i)$ 和语义不变特征 $\bm{F}_i = \mathcal{P}_i(\bm{E}_s)$。使用 BiMixCo 损失将 $\bm{F}_s$ 与 OpenCLIP 嵌入对齐（$\mathcal{L}_{spe}^{\bm{F}}$），并通过梯度反转层（GRL）阻止 $\bm{F}_i$ 与 CLIP 特征对齐（$\mathcal{L}_{inv}^{\bm{F}}$），强制更多语义信息流入 $\bm{F}_s$。

### 损失函数 / 训练策略

总损失整合七个组件：

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{dis}^{\bm{E}} + \mathcal{L}_{cls}^{\bm{E}} + \mathcal{L}_{inv}^{\bm{F}} + \mathcal{L}_{spe}^{\bm{F}} + \mathcal{L}_{sem} + \lambda \mathcal{L}_{prior}$$

其中 $\mathcal{L}_{sem} = \mathcal{L}_{cls} + \mathcal{L}_{\text{CLIP}_v} + \mathcal{L}_{\text{CLIP}_t}$，$\lambda=30$。训练 60 epochs，8 张 H800 GPU，batch size 128，AdamW 优化器，lr=1e-4。推理时使用 SDXL unCLIP 两阶段解码。

## 实验关键数据

### 主实验（NSD 数据集，被试 1/2/5/7 平均）

| 方法 | 训练方式 | PixCorr↑ | SSIM↑ | Alex(2)↑ | Alex(5)↑ | Incep↑ | CLIP↑ |
|------|---------|---------|-------|----------|----------|--------|-------|
| NeuroPictor⋆ | 零样本 | 0.057 | 0.297 | 71.4% | 74.7% | 62.5% | 66.0% |
| Our baseline | 零样本 | 0.074 | 0.316 | 70.8% | 74.0% | 63.5% | 62.5% |
| **Zebra** | **零样本** | **0.131** | **0.375** | **74.6%** | **81.2%** | **72.2%** | **71.5%** |
| MindEye2 | 少样本(1h) | 0.195 | 0.419 | 84.2% | 90.6% | 81.2% | 79.2% |
| MindTuner | 全微调 | 0.322 | 0.421 | 95.8% | 98.8% | 95.6% | 93.8% |

Zebra 零样本显著超越其他零样本方法（PixCorr +0.074，Incep +9.7%），且部分指标接近全微调模型。

### 消融实验

| 基线 | SIFE对抗 | SIFE锚 | SSFE对抗 | SSFE锚 | PixCorr | Alex(5) | CLIP |
|------|---------|--------|---------|--------|---------|---------|------|
| ✓ | | | | | 0.089 | 74.7% | 63.2% |
| ✓ | ✓ | | | | 0.129 | 77.4% | 66.8% |
| ✓ | ✓ | ✓ | | | 0.134 | 78.3% | 69.3% |
| ✓ | ✓ | ✓ | ✓ | | 0.142 | 79.6% | 70.8% |
| ✓ | ✓ | ✓ | ✓ | ✓ | **0.153** | **81.8%** | **72.3%** |

### 关键发现

- 训练被试数从 4 增到 7 时所有指标稳步提升（CLIP: 63.7% → 72.3%），说明更多被试数据有助于提升泛化
- UMAP/t-SNE 可视化确认 $\bm{E}_i$ 跨被试高度混合（无被试聚类），$\bm{E}_s$ 则清晰按被试聚类
- 零样本推理每张图约 1 秒，而传统微调方案需超过 12 小时
- Zebra 在低层感知指标上优势更大，语义准确性仍弱于少样本方法

## 亮点与洞察

- **问题定义开创性**：首次提出零样本脑视觉解码问题，将 fMRI 解码从需要被试特定微调推向即插即用
- **神经科学驱动的设计**：基于"皮层跨个体一致性编码语义"的神经科学证据，通过对抗+残差分解实现表征解耦
- **表征保持锚巧妙**：解决了对抗训练容易破坏特征空间的经典问题，fMRI 重建作为锚点保持信息完整性
- 从实用角度看，零样本方案对临床应用（脑机接口、神经康复）有巨大价值

## 局限与展望

- 语义保真度仍弱于少样本方法，在罕见物体类别上表现不佳
- 仅在 NSD 数据集（8 个被试）上验证，被试数量有限
- 仅聚焦图像重建，未探索文本或视频等更复杂模态
- 需要更多被试和 fMRI 记录来全面捕捉真实世界视觉体验

## 相关工作与启发

与 MindEye2、MindTuner 等需要微调的方法相比，Zebra 完全不需要测试被试数据。与 NeuroPictor 的统一脑编码相比，Zebra 通过显式解耦有效去除主体噪声。启发：在个体差异极大的生物医学场景中，对抗解耦可能是实现零样本泛化的通用策略。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次零样本脑视觉解码，问题定义和方法均有开创性
- 实验充分度: ⭐⭐⭐⭐ 定量/定性/消融/可视化全面，但数据集和被试规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法展示直观，实验组织合理
- 价值: ⭐⭐⭐⭐⭐ 对脑机接口和临床神经科学具有重大实用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)

</div>

<!-- RELATED:END -->

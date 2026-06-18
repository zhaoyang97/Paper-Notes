---
title: >-
  [论文解读] ViLU: Learning Vision-Language Uncertainties for Failure Prediction
description: >-
  [ICCV2025][信息检索/RAG][不确定性量化] 提出 ViLU，一个针对 VLM 零样本预测的后验不确定性量化框架，通过交叉注意力融合视觉嵌入、预测文本嵌入和图像条件文本表示，构建不确定性感知的多模态表征，在 13 个分类数据集和大规模图文数据集上显著超越现有失败预测方法。 视觉-语言模型（VLM…
tags:
  - "ICCV2025"
  - "信息检索/RAG"
  - "不确定性量化"
  - "失败预测"
  - "VLM"
  - "交叉注意力"
  - "后验估计"
---

# ViLU: Learning Vision-Language Uncertainties for Failure Prediction

**会议**: ICCV2025  
**arXiv**: [2507.07620](https://arxiv.org/abs/2507.07620)  
**代码**: [GitHub](https://github.com/ViLU-uncertainty/ViLU)  
**领域**: 信息检索  
**关键词**: 不确定性量化, 失败预测, VLM, 交叉注意力, 后验估计  

## 一句话总结

提出 ViLU，一个针对 VLM 零样本预测的后验不确定性量化框架，通过交叉注意力融合视觉嵌入、预测文本嵌入和图像条件文本表示，构建不确定性感知的多模态表征，在 13 个分类数据集和大规模图文数据集上显著超越现有失败预测方法。

## 研究背景与动机

视觉-语言模型（VLM，如 CLIP）在零样本分类中表现优异，但其预测的可靠性量化仍是开放挑战。可靠的不确定性量化（UQ）在安全关键领域至关重要。

现有方法的局限性：

**Maximum Concept Matching (MCM)**：VLM 版本的最大类别概率。虽然简单有效，但在设计上会对错误预测赋予过高置信度。例如，当 "American Eskimo dog" 被错误分类为 "Siberian husky" 时，MCM 仍给出高置信分数，无法检测错误。

**Learning Visual Uncertainties (LVU)**：仅基于视觉特征学习预测分类器的损失值。应用到 VLM 时不建模下游概念之间的关系，限制了失败预测能力。LVU 同样无法区分上述狗品种的错误。

**BayesVLM**：使用 Laplace 近似建模嵌入的不确定性，但其目标并非针对失败预测优化。

核心 insight：VLM 的不确定性来源于**两个模态**——视觉模式的模糊性和文本概念间的歧义性。有效的 UQ 须同时建模这两种不确定性及其交互。

## 方法详解

### 整体框架

ViLU 是一个**后验（post-hoc）**框架，仅利用 VLM 输出的视觉和文本嵌入，无需访问模型内部参数。框架包含三个核心组件：

1. 视觉-文本交叉注意力模块
2. ViLU 嵌入构建
3. 失败预测分类头

### 关键设计一：视觉-文本交叉注意力

给定视觉嵌入 $\bm{z}_v$ 和 $K$ 个候选文本概念的嵌入 $Z_t = \{\bm{z}_{t_j}\}_{1 \leq j \leq K}$，通过交叉注意力生成图像条件的文本表示：

$$\bm{z}_t^\alpha = h_{\theta_{\text{XA}}}(\bm{z}_v, \bm{z}_{t_1}, ..., \bm{z}_{t_K})$$

具体地，使用视觉表示作为 Query，文本嵌入作为 Key/Value：

$$\bm{\alpha} = \text{softmax}\left(\frac{(W_Q \bm{z}_v)^\top (W_K Z_t)}{\sqrt{d}}\right), \quad \bm{z}_t^\alpha = \sum_{j=1}^K \bm{\alpha}_j (W_V \bm{z}_{t_j})$$

这个加权文本嵌入根据模型对候选概念的预测分布进行重新上下文化，使得 ViLU 能捕获概念间的细粒度歧义。

### 关键设计二：ViLU 嵌入

构建三元组不确定性嵌入：

$$\bm{z}_{\text{ViLU}} = (\bm{z}_v, \bm{z}_{\hat{t}}, \bm{z}_t^\alpha)$$

- $\bm{z}_v$：视觉嵌入（捕获视觉模糊性）
- $\bm{z}_{\hat{t}}$：预测文本嵌入（模型当前的最佳猜测，近似 MCM 的信息）
- $\bm{z}_t^\alpha$：交叉注意力输出（捕获所有候选概念的加权关系）

仅用前两者可以近似 MCM 的行为，但忽略了被混淆的替代概念；加入 $\bm{z}_t^\alpha$ 后能有效捕获多概念间的歧义。

### 关键设计三：失败预测目标

不同于传统 UQ 方法预测分类器的损失值（回归任务），ViLU 将其建模为**二分类任务**——直接区分正确和错误预测：

$$\hat{y}_i = \sigma(g_{\theta_{\text{MLP}}}(\bm{z}_{\text{ViLU}}))$$

训练使用加权二分类交叉熵：

$$\mathcal{L}_{\text{wBCE}} = -\frac{1}{B}\sum_i \left[w y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

其中权重 $w$ 根据每个 mini-batch 的正确/错误样本比例自适应调整：

$$w = \log\left(1 + \frac{\sum_i(1-y_i)}{\sum_i y_i}\right)$$

这种设计使 ViLU 完全**损失无关（loss-agnostic）**，不需要知道 VLM 预训练时使用的损失函数（对比损失/sigmoid 损失），特别适合黑盒 VLM 的后验设置。

## 实验结果

### 图像分类数据集失败预测（CLIP ViT-B/32）

| 方法 | 平均 AUC↑ | 平均 FPR95↓ |
|------|----------|------------|
| MCM | 81.8 | 70.6 |
| Entropy | 80.2 | 74.0 |
| Doctor | 81.6 | 71.2 |
| Rel-U | 80.7 | 68.6 |
| LVU | 85.0 | 57.4 |
| BayesVLM | 84.2 | 65.1 |
| **ViLU** | **93.2** | **29.9** |

ViLU 在 13 个数据集的平均 AUC 上超越所有方法（+8.2 vs LVU），FPR95 降低 27.5 个百分点。在 Flowers102 上达到 98.7% AUC / 5.1% FPR95。

### 大规模图文数据集

| 方法 | CC3M AUC↑ | CC12M AUC↑ | LAION-400M AUC↑ |
|------|----------|-----------|----------------|
| MCM | 83.9 | 88.8 | 91.7 |
| LVU | 69.3 | 74.4 | 80.2 |
| BayesVLM | 87.1 | 90.9 | 95.1 |
| **ViLU** | **91.4** | **95.2** | **97.3** |

在开放词汇场景（CC12M, LAION-400M）中，LVU 表现甚至不如 MCM，证明仅建模视觉不确定性在此场景下是不够的。ViLU 在 CC12M 上 FPR95 仅为 25.2%（BayesVLM: 53.3%）。

### 消融实验

| 视觉嵌入 | 交叉注意力 | 预测文本 | CIFAR-10 AUC | ImageNet AUC | CC12M AUC |
|---------|----------|---------|-------------|-------------|----------|
| ✓ | ✗ | ✗ | 96.4 | 78.7 | 74.0 |
| ✓ | ✗ | ✓ | 97.9 | 88.8 | 88.9 |
| ✓ | ✓ | ✗ | 97.7 | 86.1 | 93.6 |
| ✓ | ✓ | ✓ | **98.3** | **89.5** | **95.2** |
| MCM | - | - | 89.9 | 80.8 | 88.8 |

- 加入预测文本嵌入在 ImageNet 上提升 +10 AUC，在 CC12M 上 +14.9 AUC
- 交叉注意力在 CC12M 上从 88.9 提升到 95.2（+6.3），因为该数据集的文本概念随 batch 动态变化
- 二分类 BCE 损失显著优于回归 MSE 损失（ImageNet AUC: 89.5 vs 85.7）

## 亮点与洞察

1. **问题定义精准**：明确指出 VLM UQ 的核心挑战在于同时建模视觉模糊性和概念间文本歧义
2. **交叉注意力设计优雅**：用视觉作 Query 查询所有候选文本，构建图像条件的文本表示，自然支持不同数量的候选概念
3. **二分类优于损失回归**：将 UQ 直接建模为失败预测（二分类）比预测损失值（回归）更有效，因为后者需要知道 VLM 的预训练损失函数
4. **Loss-agnostic 后验设计**：仅需嵌入即可工作，不需要访问 VLM 权重或知道其训练细节，适用于黑盒场景
5. **在低准确率场景仍有效**：MCM 和 BayesVLM 在 VLM 零样本精度低时性能急剧下降（如 EuroSAT 仅 64.1% AUC），但 ViLU 仍保持 90.1% AUC

## 局限性

- 需要针对每个目标数据集训练 ViLU 模型（虽然数据效率较高，2.5% ImageNet 数据即可超越 MCM）
- 跨数据集零样本迁移（在 CC12M 上训练→其他数据集评测）性能仍有提升空间
- 仅在 CLIP ViT-B/32 上做主实验，对更大规模 VLM（如 EVA-CLIP、SigLIP）的适用性有待验证
- 交叉注意力的计算复杂度随候选概念数 $K$ 线性增长

## 相关工作

- **输出分布方法**：MCM, Entropy, Doctor, Rel-U — 不依赖训练但表达能力有限
- **数据驱动预测器**：ConfidNet, LVU — 仅学习视觉不确定性，忽略语言模态
- **VLM 专用 UQ**：ProbVLM（概率嵌入适配器）, BayesVLM（Laplace 近似）— 目标非失败预测

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 5 |
| 实验充分度 | 5 |
| 写作清晰度 | 5 |
| 实用价值 | 4 |
| 总评 | 4.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Joint Vision-Language Social Bias Removal for CLIP](../../CVPR2025/information_retrieval/joint_vision-language_social_bias_removal_for_clip.md)
- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)
- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching](aligning_information_capacity_between_vision_and_language_via_dense_to_sparse_feature_distillation.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)
- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)

</div>

<!-- RELATED:END -->

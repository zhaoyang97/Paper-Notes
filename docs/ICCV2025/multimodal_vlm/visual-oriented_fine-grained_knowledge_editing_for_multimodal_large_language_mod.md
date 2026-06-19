---
title: >-
  [论文解读] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][知识编辑] 提出面向视觉的细粒度多模态知识编辑任务及 FGVEdit 基准，设计 MSCKE 框架通过多模态范围分类器融合视觉与文本信息，实现对图像中多个交互实体的精确知识更新，显著优于纯文本编辑方法。 知识编辑旨在高效、低成本地修正模型中的错误知识。随着多模态大语言模型（MLLM）的…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "知识编辑"
  - "MLLM"
  - "细粒度视觉编辑"
  - "多模态分类器"
  - "FGVEdit"
---

# Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2411.12790](https://arxiv.org/abs/2411.12790)  
**代码**: 未提供  
**领域**: 多模态VLM  
**关键词**: 知识编辑, MLLM, 细粒度视觉编辑, 多模态分类器, FGVEdit  

## 一句话总结

提出面向视觉的细粒度多模态知识编辑任务及 FGVEdit 基准，设计 MSCKE 框架通过多模态范围分类器融合视觉与文本信息，实现对图像中多个交互实体的精确知识更新，显著优于纯文本编辑方法。

## 研究背景与动机

知识编辑旨在高效、低成本地修正模型中的错误知识。随着多模态大语言模型（MLLM）的兴起，知识编辑也需要扩展到多模态场景。然而现有工作存在以下关键问题：

**粗粒度编辑的局限**：现有多模态知识编辑基准（MMEdit、KEBench、MIKE）将整张图像视为单个实体，编辑操作实际上只是简单的文本替换（如将"bird"替换为"kite"），不需要访问视觉模块，因此 LLM 编辑方法在这些设置下表现良好

**忽视多实体交互**：真实场景中一张图像通常包含多个交互实体，需要编辑与特定实体相关的知识而不影响其他实体

**视觉信息的缺失**：仅依赖文本语义无法捕捉同一图像中不同实体之间的细微关系

如图 1 所示，粗粒度编辑中更改"bird"为"kite"只需文本操作；而细粒度编辑需要在同一图像中精确定位目标实体（如其中一只风筝）并更新其关联知识，同时保持其他实体不受影响。

## 方法详解

### 问题形式化

给定预训练 MLLM $f_\theta$，编辑样本表示为 $(i_e, t_e, y_e)$，其中 $i_e$ 是包含多个实体的图像，$t_e$ 是文本提示，$y_e$ 是期望输出。知识编辑操作 $\mathcal{E}$ 更新参数为 $\theta_e = \mathcal{E}(\theta, i_e, t_e, y_e)$，使得 $y_e = f_{\theta_e}(i_e, t_e)$。

核心挑战在于**细粒度编辑范围**的定义：对于编辑样本存在一个由视觉和文本信息共同决定的编辑范围 $S(i_e, t_e, y_e)$：
- **范围内输入**：需产生修正后的输出
- **范围外输入**（同一图像但指向不同实体）：需保持原始输出不变

### MSCKE 框架

基于 SERAC 方法改进，MSCKE 包含四个核心组件：

1. **多模态编辑记忆**（Multimodal Edit Memory）：存储编辑样本，不修改基础模型参数
2. **多模态范围分类器**（Multimodal Scope Classifier）：评估输入与编辑样本的相关性
3. **基础多模态模型** $f_{\text{base}}$：参数冻结，处理范围外输入
4. **反事实多模态模型** $f_{\text{cfr}}$：可训练参数，处理范围内输入

推理时的决策逻辑：

$$y_{\text{test}} = \begin{cases} f_{\text{base}}(i_{\text{test}}, t_{\text{test}}), & \rho < 0.5 \\ f_{\text{cfr}}(t_e, y_e, i_{\text{test}}, t_{\text{test}}), & \rho \geq 0.5 \end{cases}$$

其中 $\rho$ 是多模态范围分类器计算的相似度得分。

### 多模态范围分类器

这是 MSCKE 的核心创新。与 SERAC 的纯文本分类器不同，该分类器融合视觉和文本两种模态信息：

**特征提取与对齐**：使用预训练 CLIP 模型，分别提取图像和文本特征并映射到统一空间：

$$h_e^i = A_i(E_i(i_e)), \quad h_e^t = A_t(E_t(t_e, y_e))$$

$$h_{\text{test}}^i = A_i(E_i(i_{\text{test}})), \quad h_{\text{test}}^t = A_t(E_t(t_{\text{test}}))$$

**特征融合**：使用点积注意力（dot-product attention）融合视觉和文本特征：

$$z_e = \text{Fusion}(h_e^i, h_e^t), \quad z_{\text{test}} = \text{Fusion}(h_{\text{test}}^i, h_{\text{test}}^t)$$

**相似度计算**：

$$\rho = \text{Sim}(z_e, z_{\text{test}})$$

### 损失函数

分类器作为二元分类器训练，使用二元交叉熵损失：

$$\mathcal{L}_{cls} = -\frac{1}{N} \sum_{k=1}^{N} [\log(\rho_{\text{in}}^k) + \log(1 - \rho_{\text{out}}^k)]$$

其中 $\rho_{\text{in}}^k$ 和 $\rho_{\text{out}}^k$ 分别是范围内和范围外样本的相似度得分。

### FGVEdit 基准构建

基于 VQAv2 构建，包含 11,112 个样本（训练:测试 = 3:1）：

- **Specificity 数据**：利用 GPT-4o-mini 进行两阶段分类——先基于图像+文本判断逻辑蕴含关系，再仅基于文本过滤出"难"样本
- **Locality 数据**：使用 NQ 数据集（不相关问答对）
- **Generality 数据**：GPT-4o-mini 改写问题

### 评估指标

新提出的 **Specificity** 指标包含两个组成：

$$M_{\text{specificity}} = \frac{1}{2}[M_{\text{in}}^v + M_{\text{out}}^v]$$

- $M_{\text{in}}^v$：视觉范围内问题的正确回答率（应反映编辑后的知识）
- $M_{\text{out}}^v$：视觉范围外问题的原始回答保持率（不应受编辑影响）

## 实验

### 主实验结果（BLIP-2 OPT / MiniGPT-4）

| 方法 | Reliability | Locality | Generality | Specificity |
|------|------|------|------|------|
| FT-LLM | 100.0/93.4 | 76.9/86.3 | 100.0/93.4 | 24.2/35.0 |
| IKE | 99.9/100.0 | 48.5/52.5 | 98.0/98.9 | 20.1/25.3 |
| SERAC | 93.1/99.5 | 99.9/100.0 | 96.8/92.9 | 31.9/37.9 |
| MEND | 97.0/94.9 | 98.6/98.6 | 96.4/94.8 | 65.9/67.4 |
| **MSCKE** | 99.1/99.5 | **100.0/100.0** | 98.6/93.0 | 61.6/57.2 |
| **MSCKE-MEND** | 97.4/97.1 | **100.0/100.0** | 96.5/96.7 | **68.4/72.0** |

**关键发现**：
- 在传统指标上，各方法差异不大；但在关键的 Specificity 指标上，MSCKE（61.6）显著优于 SERAC（31.9），提升接近翻倍
- MSCKE-MEND 进一步将 Specificity 提升至 68.4/72.0

### 消融与分析

| 组件 | CLIP-ViT-B/32 | CLIP-ViT-L/14 |
|------|------|------|
| 拼接融合 | 63.70 | 63.80 |
| 交叉注意力 | 64.45 | 64.35 |
| **点积注意力** | **64.73** | **64.85** |

**关键发现**：
- ViT-B/32 与 ViT-L/14 性能相当，说明轻量骨干已足够
- 点积注意力融合最优且计算开销最小

### 可迁移性实验

| 源 → 目标 | Specificity (迁移/重训) |
|------|------|
| BLIP-2 → BLIP-2 (MSCKE-MEND) | 68.35 / 68.38 |
| BLIP-2 → MiniGPT-4 (MSCKE) | 57.09 / 57.20 |
| BLIP-2 → MiniGPT-4 (MSCKE-MEND) | 72.16 / 71.98 |

分类器在不同模型间迁移后性能几乎无损，展现了优秀的模型无关性。

### 计算成本

| 组件 | 推理时间 | 模型大小 |
|------|------|------|
| 多模态分类器 | 36ms | 0.56G |
| 基础模型 | 121ms | 9.10G |
| 反事实模型 | 85ms | 4.22G |

分类器仅引入极小的额外开销（36ms / 0.56G）。

## 亮点与洞察

1. **问题定义的贡献**：首次明确提出"面向视觉的细粒度知识编辑"这一新任务，揭示了粗粒度编辑基准的不足
2. **多模态范围判别**：纯文本分类器在细粒度场景下失效（因为范围内外样本的文本高度相似），多模态分类器通过引入视觉信息才能正确区分
3. **解耦设计的优势**：分类器、基础模型和反事实模型完全解耦，支持灵活组合和迁移
4. **Specificity 指标**：填补了现有评估体系中对同一图像内多实体编辑效果的评估空白

## 局限性

1. Specificity 最高仅约 72%，仍有较大提升空间
2. 依赖 CLIP 的对齐能力进行跨模态匹配，对 CLIP 难以区分的相似实体可能失效
3. FGVEdit 基准基于 VQAv2 构建，场景多样性可能受原始数据集限制
4. 编辑记忆随编辑样本增多线性增长，可能影响检索效率

## 相关工作

- **LLM 知识编辑**：参数保持方法（SERAC、IKE、MemPrompt）vs 参数修改方法（ROME、MEMIT、MEND）
- **MLLM 知识编辑**：MMEdit 首个多模态基准，KEBench 引入泛化性指标，MIKE 和 MC-MKE 探索细粒度但仍以文本为中心
- **基座模型**：BLIP-2 OPT、MiniGPT-4 作为被编辑的 MLLM

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](../../NeurIPS2025/multimodal_vlm/vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)

</div>

<!-- RELATED:END -->

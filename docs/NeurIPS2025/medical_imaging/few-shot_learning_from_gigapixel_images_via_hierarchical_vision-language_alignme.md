---
title: >-
  [论文解读] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
description: >-
  [NeurIPS 2025][医学图像][多实例学习] 提出 HiVE-MIL，一个层级视觉-语言 MIL 框架，通过构建统一异构图建模跨尺度层级关系（5× 和 20×）和同尺度多模态对齐，配合文本引导的动态过滤机制和层级对比损失，在 TCGA 肺/乳腺/肾癌三个数据集的 16-shot 设置下全面超越已有方法，Macro F1 最高提升 4.1%。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "多实例学习"
  - "视觉语言模型"
  - "层级图"
  - "全切片图像"
  - "少样本分类"
  - "病理学"
---

# Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2505.17982](https://arxiv.org/abs/2505.17982)  
**代码**: [GitHub](https://github.com/bryanwong17/HiVE-MIL)  
**领域**: 计算病理学 / 少样本学习  
**关键词**: 多实例学习, 视觉语言模型, 层级图, 全切片图像, 少样本分类, 病理学

## 一句话总结

提出 HiVE-MIL，一个层级视觉-语言 MIL 框架，通过构建统一异构图建模跨尺度层级关系（5× 和 20×）和同尺度多模态对齐，配合文本引导的动态过滤机制和层级对比损失，在 TCGA 肺/乳腺/肾癌三个数据集的 16-shot 设置下全面超越已有方法，Macro F1 最高提升 4.1%。

## 研究背景与动机

**全切片图像分类**：WSI 具有千兆像素分辨率，包含从粗粒度组织结构到细粒度细胞形态的多尺度空间信息，标注稀缺导致必须使用弱监督 MIL 框架

**传统 MIL 的不足**：依赖大量标注数据、仅使用视觉特征，对染色变异和领域偏移敏感，在少样本场景下效果差

**VLM-MIL 的进展与瓶颈**：近期多尺度 VLM-MIL 方法引入了尺度特定提示，但存在两个关键缺陷：
   - **跨尺度同模态交互建模不足**：各尺度独立处理视觉/文本特征，仅在最终预测阶段简单求和/求均，丢失了从粗到细的语义层级结构
   - **同尺度跨模态对齐不充分**：未充分探索同一尺度上视觉与文本特征的精细对齐

## 方法详解

### 1. 多尺度层级特征提取

**视觉层级**：将 WSI 在低尺度（5×）提取 $N$ 个 patch $z_n^{(l)} = f_{\text{img}}(x_n^{(l)}) \in \mathbb{R}^D$；每个低尺度 patch 细分为 $M = (20/5)^2 = 16$ 个高尺度（20×）子 patch $z_r^{(h)} = f_{\text{img}}(x_{n,m}^{(h)})$。

**文本层级**：用 GPT-4o 生成层级文本——每类 $O=4$ 条低尺度描述（粗粒度组织特征如"腺泡模式"），每条低尺度描述对应 $K=3$ 条高尺度子描述（细粒度细胞特征如"核深染"）。采用 CoOp 的 $L=16$ 个可学习 token 前缀：

$$t_o^{(l)} = [v_1^{(l)}] \dots [v_L^{(l)}] [\text{Low-scale Text}_o]$$

### 2. 文本引导动态过滤（TGDF）

**阶段一（低尺度过滤）**：计算 patch-text 余弦相似度矩阵 $S^{(l)} \in \mathbb{R}^{N \times O}$，对每个文本 $o$ 计算均值 $\mu_o$ 和标准差 $\sigma_o$，文本自适应阈值过滤：

$$S_{\text{filtered}}^{(l)}(n,o) = \mathbb{I}\left(S^{(l)}(n,o) \geq \mu_o + \alpha \cdot \sigma_o\right)$$

**阶段二（高尺度细化）**：对保留的低尺度 patch 对应的高尺度子 patch 执行同样过滤，并用低尺度过滤结果做掩码以保持一致性：$S_{\text{masked}}^{(h)}(r,s) = S^{(h)}(r,s) \cdot S_{\text{filtered}}^{(l)}(n,o)$

### 3. 层级异构图（HHG）

图节点类型 $\mathcal{T} = \{\text{img}^{(l)}, \text{img}^{(h)}, \text{text}^{(l)}, \text{text}^{(h)}\}$

**同尺度边**（intra-scale）：基于 TGDF 过滤后的相似度矩阵连接同一尺度的视觉-文本节点对

**层级边**（hierarchical）：连接低尺度和高尺度的同模态节点（视觉层级 + 文本层级），基于空间父子关系

### 4. 模态-尺度注意力（MSA）

对层级边引入注意力机制，增强跨尺度信息传播。节点特征加入可学习的尺度嵌入，通过关系特定投影计算 QKV：

$$\beta_{vu} = \text{softmax}\left(\frac{q_v^\top k_u}{\sqrt{d}}\right), \quad h_v^{\text{hier}} = q_v + \sum_{u \in \mathcal{N}_r(v)} \beta_{vu} v_u$$

### 5. 训练目标

**层级文本对比损失（HTCL）**：对齐跨尺度文本语义，正样本为同类父子文本对，负样本为异类：

$$\mathcal{L}_{\text{HTCL}} = \frac{1}{N}\sum_{i=1}^{N}\left(-\frac{1}{|\mathcal{P}_s|}\sum_{j \in \mathcal{P}_s}\log\sigma(\text{sim}_{o,s}) - \frac{1}{|\mathcal{N}_s|}\sum_{j \in \mathcal{N}_s}\log\sigma(-\text{sim}_{o,s})\right)$$

**总损失**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_i, y_i) + \lambda \mathcal{L}_{\text{HTCL}}$，$\lambda = 0.5$

## 实验关键数据

### 16-shot 结果（PLIP 特征编码器）

| 方法 | NSCLC ACC | NSCLC F1 | BRCA ACC | BRCA F1 | RCC ACC | RCC F1 |
|------|----------|---------|---------|--------|--------|-------|
| ABMIL | 70.64 | 70.37 | 65.83 | 65.29 | 80.00 | 77.95 |
| TransMIL | 73.21 | 72.98 | 72.08 | 71.94 | 87.05 | 84.96 |
| MSCPT | 76.86 | 76.82 | 72.71 | 72.58 | 86.21 | 84.20 |
| ViLa-MIL | 74.17 | 73.90 | 71.04 | 70.56 | 85.06 | 82.51 |
| FOCUS | 71.73 | 71.65 | 71.66 | 71.36 | 87.82 | 85.54 |
| **HiVE-MIL** | **80.13** | **80.08** | **75.21** | **74.99** | **88.89** | **87.18** |
| Δ vs 2nd | +3.27 | +3.26 | +2.50 | +2.41 | +1.07 | +1.64 |

### 跨 VLM 编码器一致性

在 PLIP（208K 图文对）、QuiltNet（1M 图文对）、CONCH 三个病理 VLM 上均保持最优。

### 消融实验关键发现

- 去掉 TGDF → F1 下降 1.5-3%，说明过滤弱匹配对的必要性
- 去掉层级边 → 性能退化显著，验证跨尺度建模的重要性
- 去掉 HTCL → 文本语义跨尺度一致性降低

## 亮点

1. ⭐⭐⭐ **统一层级异构图**：首次同时建模 WSI 的跨尺度同模态层级交互和同尺度跨模态对齐，比简单融合获得显著提升
2. ⭐⭐⭐ **文本引导动态过滤**：自顶向下的两阶段过滤机制有效去除无关 patch-text 配对，且随训练动态更新
3. ⭐⭐ **一致的大幅领先**：在 3 个数据集 × 3 个 VLM × 多个 shot 设置下全面超越所有基线
4. ⭐⭐ **方法设计合理**：每个组件（MSA, TGDF, HTCL）都有充分消融验证

## 局限与展望

1. **计算开销**：构建和处理层级异构图（尤其 16× 高尺度 patch 扩展）会带来显著的内存和计算成本
2. **文本依赖 GPT-4o**：层级文本描述质量依赖 LLM 生成，对不同癌种的泛化性需要更多验证
3. **仅二/三分类**：实验仅覆盖 2-3 类分类，未验证在更细粒度亚型分类上的效果
4. **阈值超参 $\alpha$ 敏感度**：TGDF 的过滤阈值 $\alpha=0.5$ 是否对所有数据集/VLM 都最优未充分探讨

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](../../ICML2025/medical_imaging/context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](../../ICCV2025/medical_imaging/gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)

</div>

<!-- RELATED:END -->

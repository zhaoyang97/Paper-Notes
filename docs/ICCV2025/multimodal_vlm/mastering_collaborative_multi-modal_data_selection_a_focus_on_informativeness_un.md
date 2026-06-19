---
title: >-
  [论文解读] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness
description: >-
  [ICCV 2025][多模态VLM][数据选择] 提出 DataTailor——基于信息性（informativeness）、唯一性（uniqueness）和代表性（representativeness）三大原则的协同多模态数据选择框架，仅用 15% 数据即可达到全量数据微调 101.3% 的性能，充分体现"Less is More"理念。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "数据选择"
  - "指令微调"
  - "多模态大语言模型"
  - "信息性"
  - "唯一性"
  - "代表性"
---

# Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness

**会议**: ICCV 2025  
**arXiv**: [2412.06293](https://arxiv.org/abs/2412.06293)  
**代码**: 有（论文中提供 URL）  
**领域**: 多模态VLM  
**关键词**: 数据选择, 指令微调, 多模态大语言模型, 信息性, 唯一性, 代表性

## 一句话总结

提出 DataTailor——基于信息性（informativeness）、唯一性（uniqueness）和代表性（representativeness）三大原则的协同多模态数据选择框架，仅用 15% 数据即可达到全量数据微调 101.3% 的性能，充分体现"Less is More"理念。

## 研究背景与动机

MLLM 的指令微调（instruction tuning）对于增强模型的指令遵循能力至关重要，但随着视觉指令数据集的快速扩张，出现了严重的**数据冗余**问题：

**计算成本增长**：大规模但低质量的指令数据使微调过程极为耗时

**现有数据选择方法的不足**：
   - **数据特定方法**（如 IFD）：依赖大量人工设计规则，灵活性和鲁棒性不足
   - **人工反馈方法**（如 InsTag）：耗时且昂贵
   - **基于梯度的方法**（如 LESS、TIVE）：需要在下游任务上额外训练，总计算成本高

**忽视样本关系**：现有方法大多只评估单个样本的价值，忽略了样本之间的相似性和噪声关系

作者提出从系统性角度审视数据选择问题：一个有价值的样本应同时满足三个条件——包含丰富任务信息（信息性）、与其他样本具有差异性（唯一性）、能代表整体数据分布而非异常值（代表性）。

## 方法详解

### 整体框架

DataTailor 包含四个步骤：(1) 计算每个样本的信息价值；(2) 在簇内空间计算唯一价值；(3) 在簇间空间计算代表价值；(4) 自适应地整合三种价值进行协同数据选择。

### 关键设计

1. **信息价值估计（Informative Value）**：
   基于信息论和谱分析，利用 SVD 分解来衡量样本的信息密度。对于样本 $s_i$，从倒数第二层提取统一特征矩阵 $\mathbf{M_i} \in \mathbb{R}^{L_i \times d}$，进行 SVD 分解得到奇异值 $\{\sigma_j\}_{j=1}^{L_i}$，计算归一化奇异值的熵作为信息价值：
    $V_i^{Inf} = -\sum_{j=1}^{L_i} \frac{\sigma_j}{\sum_k \sigma_k} \log \frac{\sigma_j}{\sum_k \sigma_k}$
   直观理解：简单样本的特征矩阵列向量线性相关性强，少数奇异值主导，熵低；困难样本信息更丰富，奇异值分布更均匀，熵高。

2. **唯一价值估计（Unique Value）**：
   首先通过**跨模态领域聚类**（Cross-modal Domain Clustering）将相似样本聚合到同一簇中，使用 Ward 准则进行层次聚类，合并准则为：
    $\Delta \text{SSE} = \frac{n_A \cdot n_B}{n_A + n_B} \cdot \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|_2$
   当 $\Delta\text{SSE}$ 超过阈值 $\lambda \cdot \Delta\text{SSE}_{\max}$ 时停止（$\lambda=0.1$）。然后在簇内空间计算唯一价值——与簇内其他样本距离越大，唯一性越高：
    $V_i^{Uni} = \sum_{s_j \in \mathbf{C}, j \neq i} \|\mathbf{p_j} - \mathbf{p_i}\|_2 \cdot \frac{V_j^{Inf}}{\sum_{k \in \mathbf{C}} V_k^{Inf}}$
   信息价值高的样本在距离计算中获得更大权重。

3. **代表价值估计（Representative Value）**：
   在簇间空间评估样本所在簇与其他簇的关联度，防止选择孤立的噪声簇中的异常样本：
    $\tau_i^c = \frac{1}{K-1} \sum_{k \neq c}^{K} \exp(\text{sim}(\overline{\mathbf{p_k}}, \overline{\mathbf{p_c}}))$
    $V_i^{Rep} = \tau_i^c \cdot \frac{V_i^{Inf}}{\sum_{k \in \mathbf{C}} V_k^{Inf}}$
   使用最后一个 token 的特征作为样本表示（因其通过交叉注意力聚合了所有视觉和文本特征）。

4. **自适应协同选择**：
   根据指令的对话轮次自适应调整三种价值的权重：
    $V_i = \frac{r_i}{r_i + 2} \cdot V_i^{Inf} + \frac{1}{r_i + 2} \cdot (V_i^{Uni} + V_i^{Rep})$
   多轮指令（$r_i$ 大）更侧重信息价值，单轮指令更侧重唯一性和代表性。

   此外，基于每个任务的最大奇异值比例自适应确定各任务的数据选择比例 $k_p$：
    $k_p = \frac{x_p^2 \cdot |S_p|}{\sum_q x_q^2 \cdot |S_q|} \cdot k, \quad x_p = \text{avg}\left(\frac{\sigma_{\max}}{\sum_j \sigma_j}\right)$

### 损失函数 / 训练策略

DataTailor 本身是一种数据选择方法，不引入额外的训练损失。选择后的数据直接用于 MLLM 的 LoRA 微调。

## 实验关键数据

### 主实验（LLaVA-v1.5-7B on LLaVA-mix-665k）

| 方法 | 数据量 | MME-P | SEED-I | POPE | MM-Vet | SciQA | VQA-v2 | TextVQA | 相对性能 |
|------|--------|-------|--------|------|--------|-------|--------|---------|----------|
| LLaVA-v1.5 (全量) | 665k | 1476.9 | 67.4 | 86.4 | 30.9 | 70.0 | 79.1 | 58.2 | 100.0% |
| Random | 50k | 1387.5 | 59.7 | 85.7 | 29.5 | 70.0 | 73.7 | 53.1 | 95.3% |
| IFD | 50k | 1113.4 | 55.1 | 76.7 | 27.6 | 48.2 | 64.2 | 43.6 | 87.3% |
| TIVE | 50k | 1334.8 | 62.2 | 85.9 | 30.2 | 71.4 | 73.8 | 51.1 | 94.6% |
| COINCIDE | 133k | 1495.6 | - | 86.1 | - | 69.2 | 76.5 | 55.6 | 98.0% |
| ICONS | 133k | 1485.7 | - | 87.5 | 29.7 | 70.8 | 76.3 | 55.6 | 98.8% |
| **DataTailor** | **50k** | **1461.2** | **61.7** | **82.1** | **30.4** | **70.9** | **75.0** | **53.1** | **100.1%** |
| **DataTailor** | **100k** | **1476.2** | **63.6** | **85.3** | **31.8** | **71.0** | **76.7** | **55.7** | **101.3%** |

### 消融实验

| 配置 | MME | MMMU(val) | SciQA | 相对性能 |
|------|-----|-----------|-------|----------|
| Full Data (100%) | 1744.8 | 32.8 | 70.0 | 100.0% |
| Random (7.5%) | 1675.0 | 32.2 | 70.0 | 95.3% |
| 仅 $V_i^{Inf}$ | 1759.3 | 34.9 | 70.2 | 98.0% |
| 仅 $V_i^{Uni}$ | 1716.2 | 33.5 | 69.8 | 97.3% |
| 仅 $V_i^{Rep}$ | 1771.4 | 33.8 | 68.5 | 97.5% |
| DataTailor (三者协同) | 1823.7 | 33.9 | 70.9 | 100.1% |
| w/o 自适应协同 | 1770.2 | 34.0 | 70.2 | 98.8% |

### 关键发现

- **数据严重冗余**：随机选择少量数据（7.5%）即可达到 95%+ 性能，部分情况下子集还优于全量
- **三大原则互补**：单独使用任何一个原则都不如三者协同，证明设计的合理性
- **自适应机制有效**：去掉自适应协同或自适应比例都导致性能下降
- **强迁移性**：用 LLaVA-7B 选择的数据可有效迁移到 mPLUG-Owl-7B 和 Bunny-3B
- **"Less is More"**：15% 数据达到 101.3% 性能，真正实现了高质量数据优于大量低质量数据

## 亮点与洞察

- **系统性三原则框架**：首次从信息性、唯一性、代表性三维度系统评估多模态数据，对传统仅关注单一维度的方法形成降维打击
- **SVD 熵度量样本难度**：利用奇异值分布的熵来衡量信息密度，理论优雅且实践有效
- **自适应权重设计**：根据对话轮次和任务难度自适应调整，避免了繁琐的超参数搜索
- **跨模型迁移能力**：代理模型选择的数据可有效用于不同架构的目标模型

## 局限与展望

- 聚类的 $\lambda$ 设置虽然声称自适应，但仍需按总选择比例设定
- 层次聚类在超大规模数据集上的计算效率可能受限
- 仅验证了 LoRA 微调设置，全参数微调的效果未知
- 选择阈值（如 SVD 的具体层选择）可能因模型架构不同而需要调整

## 相关工作与启发

- 与 TIVE/ICONS 等 MLLM 专用方法相比，DataTailor 不需要额外的下游任务训练
- SVD 分析的思路可能启发其他场景的数据质量评估（如预训练数据过滤）
- 三原则框架可推广到文本数据选择和其他模态

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 三原则协同框架新颖且有深度，SVD 熵的使用有理论基础
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖多个数据集、模型、消融、迁移性分析，非常全面
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，图示丰富，但公式密度较高需要仔细消化
- **价值**: ⭐⭐⭐⭐⭐ 对 MLLM 训练效率有重要实用价值，"Less is More" 结论令人信服

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](../../CVPR2026/multimodal_vlm/videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)

</div>

<!-- RELATED:END -->

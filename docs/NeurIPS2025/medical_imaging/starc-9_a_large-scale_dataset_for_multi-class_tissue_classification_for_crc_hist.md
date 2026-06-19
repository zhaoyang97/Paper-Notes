---
title: >-
  [论文解读] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology
description: >-
  [NeurIPS 2025][医学图像][结直肠癌] 提出 STARC-9 大规模结直肠癌组织分类数据集（63 万张图片、9 类组织）及其构建框架 DeepCluster++，通过自编码器特征提取 + K-means 聚类 + 等频分箱采样确保形态多样性，在该数据集上训练的模型显著超越 NCT 和 HMU 训练的模型。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "结直肠癌"
  - "组织分类"
  - "数据集"
  - "组织病理学"
  - "深度聚类"
---

# STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology

**会议**: NeurIPS 2025  
**arXiv**: [2511.00383](https://arxiv.org/abs/2511.00383)  
**代码**: [GitHub](https://github.com/Path2AI/STARC-9)  
**领域**: 医学图像  
**关键词**: 结直肠癌, 组织分类, 数据集, 组织病理学, 深度聚类

## 一句话总结

提出 STARC-9 大规模结直肠癌组织分类数据集（63 万张图片、9 类组织）及其构建框架 DeepCluster++，通过自编码器特征提取 + K-means 聚类 + 等频分箱采样确保形态多样性，在该数据集上训练的模型显著超越 NCT 和 HMU 训练的模型。

## 研究背景与动机

多类别组织分类是计算病理学的基础任务，支持下游应用如组织分割、生物标志物预测和生存分析。然而，现有公开结直肠癌数据集存在三个核心问题：

**形态多样性不足**：样本无法充分代表不同组织类别的广泛外观变化

**类别不平衡**：肿瘤上皮等优势类别远多于粘液或坏死等临床重要类别

**低质量图块**：包含被错误分类的图块和含有伪影的图块，影响模型学习

以最广泛使用的 NCT-CRC-HE-100K 为例，仅 10 万张图，且存在 JPEG 压缩伪影。HMU-GC-HE-30K 包含非代表性（错误分类）图块。TCGA 数据集仅提供全切片图像（WSI），无图块级标注。

**更深层问题**：现有数据集构建方法——手工标注依赖病理学家主观判断且耗时，随机采样容易遗漏罕见形态，传统深度聚类在质心附近过采样导致多样性不足。目前没有一个统一的框架来实现大规模、平衡、多样化的数据集构建。

## 方法详解

### 整体框架

DeepCluster++ 是一个半自动化的数据集构建框架，分三个阶段：（1）在组织病理学图像上训练特定领域自编码器；（2）使用冻结编码器提取特征 → K-means 聚类 → 等频分箱采样确保多样性；（3）病理学家验证。

### 关键设计

1. **领域特定自编码器（AE_CRC）**：在 10 万张 CRC 图块上训练，编码器由 6 层卷积 + BatchNorm + LeakyReLU 组成，生成 32768 维潜向量，使用 SSIM 损失训练：

   选择自编码器而非预训练基础模型（如 UNI、CTransPath）的原因：（a）重建损失驱动的特征保留结构级形态相似性，形成更连贯的聚类；（b）分类/对比学习损失的模型倾向过度分离生物学相关组织，降低类内一致性；（c）自编码器轻量高效，适合处理 63 万张图块。

2. **聚类与多样性采样**：对每张 WSI，使用冻结编码器提取特征 → GAP 降维到 512 维 → PCA 降维到 256 维 → K-means 聚类（每簇 $m=400$ 个样本）。关键创新是**等频分箱（Equal-Frequency Binning）**：

    - 计算每个图块到簇中心的欧氏距离 $d_i = \|v_i - c\|$
    - 按距离排序后分为 $g=5$ 个等频组
    - 每组采样 20% 的图块

   此方法确保从近质心（同质）到簇边缘（多样）都有均匀表示,避免了密集区域过采样。相比等宽分箱，等频分箱保持组内图块数量一致。

3. **高效的簇标注传播**：标注一个种子簇后，利用嵌入空间的局部连续性将标签传播到相邻簇。例如将 cluster_48 标记为 "TUM" 后，发现相邻簇 2、97、53、112 也包含相同组织形态，可批量标注，大幅降低人工成本。

### 损失函数 / 训练策略

自编码器使用 SSIM 损失训练，相比 MSE 损失在验证集上表现更优（SSIM: 0.9262 vs 0.8863, PSNR: 32.48dB vs 28.53dB）。下游模型统一使用 batch size 32, lr=0.0001, weight decay=1e-5, Adam 优化器，10 epochs，数据増强包括翻转、旋转和颜色抖动。

## 实验关键数据

### 主实验（多类别组织分类，7 类公共组织）

在 STANFORD-CRC-HE-VAL-LARGE（54000 图块）上评估：

| 模型 | NCT训练 Acc | HMU训练 Acc | STARC-9训练 Acc | 提升(vs NCT) |
|------|-----------|-----------|---------------|------------|
| ResNet50 | 62.59% | 85.71% | **98.64%** | +36.05% |
| EfficientNet-B7 | 82.47% | 84.45% | **98.80%** | +16.33% |
| ViT-base | 84.25% | 90.29% | **98.09%** | +13.84% |
| DeiT-B | 81.63% | 90.05% | **98.65%** | +17.02% |
| Swin Trans-base | 79.05% | 91.88% | **98.79%** | +19.74% |
| CTransPath | 79.05% | 91.88% | **99.00%** | +19.95% |
| UNI | 80.43% | 91.80% | **98.26%** | +17.83% |

所有模型在 STARC-9 上训练后准确率均超 98%，CTransPath 达到 99%。

### 肿瘤分割实验

| 验证集 | NCT IoU | HMU IoU | STARC-9 IoU | NCT Dice | HMU Dice | STARC-9 Dice |
|--------|---------|---------|-------------|----------|----------|-------------|
| Stanford | 67.19±21.53 | 64.68±24.21 | **89.33±8.76** | 78.20±17.01 | 75.49±21.01 | **90.47±8.14** |
| TCGA-CRC | 51.94±37.94 | 58.89±29.42 | **88.81±10.90** | 58.90±31.38 | 68.85±22.10 | **89.38±9.14** |

STARC-9 训练模型在 Stanford 上 Dice 高约 14%，在 TCGA-CRC 上高约 30%，且标准差更小。

### 消融实验

| 消融项 | 验证方法 | 结果 |
|--------|---------|------|
| SSIM vs MSE 自编码器 | 重建质量 | SSIM 更优（0.9262 vs 0.8863）|
| 码字空间的编码器对比 | 特征可视化 | AE_CRC 聚类最连贯，UNI 过度分离 |
| 等频分箱 vs 等宽分箱 | 多样性 | 等频分箱保证类内异质性 |
| 每簇样本数 | 聚类纯度 | m=400 最优（800 混合组织，100 变异不足）|

### 关键发现

- **数据质量决定性**：在相同模型架构和训练配置下，仅换训练数据（STARC-9 vs NCT/HMU）即可带来 14-36% 的准确率提升
- **从零训练也能达到高精度**：在 STARC-9 上从零训练的 CNN 达到 97.81% 准确率，证明数据质量可减少对预训练的依赖
- **混合组织图块是关键挑战**：STARC-9 训练模型在混合组织图块上达 85% 准确率，远超 HMU (55%) 和 NCT (42%)
- **跨数据集泛化**：在外部 TCGA-CRC 和 IMP-CRS10K 数据集上也保持高性能
- 坏死（NCS）分类精度提升尤为显著（vs NCT 提升 90%+，vs HMU 提升 45%+）

## 亮点与洞察

- **DeepCluster++ 框架通用性强**：虽然以 CRC 为示例，框架可直接应用于其他癌症类型的 WSI 数据集构建
- **等频分箱保证多样性**：相比常用的在聚类中心附近采样，等频分箱从中心到边缘均匀覆盖，提供更全面的形态表示
- **数据 > 模型的有力证据**：所有模型在 STARC-9 上一致优于 NCT/HMU，说明数据质量和多样性对模型性能的决定性影响
- **630K 图块、9 类组织均 70K**：完美类别平衡，消除了类别不平衡对训练的干扰

## 局限与展望

- 来自单一机构（Stanford），人口统计多样性有限（黑人和原住民代表不足）
- 9 类组织可能不完全覆盖所有 CRC 切除标本中的组织类型
- 目前仅限 CRC，需验证 DeepCluster++ 在其他癌种（如 CNS 肿瘤）上的适用性
- AE_CRC 对新数据集可能需要重新训练
- 缺乏图像-文本对，未支持多模态应用

## 相关工作与启发

与 NCT-CRC-HE-100K（10 万图块，JPEG 伪影）和 HMU-GC-HE-30K（3 万图块，错误标签）相比，STARC-9 在规模、质量和多样性上全面领先。与 QuPath 手工标注相比，DeepCluster++ 大幅降低人工成本。启发：在构建大规模数据集时，数据采样策略（等频分箱）和特征提取器选择（重建损失 vs 判别损失）对最终数据质量至关重要。

## 评分

- 新颖性: ⭐⭐⭐⭐ DeepCluster++ 框架有独到设计，等频分箱采样策略实用
- 实验充分度: ⭐⭐⭐⭐⭐ 21 个模型基准测试，多验证集，分类+分割任务，对比极为充分
- 写作质量: ⭐⭐⭐⭐ 框架清晰，实验详尽，但篇幅偏长
- 价值: ⭐⭐⭐⭐⭐ 数据集、框架、基准模型全部开源，对计算病理学社区有重大贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [\[NeurIPS 2025\] Exploring and Leveraging Class Vectors for Classifier Editing](exploring_and_leveraging_class_vectors_for_classifier_editing.md)

</div>

<!-- RELATED:END -->

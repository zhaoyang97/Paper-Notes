---
title: >-
  [论文解读] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting
description: >-
  [CVPR 2025][目标检测][零样本目标计数] 提出T2ICount，利用预训练文生图扩散模型的单步去噪特征进行零样本目标计数，通过层次语义校正模块（HSCM）和表征区域一致性损失（$\mathcal{L}_{RRC}$）解决单步去噪的文本敏感性不足问题。 零样本目标计数旨在根据文本描述对图像中任意类别目标进行计数…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "零样本目标计数"
  - "扩散模型"
  - "文本敏感性"
  - "跨模态对齐"
  - "密度估计"
---

# T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting

**会议**: CVPR 2025  
**arXiv**: [2502.20625](https://arxiv.org/abs/2502.20625)  
**代码**: [https://github.com/cha15yq/T2ICount](https://github.com/cha15yq/T2ICount)  
**领域**: 图像生成/零样本计数  
**关键词**: 零样本目标计数, 扩散模型, 文本敏感性, 跨模态对齐, 密度估计

## 一句话总结

提出T2ICount，利用预训练文生图扩散模型的单步去噪特征进行零样本目标计数，通过层次语义校正模块（HSCM）和表征区域一致性损失（$\mathcal{L}_{RRC}$）解决单步去噪的文本敏感性不足问题。

## 研究背景与动机

零样本目标计数旨在根据文本描述对图像中任意类别目标进行计数，无需视觉样例。现有方法主要基于CLIP视觉-语言模型，但存在根本性问题：

1. **文本不敏感**：CLIP图像编码器在全局语义层面工作，自然倾向于关注图像中的多数类目标，当文本指定的是少数类时，模型无法正确响应
2. **数据集偏差**：FSC-147等基准数据集中，标注类别几乎总是图像中的多数类，掩盖了模型的文本敏感性问题
3. **扩散模型的潜力**：文生图扩散模型具有丰富的像素级语义理解能力，天然适合计数这种像素级任务，但多步去噪过程计算代价过高

然而使用单步去噪来提高效率时，文本-视觉对应关系建立不充分，交叉注意力图呈现严重的语义错位——不相关区域被高亮，相关目标上的注意力不一致。

## 方法详解

### 整体框架

T2ICount以Stable Diffusion为基础，输入图像和文本提示经过单步去噪，从U-Net解码器提取多尺度特征图，通过HSCM逐层精炼文本-图像对齐，最终生成密度图进行计数。

### 关键设计

**设计一：层次语义校正模块（Hierarchical Semantic Correction Module, HSCM）**

- **功能**：弥补单步去噪导致的文本-图像交互不足
- **核心思路**：三阶段级联设计。每阶段先融合相邻尺度特征 $F_i' = \text{Conv}(\text{Concat}(\text{Up}(V_{i+1}), F_i))$，然后通过SEM（双向跨模态注意力+文本-图像相似度计算）和SCM（用上一阶段的相似度图引导特征校正）交替精炼
- **设计动机**：单步去噪的跨模态对齐太弱，需要额外的多阶段校正。SEM学习生成类似分割掩码的文本-图像相似度图 $S_i$，SCM用 $V_{i+1} \odot S_{i+1}$ 将注意力重定向到文本相关区域

$$S_i = \frac{V_i \cdot c'}{\|V_i\| \|c'\|}$$

**设计二：表征区域一致性损失（Representational Regional Coherence Loss, $\mathcal{L}_{RRC}$）**

- **功能**：利用扩散模型的交叉注意力图生成可靠的正负样本监督信号
- **核心思路**：虽然单步注意力图对特定类别不敏感，但它能有效捕获整体前景区域。融合多尺度交叉注意力图 $\mathcal{A}^{cross}$，结合ground-truth密度图，生成正-负-模糊（PNA）三元掩码：密度高的为正、注意力低的为负、其余为模糊（不施加约束）
- **设计动机**：计数数据集只有点标注，传统方法用密度阈值区分前景/背景，但会将大量前景区域误判为背景。PNA掩码通过注意力图识别背景，避免了对前景的误判

$$p_{jk} = \begin{cases} 1, & \text{if } D_{jk}^{gt} \geq \tau \\ 0, & \text{if } \mathcal{A}_{jk}^{cross} \leq \theta \\ -1, & \text{otherwise} \end{cases}$$

**设计三：FSC-147-S评估基准**

- **功能**：提供更严格的文本引导计数评估协议
- **核心思路**：从FSC-147中筛选并重标注图像，使文本指定的类别与多数类不同，专门测试模型对少数类的计数能力
- **设计动机**：现有基准中标注类几乎总是多数类，模型即使完全忽略文本也能获得不错分数，无法真正评估文本敏感性

### 损失函数

总损失由回归损失和区域一致性损失组成：

$$\mathcal{L} = \mathcal{L}_{reg} + \gamma \mathcal{L}_{RRC}$$

其中 $\mathcal{L}_{RRC} = \lambda \mathcal{L}_{pos} + \mathcal{L}_{neg}$，正样本损失拉近相似度到1，负样本损失用hinge推远至0以下。

## 实验关键数据

### FSC-147测试集

| 方法 | 类型 | MAE↓ | RMSE↓ |
|------|------|------|-------|
| FamNet (3-shot) | Few-shot | 22.56 | 101.54 |
| BMNet (3-shot) | Few-shot | 14.62 | 91.83 |
| CLIP-Count | Zero-shot | 17.78 | 112.09 |
| VLCounter | Zero-shot | 17.05 | 106.16 |
| **T2ICount** | **Zero-shot** | **15.89** | **94.32** |

### FSC-147-S（文本敏感性评估）

| 方法 | MAE↓ | RMSE↓ |
|------|------|-------|
| CLIP-Count | 32.41 | 48.75 |
| VLCounter | 29.83 | 45.12 |
| **T2ICount** | **18.56** | **31.27** |

### 消融实验

| 组件 | Val MAE | Test MAE |
|------|---------|----------|
| Baseline (单步U-Net特征) | 22.15 | 21.35 |
| + HSCM | 18.42 | 17.89 |
| + $\mathcal{L}_{RRC}$ | 16.73 | 16.41 |
| + HSCM + $\mathcal{L}_{RRC}$ (完整) | **15.56** | **15.89** |

### 关键发现

1. T2ICount作为零样本方法，性能接近甚至超越部分few-shot方法（如FamNet 3-shot）
2. 在FSC-147-S上，T2ICount的MAE比CLIP-Count降低约43%，证明了显著的文本敏感性优势
3. HSCM和 $\mathcal{L}_{RRC}$ 各贡献约3-4 MAE的提升，两者互补
4. 交叉注意力图虽然对特定类别不敏感，但作为前景检测器效果良好——这一观察非常新颖

## 亮点与洞察

1. **发现并解决文本不敏感问题**：指出了零样本计数领域一个被忽视的根本性问题，并通过新方法和新评估协议双管齐下解决
2. **扩散模型注意力图的创造性利用**：虽然单步注意力图类别敏感性差，但用作前景检测器生成监督信号非常巧妙
3. **PNA三元掩码设计**：优雅地解决了点标注场景下正负样本划分的难题，避免了将前景误判为背景

## 局限与展望

1. 依赖预训练Stable Diffusion，模型体量较大，推理速度仍受限于U-Net的前向传播
2. 单步去噪的信息量有限，多步去噪+蒸馏可能进一步提升性能
3. FSC-147-S规模有限，需要更大规模的文本敏感性评估基准
4. 未探索更新的扩散模型（如SDXL、SD3）是否能提供更好的特征

## 相关工作与启发

- **CLIP-Count/VLCounter**：基于CLIP的零样本计数先驱，本文揭示了其文本不敏感的根本缺陷
- **CounTR/LOCA**：few-shot计数方法，T2ICount在零样本设定下接近其性能令人印象深刻
- **DiffusionDet**：扩散模型用于检测的探索，启发了将扩散特征用于计数的思路
- 启发：扩散模型的中间特征在多种视觉任务中展现了强大能力，值得在更多任务中探索

## 评分

⭐⭐⭐⭐ — 问题定位准确（文本不敏感），解决方案creative（PNA掩码+HSCM），实验设计周全（新建FSC-147-S）。作为将扩散模型引入零样本计数的工作，做出了实质性贡献。不足在于模型效率和FSC-147-S规模较小。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2025\] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/object_detection/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](../../NeurIPS2025/object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/object_detection/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->

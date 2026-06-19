---
title: >-
  [论文解读] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction
description: >-
  [ICCV 2025][语义分割][半监督分割] 提出ConformalSAM框架，利用Conformal Prediction校准基础分割模型SEEM在目标域的输出不确定性，筛除不可靠像素标签后作为未标注数据的监督信号，配合后期自依赖训练策略，在PASCAL VOC上1/16标注设定下达到81.21 mIoU。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "半监督分割"
  - "共形预测"
  - "SAM/SEEM"
  - "不确定性校准"
  - "伪标签"
---

# ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction

**会议**: ICCV 2025  
**arXiv**: [2507.15803](https://arxiv.org/abs/2507.15803)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 半监督分割, 共形预测, SAM/SEEM, 不确定性校准, 伪标签

## 一句话总结

提出ConformalSAM框架，利用Conformal Prediction校准基础分割模型SEEM在目标域的输出不确定性，筛除不可靠像素标签后作为未标注数据的监督信号，配合后期自依赖训练策略，在PASCAL VOC上1/16标注设定下达到81.21 mIoU。

## 研究背景与动机

半监督语义分割（SSSS）的核心挑战是如何充分利用大量未标注数据。一个自然的想法是用SAM/SEEM等基础分割模型直接为未标注数据生成伪标签——然而实验表明这**反而降低性能**：
- PASCAL VOC 1/16分割：仅用标签50.65 mIoU → 加SEEM伪标签降到42.00
- 原因：SEEM预训练数据与目标域存在域差距，在目标域上预测质量不一致

核心问题：**如何可靠地利用基础模型的强大能力，同时过滤其不可靠预测？**

本文选择Conformal Prediction (CP)作为不确定性校准工具，因为：(1) CP是黑盒方法，只需少量标注数据即可校准；(2) 提供理论保证的覆盖率；(3) 不需要修改基础模型。

## 方法详解

### 整体框架

ConformalSAM采用两阶段训练：
- **Stage I**：用CP校准后的SEEM伪标签 + 真实标签联合训练
- **Stage II**：丢弃SEEM伪标签，切换到自依赖（Self-Reliance）训练

### 关键设计

1. **CP校准的基础模型推理（Stage I）**：

    - **校准过程**：用标注数据 $D_l$ 作为校准集
        - 对每张标注图用SEEM生成概率图 $P_i \in \mathbb{R}^{K \times H \times W}$
        - 计算非一致性分数：$\hat{P}_i^j(a,b) = 1 - P_i^j(a,b)$（仅对真实类别的像素）
        - 汇总所有图像的非一致性分数，计算 $(1-\alpha)$ 分位数阈值 $\hat{q}_\alpha$
    - **校准推理**：对未标注图 $x_i$，像素 $(a,b)$ 的预测集为 $\mathcal{C}_i(a,b) = \{j: \hat{P}_i^j(a,b) \leq \hat{q}_\alpha(a,b)\}$
    - **类别条件过滤**：由于背景像素占主导地位，当背景类和非背景类同时在预测集中时，优先选择非背景类：
    $M_i(a,b) = \begin{cases} \arg\min_j \mathcal{C}_i[j], & |\mathcal{C}_i| > 0 \land 0 \notin \mathcal{C}_i \\ \arg\min_{j \neq 0} \mathcal{C}_i[j], & |\mathcal{C}_i| > 0 \land 0 \in \mathcal{C}_i \\ \text{NaN}, & |\mathcal{C}_i| = 0 \end{cases}$
    - 当预测集为空时，该像素标签设为NaN（忽略），有效滤除低置信度预测
    - 误覆盖率 $\alpha = 0.05$

2. **自依赖训练策略（Stage II）**：

    - 放弃SEEM生成的mask，使用模型自身的伪标签
    - 动态权重衰减策略：$\mathcal{L} = (1 - \lambda(t)) \times \mathcal{L}_s + \lambda(t) \times \mathcal{L}_u$
    - $\lambda(t)$ 指数衰减，使模型后期越来越依赖真实标签监督
    - PASCAL VOC: Stage I 60 epochs, Stage II 20 epochs
    - ADE20K: Stage I 30 epochs, Stage II 10 epochs

3. **灵活的插件式设计**：

    - 可替换Stage II的自训练框架为其他方法如AllSpark
    - ConformalSAM(AllSpark)：Stage I用CP校准伪标签，Stage II切换到AllSpark
    - 体现了框架的通用性和可组合性

### 损失函数 / 训练策略

- 标注数据：标准交叉熵损失
- 未标注数据（Stage I）：NaN像素被忽略，仅对CP筛选后的高置信像素计算CE
- Stage II采用指数衰减权重平衡有监督和无监督损失
- 使用SegFormer-B5作为分割骨干网络

## 实验关键数据

### 主实验

| 方法 | VOC 1/16(92) | VOC 1/8(183) | VOC 1/4(366) | VOC 1/2(732) | VOC Full |
|------|------------|------------|------------|------------|---------|
| UniMatch | 75.2 | 77.2 | 78.8 | 79.9 | - |
| AllSpark | 76.07 | 78.41 | 79.77 | 80.75 | 82.12 |
| **ConformalSAM(AllSpark)** | 80.69 | 81.29 | 81.33 | 82.69 | 83.44 |
| **ConformalSAM** | **81.21** | **82.22** | **81.84** | **83.52** | **83.85** |

| 方法 | ADE20K 1/128(158) | 1/64(316) | 1/32(632) | 1/16(1263) | 1/8(2526) |
|------|-----------------|---------|---------|----------|---------|
| AllSpark | 16.17 | 23.03 | 26.42 | 28.40 | 32.10 |
| **ConformalSAM** | **26.21** | **30.02** | **33.33** | **34.64** | **36.25** |

### 消融实验

| 配置 | SEEM | CP | SR | VOC 1/16 | VOC 1/2 |
|------|------|----|----|---------|---------|
| Semi-Baseline | ✗ | ✗ | ✗ | 52.89 | 74.22 |
| +SEEM直接用 | ✓ | ✗ | ✗ | 42.00 | 44.99 |
| +SEEM+CP | ✓ | ✓ | ✗ | 78.09 | 79.10 |
| +SEEM+CP+SR | ✓ | ✓ | ✓ | **81.21** | **83.52** |

| CP变体 | α=0.1 | α=0.05 | α=0.01 |
|--------|-------|--------|--------|
| Pixel-wise | 74.31 | **78.09** | 68.01 |
| Image-wise | 75.99 | 75.54 | 44.59 |
| K-Means | 69.36 | 69.13 | 44.16 |

### 关键发现

- **CP的关键作用**：SEEM直接用降低8.65 mIoU，加CP后提升25.2 mIoU（1/16设定）
- 类别条件过滤至关重要：对比vanilla CP，平均提升34.11 mIoU
- Pixel-wise CP优于Image/K-Means/GenAnn等其他CP变体
- $\alpha=0.05$ 是一致的最优误覆盖率
- SR策略平均再带来3.76 mIoU的提升
- ADE20K上1/128设定下提升高达10.04 mIoU（vs AllSpark 16.17→26.21）
- 作为插件整合到AllSpark，平均提升2.07 mIoU

## 亮点与洞察

- **首次将CP用于校准分割基础模型在SSSS中的伪标签**，思路简洁且实验验证有力
- 类别条件过滤解决了背景像素压倒前景的问题——这是SEEM在分割任务中的核心失败模式
- 两阶段策略的设计逻辑清晰：早期利用基础模型知识，后期避免过拟合SEEM噪声
- 作为插件框架的通用性：可与AllSpark等SSSS方法自由组合

## 局限与展望

- 效果依赖于基础模型知识与目标任务的重叠度——ADE20K/Cityscapes等含新类别的数据集收益较小
- CP校准需要标注数据，在极少标注场景（几十张）校准精度可能不足
- 仅用SEEM一种基础模型，未充分探索SAM2、GLAMM等更强模型的潜力
- SR训练的切换时机目前仅凭经验确定（60 epochs），对不同数据集可能需要调整
- 未与prompt-engineering类SAM方法深入对比

## 相关工作与启发

- **UniMatch/AllSpark**：当前SSSS SOTA方法，ConformalSAM与它们互补
- **SemiSAM/CPC-SAM**：通过改进prompt利用SAM，而本文直接用SEEM输出+CP校准
- **Conformal Prediction**：从分类/检测领域引入到分割领域的的不确定性校准工具
- CP在其他foundation model（如LLM）输出校准中有广阔应用前景

## 评分

- 新颖性: ⭐⭐⭐⭐ CP用于分割基础模型校准的思路新颖，但两阶段训练本身较朴素
- 实验充分度: ⭐⭐⭐⭐⭐ VOC/VOC-aug/ADE20K三个数据集，插件验证，CP变体消融全面
- 写作质量: ⭐⭐⭐⭐ 动机清楚，消融设计好，但方法部分公式较多
- 价值: ⭐⭐⭐⭐ 展示了如何安全利用基础模型辅助downstream训练的通用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](../../CVPR2026/segmentation/from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)
- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](../../ECCV2024/segmentation/lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)

</div>

<!-- RELATED:END -->

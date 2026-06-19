---
title: >-
  [论文解读] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model
description: >-
  [ICCV 2025][多模态VLM][零样本学习] 提出 LaZSL，通过最优传输（Optimal Transport）实现局部视觉区域与语义属性之间的细粒度对齐，在无需额外训练的前提下构建可解释的零样本分类器，在9个数据集上取得了兼顾准确性、可解释性和域泛化的优异表现。 现有CLIP类方法的局限：- CLIP 通过计算整…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "零样本学习"
  - "可解释性"
  - "最优传输"
  - "局部对齐"
  - "CLIP"
  - "属性"
---

# Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model

**会议**: ICCV 2025  
**代码**: [https://github.com/shiming-chen/LaZSL](https://github.com/shiming-chen/LaZSL)  
**领域**: 多模态VLM  
**关键词**: 零样本学习, 可解释性, 最优传输, 局部对齐, CLIP, 属性

## 一句话总结

提出 LaZSL，通过最优传输（Optimal Transport）实现局部视觉区域与语义属性之间的细粒度对齐，在无需额外训练的前提下构建可解释的零样本分类器，在9个数据集上取得了兼顾准确性、可解释性和域泛化的优异表现。

## 研究背景与动机

**现有CLIP类方法的局限：**

- CLIP 通过计算整张图像与类别文本之间的全局相似度进行零样本分类，但这种方式缺乏可解释性——无法解释模型为什么做出某个预测
- 基于属性的可解释方法（如 DCLIP、CuPL）虽然利用 LLM 生成属性描述来构建分类器，但它们仍然是将**整张图像**与属性进行匹配，无法捕捉局部视觉信息与对应属性之间的细粒度关系

**核心挑战：** 如何在预训练 VLM 的冻结网络上，实现局部视觉区域与属性之间的有效对齐？由于VLM的网络参数不能重新设计或训练，传统的注意力机制方法不适用。

**关键洞察：** 利用最优传输理论，将图像裁剪为局部patch集合、类别属性作为语义集合，然后通过OT找到两个分布之间的最优匹配方案。

## 方法详解

### 整体框架

LaZSL包含三个核心模块：(1) 语义和视觉集合构建；(2) 基于OT的局部视觉-语义对齐；(3) 零样本预测。

### 1. 语义集合与视觉集合构建

**语义集合构建：** 利用 GPT-3 为每个类别生成多个属性描述。对于类别 $y$，通过 LLM 生成语义集合：

$$S^y = h(prompt(y)) = \{s_i^y | i=1,...,M\}$$

**视觉集合构建：** 对输入图像进行多尺度随机裁剪，生成局部视觉区域集合：

$$V_r^x = \{v_i^x = P_r(x, \gamma_i \min(W,H)) | i=1,...,N\}$$

其中 $\gamma_i \sim U(\alpha, \beta)$ 控制裁剪尺度，$N$ 通常设为60-90。通过不同的 $\gamma_i$，确保视觉集合包含多种尺度的特征。

### 2. 基于最优传输的局部对齐

将视觉集合和语义集合分别通过CLIP的视觉编码器和文本编码器获得特征表示 $P^x$ 和 $Q^y$。

**Vision Selection（视觉选择）：** 计算每个局部区域与全局图像之间的余弦相似度，以平均相似度 $\delta$ 为阈值，将区域划分为正相关集合 $P_{pos}^l$ 和负相关集合 $P_{neg}^l$，仅保留正相关区域参与OT计算。更新视觉概率分布：

$$r_i^* = \begin{cases} \frac{1}{|P_{pos}^l|} & \text{if } p_i^l \in P_{pos}^l \\ 0 & \text{otherwise} \end{cases}$$

**Region-Global Hybrid Cost（混合代价矩阵）：** 为了避免随机裁剪引入的噪声和知识遗忘问题，将全局视觉信息融入代价矩阵：

$$C_i^* = 1 - (\theta \cdot sim_i + (1-\theta) \cdot p^{g\top} Q^y)$$

其中 $\theta$ 是混合系数（默认0.8），平衡局部和全局特征。

**Sinkhorn算法求解OT：** 通过迭代更新策略矩阵得到最优传输方案 $T$：

$$T = \text{diag}(\mathcal{U}) \mathcal{M} \text{diag}(\mathcal{V}), \quad \mathcal{M} = \exp(-C^*/\lambda)$$

### 3. 零样本预测

利用OT方案与混合相似度矩阵的Frobenius内积计算类别得分：

$$\psi_y = \langle T, sim^* \rangle_F$$

选择得分最高的类别作为预测结果：$y^* = \arg\max_y \psi_y$

### 损失函数

LaZSL 是 **training-free** 方法，不需要额外的损失函数或训练，直接在推理阶段通过OT计算完成分类。

## 实验关键数据

### 主实验：跨数据集零样本分类

| 方法 | ImageNet | CUB | Oxford Pets | Food101 | Place365 | 平均 |
|------|----------|-----|-------------|---------|----------|------|
| CLIP (ViT-B/16) | 66.7 | 56.0 | 88.1 | 88.4 | 39.3 | 67.7 |
| DCLIP | 67.9 | 57.1 | 86.9 | 88.5 | 40.3 | 68.1 |
| CuPL | 69.6 | 56.4 | 91.1 | 89.0 | 39.8 | 69.2 |
| **LaZSL (Ours)** | **69.2** | **60.3** | **87.4** | **89.7** | **42.0** | **69.7** |

- LaZSL 在所有三种 backbone（ViT-B/32, ViT-B/16, ViT-L/14）上均取得最佳**平均**性能
- 在细粒度数据集 CUB 上提升尤为显著（+3.1% over DCLIP on ViT-B/16），因为细粒度分类更依赖局部特征对齐

### 域泛化实验（ImageNet变体）

| 方法 | 需训练 | ImageNet-V2 | ImageNet-R | ImageNet-S | ImageNet-A | 平均 |
|------|--------|-------------|------------|------------|------------|------|
| CoOp | ✔ | 64.2 | 75.2 | 47.9 | 49.7 | 59.3 |
| MaPLe | ✔ | 64.1 | 77.0 | 49.2 | 50.9 | 60.3 |
| ArGue† | ✔ | 64.6 | 76.6 | 48.9 | 50.9 | 60.3 |
| **LaZSL† (Ours)** | **✗** | **63.3** | **75.6** | **48.2** | **56.2** | **60.9** |

- LaZSL **无需任何训练**即超越了需要额外训练的方法（如CoOp, ArGue），在ImageNet-A上领先显著（+5.3% over ArGue）

### 消融实验

| 方法 | ImageNet | CUB | Place365 |
|------|----------|-----|----------|
| Baseline (DCLIP) | 67.9 | 57.8 | 40.3 |
| + OT | 68.5 | 59.0 | 41.6 |
| + OT + VS | 69.0 | 60.0 | 41.8 |
| + OT + Hybrid | 69.0 | 59.3 | 41.9 |
| LaZSL (full) | **69.2** | **60.3** | **42.0** |

- OT局部对齐是最关键的组件，带来了主要的性能提升
- Vision Selection 和 Hybrid Cost 进一步提升了对齐质量

### 关键发现

1. 局部对齐在细粒度数据集上优势最明显（CUB +3.1%）
2. 混合系数 $\theta=0.8$ 取得最佳平衡，过高会导致全局知识遗忘
3. 裁剪尺度 $\alpha=0.6$ 对所有数据集最优
4. 额外时间开销可控（0.07s vs DCLIP 0.015s/样本）

## 亮点与洞察

1. **Training-free的创新设计**：不同于大多数CLIP变体需要额外训练，LaZSL完全在推理阶段完成，仅通过OT实现局部对齐
2. **将OT引入VLM的可解释分类**：巧妙地将图像裁剪区域和属性描述建模为两个离散分布，用OT找到最优匹配
3. **混合代价矩阵**：融合局部和全局信息避免了知识遗忘问题，这是一个简洁有效的设计
4. **可解释性强**：可以清晰展示哪些视觉区域匹配了哪些属性，类似人类的认知过程

## 局限性

1. **依赖LLM生成的属性质量**：如果LLM生成了不相关的属性描述，会影响分类准确性
2. **推理速度较慢**：需要对每张图像进行60-90次随机裁剪并通过CLIP编码器，推理时间约为DCLIP的4.7倍
3. **随机裁剪的不确定性**：基于随机裁剪构建视觉集合，多次推理结果可能不完全一致
4. **在Oxford Pets等数据集上未超越CuPL**：全局特征在某些数据集上可能已经足够

## 相关工作与启发

- **经典ZSL→VLM-based ZSL→可解释ZSL**：清晰的研究脉络，从人工标注属性到LLM生成属性
- **OT在视觉中的应用**：之前主要用于prompt tuning（PLOT），本文首次用于局部特征对齐
- **启发**：能否将此思路扩展到其他需要局部对齐的任务？如开放词汇检测、细粒度检索

## 评分

- **新颖性**: ⭐⭐⭐⭐ — OT用于局部视觉-语义对齐的想法新颖且well-motivated
- **实验**: ⭐⭐⭐⭐ — 9个数据集、3种backbone、充分的消融实验
- **写作**: ⭐⭐⭐⭐ — 逻辑清晰，图表直观
- **价值**: ⭐⭐⭐⭐ — Training-free + 可解释性，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](../../CVPR2025/multimodal_vlm/rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](../../NeurIPS2025/multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning
description: >-
  [ACL 2025][多模态VLM][句子表示学习] 提出 DALR 框架，通过跨模态一致性学习 + 模态内排序蒸馏的双层对齐策略，解决多模态句子表示中的跨模态不对齐偏差（CMB）和模态内语义分歧（ISD）问题，在 STS 和 TR 任务上取得 SOTA。 领域现状：句子表示学习将句子映射为低维向量以保留语义信息…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "句子表示学习"
  - "跨模态对齐"
  - "模态内对齐"
  - "对比学习"
  - "知识蒸馏"
---

# DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning

**会议**: ACL 2025  
**arXiv**: [2506.21096](https://arxiv.org/abs/2506.21096)  
**代码**: [GitHub](https://github.com/Hekang001/DALR)  
**领域**: 多模态VLM  
**关键词**: 句子表示学习, 跨模态对齐, 模态内对齐, 对比学习, 知识蒸馏  

## 一句话总结

提出 DALR 框架，通过跨模态一致性学习 + 模态内排序蒸馏的双层对齐策略，解决多模态句子表示中的跨模态不对齐偏差（CMB）和模态内语义分歧（ISD）问题，在 STS 和 TR 任务上取得 SOTA。

## 研究背景与动机

**领域现状**：句子表示学习将句子映射为低维向量以保留语义信息，广泛用于语义相似度、信息抽取等任务。自 SimCSE 以来，"PLM + 对比学习"范式成为主流。引入视觉信息（如 MCSE、KDMCSE）被证明能提供丰富的监督信号。

**现有痛点**：现有多模态句子表示方法在粗粒度对齐图像和文本时面临两个关键挑战：

   - **跨模态不对齐偏差（CMB）**：文本信息密集且有选择性，聚焦关键细节；图像则无差别捕获所有内容，存在大量冗余。例如图像中"船"只占小区域，但大多视觉 patch 包含无关信息。加之标注者认知偏差导致同一图像有多种语义描述，进一步放大模态异质性。
   
   - **模态内语义分歧（ISD）**：因共同引用同一图像，语义上差异很大的文本可能被错误识别为高度相似。例如对同一足球图像，"穿蓝白球衣的男孩踢球"和"一群人在看足球队员"描述重点完全不同，但都与图像高度相似，导致假阴性。

**核心矛盾**：KDMCSE 等方法通过阈值过滤假阴性来缓解问题，但这种硬阈值方式仍会误分类——高图文相似度不等于跨文本语义一致性。

**本文目标**：同时解决跨模态对齐中的 CMB 和模态内一致性中的 ISD，实现细粒度的双层对齐。

**切入角度**：跨模态层面引入辅助一致性任务生成软标签引导对齐；模态内层面利用多教师排序蒸馏捕获连续的语义排序结构。

**核心 idea**：句子间的关系不是简单的正负二元标签，而是连续的排序结构，需要跨模态+模态内双层对齐来充分利用视觉信号。

## 方法详解

### 整体框架

DALR 由三个模块组成：
1. **多模态对比学习模块**（基础）：利用视觉信息引导句子表示学习
2. **跨模态对齐模块**：通过辅助一致性任务细化图文语义匹配
3. **模态内对齐模块**：通过排序蒸馏 + KL 散度增强文本内部一致性

### 关键设计

#### 跨模态一致性学习

构造包含匹配和不匹配图文对的数据集 $\mathcal{D}'$（通过打乱图像生成不匹配对）。使用余弦嵌入损失进行二分类：

$$\mathcal{L}_{cons} = \begin{cases} 1 - \cos(h_s^{v'}, s_s^{z'}) & \text{if } y' = 1 \\ \max(0, \cos(h_s^{v'}, s_s^{z'}) - m) & \text{if } y' = 0 \end{cases}$$

其中 $m = 0.2$ 为负样本间距。该任务能捕获更深层的语义关系，与对比学习并行训练，生成跨模态软标签。

#### 跨模态对齐

计算学生模型的文本-视觉分布 $P_i^{t2v}$ 和教师模型的文本-文本 $Q_i^{t2t}$、视觉-视觉 $Q_i^{v2v}$ 分布，通过最小化 KL 散度促进对齐：

$$\mathcal{L}_{CMA} = \frac{1}{2} \sum_{i=1}^{N} \left( D_{KL}(Q_i^{t2t} \| P_i^{t2v}) + D_{KL}(Q_i^{v2v} \| (P_i^{t2v})^T) \right)$$

跨模态学习总损失：$\mathcal{L}_{CML} = \mathcal{L}_{cons} + \mathcal{L}_{CMA}$

#### 模态内排序蒸馏

认为样本关系本质上是连续的而非二元的，利用多教师模型（SimCSE + DiffCSE）生成粗粒度语义排序作为伪标签。使用 ListMLE 优化排序学习：

$$\mathcal{L}_{rank} = -\sum_{i=1}^{N} \log \left( \prod_{j=1}^{M} \frac{\exp(S(x_i)_{\pi_i^T(j)} / \tau)}{\sum_{k=j}^{M} \exp(S(x_i)_{\pi_i^T(k)} / \tau)} \right)$$

同时引入 KL 散度对齐全局分布：

$$\mathcal{L}_{IMA} = \sum_{i=1}^{N} D_{KL}(Q_i^{t2t} \| P_i^{t2t})$$

模态内总损失：$\mathcal{L}_{IML} = \mathcal{L}_{rank} + \mathcal{L}_{IMA}$

### 损失函数/训练策略

最终训练目标：

$$\mathcal{L}_{total} = \mathcal{L}_{Info} + \lambda \mathcal{L}_{CML} + \mu \mathcal{L}_{IML}$$

其中 $\mathcal{L}_{Info}$ 为 InfoNCE 对比损失，$\lambda$ 和 $\mu$ 为平衡超参数。

## 实验关键数据

### 主实验：STS 任务 (Spearman 相关系数 ×100)

**Flickr 数据集**：

| 模型 | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|------|-------|-------|-------|-------|-------|-------|--------|-----|
| SimCSE-BERT | 69.9 | 79.8 | 72.9 | 81.9 | 77.8 | 76.6 | 68.4 | 75.3 |
| MCSE-BERT | 71.4 | 81.8 | 74.8 | 83.6 | 77.5 | 79.5 | 72.6 | 77.3 |
| KDMCSE-BERT | 74.4 | 83.1 | 76.3 | 83.7 | 78.8 | 81.3 | 73.0 | 78.6 |
| **DALR-BERT** | 73.9 | **84.0** | **76.5** | **84.3** | **80.6** | **81.8** | **75.3** | **79.5** |
| KDMCSE-RoBERTa | 73.6 | 83.8 | 77.4 | 84.0 | 81.5 | 82.3 | 71.2 | 79.1 |
| **DALR-RoBERTa** | 73.6 | **84.4** | 77.2 | **84.9** | **82.0** | **82.6** | **74.6** | **79.9** |

DALR-BERT 在 Flickr 上平均得分 **79.5**，比 KDMCSE-BERT 高 **+0.9**；DALR-RoBERTa 达到 **79.9**，比 KDMCSE 高 **+0.8**。

### Transfer 任务

| 模型 | MR | CR | SUBJ | MPQA | SST | TREC | MRPC | Avg |
|------|-----|-----|------|------|-----|------|------|-----|
| KDMCSE-BERT (flickr) | 82.78 | 87.89 | 95.37 | 90.08 | 87.61 | 86.08 | 75.88 | 86.53 |
| **DALR-BERT** | **82.95** | **88.10** | **95.89** | **90.83** | **88.04** | **86.60** | **76.06** | **86.92** |
| KDMCSE-RoBERTa (flickr) | 83.21 | 88.16 | 95.73 | 90.46 | 88.05 | 86.30 | 76.18 | 86.87 |
| **DALR-RoBERTa** | **83.57** | **88.69** | **96.44** | **91.01** | **88.96** | **86.80** | **76.74** | **87.45** |

DALR-RoBERTa 在 TR 任务上平均 **87.45**，比 KDMCSE 高 **+0.58**。

### 消融实验

| 设置 | STS Avg | TR Avg |
|------|---------|--------|
| 基线 (SimCSE + 多模态对比) | 77.3 | 85.64 |
| + 跨模态对齐 (CML) | 78.8 | 86.38 |
| + 模态内对齐 (IML) | 78.6 | 86.25 |
| + CML + IML (DALR) | **79.5** | **86.92** |

两个模块均有独立贡献，组合效果最优。排序蒸馏中多教师（SimCSE+DiffCSE）优于单教师。

### 关键发现

1. 跨模态一致性学习有效缓解 CMB，SICK-R 提升最显著（+2.3 on BERT）
2. 排序蒸馏捕获连续语义结构比硬阈值过滤更有效
3. 多教师蒸馏比单教师效果更好，因为不同教师提供互补的排序视角
4. 在 Flickr 和 COCO 两个数据集上一致性提升

## 亮点与洞察

- **CMB 和 ISD 问题诊断精准**：通过可视化直观展示跨模态不对齐和模态内分歧
- **软标签替代硬阈值**：一致性学习生成连续软标签引导对齐，避免硬阈值的错误过滤
- **ListMLE + KL 互补设计**：ListMLE 保留排序结构，KL 散度学习全局分布，两者结合比单独使用更鲁棒
- **轻量级增量设计**：在现有多模态对比学习框架上增量添加，不需要额外架构修改

## 局限与展望

1. 依赖冻结的视觉/文本教师编码器（CLIP），教师质量直接影响上限
2. 仅在英文数据集上评估，跨语言泛化性未验证
3. 排序蒸馏中教师模型选择（SimCSE + DiffCSE）缺乏系统消融
4. 训练开销因多教师和辅助任务有所增加
5. 可探索与更强大的视觉编码器（如 SigLIP）结合

## 相关工作与启发

- **SimCSE → MCSE → KDMCSE → DALR**：多模态句子表示的发展脉络清晰，DALR 在前人基础上解决了两个被忽视的问题
- **ListMLE** 排序学习在 NLP 中的应用值得关注
- 启发：跨模态对齐不应只考虑正负对，还需建模样本间的连续排序关系

## 评分

- **新颖性**: ⭐⭐⭐ — CMB/ISD 问题定义有见地，但整体框架是已有技术的组合
- **实验充分度**: ⭐⭐⭐⭐ — STS 和 TR 双基准，两种 PLM 骨干，两种多模态数据集，消融完整
- **写作质量**: ⭐⭐⭐⭐ — 问题动机通过图例清晰展示，方法推导严谨
- **价值**: ⭐⭐⭐ — 在句子表示学习领域有稳定提升，但提升幅度有限（~1%）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning](../../CVPR2026/multimodal_vlm/semantic_noise_reduction_via_teacher-guided_dual-path_audio-visual_representatio.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](../../NeurIPS2025/multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](../../NeurIPS2025/multimodal_vlm/on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[ACL 2025\] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](../../NeurIPS2025/multimodal_vlm/structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths
description: >-
  [ECCV 2024][人体理解][视觉扫描路径] 提出GazeXplain，首次将视觉扫描路径预测与自然语言解释结合，通过注意力-语言解码器、语义对齐机制和跨数据集联合训练，实现对人类注视行为的可解释预测。 1. 领域现状： 人类探索视觉场景时，眼球运动形成的扫描路径（scanpath）——即注视点的时空序列——反映了其底…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "视觉扫描路径"
  - "自然语言解释"
  - "注视预测"
  - "语义对齐"
  - "跨数据集联合训练"
---

# GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths

**会议**: ECCV 2024  
**arXiv**: https://arxiv.org/abs/2408.02788  
**代码**: 无  
**领域**: 视觉注意力 / 可解释AI  
**关键词**: 视觉扫描路径, 自然语言解释, 注视预测, 语义对齐, 跨数据集联合训练

## 一句话总结

提出GazeXplain，首次将视觉扫描路径预测与自然语言解释结合，通过注意力-语言解码器、语义对齐机制和跨数据集联合训练，实现对人类注视行为的可解释预测。

## 研究背景与动机

1. **领域现状**: 人类探索视觉场景时，眼球运动形成的扫描路径（scanpath）——即注视点的时空序列——反映了其底层的注意力过程。理解视觉扫描路径对人机交互、自动驾驶、用户体验设计等应用至关重要。现有的扫描路径预测模型（如ChenLSTM、Gazeformer等）已经能较好地预测注视的"何时"（when）和"何处"（where），但无法解释"什么"（what）和"为什么"（why）。

2. **现有痛点**: 
    - **缺乏可解释性**：传统扫描路径模型仅输出注视位置和持续时间的序列，不提供任何关于注视原因的解释，存在理解gap
    - **无解释性标注数据**：现有eye-tracking数据集只标注了注视点坐标和时间，没有对每个注视点的语义解释标注
    - **任务特定性**：现有模型通常针对单一任务（如自由浏览、目标搜索或VQA）在单一数据集上训练，泛化性差
    - **视觉与语言的脱节**：注视行为蕴含丰富的语义信息，但现有方法未能将视觉注意力与自然语言理解联系起来

3. **核心矛盾**: 扫描路径预测需要深层语义理解，但现有模型只做了浅层的空间预测，无法阐明注视背后的认知过程和语义原因。

4. **本文目标**: 
    - 构建带有自然语言解释的扫描路径标注数据
    - 设计能够同时预测扫描路径和生成自然语言解释的统一模型
    - 实现跨数据集和跨任务的泛化能力

5. **切入角度**: 利用大型视觉语言模型（LLaVA）进行半自动化的注视解释标注，然后设计一个融合注意力解码和语言生成的统一架构，并通过语义对齐和跨数据集联合训练提升质量和泛化性。

6. **核心 idea**: 让扫描路径预测模型不仅预测人看哪里，还用自然语言解释人为什么看那里，从而实现可解释的人类视觉注意力建模。

## 方法详解

### 整体框架

GazeXplain建立在通用的视觉-语言编码器之上，核心创新在于注意力-语言解码器：

1. **视觉-语言编码器**：
    - 视觉编码：ResNet-50提取局部图像特征 $V_R \in \mathbb{R}^{C \times hw}$，经Transformer编码器获取全局上下文特征 $V_T \in \mathbb{R}^{d \times hw}$
    - 语言编码：RoBERTa处理任务指令，得到语义嵌入 $t_I \in \mathbb{R}^{d_{text}}$
    - 多模态融合：拼接视觉和语言特征得到 $V_I \in \mathbb{R}^{d \times hw}$

2. **注意力-语言解码器**：
    - 注意力解码器预测注视位置序列和持续时间
    - 语言解码器为每个注视点生成自然语言解释

3. **语义对齐机制**：确保注视和解释的语义一致性

4. **跨数据集联合训练**：在多个eye-tracking数据集上联合训练

### 关键设计

1. **注意力-语言解码器（Attention-Language Decoder）**: 
    - 功能：联合预测扫描路径和生成每个注视点的自然语言解释
    - 核心思路：
     - **注意力解码器**：使用Transformer模型生成显著性特征向量 $\{s_k\}_{k=1}^K$，通过与联合嵌入 $V_I$ 的余弦相似度预测注视的时空分布 $\{m_k\}$，同时预测注视持续时间的对数正态分布参数 $\{\mu_k, \sigma_k^2\}$ 和序列结束标志 $\{e_k\}$
     - **语言解码器**：(1) 根据注视位置 $y_k$ 从视觉特征 $V_T$ 中提取局部特征 $g_k$；(2) 将视觉特征 $g_k$ 和语义嵌入 $t_I$ 通过可学习参数和位置编码投影到同一维度；(3) 将融合后的特征送入预训练的语言模型（BLIP）生成解释文本 $\{w_\ell^k\}_{\ell=1}^L$
    - 设计动机：通过为每个注视点提供自然语言解释，模型被迫理解注视区域的语义内容，这反过来会提升注视预测的准确性

2. **语义对齐机制（Semantic Alignment）**: 
    - 功能：确保预测的注视、生成的解释和视觉特征在语义空间中保持一致
    - 核心思路：计算四种成对相似度：
     - 视觉相似度 $s_{i,j}^r$：预训练ResNet提取的注视区域视觉特征的余弦相似度（作为伪标签）
     - 解释相似度 $s_{i,j}^e$：不同注视的解释语言特征的余弦相似度
     - 注视相似度 $s_{i,j}^f$：注视点视觉特征的余弦相似度
     - 多模态相似度 $s_{i,j}^m$：解释语言特征与注视视觉特征的跨模态余弦相似度
    - 对齐损失：$\mathcal{L}_{aln} = \frac{1}{K'^2} \sum_{i,j} [(s_{i,j}^e - s_{i,j}^r)^2 + (s_{i,j}^f - s_{i,j}^r)^2 + (s_{i,j}^m - s_{i,j}^r)^2]$
    - 设计动机：如果两个注视看的是相似的视觉内容，那么它们的解释也应该相似，注视特征也应该相似——这种一致性约束促进了多模态表示的协调

3. **跨数据集联合训练（Cross-Dataset Co-Training）**: 
    - 功能：使模型能够同时从多个不同任务的eye-tracking数据集中学习，提升泛化性
    - 核心思路：将不同任务的指令统一为VQA格式——自由浏览转换为"What do you see in the image?"，目标搜索转换为"Is there a [target] in the image?"。图像和扫描路径统一缩放到384×512分辨率。可选地加入观察者的答案来捕获个体差异
    - 设计动机：单数据集训练容易过拟合于特定任务，联合训练可以让模型学习到跨任务的通用注意力模式

### 损失函数 / 训练策略

最终训练目标是三个损失的加和:

$$\mathcal{L} = \mathcal{L}_{fix} + \mathcal{L}_{exp} + \mathcal{L}_{aln}$$

- **扫描路径预测损失** $\mathcal{L}_{fix}$：注视位置的条件对数概率 + 持续时间的对数正态分布损失
- **语言生成损失** $\mathcal{L}_{exp}$：标准的自回归语言建模交叉熵损失
- **语义对齐损失** $\mathcal{L}_{aln}$：如上所述的多视角一致性损失
- **训练策略**：先进行8个epoch的监督学习（lr=4×10⁻⁴, batch=16），再进行2个epoch的自我批评序列训练（SCST, lr从10⁻⁵线性衰减, batch=8）

## 实验关键数据

### 主实验

在4个eye-tracking数据集/子集上的扫描路径预测结果：

| 数据集 | 指标 | GazeXplain | Gazeformer | ChenLSTM | 提升 |
|--------|------|-----------|-----------|---------|------|
| AiR-D (VQA) | SM↑ | **0.386** | 0.357 | 0.350 | +8.1% |
| AiR-D | CC↑ | **0.662** | 0.550 | 0.629 | +5.2% |
| AiR-D | NSS↑ | **1.851** | 1.512 | 1.727 | +7.2% |
| OSIE (Free-view) | SM↑ | **0.380** | 0.372 | 0.377 | +0.8% |
| OSIE | CC↑ | **0.748** | 0.685 | 0.722 | +3.6% |
| COCO-Search18 TP | SM↑ | **0.480** | 0.433 | 0.448 | +7.1% |
| COCO-Search18 TP | SS↑ | **0.541** | 0.470 | 0.475 | +13.9% |
| COCO-Search18 TA | SM↑ | **0.373** | 0.354 | 0.366 | +1.9% |

### 消融实验

在AiR-D数据集上的组件贡献分析：

| 配置 (EXP/ALN/CT) | SM↑ | CC↑ | NSS↑ | CIDEr-R↑ | 说明 |
|-------------------|-----|-----|------|----------|------|
| ✗/✗/✗ | 0.337 | 0.582 | 1.582 | 61.9 | 基线 |
| ✓/✗/✗ | 0.339 | 0.614 | 1.674 | 91.9 | 语言解码器单独有效 |
| ✓/✓/✗ | 0.346 | 0.631 | 1.733 | 115.1 | 语义对齐进一步提升 |
| ✗/✗/✓ | 0.356 | 0.582 | 1.597 | 66.7 | 联合训练独立有效 |
| ✓/✗/✓ | 0.378 | 0.647 | 1.797 | 97.3 | 解释+联训组合 |
| ✓/✓/✓ | **0.386** | **0.662** | **1.851** | **123.1** | 全部组件最优 |

### 关键发现

- 为注视点添加语言解释不仅不会损害扫描路径预测性能，反而能显著提升预测准确性（SM从0.337到0.386）
- 语义对齐机制将CIDEr-R从97.3提升到123.1，同时亦提升扫描路径预测指标
- 跨数据集联合训练对AiR-D的SM提升最大（从0.346到0.386），CT在OSIE和COCO-Search18 TA等探索性任务上效果更显著
- 竞争模型ChenLSTM和Gazeformer在跨数据集训练时性能反而下降，证明GazeXplain的设计对于利用多源数据至关重要
- GazeXplain生成的解释在忠实度（faithfulness）、多样性、词汇丰富度上均优于直接用BLIP描述

## 亮点与洞察

- **开创性新任务**：首次提出可解释的扫描路径预测任务，将"看哪里"和"为什么看"统一建模
- **数据标注创新**：利用LLaVA进行半自动化标注+人工质量控制，在4个数据集上标注了86,407个注视点的自然语言解释
- **解释促进预测**：令人意外且深刻的发现——强迫模型解释注视行为反而提升了注视预测的准确性，说明语义理解对注意力建模至关重要
- **语义对齐设计精妙**：利用视觉相似度作为自监督信号来约束解释和注视的多模态一致性
- **强泛化性**：在COCO-FreeView和WebSaliency两个额外数据集上也展现了SOTA性能

## 局限与展望

- LLaVA生成的解释标注可能存在噪声（如文本识别错误、小物体描述不准确），尽管进行了人工质量控制但仍有约0.58%的异常
- 解释的粒度固定为注视点级别，未探索更高层次的扫描路径级别的综合解释
- 当前仅使用BLIP作为语言解码器，可以探索更强大的LLM来提升解释质量
- 未利用注视持续时间信息来调整解释的详细程度（长时间注视可能需要更详细的解释）
- 跨数据集联合训练的数据混合比例可能需要更细致的调优

## 相关工作与启发

- **ChenLSTM / Gazeformer**：现有SOTA扫描路径预测方法，GazeXplain在此基础上增加了解释能力
- **BLIP / LLaVA**：视觉-语言模型，分别用于语言解码和数据标注
- **AiR-D / OSIE / COCO-Search18**：核心eye-tracking数据集
- **Image Captioning / Visual Explanation**：GazeXplain将图像描述技术引入到注视解释中
- 启发：可解释性不仅是模型输出的附加品，更能反过来提升模型的核心性能——这对其他任务也有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创了可解释扫描路径预测的新研究方向，任务定义、数据标注和模型设计均有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 4+2个数据集、全面的消融实验、多维度评估（扫描路径+显著性+解释质量+多样性+忠实度）
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，实验分析细致深入
- 价值: ⭐⭐⭐⭐⭐ 为人类视觉注意力理解开辟了全新方向，具有广泛的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](a_simple_baseline_for_spoken_language_to_sign_language_trans.md)
- [\[CVPR 2026\] Forecasting 3D Scanpaths in Egocentric Video](../../CVPR2026/human_understanding/forecasting_3d_scanpaths_in_egocentric_video.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](../../AAAI2026/human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[CVPR 2026\] Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes](../../CVPR2026/human_understanding/beyond_scanpaths_graph-based_gaze_simulation_in_dynamic_scenes.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)

</div>

<!-- RELATED:END -->

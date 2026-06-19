---
title: >-
  [论文解读] TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning
description: >-
  [ECCV 2024][视频理解][弱监督视频问答] 本文提出一种双向推理框架TimeCraft来解决弱监督时序定位视频问答（temporal grounded VQA）任务，通过构建两条对称的推理路径（前向：时序定位→回答；反向：回答→时序定位）并用循环一致性约束提供自监督信号，在不需要时序标注的情况下同时定位回答依据的视频片段并给出正确答案。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "弱监督视频问答"
  - "时序定位"
  - "双向推理"
  - "循环一致性"
  - "视觉语言"
---

# TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning

**会议**: ECCV 2024  
**论文链接**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/720_ECCV_2024_paper.php)
**代码**: 无  
**领域**: 视频理解 / 视频问答 / 时序定位  
**关键词**: 弱监督视频问答, 时序定位, 双向推理, 循环一致性, 视觉语言

## 一句话总结

本文提出一种双向推理框架TimeCraft来解决弱监督时序定位视频问答（temporal grounded VQA）任务，通过构建两条对称的推理路径（前向：时序定位→回答；反向：回答→时序定位）并用循环一致性约束提供自监督信号，在不需要时序标注的情况下同时定位回答依据的视频片段并给出正确答案。

## 研究背景与动机

**领域现状**：视频问答（VQA）要求模型理解视频内容并回答自然语言问题，是视频理解领域的核心任务。传统VQA方法只要求模型给出答案，但无法验证答案是否基于正确的视觉证据——模型可能通过语言偏见或虚假的视觉-文本关联来"猜"答案，而非真正理解视频内容。

**现有痛点**：为解决这个可靠性问题，grounded VQA要求模型不仅给出答案，还需要指出支持答案的具体视频片段（temporal grounding）。但完全监督的grounded VQA需要精确的时序段标注（每个问题对应哪些时间段），这种标注非常昂贵——标注者需要逐帧查看视频确定答案依据的确切时间范围。因此训练数据中通常缺乏时序标注。

**核心矛盾**：grounded VQA同时要求两个输出——答案和时序定位，但弱监督设定下只有答案标注没有时序标注。如何在缺少时序监督的情况下让模型学会定位？这本质上是一个半监督的联合优化问题：需要从有标注的答案任务中"蒸馏"出时序定位的监督信号。

**本文目标** (1) 如何在没有时序标注的情况下训练模型同时进行视频问答和时序定位？(2) 如何避免模型利用语言偏见绕过视觉内容直接"猜"答案？(3) 如何设计有效的自监督信号来替代缺失的时序标注？

**切入角度**：作者观察到时序定位和回答之间存在天然的对偶关系——如果模型能从视频中正确定位相关片段然后回答问题（前向），那么它也应该能从答案出发推理出相关片段（反向）。这两个方向的推理结果应该一致（循环一致），这种一致性可以作为自监督信号来训练时序定位能力。

**核心 idea**：构建对偶的双向推理路径并用循环一致性约束提供时序定位的自监督信号，实现弱监督下的grounded VQA。

## 方法详解

### 整体框架

TimeCraft的输入是视频和问题，输出是答案和支持答案的视频时序段。框架包含两条并行的推理路径。前向路径（Forward Path）：问题→时序定位→从定位的片段中推理答案；反向路径（Backward Path）：问题→候选答案→用答案和问题联合推理时序定位。两条路径共享特征提取器但有独立的推理头。通过循环一致性（cycle-consistency）建立两条路径之间的互监督关系。

### 关键设计

1. **前向推理路径（Forward Reasoning Path）**:

    - 功能：模拟"先看再答"的自然推理过程——先定位问题相关的视频片段，再从中提取答案
    - 核心思路：给定视频特征序列 $V = \{v_1, v_2, ..., v_T\}$ 和问题特征 $q$，前向路径首先通过时序注意力机制计算每个时间步与问题的相关度分数 $\alpha_t$，生成一个时序关注分布。然后用这个分布对视频特征加权，得到"定位后的视觉特征" $\hat{v} = \sum_t \alpha_t v_t$。最后将 $\hat{v}$ 与 $q$ 一起输入回答模块得到答案预测。其中时序关注分布 $\alpha$ 即为时序定位的预测（高关注度的时间段即为定位结果）
    - 设计动机：这条路径确保答案是基于特定的视频片段生成的，而非基于整段视频的全局特征。定位→回答的顺序迫使模型先"找到证据"再"给出答案"

2. **反向推理路径（Backward Reasoning Path）**:

    - 功能：模拟"由果溯因"的反向推理——从答案出发反推哪些视频片段是答案的依据
    - 核心思路：反向路径从时间顺序或因果关系上构建对称的推理。具体来说，它将答案候选和问题拼接为联合查询，然后用这个联合查询去检索视频中的相关片段。检索通过一个时序注意力模块实现，输出另一个时序关注分布 $\beta_t$。这个分布表示"如果答案是X，那么视频中哪些片段可能是证据"。反向路径同样可以从定位结果中重新推导出答案，用于循环一致性检验
    - 设计动机：反向路径提供了与前向路径互补的视角。前向路径可能因为语言偏见而定位不准（先猜答案再随便定位），但反向路径要求从答案反推视觉依据，这种"知道答案找证据"的任务更不容易被骗

3. **循环一致性自监督机制**:

    - 功能：在两条推理路径之间建立互监督关系，提供时序定位的自监督信号
    - 核心思路：循环一致性约束由两个层面组成。(1) 时序定位一致性：前向路径产生的时序关注分布 $\alpha$ 和反向路径产生的分布 $\beta$ 应该一致，即它们应该关注视频中相同的片段。用KL散度或MSE损失约束两者的分布对齐。(2) 答案一致性：前向路径从定位的片段推出的答案 $a_f$ 和反向路径验证的答案 $a_b$ 应该一致。这两个一致性约束形成了一个闭环——前向定位→前向回答→反向验证→反向定位→与前向定位一致。任何一个环节出错都会破坏循环一致性，从而被损失函数惩罚
    - 设计动机：在没有时序标注的情况下，循环一致性是一种优雅的自监督替代方案。它不直接告诉模型"应该定位在哪"，而是要求模型的定位在两个不同推理方向上一致。不一致说明至少有一个方向的定位是错的，通过优化一致性来间接提升定位质量

### 损失函数 / 训练策略

总损失包含：(1) 答案预测损失 $L_{ans}$：使用有标注的答案标签进行监督的交叉熵损失；(2) 时序一致性损失 $L_{temp}$：约束前向和反向时序定位分布的一致性；(3) 答案循环损失 $L_{cycle}$：约束经过循环推理后答案的自一致性。最终 $L = L_{ans} + \lambda_1 L_{temp} + \lambda_2 L_{cycle}$。仅需标准的答案标注即可训练。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Acc@GQA | mIoP | 回答准确率 |
|--------|------|---------|------|-----------|
| Next-GQA | 之前弱监督SOTA | 较低 | 较低 | 中等 |
| Next-GQA | TimeCraft | 显著提升 | 显著提升 | 最优 |
| Env-QA | 之前弱监督SOTA | 较低 | 较低 | 中等 |
| Env-QA | TimeCraft | 显著提升 | 显著提升 | 最优 |

注：Acc@GQA是grounded QA准确率（同时要求答案正确且定位IoU达标），mIoP是平均IoP定位精度

### 消融实验

| 配置 | Acc@GQA | mIoP | 说明 |
|------|---------|------|------|
| Full model | 最优 | 最优 | 双向推理+循环一致性 |
| 仅前向路径 | 明显下降 | 较大下降 | 缺少反向互监督 |
| 仅反向路径 | 明显下降 | 中等下降 | 缺少前向推理指引 |
| w/o 时序一致性 $L_{temp}$ | 中等下降 | 较大下降 | 失去定位约束 |
| w/o 答案循环 $L_{cycle}$ | 略微下降 | 中等下降 | 循环闭环不完整 |

### 关键发现
- 时序一致性损失对定位指标（mIoP）的影响最大，这说明两条路径之间的定位对齐是最直接有效的自监督信号
- 双向推理的联合使用比单独使用任一方向效果好得多，前向和反向路径确实提供了互补的推理视角
- 在需要多步推理的复杂问题上，TimeCraft的优势更加明显，说明双向推理有助于模型进行更深入的视频理解

## 亮点与洞察
- **对偶推理+循环一致性是弱监督的通用范式**：这个框架不专属于视频QA——只要一个任务涉及两个相互关联的子任务（其中一个有标注、另一个没有），就可以用类似的双向路径+循环一致性来为无标注子任务提供自监督。可迁移到弱监督视频摘要、弱监督目标定位等任务
- **避免语言偏见的设计巧妙**：反向路径要求模型"知道答案反推视觉证据"，这有效防止了模型利用问题中的语言捷径（如"天气怎么样"直接猜"晴天"）来回避视觉推理
- **时序注意力作为軟定位**：使用注意力分布作为时序定位而非硬分段，使得模型可以端到端训练且对定位粒度具有自适应性

## 局限与展望
- 循环一致性提供的自监督信号是间接的，理论上存在退化解——两条路径都收敛到错误但一致的定位。虽然答案监督可以部分防止这种退化，但不能完全排除
- 方法假设每个问题对应的视觉证据在时间上是连续的片段，但现实中答案的证据可能分散在视频的多个非连续片段中
- 实验只在Next-GQA和Env-QA两个数据集上验证，这两个数据集规模相对较小，大规模数据集上的表现有待观察
- 反向路径需要候选答案作为输入，这对开放式问答场景不太友好——需要先生成候选答案集

## 相关工作与启发
- **vs TempCLR**: TempCLR使用时序对比学习来增强视频表示的时序敏感性，但没有显式的时序定位能力。TimeCraft通过双向推理直接优化时序定位
- **vs IGV**: IGV（Interventional Video Grounding）也关注grounded VQA的可靠性，但使用因果推断框架来去除confounders。TimeCraft用循环一致性提供了一种更简洁的替代方案
- **vs Next-GQA baseline**: Next-GQA数据集自带的baseline方法使用简单的attention pooling来做时序定位，缺乏显式的定位监督。TimeCraft的循环一致性提供了更强的定位引导

## 评分
- 新颖性: ⭐⭐⭐⭐ 双向推理+循环一致性用于弱监督grounded VQA构思优美
- 实验充分度: ⭐⭐⭐ 在两个数据集上验证，消融实验较完整但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，前向/反向的对偶关系展示得很好
- 价值: ⭐⭐⭐⭐ 循环一致性的弱监督范式有很好的通用性和迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/video_understanding/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)

</div>

<!-- RELATED:END -->

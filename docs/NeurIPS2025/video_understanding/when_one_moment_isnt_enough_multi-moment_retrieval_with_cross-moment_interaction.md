---
title: >-
  [论文解读] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions
description: >-
  [NeurIPS 2025][视频理解][多时刻检索] 提出QV-M2数据集（首个全人工标注的多时刻检索基准）和FlashMMR框架（含后验证模块），将视频时刻检索从单时刻扩展到多时刻场景，建立了多时刻检索的标准化评价体系。 从单时刻到多时刻的需求 视频时刻检索（Moment Retrieval, MR）旨在根据自然语言查询…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "多时刻检索"
  - "视频时间定位"
  - "数据集"
  - "后验证模块"
  - "时刻检索"
---

# When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions

**会议**: NeurIPS 2025  
**arXiv**: [2510.17218](https://arxiv.org/abs/2510.17218)  
**代码**: [GitHub](https://github.com/Zhuo-Cao/QV-M2)  
**领域**: 视频理解  
**关键词**: 多时刻检索, 视频时间定位, 数据集, 后验证模块, 时刻检索

## 一句话总结

提出QV-M2数据集（首个全人工标注的多时刻检索基准）和FlashMMR框架（含后验证模块），将视频时刻检索从单时刻扩展到多时刻场景，建立了多时刻检索的标准化评价体系。

## 研究背景与动机

### 从单时刻到多时刻的需求

视频时刻检索（Moment Retrieval, MR）旨在根据自然语言查询定位视频中的相关时间片段。现有方法几乎全部基于**单时刻检索（SMR）**假设：每个查询仅对应一个视频片段。然而在现实场景中，单个查询往往对应多个不连续的时刻。例如，在烹饪教程中"切蔬菜"可能在视频中发生多次；在体育转播中"成功的三分球"也会反复出现。

### 三个核心缺失

**缺数据集**：现有MR数据集（Charades-STA、QVHighlights等）的标注以单时刻为主，每个查询平均仅1-1.8个时刻

**缺评价指标**：标准mAP和IoU指标无法衡量多目标检索的覆盖度

**缺方法**：现有方法架构天然限制在单时刻预测，对多时刻场景表现不佳

### SMR方法在多时刻场景下的失败

SMR方法在多时刻查询上的核心问题是：模型倾向于优化最高置信度的单个时刻预测，丢弃其他同样有效的片段。即使产生多个候选预测，也缺乏机制确保跨時刻的语义一致性和去冗余。

## 方法详解

### 整体框架

FlashMMR继承FlashVTG的基础架构，包含特征提取与融合、多尺度时间处理和预测三个阶段，并新增**后验证模块（Post-Verification Module）** 来专门处理多时刻场景的边界精化和语义一致性控制。

### 关键设计

#### 1. QV-M2数据集构建

**构建方法**：基于QVHighlights的原始视频，新增针对多时刻场景的人工标注。

- 标注规范：(i) 创建精确捕获演员、动作和上下文的详细查询；(ii) 包含需要理解时间关系的上下文依赖查询；(iii) 设计负查询标记特定动作不发生的片段
- 质量控制：每100个视频标注后随机抽样5%由第二名标注者复查，若时间边界重叠低于90%则由第三人重标
- 统计：2212个查询，6384个标注片段，覆盖1341个视频，**每查询平均2.9个时刻**（远超QVHighlights的1.8）

#### 2. 多时刻评价指标体系

**Generalized mAP (G-mAP)**：在多个IoU阈值 $\mathcal{T} = \{0.5, 0.55, \dots, 0.9\}$ 上平均AP：

$$\text{G-mAP} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{AP}(\tau)$$

并按查询的真实时刻数分组报告（mAP@1\_tgt、mAP@2\_tgt、mAP@3+\_tgt）。

**Mean IoU@k**：前k个预测与最佳匹配真值的平均IoU：

$$\text{mIoU}@k = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{k} \sum_{i=1}^{k} \max_{\text{gt} \in \mathcal{G}(q)} \text{IoU}(\text{pred}_i, \text{gt})$$

**Mean Recall@k**：前k个预测覆盖真值的召回率，仅对至少有k个真值的查询计算。

这三组指标在k=1时与标准SMR指标完全一致，确保了向后兼容。

#### 3. 后验证模块（Post-Verification Module）

**设计动机**：初始预测在多时刻场景下容易产生冗余或不相关的时刻预测，需要一个机制来精化边界并过滤低置信度提议。

**后处理与特征精化**：对初始预测 $\hat{B} \in \mathbb{R}^{3 \times n}$ 应用结构化约束：

$$\tilde{B} = \mathcal{F}(\hat{B}, \lambda_{\text{clip}}, \lambda_{\text{round}})$$

包括最小/最大窗口长度、时间裁剪和离散化，然后从融合特征 $F$ 中提取每个预测区间的多模态特征表示 $\mathbf{I}_i$。

**语义一致性验证**：使用GRU循环网络 $\mathcal{P}_{\text{GRU}}$ 建模检索区间间的上下文依赖关系：

$$p_i = \sigma(\mathcal{P}_{\text{GRU}}(\mathbf{I}_i))$$

输出精化置信度 $p_i$，通过tIoU监督：

$$\hat{\mathbf{IoU}} = \max(\text{tIoU}(\tilde{B}, B^*), \text{dim}=-1)$$

### 损失函数 / 训练策略

总损失包含FlashVTG的基础损失和后验证损失：

$$\mathcal{L}_{\text{PV}} = \|\mathbf{p} - \hat{\mathbf{IoU}}\|_2^2 + \mathcal{L}_{\text{repr}}$$

其中表示学习损失 $\mathcal{L}_{\text{repr}} = \sum_i \text{CE}(\mathbf{S}_i, \mathbf{T}_i)$ 通过余弦相似度矩阵 $\mathbf{S}_i$ 与真值片段一致性矩阵 $\mathbf{T}_i$ 的交叉熵，强制时间邻近帧保持高语义一致性。损失权重：$\mathcal{L}_{\text{PV}}$ 权重9，$\mathcal{L}_{\text{repr}}$ 权重7。NMS阈值0.7。

## 实验关键数据

### QV-M2测试集主实验

| 方法 | G-mAP | mAP@3+tgt | mIoU@2 | mIoU@3 | mR@2 | mR@3 |
|------|-------|-----------|--------|--------|------|------|
| M-DETR (NeurIPS'21) | 20.65 | 10.95 | 38.98 | 34.34 | 30.95 | 26.24 |
| QD-DETR (CVPR'23) | 28.95 | 18.30 | 46.79 | 40.50 | 40.58 | 36.05 |
| FlashVTG (WACV'25) | 32.14 | 20.19 | 47.85 | 40.92 | 41.30 | 35.94 |
| **FlashMMR** | **35.14** | **22.89** | **49.64** | **42.92** | **44.33** | **38.50** |
| 提升(vs FlashVTG) | +3.00 | +2.70 | +1.79 | +2.00 | +3.03 | +2.56 |

### 消融实验

| 配置 | G-mAP | mAP@2_tgt | mAP@3+tgt | mIoU@2 | mR@2 |
|------|-------|-----------|-----------|--------|------|
| FlashMMR w/o PV (QV-M2) | 32.14 | 39.48 | 20.19 | 47.85 | 41.30 |
| FlashMMR w/ PV (QV-M2) | **35.14** | **42.52** | **22.89** | **49.64** | **44.33** |
| FlashMMR w/o PV (QVHighlights) | 48.02 | 35.08 | 13.85 | 43.80 | 38.98 |
| FlashMMR w/ PV (QVHighlights) | **48.07** | **35.78** | **15.15** | **45.32** | **40.63** |

### 跨数据集实验（QV-M2训练提升性能）

| 方法 | 训练集 | G-mAP | mR@3 |
|------|--------|-------|------|
| M-DETR | QVHighlights | 32.79 | 19.55 |
| M-DETR | QV-M2 Train | **34.70** | **23.67** |
| FlashMMR | QVHighlights | 48.07 | 36.68 |
| FlashMMR | QV-M2 Train | **48.42** | **39.29** |

### 关键发现

1. 后验证模块在两个数据集上均带来一致提升，QV-M2上G-mAP提升3%，证明PV模块对多时刻场景的必要性
2. 在QV-M2上训练后，所有方法在SMR和MMR任务上都有性能提升，验证数据集质量
3. 多时刻查询（3+目标）对所有方法挑战最大，FlashMMR的mAP@3+tgt从20.19提升到22.89
4. 现有SMR方法在MMR评价中显著下降，证实了专用MMR框架的必要性

## 亮点与洞察

1. **问题定义有价值**：从SMR到MMR的范式转换反映了真实场景需求，是自然语言视频理解的重要进展
2. **评价体系完备**：G-mAP、mIoU@k、mR@k三组指标与SMR向后兼容，设计优雅
3. **数据集质量高**：全人工标注+严格质量控制，每查询2.9个时刻，比现有数据集更具挑战性
4. **方法设计简洁有效**：后验证模块仅需一个GRU网络，参数量小但效果显著

## 局限与展望

- 后验证模块较初级，可探索强化学习或对比学习进行更精细的时刻判别
- QV-M2规模有限（2212查询），随模型发展可能需要更大规模标注
- 基于SlowFast+CLIP的特征提取较固定，未探索更强的视觉骨干
- 方法假设时刻间互不重叠，对部分重叠场景的处理未明确
- NMS阈值0.7为固定设置，自适应阈值策略可能进一步提升性能

## 相关工作与启发

- **单时刻检索**: Moment-DETR, QD-DETR, FlashVTG
- **多时刻检索**: SFABD, NExT-VMR（并行工作，未公开数据集）
- **数据集**: QVHighlights, DiDeMo, CharadesSTA
- **启发**: MMR的评价指标设计思路可推广到其他one-to-many匹配问题

## 评分
- 新颖性: ⭐⭐⭐⭐☆ — 多时刻检索是自然延伸但此前缺少系统化工作
- 实验充分度: ⭐⭐⭐⭐⭐ — 6个基线方法的跨数据集对比+消融+新指标体系
- 写作质量: ⭐⭐⭐⭐☆ — 结构清晰，数据集构建流程规范
- 价值: ⭐⭐⭐⭐☆ — 数据集和指标贡献可能比方法本身更有长期影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](token_bottleneck_one_token_to_remember_dynamics.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)

</div>

<!-- RELATED:END -->

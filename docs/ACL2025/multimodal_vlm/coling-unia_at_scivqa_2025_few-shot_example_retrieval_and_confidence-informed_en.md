---
title: >-
  [论文解读] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models
description: >-
  [ACL 2025][多模态VLM][科学图表VQA] 本文提出了一种基于多模态大模型（MLLM）集成的科学图表视觉问答系统，通过 few-shot 示例检索策略和置信度感知的模型选择机制，在 SciVQA 2025 共享任务中获得第三名（平均 F1 = 85.12）。 科学图表视觉问答（Scientific VQA）要求系…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "科学图表VQA"
  - "多模态大模型"
  - "few-shot检索"
  - "置信度集成"
  - "模型校准"
---

# COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models

**会议**: ACL 2025  
**arXiv**: [2507.02357](https://arxiv.org/abs/2507.02357)  
**代码**: [有](https://github.com/coling-unia/few-shot-scivqa2025)  
**领域**: 多模态VLM  
**关键词**: 科学图表VQA, 多模态大模型, few-shot检索, 置信度集成, 模型校准

## 一句话总结

本文提出了一种基于多模态大模型（MLLM）集成的科学图表视觉问答系统，通过 few-shot 示例检索策略和置信度感知的模型选择机制，在 SciVQA 2025 共享任务中获得第三名（平均 F1 = 85.12）。

## 研究背景与动机

科学图表视觉问答（Scientific VQA）要求系统回答关于科学图表（如折线图、柱状图、架构图等）的自然语言问题。与一般 VQA 不同，科学图表 VQA 面临以下挑战：

**图表类型多样**：数据集包含折线图、散点图、饼图、混淆矩阵、神经网络架构图等多种类型，不同类型需要不同的理解能力

**问题类型复杂**：包括二值问题、四选一、无穷解集（视觉/非视觉）、不可回答等七种问题类型

**现有方法局限**：大多数图表 VQA 方法依赖针对图表领域微调的专用模型，泛化性有限

本文的核心动机是：**不做任何微调，仅通过 zero/few-shot 提示和智能集成策略**，能否让开源 MLLM 在科学图表 VQA 上达到竞争水平？

## 方法详解

### 整体框架

系统采用两阶段集成架构（Confidence-Informed Ensemble）：

1. **第一阶段**：使用校准良好的 InternVL3-78B（1s_q_img_f, BLIP2）配置处理所有实例，保留置信度 ≥ 90% 的高置信答案（约占一半实例）
2. **第二阶段**：对剩余低置信实例，根据问题类型选择最优的模型 + few-shot 配置组合

使用的模型均为开源权重模型：
- **InternVL3-78B**：性能更强，但上下文窗口有限，仅支持 0-shot 和 1-shot
- **Pixtral-Large-Instruct-2411**：上下文更大，支持 2-shot，额外示例带来更大提升

所有模型使用 16-bit 量化，温度设为 0。

### 关键设计

#### 1. Few-Shot 示例检索策略

本文评估了多种检索策略的组合：

| 维度 | 选项 |
|------|------|
| 相似度来源 | 仅问题相似度（q）/ 问题+图像相似度（q_img） |
| 嵌入模型 | SBERT / CLIP / BLIP-2 |
| 检索范围 | 按图表类型过滤（f）/ 全训练集搜索（nf） |
| 示例数量 | 0-shot / 1-shot / 2-shot |

- **问题相似度**：使用 SBERT 嵌入的余弦相似度排序
- **问题-图像相似度**：计算 CLIP 嵌入的问题和图像向量，归一化后取均值，再用余弦相似度排序
- **BLIP-2 变体**：BLIP-2 的嵌入主要反映图像内容，导致大量并列结果
- **2-shot 设计**：选择一个可回答和一个不可回答的示例，有效帮助模型区分两类

#### 2. 置信度估计与模型校准

置信度计算方法：对所有生成答案 token 的平均对数概率取指数：

$$\text{confidence} = \exp\left(\frac{1}{|T|}\sum_{t \in T} \log p(t)\right)$$

关键发现：InternVL3-78B（1s_q_img_f, BLIP2）具有良好的校准特性——高置信度（≥ 0.9）的预测确实对应高准确率。这使得该模型适合作为第一阶段的「守门员」。

#### 3. 问题/图表类型集成

为避免过拟合，将数据分为 16 个组（8 个均质图表类型 + 1 个「其他」+ 7 个按问题类型细分的折线图子集），通过 5-fold 交叉验证确定每组最优配置。关键步骤：
- 每折计算所有配置的 ROUGE-1 F1
- 减去该折最高分，计算跨折平均差距
- 重复至少 10 次直到最优配置稳定

### 损失函数 / 训练策略

本方法无需任何训练或微调，完全基于推理时的策略：
- 输入构建：图像 + 问题 + 元数据（图表标题、类型、是否含子图）
- 对有预定义选项的问题，提示模型从选项中选择
- 对开放式问题，指示模型自由回答
- 指示模型判断是否可基于提供信息回答

## 实验关键数据

### 主实验

**官方测试集排名（表1）**：

| 排名 | 团队 | ROUGE-1 F1 | ROUGE-L F1 | BERTScore F1 | 平均 |
|------|------|-----------|-----------|-------------|------|
| 1 | ExpertNeurons | 80.49 | 80.43 | 98.49 | 86.47 |
| 2 | THAii_LAB | 78.99 | 78.92 | 98.39 | 85.43 |
| **3** | **Coling-UniA** | **78.62** | **78.56** | **98.17** | **85.12** |
| 中位 | - | 75.83 | 75.75 | 98.36 | 83.31 |

**测试集上不同方法对比（表2摘要）**：

| 方法 | ROUGE-1 F1 | ROUGE-L F1 | BERTScore F1 |
|------|-----------|-----------|-------------|
| InternVL（1s_q_img_f, BLIP2）单模型 | 77.2 | 77.2 | 98.1 |
| 问题/图表类型集成 | 77.7 | 77.6 | 98.1 |
| **置信度感知集成** | **78.6** | **78.6** | **98.2** |

### 消融实验

**few-shot 对性能的影响**：
- InternVL3-78B：0-shot R1-F1 = 74.2 → 1-shot 最优 = 75.0（+0.8）
- Pixtral-Large：0-shot R1-F1 = 71.4 → 2-shot 最优 = 74.1（+2.7）
- 2-shot 对 Pixtral 几乎总是有益，增加示例对识别不可回答问题特别有帮助

**不可回答问题识别精度（表3）**：
- Pixtral 0-shot: 93.0% → 2-shot(q_img_f): 94.1%
- 2-shot 使用一个可回答 + 一个不可回答示例的策略效果最好

**不同问题类型的性能差异极大**：
- 二值问题（视觉）：~81% 
- 四选一：~76-79%
- 无穷解集（非视觉）：~65-68%
- 无穷解集（视觉）：~49-54%（最难）
- 不可回答：~77-91%

### 关键发现

1. **无微调也能竞争**：仅靠 zero/few-shot 策略，开源 MLLM 即可达到竞赛第三名，超越基线约 4 个百分点
2. **问题/图表类型显著影响最优策略**：没有单一配置在所有子集上都最优
3. **模型校准的实用价值**：InternVL3-78B 的高置信预测确实可靠，允许分层处理
4. **检索策略差异不大**：问题相似度 vs. 问题+图像相似度在整体性能上差异有限

## 亮点与洞察

1. **分层决策思路新颖**：先用高置信模型筛选「容易」实例，再对「难」实例精细调度，比单一模型或简单集成更高效
2. **置信度校准的发现**：BLIP-2 检索配置的 InternVL3 在高置信区间有优异校准，这是一个实用且可复用的发现
3. **无需微调的竞争力**：证明了 MLLM 在科学图表理解上的强大 zero-shot 能力，暗示大量 chart-specific 微调可能不再必要
4. **系统性的实验设计**：对所有配置组合进行了详尽评估，分析粒度细到每种问题类型 × 图表类型

## 局限与展望

1. **潜在数据泄露**：ACL-Fig 和 SciGraphQA 中的图像可能已在 MLLM 预训练中见过
2. **不可回答问题评估偏差**：数据集中的不可回答问题有固定模式（多指模型无法获取的材料），可能不反映真实场景
3. **计算资源需求高**：约 3600 GPU 小时的实验成本限制了更大规模的探索
4. **未探索微调**：如果对 InternVL3 或 Pixtral 进行轻量微调，性能可能进一步提升
5. **检索策略较基础**：可以考虑更复杂的示例选择方法（如基于难度的检索、图表结构感知的检索）

## 相关工作与启发

- **VQA v2, DocVQA, ChartQA, PlotQA** 等不同类型的 VQA 数据集，本文聚焦科学图表
- **InternVL3-78B, Pixtral-Large** 作为开源 MLLM 的代表，展示了大模型在科学图表理解上的能力
- **SBERT, CLIP, BLIP-2** 分别用于文本相似度、跨模态相似度和多模态嵌入检索
- 启发：模型校准（calibration）在实际系统部署中的价值——可用于构建「先易后难」的级联推理pipeline

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3 |
| 实用性 | 4 |
| 实验完整度 | 5 |
| 写作清晰度 | 4 |
| 总评 | 3.5 |

方法较为工程化，但系统设计合理、实验扎实，特别是对不同配置的详尽对比和置信度校准分析具有很好的参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[CVPR 2026\] CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis](../../CVPR2026/multimodal_vlm/cica_coupling_confidence-aware_pretraining_with_confidence-informed_attention_fo.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](../../CVPR2025/multimodal_vlm/rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ACL 2025\] SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation](semeval-2025_task_1_admire_--_advancing_multimodal_idiomaticity_representation.md)

</div>

<!-- RELATED:END -->

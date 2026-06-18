---
title: >-
  [论文解读] To See or To Read: User Behavior Reasoning in Multimodal LLMs
description: >-
  [NeurIPS 2025][多模态VLM][用户行为推理] 提出BehaviorLens基准框架，系统比较文本、散点图和流程图三种用户行为历史的表示方式对MLLM次购预测的影响，发现图像表示相比等效文本表示最高可提升87.5%的预测准确率，且无需额外计算开销。 多模态大语言模型（MLLM）正在重塑智能系统对用户行为序列数据…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "用户行为推理"
  - "模态权衡"
  - "序列推荐"
  - "视觉表示"
  - "BehaviorLens"
---

# To See or To Read: User Behavior Reasoning in Multimodal LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2511.03845](https://arxiv.org/abs/2511.03845)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 用户行为推理, 模态权衡, 序列推荐, 视觉表示, BehaviorLens

## 一句话总结
提出BehaviorLens基准框架，系统比较文本、散点图和流程图三种用户行为历史的表示方式对MLLM次购预测的影响，发现图像表示相比等效文本表示最高可提升87.5%的预测准确率，且无需额外计算开销。

## 研究背景与动机

多模态大语言模型（MLLM）正在重塑智能系统对用户行为序列数据的推理方式，但一个关键且被忽视的问题是：序列化的用户历史应该如何表示才能同时优化推理准确率和计算效率？

传统方法将用户交互历史（点击、浏览、购买）展平为逐行文本描述输入MLLM，这虽然保留了细粒度信息，但丢失了用户旅程的结构信息（如拓扑模式），可能导致模型无法掌握整体的用户叙事，降低意图预测质量。

核心问题是：文本还是图像表示对于最大化MLLM在用户行为推理上的表现更有效？这在代理推荐系统中尤为关键，因为理解用户旅程和转化直接影响个性化和营收。

本文的切入角度是：通过系统性基准测试，在控制条件下直接比较同一用户历史的文本和视觉表示，评估预测准确率和计算成本。

## 方法详解

### 整体框架

BehaviorLens框架将用户购买历史转换为三种表示形式输入MLLM：(1) 文本序列——自然语言描述每次购买事件；(2) 散点图——将时间和商品映射到二维坐标系；(3) 流程图——以节点和有向边表示购买序列的时间邻近关系。然后MLLM基于每种表示进行次购预测和推理解释。

### 关键设计

1. **文本顺序表示**: 将每个交互三元组 $(a, i, e)$（行为、商品、环境/时间）表示为自然语言描述 $\phi_{text}(a,i,e) = \text{"item \{i\} was \{a\} at timestamp \{e\}"}$，购买历史是所有事件描述的拼接。这是现有用户建模的标准方案。

2. **散点图表示**: 受时间序列建模中可视化方法启发，将购买历史转化为散点图图像。使用排名函数 $r(\cdot)$ 将商品和时间映射到坐标系：$\phi_{scatter-plot}(a,i,e) = \text{plot}(a,i|x=r(e), y=r(i))$。这种表示能直观显示购买的时间模式和商品分布。

3. **流程图表示**: 受"视觉压缩有助于LLM在结构化任务中推理"的研究启发，将购买历史转化为有向流程图。每个购买节点连接其时间上的前驱和后继节点：$\phi_{flowchart}(a,i,e) = \text{node}(a,i,e|p=\{node(a_{-1},i_{-1},e_{-1})\}, s=\{node(a_{+1},i_{+1},e_{+1})\})$。这种表示强调序列结构和时间邻近关系。

### 损失函数 / 训练策略

本文不涉及模型训练，而是纯评测工作。使用6个MLLM（Gemini-2.0-flash-lite、Gemini-2.0-flash、Gemini-2.5-flash-lite、Gemini-2.5-flash、GPT-4o、GPT-4.1-mini）在真实购买序列数据集上评估，通过LLM-as-a-Judge评估推理解释质量。

## 实验关键数据

### 主实验

| MLLM模型 | 输入类型 | 准确率 | 相似度 | Token数 | 延迟(秒) |
|----------|---------|--------|--------|---------|---------|
| Gemini-2.5-flash-lite | 文本 | 0.360 | 0.570 | 1220 | 1.444 |
| Gemini-2.5-flash-lite | 散点图 | **0.530** | **0.689** | 3623 | 2.057 |
| GPT-4o | 文本 | 0.420 | 0.602 | 1106 | 5.451 |
| GPT-4o | 散点图 | **0.560** | **0.713** | 1169 | 8.954 |
| GPT-4o | 流程图 | 0.300 | 0.527 | 1043 | 7.140 |
| GPT-4.1-mini | 文本 | 0.320 | 0.542 | 1105 | 4.680 |
| GPT-4.1-mini | 散点图 | **0.600** | **0.726** | 1039 | 7.849 |
| GPT-4.1-mini | 流程图 | 0.340 | 0.563 | 862 | 6.051 |

### 消融实验（推理解释质量对比）

| 评估维度 | 说明 | 差异 |
|---------|------|------|
| 忠实度(Faithfulness) | 推理是否忠实于输入数据 | 跨输入类型无显著差异 |
| 过度思考(Overthinking) | 是否过度推理 | 跨输入类型无显著差异 |
| 因果性(Causality) | 推理的因果逻辑 | 跨输入类型无显著差异 |
| 充分性(Sufficiency) | 推理是否充分 | 跨输入类型无显著差异 |
| 特异性(Specificity) | 推理是否针对特定用户 | 跨输入类型无显著差异 |
| 合理性(Plausibility) | 推理是否合理 | 跨输入类型无显著差异 |

### 关键发现

- 在6个MLLM中，除Gemini-2.5-flash外，散点图或流程图表示均优于文本表示。
- 最大提升：GPT-4.1-mini上散点图vs文本的准确率提升87.5%（0.320→0.600），相似度提升33.9%。
- GPT模型在不同表示下token消耗基本一致，说明图像表示不增加计算成本。
- Gemini模型中散点图消耗约3倍token，但流程图和文本消耗相近。
- 推理解释质量在不同输入类型间无显著差异（除Gemini-2.0-flash外），说明**是输入表示本身而非中间推理质量驱动了预测改善**。
- 案例分析显示：文本输入侧重购买频率，流程图侧重最近购买的互补性，散点图能更好捕捉循环性购买模式。

## 亮点与洞察

- 发现极具实用价值且反直觉：图像比文本更适合表示序列化用户行为，这挑战了"文本是MLLM最佳输入"的默认假设。
- 框架设计简洁——无需任何训练或微调，纯粹通过改变输入表示即可获得巨大提升。
- 将时间序列可视化技术引入用户行为建模是一个很好的跨领域方法迁移。
- 推理质量不变但预测准确率大幅提升的发现表明：MLLM对不同模态的信息提取效率存在根本差异。

## 局限与展望

- 仅在次购预测任务上验证，其他用户行为推理任务（如意图识别、流失预测）待探索。
- 数据集规模和多样性有限（基于单一电商场景）。
- 未探索更长的用户行为序列对结果的影响。
- 视觉表示的设计空间远未穷尽——如热力图、时间线图等其他可视化方式未被测试。
- 异常值Gemini-2.5-flash（图像反而更差）的原因未深入分析。
- 未考虑多模态组合输入（同时提供文本和图像）的效果。

## 相关工作与启发

- 与序列推荐方法（GRU4Rec、SASRec、BERT4Rec等）不同，本文将推荐问题转化为MLLM的推理任务而非端到端训练。
- Li & Jiang (2025) 关于"视觉压缩有助于LLM结构化推理"的发现为流程图表示提供了理论基础。
- Wang & Oates (2015) 的时间序列图像化方法启发了散点图表示。
- 启发：在设计MLLM应用时，不应默认使用文本输入——根据任务特性选择合适的模态表示可能带来巨大收益。

## 评分

- 新颖性: ⭐⭐⭐⭐ （切入角度新颖，发现反直觉且实用）
- 实验充分度: ⭐⭐⭐ （6个模型×3种表示的对比充分，但数据集单一，规模较小）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，形式化定义规范）
- 价值: ⭐⭐⭐⭐ （对推荐系统和代理系统设计有直接指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](enhancing_compositional_reasoning_in_clip_via_reconstruction.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ACL 2026\] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding](../../ACL2026/multimodal_vlm/do_mllms_capture_how_interfaces_guide_user_behavior_a_benchmark_for_multimodal_u.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](../../ACL2025/multimodal_vlm/cant_see_the_forest_for_the.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->

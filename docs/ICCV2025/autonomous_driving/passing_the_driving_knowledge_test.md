---
title: >-
  [论文解读] Passing the Driving Knowledge Test
description: >-
  [ICCV 2025][自动驾驶][Driving Knowledge Test] 构建DriveQA——首个大规模文本+视觉双模态驾驶知识测试基准（26K文本QA + 448K图像QA），系统评估LLM/MLLM在交通规则、标志识别和路权判断等驾驶知识上的能力，揭示其在数值推理和复杂路权场景中的显著不足，并展示DriveQA预训练对下游驾驶任务的泛化增益。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "Driving Knowledge Test"
  - "LLM"
  - "MLLM"
  - "VQA"
  - "Traffic Rules"
  - "benchmark"
  - "Fine-tuning"
---

# Passing the Driving Knowledge Test

**会议**: ICCV 2025  
**arXiv**: [2508.21824](https://arxiv.org/abs/2508.21824)  
**代码**: [driveqaiccv.github.io](https://driveqaiccv.github.io)  
**领域**: Autonomous Driving / 驾驶知识问答评估  
**关键词**: Driving Knowledge Test, LLM, MLLM, VQA, Traffic Rules, benchmark, Fine-tuning

## 一句话总结

构建DriveQA——首个大规模文本+视觉双模态驾驶知识测试基准（26K文本QA + 448K图像QA），系统评估LLM/MLLM在交通规则、标志识别和路权判断等驾驶知识上的能力，揭示其在数值推理和复杂路权场景中的显著不足，并展示DriveQA预训练对下游驾驶任务的泛化增益。

## 研究背景与动机

安全驾驶不仅要求视觉感知，还需对交通规则进行推理和决策。人类驾驶员必须通过书面知识测试才能获得驾照。然而，现有自动驾驶基准和多模态LLM评估存在关键缺口：

**现有基准聚焦感知和基础规划**：nuScenes-QA、DriveLM等关注空间理解和碰撞避免，很少评估对交通规则的理解——如限速标志、右转优先权、罕见标识

**长尾规则覆盖不足**：真实驾考覆盖大量边缘案例（如特殊施工标志、复杂路口避让规则），这些在实际驾驶数据中极少出现

**MLLM的交通知识有限**：虽然MLLM从预训练数据中可能继承部分交通知识，但实验表明这种知识和推理能力仍然有限

**商业系统的实际问题**：Tesla FSD等系统在交通规则解读上常出现错误的实例证据

DriveQA的目标是：如果让LLM今天参加驾照考试，它能通过吗？

## 方法详解

### 数据集构建

**DriveQA-T（文本QA）**：
- 26K文本QA对，覆盖5大类19子类（交通灯、交通标志、停车、法规、符号等）
- 数据来源：收集美国全部50州+DC共51本官方驾驶手册
- 构建流程：GPT-4o基于手册内容自动生成问题 → 人工质量验证 → 多轮审核删除歧义或不一致条目
- 每个QA对附带答案解释，用于评估推理能力

**DriveQA-V（视觉QA）**：
- 68K图像、448K VQA对
- **交通标志方面**：向CARLA模拟器中插入220种美国交通标志3D模型，控制视角（前方/斜视/俯视）、天气、时间、距离等变量
- **路权判断方面**：在CARLA地图中识别路口，随机生成不同颜色车辆，构造路权判断场景
- **真实数据**：从Mapillary收集标注真实世界数据补充

### 评估方法

1. **问题类型分类**：使用BERT嵌入+层次聚类将问题分为19种语义类别，配合KeyBERT提取关键词
2. **提示策略**：设计4种prompt——基础、CoT（思维链）、RAG（从驾驶手册检索增强）、CoT+RAG
3. **微调**：使用LoRA低秩适配进行高效微调
4. **损失函数**：标准交叉熵用于多选题分类

### 关键评估维度

- 文本QA准确率（跨19类）
- CoT推理质量（BLEU-4、ROUGE-L）
- 视觉QA准确率（交通标志识别、路权判断）
- 环境因素敏感性（视角、天气、时间）
- 下游任务迁移（nuScenes、BDD轨迹预测）

## 实验

### 主实验：LLM在DriveQA-T上的表现

| 模型 | 大小 | CoT | RAG | FT | 速度限制 | 停车 | 路口 | 平均 |
|------|------|-----|-----|----|----|------|------|------|
| Gemma-2 | 2B | | | | 42.2 | 35.6 | 27.9 | 44.2 |
| Gemma-2 | 9B | ✓ | ✓ | | 64.9 | 68.3 | 77.9 | 76.9 |
| Llama-3.1 | 8B | ✓ | ✓ | ✓ | 72.7 | 86.1 | **91.6** | **87.6** |
| Phi-3.5-mini | 3.8B | | | | 49.2 | 48.5 | 79.7 | 69.8 |
| Phi-3.5-mini | 3.8B | ✓ | ✓ | ✓ | 66.9 | 65.4 | 87.2 | 81.1 |
| **GPT-4o** | - | ✓ | ✓ | | **76.7** | **93.8** | **97.3** | **92.0** |

**关键发现**：
- 开源模型在基础交通规则上表现尚可，但在**数值推理**（速度/距离限制）和复杂**路权场景**上显著薄弱
- CoT+RAG持续提升性能，表明交通知识的检索增强至关重要
- 微调后开源模型大幅提升，Llama-3.1达87.6%，接近GPT-4o（92.0%）

### 消融实验：MLLM在DriveQA-V上的表现

| 模型 | 大小 | T字路口(前) | 十字路口(前) | 管制标志 | 警告标志 | 平均 |
|------|------|------------|------------|---------|---------|------|
| Mini-InternVL | 2B（原始） | 27.8 | 26.0 | 64.1 | 55.3 | 41.8 |
| Mini-InternVL | 2B（微调） | **86.7** | **74.3** | **93.8** | **92.2** | **86.6** |
| LLaVA-1.6 | 7B（原始） | 18.8 | 31.0 | 42.6 | 43.0 | 34.5 |
| LLaVA-1.6 | 7B（微调） | 86.1 | 74.4 | 82.1 | 84.1 | 83.7 |
| **GPT-4o** | -（零样本） | 55.1 | 50.5 | **93.8** | **94.0** | **75.3** |

**关键发现**：
- 未微调的MLLM路权判断准确率接近随机猜测（~25%），标志识别稍好但远不够
- 微调效果显著：Mini-InternVL微调后从41.8%→86.6%，接近GPT-4o的75.3%
- 最难的10种标志类型多为管制和警告标志（如Playground、Trauma Center等）

### CoT推理质量评估

| 模型 | BLEU-4(w/o RAG) | ROUGE-L(w/o RAG) | BLEU-4(w/ RAG) | ROUGE-L(w/ RAG) |
|------|----------------|-----------------|----------------|-----------------|
| Gemma-2(9B,FT) | **0.4112** | 0.5420 | 0.4105 | **0.5528** |
| GPT-4o | 0.3905 | 0.5354 | 0.3989 | 0.5393 |

微调后的Gemma-2(9B)在推理质量上甚至超过GPT-4o。

## 亮点与洞察

1. **首个全面驾驶知识基准**：DriveQA是唯一同时覆盖交通规则、标志、路权的多模态基准，填补了LLM/MLLM驾驶推理评估的空白
2. **可控合成数据的价值**：利用CARLA程序化生成大量受控变化（视角、天气、标志类型），DriveQA-V预训练能提升真实世界下游任务性能
3. **知识与推理的差距**：即使最强的GPT-4o在路权判断（~60%）和速度限制推理（~77%）上仍远低于人类的100%，说明规则推理比简单模式识别难得多
4. **微调的有效性**：仅用LoRA微调即可大幅弥补预训练知识不足，暗示交通规则知识在预训练语料中覆盖不足是主因

## 局限性

1. 仅聚焦美国交通规则，不同国家/地区的交通法规差异未覆盖
2. 图像数据以合成为主（CARLA），与真实场景仍有域差距
3. 评估以多选题为主，未涉及开放式驾驶决策推理
4. 仅评估了有限数量的开源MLLM，更大规模模型（如Gemini Pro、Claude）未纳入

## 相关工作

- **MLLM驾驶代理**：DriveGPT4、DriveLM、EMMA等将LLM用于驾驶决策但聚焦规划而非规则理解
- **驾驶VQA数据集**：NuScenes-QA、DriveBench、LingoQA关注空间感知而非法规推理
- **交通标志识别**：GTSRB等传统基准，但无推理要求
- **CoT与RAG**：思维链和检索增强在复杂推理任务中有效

## 评分

- 新颖性：⭐⭐⭐⭐⭐（首次系统化评估LLM的"驾考"能力，新颖且实用的视角）
- 技术深度：⭐⭐⭐（更偏benchmark贡献，方法设计相对标准）
- 实验完整度：⭐⭐⭐⭐⭐（覆盖文本&视觉QA、多模型、多策略、下游迁移验证）
- 实用价值：⭐⭐⭐⭐⭐（直接揭示当前MLLM在安全关键场景的知识盲点）
- 总体推荐：⭐⭐⭐⭐（优秀的benchmark工作，但方法创新有限）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](../../CVPR2026/autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/test-time_3d_occupancy_prediction.md)
- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)

</div>

<!-- RELATED:END -->

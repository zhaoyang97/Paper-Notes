---
title: >-
  [论文解读] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning
description: >-
  [ICCV 2025][多模态VLM][图表推理] 提出PointCoT方法，将反思性视觉定位（bounding box）集成到图表推理的思维链中，使MLLM在每个推理步骤都能与图表视觉内容交互验证，并构建了包含19.2K高质量样本的ChartPoint-SFT-62k数据集，在ChartBench上实现+5.04%的提升。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "图表推理"
  - "多模态大模型"
  - "Chain-of-Thought"
  - "视觉定位"
  - "数值幻觉"
---

# ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning

**会议**: ICCV 2025  
**arXiv**: [2512.00305](https://arxiv.org/abs/2512.00305)  
**代码**: 无  
**领域**: 图表理解 / 多模态推理  
**关键词**: 图表推理, 多模态大模型, Chain-of-Thought, 视觉定位, 数值幻觉

## 一句话总结

提出PointCoT方法，将反思性视觉定位（bounding box）集成到图表推理的思维链中，使MLLM在每个推理步骤都能与图表视觉内容交互验证，并构建了包含19.2K高质量样本的ChartPoint-SFT-62k数据集，在ChartBench上实现+5.04%的提升。

## 研究背景与动机

多模态大语言模型（MLLM）在图表理解中严重依赖OCR提取的文本信息。当图表的文本标注稀疏时（如数据点没有标注具体数值），模型往往产生严重的数值幻觉——即使推理步骤看似合理，提取的数字仍包含显著错误。

作者发现了一个关键观察：**MLLM在图表元素和比例关系上的定位能力极弱**。当提示模型指出每个推理步骤对应的图表位置时，模型要么忽略请求，要么生成完全无关的坐标。这说明：
- 传统CoT虽然增强了基于数字的逻辑推理，但未能提升模型的基础数值感知能力
- CoT虽生成更多推理token，但未能实现与图表视觉token的额外交互
- 模型缺乏人类读图时"看-指-读-算"的视觉推理逻辑

这促使作者将定位反思引入推理链：模型不仅需要说出推理步骤，还需通过输出bounding box指明其正在关注图表的哪个区域，并通过重新渲染的图表进行验证。

## 方法详解

### 整体框架

PointCoT数据构建流水线包含四个阶段：
1. **步骤分解**（Step Decomposition）：LLM生成数值问题和CoT推理步骤
2. **代码编辑**（Code Editing）：LLM修改绘图代码，在关键位置插入特殊字符
3. **代码渲染**（Code Rendering）：执行修改后的代码重新渲染图表
4. **位置定位**（Position Localization）：OCR检测特殊字符位置，提取bounding box

### 关键设计

#### 1. 结构化推理构建（Structured Reasoning）
- **功能**：将图表问答的推理过程分解为"Grounding"和"Reasoning"两类步骤
- **核心思路**：使用Qwen2.5-72B作为教师模型，基于绘图代码生成数据点相关的问题和逐步推理过程。每个子步骤被分类为：
    - **Grounding步骤**：需要从图表中提取数据（如定位坐标轴上的点、图例条目）
    - **Reasoning步骤**：基于前序grounding步骤的信息进行逻辑推理
- **设计动机**：图表阅读的思维过程天然具有结构性——人类先识别关键位置，再进行数值推理。这种结构不是人为强加的，而是源自图表阅读的内在逻辑

#### 2. 点标注构建（Point Annotation via Code Editing）
- **功能**：为每个grounding步骤生成精确的bounding box标注
- **核心思路**：不直接使用MLLM定位（不可靠），而是利用"图表-代码对"的优势：
  1. 教师模型识别每个grounding步骤对应的图表元素/位置
  2. 修改绘图代码，在关键位置插入特殊字符标记（通过`plt.text()`）
  3. 执行修改后的代码渲染新图表
  4. 使用多种OCR工具检测特殊字符位置，提取bounding box
- **设计动机**：LLM修改代码的成功率远高于MLLM直接定位图表元素，利用代码作为中介实现精确的位置标注

#### 3. 四种指令数据格式
- **Type 1 - 标准VQA**：原始图表+问题，监督信号为答案或CoT+答案（不含bbox以防数据泄露）
- **Type 2 - 定位任务**：将中间步骤加入查询提示，ground truth变为预测的bounding box
- **Type 3 - 编辑图表推理**：将前序grounding步骤的bbox标注重绘到原始图表上，引导模型关注正确区域
- **Type 4 - 推理步骤**：直接将reasoning步骤加入查询提示，最终监督信号为最终答案

最终构建ChartPoint-SFT-62k：19.2K图表 × 62.3K指令数据

### 损失函数 / 训练策略

两阶段全参数微调：
- **Stage 1 - 图表知识对齐**：使用MMC-Instruct(410K) + ChartGemma(160K) + ChartQA(28K) + ChartBench(30K)
- **Stage 2 - 图表特定退火调优**：使用ChartPoint-SFT-62k进行PointCoT方式的指令微调

训练细节：AdamW优化器，warmup lr=5e-5，权重衰减0.1，梯度裁剪1.0，等效batch size 64，bfloat16精度，约262 GPU小时（A100-40G）。坐标归一化到0-999范围。

## 实验关键数据

### 主实验

ChartQA relaxed accuracy@0.05：

| 模型 | 参数 | Human | Aug. | 平均 |
|------|------|-------|------|------|
| Qwen2-VL | 7B | 72.08 | 94.24 | 83.16 |
| Qwen2.5-VL | 7B | 78.96 | 93.76 | 86.36 |
| ChartMoE+PoT | 8B | 78.32 | 90.96 | 84.64 |
| **ChartPoint_Q2** | **7B** | 76.12 | **94.48** | **85.28** |
| **ChartPoint_Q2.5** | **7B** | **81.36** | 94.12 | **87.74** |

ChartBench准确率：

| 模型 | Regular类型 | Extra类型 | 总体 |
|------|------------|----------|------|
| Qwen2-VL | 58.36 | 59.40 | 58.90 |
| Qwen2.5-VL | 62.73 | 57.26 | 60.91 |
| ChartMoE | 56.31 | 55.58 | 51.67 |
| **ChartPoint_Q2** | **63.04** | **62.09** | **62.61** |
| **ChartPoint_Q2.5** | **66.71** | **65.03** | **65.95** |

### 消融实验

训练策略消融（基于Qwen2-VL）：

| 配置 | ChartQA | ChartBench | 说明 |
|------|---------|------------|------|
| Baseline (Qwen2-VL) | 83.16 | 58.90 | 原始模型 |
| +Stage1 | 83.74 | 60.39 | 图表知识对齐 |
| +Stage1+CoT | 84.11 | 60.76 | 文本CoT蒸馏 |
| **+Stage1+PointCoT** | **85.30** | **62.61** | 带定位的CoT |

坐标格式消融：

| 格式 | 归一化 | Human | 总体 | 说明 |
|------|--------|-------|------|------|
| Type A | [0-1], 4位小数 | 73.52 | 83.68 | 连续坐标 |
| Type B | [0-1], 3位小数 | 74.68 | 84.42 | 精度降低 |
| **Type C** | **[0-999], 整数** | **75.36** | **84.84** | 适配tokenizer |

### 关键发现

1. PointCoT在ChartBench上的提升（+3.71%/+4.28%）远大于ChartQA（+1.56%/+1.22%），因为ChartBench无数据点文本标注，更依赖视觉定位能力
2. 文本CoT蒸馏的提升有限（+0.37%），因为推理过程由LLM生成而非MLLM，未利用图表视觉信息
3. 在Extra类型图表（面积图、箱线图、雷达图等不常见类型）上PointCoT提升更显著（+7.77%），说明视觉推理逻辑具有泛化性
4. 坐标归一化到[0-999]整数比连续浮点数效果更好，因为更适配tokenizer

## 亮点与洞察

- **关键诊断**：精确指出了MLLM图表理解的核心瓶颈——不是逻辑推理能力不足，而是视觉感知（数值读取）能力弱
- **思路创新**：将"定位反思"引入CoT，使推理的每一步都能与视觉证据关联验证，而非纯文本推理
- **数据构建巧妙**：利用图表-代码对的对应关系，通过代码修改间接实现精确位置标注，避免了直接让MLLM定位的不可靠性
- **质量控制严格**：每步都有成功率追踪（96%→76%→51%→77%），最终通过三位专家审核91%达标

## 局限与展望

- 数据构建流水线成功率仅约28%（66.8K→19.2K），限制了数据规模的扩展
- 仅覆盖柱状图(57.1%)、折线图(33.6%)和饼图(9.3%)三种类型
- 问答集中在数据点读取，未涉及复杂的数值计算或多步推理
- 依赖基线模型（Qwen2-VL/2.5-VL）的定位能力，对不支持bbox的模型不适用
- 未探索推理时的scaling law（如beam search over location/reasoning steps）

## 相关工作与启发

- **MVoT**在拼图等结构化场景中也观察到类似现象——推理步骤需要与视觉输入交互
- **ChartMoE**提供了chart-code对的元数据基础，本工作创新性地利用这些数据进行位置标注
- 该思路可推广到其他需要精确视觉感知的推理任务（如科学图表、地图、工程图纸）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将视觉定位反思融入图表CoT，诊断了MLLM的核心瓶颈
- 实验充分度: ⭐⭐⭐⭐ 消融全面，跨模型验证充分，但数据集类型单一
- 写作质量: ⭐⭐⭐⭐ 流水线描述清晰，但部分公式符号不够一致
- 价值: ⭐⭐⭐⭐⭐ 为图表理解社区提供了新范式——从"文本推理"转向"视觉推理"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ACL 2025\] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](../../ACL2025/multimodal_vlm/chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown](taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co.md)

</div>

<!-- RELATED:END -->

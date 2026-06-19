---
title: >-
  [论文解读] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models
description: >-
  [ICCV 2025][多模态VLM][图表分析] GRAB 是一个面向大型多模态模型（LMM）的图表分析基准测试，包含 3284 道合成题目覆盖 5 个任务和 23 个图形属性，当前最强模型 Claude 3.5 Sonnet 仅达到 21.0% 的准确率，揭示了 LMM 在视觉分析推理方面的严重不足。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "图表分析"
  - "benchmark"
  - "大型多模态模型"
  - "合成数据"
  - "视觉推理"
---

# GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models

**会议**: ICCV 2025  
**arXiv**: [2408.11817](https://arxiv.org/abs/2408.11817)  
**代码**: [https://grab-benchmark.github.io](https://grab-benchmark.github.io)  
**领域**: 多模态VLM  
**关键词**: 图表分析, benchmark, 大型多模态模型, 合成数据, 视觉推理

## 一句话总结

GRAB 是一个面向大型多模态模型（LMM）的图表分析基准测试，包含 3284 道合成题目覆盖 5 个任务和 23 个图形属性，当前最强模型 Claude 3.5 Sonnet 仅达到 21.0% 的准确率，揭示了 LMM 在视觉分析推理方面的严重不足。

## 研究背景与动机

### 领域现状与痛点

大型多模态模型（LMM）的能力正在飞速提升，但现有基准测试正快速饱和。GPT-4o 在 MGSM、HumanEval、MMLU 等主流基准上得分已超 88-90%，意味着这些测试已难以区分模型之间的能力差距。与此同时，现有基准中标签错误据报道十分普遍，进一步压缩了可用的评估空间。

**图表分析是核心应用场景**：科学和数学图表的解读是很多分析工作的基础。在很多场景中，底层数据不可获取（如文档中的图、手绘草图），只能通过视觉解读来推断数值。这对 LMM 的精确视觉推理能力提出了很高要求。

**现有图表基准的不足**：
- 已有基准（如 ChartQA、MathVista）难度不够，GPT-4o 在 MathVista 上已超 60%
- 很多题目侧重于 OCR 类简单任务（读图例、轴标签），未考验真正的分析推理能力
- 标注质量参差不齐，人工标注引入噪声

### 核心矛盾与切入角度

作者认为，下一代基准需要具备三个关键属性：**足够的难度**（当前最强模型仍有大量上升空间）、**无噪声的高质量标注**、以及**抵抗数据污染的能力**。合成数据是满足这些要求的最佳途径——它可以精确控制题目难度、自动生成无噪声答案、且不太可能出现在预训练数据中。

基于这一思路，作者设计了 GRAB，一个以合成图表为主、覆盖广泛图表分析能力的挑战性基准。

## 方法详解

### 整体框架

GRAB 由 3284 道题目组成，覆盖 5 个核心任务和 23 个图形属性。所有合成图表均使用 Matplotlib 和 Seaborn 库生成。数据集还包含一个 500 题的轻量版 GRAB-Lite 便于快速评估。

### 关键设计

#### 1. 图形属性分类体系（23 个属性，9 大类）

- **截距与梯度**：x 截距、y 截距、梯度
- **驻点**：驻点坐标
- **三角函数**：振幅、垂直偏移、周期
- **函数方程**：函数表达式识别
- **计数**：点数、序列数
- **相关性**：Pearson、Spearman、Kendall 相关系数
- **有界面积**：总有界面积、净有界面积
- **离散度指标**：均值、中位数、四分位距、方差
- **范围与极值**：最大/最小值、域长度、值域

设计动机：覆盖分析师在解读图表时可能执行的大部分典型任务，刻意排除简单的 OCR 类问题（读标题/图例）以聚焦于视觉分析推理。

#### 2. 五大任务划分

- **Properties（660 题）**：单函数/序列的属性推导，作为基础任务
- **Functions（710 题）**：多达 10 条函数的属性均值计算，函数重叠增加难度
- **Series（490 题）**：多达 10 个数据序列的属性均值计算，数据噪声增加难度
- **Transforms（310 题）**：对单函数施加最多 10 次变换（旋转、平移、缩放、反射）后求属性
- **Real（1114 题）**：增加真实场景元素——手绘白板图、纸上草图、截图嵌入（邮件/PPT/视频会议）、添加噪声（模糊/翻转/伪影）

设计动机：从简单到复杂渐进式评估，Real 任务测试模型面对真实场景退化的鲁棒性。

#### 3. 数据生成与质量控制

- **生成流程**：为每个图形属性初始生成 250 道候选题，然后下采样保证答案均匀分布，避免答案集中在 0 附近的偏差
- **精度设计**：约 75% 的题目要求整数精度答案，~25% 要求 1 位小数精度以增加难度
- **图表美学**：Properties 任务随机采样图表外观参数，其他任务统一外观以控制变量
- **质量控制**：经过多轮人工审查，确保题目可回答、答案正确、图表清晰可读

#### 4. 评估协议

- **严格精确匹配**：不做宽松后处理，输出必须与标准答案完全一致
- **联合评估任务能力和指令遵循能力**：冗余回答（如"The answer is..."）会被判错
- 设计动机：如果模型能推理但不能精确输出，在实际应用中同样无用

### 损失函数 / 训练策略

GRAB 是评估基准，不涉及模型训练。

## 实验关键数据

### 主实验

| 模型 | Properties | Functions | Series | Transforms | Real | 总分 |
|------|-----------|-----------|--------|-----------|------|------|
| Claude 3.5 Sonnet | 41.8 | 15.5 | 11.0 | 10.0 | 19.6 | **21.0** |
| Gemini 1.5 Pro | 34.2 | 11.4 | 13.3 | 6.5 | 20.3 | 18.8 |
| Gemini 1.5 Flash | 28.5 | 11.5 | 8.4 | 9.0 | 17.1 | 16.1 |
| GPT-4o | 24.7 | 10.8 | 9.2 | 3.5 | 17.3 | 14.9 |
| GPT-4 Turbo | 18.5 | 8.5 | 4.9 | 3.5 | 7.5 | 9.2 |
| LLaVA-1.5 13b | 5.0 | 7.7 | 8.4 | 3.9 | 8.9 | 7.3 |
| CogVLM-Chat | 7.0 | 4.9 | 5.1 | 3.9 | 10.5 | 7.2 |

所有 20 个被评估的 LMM 表现都极差，最强的 Claude 3.5 Sonnet 仅 21.0%，大多数开源模型表现接近随机水平。

### 消融实验（任务难度分析）

| Real 子集 | Whiteboard | Paper | Screenshots | Noise | 总分 |
|----------|-----------|-------|------------|-------|------|
| Claude 3.5 Sonnet | 14.6 | 17.0 | 18.6 | 21.1 | 19.6 |
| GPT-4o | 29.3 | 9.8 | 18.0 | 16.3 | 17.3 |
| Gemini 1.5 Pro | 34.4 | 24.4 | 21.9 | 17.2 | 20.3 |
| Gemini 2.5 Flash | 36.6 | 22.0 | 29.5 | 29.3 | 29.4 |

| 类别 | Claude 3.5 Sonnet | GPT-4o | Gemini 1.5 Pro | 说明 |
|------|------------------|--------|----------------|------|
| Counting | 30.0 | 30.0 | 33.3 | 最简单的类别 |
| Intercepts & Gradients | 25.5 | 14.1 | 17.9 | 中等难度 |
| Correlation | 9.2 | 15.8 | 26.7 | 难度较高 |
| Area Bounded | 2.7 | 2.0 | 4.7 | 几乎全败 |
| Functions | 0.0 | 0.0 | 0.0 | 所有模型得分为0 |

### 关键发现

- **Transforms 任务最难**：所有模型在此任务上表现最差，说明 LMM 难以进行多步视觉变换推理
- **Functions 类别全军覆没**：没有任何模型在函数方程识别类别上得到非零分数
- **Area Bounded 类别极难**：需要复杂的积分估计，所有模型接近零分
- **闭源模型全面优于开源模型**：除 Reka Core 外，所有闭源模型都优于最好的开源模型
- **Real 任务表现与其他任务相当**：说明噪声/干扰对性能影响不大，核心瓶颈是推理能力而非图像质量
- **指令遵循能力是得分因素之一**：部分模型虽然计算正确但因输出格式不符被判错
- **Gemini 代际进步显著**：从 1.5 Flash 到 2.0 Flash 到 2.5 Flash，性能持续提升

## 亮点与洞察

- **前瞻性设计**：为下一代模型预留了大量提升空间，当前模型仅达 21%，5年内可能仍有价值
- **合成数据的方法论贡献**：展示了如何通过合成数据构建高质量、可控、抗污染的基准
- **精确匹配评估的哲学**：指令遵循能力与推理能力同等重要的评估理念值得借鉴
- **全面揭示 LMM 短板**：图表分析这一看似简单的任务，竟然是当前最强模型的巨大盲区

## 局限与展望

- 合成数据可能与真实世界图表存在分布差异（Matplotlib 风格很固定）
- 严格精确匹配可能低估了某些模型的真实推理能力
- 未考虑 chain-of-thought 提示等高级推理策略的效果
- Real 任务中手绘图数量有限（仅 41 张白板 + 41 张纸），代表性不足
- 未纳入更新的模型（如 GPT-4.5、Claude 4 等）
- 缺乏对模型失败模式的系统分类分析

## 相关工作与启发

- **MathVista**：覆盖更广泛的数学推理，但难度不够（GPT-4o 超 60%），GRAB 聚焦图表分析且难度更高
- **ChartQA、PlotQA**：已有图表理解基准，但已接近饱和
- **FigureQA**：二分类图表理解任务，难度太低
- 对 LMM 社区的启示：精确数值推理是比 VQA 更难的挑战，需要专门的能力提升

## 评分

- 新颖性: ⭐⭐⭐⭐ — 基准本身设计精良，但"构建高难度基准"不算全新思路
- 实验充分度: ⭐⭐⭐⭐⭐ — 20个模型、详细的类别分析和消融
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，图表丰富，论述严谨
- 价值: ⭐⭐⭐⭐ — 对理解和推动 LMM 发展有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](../../CVPR2025/multimodal_vlm/mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

</div>

<!-- RELATED:END -->

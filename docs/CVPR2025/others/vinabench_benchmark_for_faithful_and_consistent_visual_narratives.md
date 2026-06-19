---
title: >-
  [论文解读] VinaBench: Benchmark for Faithful and Consistent Visual Narratives
description: >-
  [CVPR 2025][视觉叙事] 构建了 VinaBench 基准，为视觉叙事样本标注常识链接和话语约束，提出忠实度和一致性评估指标，并验证利用这些约束可显著提升视觉叙事生成的质量。 领域现状：视觉叙事生成（Visual Narrative Generation）是将文本叙事转换为图像序列的任务，广泛应用于影视分镜、教育插…
tags:
  - "CVPR 2025"
  - "视觉叙事"
  - "一致性评估"
  - "常识约束"
  - "话语结构"
  - "基准数据集"
---

# VinaBench: Benchmark for Faithful and Consistent Visual Narratives

**会议**: CVPR 2025  
**arXiv**: [2503.20871](https://arxiv.org/abs/2503.20871)  
**代码**: [https://silin159.github.io/Vina-Bench](https://silin159.github.io/Vina-Bench)  
**领域**: LLM评测  
**关键词**: 视觉叙事, 一致性评估, 常识约束, 话语结构, 基准数据集

## 一句话总结

构建了 VinaBench 基准，为视觉叙事样本标注常识链接和话语约束，提出忠实度和一致性评估指标，并验证利用这些约束可显著提升视觉叙事生成的质量。

## 研究背景与动机

**领域现状**：视觉叙事生成（Visual Narrative Generation）是将文本叙事转换为图像序列的任务，广泛应用于影视分镜、教育插图等场景。现有方法如 ARLDM、StoryGen、MM-Interleaved 等主要依赖预训练视觉 Transformer 和扩散模型来学习文本到视觉叙事的直接映射。

**现有痛点**：现有方法存在两个核心问题：(1) **叙事对齐**——文本叙事通常是抽象的、视觉描述不充分的，模型需要推断常识知识来生成视觉内容（如"坏消息"应表现为角色悲伤的表情），但现有方法不建模这种文本与视觉之间的"具象化鸿沟"；(2) **视觉一致性**——图像序列中的角色外貌、场景位置、时间等叙事元素应跨帧保持一致，但现有方法不显式学习这些话语约束，导致生成结果在角色样貌、背景等方面前后不一。

**核心矛盾**：文本叙事与视觉叙事之间存在固有的具象化鸿沟（manifestation gap），且视觉叙事具有隐含的话语结构约束，但现有方法都没有显式建模这些约束。

**本文目标**：(1) 构建标注了常识与话语约束的视觉叙事基准数据集；(2) 设计能评估忠实度和一致性的新指标；(3) 验证约束学习可提升生成质量。

**切入角度**：利用 LLM 和 VLM（如 Llama3.1、LLaVA-OV）自动标注视觉叙事样本中隐含的常识链接和话语特征，将这些约束作为额外的学习信号和评估维度。

**核心 idea**：为视觉叙事样本构建"常识链接 + 话语约束"的双层标注体系，利用这些结构化约束作为训练辅助信号和评估标准，系统性地改善视觉叙事的忠实度和一致性。

## 方法详解

### 整体框架

VinaBench 包含约 25K 对视觉-文本叙事样本（来自 VWP、Storyboard20K、StorySalon），为每对样本标注两类约束：(1) 常识约束——将图像描述中的视觉实体链接到文本叙事中相关实体；(2) 话语约束——全局特征（角色档案、外观风格）和场景特征（出场角色、时间、地点）。基于这些约束还提出了忠实度和一致性评估指标。

### 关键设计

1. **常识约束构建（Commonsense Constraints）**:

    - 功能：桥接文本叙事与视觉叙事之间的具象化鸿沟
    - 核心思路：分三步完成——首先用 Mantis-Idefics2 对每张叙事图像生成密集描述（dense caption），同时输入文本叙事作为上下文防止幻觉；然后用 Llama3.1 从描述中提取视觉实体（名词/动词短语）；最后用 Llama3.1 将每个视觉实体链接到文本叙事中的关联实体，若无关联则标记为"no link"。例如图像中"绿色衬衫"可能没有对应文本实体，但"洗碗"可链接到叙事中"准备晚餐"。
    - 设计动机：现有视觉-语言对齐研究只关注通用的 token/区域级匹配，忽略了叙事情境中更隐含的常识对齐（如"坏消息"→悲伤表情）。通过显式标注这种对齐关系，模型可以学到更好的具象化策略。

2. **话语约束构建（Discourse Constraints）**:

    - 功能：显式表征视觉叙事的结构化特征以促进一致性
    - 核心思路：标注两类特征。**全局特征**包括角色档案（姓名、年龄、性别、社会角色、持续外貌特征）和外观风格（写实、卡通、漫画等）。**场景特征**包括每帧出场角色（用 LLaVA-OV 两步检测：先数人数再匹配档案）、时间段（清晨/上午/下午/傍晚/夜晚/不明）、地点。全局特征预期在整个叙事中保持静态，场景特征则追踪叙事元素的动态变化。
    - 设计动机：视觉叙事如同自然语言有句法结构，也有其话语结构（角色、时空元素的持续与变化）。显式标注这些结构可以为一致性提供明确的优化目标和评估标准。

3. **新评估指标体系**:

    - 功能：克服传统全参考指标的局限，提供忠实度和一致性的细粒度评估
    - 核心思路：设计了三类指标。(1) **对齐排名**（Alignment Ranking）：用 CLIP-T 或 VQAScore 从测试集 top-100 池中排名生成图像，报告 MRR，避免单参考偏差。(2) **细粒度对齐**（Fine-Grained Alignment）：5 个 VQA-based 指标分别评估非角色实体、角色数量、角色属性、时间、地点的对齐。(3) **一致性**（Consistency）：3 个 VQA-based 指标检查跨帧的风格一致性、角色一致性、地点一致性。所有 VQA 指标使用 VLM 以 Yes/No 判断形式输出。
    - 设计动机：FID 等全参考指标偏向特定参考图像（如女主角发色），但叙事的视觉表达是开放性的；CLIP-T 的绝对分值跨样本不可比。排名指标 + 基于约束的 VQA 指标可以更公平、更细粒度地衡量生成质量。

### 损失函数 / 训练策略

VinaBench 本身是一个 benchmark 而非独立模型。在实验中，作者在三种生成模型（ARLDM、StoryGen、MM-Interleaved）上测试了三种训练设置：(1) No Constraint——原始训练；(2) LLM Constraints——用 LLM 生成的约束作为额外输入进行训练；(3) Gold Constraints——用 VinaBench 标注的约束训练。约束信息以文本形式拼接到模型输入中。

## 实验关键数据

### 主实验

| 模型 | 设置 | FID↓ | CLIP-T MRR↑ | 一致性-风格 | 一致性-角色 | 一致性-地点 |
|------|------|------|-------------|------------|------------|------------|
| ARLDM | No Constraint | 42.6 | 0.110 | 0.466 | 0.379 | 0.376 |
| ARLDM | LLM Constraints | 37.6 | 0.151 | 0.859 | 0.551 | 0.689 |
| ARLDM | Gold Constraints | 35.3 | 0.155 | 0.854 | 0.569 | 0.697 |
| MM-Inter. | No Constraint | 48.3 | 0.066 | 0.947 | 0.582 | 0.449 |
| MM-Inter. | LLM Constraints | 42.2 | 0.111 | 0.986 | 0.678 | 0.764 |
| MM-Inter. | Gold Constraints | 39.3 | 0.118 | 0.976 | 0.688 | 0.856 |

### 消融实验

以 MM-Interleaved + LLM Constraints 为基础的消融：

| 设置 | FID↓ | CLIP-T MRR↑ | 一致性-地点 |
|------|------|-------------|------------|
| Full (CL + DF) | 42.2 | 0.111 | 0.764 |
| w/o 常识链接 (CL) | 42.9 | 0.109 | 0.758 |
| w/o 话语特征 (DF) | 43.3 | 0.107 | 0.684 |
| w/o 全局话语特征 | 42.6 | 0.110 | 0.760 |
| w/o 场景话语特征 | 42.6 | 0.109 | 0.685 |
| Random 约束 | 53.7 | 0.048 | 0.447 |

### 关键发现

- 在所有三个模型上，加入约束学习一致性地提升了所有指标，FID 最多降低 26 点（StoryGen）
- 话语约束对一致性指标的提升最为显著（特别是地点一致性从 0.449→0.764）
- 随机打乱约束导致性能剧烈下降，验证了约束信息的内容价值而非仅仅增加输入量的效果
- 场景话语特征（per-image）比全局话语特征对一致性的贡献更大
- 专家评审显示 VinaBench 的自动标注准确率高达 85-95%，验证了 LLM/VLM 构建流程的可靠性

## 亮点与洞察

- 开创性地将视觉叙事结构理论（话语分析中的角色/时空跟踪）引入生成式任务的约束建模
- 完整的 benchmark 设计：数据（25K 样本+约束标注）+ 指标（对齐+一致性）+ 方法验证
- 利用 LLM/VLM 做大规模自动标注的流水线设计可推广到其他需要结构化标注的任务
- 排名式评估替代绝对分值的思路值得在其他开放式生成任务中借鉴

## 局限与展望

- 标注质量依赖于 LLM/VLM 的能力，对于复杂叙事可能存在遗漏或错误
- 目前约束以文本形式拼接输入，尚未探索更结构化的约束注入方式（如图结构）
- 评估指标基于 VLM 的判断，可能引入 VLM 自身的偏差
- 生成模型仍有很大提升空间（与人工参考相比差距明显），约束学习只是方向之一
- 未来可考虑在视频生成场景中应用类似的话语约束框架

## 相关工作与启发

- 与 ConceptNet 等物理常识不同，本工作关注的是叙事情境中更隐含的常识对齐（"坏消息"→悲伤表情）
- 话语约束的思想可借鉴于长视频生成中的角色/场景一致性维护
- 排名式 MRR 评估方法可推广到其他开放式图像生成任务的评估

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 3.5 |
| 实验充分度 | 4.5 |
| 写作质量 | 4 |
| 总体评价 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](care_transformer_linear_attention.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](../../ACL2025/others/consistent_client_simulation_for_motivational_interviewing-based_counseling.md)

</div>

<!-- RELATED:END -->

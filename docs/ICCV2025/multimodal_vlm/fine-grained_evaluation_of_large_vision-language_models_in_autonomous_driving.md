---
title: >-
  [论文解读] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving
description: >-
  [ICCV 2025][多模态VLM][视觉语言模型评估] 本文提出 VLADBench，一个面向自动驾驶场景的细粒度视觉语言模型评测基准，涵盖 5 大领域、11 个二级维度和 29 个三级任务，采用封闭式 QA 形式从静态知识到动态推理逐步递进评估 VLM 能力，并基于 1.4M 领域特定 QA 数据训练小规模 DS 模型验证领域间认知交互。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉语言模型评估"
  - "自动驾驶基准"
  - "细粒度评测"
  - "视觉问答"
  - "驾驶场景理解"
---

# Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2503.21505](https://arxiv.org/abs/2503.21505)  
**代码**: 无  
**领域**: 多模态VLM / 自动驾驶  
**关键词**: 视觉语言模型评估, 自动驾驶基准, 细粒度评测, 视觉问答, 驾驶场景理解

## 一句话总结

本文提出 VLADBench，一个面向自动驾驶场景的细粒度视觉语言模型评测基准，涵盖 5 大领域、11 个二级维度和 29 个三级任务，采用封闭式 QA 形式从静态知识到动态推理逐步递进评估 VLM 能力，并基于 1.4M 领域特定 QA 数据训练小规模 DS 模型验证领域间认知交互。

## 研究背景与动机

**领域现状**：随着大型视觉语言模型（VLM）在通用视觉理解任务上取得突破性进展，研究者开始尝试将其应用于自动驾驶（AD）领域，期望利用 VLM 的强大感知和推理能力提升驾驶场景的理解与决策。

**现有痛点**：现有的自动驾驶 VLM 评测基准主要通过开放式视觉问答来评估模型的可解释性，但这种评估方式过于粗粒度——难以区分模型在不同驾驶子任务上的具体表现，例如交通标志识别、行人意图预测、自车决策规划等能力的差异无法被细致刻画。此外，开放式回答的评测一致性差、自动评分困难，导致评测结果缺乏可靠性和可比性。

**核心矛盾**：复杂驾驶场景需要模型具备从基础元素识别到高级推理的层级化认知能力，而现有基准将这些能力混为一谈，无法精准定位模型的能力短板。

**本文目标**：构建一个层次化、细粒度的封闭式评测基准，系统评估 VLM 在自动驾驶各子任务上的表现，并探索不同认知域之间的协同关系。

**切入角度**：作者观察到驾驶认知可以分解为由浅入深的五个领域——从静态交通知识理解到动态在线推理决策——这种层级结构天然适合构建细粒度评测体系。

**核心 idea**：设计涵盖 5 大领域、29 个三级任务的封闭式 QA 基准 VLADBench，通过从静态基础知识逐步过渡到动态决策推理的评测链，全面刻画 VLM 在自动驾驶中的能力画像。

## 方法详解

### 整体框架

VLADBench 的评测体系采用层级化设计：输入是驾驶场景图像和封闭式问答对，输出是模型在各维度的细粒度能力评分。整个框架包含两部分：(1) 基准数据集构建——5 个领域、11 个二级维度、29 个三级任务；(2) 领域特定模型训练——基于 1.4M QA 数据训练小规模 VLM，验证领域间的认知协同效应。

### 关键设计

1. **五大评测领域的层级化设计**:

    - 功能：从基础到高级系统评估 VLM 的驾驶认知能力
    - 核心思路：将驾驶认知分解为五个递进领域：(1) 交通知识理解（Traffic Knowledge Understanding）——评估模型对交通规则、标志含义等静态知识的掌握；(2) 通用元素识别（General Element Recognition）——评估对车辆、行人、车道线等基本元素的感知；(3) 交通拓扑生成（Traffic Graph Generation）——评估对场景拓扑关系的理解；(4) 目标属性理解（Target Attribute Comprehension）——评估对交通参与者属性和状态的细粒度识别；(5) 自车决策与规划（Ego Decision-Making and Planning）——评估动态场景下的推理和决策能力
    - 设计动机：从静态到动态、从感知到推理的递进设计，能精准定位模型在认知链条上的薄弱环节

2. **封闭式 QA 评测格式**:

    - 功能：提供标准化、可量化、可自动评分的评测方式
    - 核心思路：不同于开放式 QA，所有问题都设计为选择题或判断题形式，包含干扰选项。29 个三级任务对应不同的问题类型和难度级别，确保评测的全面性和一致性
    - 设计动机：封闭式格式避免了开放式回答的主观评分问题，使得大规模自动化评测成为可能，同时更能考察模型的精确理解而非语言生成能力

3. **领域特定（DS）模型训练与认知交互验证**:

    - 功能：验证五个评测领域之间的认知协同效应
    - 核心思路：从公开数据源收集 1.4M 领域特定 QA 数据，分别在各领域数据集上基于小规模 VLM 训练 DS 模型，研究单领域训练与跨领域训练的性能差异，揭示不同认知能力之间的相互促进关系
    - 设计动机：不仅评测现有模型，还通过实验验证"基础能力提升能否促进高级推理"的假设，为后续自动驾驶 VLM 的训练策略提供指导

### 损失函数 / 训练策略

DS 模型采用标准的视觉语言模型微调策略，使用交叉熵损失在各领域数据集上分别和联合训练，以探索领域间的迁移效应。

## 实验关键数据

### 主实验

| 模型 | 交通知识 | 元素识别 | 拓扑生成 | 属性理解 | 决策规划 | 综合 |
|------|---------|---------|---------|---------|---------|------|
| GPT-4V | 较强 | 中等 | 较弱 | 中等 | 较弱 | 中等偏上 |
| InternVL2 | 中等 | 较强 | 中等 | 中等 | 较弱 | 中等 |
| Qwen-VL | 中等 | 中等 | 较弱 | 中等 | 较弱 | 中等 |
| DS-Single | 较强 | 较强 | 中等 | 中等 | 中等 | 中等偏上 |
| DS-Joint | 较强 | 较强 | 较强 | 较强 | 中等偏上 | 较强 |

### 消融实验

| 训练配置 | 高级推理准确率 | 说明 |
|---------|-------------|------|
| 仅决策领域训练 | 基线 | 单一领域训练 |
| +交通知识 | +显著提升 | 基础知识促进推理 |
| +元素识别 | +明显提升 | 感知能力迁移 |
| 全领域联合训练 | 最优 | 领域协同效应显著 |

### 关键发现

- 通用 VLM（包括 GPT-4V）在需要驾驶领域专业知识的任务上表现不理想，尤其在交通拓扑理解和自车决策规划方面存在明显短板
- 领域特定训练能显著提升各子任务性能，且跨领域联合训练优于单领域训练，证实了认知域之间存在正向迁移
- 基础认知能力（交通知识、元素识别）的提升能有效促进高级推理能力（决策规划），验证了层级化认知建模的合理性

## 亮点与洞察

- **层级化评测设计**是本文最突出的贡献——将模糊的"自动驾驶理解能力"分解为 5→11→29 的三级体系，这种设计思路可迁移到任何需要细粒度能力评估的领域（如医学诊断、工业检测）
- **封闭式 QA 的选择很巧妙**——既保证了评测的客观性和自动化，又通过精心设计的干扰选项考察了模型的深层理解，而非表面的语言生成能力
- **认知交互实验**提供了一个重要的训练策略启示：在自动驾驶 VLM 的训练中，先建立基础感知能力再逐步引入高级推理任务可能比端到端训练更有效

## 局限与展望

- 基准目前主要基于**静态图像**的 QA，缺少对**视频序列**的时序理解评测，而实际驾驶需要对连续帧的动态变化进行推理
- 评测覆盖的**场景多样性**可能有限——极端天气、夜间驾驶、罕见交通事件等长尾场景的评测不够充分
- 1.4M QA 数据的**质量和标注一致性**未做详细分析，大规模自动生成的 QA 可能存在噪声
- 未来可以扩展到**3D 场景理解**和**多传感器融合**的评测，以更全面地反映自动驾驶系统的需求

## 相关工作与启发

- **vs DriveLM**: DriveLM 侧重对话式驾驶理解，采用开放式 QA；VLADBench 使用封闭式 QA 且评测粒度更细，更适合大规模标准化评测
- **vs NuScenes-QA**: NuScenes-QA 聚焦于基于 3D 标注的场景理解；VLADBench 更注重从 2D 图像出发的视觉认知能力层级评估
- **vs DriveVLMs**: DriveVLMs 关注端到端驾驶中 VLM 的集成；VLADBench 更专注于能力评测和诊断，两者互补
- 这项工作为"如何系统评估领域特定 VLM 能力"提供了方法论范式，可以启发构建其他垂直领域的细粒度评测基准

## 评分

- 新颖性: ⭐⭐⭐⭐ 层级化细粒度评测框架的设计思路新颖，但 benchmark 类工作的技术创新相对有限
- 实验充分度: ⭐⭐⭐⭐ 评测了多种通用和领域特定 VLM，领域交互实验设计巧妙，但部分实验细节不够透明
- 写作质量: ⭐⭐⭐⭐ 结构清晰，层级关系表达准确，图表设计直观
- 价值: ⭐⭐⭐⭐ 为自动驾驶 VLM 研究提供了重要的评测工具，领域交互发现有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)

</div>

<!-- RELATED:END -->

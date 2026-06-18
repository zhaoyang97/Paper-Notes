---
title: >-
  [论文解读] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning
description: >-
  [ICCV 2025][多模态VLM][组合式视觉推理] 本文提出 DWIM 框架，通过差异感知的工作流生成策略筛选高质量训练数据，以及指令掩码微调策略只克隆有效动作，使 LLM 在组合式视觉推理中具备工具感知能力，在多个 VR 基准上取得 SOTA。 领域现状：视觉推理（Visual Reasoning…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "组合式视觉推理"
  - "工具感知"
  - "工作流生成"
  - "指令掩码微调"
  - "LLM"
---

# DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning

**会议**: ICCV 2025  
**arXiv**: [2503.19263](https://arxiv.org/abs/2503.19263)  
**代码**: 无  
**领域**: 多模态VLM / 视觉推理  
**关键词**: 组合式视觉推理, 工具感知, 工作流生成, 指令掩码微调, LLM

## 一句话总结

本文提出 DWIM 框架，通过差异感知的工作流生成策略筛选高质量训练数据，以及指令掩码微调策略只克隆有效动作，使 LLM 在组合式视觉推理中具备工具感知能力，在多个 VR 基准上取得 SOTA。

## 研究背景与动机

**领域现状**：视觉推理（Visual Reasoning, VR）是让模型具备类人视觉理解能力的关键任务。近年来组合式视觉推理（Compositional VR）方法成为主流，这类方法借助 LLM 的推理能力，将复杂视觉问题分解为多个子步骤，调用外部工具（如目标检测器、OCR、深度估计等）逐步求解，效果优于端到端方法。

**现有痛点**：冻结的 LLM 缺乏对视觉推理工具的感知能力，不知道工具能做什么、什么时候该用什么工具、工具返回的结果是否可靠。这导致生成的工作流中包含大量无效或错误的工具调用，成为性能瓶颈。

**核心矛盾**：虽然在其他领域（如代码生成、数学推理）可以通过微调让 LLM 学习工具使用，但在视觉推理中面临三重困难：(1) 训练数据有限——高质量的视觉推理工作流数据稀缺；(2) 工具不完美——外部工具本身会引入错误，降低数据收集效率；(3) 噪声工作流——收集到的工作流中包含大量错误步骤，直接微调会让模型学到坏习惯。

**本文目标**：让 LLM 具备工具感知能力，学会在视觉推理中正确使用工具，同时解决训练数据质量差和工作流噪声大的问题。

**切入角度**：作者观察到不是工作流中的每个步骤都值得学习——有些工具调用是正确且有效的，有些则是错误的。关键在于区分工作流中的"好动作"和"坏动作"，只学习有效的部分。

**核心 idea**：提出两阶段方案——先用差异感知策略生成高质量训练工作流（过滤掉不靠谱的），再用指令掩码微调确保模型只克隆有效动作（跳过工作流中的错误步骤）。

## 方法详解

### 整体框架

DWIM 由两个核心组件构成：(1) Discrepancy-aware Workflow Generation（差异感知工作流生成），负责在数据收集阶段筛选高质量工作流用于训练；(2) Instruct-Masking Tuning（指令掩码微调），负责在训练阶段让模型只学习工作流中的有效步骤。输入是视觉问题（图像 + 问题），输出是包含工具调用的推理工作流，最终得到答案。

### 关键设计

1. **Discrepancy-aware Workflow Generation（差异感知工作流生成）**:

    - 功能：从自动收集的工作流中筛选出高质量的训练样本
    - 核心思路：利用工具执行结果与预期之间的差异（discrepancy）来评估工作流质量。具体地，对于每个候选工作流，执行其中的工具调用并检查中间结果是否合理——如果工具返回的结果与问题上下文一致（差异小），则认为该工作流是可行的；反之则过滤掉。这种评估方式避免了仅依赖最终答案正确性来判断工作流质量的缺陷（最终答案可能碰巧正确但中间步骤有误）
    - 设计动机：直接用 LLM 生成的工作流噪声太大，而仅靠最终答案过滤会保留"歪打正着"的工作流。通过评估每个工具调用步骤的差异，可以更精细地筛选出真正有效的训练数据

2. **Instruct-Masking Fine-tuning（指令掩码微调）**:

    - 功能：在微调过程中让模型只学习工作流中的有效动作，忽略无效/错误的步骤
    - 核心思路：对工作流中的每个步骤进行标注——有效的工具调用步骤保留其训练损失，无效的步骤则在计算损失时被掩码（mask）掉。这样模型在训练时不会被迫去模仿错误的工具使用模式，只克隆那些确实有效的操作
    - 设计动机：即使经过差异感知筛选，训练工作流中仍可能包含部分不理想的步骤（因为整体工作流是可行的但个别步骤次优）。指令掩码机制提供了步骤级别的精细控制，确保模型学到的都是正确的工具使用方式

3. **工具感知推理机制**:

    - 功能：使微调后的 LLM 能在推理时生成合理的工具调用工作流
    - 核心思路：通过上述两阶段训练，LLM 学会了何时调用什么工具、如何解析工具返回结果、以及如何基于中间结果继续推理。推理时模型自回归地生成包含工具调用的步骤序列，外部执行器负责运行工具并将结果反馈给模型
    - 设计动机：现有方法依赖冻结 LLM 的零样本能力或少量示例提示，工具使用能力有限。通过高质量数据微调，模型获得了内化的工具感知能力

### 损失函数 / 训练策略

训练采用标准的自回归语言建模损失，但通过指令掩码对无效步骤的 token 进行掩码处理，仅在有效动作的 token 上计算交叉熵损失。这使得模型的梯度更新只受到正确工具使用模式的引导。

## 实验关键数据

### 主实验

DWIM 在多个视觉推理基准上进行了评估，涵盖多种任务类型：

| 数据集 | 任务 | DWIM | 之前 SOTA | 提升 |
|--------|------|------|-----------|------|
| GQA | 视觉问答 | SOTA | - | 显著 |
| VQAv2 | 视觉问答 | SOTA | - | 显著 |
| NLVR2 | 视觉推理 | SOTA | - | 显著 |
| RefCOCO | 指代表达 | SOTA | - | 显著 |

实验表明 DWIM 在各种 VR 任务上均取得最优性能，尤其在需要多步工具调用的复杂推理任务中优势明显。

### 消融实验

| 配置 | 性能变化 | 说明 |
|------|---------|------|
| Full DWIM | 最优 | 完整模型 |
| w/o Discrepancy-aware | 下降 | 不筛选工作流质量，训练数据噪声大 |
| w/o Instruct-Masking | 下降 | 不掩码无效步骤，模型学到错误模式 |
| 仅用最终答案过滤 | 次优 | 保留了"歪打正着"的工作流 |

### 关键发现

- 两个组件都贡献显著：差异感知工作流生成解决了数据质量问题，指令掩码微调解决了残余噪声问题，二者缺一不可
- 在需要多步推理的复杂任务中，DWIM 的优势更为明显，说明工具感知能力对复杂推理至关重要
- 与基于提示工程的方法相比，微调方法在工具使用的准确性和一致性上都更好

## 亮点与洞察

- **差异感知评估的精细度**：不是简单看最终答案对不对，而是评估每个中间步骤的工具调用是否合理。这种思路可以迁移到其他需要多步推理的场景（如代码生成、数学证明），用于筛选高质量训练轨迹
- **指令掩码的优雅设计**：在序列级别的监督学习中引入 token 级别的选择性学习，既保留了完整工作流的上下文信息，又避免学到错误步骤。这是一个通用的"从噪声数据中学习"的技巧
- **自动化数据管线**：整个流程不需要人工标注——工作流自动生成、自动评估、自动掩码，具有很好的可扩展性

## 局限与展望

- 依赖外部工具的质量和覆盖范围——如果可用工具集不够或工具本身错误率高，方法效果会受限
- 差异感知评估本身需要执行工具调用，增加了数据准备的计算开销
- 当前评估主要在静态基准上，在动态、开放世界的视觉推理场景中的表现有待验证
- 未来可以探索让模型自适应地发现和整合新工具，而非使用固定工具集

## 相关工作与启发

- **vs VisProg/ViperGPT**: 这些经典组合式 VR 方法使用冻结 LLM 生成程序/工作流来调用工具，但不对 LLM 进行微调。DWIM 通过微调让 LLM 具备内化的工具感知能力，性能更好
- **vs Chameleon**: Chameleon 也是工具增强的 LLM 推理，但其工具选择策略是基于规则的。DWIM 通过数据驱动的方式学习工具使用模式，更灵活
- **vs InstructBLIP/LLaVA**: 这些端到端多模态模型不使用外部工具，在需要精确感知的任务上不如组合式方法。DWIM 证明了"LLM 推理 + 外部工具"范式的上限还很高

## 评分

- 新颖性: ⭐⭐⭐⭐ 差异感知 + 指令掩码的组合是新的，但整体框架在组合式 VR 范畴内
- 实验充分度: ⭐⭐⭐⭐ 多个数据集验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述层次分明
- 价值: ⭐⭐⭐⭐ 为组合式视觉推理中的数据质量和训练策略提供了实用解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[CVPR 2026\] CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization](../../CVPR2026/multimodal_vlm/codev_code_with_images_for_faithful_visual_reasoning_via_tool-aware_policy_optim.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)

</div>

<!-- RELATED:END -->

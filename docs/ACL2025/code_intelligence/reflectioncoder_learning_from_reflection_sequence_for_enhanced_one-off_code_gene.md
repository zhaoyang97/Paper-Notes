---
title: >-
  [论文解读] ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation
description: >-
  [ACL 2025][代码智能][代码生成] ReflectionCoder通过构建整合编译器反馈的"反思序列"（reflection sequence）数据，结合反思自蒸馏和动态掩码蒸馏两种训练策略，使模型在一次性代码生成中达到SOTA性能，无需运行时的多轮调试。 领域现状：代码生成是LLM的核心应用之一…
tags:
  - "ACL 2025"
  - "代码智能"
  - "代码生成"
  - "反思序列"
  - "知识蒸馏"
  - "编译反馈"
  - "一次性生成"
---

# ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation

**会议**: ACL 2025  
**arXiv**: [2405.17057](https://arxiv.org/abs/2405.17057)  
**代码**: [GitHub](https://github.com/SenseLLM/ReflectionCoder)  
**领域**: 文本生成/代码生成  
**关键词**: 代码生成, 反思序列, 知识蒸馏, 编译反馈, 一次性生成  

## 一句话总结
ReflectionCoder通过构建整合编译器反馈的"反思序列"（reflection sequence）数据，结合反思自蒸馏和动态掩码蒸馏两种训练策略，使模型在一次性代码生成中达到SOTA性能，无需运行时的多轮调试。

## 研究背景与动机

**领域现状**：代码生成是LLM的核心应用之一，现有方法主要分为两类：(1) 一次性生成（one-off），模型直接输出最终代码；(2) 迭代修复（iterative refinement），模型生成代码后通过编译器反馈多轮修正。后者通常表现更好，但推理成本高，且依赖编译器环境。

**现有痛点**：一次性生成方法的性能上限受限于训练数据质量。模型从纯"需求→代码"的数据中学习，缺乏对错误代码的理解和纠正能力。而迭代修复方法虽然效果好，但工程部署复杂、延迟高，不适合实时代码补全等场景。

**核心矛盾**：编译器反馈蕴含丰富的纠错知识，但这种知识无法直接用于一次性生成场景——推理时没有编译器，模型无法进行反思和修正。如何将"反思过程"中的知识迁移到一次性生成模型中是关键挑战。

**本文目标**：设计一种方法，让模型在训练时学习编译器反馈驱动的反思过程中的知识，同时在推理时仍能进行高效的一次性代码生成。

**切入角度**：不在推理时使用编译器反馈，而是将反思过程构造为训练数据（反思序列），通过蒸馏策略将反思知识"压缩"到一次性生成能力中。

**核心 idea**：用编译器反馈构建"尝试→错误→反思→正确"的反思序列作为训练数据，通过反思自蒸馏让模型从自己的反思过程中学习，同时用动态掩码蒸馏确保模型聚焦于最终正确代码的生成。

## 方法详解

### 整体框架
ReflectionCoder的pipeline分为两个阶段：(1) **数据构建阶段**：给定编程问题，先让教师模型（GPT-4/DeepSeek-Coder）生成初始代码，通过编译器执行获取反馈，再让模型基于反馈进行修正，整个过程构成"反思序列"；(2) **训练阶段**：使用构建好的反思序列数据和原始代码指令数据混合训练学生模型，采用反思自蒸馏和动态掩码蒸馏两种策略。推理时，模型只需要一次前向传播即可生成最终代码。

### 关键设计

1. **反思序列构建（Reflection Sequence Construction）**:

    - 功能：构建包含编译器反馈的多轮"尝试-反思-修正"训练数据
    - 核心思路：每条反思序列包含三个部分——(T) 初始尝试代码、(C) 编译器/测试的反馈信息、(E) 反思和修正后的正确代码。格式为 $[T_1, C_1, E_1, T_2, C_2, E_2, \ldots]$，其中可能包含多轮反思。使用GPT-4和DeepSeek-Coder分别构建了ReflectionSeq-GPT和ReflectionSeq-DS两个数据集。
    - 设计动机：通过保留完整的纠错过程，让模型理解"哪里容易出错、为什么出错、如何修正"的逻辑链，而非仅学习最终正确答案

2. **反思自蒸馏（Reflection Self-Distillation）**:

    - 功能：让学生模型从反思序列中学习，但在推理时只生成最终正确代码
    - 核心思路：训练时，模型的输入包含完整的反思序列，但损失函数根据block类型区分处理——对于尝试和编译反馈部分（T和C block），使用教师模型的logits进行蒸馏训练；对于最终正确代码部分（E block），直接使用标准的下一个token预测损失。整体损失为 $\mathcal{L} = \alpha \mathcal{L}_{distill} + (1-\alpha) \mathcal{L}_{CE}$。
    - 设计动机：让模型理解反思过程但不依赖它——训练时"看过"反思序列，推理时直接输出最终结果

3. **动态掩码蒸馏（Dynamically Masked Distillation）**:

    - 功能：进一步优化训练信号，防止模型在推理时尝试生成反思过程
    - 核心思路：在训练过程中动态地对反思序列中的T和C部分应用掩码，随训练进度逐步增加掩码比例——初期让模型完整看到反思过程，后期逐步隐藏更多反思内容，迫使模型将反思知识内化为一次性生成能力。掩码比例从0线性增长到接近1，训练结束时模型几乎只看到最终正确代码。
    - 设计动机：解决训练-推理不一致问题。如果训练时总是看到完整反思序列，模型可能在推理时也尝试生成反思过程，而非直接输出代码

### 损失函数 / 训练策略
整体训练采用混合数据策略：反思序列数据（ReflectionSeq-GPT + ReflectionSeq-DS）和标准代码指令数据（如Evol-CodeAlpaca）混合。使用DeepSpeed ZeRO-1进行分布式训练，学习率5e-5，cosine调度器，训练2个epoch，全局batch size 512。模型基于CodeLlama和DeepSeek-Coder两个基座进行微调。

## 实验关键数据

### 主实验
在HumanEval(+)和MBPP(+)基准上的pass@1结果：

| 模型 | 参数量 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|--------|-----------|------------|------|-------|
| WizardCoder-CL-34B | 34B | 73.2 | 64.6 | 73.2 | 59.9 |
| MagiCoder-CL-7B | 7B | 71.3 | 64.6 | 68.4 | 56.9 |
| OpenCodeInterp-DS-6.7B | 6.7B | 76.2 | 72.0 | 73.9 | 63.7 |
| **ReflectionCoder-CL-7B** | 7B | **75.0** | **68.9** | **72.2** | **61.4** |
| **ReflectionCoder-DS-6.7B** | 6.7B | **80.5** | **74.4** | **81.5** | **69.6** |
| **ReflectionCoder-DS-33B** | 33B | **82.9** | **76.8** | **84.1** | **72.0** |

### 消融实验
不同训练策略和数据组合的消融分析：

| 配置 | HumanEval | MBPP+ | 说明 |
|------|-----------|-------|------|
| 仅CodeInstruct数据 | 74.4 | 63.2 | 基线，无反思数据 |
| + ReflectionSeq（标准CE） | 77.4 | 66.5 | 直接用反思序列训练 |
| + 反思自蒸馏 | 79.3 | 68.1 | 加入蒸馏策略 |
| + 动态掩码蒸馏（完整模型） | 80.5 | 69.6 | 完整ReflectionCoder |
| 仅ReflectionSeq-GPT | 78.7 | 67.8 | 只用GPT生成的反思数据 |
| 仅ReflectionSeq-DS | 77.9 | 66.9 | 只用DeepSeek生成的反思数据 |

### 关键发现
- **反思序列数据是性能提升的最大贡献因素**：仅加入反思序列数据（即使用标准CE训练）就能带来约3%的提升，说明"错误→修正"的过程本身就包含有价值的监督信号。
- **两种蒸馏策略互补**：反思自蒸馏贡献约2%提升，动态掩码蒸馏再贡献约1.5%，两者叠加效果优于单独使用。
- **在MultiPL-E多语言基准上也表现优异**：不仅限于Python，在Java、C++、JavaScript等语言上也显著超越基线，说明反思知识具有跨语言迁移能力。
- 混合GPT和DeepSeek两套反思数据效果最好，暗示数据多样性有助于泛化。

## 亮点与洞察
- **将推理时的多轮优势"蒸馏"到一次性生成中**：这种"训练时看完整过程、推理时直接出结果"的思路非常巧妙，可以广泛迁移到其他需要迭代优化但部署时要求低延迟的任务，如数学推理、代码修复等。
- **渐进式掩码策略**：动态掩码从完全可见到完全隐藏的课程学习方式，优雅地解决了训练-推理gap的问题，这个设计模式可以迁移到任何使用中间推理步骤训练但推理时要跳过中间步骤的场景。
- **数据构建方法的可扩展性**：反思序列可以自动化大规模构建，不需要人工标注，且天然利用了编译器这一免费的验证工具。

## 局限与展望
- 反思序列的质量直接依赖教师模型的纠错能力，如果教师模型无法修正某类错误，学生也无法学到
- 目前仅在函数级代码生成上验证，对于更复杂的项目级代码生成效果未知
- 动态掩码的调度策略（线性增长）相对简单，可能存在更优的课程学习方案
- 未探索将该方法应用于非代码领域（论文中提到了可能性但未验证）
- **改进方向**：可以引入更强的代码验证器（如形式化验证工具）替代简单的编译测试，构建质量更高的反思序列

## 相关工作与启发
- **vs WizardCoder**: WizardCoder使用Evol-Instruct来增强指令复杂度提升代码生成能力，本文则从数据构建的角度（反思序列）出发，两者可能互补
- **vs Self-Repair/Self-Debug**: 后者在推理时使用编译器反馈多轮修正，效果好但成本高；本文将这种修正能力压缩到一次性生成中，显著降低推理成本
- **vs MagiCoder**: MagiCoder通过从开源代码片段生成高质量编程问题来增强训练数据，本文则通过引入错误修正过程来提供额外的监督信号，角度不同
- 本方法最有启发的点是"将多步推理知识蒸馏到单步生成"的范式，这和最近的推理蒸馏（如将CoT蒸馏到直接回答）思路一脉相承

## 评分
- 新颖性: ⭐⭐⭐⭐ 反思序列+渐进掩码蒸馏的组合是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ 多模型规模、多基准、完整消融，MultiPL-E多语言评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 实用性强，"训练时反思、推理时一步到位"的范式有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)

</div>

<!-- RELATED:END -->

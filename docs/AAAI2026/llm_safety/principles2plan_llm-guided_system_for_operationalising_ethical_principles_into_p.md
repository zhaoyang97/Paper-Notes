---
title: >-
  [论文解读] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans
description: >-
  [AAAI 2026][LLM安全][伦理规划] 提出 Principles2Plan，一个交互式原型系统，通过人类与 LLM 协作将高层伦理原则（如仁善、隐私）转化为上下文相关的伦理规则，并嵌入 PDDL 规划器生成符合伦理的行动计划。 随着机器人和自主系统越来越多地部署在人类生活环境中，确保其行为既能实现目标又尊重伦理原…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "伦理规划"
  - "LLM"
  - "自动化规划"
  - "PDDL"
  - "人机协作"
---

# Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans

**会议**: AAAI 2026  
**arXiv**: [2512.08536](https://arxiv.org/abs/2512.08536)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 伦理规划, LLM, 自动化规划, PDDL, 人机协作

## 一句话总结

提出 Principles2Plan，一个交互式原型系统，通过人类与 LLM 协作将高层伦理原则（如仁善、隐私）转化为上下文相关的伦理规则，并嵌入 PDDL 规划器生成符合伦理的行动计划。

## 研究背景与动机

随着机器人和自主系统越来越多地部署在人类生活环境中，确保其行为既能实现目标又尊重伦理原则成为关键挑战。高层伦理原则（如 beneficence 仁善、privacy 隐私）本质上是**抽象且高度依赖上下文的**。例如：

- **自动驾驶场景**：乘客需要紧急就医时，走未授权的捷径可能是合理的（仁善原则优先于交通规则）
- **休闲出行场景**：同样的仁善原则下，遵守标准交通规则更为合适

这种**同一原则在不同上下文下产生不同行为**的特性，使得完全自动化的伦理规划极为困难。

现有计算机器伦理（CME）方法的局限：

**自顶向下方法**: 预先指定规则，透明但缺乏适应性，手动编码工作量大

**自底向上方法**: 从数据推断伦理行为，灵活但缺乏可解释性

**混合方法**: 仍需大量人工编码伦理规则

**核心动机**: LLM 的出现为减少人工编码伦理规则的工作量提供了可能。能否利用 LLM 的理解和生成能力，在人类监督下自动生成上下文敏感的伦理规则？

## 方法详解

### 整体框架

Principles2Plan 采用**四步交互式流程**，引导用户从输入到生成伦理计划：

```
输入页面 → 伦理规则编辑器 → 代码编辑器 → 输出计划页面
```

系统面向的用户包括：伦理敏感领域的领域专家、AI 伦理研究者、对伦理-LLM-规划交叉领域感兴趣的人。

### 关键设计

#### 1. **输入页面（Input Page）**

用户提供以下关键信息来启动伦理规划：

- **PDDL 文件**: 上传 `problem.pddl` 和 `domain.pddl` 定义规划问题和领域
- **初始状态与假设**: 关于问题或领域的背景信息
- **高层伦理原则**: 如 beneficence（仁善）、privacy（隐私）等
- **模型选择**: 用户可选择使用的 LLM

系统提供了三个伦理敏感领域的示例问题：自动驾驶、老年护理、消防救援，用户可直接加载示例填充输入。

LLM 根据所有输入信息实时生成上下文相关的伦理规则，每条规则包含**伦理特征**（ethical features），表示该规则的正面或负面伦理特性（如"dishonesty"是负面特征）。

#### 2. **伦理规则编辑器（Ethical Rules Editor）**

由于 LLM 生成的规则可能存在不一致或不完善，系统提供人机协作的审查机制：

- **增删改**: 用户可添加缺失规则、删除不当规则、修改现有规则
- **LLM 解释**: 系统为每条规则提供 LLM 的推理解释，说明为何在当前问题和原则下生成该规则
- **优先级设定**: 用户为每个伦理特征分配 1-5 的重要性等级
- **正负特征高亮**: 系统高亮显示正面和负面伦理特征，方便用户快速识别

这一步体现了"**human-in-the-loop**"的设计理念，确保最终规则既利用了 LLM 的生成能力，又经过了人类专家的审查和校正。

#### 3. **代码编辑器（Code Editor）**

经用户确认的伦理规则通过 LLM 自动转换为 **PDDL-Ethical 代码**（一种支持伦理构造的 PDDL 扩展）。用户可以：

- 审查语法高亮的代码
- 对照上一步的伦理规则进行交叉检查
- 确保规则与代码之间的一致性

代码随后通过**转译器**（基于 jedwabny et al. 2022 的方法）转换为带有动作代价（action costs）的标准 PDDL，并提交给领域无关的经典规划器（Fast Downward）。

#### 4. **输出计划页面（Output Plan Page）**

系统并排展示两个计划：

- **带伦理规则的计划**: 考虑了伦理约束后生成的计划
- **原始计划**: 使用原始 PDDL 文件、不考虑伦理约束的计划

这种**对比展示**使用户能直观评估伦理规则对规划结果的影响。

### 损失函数 / 训练策略

本文是系统演示论文，不涉及模型训练。核心技术栈为：

- **LLM**: 使用 DeepSeek-R1-Distill-Llama-70B 作为后端
- **规划器**: Fast Downward（领域无关经典规划器）
- **PDDL-Ethical**: 规划领域特定语言的伦理扩展
- **评估指标**: Sentence-BERT 相似度（0.82）、代码生成成功率（82.2%）

## 实验关键数据

### 主实验

由于是系统演示论文，没有传统的对比实验。核心性能指标来自底层方法的评估：

| 指标 | 数值 | 说明 |
|---|---|---|
| Sentence-BERT 相似度 | 0.82 | LLM 生成规则与参考规则的语义相似度 |
| 代码生成成功率 | 82.2% | PDDL-Ethical 代码的正确生成率 |
| 支持领域 | 3 个 | 自动驾驶、老年护理、消防救援 |

### 消融实验

| 系统组件 | 功能 | 必要性 |
|---|---|---|
| LLM 规则生成 | 自动生成上下文相关的伦理规则 | 核心（减少手动编码） |
| 人类审查编辑 | 修正 LLM 输出的不一致和错误 | 核心（保证质量） |
| 优先级设定 | 区分不同伦理特征的重要性程度 | 关键（影响计划选择） |
| 代码交叉检查 | 确保规则到代码的一致性 | 重要（减少转译错误） |

### 关键发现

1. **LLM 在伦理规则生成方面有潜力但不完美**: 0.82 的语义相似度表明 LLM 能捕捉大部分伦理意图，但仍需人类审查
2. **上下文敏感性是核心难点**: 同一伦理原则在不同上下文下需要生成截然不同的规则
3. **代码生成成功率有提升空间**: 82.2% 意味着近 1/5 的情况需要人工修复代码

## 亮点与洞察

- **首个支持原则到规划的完整系统**: 将高层伦理原则通过 LLM 转化为可执行的规划约束，实现了从抽象到具体的完整链路
- **透明性设计**: 每条规则都有 LLM 的推理解释，规则到代码的映射可审查，最终计划可与无伦理约束版本对比
- **实际可操作性**: 提供了三个领域的示例问题，用户可立即体验系统
- **开创性方向**: 首次在规划社区中实现了人-LLM 协作的伦理规则系统化生成

## 局限与展望

1. **需要 PDDL 专业知识**: 当前用户需要理解 PDDL 和 PDDL-Ethical，这限制了非技术用户的使用
2. **LLM 生成质量有限**: 82.2% 的代码成功率不够高，当前版本可能需要较多人工干预
3. **缺乏大规模用户研究**: 未评估真实用户对系统的使用体验和接受度
4. **伦理冲突处理不足**: 当多个伦理原则产生冲突时（如仁善 vs 隐私），系统仅通过优先级简单排序，缺乏更深层的冲突解决机制
5. **仅限经典规划**: 不支持不确定性规划或多智能体场景

## 相关工作与启发

- **CME 三分类法（top-down / bottom-up / hybrid）** 提供了很好的伦理 AI 研究分类框架
- **PDDL-Ethical** 作为经典规划的伦理扩展，通过 action costs 编码伦理偏好是一个优雅的技术方案
- **LLM + 规划** 的交叉领域正在快速发展，从直接用 LLM 生成计划到用 LLM 辅助规划过程（构建模型、翻译约束）
- 未来可结合迭代对话让 LLM 根据规划结果反馈改进伦理规则

## 评分

- 新颖性: ⭐⭐⭐⭐ (首个将伦理原则到自动规划的完整人机协作系统)
- 实验充分度: ⭐⭐⭐ (系统演示论文，缺乏全面的定量评估和用户研究)
- 写作质量: ⭐⭐⭐⭐ (系统流程描述清晰，动机充分)
- 价值: ⭐⭐⭐⭐ (开创性工作，为伦理自动规划提供实用工具)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/llm_safety/representation-guided_parameter-efficient_llm_unlearning.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](perturb_your_data_paraphrase-guided_training_data_watermarking.md)

</div>

<!-- RELATED:END -->

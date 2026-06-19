---
title: >-
  [论文解读] Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework
description: >-
  [CVPR 2025][LLM推理][Lyapunov函数] 提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。 领域现状 领域现状：领域现状：Lyapunov 函数是验证动力系统稳定性的核心工具…
tags:
  - "CVPR 2025"
  - "LLM推理"
  - "Lyapunov函数"
  - "多项式网络"
  - "形式验证"
  - "反例引导"
  - "稳定性分析"
---

# Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework

**会议**: CVPR 2025  
**代码**: 无  
**领域**: 音频语音  
**关键词**: Lyapunov函数, 多项式网络, 形式验证, 反例引导, 稳定性分析

## 一句话总结
提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：Lyapunov 函数是验证动力系统稳定性的核心工具，多项式形式的 Lyapunov 函数可将稳定性分析转化为高效可解的优化问题（如 SOS 优化）。

**现有痛点**：

### 现有痛点

**现有痛点**：传统数值方法（如 SOS 编程）依赖用户预定义的多项式模板，模板选择不当会导致搜索失败

### 核心矛盾

**核心矛盾**：新兴的神经 Lyapunov 方法虽有灵活性，但基于朴素平方多项式网络的泛化能力差，且缺乏形式化正确性保证

### 解决思路

**解决思路**：反例引导的验证方法在反例精度不够高时，迭代收敛慢甚至发散

**核心矛盾**：灵活性与可验证性的权衡 — 神经网络方法灵活但不可验证，传统 SOS 方法可验证但不够灵活。

**本文目标** 如何在保持多项式 Lyapunov 函数数学严格性的同时，利用机器学习自动发现合适的多项式形式。

**切入角度**：将问题分解为"学习引导搜索方向"和"形式验证确保正确性"两个互补阶段，并通过高精度反例作为桥梁连接两者。

**核心 idea**：数据驱动的 ML 模型负责提出候选 Lyapunov 函数，高精度反例引导框架负责验证和迭代修正，实现 learning-enabled + formally-verified 的统一。

## 方法详解

### 整体框架
框架包含两个交替迭代的阶段：(1) **学习阶段**：基于采样数据和反例训练 ML 模型，自动选择多项式形式并估计系数；(2) **验证阶段**：对候选 Lyapunov 函数进行形式化验证，找到高精度反例反馈给学习阶段。

### 关键设计

1. **数据驱动的多项式形式选择**:
    - 功能：利用 ML 模型从数据中学习适合目标系统的多项式结构
    - 核心思路：不再要求用户手工指定多项式模板，而是通过神经网络在多项式基函数空间中搜索，学习系数和结构
    - 设计动机：多项式模板的选择高度依赖专家经验，自动化选择可以扩展方法的适用范围

2. **高精度反例引导（High-Accuracy Counterexample-Guided）**:
    - 功能：当验证阶段发现候选函数在某些状态点不满足 Lyapunov 条件时，精确定位这些违反点并将它们作为高精度反例反馈
    - 核心思路：相比随机采样或低精度近似反例，高精度反例更精准地指出候选函数的缺陷位置，使得下一轮学习迭代更高效
    - 设计动机：低精度反例会导致学习信号嘈杂，收敛缓慢；高精度反例可以加速收敛并减少迭代次数

3. **形式验证保证正确性**:
    - 功能：通过 SOS（Sum-of-Squares）分解或 SMT（Satisfiability Modulo Theories）求解器对候选 Lyapunov 函数进行数学严格验证
    - 核心思路：最终输出的 Lyapunov 函数经过形式化证明，不仅仅是数值近似
    - 设计动机：在安全关键系统中，数值近似不够可靠，需要形式化证明

## 实验关键数据

### 主要对比


### 主实验

| 方法 | 合成成功率 | 迭代次数 | 适用系统维度 |
|------|-----------|---------|-------------|
| 传统 SOS（固定模板） | 受限于模板选择 | N/A | 中低维 |
| 神经 Lyapunov | 灵活但无保证 | 多 | 中高维 |
| 本文方法 | 高成功率 + 形式验证 | 较少 | 中高维 |

### 关键发现
- 高精度反例引导相比随机采样反例显著减少了迭代次数
- 学习引导的多项式搜索在高维系统上优于固定模板方法
- 最终合成的 Lyapunov 函数具有数学严格性保证，可直接用于控制器设计

## 亮点与洞察
- **Learning + Verification 范式**：将 ML 的灵活性和形式验证的严格性有机结合，是 AI for Science 领域的一个有价值的范式
- **高精度反例的关键作用**：反例质量直接决定了迭代效率，这一洞察对其他 counterexample-guided 方法也有启发
- **多项式形式的自动发现**：去除了人工模板选择的环节，降低了使用门槛

## 局限与展望
- 多项式 Lyapunov 函数本身的表达力有限，对某些非线性系统可能不存在多项式形式的 Lyapunov 函数
- 形式验证阶段（SOS/SMT）随系统维度增长计算量急剧增大
- 目前主要针对自治系统的稳定性分析，受控系统和切换系统的扩展有待探索
- 与 Barrier Function 等安全性证书的统一框架是有价值的后续方向

## 相关工作与启发

- **vs 传统 SOS 方法**: 传统方法依赖用户手工指定多项式模板，模板选择很大程度决定了搜索成败。本文通过ML自动学习多项式结构，去除了人工模板设计环节，适用范围更广
- **vs 神经 Lyapunov 方法（如 SMT-based）**: 神经网络方法灵活但缺乏形式化正确性保证，本文的 Learner-Verifier 迭代框架同时兼顾灵活性和可验证性
- **vs CEGIS 框架**: 传统 CEGIS（Counterexample-Guided Inductive Synthesis）使用的反例精度较低，本文强调高精度反例引导是提升收敛效率的关键
- 该框架的 Learning + Verification 范式对其他形式化验证问题（如 Barrier Certificate、不变式合成）有迁移价值

## 评分

- 新颖性: ⭐⭐⭐⭐ Learning-enabled + 形式验证的结合思路有价值，高精度反例引导是有效的工程贡献
- 实验充分度: ⭐⭐⭐ 与 SMT-based 和传统 SOS 的对比充分，但实验系统规模和多样性有待扩展
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，Learner-Verifier框架描述规范
- 价值: ⭐⭐⭐ 在AI4Science / 形式化验证交叉领域有贡献，但应用场景相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/llm_reasoning/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](../../ACL2025/llm_reasoning/logicpro_program_guided_reasoning.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](../../NeurIPS2025/llm_reasoning/expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)
- [\[ACL 2025\] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](../../ACL2025/llm_reasoning/self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)

</div>

<!-- RELATED:END -->

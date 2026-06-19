---
title: >-
  [论文解读] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
description: >-
  [ACL 2026 Findings][LLM Agent][时间感知盲区] 揭示 LLM Agent 在多轮交互中的"时间盲区"(Temporal Blindness)——无法根据消息间流逝的真实时间来调整工具调用决策，并构建 TicToc 基准评估这一问题。 领域现状：LLM Agent 正越来越多地用于动态环境中的任务…
tags:
  - "ACL 2026 Findings"
  - "LLM Agent"
  - "时间感知盲区"
  - "工具调用决策"
  - "人类偏好对齐"
  - "多轮对话"
  - "时间敏感性"
---

# Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception

**会议**: ACL 2026 Findings  
**arXiv**: [2510.23853](https://arxiv.org/abs/2510.23853)  
**代码**: [GitHub](https://github.com/chengez/TicToc)  
**领域**: LLM Agent / Tool Use  
**关键词**: 时间感知盲区, 工具调用决策, 人类偏好对齐, 多轮对话, 时间敏感性

## 一句话总结

揭示 LLM Agent 在多轮交互中的"时间盲区"(Temporal Blindness)——无法根据消息间流逝的真实时间来调整工具调用决策，并构建 TicToc 基准评估这一问题。

## 研究背景与动机

**领域现状**：LLM Agent 正越来越多地用于动态环境中的任务执行，通过调用外部工具（搜索引擎、数据库等）来获取实时信息。现有工具调用评估主要关注调用的**准确性**（是否调用了正确的工具和参数），但忽略了**何时应该调用**的问题。

**现有痛点**：LLM Agent 默认假设上下文是静态的，不考虑消息之间流逝的真实世界时间。这导致两种失败模式：(1) **过度依赖**(over-reliance)——过度信任已过时的上下文而跳过必要的工具调用，产生错误输出；(2) **不足依赖**(under-reliance)——对稳定信息（如地球半径）也重复调用工具，造成不必要的延迟。

**核心矛盾**：人类天然地将时间流逝整合到决策中——知道什么时候需要重新查询，什么时候可以依赖之前的信息。但 LLM Agent 缺乏这种时间感知能力，即使提供了显式时间戳，也无法很好地利用。

**本文目标**：(1) 系统识别和量化 LLM Agent 的时间盲区问题；(2) 构建评估基准 TicToc；(3) 探索缓解策略。

**核心idea**：时间盲区是 LLM Agent 的一个基本局限，不是简单的提示工程能解决的——需要专门的后训练对齐才能有效缓解。

## 方法详解

### 整体框架

本文不是提出一个新模型，而是构建 TicToc 这套基准与诊断框架，系统回答「LLM Agent 到底会不会根据流逝的真实时间决定要不要调用工具」。输入是带时间戳的多轮对话轨迹，中间通过人类偏好标注确定每个时刻「该调工具还是该直接回答」的金标准，输出是对 18 个 LLM 的时间对齐能力评估、以及对失败根因的拆解。TicToc 覆盖 76 个场景、高/中/低三档时间敏感度，每条轨迹注入不同时间间隔的时间戳生成多个版本，从而把「时间变化」这一变量隔离出来单独考察。

### 关键设计

**1. TicToc 数据集构建：把「时间」做成可控变量的工具调用基准**

现有工具调用评估只看「调得对不对」、完全不含时间维度，于是作者专门设计了带时间变化的多轮场景。76 个场景按时间敏感度分为低（29）、中（25）、高（22）三档，兼顾只读和读写两种交互模式；在此之上定义 8 种对话变体（重复询问、对比、多检索单询问、简单推理、失败后重试、用户确认、重复请求、资源耗尽）覆盖典型决策情境。关键操作是为每条轨迹注入小/中/大三种时间间隔的时间戳，最终生成 5592 个样本，每个样本至少由 5 名标注者投票，Krippendorff's alpha 达 0.8574 表明标注高度一致。

**2. 时间盲区诊断分析：定位模型为什么用不上时间信息**

仅给出归一化对齐率（NAR，下文实验给出定义）这一总分不足以解释失败，作者进一步统计推理链中时间信息的实际使用情况：时间戳在推理轨迹里出现率不足 4%，「timestamp」关键词出现率低于 1.5%，所有时间相关词汇加起来也不到 15%——模型在思考时几乎从不主动引用时间。更进一步，他们发现「思考—回答不一致」现象：模型在推理中已经决定要调用工具，最终输出却直接作答。正是这种分析把「调用不对」的表象，归因到「根本没把时间纳入决策」这一根因。

**3. 对齐策略探索：从识别问题走到初步解法**

在确认时间盲区是普遍缺陷后，作者对比了两条缓解路径：朴素提示工程对多数模型收效甚微，说明这不是改改 prompt 就能解决的；而用 TicToc 子集做后训练对齐（监督微调）则能明显提升时间感知能力。这条对比给出了「提示无效、后训练有潜力」的明确结论，为后续工作指明方向。

### 损失函数 / 训练策略

评估指标为归一化对齐率 $NAR = \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$，50% 等同于随机猜测。后训练对齐使用 TicToc 的子集做监督微调。评估时所有模型使用 Temperature=0（Qwen3 推理模式除外）。

## 实验关键数据

### 主实验

| 条件 | 最高 NAR | 说明 |
|------|---------|------|
| 无时间戳 | ~55% | 接近随机猜测 |
| 有时间戳 | <65% | 最佳模型仍然很差 |

### 关键分析

| 分析维度 | 发现 | 说明 |
|---------|------|------|
| 对话长度 | 正相关 with 工具调用频率 | 模型用"轮次"启发式代替"时间"判断 |
| 推理模式(CoT) | 几乎无改善 | 时间感知不是推理问题 |
| 思考-回答不一致 | 高达 61.26% 的 FP | 推理与行动严重脱节 |
| 时间敏感度分层 | 模型在高/中/低敏感度上均匀失败 | 不是特定场景的问题 |

### 关键发现
- **没有一个模型**在提供时间戳后的 NAR 超过 65%，说明时间盲区是普遍且严重的问题
- 模型将**对话轮次**而非**实际时间**作为信息"过期"的启发式指标
- 推理(CoT)无法改善时间对齐，因为模型在推理过程中根本不自发地引用时间信息
- 后训练对齐显示出**强大的潜力**，是解决时间盲区的可行路径

## 亮点与洞察
- **问题识别的前瞻性**：时间盲区是一个此前被完全忽视但极其重要的 Agent 能力缺陷
- **过度/不足依赖的双向分析**：不仅分析了何时应该调用工具而没调用，还分析了不必要的重复调用
- **推理轨迹的深度分析**：通过统计推理链中的时间关键词揭示了问题的根源
- **"思考-回答不一致"的发现**：揭示了 LLM 推理与行动之间的系统性断裂
- **TicToc 数据集的设计质量**：场景多样性、时间戳注入方法、人类标注质量控制都非常精心

## 局限与展望
- 主要评估英语场景，跨语言的时间感知差异未知
- 后训练对齐的详细方法和大规模验证仍需进一步公开和扩展
- 时间戳以固定格式(ISO 8601)提供，不同时间表示方式的影响未探索
- 场景设计虽然多样但仍有限（76个），更多领域的覆盖有待扩展
- 未来应研究如何在预训练或对齐阶段系统性地注入时间感知能力

## 相关工作与启发
- **vs 现有工具调用评估(ToolBench等)**：现有工作关注"是否调用了正确工具"，本文首次关注"何时应该调用工具"
- **vs 时间推理研究**：已有工作研究 LLM 的时间推理能力，但都在孤立场景中，未结合 Agent 的工具调用决策
- **vs LLM 对齐**：将时间感知的工具调用决策作为对齐问题的新维度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次识别并系统研究 LLM Agent 的时间盲区，问题重要且此前完全被忽视
- 实验充分度: ⭐⭐⭐⭐⭐ 18个模型、76个场景、5592个标注样本、多维度分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数据集构建方法详尽，分析深入
- 价值: ⭐⭐⭐⭐⭐ 揭示了 LLM Agent 的基本能力缺陷，对 Agent 系统设计有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ACL 2026\] ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"](toolgrad_efficient_tool-use_dataset_generation_with_textual_gradients.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)

</div>

<!-- RELATED:END -->

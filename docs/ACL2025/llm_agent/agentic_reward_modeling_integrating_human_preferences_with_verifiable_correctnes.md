---
title: >-
  [论文解读] Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems
description: >-
  [ACL 2025][LLM Agent][奖励模型] 本文提出Agentic Reward Modeling范式和RewardAgent实现，将传统基于人类偏好的奖励模型与来自事实性验证和指令遵循验证的可验证正确性信号进行整合，通过Router-验证Agent-Judger三模块架构显著提升奖励模型的可靠性。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "奖励模型"
  - "可验证信号"
  - "事实性验证"
  - "指令遵循"
  - "Agentic workflow"
---

# Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems

**会议**: ACL 2025  
**arXiv**: [2502.19328](https://arxiv.org/abs/2502.19328)  
**代码**: [https://github.com/THU-KEG/Agentic-Reward-Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling)  
**领域**: LLM Agent / 对齐RLHF  
**关键词**: 奖励模型、可验证信号、事实性验证、指令遵循、Agentic workflow

## 一句话总结
本文提出Agentic Reward Modeling范式和RewardAgent实现，将传统基于人类偏好的奖励模型与来自事实性验证和指令遵循验证的可验证正确性信号进行整合，通过Router-验证Agent-Judger三模块架构显著提升奖励模型的可靠性。

## 研究背景与动机

**领域现状**：奖励模型（RM）在LLM的后训练（RLHF/DPO）和推理时扩展（best-of-n搜索）中扮演核心角色。当前主流的RM主要基于人类偏好训练，通过Bradley-Terry模型学习偏好排序。

**现有痛点**：基于人类偏好的RM存在主观偏差——倾向于偏好更长、更详细的回答（length bias），而忽略回答中的事实性错误和指令违反。例如，一个包含事实错误但语言流畅的回答可能获得高于事实正确但简洁的回答的奖励分数。

**核心矛盾**：人类偏好信号和可验证正确性信号是互补但不同维度的质量指标。人类偏好反映了语言风格等主观因素，可验证信号则提供了客观的事实和约束检查。现有RM只用了前者，导致奖励信号不可靠。

**本文目标**：设计一种新的奖励系统，能够结合人类偏好和多维度可验证信号，提供更可靠的奖励打分。

**切入角度**：受DeepSeek-R1等工作中可验证奖励（verifiable rewards）在训练LLM中展现的潜力启发，将可验证信号引入奖励模型，并通过Agentic workflow灵活集成多种验证维度。

**核心 idea**：提出"Agentic Reward Modeling"范式，奖励分数 = 人类偏好基础分 + 来自各验证Agent的正确性信号加权和，并通过Router动态选择需要的验证Agent。

## 方法详解

### 整体框架
RewardAgent的奖励公式为 $r(x,y) = \lambda \cdot r_{RM}(x,y) + \sum_{i \in A_x} w_i \cdot a_i(x,y)$，其中 $r_{RM}$ 是基础偏好奖励，$a_i$ 是第 $i$ 个验证Agent的正确性信号，$A_x$ 是根据指令 $x$ 选择的验证Agent子集。系统由三个模块组成：Router、Verification Agents、Judger。

### 关键设计

1. **Router（路由器）**:

    - 功能：分析指令内容，动态选择需要调用的验证Agent
    - 核心思路：将所有验证Agent的功能描述和触发条件输入LLM backbone，让LLM判断当前指令需要哪些维度的验证。例如，事实性问答需要调用事实性验证Agent，格式约束类指令需要调用指令遵循验证Agent
    - 设计动机：不是所有指令都需要全部维度的验证，动态选择可以减少推理成本并避免不相关验证带来的累积误差

2. **Factuality Verification Agent（事实性验证）**:

    - 功能：评估两个回答之间的事实性差异
    - 核心思路：采用成对比较而非独立评估，包含四个步骤：（1）Difference Proposal——识别两个回答在关键事实上的差异；（2）Query Generation——根据差异构造查询以获取支持证据；（3）Evidence Generation——使用搜索引擎或LLM参数知识生成支持证据；（4）Verification——基于证据和原始回答对每个回答打0或1的事实性分数。所有步骤由LLM backbone实现
    - 设计动机：相比FactScore等逐事实验证方法（需要大量搜索引擎调用），成对比较只需验证两个回答的差异部分，大幅减少推理成本。同时能捕捉到细微的事实性差异

3. **Instruction-Following Verification Agent（指令遵循验证）**:

    - 功能：检查回答是否满足指令中的硬性约束
    - 核心思路：三步流程：（1）Constraint Parsing——从指令中提取硬性约束（如长度限制、格式要求、关键词包含）；（2）Code Generation & Refinement——为每个约束生成Python检查脚本，输入回答输出0/1。加入自我修正步骤，如果代码执行报错则反馈错误信息让LLM重新生成；（3）Verification——执行Python脚本获取每个约束的二值分数，最终分数为所有约束分数的均值
    - 设计动机：硬约束（如"回答不超过100词"）可以通过代码精确验证，但现有RM很难准确评估这类约束。代码验证比LLM判断更可靠，且不需要额外训练

### 损失函数 / 训练策略
RewardAgent本身不需要训练——它在推理时通过Agentic workflow组合各模块。基础RM使用ArmoRM（预训练）。在DPO训练实验中，使用RewardAgent构造的偏好对在Zephyr-7B上进行DPO训练。

## 实验关键数据

### 主实验

| 模型 | RM-Bench Normal | RM-Bench Hard | JudgeBench | IFBench | Overall |
|------|----------------|---------------|------------|---------|---------|
| ArmoRM-8B | 76.7 | 34.6 | 66.2 | 59.5 | 56.5 |
| GPT-4o | 71.4 | 27.9 | 66.2 | 54.4 | 56.3 |
| Skywork-Gemma-27B | 82.7 | 35.1 | 68.4 | 56.1 | 59.2 |
| DeepSeek-R1 | 83.7 | 50.1 | 74.4 | 64.0 | 69.1 |
| **RewardAgent-mini** | **86.0** | **60.2** | 69.2 | **78.0** | **72.5** |
| **RewardAgent-Llama** | 79.3 | 53.5 | 63.9 | 67.8 | 63.2 |

### 消融实验

| 配置 | RM-Bench | JudgeBench | IFBench | 说明 |
|------|---------|------------|---------|------|
| RewardAgent-mini | 73.1 | 68.2 | 75.5 | 完整模型 |
| – factuality verifier | 54.0 | 52.9 | 73.6 | 事实性验证贡献最大 |
| – if verifier | 74.7 | 66.2 | 60.4 | 指令遵循验证对IFBench贡献大 |
| – both | 55.4 | 58.8 | 58.8 | 退化为基础RM |
| Oracle setting | 76.7 | 70.1 | 77.5 | Router完美选择时的上限 |

### 关键发现
- 事实性验证Agent贡献最大——去掉后RM-Bench从73.1暴跌到54.0，证实现有RM严重忽视事实性
- 使用开源Llama-8B作为backbone的RewardAgent-Llama已超越参数量多得多的专用RM和GPT-4o
- 搜索引擎辅助的证据生成在部分场景反而降低性能（噪声干扰），说明LLM内部知识检索可能更稳定
- 使用RewardAgent构造的DPO训练数据训练出的模型，在多个NLP基准上均优于用原始RM标注的数据

## 亮点与洞察
- Agentic Reward Modeling范式将奖励建模从"单一分数回归"扩展为"多Agent协作评估"，提供了一个可扩展的框架——未来可以方便地添加新的验证维度（如代码正确性、数学推理正确性）
- 用Python代码验证硬约束是一个巧妙设计，将不可微的约束检查转化为可执行的程序验证，比LLM判断更精确且可解释
- 成对事实性验证只需检查差异部分，比FactScore高效得多，这个思路可以迁移到其他pair-wise评估任务

## 局限与展望
- 当前只实现了事实性和指令遵循两种验证Agent，对于代码生成、数学推理等需要不同类型的验证
- Router的准确性影响整体表现——Oracle实验显示Router选择还有提升空间
- 推理成本较高——每次奖励打分需要多次LLM调用（Router + 验证Agent + Judger）
- Judger目前使用简单加权求和（$\lambda = w_i = 1.0$），自适应权重调整留作未来工作

## 相关工作与启发
- **vs ArmoRM**: ArmoRM是纯偏好RM，RewardAgent在其基础上叠加验证信号，性能大幅提升
- **vs FactScore**: FactScore需要逐事实验证，RewardAgent的成对比较更高效
- **vs DeepSeek-R1的verifiable rewards**: DeepSeek-R1将可验证奖励用于RL训练，本文将其扩展到通用RM框架中

## 评分
- 新颖性: ⭐⭐⭐⭐ Agentic reward modeling范式有开创性，但各模块技术相对成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 基准评测、BoN搜索、DPO训练三个应用场景全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，方法描述详尽，图表设计好
- 价值: ⭐⭐⭐⭐⭐ 开辟了奖励模型的新方向，开源代码促进后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2026\] AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning](../../ACL2026/llm_agent/adarubric_task-adaptive_rubrics_for_reliable_llm_agent_evaluation_and_reward_lea.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](../../ACL2026/llm_agent/exploring_reasoning_reward_model_for_agents.md)
- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](../../ICML2026/llm_agent/process_reward_agents_for_steering_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->

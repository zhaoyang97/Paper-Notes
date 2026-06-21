---
title: >-
  [论文解读] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
description: >-
  [ICLR 2026 (Oral)][LLM Agent][动态环境] 提出 Gaia2 基准，在动态异步环境中评估 LLM Agent 的能力，引入时间约束、噪声事件、歧义解析和多 Agent 协作等现实场景，配合可验证奖励的写操作验证器，使基准可直接用于 RLVR 训练，评估显示最强模型 GPT-5 (high) 仅达42% pass@1。
tags:
  - "ICLR 2026 (Oral)"
  - "LLM Agent"
  - "动态环境"
  - "异步交互"
  - "benchmark"
  - "强化学习"
---

# Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments

**会议**: ICLR 2026 (Oral)  
**arXiv**: [2602.11964](https://arxiv.org/abs/2602.11964)  
**代码**: 基于 Agents Research Environments (ARE) 平台，开源  
**领域**: LLM Agent 评估  
**关键词**: LLM Agent, 动态环境, 异步交互, benchmark, 强化学习

## 一句话总结

提出 Gaia2 基准，在动态异步环境中评估 LLM Agent 的能力，引入时间约束、噪声事件、歧义解析和多 Agent 协作等现实场景，配合可验证奖励的写操作验证器，使基准可直接用于 RLVR 训练，评估显示最强模型 GPT-5 (high) 仅达42% pass@1。

## 研究背景与动机

当前 LLM Agent 的评估存在根本性缺陷：大多数基准依赖**静态**或**同步**环境。在这些设置中，环境不会独立于 Agent 的操作而变化——Agent 拥有完全的时间控制权，可以任意暂停、思考，环境状态始终等待 Agent 的下一步操作。

然而，真实世界的任务环境完全不同：
- **时间敏感性**：航班价格波动、库存变化、截止日期临近
- **异步事件**：新消息到达、状态更新独立发生
- **噪声与歧义**：不完整信息、矛盾的上下文、需要澄清的需求
- **多方协作**：需要与其他 Agent 或人类协调

现有基准（如原始 GAIA）只测试静态问答和工具调用，无法评估 Agent 在这些**现实维度**上的能力。这导致了一个严重的"**sim2real gap**"——基准上的好成绩不能预测真实部署中的表现。

Gaia2 的设计目标是创建一个更贴近现实的评估平台，同时保持可量化和可复现性。

## 方法详解

### 整体框架

Gaia2 把评估场景建在开源的 Agents Research Environments (ARE) 平台之上，每个场景由一个独立于 Agent 操作而持续演化的动态环境、一段任务描述、以及一组细粒度的写操作验证器组成。Agent 在这个"会自己往前走"的环境里被要求一边感知变化、一边在时间窗口内做出并执行决策，验证器则在每个关键行动点上判断对错，从而把评估从"最终答案对不对"推进到"每一步行动对不对"。

### 关键设计

**1. 动态异步环境：打破"请求-响应"的时间控制权**

传统 Agent 基准默认环境会一直等 Agent 想清楚再响应，Agent 因此握有完全的时间控制权，可以任意暂停、回溯。Gaia2 反其道而行，让环境独立于 Agent 操作持续往前走：价格会波动、库存会变化、新消息会异步到达，机会窗口一旦错过就消失。Agent 必须在限定时间窗口内决策、持续监控环境状态、并对意外事件和状态转换做出反应。这一改动把测试焦点从"能否规划出一条静态最优路径"转向"能否在不确定且不断变化的条件下持续适应"，正是现实部署里最容易暴露差距的能力。

**2. 多维度能力覆盖：把"现实"拆成可测量的轴**

只看一个笼统的通过率无法定位模型到底弱在哪里，所以 Gaia2 刻意把场景设计成覆盖五个核心维度——时间敏感决策（限时条件下选最优行动）、噪声鲁棒性（从不完整或矛盾信息中提取关键事实）、歧义解析（主动澄清或在多义理解中选最合理的解释）、多 Agent 协作（与其他 Agent 交换信息、协调行动）、以及环境适应（响应动态变化并修正计划）。这样的拆分让评估能给出按维度分解的能力剖面，直接看出模型究竟弱在"反应快慢"还是"信息抗噪"，而不是只剩一个总分。

**3. 写操作验证器：让奖励既可评估又可训练**

如果只看 Agent 最终交出的答案对不对，过程中每一步决策的好坏就被冲掉了。Gaia2 最核心的技术创新是在每个场景里预先定义若干"写操作"（write action）检查点，验证器逐点判断 Agent 在这些关键行动上的操作是否正确，从而把评估粒度从结果级别细化到过程中的每一步决策质量。更关键的是，这种逐步、可验证的奖励信号天然适配强化学习——即 RLVR（Reinforcement Learning from Verifiable Rewards），使同一套基准既能用来打分，又能直接作为训练信号驱动 Agent 自我改进，打通了"基准 → 训练"的闭环。

**4. 基于 ARE 的可扩展架构：环境逻辑与验证逻辑解耦**

要让上面三点能持续扩展而不沦为一次性测试集，系统建在开源的 Agents Research Environments（ARE）框架上，把环境演化逻辑和验证逻辑分离开来：新场景可以通过标准接口接入，并兼容多种 Agent 框架。场景取材于购物、旅行规划等贴近日常的消费者环境，既保证了任务的现实感，也让基准本身成为可持续扩展的研究基础设施。评估上以 pass@1（单次尝试通过率）为主指标，配合按维度分解的性能剖面和"完成速度 vs API 调用成本"的效率权衡，给出多面而非单点的画像。

## 实验关键数据

### 主实验：模型整体表现

| 模型 | pass@1 | 类型 | 突出特点 |
|------|--------|------|---------|
| GPT-5 (high) | 42% | 闭源 | 综合最强但时间敏感任务弱 |
| Claude-4 Sonnet | ~35-38% | 闭源 | 准确性与速度平衡，成本更优 |
| Kimi-K2 | 21% | 开源 | 开源模型中最佳 |
| 其他开源模型 | <20% | 开源 | 显著落后于闭源 |

### 能力维度分析

| 能力维度 | GPT-5 | Claude-4 | Kimi-K2 | 说明 |
|---------|-------|----------|---------|------|
| 时间敏感决策 | 弱 | 中等 | 弱 | 最具挑战的维度 |
| 噪声鲁棒性 | 强 | 强 | 中 | 闭源模型优势明显 |
| 歧义解析 | 强 | 中 | 弱 | 需要强推理能力 |
| 多Agent协作 | 中 | 中 | 弱 | 所有模型的薄弱环节 |
| 环境适应 | 中 | 中 | 弱 | 动态调整计划的能力 |

### 消融实验

| 对比维度 | 关键发现 |
|---------|---------|
| 静态 vs 动态环境 | 动态环境下所有模型性能显著下降 |
| 同步 vs 异步 | 异步事件进一步拉大了模型间差距 |
| 单 Agent vs 多 Agent | 多 Agent 场景是当前最大瓶颈 |
| 无时间限制 vs 有时间限制 | 时间约束对开源模型影响更大 |

### 关键发现

1. **没有模型在所有维度上占优**：GPT-5 综合最强但在时间敏感任务上失败，Claude-4 在成本效率上更好
2. **42% pass@1 暴露了巨大差距**：即使最强模型也有近60%的场景无法通过，说明现实Agent任务仍极具挑战
3. **开源与闭源的鸿沟**：21% vs 42% 的差距表明开源模型在Agent场景中的能力仍然不足
4. **"sim2real gap"确实存在**：在静态基准上表现接近的模型，在Gaia2的动态环境中差异被放大
5. **RLVR 的潜力**：写操作验证器提供的细粒度奖励信号为基于强化学习的Agent训练开辟了道路

## 亮点与洞察

- **从"能问答"到"能行动"的范式转变**：Gaia2 评估的不是 Agent 的知识或推理，而是在动态环境中采取正确行动的能力
- **写操作验证器是关键创新**：使基准同时服务于评估和训练两个目的，大大提升了基准的实用价值
- **异步性是被忽视的核心挑战**：现有 Agent 系统几乎都假设同步交互，Gaia2 首次系统性地测试了异步场景
- **ICLR 2026 Oral 说明其重要性**：被选为口头报告反映了社区对真实Agent评估的迫切需求
- **开源 ARE 平台的生态价值**：不仅是一个基准，更是一个可持续扩展的研究基础设施

## 局限与展望

1. **消费者环境可能不代表所有领域**：购物、旅行等场景与科学研究、软件开发等专业领域的Agent需求不同
2. **评估的可复现性挑战**：动态环境的随机性可能导致不同运行间结果波动
3. **写操作验证器的设计需要人工**：每个场景的验证器需要人工定义检查点和正确性标准，限制了自动化扩展
4. **未充分测试工具使用能力**：虽然环境是动态的，但工具集和API接口的复杂度可能不够
5. **多 Agent 场景的规模有限**：当前可能主要是双 Agent 场景，更大规模的协作测试有待开发

## 相关工作与启发

- **与 GAIA (2023) 的继承关系**：Gaia2 在前代基础上引入了动态性和异步性这两个质变维度
- **与 WebArena、AgentBench 的区别**：这些基准侧重于静态网页交互或API调用，Gaia2 强调环境的时间演化
- **与 SWE-bench 互补**：后者测试代码生成能力，Gaia2 测试环境交互和决策能力
- **对 Agent 训练方法的影响**：RLVR-ready 的设计使 Gaia2 可能成为训练更强 Agent 的关键数据来源
- **对 Agent 架构设计的启发**：需要考虑时间感知、异步事件处理模块和动态计划调整机制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (动态异步Agent评估 + RLVR-ready设计, 领域引领性)
- 实验充分度: ⭐⭐⭐⭐ (覆盖主流模型但场景数量未知)
- 写作质量: ⭐⭐⭐⭐ (结构合理，分析清晰)
- 价值: ⭐⭐⭐⭐⭐ (Agent评估的重要里程碑，Oral接收实至名归)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](../../CVPR2025/llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)
- [\[ICLR 2026\] Real-Time Reasoning Agents in Evolving Environments](real-time_reasoning_agents_in_evolving_environments.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](../../AAAI2026/llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[ICLR 2026\] Benchmarking LLM Tool-Use in the Wild](benchmarking_llm_tool-use_in_the_wild.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [论文解读] TAI3: Testing Agent Integrity in Interpreting User Intent
description: >-
  [NeurIPS 2025][LLM Agent][Agent 测试] 提出 TAI3，一个以 API 为中心的 LLM Agent 意图完整性压力测试框架，通过语义分区（Semantic Partitioning）将自然语言输入空间组织为结构化测试网格，再利用意图保持变异（Intent-Preserving Mutation）和策略记忆（Strategy Memory）高效暴露 Agent 在执行用户任务时的意图理解错误。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Agent 测试"
  - "意图完整性"
  - "等价类划分"
  - "压力测试"
  - "API 调用"
---

# TAI3: Testing Agent Integrity in Interpreting User Intent

**会议**: NeurIPS 2025  
**arXiv**: [2506.07524](https://arxiv.org/abs/2506.07524)  
**代码**: 无  
**领域**: LLM Agent / AI Safety  
**关键词**: Agent 测试, 意图完整性, 等价类划分, 压力测试, API 调用

## 一句话总结

提出 TAI3，一个以 API 为中心的 LLM Agent 意图完整性压力测试框架，通过语义分区（Semantic Partitioning）将自然语言输入空间组织为结构化测试网格，再利用意图保持变异（Intent-Preserving Mutation）和策略记忆（Strategy Memory）高效暴露 Agent 在执行用户任务时的意图理解错误。

## 研究背景与动机

### 1. 领域现状

LLM Agent 正被广泛部署于软件开发、电商、智能家居等领域，通过自然语言指令调用外部 API 完成任务。这些 Agent 将高层用户意图翻译为具体的 API 调用序列，但自然语言的模糊性使得 Agent 行为可能偏离用户真实意图。

### 2. 现有痛点

- **固定基准不足**：现有 LLM Agent 安全基准（如 AgentSafetyBench、ToolEmu）依赖固定测试用例，无法跟上工具包快速演化的节奏
- **对抗测试错位**：大量工作聚焦于 jailbreaking 和 prompt injection，而非确保 Agent 在正常使用下能鲁棒地执行良性任务
- **传统测试失效**：经典软件测试假设结构化输入接口，无法处理自然语言的开放性和模糊性
- **覆盖无法量化**：缺乏类似代码覆盖率的指标来衡量 Agent 行为空间被测试了多少

### 3. 核心矛盾

API 规格说明是精确的、形式化的，但用户自然语言指令是模糊的、多变的。这一差距使得 Agent 可能在看似合理的输入上发生意图误解，而现有测试方法缺乏系统手段来发现这些隐蔽错误。

### 4. 本文目标

为 LLM Agent 设计一个系统化的意图完整性（Intent Integrity）测试框架，能够：(1) 量化验证 Agent 意图保真度；(2) 生成现实任务作为测试用例；(3) 在合理查询预算下高效发现错误。

### 5. 切入角度

核心洞察：Agent 的行为（及其潜在漏洞）可以通过底层 API 的结构来系统描述。借鉴经典黑盒测试中的"等价类划分"技术，将 API 参数的值域按意图类别分区，从而获得有限且可解释的测试网格。

### 6. 核心 idea

通过 API 参数的语义分区构建结构化测试空间，再用意图保持变异和策略记忆高效搜索 Agent 意图理解失败的边界情况。

## 方法详解

### 整体框架

TAI3 分为两个阶段：

- **Stage 1 — Semantic Partitioning**：对每个 API 参数，按 VALID / INVALID / UNDERSPEC 三类意图类别进行等价类划分，生成参数-分区表，每个 cell 生成一个种子任务
- **Stage 2 — Intent-Preserving Mutation**：对种子任务进行意图保持变异，用轻量预测模型排序，选择最可能触发错误的变体送给 Agent 测试

### 关键设计

#### 1. 语义分区（Semantic Partitioning）

- **功能**：将 API 参数值域按意图类别分为 VALID（合法值）、INVALID（非法值）、UNDERSPEC（信息不足）三大类，再在每类内做等价类划分
- **核心思路**：对参数 $p$ 的值域 $\mathcal{D}_p$ 按类别 $c \in \{VA, IV, US\}$ 分解为 $\mathcal{D}_p^c = \mathcal{E}_{p,c}^1 \cup \cdots \cup \mathcal{E}_{p,c}^{m(p,c)}$，每个等价类代表一种语义不同的输入方式
- **举例**：`start_time` 参数在 VALID 下可分为"标准日期格式"、"相对时间表达"等；INVALID 下有"不存在的日期"、"不支持的功能"等
- **设计动机**：API 是形式化定义的，借此可精确完整地规格化 Agent 行为空间，类似代码覆盖率的概念

#### 2. 种子任务生成

- **功能**：对分区表的每个 cell $(p, c, i)$，用 LLM 生成一条现实的自然语言用户指令
- **约束**：每条指令需选取对应等价类的代表值，设计为能触发类别 $c$ 对应行为
- **保证**：每个分区 cell 都有对应种子任务 → 语义输入空间的完整覆盖

#### 3. 意图保持变异（Intent-Preserving Mutation）

- **功能**：从种子任务出发，迭代生成变体，保持原始意图不变但增加 Agent 出错概率
- **意图一致性检查**：对每个候选变体 $u'$，用 LLM 验证其是否与原始意图 $\mathcal{I}(u)$ 一致（检查比推断更容易）
- **错误概率估计**：用小语言模型（phi4-mini）计算变异任务的错误似然：$\sum_i \log P(\mathcal{I}(u)_i | u' \cdot \mathcal{I}(u)_{<i}; \theta)$，分数越低说明越难从变异任务还原原始意图，越可能触发 Agent 错误
- **设计动机**：实际运行 Agent 代价高（每个动作 5-26 秒），用轻量代理模型排序可大幅减少查询次数

#### 4. 常青策略记忆（Evergreen Strategy Memory）

- **功能**：记录成功触发错误的变异策略模式，按参数数据类型和意图类别索引
- **策略示例**："在两个枚举选项间犹豫"、"将金额拆成两句引入数学表达式"
- **复用机制**：新种子任务到来时，检索相关策略 → LLM 重排序 → 取 Top-3 指导变异
- **设计动机**：类似人类测试者随经验积累变得更高效，框架也能从历史成功模式中学习

### 损失函数 / 训练策略

框架不涉及模型训练。核心优化目标是在固定查询预算内最大化 EESR（Error-Exposing Success Rate），即在语义分区中发现至少一个 Agent 错误的比例。效率指标为 AQFF（Average Queries to First Failure），即触发首次失败所需的平均查询数。

## 实验关键数据

### 主实验

数据集：80 个 API（233 参数），覆盖金融、医疗、智能家居、物流、办公 5 个领域。

| 领域 | 目标模型 | VALID EESR (SelfRef→TAI3) | INVALID EESR (SelfRef→TAI3) | UNDERSPEC EESR (SelfRef→TAI3) |
|-----|---------|--------------------------|----------------------------|------------------------------|
| Finance | Llama-3.1-8B | 65.0→80.5 (+15.5) | 78.0→85.4 (+7.4) | 58.5→73.2 (+14.7) |
| Finance | GPT-4o-mini | 41.5→61.0 (+19.5) | 65.9→73.2 (+7.3) | 61.0→65.9 (+4.9) |
| Healthcare | Llama-3.1-8B | 66.0→70.2 (+4.2) | 51.1→55.3 (+4.2) | 57.4→61.7 (+4.3) |
| Smart Home | GPT-4o-mini | 63.0→72.2 (+9.2) | 57.4→63.0 (+5.6) | 61.1→63.0 (+1.9) |
| Office | Llama-3.1-8B | 60.0→64.0 (+4.0) | 54.0→58.0 (+4.0) | 65.7→82.0 (+16.3) |

TAI3 在 **所有领域、所有意图类别、所有目标模型** 上均超越 SelfRef 基线。

### 消融实验

| 组件 | 效果 |
|-----|------|
| 错误似然排序 vs 随机选择 | 排序方法在所有 k 值上触发更多错误，验证预测模型有效 |
| 错误似然排序 vs Select Last 5 | 基于排序显著优于"后 5 轮反思选最后 5 个" |
| 策略记忆 | 跨 API 和领域可迁移有效变异模式 |
| 语义分区覆盖度 | AgentSafetyBench 和 ToolEmu 的测试用例仅覆盖 TAI3 分区的 8.3%-50%，大量分区未被覆盖 |

### 泛化性验证

弱测试模型（Llama-3.1-8B, Qwen3-30B-A3B）也能有效暴露强目标模型（Claude-3.5-Haiku, Gemini-2.5-Pro, GPT-o3-mini）的错误。Top-1 设置下弱强差距小；Top-5 设置下 GPT-4o-mini 排序能力最强。开源目标模型（Llama-3.3-70B, DeepSeek-R1-70B）比闭源模型更脆弱。

### 关键发现

1. **EESR 提升**：TAI3 在 Finance+VALID 场景最高提升 19.5 个百分点（GPT-4o-mini 作为目标）
2. **查询效率**：AQFF 在 UNDERSPEC 类别降低最高 12%，变异排序显著减少搜索开销
3. **现有基准不足**：AgentSafetyBench 几乎无 INVALID 测试用例，ToolEmu 完全缺失 INVALID 输入，两者覆盖率普遍低于 50%
4. **弱打强**：较弱的测试 LLM 可以成功发现强模型的意图完整性漏洞，说明漏洞是 Agent 本身的问题而非测试模型能力限制

## 亮点与洞察

1. **桥接 API 形式化与自然语言模糊性**：将软件测试中经典的等价类划分技术首次应用于 LLM Agent 测试，提供了量化测试覆盖的方法论
2. **三类意图完整性分类**：VALID/INVALID/UNDERSPEC 的划分简洁但全面，为衡量 Agent 鲁棒性提供了清晰框架
3. **轻量预测模型排序**：用小模型（phi4-mini）估计错误概率，避免昂贵的 Agent 运行，实现"便宜探测、昂贵验证"的分层策略
4. **策略记忆的可迁移性**：按数据类型和意图类别索引的策略可跨 API 和领域复用，类似人类测试经验积累
5. **揭示现有基准短板**：定量证明 AgentSafetyBench 和 ToolEmu 在意图完整性维度覆盖严重不足

## 局限与展望

1. **轨迹可观测性假设**：TAI3 需要访问 Agent 的 API 调用轨迹，对于仅暴露高层输出（如网页交互）的商业 Agent 不适用
2. **仅关注 API 调用层**：未涉及高层安全问题（政策违规、隐私泄露、有害内容），范围限于 API 参数层面的意图理解
3. **依赖 LLM 做意图一致性检查**：意图保持的判断本身也依赖 LLM，可能引入噪声
4. **分区粒度**：等价类划分依赖 LLM 的语义分析，不同 API 的分区质量可能有较大差异
5. **多步交互**：当前方法主要测试单轮 API 调用，对多步规划场景的覆盖有待扩展

## 相关工作与启发

- **软件测试经典方法**：等价类划分（Equivalence Class Partitioning）从传统黑盒测试迁移到 NL 输入空间，展示了传统方法的新生命力
- **NLP 鲁棒性测试**（CheckList, TextAttack）：聚焦于模型级对抗扰动，TAI3 扩展到 Agent 级的意图完整性
- **ToolFuzz / PDoctor**：ToolFuzz 关注工具文档和实现的 bug，PDoctor 检查高层规划的约束遵循，TAI3 聚焦底层动作与用户意图的对齐
- **Agent Safety Benchmarks**（AgentSafetyBench, ToolEmu）：TAI3 的分区分析定量揭示了这些基准的覆盖率不足
- **启发**：策略记忆 + 预测排序的组合可推广到任何 LLM 系统的高效自动化测试场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将等价类划分应用于 Agent 意图完整性测试的思路新颖，三类意图分类清晰实用
- 实验充分度: ⭐⭐⭐⭐ — 80 个 API、5 领域、多目标模型、泛化实验和消融全面，缺少多步交互场景
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、形式化与直觉并重，图表优秀（尤其 Figure 1-3 的示例极具说服力）
- 价值: ⭐⭐⭐⭐ — 填补了 Agent 意图完整性系统化测试的空白，对 Agent 部署质量保障具有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Grounding Agent Memory in Contextual Intent](../../ACL2026/llm_agent/grounding_agent_memory_in_contextual_intent.md)
- [\[ACL 2025\] iAgent: LLM Agent as a Shield between User and Recommender Systems](../../ACL2025/llm_agent/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](../../ICLR2026/llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](../../ICML2026/llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)

</div>

<!-- RELATED:END -->

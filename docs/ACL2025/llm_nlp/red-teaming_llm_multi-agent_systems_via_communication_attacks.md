---
title: >-
  [论文解读] Red-Teaming LLM Multi-Agent Systems via Communication Attacks
description: >-
  [ACL2025][LLM 其他][multi-agent systems] 提出 Agent-in-the-Middle (AiTM) 攻击，通过拦截和篡改 LLM 多智能体系统中的 agent 间通信消息（而非直接修改 agent 本身），利用一个带反思机制的对抗性 agent 生成上下文感知的恶意指令，在多种框架/通信结构/真实应用上均取得 40%~100% 的攻击成功率。
tags:
  - "ACL2025"
  - "LLM 其他"
  - "multi-agent systems"
  - "red teaming"
  - "communication attack"
  - "man-in-the-middle"
  - "LLM safety"
---

# Red-Teaming LLM Multi-Agent Systems via Communication Attacks

**会议**: ACL2025  
**arXiv**: [2502.14847](https://arxiv.org/abs/2502.14847)  
**代码**: 待发布  
**领域**: LLM/NLP  
**关键词**: multi-agent systems, red teaming, communication attack, man-in-the-middle, LLM safety

## 一句话总结

提出 Agent-in-the-Middle (AiTM) 攻击，通过拦截和篡改 LLM 多智能体系统中的 agent 间通信消息（而非直接修改 agent 本身），利用一个带反思机制的对抗性 agent 生成上下文感知的恶意指令，在多种框架/通信结构/真实应用上均取得 40%~100% 的攻击成功率。

## 研究背景与动机

**领域现状**：LLM 多智能体系统（LLM-MAS）通过多个专长 agent 协作（辩论、任务分解、投票等）解决复杂问题，在软件开发（MetaGPT、ChatDev）、科学研究等领域取得成功。通信是 LLM-MAS 的核心基础设施，决定了 agent 间的信息共享与协调。

**现有痛点**：已有安全研究主要聚焦于攻击单个 agent——将良性 agent 改为恶意 agent，或在输入端注入对抗样本，但对通信信道本身的安全性几乎未被探索。

**核心矛盾**：通信是 LLM-MAS 高效协作的命脉，但同时也是潜在的攻击面——恶意信息可通过通信链路传播并放大，影响整个系统。在去中心化部署场景中，agent 间消息依赖网络传输，非常容易被窃听/篡改。

**本文目标**：验证仅通过拦截和篡改 agent 间通信消息（不修改 agent 配置、能力和工具），是否能有效攻击整个多智能体系统。

**切入角度**：借鉴传统网络安全中的中间人攻击（Man-in-the-Middle），提出 Agent-in-the-Middle 框架，使用一个外部 LLM 对抗 agent 来拦截消息、生成上下文感知的恶意指令，间接操控受害 agent 的行为。

**核心 idea**：通过在 agent 通信信道上部署带反思机制的对抗 agent，仅篡改消息即可系统性地攻破 LLM-MAS，无需直接控制任何系统内 agent。

## 方法详解

### 整体框架

AiTM 在 LLM-MAS 的通信链路上插入一个外部对抗 agent $A^{ad}$，该 agent 拦截发送给受害 agent $A^{vic}$ 的所有消息，分析上下文后生成恶意指令替换或附加到原始消息中，诱导受害 agent 产生符合攻击目标的响应，进而影响系统中的其他 agent。

### 关键设计 1：威胁模型

- **功能**：定义攻击者的能力边界——仅能拦截和篡改发送给某个特定受害 agent 的消息，不能修改其他 agent、通信结构或外部工具。
- **为什么**：确保攻击场景的实际可行性，对应去中心化系统中的窃听攻击场景。
- **怎么做**：攻击者知道 LLM-MAS 正在处理的任务，但不知道系统内部配置（通信结构、模型类型）。攻击目标包括拒绝服务（DoS）和目标行为诱导（Targeted Behavior）。

### 关键设计 2：反思机制（Reflection Mechanism）

- **功能**：让对抗 agent 在每轮消息拦截时，评估上一轮指令的攻击效果，并据此生成更优的恶意指令。
- **为什么**：单次静态指令注入效果有限，迭代反思可以让攻击越来越精准，类似于 prompt 优化器，用上一轮反馈作为奖励信号。
- **怎么做**：在第 $t$ 轮，对抗 agent 接收拦截的消息 $M^t_{vic,r}$ 和上一轮指令 $I^{t-1}$，生成新指令 $I^t = A^{ad}(I^{t-1}, M^t_{vic,r}, G)$，其中 $G$ 是恶意目标。新指令要求与受害 agent 的预定义角色对齐，保持上下文一致性。

### 关键设计 3：攻击目标适配

- **Targeted Behavior**：在 MMLU 上诱导输出标签 Unicode 偏移 4（如 A→E）；在 HumanEval/MBPP 上注入 `safety_check` 恶意函数扫描 `/Users` 目录。
- **DoS 攻击**：诱导系统拒绝服务，输出 "I can not assist the request"。

### 损失函数/训练策略

AiTM 不需要训练——对抗 agent 是一个 prompt 驱动的 LLM，通过精心设计的 system prompt 编码恶意目标，并在运行时通过反思机制动态优化攻击指令。

## 实验关键数据

### 主实验：多框架多结构攻击成功率（Table 1）

| 数据集 | 攻击类型 | AutoGen-Chain | AutoGen-Tree | AutoGen-Complete | Camel-Chain | Camel-Complete |
|--------|---------|--------------|-------------|-----------------|------------|---------------|
| MMLU-bio | Target | 93.1 | 40.7 | 43.9 | 81.7 | 68.2 |
| HumanEval | Target | 95.2 | 90.4 | 96.3 | 97.6 | 96.2 |
| MBPP | Target | 96.9 | 90.5 | 92.4 | 98.5 | 95.9 |
| MMLU-bio | DoS | 96.3 | 93.7 | 94.9 | 98.4 | 96.5 |
| HumanEval | DoS | 86.5 | 83.9 | 87.3 | 93.6 | 95.8 |

### 消融实验：说服力等级影响（Table 3）

| 说服力等级 | Tree-Target | Tree-DoS | Complete-Target | Complete-DoS |
|-----------|------------|---------|----------------|-------------|
| Level 1 | 19.5 | 79.8 | 13.6 | 74.1 |
| Level 2 | 27.1 | 85.2 | 30.3 | 80.8 |
| Level 3 | 40.7 | 93.7 | 43.9 | 94.9 |

### 真实应用攻击（Table 4）

| 应用 | 受害Agent | SoftwareDev | HumanEval | MBPP |
|------|----------|------------|-----------|------|
| MetaGPT | Product Manager | 100.0 | 90.4 | 95.1 |
| MetaGPT | Engineer | 100.0 | 75.7 | 80.4 |
| ChatDev | CTO | 45.4 | 52.7 | 55.9 |
| ChatDev | Programmer | 63.6 | 60.1 | 69.3 |

### 关键发现

1. **AiTM 全场景有效**：所有配置下 ASR 均超 40%，多数超 70%，Chain 结构最脆弱（95%+）。
2. **DoS 比 Targeted 更容易**：拒绝服务比注入特定行为更简单，23/32 个 DoS 实验 ASR 超 80%。
3. **代码注入比答案篡改更隐蔽**：HumanEval/MBPP 上 Target ASR 远高于 MMLU，因为辅助函数不干扰正常功能。
4. **说服力显著影响攻击效果**：最高等级比最低等级提升 20-30 个百分点。
5. **攻击后期 agent（靠近决策端）效果更强**：Complete 结构中攻击第 3 个 agent 比第 2 个提升 30%+。
6. **更强的对抗模型 → 更高 ASR**：GPT-4o 对抗 agent 比 GPT-3.5-turbo 提升约 14%。
7. **MetaGPT 几乎完全被攻破**（Chain 结构无监控），ChatDev 因额外的角色约束提供了部分抵抗力。

## 亮点与洞察

1. **新攻击面的开拓**：首次系统性研究 LLM-MAS 通信信道的安全性，将传统 MITM 攻击迁移到 AI 多智能体场景，视角非常新颖。
2. **最小权限攻击**：攻击者仅篡改消息，不修改任何系统组件，这种最小化假设更贴近真实威胁场景。
3. **反思机制的精妙设计**：将 prompt 优化的思想用于攻击迭代，对抗 agent 会根据拦截到的上下文动态调整策略。
4. **揭示通信结构与安全性的关系**：双向讨论结构（Complete）比单向传递结构（Chain）更安全，为 MAS 设计提供安全指导。
5. **真实框架的验证**：MetaGPT 和 ChatDev 的攻击实验证明威胁在实际系统中真实存在。

## 局限与展望

1. **仅使用黑盒 GPT 模型**：未测试开源模型（LLaMA、Mistral 等），攻击效果在开源模型上是否一致尚不清楚。
2. **通信结构覆盖有限**：仅测试 4 种结构 + 2 个真实应用，更复杂的动态通信拓扑未被探索。
3. **缺乏防御方案**：论文揭示了攻击但未提出有效的防御策略（如消息签名、异常检测、通信加密等）。
4. **攻击成本分析缺失**：未报告对抗 agent 的 token 开销和 API 调用成本。
5. **多受害 agent 场景**：仅拦截单个 agent 的消息，多 agent 同时被攻击的场景未被考虑。

## 相关工作与启发

### vs 恶意 Agent 攻击（Yu et al., 2024; Huang et al., 2024）

恶意 agent 攻击需要将系统内的某个 agent 替换为攻击者控制的版本，这要求攻击者拥有更高权限。AiTM 仅需拦截通信信道，攻击假设更弱但仍然高效——甚至在某些场景下 ASR 更高，因为不受 agent 角色限制的约束。

### vs 对抗性输入/Prompt 注入（Zhang et al., 2024）

传统 prompt 注入针对单个 agent 的输入，AiTM 通过操控 agent 间通信实现系统级攻击。区别在于：(1) AiTM 利用反思机制迭代优化，而非一次性注入；(2) 攻击影响沿通信链路传播，单点攻击可影响全局；(3) 更难被单个 agent 的安全过滤器检测到。

### vs 多智能体辩论安全（Amayuelas et al., 2024）

Amayuelas 等研究了 agent 在辩论中被说服放弃任务的场景，但攻击发生在 agent 层面。AiTM 将攻击下沉到通信层，揭示了即使每个 agent 都是良性的，通信信道被攻破仍然足以瘫痪系统。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统性提出 LLM-MAS 通信层面的中间人攻击，攻击面定义清晰、假设合理，是安全领域的重要贡献。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 2 个框架、4 种结构、4 个数据集、2 种攻击目标、2 个真实应用，并对受害位置/说服力/模型做了消融，但缺乏防御实验和开源模型测试。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰、威胁模型定义严谨、通信形式化表述完整，但部分符号定义冗余。
- **价值**: ⭐⭐⭐⭐⭐ — 对 LLM-MAS 安全领域有重要警示作用，揭示了一个被忽视的关键攻击面，对多智能体系统的部署和安全设计具有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)
- [\[ACL 2025\] Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems](understanding_and_meeting_practitioner_needs_when_measuring_representational_har.md)

</div>

<!-- RELATED:END -->

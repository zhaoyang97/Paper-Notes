---
title: >-
  [论文解读] LLM Agent Communication Protocol (LACP) Requires Urgent Standardization: A Telecom-Inspired Protocol is Necessary
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent communication] 这篇 position paper 指出当前 LLM Agent 通信的碎片化生态类似早期网络的"协议战争"，提出受电信标准化启发的三层协议 LACP（语义层、事务层、传输层），强调安全内建、事务完整性和语义互操作性对多智能体系统至关重要。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "multi-agent communication"
  - "protocol standardization"
  - "agent interoperability"
  - "telecom-inspired design"
  - "security-by-design"
---

# LLM Agent Communication Protocol (LACP) Requires Urgent Standardization: A Telecom-Inspired Protocol is Necessary

**会议**: NeurIPS 2025  
**arXiv**: [2510.13821](https://arxiv.org/abs/2510.13821)  
**代码**: 待确认  
**领域**: LLM Agent  
**关键词**: multi-agent communication, protocol standardization, agent interoperability, telecom-inspired design, security-by-design

## 一句话总结
这篇 position paper 指出当前 LLM Agent 通信的碎片化生态类似早期网络的"协议战争"，提出受电信标准化启发的三层协议 LACP（语义层、事务层、传输层），强调安全内建、事务完整性和语义互操作性对多智能体系统至关重要。

## 研究背景与动机

当前 LLM Agent 通信面临严重的**碎片化问题**，各种协议各自为政：

| 协议 | 发布时间 | 开发者 | 接口类型 | 安全特性 |
|------|---------|--------|---------|---------|
| Function Calling | 2023.06 | OpenAI | JSON schema | 仅 API key |
| Agent Protocol | 2024.11 | LangChain | REST API | HTTP/JWT |
| MCP | 2024.11 | Anthropic | JSON-RPC/HTTP | OAuth 2.1 |
| ACP | 2024 | IBM/LF | JSON-RPC | 签名 token, RBAC |
| ANP | 2024.03 | 社区 | DID/JSON-LD | W3C DIDs |
| Agora | 2024.10 | 学术界 | 元协议 | 基于哈希的 ID |
| A2A | 2025.04 | Google | HTTP/Protobuf | 能力发现 |

这种碎片化导致三大系统性风险：

**互操作性缺失**：不同框架的 Agent 无法无缝协作，异构多智能体系统开发困难

**安全性作为事后补丁**：安全不是核心组件，系统面临数据篡改、Agent 冒充等攻击风险

**单体设计与缺乏事务完整性**：通信逻辑与 Agent 实现紧耦合，复杂多步操作缺乏原子性保证

作者类比历史：这与 1970-1990 年代的"协议战争"如出一辙，直到 TCP/IP 的广泛采用才打破了网络碎片化。**没有共同的通信基底，分布式 AI 的变革潜力将无法实现。**

## 方法详解

### 整体框架

LACP 是一个**三层协议架构**，灵感来自电信工程的分层抽象原则：

```
┌─────────────────────────────┐
│    Semantic Layer (语义层)     │ ← 传达通信意图
├─────────────────────────────┤
│  Transactional Layer (事务层)  │ ← 确保可靠性和完整性
├─────────────────────────────┤
│    Transport Layer (传输层)    │ ← 高效安全的消息传递
└─────────────────────────────┘
```

三层相互隔离，接口清晰定义，可独立演进。

### 关键设计

**语义层**（Semantic Layer）：

定义最小通用消息类型集，采用"窄腰"设计原则：

| 消息类型 | 必填字段 | 可选字段 | 用途 |
|---------|---------|---------|------|
| PLAN | intent_id, role, natural_language | graph_ops | 表达高层意图 |
| ACT | intent_id, tool_call, params | deadline, cost_cap | 调用外部工具 |
| OBSERVE | intent_id, status, output | metrics | 返回结果/状态 |

所有消息包裹在 JWS（JSON Web Signature）信封中，确保语义安全清晰。

**事务层**（Transactional Layer）：

提供以下机制确保通信可靠性：
- **消息签名**：端到端密码学完整性验证
- **消息排序**：确保多步操作的顺序一致性
- **唯一事务 ID**：保证幂等性，防止重放攻击
- **原子事务支持**：借鉴两阶段提交（2PC）概念

**传输层**（Transport Layer）：

传输无关设计，可运行在多种网络协议上：
- HTTP/2
- QUIC
- WebSockets

这使 LACP 能适配不同网络环境，包括 6G 等下一代网络。

### 设计原则（来自电信标准化的洞见）

1. **共识驱动的开放标准**：借鉴 ITU 和 3GPP 的协作模式
2. **安全即设计**（Security by Construction）：安全不是附加组件，而是每一层的基础组件
3. **分层抽象**：借鉴 OSI 模型，关注点分离
4. **窄腰原则**：最小核心 + 可扩展边缘，如同 IP 协议

## 实验关键数据

### 主实验

**性能与开销分析**（10,000 次顺序请求）：

| 场景 | 基线大小 | LACP 大小 | 大小开销 | 基线延迟 | LACP 延迟 | 延迟开销 |
|------|---------|----------|---------|---------|----------|---------|
| 小消息 (51B) | 51 bytes | 306 bytes | +500% | 0.85 ms | 0.88 ms | +3.5% |
| 中消息 (151B) | 151 bytes | 442 bytes | +191% | 0.86 ms | 0.89 ms | +3.1% |
| 大消息 (1964B) | 1,964 bytes | 2,560 bytes | +30% | 0.89 ms | 0.92 ms | +2.9% |

关键结论：**延迟开销极小**（大负载仅增加 0.03ms），有效负载开销对实际应用大小的消息仅约 30%。

**互操作性验证**：

成功实现 LangChain ReAct Agent 与独立 LACP Tool Server 的透明通信：
- LangChain Agent 构造并发送签名的 LACP ACT 消息
- LACP 服务器验证、执行、返回签名的 OBSERVE 消息
- 全程无需框架特定的定制集成代码

### 消融实验

**安全验证——篡改攻击**：
- 生成有效签名消息（转账金额 100）
- 签名后篡改金额为 10000
- 服务器签名验证立即失败，返回 HTTP 403

**安全验证——重放攻击**：
- 先发送合法消息并成功处理
- 重发同一消息
- 签名验证通过，但事务层检测到重复 transaction_id，返回 HTTP 409

### 关键发现

1. **LACP 的延迟开销可以忽略不计**：对实际负载，绝对延迟增加仅 0.03ms
2. **端到端安全不等于传输层安全**：TLS 仅保护传输，无法防止端点处的篡改和重放，LACP 的应用层安全是必要补充
3. **PLAN/ACT/OBSERVE 语义模型可与现有框架兼容**：成功嵌入 LangChain 工作流

## 亮点与洞察

1. **电信类比精准**：将 Agent 通信碎片化与网络"协议战争"类比，论证有力
2. **三层架构设计合理**：语义-事务-传输的分离清晰，各层职责明确
3. **安全内建而非事后补丁**：对高风险多智能体应用（金融、医疗、自动驾驶）至关重要
4. **"窄腰"设计理念**：PLAN/ACT/OBSERVE 三种核心消息类型足够简洁又不失表达力
5. **有实验原型验证**：不仅是概念性讨论，还提供了可工作的代码实现和定量评估

## 局限与展望

1. **仅为 Position Paper**：缺乏大规模真实多智能体系统的部署验证
2. **性能测试规模有限**：实验在本地单机上进行，未测试跨地域、高并发场景
3. **缺少与已有协议的渐进迁移方案**：现实中协议替换成本高，需要桥接方案
4. **语义层表达力待检验**：PLAN/ACT/OBSERVE 三种类型是否覆盖所有 Agent 交互模式（如协商、竞价）
5. **与 MCP 和 A2A 的竞争关系**：Google A2A 和 Anthropic MCP 已有产业支持，标准之争尚未结束
6. **缺少动态发现和注册机制**：Agent 如何发现并注册到 LACP 网络中未充分讨论
7. **加密开销在高频场景的影响**：每条消息 JWS 签名在 Agent 高频短消息通信中可能成为瓶颈

## 相关工作与启发

- **与 MCP 的对比**：MCP 专注于工具-资源-提示的上下文标准化，LACP 关注更广泛的 Agent 间通信安全与互操作
- **与 A2A 的对比**：A2A 支持对等通信和 Agent Cards，但缺乏事务层的原子性保证
- **与 ANP 的对比**：ANP 采用 W3C DID 去中心化身份，LACP 提供更完整的分层安全
- **OSI 七层模型启发**：LACP 简化为三层，更适合 Agent 通信的实际需求
- **启发**：多智能体系统的标准化窗口期确实存在，早期制定标准比后期修补代价小得多

## 评分
- 新颖性: ⭐⭐⭐ 电信启发的 Agent 协议设计有意义，但核心架构思路不算全新
- 实验充分度: ⭐⭐⭐ 有原型实验但规模有限，作为 position paper 可以接受
- 写作质量: ⭐⭐⭐⭐ 论证结构清晰，历史类比生动，技术描述准确
- 价值: ⭐⭐⭐⭐ 在 Agent 协议竞争的关键时期提出了重要的标准化呼吁

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[CVPR 2026\] Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding](../../CVPR2026/llm_agent/symphony_a_cognitively-inspired_multi-agent_system_for_long-video_understanding.md)
- [\[ACL 2026\] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](../../ACL2026/llm_agent/dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[CVPR 2025\] Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](../../CVPR2025/llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)

</div>

<!-- RELATED:END -->

---
title: >-
  ACL2025 Multi-Agent论文汇总 · 8篇论文解读
description: >-
  8篇ACL2025的 Multi-Agent 方向论文解读，涵盖 Agent、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "推理"
item_list:
  - u: "beyond_frameworks_multi_agent_collaboration/"
    t: "Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems"
  - u: "comet_metaphor-driven_covert_communication_for_multi-agent_language_games/"
    t: "CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games"
  - u: "cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate/"
    t: "CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate"
  - u: "docagent_a_multi-agent_system_for_automated_code_documentation_generation/"
    t: "DocAgent: A Multi-Agent System for Automated Code Documentation Generation"
  - u: "getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re/"
    t: "GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning"
  - u: "multi-agent_collaboration_via_cross-team_orchestration/"
    t: "Multi-Agent Collaboration via Cross-Team Orchestration"
  - u: "preventing_rogue_agents_improves_multi-agent_collaboration/"
    t: "Preventing Rogue Agents Improves Multi-Agent Collaboration"
  - u: "voting_or_consensus_decision-making_in_multi-agent_debate/"
    t: "Voting or Consensus? Decision-Making in Multi-Agent Debate"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**💬 ACL2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×8

**[Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games](comet_metaphor-driven_covert_communication_for_multi-agent_language_games.md)**

:   本文提出 CoMet 框架，通过整合基于假设检验的隐喻推理器和自改进式隐喻生成器，使 LLM 智能体能在多智能体语言博弈中运用隐喻进行隐蔽通信和语义规避，在 Undercover 和 Adversarial Taboo 两个游戏中显著提升了智能体的策略沟通能力（胜率从 0.20 提升至 0.70）。

**[CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)**

:   提出 CortexDebate，一种受人脑皮层工作机制启发的多智能体辩论方法，通过构建稀疏动态辩论图和基于 McKinsey 信任公式的评估模块（MDM），同时解决了现有 MAD 方法中"输入上下文过长"和"过度自信导致不平等辩论"两大核心问题。

**[DocAgent: A Multi-Agent System for Automated Code Documentation Generation](docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)**

:   提出 DocAgent，一个基于拓扑依赖排序的多智能体代码文档生成系统，通过 Reader-Searcher-Writer-Verifier 协作流程增量构建上下文，在完整性、实用性和真实性三个维度上显著优于 FIM 和 Chat 基线。

**[GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)**

:   提出 GETReason，一个层级化多智能体框架，通过将公共事件图像的上下文提取分解为地理空间、时间和事件三个子任务，并由专门化的 Agent 协作完成，实现比现有方法更准确的图像上下文推理。

**[Multi-Agent Collaboration via Cross-Team Orchestration](multi-agent_collaboration_via_cross-team_orchestration.md)**

:   提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互，利用层次化分组 (Hierarchy Partitioning) 和贪心聚合 (Greedy Aggregation) 机制将各团队的多样化解决方案融合为更优结果。

**[Preventing Rogue Agents Improves Multi-Agent Collaboration](preventing_rogue_agents_improves_multi-agent_collaboration.md)**

:   提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。

**[Voting or Consensus? Decision-Making in Multi-Agent Debate](voting_or_consensus_decision-making_in_multi-agent_debate.md)**

:   系统性对比了多智能体辩论中 7 种决策协议（投票 vs 共识），发现共识协议在知识任务上提升 2.8%、投票协议在推理任务上提升 13.2%，并提出 AAD 和 CI 两种增强答案多样性的新方法，分别带来 3.3% 和 7.4% 的性能提升。

---
title: >-
  CVPR2025 Multi-Agent论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2025的 Multi-Agent 方向论文解读，涵盖 Agent、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
item_list:
  - u: "collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration/"
    t: "Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration"
  - u: "comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c/"
    t: "ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems"
  - u: "nader_neural_architecture_design_via_multi-agent_collaboration/"
    t: "NADER: Neural Architecture Design via Multi-Agent Collaboration"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**📷 CVPR2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×2

**[Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)**

:   提出 Cooperative Tree Search (CoTS) 框架，将修改版蒙特卡洛树搜索与 LLM 驱动的奖励函数结合，引导多个具身智能体进行长期战略规划和高效协作，并通过计划评估模块避免频繁计划更新带来的行为混乱，在 CWAH 和 TDW-MAT 环境上显著超越现有方法。

**[ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)**

:   ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。

**[NADER: Neural Architecture Design via Multi-Agent Collaboration](nader_neural_architecture_design_via_multi-agent_collaboration.md)**

:   NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验，仅 10 次试验即突破 NAS-Bench-201 搜索空间的准确率上限，在 CIFAR-100 上达 74.51%（搜索空间最优 73.51%）。

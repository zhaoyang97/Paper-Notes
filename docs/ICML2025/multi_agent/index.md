---
title: >-
  ICML2025 Multi-Agent论文汇总 · 7篇论文解读
description: >-
  7篇ICML2025的 Multi-Agent 方向论文解读，涵盖 Agent、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "推理"
item_list:
  - u: "automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl/"
    t: "AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML"
  - u: "cross-environment_cooperation_enables_zero-shot_multi-agent_coordination/"
    t: "Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination"
  - u: "from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_/"
    t: "From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium"
  - u: "is_your_llm-based_multi-agent_a_reliable_real-world_planner_exploring_fraud_dete/"
    t: "Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning"
  - u: "metaagent_automatically_constructing_multi-agent_systems_based_on_finite_state_m/"
    t: "MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines"
  - u: "researchtown_simulator_of_human_research_community/"
    t: "ResearchTown: Simulator of Human Research Community"
  - u: "theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive/"
    t: "Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**🧪 ICML2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×6 · LLM ×3 · 推理 ×2

**[AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)**

:   本文提出 AutoML-Agent，一个基于多智能体 LLM 协作的全流水线 AutoML 框架，通过检索增强规划策略（Retrieval-Augmented Planning）扩大搜索空间、将任务分解为并行执行的子任务由专业化 Agent 分别完成、并引入多阶段验证机制保障代码生成质量，在 7 类任务 14 个数据集上实现了更高的自动化成功率和模型性能。

**[Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)**

:   > 提出跨环境合作（CEC）范式，通过在程序化生成的大量多样化环境中进行自对弈训练（而非增加伙伴多样性），使智能体学习到通用的合作规范，从而在从未见过的新环境中与从未见过的新伙伴实现零样本协调。

**[From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_.md)**

:   将多 LLM 协调建模为不完全信息博弈，提出 ECON 框架，通过贝叶斯纳什均衡（BNE）实现隐式信念驱动的多 Agent 协调推理，无需显式消息传递即可获得理论收敛保证，在六个推理基准上平均提升 11.2%。

**[Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning](is_your_llm-based_multi-agent_a_reliable_real-world_planner_exploring_fraud_dete.md)**

:   提出 WandaPlan 评估环境，通过在旅行规划场景中注入三种递进式欺诈（单源误导、团队协调刷单、逐级升级），系统性评估 LLM 多智能体规划系统对虚假信息的脆弱性，并设计反欺诈 Agent 来缓解风险。

**[MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](metaagent_automatically_constructing_multi-agent_systems_based_on_finite_state_m.md)**

:   提出 MetaAgent，一个基于有限状态机（FSM）的框架，给定任务描述即可自动设计多智能体系统，无需外部训练数据，支持工具调用和状态回溯，在文本任务、ML 任务和软件开发任务上超越现有自动设计方法并逼近人工设计系统性能。

**[ResearchTown: Simulator of Human Research Community](researchtown_simulator_of_human_research_community.md)**

:   提出 ResearchTown，一个基于 agent-data 图和 TextGNN（文本空间消息传递）的多智能体框架，将人类科研社区建模为异构图，统一模拟论文阅读、论文写作和审稿三大核心研究活动，并通过节点掩码预测任务 (ResearchBench) 进行可扩展、客观的仿真质量评估。

**[Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   提出 Theorem-of-Thought (ToTh) 框架，通过三个分别模拟溯因、演绎和归纳推理的 Agent 独立生成推理轨迹，将其构建为形式化推理图 (FRG)，再用 NLI 校准的贝叶斯置信传播进行一致性评分，选取最优图的终端节点作为最终答案，在符号和数值推理任务上一致超越 CoT、Self-Consistency 和 CoT-Decoding。

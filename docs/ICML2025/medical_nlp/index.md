---
title: >-
  ICML2025 医疗LLM论文汇总 · 4篇论文解读
description: >-
  4篇ICML2025的医疗 LLM 方向论文解读，涵盖个性化生成、Agent、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "医疗 LLM"
  - "论文解读"
  - "论文笔记"
  - "个性化生成"
  - "Agent"
  - "RAG"
item_list:
  - u: "agent_warpp_workflow_adherence_via_runtime_parallel_personalization/"
    t: "Agent WARPP: Workflow Adherence via Runtime Parallel Personalization"
  - u: "autoformulation_of_mathematical_optimization_models_using_llms/"
    t: "Autoformulation of Mathematical Optimization Models Using LLMs"
  - u: "evolve_evaluating_and_optimizing_llms_for_in-context_exploration/"
    t: "EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration"
  - u: "on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge/"
    t: "On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 LLM

**🧪 ICML2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/medical_nlp/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/medical_nlp/index.md) · [💬 ACL2026 (47)](../../ACL2026/medical_nlp/index.md) · [🧪 ICML2026 (4)](../../ICML2026/medical_nlp/index.md) · [🤖 AAAI2026 (12)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/medical_nlp/index.md)

**[Agent WARPP: Workflow Adherence via Runtime Parallel Personalization](agent_warpp_workflow_adherence_via_runtime_parallel_personalization.md)**

:   提出 WARPP，一个无需训练的多智能体框架，在运行时根据用户属性动态剪枝条件分支工作流，并通过并行化的 Personalizer 智能体与模块化域特定智能体协同执行，在提升工具调用精度和参数保真度的同时减少 token 消耗。

**[Autoformulation of Mathematical Optimization Models Using LLMs](autoformulation_of_mathematical_optimization_models_using_llms.md)**

:   本文提出一种利用大语言模型结合蒙特卡洛树搜索（MCTS）自动将自然语言描述的优化问题转化为可求解器求解的数学规划模型的方法，通过符号剪枝和 LLM 价值评估显著提升了搜索效率。

**[EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](evolve_evaluating_and_optimizing_llms_for_in-context_exploration.md)**

:   提出 BanditBench 基准和三种增强策略（推理时算法引导、Few-shot 示范、Oracle 行为微调），系统评估并改善 LLM 在 bandit 环境中的上下文探索能力，使小模型通过算法蒸馏超越大模型。

**[On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)**

:   本文系统揭示了 RAG 检索系统在知识密集型领域（医疗、法律）中面临的**通用投毒攻击**漏洞，提出"正交增强"性质解释攻击成因，并设计基于分布感知距离的检测防御方法，在几乎所有场景中达到近乎完美的检测率。

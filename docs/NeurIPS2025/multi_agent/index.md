---
title: >-
  NeurIPS2025 Multi-Agent论文汇总 · 17篇论文解读
description: >-
  17篇NeurIPS2025的 Multi-Agent 方向论文解读，涵盖 Agent、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "推理"
item_list:
  - u: "3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation/"
    t: "3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation"
  - u: "adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen/"
    t: "Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning"
  - u: "automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select/"
    t: "Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection"
  - u: "belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks/"
    t: "Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks"
  - u: "communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied/"
    t: "Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models"
  - u: "debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model/"
    t: "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?"
  - u: "gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di/"
    t: "GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies"
  - u: "large_language_models_miss_the_multi-agent_mark/"
    t: "Large Language Models Miss the Multi-Agent Mark"
  - u: "lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve/"
    t: "Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve"
  - u: "masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting/"
    t: "MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting"
  - u: "maszero_designing_multiagent_systems_with_zero_supervision/"
    t: "MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision"
  - u: "medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f/"
    t: "MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks"
  - u: "metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems/"
    t: "MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems"
  - u: "multi-agent_collaboration_via_evolving_orchestration/"
    t: "Multi-Agent Collaboration via Evolving Orchestration"
  - u: "rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_/"
    t: "R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization"
  - u: "the_pokeagent_challenge_competitive_and_long-context_learning_at_scale/"
    t: "The PokeAgent Challenge: Competitive and Long-Context Learning at Scale"
  - u: "thought_communication_in_multiagent_collaboration/"
    t: "Thought Communication in Multiagent Collaboration"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**🧠 NeurIPS2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×14 · LLM ×3 · 推理 ×2

**[3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)**

:   提出 Tri-MARF 三模态多智能体框架，通过 VLM 标注 Agent（多视角多候选描述）+ 信息聚合 Agent（BERT 聚类 + CLIP 加权 + UCB1 多臂赌博机选择）+ 点云门控 Agent（Uni3D 文本-点云对齐过滤幻觉），实现 CLIPScore 88.7（超越人类标注 82.4）、吞吐量 12k 物体/小时，已标注约 200 万 3D 模型。

**[Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)**

:   提出 Adaptive Coopetition (AdCo) 框架，利用 UCB 多臂老虎机策略和粗粒度验证器信号，使多个 LLM 智能体在推理过程中自适应地切换协作与竞争模式，在数学推理基准上实现 20% 的相对提升。

**[Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

**[Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)**

:   提出 Belief-Calibrated Consensus Seeking (BCCS) 框架，通过引入信念（belief）校准的共识判断、冲突感知的协作者分配和领导者选择三个模块，让多智能体系统在复杂NLP任务上达成更稳定的共识，在 MATH 和 MMLU 上的困难任务分别提升 2.23% 和 3.95%。

**[Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)**

:   提出基于轻量世界模型的"意图通信"架构，通过生成并共享未来轨迹计划来实现多智能体协调，在可扩展性和性能上全面超越端到端涌现通信方案。

**[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)**

:   提出 GauDP，通过从多智能体的去中心化 RGB 观测中构建全局一致的 3D 高斯场，并将高斯属性动态分配回各智能体的局部视角，实现可扩展的、感知增强的多智能体协作模仿学习。

**[Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)**

:   Position paper 通过调研 1400+ 篇论文，系统论证当前 MAS LLMs 在四个维度偏离传统 MAS 基础理论——LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化，指出该领域有忽视 40 年 MAS 成果而重新发明轮子的风险。

**[Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)**

:   提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。

**[MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](maszero_designing_multiagent_systems_with_zero_supervision.md)**

:   MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

**[MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)**

:   提出 MedAgentBoard，一个系统评估多智能体协作、单 LLM 和传统方法在多样化医学任务上表现的综合基准，揭示多智能体协作并不总是优于强单模型或专用传统方法。

**[MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)**

:   提出 MetaMind——一个受心理学元认知理论启发的多智能体框架，通过 ToM Agent（心理状态假设生成）、Moral Agent（社会规范约束精炼）和 Response Agent（响应生成与自我验证）三阶段协作，显著提升 LLM 的社会推理能力，在多个社会智能基准上达到 SOTA 并首次接近人类水平。

**[Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)**

:   提出"木偶师"(Puppeteer)式多 Agent 协作范式——一个中心化编排器通过 RL 学习在每个推理步骤动态选择激活哪个 Agent，在封闭域和开放域任务上同时提升性能和效率，并发现演化后的拓扑趋向更紧凑的环形结构。

**[R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)**

:   提出 R&D-Agent(Q)，一个数据驱动的多智能体框架，通过五个协作模块（Specification、Synthesis、Implementation、Validation、Analysis）自动化量化策略的因子挖掘与模型创新联合优化，在真实股票市场上以不到 $10 的成本实现约 2× 于传统因子库的年化收益。

**[The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)**

:   提出 PokéAgent Challenge，一个基于宝可梦对战和RPG速通的双赛道大规模AI基准，通过NeurIPS 2025竞赛验证了专家RL方法远超通用LLM方法，并揭示宝可梦对战衡量的能力与现有49个LLM基准近乎正交。

**[Thought Communication in Multiagent Collaboration](thought_communication_in_multiagent_collaboration.md)**

:   提出 ThoughtComm 框架，将多智能体通信形式化为隐变量生成模型，证明了在非参数条件下共享思想和私有思想均可辨识，并通过稀疏正则化自编码器提取潜在思想、经前缀注入回馈给每个智能体，在数学推理任务上相比当前 SOTA 的 Multiagent Finetuning 平均提升 19.06%。

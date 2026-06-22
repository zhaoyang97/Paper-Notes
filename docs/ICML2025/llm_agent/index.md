---
title: >-
  ICML2025 LLMAgent论文汇总 · 11篇论文解读
description: >-
  11篇ICML2025的 LLM Agent 方向论文解读，涵盖 LLM、推理、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "Agent"
item_list:
  - u: "advagent_controllable_blackbox_red-teaming_on_web_agents/"
    t: "AdvAgent: Controllable Blackbox Red-teaming on Web Agents"
  - u: "agacci_affiliated_grading_agents_for_criteria-centric_interface_in_educational_c/"
    t: "AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts"
  - u: "aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction/"
    t: "Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction"
  - u: "evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco/"
    t: "Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics"
  - u: "from_passive_to_active_reasoning_can_large_language_models_ask_the_right_questio/"
    t: "From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?"
  - u: "guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning/"
    t: "GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning"
  - u: "improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati/"
    t: "Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search"
  - u: "kbqa-o1_agentic_knowledge_base_question_answering_with_monte_carlo_tree_search/"
    t: "KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search"
  - u: "open_source_planning_control_system_with_language_agents_for_autonomous_scientif/"
    t: "Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery"
  - u: "towards_llm_agents_for_earth_observation/"
    t: "Towards LLM Agents for Earth Observation"
  - u: "xchemagents_agentic_ai_for_explainable_quantum_chemistry/"
    t: "xChemAgents: Agentic AI for Explainable Quantum Chemistry"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧪 ICML2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (42)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/llm_agent/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_agent/index.md) · [🧪 ICML2026 (59)](../../ICML2026/llm_agent/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md)

🔥 **高频主题：** LLM ×4 · 推理 ×2 · Agent ×2

**[AdvAgent: Controllable Blackbox Red-teaming on Web Agents](advagent_controllable_blackbox_red-teaming_on_web_agents.md)**

:   提出 AdvAgent，一个基于强化学习（DPO）的黑盒红队测试框架，训练一个对抗 prompter 模型自动生成不可见的 HTML 对抗 prompt，注入网页后可误导 GPT-4V 驱动的 Web Agent 执行攻击者指定的目标动作（如将买微软股票改为买英伟达），在 440 个任务上达到 97.5% 攻击成功率，且对现有防御手段仍保持 88.8% 以上的有效性。

**[AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts](agacci_affiliated_grading_agents_for_criteria-centric_interface_in_educational_c.md)**

:   AGACCI 提出一个由 9 个专门化 Agent 组成的多 Agent 评估框架，将教育编程作业的评估任务分解为 rubric 解析、代码执行验证、可视化评估、解释性推理评估等角色，通过协作实现比单模型 baseline 更准确、一致且可解释的 rubric 对齐反馈。

**[Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)**

:   提出 Aguvis，首个完全基于纯视觉的跨平台自主 GUI Agent 框架，通过统一视觉观察空间、标准化动作空间和内心独白（inner monologue）机制，在离线和在线基准上取得 SOTA，无需依赖闭源模型。

**[Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)**

:   本文构建了宇宙学领域的 RAG 评测基准 CosmoPaperQA（105 个专家 QA 对），系统评估了 9 种 RAG agent 配置（涵盖商业 API、混合架构、学术工具），发现 OpenAI RAG 方案以 91.4% 准确率领先，并校准了可替代人工评审的 LLM-as-a-Judge 系统。

**[From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](from_passive_to_active_reasoning_can_large_language_models_ask_the_right_questio.md)**

:   本文提出 AR-Bench，一个专门评估 LLM 主动推理能力的基准，包含侦探案件、情境谜题和猜数字三类任务，实验发现 GPT-4o 等最先进模型在需要主动提问获取缺失信息的场景中表现远逊于人类，揭示了被动推理与主动推理之间的巨大鸿沟。

**[GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)**

:   GuardAgent 是首个"用 Agent 守护 Agent"的框架，通过将安全规则动态转化为可执行的护栏代码来检查目标 Agent 的动作是否违规，在医疗访问控制和 Web 安全控制两个新基准上分别达到 98%+ 和 83%+ 的护栏准确率。

**[Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)**

:   提出 LWM-Planner，从交互轨迹中提取"原子事实"增强 LLM 世界模型模拟，结合递归前瞻搜索实现纯 in-context 的 Agent 规划改进，在 ALFWorld 等任务上显著优于 ReAct 和 Reflexion。

**[KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](kbqa-o1_agentic_knowledge_base_question_answering_with_monte_carlo_tree_search.md)**

:   提出 KBQA-o1，将 ReAct Agent 与蒙特卡洛树搜索（MCTS）结合，通过策略模型和奖励模型驱动的启发式搜索实现知识库问答，在低资源设置下以 Llama-3.1-8B 将 GrailQA F1 从 48.5%（GPT-3.5-turbo SOTA）提升至 78.5%。

**[Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)**

:   本文提出 cmbagent，一个由约 30 个 LLM Agent 组成的多智能体系统，采用 Planning & Control 策略编排无人干预的科研工作流，各 Agent 分别负责论文检索、代码编写、结果解读、输出评审等专业任务，并可在本地执行代码；该系统成功完成了博士级别的宇宙学任务（用超新星数据测量宇宙学参数），在两个基准测试集上优于当前最先进的 LLM。

**[Towards LLM Agents for Earth Observation](towards_llm_agents_for_earth_observation.md)**

:   本文提出 UnivEARTH——一个包含 140 个 yes/no 问题的地球观测基准，涵盖 13 个主题和 17 种卫星传感器，评估发现最佳 LLM Agent（使用 Google Earth Engine 生成代码）的准确率仅 33%，主要受限于 58% 的代码无法运行。

**[xChemAgents: Agentic AI for Explainable Quantum Chemistry](xchemagents_agentic_ai_for_explainable_quantum_chemistry.md)**

:   xChemAgents 提出了一个 Selector-Validator 双 Agent 协作框架，将物理感知的推理注入多模态分子性质预测中：Selector Agent 自适应选择稀疏加权描述符子集并给出自然语言解释，Validator Agent 通过量纲一致性和标度律检验迭代验证，在 QM9 基准上实现最高 22% 的 MAE 降低。

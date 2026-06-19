---
title: >-
  ICLR2026 Multi-Agent论文汇总 · 24篇论文解读
description: >-
  24篇ICLR2026的 Multi-Agent 方向论文解读，涵盖 Agent、LLM、强化学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "强化学习"
  - "推理"
item_list:
  - u: "adaptive_collaboration_with_humans_metacognitive_policy_optimization_for_multi-a/"
    t: "Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning"
  - u: "agentpo_enhancing_multi-agent_collaboration_via_reinforcement_learning/"
    t: "AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning"
  - u: "agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_/"
    t: "AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems"
  - u: "aligned_agents_biased_swarm_measuring_bias_amplification_in_multi-agent_systems/"
    t: "Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems"
  - u: "atlas_constraints-aware_multi-agent_collaboration_for_real-world_travel_planning/"
    t: "ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning"
  - u: "auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut/"
    t: "Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution"
  - u: "automated_stateful_specialization_for_adaptive_agent_systems/"
    t: "Automated Stateful Specialization for Adaptive Agent Systems"
  - u: "benefits_and_limitations_of_communication_in_multi-agent_reasoning/"
    t: "Benefits and Limitations of Communication in Multi-Agent Reasoning"
  - u: "breaking_and_fixing_defenses_against_control_flow_hijacking_in_multi-agent_syste/"
    t: "Breaking and Fixing Defenses Against Control Flow Hijacking in Multi-Agent Systems"
  - u: "bridge_bi-level_reinforcement_learning_for_dynamic_group_structure_in_coalition_/"
    t: "BRIDGE: Bi-level Reinforcement Learning for Dynamic Group Structure in Coalition Formation Games"
  - u: "cache-to-cache_direct_semantic_communication_between_large_language_models/"
    t: "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
  - u: "completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev/"
    t: "Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment"
  - u: "hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat/"
    t: "HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre"
  - u: "kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing/"
    t: "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing"
  - u: "lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo/"
    t: "LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions"
  - u: "mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim/"
    t: "MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design"
  - u: "mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni/"
    t: "MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning"
  - u: "multi-agent_coordination_via_flow_matching/"
    t: "Multi-agent Coordination via Flow Matching"
  - u: "multi-agent_design_optimizing_agents_with_better_prompts_and_topologies/"
    t: "Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies"
  - u: "stochastic_self-organization_in_multi-agent_systems/"
    t: "Stochastic Self-Organization in Multi-Agent Systems"
  - u: "stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems/"
    t: "Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems"
  - u: "uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed/"
    t: "UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking"
  - u: "when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m/"
    t: "When Agents \"Misremember\" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems"
  - u: "which_llm_multi-agent_protocol_to_choose/"
    t: "Which LLM Multi-Agent Protocol to Choose?"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**🔬 ICLR2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×20 · LLM ×5 · 强化学习 ×2 · 推理 ×2

**[Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning](adaptive_collaboration_with_humans_metacognitive_policy_optimization_for_multi-a.md)**

:   提出 HILA 框架，让多智能体 LLM 学会一套"元认知策略"——自己判断什么时候能独立解题、什么时候该把问题交给人类专家；再用 Dual-Loop Policy Optimization 把"何时求助"（内环强化学习）和"如何从求助中长本事"（外环持续学习）分开优化，在数学推理等基准上稳定超过现有自主多智能体系统。

**[AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning](agentpo_enhancing_multi-agent_collaboration_via_reinforcement_learning.md)**

:   AgentPO 不去搜索多智能体拓扑结构，而是在一个固定拓扑里冻结强大的 Actor、只用强化学习（GRPO）训练一个轻量级 Collaborator 学会"怎么辅助队友"，仅用 500 条训练样本和 EvoAgent 7.8% 的推理开销，就在多个数学推理基准上稳定超越 Role Assignment、EvoAgent 等强基线。

**[AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)**

:   提出AgentTrace框架，从多智能体系统的执行日志中构建因果图，通过反向追踪+轻量级特征排序（五组特征的加权线性组合）定位根因节点，在550个合成故障场景上Hit@1达94.9%，延迟0.12秒，比LLM分析快69倍。

**[Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems](aligned_agents_biased_swarm_measuring_bias_amplification_in_multi-agent_systems.md)**

:   这篇论文用一个强制三选一的开放式偏见基准 Discrim-Eval-Open，把多智能体系统（MAS）建模成有向无环图、用基尼系数追踪偏见在层间的"放大率"，系统性地证明了一个反直觉结论：人们以为多智能体协作会"稀释"偏见，实际上各种角色分工、复杂拓扑、加深迭代反而把单体模型里微小的随机偏好放大成系统性的人群歧视，甚至一句客观中性的外部信息就能触发剧烈极化。

**[ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning](atlas_constraints-aware_multi-agent_collaboration_for_real-world_travel_planning.md)**

:   ATLAS 把"带搜索的真实旅行规划"形式化成一个动态约束满足问题（CSP），用 5 个分工明确的 LLM 智能体（搜索、约束管理、规划、检查、搜索顾问）协同补全约束、迭代纠错、并在卡死时反过来指导搜索，把 TravelPlanner 最终通过率从 23.3% 提到 44.4%，并首次在带实时网络搜索的真实多轮场景里做到 84% 通过率。

**[Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)**

:   提出 SCCAL 框架，通过耦合语义流（semantic flow）和交互图的 Ollivier–Ricci 曲率（ORC）来建模多智能体系统中语义-几何的协同演化，利用两者的一致性残差作为级联风险的早期预警信号，在语义违规显现前数轮即可检测异常。

**[Automated Stateful Specialization for Adaptive Agent Systems](automated_stateful_specialization_for_adaptive_agent_systems.md)**

:   ASPEC 提出一套全自动的"有状态专家智能体团队"生命周期框架：先用进化搜索离线发现一批领域专家算子、再让它们在经验中反思培养出持久记忆，最后在线用一个轻量的 "retain-then-escalate"（先保留、再升级）元控制器决定每个查询是沿用现有团队还是重新搜索架构，从而在专家级科学基准 GPQA 上把 Gemini 2.0 Flash 从 56.3% 拉到 62.8%，同时训练+推理成本远低于同类自动框架。

**[Benefits and Limitations of Communication in Multi-Agent Reasoning](benefits_and_limitations_of_communication_in_multi-agent_reasoning.md)**

:   本文给"把长上下文切块、多个 LLM agent 分头处理再汇总"这类多智能体推理系统建了一套基于 Transformer 表达力的理论框架，在关联召回、状态追踪、k-hop 推理三类任务上证明了**需要多少个 agent、多少通信、能换来多少并行加速**的紧界，划出三种 depth–通信权衡区间，并用 Llama 在合成基准上验证理论预测的拐点确实出现。

**[Breaking and Fixing Defenses Against Control Flow Hijacking in Multi-Agent Systems](breaking_and_fixing_defenses_against_control_flow_hijacking_in_multi-agent_syste.md)**

:   这篇论文先证明了现有"对齐检查"类防御（如 LlamaFirewall）能被精心改写的控制流劫持攻击绕过，再提出 CONTROLVALVE——一个借鉴程序控制流完整性思想的协同层防御：在任务规划期生成"允许的智能体调用图 + 每条边的上下文规则"，运行期对每次智能体跳转只做"是否在图里、是否满足边规则"的窄判定，从而在不掉基准任务性能的前提下把所有评测攻击的成功率压到 0%。

**[BRIDGE: Bi-level Reinforcement Learning for Dynamic Group Structure in Coalition Formation Games](bridge_bi-level_reinforcement_learning_for_dynamic_group_structure_in_coalition_.md)**

:   把"把一群智能体最优地划分成若干联盟"（NP 完全的联盟结构生成问题）建模成一个紧凑、可被强化学习吃下的 MDP，再用双层 RL（上层学合并联盟、下层学每个智能体的最优策略）联合求解，使得在 3 个智能体上训练的模型能泛化到 100 个智能体，并在推理速度和混合动机马尔可夫博弈上超过传统启发式方法。

**[Cache-to-Cache: Direct Semantic Communication Between Large Language Models](cache-to-cache_direct_semantic_communication_between_large_language_models.md)**

:   让多个大语言模型不再靠"互相说话"协作，而是用一个轻量神经网络把 Sharer 模型的 KV-Cache 直接投影、融合进 Receiver 模型的 KV-Cache，绕开 token-by-token 的文本生成，既保住了文本会丢失的深层语义，又把延迟平均压低 2.5×，准确率比纯文本协作高约 3–5%。

**[Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)**

:   提出DREAM——基于对立立场初始化的多Agent多轮辩论框架用于IR相关性标注：一致时自动标注、分歧时交给人工(含辩论历史辅助)。达到95.2% balanced accuracy且仅3.5%需人工介入，据此构建BRIDGE基准数据集，发现29,824个原有基准缺失的相关标注(原标注的428%)，修正了检索系统排名偏差和RAG中检索-生成性能不匹配问题。

**[HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)**

:   提出 HAMLET 多智能体框架，将 AI 戏剧创作和在线表演解耦为离线规划和在线表演两阶段，通过叙事蓝图、感知与决策（PAD）模块和层级控制系统，实现了具有主动性、物理环境交互能力和即兴表演自由的 AI 戏剧体验。

**[KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)**

:   提出 KVComm 框架通过选择性共享 KV pairs 实现 LLM 间高效通信，发现 hidden states 存在"信息集中偏差"使其不适合跨模型传递，设计基于注意力重要性 + 高斯先验的层选择策略，仅传输 30% 层即可超越大多数 baseline。

**[LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)**

:   提出首个面向长时域交互的 LLM 欺骗行为仿真框架 LH-Deception，采用执行者-监督者-审计者三角色多智能体架构，结合社会科学理论驱动的概率事件系统，在 11 个前沿模型上系统量化了欺骗频率、严重性、类型分布及其对信任关系的侵蚀效应，揭示了静态单轮评估完全无法捕捉的"欺骗链"涌现现象。

**[MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)**

:   提出 MAC-AMP，首个闭环多智能体协作系统，将抗菌肽（AMP）设计重构为协调多智能体优化问题，通过 AI 模拟同行评审和自适应奖励设计实现多目标优化。

**[MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)**

:   提出 MMedAgent-RL，通过 RL 优化模拟临床会诊流程（分诊→专科→主治）的多智能体系统，核心创新是课程学习引导的熵感知 RL（C-MARL），让主治医师智能体在面对正确/冲突/错误的专科意见时分别采取不同的探索-利用策略，在域内外共 5 个医学 VQA 基准上实现 SOTA。

**[Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)**

:   提出 MAC-Flow，先用 Flow Matching 学习中心化联合行为分布，再通过 IGM（Individual-Global-Max）分解将其蒸馏为去中心化的单步策略，结合 Q 值最大化进行行为正则化训练，在 4 个基准 12 个环境 34 个数据集上实现了约 14.5 倍于扩散方法的推理加速，同时保持了与扩散策略可比的协调性能。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   提出Multi-Agent System Search（MASS）框架，通过交错优化提示词和拓扑结构的三阶段策略（局部提示优化→拓扑搜索→全局提示优化），自动发现高性能的多智能体系统设计。

**[Stochastic Self-Organization in Multi-Agent Systems](stochastic_self-organization_in_multi-agent_systems.md)**

:   提出 SelfOrg 框架，基于 Agent 响应的语义相似度和 Shapley 值贡献估计，动态构建有向无环通讯图（DAG），实现多 Agent 系统的自组织协作。在弱模型场景下优势尤为显著。

**[Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)**

:   提出 SupervisorAgent，一个轻量级的实时自适应监督框架，通过无 LLM 的自适应过滤器在关键交互节点主动干预（纠错、指导、观察净化），在 GAIA 基准上将 Smolagent 的 token 消耗降低 29.68% 而不损失成功率。

**[UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)**

:   识别并形式化"未索引信息检索"(UIS) 问题——搜索引擎无法直接检索的动态网页/嵌入文件/交互式内容，提出首个 UIS 基准 UIS-QA（110 题）和多 Agent 框架 UIS-Digger，以 ~30B 参数模型经 SFT+RFT 训练后达到 27.27% 准确率，超越集成 O3/GPT-4.1 的系统。

**[When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)**

:   本文首次系统研究了 LLM 多智能体系统中的曼德拉效应（集体虚假记忆），提出 ManBench 基准（4838 个问题、5 种交互协议），发现所有 13 个被评估的 LLM 均易受此效应影响，并提出 prompt 级和模型级缓解策略，平均减少 74.40% 的虚假记忆。

**[Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)**

:   本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异，并通过可学习的协议路由器实现场景自适应的协议选择，最高降低18.1%的故障恢复时间。

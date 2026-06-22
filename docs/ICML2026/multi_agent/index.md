---
title: >-
  ICML2026 Multi-Agent论文汇总 · 24篇论文解读
description: >-
  24篇ICML2026的 Multi-Agent 方向论文解读，涵盖 Agent、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "推理"
item_list:
  - u: "beyond_majority_voting_llm_aggregation_by_leveraging_higher-order_information/"
    t: "Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information"
  - u: "coot_learning_to_coordinate_in-context_with_coordination_transformers/"
    t: "CoOT: Learning to Coordinate In-Context with Coordination Transformers"
  - u: "does_persona_make_llms_k-pop_fans_a_pilot_study_of_llm-based_online_concert_audi/"
    t: "Does Persona Make LLMs K-pop Fans? A Pilot Study of LLM-Based Online Concert Audience Agents"
  - u: "e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory/"
    t: "E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory"
  - u: "edumirror_modeling_educational_social_dynamics_with_value-driven_multi-agent_sim/"
    t: "EduMirror: Modeling Educational Social Dynamics with Value-driven Multi-agent Simulation"
  - u: "engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi/"
    t: "EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions"
  - u: "learning_multi-agent_coordination_via_sheaf-admm/"
    t: "Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM"
  - u: "mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic/"
    t: "MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks"
  - u: "maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems/"
    t: "MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems"
  - u: "maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura/"
    t: "MASPOB: 用 GNN 代理 + LinUCB + 坐标上升做多智能体提示优化"
  - u: "more_capable_less_cooperative_when_llms_fail_at_zero-cost_collaboration/"
    t: "More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration"
  - u: "multi-agent_systems_are_mixtures_of_experts_who_becomes_an_influencer/"
    t: "Multi-Agent Systems are Mixtures of Experts: Who Becomes an Influencer?"
  - u: "omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration/"
    t: "OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration"
  - u: "protocolbench_which_llm_multiagent_protocol_to_choose/"
    t: "ProtocolBench: Which LLM MultiAgent Protocol to Choose?"
  - u: "radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat/"
    t: "RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation"
  - u: "representational_similarity_and_model_behavior_in_multi-agent_interaction/"
    t: "Representational Similarity and Model Behavior in Multi-Agent Interaction"
  - u: "searching_for_synergy_in_shared_workspace_human-ai_collaboration/"
    t: "Searching for Synergy in Shared Workspace Human-AI Collaboration"
  - u: "securing_multi-agent_systems_against_corruptions_via_node_contribution_backpropa/"
    t: "Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation"
  - u: "smarter_saboteurs_better_fixers_scaling_security_in_linear_multi-agent_workflows/"
    t: "Smarter Saboteurs, Better Fixers: Scaling & Security in Linear Multi-Agent Workflows"
  - u: "systematic_failures_in_collective_reasoning_under_distributed_information_in_mul/"
    t: "Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs"
  - u: "toward_culturally_aligned_llms_through_ontology-guided_multi-agent_reasoning/"
    t: "Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning"
  - u: "voting_protocols_as_coordination_mechanisms_for_role-constrained_multi-agent_tut/"
    t: "Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems"
  - u: "when_cloud_agents_meet_device_agents_lessons_from_hybrid_multi-agent_systems/"
    t: "When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems"
  - u: "why_specialist_models_still_matter_a_heterogeneous_multi-agent_paradigm_for_medi/"
    t: "Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**🧪 ICML2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/multi_agent/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×16 · LLM ×7 · 推理 ×3

**[Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information](beyond_majority_voting_llm_aggregation_by_leveraging_higher-order_information.md)**

:   本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票，并在 UltraFeedback、MMLU 和医疗健康数据集上验证了一致的提升。

**[CoOT: Learning to Coordinate In-Context with Coordination Transformers](coot_learning_to_coordinate_in-context_with_coordination_transformers.md)**

:   把"如何与陌生伙伴协作"从 task-generalization 改写成 partner-generalization 的 in-context 学习问题：训练一个 Decision Transformer 在跨 episode 的交互轨迹上预测最佳响应动作，让模型不更新参数就能在几局之内适应任何未见过的伙伴。

**[Does Persona Make LLMs K-pop Fans? A Pilot Study of LLM-Based Online Concert Audience Agents](does_persona_make_llms_k-pop_fans_a_pilot_study_of_llm-based_online_concert_audi.md)**

:   作者搭了一套十个 LLM 智能体实时刷弹幕的"虚拟观众"系统，给录播的 K-pop 演出配上真人感的粉丝聊天，并通过一次 N=11 的被试内试点实验发现：给每个智能体加上独立人格能显著提升模型输出层面的多样性和"自然度"，但**并不能**转化为更强的社交连接感、参与度或情感共鸣——因为 K-pop 弹幕本质上是"集体独白"而非人际对话。

**[E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)**

:   E-mem 把"预处理压缩成嵌入/图"的传统记忆范式改成"保留原始上下文 + 小模型助手就地推理"的情景重构范式：master agent 只做全局规划，多个 SLM assistant 各自守着一段未压缩的原文，按多路由检索激活后再做局部推理回传证据，在 LoCoMo 上 F1 反超 SOTA 7.75 个点的同时把 token 消耗砍掉 70%。

**[EduMirror: Modeling Educational Social Dynamics with Value-driven Multi-agent Simulation](edumirror_modeling_educational_social_dynamics_with_value-driven_multi-agent_sim.md)**

:   EduMirror 把"校园欺凌""同伴合作"这类教育社会现象搬进一个由 LLM 驱动的多智能体沙盒：用基于马斯洛需求层次和社会价值取向（SVO）的"价值驱动智能体"扮演学生/老师，再配一套"双轨测量"协议把可观测行为和潜在心理状态都量化出来，从而在伦理安全的数字环境里做"如果换一种干预会怎样"的反事实实验。

**[EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)**

:   EngiAgent 把工程问题求解拆成 Analyzer/Modeler/Verifier/Solver/Evaluator 五个专家 Agent，再用一个**全连接协调器**动态路由反馈（而不是走固定流水线），让 GPT-4o 上工程任务的可行解率从 5.66%（zero-shot）/7.55%（MM-Agent）一跃到 64.15%，平均比此前 SOTA 提升约 7 倍。

**[Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM](learning_multi-agent_coordination_via_sheaf-admm.md)**

:   Sheaf-ADMM 把多智能体协调问题做成端到端可微的 ADMM 展开——每个 agent 只看局部 patch，独立解 ADMM 子问题（$\bm x$-update）、通过 cellular sheaf 定义的"边空间投影"协商一致（$\bm z$-update）、用对偶变量 $\bm u$ 累积分歧；在 maze pathfinding / MNIST / Sudoku 上 agents 协同得出正确全局解，且推理路径有可分析的 primal/consensus/dual 三态——比 MPNN 更可干预。

**[MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)**

:   把"自动多智能体系统设计"重新表述为一次性输出整张 MAS 的函数调用 RL 问题，并配套 MASBench 从 Depth/Horizon/Breadth/Parallel/Robustness 五个轴说清楚"什么时候多智能体真的比单智能体强"。

**[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)**

:   MASPO 通过多粒度联合评价（局部有效性 + 前瞻潜力 + 全局对齐）+ 错位案例驱动的进化束搜索，在不依赖标注的前提下端到端地为整条多智能体链路联合优化角色提示词，6 个任务上平均提升约 2.9 分。

**[MASPOB: 用 GNN 代理 + LinUCB + 坐标上升做多智能体提示优化](maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)**

:   MASPOB 把多智能体系统的 prompt 优化看作预算紧缩的黑盒优化，用 GAT 代理模型捕获 workflow topology 下的 prompt 耦合、用 LinUCB 在嵌入空间算 epistemic uncertainty、用坐标上升把联合搜索拆成序贯单体问题，复杂度从 $\mathcal{O}(\prod |\mathcal{P}_i|)$ 降到 $\mathcal{O}(\sum |\mathcal{P}_i|)$；在 6 个基准（QA/Code/Math）上平均 80.58 超越 MIPRO 78.87、AFlow 78.52、IO 68.56。

**[More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration](more_capable_less_cooperative_when_llms_fail_at_zero-cost_collaboration.md)**

:   作者搭了一个"帮人零成本、合作是显然最优解"的回合制多智能体环境，发现 8 个主流 LLM 中能力强弱完全预测不了合作度（o3 只达到最优的 17%，更弱的 o3-mini 反而 50%），并用"自动化一侧通信"的因果分解把失败拆成"不愿合作"和"不会执行"两类，再用显式协议、微小分享激励、隐藏对比信息三种低成本干预分别对症下药。

**[Multi-Agent Systems are Mixtures of Experts: Who Becomes an Influencer?](multi-agent_systems_are_mixtures_of_experts_who_becomes_an_influencer.md)**

:   本文用社会学里的 **Friedkin-Johnsen（FJ）意见动力学**给"多个 LLM 智能体辩论"建模，证明 FJ 参数是**随输入变化**的——这等价于让多智能体系统（MAS）实现了一个**专家混合（MoE）+ 隐式路由**；进而从理论上刻画 MAS 何时能赢过单智能体和静态集成，并通过实验揭示"谁会成为意见领袖"主要由**置信度（尤其是相对置信度）**决定。

**[OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)**

:   本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在每个维度上做监督式优化，再迭代联合优化多个维度，在 HumanEval / MMLU / MATH 上稳定打败 DyLAN、ADAS、AFlow 等基线。

**[ProtocolBench: Which LLM MultiAgent Protocol to Choose?](protocolbench_which_llm_multiagent_protocol_to_choose.md)**

:   ProtocolBench 首次系统对比四大 LLM 多智能体通信协议（A2A、ACP、ANP、Agora）在任务成功、端到端延迟、消息字节开销、失败鲁棒性四轴上的表现——发现协议选择对系统行为有 36.5% 完成时间差、3.48s 延迟差；进一步提出 ProtocolRouter 按场景/模块动态选协议，将 Fail-Storm 恢复时间降 18.1%。

**[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)**

:   RADAR 把多 LLM-Agent 系统的通信拓扑设计建模为一个"冗余感知"的离散图扩散过程，用 effective size 作为指导信号一步步增量生成 query-自适应的协作图，在 6 个基准上同时拿到更高准确率、更低 token 消耗和更强鲁棒性。

**[Representational Similarity and Model Behavior in Multi-Agent Interaction](representational_similarity_and_model_behavior_in_multi-agent_interaction.md)**

:   这篇论文把 276 个 LLM 配对放进 8 个游戏里互动，发现一个稳健规律：内部表示越相似（用 CKA 量化）的两个模型越能合作，但联手产出的内容越缺乏新颖性——合作与创造力之间存在一条由表示相似性驱动的权衡线。

**[Searching for Synergy in Shared Workspace Human-AI Collaboration](searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)**

:   这篇论文在共享工作区（shared-workspace）的人–AI 协作环境里发现一个反直觉现象——给 AI 智能体**加上有相关专长的（模拟）人类协作者反而会拉低成绩**，根因是团队缺乏协调结构导致"过程损失"（process loss）；作者借用群体心理学的两个机制（共享组记忆 + 模拟 HITL 审批门控）做脚手架，在三人队上把平均成绩从 0.63 拉回到 0.76。

**[Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation](securing_multi-agent_systems_against_corruptions_via_node_contribution_backpropa.md)**

:   BPD 把 LLM 多智能体系统的多轮交互重构成 "带符号有向无环图"，把每条消息打成 $\{-1, 0, 1\}$ 的同意 / 漠视 / 反对分数，再用 PageRank 式的一次反向拓扑传播算出每个 agent 对最终答案的贡献分，分数离群者直接判定为恶意 agent 并切掉其出边——免训练、单次查询即用、对动态拓扑天然鲁棒。

**[Smarter Saboteurs, Better Fixers: Scaling & Security in Linear Multi-Agent Workflows](smarter_saboteurs_better_fixers_scaling_security_in_linear_multi-agent_workflows.md)**

:   本文用一条「产品经理→架构师→项目经理→工程师」的线性 MetaGPT 流水线，往工程师身上注入一个偷偷埋 bug 的恶意 agent，发现**模型越大、被破坏得越狠**（27B 时 Pass@1 掉 53.7pp），但只要在链尾接一个轻量的 QA+Fixer 纠错环节，掉点就被压回 0.6pp——说明此前归咎于「线性拓扑天然脆弱」的结论，其实是因为缺少终端纠错。

**[Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)**

:   本文将社会心理学的 Hidden Profile 范式搬到多智能体 LLM 评测里，构建 65 任务的 HiddenBench，在 15 个前沿 LLM 上系统揭示：单 agent 在 Full Profile 下能 80.7% 答对的同类任务，多 agent 在分布式信息下仅 30.1%，根本失败模式是**不会主动 elicit 别人没说出来的信息**，而轻量结构化沟通协议能跨家族大幅缓解。

**[Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning](toward_culturally_aligned_llms_through_ontology-guided_multi-agent_reasoning.md)**

:   OG-MAR 把世界价值观调查（WVS）的原始问卷整理成「带结构关系的文化本体 + 个体价值画像」，推理时检索出与目标人群相关的本体三元组和人口学相似的真实受访者，实例化多个「价值人格智能体」各自作答，再由一个判决智能体按「证据优先、本体一致」的协议综合出最终答案，从而在六个地区社会调查基准上提升文化对齐度并给出可解释的推理轨迹。

**[Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems](voting_protocols_as_coordination_mechanisms_for_role-constrained_multi-agent_tut.md)**

:   把四个职责互不重叠的"辅导智能体"（脚手架/纠错/激励/元认知）放进同一个辅导回合里，让它们各自提案、互评、修订，再用四种不同的投票协议（简单多数 / 排序 / 累积 / 赞同）把分歧收敛成一个最终答复；论文不是想证明"投票让辅导更好"，而是把辅导当成一个**目标部分对齐但局部冲突**的协调实验场，系统地刻画不同投票规则会诱导出怎样不同的协调行为。

**[When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](when_cloud_agents_meet_device_agents_lessons_from_hybrid_multi-agent_systems.md)**

:   这篇论文系统研究云端 GPT-4o 监督者与端侧 Qwen3 执行者组成的混合多智能体系统，发现 PEVR 和 EVA 在 UI assistance 与 deep search 上各有优势，更多云端介入不一定更好，而上下文重置与摘要能显著改善端侧长任务的成本和 KV-cache 压力。

**[Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence](why_specialist_models_still_matter_a_heterogeneous_multi-agent_paradigm_for_medi.md)**

:   HetMedAgent 将通用 LLM、模态专科模型和临床医生组织成异构多智能体系统，通过冲突感知证据融合与不确定性路由，在心血管和胸片临床决策任务上证明专科模型与人类监督仍是医疗 AI 中不可替代的组成部分。

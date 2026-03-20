<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧠 NeurIPS2025** · 共 **32** 篇

**[A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)**

:   提出 A-Mem，一种受 Zettelkasten 启发的 LLM Agent 智能记忆系统，每条记忆自动生成结构化笔记（关键词/标签/上下文描述），动态建立记忆间链接，并在新记忆加入时触发旧记忆的演化更新，在 LoCoMo 长对话 QA 上显著超越 MemGPT 等基线。

**[A Differentiable Model Of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)**

:   用JAX自动微分实现可微的供应链冲击Agent-Based Model，支持GPU加速的参数校准，使大规模供应链网络冲击传播模拟变得可行。

**[A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[Adaptive Cooperative Transmission Design For Ultra-Reliable Low-Latency Communic](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   用双Agent DRL优化两跳中继系统中的传输参数，在严格延迟约束下实现超可靠低延迟通信(URLLC)。

**[AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)**

:   提出 AgentAuditor，一个通用的无训练记忆增强推理框架，使 LLM 评估者能模拟人类专家评估 agent 的安全与安全性——通过自适应提取结构化语义特征并生成CoT推理轨迹构建经验记忆，多阶段上下文感知 RAG 检索相关经验指导新案例评估，在自建的 ASSEBench（2293条记录×15类風险×29场景）上达到人类水平准确率。

**[AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)**

:   提出 AgentDAM，首个在真实 Web 环境中端到端评估 AI Agent 数据最小化能力的基准，包含 246 个跨 Reddit/GitLab/Shopping 的任务，发现 GPT-4o 等主流模型在无缓解措施时隐私泄露率高达 36-46%，而 CoT 隐私提示可将泄露率降至 6-8%。

**[Agentic NL2SQL to Reduce Computational Costs](agentic_nl2sql_to_reduce_computational_costs.md)**

:   提出 Datalake Agent，一个基于交互循环的 agentic NL2SQL 系统，通过分层的信息获取策略（GetDBDescription -> GetTables -> GetColumns -> DBQueryFinalSQL）让 LLM 按需请求数据库 schema 信息而非一次性接收全部，在 319 张表的场景下将 token 使用量减少 87%、成本降低 8 倍，同时在复杂查询上保持更好的性能。

**[Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)**

:   提出 Agentic Plan Caching (APC)——从 agent 执行日志中提取结构化计划模板，通过关键词匹配缓存命中后用小模型适配复用，平均降低 50.31% 成本和 27.28% 延迟，同时保持 96.61% 的最优准确率。

**[AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)**

:   提出 AgentMisalignment 基准套件，包含 9 个现实场景评估任务，测量 LLM Agent 在非恶意指令下 **自发偏离** 部署者意图的倾向（而非能力），发现更强的模型倾向于更高的错误对齐，且人格提示（persona prompt）有时比模型选择本身对错误对齐行为的影响更大。

**[AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)**

:   提出 AgentTTS，一个用 LLM agent 自动搜索多阶段复杂任务中**测试时计算最优缩放策略**（模型选择+预算分配）的框架，通过迭代反馈驱动的交互显著提升搜索效率和性能。

**[Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择建模为在线背包问题，通过实际测试而非静态语义检索评估组件价值，用 ZCL 算法在预算约束下动态选取最优组合，单 Agent 成功率提升 31.6%，多 Agent 从 37%→87%。

**[Benchmarking Agentic Systems In Automated Scientific Information Extraction With](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)**

:   提出ChemX基准，用10个精心策划的化学数据集全面评估agentic系统从科学文献中自动提取化学信息的能力。

**[BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](btlui_blinkthinklink_reasoning_model_for_gui_agent.md)**

:   提出"Blink-Think-Link"（BTL）脑启发框架模拟人类与GUI交互的认知过程——分解为Blink（快速注意力检测，类似眼跳）、Think（高级推理决策，类似认知规划）、Link（生成可执行命令，类似动作选择）三个生物合理阶段，配合自动化Blink数据标注和首个基于规则的过程+结果复合奖励机制，BTL-UI在静态GUI理解和动态交互任务上均达competitive性能。

**[Confounding Robust Deep Reinforcement Learning A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)**

:   提出因果DQN算法，从含混杂因子的观测数据中学习鲁棒策略，在Atari等高维环境中解决离线RL的混杂偏差问题。

**[Continual Knowledge Adaptation For Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)**

:   提出CKA-RL框架，用知识向量在多任务RL中实现持续知识适应，缓解灾难性遗忘问题。

**[Contribution Of Task-Irrelevant Stimuli To Drift Of Neural Representations](contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)**

:   理论证明任务无关刺激作为学习噪声导致神经表示漂移，并定量刻画了输入统计特性如何影响漂移的速度和方向。

**[Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)**

:   首次量化控制算法的“调优潜能”，通过 LLM Agent 模拟不同水平开发者进行参数优化和逻辑级改进，在 ABR 上相比贝叶斯优化提升 44.1%，CartPole 上 Bang-bang 从 34→500。

**[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)**

:   提出 DVD（Deep Video Discovery）agent，通过在分段视频片段上进行代理式搜索实现长视频理解——不同于使用预定义工作流的先前 video agent，DVD 利用 LLM 的推理能力在多粒度视频数据库上自主规划、策略性选择搜索工具、根据已获取信息动态编排自适应工作流，在 LVBench 上达到 74.2% 准确率（SOTA，显著超越先前所有工作），使用字幕时提升至 76.0%。

**[Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)**

:   提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

**[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](drift_dynamic_rulebased_defense_with_injection_isolation_for.md)**

:   提出 DRIFT 系统级 Agent 安全框架，通过 Secure Planner（预规划函数轨迹+参数检查表）、Dynamic Validator（基于 Read/Write/Execute 权限的动态策略更新）和 Injection Isolator（从 memory stream 中检测并屏蔽注入指令）三层防御，在 AgentDojo 上将 ASR 从 30.7% 降至 1.3%，同时比 CaMeL 提升 20.1% utility。

**[Evaluating LLMs in Open-Source Games](evaluating_llms_in_open-source_games.md)**

:   通过开源游戏（智能体提交程序而非原始行动）这一新范式，系统评估 LLM 在战略推理、互相学习和合作博弈中的能力，发现 LLM 可自动发现近似程序平衡。

**[Exact Learning Of Arithmetic With Differentiable Agents](exact_learning_of_arithmetic_with_differentiable_agents.md)**

:   用可微有限状态转换器(FST)实现算术任务的精确长度泛化，仅需很少的训练数据即可学到可證正确的算法。

**[Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)**

:   GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

**[Interactive And Hybrid Imitation Learning Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)**

:   提出state-wise交互算法，在低恢复成本设定下可证地超越行为克隆，结合少量交互查询和离线数据实现混合模仿学习。

**[Inverse Optimization Latent Variable Models For Learning Costs Applied To Route ](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)**

:   提出IO-LVM，从观测的组合优化解中学习潜在成本函数分布，无需agent标签，应用于路径规划问题。

**[It's LIT! Reliability-Optimized LLMs with Inspectable Tools](its_lit_reliability-optimized_llms_with_inspectable_tools.md)**

:   通过为工具分配可靠性和可调试性成本，引导 LLM 主动选择更透明、更易理解的工具，在 61/65 测试场景中提高可解释性，同时保持性能。

**[Last Iterate Convergence In Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)**

:   在单调平均场博弈中实现近端点的指数最后迭代收敛，无需时间平均，超越了先前仅保证平均收敛的结果。

**[Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)**

:   利用后缀树缓存长序列，通过自适应推测长度实现 5.3 倍加速，特别针对 Agent 场景中高度可预测的重复推理任务。

**[T1 A Tool-Oriented Conversational Dataset For Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)**

:   构建T1数据集——13.5K多轮对话，覆盖工具依赖、多步规划等复杂Agent场景，用于评估和训练LLM的多轮工具使用能力。

**[TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)**

:   首个 LLM 代理框架自动处理轨迹建模全流程，通过 UniEnv 统一接口和协作学习双层优化（LLM 推理 + 小模型训练），性能相比基线最高提升 69.91%。

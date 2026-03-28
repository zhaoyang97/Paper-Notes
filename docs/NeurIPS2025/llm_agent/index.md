<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧠 NeurIPS2025** · 共 **35** 篇

**[A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)**

:   提出 A-Mem，一种受 Zettelkasten 启发的 LLM Agent 智能记忆系统，每条记忆自动生成结构化笔记（关键词/标签/上下文描述），动态建立记忆间链接，并在新记忆加入时触发旧记忆的演化更新，在 LoCoMo 长对话 QA 上显著超越 MemGPT 等基线。

**[A Differentiable Model of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)**

:   本文用 JAX 实现了一个可微分的供应链 Agent-Based Model（ABM），通过 GPU 并行化和自动微分实现了比传统无梯度方法快 3 个数量级的贝叶斯参数校准，为大规模供应网络建模打开了可能性。

**[A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[Adaptive Cooperative Transmission Design for URLLC via Deep Reinforcement Learning](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   针对两跳协作中继通信中的URLLC难题，提出DRL-CoLA算法：将每跳传输参数配置建模为MDP，用双agent DQN在仅观测本地CSI和ARQ反馈下学习分布式时延感知传输策略，接近全局最优可靠性。

**[AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)**

:   提出 AgentAuditor，一个通用的无训练记忆增强推理框架，使 LLM 评估者能模拟人类专家评估 agent 的安全与安全性——通过自适应提取结构化语义特征并生成CoT推理轨迹构建经验记忆，多阶段上下文感知 RAG 检索相关经验指导新案例评估，在自建的 ASSEBench（2293条记录×15类風险×29场景）上达到人类水平准确率。

**[AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)**

:   AgentChangeBench 是首个系统评估 LLM agent 在对话中途目标切换时适应能力的 benchmark：315 基础任务 × 9 变体 = 2835 序列，覆盖 3 个企业领域（银行/零售/航空）和 5 种 user persona，引入 GSRT（目标切换恢复时间）等 4 个互补指标，揭示高 pass@k 掩盖的效率和鲁棒性差距——如 GPT-4o 航空恢复率 92.2% 但零售冗余率达 89.1%。

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

**[Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)**

:   系统评估 LLM 对通信动机的敏感性（motivational vigilance）——在控制实验中 LLM 能像人类一样折扣有偏见信源的建议（与理性模型相关系数 r>0.78），但在真实场景（YouTube 赞助广告）中表现大幅下降（r<0.2），通过简单的 prompt steering 可部分恢复。

**[Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools](attractive_metadata_attack_inducing_llm_agents_to_invoke_malicious_tools.md)**

:   AMA（Attractive Metadata Attack）证明仅通过精心设计恶意工具的元数据（名称、描述、参数模式），不需要提示注入或模型内部访问，就能诱导 LLM Agent 以 81-95% 的成功率调用攻击者工具并泄露隐私，同时几乎不影响原始任务完成（98%+），且现有防御（审计器、提示重写）效果有限。

**[Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

**[VeriMaAS: Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 提出自动组合 agent 工作流的框架用于 RTL 代码生成——关键创新是将 HDL 工具的形式化验证反馈直接整合到工作流生成中，无需梯度更新或长推理链，在 pass@k 上超过微调基线 5-7%，且训练样本需求降低一个量级。

**[Benchmarking Agentic Systems in Automated Scientific Information Extraction](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)**

:   构建 ChemX——10 个人工标注的多模态化学数据提取数据集，系统评估 SOTA Agent 系统（ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER）和 LLM（GPT-5 等），发现单 Agent 方法（GPT-5）在纳米酶数据集上达到 F1=0.58，超越专用多 Agent 系统。

**[BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](btlui_blinkthinklink_reasoning_model_for_gui_agent.md)**

:   提出"Blink-Think-Link"（BTL）脑启发框架模拟人类与GUI交互的认知过程——分解为Blink（快速注意力检测，类似眼跳）、Think（高级推理决策，类似认知规划）、Link（生成可执行命令，类似动作选择）三个生物合理阶段，配合自动化Blink数据标注和首个基于规则的过程+结果复合奖励机制，BTL-UI在静态GUI理解和动态交互任务上均达competitive性能。

**[ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)**

:   提出 ContextAgent，首个利用可穿戴设备多模态感知（视频+音频+通知）来理解用户意图并主动提供工具增强服务的 LLM Agent 框架，同时构建了包含 1000 个样本的 ContextAgentBench 基准，在主动预测准确率和工具调用上分别提升 8.5% 和 6.0%。

**[Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)**

:   首次提出"调优潜能"（Tuning Potential）概念并给出形式化度量，通过 LLM Agent 模拟不同能力水平的开发者对控制算法进行参数调优和逻辑级改进，在 ABR 任务上相比贝叶斯优化提升 44.1%，CartPole 上 Bang-bang 从 34→500 达到 DQN 水平。

**[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)**

:   提出 DVD（Deep Video Discovery）agent，将长视频理解建模为多步信息搜索问题：先将长视频构建为多粒度结构化数据库（全局摘要 + clip 级字幕嵌入 + 帧级像素），再提供三种搜索工具（Global Browse / Clip Search / Frame Inspect），由 reasoning LLM 通过 observe-reason-act 循环自主编排搜索轨迹，在 LVBench 达 74.2%（超先前 SOTA MR.Video 13.4 pp），加字幕 76.0%。

**[Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)**

:   提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

**[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](drift_dynamic_rulebased_defense_with_injection_isolation_for.md)**

:   提出 DRIFT 系统级 Agent 安全框架，通过 Secure Planner（预规划函数轨迹+参数检查表）、Dynamic Validator（基于 Read/Write/Execute 权限的动态策略更新）和 Injection Isolator（从 memory stream 中检测并屏蔽注入指令）三层防御，在 AgentDojo 上将 ASR 从 30.7% 降至 1.3%，同时比 CaMeL 提升 20.1% utility。

**[Evaluating LLMs in Open-Source Games](evaluating_llms_in_open-source_games.md)**

:   通过开源游戏（智能体提交程序而非原始行动）这一新范式，系统评估 LLM 在战略推理、互相学习和合作博弈中的能力，发现 LLM 可自动发现近似程序平衡。

**[Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)**

:   提出 Ground-Compose-Reinforce (GCR)，一个端到端的神经符号框架，通过少量标注轨迹（仅350条）学习原子命题的接地语义（Ground），将其通过 Reward Machine 组合成复杂任务规范（Compose），然后用自生成的稠密奖励训练 RL 智能体（Reinforce），无需手工奖励函数即可引出分布外的复杂行为。

**[Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)**

:   GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

**[It's LIT! Reliability-Optimized LLMs with Inspectable Tools](its_lit_reliability-optimized_llms_with_inspectable_tools.md)**

:   通过为工具分配可靠性和可调试性成本，引导 LLM 主动选择更透明、更易理解的工具，在 61/65 测试场景中提高可解释性，同时保持性能。

**[Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)**

:   基于真实 ML 会议竞赛构建动态基准 MLRC-Bench，评估 LLM Agent 提出和实现新颖研究方法的能力，发现最强 Agent（Gemini）仅达人类顶级方案 9.3% 的相对改进，且提供 AI/人类想法并不能一致改善实现质量。

**[Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)**

:   本文是一篇 position paper，主张将形式化最优控制理论作为 AI 对齐研究的核心工具，并提出"对齐控制栈"(Alignment Control Stack, ACS)——一个从物理硬件层到社会治理层的十层分层框架，用于系统地组织和分析不同对齐方法的测量、控制与互操作性。

**[SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)**

:   利用后缀树缓存长序列，通过自适应推测长度实现 5.3 倍加速，特别针对 Agent 场景中高度可预测的重复推理任务。

**[T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)**

:   构建 T1 数据集——13.5K 多轮对话覆盖 9 个领域（4 单领域 + 5 跨领域）、14 个工具，聚焦工具间依赖和动态重规划，并提出 T1-Agent（代码生成 + 缓存机制）作为基线系统；实验发现 SFT 后的 Llama 8B 在 Tool Call F1 上达 87.17%，超越未微调的 70B 模型，但仍落后于 GPT-5/o3 等闭源模型。

**[The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)**

:   提出 CGI（Critique-Guided Improvement）双角色框架，训练专门的 Critic 模型为 Actor Agent 提供结构化自然语言反馈（判别+修正建议），并通过迭代动作精炼让 Actor 学会利用这些反馈，在 WebShop/ScienceWorld/TextCraft 三个环境中平均得分 74.20%，超越 GPT-4o（45.46%）和 Iterative SFT（58.21%）。

**[TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)**

:   首个 LLM 代理框架自动处理轨迹建模全流程，通过 UniEnv 统一接口和协作学习双层优化（LLM 推理 + 小模型训练），性能相比基线最高提升 69.91%。

**[Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](web-shepherd_advancing_prms_for_reinforcing_web_agents.md)**

:   提出首个针对网页导航的过程奖励模型 Web-Shepherd，通过检查清单分解任务目标为可评估的子目标，3B/8B 模型在轨迹准确率上碾压 GPT-4o（85% vs 10%），同时成本仅为 1/10，使网页 Agent 的强化学习和推理时搜索变得实际可行。

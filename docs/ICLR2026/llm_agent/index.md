<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🔬 ICLR2026** · 共 **13** 篇

**[A Benchmark for Deep Information Synthesis (DeepSynth)](a_benchmark_for_deep_information_synthesis.md)**

:   提出 DeepSynth 基准，包含 120 个跨 7 领域 67 国的真实信息综合任务（平均需 5.5 小时人工标注），要求 agent 从多个网页收集信息并进行结构化推理，当前最强 agent（o3-deep-research）仅获 8.97 F1 / 17.5% LLM-Judge，揭示了 LLM agent 在信息综合方面的严重不足。

**[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)**

:   提出 ACE（Agentic Context Engineering）框架，将 context 视为不断演化的"策略手册"（playbook），通过 Generator-Reflector-Curator 三角色分工和增量式 delta 更新来持续积累和精炼策略，解决了现有 prompt 优化中的简洁偏差和上下文坍塌问题，在 agent 任务上平均提升 10.6%、金融任务提升 8.6%，且自适应延迟降低 86.9%。

**[AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)**

:   提出AgentSynth pipeline，利用信息不对称原理（正向逐步生成简单、反向整体求解困难）将简单子任务链式组合为复杂长程计算机使用任务，自动生成6000+多样化任务和轨迹，每条轨迹仅需$0.60，SOTA Agent在最高难度下成功率仅4%。

**[Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)**

:   构建 Ambig-SWE（基于 SWE-Bench Verified 的欠指定变体），系统评估 LLM 编程 agent 在三个维度上的交互能力——检测欠指定、提出澄清问题、利用交互信息——发现交互可将欠指定场景下的解决率提升最高 74%，但模型默认非交互行为且难以区分指定充分/不足的指令。

**[ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)**

:   揭示 LLM Agent 中 chat template 的结构性漏洞：通过在工具返回的数据中伪造角色标签（如 `<system>`, `<user>`），攻击者可以劫持模型的角色层级认知，将恶意指令伪装为高优先级指令，ASR 从 5-15% 提升至 32-52%。

**[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)**

:   提出 FeatureBench，面向复杂特征开发的 Agent 编程基准，200 个任务/24 个仓库，平均 790 行代码/15.7 文件——Claude Opus 4.5 仅解决 11%（vs SWE-bench 74.4%），揭示当前 Agent 在跨文件特征开发上的巨大差距。

**[Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)**

:   发现现代 LLM agents 虽然对直接对抗性压力具有鲁棒性（目标偏移为 0），但会从弱模型的上下文中"继承"目标偏移行为；更反直觉的是，指令层级遵循能力（system vs user prompt 优先级）与偏移抗性之间缺乏相关性——Gemini 不遵循 system prompt 但偏移抗性不差，Qwen3 遵循 system prompt 但仍被传染。

**[Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)**

:   提出 Judge Reliability Harness（JRH），一个开源框架，通过 label flip、格式不变性、语义改写、冗余偏差、随机稳定性 等合成测试系统评估 LLM Judge 的可靠性，在四个基准（FORTRESS、HarmBench、Persuade、AgentHarm）上对四个 SOTA Judge 进行压力测试，发现没有任何一个 Judge 在所有场景下都可靠。

**[MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)**

:   提出 MC-Search 基准（3333 个多跳多模态检索推理样本，平均 3.7 跳），通过 HAVE 验证和 Search-Align 过程监督微调，使开源模型达到接近商业模型的多模态搜索推理能力。

**[RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)**

:   提出 RefTool 框架基于外部参考资料（教材、知识片段）自动创建可执行 Python 工具，解决了现有工具创建方法依赖 LLM 内在知识在专业领域失败的问题，在因果推理、物理和化学任务上平均超过已有方法 12.3%。

**[Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)**

:   提出 HPL 框架解决长时序 LLM Agent 中偏好学习的粒度不匹配问题，通过三级 DPO（轨迹级+步骤级+动作组级）和双层课程学习（子任务复杂度×样本难度），在 ALFWorld/WebShop/InterCode-SQL 上显著超越 ETO 和 IPR 等基线（平均 59.44 vs 55.43/55.49）。

**[The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)**

:   系统评估当前 LLM 在长上下文代码调试中的能力极限，发现 agentic 工作流的成功来自任务分解而非长上下文推理（成功轨迹仅消耗 20-30K token），64K token 单次补丁生成中性能急剧下降（GPT-5-nano 0%），揭示名义上下文长度与实际可用上下文能力之间的显著差距。

**[ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)**

:   提出首个评估 LLM Agent 发现并修补新型零日漏洞的 benchmark，通过将真实 CVE 移植到不同代码库创建 22 个新颖高危漏洞任务，在 5 个信息层级评估 Agent 能力，发现最强模型在 zero-day 级别仅 14.4% 通过率，说明自主漏洞发现仍是重大挑战。

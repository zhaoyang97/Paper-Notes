---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**💬 ACL2025** · 共 **19** 篇

**[Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)**

:   Agentic Reasoning 提出了一个将 Web 搜索、代码执行和知识图谱记忆（Mind-Map）三种 Agent 工具集成到 LLM 推理过程中的框架，在 DeepSeek-R1 上将 Humanity's Last Exam 准确率从 9.4% 提升到 23.8%（+14.4%），GPQA 从 71.5% 到 81.2%，接近 OpenAI Deep Research 水平。

**[Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)**

:   Astute RAG 提出了一种对不完美检索具有鲁棒性的 RAG 方法，通过自适应生成 LLM 内部知识作为补充、带有来源标注的知识整合、以及基于可靠性的答案生成三个步骤，在 Gemini 和 Claude 上显著优于现有鲁棒 RAG 方法，且是唯一在最坏情况下（检索全部无用）不劣于无 RAG 基线的方法。

**[Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions](caution_environment_gui_agent_distractions.md)**

:   本文首次系统研究了多模态 GUI Agent 对环境干扰（弹窗广告、推荐内容等）的脆弱性，在无恶意攻击的自然场景下，即使最强的 MLLM（包括GPT-4o）也有 20-40% 的概率被环境中的无关内容分散注意力而执行偏离用户目标的操作。

**[Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](dpt_agent_dual_process.md)**

:   提出 DPT-Agent，首个将双过程理论（Dual Process Theory）系统化地融入语言智能体框架的方法——用有限状态机(FSM)+code-as-policy 作为快速直觉的 System 1，用心智理论(ToM)+异步反思的 LLM 作为慢速深思的 System 2，首次实现了自主的实时同步人机协作（在 Overcooked 困难版中）。

**[FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](fact_audit_factcheck.md)**

:   提出FACT-AUDIT——一个基于重要性采样和多智能体协作的自适应动态事实核查评估框架，通过动态生成测试数据、迭代探测模型弱点、并同时评估verdict预测和justification质量，全面审计LLM的事实核查能力边界。

**[GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)**

:   提出 GUI-explorer，一个无需训练的 GUI agent，通过自主探索收集功能感知的交互轨迹，并以无监督方式从状态转换三元组中挖掘 transition-aware 知识，在 SPA-Bench 和 AndroidWorld 上分别达到 53.7% 和 47.4% 的任务成功率。

**[GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)**

:   提出 GuideBench 基准测试，系统评估 LLM 在领域导向指南遵循方面的能力，覆盖 7 个任务类别共 1272 个实例，从规则遵循、规则更新鲁棒性和人类偏好对齐三个维度评估 18 个 LLM，发现当前模型在复杂领域规则遵循上仍有较大提升空间。

**[Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent_image_classification.md)**

:   提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

**[Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)**

:   提出 MeCo（Meta-Cognition Trigger），通过表示工程从 LLM 内部提取"元认知信号"——模型对自身能力的自我评估——来自适应决定是否需要调用外部工具，无需微调且计算开销极小，在多个骨干模型和基准上显著改善工具使用决策的准确性。

**[Unveiling Privacy Risks in LLM Agent Memory](mextra_agent_memory_privacy.md)**

:   本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

**[Multiple LLM Agents Debate for Equitable Cultural Alignment](multiple_llm_agents_debate_for_equitable.md)**

:   提出 Multi-Agent Debate 框架，让两个 LLM agent 围绕文化场景进行辩论并由 judge LLM 仲裁，在 NormAd-eti 基准上显著提升文化适应准确率和跨文化群体公平性，使 7-9B 小模型达到 27B 模型的性能水平。

**[NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](nexussum_narrative_summarization.md)**

:   提出 NexusSum，一个三阶段多Agent LLM框架（对话转描述→层次摘要→迭代压缩），无需微调即可处理书籍/电影/电视剧等长叙事文本的摘要生成，在 BookSum 上 BERTScore 提升达 30%。

**[OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use](os_agents_survey_mllm.md)**

:   首个系统性综述基于（多模态）大语言模型的操作系统智能体（OS Agents），覆盖基础概念、构建方法（基础模型+Agent框架）、评估基准和商业产品，全面梳理了从CogAgent到Anthropic Computer Use等50+工作的技术演进。

**[OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)**

:   提出OS-Genesis，一种"先交互探索再逆向生成任务"的GUI agent轨迹数据合成范式，通过无人监督的UI元素遍历收集状态转移三元组，逆向合成任务指令后用Trajectory Reward Model质量控制，在AndroidWorld上将Qwen2-VL-7B性能从9.82%提升至17.41%，接近GPT-4o的23.70%。

**[R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](r2d2_reflective_agentic_memory.md)**

:   R2D2 提出了一个结合 Remember（经验回放缓冲区 + A* 搜索导航）和 Reflect（错误反思 + 反思记忆存储）两范式的 Web Agent 框架，将 Web 导航从 Unknown MDP 转化为 Known MDP，在 WebArena 上导航错误减少 50%，任务完成率提升 3 倍，超越 SOTA 17%。

**[Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)**

:   提出 AgenticLU 框架，通过 Chain-of-Clarifications (CoC) 工作流让 LLM 自主生成澄清问题并检索相关上下文，再通过 SFT+DPO 两阶段微调将树搜索路径蒸馏到模型中，使 8B 模型在 128K 长上下文 QA 任务上大幅超越基线。

**[SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement](synworld_agentic_action_knowledge.md)**

:   SynWorld 提出让 Agent 在合成的虚拟场景中通过蒙特卡洛树搜索（MCTS）来探索和优化动作知识（工具描述和工作流），使 Agent 能够自主适应新环境的工具使用，在 ToolBench 上比 ReAct 基线提升约 9 个百分点。

**[Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)**

:   提出ToolHop——首个query-driven构建的多跳工具使用评估数据集（995个多跳查询+3912个本地可执行工具），评估14个LLM后发现最强的GPT-4o仅达49.04%准确率，揭示了不同模型家族在工具使用策略上的显著差异。

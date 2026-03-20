---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🤖 AAAI2026** · 共 **21** 篇

**[A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)**

:   提出 KDR-Agent 多智能体框架，通过知识检索（Wikipedia）、歧义消解和反思式自我纠错三个专业智能体协同工作，在仅使用少量静态标注示例的条件下，在5个领域10个NER数据集上显著超越现有零样本和少样本ICL NER方法。

**[AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)**

:   利用LLM驱动的具身智能体在模拟智能家居中"生活"，生成虚拟环境传感器数据用于预训练HAR模型，在低资源场景下显著提升活动识别性能。

**[AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)**

:   提出AquaSentinel，一个物理信息驱动的AI系统，通过稀疏传感器部署+物理增强虚拟传感器+MoE时空GNN集成+双阈值RTCA检测算法+因果流定位+LLM报告生成，仅用20-30%节点覆盖即可实现全网管道泄漏检测，在110个泄漏场景中达到100%检测率。

**[ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)**

:   提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

**[AutoTool: Efficient Tool Selection for Large Language Model Agents](autotool_efficient_tool_selection_for_large_language_model_a.md)**

:   提出AutoTool，一个基于图的免训练工具选择框架，通过发现并利用"工具使用惯性"（tool usage inertia）——即工具调用遵循可预测的顺序模式这一经验现象——构建工具惯性图（TIG），用统计方法代替部分LLM推理来高效选择工具和填充参数，在保持任务完成率的同时减少15-40%的推理成本。

**[BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)**

:   提出 vPGM 框架，通过自然语言引导 LLM Agent 模拟概率图模型（PGM）的贝叶斯推理过程，发现隐变量并推断后验分布，再用 Dirichlet 先验做数值贝叶斯校准（BayesVPGM），在多个推理任务上同时提升准确率和置信度校准。

**[Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](beyond_react_a_planner-centric_framework_for_complex_tool-au.md)**

:   提出以Planner为核心的Plan-Execute框架，将复杂查询转化为DAG执行计划，通过SFT+GRPO两阶段训练专门的Planner模型，在ComplexTool-Plan和StableToolBench上超越ReAct等反应式方法，用更少推理步骤实现更高成功率。

**[Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)**

:   提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

**[Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](cook_and_clean_together_teaching_embodied_agents_for_paralle.md)**

:   提出ORS3D任务——将运筹学(OR)知识引入具身AI的任务调度，要求智能体利用可并行子任务的等待时间执行其他任务以最小化总完成时间，同时在3D场景中定位目标物体；构建60K级数据集ORS3D-60K，并提出GRANT模型通过调度token机制连接外部动态规划求解器，在时间效率上比baseline提升30.53%。

**[COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)**

:   提出 VLM 与 RL 双向协同优化框架 COVR：RL 生成的高质量交互数据用于微调 VLM，增强后的 VLM 反过来通过 action prior 指导 RL 策略学习，在 CARLA 和 DMControl 上取得 SOTA。

**[D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)**

:   提出D-GARA，首个系统评估GUI Agent在真实世界异常情况（如UI变化、弹窗干扰、网络延迟、元素缺失等）下鲁棒性的动态benchmark框架，揭示现有GUI Agent在面对环境扰动时性能急剧下降的严重问题。

**[DEPO: Dual-Efficiency Preference Optimization for LLM Agents](depo_dual-efficiency_preference_optimization_for_llm_agents.md)**

:   提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

**[EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)**

:   提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

**[Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[History-Aware Reasoning for GUI Agents](history-aware_reasoning_for_gui_agents.md)**

:   提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

**[iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)**

:   iMAD 提出选择性触发多Agent辩论的框架：先让单Agent生成带自我批判的结构化响应，从中提取 41 个可解释的语言/语义特征，用轻量 MLP 分类器（FocusCal 损失训练）判断是否需要触发 MAD，在 6 个 QA/VQA 数据集上减少高达 92% 的 Token 开销，同时提升准确率高达 13.5%。

**[MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)**

:   提出 MedLA，首个基于三段论逻辑树的医学多 Agent 推理框架：每个 Agent 将推理组织为显式的逻辑树（大前提-小前提-结论三段论节点），多个 Agent 通过图引导的多轮讨论在前提级别对齐和修正逻辑树，在 MedDDx 上超越所有基线 7.4%（8B 模型），在医学 QA 上以 8B 模型达到 69.9% 平均准确率（超 70B RAG 模型）。

**[ProBench: Benchmarking GUI Agents with Accurate Process Information](probench_benchmarking_gui_agents_with_accurate_process_infor.md)**

:   提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

**[Prune4Web: DOM Tree Pruning Programming for Web Agent](prune4web_dom_tree_pruning_programming_for_web_agent.md)**

:   提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

**[SoMe: A Realistic Benchmark for LLM-based Social Media Agents](some_a_realistic_benchmark_for_llm-based_social_media_agents.md)**

:   提出 SoMe，首个全面评估 LLM 社交媒体 Agent 的 benchmark：8 个任务覆盖帖子分析、用户理解、综合推理，917 万帖子 + 6591 用户 + 1.7 万标注查询，配套 8 个 MCP 兼容工具，评估 13 个主流 LLM 发现最强模型仅 54.33 分（满分 100），揭示了推理能力≠Agent能力、工具调用幻觉普遍存在等关键发现。

**[TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents](tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)**

:   TongUI 提出从互联网上的多模态教程（视频+图文）自动转化为 GUI 操作轨迹数据的框架，构建了百万级的 GUI-Net-1M 数据集，用于微调 Qwen2.5-VL 模型，在多个 grounding 和 navigation 基准上超越或接近 UI-TARS 等 SOTA。

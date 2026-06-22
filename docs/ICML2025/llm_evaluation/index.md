---
title: >-
  ICML2025 LLM评测论文汇总 · 22篇论文解读
description: >-
  22篇ICML2025的 LLM 评测方向论文解读，涵盖 LLM、对齐/RLHF、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
  - "推理"
item_list:
  - u: "aaar-10_assessing_ais_potential_to_assist_research/"
    t: "AAAR-1.0: Assessing AI's Potential to Assist Research"
  - u: "are_llm_belief_updates_consistent_with_bayes_theorem/"
    t: "Are LLM Belief Updates Consistent with Bayes' Theorem?"
  - u: "bounded_rationality_for_llms_satisficing_alignment_at_inference-time/"
    t: "Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time"
  - u: "communicating_activations_between_language_model_agents/"
    t: "Communicating Activations Between Language Model Agents"
  - u: "consistency_in_language_models_current_landscape_challenges_and_future_direction/"
    t: "Consistency in Language Models: Current Landscape, Challenges, and Future Directions"
  - u: "correlated_errors_in_large_language_models/"
    t: "Correlated Errors in Large Language Models"
  - u: "datadecide_how_to_predict_best_pretraining_data_with_small_experiments/"
    t: "DataDecide: How to Predict Best Pretraining Data with Small Experiments"
  - u: "disentangling_and_integrating_relational_and_sensory_information_in_transformer_/"
    t: "Disentangling and Integrating Relational and Sensory Information in Transformer Architectures"
  - u: "enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln/"
    t: "EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities"
  - u: "evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_/"
    t: "MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels"
  - u: "fleet_of_agents_coordinated_problem_solving_with_large_language_models/"
    t: "Fleet of Agents: Coordinated Problem Solving with Large Language Models"
  - u: "g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr/"
    t: "G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration"
  - u: "how_much_can_we_forget_about_data_contamination/"
    t: "How Much Can We Forget about Data Contamination?"
  - u: "hyperband-based_bayesian_optimization_for_black-box_prompt_selection/"
    t: "Hyperband-based Bayesian Optimization for Black-box Prompt Selection"
  - u: "learning_distribution-wise_control_in_representation_space_for_language_models/"
    t: "Learning Distribution-Wise Control in Representation Space for Language Models"
  - u: "leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati/"
    t: "Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation"
  - u: "llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag/"
    t: "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs"
  - u: "phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation/"
    t: "PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation"
  - u: "position_theory_of_mind_benchmarks_are_broken_for_large_language_models/"
    t: "Position: Theory of Mind Benchmarks are Broken for Large Language Models"
  - u: "sample_efficient_demonstration_selection_for_in-context_learning/"
    t: "Sample Efficient Demonstration Selection for In-Context Learning"
  - u: "the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi/"
    t: "The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph"
  - u: "unlocking_post-hoc_dataset_inference_with_synthetic_data/"
    t: "Unlocking Post-hoc Dataset Inference with Synthetic Data"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🧪 ICML2025** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (131)](../../ICLR2026/llm_evaluation/index.md) · [💬 ACL2026 (96)](../../ACL2026/llm_evaluation/index.md) · [🧪 ICML2026 (40)](../../ICML2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×6

**[AAAR-1.0: Assessing AI's Potential to Assist Research](aaar-10_assessing_ais_potential_to_assist_research.md)**

:   提出 AAAR-1.0 基准，通过公式推断、实验设计、论文弱点发现、审稿质量鉴别四个专家级任务，系统评估 LLM 辅助科研的真实能力，揭示当前模型在深度研究任务上仍有显著不足。

**[Are LLM Belief Updates Consistent with Bayes' Theorem?](are_llm_belief_updates_consistent_with_bayes_theorem.md)**

:   本文提出贝叶斯一致性系数（BCC）来量化 LLM 的信念更新是否符合贝叶斯定理，发现更大、更强的预训练模型在给定新证据时，其信念更新与贝叶斯定理更一致。

**[Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](bounded_rationality_for_llms_satisficing_alignment_at_inference-time.md)**

:   提出 SITAlign——基于有界理性的满意决策框架，在推理时最大化主要目标（如有用性）同时确保次要目标（如无害性）满足阈值约束，通过对偶理论求解，在 GPT-4 评估上相比多目标解码 SOTA 提升 22.3% 胜率。

**[Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)**

:   提出让 LLM 智能体通过中间层激活（而非自然语言）进行通信的方法——在模型 B 的前向传播中间层注入模型 A 的激活向量，无需额外参数和数据，在多项推理基准上比自然语言通信提升 27%，计算量仅为 1/4。

**[Consistency in Language Models: Current Landscape, Challenges, and Future Directions](consistency_in_language_models_current_landscape_challenges_and_future_direction.md)**

:   系统综述了 LLM 一致性研究的全景，提出包含逻辑一致性（否定/对称/传递）、语义一致性、事实/信息一致性和非逻辑一致性（道德/规范）的分类体系，分析了 2019-2025 年间评测方法的不足，并呼吁建立标准化多语言基准和跨学科方法。

**[Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)**

:   本文通过对超过350个LLM的大规模实证分析，发现不同LLM之间存在高度相关的错误模式——在两个模型都出错时约60%的情况下会选择同一个错误答案，且越准确的模型相关性越高；进而研究了这种相关性对LLM-as-Judge评估和招聘市场的下游影响。

**[DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)**

:   > 本文构建了 DataDecide——迄今最大规模的开放模型套件（25 种数据配方 × 14 种模型规模 × 3 个随机种子），系统研究如何用小规模实验预测最佳预训练数据，发现单一小规模排名（如 150M 参数）即可达到约 80% 的成对决策准确率，且连续似然代理指标仅需目标计算量 0.01% 即可让多个基准任务的预测准确率超过 80%。

**[Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](disentangling_and_integrating_relational_and_sensory_information_in_transformer_.md)**

:   本文提出了 Dual Attention Transformer（DAT），通过在标准注意力机制中引入"关系注意力"头，将感知信息和关系信息解耦后并行处理再整合，在关系推理基准、数学问题求解、图像识别和语言建模等任务上均展现出显著的数据效率和参数效率提升。

**[EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)**

:   EnIGMA 是一个用于自主解决 Capture The Flag (CTF) 挑战的 LM agent，通过引入新型交互式 Agent 工具（调试器和服务器连接工具），首次使 LM agent 能够运行交互式终端程序，在 4 个基准的 390 个 CTF 挑战上取得 SOTA，并发现了 "soliloquizing" 这一新的幻觉现象。

**[MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)**

:   受 Bloom 分类法启发，提出多认知层次评估框架 MultiCogEval，从知识掌握、综合应用、情景问题解决三个层次评估 LLM 医学能力，发现所有模型性能随认知复杂度增加显著下降，且模型规模在高层次更关键。

**[Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)**

:   提出Fleet of Agents(FoA)——用遗传粒子滤波思想协调多Agent的LLM推理：生成多个Agent各自探索→基于启发式价值函数重采样→动态分支适应发现的方案，平均比SOTA方法提升5%质量同时仅需40%的成本。

**[G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)**

:   提出 G-Sim 混合框架，利用 LLM 自动设计仿真器的因果结构（子模块与连接关系），再通过无梯度优化（GFO）或仿真推断（SBI）对数值参数进行经验校准，在迭代循环中不断改进，生成可靠、可干预的通用仿真器。

**[How Much Can We Forget about Data Contamination?](how_much_can_we_forget_about_data_contamination.md)**

:   通过受控实验系统量化数据污染对 LLM benchmark 评估的影响，发现在超过 Chinchilla 最优五倍以上的训练数据量下，即使 144 次重复的污染数据也能被完全遗忘；进一步证明权重衰减是遗忘的关键机制，并据此推断 Llama 3 405B 等大型模型已遗忘训练早期的数据。

**[Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)**

:   提出 HbBoPs 方法，结合结构感知深度核高斯过程（对 instruction 和 few-shot exemplar 分别编码）与 Hyperband 多保真度调度器，在黑盒 LLM 的 prompt 选择问题上同时实现样本高效和查询高效，在十个基准和三个 LLM 上超越所有 SOTA 方法。

**[Learning Distribution-Wise Control in Representation Space for Language Models](learning_distribution-wise_control_in_representation_space_for_language_models.md)**

:   将表示微调（Representation Fine-tuning）中的确定性节点替换为随机节点，通过重参数化技巧学习潜在分布而非单点变换，在常识推理和数学推理任务上取得了一致性能提升，尤其在早期层的干预效果最为显著。

**[Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)**

:   利用 Art of Problem Solving (AoPS) 论坛的社区内容，构建了 652K 奥赛级数学 QA 对的训练集 AoPS-Instruct 和带时间戳的抗污染评估集 LiveAoPSBench，揭示了 LLM 在旧数据上的高表现可能源于预训练数据泄露而非真正推理能力。

**[LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs](llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag.md)**

:   提出LLM-SRBench基准（239题/4个科学领域），通过方程变换(LSR-Transform)和合成问题(LSR-Synth)防止LLM的记忆化，当前最好方法仅达31.5%符号准确率。

**[PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)**

:   提出 PhantomWiki——一个按需生成虚构世界语料库和 QA 对的评测框架，通过上下文无关文法（CFG）控制推理难度、调节宇宙规模控制检索难度，实现对 LLM 推理与检索能力的解耦评估，同时天然抵抗数据泄漏。

**[Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)**

:   这篇 Position Paper 指出当前大多数 LLM Theory of Mind（ToM）基准只测“能否预测他人行为”（Literal ToM），却没有测“能否基于该预测采取最优响应”（Functional ToM），因此会系统性高估模型在真实交互中的适应能力。

**[Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)**

:   本文提出了一种样本高效的上下文学习(ICL)示例选择方法，能够在有限的标注预算下高效地选择最佳示例组合，显著提升 LLM 的 ICL 性能，同时大幅减少所需的标注数据量。

**[The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)**

:   提出 GraphFilter 方法，将 SFT 数据集建模为句子-n-gram 的二部图，通过乘法优先级函数同时优化数据质量和多样性，在 3 个模型 6 个基准上全面超越 9 种基线方法。

**[Unlocking Post-hoc Dataset Inference with Synthetic Data](unlocking_post-hoc_dataset_inference_with_synthetic_data.md)**

:   提出通过合成生成held-out数据集并结合后校准（post-hoc calibration）来实现无需真实held-out集的数据集推断（Dataset Inference），通过suffix completion生成高质量合成数据、双分类器校准解耦生成偏移与成员信号，在15个多样化文本数据集上实现高置信度版权检测且低误报率。

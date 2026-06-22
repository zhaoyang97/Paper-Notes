---
title: >-
  NeurIPS2025 代码智能论文汇总 · 19篇论文解读
description: >-
  19篇NeurIPS2025的代码智能方向论文解读，涵盖 LLM、代码智能、Agent、推理、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "推理"
  - "布局/合成"
item_list:
  - u: "a_selfimproving_coding_agent/"
    t: "A Self-Improving Coding Agent"
  - u: "a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction/"
    t: "A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions"
  - u: "astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast/"
    t: "AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy"
  - u: "automated_multi-agent_workflows_for_rtl_design/"
    t: "Automated Multi-Agent Workflows for RTL Design"
  - u: "co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning/"
    t: "Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning"
  - u: "codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin/"
    t: "CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning"
  - u: "embedding_alignment_in_code_generation_for_audio/"
    t: "Embedding Alignment in Code Generation for Audio"
  - u: "learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation/"
    t: "Learning From Design Procedure To Generate CAD Programs for Data Augmentation"
  - u: "learning_to_solve_complex_problems_via_dataset_decomposition/"
    t: "Learning to Solve Complex Problems via Dataset Decomposition"
  - u: "maintaincoder_maintainable_code_generation_under_dynamic_requirements/"
    t: "MaintainCoder: Maintainable Code Generation Under Dynamic Requirements"
  - u: "mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research/"
    t: "MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research"
  - u: "once_upon_an_input_reasoning_via_per-instance_program_synthesis/"
    t: "Once Upon an Input: Reasoning via Per-Instance Program Synthesis"
  - u: "preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o/"
    t: "Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization"
  - u: "principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio/"
    t: "Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward"
  - u: "program_synthesis_via_test-time_transduction/"
    t: "Program Synthesis via Test-Time Transduction"
  - u: "qimeng-salv_signal-aware_learning_for_verilog_code_generation/"
    t: "QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation"
  - u: "searching_latent_program_spaces/"
    t: "Searching Latent Program Spaces"
  - u: "swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat/"
    t: "SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents"
  - u: "text-to-code_generation_for_modular_building_layouts_in_building_information_mod/"
    t: "Text-to-Code Generation for Modular Building Layouts in Building Information Modeling"
item_total: 19
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧠 NeurIPS2025** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (58)](../../ICLR2026/code_intelligence/index.md) · [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🧪 ICML2026 (22)](../../ICML2026/code_intelligence/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (9)](../../ICML2025/code_intelligence/index.md)

🔥 **高频主题：** LLM ×4 · 代码智能 ×4 · Agent ×2 · 推理 ×2 · 布局/合成 ×2

**[A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)**

:   将 LLM 迭代交互中的多目标优化建模为 SDE（漂移-扩散过程），通过干扰矩阵量化目标间的耦合模式，通过特征值谱分析策略收敛行为，在代码生成（安全性、效率、功能性三目标）上验证了不同策略的收敛率（0.33-1.29）和可预测性（$R^2$ 达 0.74）。

**[AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)**

:   AstroVisBench 构建了首个评估 LLM 天文科学计算和可视化能力的代码基准——从 110 个 Jupyter Notebook 提取 864 个任务（处理+可视化），设计双重评估管线（执行式变量检查 + VLM-as-Judge 可视化评分，与专家 Spearman ρ=0.822），评测 8 个 SOTA 模型后发现 Gemini 2.5 Pro 最佳但无错误率仅 15.7%，FileNotFoundError 占 43% 错误。

**[Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 是一个多智能体框架，通过将 HDL 形式化验证反馈（Yosys + OpenSTA）集成到工作流自动生成过程中，自适应地为 RTL 代码生成任务选择推理算子（I/O → CoT → ReAct → SelfRefine → Debate），以仅数百个训练样本实现比微调基线高 5-7% 的 pass@k 性能。

**[Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出 CURE 框架，让同一个 LLM 同时扮演代码生成器和单元测试生成器两个角色，通过生成代码与生成测试的交叉执行构建成对奖励矩阵，用基于理论推导的奖励信号进行强化学习，在完全不需要 ground-truth 代码标注的情况下实现代码生成能力和单元测试生成能力的共同进化，在五个编程基准上大幅超过同规模的专用 Coder 模型。

**[CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning](codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin.md)**

:   提出 CodeCrash 压力测试框架，通过功能等价的结构扰动和误导性自然语言注入（注释/print/暗示），系统评估 17 个 LLM 的代码推理鲁棒性，揭示模型平均性能下降 23.2%，CoT 仅能挽回至 13.8%，并首次发现大推理模型（LRM）中的 "Reasoning Collapse" 现象。

**[Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)**

:   提出双 MLP + InfoNCE 对比学习框架，将代码嵌入（distilroberta-base）和音频嵌入（wav2vec2）对齐到共享空间，使 LLM 代码生成流程无需编译执行即可从代码推断音乐相似性，CKA 从 0.090 提升至 0.590。

**[Learning From Design Procedure To Generate CAD Programs for Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)**

:   提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。

**[Learning to Solve Complex Problems via Dataset Decomposition](learning_to_solve_complex_problems_via_dataset_decomposition.md)**

:   提出Decomp方法，利用教师模型将复杂数学题按推理步骤递归分解为更简单的子问题，构建概念依赖图量化难度，再按从易到难的课程顺序训练学生模型——Qwen2.5-1.5B在MATH-500上达51.6%（超MuggleMath用147K数据的50.4%），Qwen3-4B在AIME2025仅用385样本达16.7%（超Qwen2.5-72B的15%）。

**[MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)**

:   首次系统定义并解决 LLM 代码生成的**可维护性**问题，同时贡献基准和方法：MaintainBench 通过 4 种需求变化模式 + 动态指标评测代码在需求演化下的可维护性；MaintainCoder 将 Waterfall 模型、设计模式与 6 个专业化 Agent 结合，动态可维护性指标提升 60%+，且初始代码正确性也一并提高。

**[MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)**

:   提出 MLR-Bench，一个包含 201 个开放式 ML 研究任务的综合基准，配套 MLR-Judge（LLM 评审框架）和 MLR-Agent（模块化研究代理），发现当前最先进的编码代理在约 80% 的情况下会生成伪造或未验证的实验结果，揭示了 AI 自动化科学研究的核心瓶颈。

**[Once Upon an Input: Reasoning via Per-Instance Program Synthesis](once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)**

:   提出 PIPS（Per-Instance Program Synthesis），通过实例级别的程序合成与结构化反馈迭代改进，结合置信度度量动态选择直接推理或程序合成，在30个基准上将调和平均准确率提升8.6%。

**[Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)**

:   系统研究了校准数据的组成特性（序列长度/样本量/来源/格式）和领域对应关系对LLM压缩后能力保持的影响，发现激活空间中的代表性和多样性是数据质量的本质决定因素，并据此提出三阶段校准数据策展框架COLA。

**[Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)**

:   系统研究如何利用用户编辑数据微调 LLM，将偏好、监督标签和代价三种反馈类型统一起来，并提出一种简单的集成方法，在不同用户分布下实现鲁棒适应。

**[Program Synthesis via Test-Time Transduction](program_synthesis_via_test-time_transduction.md)**

:   提出 SYNTRA 框架，将程序合成重新定义为转导式学习——在测试时利用可见的 test inputs 和 LLM 的判断来迭代消除不一致的候选程序假设，通过 greedy maximin 算法最小化 LLM 查询次数，在 4 个 benchmark 上准确率提升最高达 196%。

**[QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)**

:   提出信号级感知学习方法 QiMeng-SALV，通过从部分错误的 Verilog 模块中提取信号级功能正确的代码片段作为 DPO 训练的奖励信号，将优化粒度从模块级提升到信号级，在 VerilogEval 和 RTLLM 上达到 SOTA。

**[Searching Latent Program Spaces](searching_latent_program_spaces.md)**

:   提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。

**[SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)**

:   构建全自动化流水线从 GitHub 持续挖掘真实软件工程交互任务，生成 21,000+ 可执行 Python 任务的 SWE-rebench 数据集和去污染 benchmark，揭示部分模型在 SWE-bench Verified 上的性能存在污染膨胀问题（如 DeepSeek-V3 在 SWE-bench 上 39.7% vs SWE-rebench 上 21.3%）。

**[Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)**

:   提出 Text2MBL 框架，将自然语言描述转化为可执行的 BIM 代码（而非坐标序列），通过面向对象的代码架构和 LLM 微调实现模块化建筑布局的自动生成，在几何一致性上比坐标驱动方法提升 10%+ IoU。

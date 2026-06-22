---
title: >-
  ACL2025 代码智能论文汇总 · 28篇论文解读
description: >-
  28篇ACL2025的代码智能方向论文解读，涵盖代码智能、LLM、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
item_list:
  - u: "benchmarking_long-context_language_models_on_long_code_understanding/"
    t: "LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding"
  - u: "beyond_sequences_two-dimensional_representation_and_dependency_encoding_for_code/"
    t: "Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation"
  - u: "coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev/"
    t: "CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation"
  - u: "codedpo_code_alignment/"
    t: "CodeDPO: Aligning Code Models with Self Generated and Verified Source Code"
  - u: "codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod/"
    t: "CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation"
  - u: "codereviewqa_the_code_review_comprehension_assessment_for_large_language_models/"
    t: "CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models"
  - u: "compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll/"
    t: "CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System"
  - u: "coret_improved_retriever_for_code_editing/"
    t: "CoRet: Improved Retriever for Code Editing"
  - u: "dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_/"
    t: "DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal"
  - u: "dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language/"
    t: "DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation"
  - u: "exploracoder_advancing_code_generation_for_multiple_unseen_apis_via_planning_and/"
    t: "ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration"
  - u: "feabench_repo_code_gen/"
    t: "FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation"
  - u: "galla_graph_aligned_large_language_models/"
    t: "GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding"
  - u: "gift_gibbs_fine_tuning_code_gen/"
    t: "GiFT: Gibbs Fine-Tuning for Code Generation"
  - u: "mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios/"
    t: "MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios"
  - u: "oasis_order-augmented_strategy_for_improved_code_search/"
    t: "OASIS: Order-Augmented Strategy for Improved Code Search"
  - u: "personality_guided_code_gen/"
    t: "Personality-Guided Code Generation Using Large Language Models"
  - u: "program_synthesis_benchmark_for_visual_programming_in_xlogoonline_environment/"
    t: "Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment"
  - u: "reflectioncoder_learning_from_reflection_sequence_for_enhanced_one-off_code_gene/"
    t: "ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation"
  - u: "rethinking_repetition_problems_of_llms_in_code_generation/"
    t: "Rethinking Repetition Problems of LLMs in Code Generation"
  - u: "revisit_self-debugging_with_self-generated_tests_for_code_generation/"
    t: "Revisit Self-Debugging with Self-Generated Tests for Code Generation"
  - u: "scenegenagent_precise_industrial_scene_generation_with_coding_agent/"
    t: "SceneGenAgent: Precise Industrial Scene Generation with Coding Agent"
  - u: "share_text_to_sql_correction/"
    t: "SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL"
  - u: "star-sql_self-taught_reasoner_for_text-to-sql/"
    t: "STaR-SQL: Self-Taught Reasoner for Text-to-SQL"
  - u: "texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms/"
    t: "TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs"
  - u: "tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio/"
    t: "Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation"
  - u: "tree_of_evolution_code_gen/"
    t: "Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models"
  - u: "utboost_rigorous_evaluation_of_coding_agents_on_swe-bench/"
    t: "UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench"
item_total: 28
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**💬 ACL2025** · **28** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (58)](../../ICLR2026/code_intelligence/index.md) · [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🧪 ICML2026 (22)](../../ICML2026/code_intelligence/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md)

🔥 **高频主题：** 代码智能 ×13 · LLM ×8 · Agent ×4

**[LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)**

:   提出 LongCodeU 基准，从代码单元感知、单元内理解、单元间关系理解和长文档理解四个维度设计 8 个任务，评估 9 个长上下文语言模型在真实仓库级长代码上的理解能力，揭示 32K token 是当前 LCLM 长代码理解的实际上限。

**[Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation](beyond_sequences_two-dimensional_representation_and_dependency_encoding_for_code.md)**

:   本文提出超越传统一维序列表示的二维代码表示方法，通过显式编码代码的结构依赖关系（如语法树结构和变量依赖），显著提升了代码生成的准确性和结构正确性。

**[CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)**

:   提出 CoCo-Bench（Comprehensive Code Benchmark），一个覆盖代码理解、代码生成、代码修改和代码审查四个维度的综合代码基准，支持多编程语言和多难度等级，通过严格的人工审核确保数据质量，揭示了现有 LLM 在代码能力上的不均衡表现。

**[CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)**

:   提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。

**[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)**

:   提出 CodeReviewQA 基准，将代码审查自动修正（ACR）任务分解为三个中间推理步骤——变更类型识别（CTR）、变更定位（CL）、解决方案识别（SI），各自设计为不同难度的多选题探测，在 900 个人工验证的高质量样例（9 种语言）上评测 72 个 LLM，揭示了模型在代码审查理解中的具体弱点。

**[CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)**

:   提出 CompileAgent，首个面向仓库级代码编译的 LLM Agent 框架，集成五种专用工具和流程化 Agent 策略，在 100 个 C/C++ 真实项目的 CompileAgentBench 上将编译成功率最高提升 71%，平均每个项目仅需 $0.22。

**[CoRet: Improved Retriever for Code Editing](coret_improved_retriever_for_code_editing.md)**

:   提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。

**[DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)**

:   本文提出 DARS（动态动作重采样），一种针对编程智能体的推理时计算扩展方法，在智能体做出次优决策的关键节点上动态分支并尝试替代动作，在 SWE-Bench Lite 上以 Claude 3.5 Sonnet V2 实现 55% 的 pass@k 和 47% 的 pass@1，超越当时开源 SOTA 框架。

**[DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)**

:   提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。

**[ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration](exploracoder_advancing_code_generation_for_multiple_unseen_apis_via_planning_and.md)**

:   提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。

**[FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)**

:   本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。

**[OASIS: Order-Augmented Strategy for Improved Code Search](oasis_order-augmented_strategy_for_improved_code_search.md)**

:   提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。

**[Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)**

:   用 GPT-4o 为每个编程任务动态生成适配的 MBTI 人格类型和详细描述，再让目标 LLM 以该人格角色扮演程序员生成代码，在 7 个 LLM × 4 个数据集的 28 个组合中 23 个取得 pass rate 提升，最高达 12.9%，关键因素是人格多样性而非某个特定人格。

**[Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment](program_synthesis_benchmark_for_visual_programming_in_xlogoonline_environment.md)**

:   本文基于 XLogoOnline 视觉编程环境构建了一个需要空间规划、编程和逻辑推理等多技能组合的程序合成基准，发现 GPT-4V 仅能解决 20% 的任务，但通过 8 万+合成数据微调加上模拟器驱动的课程学习，Llama3-8B 大幅超越了 GPT-4V 和 Llama3-70B。

**[ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation](reflectioncoder_learning_from_reflection_sequence_for_enhanced_one-off_code_gene.md)**

:   ReflectionCoder通过构建整合编译器反馈的"反思序列"（reflection sequence）数据，结合反思自蒸馏和动态掩码蒸馏两种训练策略，使模型在一次性代码生成中达到SOTA性能，无需运行时的多轮调试。

**[Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)**

:   本文重新定义了代码生成中的重复问题，区分出比内容重复更普遍且更难处理的"结构性重复"，并提出基于语法规则的重复惩罚解码方法RPG（Repetition Penalization based on Grammar），在新构建的CodeRepetEval和标准基准上大幅缓解重复问题。

**[Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)**

:   系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。

**[SceneGenAgent: Precise Industrial Scene Generation with Coding Agent](scenegenagent_precise_industrial_scene_generation_with_coding_agent.md)**

:   提出 SceneGenAgent，一个基于 LLM 的代码生成 Agent，通过结构化布局规划、布局验证和迭代优化流程，利用 C# 代码精确生成工业场景，在真实工业任务上达到 81% 成功率，并构建 SceneInstruct 数据集使开源 LLM 接近 GPT-4o 水平。

**[SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)**

:   提出 SHARE 框架，用三个 <8B 参数的专用小语言模型（SLM）组成顺序管道，将声明式 SQL 转换为可暴露推理路径的步进动作轨迹，再分阶段修正 schema 链接错误与逻辑推理错误，以极低成本实现 LLM 的 Text-to-SQL 自纠正。

**[STaR-SQL: Self-Taught Reasoner for Text-to-SQL](star-sql_self-taught_reasoner_for_text-to-sql.md)**

:   将 Text-to-SQL 任务重新定义为推理驱动的过程，通过 STaR（Self-Taught Reasoner）自举方法让 LLM 学习生成逐步推理来辅助 SQL 生成，并集成 ORM 验证器进行 best-of-N 采样，在 Spider 基准上达到 86.6% 执行准确率。

**[TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)**

:   提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。

**[Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)**

:   提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。

**[Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构代码指令合成框架，通过多路径进化和质量驱动优化克服Code Evol-Instruct和OSS-Instruct的单向合成与随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)**

:   本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。

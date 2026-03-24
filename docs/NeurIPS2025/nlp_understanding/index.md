<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**🧠 NeurIPS2025** · 共 **7** 篇

**[AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)**

:   提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

**[Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)**

:   构建Braingle Brainteaser基准（242数学+236逻辑谜题），系统评估LLM在脑筋急转弯上的推理策略——发现模型有时能产生创造性洞察式解法，但也常在有巧妙解法可用时退回暴力穷举，且自纠错能力有限、将叙事→数学格式翻译可小幅提升性能。

**[Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)**

:   提出 diversity-steered sampling 框架：在解码时注入基于 NLI 的语义相似度惩罚来驱动生成语义多样化的样本，并用重要性加权+控制变量纠正偏差降低方差，在仅 16 个样本下即可准确估计 LLM 的语义熵（偶然不确定性）和互信息（认知不确定性）。

**[Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)**

:   将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

**[How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)**

:   在高维渐近框架下证明了带非线性MLP头的Transformer在ICL误差上等价于结构化多项式预测器，揭示了非线性MLP对非线性任务的增益机制，以及多源数据混合中低噪声和结构化协方差是高质量数据源的关键特征。

**[Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)**

:   提出PNLC方法，通过训练轻量级目标条件价值函数作为"自然语言评论家"，在推理步骤层面引导LLM智能体进行多轮规划和自我精化，无需直接微调或推理时搜索，在Web导航、社交推理、劝服等复杂交互任务上显著超越现有方法且推理速度快8-10倍。

**[The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)**

:   通过 AttnLRP 归因方法系统解剖 LLM 在 in-context retrieval augmented QA 中的内部机制，发现三类功能特化的注意力头——Task heads（中间层，解析指令/问题）、Retrieval heads（后层，逐字复制上下文答案）、Parametric heads（编码参数化知识），并通过 Function Vector 注入和来源追踪探针验证其功能，在 Llama-3.1/Mistral/Gemma 上 ROC AUC ≥94%。

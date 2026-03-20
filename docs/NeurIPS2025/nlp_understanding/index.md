<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**🧠 NeurIPS2025** · 共 **4** 篇

**[AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)**

:   提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

**[Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)**

:   构建Braingle Brainteaser基准（242数学+236逻辑谜题），系统评估LLM在脑筋急转弯上的推理策略——发现模型有时能产生创造性洞察式解法，但也常在有巧妙解法可用时退回暴力穷举，且自纠错能力有限、将叙事→数学格式翻译可小幅提升性能。

**[Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)**

:   将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

**[The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)**

:   通过 AttnLRP 归因方法系统解剖 LLM 在 in-context retrieval augmented QA 中的内部机制，发现三类功能特化的注意力头——Task heads（中间层，解析指令/问题）、Retrieval heads（后层，逐字复制上下文答案）、Parametric heads（编码参数化知识），并通过 Function Vector 注入和来源追踪探针验证其功能，在 Llama-3.1/Mistral/Gemma 上 ROC AUC ≥94%。

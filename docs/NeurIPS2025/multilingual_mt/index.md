---
title: >-
  NeurIPS2025 多语言/翻译论文汇总 · 11篇论文解读
description: >-
  11篇NeurIPS2025的多语言/翻译方向论文解读，涵盖 LLM、翻译、对齐/RLHF、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "多语言/翻译"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "翻译"
  - "对齐/RLHF"
  - "少样本学习"
item_list:
  - u: "adaptive_originality_filtering_rejection_based_prompting_and_riddlescore_for_cul/"
    t: "Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation"
  - u: "exploring_the_translation_mechanism_of_large_language_models/"
    t: "Exploring the Translation Mechanism of Large Language Models"
  - u: "helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_/"
    t: "HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages"
  - u: "how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme/"
    t: "How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs"
  - u: "merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query/"
    t: "MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query"
  - u: "on_extending_direct_preference_optimization_to_accommodate_ties/"
    t: "On Extending Direct Preference Optimization to Accommodate Ties"
  - u: "parallelprompt_extracting_parallelism_from_large_language_model_queries/"
    t: "ParallelPrompt: Extracting Parallelism from Large Language Model Queries"
  - u: "quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-/"
    t: "Quantifying Climate Policy Action and Its Links to Development Outcomes: A Cross-National Data-Driven Analysis"
  - u: "reflective_translation_improving_low-resource_machine_translation_via_structured/"
    t: "Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection"
  - u: "xifbench_evaluating_large_language_models_on_multilingual_instruction_following/"
    t: "XIFBench: Evaluating Large Language Models on Multilingual Instruction Following"
  - u: "zero-shot_performance_prediction_for_probabilistic_scaling_laws/"
    t: "Zero-Shot Performance Prediction for Probabilistic Scaling Laws"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🧠 NeurIPS2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (8)](../../ICLR2026/multilingual_mt/index.md) · [💬 ACL2026 (63)](../../ACL2026/multilingual_mt/index.md) · [🧪 ICML2026 (3)](../../ICML2026/multilingual_mt/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/multilingual_mt/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/multilingual_mt/index.md) · [🧪 ICML2025 (1)](../../ICML2025/multilingual_mt/index.md)

🔥 **高频主题：** LLM ×3 · 翻译 ×2

**[Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation](adaptive_originality_filtering_rejection_based_prompting_and_riddlescore_for_cul.md)**

:   提出 Adaptive Originality Filtering (AOF)——一种基于语义拒绝采样的提示策略，通过 MiniLM 嵌入的余弦相似度过滤重复/模板化输出，强制 LLM 生成更新颖、多样且文化匹配的多语言谜语；同时提出 RiddleScore 复合评估指标（Novelty + Diversity + Fluency + Alignment），与人类评分相关性达 $\rho=0.83$。

**[Exploring the Translation Mechanism of Large Language Models](exploring_the_translation_mechanism_of_large_language_models.md)**

:   提出 subspace-intervened path patching 方法对 LLM 翻译机制进行精细因果分析，发现翻译由不到 5% 的稀疏 attention head 驱动——分为 source head、indicator head、positional head 三类功能角色，MLP 将其特征整合为以英语为中心的中间表示，仅微调 64 个关键 head 即可匹配全参数微调性能。

**[HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)**

:   NVIDIA 发布的 40K+ 开源人工标注偏好数据集，覆盖通用/STEM/代码/多语言（13 种语言），训练的奖励模型在 RM-Bench 上达 82.4%（+10%），CC-BY-4.0 许可对商业友好。

**[How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)**

:   在高维渐近框架下证明了带非线性MLP头的Transformer在ICL误差上等价于结构化多项式预测器，揭示了非线性MLP对非线性任务的增益机制，以及多源数据混合中低噪声和结构化协方差是高质量数据源的关键特征。

**[MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query](merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)**

:   提出首个多语言交错多条件语义检索数据集 MERIT（320K queries, 135K products, 5种语言, 7大品类），揭示现有检索模型仅关注全局语义而忽略条件细节的瓶颈，并设计 Coral 微调框架通过嵌入重建+对比学习将检索性能提升 45.9%。

**[On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)**

:   将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

**[ParallelPrompt: Extracting Parallelism from Large Language Model Queries](parallelprompt_extracting_parallelism_from_large_language_model_queries.md)**

:   构建了首个查询内并行（intra-query parallelism）基准数据集ParallelPrompt，包含37000+条真实用户提示的结构化分解标注，证明约10%的用户查询包含可并行的潜在结构，并行执行可实现最高5.7×的延迟加速且质量损失有限。

**[Quantifying Climate Policy Action and Its Links to Development Outcomes: A Cross-National Data-Driven Analysis](quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-.md)**

:   本文构建了一个NLP-计量经济学一体化框架，先用微调的多语言DistilBERT对全球气候政策文档按主题（减缓/适应/灾害风险管理/损失与损害）自动分类（F1=0.90），再与世界银行发展指标做固定效应面板回归，发现减缓政策与较高GDP/GNI显著正相关，而损失与损害政策全球仍然缺乏实质性实施。

**[Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](reflective_translation_improving_low-resource_machine_translation_via_structured.md)**

:   提出 Reflective Translation 框架，让 LLM 在推理时对自身的初始翻译进行结构化自我批判（识别误译/遗漏/语义扭曲），再根据批判生成修正翻译，无需微调或额外标注数据即可在 isiZulu/isiXhosa 等低资源非洲语言上取得 BLEU 和 COMET 的统计显著提升。

**[XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)**

:   提出XIFBench——首个系统评估LLM多语言指令遵循能力的约束驱动基准，包含558条指令（0-5个约束，5大类21维度）×6种语言（高/中/低资源），并引入英语需求锚定评估协议，实现94.7%的跨语言评估一致性。

**[Zero-Shot Performance Prediction for Probabilistic Scaling Laws](zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)**

:   将 NLP 学习曲线预测建模为多任务学习问题，利用潜变量多输出高斯过程（MaGP）捕捉数据集中的双层层次结构和任务间相关性，实现学习曲线的零样本预测，并通过蒙特卡洛模拟推导概率化的 Scaling Laws。

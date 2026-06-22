---
title: >-
  ACL2026 文本生成论文汇总 · 17篇论文解读
description: >-
  17篇ACL2026的文本生成方向论文解读，涵盖文本摘要、LLM、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
  - "文本摘要"
  - "LLM"
  - "Agent"
item_list:
  - u: "adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl/"
    t: "Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search"
  - u: "are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_/"
    t: "Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering"
  - u: "can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl/"
    t: "Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style"
  - u: "childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact/"
    t: "Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety"
  - u: "conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline/"
    t: "ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline"
  - u: "difficulty-controllable_cloze_question_distractor_generation/"
    t: "Difficulty-Controllable Cloze Question Distractor Generation"
  - u: "edumath_generating_standards-aligned_educational_math_word_problems/"
    t: "EDUMATH: Generating Standards-aligned Educational Math Word Problems"
  - u: "facts_table_summarization_via_offline_template_generation_with_agentic_workflows/"
    t: "FACTS: Table Summarization via Offline Template Generation with Agentic Workflows"
  - u: "frankentext_stitching_random_text_fragments_into_long-form_narratives/"
    t: "Frankentext: Stitching Random Text Fragments into Long-Form Narratives"
  - u: "in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an/"
    t: "In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis"
  - u: "investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu/"
    t: "Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models"
  - u: "losses_that_cook_topological_optimal_transport_for_structured_recipe_generation/"
    t: "Losses that Cook: Topological Optimal Transport for Structured Recipe Generation"
  - u: "planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation/"
    t: "Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation"
  - u: "right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si/"
    t: "Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification"
  - u: "scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha/"
    t: "SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization"
  - u: "threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts/"
    t: "ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts"
  - u: "xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll/"
    t: "XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**💬 ACL2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (12)](../../ICLR2026/nlp_generation/index.md) · [🧪 ICML2026 (2)](../../ICML2026/nlp_generation/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md) · [💬 ACL2025 (27)](../../ACL2025/nlp_generation/index.md)

🔥 **高频主题：** 文本摘要 ×5 · LLM ×3 · Agent ×2

**[Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)**

:   本文提出 PACO，把"多属性可控摘要"重新表述为一个寻找"属性控制顺序"的规划问题，并用一个定制的 Monte Carlo Tree Search（节点是完整摘要、动作是单属性调整）在 prompt 阶段就找到最优调整路径，无需任何属性专用训练，用 Llama-3.2-1B 即可达到 Llama-3.3-70B baseline 的可控性，70B+PACO 全面超越所有现有方法。

**[Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)**

:   系统研究LLM中情感和修辞神经元的表征机制及其内在关联，提出结合多维筛选的神经元识别框架和自适应遮蔽验证方法，实现了情感/修辞预测的定向诱导和修辞神经元辅助情感识别。

**[Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)**

:   作者设计一项 81 人预注册在线研究，让被试用 GPT-o4-mini 起草+人工 post-edit 重写婚礼誓词、道歉信等"在意个人风格"的文本，发现 post-edit 确实能显著拉近被试自身风格、远离 LLM 风格，但被编辑后的文本仍系统性地比独立写作更"AI 味"——而被试自己却感知不到这种残留风格痕迹。

**[Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)**

:   作者用 UFLI K–2 英语阅读课程对应的 2,580 篇 GPT-4o / Llama-3.3-70B 生成故事，对三个 8B 模型（Llama 3 / Granite 3.3 / Apertus）做 4 种 SFT 设计（baseline / Good Stories / Rewarded SFT / 模拟儿童读音错误），证明 **小模型 + 合适 SFT 策略** 可在 Spache 可读性、句法复杂度、毒性等 K-2 关键指标上**超过 zero-shot GPT-4o 与 Llama-3.3-70B**，其中 Rewarded SFT 最稳定、几乎无幻觉。

**[ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)**

:   本文提出 ConlangCrafter，一个基于 LLM 的多跳管道，将构造语言（conlang）设计分解为音系、语法、词汇三个模块化阶段，通过随机性注入保证类型学多样性、通过自精炼循环保证内部一致性，并提出了一个包含类型学多样性分析和翻译一致性评估的自动评估框架。

**[Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)**

:   这篇论文提出 DCDG，通过双路干扰项数据增强、QA ensemble 难度聚类和多任务 seq2seq 训练，让完形填空干扰项生成模型可以按 easy/hard 控制难度，并在自动与人工评测中明显优于 GPT-4o。

**[EDUMATH: Generating Standards-aligned Educational Math Word Problems](edumath_generating_standards-aligned_educational_math_word_problems.md)**

:   作者把"按 K-12 数学课程标准生成应用题（MWP）"任务系统化，搜集了 11,000+ 由真实美国教师标注的 MWP 训练数据 STEM，用 SFT + KTO + ModernBERT 过滤训出 EDUMATH-12B/30B 两个开源 SOTA 生成器，并在 3-5 年级真实学生身上做了第一个 RCT，发现学生在 LLM 题与人写题上正确率相当但**几乎一致偏好定制 LLM 题**。

**[FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)**

:   本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板），实现快速、准确、隐私合规的查询聚焦表格摘要，在 FeTaQA、QTSumm 和 QFMTS 三个基准上全面超越基线。

**[Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)**

:   提出Frankentext范式，让LLM在极端约束下（90%文本逐字复制自人类写作）拼接随机人类文本片段为连贯长篇叙事，揭示现有AI文本检测器在混合作者场景下的严重失败（72%的Frankentext被误判为人类写作）。

**[In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)**

:   这篇论文提出“科研影响力摘要”任务：先从论文的引文上下文中识别真正揭示影响的细粒度意图，再生成随时间演化的影响力叙事，比单纯引用数更能说明一篇论文如何被后续工作采用、批评和改造。

**[Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models](investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu.md)**

:   论文在英日双语口语对话语料上用 MASK / NTP / TTP 三种微调任务训练 BERT / GPT-2 / TurnGPT / LLaMA-3 8B / Qwen-3 8B，再用 t-SNE 可视化和 silhouette 聚类量化"哼哈词"（backchannel 如 *uh-huh*）和"语气词"（filler 如 *um*）的表征质量；发现微调能让这些被视为"语义漂白"的功能词在嵌入空间里被显著区分开，并让模型在 NLG 生成时自然地说出多种 backchannel/filler，向"像人一样对话的 LM"迈出可量化的第一步。

**[Losses that Cook: Topological Optimal Transport for Structured Recipe Generation](losses_that_cook_topological_optimal_transport_for_structured_recipe_generation.md)**

:   提出一种基于 Sinkhorn 散度的拓扑损失函数，将食材列表表示为嵌入空间中的点云，最小化预测与真实食材之间的几何差异，显著提升结构化食谱生成中食材召回率和数量精度，在人类评估中 62% 的情况被偏好。

**[Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)**

:   本文提出 PLOTTER 框架，首次将叙事规划从文本表示转移到图结构表示（事件图+角色图），通过多 agent 的 Evaluate-Plan-Revise 迭代循环在图拓扑上诊断和修复叙事缺陷，在叙事性、角色塑造、戏剧张力等维度上显著优于现有方法。

**[Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)**

:   提出 Re-RIGHT 框架，通过三模块奖励（词汇覆盖率+语义保持+连贯性）的 GRPO 训练，用 4B 策略模型在英日韩中四种语言上实现按学习者熟练度等级（CEFR/JLPT/TOPIK/HSK）精确简化文本，超越 GPT-5.2 和 Gemini 2.5 等大模型。

**[SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)**

:   本文提出 SCURank，一种基于摘要内容单元（SCU）的排序框架，通过提取 SCU、跨摘要聚类估计信息重要性、按信息丰富度评分来排序候选摘要，替代不稳定的 LLM 直接排序和粗粒度的 ROUGE 排序，在多 LLM 蒸馏场景中配合 BRIO 对比学习显著提升了蒸馏模型的摘要性能。

**[ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)**

:   本文提出 ThreadSumm，一个多阶段 LLM 管道框架，将嵌套话语线程摘要建模为层次推理问题——先提取方面和原子内容单元进行内容规划，再通过句子排序构建线程感知序列，最后用 Tree of Thoughts 搜索生成和评分多个段落候选，在 Reddit/StackExchange 数据集上优于基线。

**[XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)**

:   本文提出 XtraGPT——首个面向学术论文修改的开源 LLM 套件（1.5B-14B），通过在 7,000 篇顶会论文和 140,000 个标准引导的指令-修改对上微调，实现上下文感知的段落级可控修改，7B 版本匹配 GPT-4o-mini，14B 版本超越 GPT-4o-mini，人类评估显示修改后论文预测评分平均提升 0.65 分。

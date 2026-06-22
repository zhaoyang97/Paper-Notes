---
title: >-
  ACL2025 文本生成论文汇总 · 27篇论文解读
description: >-
  27篇ACL2025的文本生成方向论文解读，涵盖文本摘要、LLM、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
  - "文本摘要"
  - "LLM"
  - "个性化生成"
item_list:
  - u: "a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors/"
    t: "A Representation Level Analysis of NMT Model Robustness to Grammatical Errors"
  - u: "abstractive_snippet_generation/"
    t: "Abstractive Snippet Generation"
  - u: "an_empirical_study_of_manytomany_summarization/"
    t: "An Empirical Study of Many-to-Many Summarization with Large Language Models"
  - u: "atgen_a_framework_for_active_text_generation/"
    t: "ATGen: A Framework for Active Text Generation"
  - u: "balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param/"
    t: "Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation"
  - u: "cocolex_legal_text_gen/"
    t: "CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation"
  - u: "context-aware_hierarchical_merging_for_long_document_summarization/"
    t: "Context-Aware Hierarchical Merging for Long Document Summarization"
  - u: "decomposed_opinion_summarization_with_verified_aspect-aware_modules/"
    t: "Decomposed Opinion Summarization with Verified Aspect-Aware Modules"
  - u: "dehumanizing_machines_anthropomorphic/"
    t: "Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems"
  - u: "doc_level_mbr_optimal_transport/"
    t: "Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport"
  - u: "dpp_diverse_multidoc_summary/"
    t: "Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries"
  - u: "dtcrs_dynamic_tree_construction_for_recursive_summarization/"
    t: "DTCRS: Dynamic Tree Construction for Recursive Summarization"
  - u: "enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study/"
    t: "Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study"
  - u: "event_graph_bias_mitigation_summarization/"
    t: "Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs"
  - u: "gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation/"
    t: "gec-metrics: A Unified Library for Grammatical Error Correction Evaluation"
  - u: "impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er/"
    t: "IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator"
  - u: "odysseus_dynamic_focus_decoding/"
    t: "Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation"
  - u: "personalized_text_generation_with_contrastive_activation_steering/"
    t: "Personalized Text Generation with Contrastive Activation Steering"
  - u: "persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_/"
    t: "PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization"
  - u: "rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe/"
    t: "Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?"
  - u: "tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta/"
    t: "TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks"
  - u: "tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera/"
    t: "Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes"
  - u: "theme-explanation_structure_for_table_summarization_using_large_language_models_/"
    t: "Theme-Explanation Structure for Table Summarization Using Large Language Models"
  - u: "towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework/"
    t: "Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework"
  - u: "unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_/"
    t: "Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing"
  - u: "video_text_summarization/"
    t: "What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations"
  - u: "writing_like_best_exemplar/"
    t: "Writing Like the Best: Exemplar-Based Expository Text Generation"
item_total: 27
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**💬 ACL2025** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (12)](../../ICLR2026/nlp_generation/index.md) · [💬 ACL2026 (17)](../../ACL2026/nlp_generation/index.md) · [🧪 ICML2026 (2)](../../ICML2026/nlp_generation/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md)

🔥 **高频主题：** 文本摘要 ×8 · LLM ×4 · 个性化生成 ×2

**[A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)**

:   从表示层面系统分析 NMT 编码器如何处理语法错误——发现编码器先在浅层"检测"错误（GED 探测 F1 上升），再在深层"纠正"错误（CKA 距离下降），并提出 Robustness Heads 概念识别出参与纠正的具体注意力头，在 4 个模型×5 个语言方向上验证了该"检测→纠正"两阶段机制。

**[Abstractive Snippet Generation](abstractive_snippet_generation.md)**

:   本文提出了一种面向搜索引擎的抽象化片段生成方法，通过查询感知的摘要生成技术，为搜索结果页面生成比传统抽取式片段更简洁、信息量更大的文本摘要，显著提升用户搜索体验。

**[An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)**

:   首次系统研究LLM在多对多摘要（M2MS）任务上的表现，整合8个数据集构建涵盖5个领域6种语言的47.8K样本基准，评测18个LLM发现零样本LLM可媲美微调传统模型，指令微调后显著超越，但事实性问题仍是关键瓶颈。

**[ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)**

:   提出ATGen——首个系统化的NLG主动学习框架，集成SOTA AL策略、人工/LLM标注界面、PEFT高效训练和vLLM推理优化，在TriviaQA/GSM8K等4个NLG任务上验证主动学习可将标注成本降低2-4倍。

**[Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)**

:   本文提出了一种基于上下文保持前缀树（CP-Trie）的系统性评估框架，通过不依赖概率和参数调优的指标来评估截断采样方法在多样性与风险之间的内在适应能力，并为实际应用中的参数选择提供指导。

**[CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)**

:   提出 CoCoLex，一种无需训练的解码策略，利用解码过程中隐状态与上下文 token 隐状态的欧氏距离构造复制分布，并通过基于预测熵的置信度分数动态平衡"从上下文复制"与"自由生成"的比例，在五个法律基准上一致提升忠实性和正确性，尤其在长文本生成任务中效果突出。

**[Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)**

:   提出上下文感知的层次合并（CAHM）方法，通过在层次合并摘要过程中引入源文档的相关上下文（抽取/检索/引用三种方式），有效缓解 LLM 在超长文档（>100K tokens）摘要中的幻觉问题。

**[Decomposed Opinion Summarization with Verified Aspect-Aware Modules](decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)**

:   本文将观点摘要（opinion summarization）任务分解为三个可逐步验证的模块——方面识别、观点汇总、元评论合成，通过 LLM 零样本提示实现领域无关的模块化处理，在科研论文、商业评论和产品评论三个领域生成了更可追溯、更全面的摘要。

**[Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)**

:   通过文献综述和众包研究，系统整理出 21 类干预措施来降低文本生成系统输出的拟人化程度，提出包含干预类型、目标行为、操作化方式和负面影响四个维度的概念框架，为去拟人化研究提供最全面的基础设施。

**[Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)**

:   提出 MBR-OT，将最优传输（Wasserstein距离）引入最小贝叶斯风险（MBR）解码，实现用句子级效用函数评估文档级输出质量，在文档级机器翻译、文本简化和密集图像描述任务上显著优于标准 MBR 解码。

**[Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)**

:   提出将多文档摘要解耦为关键点抽取→DPP多样性选择→重写三步流水线，通过行列式点过程（DPP）进行原则性内容选择，显著提升LLM多文档摘要的源文档覆盖率。

**[DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)**

:   提出 DTCRS 方法，根据文档结构和查询语义动态构建摘要树，通过问题分解和子问题引导聚类减少冗余摘要节点，在三个 QA 数据集上显著优于静态摘要树方法 RAPTOR。

**[Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)**

:   本文提出一种无需语言特定编辑集的通用文本编辑方法（SWEET），通过数据驱动的编辑标签自动提取和压缩策略，首次成功将文本编辑范式应用于阿拉伯语语法纠错，在多个基准上达到SOTA且推理速度提升6倍以上。

**[Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)**

:   构建多文档事件关系图（包含四类文档内事件关系、跨文档事件共指、事件级道德观点），通过图文本化和图提示微调两种策略将偏见信息注入 LLM，生成去偏见的中立化摘要，在内容保留和偏见消除上均优于基线。

**[gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)**

:   提出 gec-metrics 统一库，将 10 种语法纠错 (GEC) 评估指标整合到统一接口中，并提供元评估功能，解决了现有 GEC 评估实现碎片化、不可复现、难以扩展的问题。

**[IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)**

:   在 IMPARA 的质量估计器构建之前，增加一步语法错误检测（GED）预训练，同时去掉失效的相似度估计器，使无参考 GEC 评估在 SEEDA 上达到句子级最高相关性。

**[Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)**

:   提出动态聚焦解码（DFD），通过追踪 LLM 各层间分布差异（KL 散度）来识别知识密集型解码步骤，自适应调整温度——知识密集步用低温保事实性，非知识密集步用高温促多样性——在七个数据集上同时提升事实性和多样性。

**[Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)**

:   提出 StyleVector——一个无需训练的个性化文本生成框架，通过对比用户真实响应与模型生成的无风格响应之间的隐层激活差异来提取"风格向量"，在推理时通过简单的线性激活干预引导 LLM 生成符合用户写作风格的文本，在 LaMP 和 LongLaMP 基准上实现 8% 的相对提升，同时将存储需求降低至 PEFT 方法的 1/1700。

**[PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)**

:   > 提出 PerSphere 基准数据集和 MURS（Multi-faceted perspective retrieval and summarization）任务，旨在从文档集中检索并全面总结争议性问题的多面向观点，并提出分层多智能体总结系统 HierSphere 来缓解长上下文和观点提取的挑战。

**[Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)**

:   本文指出当前 GEC 自动评估与人工评估在"从句级分数到系统排名"的聚合流程上存在根本差异——人工评估用句级两两比较+TrueSkill 排名算法，而自动评估用平均绝对分数+排序——并通过在自动评估中也采用 TrueSkill 聚合来弥补这一差距，在 SEEDA 基准上大幅提升多数指标与人工评估的相关性，甚至使 BERT 级指标超越 GPT-4。

**[TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)**

:   这篇论文提出 TagRouter，用一个小型标签生成器把开放域文本生成请求先压缩成一组语义标签，再基于标签统计每个候选 LLM 的相对优势并进行路由，从而在不重新训练路由器的前提下，把多模型系统的接受率做得比单个大模型更高，同时显著降低推理成本。

**[Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes](tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)**

:   提出 Retell 方法：利用小型 LM 对文学段落进行抽象复述（abstractive retelling），

**[Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)**

:   提出 Tabular-TX 管线，通过多步 CoT 推理实现深度表格理解、记者角色 prompt 生成清晰句子、并将输出结构化为 Theme（主题状语）+ Explanation（解释谓语）的格式，在韩语行政表格摘要基准上不依赖微调即实现 ROUGE-1 0.51 的最佳性能，显著超越微调和纯 ICL 方法。

**[Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)**

:   针对开放式文本生成中多指标（coherence/diversity/perplexity）之间的权衡问题，提出三种互补的多准则评估方法——Extended Bradley-Terry 模型（序数排名）、Union-Free Generic Depth（允许不可比性的偏序）和 Q*Text（基数评估综合指标），在6个 LLM × 59种解码策略 × 180万+生成文本上验证，发现中等超参配置普遍优于极端配置，小模型+合理解码策略可匹敌大模型。

**[Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)**

:   本文从动力系统理论出发，发现LLM在连续释义（successive paraphrasing）过程中输出会收敛至稳定的2-周期吸引子循环，而非探索广阔的释义空间，揭示了LLM生成能力的固有局限性。

**[What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations](video_text_summarization.md)**

:   提出VISTA数据集——18,599个AI会议演讲视频与论文摘要配对，并引入plan-based摘要框架，通过生成中间问题序列引导科学视频的结构化摘要生成，显著提升事实一致性。

**[Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)**

:   定义"基于范例的说明文生成"新任务——给定一篇关于源主题的范例文本，生成关于目标主题的说明文，提出 Recurrent Plan-then-Adapt（RePA）框架，通过逐段模仿规划+检索增强自适应生成+双记忆机制，在 Wikipedia/RoleEE/USNews 三个数据集上显著优于 GPT-4 和 o1 基线。

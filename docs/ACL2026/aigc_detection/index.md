---
title: >-
  ACL2026 AIGC检测论文汇总 · 17篇论文解读
description: >-
  17篇ACL2026的 AIGC 检测方向论文解读，涵盖 LLM、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "个性化生成"
item_list:
  - u: "aegis_a_holistic_benchmark_for_evaluating_forensic_analysis_of_ai-generated_acad/"
    t: "AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images"
  - u: "authorship_attribution_in_multilingual_machine-generated_texts/"
    t: "Authorship Attribution in Multilingual Machine-Generated Texts"
  - u: "beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr/"
    t: "Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection"
  - u: "biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu/"
    t: "BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories"
  - u: "c-red_a_comprehensive_chinese_benchmark_for_ai-generated_text_detection_derived_/"
    t: "C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts"
  - u: "can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l/"
    t: "Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences"
  - u: "detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec/"
    t: "DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection"
  - u: "exagpt_example-based_machine-generated_text_detection_for_human_interpretability/"
    t: "ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability"
  - u: "frame_in_frame_out_measuring_framing_bias_in_llm-generated_news_summaries/"
    t: "Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries"
  - u: "from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based/"
    t: "From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment"
  - u: "gigacheck_detecting_llm-generated_content_via_object-centric_span_localization/"
    t: "GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization"
  - u: "mash_evading_black-box_ai-generated_text_detectors_via_style_humanization/"
    t: "MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization"
  - u: "mdok-style_at_semeval-2026_task_10_finetuning_llms_for_conspiracy_detection/"
    t: "mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection"
  - u: "reflex_self-refining_explainable_fact-checking_via_verdict-anchored_style_contro/"
    t: "REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control"
  - u: "temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec/"
    t: "Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories"
  - u: "when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene/"
    t: "When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection"
  - u: "who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_/"
    t: "Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**💬 ACL2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

🔥 **高频主题：** LLM ×8

**[AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images](aegis_a_holistic_benchmark_for_evaluating_forensic_analysis_of_ai-generated_acad.md)**

:   AEGIS 是首个面向学术图像伪造取证的综合基准，覆盖 7 大学术图类与 39 子类、4 种伪造策略（整图捏造、参考图改写、局部修复、局部编辑）和 25 个生成模型，提出取证范围判别、文字伪影识别、操作类型分类、篡改像素定位四项任务，对 25 个 MLLM 与 9 个专家模型联评后发现：即使 GPT-5.1 综合分仅 48.80%，专家模型像素 IoU 仅 30.09%，凸显「生成进化快于取证」与「MLLM 推理 vs 专家模型敏感度」的结构性互补。

**[Authorship Attribution in Multilingual Machine-Generated Texts](authorship_attribution_in_multilingual_machine-generated_texts.md)**

:   现有「机器生成文本作者归属（attribution，即判断一段文本出自哪个具体 LLM 还是人类）」研究几乎全是单语种（尤其英语）的，这篇论文首次形式化定义了**多语言作者归属（ML-MGT）**和**跨语言迁移（CL-MGT）**两个问题，在 18 种语言 × 8 个生成者（7 个 LLM + 人类）上系统评测了统计法、微调编码器、对比学习、微调解码器等一整套现有方法，发现微调/对比方法能适配多语言（最佳 macro-F1 > 0.9），但**跨不同语系/书写体系迁移时严重退化**，揭示了真实多语言场景的难度。

**[Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)**

:   提出 RACE（Rhetorical Analysis for Creator-Editor Modeling），利用修辞结构理论(RST)构建逻辑图来建模文本"创作者"的思维架构，同时提取篇章单元级特征捕获"编辑者"的语言风格，实现四类细粒度 LLM 生成文本检测（人写/LLM写/LLM润色人文/人改写LLM文）。

**[BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)**

:   BiasedTales-ML 构建了约 35 万篇覆盖 8 种语言的 LLM 生成儿童故事语料库，通过全排列提示设计和分布分析框架，揭示了**叙事中社会属性分布在不同语言间存在显著差异**，英语中心的评估无法反映多语言场景下的偏见模式。

**[C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts](c-red_a_comprehensive_chinese_benchmark_for_ai-generated_text_detection_derived_.md)**

:   C-ReD 构建了一个覆盖五类中文写作场景、九个 LLM 生成器和真实使用式 prompt 的中文 AI 生成文本检测基准，并显示检测难度强烈依赖领域、生成器和 prompt，而在 C-ReD 上微调能显著提升对未见模型和外部中文数据的泛化。

**[Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)**

:   本文引入 Persuaficial——一个覆盖六种语言的高质量 AI 生成说服性文本多语言基准，系统评估了 LLM 生成的说服性文本与人类撰写的说服性文本在自动检测难度上的差异，发现微妙的 AI 说服比人类说服更难检测（F1 下降约 20%），而过度强化的说服反而更容易被发现。

**[DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)**

:   DetectRL-X 构建了一个 345.6 万样本、多语言、多域、多攻击、多长度和二/三分类并行的 LLM 生成文本检测基准，证明现有检测器在真实多语言和人机协作写作场景下仍存在明显鲁棒性短板。

**[ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](exagpt_example-based_machine-generated_text_detection_for_human_interpretability.md)**

:   ExaGPT 把"判定一段文本是人写还是 LLM 生成"这件事重构成"在数据存储里找哪一侧的相似 span 更多"，通过 BERT 嵌入 + k-NN 检索 + 动态规划做最优 span 切分，既给出可解释证据（最相似的检索 span 例子）又在 1% FPR 下把准确率刷到比此前可解释检测器最高高出 +37.0 个点。

**[Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries](frame_in_frame_out_measuring_framing_bias_in_llm-generated_news_summaries.md)**

:   本文提出 FIFO，用 LLM jury 加专家校准的方式在 XSum 上大规模测量 LLM 新闻摘要是否引入 framing bias，并发现若干高容量模型的框架化表达比例高于人工摘要基线。

**[From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment](from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based.md)**

:   这篇论文提出一个面向 rubric 自动评分的句子级解释评估框架，在课堂教学反馈质量评分任务上比较微调 PLM、prompted LLM、SHAP 归因和 LLM rationale，发现 fine-tuned PLM 更准，而 SHAP 比 LLM 生成的解释更忠实、更可迁移。

**[GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)**

:   提出 GigaCheck，一个双策略框架：文档级使用微调 LLM 进行分类，片段级创新地将 AI 生成文本片段视为"目标"，用 DETR-like 架构实现端到端的字符级定位。

**[MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)**

:   本文提出 MASH（多阶段风格人性化对齐），通过风格注入 SFT → DPO 对齐 → 推理时精炼三阶段流水线，训练一个仅 0.1B 参数的改写器，在黑盒设置下以 92% 的平均攻击成功率规避 AI 文本检测器，同时保持优秀的语言质量。

**[mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection](mdok-style_at_semeval-2026_task_10_finetuning_llms_for_conspiracy_detection.md)**

:   作者把自己在 PAN@CLEF2025 拿冠军的 mdok（机器生成文本检测器）的 finetuning 范式平移到阴谋论检测：用四种数据增强（匿名化 / 大小写 / 同形字 / 去重）扩训练集，再做一轮自训练（只保留 ≥0.99 或 ≤0.01 的高置信伪标签），用 QLoRA 4-bit PEFT 微调 Qwen3-32B，最终在 SemEval-2026 Task 10 subtask 2 拿 Macro F1 = 0.78，排名 8/52（85 百分位）。

**[REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control](reflex_self-refining_explainable_fact-checking_via_verdict-anchored_style_contro.md)**

:   REFLEX 将事实核查中的判决预测和解释生成绑定起来，用 backbone 与微调模型之间的自分歧样本构造内部 steering 向量，在不依赖搜索 API 或闭源教师模型的情况下提升判决 Macro-F1，并让解释更短、更一致、更少误导。

**[Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)**

:   本文通过构建跨12年的纵向写作数据集，发现LLM生成文本存在"时间扁平化"现象——虽然词汇多样性高，但在语义和认知情感维度上的时间漂移显著低于人类，仅凭时间变异模式就能以94%准确率区分人类与LLM文本。

**[When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)**

:   揭示了个性化场景下 MGT 检测器的"特征反转陷阱"——通用域中区分人写文本和机器文本的特征在个性化域中发生反转，导致检测器性能骤降甚至翻转，并提出 StyloCheck 框架通过量化检测器对反转特征的依赖程度来预测跨域性能变化，预测相关性达 0.85 以上。

**[Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)**

:   本文构建了首个面向LLM生成古典中文诗词的检测基准ChangAn（含30,664首诗），系统评估了12种AI检测方法在不同文本粒度和生成策略下的表现，揭示了当前中文文本检测器在古典诗词领域的严重局限性。

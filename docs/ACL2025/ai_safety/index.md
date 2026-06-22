---
title: >-
  ACL2025 AI安全论文汇总 · 14篇论文解读
description: >-
  14篇ACL2025的 AI 安全方向论文解读，涵盖水印/隐写、对抗鲁棒、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "水印/隐写"
  - "对抗鲁棒"
  - "语音"
item_list:
  - u: "building_a_long_text_privacy_policy_corpus_with_multi-class_labels/"
    t: "Building a Long Text Privacy Policy Corpus with Multi-Class Labels"
  - u: "centaur_bridging_the_impossible_trinity_of/"
    t: "CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference"
  - u: "crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf/"
    t: "Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference"
  - u: "fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s/"
    t: "FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes"
  - u: "gifi_gender_fairness/"
    t: "Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework"
  - u: "multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries/"
    t: "Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries"
  - u: "privacibench_evaluating_privacy_with_contextual_integrity/"
    t: "PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance"
  - u: "quantifying_misattribution_unfairness_in_authorship_attribution/"
    t: "Quantifying Misattribution Unfairness in Authorship Attribution"
  - u: "robust_and_minimally_invasive_watermarking_for_eaas/"
    t: "Robust and Minimally Invasive Watermarking for EaaS"
  - u: "sandcastles_watermarking_impossibility/"
    t: "Sandcastles in the Storm: Revisiting Watermarking Impossibility"
  - u: "speechfake_a_largescale_multilingual_speech_deepfake/"
    t: "SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods"
  - u: "towards_fairness_assessment_of_dutch_hate_speech_detection/"
    t: "Towards Fairness Assessment of Dutch Hate Speech Detection"
  - u: "watermark_segment_detection/"
    t: "Efficiently Identifying Watermarked Segments in Mixed-Source Texts"
  - u: "wet_eaas_watermark/"
    t: "WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**💬 ACL2025** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (145)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/ai_safety/index.md) · [💬 ACL2026 (5)](../../ACL2026/ai_safety/index.md) · [🧪 ICML2026 (114)](../../ICML2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md)

🔥 **高频主题：** 水印/隐写 ×4 · 对抗鲁棒 ×3 · 语音 ×2

**[Building a Long Text Privacy Policy Corpus with Multi-Class Labels](building_a_long_text_privacy_policy_corpus_with_multi-class_labels.md)**

:   本文构建了一个包含149家公司隐私政策的多维度标注语料库（64个标注维度），涵盖欧盟和美国隐私法规中的争议条款和法律规则，并使用当前大语言模型建立了分类基准。

**[CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)**

:   本文提出一种通过构造隐私保护型对抗样本来防御成员推理攻击（MIA）的方法，在模型预测输出中注入精心设计的扰动，使攻击者无法判断某条数据是否属于训练集，同时保持模型对正常用户的服务质量。

**[FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)**

:   本文提出 Indic-Bias，首个面向印度多元社会的大规模 LLM 公平性基准，通过 20,000 个人工验证的场景模板在三大评估任务上测试 14 个 LLM，揭示模型对达利特等边缘化群体存在严重负面偏见，且超过 70% 的情况下会强化刻板印象。

**[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和数学推理一致性七个维度的多层次评估框架，在 22 个主流 LLM 上系统量化二元与非二元性别的公平性，揭示新代词在无提示时完全缺席、"she" 过度矫正等深层偏见模式。

**[Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)**

:   提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。

**[PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Quantifying Misattribution Unfairness in Authorship Attribution](quantifying_misattribution_unfairness_in_authorship_attribution.md)**

:   本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。

**[Robust and Minimally Invasive Watermarking for EaaS](robust_and_minimally_invasive_watermarking_for_eaas.md)**

:   提出 ESpeW（Embedding-Specific Watermark），一种嵌入特异性水印方法，通过在每个嵌入向量的不同位置注入独特水印，实现对 Embeddings as a Service (EaaS) 的鲁棒版权保护，抵抗各种水印移除攻击且对嵌入质量的影响小于 1%。

**[Sandcastles in the Storm: Revisiting Watermarking Impossibility](sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake 大规模语音深伪数据集，包含 300 万+深伪样本、3000+ 小时音频、40 种生成工具和 46 种语言，并通过基线实验系统分析了生成方法、语言多样性和说话人变化对检测性能的影响。

**[Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)**

:   本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。

**[Efficiently Identifying Watermarked Segments in Mixed-Source Texts](watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark](wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。

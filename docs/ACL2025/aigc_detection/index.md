---
title: >-
  ACL2025 AIGC检测论文汇总 · 15篇论文解读
description: >-
  15篇ACL2025的 AIGC 检测方向论文解读，涵盖 LLM、对抗鲁棒、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "少样本学习"
item_list:
  - u: "a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e/"
    t: "A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI"
  - u: "aigt_social_media_monitoring/"
    t: "Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media"
  - u: "an_empirical_study_on_detecting_ai-generated_text_in_financial_reports/"
    t: "An Empirical Study on Detecting AI-Generated Text in Financial Reports"
  - u: "chatgpt_user_ai_text_detection/"
    t: "People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text"
  - u: "chemactor_enhancing_automated_extraction_of_chemical_synthesis_actions_with_llm-/"
    t: "ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data"
  - u: "cognitive_framework_for_detecting_ai-generated_fiction/"
    t: "Cognitive Framework for Detecting AI-Generated Fiction"
  - u: "greater_adversarial_mgt_detection/"
    t: "Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training"
  - u: "haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum/"
    t: "HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring"
  - u: "katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys/"
    t: "KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis"
  - u: "learning_to_rewrite_generalized_llm-generated_text_detection/"
    t: "Learning to Rewrite: Generalized LLM-Generated Text Detection"
  - u: "llm_vs_human_formal_syntax/"
    t: "Comparing LLM-generated and human-authored news text using formal syntactic theory"
  - u: "low-perplexity_llm-generated_sequences_and_where_to_find_them/"
    t: "Low-Perplexity LLM-Generated Sequences and Where To Find Them"
  - u: "multisocial_mgt_detection/"
    t: "MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts"
  - u: "reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f/"
    t: "Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction"
  - u: "who_writes_what_ai_detection/"
    t: "Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**💬 ACL2025** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md)

🔥 **高频主题：** LLM ×6 · 对抗鲁棒 ×2

**[A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)**

:   提出用 LLM 生成的 NLI 解释替代昂贵的人工解释来近似人工判断分布（HJD），实验表明在提供人工标签引导的条件下，LLM 生成的解释与人工解释在 KL 散度、JSD 等指标上效果相当，并可推广到无人工解释的数据集（MNLI）和域外测试集（ANLI）。

**[Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigt_social_media_monitoring.md)**

:   首次大规模量化社交媒体上 AI 生成文本(AIGT)的占比变化——收集 Medium/Quora/Reddit 上 240 万帖子，构建 AIGTBench 训练最佳检测器 OSM-Det，发现 2022-2024 年间 Medium 和 Quora 的 AIGT 占比从~2% 飙升至~37-39%，而 Reddit 仅从 1.3% 增至 2.5%。

**[An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)**

:   本文针对金融报告这一高监管领域，系统评估了多种AI生成文本检测方法（统计特征、神经网络分类器、水印检测等）在识别金融文档中AI生成内容方面的表现，揭示了领域特异性对检测效果的显著影响。

**[People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text](chatgpt_user_ai_text_detection.md)**

:   通过 1,740 条标注实验发现，经常使用 LLM 进行写作任务的人类标注者可以极高精度（5人投票仅错 1/300）检测 AI 生成文本，即使面对改写和人性化逃逸策略也显著优于大多数自动检测器。

**[ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data](chemactor_enhancing_automated_extraction_of_chemical_synthesis_actions_with_llm-.md)**

:   本文提出 ChemActor，一个经过完全微调的 LLM 化学执行器，通过序列化 LLM 生成数据框架和分布散度数据筛选模块来解决化学合成动作提取中的数据稀缺问题，在 R2D 和 D2A 任务上超越基线模型 10%。

**[Cognitive Framework for Detecting AI-Generated Fiction](cognitive_framework_for_detecting_ai-generated_fiction.md)**

:   本文提出一种基于认知语言学特征的AI生成小说/虚构文本检测框架，通过建模人类创意写作中的认知模式（如叙事节奏、情感弧线、隐喻密度）来区分人类和AI创作的虚构文本，在长文本场景下显著优于现有检测方法。

**[Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](greater_adversarial_mgt_detection.md)**

:   提出 GREATER 对抗训练框架，同步训练对抗攻击器（Greater-A）和 MGT 检测器（Greater-D），对抗器通过代理模型梯度识别关键 token 并在嵌入空间扰动生成对抗样本，检测器从课程式对抗样本中学习泛化防御，在 16 种攻击下 ASR 降至 5.53%（SOTA 为 6.20%），攻击效率比 SOTA 快 4 倍。

**[HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)**

:   提出面向人机协作写作场景的细粒度机器生成文本（MGT）检测基准 HACo-Det，通过多轮局部改写流水线自动构建带词级归属标注的 11,200 篇人机共创文本，并将七种主流检测器改造为词级序列标注模式进行系统评估，揭示当前方法在细粒度检测上的巨大改进空间。

**[KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)**

:   本文构建了首个韩语 LLM 生成文本检测基准 KatFish（涵盖三种文体、四种 LLM），通过分析词间距、词性多样性和逗号使用三类韩语语言学特征，提出 KatFishNet 检测方法，在 OOD（未见过的 LLM）设置下平均 AUROC 比最佳基线高 19.78%。

**[Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)**

:   提出Learning2Rewrite（L2R）框架，通过微调LLM的改写模型来放大人写文本和AI生成文本在改写编辑距离上的差异，从而实现跨领域高度泛化的AI文本检测——在21个独立领域上平均AUROC达0.9009，域外测试超越RAIDAR达4.67%、超越直接分类微调达51.35%。

**[Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)**

:   首次使用 **HPSG 形式句法理论**（通过英语资源语法 ERG）从句法构式（298 种）、词汇类型（1398 种）和词法规则（100 种）三个层级系统比较 6 个 LLM 与人类 NYT 新闻写作的语法差异，发现 LLM 在语法特征上是人类作者的 **"均值化"投影**——人类个体作者间的语法差异反而大于任何人类与 LLM 的差异，而 LLM 之间几乎无差别。

**[Low-Perplexity LLM-Generated Sequences and Where To Find Them](low-perplexity_llm-generated_sequences_and_where_to_find_them.md)**

:   提出系统化 pipeline 分析 LLM 生成的低困惑度序列（token 预测概率 ≥0.9）并追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，将可匹配片段分为四种记忆行为类别。

**[MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](multisocial_mgt_detection.md)**

:   构建了首个覆盖 22 种语言、5 个社交媒体平台、7 个 LLM 生成器的大规模机器生成文本检测基准 MultiSocial（47.2 万文本），实验表明 fine-tuned 检测器（Llama-3-8B/Mistral-7B, AUC ROC 0.977）在社交媒体文本上表现优异，且训练平台选择对跨平台泛化影响显著。

**[Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)**

:   提出基于多尺度保形预测（MCP）的零样本机器生成文本检测框架，通过文本长度感知的分组分位数计算，在严格约束假阳性率（FPR）上界的同时显著提升检测性能，并构建了覆盖15个领域、22个LLM的大规模双语基准数据集RealDet。

**[Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)**

:   揭示作者的社会语言学属性（性别、CEFR水平、学科领域、语言环境）会系统性地影响AI生成文本检测器的准确率，其中语言水平和语言环境的偏差最为显著且一致，提出了基于多因素WLS+ANOVA的偏差量化框架。

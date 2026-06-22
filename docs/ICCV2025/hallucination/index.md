---
title: >-
  ICCV2025 幻觉检测论文汇总 · 5篇论文解读
description: >-
  5篇ICCV2025的幻觉检测方向论文解读，涵盖多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
item_list:
  - u: "chartcap_mitigating_hallucination_of_dense_chart_captioning/"
    t: "ChartCap: Mitigating Hallucination of Dense Chart Captioning"
  - u: "dash_detection_and_assessment_of_systematic_hallucinations_of_vlms/"
    t: "DASH: Detection and Assessment of Systematic Hallucinations of VLMs"
  - u: "mitigating_object_hallucinations_via_sentence-level_early_intervention/"
    t: "Mitigating Object Hallucinations via Sentence-Level Early Intervention"
  - u: "only_onelayer_intervention_sufficiently_mitigates_hallucinat/"
    t: "ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models"
  - u: "why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte/"
    t: "Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**📹 ICCV2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md)

**[ChartCap: Mitigating Hallucination of Dense Chart Captioning](chartcap_mitigating_hallucination_of_dense_chart_captioning.md)**

:   构建了包含56.5万张真实图表-描述对的大规模数据集ChartCap，通过类型特定的描述模式排除无关信息、强调结构与关键洞察，并提出无参考的Visual Consistency Score评估指标，有效减少VLM在图表描述中的幻觉问题。

**[DASH: Detection and Assessment of Systematic Hallucinations of VLMs](dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)**

:   提出DASH自动化流水线，通过LLM生成文本查询（DASH-LLM）和扩散模型优化图像查询（DASH-OPT）两种策略，在ReLAION-5B中系统性地发现VLM的假阳性对象幻觉聚类，共发现19k+聚类和950k+图像，并构建了更具挑战性的DASH-B基准。

**[Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)**

:   本文提出SENTINEL框架，基于"幻觉在生成早期出现并向后传播"的关键观察，通过域内候选引导、双检测器交叉验证构建句子级偏好数据，使用上下文感知DPO（C-DPO）实现早期干预，在Object HalBench上减少92%幻觉且保持通用能力。

**[ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)**

:   提出ONLY，一种training-free的单层干预解码方法——通过Text-to-Visual Entropy Ratio（TVER）选择偏向文本的attention head生成textually-enhanced logits，然后与原始logits做自适应对比/协作解码，仅增加1.07×推理时间就在POPE上比VCD/M3ID高3.14%，在CHAIR上降低CHAIR_S 6.2个点。

**[Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context](why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte.md)**

:   深入探究 LVLM 长文本生成中幻觉频发的根本原因——不是长度本身，而是上下文的连贯性（coherence）和完备性（completeness）需求驱动模型外推产生幻觉，并据此提出 HalTrapper 的"诱导-检测-抑制"三阶段框架。

---
title: >-
  AAAI2026 社会计算论文汇总 · 10篇论文解读
description: >-
  10篇AAAI2026的社会计算方向论文解读，涵盖推理、LLM、情感分析、对抗鲁棒、个性化生成、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "情感分析"
  - "对抗鲁棒"
  - "个性化生成"
  - "多模态"
item_list:
  - u: "argumentative_debates_for_transparent_bias_detection_technic/"
    t: "Argumentative Debates for Transparent Bias Detection"
  - u: "bias_association_discovery_framework_for_open-ended_llm_generations/"
    t: "Bias Association Discovery Framework for Open-Ended LLM Generations"
  - u: "cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition/"
    t: "Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition"
  - u: "fact2fiction_targeted_poisoning_attack_to_agentic_fact-check/"
    t: "Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System"
  - u: "factguard_event-centric_and_commonsense-guided_fake_news_detection/"
    t: "FactGuard: Event-Centric and Commonsense-Guided Fake News Detection"
  - u: "from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha/"
    t: "From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks"
  - u: "multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering/"
    t: "Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering"
  - u: "reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in/"
    t: "Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference"
  - u: "scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval/"
    t: "SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation"
  - u: "t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca/"
    t: "T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🤖 AAAI2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

🔥 **高频主题：** 推理 ×2

**[Argumentative Debates for Transparent Bias Detection](argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出 ABIDE（Argumentative BIas Detection by DEbate），通过基于邻域属性的论证方案（argument schemes）构建量化双极论证框架（QBAF），将偏见检测过程建模为结构化辩论，实现从单邻域到全局的透明偏见推理，并形式化证明 QBAF 语义与偏见检测期望行为之间的对应关系。

**[Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)**

:   提出偏见关联发现框架 BADF，通过分析 LLM 开放式故事生成中的叙事内容，系统性地提取人口统计身份与描述性概念之间的已知和未知偏见关联，突破了以往依赖预定义偏见概念的局限。

**[Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)**

:   提出 Cross-modal Prompting (ComP) 方法，通过渐进式提示生成+跨模态知识传播+动态调度器来解决不完整多模态情感识别中的模态不平衡问题，在 4 个数据集、 7 种缺失率下均达到 SOTA。

**[Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](factguard_event-centric_and_commonsense-guided_fake_news_detection.md)**

:   提出 FactGuard 框架，利用 LLM 提取事件核心内容（去风格化）并生成常识推理，通过 Rationale Usability Evaluator 动态评估 LLM 建议的可信度，并通过知识蒸馏获得无需 LLM 的轻量版 FactGuard-D，在假新闻检测中兼顾鲁棒性和效率。

**[From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)**

:   提出 CAPO（Curriculum Advantage Policy Optimization），一种基于优势信号的自适应课程机制，通过先模仿（仅正向优势样本）再判别（引入负向信号）的两阶段策略，稳定且显著提升 LLM 在数学推理和多模态 GUI 推理任务上的表现。

**[Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)**

:   本文提出Multi-DProxy框架，通过门控跨模态融合、双约束代理优化和动态候选词管理三大创新机制，利用可学习的文本代理实现个性化多重聚类，在全部公开基准上达到SOTA。

**[Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)**

:   提出OmiGraph，首个基于"遗漏感知"的虚假信息检测框架，通过构建遗漏感知图、利用LLM推理遗漏意图、以及遗漏导向的消息传递与聚合机制，从"未说出的内容"中提取欺骗模式，在双语数据集上平均提升+5.4% F1和+5.3% ACC。

**[SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)**

:   提出 T2Agent，一个集成可扩展工具集与蒙特卡洛树搜索（MCTS）的虚假信息检测智能体，通过多源验证机制将检测任务分解为针对不同伪造源的子任务，在 MMfakebench 上以 GPT-4o 为骨干将基线 MMDAgent 的准确率提升 28.7%，达到新 SOTA。

---
title: >-
  ACL2025 因果推理论文汇总 · 10篇论文解读
description: >-
  10篇ACL2025的因果推理方向论文解读，涵盖推理、LLM、RAG、情感分析等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "RAG"
  - "情感分析"
item_list:
  - u: "causal_graph_based_event_reasoning_using_semantic_relation_experts/"
    t: "Causal Graph based Event Reasoning using Semantic Relation Experts"
  - u: "causalrag_integrating_causal_graphs_into_retrieval-augmented_generation/"
    t: "CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation"
  - u: "coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l/"
    t: "CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs"
  - u: "counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg/"
    t: "Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models"
  - u: "counterfactual_explanations_for_aspect-based_sentiment_analysis/"
    t: "Counterfactual Explanations for Aspect-Based Sentiment Analysis"
  - u: "fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example/"
    t: "FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation"
  - u: "iris_an_iterative_and_integrated_framework/"
    t: "IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery"
  - u: "leveraging_variation_theory_in_counterfactual_data_augmentation_for_optimized_ac/"
    t: "Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning"
  - u: "llm_causal_discovery_reliability/"
    t: "On the Reliability of Large Language Models for Causal Discovery"
  - u: "reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi/"
    t: "Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**💬 ACL2025** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md)

🔥 **高频主题：** 推理 ×3 · LLM ×2

**[Causal Graph based Event Reasoning using Semantic Relation Experts](causal_graph_based_event_reasoning_using_semantic_relation_experts.md)**

:   提出基于四类语义关系专家（时间、篇章、条件、常识）多轮协作讨论的因果事件图生成框架，在零样本设置下于事件预测、事件预报等多个下游任务上取得与微调模型竞争的结果，并提供可解释的因果事件链。

**[CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)**

:   提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

**[CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)**

:   本文提出CoA-Reasoning框架，通过构造反事实场景来系统性地评估和增强大型视觉语言模型（LVLMs）在物理世界推理中的因果理解能力，揭示了现有模型在反事实物理推理上的显著不足。

**[Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)**

:   本文提出了一种反事实一致性提示（Counterfactual-Consistency Prompting）方法，通过生成反事实问题并施加集体约束来解决大语言模型在时序推理中的不一致性问题，在多个时序理解数据集上取得了显著改进。

**[Counterfactual Explanations for Aspect-Based Sentiment Analysis](counterfactual_explanations_for_aspect-based_sentiment_analysis.md)**

:   本文提出一种为方面级情感分析（ABSA）生成反事实解释的方法，通过找到能翻转特定方面情感极性的最小文本修改，为 ABSA 模型的预测提供直观的因果解释。

**[FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)**

:   提出 FitCF 框架，利用 BERT 特征归因方法（LIME/IG/SHAP等）提取重要词来引导 LLM 在 zero-shot 下生成反事实样本（ZeroCF），再经标签翻转验证筛选后作为 few-shot 示例，在新闻分类和情感分析任务上一致性超越 Polyjuice、BAE、FIZLE 三种基线。

**[IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](iris_an_iterative_and_integrated_framework.md)**

:   提出 IRIS 框架——仅需一组初始变量名作为输入，即可自动检索文档、提取变量值构建结构化数据、通过混合因果发现（GES 统计算法 + LLM 因果关系验证）构建因果图，并通过缺失变量提议组件迭代扩展变量集合，放松了传统方法的无环和因果充分性假设，在 Cancer、Diabetes、Obesity、ADNI、Insurance 等 6 个数据集上 F1 全面超越 0-shot/CoT/RAG 基线。

**[Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning](leveraging_variation_theory_in_counterfactual_data_augmentation_for_optimized_ac.md)**

:   本文将变异理论(Variation Theory)引入反事实数据增强(CDA)框架，通过保留神经符号模式的方式使用LLM生成反事实样本，并结合三级过滤流水线筛选高质量数据，用于优化主动学习中的少样本文本分类，在多个数据集上取得显著F1提升。

**[On the Reliability of Large Language Models for Causal Discovery](llm_causal_discovery_reliability.md)**

:   利用开源 LLM（OLMo、BLOOM）可访问的预训练语料库，实证验证了"因果鹦鹉"假说——LLM 识别因果关系的能力与预训练数据中该关系的出现频率高度相关（Spearman r=0.9），且错误因果关系的存在和上下文变化都会显著影响预测可靠性。

**[Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)**

:   提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

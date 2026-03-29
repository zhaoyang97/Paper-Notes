<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**💬 ACL2025** · 共 **5** 篇

**[CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)**

:   提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

**[FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)**

:   提出 FitCF 框架，利用特征归因方法（LIME/IG等）提取重要词来引导 LLM 生成反事实样本，并通过标签翻转验证构建可靠的 few-shot 示例，在新闻分类和情感分析任务上一致性超越三种 SOTA 基线。

**[IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](iris_an_iterative_and_integrated_framework.md)**

:   提出 IRIS 框架——仅需一组初始变量名作为输入，即可自动检索文档、提取变量值构建结构化数据、通过混合因果发现（GES 统计算法 + LLM 因果关系验证）构建因果图，并通过缺失变量提议组件迭代扩展变量集合，放松了传统方法的无环和因果充分性假设，在 Cancer、Diabetes、Obesity、ADNI、Insurance 等 6 个数据集上 F1 全面超越 0-shot/CoT/RAG 基线。

**[On the Reliability of Large Language Models for Causal Discovery](llm_causal_discovery_reliability.md)**

:   利用开源 LLM（OLMo、BLOOM）可访问的预训练语料库，实证验证了"因果鹦鹉"假说——LLM 识别因果关系的能力与预训练数据中该关系的出现频率高度相关（Spearman r=0.9），且错误因果关系的存在和上下文变化都会显著影响预测可靠性。

**[Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)**

:   提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

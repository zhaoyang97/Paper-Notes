---
title: >-
  ICLR2026 社会计算论文汇总 · 14篇论文解读
description: >-
  14篇ICLR2026的社会计算方向论文解读，涵盖 LLM、多模态、对抗鲁棒、语音、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
  - "对抗鲁棒"
  - "语音"
  - "对齐/RLHF"
item_list:
  - u: "adaptive_debiasing_tsallis_entropy_for_test-time_adaptation/"
    t: "Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation"
  - u: "adaptive_logit_adjustment_for_debiasing_multimodal_language_models/"
    t: "Adaptive Logit Adjustment for Debiasing Multimodal Language Models"
  - u: "biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses/"
    t: "BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses"
  - u: "from_five_dimensions_to_many_large_language_models_as_precise_and_interpretable_/"
    t: "From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers"
  - u: "functional_embeddings_enable_aggregation_of_multi-area_seeg_data_for_robust_bci/"
    t: "Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI"
  - u: "gradiend_feature_learning_within_neural_networks_exemplified_through_biases/"
    t: "GRADIEND: Feature Learning within Neural Networks Exemplified through Biases"
  - u: "human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction/"
    t: "Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction"
  - u: "mitigating_mismatch_within_reference-based_preference_optimization/"
    t: "Mitigating Mismatch within Reference-based Preference Optimization"
  - u: "propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models/"
    t: "Propaganda AI: An Analysis of Semantic Divergence in Large Language Models"
  - u: "sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn/"
    t: "SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition"
  - u: "scalable_multi-task_low-rank_model_adaptation/"
    t: "Scalable Multi-Task Low-Rank Model Adaptation"
  - u: "socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests/"
    t: "SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests"
  - u: "tracing_and_reversing_edits_in_llms/"
    t: "Tracing and Reversing Edits in LLMs"
  - u: "when_agents_persuade_propaganda_generation_and_mitigation_in_llms/"
    t: "When Agents Persuade: Propaganda Generation and Mitigation in LLMs"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🔬 ICLR2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

🔥 **高频主题：** LLM ×4

**[Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)**

:   提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

**[Adaptive Logit Adjustment for Debiasing Multimodal Language Models](adaptive_logit_adjustment_for_debiasing_multimodal_language_models.md)**

:   ALA 是一种**后处理**去偏方法：在自回归生成的每一步，用外部图像/文本分类器测出"图像该有的属性"与"文本当前流露的偏见"之间的偏差，再沿梯度方向**只对偏见相关词的 logit** 做按比例微调，从而在不改动模型内部表征、不重训的前提下，把图文属性对齐或中和有害刻板印象，且几乎不掉模型实用性。

**[BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)**

:   本文构建了 BiasFreeBench 基准，首次在统一框架下系统比较 8 种主流去偏方法（4 种 prompting + 4 种 training），聚焦于 LLM 响应层面的偏差评估，并提出了 Bias-Free Score 指标，发现 prompting 方法（尤其是 CoT）整体优于 training 方法，而 DPO 在跨偏差类型泛化上表现突出。

**[From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers](from_five_dimensions_to_many_large_language_models_as_precise_and_interpretable_.md)**

:   只给 LLM 一个人的 20 道大五人格题答案，让它角色扮演去预测这个人在另外 9 个心理量表上的作答，结果 LLM 重建出的"量表间相关结构"与真实人类数据高度对齐（$R^2>0.88$），并且通过分析推理链发现 LLM 走的是"先把原始分压缩成自然语言人格摘要、再据此推理"的两阶段抽象过程——它不是语义模式匹配，而是在做真正的心理推理。

**[Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](functional_embeddings_enable_aggregation_of_multi-area_seeg_data_for_robust_bci.md)**

:   提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建，在 20 名被试的多脑区 SEEG 数据集上验证有效。

**[GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)**

:   提出GRADIEND——一个基于梯度的编码器-解码器架构，通过单个瓶颈神经元从模型梯度中学习可解释的单语义特征（以性别为例），不仅可以识别哪些权重编码了特定特征，还能通过解码器直接修改模型权重来消除偏见，与INLP结合在所有基线模型上达到SOTA去偏效果。

**[Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)**

:   对9个SOTA语音对话系统开展首次语音图灵测试（2968次人类判断），发现所有系统均未通过（成功率7%-31%），瓶颈不在语义理解而在副语言特征、情感表达和对话人格，并构建了18维细粒度评估框架和可解释AI评审模型。

**[Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)**

:   揭示 DPO 的"过早满足"问题——当 reference 策略对 chosen 的概率低于 rejected 时（~45% pairs），DPO 的梯度被 reference 的悲观信号不必要地衰减（即使策略仍然错误即 $\Delta_\theta < 0$）；提出 HyPO（一行代码修改：$\max(0, \Delta_{ref})$ 裁剪 reference margin），在 AlpacaEval 2.0 上相对 DPO 提升 41.2%。

**[Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)**

:   提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。

**[SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)**

:   提出 SAGE，一个统一的 VPR 训练框架：引入轻量 Soft Probing 模块增强局部特征判别力，每个 epoch 在线重建融合地理距离与视觉相似度的亲和图，再通过贪心加权团扩展聚焦最难样本，冻结 DINOv2 骨干仅训练 1.96M 参数即在 8 个基准上全面 SOTA。

**[Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)**

:   系统分析多任务 LoRA 在任务数量增大时崩溃的根因（均匀正则化破坏共享知识 + 组件级 LoRA 放大梯度冲突），提出 mtLoRA：谱感知正则化 + 块级适配 + 细粒度路由，在 15-25 个任务上平均超越 SOTA 2.3%，同时减少 47% 参数和 24% 训练时间。

**[SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)**

:   提出首个专门针对社会政治危害的LLM安全评估基准 SocialHarmBench，包含585条覆盖7个领域、34个国家的提示，揭示了当前LLM在历史修正主义、宣传操纵等政治敏感场景中的系统性安全漏洞。

**[Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)**

:   针对知识编辑（Knowledge Editing）的双重使用风险，提出 EditScope 方法从编辑后的权重中推断被编辑的目标实体（准确率高达 99%），以及基于 SVD bottom-rank 近似的无训练编辑逆转方法（逆转率高达 94%），仅依赖编辑后的权重、不需要编辑 prompt 或原始权重信息。

**[When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)**

:   系统研究LLM的宣传生成行为，训练专用检测器量化3个LLM使用的6种修辞技术，发现所有LLM均能生成宣传且大量使用Loaded Language和Flag-Waving，通过SFT/DPO/ORPO三种微调方法缓解，ORPO将宣传分类率从77%降至10%、修辞技术使用减少13.4倍。

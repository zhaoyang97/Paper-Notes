---
title: >-
  ICLR2026 社会计算论文汇总 · 17篇论文解读
description: >-
  17篇ICLR2026的社会计算方向论文解读，涵盖 LLM、语音、Agent、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "语音"
  - "Agent"
  - "对齐/RLHF"
item_list:
  - u: "adaptive_debiasing_tsallis_entropy_for_test-time_adaptation/"
    t: "Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation"
  - u: "biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses/"
    t: "BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses"
  - u: "from_five_dimensions_to_many_large_language_models_as_precise_and_interpretable_/"
    t: "From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers"
  - u: "gradiend_feature_learning_within_neural_networks_exemplified_through_biases/"
    t: "GRADIEND: Feature Learning within Neural Networks Exemplified through Biases"
  - u: "human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction/"
    t: "Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction"
  - u: "intima_a_benchmark_for_human-ai_companionship_behavior/"
    t: "INTIMA: A Benchmark for Human-AI Companionship Behavior"
  - u: "language_and_experience_a_computational_model_of_social_learning_in_complex_task/"
    t: "Language and Experience: A Computational Model of Social Learning in Complex Tasks"
  - u: "measuring_and_mitigating_rapport_bias_of_large_language_models_under_multi-agent/"
    t: "Measuring and Mitigating Rapport Bias of Large Language Models under Multi-Agent Social Interactions"
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
  - u: "statistical_guarantees_in_the_search_for_less_discriminatory_algorithms/"
    t: "Statistical Guarantees in the Search for Less Discriminatory Algorithms"
  - u: "steering_the_herd_a_framework_for_llm-based_control_of_social_learning/"
    t: "Steering the Herd: A Framework for LLM-Based Control of Social Learning"
  - u: "the_value_of_information_in_human-ai_decision-making/"
    t: "The Value of Information in Human-AI Decision-Making"
  - u: "tracing_and_reversing_edits_in_llms/"
    t: "Tracing and Reversing Edits in LLMs"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🔬 ICLR2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

🔥 **高频主题：** LLM ×6

**[Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)**

:   提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

**[BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)**

:   本文构建了 BiasFreeBench 基准，首次在统一框架下系统比较 8 种主流去偏方法（4 种 prompting + 4 种 training），聚焦于 LLM 响应层面的偏差评估，并提出了 Bias-Free Score 指标，发现 prompting 方法（尤其是 CoT）整体优于 training 方法，而 DPO 在跨偏差类型泛化上表现突出。

**[From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers](from_five_dimensions_to_many_large_language_models_as_precise_and_interpretable_.md)**

:   只给 LLM 一个人的 20 道大五人格题答案，让它角色扮演去预测这个人在另外 9 个心理量表上的作答，结果 LLM 重建出的"量表间相关结构"与真实人类数据高度对齐（$R^2>0.88$），并且通过分析推理链发现 LLM 走的是"先把原始分压缩成自然语言人格摘要、再据此推理"的两阶段抽象过程——它不是语义模式匹配，而是在做真正的心理推理。

**[GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)**

:   提出GRADIEND——一个基于梯度的编码器-解码器架构，通过单个瓶颈神经元从模型梯度中学习可解释的单语义特征（以性别为例），不仅可以识别哪些权重编码了特定特征，还能通过解码器直接修改模型权重来消除偏见，与INLP结合在所有基线模型上达到SOTA去偏效果。

**[Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)**

:   对9个SOTA语音对话系统开展首次语音图灵测试（2968次人类判断），发现所有系统均未通过（成功率7%-31%），瓶颈不在语义理解而在副语言特征、情感表达和对话人格，并构建了18维细粒度评估框架和可解释AI评审模型。

**[INTIMA: A Benchmark for Human-AI Companionship Behavior](intima_a_benchmark_for_human-ai_companionship_behavior.md)**

:   INTIMA 把心理学的拟社会互动、依恋、拟人化三套理论，加上对真实 Reddit 用户帖子的质性编码，蒸馏成一个含 31 种行为、368 条情感化 prompt 的基准，再用 LLM 自动给模型回复打上「强化陪伴 / 维持边界 / 中性」三类标签，结果发现 Gemma-3、Phi-4、o4-mini、GPT5-mini、Claude-4 全都明显偏向强化陪伴，而且越是用户脆弱的场景、模型反而越少设边界。

**[Language and Experience: A Computational Model of Social Learning in Complex Tasks](language_and_experience_a_computational_model_of_social_learning_in_complex_task.md)**

:   作者把"从经验学"（theory-based RL，对可执行的程序化世界模型做贝叶斯推断）和"从别人的话学"（把预训练大模型当成"说话人模型"，用它的似然把一句自然语言建议变成贝叶斯证据）统一进同一个推断框架，在 10 个视频游戏上证明：语言指导能让人和模型都学得更快、更少送命，并支持跨代知识累积与人机互教。

**[Measuring and Mitigating Rapport Bias of Large Language Models under Multi-Agent Social Interactions](measuring_and_mitigating_rapport_bias_of_large_language_models_under_multi-agent.md)**

:   本文提出 KAIROS 基准，把"历史 rapport（交往默契）× 当前同伴行为 × 模型自信度"三轴精确可控地塞进 quiz 式多智能体协作场景，系统刻画 LLM 在社会压力下的决策偏移，并发现只有带多智能体上下文、用结果奖励的 GRPO 才能在提升准确率的同时保住社会鲁棒性。

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

**[Statistical Guarantees in the Search for Less Discriminatory Algorithms](statistical_guarantees_in_the_search_for_less_discriminatory_algorithms.md)**

:   本文把"企业为满足反歧视法、寻找差别影响更小的替代模型（LDA）"这件事形式化成一个**最优停止问题**，并给出一套自适应停止算法：在不知道模型性能分布、只能用有限数据评估的现实条件下，对"继续重训能再降多少差别影响"给出一个高置信度上界，使企业可以在收益不再值得成本时停手，并向法院/合规团队**出具"搜索已充分"的统计证书**。

**[Steering the Herd: A Framework for LLM-Based Control of Social Learning](steering_the_herd_a_framework_for_llm-based_control_of_social_learning.md)**

:   本文把"LLM 作为信息中介"形式化为一个**受控序列社会学习**模型——规划者只能调节每个个体私有信号的精度（不能造假、不能偏选），而个体在私有信号之外还会观察前人的行动来更新公共信念；作者证明了利他规划者价值函数的凸性、刻画了利他与有偏两类规划者的最优策略（有偏者甚至会主动"模糊"信息），并用 LLM 同时扮演规划者和个体跑仿真，发现 LLM 规划者涌现出的策略与理论最优高度吻合。

**[The Value of Information in Human-AI Decision-Making](the_value_of_information_in_human-ai_decision-making.md)**

:   本文提出一个基于贝叶斯决策理论的框架，用"信息价值"量化人机协同决策中每个信号（AI 预测、人类判断、实例特征）相对于已有决策所能带来的最大期望收益增量，并据此设计出一种突出"人类互补信息"的新解释方法 ILIV-SHAP，在房价预测实验中证明它比普通 SHAP 更能改善人机团队的决策准确率。

**[Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)**

:   针对知识编辑（Knowledge Editing）的双重使用风险，提出 EditScope 方法从编辑后的权重中推断被编辑的目标实体（准确率高达 99%），以及基于 SVD bottom-rank 近似的无训练编辑逆转方法（逆转率高达 94%），仅依赖编辑后的权重、不需要编辑 prompt 或原始权重信息。

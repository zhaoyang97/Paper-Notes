---
title: >-
  ICLR2026 推荐系统论文汇总 · 24篇论文解读
description: >-
  24篇ICLR2026的推荐系统方向论文解读，涵盖推荐系统、LLM、个性化生成、扩散模型、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "个性化生成"
  - "扩散模型"
  - "推理"
item_list:
  - u: "adaptive_regularization_for_large-scale_sparse_feature_embedding_models/"
    t: "Adaptive Regularization for Large-Scale Sparse Feature Embedding Models"
  - u: "beyond_markovian_drifts_action-biased_geometric_walks_with_memory_for_personaliz/"
    t: "Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization"
  - u: "catalog-native_llm_speaking_item-id_dialect_with_less_entanglement_for_recommend/"
    t: "Catalog-Native LLM: Speaking Item-ID dialect with Less Entanglement for Recommendation"
  - u: "collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco/"
    t: "CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation"
  - u: "continual_low-rank_adapters_for_llm-based_generative_recommender_systems/"
    t: "Continual Low-Rank Adapters for LLM-based Generative Recommender Systems"
  - u: "discrete_diffusion_for_bundle_construction/"
    t: "Discrete Diffusion for Bundle Construction"
  - u: "from_evaluation_to_defense_advancing_safety_in_video_large_language_models/"
    t: "From Evaluation to Defense: Advancing Safety in Video Large Language Models"
  - u: "goalrank_group-relative_optimization_for_a_large_ranking_model/"
    t: "GoalRank: Group-Relative Optimization for a Large Ranking Model"
  - u: "ifusion_integrating_dynamic_interest_streams_via_diffusion_model_for_click-throu/"
    t: "iFusion: Integrating Dynamic Interest Streams via Diffusion Model for Click-Through Rate Prediction"
  - u: "in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g/"
    t: "In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations"
  - u: "low-pass_personalized_subgraph_federated_recommendation/"
    t: "Low-pass Personalized Subgraph Federated Recommendation"
  - u: "massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequential_tra/"
    t: "Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders"
  - u: "more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user/"
    t: "More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences"
  - u: "off-policy_evaluation_for_ranking_policies_under_deterministic_logging_policies/"
    t: "Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies"
  - u: "on_the_mechanisms_of_collaborative_learning_in_vae_recommenders/"
    t: "On the Mechanisms of Collaborative Learning in VAE Recommenders"
  - u: "propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis/"
    t: "ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation"
  - u: "rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme/"
    t: "Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning"
  - u: "reinforced_latent_reasoning_for_llm-based_recommendation/"
    t: "Reinforced Latent Reasoning for LLM-based Recommendation"
  - u: "rpm_reasoning-level_personalization_for_black-box_large_language_models/"
    t: "RPM: Reasoning-Level Personalization for Black-Box Large Language Models"
  - u: "search_arena_analyzing_search-augmented_llms/"
    t: "Search Arena: Analyzing Search-Augmented LLMs"
  - u: "steering_diffusion_models_towards_credible_content_recommendation/"
    t: "Steering Diffusion Models Towards Credible Content Recommendation"
  - u: "supporting_high-stakes_decision_making_through_interactive_preference_elicitatio/"
    t: "Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space"
  - u: "token-efficient_item_representation_via_images_for_llm_recommender_systems/"
    t: "Token-Efficient Item Representation via Images for LLM Recommender Systems"
  - u: "token-efficient_long-term_interest_sketching_and_internalized_reasoning_for_llm-/"
    t: "Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🔬 ICLR2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (22)](../../ACL2026/recommender/index.md) · [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** 推荐系统 ×12 · LLM ×10 · 个性化生成 ×4 · 扩散模型 ×3 · 推理 ×3

**[Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)**

:   本文用 Rademacher 复杂度从理论上解释了 CTR/CVR 模型「训练超过一个 epoch 就严重过拟合」的根因——embedding 层范数无约束增长撑大了泛化界，并据此提出按特征出现频率自适应分配范数预算的正则方法 AdamAR：高频特征轻正则、低频特征重正则，既消除多 epoch 过拟合又能提升单 epoch 性能，已在阿里搜索广告线上部署。

**[Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization](beyond_markovian_drifts_action-biased_geometric_walks_with_memory_for_personaliz.md)**

:   本文提出"结构化游走假设"(SWH)质疑个性化摘要中通用的马尔可夫漂移假设(MDH)，并给出轻量编码-解码模型 Walk2Pers——把用户偏好演化建模成带双记忆通道、可分解为幅度与方向(连续 vs 新颖)的动作偏置几何游走，在三个基准上显著超越专用摘要器与大模型。

**[Catalog-Native LLM: Speaking Item-ID dialect with Less Entanglement for Recommendation](catalog-native_llm_speaking_item-id_dialect_with_less_entanglement_for_recommend.md)**

:   针对"把 item-ID 塞进 LLM 会让协同信号和语言语义互相打架"这个问题，本文提出 IDIOMoE：把预训练 LLM 每个 block 的 FFN 拆成一个**文本专家**和一个**item 专家**，用静态的 token-type 门控按 token 类型分流（item-id token 走 item 专家，其余走文本专家），从而把"协同过滤"和"语义理解"解耦到不同子网络里，在公开和工业级数据集上都取得最强推荐效果，同时几乎不损伤原 LLM 的语言能力。

**[CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)**

:   观察到序列推荐中不同用户的 KV cache 具有显著跨用户相似性（协同信号），提出 CollectiveKV 将 KV 分解为低维用户特有部分和从全局 KV 池检索的高维共享部分，实现 0.8% 的压缩率且性能不降。

**[Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](continual_low-rank_adapters_for_llm-based_generative_recommender_systems.md)**

:   PESO 把基于 LLM 的生成式推荐的持续学习从"堆叠多个冻结适配器"改成"单个不断演化的 LoRA + 一个近端正则项"，让适配器每次更新都被轻轻锚向上一阶段的状态，从而在保留长期偏好与吸收新偏好之间自动找平衡，在三个真实数据集上稳定超过累积式 LoRA 和单纯演化的 LoRA。

**[Discrete Diffusion for Bundle Construction](discrete_diffusion_for_bundle_construction.md)**

:   DDBC 把"捆绑构建"（从大商品库里挑一组商品凑成一个完整 bundle，或补全一个残缺 bundle）重新建模成**掩码离散扩散**过程：用残差向量量化（RVQ）把每件商品压成几位共享码本里的离散码以化解海量商品库带来的维度灾难，再用一个双向 Transformer 以顺序无关的方式逐步把 `[MASK]` 去噪还原成完整 bundle，在长 bundle 数据集上相对最强基线取得 100%+ 的相对提升。

**[From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)**

:   构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[iFusion: Integrating Dynamic Interest Streams via Diffusion Model for Click-Through Rate Prediction](ifusion_integrating_dynamic_interest_streams_via_diffusion_model_for_click-throu.md)**

:   iFusion 把"长短期用户兴趣融合"重新表述为一个条件生成问题——以短期兴趣为引导，对长期兴趣表示做扩散去噪，从而摆脱传统线性融合（拼接/注意力/门控）的假设，在公开数据集、工业数据集和线上 A/B 上都拿到 CTR 提升。

**[In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)**

:   通过对来自6家提供商的12个LLM在新闻、学术、电商三大领域的大规模控制实验，揭示了LLM存在系统性的**隐式信息源偏好**（latent source preferences）——当内容语义完全相同时，仅更换来源标签就能显著改变模型的信息选择行为，且这种偏好无法通过提示工程消除。

**[Low-pass Personalized Subgraph Federated Recommendation](low-pass_personalized_subgraph_federated_recommendation.md)**

:   针对联邦推荐里各客户端"子图大小/连通度差异巨大"导致的表示错位与流行度偏置问题，LPSFed 用低通谱滤波抽取跨子图稳定的低频结构信号来度量客户端与中性锚图的相似度、据此个性化聚合参数，并叠加一个感知局部流行度的自适应 margin 校正长尾，五个数据集上 NDCG 最高涨 24%。

**[Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders](massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequential_tra.md)**

:   VISTA 把候选物品对超长用户历史的 target attention 拆成「先把上百万长度的历史压成几百个摘要 token 并缓存」+「下游只对缓存 token 做轻量注意力」两阶段，让训练/推理成本固定，已在 Meta 服务数十亿用户的推荐平台上线。

**[More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)**

:   用户点过的东西未必是真喜欢的——本文提出"一致性偏好"(Coherent Preference)来补充传统"显示偏好"(Revealed Preference)，并设计冲突感知的 DPO 变体 C-APO，在 RP 与 CP 一致时放大、冲突时压制其影响，从而同时提升推荐准确率和理由的说服力。

**[Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies](off-policy_evaluation_for_ranking_policies_under_deterministic_logging_policies.md)**

:   针对工业排序系统常用「完全确定性」日志策略导致传统 IPS 类估计器严重偏置的痛点，本文提出用「用户点击概率之比」代替「策略概率之比」作为重要性权重的 CIPS 估计器（及其双重稳健扩展 CDR），把无偏性所需的支撑条件从「日志策略要足够随机」放宽到「点击行为本身有随机性」，从而在确定性日志下实现低偏甚至无偏评估。

**[On the Mechanisms of Collaborative Learning in VAE Recommenders](on_the_mechanisms_of_collaborative_learning_in_vae_recommenders.md)**

:   本文从理论上揭示 VAE 协同过滤里"用户之间能否互相帮助"由潜在空间的距离（一个可推导的"共享半径"）决定，并指出干净输入只用得上局部协作、而 β-KL 与输入掩码各自有代价地促进全局协作；据此提出训练期专用的锚点正则 PIA，把掩码后的用户表示拉向其交互物品的锚点中心，稳住几何结构并促成语义对齐的全局协作，在三个公开数据集和 Amazon 流媒体平台的线上 A/B 测试中都拿到提升。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4，首次验证了主动性+个性化统一的可行性。

**[Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme.md)**

:   本文提出 ConvRec-R1 两阶段框架训练 LLM 对话推荐系统：先用 Remap–Reflect–Adjust 蒸馏管线从黑盒教师造出「落在目标 catalog 内」的高质量演示做 SFT 预热，再用 Rank-GRPO（把推荐列表中的「每个 rank」当作动作单元的 GRPO 改造）做 RL 对齐，让 0.5B–3B 的小模型在 REDDIT-V2 上的 Recall/NDCG 收敛更快、并能追平甚至超过 GPT-4o。

**[Reinforced Latent Reasoning for LLM-based Recommendation](reinforced_latent_reasoning_for_llm-based_recommendation.md)**

:   针对 LLM 推荐中显式思维链（CoT）既难拿到监督数据、推理又慢的痛点，本文提出 LatentR3：在 LLM 顶层加一个注意力层把推理压进**连续潜在空间**（只需 1 个潜在 token），再用一套改造过的 GRPO（PPL 连续奖励 + batch 级优势）在完全没有 CoT 监督的情况下端到端学会推理，套在 BIGRec / D3 上分别带来 17.0% / 8.4% 的相对提升。

**[RPM: Reasoning-Level Personalization for Black-Box Large Language Models](rpm_reasoning-level_personalization_for_black-box_large_language_models.md)**

:   RPM 把黑盒 LLM 的个性化从"对齐最终回答"升级为"对齐底层推理过程"：它从原始用户历史里自动抽取「特征→因子→统计量」的结构化用户模型，为每条历史构造个性化推理路径，再用基于特征的检索把这些推理示例喂给模型，让 LLM 沿着用户自己的逻辑来推理，在四类任务上一致超过现有的回答级个性化方法且更可解释。

**[Search Arena: Analyzing Search-Augmented LLMs](search_arena_analyzing_search-augmented_llms.md)**

:   构建 Search Arena——首个大规模搜索增强 LLM 人类偏好数据集（24069 对话 + 12652 偏好投票，71 种语言），发现用户偏好受引用数量影响（即使引用不支持声明），社区驱动平台比 Wikipedia 更受偏好，搜索增强不降低通用聊天性能但通用 LLM 在搜索场景显著退化。

**[Steering Diffusion Models Towards Credible Content Recommendation](steering_diffusion_models_towards_credible_content_recommendation.md)**

:   针对扩散模型做序列推荐时会推送假新闻、虚假信息等不可信内容的问题，本文提出 Disco：用一个"解耦扩散模型"把用户真实偏好信号和不可信内容信号分离开、再把扩散目标投影到不可信特征的零空间里抑制不可信内容，并在标签稀缺时渐进地检测潜在不可信物品来补全这个零空间，最终在三个真实数据集上同时拿到更高的推荐准确率和可信率。

**[Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space](supporting_high-stakes_decision_making_through_interactive_preference_elicitatio.md)**

:   本文面向租房这类高风险、低频、反馈稀疏的决策场景，把用户访谈得到的 LLM 偏好先验、Autoencoder 潜空间压缩和 Preferential Bayesian Optimization 结合起来，用更少的成对比较学习用户效用函数，并在真实房源数据上比普通 PBO 获得更高的排序准确率。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

**[Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation](token-efficient_long-term_interest_sketching_and_internalized_reasoning_for_llm-.md)**

:   本文提出 SIREN，用「长期兴趣草图」把动辄上百条的用户历史压成一小串「喜欢/不喜欢的语义主题」喂给 LLM，再用「两阶段训练」先 RL 学会显式 CoT 推理、后把推理通过隐状态对齐内化进参数，从而在 answer-only 解码下保住 CoT 精度，输入 token 降 48.7%、推理延迟比 CoT 低 100×。

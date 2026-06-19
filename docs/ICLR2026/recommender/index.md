---
title: >-
  ICLR2026 推荐系统论文汇总 · 13篇论文解读
description: >-
  13篇ICLR2026的推荐系统方向论文解读，涵盖 LLM、推荐系统、扩散模型、对抗鲁棒、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "对抗鲁棒"
  - "个性化生成"
item_list:
  - u: "adaptive_regularization_for_large-scale_sparse_feature_embedding_models/"
    t: "Adaptive Regularization for Large-Scale Sparse Feature Embedding Models"
  - u: "bed-llm_intelligent_information_gathering_with_llms_and_bayesian_experimental_de/"
    t: "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design"
  - u: "c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system/"
    t: "C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems"
  - u: "catalog-native_llm_speaking_item-id_dialect_with_less_entanglement_for_recommend/"
    t: "Catalog-Native LLM: Speaking Item-ID dialect with Less Entanglement for Recommendation"
  - u: "collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco/"
    t: "CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation"
  - u: "discrete_diffusion_for_bundle_construction/"
    t: "Discrete Diffusion for Bundle Construction"
  - u: "from_evaluation_to_defense_advancing_safety_in_video_large_language_models/"
    t: "From Evaluation to Defense: Advancing Safety in Video Large Language Models"
  - u: "goalrank_group-relative_optimization_for_a_large_ranking_model/"
    t: "GoalRank: Group-Relative Optimization for a Large Ranking Model"
  - u: "in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g/"
    t: "In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations"
  - u: "propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis/"
    t: "ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation"
  - u: "rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese/"
    t: "RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search"
  - u: "search_arena_analyzing_search-augmented_llms/"
    t: "Search Arena: Analyzing Search-Augmented LLMs"
  - u: "token-efficient_item_representation_via_images_for_llm_recommender_systems/"
    t: "Token-Efficient Item Representation via Images for LLM Recommender Systems"
item_total: 13
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🔬 ICLR2026** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [💬 ACL2026 (22)](../../ACL2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** LLM ×5 · 推荐系统 ×4

**[Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)**

:   本文用 Rademacher 复杂度从理论上解释了 CTR/CVR 模型「训练超过一个 epoch 就严重过拟合」的根因——embedding 层范数无约束增长撑大了泛化界，并据此提出按特征出现频率自适应分配范数预算的正则方法 AdamAR：高频特征轻正则、低频特征重正则，既消除多 epoch 过拟合又能提升单 epoch 性能，已在阿里搜索广告线上部署。

**[BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design](bed-llm_intelligent_information_gathering_with_llms_and_bayesian_experimental_de.md)**

:   把序贯贝叶斯实验设计（BED）套到 LLM 上，让模型每一轮都挑"期望信息增益（EIG）最大"的问题去问用户，从而把 LLM 变成会主动、自适应收集信息的多轮对话 agent；在 20 Questions 和电影偏好推断上，平均成功率比直接 prompting 高出 37.4 个百分点。

**[C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)**

:   提出 C2AL（Cohort-Contrastive Auxiliary Learning），通过数据驱动地发现分布差异最大的用户群体对，构建对比性辅助二分类任务正则化共享编码器，使 FM 注意力权重从稀疏变为稠密，缓解大规模推荐系统中少数群体的表征偏差，在 Meta 6 个生产模型（数十亿数据点）上验证有效。

**[Catalog-Native LLM: Speaking Item-ID dialect with Less Entanglement for Recommendation](catalog-native_llm_speaking_item-id_dialect_with_less_entanglement_for_recommend.md)**

:   针对"把 item-ID 塞进 LLM 会让协同信号和语言语义互相打架"这个问题，本文提出 IDIOMoE：把预训练 LLM 每个 block 的 FFN 拆成一个**文本专家**和一个**item 专家**，用静态的 token-type 门控按 token 类型分流（item-id token 走 item 专家，其余走文本专家），从而把"协同过滤"和"语义理解"解耦到不同子网络里，在公开和工业级数据集上都取得最强推荐效果，同时几乎不损伤原 LLM 的语言能力。

**[CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)**

:   观察到序列推荐中不同用户的 KV cache 具有显著跨用户相似性（协同信号），提出 CollectiveKV 将 KV 分解为低维用户特有部分和从全局 KV 池检索的高维共享部分，实现 0.8% 的压缩率且性能不降。

**[Discrete Diffusion for Bundle Construction](discrete_diffusion_for_bundle_construction.md)**

:   DDBC 把"捆绑构建"（从大商品库里挑一组商品凑成一个完整 bundle，或补全一个残缺 bundle）重新建模成**掩码离散扩散**过程：用残差向量量化（RVQ）把每件商品压成几位共享码本里的离散码以化解海量商品库带来的维度灾难，再用一个双向 Transformer 以顺序无关的方式逐步把 `[MASK]` 去噪还原成完整 bundle，在长 bundle 数据集上相对最强基线取得 100%+ 的相对提升。

**[From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)**

:   构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)**

:   通过对来自6家提供商的12个LLM在新闻、学术、电商三大领域的大规模控制实验，揭示了LLM存在系统性的**隐式信息源偏好**（latent source preferences）——当内容语义完全相同时，仅更换来源标签就能显著改变模型的信息选择行为，且这种偏好无法通过提示工程消除。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4，首次验证了主动性+个性化统一的可行性。

**[RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)**

:   提出 RAE（Regularized Auto-Encoder），通过线性自编码器 + Frobenius 范数正则化实现降维，理论证明正则化系数 $\lambda$ 通过 Rayleigh 商性质约束编码器矩阵的条件数 $\kappa(W)$，从而保证范数失真率有界、k-NN 结构被保持。在 4 个数据集上一致优于 PCA/UMAP/MDS/ISOMAP，余弦距离下比 PCA 至少高 12%，且训练仅需 8 秒、推理毫秒级。

**[Search Arena: Analyzing Search-Augmented LLMs](search_arena_analyzing_search-augmented_llms.md)**

:   构建 Search Arena——首个大规模搜索增强 LLM 人类偏好数据集（24069 对话 + 12652 偏好投票，71 种语言），发现用户偏好受引用数量影响（即使引用不支持声明），社区驱动平台比 Wikipedia 更受偏好，搜索增强不降低通用聊天性能但通用 LLM 在搜索场景显著退化。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

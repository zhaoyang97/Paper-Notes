<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🔬 ICLR2026** · 共 **8** 篇

**[C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)**

:   提出 C2AL（Cohort-Contrastive Auxiliary Learning），通过数据驱动地发现分布差异最大的用户群体对，构建对比性辅助二分类任务正则化共享编码器，使 FM 注意力权重从稀疏变为稠密，缓解大规模推荐系统中少数群体的表征偏差，在 Meta 6 个生产模型（数十亿数据点）上验证有效。

**[From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)**

:   构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出 ProPerSim 模拟框架和 ProPerAssistant 基线，通过用户-助手模拟环境结合 DPO 偏好学习，开发能同时具备主动性和个性化的 AI 家庭助手。

**[RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)**

:   提出 RAE（Regularized Auto-Encoder），通过线性自编码器 + Frobenius 范数正则化实现降维，理论证明正则化系数 $\lambda$ 通过 Rayleigh 商性质约束编码器矩阵的条件数 $\kappa(W)$，从而保证范数失真率有界、k-NN 结构被保持。在 4 个数据集上一致优于 PCA/UMAP/MDS/ISOMAP，余弦距离下比 PCA 至少高 12%，且训练仅需 8 秒、推理毫秒级。

**[Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)**

:   理论证明 CE 损失在推荐系统 KD 中最大化 NDCG 下界需满足"闭合性假设"（子集需包含学生 top 项目），但实际目标是蒸馏教师 top 项目的排序——两者冲突导致 vanilla CE 表现差。据此提出 RCE-KD：将教师 top-K 项目按是否在学生 top-K 中分两组，分别用精确 CE 和采样近似闭合性 CE，自适应融合权重随训练动态调整。

**[Search Arena: Analyzing Search-Augmented LLMs](search_arena_analyzing_search-augmented_llms.md)**

:   构建 Search Arena——首个大规模搜索增强 LLM 人类偏好数据集（24069 对话 + 12652 偏好投票，71 种语言），发现用户偏好受引用数量影响（即使引用不支持声明），社区驱动平台比 Wikipedia 更受偏好，搜索增强不降低通用聊天性能但通用 LLM 在搜索场景显著退化。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

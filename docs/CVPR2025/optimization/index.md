---
title: >-
  CVPR2025 优化/理论论文汇总 · 11篇论文解读
description: >-
  11篇CVPR2025的优化/理论方向论文解读，涵盖联邦学习、对抗鲁棒、压缩/编码、模型压缩、少样本学习、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "对抗鲁棒"
  - "压缩/编码"
  - "模型压缩"
  - "少样本学习"
  - "多模态"
item_list:
  - u: "automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network/"
    t: "Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression"
  - u: "conformal_prediction_for_zero-shot_models/"
    t: "Conformal Prediction for Zero-Shot Models"
  - u: "convex_relaxation_for_robust_vanishing_point_estimation_in_manhattan_world/"
    t: "Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World"
  - u: "federated_learning_with_domain_shift_eraser/"
    t: "Federated Learning with Domain Shift Eraser"
  - u: "how_to_merge_your_multimodal_models_over_time/"
    t: "How to Merge Your Multimodal Models Over Time?"
  - u: "mind_the_gap_confidence_discrepancy_can_guide_federated_semi-supervised_learning/"
    t: "Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning"
  - u: "model_poisoning_attacks_to_federated_learning_via_multi-round_consistency/"
    t: "Model Poisoning Attacks to Federated Learning via Multi-Round Consistency"
  - u: "scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn/"
    t: "SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning"
  - u: "stop_walking_in_circles_bailing_out_early_in_projected_gradient_descent/"
    t: "Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent"
  - u: "test-time_augmentation_improves_efficiency_in_conformal_prediction/"
    t: "Test-Time Augmentation Improves Efficiency in Conformal Prediction"
  - u: "towards_stable_and_storage-efficient_dataset_distillation_matching_convexified_t/"
    t: "Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**📷 CVPR2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×4 · 对抗鲁棒 ×2

**[Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)**

:   提出 GETA 框架实现自动联合结构化剪枝和量化感知训练：量化感知依赖图（QADG）构建通用剪枝搜索空间 + 部分投影 SGD 保证逐层比特约束 + 可解释的联合学习策略，在 CNN 和 Transformer 上均达到竞争力或领先的压缩性能。

**[Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)**

:   将保形预测（Conformal Prediction）应用于零样本模型，为 CLIP 等模型的预测提供有理论保证的不确定性量化和校准预测集

**[Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World](convex_relaxation_for_robust_vanishing_point_estimation_in_manhattan_world.md)**

:   GlobustVP 首次将凸松弛技术引入曼哈顿世界消失点估计问题，通过将联合估计消失点位置与直线-消失点关联的问题转化为 QCQP 再松弛为 SDP，实现了全局最优且对 70% 外点鲁棒的高效求解器（~50ms/图）。

**[Federated Learning with Domain Shift Eraser](federated_learning_with_domain_shift_eraser.md)**

:   提出FDSE方法，将每层网络分解为域无关特征提取器（DFE，全局聚合增强共识）和域特异偏移消除器（DSE，个性化聚合保留本地特性），结合BN一致性正则化，在DomainNet上达到76.77%（超Ditto 1.6%），在Office-Caltech10上达到91.58%（超FedBN 4.6%）。

**[How to Merge Your Multimodal Models Over Time?](how_to_merge_your_multimodal_models_over_time.md)**

:   本文提出 TIME（Temporal Integration of Model Expertise）框架，系统研究了多模态专家模型随时间渐进融合的问题，通过初始化策略、部署策略和融合技术三个轴定义搜索空间，在 FoMo-in-Flux 基准上揭示了时序模型融合的关键设计原则。

**[Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning](mind_the_gap_confidence_discrepancy_can_guide_federated_semi-supervised_learning.md)**

:   提出 TABASCO，一个两阶段二维样本选择框架解决同时存在标签噪声和长尾分布的联邦半监督学习：用加权 JSD（WJSD）和自适应质心距离（ACD）两个互补指标识别干净样本，GMM 聚类后以半监督方式利用剩余噪声数据，在 CIFAR-10（0.1 不平衡+0.4 噪声）上达 85.53%。

**[Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)**

:   发现现有联邦学习模型投毒攻击因跨轮次方向不一致导致效果自相抵消，提出 PoisonedFL 通过固定随机方向向量 + 动态幅度调节 + 假设检验机制实现多轮一致性攻击，在无需任何真实客户端信息的前提下击穿 8 种 SOTA 防御。

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 提出了一种面向联邦学习的语义 coreset 选择框架，利用 VLM（MobileCLIP-S2）零样本提取三种标量指标（表示分数、多样性分数、边界接近度），通过服务器聚合全局共识后指导客户端进行两阶段剪枝（异常过滤+冗余消除），在 128-512× 上行带宽减少和 7.72× 加速的同时保持竞争精度。

**[Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent](stop_walking_in_circles_bailing_out_early_in_projected_gradient_descent.md)**

:   发现 PGD 攻击在 L∞ 球上对鲁棒样本会产生循环行为，通过哈希检测循环实现提前终止（PGD_CD），在保持完全相同鲁棒性评估结果的前提下实现最高 96% 的迭代次数减少。

**[Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)**

:   发现测试时数据增强（TTA）可以系统性地提升共形预测的效率——通过在校准集上学习增强权重来优化增强聚合策略，在 ImageNet ResNet-50 上将预测集大小减少 10-17%，同时严格保持覆盖率保证。

**[Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory](towards_stable_and_storage-efficient_dataset_distillation_matching_convexified_t.md)**

:   提出 MCT (Matching Convexified Trajectory) 方法，通过将 SGD 专家轨迹替换为从随机初始化到最优点的凸组合线性轨迹，同时解决了传统 MTT 方法的轨迹不稳定、收敛慢和存储消耗高三大问题。

---
title: >-
  ICML2025 自监督/表示学习论文汇总 · 22篇论文解读
description: >-
  22篇ICML2025的自监督/表示学习方向论文解读，涵盖自监督学习、少样本学习、强化学习、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "少样本学习"
  - "强化学习"
  - "对齐/RLHF"
  - "对抗鲁棒"
item_list:
  - u: "a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints/"
    t: "A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints"
  - u: "adaworld_learning_adaptable_world_models_with_latent_actions/"
    t: "AdaWorld: Learning Adaptable World Models with Latent Actions"
  - u: "alpha-sql_zero-shot_text-to-sql_using_monte_carlo_tree_search/"
    t: "Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search"
  - u: "beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h/"
    t: "Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions"
  - u: "clarify_contrastive_preference_reinforcement_learning_for_untangling_ambiguous_q/"
    t: "CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries"
  - u: "clustering_properties_of_self-supervised_learning/"
    t: "ReSA: Clustering Properties of Self-Supervised Learning"
  - u: "collapse-proof_non-contrastive_self-supervised_learning/"
    t: "Collapse-Proof Non-Contrastive Self-Supervised Learning"
  - u: "contextures_representations_from_contexts/"
    t: "Contextures: Representations from Contexts"
  - u: "deep_learning_is_not_so_mysterious_or_different/"
    t: "Deep Learning is Not So Mysterious or Different"
  - u: "discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le/"
    t: "Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning"
  - u: "foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o/"
    t: "Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection"
  - u: "generalization_analysis_for_supervised_contrastive_representation_learning_under/"
    t: "Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings"
  - u: "griffin_towards_a_graph-centric_relational_database_foundation_model/"
    t: "Griffin: Towards a Graph-Centric Relational Database Foundation Model"
  - u: "mtl-ue_learning_to_learn_nothing_for_multi-task_learning/"
    t: "MTL-UE: Learning to Learn Nothing for Multi-Task Learning"
  - u: "neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur/"
    t: "Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling"
  - u: "pde-transformer_efficient_and_versatile_transformers_for_physics_simulations/"
    t: "PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations"
  - u: "proxy-fda_proxy-based_feature_distribution_alignment_for_fine-tuning_vision_foun/"
    t: "Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting"
  - u: "test-time_canonicalization_by_foundation_models_for_robust_perception/"
    t: "Test-Time Canonicalization by Foundation Models for Robust Perception"
  - u: "test-time_training_provably_improves_transformers_as_in-context_learners/"
    t: "Test-Time Training Provably Improves Transformers as In-Context Learners"
  - u: "towards_benchmarking_foundation_models_for_tabular_data_with_text/"
    t: "Towards Benchmarking Foundation Models for Tabular Data With Text"
  - u: "update_your_transformer_to_the_latest_release_re-basin_of_task_vectors/"
    t: "Update Your Transformer to the Latest Release: Re-Basin of Task Vectors"
  - u: "what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models/"
    t: "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🧪 ICML2025** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×3

**[A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)**

:   引入"下游自由能"作为预训练检查点可适应性的贝叶斯模型选择准则，证明"预训练自由能"可作为其上界代理（无需下游数据），并实验验证大学习率/小 batch/高 momentum 通过降低预训练自由能改善下游迁移性能。

**[AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)**

:   提出 AdaWorld——通过从视频中自监督提取潜在动作（latent actions）进行动作感知预训练，构建高度可适应的世界模型，支持零样本动作迁移和少量交互快速适应新环境。

**[Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search](alpha-sql_zero-shot_text-to-sql_using_monte_carlo_tree_search.md)**

:   Alpha-SQL 将零样本 Text-to-SQL 建模为树搜索问题，通过蒙特卡洛树搜索 (MCTS) 框架结合 LLM-as-Action-Model 和自监督奖励函数，无需微调即可在 BIRD 数据集上以 32B 开源模型达到 69.7% 执行精度，超越基于 GPT-4o 的零样本 SOTA 2.5 个百分点。

**[Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions](beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h.md)**

:   在 Apple Heart and Movement Study 的 162K 参与者、25 亿小时可穿戴行为数据上，系统探索 tokenizer 和架构组合，以 TST+Mamba-2+对比学习构建行为数据基础模型 WBM，在 57 项健康检测任务上显著优于手工特征基线，并与 PPG 传感器模型形成互补。

**[CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries](clarify_contrastive_preference_reinforcement_learning_for_untangling_ambiguous_q.md)**

:   提出 CLARIFY 方法，通过对比学习构建融合偏好信息的轨迹嵌入空间，利用拒绝采样选择更清晰可区分的偏好查询，从而提升离线 PbRL 在非理想反馈下的标注效率和策略性能。

**[ReSA: Clustering Properties of Self-Supervised Learning](clustering_properties_of_self-supervised_learning.md)**

:   系统分析了 JEA-based SSL 中各组件的聚类性质，发现 encoding 比 embedding 和 projector 隐层具有更优更稳定的聚类能力，据此提出 ReSA（Representation Self-Assignment）利用 encoding 的聚类信息引导 embedding 学习，形成正反馈 SSL 框架，在多个标准基准上大幅超越 SOTA。

**[Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)**

:   提出 FALCON 方法，基于超维计算 (hyperdimensional computing) 原理设计投影器和损失函数，理论证明可同时避免四种已知训练失败模式（表示崩塌、维度崩塌、聚类崩塌、簇内崩塌），并使表征自然具备去相关和聚类特性。

**[Contextures: Representations from Contexts](contextures_representations_from_contexts.md)**

:   建立 contexture 理论，统一证明监督学习、自监督学习和流形学习等多种表示学习范式都可被理解为学习上下文变量诱导的期望算子的 top-$d$ 奇异函数，并揭示模型规模增大的边际递减效应以及提出上下文质量评估指标。

**[Deep Learning is Not So Mysterious or Different](deep_learning_is_not_so_mysterious_or_different.md)**

:   本文是一篇 position paper，论证深度学习中被认为"神秘"的泛化现象（良性过拟合、双重下降、过参数化的成功）并非深度学习独有，也不神秘，可以通过长期存在的泛化框架（PAC-Bayes、可数假设界）形式化描述，并提出**软归纳偏置（soft inductive biases）**作为统一解释原则。

**[Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)**

:   提出 GloFND，通过为每个锚点样本学习动态阈值，在训练过程中实时发现并过滤全局假阴性（false negatives），以低额外开销提升对比学习表示质量。

**[Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)**

:   本文系统研究了基础模型（FM）替代传统信息提取器（IE）用于子集选择的优劣，发现 FM 在细粒度数据集上显著优于传统 IE，并提出 RAM-APL 方法，利用多个 FM（DINOv2 + CLIP）从类内和类间两个维度联合衡量样本重要性，在三个细粒度数据集上达到 SOTA。

**[Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)**

:   本文首次在非独立同分布（non-IID）条件下为监督对比表征学习（CRL）建立了泛化界，利用 U-统计量分解技术处理训练元组重叠样本的依赖性问题，给出了以标记样本数 $N$ 为自变量的 excess risk 收敛速率。

**[Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)**

:   Griffin 是首个面向关系数据库（RDB）的基础模型，通过将多表结构转化为异构图，结合统一编码器/解码器、交叉注意力和层级聚合的 MPNN，在 150M+ 行数据上进行自监督掩码补全预训练 + 联合 SFT，实现跨数据库、跨域、跨任务的泛化预测。

**[MTL-UE: Learning to Learn Nothing for Multi-Task Learning](mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)**

:   MTL-UE是首个针对多任务学习的不可学习样本生成框架，通过编码器-解码器架构注入任务特定的类别先验嵌入来降低虚假特征的类内方差，配合任务内/间嵌入余弦正则化增大类间距离和减少冗余，在CelebA(40任务)上将MTL模型准确率从91%降至59%，在4个数据集、3种基础UE方法、5种backbone和5种MTL策略上一致有效。

**[Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)**

:   提出两种基于最近邻的高斯过程先验近似方法（HPA 和 SPA），将近邻驱动的稀疏性引入 GPVAE 的潜空间推断，在保留关键潜变量依赖的同时实现可扩展的 mini-batch 训练，避免了对大量诱导点或受限核函数的依赖。

**[PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)**

:   提出 PDE-Transformer，一种面向物理模拟的改进 Transformer 架构，通过分离通道嵌入、移位窗口注意力和多尺度 U 形结构，在 16 种 PDE 类型上超越现有 SOTA，并展现出强大的下游任务迁移能力。

**[Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting](proxy-fda_proxy-based_feature_distribution_alignment_for_fine-tuning_vision_foun.md)**

:   提出结构级特征正则化方法 Proxy-FDA：通过迁移预训练特征空间的最近邻图到微调特征空间，并用轻量代理生成器合成新特征增强分布覆盖，在不牺牲下游精度的前提下实现所有微调任务的正向迁移。

**[Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)**

:   提出 FoCal 框架，在推理阶段利用 CLIP 和 Stable Diffusion 的视觉先验，通过"变换-排序"策略将输入图像变换为最具视觉典型性的版本，无需重训练即可提升模型对视角、光照、旋转等变换的鲁棒性。

**[Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)**

:   本文从理论上严格证明了测试时训练（TTT）能够可证明地提升 Transformer 的上下文学习（ICL）能力，并在表格基础模型 TabPFN 上验证 TTT 可将所需样本量减少 3-5 倍，同时带来显著的推理效率提升。

**[Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)**

:   首个系统性研究含文本特征的表格数据建模：设计定性反例暴露三类文本嵌入的失败模式，手动策划 13 个真实数据集，发现文本特征在 11/13 数据集上提升预测精度，但无单一最优嵌入方法，表明表格+文本仍是未解决问题。

**[Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)**

:   提出 TransFusion，一种专为 Transformer 设计的两级权重置换方法（头间+头内），实现将旧模型的微调知识（任务向量）免数据免训练地迁移至新版基础模型。

**[What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)**

:   本文提出"归纳偏置探针"（Inductive Bias Probe），通过在合成数据上反复微调基础模型来测试其外推行为是否符合预设的世界模型，发现在轨道力学、Othello、格问题等领域中，基础模型虽然能准确预测序列但未真正学到底层世界模型，而是发展出特定于任务的启发式策略。

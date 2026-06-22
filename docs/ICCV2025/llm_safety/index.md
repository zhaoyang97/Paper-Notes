---
title: >-
  ICCV2025 LLM安全论文汇总 · 10篇论文解读
description: >-
  10篇ICCV2025的 LLM 安全方向论文解读，涵盖联邦学习、对抗鲁棒、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "对抗鲁棒"
  - "多模态"
item_list:
  - u: "adversarial_robust_memory-based_continual_learner/"
    t: "Adversarial Robust Memory-Based Continual Learner"
  - u: "asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset/"
    t: "Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset"
  - u: "cooperative_pseudo_labeling_for_unsupervised_federated_classification/"
    t: "Cooperative Pseudo Labeling for Unsupervised Federated Classification"
  - u: "enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_/"
    t: "Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling"
  - u: "forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re/"
    t: "Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation"
  - u: "geminio_language-guided_gradient_inversion_attacks_in_federated_learning/"
    t: "Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning"
  - u: "latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_/"
    t: "LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning"
  - u: "munba_machine_unlearning_via_nash_bargaining/"
    t: "MUNBa: Machine Unlearning via Nash Bargaining"
  - u: "sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc/"
    t: "SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders"
  - u: "temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e/"
    t: "Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📹 ICCV2025** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (184)](../../ICLR2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [🧪 ICML2025 (41)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** 联邦学习 ×4 · 对抗鲁棒 ×3 · 多模态 ×2

**[Adversarial Robust Memory-Based Continual Learner](adversarial_robust_memory-based_continual_learner.md)**

:   揭示持续学习与对抗训练结合时的双重挑战（加速遗忘 + 梯度混淆），提出抗遗忘 Logit 校准（AFLC）和鲁棒感知经验回放（RAER）两个即插即用模块，在 Split-CIFAR10/100 和 Split-Tiny-ImageNet 上有效提升对抗鲁棒性达 8.13%。

**[Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)**

:   提出首个面向异步事件数据的不可学习样本生成方法（UEvs），设计了事件误差最小化噪声（E²MN）及自适应投影机制，使事件数据集在保持合法使用功能的同时阻止未授权模型从中学习。

**[Cooperative Pseudo Labeling for Unsupervised Federated Classification](cooperative_pseudo_labeling_for_unsupervised_federated_classification.md)**

:   FedCoPL 首次将无监督联邦学习扩展到分类任务，通过协作伪标签策略（全局分配伪标签确保类别平衡）和部分 prompt 聚合协议（仅聚合视觉 prompt、保留文本 prompt 本地化）有效应对 CLIP 固有偏差和标签偏移挑战。

**[Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)**

:   提出Gradient-Guided Sampling (GGS)内迭代采样策略，通过使用上一内迭代的梯度方向引导采样，在平衡Exploitation（攻击强度/损失极大值）和Exploration（跨模型泛化/平坦损失面）的困境中取得突破，在CNN/ViT/MLLM等多架构上显著超越现有迁移攻击方法。

**[Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)**

:   提出 FUCRT 方法，通过类感知表征变换实现联邦遗忘：将遗忘类的表征“变换”到语义最近的保留类，而非直接消除，配合双重对比学习对齐跨客户端的变换一致性，在四个数据集上实现 100% 遗忘保障的同时保持甚至提升剩余类性能。

**[Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)**

:   本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。

**[LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)**

:   提出 Latte 框架，在联邦学习的去中心化场景下，通过本地记忆与外部记忆的协同机制，实现视觉语言模型（如 CLIP）的协作式测试时自适应，兼顾跨客户端知识共享与个性化。

**[MUNBa: Machine Unlearning via Nash Bargaining](munba_machine_unlearning_via_nash_bargaining.md)**

:   将机器遗忘（Machine Unlearning）建模为双玩家合作博弈问题，利用 Nash 讨价还价理论推导闭式解来同时解决遗忘目标与保留目标之间的梯度冲突和梯度支配问题，在分类和生成任务上实现遗忘与保留的最优平衡。

**[SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)**

:   SAUCE 利用稀疏自编码器（SAE）在 VLM 的中间表征中识别并选择性抑制与目标概念相关的特征，实现了无需权重更新的细粒度概念遗忘，在 60 个概念的测试中遗忘质量超越 SOTA 18%。

**[Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)**

:   本文首次研究防止视频数据被深度跟踪器未授权使用的问题，提出基于 DiT 的生成式框架生成时序不可学习样本（TUE），通过时间对比损失使跟踪器依赖扰动噪声进行时序匹配而非学习真实数据结构，实现了跨模型、跨数据集和跨任务的强可迁移性。

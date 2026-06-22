---
title: >-
  CVPR2025 AI安全论文汇总 · 27篇论文解读
description: >-
  27篇CVPR2025的 AI 安全方向论文解读，涵盖对抗鲁棒、联邦学习、人脸/视线、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "联邦学习"
  - "人脸/视线"
  - "对齐/RLHF"
item_list:
  - u: "a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning/"
    t: "A Simple Data Augmentation for Feature Distribution Skewed Federated Learning"
  - u: "data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior/"
    t: "Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior"
  - u: "deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging/"
    t: "DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging"
  - u: "dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders/"
    t: "DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders"
  - u: "detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect/"
    t: "Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection"
  - u: "detecting_out-of-distribution_through_the_lens_of_neural_collapse/"
    t: "Detecting Out-of-Distribution through the Lens of Neural Collapse"
  - u: "dynamic_integration_of_task-specific_adapters_for_class_incremental_learning/"
    t: "Dynamic Integration of Task-Specific Adapters for Class Incremental Learning"
  - u: "fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_/"
    t: "FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors"
  - u: "forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection/"
    t: "Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection"
  - u: "geometric_knowledge-guided_localized_global_distribution_alignment_for_federated/"
    t: "Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning"
  - u: "gradient_inversion_attacks_on_parameter-efficient_fine-tuning/"
    t: "Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning"
  - u: "h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection/"
    t: "H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection"
  - u: "infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning/"
    t: "Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning"
  - u: "invisible_backdoor_attack_against_self-supervised_learning/"
    t: "INACTIVE: Invisible Backdoor Attack against Self-supervised Learning"
  - u: "joint_out-of-distribution_filtering_and_data_discovery_active_learning/"
    t: "Joint Out-of-Distribution Filtering and Data Discovery Active Learning"
  - u: "leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection/"
    t: "Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection"
  - u: "lyapunov_stable_graph_neural_flow/"
    t: "Lyapunov Stable Graph Neural Flow"
  - u: "mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query/"
    t: "Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis"
  - u: "mos-attack_a_scalable_multi-objective_adversarial_attack_framework/"
    t: "MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework"
  - u: "not_federated_unlearning_via_weight_negation/"
    t: "NoT: Federated Unlearning via Weight Negation"
  - u: "oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary/"
    t: "OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary"
  - u: "psbd_prediction_shift_uncertainty_unlocks_backdoor_detection/"
    t: "PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection"
  - u: "split_adaptation_for_pre-trained_vision_transformers/"
    t: "Split Adaptation for Pre-trained Vision Transformers"
  - u: "stacking_brick_by_brick_aligned_feature_isolation_for_incremental_face_forgery_d/"
    t: "Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection"
  - u: "towards_general_visual-linguistic_face_forgery_detection/"
    t: "Towards General Visual-Linguistic Face Forgery Detection"
  - u: "towards_source-free_machine_unlearning/"
    t: "Towards Source-Free Machine Unlearning"
  - u: "where_the_devil_hides_deepfake_detectors_can_no_longer_be_trusted/"
    t: "Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted"
item_total: 27
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**📷 CVPR2025** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (145)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/ai_safety/index.md) · [💬 ACL2026 (5)](../../ACL2026/ai_safety/index.md) · [🧪 ICML2026 (114)](../../ICML2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×9 · 联邦学习 ×6 · 人脸/视线 ×3 · 对齐/RLHF ×2

**[A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)**

:   提出FedRDN——一种极其简单的联邦学习数据增强方法，在训练时随机使用其他客户端的通道级均值/标准差做数据归一化（而非固定用本地统计），仅需几行代码即可显著缓解特征分布偏移问题，在多种FL方法上一致提升性能。

**[Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior](data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior.md)**

:   提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。

**[DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging](deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)**

:   提出 DEAL（Data-Efficient Adversarial Learning），一种仅需 50 张清晰红外图像训练的对抗学习框架，通过动态对抗退化合成和双通道交互网络（Scale Transform + Spiking Neurons），以 0.96M 超轻量参数同时处理条纹噪声、低分辨率和低对比度三种红外退化。

**[DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)**

**[Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)**

:   提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。

**[Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)**

:   从 Neural Collapse 理论出发，发现中心化后的 ID 特征聚集在预测类别的权重向量附近且远离原点（形成 simplex ETF），据此设计 NCI 检测器——结合特征与权重向量的角度近邻度（pScore）和特征范数过滤，在 CIFAR-10/100 和 ImageNet 多架构上实现最佳综合 OOD 检测性能且推理延迟与 softmax 基线持平。

**[Dynamic Integration of Task-Specific Adapters for Class Incremental Learning](dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)**

:   通过动态集成任务特定适配器实现类增量学习，每个任务训练轻量适配器，推理时动态选择和组合相关适配器

**[FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_.md)**

:   提出 FedAWA，受任务算术（task arithmetic）启发，用客户端向量（本地参数与全局参数的差值）来自适应优化联邦学习中的聚合权重——与全局优化方向一致的客户端获得更高权重，在 non-IID 场景下稳定提升 FedAvg 1-4 个点。

**[Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)**

:   提出 Forensics Adapter，一个仅 5.7M 参数的轻量适配器网络，与冻结 CLIP 并行学习人脸伪造的融合边界特征，通过掩码边界预测+逐块对比+样本级对比三重目标实现跨数据集的高泛化性人脸伪造检测，CDF-v1 上 AUC 达 0.914。

**[Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)**

:   在联邦学习中通过从局部协方差矩阵精确重建全局协方差来获取全局嵌入分布的几何形状，沿全局主方向生成增强样本本地化全局分布信息，在 CIFAR-100 极端异质场景（β=0.01）下提升 17 个百分点。

**[Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning](gradient_inversion_attacks_on_parameter-efficient_fine-tuning.md)**

:   首次证明 Adapter-based PEFT 在联邦学习中不是隐私安全的——恶意服务器可以将预训练模型设计为恒等映射使 patch embedding 原样传播到 adapter 层，从 adapter 梯度中解析式恢复训练图像（CIFAR-100 SSIM 0.88）。

**[H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)**

:   提出H2ST方法，用层次化的两样本检验架构实现增量学习中的OOD检测——每个任务对应一个特征级别的源-目标二分类器层，通过Clopper-Pearson置信区间假设检验自动判定ID/OOD（无需手动阈值），同时提供任务ID预测能力，在7个基准上优于MSP/Energy/ODIN且计算效率提升$(T+1)/2$倍。

**[Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)**

:   本文首次研究了联邦学习中非合作多标签后门攻击(MBA)场景，揭示了现有单标签后门攻击方法扩展到多标签场景时因构建相似的分布外(OOD)映射而导致攻击者间相互排斥的内在缺陷，提出 Mirage 方法通过构建分布内(ID)后门映射，使多个攻击者可以独立且持久地植入后门，平均攻击成功率超过97%且在900轮后仍保持90%以上。

**[INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)**

:   提出 INACTIVE，首个对自监督学习（SSL）有效的不可见后门攻击——通过在 HSV/HSL 色彩空间中设计触发器以逃离 SSL 数据增强的分布空间，实现 99.09% 平均攻击成功率，同时保持 SSIM 0.9763/PSNR 41.07dB 的高隐蔽性，抵抗 7 种防御方法。

**[Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)**

:   提出 Open-Set Discovery Active Learning (OSDAL) 场景，并设计 Joda 算法，通过训练-过滤-选择三阶段流程，用单一模型同时过滤 OOD 数据和发现新类别，无需额外辅助模型，在 18 个配置上持续达到最高准确率。

**[Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)**

:   发现 OOD 样本的检测得分比 IND 样本更容易被对抗扰动降低，提出 PRO 方法——在推理时用梯度下降搜索 ε-球内的最小 OOD 得分，增强 IND/OOD 可分性，在 CIFAR-10 上 FPR@95 从 44.35% 降至 19.95%。

**[Lyapunov Stable Graph Neural Flow](lyapunov_stable_graph_neural_flow.md)**

:   将 Lyapunov 稳定性理论（整数阶和分数阶）与图神经流集成，通过可学习 Lyapunov 函数和投影机制将 GNN 特征动态约束在稳定空间中，首次为图神经流提供可证明的对抗鲁棒性保证，且与对抗训练正交可叠加。

**[Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)**

:   本文提出了一种基于查询更新模式(而非输入模式)的黑盒对抗攻击检测框架 GWAD，引入 Delta Similarity 指标来捕获基于查询的攻击中零阶优化的固有模式，在8种SOTA攻击(包括自适应攻击OARS)上实现了接近100%的检测率且误报率极低，显著优于现有的状态化防御方法。

**[MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)**

:   提出MOS Attack框架，将对抗攻击建模为多目标集合优化问题，结合smooth max/min近似实现多损失函数联合优化，并自动发现损失函数间的协同模式，在CIFAR-10和ImageNet上超越现有SOTA单目标攻击和集成攻击。

**[NoT: Federated Unlearning via Weight Negation](not_federated_unlearning_via_weight_negation.md)**

:   提出 NoT 算法，通过对全局模型特定层的权重乘以 -1（取反）来破坏层间协同适应从而实现遗忘，再用保留数据微调恢复性能，无需额外存储或访问目标数据，在 CIFAR-10/100、Caltech-101 上以最低通信/计算开销显著优于七种基线方法。

**[OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)**

:   提出 OODD，通过优先队列维护动态 OOD 字典在测试时实时收集潜在 OOD 样本特征来校准 OOD 分数，在 CIFAR-100 Far OOD 上相比 SOTA 方法 FPR95 降低 26.0%，且无需微调。

**[PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection](psbd_prediction_shift_uncertainty_unlocks_backdoor_detection.md)**

:   提出 PSBD 方法，发现被植入后门的模型在推理时开启 dropout 后，干净数据的预测会偏移向目标类别而后门数据预测保持稳定（Prediction Shift 现象），基于此设计 Prediction Shift Uncertainty (PSU) 指标实现 SOTA 后门训练数据检测。

**[Split Adaptation for Pre-trained Vision Transformers](split_adaptation_for_pre-trained_vision_transformers.md)**

:   本文提出 Split Adaptation (SA)，将预训练 ViT 分割为前端（量化后发送给客户端）和后端（留在服务器），通过双层噪声注入保护数据隐私，配合OOD增强和patch检索增强缓解噪声影响和过拟合，在保护模型和数据的前提下实现高效少样本下游适配。

**[Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection](stacking_brick_by_brick_aligned_feature_isolation_for_incremental_face_forgery_d.md)**

:   提出 SUR-LID 方法解决增量人脸伪造检测 (IFFD) 中的灾难性遗忘问题：通过稀疏均匀回放 (SUR) 保留旧任务的全局特征分布，通过隐空间增量检测器 (LID) 中的特征隔离和决策对齐策略将新旧任务分布"逐块堆叠"而非相互覆盖。

**[Towards General Visual-Linguistic Face Forgery Detection](towards_general_visual-linguistic_face_forgery_detection.md)**

:   VLFFD 提出了一种视觉-语言范式的深度伪造检测方法，通过 Prompt Forgery Image Generator (PFIG) 自动生成带有细粒度文本描述的混合伪造图像，再用 Coarse-and-Fine Co-training (C2F) 框架联合训练粗粒度和细粒度数据，显著提升了检测模型的泛化性和可解释性。

**[Towards Source-Free Machine Unlearning](towards_source-free_machine_unlearning.md)**

:   本文提出了一种无源机器遗忘（Source-Free Machine Unlearning）算法，在无法获取原始训练数据的条件下，通过近似估计保留数据的 Hessian 矩阵（仅使用待遗忘数据和训练好的模型），实现了对线性和混合线性分类器的高效遗忘，并提供了严格的理论上界保证。

**[Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted](where_the_devil_hides_deepfake_detectors_can_no_longer_be_trusted.md)**

:   揭示了 Deepfake 检测器面临的严重安全风险——第三方数据提供者可以通过注入密码控制的、自适应的、不可见的触发器来植入后门，使被污染的检测器在遇到带特定触发器的样本时产生错误判断，同时在正常样本上保持正常性能。支持 dirty-label 和 clean-label 两种攻击场景。

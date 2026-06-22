---
title: >-
  CVPR2026 优化/理论论文汇总 · 22篇论文解读
description: >-
  22篇CVPR2026的优化/理论方向论文解读，涵盖联邦学习、压缩/编码、扩散模型、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "压缩/编码"
  - "扩散模型"
  - "对抗鲁棒"
item_list:
  - u: "ace-merging_data-free_model_merging_with_adaptive_covariance_estimation/"
    t: "ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation"
  - u: "bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea/"
    t: "BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning"
  - u: "beyond_single_solution_multi-hypothesis_collaborative_deep_unfolding_network_for/"
    t: "Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing"
  - u: "conditional_factuality_controlled_llms_with_generalization_certificates_via_conf/"
    t: "Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling"
  - u: "dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors/"
    t: "DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors"
  - u: "dc-merge_improving_model_merging_with_directional_consistency/"
    t: "DC-Merge: Improving Model Merging with Directional Consistency"
  - u: "defending_unauthorized_model_merging_via_dual-stage_weight_protection/"
    t: "Defending Unauthorized Model Merging via Dual-Stage Weight Protection"
  - u: "dynamic_momentum_recalibration_in_online_gradient_learning/"
    t: "Dynamic Momentum Recalibration in Online Gradient Learning"
  - u: "end-to-end_hyper-relational_information_extraction_for_engineering_diagrams_via_/"
    t: "End-to-End Hyper-Relational Information Extraction for Engineering Diagrams via Dynamically Tokenized Relation Transformer"
  - u: "enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p/"
    t: "Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning"
  - u: "fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_/"
    t: "Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift"
  - u: "few-for-many_personalized_federated_learning/"
    t: "Few-for-Many Personalized Federated Learning"
  - u: "globscope_toward_a_global_view_of_the_loss_landscape/"
    t: "Globscope: Toward a Global View of the Loss Landscape"
  - u: "gr-gauge_cost-efficient_training_configuration_by_gauging_the_gradient_redundanc/"
    t: "GR-Gauge: Cost-efficient Training Configuration By Gauging the Gradient Redundancy"
  - u: "hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r/"
    t: "HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation"
  - u: "label-free_cross-task_lora_merging_with_null-space_compression/"
    t: "Label-Free Cross-Task LoRA Merging with Null-Space Compression"
  - u: "learning_to_learn_weight_generation_via_local_consistency_diffusion/"
    t: "Learning to Learn Weight Generation via Local Consistency Diffusion"
  - u: "mapping_networks/"
    t: "Mapping Networks"
  - u: "model_merging_in_the_essential_subspace/"
    t: "Model Merging in the Essential Subspace"
  - u: "semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score/"
    t: "Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score"
  - u: "the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s/"
    t: "The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers"
  - u: "unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source/"
    t: "UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**📷 CVPR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md) · [🧪 ICML2025 (61)](../../ICML2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×4 · 压缩/编码 ×2 · 扩散模型 ×2 · 对抗鲁棒 ×2

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)**

:   提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。

**[Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing](beyond_single_solution_multi-hypothesis_collaborative_deep_unfolding_network_for.md)**

:   针对压缩感知（CS）问题"欠定、解不唯一"的本质，本文提出 MHC-DUN：把传统深度展开网络（DUN）里"只重建一个解"的范式扩展成"同时重建 $T$ 个假设解并让它们协同优化"，在梯度下降步用 AlphaNet 给每个假设预测逐像素自适应步长、在近端映射步用 MHCB 挖掘假设间相关性融合，在 Set11/Urban100/CS-MRI 上全面超过现有 SOTA（Set11 平均 PSNR 比 USB-Net 高 0.45 dB）。

**[Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)**

:   提出 CFC（Conditional Factuality Control），一种后验保形框架，通过增广分位数回归学习特征条件化的接受阈值，为LLM/VLM采样输出提供条件覆盖率保证，在保持紧凑预测集的同时显著改善难题子群的可靠性。

**[DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)**

:   DABO 把"优化难度"作为一等条件变量贯穿整条 freeze-thaw 超参数优化流水线——用三层难度刻画 + 条件扩散模型生成 100 万条带难度标注的合成学习曲线，训出难度感知的 PFN 代理与自适应采集函数，在 75 个任务上比当前 SOTA（ifBO）平均降低 11–18% 的 regret，且越难的任务收益越大。

**[DC-Merge: Improving Model Merging with Directional Consistency](dc-merge_improving_model_merging_with_directional_consistency.md)**

:   DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。

**[Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)**

:   提出 MergeGuard，一种主动式双阶段权重保护框架：Stage 1通过L2正则化分散任务关键权重，Stage 2注入结构化扰动破坏合并兼容性，在保持保护模型<1.5%性能损失的同时使合并模型精度下降高达90%。

**[Dynamic Momentum Recalibration in Online Gradient Learning](dynamic_momentum_recalibration_in_online_gradient_learning.md)**

:   从信号处理视角揭示固定动量系数在偏差-方差权衡上的固有缺陷，提出SGDF优化器，通过在线计算最优时变增益（基于最小均方误差原则）动态平衡梯度估计的噪声抑制和信号保持，在多种视觉任务上超越SGD动量和Adam变体。

**[End-to-End Hyper-Relational Information Extraction for Engineering Diagrams via Dynamically Tokenized Relation Transformer](end-to-end_hyper-relational_information_extraction_for_engineering_diagrams_via_.md)**

:   把工程图纸（管道仪表图 P&ID、电气图 ED）的解析从"多模型分别检测符号/线/文字"重构成一次性的场景图生成任务，用一个带动态令牌剪枝的视觉主干 + 一阶段关系 Transformer（DTRT）端到端输出"实体 + 连接关系 + 文字限定词"的超关系知识图谱，在 P&ID 上 SGDET R@2000 达 94.84%、计算量却只有两阶段方法的约 1/8。

**[Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)**

:   针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。

**[Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)**

:   提出 Fed-ADE 框架，通过 uncertainty dynamics estimation 和 representation dynamics estimation 两个轻量级分布漂移信号，为每个客户端在每个时间步自适应调整学习率，实现联邦部署后无监督适应。

**[Few-for-Many Personalized Federated Learning](few-for-many_personalized_federated_learning.md)**

:   把个性化联邦学习重构成"用 K 个共享模型服务 M 个客户端"（K≪M）的少对多多目标优化问题，并用可微的平滑 Tchebycheff 集合标量化（STCH-Set）联合训练这 K 个模型，只用 3 个模型就在视觉、NLP 和真实医学影像上稳定超过现有方法。

**[Globscope: Toward a Global View of the Loss Landscape](globscope_toward_a_global_view_of_the_loss_landscape.md)**

:   用一个**可逆的自编码器**把一堆独立训练好的网络（每个模型展平成参数向量）压进二维潜空间，再在这个潜空间上把"loss"当作标量场做拓扑分析（merge tree），第一次给出能同时容纳**多个极小值/盆地及其连通关系**的全局损失景观可视化，并用它复现 mode connectivity 与置换对称（re-basin）等理论现象。

**[GR-Gauge: Cost-efficient Training Configuration By Gauging the Gradient Redundancy](gr-gauge_cost-efficient_training_configuration_by_gauging_the_gradient_redundanc.md)**

:   把模型训练看成"梯度在时间和样本两个维度上的投票过程"，提出梯度冗余度指标 GRT/GRS 作为一把跨模型通用的"健康度尺子"，用它来指导学习率和 batch size 的超参搜索、早停和状态复用，从而在不跑昂贵验证集的前提下把达到目标精度的总时间最多压掉 80%+。

**[HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r.md)**

:   本文首次形式化「分层联邦域泛化（HFedDG）」并推导出按客户端/中转站/服务器三级分解的泛化误差界，据此提出 HFedATM——一个无需访问数据、只改服务器聚合步骤的即插即用方法：先用 Filter-wise Optimal Transport 把各中转站的卷积滤波器对齐，再用 Shrinkage-aware RegMean 闭式融合线性层，在视觉与 NLP 基准上稳定提升 FedAvg/FedProx/FedSR/FedIIR 等多种基线。

**[Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)**

:   观察到LoRA微调过程中下投影矩阵A的零空间比率随训练下降且与性能强相关，据此提出NSC Merging，一种无标签、任务无关的LoRA合并方法，在20个异构视觉任务、6个NLI任务和VLM评估上达到SOTA。

**[Learning to Learn Weight Generation via Local Consistency Diffusion](learning_to_learn_weight_generation_via_local_consistency_diffusion.md)**

:   Mc-Di 把元学习的双层优化和扩散式权重生成结合起来，并把原本只学"全局最优权重"的扩散过程改造成"局部一致性扩散"——让模型沿优化轨迹上的多个中间权重逐段重建，从而在迁移学习、小样本、域泛化、语言模型微调等需要频繁更新权重的任务上同时拿到更高精度和更低推理延迟。

**[Mapping Networks](mapping_networks.md)**

:   本文提出 Mapping Networks——一种"元参数化"方法，用一个低维可训练隐向量 $z$（配合固定的、被 $z$ 调制的映射权重）来生成目标网络的全部参数，从而把训练从高维权重空间搬到低维隐空间，在图像分类、deepfake 检测、分割等任务上以约 500× 更少的可训练参数达到甚至超过原网络的精度，同时显著抑制过拟合。

**[Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)**

:   提出 ESM 框架，通过对参数更新引起的激活偏移做 PCA 构建"本质子空间"（而非直接对参数做 SVD），并用三级极化缩放增强关键参数、抑制噪声，在 ViT-B/32 的 20 任务合并中比 Iso-CTS 提升 3.2%（绝对准确率）。

**[Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)**

:   提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

**[The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)**

:   将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。

**[UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)**

:   提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。

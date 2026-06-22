---
title: >-
  ICML2025 医学图像论文汇总 · 21篇论文解读
description: >-
  21篇ICML2025的医学图像方向论文解读，涵盖医学影像、语义分割、多模态、目标检测、域适应、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "语义分割"
  - "多模态"
  - "目标检测"
  - "域适应"
  - "推理"
item_list:
  - u: "bayesian_inference_for_correlated_human_experts_and_classifiers/"
    t: "Bayesian Inference for Correlated Human Experts and Classifiers"
  - u: "boosting_masked_ecg-text_auto-encoders_as_discriminative_learners/"
    t: "Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)"
  - u: "certification_for_differentially_private_prediction_in_gradient-based_training/"
    t: "Certification for Differentially Private Prediction in Gradient-Based Training"
  - u: "context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images/"
    t: "Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images"
  - u: "do_multiple_instance_learning_models_transfer/"
    t: "Do Multiple Instance Learning Models Transfer?"
  - u: "eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping/"
    t: "EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping"
  - u: "efficient_noise_calculation_in_deep_learning-based_mri_reconstructions/"
    t: "Efficient Noise Calculation in Deep Learning-based MRI Reconstructions"
  - u: "enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza/"
    t: "Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing"
  - u: "from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining/"
    t: "From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining"
  - u: "i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts/"
    t: "I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts"
  - u: "idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio/"
    t: "iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection"
  - u: "implementing_adaptations_for_vision_autoregressive_model/"
    t: "Implementing Adaptations for Vision AutoRegressive Model"
  - u: "langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me/"
    t: "LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation"
  - u: "mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees/"
    t: "Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees"
  - u: "medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding/"
    t: "MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding"
  - u: "neural_stochastic_differential_equations_on_compact_state_spaces_theory_methods_/"
    t: "Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods and Applications"
  - u: "raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain/"
    t: "Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models"
  - u: "sgd_jittering_a_training_strategy_for_robust_and_accurate_model-based_architectu/"
    t: "SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures"
  - u: "the_brains_bitter_lesson_scaling_speech_decoding_with_self-supervised_learning/"
    t: "The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning"
  - u: "the_disparate_benefits_of_deep_ensembles/"
    t: "The Disparate Benefits of Deep Ensembles"
  - u: "the_four_color_theorem_for_cell_instance_segmentation/"
    t: "The Four Color Theorem for Cell Instance Segmentation"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🧪 ICML2025** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (172)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (86)](../../ICLR2026/medical_imaging/index.md) · [🧪 ICML2026 (28)](../../ICML2026/medical_imaging/index.md) · [🤖 AAAI2026 (75)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (77)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×6 · 语义分割 ×2

**[Bayesian Inference for Correlated Human Experts and Classifiers](bayesian_inference_for_correlated_human_experts_and_classifiers.md)**

:   提出通用贝叶斯框架来建模相关人类专家和分类器之间的联合标注行为，通过潜在表示捕捉专家间相关性，用模拟推断评估额外查询的效用，在医学分类和图像标注中大幅减少专家查询次数同时保持预测准确率。

**[Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)](boosting_masked_ecg-text_auto-encoders_as_discriminative_learners.md)**

:   D-BETA 提出了一种融合生成式掩码自编码器与增强判别能力的对比学习框架，通过 ECG-Text Sigmoid (ETS) 损失和最近邻负采样策略 (N3S)，在 ECG-文本跨模态表征学习中显著超越现有方法，在仅用 1% 训练数据的线性探测中平均 AUC 提升 15%，零样本性能提升 2%。

**[Certification for Differentially Private Prediction in Gradient-Based Training](certification_for_differentially_private_prediction_in_gradient-based_training.md)**

:   提出 Abstract Gradient Training (AGT) 框架，通过凸松弛与界传播技术计算训练过程中模型参数的可达集上界，从而利用平滑敏感度机制大幅收紧隐私预测的隐私分析，在医学影像和 NLP 任务上实现比全局敏感度紧数个数量级的隐私界。

**[Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)**

:   提出 Querent 框架——通过 query-aware 的动态区域重要性评估实现千亿像素全切片图像（WSI）中的高效长程上下文建模，在理论上有界逼近完整自注意力，在 10+ 个 WSI 数据集的生物标志物预测/基因突变预测/癌症分型/生存分析中超越 SOTA。

**[Do Multiple Instance Learning Models Transfer?](do_multiple_instance_learning_models_transfer.md)**

:   首次系统评估计算病理学中 MIL 模型的迁移学习能力，发现在 pancancer 数据集上预训练的 MIL 模型能够跨器官、跨任务泛化，以不到 10% 的预训练数据超越自监督 slide foundation model（CHIEF、GigaPath）。

**[EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping](eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping.md)**

:   本文首创 EEG-语言模型（ELM），在15000份EEG记录和临床报告上训练，结合时间序列裁剪、文本分割和多实例学习策略，首次实现了EEG的零样本分类和跨模态检索，在低标注场景下病理检测性能显著优于纯EEG自监督方法。

**[Efficient Noise Calculation in Deep Learning-based MRI Reconstructions](efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)**

:   提出基于 Jacobian Sketching 的高效方法，通过随机相向量探测 DL 重建网络的 Jacobian 对角元，以无偏估计加速 MRI 重建中的体素级噪声方差，计算和内存需求降低一个数量级以上，与 Monte Carlo 参考相关系数达 99.8%。

**[Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing](enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)**

:   提出基于 Fisher 随机化检验（FRT）+ 保形选择性借用（CSB）的混合对照试验推断框架，实现有限样本精确的 I 类错误率控制和模型无关的统计推断，通过自适应阈值最小化 MSE，在保持严格 I 类错误控制的同时提升检验功效。

**[From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)**

:   MELP 提出了一种多尺度 ECG-语言预训练模型，通过 Token/Beat/Rhythm 三个层次的跨模态监督信号，结合心脏学专业语言模型预训练，在零样本分类、线性探测和迁移学习中全面超越现有 ECG 自监督和多模态方法。

**[I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)**

:   I2MoE 提出了一种可解释的多模态交互感知混合专家框架，通过四种交互专家（唯一性×2 + 协同 + 冗余）结合弱监督交互损失显式建模模态间的异质交互，并通过重加权模型提供样本级和数据集级的可解释性，在 ADNI 数据集上提升准确率 5.5%。

**[iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection](idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio.md)**

:   提出 iDPA 框架，通过实例级 Prompt 生成（IPG）和解耦 Prompt 注意力（DPA）两大模块，在冻结的视觉-语言目标检测模型上实现增量医学目标检测（IMOD），仅训练 1.4% 的参数即在 13 个跨模态医学数据集上全面超越 SOTA。

**[Implementing Adaptations for Vision AutoRegressive Model](implementing_adaptations_for_vision_autoregressive_model.md)**

:   本文首次系统实现并评测了Vision AutoRegressive（VAR）模型的各种适配方法（FFT/LoRA/LNTuning）及差分隐私适配，发现VAR在非DP场景下显著超越扩散模型适配（DiffFit），收敛速度更快、计算效率更高，但DP适配性能仍然不佳，揭示了隐私保护图像生成领域的重要研究空白。

**[LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation](langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)**

:   LangDAug 利用基于对比散度训练的能量模型(EBM)，通过 Langevin 动力学在源域之间遍历生成中间样本，实现医学图像分割的多源域泛化，理论证明其诱导正则化效果并上界 Rademacher 复杂度。

**[Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees](mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)**

:   本文为多专家路由(learning to defer)问题提出了新的代理损失函数和高效算法，建立了可实现 H-一致性、H-一致性界和 Bayes 一致性的理论保证，覆盖单阶段和两阶段两种学习场景。

**[MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding](medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding.md)**

:   MedXpertQA 构建了包含 4460 题、覆盖 17 个专科和 11 个身体系统的专家级医学 QA 基准，通过严格的筛选增强和数据合成防泄漏，评估了 18 个主流模型，并专门设计了推理子集用于评估 o1 类推理模型。

**[Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods and Applications](neural_stochastic_differential_equations_on_compact_state_spaces_theory_methods_.md)**

:   本文提出基于随机生存理论的神经 SDE 参数化方法 (WSP)，确保 SDE 轨迹可证明地约束在紧多面体空间内，具有连续动力学和良好归纳偏置，克服了 chain-rule 方法和反射 SDE 的缺陷。

**[Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)**

:   提出 Raptor（Random Planar Tensor Reduction），一种完全免训练的方法，利用冻结的 2D 基础模型（DINOv2-L）对 3D 医学体积沿三轴提取视觉 token，再通过随机投影大幅压缩维度，在 10 个医学任务上超越所有需要大规模预训练的 SOTA 方法。

**[SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures](sgd_jittering_a_training_strategy_for_robust_and_accurate_model-based_architectu.md)**

:   提出 SGD jittering 训练策略，在模型迭代重建过程中逐步注入零均值高斯噪声，理论证明其同时提升模型鲁棒性和泛化精度，且无需对抗训练的高计算开销。

**[The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning](the_brains_bitter_lesson_scaling_speech_decoding_with_self-supervised_learning.md)**

:   开发神经科学启发的自监督 pretext 任务和异构脑信号处理架构，将 MEG 语音解码扩展至约 400 小时/900 名被试，超越 SOTA 15-27%，首次以非侵入式数据匹配手术级解码性能，并展现跨数据集、跨被试、跨任务的泛化能力。

**[The Disparate Benefits of Deep Ensembles](the_disparate_benefits_of_deep_ensembles.md)**

:   这篇论文通过对人脸分析与医学影像数据集的大规模实证研究，揭示了一个被忽视的现象——"差异化收益效应"（disparate benefits effect）：深度集成（Deep Ensembles）在提升整体性能的同时，会**不均衡地**惠及不同受保护群体（往往偏向本就占优势的群体），从而损害群体公平；作者进一步指出其根因是**群体间预测多样性的差异**，并证明经典的 Hardt 后处理（HPP）能在保住性能增益的前提下有效修复公平。

**[The Four Color Theorem for Cell Instance Segmentation](the_four_color_theorem_for_cell_instance_segmentation.md)**

:   将四色定理引入细胞实例分割，将每个细胞视为"国家"、背景为"海洋"，用仅 4 类语义分割替代实例分割，并设计渐进训练策略和编码变换方法解决四色编码的非唯一性问题，在多种成像模式上达到 SOTA 性能同时大幅降低模型复杂度。

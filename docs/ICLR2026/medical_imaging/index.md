---
title: >-
  ICLR2026 医学图像论文汇总 · 22篇论文解读
description: >-
  22篇ICLR2026的医学图像方向论文解读，涵盖医学影像、扩散模型、语义分割等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "扩散模型"
  - "语义分割"
item_list:
  - u: "adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation/"
    t: "Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation"
  - u: "biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo/"
    t: "Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces"
  - u: "boosting_medical_visual_understanding_from_multi-granular_language_learning/"
    t: "Boosting Medical Visual Understanding From Multi-Granular Language Learning"
  - u: "brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer/"
    t: "Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer"
  - u: "brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled/"
    t: "Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model"
  - u: "care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev/"
    t: "CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework"
  - u: "characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i/"
    t: "Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space"
  - u: "compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics/"
    t: "COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics"
  - u: "disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo/"
    t: "DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring"
  - u: "distributional_consistency_loss_beyond_pointwise_data_terms_in_inverse_problems/"
    t: "Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems"
  - u: "dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction/"
    t: "DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction"
  - u: "glance_and_focus_reinforcement_for_pan-cancer_screening/"
    t: "Glance and Focus Reinforcement for Pan-cancer Screening"
  - u: "heegnet_hyperbolic_embeddings_for_eeg/"
    t: "HEEGNet: Hyperbolic Embeddings for EEG"
  - u: "improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent/"
    t: "Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity"
  - u: "inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati/"
    t: "Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification"
  - u: "lavca_llm-assisted_visual_cortex_captioning/"
    t: "LaVCa: LLM-assisted Visual Cortex Captioning"
  - u: "learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu/"
    t: "Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation"
  - u: "neuro-symbolic_decoding_of_neural_activity/"
    t: "Neuro-Symbolic Decoding of Neural Activity"
  - u: "neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain/"
    t: "NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification"
  - u: "q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin/"
    t: "Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering"
  - u: "seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding/"
    t: "SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding"
  - u: "towards_interpretable_visual_decoding_with_attention_to_brain_representations/"
    t: "Towards Interpretable Visual Decoding with Attention to Brain Representations"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🔬 ICLR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (176)](../../CVPR2026/medical_imaging/index.md) · [🧪 ICML2026 (24)](../../ICML2026/medical_imaging/index.md) · [🤖 AAAI2026 (75)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (77)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md) · [🧪 ICML2025 (21)](../../ICML2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×5 · 扩散模型 ×3 · 语义分割 ×2

**[Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)**

:   提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

**[Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces](biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo.md)**

:   提出一种无需BPTT的在线SNN解码器，通过三因子Hebbian局部学习规则结合双时间尺度eligibility trace和自适应学习率控制，在O(1)内存下实现可比离线训练方法的BCI神经解码精度（Pearson R≥0.63/0.81），并在闭环仿真中展现了对神经信号非平稳性的持续适应能力。

**[Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)**

:   提出 Multi-Granular Language Learning (MGLL)，一个即插即用的对比学习框架，通过 soft CLIP loss、point-wise loss 和 smooth KL 散度联合优化，实现医学图像与多标签多粒度文本描述的对齐，在眼底和 X 光数据集上全面超越 SOTA 方法，并可作为视觉编码器嵌入多模态大语言模型提升诊断准确率最高达 34.1%。

**[Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)**

:   提出 Brain-IT 框架，通过脑启发式的 Brain Interaction Transformer (BIT) 将功能相似的脑体素聚类为跨被试共享的 Brain Token，并从中预测局部化的语义和结构图像特征，实现从 fMRI 到图像的高保真重建，仅用 1 小时数据即达到先前方法 40 小时的性能。

**[Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)**

:   提出 Brain-Semantoks，一种基于语义分词器和自蒸馏目标的 fMRI 基础模型，将大脑功能网络聚合为鲁棒的语义 token，并通过跨时间视角的一致性学习抽象的脑动态表征，在线性探测设置下即可达到 SOTA 性能。

**[CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)**

:   提出 CARE 框架——将医学 VQA 拆分为"实体提议→指称分割→证据引导问答"三阶段专家管道，用 RLVR 微调各 VLM，并引入 GPT-5 作为动态协调器进行工具规划与 CoT 审查，在 4 个医学 VQA 基准上以 10B 参数量（77.54% 平均准确率）超越 32B 端到端 SOTA（72.29%）。

**[Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)**

:   提出将人类概念产生过程建模为 Transformer 嵌入空间中的累积轨迹，定义 5 个运动学指标（距离、速度、加速度、熵、质心距离），在 4 个数据集（3 种语言、神经退行性疾病/脏话流畅性/属性列举）上成功区分临床组和概念类别，且不同嵌入模型产生高度一致的结果。

**[COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)**

:   COMPASS 通过在分割网络的中间特征空间沿**对目标度量最敏感的低维子空间**进行线性扰动来构建 conformal prediction 区间，在四个医学分割任务上实现了比传统 CP 方法显著更窄的预测区间，同时保持有效覆盖率。

**[DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)**

:   将密集重叠细胞实例分割建模为图着色问题，提出"显式标记冲突节点 + 隐式邻接约束消歧"的分治框架 Disco，通过 BFS 分解细胞邻接图并引入五种协同损失函数，在高密度病理数据集 GBC-FS 2025 上 PQ 提升 7.08%，同时在四个异质数据集上均取得 SOTA。

**[Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems](distributional_consistency_loss_beyond_pointwise_data_terms_in_inverse_problems.md)**

:   提出分布一致性（DC）损失，用分布级别的校准替代传统逐点数据保真项（如MSE/NLL），避免对噪声的过拟合，在DIP去噪和PET图像重建中显著提升性能且无需早停。

**[DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)**

:   提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。

**[Glance and Focus Reinforcement for Pan-cancer Screening](glance_and_focus_reinforcement_for_pan-cancer_screening.md)**

:   提出 GF-Screen 两阶段框架——轻量 Glance 模型用强化学习快速定位含病灶的 CT 子体积，Focus 模型只对选中区域做精细分割；通过将 GRPO 的"组内相对比较"思想从 NLP 迁移到视觉子体积组，首次在纯视觉任务中实现无价值网络的 RL 优化，在 FLARE25 泛癌挑战中以 +25.6% DSC 大幅领先冠军方案且推理快 5.7 倍。

**[HEEGNet: Hyperbolic Embeddings for EEG](heegnet_hyperbolic_embeddings_for_eeg.md)**

:   首次系统验证EEG数据具有双曲性（层次结构），提出HEEGNet混合双曲网络架构，结合欧几里得编码器提取时空频谱特征和双曲编码器捕捉层次关系，配合创新的粗到细域适应策略(DSMDBN)，在视觉诱发电位、情感识别和颅内EEG多个跨域任务上达到SOTA。

**[Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)**

:   提出 Inter-Slice Consistent Stochasticity (ISCS)，通过球面线性插值(Slerp)在扩散采样的 re-noising 步骤中生成层间相关噪声，从根源消除 2D 扩散先验做 3D 医学重建时的层间不连续伪影——零额外计算/超参数/训练开销，即插即用到任何 2D 扩散逆问题求解器，在稀疏视角 CT、限角 CT 和 MRI 超分辨率上均持续提升。

**[Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)**

:   提出DyMo——推理时动态模态选择框架，通过理论推导将多模态任务相关信息增益转化为可计算的MTIR奖励函数（基于分类损失降低代理 + 类原型距离 + 类内相似性校准），在推理时迭代选择性融合可靠的恢复模态，首次系统性解决"丢弃缺失模态损失信息 vs 补全可能引入噪声"的困境。

**[LaVCa: LLM-assisted Visual Cortex Captioning](lavca_llm-assisted_visual_cortex_captioning.md)**

:   提出 LaVCa 方法，利用 LLM 为人类视觉皮层的每个体素生成自然语言描述（caption），通过"编码模型→最优图像选取→MLLM生成描述→LLM关键词提炼+句子组合"四步流程，比已有方法 BrainSCUBA 更准确、更多样地揭示了体素级视觉选择性。

**[Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)**

:   提出 Δ-LFM 框架：用 ArcRank 损失在潜在空间构建患者特异性时间对齐轨迹（角度一致 + 幅度单调递增），将流匹配时间范围从 [0,1] 扩展到 [0,T] 实际时间间隔实现任意时间点预测，在三个阿尔茨海默纵向 MRI 基准上全面超越 8 种基线方法，并提出进展专用指标 Δ-RMAE。

**[Neuro-Symbolic Decoding of Neural Activity](neuro-symbolic_decoding_of_neural_activity.md)**

:   提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。

**[NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)**

:   提出 NH-GCAT 框架，将神经科学中的抑郁症神经环路先验知识显式融入 GNN，在区域、环路和网络三个空间尺度上建模，在 REST-meta-MDD 数据集上取得 SOTA 分类效果（AUC 78.5%、ACC 73.8%），并提供与神经科学相符的可解释性分析。

**[Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin.md)**

:   提出 Q-FSRU 框架，通过 FFT 将医学图像和文本特征变换到频率域进行融合，并引入量子启发的检索增强机制（Quantum RAG）从外部知识库中获取医学事实，在 VQA-RAD 数据集上取得 90.0% 准确率。

**[SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)**

:   提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。

**[Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)**

:   提出 NeuroAdapter，将 fMRI 信号按脑区分割为独立 token 并通过交叉注意力直接条件化 Stable Diffusion，跳过传统的 CLIP/DINO 中间嵌入空间，在 NSD 等数据集上高层语义指标超越或持平现有方法，同时引入 IBBI 双向可解释性框架，首次动态揭示不同皮层区域在去噪轨迹中如何驱动图像生成。

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🔬 ICLR2026** · 共 **34** 篇

**[Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)**

:   提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)**

:   构建了首个大规模抗体功能注释指令数据集AFD-Instruction（430K+条目），通过多智能体文献抽取pipeline对齐抗体序列与自然语言功能描述，用于指令微调通用LLM使其掌握抗体理解和功能导向设计能力，在5类分类任务上平均准确率提升20+点。

**[An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)**

:   从因果推断视角重新审视Q函数估计问题，揭示传统Q回归和FQE是具有插入偏差的plug-in学习器，提出DRQQ-learner——一种双重鲁棒、Neyman正交、准oracle高效的Q函数估计器，通过推导有效影响函数构建去偏两阶段损失函数，在Taxi和Frozen Lake环境中验证了其优越性。

**[AntigenLM: Structure-Aware DNA Language Modeling for Influenza](antigenlm_structure-aware_dna_language_modeling_for_influenza.md)**

:   AntigenLM 是一个保留基因组功能单元完整性的 GPT-2 风格 DNA 语言模型，通过在流感病毒全基因组上预训练并微调，能够自回归预测未来流行毒株的抗原序列，在氨基酸错配率上显著优于进化模型 beth-1 和通用基因组模型。

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[Augmenting Representations With Scientific Papers](augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

**[Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)**

:   对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。

**[Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)**

:   提出 Multi-Granular Language Learning (MGLL)，一个即插即用的对比学习框架，通过 soft CLIP loss、point-wise loss 和 smooth KL 散度联合优化，实现医学图像与多标签多粒度文本描述的对齐，在眼底和 X 光数据集上全面超越 SOTA 方法，并可作为视觉编码器嵌入多模态大语言模型提升诊断准确率最高达 34.1%。

**[Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)**

:   提出 Brain-IT 框架，通过脑启发式的 Brain Interaction Transformer (BIT) 将功能相似的脑体素聚类为跨被试共享的 Brain Token，并从中预测局部化的语义和结构图像特征，实现从 fMRI 到图像的高保真重建，仅用 1 小时数据即达到先前方法 40 小时的性能。

**[Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)**

:   提出 Brain-Semantoks，一种基于语义分词器和自蒸馏目标的 fMRI 基础模型，将大脑功能网络聚合为鲁棒的语义 token，并通过跨时间视角的一致性学习抽象的脑动态表征，在线性探测设置下即可达到 SOTA 性能。

**[Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)**

:   提出BEE框架，通过分析微调如何扰动预训练表征的权重空间几何结构，直接从分类器学到的权重中识别和命名虚假相关性（spurious correlations），无需反例样本即可发现隐藏的数据偏差，在ImageNet-1k上发现可导致准确率下降高达95%的虚假关联。

**[Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)**

:   研究稀疏自编码器（SAE）能否揭示和缓解 LLM 在医疗场景中的种族偏见：发现 SAE 能识别出与种族相关的有害联想（如黑人与暴力），但在复杂临床任务中缓解偏见的效果有限（FLDD < 3%），远不如简单的提示策略（FLDD 8-15%）。

**[CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)**

:   提出 CARE Agent 框架，将医学 VQA 分解为实体提议、指称分割和证据引导推理三个专家模块，通过 GPT-5 作为动态协调器，在医学 VQA 基准上以 77.54% 准确率超越 32B 模型。

**[Causal Interpretation of Neural Network Computations with Contribution Decomposition](causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)**

:   提出 CODEC（Contribution Decomposition），用 Integrated Gradients 计算隐藏层神经元对输出的贡献（而非仅分析激活），再用 Sparse Autoencoder 将贡献分解为稀疏模式（modes），实现比激活分析更强的因果可解释性和网络控制能力，并成功应用于 ResNet-50 和视网膜生物神经网络模型。

**[Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)**

:   提出将人类概念产生过程建模为 Transformer 嵌入空间中的累积轨迹，定义 5 个运动学指标（距离、速度、加速度、熵、质心距离），在 4 个数据集（3 种语言、神经退行性疾病/脏话流畅性/属性列举）上成功区分临床组和概念类别，且不同嵌入模型产生高度一致的结果。

**[COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)**

:   提出 COMPASS 框架，在分割模型的中间特征空间而非输出空间做共形预测，通过沿 Jacobian 确定的低维敏感子空间扰动特征来构建预测区间，在多个医学分割数据集上以更紧凑的区间达到目标覆盖率。

**[ConfHit: Conformal Generative Design with Oracle Free Guarantees](confhit_conformal_generative_design_with_oracle_free_guarantees.md)**

:   提出 ConfHit，一个模型无关的保理推断框架，通过密度比加权的共形 p 值和嵌套检验策略，在无需实验验证（oracle-free）和分布偏移条件下，为生成模型（药物发现等）提供有限样本统计保证——生成的候选集以 $1-\alpha$ 概率包含至少一个 hit。

**[Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)**

:   首次系统性研究蛋白质语言模型（PLM）中的病态重复问题，提出统一的重复度量指标 $R(x)$ 和效用指标 $U(x)$，并设计 UCCS（Utility-Controlled Contrastive Steering）方法，通过在隐层注入与重复解耦的引导向量，在不重训模型的前提下有效抑制重复同时保持折叠可信度。

**[CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)**

:   提出 CryoNet.Refine，首个基于 AI 的冷冻电镜 (cryo-EM) 原子模型精修框架，利用单步扩散模型结合可微密度损失和几何约束损失，在 120 个复合物基准上全面超越 Phenix.real_space_refine（$\text{CC}_{\text{mask}}$ 0.59 vs 0.54，Ramachandran favored 98.92% vs 96.39%）。

**[Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)**

:   提出 TeCh 框架，用核心 Token 聚合-再分配（CoTAR）替代标准注意力处理医学时间序列的通道依赖，将复杂度从 $O(n^2)$ 降至 $O(n)$，在 APAVA 上精度 86.86%（超 Medformer 12.13%），内存仅 33%、推理时间仅 20%。

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)**

:   提出基于图着色理论的密集重叠细胞实例分割框架 DISCO，通过"显式标记冲突+隐式消歧邻接约束"的分治策略，在高密度病理图像上 PQ 提升 7.08%。

**[DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)**

:   提出 DistMLIP 分布式推理平台，基于零冗余图级并行化策略（graph-level parallelization），解决现有机器学习原子间势（MLIP）缺乏多 GPU 支持的问题，在 8 GPU 上实现接近百万原子的模拟，比空间分区方法快达 8 倍且能模拟 3.4 倍更大的系统。

**[DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)**

:   DriftLite 提出在 Fokker-Planck 方程中利用漂移-势函数的自由度，通过轻量级线性系统求解最优控制漂移来主动稳定粒子权重，以最小代价解决 Sequential Monte Carlo 中的权重退化问题，在高斯混合、分子系统和蛋白质-配体共折叠任务上大幅超越 Guidance-SMC 基线。

**[Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)**

:   提出双蒸馏框架 D24FAD，结合 query 图像上的教师-学生蒸馏（TSD）和 support 图像上的学生自蒸馏（SSD），辅以学习权重机制（L2W）自适应评估 support 重要性，在 APTOS 眼底数据集上仅用 2-shot 达到 100% AUROC。

**[Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)**

:   挑战基因表达预测中"越长越好"的长序列建模范式，发现当前 SSM 模型本质上只利用近端信息；进而识别出背景染色质信号（DNase-seq/Hi-C）作为混杂变量引入虚假关联，提出 Prism 框架通过后门调整去混杂，仅用 2k 短序列即超越 200k 长序列的 SOTA。

**[HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)**

:   提出 HistoPrism，基于 Transformer 的跨癌种基因表达预测框架，通过交叉注意力注入癌种条件、引入基因通路一致性（GPC）评估指标，仅用 500 张 WSI 即实现优于 STPath（需 38K WSI）的通路级预测，在 86% 的 Hallmark 通路上更优。

**[Intrinsic Lorentz Neural Network](intrinsic_lorentz_neural_network.md)**

:   提出完全内禀（fully intrinsic）的双曲神经网络 ILNN，所有运算均在 Lorentz 模型内完成，消除了现有方法中混合欧几里得操作的几何不一致性，在图像分类、基因组学和图分类上取得 SOTA。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，基准评估 29 个 LLM，并通过离线/在线 RL 训练出 Med-Copilot（分别 +43%/+45% 提升），达到与 GPT-4o 竞争的性能同时保持成本效益和隐私保护。

**[Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)**

:   构建 Omni-iEEG，首个多中心标准化颅内脑电（iEEG）数据集（302名患者/8中心/36177标注事件），配套 HFO 分类和致病脑区识别基准，发现端到端分段模型（AUC 0.8061）可匹配临床验证的生物标志物方法（AUC 0.7351），且音频预训练模型可有效迁移。

**[Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)**

:   提出 STAR-MD，一个 SE(3) 等变的因果扩散 Transformer，通过联合时空注意力和上下文噪声扰动实现微秒级蛋白质动力学轨迹生成，在 ATLAS 基准上所有指标达到 SOTA，且能稳定外推到训练中未见的微秒时间尺度。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)**

:   提出 SynCoGen，联合建模 2D 合成路线（掩码图扩散）和 3D 几何（流匹配），确保生成分子既化学可合成又空间有效，在无条件 3D+图生成和连接子设计中达到 SOTA。

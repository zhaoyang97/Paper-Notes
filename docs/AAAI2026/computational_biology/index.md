---
title: >-
  AAAI2026 计算生物论文汇总 · 20篇论文解读
description: >-
  20篇AAAI2026的计算生物方向论文解读，涵盖生物分子、对齐/RLHF、扩散模型、多模态、持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "生物分子"
  - "对齐/RLHF"
  - "扩散模型"
  - "多模态"
  - "持续学习"
item_list:
  - u: "apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff/"
    t: "Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models"
  - u: "beerna_tertiary_structure-based_rna_inverse_folding_using_artificial_bee_colony/"
    t: "BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony"
  - u: "cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce/"
    t: "CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data"
  - u: "constrained_best_arm_identification_with_tests_for_feasibility/"
    t: "Constrained Best Arm Identification with Tests for Feasibility"
  - u: "consurv_multimodal_continual_learning_for_survival_analysis/"
    t: "ConSurv: Multimodal Continual Learning for Survival Analysis"
  - u: "distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r/"
    t: "Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes"
  - u: "dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv/"
    t: "Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics"
  - u: "efficient_chromosome_parallelization_for_precision_medicine_genomic_workflows/"
    t: "Efficient Chromosome Parallelization for Precision Medicine Genomic Workflows"
  - u: "epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti/"
    t: "EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization"
  - u: "gene_incremental_learning_for_single-cell_transcriptomics/"
    t: "Gene Incremental Learning for Single-Cell Transcriptomics"
  - u: "gp-molformer-sim_test_time_molecular_optimization_through_contextual_similarity_/"
    t: "GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance"
  - u: "hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati/"
    t: "HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology"
  - u: "investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale/"
    t: "Investigating Data Pruning for Pretraining Biological Foundation Models at Scale"
  - u: "mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m/"
    t: "MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging"
  - u: "on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini/"
    t: "On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples"
  - u: "protsae_disentangling_and_interpreting_protein_language_models_via_semantically-/"
    t: "ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders"
  - u: "qgshap_quantum_acceleration_for_faithful_gnn_explanations/"
    t: "QGShap: Quantum Acceleration for Faithful GNN Explanations"
  - u: "s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_/"
    t: "S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening"
  - u: "spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance/"
    t: "SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection"
  - u: "trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod/"
    t: "TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**🤖 AAAI2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (21)](../../CVPR2026/computational_biology/index.md) · [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/computational_biology/index.md)

🔥 **高频主题：** 生物分子 ×4 · 对齐/RLHF ×3 · 扩散模型 ×2 · 多模态 ×2 · 持续学习 ×2

**[Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)**

:   提出Apo2Mol，一个基于扩散的全原子框架，从蛋白质apo（未结合）构象出发，同时生成3D配体分子和对应的holo（结合态）口袋构象，使用24K实验解析的apo-holo结构对训练，在结合亲和力（Vina min -7.86）和药物类似性上达到SOTA。

**[BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony](beerna_tertiary_structure-based_rna_inverse_folding_using_artificial_bee_colony.md)**

:   提出 BeeRNA，将人工蜂群（ABC）优化算法应用于 RNA 三级结构逆折叠问题，通过碱基对距离预筛选 + RMSD 两阶段适应度评估，在短/中长度 RNA（<100 nt）上超越深度学习方法 gRNAde 和 RiboDiffusion。

**[CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)**

:   提出 CellStream，一种将自编码器与非平衡动态最优传输（unbalanced dynamical OT）联合学习的深度学习框架，从离散时间点的单细胞快照数据中同时学习低维嵌入和连续细胞动态轨迹，在时间一致性和速度一致性上显著优于现有方法。

**[Constrained Best Arm Identification with Tests for Feasibility](constrained_best_arm_identification_with_tests_for_feasibility.md)**

:   提出带可行性约束的最优臂识别新框架，允许决策者分别测试臂的性能或可行性约束，设计了渐近最优算法，可自适应地选择通过性能或可行性中更容易的方式淘汰次优臂。

**[ConSurv: Multimodal Continual Learning for Survival Analysis](consurv_multimodal_continual_learning_for_survival_analysis.md)**

:   本文提出 ConSurv，首个面向生存分析的多模态持续学习方法，通过多阶段混合专家（MS-MoE）和特征约束回放（FCR）两个核心组件，在整合全切片病理图像和基因组数据的场景下有效缓解灾难性遗忘，并在新构建的 MSAIL 基准上全面超越现有方法。

**[Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)**

:   本文提出 GODD（Geometric OOD Diffusion Model），通过等变非对称自编码器捕捉分布结构先验来引导扩散模型的生成过程，使得在数据丰富的分子分布上训练的模型能够泛化到数据稀缺的分布，在 OOD 结构偏移基准上成功率提升 12.6%。

**[Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)**

:   提出 DKAN，一个双路径知识增强对比对齐网络，通过整合外部基因数据库的语义信息作为跨模态协调器，结合统一的一阶段对比学习范式和自适应加权机制，从病理组织切片图像（H&E WSI）预测空间分辨率的基因表达，在三个公开ST数据集上全面超越SOTA。

**[Efficient Chromosome Parallelization for Precision Medicine Genomic Workflows](efficient_chromosome_parallelization_for_precision_medicine_genomic_workflows.md)**

:   提出三种互补的染色体级基因组并行化调度方案——静态调度（优化处理顺序）、动态调度（背包问题式批处理+在线RAM预测）和符号回归RAM预测器，在模拟和真实精准医学流水线中显著降低了内存溢出和执行时间。

**[EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)**

:   提出EPO（Energy Preference Optimization），将反向SDE采样与listwise能量排序偏好优化结合，用能量信号对齐预训练蛋白质生成器与目标Boltzmann分布，在Tetrapeptides/ATLAS/Fast-Folding三个基准9个指标上达到SOTA，完全消除了昂贵的分子动力学（MD）模拟需求。

**[Gene Incremental Learning for Single-Cell Transcriptomics](gene_incremental_learning_for_single-cell_transcriptomics.md)**

:   本文提出了基因增量学习（GIL）框架，利用单细胞转录组学数据的无序性特点，将类增量学习（CIL）的范式扩展到 token（基因）维度，设计了基因回放和基因蒸馏两种基线方法，并建立了包含基因级回归和基因级分类两种评估方式的完整基准。

**[GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance](gp-molformer-sim_test_time_molecular_optimization_through_contextual_similarity_.md)**

:   提出 GP-MoLFormer-Sim，一种无需训练的测试时分子生成引导方法：利用化学语言模型（GP-MoLFormer）自身的上下文嵌入计算与目标分子的相似度，在自回归解码时动态调整logits来引导生成，结合遗传算法（GP-MoLFormer-Sim+GA）后在PMO基准的23个任务上平均排名第2，且在黑盒oracle设定下优于依赖GPT-4的MOLLEO。

**[HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)**

:   提出 HiFusion 框架，通过层次化 spot 内建模（HISM）和上下文感知跨尺度融合（CCF）两个互补模块，从 H&E 染色全切片图像中准确预测空间基因表达，在两个基准数据集的 2D 切片交叉验证和 3D 样本特异性评估中均达到 SOTA。

**[Investigating Data Pruning for Pretraining Biological Foundation Models at Scale](investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale.md)**

:   提出一个基于影响函数的后验数据剪枝框架，通过子集自影响估计（Subset-Based Self-Influence）和两种选择策略（Top-k Influence 和 Coverage-Centric Influence），在超过 99% 的极端剪枝率下，用仅 0.2M 序列预训练的 RNA-FM 在多项下游任务上媲美甚至超越用 23M 序列训练的完整模型，揭示了生物序列数据集的巨大冗余性。

**[MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)**

:   提出 MergeDNA，通过可微分 Token Merging 实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder 和自适应 masked token modeling 预训练，380M 参数超越 1.3B GENERator。

**[On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)**

:   本文通过Poisson过程框架，解析刻画了一维Wasserstein距离在有限样本下同时编码概率密度函数的逐点密度差异（rate difference）和支撑差异（support difference）的能力，并在神经脉冲数据和氨基酸接触频率数据上验证了其实际价值。

**[ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)**

:   提出 ProtSAE，在稀疏自编码器训练中引入语义标注和领域本体知识作为引导信号，解决传统 SAE 的语义纠缠问题，使蛋白质语言模型的隐层特征与生物学概念（分子功能、生物过程、离子结合位点等）精准对齐，同时保持高重建保真度并支持概念级别的生成控制。

**[QGShap: Quantum Acceleration for Faithful GNN Explanations](qgshap_quantum_acceleration_for_faithful_gnn_explanations.md)**

:   提出 QGShap，一种利用量子振幅放大技术加速精确 Shapley 值计算的图神经网络可解释性框架，在保持精确计算（非近似）的同时实现了相对经典 Monte Carlo 方法的二次加速。

**[S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)**

:   提出 S2Drug，一个两阶段对比学习框架，第一阶段在 ChemBL 大规模数据上用蛋白质序列-配体对比预训练（含双边数据采样策略降噪去冗），第二阶段在 PDBBind 上通过残基级门控模块融合序列与 3D 结构信息并引入结合位点预测辅助任务，在 DUD-E 和 LIT-PCBA 虚拟筛选基准上大幅超越现有方法。

**[SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)**

:   提出 SpaCRD，一个基于迁移学习的多模态深度融合框架，通过类别正则化变分重建引导的双向交叉注意力融合网络（VRBCA），将组织学图像与空间转录组学数据深度整合，在 23 个配对数据集上跨样本、跨平台/批次实现了癌症组织区域（CTR）检测的 SOTA 性能。

**[TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)**

:   提出 TrinityDNA，一个生物启发的DNA基础模型，整合三大创新：Groove Fusion模块捕获DNA大小沟槽结构特征、Gated Reverse Complement机制处理双链互补对称性、Sliding Multi-Window Attention实现多尺度长程依赖建模，配合从原核到真核的进化训练策略（ETS），在GUE基准15个任务上平均MCC达0.708（超越2.5B参数的NT），在19个零样本任务上的原核/真核表现均领先，并提出新的CDS标注基准供长序列推理评估。

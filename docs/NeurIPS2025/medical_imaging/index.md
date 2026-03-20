---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🧠 NeurIPS2025** · 共 **9** 篇

**[3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)**

:   提出 3D-RAD——首个大规模3D医学VQA基准，包含170K条CT影像问答数据，覆盖六类临床任务（含创新性的多时相诊断任务），并配套136K训练集，揭示了现有VLM在3D时序推理上的严重不足。

**[A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)**

:   将 Latent ODE 编码器与梯度提升决策树结合，构建端到端 ECG 心律失常分类流水线，在 360Hz→45Hz 降采样下 AUC-ROC 仅从 0.984 降至 0.976，展示了对低采样率的鲁棒性。

**[A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)**

:   提出首个统一视频融合框架 UniVF（基于多帧学习 + 光流特征 warping + 时序一致性损失），并构建首个覆盖四大融合任务（多曝光、多焦点、红外-可见光、医学）的视频融合基准 VF-Bench，在全部子任务上取得 SOTA。

**[A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)**

:   提出一种变分流形嵌入框架，将降维问题形式化为最优嵌入映射的优化问题（最小化先验分布与数据分布pullback之间的KL散度），在理论上统一了PCA与非线性降维方法，并利用变分法（Euler-Lagrange方程）和Noether定理为最优嵌入提供了可解释性约束。

**[AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)**

:   针对现实药物发现中蛋白质 holo 结构不可用的问题，提出 AANet——通过三模态对比学习（配体-holo pocket-检测cavity）对齐表征并用交叉注意力聚合多个候选结合位点，在 apo/predicted 蛋白质结构上的盲筛性能远超 SOTA（DUD-E 上 EF1% 从 11.75 提升至 37.19）。

**[Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)**

:   提出 EM-PTDM 框架，受神经科学双记忆系统启发，利用预训练扩散模型作为"永久记忆"并结合基于 Doob's h-transform 的轻量"瞬时记忆"模块，在**无领域先验数据**的条件下实现高效的主动目标发现，理论保证先验单调改进。

**[Amortized Active Generation of Pareto Sets](amortized_active_generation_of_pareto_sets.md)**

:   提出 A-GPS 框架，通过学习 Pareto 集的条件生成模型实现在线离散黑箱多目标优化——用非支配类概率估计器（CPE）作为 PHVI 的隐式估计替代显式超体积计算，并通过偏好方向向量实现摊还式后验偏好条件化（无需重新训练），在合成基准和蛋白质设计任务上展示了优越的样本效率。

**[Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)**

:   提出 ChefNMR，首个基于 3D 原子扩散模型的端到端框架，仅从 1D NMR 光谱和化学式直接预测未知小分子（尤其是复杂天然产物）的分子结构，在合成和实验数据集上均达到 SOTA。

**[Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)**

:   首个统一脑结构形态（T1 sMRI）与功能动态（fMRI）的多模态脑基础模型，通过几何谐波预对齐和时序自适应 Patch Embedding（TAPE）将高维神经影像压缩为紧凑的 1D token 表示，在神经发育/退行性疾病诊断和认知预测任务上全面超越先前方法。

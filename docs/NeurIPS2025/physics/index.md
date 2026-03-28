<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理学

**🧠 NeurIPS2025** · 共 **14** 篇

**[Astroco Self-Supervised Conformer-Style Transformers For Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)**

:   提出 AstroCo，一种将 Conformer（注意力 + 深度可分离卷积 + 门控）引入天文不规则光变曲线的自监督编码器，在 MACHO 数据集上重建误差比 Astromer v1/v2 降低 61-70%，少样本分类 macro-F1 提升约 7%。

**[Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)**

:   探究LLM嵌入是否能编码从X射线天文观测导出的物理量（硬度比、幂律指数、变异性），发现结构化prompt设计可将物理属性聚类纯度提升5.9%-57.5%，稀疏自编码器揭示LLM通过识别天体类型来推断未显式给出的物理参数。

**[Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)**

:   用条件可逆神经网络（cINN）训练于15,777颗合成行星数据，从观测量（行星质量、轨道距离）快速推断行星形成参数（盘质量、湍流α、尘气比），实现比物理模型快~10⁶倍的概率性参数回溯，并证明多行星系统数据比单行星数据更鲁棒。

**[FAIR Universe HiggsML Uncertainty Dataset and Competition](fair_universe_higgsml_uncertainty_dataset_and_competition.md)**

:   提供2.8亿模拟LHC碰撞事件的标准化数据集和竞赛平台，包含6种参数化系统偏差（探测器校准+背景成分）及不确定性感知评估指标，要求参赛者为Higgs信号强度μ估计鲁棒的68.27%置信区间，优胜方案通过无聚焦替代建模实现比传统方法窄20%的置信区间。

**[Feat Free Energy Estimators With Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)**

:   提出 FEAT 框架，基于随机插值学习自适应传输，通过 escorted Jarzynski 等式和 Crooks 定理提供一致、最小方差的自由能差估计器，统一了平衡与非平衡方法。

**[From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)**

:   构建从模拟星系（TNG50）到真实巡天观测（SDSS）的域适应 pipeline，通过特征级对齐（欧几里得距离 + 最优传输 + top-$k$ 软匹配损失）和可训练权重调度，将星系形态分类的目标域准确率从 46.8%（无适应）提升到 87.3%，Macro F1 从 0.298 提升到 0.626。

**[Latent Representation Learning In Heavy-Ion Collisions With Maskpoint Transforme](latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)**

:   将掩码点云 Transformer 自编码器引入重离子碰撞分析，通过自监督预训练+监督微调的两阶段范式，学习到比 PointNet 更强的非线性潜在表征（PC1 分布重叠从 2.42% 降至 0.27%），为 QGP 性质研究提供了通用特征学习框架。

**[Multi-Modal Masked Autoencoders for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器 (MMAE) 应用于星系图像和光谱的联合重建，构建了 134,533 个星系的图像+光谱数据集，实现了光谱和图像的交叉重建以及仅从图像的红移回归，$\sigma_{\text{NMAD}} = 0.016$ 优于 AstroCLIP。

**[Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)**

:   用神经网络解决星系恒星质量轮廓的去投影问题——从 2D 投影光度轮廓恢复 3D 空间质量分布，替代传统的 Abel 反演解析方法，在处理噪声数据和复杂轮廓时更鲁棒且更快。

**[Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)**

:   提出 QDSFormer，用变分量子电路（QontOT）替换 ViT 中的 softmax 生成双随机注意力矩阵，在多个小规模图像识别任务上超越标准 ViT 和 Sinkformer，并显著稳定训练。

**[Toward Complete Merger Identification at Cosmic Noon with Deep Learning](toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)**

:   在 IllustrisTNG50 模拟生成的模拟 HST CANDELS 图像上训练 ResNet18，首次证明深度学习可以在高红移 $1<z<1.5$ 下成功识别包括小质量比合并（minor merger, $\mu \geq 1/10$）和低质量星系（$M_\star > 10^8 M_\odot$）在内的星系合并，总体准确率约 73%，并通过 Grad-CAM 和 UMAP 深入分析了模型行为。

**[Tropical Attention Neural Algorithmic Reasoning For Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)**

:   提出 Tropical Attention，将注意力机制提升到热带射影空间中进行分段线性推理，在组合算法的 OOD 泛化上大幅超越 softmax 基线，同时推理速度快 3-9 倍、参数少 ~20%。

**[Unsupervised Discovery Of High-Redshift Galaxy Populations With Variational Auto](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)**

:   用变分自编码器(VAE)对 2743 条 JWST 高红移($z>4$)星系光谱进行无监督聚类，发现 12 个不同的天体物理类别，使已知的后星暴星系、Lyman-α 发射星系、极端发射线星系、Little Red Dots 等稀有种群数量翻倍。

**[Why Is Attention Sparse In Particle Transformer](why_is_attention_sparse_in_particle_transformer.md)**

:   分析 Particle Transformer (ParT) 在jet tagging中出现的二值化稀疏attention现象：稀疏性来自attention机制本身而非物理启发的interaction矩阵，但两者对性能都不可或缺。

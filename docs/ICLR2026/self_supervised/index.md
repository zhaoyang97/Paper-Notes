<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🔬 ICLR2026** · 共 **14** 篇

**[Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)**

:   提出一种计算高效、与性能无关的动态丰富度度量 $\mathcal{D}_{LR}$，通过比较最后一层前后的激活来衡量 rich/lazy 训练动态，并证明 neural collapse 是该度量的特殊情况。

**[Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)**

:   通过相似度图模型理论分析证明"困难样本"（跨类高相似度样本）会损害无监督对比学习性能，提出删除困难样本、调节 margin 和温度缩放三种策略，在 TinyImageNet 上带来 15% 的提升。

**[Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions](enhancing_molecular_property_predictions_by_learning_from_bond_modelling_and_int.md)**

:   提出 DeMol 双图增强多尺度交互框架，通过并行的原子中心图和键中心图通道以及 Double-Helix Blocks 显式建模原子-原子、原子-键、键-键三类交互，在 PCQM4Mv2、OC20、QM9 等基准上取得 SOTA。

**[Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)**

:   受果蝇嗅觉回路启发，提出 Fly-CL 框架，通过稀疏随机投影+top-k操作+流式岭分类三阶段渐进去相关，在预训练模型持续学习中大幅降低训练时间的同时达到SOTA水平。

**[G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)**

:   提出 G-reasoner，通过 QuadGraph 四层统一图接口将异构知识源标准化，训练 34M 参数的 GNN 图基础模型联合推理图拓扑和文本语义，配合 LLM 在 6 个基准上全面超越 SOTA GraphRAG 方法。

**[Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)**

:   提出 GradFix 方法，利用目标模型的梯度符号作为掩码过滤源模型的任务向量，仅保留与目标损失景观对齐的分量，在少样本下实现跨预训练模型的任务知识迁移，并提供一阶下降保证。

**[InfoNCE Induces Gaussian Distribution](infonce_induces_gaussian_distribution.md)**

:   从理论上证明 InfoNCE 损失函数在两种互补机制下会诱导表征趋向高斯分布：经验理想化路线（对齐+球面均匀性→高斯）和正则化路线（消失正则项→各向同性高斯），并在合成数据和 CIFAR-10 上验证。

**[No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](no_other_representation_component_is_needed_diffusion_transformers_can_provide_r.md)**

:   提出 Self-Representation Alignment (SRA)，利用扩散 Transformer 内部从"差到好"的判别过程，将早层高噪声的表征对齐到晚层低噪声的表征，无需外部表征组件即可加速生成训练并提升质量。

**[PICS: Pairwise Image Compositing with Spatial Interactions](pics_pairwise_image_compositing_with_spatial_interactions.md)**

:   提出 PICS——一种并行成对图像合成方法，通过 Interaction Transformer 中的掩码引导 MoE 和自适应 α-blending 策略，在单次推理中同时合成两个对象并显式建模遮挡、接触等空间交互关系，全面超越现有序列合成方法。

**[PonderLM: Pretraining Language Models to Ponder in Continuous Space](ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)**

:   提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。

**[Revela: Dense Retriever Learning via Language Modeling](revela_dense_retriever_learning_via_language_modeling.md)**

:   提出 Revela，通过 in-batch attention 机制将检索器学习融入语言建模——NTP 不仅依赖本序列上下文，还依赖批内其他序列（由检索器相似度加权），无需标注 query-document 对即可训练强大的密集检索器。

**[Temporal Slowness in Central Vision Drives Semantic Object Learning](temporal_slowness_in_central_vision_drives_semantic_object_learning.md)**

:   通过模拟人类中央视觉（注视点裁剪）和时间慢性原则（时间对比学习），在 Ego4D 数据上训练 SSL 模型，发现两者组合能有效提升语义对象表征——中央视觉强化前景提取，时间慢性在注视凝视期间蒸馏语义信息。

**[Weak-SIGReg: Covariance Regularization for Stable Deep Learning](weak-sigreg_covariance_regularization_for_stable_deep_learning.md)**

:   将 LeJEPA 的 SIGReg 正则化从自监督学习迁移到监督学习，并提出计算高效的 Weak-SIGReg 变体——只约束协方差矩阵趋向单位矩阵（而非全部矩），用随机投影将内存从 $O(C^2)$ 降至 $O(CK)$，在 ViT 无 BN/残差连接时将 CIFAR-100 准确率从 20.73%（坍缩）恢复到 72.02%，且匹配或超越专家精调的基线。

**[Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)**

:   诊断出原型自监督学习中部分原型坍缩的根因是编码器与原型的联合优化导致的快捷学习，提出全解耦训练策略——用在线 GMM 独立估计原型——彻底消除坍缩并提升下游性能。

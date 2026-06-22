---
title: >-
  ECCV2024 模型压缩论文汇总 · 24篇论文解读
description: >-
  24篇ECCV2024的模型压缩方向论文解读，涵盖压缩/编码、模型压缩、知识蒸馏等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
  - "知识蒸馏"
item_list:
  - u: "a_simple_lowbit_quantization_framework_for_video_snapshot_co/"
    t: "A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging"
  - u: "adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith/"
    t: "AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer"
  - u: "adaptive_compressed_sensing_with_diffusionbased_posterior_sa/"
    t: "Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling"
  - u: "adaptive_selection_of_samplingreconstruction_in_fourier_comp/"
    t: "Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing"
  - u: "adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap/"
    t: "Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap"
  - u: "anytime_continual_learning_for_open_vocabulary_classification/"
    t: "Anytime Continual Learning for Open Vocabulary Classification"
  - u: "auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec/"
    t: "Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search"
  - u: "basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre/"
    t: "BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression"
  - u: "bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model/"
    t: "Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model"
  - u: "category_adaptation_meets_projected_distillation_in_generalized_continual_catego/"
    t: "Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery"
  - u: "else_efficient_deep_neural_network_inference_through_line-based_sparsity_explora/"
    t: "ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration"
  - u: "improving_knowledge_distillation_via_regularizing_feature_direction_and_norm/"
    t: "Improving Knowledge Distillation via Regularizing Feature Direction and Norm"
  - u: "improving_zero-shot_generalization_for_clip_with_variational_adapter/"
    t: "Improving Zero-Shot Generalization for CLIP with Variational Adapter"
  - u: "is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear/"
    t: "Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images"
  - u: "isomorphic_pruning_for_vision_models/"
    t: "Isomorphic Pruning for Vision Models"
  - u: "leveraging_hierarchical_feature_sharing_for_efficient_dataset_condensation/"
    t: "Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation"
  - u: "metaaug_meta-data_augmentation_for_post-training_quantization/"
    t: "MetaAug: Meta-Data Augmentation for Post-Training Quantization"
  - u: "papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i/"
    t: "PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference"
  - u: "pq-sam_post-training_quantization_for_segment_anything_model/"
    t: "PQ-SAM: Post-training Quantization for Segment Anything Model"
  - u: "simple_unsupervised_knowledge_distillation_with_space_similarity/"
    t: "Simple Unsupervised Knowledge Distillation With Space Similarity"
  - u: "spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o/"
    t: "SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images"
  - u: "token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin/"
    t: "Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning"
  - u: "uncertainty-driven_spectral_compressive_imaging_with_spatial-frequency_transform/"
    t: "Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer"
  - u: "unic_universal_classification_models_via_multi-teacher_distillation/"
    t: "UNIC: Universal Classification Models via Multi-teacher Distillation"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🎞️ ECCV2024** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (108)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (239)](../../ICLR2026/model_compression/index.md) · [💬 ACL2026 (59)](../../ACL2026/model_compression/index.md) · [🧪 ICML2026 (116)](../../ICML2026/model_compression/index.md) · [🤖 AAAI2026 (60)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/model_compression/index.md)

🔥 **高频主题：** 压缩/编码 ×6 · 模型压缩 ×6 · 知识蒸馏 ×2

**[A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)**

:   首个面向视频快照压缩成像（Video SCI）重建任务的低比特量化框架Q-SCI，通过高质量特征提取模块、精确视频重建模块和Transformer分支的query/key分布偏移操作，在4-bit量化下实现7.8倍理论加速且性能仅下降2.3%。

**[AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)**

:   提出自适应对数底量化器AdaLog，通过可搜索的对数底替代固定log₂/log√2量化器来处理ViT中post-Softmax和post-GELU激活的幂律分布，并设计快速渐进组合搜索(FPCS)策略高效确定量化超参，在极低比特(3/4-bit)下显著优于现有ViT PTQ方法。

**[Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)**

:   本文提出 AdaSense，利用预训练扩散模型的零样本后验采样能力来量化重建不确定性，从而自适应地选择最优测量矩阵，在人脸图像、MRI 和 CT 等多个领域实现了无需额外训练的自适应压缩感知，性能超越非自适应方法甚至基于 PCA 的最优非自适应方案。

**[Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)**

:   本文提出"自适应选择采样-重建对"($\mathcal{H}_{1.5}$)框架，利用超分辨率空间生成模型量化高频贝叶斯不确定性，为每个输入数据选择最佳的采样掩码-重建网络对，在理论和实验上同时优于非自适应联合优化方法（$\mathcal{H}_1$）和自适应采样方法（$\mathcal{H}_2$），在人脸图像和多线圈 MRI 重建中取得显著 SSIM 提升。

**[Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)**

:   本文提出了一种基于特征分布统计对齐的对抗鲁棒知识蒸馏方法，通过减小 student 和 teacher 模型在对抗样本和干净样本之间的特征方差差距(variance gap)来提升 student 模型的对抗鲁棒性，发现鲁棒精度与方差差距存在强负相关线性关系。

**[Anytime Continual Learning for Open Vocabulary Classification](anytime_continual_learning_for_open_vocabulary_classification.md)**

:   提出 AnytimeCL 框架，通过部分微调 CLIP 最后一个 transformer block 并动态加权融合微调模型与原始模型的预测，实现任意时刻接收样本、任意标签集推理的开放词汇持续学习。

**[Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)**

:   本文提出 Auto-DAS，一个基于进化算法的自动化代理发现框架，用于免训练的蒸馏感知架构搜索（DAS），通过在由学生内在统计量和师生交互统计量构成的搜索空间中自动发现最优代理指标，避免了手工设计代理的局限性，在 ResNet、ViT、NAS-Bench-101/201 等多种架构和搜索空间上达到了 SOTA 的排序相关性和搜索精度。

**[BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)**

:   本文提出 BaSIC 框架，通过学习神经图像压缩（NIC）系统的贝叶斯网络结构，同时控制骨干网络复杂度和自回归单元的并行计算能力，首次实现了对 NIC 全流程的计算可扩展性控制。

**[Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)**

:   提出双向对称的立体图像压缩框架 BiSIC，采用 3D 卷积联合编解码器和跨维度熵模型，在 PSNR 和 MS-SSIM 上均超越传统标准和已有学习方法，同时消除了单向方法中左右视图压缩质量不平衡的问题。

**[Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)**

:   提出 CAMP 方法，通过可学习投影器蒸馏与类别中心适应网络的协同组合，在广义持续类别发现（GCCD）场景中显著提升了新类别学习与旧知识保持之间的平衡。

**[ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration](else_efficient_deep_neural_network_inference_through_line-based_sparsity_explora.md)**

:   提出基于行稀疏性探索的事件抑制方法ELSE，利用激活图中相邻行的空间相关性来减少非零激活（事件）数量，在目标检测和姿态估计任务上实现3.14~6.49倍的计算节省，且可与现有事件抑制方法互补。

**[Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)**

:   提出 ND 损失函数，通过同时对齐学生特征方向至教师类均值方向并鼓励学生产生大范数特征，显著提升了现有知识蒸馏方法在 ImageNet、CIFAR100 和 COCO 上的性能。

**[Improving Zero-Shot Generalization for CLIP with Variational Adapter](improving_zero-shot_generalization_for_clip_with_variational_adapter.md)**

:   提出 Prompt-based Variational Adapter (PVA)，通过变分适配器将 base 和 novel 类别样本在隐空间中分离，采用分治策略分别处理，结合残差连接增强 novel 类别的迁移能力，在广义零样本学习和跨数据集迁移学习基准上达到 SOTA。

**[Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images](is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)**

:   提出 SCAR（Selective-distillation for Class and Architecture-agnostic unleaRning），一种无需保留集的近似遗忘算法，通过 Mahalanobis 距离引导遗忘样本特征向量向最近错误类分布迁移，并利用 OOD 图像蒸馏保持模型性能。

**[Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)**

:   提出 Isomorphic Pruning，通过将网络子结构建模为图并按图同构性分组，在同构组内独立排序剪枝，解决异构子结构间重要性不可比的问题，在 ViT 和 CNN 上均取得优于专门设计的剪枝方法的效果。

**[Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation](leveraging_hierarchical_feature_sharing_for_efficient_dataset_condensation.md)**

:   提出层级记忆网络（HMN），将数据蒸馏中的合成数据存储为三层结构（数据集级-类级-实例级记忆），通过层级化特征共享提升存储效率，并利用实例级剪枝进一步去除冗余，仅用低GPU内存的 batch-based loss 即超越所有基线方法。

**[MetaAug: Meta-Data Augmentation for Post-Training Quantization](metaaug_meta-data_augmentation_for_post-training_quantization.md)**

:   提出 MetaAug，一种基于元学习的训练后量化（PTQ）方法，通过可学习的变换网络对校准数据进行增强，并以双层优化框架同时优化变换网络和量化模型，有效缓解 PTQ 在小校准集上的过拟合问题。

**[PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)**

:   提出 PaPr，利用轻量级 ConvNet 的卷积特征图生成 Patch Significance Map (PSM)，在**无需重训练**的情况下对 ViT/ConvNet/混合架构进行**一步式** patch 剪枝，实现显著的计算量削减（视频场景最高 3.7× FLOPs 减少），且精度损失极小。

**[PQ-SAM: Post-training Quantization for Segment Anything Model](pq-sam_post-training_quantization_for_segment_anything_model.md)**

:   本文提出PQ-SAM，首个专为Segment Anything Model定制的训练后量化方法，通过分组激活分布变换(GADT)和两阶段异常值层次聚类(OHC)方案解决SAM的高度不对称激活分布和有害异常值问题，将4-bit量化的SAM推进到可用水平。

**[Simple Unsupervised Knowledge Distillation With Space Similarity](simple_unsupervised_knowledge_distillation_with_space_similarity.md)**

:   CoSS 提出在无监督知识蒸馏中，除了常规的**特征维度余弦相似度**外，额外引入一个**空间维度余弦相似度（Space Similarity）**损失——将特征矩阵转置后在维度方向上对齐，从而弥补 $L_2$ 归一化导致的流形结构信息丢失，以极简的方式在多个 UKD benchmark 上达到 SOTA。

**[SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)**

:   提出 SpaceJAM，一种仅约 16K 可训练参数的无监督图像联合对齐方法，无需正则化项或 atlas 维护，在 SPair-71K 和 CUB 数据集上匹配现有方法的对齐能力同时实现 10 倍以上加速。

**[Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)**

:   提出 ToCom（Token Compensator），一个模型算术框架的轻量插件，通过快速的参数高效自蒸馏获得，可在推理时直接插入任意下游已训练模型以弥补 token 压缩度不匹配造成的性能损失，无需重新训练。

**[Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer](uncertainty-driven_spectral_compressive_imaging_with_spatial-frequency_transform.md)**

:   本文提出 Specformer，通过并行的空间局部窗口自注意力（LWSA）和频率域自注意力（FWSA）模块充分捕获高光谱图像（HSI）的空间稀疏性和光谱间相似性先验，并引入不确定性驱动的损失函数增强网络对纹理丰富和边缘区域的重建能力，在模拟和真实 HSI 数据集上以更低计算量超越 SOTA。

**[UNIC: Universal Classification Models via Multi-teacher Distillation](unic_universal_classification_models_via_multi-teacher_distillation.md)**

:   提出UNIC框架，通过改进的多教师蒸馏策略（包括梯形投影器和教师丢弃技术），将多个互补预训练模型的知识融合到单一学生模型中，实现跨任务的通用分类。

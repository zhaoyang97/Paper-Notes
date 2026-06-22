---
title: >-
  CVPR2025 可解释性论文汇总 · 21篇论文解读
description: >-
  21篇CVPR2025的可解释性方向论文解读，涵盖少样本学习、域适应、持续学习、布局/合成、水印/隐写、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "域适应"
  - "持续学习"
  - "布局/合成"
  - "水印/隐写"
  - "自监督学习"
item_list:
  - u: "albm_attribute_concept_space/"
    t: "Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability"
  - u: "attribute-formed_class-specific_concept_space_endowing_language_bottleneck_model/"
    t: "Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability"
  - u: "differentiable_inverse_rendering_with_interpretable_basis_brdfs/"
    t: "Differentiable Inverse Rendering with Interpretable Basis BRDFs"
  - u: "geometry-guided_camera_motion_understanding_in_videollms/"
    t: "Geometry-Guided Camera Motion Understanding in VideoLLMs"
  - u: "interpretable_image_classification_via_non-parametric_part_prototype_learning/"
    t: "Interpretable Image Classification via Non-parametric Part Prototype Learning"
  - u: "kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception/"
    t: "KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception"
  - u: "l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas/"
    t: "L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers"
  - u: "language_guided_concept_bottleneck_models_for_interpretable_continual_learning/"
    t: "Language Guided Concept Bottleneck Models for Interpretable Continual Learning"
  - u: "learning_on_model_weights_using_tree_experts/"
    t: "Learning on Model Weights using Tree Experts"
  - u: "learning_visual_composition_through_improved_semantic_guidance/"
    t: "Learning Visual Composition through Improved Semantic Guidance"
  - u: "lswag_zero_shot_nas/"
    t: "L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers"
  - u: "on_the_possible_detectability_of_image-in-image_steganography/"
    t: "On the Possible Detectability of Image-in-Image Steganography"
  - u: "open_ad-hoc_categorization_with_contextualized_feature_learning/"
    t: "Open Ad-Hoc Categorization with Contextualized Feature Learning"
  - u: "probing_the_mid-level_vision_capabilities_of_self-supervised_learning/"
    t: "Probing the Mid-Level Vision Capabilities of Self-Supervised Learning"
  - u: "prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis/"
    t: "Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis"
  - u: "sample-_and_parameter-efficient_auto-regressive_image_models/"
    t: "Sample- and Parameter-Efficient Auto-Regressive Image Models"
  - u: "scaling_vision_pre-training_to_4k_resolution/"
    t: "Scaling Vision Pre-Training to 4K Resolution"
  - u: "tide_domain_generalization/"
    t: "TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction"
  - u: "tide_training_locally_interpretable_domain_generalization_models_enables_test-ti/"
    t: "TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction"
  - u: "towards_human-understandable_multi-dimensional_concept_discovery/"
    t: "Towards Human-Understandable Multi-Dimensional Concept Discovery"
  - u: "why_does_it_look_there_structured_explanations_for_image_classification/"
    t: "Why Does It Look There? Structured Explanations for Image Classification"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**📷 CVPR2025** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (34)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (195)](../../ICLR2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🧪 ICML2026 (91)](../../ICML2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (80)](../../NeurIPS2025/interpretability/index.md)

🔥 **高频主题：** 少样本学习 ×2 · 域适应 ×2

**[Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](albm_attribute_concept_space.md)**

:   本文提出ALBM（属性形成的语言瓶颈模型），通过构建属性引导的类特异概念空间避免虚假线索推理问题，并利用视觉属性提示学习提取细粒度属性特征，结合描述-摘要-补充（DSS）策略自动生成高质量概念集，在9个基准上实现了更好的可解释性和可扩展性。

**[Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](attribute-formed_class-specific_concept_space_endowing_language_bottleneck_model.md)**

:   提出 ALBM 模型，用属性化的类特定概念空间（ACCS）取代现有语言瓶颈模型的类共享概念空间，避免虚假线索推理问题并支持跨类泛化，配合视觉属性提示学习（VAPL）提取细粒度属性特征，在 9 个 few-shot 基准上全面超越现有可解释分类方法。

**[Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)**

:   提出基于可解释基 BRDF 的可微逆渲染方法，将材质分解为有物理意义的基函数组合，实现可解释的材质估计

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)**

:   提出一个从基准构建、诊断到注入的完整框架，通过 3D 基础模型（VGGT）提取相机运动线索并以结构化提示注入 VideoLLM，实现无需训练的相机运动感知增强。

**[Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)**

:   本文提出一种基于非参数原型学习的可解释图像分类框架，通过对自监督ViT特征进行最优传输聚类来发现语义上不同的物体部件原型，解决了现有ProtoPNet方法中原型重复冗余的问题，同时引入了Distinctiveness和Comprehensiveness两个新指标来量化解释质量。

**[KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)**

:   KVQ 受人类视觉系统启发，将视频全局质量显式解耦为视觉显著性和局部纹理两个因素，通过 Fusion-Window Attention 提取跨区域显著性、Local Perception Constraint 增强独立区域的纹理感知，在五个 VQA benchmark 上显著超越 SOTA。

**[L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers](l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas.md)**

:   本文提出L-SWAG（Layer-Sample Wise Activation with Gradients），一种新型通用零代价代理，通过结合层级和样本级的激活值与梯度信息来评估网络架构质量，首次将零代价NAS系统性地扩展到Vision Transformer搜索空间，并在Autoformer搜索空间的6个任务上建立了新的benchmark。

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   本文将语言引导的概念瓶颈模型（CBM）引入持续学习，用 ChatGPT 生成人类可理解的概念、CLIP 编码概念嵌入构建概念瓶颈层，在缓解灾难性遗忘的同时提供透明的决策解释，在 ImageNet-subset 上超越 SOTA 3.06%。

**[Learning on Model Weights using Tree Experts](learning_on_model_weights_using_tree_experts.md)**

:   发现公开模型大多属于少数 Model Tree（从共同祖先微调而来），在同一 Tree 内学习权重远比跨 Tree 简单；提出 ProbeX——首个针对单隐藏层权重的轻量 probing 方法，通过 Tucker 张量分解实现参数量 30 倍压缩，并首次实现了将模型权重与文本表示对齐的零样本模型分类（89.8% 准确率）。

**[Learning Visual Composition through Improved Semantic Guidance](learning_visual_composition_through_improved_semantic_guidance.md)**

:   本文提出通过改善训练数据的语义监督信号（使用基础模型重新生成高质量描述+使用预训练文本编码器替代从头训练）来大幅提升标准 CLIP 模型的视觉组合理解能力，在 ARO 基准上从CLIP的59%/63%提升到92%/94%，在DOCCI图像检索上从58.4%提升到94.5% recall@1，且无需任何架构改动。

**[L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](lswag_zero_shot_nas.md)**

:   本文提出L-SWAG指标，通过分层梯度方差和激活模式基数的乘积来表征CNN和ViT网络的可训练性和表达性，并设计LIBRA-NAS算法组合互补代理指标，在ViT搜索空间和14个任务上实现了SOTA级别的零样本NAS性能。

**[On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)**

:   本文从理论和实验两个层面揭示了当下流行的基于深度学习的 image-in-image 隐写方案存在严重的可检测性漏洞——其嵌入过程本质是一个混合过程，可被独立成分分析（ICA）轻松识别，仅用小波域独立成分的前四阶矩构成的 8 维特征就能达到 84.6% 的检测准确率，而经典的 SRM+SVM 方法更是达到 99% 以上。

**[Open Ad-Hoc Categorization with Contextualized Feature Learning](open_ad-hoc_categorization_with_contextualized_feature_learning.md)**

:   本文提出了 OAK（Open Ad-hoc Categorization with Contextualized Feature Learning），通过在冻结 CLIP 的输入层引入少量可学习的上下文 token，联合 CLIP 的图文对齐目标和 GCD 的视觉聚类目标，在仅有少数标注样本的条件下实现了自适应的 ad-hoc 类别发现和上下文切换，Stanford Mood 数据集新类别准确率达 87.4%，超过 CLIP 和 GCD 50% 以上。

**[Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)**

:   本文从儿童视觉发育的视角出发，系统评估了 22 种自监督学习（SSL）模型在中层视觉任务（深度估计、表面法线、物体分割、几何对应等）上的能力，发现尽管 SSL 模型在高层语义任务上与监督模型存在较大差距，但在 3D 空间感知等中层视觉能力上差距显著更小。

**[Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)**

:   提出 Prompt-CAM，通过为预训练 ViT 注入类别特定的可学习 prompt token，利用最后一层的多头注意力图来识别和定位区分细粒度类别的关键特征（traits），实现了近乎"免费"的可解释细粒度分析。

**[Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)**

:   本文提出 XTRA，通过在 ViT 中引入 Block Causal Mask（以 k×k token 块为因果单元），使自回归图像模型在仅用 1/152 训练样本的情况下超越了先前最佳自回归模型在 15 个图像识别基准上的平均准确率，同时以 1/7~1/16 的参数量达到更优的探测性能。

**[Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)**

:   本文提出PS3（Pre-training with Scale-Selective Scaling），通过局部区域与局部caption的对比学习代替全图对比，以近常数的计算开销将CLIP式视觉预训练扩展到4K分辨率，并结合top-down/bottom-up patch选择机制构建VILA-HD多模态大模型，在高分辨率感知任务上大幅超越GPT-4o和Qwen2.5-VL。

**[TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](tide_domain_generalization.md)**

:   本文提出TIDE方法，通过利用扩散模型和LLM自动生成概念级显著图标注，训练可局部解释的域泛化模型，并在测试时利用概念签名进行预测矫正，在四个标准DG基准上平均超越SOTA 12%。

**[TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](tide_training_locally_interpretable_domain_generalization_models_enables_test-ti.md)**

:   本文提出TIDE，一种针对单源域泛化的新型训练方案，利用扩散模型和LLM自动生成类别级概念标注（如"鸟类=尖嘴+翅膀+爪子"），通过概念显著性对齐损失训练模型关注域不变的局部概念而非全局背景特征，使模型在测试时能通过概念显著图自动矫正域偏移导致的错误预测。

**[Towards Human-Understandable Multi-Dimensional Concept Discovery](towards_human-understandable_multi-dimensional_concept_discovery.md)**

:   提出 HU-MCD 框架，用 SAM 替代传统分割方法发现人类可理解的视觉概念，配合 CNN 专用的输入遮罩方案减少噪声干扰，在 MCD 的完备性框架下实现可理解性和忠实性兼顾的概念级模型解释。

**[Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   本文提出 I2X 框架，通过在训练检查点上追踪从 GradCAM 显著性图中提取的抽象原型（prototype）的强度变化与模型置信度的对应关系，将非结构化的可解释性转化为结构化的可解释性，并利用识别出的"不确定原型"来指导微调、减少类间混淆、提升分类精度。

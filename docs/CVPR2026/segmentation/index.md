---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**📷 CVPR2026** · 共 **20** 篇

**[MEDISEG: 药物图像实例分割数据集——预防不良药物事件](a_dataset_of_medication_images_with_instance_segme.md)**

:   构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），用YOLOv8/v9验证在3类上达99.5% mAP@0.5、32类达80.1%，并通过FsDet few-shot协议证明MEDISEG预训练比CURE数据集在遮挡场景中显著提升未见药片类别的识别（1-shot准确率0.406 vs 0.131）。

**[CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](clip_shortsighted_beyond_first_sentence.md)**

:   发现CLIP对长描述"只看第一句"的根本原因在于训练数据中长caption普遍以摘要句开头形成捷径，提出DeBias-CLIP通过去除摘要句+句子子采样+token填充来分散监督信号，实现长短文本检索双SOTA。

**[Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging. Review Paper](comparative_evaluation_of_traditional_methods_and.md)**

:   系统综述脑胶质瘤 MRI 分割与分类方法，比较传统方法（阈值、区域生长、聚类等）与深度学习方法（CNN 架构），结论是 CNN 在分割和分类任务上全面优于传统技术，但半自动方法因可控性更受放射科医生青睐。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 基础上引入物理引导的稀疏 MoE 架构（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由），配套 200K 级预训练数据集和 22 个子基准，在 20/22 个跨域分割任务上达到 SOTA。

**[DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](dsflash_panoptic_scene_graph_realtime.md)**

:   DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。

**[DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](dss_discover_segment_select_zero_shot_cos.md)**

:   提出三阶段零样本伪装目标分割框架DSS：先用DINOv2特征聚类+部件组合发现候选区域（Discover），再用SAM分割（Segment），最后用MLLM逐对比较选最优mask（Select），无需任何训练即在四个COD基准上全面超越先前零样本方法，尤其在多实例场景中优势显著。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)**

:   提出一种高效 RGB-D 多任务场景理解网络，通过部分通道卷积融合编码器、归一化焦点通道层(NFCL)、上下文特征交互层(CFIL)和多任务自适应损失，在 NYUv2 上以 20+ FPS 同时完成语义/实例/全景分割、方向估计和场景分类。

**[Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)**

:   提出 VLM 驱动的语义敏感学习策略，通过 VLM 生成目标物体描述、BLIP 构建空间语义引导图、双重引导机制（cross-attention + 语义对齐损失）注入 UIE decoder，使增强结果在感知质量和检测/分割下游任务上同时提升。

**[EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](erecu_pseudolabel_evolution_unsupervised_camouflage.md)**

:   提出EReCu框架，在DINO师生架构上通过多线索原生感知(MNP)提取纹理+语义先验来引导伪标签进化融合(PEF)和局部伪标签精修(LPR)，实现无标注下的伪装目标检测，在4个COD数据集上达到UCOD SOTA。

**[A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)**

:   解耦双手重建为2D结构对齐+3D空间交互对齐：Stage 1用Fusion Alignment Encoder隐式蒸馏Sapiens的关键点/分割/深度三种2D先验(推理时免基础模型)，Stage 2用穿透感知扩散模型+碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M上MPJPE降至5.36mm(超SOTA 4DHands 2.13mm)，穿透体积降7倍。

**[GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)**

:   提出GKD框架，通过将表示学习与任务学习解耦的多阶段蒸馏策略和查询式软蒸馏机制，从VFM（如DINOv2）中蒸馏出具有跨域泛化能力的轻量学生模型，在F2L设置下平均mIoU提升+10.6%，F2F设置下+1.9%。

**[A Mixed Diet Makes DINO An Omnivorous Vision Encoder](mixed_diet_dino_omnivorous_encoder.md)**

:   发现DINOv2的特征在不同模态间几乎零对齐（同一场景RGB和深度图的特征相似度≈随机图像对），提出Omnivorous Vision Encoder通过跨模态对齐+冻结教师蒸馏的双目标训练，让单一编码器产出模态无关的统一特征空间。

**[Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)**

:   提出在潜在空间而非输入空间做掩码建模的辅助任务MRM，通过轻量级Rebuilder模块对编码器特征做掩码-重建并用分割损失监督，在GTA→Cityscapes上为四种UDA基线平均带来+2.3 mIoU提升，推理时零额外开销。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)**

:   提出SAM FTI-FDet，通过设计一个基于Transformer decoder的自提示生成器（Prompt Generator），让轻量化的TinyViT-SAM自动生成任务相关的query prompt，无需人工交互即可完成货运列车部件的实例级故障检测，在自建数据集上达到74.6 AP_box / 74.2 AP_mask。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)**

:   提出RDNet，通过区域比例感知机制动态选择不同大小卷积核组合，结合小波域频率匹配上下文增强和跨注意力定位模块，在遥感图像显著性检测三个数据集上全面超越SOTA。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_regionguided_selective_optimization_network.md)**

:   提出RSONet两阶段RGB-T显著性检测框架：先通过三分支并行编码器生成区域引导图并基于相似度比较选择主导模态，再通过选择性优化模块融合双模态特征，在VT5000/VT1000/VT821上MAE达0.020/0.014/0.021，超越27个SOTA方法。

**[SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)**

:   将全景分割重构为拓扑-记忆对齐问题，通过列优先锯齿扫描将ERP全景图转为透视伪视频序列，完美复用SAM2的流式记忆机制，在零样本4K全景分割上比vanilla SAM2平均提升+17.2 mIoU。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semanticguided_modalityaware_segmentation_for.md)**

:   提出SGMA——语义引导模态感知分割框架，通过语义引导融合(SGF)降低类内变异和协调跨模态冲突，模态感知采样(MAS)平衡脆弱模态训练，在ISPRS上Average mIoU +9.20%且弱模态Last-1 mIoU +18.26%(vs SOTA IMLT)。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_re.md)**

:   SPARROW通过目标特定跟踪特征(TSF)和双提示[BOX]+[SEG]定位机制增强视频MLLM的时空一致性，在MeViS上J&F +8.9、VidSTG上mIoU +5.49，可即插即用到三种backbone上。

**[VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_encoder_only_video_segmentation.md)**

:   提出encoder-only视频分割模型VidEoMT，通过查询传播和查询融合将分割与时序关联统一在单个ViT编码器中，消除所有专用追踪模块，在YouTube-VIS 2019上达到160 FPS（比CAVIS快10×+），同时AP仅差0.3。

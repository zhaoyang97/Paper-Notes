<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📷 CVPR2025 论文笔记

共 **280** 篇笔记，覆盖 **27** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 🏥 [医学图像](#medical_imaging) | 39 |
| 💬 [LLM / NLP](#llm_nlp) | 32 |
| 🧊 [3D 视觉](#3d_vision) | 28 |
| 🎨 [图像生成](#image_generation) | 19 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 15 |
| 👁️ [多模态 VLM](#multimodal_vlm) | 15 |
| 🚗 [自动驾驶](#autonomous_driving) | 14 |
| ✂️ [语义分割](#segmentation) | 13 |
| ⚡ [LLM 效率](#llm_efficiency) | 12 |
| 🦾 [LLM Agent](#llm_agent) | 11 |
| 🤖 [机器人/具身智能](#robotics) | 9 |
| 🧑 [人体理解](#human_understanding) | 8 |
| 🎬 [视频理解](#video_understanding) | 8 |
| 💡 [LLM 推理](#llm_reasoning) | 7 |
| 🎯 [目标检测](#object_detection) | 6 |
| 📦 [模型压缩](#model_compression) | 5 |
| 🛡️ [AI 安全](#ai_safety) | 4 |
| ✍️ [文本生成](#nlp_generation) | 4 |
| 🛰️ [遥感](#remote_sensing) | 4 |
| 🖼️ [图像恢复](#image_restoration) | 3 |
| 🔄 [自监督/表示学习](#self_supervised) | 3 |
| 📖 [NLP 理解](#nlp_understanding) | 2 |
| 🎵 [音频/语音](#audio_speech) | 1 |
| 📐 [优化/理论](#optimization) | 1 |
| 🎮 [强化学习](#reinforcement_learning) | 1 |
| 📈 [时间序列](#time_series) | 1 |
| 📂 [其他](#others) | 15 |

---

## 🏥 医学图像 { #medical_imaging }

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](medical_imaging/accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型范式，在大规模公开脑 MRI 数据上预训练扩散概率模型（DPM），再在仅 20 例中风患者数据上微调，实现数据受限场景下加速 MRI 重建，临床读者研究证实 2× 加速图像质量不劣于标准治疗。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](medical_imaging/adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](medical_imaging/association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST 7980 例、SUMMIT 8561 例）中验证了基于深度学习自动量化的 PPFE（胸膜肺实质纤维弹性组织增生）进展与全因死亡率独立相关，提出 PPFE 纵向变化可作为筛查人群中识别高呼吸发病风险个体的影像生物标志物。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](medical_imaging/automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   BiCLIP 提出了一种双向一致性视觉-语言分割框架，通过双向多模态融合（BMF，让视觉特征反向精炼文本嵌入）和图像增强一致性（IAC，跨弱/强扰动正则化），在 COVID-19 CT 分割上以仅 1% 标注数据即可保持鲁棒性能，且对临床图像退化（噪声/模糊）具有容忍力。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE 框架，将缺失模态分割的鲁棒性问题重新定义为决策层专家一致性控制问题，通过全局模态专家一致性(MEC)和区域专家一致性(REC)双分支约束减少专家漂移，并用轻量门控网络将一致性分数转化为可靠性权重指导特征融合，在 BraTS 2020 和 MSD Prostate 上超越 SOTA。

**[CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](medical_imaging/cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)**

:   提出 CycleULM，首个统一的无标签深度学习超声定位显微(ULM)框架，通过 CycleGAN 学习 CEUS 帧到简化微泡域的物理仿真双向翻译来弥合仿真-真实域差距，实现微泡定位精度提升达40% recall、46% precision，并以18.3 fps 实现实时处理。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为核心的混合 Transformer-CNN-Mamba 架构，通过 Co-Attention Gate、Vision State Space Module 和可变形卷积精炼块增强解码器能力，并引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA 性能，同时保持适中的模型复杂度。

**[Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出 ScleraGluNet，通过五方向巩膜血管图像结合多分支 CNN + MRFO 特征筛选 + Transformer 跨视图融合，实现三分类代谢状态判别（93.8% 准确率）和连续空腹血糖估计（MAE = 6.42 mg/dL），为无创血糖监测提供了新途径。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建了最大规模 PET 分割数据集 PETWB-Seg11K（11,041 例全身 PET + 59,831 个分割掩码），并提出 SegAnyPET——基于 3D 架构 + prompt 工程的 PET 通用分割基础模型，在多中心、多示踪剂、多疾病场景下展现强零样本泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   提出一种结合非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量 CNN 分类和扩散式特征空间去噪的脑肿瘤分类框架，在保持 ~85% 干净准确率的同时，将 AutoAttack 下的鲁棒准确率从 0.47% 提升至 59.5%。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出 EquivAnIA，一种基于 cake wavelet 和 ridge filter 的频谱方法，用于对图像进行旋转等变的各向异性分析，在合成和真实图像（含 CT）上展现出优于传统 angular binning 的旋转鲁棒性。

**[Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)**

:   提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器（全局联邦）和部分个性化融合解码器，同时解决多模态 MRI 脑肿瘤分割中的模态间异质性和客户端个性化问题，在 BraTS 2018/2020 上客户端平均 mDSC 达 75.70%/75.90%。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](medical_imaging/giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM，一种基于多异构图（MHG）的多视图医学图像分类框架，同时建模视图内（intra-view）和视图间（inter-view）的病灶依赖关系，在肝脏 CT、乳腺 X 线和乳腺 MRI 三种模态上均显著优于现有多视图方法，并对缺失视图具有鲁棒性。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）理论量化因果覆盖差距，并将人类专家知识与 MedGemma-4B 视觉语言模型结合，实现单源域泛化（SDG），在糖尿病视网膜病变分级（8 个数据集）和癫痫灶检测（2 个数据集）上大幅超越现有方法。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](medical_imaging/mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D 深度学习模型，融合纵向 MRI 数据与放疗剂量分布图，用于区分胶质母细胞瘤术后放射性对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM 框架，在潜在空间中显式解耦风格与结构信息，通过高频注入块 (HFIB)、多模态结构特征融合 (MMSF) 和多尺度结构增强 (MSSE) 提取模态不变的多尺度结构先验来引导扩散过程，解决任意模态缺失下 MRI 翻译的解剖不一致和纹理退化问题。

**[NOIR: Neural Operator Mapping for Implicit Representations](medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)**

:   NOIR 将医学图像计算任务重新建模为连续函数空间之间的算子学习问题，通过隐式神经表示(INR)将离散医学信号嵌入连续函数空间，再用神经算子(NO)学习函数间的映射，实现分辨率无关的分割、形状补全、图像翻译和合成。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   本文将 Singleton 和 Batch Processing 设计模式集成到基于 Python 的 RPA 自动化管道中，结合 EfficientNetV2B1 模型实现口腔癌病灶检测，相比 UiPath/Automation Anywhere 等传统 RPA 平台实现 60-100× 的推理加速。

**[Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era](medical_imaging/nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)**

:   Nyxus 是一个面向大数据和 AI 时代的下一代图像特征提取库，支持 2D/3D 数据的 out-of-core 可扩展提取，覆盖 radiomics 和细胞分析两大领域共 261+ 特征，在速度上比 CellProfiler 快 3–131×、比 PyRadiomics/MITK 快数倍至数百倍。

**[LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments](medical_imaging/paper_title_lov3d_grounding_cognitive_prognosis_reasoning_in_longitudinal_3d_bra.md)**

:   LoV3D 提出一套端到端纵向 3D 脑 MRI 视觉-语言模型管线，通过结构化可验证输出设计实现解剖区域评估 + 纵向对比 + 三分类诊断推理，并利用临床加权 Verifier 驱动 DPO 训练（无需人工标注），在 ADNI 上达到 93.7% 三分类准确率且零非相邻诊断错误。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](medical_imaging/prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   ProtoSR 提出从大规模自由文本放射学报告中挖掘模板对齐的原型知识库，并通过原型条件化的后期融合残差模块注入结构化报告预测，在 Rad-ReStruct 基准上实现 SOTA，尤其在细粒度属性问题 (L3) 上获得 72.1% 的相对提升。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](medical_imaging/reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   将深度学习模块（SynthStrip/SynthSeg）模块化替换 SIENA 管线中的经典颅骨剥离和组织分割步骤，在保留管线可解释性的前提下显著提升纵向脑萎缩（PBVC）估计的临床敏感性和鲁棒性。在 ADNI 和 PPMI 两个纵向队列上验证。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

**[SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](medical_imaging/salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)**

:   提出 SALIENT，一个基于小波域扩散的掩码条件生成框架，通过频率感知的可解释优化目标和配对的病灶-掩码体积生成，实现长尾 CT 检测中可控、高效的合成数据增强与精度拯救。首次系统表征增强剂量-反应曲线。

**[SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](medical_imaging/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)**

:   提出 SAW（Surgical Action World），通过四种轻量级条件信号（语言提示、参考帧、组织功能图、工具轨迹）驱动视频扩散模型，实现可控、可扩展的手术动作视频生成，用于罕见动作增强和手术仿真。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出 SCDL 即插即用模块，通过学习类条件代理分布并进行双向对齐（CDBA）+ 语义锚约束（SAC），在嵌入空间显式重塑类条件特征结构，缓解半监督医学影像分割中的监督偏差和表示不平衡。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 多教师多学生半监督框架，通过 Stricter Weighted-Confidence Constraint 实现多源 CBCT 牙齿分割的跨域泛化。

**[Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](medical_imaging/transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   首次将 SegFormer 用于 HR-pQCT 影像的多区域（骨+软组织）自动分割与放射组学分析，发现肌腱组织特征在骨质疏松分类中优于传统骨指标。

**[UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](medical_imaging/ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)**

:   提出 UltrasoundAgents 层次化多智能体框架，通过主智能体定位病灶+子智能体识别属性+证据链推理的流程，对齐乳腺超声临床诊断工作流并实现可追溯的 BI-RADS 分级与良恶性判断。

**[Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的教师-学生结构结合文本概念提示、置信度感知一致性正则化和双流时序一致性，实现 X 光冠脉造影视频的半监督血管分割。

**[UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结病理基础模型 UNI 的稠密空间 token 作为生成器的直接条件信号，实现 H&E 到 IHC 的虚拟染色，单一统一模型同时服务四种 IHC 标记物并达到 SOTA。

**[Unleashing Video Language Models for Fine-grained HRCT Report Generation](medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

**[Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)**

:   通过从 13 个癌症病理基准数据集的图像背景区域（不含临床信息的 20×20 像素裁剪）训练 CNN，发现分类准确率远高于随机猜测（最高 93%），揭示了 CNN 可能依赖数据集偏差而非真正的病理特征来做出诊断判断。

---

## 💬 LLM / NLP { #llm_nlp }

**[Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](llm_nlp/attribute-formed_class-specific_concept_space_endowing_language_bottleneck_model.md)**

:   > 基于摘要：Language Bottleneck Models (LBMs) are proposed to achieve interpretable image recognition by classifying images based on textual concept bottlenecks. However, current LBMs simply list all concepts together as the bottleneck layer, leading to the spurious cue inference problem and cannot generalized

**[Breaking the Low-Rank Dilemma of Linear Attention](llm_nlp/breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   > 基于摘要：The Softmax attention mechanism in Transformer models is notoriously computationally expensive, particularly due to its quadratic complexity, posing significant challenges in vision applications. In contrast, linear attention provides a far more efficient solution by reducing the complexity to linea

**[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](llm_nlp/bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)**

:   > 基于摘要：Can our brain signals faithfully reflect the original visual stimuli, even including high-frequency details? Although human perceptual and cognitive capacities enable us to process and remember visual information, these abilities are constrained by several factors, such as limited attentional resour

**[Building Vision Models upon Heat Conduction](llm_nlp/building_vision_models_upon_heat_conduction.md)**

:   > 基于摘要：Visual representation models leveraging attention mechanisms are challenged by significant computational overhead, particularly when pursuing large receptive fields. In this study, we aim to mitigate this challenge by introducing the Heat Conduction Operator (HCO) built upon the physical heat conduc

**[Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)**

:   > 基于摘要：Traditional text-based person retrieval (TPR) relies on a single-shot text as query to retrieve the target person, assuming that the query completely captures the user's search intent. However, in real-world scenarios, it can be challenging to ensure the information completeness of such single-shot

**[Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](llm_nlp/classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)**

:   > 基于摘要：A person downloading a pre-trained model from the web should be aware of its biases. Existing approaches for bias identification rely on datasets containing labels for the task of interest, something that a non-expert may not have access to, or may not have the necessary resources to collect: this g

**[ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](llm_nlp/comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)**

:   > 基于摘要：Much previous AI research has focused on developing monolithic models to maximize their intelligence, with the primary goal of enhancing performance on specific tasks. In contrast, this work attempts to study using LLM-based agents to design collaborative AI systems autonomously. To explore this pro

**[ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices](llm_nlp/comrope_scalable_and_robust_rotary_position_embedding_parameterized_by_trainable.md)**

:   > 基于摘要：The Transformer architecture has revolutionized various fields since it was proposed, where positional encoding plays an essential role in effectively capturing sequential order and context. Therefore, Rotary Positional Encoding (RoPE) was proposed to alleviate these issues, which integrates positio

**[Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders](llm_nlp/dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   > 基于摘要：Recent 3D content generation pipelines commonly employ Variational Autoencoders (VAEs) to encode shapes into compact latent representations for diffusion-based generation. However, the widely adopted uniform point sampling strategy in Shape VAE training often leads to a significant loss of geometric

**[Empowering LLMs to Understand and Generate Complex Vector Graphics](llm_nlp/empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   > 基于摘要：The unprecedented advancements in Large Language Models (LLMs) have profoundly impacted natural language processing but have yet to fully embrace the realm of scalable vector graphics (SVG) generation. While LLMs encode partial knowledge of SVG data from web pages during training, recent findings su

**[Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention for Region-aware Exposure Correction](llm_nlp/exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)**

:   > 基于摘要：Image exposure correction enhances images captured under diverse real-world conditions by addressing issues of under- and over-exposure, which can result in the loss of critical details and hinder content recognition. While significant advancements have been made, current methods often fail to achie

**[Guiding Human-Object Interactions with Rich Geometry and Relations](llm_nlp/guiding_human-object_interactions_with_rich_geometry_and_relations.md)**

:   > 基于摘要：Human-object interaction (HOI) synthesis is crucial for creating immersive and realistic experiences for applications such as virtual reality. Existing methods often rely on simplified object representations, such as the object's centroid or the nearest point to a human, to achieve physically plausi

**[Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](llm_nlp/imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   > 基于摘要：The Zero-shot Composed Image Retrieval (ZSCIR) requires retrieving images that match the query image and the relative captions.Current methods focus on projecting the query image into the text feature space, subsequently combining them with features of query texts for retrieval. However, retrieving

**[Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](llm_nlp/improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)**

:   > 基于摘要：Employing LLMs for visual generation has recently become a research focus. However, the existing methods primarily transfer the LLM architecture to visual generation but rarely investigate the fundamental differences between language and vision. This oversight may lead to suboptimal utilization of v

**[L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers](llm_nlp/l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas.md)**

:   > 基于摘要：Training-free Neural Architecture Search (NAS) efficiently identifies high-performing neural networks using zero-cost (ZC) proxies. Unlike multi-shot and one-shot NAS approaches, ZC-NAS is both (i) time-efficient, eliminating the need for model training, and (ii) interpretable, with proxy designs of

**[Learning Textual Prompts for Open-World Semi-Supervised Learning](llm_nlp/learning_textual_prompts_for_open-world_semi-supervised_learning.md)**

:   > 基于摘要：Traditional semi-supervised learning achieves significant success in closed-world scenarios. To better align with the openness of the real world, researchers propose open-world semi-supervised learning (OWSSL), which enables models to effectively recognize known and unknown classes even without labe

**[Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness of Samples](llm_nlp/let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)**

:   > 基于摘要：Deep learning models are known to often learn features that spuriously correlate with the class label during training but are irrelevant to the prediction task. Existing methods typically address this issue by annotating potential spurious attributes, or filtering spurious features based on some emp

**[Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](llm_nlp/lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)**

:   > 基于摘要：Our objective is to translate continuous sign language into spoken language text. Inspired by the way human interpreters rely on context for accurate translation, we incorporate additional contextual cues together with the signing video, into a new translation framework. Specifically, besides visual

**[Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration](llm_nlp/making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res.md)**

:   > 基于摘要：Unlike modern native digital videos, the restoration of old films requires addressing specific degradations inherent to analog sources. However, existing specialized methods still fall short compared to general video restoration techniques. In this work, we propose a new baseline to re-examine the c

**[PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](llm_nlp/postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)**

:   > 基于摘要：In poster design, content-aware layout generation is crucial for automatically arranging visual-textual elements on the given image. With limited training data, existing work focused on image-centric enhancement. However, this neglects the diversity of layouts and fails to cope with shape-variant el

**[Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](llm_nlp/rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)**

:   > 基于摘要：Transformers significantly raise the performance limits across various tasks, spurring research into integrating them into spiking neural networks. However, a notable performance gap remains between existing spiking Transformers and their artificial neural network counterparts. Here, we first analyz

**[RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives](llm_nlp/roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_.md)**

:   > 基于摘要：We introduce RoadSocial, a large-scale, diverse VideoQA dataset tailored for generic road event understanding from social media narratives. Unlike existing datasets limited by regional bias, viewpoint bias and expert-driven annotations, RoadSocial captures the global complexity of road events with v

**[Robust Message Embedding via Attention Flow-Based Steganography](llm_nlp/robust_message_embedding_via_attention_flow-based_steganography.md)**

:   > 基于摘要：Image steganography can hide information in a host image and obtain a stego image that is perceptually indistinguishable from the original one. This technique has tremendous potential in scenarios like copyright protection and information retrospection. Some previous studies have proposed to enhance

**[SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](llm_nlp/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)**

:   > 基于摘要：Over the past few years, vision transformers (ViTs) have consistently demonstrated remarkable performance across various visual recognition tasks. However, attempts to enhance their robustness have yielded limited success, mainly focusing on different training strategies, input patch augmentation, o

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](llm_nlp/scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   > 基于摘要：The scaling law has been validated in various domains, such as natural language processing (NLP) and massive computer vision tasks; however, its application to motion generation remains largely unexplored. In this paper, we introduce a scalable motion generation framework that includes the motion to

**[SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](llm_nlp/softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)**

:   > 基于摘要：Recent advancements in deep learning have yielded promising results for the image shadow removal task. However, most existing methods rely on binary pre-generated shadow masks. The binary nature of such masks could potentially lead to artifacts near the boundary between shadow and non-shadow areas.

**[Spiking Transformer with Spatial-Temporal Attention](llm_nlp/spiking_transformer_with_spatial-temporal_attention.md)**

:   > 基于摘要：Spike-based Transformer presents a compelling and energy-efficient alternative to traditional Artificial Neural Network (ANN)-based Transformers, achieving impressive results through sparse binary computations. However, existing spike-based transformers predominantly focus on spatial attention while

**[STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](llm_nlp/staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   > 基于摘要：Spiking Neural Networks (SNNs) have gained significant attention due to their biological plausibility and energy efficiency, making them promising alternatives to Artificial Neural Networks (ANNs). However, the performance gap between SNNs and ANNs remains a substantial challenge hindering the wides

**[Test-Time Visual In-Context Tuning](llm_nlp/test-time_visual_in-context_tuning.md)**

:   > 基于摘要：Visual in-context learning (VICL), as a new paradigm in computer vision, allows the model to rapidly adapt to various tasks with only a handful of prompts and examples. While effective, the existing VICL paradigm exhibits poor generalizability under distribution shifts. In this work, we propose test

**[The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generationf](llm_nlp/the_change_you_want_to_detect_semantic_change_detection_in_earth_observation_wit.md)**

:   > 基于摘要：Bi-temporal change detection at scale based on Very High Resolution (VHR) images is crucial for Earth monitoring. Such task remains poorly addressed even in the deep learning era: it either requires large volumes of annotated data - in the semantic case - or is limited to restricted datasets for bin

**[TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](llm_nlp/tide_training_locally_interpretable_domain_generalization_models_enables_test-ti.md)**

:   > 基于摘要：We consider the problem of single-source domain generalization. Existing methods typically rely on extensive augmentations to synthetically cover diverse domains during training. However, they struggle with semantic shifts (e.g., background and viewpoint changes), as they often learn global features

**[VIRES: Video Instance Repainting via Sketch and Text Guided Generation](llm_nlp/vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)**

:   > 基于摘要：We introduce VIRES, a video instance repainting method with sketch and text guidance, enabling video instance repainting, replacement, generation, and removal. Existing approaches struggle with temporal consistency and accurate alignment with the provided sketch sequence. VIRES leverages the generat

---

## 🧊 3D 视觉 { #3d_vision }

**[3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](3d_vision/3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d_vision/3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)**

:   提出3D-GSW，首个专为3D Gaussian Splatting设计的鲁棒数字水印方法，通过频率引导致密化（FGD）移除冗余高斯并在高频区域分裂高斯来增强鲁棒性，结合梯度掩码和小波子带损失保持渲染质量，在Blender/LLFF/Mip-NeRF 360数据集上同时实现了最优的水印鲁棒性和渲染质量。

**[3D-HGS: 3D Half-Gaussian Splatting](3d_vision/3d-hgs_3d_half-gaussian_splatting.md)**

:   提出3D Half-Gaussian (3D-HGS)核函数——用一个分割平面将3D高斯分成两半，每半有独立不透明度，作为**即插即用**的重建核替换标准高斯核，在不牺牲渲染速度的前提下显著提升形状和颜色不连续处的渲染质量，在Mip-NeRF360/T&T/Deep Blending上全面超越所有SOTA方法。

**[3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](3d_vision/3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)**

:   提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

**[3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d_vision/3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)**

:   提出3D-Mem——基于"记忆快照"的3D场景记忆框架，用少量精选多视角图像紧凑表示已探索区域，结合Frontier Snapshot表示未探索区域，配合VLM实现高效的具身探索与推理。

**[3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d_vision/3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)**

:   提出3D-SLNR，一种超轻量神经3D表示——基于锚定在点云支撑点上的带限局部SDF集合定义全局SDF，每个局部SDF仅由一个微型MLP参数化（无隐特征），通过可学习的位置/旋转/缩放适应复杂几何，配合并行查找算法和剪枝-扩展策略，以不到先前方法1/5的内存实现SOTA重建质量。

**[3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)**

:   用3D光滑凸体（Smooth Convex）替代高斯基元进行辐射场渲染，通过点集定义凸包+LogSumExp平滑化+自定义CUDA光栅化器，在T&T和Deep Blending上超越3DGS，且所需基元更少。

**[3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_vision/3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)**

:   提出 CrossTooth，通过基于曲率先验的选择性下采样（边界区域顶点密度提升 10-15%）和多视角渲染图像的跨模态边界特征融合，在 3DTeethSeg'22 公开数据集上实现 95.86% mIoU 和 82.05% boundary IoU，分别比之前 SOTA（ToothGroupNet）提升 2.3% 和 5.7%。

**[3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations](3d_vision/3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)**

:   提出一种紧凑张量表示的3D高斯头部头像方法——用三平面存储中性表情的静态外观，用轻量1D特征线存储每个blendshape的动态纹理（不透明度偏移），仅需**10MB存储**即可实现300FPS实时渲染和准确的动态面部细节捕捉，在Nersemble数据集上PSNR和存储效率全面超越GA、GBS和GHA。

**[3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_vision/3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)**

:   提出3DGIC，通过**深度引导的跨视角一致修复**框架实现3D高斯场景中的物体移除与修补——利用渲染深度图从其他视角发现被掩码区域中的可见背景像素来精化修补掩码，再用参考视角的2D修补结果通过3D投影约束其他视角的一致性，在SPIn-NeRF数据集上FID和LPIPS全面超越现有方法。

**[3D Student Splatting and Scooping (SSS)](3d_vision/3d_student_splatting_and_scooping.md)**

:   提出SSS（Student Splatting and Scooping），用前所未有的三重创新改进3DGS范式：(1) 用**Student-t分布**替代高斯分布作为混合组件（可学习的尾部厚度，从Cauchy到Gaussian连续变化）；(2) 引入**负密度组件**（scooping减去颜色）扩展到非单调混合模型；(3) 用**SGHMC采样**替代SGD解耦参数优化，在Mip-NeRF360/T&T/Deep Blending上6/9指标取得最优，且参数效率极高——用**最少18%**的组件数即可匹配或超越3DGS。

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

**[FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT](3d_vision/framevggt_frame_evidence_rolling_memory_for_streaming_vggt.md)**

:   提出 FrameVGGT，将流式 VGGT 的 KV 缓存从 token 级保留重组为帧级证据块保留，通过中期记忆库+稀疏锚点的双层有界内存结构，在固定内存预算下保持更连贯的几何支撑，实现长序列3D重建/深度/位姿估计的精度-内存最优权衡。

**[HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)**

:   提出 HOI3DGen 框架，通过MLLM自动标注高质量交互数据 + 视角条件化微调扩散模型 + 3D提升与SMPL配准，首次实现从文本精确控制接触语义的高质量3D人物交互生成，在文本一致性上超越基线4-15倍。

**[Hybrid eTFCE-GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](3d_vision/hybrid_etfce-grf_exact_cluster-size_retrieval_with_analytical_p-values_for_voxel.md)**

:   将 eTFCE 的并查集精确聚类大小查询与 pTFCE 的解析 GRF p 值推断结合，首次在单一框架中实现精确聚类检索+无需置换检验的统计推断，速度比置换 TFCE 快 1300 倍，在全脑体素形态测量中保持严格 FWER 控制。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)**

:   提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模进行多曝光融合 + MetaNet 预测场景自适应色调映射器，从未标定多曝光 LDR 图像一次前向推理重建 HDR 3D 高斯，速度比优化方法快 ~700 倍。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](3d_vision/jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)**

:   提出 JOPP-3D 框架，通过将全景图切线分解为透视图像、利用 SAM+CLIP 进行3D实例-语义对齐，首次实现对3D点云和全景图像的联合开放词汇语义分割，在 Stanford-2D-3D-s 和 ToF-360 数据集上超越现有方法。

**[Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](3d_vision/mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)**

:   提出 Mobile-GS，通过深度感知的无序渲染（消除排序瓶颈）+ 神经视角依赖增强 + 一阶SH蒸馏 + 神经向量量化 + 贡献度剪枝，首次在 Snapdragon 8 Gen 3 手机 GPU 上实现 116 FPS 实时高斯溅射渲染，存储仅 4.6MB 且视觉质量与原始 3DGS 相当。

**[MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出 MotionAnyMesh，一种零样本框架，通过 SP4D 运动学先验引导 VLM 推理消除幻觉 + 物理约束轨迹优化保证无碰撞，将静态3D网格自动转化为仿真可用的铰接数字孪生，物理可执行率达 87%，是现有最好方法的近两倍。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](3d_vision/node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   提出 Node-RF，将 Neural ODE 与动态 NeRF 紧密耦合，用潜在向量的 ODE 演化建模场景连续时间动力学，实现超出训练序列的长程时序外推和跨轨迹泛化，无需光流或深度监督。

**[P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](3d_vision/p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)**

:   提出 P-SLCR，一种原型库驱动的无监督点云语义分割方法，通过将点分离为"一致"和"模糊"两类，用一致结构学习对齐一致点与原型 + 语义关系一致性推理约束两个原型库，在 S3DIS 上无监督达 47.1% mIoU，超越全监督 PointNet。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，首个在3D摄影测量空间进行全景拼接的 Transformer 框架，利用预训练 VGGT 骨干获取3D感知的多视角特征对齐 + 多特征联合优化接缝检测，支持2到数百张输入图像，在弱纹理/大视差/重复模式场景下成功率达97.8%。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，将隐式神经表示 (INR/SIREN) 与预训练扩散模型先验结合，通过 proximal loss 在每个 DDIM 时间步用扩散去噪输出正则化 INR 重建，在稀疏视角中子 CT（低至 4-5 个视角）上超越 FBP、纯 INR、DD3IP 和经典 MBIR(qGGMRF) 方法。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](3d_vision/rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)**

:   Rewis3d 利用前馈 3D 重建（MapAnything）从 2D 视频中获取 3D 点云作为辅助监督信号，通过双 Student-Teacher 架构和加权跨模态一致性 (CMC) 损失，在仅使用稀疏标注（点/涂鸦/粗标记）的情况下将弱监督 2D 语义分割性能提升 2-7% mIoU，推理时仍为纯 2D。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](3d_vision/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一个即插即用的背景引导原型富化框架，在基类训练后用类无关分割模型从背景区域挖掘伪实例建立 Instance Prototype Bank (IPB)，当新类别以少样本方式出现时，通过 Contextual Prototype Retrieval (CPR) 和 Attention-Based Prototype Enrichment (APE) 融合背景原型与少样本原型，在 ScanNet/S3DIS 上新类 IoU 提升最高 6.98%。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   针对 3DGS 的资源瞄准攻击（通过投毒训练图像触发高斯过度增长导致资源耗尽），提出频域防御：3D 频率滤波器通过将高斯协方差与频谱响应关联实现频率感知剪枝，2D 频谱正则化通过熵惩罚渲染图像的角向能量各向异性来抑制攻击噪声，实现高斯数量压缩 5.92×、内存减少 3.66×、速度提升 4.34×。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](3d_vision/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。

---

## 🎨 图像生成 { #image_generation }

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，用双向布朗桥扩散模型建模地面 LSST 和太空 Euclid 两大天文巡天之间的随机映射关系，实现概率性跨巡天翻译与稀有事件检测（强引力透镜），并证明 epsilon-prediction 训练目标兼具重建质量和似然性优势。

**[Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](image_generation/beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)**

:   系统性地将学习式图像处理中卷积的替代/扩展算子组织为五大家族（分解型、自适应加权型、基自适应型、积分/核型和注意力型），并从线性、局部性、等变性、计算成本和任务适用性等多个维度进行比较分析。

**[BiGain: Unified Token Compression for Joint Generation and Classification](image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 首次将扩散模型的 token 压缩重新定义为生成+分类的双目标优化问题，提出拉普拉斯门控 token 合并（L-GTM）和插值-外推 KV 下采样（IE-KVD）两个频率感知算子，在保持生成质量同时显著提升分类准确率（ImageNet-1K 70%合并比下 Acc +7.15%，FID -0.34）。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)**

:   提出 coDrawAgents，由 Interpreter、Planner、Checker、Painter 四个专家 agent 组成的交互式多智能体对话框架，通过分而治之的增量布局规划、视觉上下文感知推理和显式错误纠正，在 GenEval 上达到 0.94（SOTA）、DPG-Bench 上 85.17（SOTA）。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)**

:   DiT-IC 将预训练 T2I 扩散 Transformer 适配为单步图像压缩重建模型，在 32x 下采样的深层潜空间工作，通过方差引导重建流、自蒸馏对齐和潜变量条件引导三种对齐机制，实现 SOTA 感知质量且解码比现有扩散 codec 快 30 倍。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。

**[Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](image_generation/enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)**

:   提出 DIAE，通过多模态美学感知模块（MAP）将模糊美学指令转化为 HSV/轮廓图+文本的多模态控制信号，并构建"非完美配对"数据集 IIAEData 配合双分支监督策略实现弱监督美学增强，在 LAION 和 MLLM 美学评分上达 SOTA。

**[EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](image_generation/evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)**

:   EvoTok 提出了一种基于残差潜在演化（Residual Latent Evolution）的统一图像 tokenizer，通过在共享潜空间中级联残差向量量化，使表示从浅层的像素级细节渐进演化到深层的语义级抽象，在仅用 13M 图像训练的情况下实现了 0.43 rFID 的重建质量，并在 7/9 个理解 benchmark 和 GenEval/GenAI-Bench 上取得优异效果。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](image_generation/fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。

**[Generation of Maximal Snake Polyominoes Using a Deep Neural Network](image_generation/generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)**

:   将 DDPM 应用于生成最大蛇形多联骨牌，提出精简版 Structured Pixel Space Diffusion（SPS Diffusion），在训练到 14x14 正方网格的情况下泛化到 28x28 并生成有效蛇形，部分结果超越已知最大长度下界。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](image_generation/interedit_navigating_text-guided_multi-human_3d_motion_editing.md)**

:   提出 InterEdit，首个文本引导的多人 3D 运动交互编辑框架，通过 Semantic-Aware Plan Token Alignment 和 Interaction-Aware Frequency Token Alignment 在扩散模型中实现语义编辑的同时保持多人之间的时空耦合关系。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   揭示 DiT 的计算在空间 token 上均匀分配（不会把多余计算重分配到困难区域），提出 ELIT——在 DiT 中插入可变长度的 latent interface（Read/Write 交叉注意力），训练时随机丢弃尾部 latent 学出重要性排序，推理时通过调节 latent 数量实现平滑的质量-FLOPs 权衡，ImageNet 512px 上 FID 降低 53%。

**[Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](image_generation/overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)**

:   提出 Concept-Gated Visual Distillation (CGVD)，一种无需训练的推理时框架，通过语言指令解析 → SAM3 分割 → 集合论交叉验证 → LaMa 修复的流水线，从 VLA 模型的视觉输入中选择性移除语义干扰物，在高度杂乱场景中将 π₀ 的操作成功率从 43.0% 提升至 77.5%。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 去噪器（Auto-Correction + Directional Correction + Score-Based Denoising 三阶段），解决将 score-based 扩散先验嵌入 ADMM-PnP 框架时的流形不匹配问题，并首次建立了 score-based 去噪器在 ADMM 中的收敛性理论保证，在去噪、修复、去模糊、超分辨、相位恢复、HDR 等逆问题上一致超越现有基线。

**[Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](image_generation/trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)**

:   提出 FIRM 框架——通过"差异优先"（编辑）和"计划-打分"（生成）的数据构建流水线训练专用奖励模型（FIRM-Edit-8B / FIRM-Gen-8B），配合"Base-and-Bonus"奖励策略（CME/QMA）解决 RL 中的奖励 hacking 问题，在图像编辑和 T2I 生成任务上均取得 SOTA。

**[Unicom Unified Multimodal Modeling Via Compressed Continuous Semantic Representa](image_generation/unicom_unified_multimodal_modeling_via_compressed_continuous_semantic_representa.md)**

:   提出 UniCom，通过对 VLM 连续语义特征进行**通道维度压缩**（而非空间下采样），构建紧凑连续表示空间，用 Transfusion 架构统一多模态理解与生成，在统一模型中达到 SOTA 生成质量。

**[V-Bridge Bridging Video Generative Priors To Versatile Few-Shot Image Restoratio](image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像复原重新定义为**渐进式视频生成过程**，利用预训练视频生成模型（Wan2.2-TI2V-5B）的先验知识，仅用 1,000 个多任务训练样本（不到现有方法的 2%）即可实现竞争力的多任务图像复原。

**[Visual-Erm Reward Modeling For Visual Equivalence](image_generation/visual-erm_reward_modeling_for_visual_equivalence.md)**

:   提出 Visual-ERM，一个多模态生成式奖励模型，在视觉空间中直接评估 vision-to-code 任务的渲染质量，提供细粒度、可解释、任务无关的奖励信号，用于 RL 训练和测试时缩放。

**[When To Lock Attention Training-Free Kv Control In Video Diffusion](image_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，一种基于扩散幻觉检测的免训练视频编辑框架，通过动态调度 KV 缓存融合比例和 CFG 引导尺度，在保持背景一致性的同时增强前景生成质量。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](llm_alignment/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)**

:   > 注：本笔记基于论文摘要撰写，尚未阅读全文。待下载完整论文后补充详细方法和实验分析。

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](llm_alignment/bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种求解可转向等变 CNN 核约束方程的替代方法，通过在不动点处求解更简单的不变性条件再"转向"到任意点，绕过了计算 Clebsch-Gordan 系数的需要，为 SO(2)、O(2)、SO(3)、O(3) 及 Lorentz 群给出了显式的核基底公式。

**[Boost Your Human Image Generation Model via Direct Preference Optimization](llm_alignment/boost_your_human_image_generation_model_via_direct_preference_optimization.md)**

:   提出 HG-DPO，以真实人像作为 DPO 的 winning image（而非生成图像对）+ 三阶段课程学习（Easy/Normal/Hard）渐进弥合生成-真实图像分布 gap + 统计匹配损失解决色偏，FID 从 37.34 降至 29.41（-21.4%），CI-Q 0.906→0.934，win-rate 超越 Diffusion-DPO 达 99.97%。

**[CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation](llm_alignment/cad-llama_leveraging_large_language_models_for_computer-aided_design_parametric_.md)**

:   > 基于摘要：Recently, Large Language Models (LLMs) have achieved significant success, prompting increased interest in expanding their generative capabilities beyond general text into domain-specific areas. This study investigates the generation of parametric sequences for computer-aided design (CAD) models usin

**[Calibrated Multi-Preference Optimization for Aligning Diffusion Models](llm_alignment/calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)**

:   > 基于摘要：Aligning text-to-image (T2I) diffusion models with prefer-ence optimization is valuable for human-annotated datasets, but the heavy cost of manual data collection limits scalability. Using reward models offers an alternative, however, current preference optimization methods fall short in exploiting

**[Continual SFT Matches Multimodal RLHF with Negative Supervision](llm_alignment/continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)**

:   通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

**[Curriculum Direct Preference Optimization for Diffusion and Consistency Models](llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)**

:   > 基于摘要：Direct Preference Optimization (DPO) has been proposed as an effective and efficient alternative to reinforcement learning from human feedback (RLHF). In this paper, we propose a novel and enhanced version of DPO based on curriculum learning for text-to-image generation. Our method is divided into t

**[Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](llm_alignment/debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)**

:   > 基于摘要：Multimodal Large Language Models (MLLMs) excel in various tasks, yet often struggle with modality bias, tending to rely heavily on a single modality or prior knowledge when generating responses. In this paper, we propose a debiased preference optimization dataset, RLAIF-V-Bias, and introduce a Noise

**[Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal Large Language Models?](llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)**

:   > 基于摘要：Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically design

**[Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation](llm_alignment/enhancing_sam_with_efficient_prompting_and_preference_optimization_for_semi-supe.md)**

:   > 基于摘要：Foundational models such as the Segment Anything Model (SAM) are gaining traction in medical imaging segmentation, supporting multiple downstream tasks. However, such models are supervised in nature, still relying on large annotated datasets or prompts supplied by experts. Conventional techniques su

**[InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](llm_alignment/inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)**

:   > 基于摘要：Without using explicit reward, direct preference optimization (DPO) employs paired human preference data to fine-tune generative models, a method that has garnered considerable attention in large language models (LLMs). However, exploration of aligning text-to-image (T2I) diffusion models with human

**[Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](llm_alignment/jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)**

:   > 基于摘要：Non-transferable learning (NTL) has been proposed to protect model intellectual property (IP) by creating a "non-transferable barrier" to restrict generalization from authorized to unauthorized domains. Recently, well-designed attack, which restores the unauthorized-domain performance by fine-tuning

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](llm_alignment/physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)**

:   提出 PhysMoDPO，将 Direct Preference Optimization 应用于文本驱动的人体运动生成，通过将全身控制器（WBC）集成到训练 pipeline 中计算基于物理的奖励来构造偏好数据，使生成运动同时满足物理约束和文本指令，并在 Unitree G1 机器人上实现零样本部署。

**[SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](llm_alignment/symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)**

:   > 基于摘要：As language models continue to scale, Large Language Models (LLMs) have exhibited emerging capabilities in In-Context Learning (ICL), enabling them to solve language tasks by prefixing a few in-context demonstrations (ICDs) as context. Inspired by these advancements, researchers have extended these

**[Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](llm_alignment/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)**

:   提出 Task Preference Optimization（TPO），通过可学习的任务 token 将视觉任务专用头（区域定位/时序定位/分割）接入 MLLM，利用视觉任务标注作为"任务偏好"反向优化 MLLM，在不损害对话能力的前提下大幅提升细粒度视觉理解，VideoChat 基线上平均提升 14.6%。

---

## 👁️ 多模态 VLM { #multimodal_vlm }

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](multimodal_vlm/a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

**[Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](multimodal_vlm/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)**

:   提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](multimodal_vlm/codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过 scaling 分析发现 STEM 视觉推理的真正瓶颈是感知而非推理，提出用可执行 Python 代码作为精确感知媒介——构建 ICC-1M 数据集（Image-Caption-Code 三元组）训练模型，在 STEM 感知基准上 CodePercept-8B 比 Qwen3-VL-8B 提升 +3.0%-12.3%。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL 框架，通过对抗性锚点（DPGD）精准探测新旧任务语义边界的脆弱区域，结合跨模态几何蒸馏（ACGD）和文本语义正则化（TSGR）保护 VLM 的跨模态几何结构，在五个持续学习 benchmark 上达到 SOTA。

**[ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)**

:   提出 Espire，一个基于仿真环境的具身空间推理诊断基准，将 VLM 评估分解为定位和执行两阶段，通过全生成式范式系统评估 VLM 在多种空间推理维度和粒度上的能力。

**[ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](multimodal_vlm/forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)**

:   发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](multimodal_vlm/geometry-guided_camera_motion_understanding_in_videollms.md)**

:   提出一个从基准构建、诊断到注入的完整框架，通过 3D 基础模型（VGGT）提取相机运动线索并以结构化提示注入 VideoLLM，实现无需训练的相机运动感知增强。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](multimodal_vlm/hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)**

:   通过对 attention 公式的精确分解，揭示 ICL 的效果本质上是 query-dependent 的标准自注意力输出与上下文 value 的动态混合，据此提出直接参数化"虚拟 KV 对"（低秩分解）来高保真模拟 ICL，仅 2.2M 参数即超越 MimIC/LoRA，且训练快 7.5 倍。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](multimodal_vlm/mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   构建首个包含正负语义描述的视觉定位数据集 D-Negation，并提出 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义约束显著增强 grounding 模型对否定语义的理解能力。

**[Multimodal OCR: Parse Anything from Documents](multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)**

:   提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。

**[Revisiting Model Stitching in the Foundation Model Era](multimodal_vlm/revisiting_model_stitching_in_the_foundation_model_era.md)**

:   系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](multimodal_vlm/spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架中系统替换图像编码器（CLIP/SigLIP/SigLIP2/AIMv2）和引入 2D-RoPE 位置编码，发现 VLM 的空间推理能力主要由编码器的训练目标决定，指望仅靠 2D 位置结构改善空间理解是不够的。

**[CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   发现现有 VLM（包括 GPT-5.2、Gemini-2.5）在拓扑异常检测上几乎为零（F1@0.5 < 1.5%），提出 Topo-R1 框架通过 SFT + GRPO（含拓扑感知复合 reward，集成 type-aware Hungarian matching + clDice）赋予 VLM 拓扑感知能力，最佳 F1@0.5 达 45.2%。

**[Towards Faithful Multimodal Concept Bottleneck Models](multimodal_vlm/towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](autonomous_driving/3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)**

:   提出3D-AVS，首个针对LiDAR点云的**自动词表分割**方法：无需用户指定目标类别，系统自动从图像和点云中识别场景中存在的语义实体并生成词表，再用开放词表分割器完成逐点语义分割，在nuScenes和ScanNet200上展示了生成精细语义类别的能力。

**[ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](autonomous_driving/3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)**

:   提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

**[A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](autonomous_driving/a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)**

:   本文提出首个将 ASP 符号推理决策以可学习嵌入形式直接嵌入端到端规划器轨迹解码的神经-符号框架，用 LLM 动态提取场景规则、Clingo 求解器进行逻辑仲裁、可微 KBM 生成物理可行轨迹并配合神经残差修正，在 nuScenes 上 L₂ 误差 0.57m、碰撞率 0.075%、TPC 0.47m 全面超越 MomAD。

**[PAP: A Prediction-as-Perception Framework for 3D Object Detection](autonomous_driving/a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   PAP 受人脑"预测性感知"启发，将上一帧轨迹预测结果作为当前帧感知模块的 query 输入替代部分随机 query，在 UniAD 上实现 AMOTA 提升 10%（0.359→0.395）、推理速度提升 15%（14→16 FPS）和训练时间缩短 14%。

**[CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](autonomous_driving/cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)**

:   CAWM-Mamba 首次提出端到端统一处理红外-可见光图像融合与复合恶劣天气（如雾+雨、雨+雪）场景的框架，通过天气感知预处理、跨模态特征交互和小波域频率-SSM 解耦多频退化，在 AWMM-100K 和标准融合数据集上全面超越 SOTA。

**[CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](autonomous_driving/composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)**

:   CompoSIA 提出一种基于 Flow Matching DiT 的组合式驾驶视频生成框架，通过解耦结构（3D bbox）、身份（单参考图像）和自车动作（相机轨迹）三类控制信号的注入方式，实现精细独立控制和组合编辑，用于系统化合成对抗性驾驶场景，FVD 提升 17%，碰撞率增加 173%。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出基于 LiDAR 反射率引导的显著高斯泼溅方法，引入结构感知的显著高斯表示（由 LiDAR 几何和反射率特征点初始化）和光照不变的反射率通道作为额外约束，在 Waymo 数据集挑战场景（复杂光照）上 PSNR 超越 OmniRe 1.18 dB。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](autonomous_driving/m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对多相机输入不完整时的语义占用预测问题，提出多视角掩码重建（MMR）模块利用相邻相机重叠区域恢复缺失视角特征，以及特征记忆模块（FMM）通过类级语义原型精炼不确定体素特征，在缺失后视角设置下 IoU 提升 4.93%。

**[MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)**

:   MapGCLR 提出地理空间对比学习方法，通过强制多次行驶中地理空间重叠区域的 BEV 特征一致性来改善在线矢量化 HD 地图构建的 BEV 编码器，在仅 5% 标注数据下实现 42% 的相对 mAP 提升。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出纯视觉端到端的全向开放词汇占用预测框架，通过极坐标螺旋 Mamba（PsM）建模全向空间连续性、占用代价聚合（OCA）统一几何和语义监督、以及无梯度自然模态对齐（NMA）桥接像素-体素-文本模态间隙，在 QuadOcc 和 Human360Occ 上达到 SOTA。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)**

:   首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](autonomous_driving/single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)**

:   利用 microLED-on-CMOS 超快数字光投影器实现基于单像素成像（SPI）的 MNIST 图像分类，在 1.2 kfps 帧率下达到 >90% 分类精度，完全绕过图像重建直接从时序光信号分类。

**[Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)**

:   SG-NLF 提出一种无需精确位姿的 LiDAR NeRF 框架，通过混合频谱-几何表征重建平滑几何、置信度感知位姿图实现全局对齐、对抗学习增强跨帧一致性，在低频 LiDAR 场景下重建质量和位姿精度分别超越 SOTA 35.8% 和 68.8%。

**[VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](autonomous_driving/vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   VIRD 通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视角不变表征，实现无需方向先验的全向跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。

---

## ✂️ 语义分割 { #segmentation }

**[2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](segmentation/2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)**

:   提出2DMamba，首个具有高效并行算法的**原生2D选择性状态空间模型**，通过保持2D空间连续性（而非展平为1D序列）来建模WSI中的patch间关系，在10个公共病理数据集上全面超越1D Mamba方法，并在ImageNet分类和ADE20K分割上也有提升。

**[Binwang2Hfnet Geogran-Aware Hierarchical Feature Fusion Network For Salient Obje](segmentation/binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)**

:   提出 G2HFNet，通过多尺度细节增强 (MDE)、双分支几何-粒度互补 (DGC)、深层语义感知 (DSP) 和局部-全局引导融合 (LGF) 四个模块，针对不同层级特征设计差异化优化策略，在三个遥感显著性检测数据集上全面超越 SOTA。

**[Crossearth-Sar A Sar-Centric And Billion-Scale Geospatial Foundation Model For D](segmentation/crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，基于物理引导的稀疏混合专家 (MoE) 架构，构建了包含 200K 图像的训练集和 22 个子基准的评估体系，在 20/22 个跨域语义分割基准上达到 SOTA。

**[Efficient Rgb-D Scene Understanding Via Multi-Task Adaptive Learning And Cross-D](segmentation/efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一个高效 RGB-D 多任务场景理解网络，通过改进融合编码器利用冗余特征加速推理，引入归一化聚焦通道层 (NFCL) 和上下文特征交互层 (CFIL) 进行跨维度特征引导，并设计多任务自适应损失函数动态调整任务权重，在 NYUv2/SUN RGB-D/Cityscapes 上达到 SOTA。

**[HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](segmentation/hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)**

:   HFP-SAM 提出分层频率提示的 SAM 框架，通过频率引导适配器（FGA）注入海洋场景信息、频率感知点选择（FPS）自动生成高质量点提示、全视图 Mamba（FVM）高效解码，在四个海洋动物分割数据集上取得 SOTA。

**[PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](segmentation/picosam3_real-time_in-sensor_region-of-interest_segmentation.md)**

:   PicoSAM3 是一个 1.3M 参数的超轻量可提示分割模型，通过 ROI 隐式提示编码、密集 CNN 架构（无 Transformer）、SAM3 知识蒸馏和 INT8 量化，在 COCO 上达 65.45% mIoU，并实现在 Sony IMX500 视觉传感器上 11.82ms 实时推理。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](segmentation/prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)**

:   SAM FTI-FDet 提出基于轻量 SAM 的自动提示实例分割框架，通过 Transformer 解码器式的提示生成器自动产生任务相关提示、自适应特征分发器融合多尺度特征、TinyViT backbone 降低计算开销，在货运列车故障检测数据集上达 74.6 $AP^{box}$ / 74.2 $AP^{mask}$。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](segmentation/rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   RDNet 针对遥感图像中目标尺度剧烈变化的问题，提出区域比例感知的动态自适应显著性检测网络，通过动态自适应细节感知模块（DAD，根据目标区域比例选择不同大小卷积核组合）、频率匹配上下文增强模块（FCE，小波域特征交互）和区域比例感知定位模块（RPL，交叉注意力+比例引导），在 EORSSD/ORSSD/ORSI-4199 三个数据集上取得 SOTA。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出区域引导选择性优化网络 RSONet，通过两阶段（区域引导+显著性生成）解决 RGB 与热红外图像中显著区域不一致问题，利用相似度分数自动选择信息更准确的模态主导后续融合。

**[SAP: Segment Any 4K Panorama](segmentation/sap_segment_any_4k_panorama.md)**

:   将 360° 全景图分割重新定义为透视视频分割问题，通过沿 zigzag 轨迹分解全景图为重叠 patch 序列并微调 SAM2 的 memory 模块，配合 183K 合成 4K 全景图的大规模训练，实现零样本全景分割 +17.2 mIoU 的提升。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出SGMA框架，通过语义引导融合(SGF)模块构建全局语义原型估计模态鲁棒性并自适应加权融合，以及模态感知采样(MAS)模块动态优先训练脆弱模态，解决遥感不完整多模态分割中的模态不平衡、类内变化和跨模态异质性三大挑战。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](segmentation/sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出SPARROW框架，通过目标特定跟踪特征(TSF)和双提示(BOX+SEG)机制，解决视频MLLM中时序引用一致性差和首帧初始化不稳定的问题，在6个基准上对3个主流视频MLLM均取得一致提升。

**[Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在预训练视觉语言模型中引入轻量级表达感知的混合专家（MoE）精细化，分别在 backbone 层（SERA-Adapter）和融合层（SERA-Fusion）进行专家路由，仅更新 <1% 参数即在参考图像分割基准上达到 SOTA。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[Associative Transformer](llm_efficiency/associative_transformer.md)**

:   > 基于摘要：Emerging from the pairwise attention in conventional Transformers, there is a growing interest in sparse attention mechanisms that align more closely with localized, contextual learning in the biological brain. Existing studies such as the Coordination method employ iterative cross-attention mechani

**[CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](llm_efficiency/care_transformer_mobile-friendly_linear_visual_transformer_via_decoupled_dual_in.md)**

:   > 基于摘要：Recently, large efforts have been made to design efficient linear-complexity visual Transformers. However, current linear attention models are generally unsuitable to be deployed in resource-constrained mobile devices, due to suffering from either few efficiency gains or significant accuracy drops.

**[Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](llm_efficiency/efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](llm_efficiency/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)**

:   > 基于摘要：Deep Neural Networks (DNNs) have achieved remarkable success in a variety of tasks, particularly in terms of prediction accuracy. However, in real-world scenarios, especially in safety-critical applications, accuracy alone is insufficient; reliable uncertainty estimates are essential. Modern DNNs, o

**[KAC: Kolmogorov-Arnold Classifier for Continual Learning](llm_efficiency/kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   > 基于摘要：Continual learning requires models to train continuously across consecutive tasks without forgetting. Most existing methods utilize linear classifiers, which struggle to maintain a stable classification space while learning new tasks. Inspired by the success of Kolmogorov-Arnold Networks (KAN) in pr

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](llm_efficiency/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   > 基于摘要：Continual learning (CL) aims to enable learning systems to acquire new knowledge constantly without forgetting previously learned information. CL faces the challenge of mitigating catastrophic forgetting while maintaining interpretability across tasks.Most existing CL methods focus primarily on pres

**[LOCORE: Image Re-ranking with Long-Context Sequence Modeling](llm_efficiency/locore_image_re-ranking_with_long-context_sequence_modeling.md)**

:   > 基于摘要：We introduce LOCORE, Long-Context Re-ranker, a model that takes as input local descriptors corresponding to an image query and a list of gallery images and outputs similarity scores between the query and each gallery image. This model is used for image retrieval, where typically a first ranking is p

**[Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](llm_efficiency/low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)**

:   > 基于摘要：In security-sensitive fields, data should be encrypted to protect against unauthorized access and maintain confidentiality throughout processing. However, traditional networks like ViTs and CNNs return different results when processing original data versus its encrypted form, meaning that they requi

**[SEC-Prompt:SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](llm_efficiency/sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)**

:   > 基于摘要：Few-shot class-incremental learning (FSCIL) presents a significant challenge in machine learning, requiring models to integrate new classes from limited examples while preserving performance on previously learned classes. Recently, prompt-based CIL approaches leverage ample data to train prompts, ef

**[Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](llm_efficiency/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)**

:   > 基于摘要：The CLIP model has demonstrated significant advancements in aligning visual and language modalities through large-scale pre-training on image-text pairs, enabling strong zero-shot classification and retrieval capabilities on various domains. However, CLIP's training remains computationally intensive

**[Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](llm_efficiency/spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)**

:   > 基于摘要：Humans perceive and understand real-world spaces through a stream of visual observations. Therefore, the ability to streamingly maintain and update spatial evidence from potentially unbounded video streams is essential for spatial intelligence. The core challenge is not simply longer context windows

**[Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](llm_efficiency/spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)**

:   > 基于摘要：Transformers have demonstrated outstanding performance across a wide range of tasks, owing to their self-attention mechanism, but they are highly energy-consuming. Spiking Neural Networks have emerged as a promising energy-efficient alternative to traditional Artificial Neural Networks, leveraging e

---

## 🦾 LLM Agent { #llm_agent }

**[ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Inpainting](llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)**

:   提出 ATA（Adaptive Transformation Agent），解决文本引导的主体位置可变背景修复任务，通过 PosAgent Block 自适应预测位移、Reverse Displacement Transform 模块和 Position Switch Embedding，在保持修复质量的同时实现主体位置的灵活调整。

**[ChatHuman: Chatting about 3D Humans with Tools](llm_agent/chathuman_chatting_about_3d_humans_with_tools.md)**

:   提出 ChatHuman，一个基于 LLM 的语言驱动系统，通过自动选择和集成专门的 3D 人体分析工具（3D 姿态估计、形状恢复、接触检测、人物交互分析、情感识别等），利用学术论文作为工具使用说明和 RAG（检索增强生成）创建 in-context 示例以管理新工具，在工具选择准确率和整体 3D 人体任务性能上超越现有 LLM 模型。

**[Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields](llm_agent/feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_.md)**

:   提出 Feature4X，一个通用框架，从任意单目视频通过动态优化策略将多种 2D 视觉基础模型（SAM2、InternVideo2 等）的功能蒸馏到统一的 4D 高斯特征场中，首次实现基于 Gaussian Splatting 的视频基础模型 4D 特征提升，支持新视角下的 segment anything、几何/外观编辑和自由形式 VQA。

**[GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)**

:   提出 GUI-Xplore 数据集（312 个应用、32K+ QA 对、五层级任务）和 Xplore-Agent 框架（Action-aware GUI 建模 + GUI Transition Graph 推理），通过模拟"先探索再推理"的人类策略，在陌生应用上比 SOTA GUI Agent 提升约 10% StepSR。

**[RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression](llm_agent/rl-rc-dot_a_block-level_rl_agent_for_task-aware_video_compression.md)**

:   提出 RL-RC-DoT，一个基于强化学习的宏块级量化参数（QP）控制 agent，用于任务感知视频压缩。通过将 QP 选择建模为 RL 的顺序决策问题，agent 学习在给定码率约束下为任务相关区域分配更多码率，在车辆检测和 ROI 显著性编码两个任务上显著提升性能。关键优势在于推理时不需要运行下游任务模型，适合边缘设备部署。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](llm_agent/sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，一个基于视觉反馈的闭环 agentic 框架，通过为 VLM 设计一套功能完备的 Action API（13个原子操作覆盖物体增删、6DoF空间操作、相机控制），让 VLM 以 ReAct 范式迭代生成开放词汇的 3D 场景，在室内（偏好率61.25%）和开放域（偏好率65.00%）场景中均大幅优于 Holodeck 和 SceneWeaver。

**[Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)**

:   提出 Sketchtopia 大规模数据集（20K+ 游戏会话、263K 草图、916 名玩家）和三组件 Agent 框架（ActionDecider + DRAWBOT + GUESSBOT），在 Pictionary 场景下研究异步、目标驱动的多模态协作通信，引入 AAO/FRS/MATS 三个新评估指标。

**[SpiritSight Agent: Advanced GUI Agent with One Look](llm_agent/spiritsight_agent_advanced_gui_agent_with_one_look.md)**

:   提出 SpiritSight，一个基于视觉的端到端 GUI agent，通过 573 万样本的多层级数据集 GUI-Lasagne 和 Universal Block Parsing (UBP) 方法解决动态高分辨率输入的定位歧义，SpiritSight-8B 在 Multimodal-Mind2Web 上非候选元素设置下 Step SR 达 52.7%，全面超越所有视觉/语言/混合方法。

**[TANGO: Training-free Embodied AI Agents for Open-world Tasks](llm_agent/tango_training-free_embodied_ai_agents_for_open-world_tasks.md)**

:   提出 TANGO，通过 LLM 的程序组合能力编排两个最小化的导航基础原语（PointGoal Navigation + 记忆驱动探索策略），无需任何任务特定训练，仅用 few-shot 示例即可在 Open-Set ObjectGoal Navigation、Multi-Modal Lifelong Navigation 和 Open Embodied QA 三个不同的具身 AI 任务上达到 SOTA，体现了"最小原语集 + LLM 组合"的通用性。

**[V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](llm_agent/v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)**

:   提出 V-Stylist，一个基于 MLLM 多 agent 协作和反思的视频风格化系统，通过 Video Parser（视频分镜）、Style Parser（风格树搜索）和 Style Artist（多轮自反思渲染）三个角色协作，在复杂转场视频和开放风格描述上实现 SOTA，整体指标超越 FRESCO 6.05%。

**[Visual Agentic AI for Spatial Reasoning with a Dynamic API](llm_agent/visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)**

:   提出 VADAR，一种 agentic 程序合成方法用于 3D 空间推理。多个 LLM agent 协作生成 Pythonic API 并在求解过程中动态扩展新函数来解决常见子问题，克服了 VisProg/ViperGPT 等先前方法依赖静态人工定义 API 的局限。同时引入涉及多步空间定位和推理的新 benchmark，在 3D 理解任务上超越现有零样本方法。

---

## 🤖 机器人/具身智能 { #robotics }

**[3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](robotics/3d-mvp_3d_multiview_pretraining_for_manipulation.md)**

:   提出3D-MVP，将Masked Autoencoder预训练从2D扩展到3D多视角设定——在Objaverse的200K个3D物体上预训练RVT的多视角Transformer编码器，下游微调后在RLBench上平均成功率从62.9%提升到67.5%，在COLOSSEUM上显著提升对纹理、大小、光照等环境变化的鲁棒性。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](robotics/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   提出 Expert Pyramid Tuning (EPT)，将计算机视觉中的多尺度特征金字塔思想引入 LoRA-based MoE，通过共享元知识子空间 + 反卷积金字塔投影机制构建不同粒度的专家，实现更高效的多任务参数微调。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA，将 7-DoF 机器人动作解耦为平移/旋转/夹爪三类运动原语并与语言语义建立对应，通过软标签对比学习和自适应损失加权，以 1.3B 参数在 LIBERO 上达到 93.6% 平均成功率。

**[One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](robotics/one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

**[SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

**[SortScrews: A Dataset and Baseline for Real-time Screw Classification](robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)**

:   提出SortScrews数据集——一个包含560张512×512 RGB图像、覆盖6类螺丝的工业分类数据集，配套可复用的数据采集流水线，并以迁移学习的EfficientNet-B0和ResNet-18作为基线，ResNet-18在该数据集上达到96.4%验证准确率。

**[TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)**

:   在 ESP32 微控制器上部署端到端量化 CNN，仅用 23k 参数和 ToF 深度相机实现 30ms 延迟的实时自主导航。

---

## 🧑 人体理解 { #human_understanding }

**[3D Face Reconstruction From Radar Images](human_understanding/3d_face_reconstruction_from_radar_images.md)**

:   首次从毫米波雷达图像进行3D人脸重建：用物理雷达渲染器生成合成数据集训练CNN编码器估计BFM参数，再通过学习一个可微分雷达渲染器构建model-based autoencoder，在合成数据上实现2.56mm平均点距精度，并可在推理时无监督优化参数。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](human_understanding/breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   用预训练的Foundation Model（TabPFN）替代传统手工先验，实现零超参数调优的电路Yield Multi-Corner Analysis：冻结backbone做in-context learning，自动跨corner迁移知识，结合自动特征选择（1152D→48D），在SRAM benchmarks上达到SOTA精度（MRE低至0.11%）且验证成本降低10倍以上。

**[L2GTX: From Local to Global Time Series Explanations](human_understanding/l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的时间序列分类全局解释方法，通过聚合 LOMATCE 产生的参数化时间事件原语（PEPs）构建类级全局解释，在六个基准数据集上保持稳定的全局忠实度（R²）。

**[MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](human_understanding/mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co.md)**

:   MM-CondChain 是首个针对视觉基础深层组合推理的 MLLM 基准，通过可验证程序中间表示（VPIR）自动构建多层条件链和链式硬负样本，最强模型仅获 53.33 Path F1，揭示深层组合推理是根本挑战。

**[NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction](human_understanding/nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)**

:   NBAvatar 提出 Neural Billboard 原语——将可学习平面几何原语与神经纹理延迟渲染结合，实现手脸交互场景下的照片级真实头部 avatar 渲染，在百万像素分辨率下 LPIPS 比 Gaussian 方法降低 30%。

**[Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception](human_understanding/perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)**

:   提出一种面向人机协作的感知调度框架，基于信息增益和计算代价的权衡来选择性激活感知模块（目标检测/姿态估计），在流式感知场景下将计算延迟降低最多 27.52%，同时 MMPose 激活召回提升 72.73%。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](human_understanding/team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

---

## 🎬 视频理解 { #video_understanding }

**[Behaviorvlm Unified Finetuning-Free Behavioral Understanding With Vision-Languag](video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)**

:   提出 BehaviorVLM，一个统一的无需微调的视觉语言框架，通过多阶段结构化推理管线同时解决动物姿态估计和行为理解两大任务，仅需 3 帧人工标注即可实现可靠的关键点追踪，并通过深度嵌入聚类 + VLM 描述 + LLM 语义合并实现可解释的多动物行为分割。

**[Beyond Single-Sample Reliable Multi-Sample Distillation For Video Understanding](video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   提出 R-MSD（Reliable Multi-Sample Distillation），通过对每个输入采样多个教师响应并结合任务自适应质量匹配，解决视频 LVLM 黑盒蒸馏中单样本教师监督不可靠的问题，4B 学生模型在 VideoMME (+1.5%)、Video-MMMU (+3.2%)、MathVerse (+3.6%) 等基准上取得一致提升。

**[Fc-Track Overlap-Aware Post-Association Correction For Online Multi-Object Track](video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一个轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的外观特征过滤和重叠 tracklet 对内的相似度比较，在线纠正因目标重叠导致的检测-轨迹错误匹配，将长期身份切换比例从 36.86% 降至 29.55%，同时在 MOT17/MOT20 上保持 SOTA 性能。

**[Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](video_understanding/reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](video_understanding/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出面向卫星场景的自适应多模态语义传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）灵活切换、动态知识库更新机制和 LLM 决策代理，在极低带宽下实现高保真音视频同步重建。

**[VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](video_understanding/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)**

:   VCBench 将计数重新定位为诊断视频模型"时空状态维护"能力的最小探针，提出了覆盖物体计数（当前状态/身份追踪）和事件计数（瞬时事件/周期活动）的 8 种子类别，通过沿时间线的流式多点查询观察模型预测轨迹，在 406 个视频/4576 个查询点上评估主流模型，发现当前模型在时空状态维护上仍存在显著缺陷。

**[Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](video_understanding/video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)**

:   提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。

**[World2Act: Latent Action Post-Training via Skill-Compositional World Models](video_understanding/world2act_latent_action_post-training_via_skill-compositional_world_models.md)**

:   World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)**

:   > 基于摘要：Recent advances in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language tasks, yet they often struggle with vision-centric scenarios where precise visual focus is needed for accurate reasoning. In this paper, we introduce Argus to address these limita

**[CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](llm_reasoning/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   > 基于摘要：Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLA

**[Interleaved-Modal Chain-of-Thought](llm_reasoning/interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework](llm_reasoning/learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte.md)**

:   > 基于摘要：Polynomial Lyapunov function \mathcal V (x) provides mathematically rigorous that converts stability analysis into efficiently solvable optimization problem. Traditional numerical methods rely on user-defined templates, while emerging neural \mathcal V (x) offer flexibility but exhibit poor generali

**[Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval](llm_reasoning/reason-before-retrieve_one-stage_reflective_chain-of-thoughts_for_training-free_.md)**

:   > 基于摘要：Composed Image Retrieval (CIR) aims to retrieve target images that closely resemble a reference image while integrating user-specified textual modifications, thereby capturing user intent more accurately. Existing training-free zero-shot CIR (ZS-CIR) methods often employ a two-stage process: they fi

**[Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](llm_reasoning/style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   > 基于摘要：Recently, a task of Single-Domain Generalized Object Detection (Single-DGOD) is proposed, aiming to generalize a detector to multiple unknown domains never seen before during training. Due to the unavailability of target-domain data, some methods leverage the multimodal capabilities of vision-langua

**[VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](llm_reasoning/videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)**

:   > 基于摘要：The advancement of Large Vision Language Models (LVLMs) has significantly improved multimodal understanding, yet challenges remain in video reasoning tasks due to the scarcity of high-quality, large-scale datasets. Existing video question-answering (VideoQA) datasets often rely on costly manual anno

---

## 🎯 目标检测 { #object_detection }

**[Abra Teleporting Fine-Tuned Knowledge Across Domains For Open-Vocabulary Object ](object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA（Aligned Basis Relocation for Adaptation），通过在权重空间中进行 SVD 分解与正交旋转对齐，将源域的类别特定检测知识"传送"到无标注数据的目标域，实现零样本跨域目标检测。

**[Dreamvideo-Omni Omni-Motion Controlled Multi-Subject Video Customization With La](object_detection/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)**

:   提出 DreamVideo-Omni，通过渐进式两阶段训练范式（Omni-Motion SFT + Latent Identity Reward Feedback Learning），在统一的 DiT 框架中实现多主体定制与全运动控制（全局 bbox + 局部轨迹 + 相机运动）的协同生成。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](object_detection/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)**

:   提出 RAPTA（训练时基于目标检测的区域感知 prompt 变体增强）和 ADMCD（推理时三流注意力融合的多模态复制检测），从缓解和检测两个角度端到端地应对文生图扩散模型的训练数据记忆化问题。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](object_detection/mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)**

:   提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](object_detection/sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶作为刚体的物理先验，在 ViT 中间层提取尺度不变的梯度能量统计量作为跨模态几何锚点，并在终端层将特征解耦为模态不变共享特征和模态特定特征后通过加性残差融合，实现光学-SAR 船舶重识别 SOTA。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一种轻量级可见光-红外融合网络，通过 MEAF 模块（可学习空间掩码+空间注意力的像素级融合）和训练时结构表示增强（SR，推理时无开销的超分辅助监督），在 VEDAI 上达到 84.71% mAP 同时参数量仅 5.1M（减少 93.6%）。

---

## 📦 模型压缩 { #model_compression }

**[Alternating Gradient Flow Utility A Unified Metric For Structural Pruning And Dy](model_compression/alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)**

:   提出基于交替梯度流(AGF)的统一效用度量，将特征空间总变差作为结构化剪枝指标，并结合置信度级联路由实现离线拓扑构建与在线动态推理的解耦，在ImageNet-1K极端压缩下避免传统指标导致的结构崩溃，在ImageNet-100动态推理中以0.92x计算代价匹配全模型精度。

**[An Fpga Implementation Of Displacement Vector Search For Intra Pattern Copy In J](model_compression/an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移向量(DV)搜索模块的FPGA架构实现，采用四级流水线设计和优化的存储组织方式，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐量和277 mW功耗，为IPC实际硬件部署和ASIC转化奠定基础。

**[Arche Autoregressive Residual Compression With Hyperprior And Excitation](model_compression/arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出ARCHE端到端学习型图像压缩框架，在统一概率架构中整合分层Hyperprior、掩码空间自回归上下文、通道条件化和SE激励通道重校准，无需Transformer或循环组件，在Kodak上相对Ballé基线BD-Rate降低约48%，相对VVC Intra降低约5.6%，仅95M参数和222ms解码时间。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](model_compression/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。

---

## 🛡️ AI 安全 { #ai_safety }

**[Lyapunov Stable Graph Neural Flow](ai_safety/lyapunov_stable_graph_neural_flow.md)**

:   将 Lyapunov 稳定性理论（整数阶和分数阶）与图神经流集成，通过可学习 Lyapunov 函数和投影机制将 GNN 特征动态约束在稳定空间中，首次为图神经流提供可证明的对抗鲁棒性保证，且与对抗训练正交可叠加。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](ai_safety/neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[Rethinking VLMs for Image Forgery Detection and Localization](ai_safety/rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM，揭示 VLM 先验对伪造检测/定位几乎无益，通过将检测/定位与语言解释解耦的两阶段框架，用 ViT+SAM 专家模型做检测定位、再将定位 mask 作为辅助输入增强 VLM 训练以生成可解释文字说明。

**[STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](ai_safety/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)**

:   STRAP-ViT 提出一种无需训练的即插即用 ViT 防御模块，利用 Jensen-Shannon 散度将受对抗补丁影响的 token 从正常 token 中分离出来，再通过随机复合变换消除其对抗效应，在多种 ViT 架构和攻击方法下实现了接近干净基线 2-3% 的鲁棒精度。

---

## ✍️ 文本生成 { #nlp_generation }

**[ArtFormer: Controllable Generation of Diverse 3D Articulated Objects](nlp_generation/artformer_controllable_generation_of_diverse_3d_articulated_objects.md)**

:   > 基于摘要：This paper presents a novel framework for modeling and conditional generation of 3D articulated objects. Troubled by flexibility-quality tradeoffs, existing methods are often limited to using predefined structures or retrieving shapes from static datasets. To address these challenges, we parameteriz

**[Dense Match Summarization for Faster Two-view Estimation](nlp_generation/dense_match_summarization_for_faster_two-view_estimation.md)**

:   > 基于摘要：In this paper, we speed up robust two-view relative pose from dense correspondences. Previous work has shown that dense matchers can significantly improve both accuracy and robustness in the resulting pose. However, the large number of matches comes with a significantly increased runtime during robu

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](nlp_generation/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   > 基于摘要：Approximate nearest neighbor search (ANNS) is an essential building block for applications like RAG but can sometimes yield results that are overly similar to each other. In certain scenarios, search results should be similar to the query and yet diverse. We propose LotusFilter, a post-processing mo

**[The Scene Language: Representing Scenes with Programs, Words, and Embeddings](nlp_generation/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)**

:   提出 Scene Language——一种用程序（P, 编码层级结构）+ 词语（W, 语义类别）+ 嵌入（Z, 视觉身份）三元组 $\Phi(s)=(W,P,Z)$ 表示视觉场景的新范式，通过 Claude 3.5 Sonnet 的 training-free 推理从文本/图像输入生成场景表示，支持传统/神经/混合渲染，在 3D/4D 场景生成质量和可控编辑上超越场景图等现有表示。

---

## 🛰️ 遥感 { #remote_sensing }

**[Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](remote_sensing/hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)**

:   提出 UAV 场景变化描述（UAV-SCC）新任务及 HDC-CL 框架，通过动态自适应布局 Transformer 建模移动视角下的图像对重叠/非重叠区域，结合层级跨模态方向一致性校准增强视角偏移方向感知，并构建了专用基准数据集。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](remote_sensing/joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   研究多 LEO 卫星联合服务多天线地面用户的分布式 MIMO 下行通信，提出联合传输与流式传输两种模式：前者通过 WMMSE 迭代优化预编码器最大化和频谱效率，后者通过匈牙利算法的流-卫星关联减少前传开销，实现性能与前传负载的灵活权衡。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](remote_sensing/metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   提出 MetaSpectra+，一种基于超表面-折射混合光学的紧凑多功能相机，通过双层超表面独立控制各通道色散/曝光/偏振，在约 250nm 可见光带宽内实现快照式高光谱+HDR 或高光谱+偏振联合成像，重建精度在基准数据集上达到 SOTA。

**[Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing](remote_sensing/think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)**

:   构建遥感多实体推理定位基准 ME-RSRG（首个显式标注主体-客体角色的遥感定位数据集），提出 Entity-Aware Reasoning (EAR) 框架，结合 SFT 冷启动与实体感知奖励驱动的 GRPO 优化，实现结构化推理链输出和主-客体联合定位，Qwen2.5-VL 系列在 EAR 优化后 mAcc@0.5 提升超 10%。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Polishing The Sky Wide-Field And High-Dynamic Range Interferometric Image Recons](image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+/++，通过**分块训练+拼接推理**和**arcsinh 非线性变换**两项改进，使深度学习方法首次能处理宽视场（12,960×12,960 像素）、高动态范围（~10⁶）的射电干涉成像，并展示了超分辨率对强引力透镜发现的 10× 提升潜力。

**[OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras](image_restoration/towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)**

:   扩展 OptiFusion 自动设计 120 种多样化镜头，提出 ODE 综合评估指标和大规模 benchmark，系统对比 24 种算法，发现 CNN 模型在像差校正中提供最佳速度-精度权衡，反直觉地超越 Transformer。

**[Variational Garrote for Sparse Inverse Problems](image_restoration/variational_garrote_for_sparse_inverse_problems.md)**

:   系统比较 $\ell_1$ 正则化 (LASSO) 与 Variational Garrote (VG, 概率 $\ell_0$ 近似) 在信号重采样、去噪和稀疏视角 CT 重建三种逆问题上的表现，发现 VG 在强欠定情况下（采样率低/角度稀疏）通常获得更低的泛化误差，因为 spike-and-slab 先验与真实稀疏分布更匹配。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[Representation Learning for Spatiotemporal Physical Systems](self_supervised/representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。

---

## 📖 NLP 理解 { #nlp_understanding }

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](nlp_understanding/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](nlp_understanding/as_language_models_scale_low-order_linear_depth_dynamics_emerge_v2.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

---

## 🎵 音频/语音 { #audio_speech }

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](audio_speech/team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   本文提出面向视频级矛盾/犹豫（A/H）识别的多模态方法，整合场景（VideoMAE）、面部（EmotionEfficientNetB0）、音频（EmotionWav2Vec2.0+Mamba）和文本（EmotionDistilRoBERTa）四种模态，通过原型增强的 Transformer 融合模型实现 83.25% 平均 MF1，最终以五模型集成在测试集达到 71.43%。

---

## 📐 优化/理论 { #optimization }

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 提出了一种面向联邦学习的语义 coreset 选择框架，利用 VLM（MobileCLIP-S2）零样本提取三种标量指标（表示分数、多样性分数、边界接近度），通过服务器聚合全局共识后指导客户端进行两阶段剪枝（异常过滤+冗余消除），在 128-512× 上行带宽减少和 7.72× 加速的同时保持竞争精度。

---

## 🎮 强化学习 { #reinforcement_learning }

**[ThinkStream: Thinking in Streaming Video](reinforcement_learning/thinking_in_streaming_video.md)**

:   提出 ThinkStream，采用 Watch-Think-Speak 范式实现流式视频的实时连续推理，通过 RCSM（推理压缩流式记忆）将推理 trace 作为紧凑语义锚点替代旧视觉 token，配合 Streaming RLVR 训练策略，在保持低延迟/低内存的同时超越现有在线视频模型。

---

## 📈 时间序列 { #time_series }

**[Competition-Aware CPC Forecasting with Near-Market Coverage](time_series/competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   将付费搜索CPC预测重构为"部分可观测竞争下的预测"问题，通过语义邻域（Transformer嵌入）、行为邻域（DTW对齐）和地理意图三类竞争代理逼近不可观测的竞争状态，在1811个关键词×127周的Google Ads数据上显示竞争感知增强在中长期预测（6/12周）上显著优于单变量和弱上下文baseline。

---

## 📂 其他 { #others }

**[3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](others/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)**

:   提出跨任务少样本2D视线估计——利用预训练3D视线模型作为先验，通过**基于物理的可微投影模块**（6个可学习屏幕参数）将3D视线方向投影到2D屏幕坐标，仅需10张标注图像即可在未知设备上适配2D视线估计，在MPIIGaze/EVE/GazeCapture上比EFE和IVGaze提升超25%。

**[A2Z-10M Geometric Deep Learning With A-To-Z Brep Annotations For Ai-Assisted Cad](others/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含100万+复杂CAD模型、超1000万多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的A2Z数据集，是目前最大的CAD逆向工程数据集，并基于此训练了BRep边界和角点检测的基础模型。

**[Bendfm A Taxonomy And Synthetic Cad Dataset For Manufacturability Assessment In ](others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出一个面向板金弯曲工艺的可制造性度量分类法（按配置依赖性×可行性/复杂度两个维度划分为四象限），并构建首个包含20,000个零件（含可制造与不可制造样本）的合成数据集BenDFM，基准测试表明图结构表示（UV-Net）优于点云（PointNext），配置依赖性指标的预测更具挑战性。

**[Bounds On Agreement Between Subjective And Objective Measurements](others/bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   通过仅假设投票均值收敛于真实质量，推导出主观测试（MOS）与客观估计器之间PCC（上界）和MSE（下界）的数学界限，并提出基于二项分布的投票模型BinoVotes，使得即使在投票方差不可用时也能计算这些界限，18个主观测试数据的验证表明BinoVotes界限与全数据驱动界限高度吻合。

**[Deconstructing The Failure Of Ideal Noise Correction A Three-Pillar Diagnosis](others/deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   通过提供完美的oracle噪声转移矩阵T，证明Forward Correction在理想条件下仍会训练崩塌（先升后降最终与无校正基线收敛），从宏观（收敛终态）、微观（梯度动力学）、信息论（噪声信道不可逆信息损失）三个层面系统诊断了失败的根本原因——这不是T估计不准的问题，而是有限样本下高容量网络的结构性缺陷。

**[HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](others/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)**

:   HSEmotion 团队在 ABAW-10 竞赛中提出了一个轻量级 pipeline：用预训练 EfficientNet 提取面部 embedding，结合 MLP + GLA（Generalized Logit Adjustment）+ 滑窗平滑，在四项任务（EXPR/VA/AU/VD）上均大幅超过官方 baseline，其中暴力检测任务使用 ConvNeXt-T + TCN 达到 0.783 macro F1。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](others/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，在制药 BFS 高速产线上实现了仅用正常样本训练、单 patch 推理 0.17ms 的实时在线质检部署，通过 Perlin 噪声增强和 Noise Loss 优化重建质量。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](others/mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)**

:   MXNorm 提出复用 MXFP 量化过程中已计算的 block absmax 来近似 RMS，将归一化与 MX 量化融合为单次统计收集操作，实现 RMSNorm 的 drop-in 替换，在 Llama 3 8B 预训练中保持训练精度的同时获得最高 2.4× 的 kernel 加速。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](others/out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   StEvo-Bench 提出了一个评估视频世界模型"不可观测状态演化"能力的 benchmark——测试当物理过程不被观察时（相机移开/遮挡/关灯），世界模型能否继续正确推理状态变化，结果发现当前所有前沿模型（Veo 3、Sora 2 Pro 等）的任务成功率均低于 10%，揭示了"眼不见，心不在"的严重缺陷。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](others/rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   提出基于风洞PIV实验数据的学习框架，系统对比Kriging插值与三种深度学习模型（UNet、ViTAE、CWGAN）在稀疏传感器屋顶风场重建任务中的表现，并结合QR分解优化传感器布局。深度学习在混合风向训练下全面优于Kriging，SSIM提升最高33.5%。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](others/shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出SHREC算法，利用谱嵌入技术从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群Cn，实现从头(ab-initio)螺旋结构重建。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   构建包含24.2万工业零件的大规模多模态CAD数据集SldprtNet，提供3D模型、多视图图像、参数化文本脚本和自然语言描述的完整对齐，支持语义驱动的CAD建模任务。

**[Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](others/wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)**

:   针对柔性磨料翻页轮的复杂磨损模式，提出三级层次化深度学习分类框架，将磨损评估分解为使用状态检测、磨损类型识别和严重程度评估三个子任务，使用EfficientNetV2迁移学习实现93.8%–99.3%的分类精度。

**[Why Does It Look There? Structured Explanations for Image Classification](others/why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   提出I2X框架，通过追踪训练过程中模型置信度与原型强度的协同演化，将非结构化的显著性图解释转化为结构化的解释，回答模型"为什么看那里"的问题，并可指导微调提升性能。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，将零阶优化策略性地整合到 SAM 的扰动步骤中，仅需一次反向传播即可获得 SAM 的平坦最小值优势，在稀疏训练场景下将计算开销减半的同时提升精度和鲁棒性。

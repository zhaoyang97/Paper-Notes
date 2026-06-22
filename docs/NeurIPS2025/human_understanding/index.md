---
title: >-
  NeurIPS2025 人体理解论文汇总 · 21篇论文解读
description: >-
  21篇NeurIPS2025的人体理解方向论文解读，涵盖人脸/视线、人体姿态、推理、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
  - "人脸/视线"
  - "人体姿态"
  - "推理"
  - "语音"
item_list:
  - u: "a_generalized_label_shift_perspective_for_crossdomain_gaze_e/"
    t: "A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation"
  - u: "bedlam20_synthetic_humans_and_cameras_in_motion/"
    t: "BEDLAM2.0: Synthetic Humans and Cameras in Motion"
  - u: "conceptscope_characterizing_dataset_bias_via_disentangled_visual_concepts/"
    t: "ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts"
  - u: "cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si/"
    t: "CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals"
  - u: "cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten/"
    t: "Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization"
  - u: "devfd_developmental_face_forgery_detection_by_learning_shared_and_orthogonal_lor/"
    t: "DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces"
  - u: "foundation_cures_personalization_improving_personalized_models_prompt_consistenc/"
    t: "Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge"
  - u: "hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion/"
    t: "HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion"
  - u: "k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso/"
    t: "K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning"
  - u: "kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_/"
    t: "KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills"
  - u: "learning_dense_hand_contact_estimation_from_imbalanced_data/"
    t: "Learning Dense Hand Contact Estimation from Imbalanced Data"
  - u: "mospa_human_motion_generation_driven_by_spatial_audio/"
    t: "MOSPA: Human Motion Generation Driven by Spatial Audio"
  - u: "omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild/"
    t: "OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild"
  - u: "pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio/"
    t: "PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space"
  - u: "part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti/"
    t: "Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection"
  - u: "raptr_radar-based_3d_pose_estimation_using_transformer/"
    t: "RAPTR: Radar-Based 3D Pose Estimation Using Transformer"
  - u: "some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai/"
    t: "Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness"
  - u: "switchable_token-specific_codebook_quantization_for_face_image_compression/"
    t: "Switchable Token-Specific Codebook Quantization for Face Image Compression"
  - u: "uncle_towards_scalable_dynamic_causal_discovery_in_non-linear_temporal_systems/"
    t: "UnCLe: Towards Scalable Dynamic Causal Discovery in Non-Linear Temporal Systems"
  - u: "vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image/"
    t: "VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image"
  - u: "vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language/"
    t: "VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🧠 NeurIPS2025** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (151)](../../CVPR2026/human_understanding/index.md) · [🔬 ICLR2026 (45)](../../ICLR2026/human_understanding/index.md) · [🧪 ICML2026 (5)](../../ICML2026/human_understanding/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/human_understanding/index.md) · [📹 ICCV2025 (41)](../../ICCV2025/human_understanding/index.md) · [🧪 ICML2025 (3)](../../ICML2025/human_understanding/index.md)

🔥 **高频主题：** 人脸/视线 ×4 · 人体姿态 ×3 · 推理 ×2 · 语音 ×2

**[A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)**

:   本文将跨域视线估计(CDGE)问题建模为广义标签偏移(GLS)问题，指出现有域不变表示学习方法在标签偏移存在时理论上不充分，提出基于截断高斯分布的连续重要性重加权和概率感知条件算子差异(PCOD)来联合纠正标签偏移和条件偏移，在多个backbone上平均降低误差12%~27%。

**[BEDLAM2.0: Synthetic Humans and Cameras in Motion](bedlam20_synthetic_humans_and_cameras_in_motion.md)**

:   BEDLAM2.0 在 BEDLAM 基础上全面升级——引入多样化相机运动（合成平移/追踪/环绕 + 手持/头戴设备捕捉）、更广体型覆盖（BMI 18-41）、strand-based 发型、鞋子、分级服装和更多3D环境，构建 27K+ 序列 / 8M+ 帧的合成数据集，仅用合成数据训练即可在世界坐标系人体运动估计上超越 SOTA。

**[ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts](conceptscope_characterizing_dataset_bias_via_disentangled_visual_concepts.md)**

:   提出 ConceptScope 框架，利用在视觉基础模型表征上训练的稀疏自编码器（SAE）自动发现和量化数据集中的视觉概念偏差，无需人工标注即可将概念分类为 target / context / bias 三类。

**[CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals](cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)**

:   提出 CPEP 框架，通过对比学习将低质量 EMG 信号表征与高质量手部姿态表征对齐，使 EMG 编码器获得姿态感知能力，首次实现从 EMG 信号零样本识别未见手势，分布内手势分类提升 21%、未见手势分类提升 72%。

**[Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)**

:   提出 Cycle-Sync 全局相机位姿估计框架，通过将消息传递最小二乘 (MPLS) 扩展到相机位置估计、引入 Welsch 型鲁棒损失和环一致性加权，在无需 bundle adjustment 的情况下超越了包括完整 SfM pipeline（含 BA）在内的所有基线方法。

**[DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces](devfd_developmental_face_forgery_detection_by_learning_shared_and_orthogonal_lor.md)**

:   提出 DevFD——一种发展式 MoE 架构，用共享 Real-LoRA 建模真实人脸共性、正交 Fake-LoRA 序列逐步建模新伪造类型，并通过将正交梯度集成到正交损失中缓解灾难性遗忘，在持续学习人脸伪造检测中达到最高准确率和最低遗忘率。

**[Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge](foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)**

:   FreeCure发现面部个性化模型的身份嵌入会覆盖但不破坏基础模型的prompt控制能力，据此提出无训练框架，通过Foundation-Aware Self-Attention（FASA）将基础模型的属性信息注入个性化生成过程，在保持身份保真度的同时大幅提升prompt一致性，可无缝集成到SD/SDXL/FLUX等主流模型。

**[HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)**

:   将人体-物体交互（HOI）生成建模为 Driver-Responder 系统，通过轻量级 Transformer 交互动力学模型显式预测物体对人体动作的响应，利用残差动力学损失在训练时增强因果一致性，同时保持推理效率。

**[K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)**

:   提出 K-DeCore 框架，通过知识解耦将结构化知识推理分为任务无关的 schema 过滤和任务特定的 query 构建两阶段，配合双视角记忆构建和结构引导的伪数据合成策略，在固定参数量下实现跨异构 SKR 任务的有效知识迁移。

**[KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)**

:   提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。

**[Learning Dense Hand Contact Estimation from Imbalanced Data](learning_dense_hand_contact_estimation_from_imbalanced_data.md)**

:   提出 HACO 框架，通过平衡接触采样（BCS）解决类别不平衡和顶点级类别平衡损失（VCB Loss）解决空间不平衡，首次在 14 个数据集（65.5 万图像）上训练稠密手部接触估计模型，在多种交互场景下达到 SOTA。

**[MOSPA: Human Motion Generation Driven by Spatial Audio](mospa_human_motion_generation_driven_by_spatial_audio.md)**

:   首次定义"空间音频驱动人体运动生成"这一新任务，构建包含 9+ 小时、27 种场景、12 名受试者的双耳音频-运动配对 SAM 数据集，提出 MOSPA 扩散模型在融合 MFCC/tempogram/RMS 等音频特征与声源位置及运动风格条件后，以 FID 7.98 大幅领先 EDGE（14.0）、POPDG（21.0）等音乐/舞蹈基线。

**[OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)**

:   提出OmniGaze，一个半监督3D注视估计框架，利用融合视觉嵌入、MLLM生成的语义注视描述和几何方向向量的奖励模型来评估伪标签质量，在140万无标签人脸数据上训练，在5个数据集的域内/跨域设置下达到SOTA，并在4个未见数据集上展示零样本泛化能力。

**[PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)**

:   提出 PandaPose，通过将 2D 姿态先验传播到 3D 锚点空间作为统一中间表示，结合自适应关节级 3D 锚点设置和关节级深度分布估计，实现对遮挡和 2D 姿态误差鲁棒的单帧 3D 人体姿态提升。

**[Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)**

:   提出一种部位感知的自底向上群组推理框架，通过姿态引导的身体部位特征增强和基于相似度的个体关联来推断社交群组和细粒度交互，在 NVI 和 Café 数据集上达到新 SOTA。

**[RAPTR: Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)**

:   提出RAPTR，首个利用弱监督（3D BBox + 2D关键点标签）进行雷达3D人体姿态估计的Transformer框架，通过伪3D可变形注意力和结构化损失函数在两个室内数据集上大幅超过基线。

**[Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)**

:   本文首次系统研究了优化算法选择对深度学习群体公平性的影响，通过随机微分方程（SDE）分析和两个新定理证明，自适应优化器（RMSProp/Adam）比SGD更容易收敛到公平的极小值点，特别是在数据严重不平衡时。

**[Switchable Token-Specific Codebook Quantization for Face Image Compression](switchable_token-specific_codebook_quantization_for_face_image_compression.md)**

:   提出可切换的token专属码本量化机制（STSCQ），通过图像级码本路由和token级码本分割的层次动态结构，在超低比特率下显著提升人脸图像的压缩重建质量和识别精度。

**[UnCLe: Towards Scalable Dynamic Causal Discovery in Non-Linear Temporal Systems](uncle_towards_scalable_dynamic_causal_discovery_in_non-linear_temporal_systems.md)**

:   提出 UnCLe，一种基于 TCN 自编码器解耦和自回归依赖矩阵的可扩展动态因果发现方法，通过时序扰动后逐数据点预测误差增量推断时变因果关系，在静态和动态因果发现基准上均达到 SOTA。

**[VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)**

:   提出VASA-3D，通过将VASA-1的2D运动隐空间适配到3D高斯溅射表征，并利用VASA-1合成训练数据进行单图定制优化，实现了从单张肖像照到逼真音频驱动3D头部化身的实时生成（512×512, 75fps）。

**[VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)**

:   提出 VimoRAG 框架，利用大规模野外视频数据库作为2D运动先验来增强3D运动生成，通过 Gemini-MVR 检索器和 McDPO 训练策略解决人体动作视频检索和错误传播两大瓶颈。

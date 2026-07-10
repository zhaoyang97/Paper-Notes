---
title: >-
  ECCV2026 人体理解论文汇总 · 18篇论文解读
description: >-
  18篇ECCV2026的人体理解方向论文解读，涵盖人脸/视线、翻译、推理、扩散模型、多模态、虚拟人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
  - "人脸/视线"
  - "翻译"
  - "推理"
  - "扩散模型"
  - "多模态"
  - "虚拟人"
item_list:
  - u: "backtranslation20_--_a_linguistically_motivated_metric_to_assess_sign_language_p/"
    t: "BackTranslation2.0 -- A Linguistically Motivated Metric to Assess Sign Language Production"
  - u: "dress-ed_instruction-guided_editing_for_virtual_try-on_and_try-off/"
    t: "Dress-ED: Instruction-Guided Editing for Virtual Try-On and Try-Off"
  - u: "facemoe_mixture_of_experts_for_low-resolution_face_recognition/"
    t: "FaceMoE: Mixture of Experts for Low-Resolution Face Recognition"
  - u: "flowerdance_meanflow_for_efficient_and_refined_3d_dance_generation/"
    t: "FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation"
  - u: "interedit_navigating_text-guided_3d_dyadic_human_motion_editing/"
    t: "InterEdit: Navigating Text-Guided 3D Dyadic Human Motion Editing"
  - u: "multi-scale_object-aware_gaze_estimation_via_geometric_reasoning/"
    t: "Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning"
  - u: "odoriko_a_shape-aware_multimodal_diffusion_framework_for_human_motion/"
    t: "Odoriko: A Shape-Aware Multimodal Diffusion Framework for Human Motion"
  - u: "piavatar_physically_interactive_avatars_via_deformation_gradient_decoupling/"
    t: "PIAvatar: Physically Interactive Avatars via Deformation Gradient Decoupling"
  - u: "reweighting_framewise_attention_in_video_transformers_for_facial_expression_unde/"
    t: "Reweighting Framewise Attention in Video Transformers for Facial Expression Understanding"
  - u: "self-supervised_garment_dynamics_with_persistent_wrinkles/"
    t: "Self-supervised Garment Dynamics with Persistent Wrinkles"
  - u: "sicage_speaker-independent_culture-aware_gesture_generation_using_ted4c-l_datase/"
    t: "SICAGE: Speaker-Independent Culture-Aware Gesture Generation using TED4C-L Dataset"
  - u: "signer_temporally_grounded_sign_language_generation_via_time-resolved_conditioni/"
    t: "SIGNER: Temporally Grounded Sign Language Generation via Time-Resolved Conditioning"
  - u: "signet_motion-level_knowledge_transfer_for_cross-language_sign_language_translat/"
    t: "SIGNET: Motion-Level Knowledge Transfer for Cross-Language Sign Language Translation"
  - u: "signnet-1m_large-scale_multilingual_sign_language_video_dataset_with_downstream_/"
    t: "SignNet-1M: Large-Scale Multilingual Sign Language Video Dataset with Downstream Benchmarks"
  - u: "synccache_exploiting_asymmetric_dynamics_for_fast_audio-driven_portrait_animatio/"
    t: "SyncCache: Exploiting Asymmetric Dynamics for Fast Audio-Driven Portrait Animation"
  - u: "text_dictates_music_decorates_energy_based_attention_for_editable_dance_motion_generation/"
    t: "Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation"
  - u: "unimotion_a_unified_framework_for_motion-text-vision_understanding_and_generatio/"
    t: "UniMotion: A Unified Framework for Motion-Text-Vision Understanding and Generation"
  - u: "unsupervised_multi-person_tracking_via_point_visual_prompting/"
    t: "PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🎞️ ECCV2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (151)](../../CVPR2026/human_understanding/index.md) · [🔬 ICLR2026 (45)](../../ICLR2026/human_understanding/index.md) · [🧪 ICML2026 (5)](../../ICML2026/human_understanding/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/human_understanding/index.md) · [🧠 NeurIPS2025 (21)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (41)](../../ICCV2025/human_understanding/index.md)

🔥 **高频主题：** 人脸/视线 ×3 · 翻译 ×2

**[BackTranslation2.0 -- A Linguistically Motivated Metric to Assess Sign Language Production](backtranslation20_--_a_linguistically_motivated_metric_to_assess_sign_language_p.md)**

:   提出 BackTranslation2.0，一个语言学驱动的文本到手语翻译评估指标，通过两阶段智能体框架（10 个专用工具提取证据 + 4 个 LLM 交叉比对模块）在语法正确性、音系准确性、运动流畅度和生成保真度四个维度上给出确定性评分，在已知损坏和合成数据上与人类判断达到 Pearson r=1.00 / Spearman ρ=1.00，远超现有反向翻译和运动相似度基线。

**[Dress-ED: Instruction-Guided Editing for Virtual Try-On and Try-Off](dress-ed_instruction-guided_editing_for_virtual_try-on_and_try-off.md)**

:   Dress-ED 构建了首个统一 VTON（虚拟试穿）、VTOFF（虚拟试脱）和文本指令驱动服装编辑的大规模基准数据集（146k 已验证四元组），并提出一套基于 MLLM + 双路径连接器 + DiT 的统一多模态扩散基线模型 Dress-EM，在指令驱动的服装编辑任务上全面超越通用编辑模型和领域专用 VTON 模型。

**[FaceMoE: Mixture of Experts for Low-Resolution Face Recognition](facemoe_mixture_of_experts_for_low-resolution_face_recognition.md)**

:   FaceMoE 将 Transformer 中单一 FFN 替换为多个稀疏激活的 MoE 专家加 Top-k 路由器，使不同专家自动特化于人脸不同语义区域（高频纹理/低频平滑/关键点），实现分辨率感知的特征提取；在 BRIAR、IJB-S、TinyFace 三个低分辨率基准上全面超越 SOTA，同时几乎不损失 HR 预训练性能。

**[FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation](flowerdance_meanflow_for_efficient_and_refined_3d_dance_generation.md)**

:   FlowerDance 将 MeanFlow（用区间平均速度替代瞬时速度预测的流匹配变体）与物理一致性约束、双向 Mamba 骨干和通道级跨模态融合相结合，仅需 5-20 步采样即可生成高质量的 3D 舞蹈，在 FineDance 和 AIST++ 上达到 SOTA 的同时推理速度（2008 FPS）远超此前最佳方法（MatchDance 345 FPS），为实时 3D 渲染留出充足算力空间。

**[InterEdit: Navigating Text-Guided 3D Dyadic Human Motion Editing](interedit_navigating_text-guided_3d_dyadic_human_motion_editing.md)**

:   InterEdit 提出了文本引导双人3D运动编辑（TMME）这一新任务，构建了首个大规模双人运动编辑数据集 InterEdit3D（5161组源-目标-文本三元组），并设计了基于条件扩散模型的 InterEdit 方法——通过语义感知计划令牌对齐捕获高层编辑意图、交互感知频率令牌对齐利用DCT频段能量约束交互节奏——在编辑忠实度和运动真实性上显著超越四种基线方法。

**[Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning](multi-scale_object-aware_gaze_estimation_via_geometric_reasoning.md)**

:   将视线目标估计从像素级回归重新建模为层次化推理问题，先用目标级语义表征建立候选注意对象，再用视线方向构造视场锥几何先验约束空间，最后通过多尺度残差融合精确定位，在 GazeFollow/VideoAttentionTarget/ChildPlay/GOO-Real 上以 7.1M 参数取得 SOTA。

**[Odoriko: A Shape-Aware Multimodal Diffusion Framework for Human Motion](odoriko_a_shape-aware_multimodal_diffusion_framework_for_human_motion.md)**

:   Odoriko 提出首个统一的多模态人体运动生成框架，通过将性别和 SMPL 体形参数作为显式条件信号分层注入扩散骨干网络，使生成的运动能够反映主体的生物形态学特征，同时在文本到运动、音乐到舞蹈、视频到运动估计三个任务上以极少的参数量达到甚至超越当前专用方法的性能。

**[PIAvatar: Physically Interactive Avatars via Deformation Gradient Decoupling](piavatar_physically_interactive_avatars_via_deformation_gradient_decoupling.md)**

:   PIAvatar 提出基于 MPM 的物理可交互 3D 人体化身框架，通过将用户定义的 kinematic velocity 从变形梯度更新中显式解耦，消除运动驱动时产生的非预期内部应力，并嵌入骨骼结构以闭式优化实现形变后姿态的实时追踪，首次在统一 MPM 仿真框架内同时支持人-人、人-物的双向物理交互与非刚性表面形变。

**[Reweighting Framewise Attention in Video Transformers for Facial Expression Understanding](reweighting_framewise_attention_in_video_transformers_for_facial_expression_unde.md)**

:   MiRA 提出一种零额外参数的即插即用帧边缘注意力重分配模块，通过帧级置信度和帧内集中度两种互补统计量重新分配 ViT 自注意力，使模型在不依赖人脸对齐的前提下更精准地聚焦细微面部动态变化，在 DFEW、MAFW、FERV39k 等基准上取得 video-only 方法的 SOTA，其 FlashLite 近似还可兼容 FlashAttention 实现在线高效重分配。

**[Self-supervised Garment Dynamics with Persistent Wrinkles](self-supervised_garment_dynamics_with_persistent_wrinkles.md)**

:   提出首个自监督神经网络服装仿真器，通过动态静止弯曲能量和物理启发的课程学习，显式建模织物的弹塑性变形，首次在自监督框架下生成自然的持久褶皱，在多种服装、体形和动作上均优于现有方法。

**[SICAGE: Speaker-Independent Culture-Aware Gesture Generation using TED4C-L Dataset](sicage_speaker-independent_culture-aware_gesture_generation_using_ted4c-l_datase.md)**

:   SICAGE 将文化感知的共语手势生成建模为领域泛化问题：把每位说话人视为一个域，用 Fishr 正则化或对抗学习从音视频和文本中提取说话人无关的文化表征，再以此条件一个实时扩散生成器 ALaDiT，在自建的 106 小时四文化 TED4C-L 数据集上显著提升手势的真实性、多样性、节拍同步和文化一致性。

**[SIGNER: Temporally Grounded Sign Language Generation via Time-Resolved Conditioning](signer_temporally_grounded_sign_language_generation_via_time-resolved_conditioni.md)**

:   SIGNER 提出时序化解码条件（time-resolved conditioning）框架，通过构造时序化 Gloss 条件序列并在扩散去噪中以局部时序融合（LTF）将其注入，显式保留下游手语片段的时序对应关系（temporal grounding），从而解决现有手语生成方法中词序错乱和语义不准的核心问题，在 CSL-Daily 和 Phoenix-2014T 上大幅超越此前 SOTA。

**[SIGNET: Motion-Level Knowledge Transfer for Cross-Language Sign Language Translation](signet_motion-level_knowledge_transfer_for_cross-language_sign_language_translat.md)**

:   SIGNET 提出将多个在大型语料上预训练的手语骨架骨干网络视为冻结核运动视觉专家，通过手部先验驱动的注意力聚合与可学习门控融合实现跨语言运动级知识迁移，仅用 150 万可训练参数即在四个翻译基准和 WLASL 识别基准上取得 SOTA 或可比结果。

**[SignNet-1M: Large-Scale Multilingual Sign Language Video Dataset with Downstream Benchmarks](signnet-1m_large-scale_multilingual_sign_language_video_dataset_with_downstream_.md)**

:   SignNet-1M 利用 3DGS 新视角渲染、扩散模型场景/身份编辑和后渲染增强，将 7 个公开手语语料库扩充为约 100 万条多语言手语视频片段（ASL/CSL/DGS），并配套一套统一评估协议（Orig/Zero-shot/Trained）揭示并弥补现有模型在视角、背景、身份变化下的鲁棒性漏洞。

**[SyncCache: Exploiting Asymmetric Dynamics for Fast Audio-Driven Portrait Animation](synccache_exploiting_asymmetric_dynamics_for_fast_audio-driven_portrait_animatio.md)**

:   SyncCache 针对 DiT 架构的音频驱动肖像动画提出一种训练无关的特征缓存加速方案，通过空间非对称探测（SAP）在人脸区域优先触发重新计算、模态解耦缓存（MDC）跳过沉重视觉模块却逐时间步刷新轻量音频模块，以及记忆自适应的离线 DP 缓存规划，在 HunyuanVideo-Avatar 和 Wan-S2V 上分别实现 4.12x 和 3.75x 加速且视觉质量和唇音同步几乎无损。

**[Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation](text_dictates_music_decorates_energy_based_attention_for_editable_dance_motion_generation.md)**

:   STREAM 提出双模态解耦的能量注意力机制 BEAM，让文本控制舞蹈动作的语义结构（"做什么动作"）、音乐仅装饰其时间节奏（"何时做"），通过数学上保证的层次化无分类器引导解决文本-音乐联合生成中的模态坍塌问题，并在新标注的 Motorica++ 数据集上达到 SOTA。

**[UniMotion: A Unified Framework for Motion-Text-Vision Understanding and Generation](unimotion_a_unified_framework_for_motion-text-vision_understanding_and_generatio.md)**

:   UniMotion 把人体运动当作与 RGB 平权的**连续**模态塞进一个共享 LLM，用连续运动 VAE（CMA-VAE）+ 双路 embedder 替代离散 token，配合双后验 KL 对齐（DPA）注入视觉语义先验、潜空间重建对齐（LRA）解决运动通路冷启动，在 Motion-Text-RGB 三模态的理解、生成、编辑七个任务上做到统一并全面 SOTA。

**[PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking](unsupervised_multi-person_tracking_via_point_visual_prompting.md)**

:   PS-Track 提出数据-模型-损失三层递进的点监督多目标追踪框架，通过时间反馈提示、点激发小波注意力和不确定性高斯学习，仅用逐帧单像素标注即接近全监督方法的追踪性能，并在密集/极端运动场景下反超全监督基线。

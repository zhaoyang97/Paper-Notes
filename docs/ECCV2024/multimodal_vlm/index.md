<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**🎞️ ECCV2024** · 共 **82** 篇

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)**

:   AdaShield通过在MLLM输入前添加防御提示(defense prompt)来防御结构化越狱攻击（图像中嵌入有害文本），提出静态手动提示和自适应自动精化框架两种方案，无需微调模型即可显著提升安全性且不损害正常能力。

**[AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_visionlanguage_models_for_citywide_im.md)**

:   AddressCLIP 定义了"图像地址定位"(IAL) 新任务，提出端到端框架通过图像-文本对齐（图像↔地址/场景描述的对比学习）和图像-地理匹配（流形学习约束特征空间距离与地理距离一致）直接预测图像拍摄的可读文本地址，在自建的 Pittsburgh 和 San Francisco 数据集上优于现有 VLM 迁移方法。

**[AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation](anycontrol_create_your_artwork_with_versatile_control_on_tex.md)**

:   AnyControl提出Multi-Control Encoder，通过交替执行多控制融合块和多控制对齐块，从任意组合的多种空间控制信号中提取统一的多模态embedding，实现高质量、语义对齐的多条件可控图像生成。

**[ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)**

:   本文提出将视觉属性识别问题重新建模为基于图像条件的前缀语言模型（PrefixLM）下的句子生成概率问题，通过"生成式检索"（Generative Retrieval）替代传统的"对比式检索"（Contrastive Retrieval），显式建模物体-属性间的条件依赖关系，在VAW和新提出的VGARank数据集上显著超越对比检索方法。

**[ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](artvlm_attribute_recognition_through_visionbased_prefix_lang.md)**

:   ArtVLM将属性识别定义为语言建模问题，PrefixLM生成式检索灵活建模物体-属性条件依赖。

**[Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_vision-language_models.md)**

:   本文提出Attention Prompting on Image（API），通过辅助模型（如CLIP或LLaVA）根据文本查询生成注意力热力图，将热力图叠加到原始图像上作为视觉提示输入LVLM，在不修改模型参数的情况下在MM-Vet、LLaVA-Bench等多个VL基准上稳定提升多种LVLM的性能（LLaVA-1.5提升3.8%/2.9%）。

**[Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)**

:   API用辅助VLM根据文本查询生成注意力热力图叠加原图，引导LVLM关注相关区域。

**[Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)**

:   Be Yourself深入分析了扩散模型中Cross-Attention和Self-Attention导致的多主体语义泄漏问题，提出Bounded Attention机制，通过在去噪过程中限制不同主体间的信息流动来生成语义独立的多主体图像，免训练即可生成5+个语义相似主体。

**[BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](beaf_observing_before-after_changes_to_evaluate_hallucination_in_vision-language.md)**

:   提出 BEAF 幻觉评估基准，通过图像编辑（移除物体）构造"前后对比"场景，设计 TU/IG/SB/ID 四个变化感知指标，揭示现有 VLM 即使传统 accuracy 高也可能存在严重幻觉。

**[BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](beaf_observing_beforeafter_changes_to_evaluate_hallucination.md)**

:   BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。

**[BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)**

:   提出 BI-MDRG 框架，通过桥接图像历史信息来增强多模态对话中文本回复的图像 grounding 能力和连续图像回复中物体的一致性。

**[BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](bimdrg_bridging_image_history_in_multimodal_dialogue_respons.md)**

:   BI-MDRG通过视觉交叉注意力和Citation Module桥接图像历史增强多模态对话。

**[BLINK: Multimodal Large Language Models Can See but Not Perceive](blink_multimodal_large_language_models_can_see_but_not_perceive.md)**

:   提出BLINK——一个包含14个经典计算机视觉感知任务的多模态评测基准（3807道选择题），这些任务人类可以"眨眼间"解决（95.7%准确率），但最强的GPT-4V仅达51.26%（仅高于随机猜测13.17%），揭示了当前MLLM在核心视觉感知能力上的严重缺失。

**[BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)**

:   本文系统性地分析了不同视觉编码器（CLIP、DINOv2、EVA-CLIP等）对VLM性能的影响，发现没有单一编码器能在所有任务上最优，基于此提出BRAVE方法，通过轻量级的MEQ-Former将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在captioning和VQA任务上取得SOTA，并显著降低视觉幻觉。

**[BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_visionlanguage_model.md)**

:   通过系统benchmarking发现没有单一视觉编码器在所有VLM任务上最优，提出BRAVE方法用Multi-Encoder Querying Transformer（MEQ-Former）将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在多个captioning和VQA基准上达到SOTA。

**[CLAP: Isolating Content from Style Through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learni.md)**

:   从因果生成模型视角出发，提出CLAP（Contrastive Learning with Augmented Prompts），通过文本增强（而非图像增强）在预训练CLIP的特征空间中解耦内容与风格信息，以极低训练成本（<1小时）显著提升CLIP在零样本/少样本分类和对抗鲁棒性上的表现。

**[CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)**

:   从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。

**[Dataset Growth (InfoGrowth)](dataset_growth.md)**

:   提出 InfoGrowth，一种高效的在线数据清洗与选择算法，通过近邻搜索估计每个样本的信息增益，实现数据集的持续增长，同时保证清洁度和多样性，在 CC3M 上仅用 1/6 数据即超过全量训练效果。

**[DeCUR: Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_.md)**

:   DeCUR将嵌入维度分为跨模态公共和模态独特维度进行多模态自监督学习。

**[Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_self-supervised_lear.md)**

:   提出 DeCUR，在多模态自监督学习中将嵌入维度显式拆分为跨模态共有 (common) 和模态独有 (unique) 两部分，通过互相关矩阵分别驱动对齐与去相关，同时引入模态内训练保证独有维度学到有意义信息，在 SAR-光学、RGB-DEM、RGB-Depth 三类多模态场景上均优于 Barlow Twins / CLIP 等基线。

**[Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](difftracker_texttoimage_diffusion_models_are_unsupervised_tr.md)**

:   Diff-Tracker利用预训练T2I扩散模型知识进行无监督跟踪，学习prompt在cross-attention上激活目标区域。

**[DreamDrone: Text-to-Image Diffusion Models Are Zero-Shot Perpetual View Generators](dreamdrone_texttoimage_diffusion_models_are_zeroshot_perpetu.md)**

:   DreamDrone提出零样本、免训练的无限场景飞越生成pipeline，核心创新是在扩散模型的latent空间进行视角变换（而非像素空间），并通过特征对应引导和高通滤波策略保证帧间的几何一致性和高频细节一致性。

**[DreamView: Injecting View-Specific Text Guidance Into Text-to-3D Generation](dreamview_injecting_viewspecific_text_guidance_into_textto3d.md)**

:   DreamView通过自适应引导注入模块协调全局和视角特定文本实现3D定制化生成。

**[SpLIP: 通过多模态提示学习提升所有零样本草图检索任务](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)**

:   提出 SpLIP，一种基于冻结 CLIP 的双向多模态提示学习框架，通过视觉-文本编码器间的双向知识交换、自适应 margin 的三元组损失和条件跨模态拼图任务，在 ZS-SBIR、GZS-SBIR 和 FG-ZS-SBIR 三种草图检索设定下均取得 SOTA。

**[SpLIP: Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zeroshot_sketchbased_image_retrieval_through_m.md)**

:   SpLIP提出双向prompt共享用于零样本sketch检索，配合自适应margin和跨模态拼图任务。

**[Elysium: Exploring Object-Level Perception in Videos via MLLM](elysium_exploring_objectlevel_perception_in_videos_via_mllm.md)**

:   提出Elysium，首个端到端可训练的多模态大语言模型系统化处理视频目标级任务（如目标跟踪），构建了百万级ElysiumTrack-1M视频数据集支持SOT/RSOT/Video-REG三类任务，并设计T-Selector token压缩网络在保持性能的同时大幅减少视觉token消耗。

**[Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](exploring_pretrained_texttovideo_diffusion_models_for_referr.md)**

:   VD-IT首次探索预训练T2V扩散模型（ModelScopeT2V）在视频理解任务中的应用，通过Text-Guided Image Projection和Video-specific Noise Prediction设计，从固定T2V模型中提取语义对齐、时序一致的视频特征，在Referring VOS任务上超越传统判别式backbone。

**[Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)**

:   提出ECSO（Eyes Closed, Safety On），一种无需训练的MLLM保护方法，通过检测自身响应的安全性，并将不安全查询中的图像自适应转换为文本描述，从而恢复预对齐LLM的内在安全机制，在MM-SafetyBench上实现最高71.3%的安全性提升，且不损害常规性能。

**[Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_imageto.md)**

:   发现MLLM虽易受图像输入的越狱攻击但具备内省能力（能检测自身不安全回复）、且去除图像后安全机制恢复，据此提出ECSO——通过自检不安全回复后将图像转为query-aware文本描述来恢复预对齐LLM的固有安全机制，无需额外训练即可大幅提升安全性。

**[FlexAttention: 面向高效高分辨率视觉语言模型的灵活注意力机制](flexattention_for_efficient_high-resolution_vision-language_models.md)**

:   提出 FlexAttention，通过基于注意力图的高分辨率token动态选择和层次化自注意力融合机制，在保持甚至超越现有高分辨率VLM性能的同时，将计算成本降低近40%。

**[FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_highresolution_visionlanguage_mo.md)**

:   FlexAttention动态选择约10%高分辨率token进行层次自注意力，计算成本降40%且性能超越。

**[FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)**

:   首次在**完全不使用动捕数据**的情况下，利用 MLLM（GPT-4V）作为关键帧设计师和动画师，结合基于物理的运动跟踪，实现开放集人体运动合成。

**[FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocapfree_human_motion_synthesis_with_multimodal_.md)**

:   FreeMotion首次在不使用任何动捕数据的情况下，利用GPT-4V作为关键帧设计师和动画师，将自然语言指令分解为关键帧序列，再通过插值和基于物理的运动跟踪填充帧间运动，实现了开放集人体动作合成。

**[GENIXER: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powe.md)**

:   Genixer提出一套完整的视觉指令微调数据生成pipeline，通过训练现有MLLM（LLaVA1.5和Shikra）使其具备数据生成能力，无需GPT-4即可生成高质量的VQA和REC指令数据，并通过Fuyu驱动和CLIP驱动的自动过滤框架保证数据质量。

**[Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)**

:   提出 Genixer 数据生成流水线，训练 MLLM 自身作为数据生成器，无需依赖 GPT-4V 即可自动生成高质量视觉指令微调数据，生成的 915K VQA 数据和 350K REC 数据分别提升 LLaVA1.5 和 Shikra 在多个基准上的表现。

**[Getting it Right: Improving Spatial Consistency in Text-to-Image Models](getting_it_right_improving_spatial_consistency_in_texttoimag.md)**

:   发现现有VL数据集严重缺乏空间关系描述（如left/right/above/behind出现率极低），构建了首个空间聚焦的大规模数据集SPRIGHT（600万张图像重描述），仅用0.25%数据微调即可提升22%空间一致性得分，用<500张多物体图像微调达到T2I-CompBench空间SOTA 0.2133。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal.md)**

:   提出Groma，通过在视觉tokenizer中引入区域提议和区域编码机制，将定位能力嵌入图像token化过程，实现统一的referring和grounding能力，在标准基准上超越同类MLLM。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)**

:   Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

**[AutoVER: Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)**

:   提出 AutoVER，在多模态大语言模型中统一集成对比检索和前缀树约束解码，将 600 万级 Wikipedia 实体空间先缩小到数百候选再做受限生成，在 Oven-Wiki 上将 entity seen 准确率从 PaLI-17B 的 30.6% 翻倍到 61.5%，同时在 unseen/query split 上也大幅领先。

**[Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multimodal_contrastive_learning_with_exper.md)**

:   提出eCLIP，通过引入放射科医生的眼动热力图（eye-gaze heatmap）作为专家标注，利用热力图处理器和mixup增强策略扩充高质量正样本对，有效缓解医学CLIP中的"模态间隙"问题，在零样本推理、线性探测、跨模态检索和RAG报告生成等任务上取得一致性提升。

**[JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation](jointdreamer_ensuring_geometry_consistency_and_text_congruen.md)**

:   JointDreamer提出JSD通过能量函数建模多视角联合分布确保3D一致性。

**[Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multimodal_motion_generation.md)**

:   LMM是首个多模态通用人体动作生成模型，统一了文本/动作/音乐/语音等10种任务、16个数据集（320K序列/1亿帧），通过身体部位感知的ArtAttention机制和可变帧率+随机遮掩的预训练策略，在多个标准benchmark上与专家模型竞争甚至超越。

**[Latent Guard: A Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_texttoimage_generation.md)**

:   Latent Guard在T2I文本编码器上学习潜在空间检测黑名单概念。

**[LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](layoutdetr_detection_transformer_is_a_good_multimodal_layout.md)**

:   将版式设计问题重新构建为基于背景图像的目标检测问题，提出LayoutDETR框架，利用DETR的transformer编解码器结构结合GAN/VAE生成先验，以多模态前景元素（图像+文本）为输入，生成考虑背景语义的排版布局，在公开基准和自建广告横幅数据集上均达到SOTA。

**[LCM-Lookahead for Encoder-Based Text-to-Image Personalization](lcmlookahead_for_encoderbased_texttoimage_personalization.md)**

:   本文提出利用LCM（Latent Consistency Model）作为"快捷通道"，在扩散模型encoder训练中实现图像空间损失（如身份识别loss）的反向传播，配合自注意力特征共享和一致性数据生成，显著提升encoder-based人脸个性化的身份保持和prompt对齐能力。

**[Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audiovisual_question_answerin.md)**

:   提出面向音视觉问答（AVQA）的缺失模态处理框架，通过Relation-aware Missing Modal生成器利用三模态关系召回缺失信息，再通过Audio-Visual Relation-aware扩散模型增强特征表示，即使缺少一个模态也能准确回答问题。

**[MarvelOVD: 融合目标检测器与视觉语言模型实现鲁棒开放词汇目标检测](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)**

:   提出 MarvelOVD 框架，通过将检测器的上下文感知能力和背景识别能力融入 VLM 的伪标签生成与训练流程，在线净化噪声伪标签并自适应重加权训练框，在 COCO 和 LVIS 上大幅超越已有方法。

**[MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)**

:   分析了VLM（CLIP）在局部区域预测中产生噪声伪标签的两大根因——缺乏上下文信息和无"背景"概念，提出MarvelOVD结合检测器的上下文和背景感知能力进行在线伪标签挖掘，配合自适应提案重加权和分层标签分配，在COCO和LVIS上显著超越SOTA。

**[MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)**

:   提出MathVerse——一个包含2612道视觉数学题目（转化为6个版本共15K测试样本）的多模态数学推理评测基准，通过系统性地调控文本与图像中的信息分配来检验MLLM是否真正"看懂"了数学图表，并提出CoT评估策略进行细粒度推理过程评分，揭示了大多数MLLM严重依赖文本而非视觉图表进行数学推理。

**[MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math?](mathverse_does_your_multimodal_llm_truly_see_the_diagrams_in.md)**

:   提出MathVerse——一个专门评估MLLM视觉数学推理能力的基准，通过将每道题转化为6个版本（从文本主导到纯视觉），揭示大多数MLLM严重依赖文本提示而非真正理解数学图表，并提出CoT评估策略进行细粒度推理过程评分。

**[MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization](mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)**

:   针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。

**[MMBench: Is Your Multi-modal Model an All-Around Player?](mmbench_is_your_multimodal_model_an_allaround_player.md)**

:   提出MMBench——一个系统设计的双语多模态评测基准，包含3000+多选题覆盖20个能力维度，并引入CircularEval策略和LLM辅助选项匹配，实现对VLM的鲁棒、细粒度评估。

**[MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal.md)**

:   MotionChain构建视觉-运动语言模型，通过VQ-VAE将动作token化支持多轮对话运动生成。

**[MyVLM: Personalizing VLMs for User-Specific Queries](myvlm_personalizing_vlms_for_userspecific_queries.md)**

:   MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。

**[Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](nymeria_a_massive_collection_of_multimodal_egocentric_daily_.md)**

:   Nymeria是全球最大野外人体运动数据集，300h/264人多设备多模态自我中心数据和310.5K句语言描述。

**[OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)**

:   提出OccGen，首次将扩散模型的"噪声到占据"生成范式引入3D语义占据预测任务，通过条件编码器+渐进式精炼解码器实现由粗到精的占据图生成，在nuScenes-Occupancy上多模态/纯LiDAR/纯相机设置下分别提升mIoU 9.5%/6.3%/13.3%。

**[Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniviewtuning_boosting_viewpoint_invariance_of_visionlangua.md)**

:   OVT构建400万+MVCap数据集+Cross-Viewpoint Alignment提升VLP视角不变性。

**[OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_openset_panoptic_scene_graph_generation_via_large_mu.md)**

:   本文首次提出开放集全景场景图生成任务（OpenPSG），利用大型多模态模型（BLIP-2）以自回归方式预测物体间的开放集关系，通过关系查询Transformer高效提取物体对特征并过滤无关对，在闭集和开放集设置下均取得SOTA。

**[Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_texttoimage_generation_vi.md)**

:   将个性化T2I建模为DPG框架，引入Q函数和向前看机制捕获长期视觉一致性。

**[Quantized Prompt for Efficient Generalization of Vision-Language Models](quantized_prompt_for_efficient_generalization_of_visionlangu.md)**

:   发现适度噪声可以抑制VLM prompt tuning中的过拟合和灾难性遗忘，首次将量化误差视为正则化，设计了基于K-Means聚类的量化感知训练算法，在11个数据集上以极小存储开销（0.26KB）超越了众多SOTA方法。

**[Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_i.md)**

:   发现训练图文对齐模型时正负caption之间存在被忽视的数据集级别分布偏差（如GPT生成负样本时倾向用elephant替换giraffe），提出用纯文本分类器过滤高置信样本来消除偏差，结合替换型+交换型两类负样本微调LLaVA-1.5，在Winoground、SeeTRUE等多个基准上大幅超越现有方法。

**[Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_visionlanguage_adapters.md)**

:   CLIP适配方法OOD校准退化的根因是logit范围增大，提出SaLS等方案。

**[ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](scaledreamer_scalable_textto3d_synthesis_with_asynchronous_s.md)**

:   提出异步分数蒸馏(ASD)，通过将扩散时间步前移（而非微调扩散模型）来减小噪声预测误差，实现稳定的3D生成器训练并可扩展到100K文本提示，保持扩散模型的文本理解能力不受损。

**[SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](scenegraphloc_crossmodal_coarse_visual_localization_on_3d_sc.md)**

:   提出SceneGraphLoc，首次将queryimage在多模态3D场景图数据库中进行粗定位，通过学习场景图节点和图像patch的统一嵌入空间，在存储效率提升1000倍的同时接近图像检索方法的定位精度。

**[SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_visionlanguage_learning_for_grounded_s.md)**

:   提出SceneVerse——首个百万级3D视觉语言数据集（68K场景+250万语言描述），通过结合人工标注和基于场景图的自动生成pipeline构建多粒度描述，并设计GPS预训练框架实现多层次场景-文本对齐，在3D grounding和QA基准上达到SOTA。

**[SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_selfattention_for_dense_visionlanguage_infe.md)**

:   发现CLIP在密集预测中失败的根因是自注意力机制导致的空间位置错配（spatial-invariant features），提出Correlative Self-Attention(CSA)机制——仅用一个投影矩阵计算token间相关性作为注意力分数，无需任何训练/额外参数即可将CLIP的零样本语义分割mIoU从14.1%提升至38.2%（8个基准平均），大幅超越现有SOTA的33.9%。

**[ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)**

:   ShareGPT4V 构建了一个120万条高质量描述性caption数据集（由GPT4-Vision生成100K种子 + Share-Captioner扩展至1.2M），通过在预训练和SFT两阶段使用该数据集训练LLaVA架构的模型ShareGPT4V-7B，在11个多模态benchmark中9个取得最优，证明了高质量caption是LMM模态对齐的关键瓶颈。

**[ShareGPT4V: Improving Large Multi-modal Models with Better Captions](sharegpt4v_improving_large_multimodal_models_with_better_cap.md)**

:   指出现有LMM训练中低质量caption是模态对齐的瓶颈，构建了1.2M高质量详细描述的ShareGPT4V数据集（100K来自GPT4-Vision + 1.2M来自训练得到的Share-Captioner），在预训练和SFT两阶段使用该数据，以简单架构的7B模型在11个基准中9个取得最优。

**[SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sqllava_selfquestioning_for_large_visionlanguage_assistant.md)**

:   提出SQ-LLaVA，首次将指令数据中问题作为额外学习目标，训练MLLM不仅回答问题还学会"自问"，通过视觉自提问（visual self-questioning）任务挖掘指令数据中被忽视的问题上下文信息，配合原型提取器和LoRA微调，在10个VQA基准中9个超越基线。

**[The Hard Positive Truth About Vision-Language Compositionality](the_hard_positive_truth_about_visionlanguage_compositionalit.md)**

:   本文揭示了现有CLIP组合性基准的评估盲区——缺少hard positives测试，发现hard negative微调会导致模型"过敏"（对语义保持的改写也错误地降低匹配分数），并通过同时加入hard positives和hard negatives训练来缓解这一问题。

**[TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data](tip_tabularimage_pretraining_for_multimodal_classification_w.md)**

:   提出TIP框架，通过掩码表格重建、图像-表格匹配和对比学习三个自监督任务，在表格数据不完整的条件下学习鲁棒的多模态表示，在自然图像和医学图像分类任务上超越现有方法。

**[Towards Multi-modal Transformers in Federated Learning](towards_multimodal_transformers_in_federated_learning.md)**

:   首次探索Transformer架构在转移式多模态联邦学习中的应用，提出FedCola框架，通过互补式本地训练（利用跨模态Transformer blocks）和协作式服务器聚合（选择性聚合self-attention层），在保护数据隐私的前提下有效训练多模态Transformer。

**[Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models](towards_realworld_adverse_weather_image_restoration_enhancin.md)**

:   提出WResVLM半监督框架，利用VLM评估图像清晰度和提供语义信息，通过伪标签选择+天气prompt学习增强清晰度、VLM描述引导的语义正则化增强语义，首次有效地将合成数据训练的复原模型泛化到真实恶劣天气场景。

**[Towards Reliable Advertising Image Generation Using Human Feedback](towards_reliable_advertising_image_generation_using_human_fe.md)**

:   针对电商广告图像生成中大量不可用图像（空间不匹配、尺寸不匹配、不显著、形状幻觉）的问题，构建了百万级RF1M数据集训练多模态检测网络RFNet，并提出基于RFNet反馈微调扩散模型的RFFT方法（含Consistent Condition正则化），将可用率从约50%提升至接近100%且不损失美观性。

**[UMBRAE: Unified Multimodal Brain Decoding](umbrae_unified_multimodal_brain_decoding.md)**

:   提出UMBRAE，通过通用脑编码器将fMRI信号与图像特征对齐后送入冻结的MLLM，实现多模态脑解码（描述、定位、检索、视觉重建），并创新性地引入跨被试训练策略，使单一模型服务多个被试且优于单被试模型。

**[UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_lan.md)**

:   提出UniCode，通过语言驱动的迭代训练范式学习一个统一码本，使LLM的词表可同时量化视觉和文本信号，无需额外对齐模块即可实现多模态理解与生成，并引入上下文图像解压缩任务提升生成质量。

**[UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)**

:   UniCode提出学习一个统一的codebook来同时tokenize视觉和文本信号，通过language-driven iterative training范式将视觉tokenizer的码本与LLM的词表渐进对齐，并引入in-context image decompression预训练任务提升图像生成质量，使MLLM无需额外对齐模块即可实现多模态理解与生成。

**[Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

**[View Selection for 3D Captioning via Diffusion Ranking](view_selection_for_3d_captioning_via_diffusion_ranking.md)**

:   DiffuRank用预训练text-to-3D扩散模型评估视角对齐度选择最佳视角减少幻觉。

**[X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](xformer_unifying_contrastive_and_reconstruction_learning_for.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP（全局语义）和MAE（局部细节）两种视觉编码器的互补特征，结合ITC/ITM/ITG和重建四个损失联合优化，提升MLLM的细粒度视觉理解能力。

**[XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)**

:   XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🧠 NeurIPS2025** · 共 **20** 篇

**[Alligat0R: Pre-Training through Covisibility Segmentation for Relative Camera Pose Regression](alligat0r_pre-training_through_co-visibility_segmentation_for_relative_camera_po.md)**

:   用共视性分割（covisibility segmentation）替代 CroCo 的跨视图补全作为双目视觉预训练任务，对每个像素预测"共视/遮挡/视野外"三类标签，在低重叠场景下显著超越 CroCo，RUBIK 基准总体成功率 60.3% 排第一。

**[ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](argenseg_image_segmentation_with_autoregressive_image_generation_model.md)**

:   提出ARGenSeg——首个利用自回归图像生成范式实现图像分割的统一MLLM框架，让模型直接输出visual tokens并通过VQ-VAE解码为分割mask，无需额外分割头，搭配next-scale prediction并行生成策略实现4×加速，在RefCOCO/+/g上以更少训练数据超越SOTA。

**[Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)**

:   将 softmax 归一化后的注意力矩阵重新解读为离散时间 Markov 链（DTMC）的转移概率矩阵，提出多跳注意力（Multi-Bounce）和 TokenRank（稳态分布，类似 PageRank）来捕获间接注意力路径和全局 token 重要性，在 ImageNet 分割上达 94.29% mAP，并增强 Self-Attention Guidance 的图像生成质量。

**[Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)**

:   揭示 LLM 能秘密处理非标准分词（如将"Hello"拆为"He"+"llo"而非标准的"Hello"整词token）——即使输入的 token 序列与训练时不同，模型表现出惊人的鲁棒性，且这种能力来自嵌入空间中子词嵌入的线性组合近似整词嵌入的特性。

**[ConnectomeBench: Can LLMs Proofread the Connectome?](connectomebench_can_llms_proofread_the_connectome.md)**

:   构建ConnectomeBench评估LLM在3D神经元网格理解上的能力——包括片段识别、分裂错误修正和合并错误检测三个任务，Claude 3.7在片段识别达82%和分裂修正达85%（远超baseline 20-50%），但合并检测仍落后于人类专家。

**[COS3D: Collaborative Open-Vocabulary 3D Segmentation](cos3d_collaborative_open-vocabulary_3d_segmentation.md)**

:   提出COS3D协作式开放词汇3D分割框架，在3D Gaussian Splatting中同时维护instance field（学习清晰边界）和language field（学习语义），通过两阶段训练实现Ins2Lang映射，推理时Language→Instance prompt精化实现互补协作，在LeRF数据集上mIoU达50.76%，大幅超越Dr.Splat（43.58%）。

**[Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)**

:   解决扩散语言模型（DLM）的长解码窗口问题——通过卷积归一化（替代硬半自回归分块）和基于规则的拒绝微调（R2FT, DPO风格下采权高先验/重复token），在128步（而非512+步）下实现DLM的SOTA生成质量。

**[Fast Foreground-Aware Diffusion With Accelerated Sampling Trajectory For Segment](fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)**

:   提出 FAST，一个面向分割的工业异常合成框架，通过前景感知重建模块（FARM）和异常感知加速采样（AIAS）在仅 10 步去噪下生成高质量合成异常，在 MVTec-AD 上 mIoU 达 76.72%，超越所有先前方法。

**[FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)**

:   提出FineRS两阶段MLLM强化学习框架（全局语义探索GSE→局部感知精化LPR），通过locate-informed retrospective reward耦合两阶段，在自建FineRS-4k UAV高分辨率数据集上实现超小目标的推理与分割，gIoU达55.1%（超Seg-Zero† 8.5%），同时支持VQA（MVQA 83.3%）。

**[HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)**

:   提出HAODiff——人体感知的单步扩散复原模型，设计包含人体运动模糊(HMB)的退化流水线、三分支双提示引导(DPG)网络产生正/负prompt对，通过CFG指导SD2.1单步去噪，在合成和真实场景均大幅领先（FID 8.36 vs 14.41，HMB残留率0.09 vs 0.19）。

**[LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)**

:   提出LangHOPS——首个基于MLLM的开放词汇物体-部件实例分割框架，在语言空间中建立object-part层次结构，利用MLLM的知识和推理能力进行多粒度概念链接，在PartImageNet上in-domain达56.9 AP（超SOTA 6.5），cross-dataset设置下超越5.7 AP。

**[OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)**

:   OmniSegmentor 构建了含 5 种视觉模态的大规模 ImageNeXt 数据集（1.2M 样本），提出随机选择补充模态与 RGB 对齐的高效预训练策略，首次实现灵活的多模态预训练-微调流水线，在 6 个多模态语义分割基准上刷新 SOTA。

**[Panoptic Captioning An Equivalence Bridge For Image And Text](panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)**

:   提出 Panoptic Captioning 新任务，追求图像的"最小文本等价"——生成包含所有实体、位置、属性、关系和全局状态的全面描述，13B 模型配合解耦学习即超越 78B 开源和 GPT-4o 等商业模型。

**[PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](partonomy_large_multimodal_models_with_part-level_visual_understanding.md)**

:   提出 Partonomy 部件级分割 benchmark（862 部件标签/534 物体标签）和 Plum 模型（用 span 标记替代 [SEG] token + mask 反馈循环），发现 SOTA 分割 LMM 在部件理解上仅 5.9% gIoU，Plum 通过避免分布偏移和利用历史预测显著提升。

**[HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)**

:   HCLFuse 基于信息瓶颈原理和最优传输理论进行模态对齐，设计变分瓶颈编码器（VBE）+ 物理引导条件扩散模型，融合热传导/结构保持/物理一致性三种约束到扩散过程中，在 MSRS 数据集上梯度指标 AG 提升 69.87%，空间频率 SF 提升 39.41%。

**[Robust Ego-Exo Correspondence with Long-Term Memory](robust_ego-exo_correspondence_with_long-term_memory.md)**

:   提出LM-EEC，基于SAM 2的自中心-外中心(ego-exo)视频跨视角目标分割框架，通过Memory-View MoE自适应融合记忆特征与跨视角特征，配合双记忆库压缩策略保持长期信息，在EgoExo4D基准上大幅超越现有方法（Ego2Exo IoU 54.98 vs 38.26）。

**[Robust Egocentric Referring Video Object Segmentation Via Dual-Modal Causal Inte](robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)**

:   提出CERES框架，通过双模态因果干预解决自中心指代视频分割(Ego-RVOS)中的鲁棒性问题：对语言偏见用后门调整（消除目标-动作频率偏差），对视觉混淆用前门调整（以深度信息引导视觉中介变量聚合），在VISOR/VOST/VSCOS上达到SOTA。

**[Sam-R1 Leveraging Sam For Reward Feedback In Multimodal Segmentation Via Reinfor](sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)**

:   提出 SAM-R1，将 SAM 作为 RL 训练循环中的奖励提供者（而非仅下游模块），设计分层 IoU 分割精度奖励 + 推理/分割格式奖励，配合改进的非对称裁剪 GRPO 算法，仅用 3k 训练样本在 ReasonSeg 零样本基准上超越 Seg-Zero 等方法，证明了细粒度分割奖励对 MLLM 推理-分割对齐的有效性。

**[Sansa Unleashing The Hidden Semantics In Sam2 For Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)**

:   发现 SAM2 虽然是类无关预训练，但其特征中已隐含编码了丰富的语义结构（只是与跟踪特征纠缠在一起），通过在 Image Encoder 最后两层加入轻量 AdaptFormer 即可将语义结构显式化，将 SAM2 的"对象跟踪"重新诠释为"语义跟踪"，在少样本分割基准上达到 SOTA 且比竞争方法快 3× 小 4-5×。

**[Towards Robust Pseudo-Label Learning In Semantic Segmentation An Encoding Perspe](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)**

:   提出 ECOCSeg，用纠错输出码（ECOC）替代 one-hot 编码来表示伪标签，将 N 类分类分解为 K 个二分类子任务，通过 bit 级去噪和可靠位挖掘生成更鲁棒的伪标签，在 UDA 和 SSL 分割任务上一致提升。

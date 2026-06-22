---
title: >-
  NeurIPS2025 视频生成论文汇总 · 23篇论文解读
description: >-
  23篇NeurIPS2025的视频生成方向论文解读，涵盖视频生成、扩散模型、对抗鲁棒、对齐/RLHF、布局/合成、推荐系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "布局/合成"
  - "推荐系统"
item_list:
  - u: "autoregressive_adversarial_posttraining_for_realtime_interac/"
    t: "Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation"
  - u: "densedpo_finegrained_temporal_preference_optimization_for_vi/"
    t: "DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models"
  - u: "dismo_disentangled_motion_representations_for_openworld_moti/"
    t: "DisMo: Disentangled Motion Representations for Open-World Motion Transfer"
  - u: "force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c/"
    t: "Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals"
  - u: "foresight_adaptive_layer_reuse_for_accelerated_and_highquali/"
    t: "Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation"
  - u: "lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge/"
    t: "LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation"
  - u: "magcache_fast_video_generation_with_magnitudeaware_cache/"
    t: "MagCache: Fast Video Generation with Magnitude-Aware Cache"
  - u: "photography_perspective_composition_towards_aesthetic_perspective_recommendation/"
    t: "Photography Perspective Composition: Towards Aesthetic Perspective Recommendation"
  - u: "physctrl_generative_physics_for_controllable_and_physicsgrou/"
    t: "PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation"
  - u: "posecrafter_extreme_pose_estimation_with_hybrid_video_synthesis/"
    t: "PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis"
  - u: "radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener/"
    t: "Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation"
  - u: "rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video/"
    t: "RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation"
  - u: "s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp/"
    t: "S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation"
  - u: "safesora_safe_texttovideo_generation_via_graphical_watermark/"
    t: "Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking"
  - u: "scaling_rl_to_long_videos/"
    t: "Scaling RL to Long Videos"
  - u: "seeing_the_wind_from_a_falling_leaf/"
    t: "Seeing the Wind from a Falling Leaf"
  - u: "self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion/"
    t: "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
  - u: "stable_cinemetrics_structured_taxonomy_and_evaluation_for_professional_video_gen/"
    t: "Stable Cinemetrics: Structured Taxonomy and Evaluation for Professional Video Generation"
  - u: "training-free_efficient_video_generation_via_dynamic_token_carving/"
    t: "Training-Free Efficient Video Generation via Dynamic Token Carving"
  - u: "video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi/"
    t: "Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision"
  - u: "video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o/"
    t: "Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models"
  - u: "vorta_efficient_video_diffusion_via_routing_sparse_attention/"
    t: "VORTA: Efficient Video Diffusion via Routing Sparse Attention"
  - u: "vsa_faster_video_diffusion_with_trainable_sparse_attention/"
    t: "VSA: Faster Video Diffusion with Trainable Sparse Attention"
item_total: 23
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🧠 NeurIPS2025** · **23** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (182)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (97)](../../ICLR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🧪 ICML2026 (32)](../../ICML2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [📹 ICCV2025 (49)](../../ICCV2025/video_generation/index.md)

🔥 **高频主题：** 视频生成 ×12 · 扩散模型 ×7

**[Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](autoregressive_adversarial_posttraining_for_realtime_interac.md)**

:   本文提出 AAPT（Autoregressive Adversarial Post-Training），通过对抗训练将预训练视频扩散模型转化为自回归实时视频生成器，每帧仅需一次前向传播（1NFE），基于 student-forcing 训练减少误差累积，8B 模型在单张 H100 上实现 736×416 分辨率 24fps 实时流式生成，最长可达一分钟（1440帧）。

**[DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](densedpo_finegrained_temporal_preference_optimization_for_vi.md)**

:   识别并解决视频 DPO 的运动偏差问题——通过从 GT 视频加噪去噪构造结构对齐的视频对来固定运动维度、在时间片段级标注密集偏好来获取更精准的学习信号、用现成 VLM 自动标注来降低成本，仅用 1/3 标注数据即大幅提升运动生成质量同时匹配视觉质量和文本对齐。

**[DisMo: Disentangled Motion Representations for Open-World Motion Transfer](dismo_disentangled_motion_representations_for_openworld_moti.md)**

:   DisMo 通过双流架构（运动提取器 + 帧生成器）和图像空间重建目标，从原始视频中学习与外观、姿态、类别无关的抽象运动表征，实现跨类别/跨视角的开放世界运动迁移，并在零样本动作分类上大幅超越 V-JEPA 等视频表征模型。

**[Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)**

:   提出Force Prompting，将物理力（局部点力和全局风力）作为视频生成模型的控制信号，仅用~15K合成训练视频（Blender旗帜和滚球）和单日4xA100训练，即可在多样真实场景图像上展现跨物体/材质/几何的惊人泛化，包括初步的质量理解能力。

**[Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)**

:   提出 Foresight，一种训练无关的自适应层复用框架，通过在 warmup 阶段建立逐层 MSE 阈值、在 reuse 阶段按阈值动态决策每层是复用缓存还是重新计算，在 5 个视频生成模型上实现了比静态方法更高质量和更快速度的推理加速（最高 2.23×）。

**[LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)**

:   提出 LeMiCa，一种免训练的扩散视频生成加速框架，将缓存调度建模为有向无环图上的字典序极小极大路径优化问题，通过全局误差控制实现速度和质量的双重提升（Latte 上 2.9× 加速，Open-Sora 上 LPIPS 低至 0.05）。

**[MagCache: Fast Video Generation with Magnitude-Aware Cache](magcache_fast_video_generation_with_magnitudeaware_cache.md)**

:   发现视频扩散模型中相邻时间步残差输出的幅度比（magnitude ratio）遵循一条跨模型、跨 prompt 普遍成立的单调递减规律（"统一幅度定律"），由此提出 MagCache：基于幅度比对跳步误差进行精确累积建模，自适应跳过冗余时间步并复用缓存，仅需 1 个样本校准，即可在 Open-Sora、CogVideoX、Wan 2.1、HunyuanVideo 等模型上实现 2.10–2.68× 加速，且在 LPIPS/SSIM/PSNR 三个指标上全面优于 TeaCache 等已有方法。

**[Photography Perspective Composition: Towards Aesthetic Perspective Recommendation](photography_perspective_composition_towards_aesthetic_perspective_recommendation.md)**

:   提出"摄影透视构图"(PPC) 新范式，超越传统裁剪方法，通过 3D 重建构建透视变换数据集 + Image-to-Video 生成推荐视角 + RLHF 对齐人类偏好 + PQA 模型评估透视质量。

**[PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](physctrl_generative_physics_for_controllable_and_physicsgrou.md)**

:   PhysCtrl用扩散模型学习四种材料（弹性/沙/橡皮泥/刚体）的物理动力学分布，将动态表示为3D点轨迹，在55万合成动画上训练含时空注意力+物理约束的扩散模型，生成的轨迹驱动预训练视频模型实现力和材料参数可控的高保真物理视频生成。

**[PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis](posecrafter_extreme_pose_estimation_with_hybrid_video_synthesis.md)**

:   提出 PoseCrafter，一种无需训练的极端位姿估计框架：通过混合视频生成（HVG，DynamiCrafter+ViewCrafter双阶段）合成高保真中间帧解决极小/无重叠图像对的位姿估计，配合特征匹配选择器（FMS）高效选取最有用的中间帧，在四个数据集上显著提升极端位姿估计精度。

**[Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)**

:   Radial Attention 发现了视频扩散模型中注意力分数随时空距离指数衰减的"时空能量衰减"现象，据此设计了一种 O(n log n) 复杂度的静态稀疏注意力掩码，在 HunyuanVideo/Wan2.1 等模型上实现最高 3.7× 推理加速，并通过 LoRA 微调支持 4× 更长视频生成。

**[RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)**

:   本文首次系统量化自动驾驶视频生成中的几何失真问题，提出 RLGF 框架通过层次化几何奖励（消失点-车道线-深度-占用）和潜空间滑动窗口优化策略，将 3D 目标检测 mAP 提升 12.7 个绝对百分点（25.75→31.42），大幅缩小合成数据与真实数据的性能差距。

**[S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)**

:   针对视频扩散 Transformer 的超长 token 序列导致的量化校准高方差和学习困难问题，提出 S²Q-VDiT 框架，利用 Hessian 感知的显著数据选择和注意力引导的稀疏 token 蒸馏两项技术，首次在 W4A6 设置下实现无损量化，带来 3.9× 模型压缩和 1.3× 推理加速。

**[Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking](safesora_safe_texttovideo_generation_via_graphical_watermark.md)**

:   Safe-Sora 首次将**图形水印**（如logo图像）直接嵌入到视频生成管线中，通过分层粗到细自适应匹配将水印patch分配到视觉最相似的帧和区域，并设计3D小波变换增强Mamba架构实现时空融合，在视频质量（FVD 3.77 vs 次优154.35）和水印保真度上大幅超越所有基线。

**[Scaling RL to Long Videos](scaling_rl_to_long_videos.md)**

:   提出 LongVILA-R1 全栈框架，通过 104K 长视频推理数据集、两阶段 CoT-SFT + RL 训练管线、以及 MR-SP 多模态强化序列并行系统，将 VLM 的推理能力扩展到长视频（最高 8192 帧），在 VideoMME 上达到 65.1%/71.1%。

**[Seeing the Wind from a Falling Leaf](seeing_the_wind_from_a_falling_leaf.md)**

:   提出端到端可微逆图形学框架，通过联合建模物体几何/物理属性、力场表示和物理过程，从视频中反向传播恢复不可见的力场（如风场），并支持基于物理的视频生成和编辑。

**[Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)**

:   提出 Self Forcing 训练范式，通过在训练时执行自回归自展开（self-rollout）并使用整体视频级分布匹配损失（DMD/SiD/GAN），消除了 Teacher Forcing 和 Diffusion Forcing 中训练-推理分布不匹配导致的暴露偏差问题，基于 Wan2.1-1.3B 实现了单 GPU 上 17 FPS 实时流式视频生成，同时质量匹敌甚至超越慢几十倍的双向扩散模型。

**[Stable Cinemetrics: Structured Taxonomy and Evaluation for Professional Video Generation](stable_cinemetrics_structured_taxonomy_and_evaluation_for_professional_video_gen.md)**

:   提出 SCINE（Stable Cinemetrics），首个面向专业视频制作的结构化评估框架，定义了 76 个细粒度电影控制节点的分层分类体系，配合大规模专业人员评估（80+ 影视从业者、20K+ 视频、248K 标注），揭示当前最强 T2V 模型在专业控制上的显著不足。

**[Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)**

:   本文提出 Jenga，一种免训练的视频 DiT 推理加速方案，通过动态块注意力裁剪（基于 3D 空间填充曲线重排 token 后进行稀疏 KV block 选择）和渐进分辨率策略（从低分辨率逐步提升）正交结合，在 HunyuanVideo 上实现 8.83 倍加速且 VBench 仅下降 0.01%。

**[Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision](video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)**

:   发现预训练视频扩散模型在高噪声去噪阶段天然学到了适合追踪的运动表示，提出 TED 框架融合运动和外观特征，在追踪外观相似物体时比现有自监督方法提升多达 10 个百分点。

**[Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)**

:   对开源文本到视频 (T2V) 模型进行系统性延迟和能耗分析，建立了基于 FLOP 的计算分析模型预测 WAN2.1 的缩放规律（空间/时间维度二次缩放、去噪步数线性缩放），并在 7 个 T2V 模型上提供跨模型能耗基准。

**[VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)**

:   提出VORTA框架，通过桶化核心集注意力（建模长程依赖）和信号感知路由机制（自适应选择稀疏注意力分支），在不损失生成质量的前提下实现视频扩散Transformer端到端1.76×加速，并可与缓存和蒸馏方法叠加达到14.41×加速。

**[VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)**

:   提出 VSA (Video Sparse Attention)，一种端到端可训练的硬件对齐稀疏注意力机制，通过粗粒度阶段（cube 池化预测关键 token）和细粒度阶段（在预测的块稀疏区域执行 token 级注意力）的层次化设计，在视频 DiT 的训练和推理中同时实现加速：从头预训练实现 2.53× 训练 FLOPs 减少且无质量损失，适配 Wan2.1-1.3B 实现注意力 6× 加速和端到端推理从 31s 降至 18s。

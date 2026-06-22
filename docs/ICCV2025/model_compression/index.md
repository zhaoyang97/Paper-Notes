---
title: >-
  ICCV2025 模型压缩论文汇总 · 52篇论文解读
description: >-
  52篇ICCV2025的模型压缩方向论文解读，涵盖模型压缩、知识蒸馏、压缩/编码、持续学习、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "知识蒸馏"
  - "压缩/编码"
  - "持续学习"
  - "对抗鲁棒"
item_list:
  - u: "a_good_teacher_adapts_their_knowledge_for_distillation/"
    t: "A Good Teacher Adapts Their Knowledge for Distillation"
  - u: "acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation/"
    t: "ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation"
  - u: "achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme/"
    t: "Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning"
  - u: "argmatch_adaptive_refinement_gathering_for_efficient_dense_matching/"
    t: "ARGMatch: Adaptive Refinement Gathering for Efficient Dense Matching"
  - u: "b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens/"
    t: "B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens"
  - u: "beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer/"
    t: "Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes"
  - u: "bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation/"
    t: "Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation"
  - u: "ciard_cyclic_iterative_adversarial_robustness_distillation/"
    t: "CIARD: Cyclic Iterative Adversarial Robustness Distillation"
  - u: "color_matching_using_hypernetwork-based_kolmogorov-arnold_networks/"
    t: "Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)"
  - u: "colors_see_colors_ignore_clothes_changing_reid_with_color_disentanglement/"
    t: "Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement"
  - u: "competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif/"
    t: "Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification"
  - u: "context_guided_transformer_entropy_modeling_for_video_compression/"
    t: "Context Guided Transformer Entropy Modeling for Video Compression"
  - u: "cross-architecture_distillation_made_simple_with_redundancy_suppression/"
    t: "Cross-Architecture Distillation Made Simple with Redundancy Suppression"
  - u: "dataset_distillation_via_the_wasserstein_metric/"
    t: "Dataset Distillation via the Wasserstein Metric"
  - u: "dlf_extreme_image_compression_with_dual-generative_latent_fusion/"
    t: "DLF: Extreme Image Compression with Dual-generative Latent Fusion"
  - u: "duolora_cycle-consistent_and_rank-disentangled_content-style_personalization/"
    t: "DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization"
  - u: "ea-kd_entropy-based_adaptive_knowledge_distillation/"
    t: "EA-KD: Entropy-based Adaptive Knowledge Distillation"
  - u: "ea-vit_efficient_adaptation_for_elastic_vision_transformer/"
    t: "EA-ViT: Efficient Adaptation for Elastic Vision Transformer"
  - u: "efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat/"
    t: "Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory"
  - u: "fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning/"
    t: "FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning"
  - u: "free-merging_fourier_transform_for_efficient_model_merging/"
    t: "FREE-Merging: Fourier Transform for Efficient Model Merging"
  - u: "frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast/"
    t: "SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting"
  - u: "fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation/"
    t: "Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation"
  - u: "gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp/"
    t: "Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP"
  - u: "generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform/"
    t: "Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations"
  - u: "gradient_short-circuit_efficient_out-of-distribution_detection_via_feature_inter/"
    t: "Gradient Short-Circuit: Efficient Out-of-Distribution Detection via Feature Intervention"
  - u: "heavy_labels_out_dataset_distillation_with_label_space_lightening/"
    t: "Heavy Labels Out! Dataset Distillation with Label Space Lightening"
  - u: "integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla/"
    t: "Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning"
  - u: "knowledge_distillation_with_refined_logits/"
    t: "Knowledge Distillation with Refined Logits"
  - u: "learned_image_compression_with_hierarchical_progressive_context_modeling/"
    t: "Learned Image Compression with Hierarchical Progressive Context Modeling"
item_total: 52
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📹 ICCV2025** · **52** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (108)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (239)](../../ICLR2026/model_compression/index.md) · [💬 ACL2026 (59)](../../ACL2026/model_compression/index.md) · [🧪 ICML2026 (116)](../../ICML2026/model_compression/index.md) · [🤖 AAAI2026 (60)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/model_compression/index.md)

🔥 **高频主题：** 模型压缩 ×10 · 知识蒸馏 ×5 · 压缩/编码 ×5 · 持续学习 ×3 · 对抗鲁棒 ×2

**[A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)**

:   本文揭示了知识蒸馏中教师-学生容量差距问题的本质原因在于**输出分布的类内分布不匹配**，并提出 AID（Adapted Intra-class Distribution）方法，在蒸馏前对教师模型进行微调以优化其类内分布使之更符合学生的学习能力，在多种架构组合上取得了SOTA性能。

**[ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)**

:   提出 ACAM-KD，一种自适应学生-教师协作注意力掩码知识蒸馏方法，通过跨注意力特征融合（STCA-FF）和自适应空间-通道掩码（ASCM）动态调整蒸馏焦点，在 COCO 检测上超越 SOTA 最高 1.4 mAP，在 Cityscapes 分割上提升 3.09 mIoU。

**[Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)**

:   提出 APT（Additive Prompt Tuning），用加法操作替代传统的提示拼接范式，仅在 CLS token 的 key/value 上添加两个可学习向量，在大幅降低计算开销（GFLOPs 减少 41.5%）和可训练参数（减少 78.2%）的同时实现 SOTA 的类增量学习性能。

**[ARGMatch: Adaptive Refinement Gathering for Efficient Dense Matching](argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)**

:   提出自适应精炼聚合（Adaptive Refinement Gathering）管线，包含内容感知偏移估计器、局部一致匹配校正器和局部一致上采样器三个模块，配合自适应门控机制，大幅减少了稠密匹配对重量级特征提取器和全局匹配器的依赖，以轻量级模型实现与SOTA可比的性能。

**[B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)**

:   本文提出B-VLLM框架，通过文本条件自适应帧选择、时序帧Token合并和空间Token采样三个模块，在VLLM的上下文窗口限制内动态平衡视频的时空线索，在MVBench上带来10%的性能提升。

**[Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)**

:   提出SR-LoRA（Stable Rank-Guided LoRA），利用预训练权重矩阵的稳定秩（Stable Rank）作为自然先验为每层LoRA模块分配最优秩，无需搜索即可实现灵活的逐层秩分配，在大域差距+少样本迁移场景（如医学影像）中显著优于固定低秩LoRA和其他自适应秩方法。

**[Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)**

:   提出TokenBridge，通过对预训练连续VAE特征进行后训练维度级量化，将连续token转化为离散token，在保持连续token高保真表示能力的同时，使用标准交叉熵损失进行简洁的自回归建模，在ImageNet 256×256上达到与连续方法可比的生成质量。

**[CIARD: Cyclic Iterative Adversarial Robustness Distillation](ciard_cyclic_iterative_adversarial_robustness_distillation.md)**

:   提出CIARD，通过对比推离损失（Contrastive Push Loss）解决双教师ARD框架中clean teacher和robust teacher的优化目标冲突，并设计迭代教师训练（ITT）策略持续更新robust teacher以防止性能退化，在CIFAR-10/100和Tiny-ImageNet上同时提升对抗鲁棒性+3.53%和干净准确率+5.87%。

**[Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)](color_matching_using_hypernetwork-based_kolmogorov-arnold_networks.md)**

:   提出cmKAN，利用超网络驱动的Kolmogorov-Arnold Network进行颜色匹配，通过生成器预测空间变化的KAN样条参数，支持有监督/无监督/配对优化三种场景和raw-to-raw/raw-to-sRGB/sRGB-to-sRGB三种任务，在所有任务上平均超越现有方法37.3%且极轻量（76.4K参数）。

**[Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement](colors_see_colors_ignore_clothes_changing_reid_with_color_disentanglement.md)**

:   提出CSCI方法，通过引入Color token学习颜色表示（Color See），并利用新颖的S2A自注意力机制将颜色信息与ReID特征解耦（Color Ignore），在无需外部标注的情况下有效消除换衣行人重识别中的外观偏差。

**[Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)**

:   提出竞争蒸馏策略，在多网络联合训练中，每个迭代动态选择表现最好的网络作为教师，配合随机扰动机制引入类似遗传算法的变异操作，显著提升视觉分类性能。

**[Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)**

:   提出Context Guided Transformer (CGT) 条件熵模型，通过时间上下文重采样器降低计算开销、依赖加权空间上下文分配器显式建模空间依赖关系，在视频压缩中将熵建模时间减少约65%，同时实现11% BD-Rate改进。

**[Cross-Architecture Distillation Made Simple with Redundancy Suppression](cross-architecture_distillation_made_simple_with_redundancy_suppression.md)**

:   提出RSD（Redundancy Suppression Distillation），通过跨架构不变性最大化和特征去相关来提取架构无关知识，仅用一个简单的RSD损失和轻量MLP解耦模块，在CIFAR-100和ImageNet-1k上大幅超越跨架构蒸馏先驱方法OFA，且参数开销仅为其小部分。

**[Dataset Distillation via the Wasserstein Metric](dataset_distillation_via_the_wasserstein_metric.md)**

:   提出 WMDD（Wasserstein Metric-based Dataset Distillation），使用 Wasserstein 重心替代 MMD 进行分布匹配，结合逐类 BatchNorm 正则化，在 ImageNet-1K 等大规模数据集上达到 SOTA 数据集蒸馏性能。

**[DLF: Extreme Image Compression with Dual-generative Latent Fusion](dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)**

:   提出双分支生成式隐空间融合（DLF）框架，将图像隐空间分解为语义和细节两个分支分别压缩，通过跨分支交互设计消除冗余，在极低码率（<0.01 bpp）下实现了超越 MS-ILLM 高达 67.82% BD-Rate 节省的 SOTA 重建质量，同时解码速度远快于扩散模型方案。

**[DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization](duolora_cycle-consistent_and_rank-disentangled_content-style_personalization.md)**

:   DuoLoRA 提出在 LoRA 的秩维度上学习掩码（ZipRank），结合 SDXL 层先验信息和循环一致性损失（Constyle loss），实现了高效的内容-风格 LoRA 合并，在多个基准上超过 ZipLoRA 等 SOTA 方法，且可训练参数减少 19 倍。

**[EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)**

:   提出 EA-KD，一种基于信息熵的即插即用知识蒸馏方法：通过结合 teacher 和 student 输出的熵值动态重加权蒸馏损失，优先学习高熵（高信息量）样本，在图像分类、目标检测和 LLM 蒸馏任务上均一致提升多种 KD 框架的性能，且计算开销可忽略。

**[EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)**

:   提出首个在适配（adaptation）阶段引入弹性结构的ViT框架，通过多维弹性架构+课程学习+轻量路由器，一次适配即生成覆盖10^26种配置的子模型，在多个下游任务上持续优于现有弹性方法。

**[Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)**

:   本文发现预训练 ViT 权重矩阵的行/列向量具有近似正交性，而 LoRA/Adapter 的投影矩阵不具备此性质；提出 AOFT 策略，用单个可学习向量生成近似正交的下/上投影矩阵，使其与骨干网络性质对齐，从而降低泛化误差上界，在 FGVC 和 VTAB-1k 上用更少参数达到竞争性能。

**[FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)**

:   FastVAR 提出一种无需训练的后处理加速方法，通过观察 VAR 模型中大尺度步骤主要建模高频纹理且对剪枝鲁棒的特性，利用频域引导的关键 token 选择（PTS）仅保留高频 token 参与前向，并用缓存的早期尺度 token 恢复被剪枝的位置（CTR），在 FlashAttention 基础上实现额外 2.7× 加速且性能损失 <1%，并首次实现单张 3090 GPU 上 1.5 秒生成 2K 图像。

**[FREE-Merging: Fourier Transform for Efficient Model Merging](free-merging_fourier_transform_for_efficient_model_merging.md)**

:   首次发现模型合并中任务干扰在频域上的表现，提出 FR-Merging 通过高通滤波去除低频干扰构建高质量合并骨干网络，并结合轻量级任务专家模块（FREE-Merging），在视觉、语言和多模态任务上实现性能-成本的最优平衡。

**[SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)**

:   提出SDKD（频域解耦知识蒸馏）框架，通过频率感知的教师模型和频率对齐的蒸馏策略，将复杂时空预测模型的多尺度频域知识迁移到轻量级学生网络，在Navier-Stokes数据集上MSE最高降低81.3%。

**[Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)**

:   提出 FBT（Fuse Before Transfer），通过在知识传递前先融合异构教师和学生的模块（CNN/MSA/MLP），构建一个自适应的中间融合模型来缓解跨架构蒸馏（CAKD）中的特征差距，并用空间无关的 InfoNCE 损失替代传统 MSE 损失，在 CIFAR-100 上平均提升 8.38%，在 ImageNet-1K 上平均提升 2.31%。

**[Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP](gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp.md)**

:   提出使用 10KB 轻量级 MLP 网络替代传统 JPEG/HEIC 压缩来编码 HDR gain map，以 SDR 图像的颜色和位置坐标 (r,g,b,x,y) 作为输入，结合指数残差编码（gamma map），在多个 HDR 重建指标上超越现有方法和传统压缩技术。

**[Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)**

:   提出 LieRA，利用李群理论将矩阵级 PEFT 方法（如 LoRA）推广到高维参数空间（如卷积核），通过在李代数中表示扰动并用指数映射回李群，在保持参数空间结构性质的同时实现高效微调。

**[Gradient Short-Circuit: Efficient Out-of-Distribution Detection via Feature Intervention](gradient_short-circuit_efficient_out-of-distribution_detection_via_feature_inter.md)**

:   本文发现 ID 样本的局部梯度方向一致而 OOD 样本梯度方向混乱，据此提出在推理阶段"短路"被虚假梯度利用的特征坐标来降低 OOD 置信度，并通过一阶近似避免二次前向传播，实现轻量高效的 OOD 检测。

**[Heavy Labels Out! Dataset Distillation with Label Space Lightening](heavy_labels_out_dataset_distillation_with_label_space_lightening.md)**

:   提出 HeLlO 框架，利用 CLIP 预训练模型和 LoRA-like 低秩知识迁移构建轻量级图像-标签投影器，将数据集蒸馏中软标签的存储需求降低至原来的 0.003%，同时保持甚至超越 SOTA 性能。

**[Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)**

:   提出 TUNA 方法，通过为每个增量任务训练正交的 task-specific adapter，并将它们融合为一个 universal adapter，结合基于熵的 adapter 选择机制和双 adapter 集成推理策略，在无 exemplar 的 PTM-based CIL 中实现 SOTA。

**[Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)**

:   RLD 通过 Sample Confidence（样本置信度）和 Masked Correlation（掩码相关性）两种精炼知识，在不破坏类别相关性的前提下修正教师错误预测的负面影响，在 CIFAR-100 和 ImageNet 上全面超越现有 logit 蒸馏方法。

**[Learned Image Compression with Hierarchical Progressive Context Modeling](learned_image_compression_with_hierarchical_progressive_context_modeling.md)**

:   提出分层渐进上下文模型 (HPCM)，通过将 latent 划分为多尺度子表征并从小到大依次编码，结合跨编码步的渐进上下文融合机制（基于交叉注意力），实现更高效的远程依赖建模和更准确的熵参数估计，在图像压缩性能和计算复杂度之间取得更好的平衡。

**[Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)**

:   本文提出局部稠密关系 logit 蒸馏（LDRLD），通过递归解耦和重组 logit 知识来捕获细粒度的类间关系，结合自适应衰减权重（ADW）策略对关键类别对赋予更高权重，在 CIFAR-100、ImageNet-1K 和 Tiny-ImageNet 上持续优于现有 logit 蒸馏 SOTA。

**[MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)**

:   提出 MixA-Q，一种混合精度激活量化框架，将窗口级激活稀疏性（原本用于剪枝）转化为量化维度的利用——对不重要的窗口分配更低比特宽度而非完全跳过计算，在 COCO 目标检测上实现 PTQ 无损 1.35× 加速和 QAT 无损 1.25× 加速，同时具有更好的 OOD 鲁棒性。

**[MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)**

:   提出 MotionFollower，通过两个轻量卷积控制器（姿态+外观）和基于分数函数正则化的一致性引导机制，实现视频运动编辑，在 GPU 显存消耗减少约 80% 的同时超越 MotionEditor 等强基线。

**[MSQ: Memory-Efficient Bit Sparsification Quantization](msq_memory-efficient_bit_sparsification_quantization.md)**

:   提出MSQ，通过RoundClamp量化器从权重直接计算最低有效位(LSB)并施加L1正则化诱导稀疏性，无需显式创建bit-level可训练参数即可实现混合精度量化发现，训练参数减少8倍、训练时间减少86%，同时保持竞争性的精度-压缩权衡。

**[Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)**

:   MoSketch 首次解决多物体草图动画问题，通过 LLM 场景分解 + LLM 运动规划 + 运动精炼网络 + 组合式 SDS 四个模块，以分治策略处理物体感知运动建模和复杂运动优化两大挑战，无需任何训练数据实现高质量多物体草图动画。

**[OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)**

:   首个面向 Vision Mamba 模型（VMM）的无数据后训练量化框架，通过增强隐式注意力生成高质量合成数据，并结合动态异常值检测的混合精度量化方案，在 W4A4 设置下显著超越现有数据驱动 PTQ 方法。

**[Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration](partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)**

:   提出 Partial Forward Blocking (PFB)，在前向传播的浅层阶段计算样本重要性并剪枝，阻断被剪枝样本的后续深层前向传播，实现 ImageNet 上 40% 剪枝下 0.5% 精度提升 + 33% 训练时间缩减。

**[Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)**

:   提出PAT（Perspective-Aware Teaching）框架，通过区域感知注意力（RAA）解决异构架构间的视角不匹配问题，通过自适应反馈提示（AFP）解决教师无感知问题，使得特征级蒸馏首次在异构知识蒸馏场景中全面超越logits级方法。

**[PLAN: Proactive Low-Rank Allocation for Continual Learning](plan_proactive_low-rank_allocation_for_continual_learning.md)**

:   提出 PLAN 框架，通过为每个任务前瞻性地分配正交低秩子空间并使用扰动策略最小化任务间干扰，在持续学习场景下实现了高效且无遗忘的大模型微调，在标准 CL 基准上建立了新的 SOTA。

**[Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)**

:   提出 Representation Shift，一种无需训练、模型无关的 token 重要性度量方法，通过计算 token 在网络层前后的表征变化量来衡量重要性，从而首次实现 token 压缩与 FlashAttention 的兼容，在视频理解和图像分类上取得高达 5.5× 的加速。

**[SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)**

:   提出 SAMO，一种轻量级锐度感知多任务优化方法，通过全局-局部联合扰动缓解任务梯度冲突，并利用零阶梯度近似和层级归一化大幅降低计算开销。

**[Scheduling Weight Transitions for Quantization-Aware Training](scheduling_weight_transitions_for_quantization-aware_training.md)**

:   指出传统学习率调度对量化感知训练（QAT）中量化权重的有效步长控制失效，提出转换率（Transition Rate）调度技术，通过自适应学习率（TALR）显式控制量化权重的离散跳变次数，显著提升低比特量化模型性能。

**[Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)**

:   提出 SSD（Soft Separation and Distillation）框架，通过维度缩放正则化（DSR）和投影器蒸馏（PD）两个模块，改善联邦无监督学习中跨客户端的全局表征均匀性（inter-client uniformity），在不增加通信开销的前提下显著提升表征质量和下游任务性能。

**[SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)**

:   提出 Sign-Splitting Vector Quantization (SSVQ)，将权重的符号位与码本解耦，引入可学习符号位和增强的迭代冻结策略，使 VQ 微调时每个量化权重可以沿各自梯度方向独立更新，在极端压缩率下显著优于传统 VQ 和标量量化。

**[StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data](stolenlora_exploring_lora_extraction_attacks_via_synthetic_data.md)**

:   StolenLoRA 首次提出针对 LoRA 自适应模型的模型提取攻击方向，利用 LLM 驱动的 Stable Diffusion 生成高质量合成数据替代真实数据集搜索，并设计基于分歧的半监督学习（DSL）策略通过选择性查询最大化信息增益，仅需 10k 次查询即可达到高达 96.60% 的攻击成功率，揭示了 LoRA 适配模型的严重安全漏洞。

**[Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)**

:   本文提出对任务向量（fine-tuned 与 pre-trained 权重之差）而非 fine-tuned 权重本身进行量化，利用任务向量更窄的数值范围实现低至 3-bit 的量化而不损失精度；进一步提出残差任务向量量化（RTVQ），将任务向量分解为共享高精度基向量和低精度偏移量，在仅用 8% 原始存储的情况下维持甚至提升模型合并性能。

**[Time-Aware Auto White Balance in Mobile Photography](time-aware_auto_white_balance_in_mobile_photography.md)**

:   本文提出一种利用手机上下文元数据（时间戳和地理位置）辅助图像颜色信息的轻量化光照估计方法（约 5K 参数），在自建 3224 张智能手机数据集上达到或超过大模型性能，且可在旗舰手机 DSP 上 0.25ms 内完成推理。

**[TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)**

:   提出 TR-PTS 框架，通过 Fisher 信息矩阵进行任务驱动的逐层参数选择，同时利用 CLS 注意力分数动态筛选/合并 token，在仅微调 0.34%-0.60% 参数的情况下超越全量微调 3.40%（FGVC）和 10.35%（VTAB）。

**[UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale](uniconvnet_expanding_effective_receptive_field_while_maintaining_asymptotically_.md)**

:   提出UniConvNet，通过合理组合较小卷积核（7×7, 9×9, 11×11）的三层感受野聚合器（RFA），在扩大有效感受野（ERF）的同时保持其渐近高斯分布（AGD），从而在轻量级到大规模模型上全面超越现有CNN和ViT。

**[Variance-Based Pruning for Accelerating and Compressing Trained Networks](variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)**

:   提出基于方差的一次性结构化剪枝方法（VBP），通过移除MLP隐藏层中方差最小的神经元，并将其均值激活补偿到下一层偏置中，以极少微调（10 epoch）即可恢复99%原始精度，同时减少35%计算量和36%参数。

**[ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)**

:   提出 ViT-Linearizer，一种跨架构蒸馏框架，通过**激活匹配**和**掩码预测**两个核心机制，将 ViT 自注意力中学习到的"二次知识"高效迁移到线性复杂度的循环模型（Mamba-based Adventurer），在 ImageNet 上达到 84.3% 准确率，同时在高分辨率任务中实现最高 4.2× 的推理加速。

**[VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)**

:   > 提出 VQ-SGen，将每个笔画视为独立实体并解耦其形状与位置信息，通过向量量化（VQ）构建紧凑离散的笔画码本，再用级联自回归 Transformer 逐步生成笔画的语义标签、形状和位置，在 CreativeSketch 数据集上显著超越现有方法。

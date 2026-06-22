---
title: >-
  ICLR2026 VLMEfficiency论文汇总 · 18篇论文解读
description: >-
  18篇ICLR2026的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩、压缩/编码、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
  - "压缩/编码"
  - "LLM"
item_list:
  - u: "enhancing_visual_token_representations_for_video_large_language_models_via_train/"
    t: "Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding"
  - u: "hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p/"
    t: "HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit"
  - u: "illava_an_image_is_worth_fewer_than_13_input_tokens_in_large_multimodal_models/"
    t: "iLLaVA: An Image is Worth Fewer Than 1/3 Input Tokens in Large Multimodal Models"
  - u: "ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr/"
    t: "IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning"
  - u: "learnpruner_rethinking_attention-based_token_pruning_in_vision_language_models/"
    t: "LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models"
  - u: "lightweight_spatio-temporal_modeling_via_temporally_shifted_distillation_for_rea/"
    t: "Lightweight Spatio-Temporal Modeling via Temporally Shifted Distillation for Real-Time Accident Anticipation"
  - u: "mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_/"
    t: "Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models"
  - u: "nüwa_mending_the_spatial_integrity_torn_by_vlm_token_pruning/"
    t: "Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning"
  - u: "photon_speedup_volume_understanding_with_efficient_multimodal_large_language_mod/"
    t: "Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models"
  - u: "ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_/"
    t: "PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models"
  - u: "prune_redundancy_preserve_essence_vision_token_compression_in_vlms_via_synergist/"
    t: "Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity"
  - u: "sp-vla_a_joint_model_scheduling_and_token_pruning_approach_for_vla_model_acceler/"
    t: "SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration"
  - u: "st-simdiff_balancing_spatiotemporal_similarity_and_difference_for_efficient_vide/"
    t: "ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs"
  - u: "surge_surprise-guided_token_reduction_for_efficient_video_understanding_with_vlm/"
    t: "SURGE: Surprise-Guided Token Reduction for Efficient Video Understanding with VLMs"
  - u: "task-related_token_compression_in_multimodal_large_language_models_from_an_expla/"
    t: "Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective"
  - u: "tiny_but_mighty_a_software-hardware_co-design_approach_for_efficient_multimodal_/"
    t: "Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices"
  - u: "visiontrim_unified_vision_token_compression_for_training-free_mllm_acceleration/"
    t: "VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration"
  - u: "vq-transplant_efficient_vq-module_integration_for_pre-trained_visual_tokenizers/"
    t: "VQ-Transplant: Efficient VQ-Module Integration for Pre-trained Visual Tokenizers"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🔬 ICLR2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×8 · 模型压缩 ×6 · 压缩/编码 ×5 · LLM ×4

**[Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding](enhancing_visual_token_representations_for_video_large_language_models_via_train.md)**

:   针对视频大语言模型把成千上万视觉 token 压缩进有限上下文时丢失时空信息的问题，提出训练无关的 ST-GridPool：用「金字塔时序网格化」在不同时间尺度上聚合帧 token 注入多粒度运动信息，再用「基于范数的空间池化」依据 token 的 L2 范数加权保留高信息量区域，在 LLaVA-Video / LLaVA-OneVision 上即插即用、不需重训就稳定涨点。

**[HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)**

:   提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

**[iLLaVA: An Image is Worth Fewer Than 1/3 Input Tokens in Large Multimodal Models](illava_an_image_is_worth_fewer_than_13_input_tokens_in_large_multimodal_models.md)**

:   iLLaVA 跳出"只在 LLM 阶段压缩 token"的惯性，把 token 合并同时插进**图像编码器**和 **LLM** 两个阶段，并用"信息 token + 回收 token"的合并策略把被丢弃 token 的有用信息收回来，训练-free 实现端到端 2× 吞吐、4× prefilling 加速且保持 >95% 性能。

**[IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)**

:   揭示了LVLM中RoPE位置编码隐式建立的视觉坐标系统（IVC tokens），提出一种训练免的、提示感知的视觉token剪枝策略，在保留IVC tokens和语义前景token的同时，削减约50%视觉token并维持≥99%原始性能。

**[LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models](learnpruner_rethinking_attention-based_token_pruning_in_vision_language_models.md)**

:   LearnPruner 通过实证拆穿了"attention 分数 = token 重要性"这一通行假设，指出视觉编码器的 [CLS] attention 被 attention sink 污染、而 LLM 中只有"文本→视觉"的中层注意力才可靠，进而用一个可学习剪枝模块替代 [CLS] attention、再叠加 LLM 中层的文本引导剪枝，仅保留约 5.5% 视觉 token 即可维持 95% 性能并取得 3.2× 加速。

**[Lightweight Spatio-Temporal Modeling via Temporally Shifted Distillation for Real-Time Accident Anticipation](lightweight_spatio-temporal_modeling_via_temporally_shifted_distillation_for_rea.md)**

:   用一个**冻结的纯图像 CLIP 教师 + 时间偏移蒸馏**，让轻量 RepMixer+RWKV 学生在不做大规模视频预训练的前提下学到"预测未来帧"的时序能力，在 DAD/CCD 事故预测基准上达到 SOTA，且模型比对手小 3–7×、能在 Jetson Orin Nano 上 80 FPS 实时跑。

**[Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)**

:   发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

**[Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning](nüwa_mending_the_spatial_integrity_torn_by_vlm_token_pruning.md)**

:   本文发现现有视觉 token 剪枝方法之所以在视觉定位（visual grounding, VG）任务上崩盘，是因为它们破坏了由位置编码构建的"全局空间参考系"，于是提出 Nüwa——一个受群体智能（Boids）启发的两阶段剪枝框架，先在视觉编码器侧用"分区-对齐-聚合"保住空间锚点、再在 LLM 中段做文本引导的精筛，把 VG 任务的性能保持率从 ~7% 拉到 47%，同时 VQA 维持在 95%。

**[Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models](photon_speedup_volume_understanding_with_efficient_multimodal_large_language_mod.md)**

:   Photon 是一个直接吃整段 3D 医学体数据（CT/MRI）的多模态大模型，用「指令条件 Token 调度（ITS）」按每个问题自适应地决定保留多少视觉 token，再用「代理梯度传播（SGP）」让离散丢 token 这件事在训练时仍然可微，从而在医学视觉问答上同时拿到 SOTA 精度、约 5 倍训练加速和约三分之二的显存节省。

**[PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)**

:   提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息。PPE是零参数、即插即用的通用算子，在55%压缩率下图像任务平均仅降3.6%、在90%压缩率下通过级联压缩仍保持可比性能。

**[Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity](prune_redundancy_preserve_essence_vision_token_compression_in_vlms_via_synergist.md)**

:   PRUNESID 是一个训练免训（training-free）的视觉 token 压缩框架，用「语义主成分聚类（PSCA）+ 组内非极大值抑制（NMS）」两阶段流水线同时兼顾 token 的语义重要性和信息多样性，并按图像复杂度动态分配 token 预算，在 LLaVA-1.5 上只保留 11.1% token 就拿到 96.3% 的相对精度，在 LLaVA-NeXT 极端压缩（5.6%）下仍保 92.8%。

**[SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration](sp-vla_a_joint_model_scheduling_and_token_pruning_approach_for_vla_model_acceler.md)**

:   SP-VLA 将 VLA 动作序列分为"深思型"与"直觉型"两类，深思型调用大模型、直觉型用轻量 Ridge Regression 近似，同时对 token 做空间-语义双感知剪枝，在 LIBERO 实现 1.5× 无损加速、SimplerEnv 实现 2.4× 加速。

**[ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs](st-simdiff_balancing_spatiotemporal_similarity_and_difference_for_efficient_vide.md)**

:   针对多模态大模型处理长视频时视觉 token 爆炸的问题，本文提出训练无关框架 ST-SimDiff：把所有视觉 token 建成一张时空图，并行地用「相似度」做社区检测保留代表 token、用「差异」检测时间轴上的突变点保留事件 token，最后按注意力做预算微调；在 30%/50% token 预算下全面超过 FastV、FrameFusion 等 SOTA，且部分基准甚至追平 100% token 的原模型。

**[SURGE: Surprise-Guided Token Reduction for Efficient Video Understanding with VLMs](surge_surprise-guided_token_reduction_for_efficient_video_understanding_with_vlm.md)**

:   SURGE 用「token 在时间上是否可预测」来度量惊奇度（surprise）——可预测的冗余 token 被裁掉、不可预测的新信息 token 被保留，免训练、不挑骨干网络，在五个视频理解基准上把 token 数压到原来的 1/7、prefill 成本砍掉 86–98%，精度却与全 token 基线相差不超过 ±1 分。

**[Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective](task-related_token_compression_in_multimodal_large_language_models_from_an_expla.md)**

:   这篇论文用 Transformer 可解释性方法估计视觉 token 对当前指令的任务相关性，并训练一个轻量卷积压缩器在 LLM 输入端提前剪掉低相关 token，从而在 Qwen2-VL、LLaVA-OneVision 和 VILA1.5 上显著减少 FLOPs、prefill 时间与 KV-cache，同时尽量保持图像和视频理解性能。

**[Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices](tiny_but_mighty_a_software-hardware_co-design_approach_for_efficient_multimodal_.md)**

:   本文提出 NANOMIND，把大型多模态模型（LMM）拆成视觉、投影、语言、音频四个独立"积木"，按各加速器（NPU/GPU/CPU）所长分别调度，并在统一内存上用零拷贝缓冲管理器（TABM）传递 embedding，配合自研硬件、低比特融合 GEMM 内核和电量感知调度，让一台 2000 mAh 电池供电的小设备完全离线跑多模态推理，端到端能耗比主流边缘框架降低 42.3%，低功耗模式下可续航近 18.8 小时。

**[VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration](visiontrim_unified_vision_token_compression_for_training-free_mllm_acceleration.md)**

:   VisionTrim 是一个免训练的多模态大模型（MLLM）加速框架，用 DVTS（兼顾全局语义和局部空间连续性挑选主导视觉 token）+ TGVC（用文本引导把被丢弃的 token 聚类合并成补充 token）两个即插即用模块，在视觉编码和 LLM 解码两个阶段同时压缩视觉 token，在 LLaVA-1.5 上砍掉 88.9% 视觉 token 仍能保持 98.8% 的平均性能。

**[VQ-Transplant: Efficient VQ-Module Integration for Pre-trained Visual Tokenizers](vq-transplant_efficient_vq-module_integration_for_pre-trained_visual_tokenizers.md)**

:   VQ-Transplant 把预训练视觉 tokenizer 的 encoder-decoder 固定住，只替换并轻量适配 VQ 模块，使新量化算法能以约 22 小时训练成本接入 VAR 这类强 tokenizer，同时用 MMD-VQ 在 ImageNet-1K 上达到 0.81 r-FID，超过原始 VAR tokenizer 的 0.92 r-FID。

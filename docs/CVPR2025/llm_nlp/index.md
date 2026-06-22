---
title: >-
  CVPR2025 LLM其他论文汇总 · 15篇论文解读
description: >-
  15篇CVPR2025的 LLM 其他方向论文解读，涵盖对齐/RLHF、对话系统、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "对话系统"
  - "对抗鲁棒"
item_list:
  - u: "building_vision_models_upon_heat_conduction/"
    t: "Building Vision Models upon Heat Conduction"
  - u: "chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment/"
    t: "Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment"
  - u: "comrope_rotary_position/"
    t: "ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices"
  - u: "dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders/"
    t: "Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders"
  - u: "exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent/"
    t: "Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention"
  - u: "imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy/"
    t: "Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy"
  - u: "learning_textual_prompts_for_open-world_semi-supervised_learning/"
    t: "Learning Textual Prompts for Open-World Semi-Supervised Learning"
  - u: "making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res/"
    t: "Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration"
  - u: "mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_/"
    t: "MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities"
  - u: "rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu/"
    t: "Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers"
  - u: "spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo/"
    t: "Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer"
  - u: "spiking_transformer_with_spatial-temporal_attention/"
    t: "Spiking Transformer with Spatial-Temporal Attention"
  - u: "staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks/"
    t: "STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks"
  - u: "test-time_visual_in-context_tuning/"
    t: "Test-Time Visual In-Context Tuning"
  - u: "the_change_you_want_to_detect_semantic_change_detection_in_earth_observation_wit/"
    t: "The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**📷 CVPR2025** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (55)](../../ICLR2026/llm_nlp/index.md) · [💬 ACL2026 (61)](../../ACL2026/llm_nlp/index.md) · [🧪 ICML2026 (39)](../../ICML2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md)

**[Building Vision Models upon Heat Conduction](building_vision_models_upon_heat_conduction.md)**

:   提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

**[Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)**

:   本文提出基于对话的行人检索（ChatPR）新范式，构建了首个对话-图像配对数据集ChatPedes，并设计了DiaNA框架通过自适应属性精炼器实现对话与图像间的细粒度跨模态对齐，显著优于传统单句文本检索方法。

**[ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices](comrope_rotary_position.md)**

:   本文提出ComRoPE，通过将RoPE推广为由可训练交换角矩阵参数化的旋转位置编码，理论证明了角矩阵的成对交换性是RoPE满足相对位置依赖性的充要条件，在ImageNet-1K上比SOTA方法LieRE提升1.6%（训练分辨率）和2.9%（更高分辨率）。

**[Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders](dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)**

:   本文提出Exposure-slot框架，将Slot Attention算法扩展为层次化的slot-in-slot结构，通过可学习的曝光prompt引导特征聚类，实现以曝光为中心的区域感知表征学习，在欠曝/过曝图像矫正任务上取得SOTA性能。

**[Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

**[Learning Textual Prompts for Open-World Semi-Supervised Learning](learning_textual_prompts_for_open-world_semi-supervised_learning.md)**

:   本文提出了一种针对开放世界半监督学习（OWSSL）的新方法，通过全局-局部文本提示学习策略增强图文对齐效果，并设计前向-反向策略降低无标签样本中图文匹配的噪声，在多个细粒度数据集上显著超越SOTA。

**[Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration](making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res.md)**

:   本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。

**[MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)**

:   MG-MotionLLM 提出了一个统一的多粒度动作-语言模型，通过 Motion VQ-VAE + T5 语言模型的架构和精心设计的多粒度协同预训练方案（含 28 种任务），同时支持粗粒度和细粒度的动作理解与生成，在经典任务上达到 SOTA 的同时开启了细粒度动作编辑等新应用。

**[Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)**

:   本文深入分析了点积在脉冲查询-键对中因大量"非脉冲事件"导致相似度度量失效的根本原因，提出专为脉冲序列设计的a-XNOR相似度度量，将非脉冲对的相关性重定义为特定值a，在多种脉冲Transformer架构和数据集上显著提升性能。

**[Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)**

:   本文提出 Accurate Addition-Only Spiking Self-Attention（A²OS²A），通过融合二值、ReLU 和三值脉冲神经元的混合策略，在保持纯加法计算（无乘法）的前提下显著提升脉冲Transformer精度，ImageNet-1K 上达到 78.66%。

**[Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)**

:   将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

**[STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   通过在SNN中集成全局上下文自注意(GC)、位置编码(PE)、步骤注意(SA)和时间步随机退出(TSRD)四大模块，STAA-SNN在CIFAR-10/100和ImageNet上达到97.14%/82.05%/70.40%的SNN SOTA性能。

**[Test-Time Visual In-Context Tuning](test-time_visual_in-context_tuning.md)**

:   本文提出VICT（Visual In-Context Tuning），通过翻转任务提示和测试样本的角色并利用循环一致性损失，在测试时对视觉上下文学习模型（如Painter）进行单样本自适应，显著提升其在分布偏移下的泛化能力。

**[The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation](the_change_you_want_to_detect_semantic_change_detection_in_earth_observation_wit.md)**

:   本文提出HySCDG（Hybrid Semantic Change Detection Data Generation），一种混合数据生成流水线，结合真实超高分辨率（VHR）遥感影像和图像inpainting技术生成大规模语义变化检测训练数据，在简洁的架构设计下实现了强大的时间和空间泛化能力。

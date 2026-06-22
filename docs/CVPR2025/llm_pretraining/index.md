---
title: >-
  CVPR2025 预训练论文汇总 · 15篇论文解读
description: >-
  15篇CVPR2025的预训练方向论文解读，涵盖医学影像、三维重建、人脸/视线、水印/隐写、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "三维重建"
  - "人脸/视线"
  - "水印/隐写"
  - "对抗鲁棒"
item_list:
  - u: "a_unified_framework_for_heterogeneous_semi-supervised_learning/"
    t: "A Unified Framework for Heterogeneous Semi-supervised Learning"
  - u: "amo_sampler_enhancing_text_rendering_with_overshooting/"
    t: "AMO Sampler: Enhancing Text Rendering with Overshooting"
  - u: "bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior/"
    t: "Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior"
  - u: "context-cir_learning_from_concepts_in_text_for_composed_image_retrieval/"
    t: "ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval"
  - u: "dreamtext_high_fidelity_scene_text_synthesis/"
    t: "DreamText: High Fidelity Scene Text Synthesis"
  - u: "exploration-driven_generative_interactive_environments/"
    t: "Exploration-Driven Generative Interactive Environments"
  - u: "improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio/"
    t: "Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction"
  - u: "influence_malleability_in_linearized_attention_dual_implications_of_non-converge/"
    t: "Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics"
  - u: "mr_plip_multi_resolution_pathology/"
    t: "MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation"
  - u: "planarsplatting_accurate_planar_surface_reconstruction_in_3_minutes/"
    t: "PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes"
  - u: "precise_event_spotting_in_sports_videos_solving_long-range_dependency_and_class_/"
    t: "Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance"
  - u: "robust_message_embedding_via_attention_flow-based_steganography/"
    t: "Robust Message Embedding via Attention Flow-Based Steganography"
  - u: "scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model/"
    t: "ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model"
  - u: "seeing_what_matters_empowering_clip_with_patch_generation-to-selection/"
    t: "Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection"
  - u: "the_scene_language_representing_scenes_with_programs_words_and_embeddings/"
    t: "The Scene Language: Representing Scenes with Programs, Words, and Embeddings"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**📷 CVPR2025** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md)

**[A Unified Framework for Heterogeneous Semi-supervised Learning](a_unified_framework_for_heterogeneous_semi-supervised_learning.md)**

:   提出异构半监督学习(HSSL)新问题设定——标记数据和无标记数据来自不同分布的域，目标是训练能在两个域上都泛化的模型；通过将C类问题扩展为2C类分类（每个域的同一语义类视为不同类），结合WMA伪标签、跨域原型对齐和渐进式跨域Mixup三个组件统一解决。

**[AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)**

:   提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。

**[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)**

:   首次提出"系统差距"（System GAP）和"随机差距"（Random GAP）的概念来描述脑信号与视觉刺激之间的信息不匹配，通过不确定性感知的模糊先验（UBP）动态调整图像模糊程度来缓解训练中的过拟合，在 200-way 零样本脑-图像检索任务上实现 50.9% top-1 准确率，超越前 SOTA 13.7 个百分点。

**[ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)**

:   提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

**[DreamText: High Fidelity Scene Text Synthesis](dreamtext_high_fidelity_scene_text_synthesis.md)**

:   DreamText重构扩散模型训练流程，引入字符级别的均衡监督(balanced supervision)和启发式交替优化策略来校正字符注意力，结合文本编码器与生成器的联合训练学习多样化字体风格，在场景文字合成任务上大幅超越SOTA方法（SeqAcc从UDiffText的0.763提升至0.940）。

**[Exploration-Driven Generative Interactive Environments](exploration-driven_generative_interactive_environments.md)**

:   开源实现 Genie 世界模型（GenieRedux），增加真实动作条件、Token 距离交叉熵（TDCE）损失和 token 跳连得到 GenieRedux-G，并提出 AutoExplore 探索智能体用世界模型的 token 预测不确定性作为内在奖励驱动多样数据收集，将仿真质量提升高达 7.4 PSNR。

**[Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)**

:   提出 IAR，通过平衡 K-means 重排 VQGAN 码本使相似 embedding 具有相邻索引，配合簇导向交叉熵损失引导模型正确预测目标 token 所在的语义簇，在 LlamaGen 100M-1.4B 各规模上将训练时间减半且提升生成质量。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)**

:   提出 MR-PLIP，首个多分辨率病理学视觉语言预训练模型，在 TCGA 数据集的 3400 万张多分辨率图文对上预训练，通过跨分辨率视觉-文本对齐和文本引导视觉表示，在 26 个数据集上超越 SOTA。

**[PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes](planarsplatting_accurate_planar_surface_reconstruction_in_3_minutes.md)**

:   本文提出 PlanarSplatting，通过直接优化可学习的 3D 矩形平面基元，利用新设计的矩形 splatting 函数将平面可微地渲染为深度和法线图，仅需 3 分钟即可从多视角图像重建精确的室内平面场景，无需任何平面标注。

**[Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance](precise_event_spotting_in_sports_videos_solving_long-range_dependency_and_class_.md)**

:   提出端到端可训练的精确事件定位框架，通过自适应时空精炼模块（ASTRM）增强特征的时空信息，并引入Soft Instance Contrastive（SoftIC）损失解决类别不平衡问题，在SoccerNet V2 tight设置上以73.74 mAP超越SOTA。

**[Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)**

:   本文提出RMSteg（Robust Message Steganography）框架，首次将Transformer注意力机制集成到归一化流网络中（AttnFlow），配合可逆QR码转换和可逆Token融合模块，实现了高质量、高容量且鲁棒的消息-图像隐写，隐写图像即使经过打印-拍照等极端扭曲仍可准确解码。

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   首次在人类动作生成领域系统验证缩放律，提出包含Motion FSQ-VAE（解决codebook collapse）、260小时MotionUnion数据集和文本前缀自回归Transformer的可扩展系统ScaMo，发现归一化测试损失与FLOPs的对数律以及词汇参数/模型参数/数据量与FLOPs的幂律关系，并在$1\times 10^{18}$FLOPs预算下成功预测最优配置。

**[Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)**

:   提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。

**[The Scene Language: Representing Scenes with Programs, Words, and Embeddings](the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)**

:   提出 Scene Language——一种用程序（P, 编码层级结构）+ 词语（W, 语义类别）+ 嵌入（Z, 视觉身份）三元组 $\Phi(s)=(W,P,Z)$ 表示视觉场景的新范式，通过 Claude 3.5 Sonnet 的 training-free 推理从文本/图像输入生成场景表示，支持传统/神经/混合渲染，在 3D/4D 场景生成质量和可控编辑上超越场景图等现有表示。

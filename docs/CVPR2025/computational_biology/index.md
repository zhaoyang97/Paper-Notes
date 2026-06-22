---
title: >-
  CVPR2025 计算生物论文汇总 · 7篇论文解读
description: >-
  7篇CVPR2025的计算生物方向论文解读，涵盖扩散模型、多模态、生物分子等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "多模态"
  - "生物分子"
item_list:
  - u: "diffvsgg_diffusion-driven_online_video_scene_graph_generation/"
    t: "DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation"
  - u: "multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_/"
    t: "Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation"
  - u: "semantic_and_expressive_variations_in_image_captions_across_languages/"
    t: "Semantic and Expressive Variation in Image Captions Across Languages"
  - u: "shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica/"
    t: "SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules"
  - u: "synthetic_visual_genome/"
    t: "Synthetic Visual Genome"
  - u: "towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos/"
    t: "Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos"
  - u: "unsupervised_foundation_model-agnostic_slide-level_representation_learning/"
    t: "Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**📷 CVPR2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (21)](../../CVPR2026/computational_biology/index.md) · [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md)

**[DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)**

:   提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[Semantic and Expressive Variation in Image Captions Across Languages](semantic_and_expressive_variations_in_image_captions_across_languages.md)**

:   系统性证明了不同语言的图像描述在语义内容（对象、关系、属性）和表达方式（具象度、语调、真实性）上存在显著的分布差异，多语言描述集相比单语言提供更丰富的视觉信息（+46% 对象、+66.1% 关系、+66.8% 属性），为多语言数据训练视觉模型提供了实证支撑。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，利用图拉普拉斯算子的谱嵌入技术，从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群 $C_n$，在多个公开数据集上实现了接近原子分辨率的从头螺旋结构重建。

**[Synthetic Visual Genome](synthetic_visual_genome.md)**

:   提出**SVG**（Synthetic Visual Genome）数据引擎，通过GPT-4在已有人工标注基础上**补全缺失关系**（Stage 1）和**Robin自蒸馏+GPT-4编辑**（Stage 2/SG-Edit）两阶段管道，生成146K图像、2.6M物体、5.6M关系的密集场景图数据集，训练的**Robin-3B**模型仅用<3M实例即超越300M实例训练的同尺寸模型，在指代表达理解上达到88.9的SOTA。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning](unsupervised_foundation_model-agnostic_slide-level_representation_learning.md)**

:   提出 Cobra，一种无监督的基础模型无关 (FM-agnostic) 全切片图像 (WSI) 级别表征学习框架：将来自多个预训练 patch 级基础模型的嵌入作为特征空间增广，通过 Mamba-2 编码器和对比学习训练 slide 编码器，仅用 3048 张 WSI 预训练即在 15 个下游任务上平均 AUC 超过现有 slide 编码器至少 +4.4%。

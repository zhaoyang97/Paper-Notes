---
title: >-
  ICML2026 语义分割论文汇总 · 14篇论文解读
description: >-
  14篇ICML2026的语义分割方向论文解读，涵盖语义分割、目标跟踪、LLM、对齐/RLHF、语音、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "语义分割"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "LLM"
  - "对齐/RLHF"
  - "语音"
  - "对抗鲁棒"
item_list:
  - u: "activation-free_backbones_for_image_recognition_polynomial_alternatives_within_m/"
    t: "Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models"
  - u: "beyond_detection_a_structure-aware_framework_for_scene_text_tracking/"
    t: "Beyond Detection: A Structure-Aware Framework for Scene Text Tracking"
  - u: "flowseg_dynamic_semantic_guidance_for_llm-conditioned_segmentation/"
    t: "FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation"
  - u: "functional_attention_from_pairwise_affinities_to_functional_correspondences/"
    t: "Functional Attention: From Pairwise Affinities to Functional Correspondences"
  - u: "geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models/"
    t: "Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models"
  - u: "lightavseg_lightweight_audio-visual_segmentation/"
    t: "LightAVSeg: Lightweight Audio-Visual Segmentation"
  - u: "mvr-cache_optimizing_semantic_caching_via_multi-vector_retrieval_and_learned_pro/"
    t: "MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation"
  - u: "refining_context-entangled_content_segmentation_via_curriculum_selection_and_ant/"
    t: "Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion"
  - u: "segment_anything_with_robust_uncertainty-accuracy_correlation/"
    t: "Segment Anything with Robust Uncertainty-Accuracy Correlation"
  - u: "supervise_less_see_more_training-free_nuclear_instance_segmentation_with_prototy/"
    t: "SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting"
  - u: "towards_effective_waste_segmentation_for_automated_waste_recycling_in_cluttered_/"
    t: "Towards Effective Waste Segmentation for Automated Waste Recycling in Cluttered Background"
  - u: "uground_towards_unified_visual_grounding_with_unrolled_transformers/"
    t: "UGround: Towards Unified Visual Grounding with Unrolled Transformers"
  - u: "unsupervised_hierarchical_skill_discovery/"
    t: "无监督层级技能发现"
  - u: "what_makes_synthetic_data_effective_in_image_segmentation/"
    t: "What Makes Synthetic Data Effective in Image Segmentation"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🧪 ICML2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (122)](../../CVPR2026/segmentation/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/segmentation/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (45)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (73)](../../ICCV2025/segmentation/index.md) · [🧪 ICML2025 (18)](../../ICML2025/segmentation/index.md)

🔥 **高频主题：** 语义分割 ×7

**[Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models](activation-free_backbones_for_image_recognition_polynomial_alternatives_within_m.md)**

:   本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax，在 MetaFormer 风格骨干中无需常规激活函数也能在 ImageNet、鲁棒性和 ADE20K 分割上达到或超过激活式模型。

**[Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)**

:   提出 SymTrack，一个无需检测的双分支场景文字跟踪框架，通过预测性 Token 校正（PTR）解决透视畸变导致的特征瓶颈，跨专家校准（CEC）消除文字实例间的高视觉歧义，自适应推理引擎（AIE）稳定细粒度定位，在三个基准上大幅刷新 SOTA（最高 +12.32% AUC）。

**[FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation](flowseg_dynamic_semantic_guidance_for_llm-conditioned_segmentation.md)**

:   本文指出当前基于 query 的 LLM-conditioned 分割是"propose-then-select"——候选 mask 往往已经够准，错就错在选不对；为此提出 FlowSeg，让 LLM 条件嵌入在 decoder 每一层都参与 query refinement 并被新的视觉证据持续更新，再叠一个轻量边界细化模块，在 RefCOCO/+/g 和 ReasonSeg 上一致刷点。

**[Functional Attention: From Pairwise Affinities to Functional Correspondences](functional_attention_from_pairwise_affinities_to_functional_correspondences.md)**

:   本文把 Transformer 里的 softmax 注意力重新解释为"两个学得到的函数基之间的最小二乘线性算子"，借用形状匹配里的 functional maps 思想，把 $n\times n$ 的点对亲和矩阵压缩成 $k\times k$ 的紧致谱算子，在 PDE 求解、3D 点云分割和 OOD 推广上同时拿到 SOTA。

**[Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models.md)**

:   GPUA 把 CLIP 这种语义有余而局部精度不足的 VLM 和 DINOv3 这种细粒度足但缺语义的 VFM 看作两种"视觉语言"，用最优传输挖软对应再解正交 Procrustes 学一个保几何的线性映射，把 VFM 翻译进 VLM 空间——全程无监督、不更新任何预训练参数，零样本分类平均涨 11.8%。

**[LightAVSeg: Lightweight Audio-Visual Segmentation](lightavseg_lightweight_audio-visual_segmentation.md)**

:   LightAVSeg 通过解耦 "语义筛选 (what)" 和 "空间定位 (where)"，用全局通道调制替换 $\mathcal{O}(N^2)$ 的跨模态注意力，让 AVS 模型在 20.5M 参数下达到 50.4 mIoU (MS3)，并在 Snapdragon 8 Elite 上做到 163.4 ms 的端侧延迟，比 AVSegFormer-R50 快约 $8\times$。

**[MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation](mvr-cache_optimizing_semantic_caching_via_multi-vector_retrieval_and_learned_pro.md)**

:   MVR-cache 把 LLM 语义缓存的相似度度量从"单向量 cosine"升级为"可学习分段后的多向量 MaxSim"，并用 REINFORCE 训练一个轻量分段模型，在保证错误率上界 $\delta$ 不变的前提下把缓存命中率最多再抬 37%。

**[Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion](refining_context-entangled_content_segmentation_via_curriculum_selection_and_ant.md)**

:   CurriSeg 不动分割网络结构，只换训练计划：先用"时间损失统计 + 像素熵加权"的稳健课程把模型推到稳态，再用反课程的"频谱失明"微调（砍掉高频迫使模型读结构语义），就让 FEDER / FSEL / RUN 在 CHAMELEON / CAMO / COD10K / NC4K 等伪装/息肉分割基准上稳定涨 2–4%，零额外参数、训练时间还更短。

**[Segment Anything with Robust Uncertainty-Accuracy Correlation](segment_anything_with_robust_uncertainty-accuracy_correlation.md)**

:   针对 SAM 系列只输出 mask-level 单一置信度、在域漂移下出现"Mask-level Confidence Confusion"的问题，本文给 SAM2 接上 Weibull 双粒度贝叶斯 mask decoder 做像素级 epistemic 估计，并配以受人类视觉启发的 style + deformation 协同对抗扰动 + 校准损失，让 uncertainty 在 23 个 zero-shot 目标域始终与误差对齐，平均 J&F 达 79.87 同时不确定性图变得显著可信。

**[SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting](supervise_less_see_more_training-free_nuclear_instance_segmentation_with_prototy.md)**

:   SPROUT 是首个完全训练无关、零标注的病理核分割框架——用 H&E 染色先验在每张切片上自构高置信度前景/背景区域→提取原型→用部分最优传输（POT）做特征-原型软对齐→输出 SAM 的正/负点提示；在 MoNuSeg 等基准上 AJI 比训练方法高 8.2%。

**[Towards Effective Waste Segmentation for Automated Waste Recycling in Cluttered Background](towards_effective_waste_segmentation_for_automated_waste_recycling_in_cluttered_.md)**

:   针对自动垃圾回收里"杂乱背景 + 半透明/可变形废品 + 现有方法靠大 backbone 太重"的痛点，提出轻量分割网络 EWSegNet：用**空间域模块抓局部结构、频域模块抓全局上下文**级联互补，再加一个用**高斯差分 + 池化注意力**强化边界与团块（blob）的辅助增强模块 AFEM，在更小参数/更低延迟下达到与 SOTA 相当甚至更好的分割精度。

**[UGround: Towards Unified Visual Grounding with Unrolled Transformers](uground_towards_unified_visual_grounding_with_unrolled_transformers.md)**

:   UGround 把 LMM-based 视觉定位从"用最后一层 $\langle\text{SEG}\rangle$ token 当 prompt"的范式翻转为"用动态选中的中间层相似度图当 prompt"，通过强化学习策略 SSC 让 $\langle\text{SEG}\rangle$ 滑过所有 transformer 层、把相似度图同时当作 SAM 的软 logit mask 和反向监督信号，首次在单一框架内统一了 RES / RS / FP-RES / gRES / Multi-RS 五种视觉定位任务，并在 ReasonSeg test 上 cIoU +9.0%、gRefCOCO val N-acc +12.1%。

**[无监督层级技能发现](unsupervised_hierarchical_skill_discovery.md)**

:   HiSD 从无标签观测轨迹出发——通过最优传输进行技能分割，再用 Sequitur 文法诱导发现多层级技能层次，无需动作标签或奖励信号。

**[What Makes Synthetic Data Effective in Image Segmentation](what_makes_synthetic_data_effective_in_image_segmentation.md)**

:   这篇论文系统分析了合成图像对语义分割有效的两个关键因素：复杂场景组合和高实例保真度，并提出 SENSE 用最优传输稳定合成图像的伪标签分配，从而在 Cityscapes、COCO、ADE20K 上稳定提升 DPT 和 Mask2Former。

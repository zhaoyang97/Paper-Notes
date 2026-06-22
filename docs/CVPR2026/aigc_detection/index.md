---
title: >-
  CVPR2026 AIGC检测论文汇总 · 10篇论文解读
description: >-
  10篇CVPR2026的 AIGC 检测方向论文解读，涵盖图像修复、多模态、自监督学习、语音、推理、水印/隐写等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "图像修复"
  - "多模态"
  - "自监督学习"
  - "语音"
  - "推理"
  - "水印/隐写"
item_list:
  - u: "common_inpainted_objects_in-n-out_of_context/"
    t: "Common Inpainted Objects In-N-Out of Context"
  - u: "enabling_supervised_learning_of_generative_signatures_for_generalized_ai-generat/"
    t: "Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection"
  - u: "fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela/"
    t: "Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks"
  - u: "inconsistency-aware_multimodal_schrodinger_bridge_for_deepfake_localization/"
    t: "Inconsistency-aware Multimodal Schrodinger Bridge for Deepfake Localization"
  - u: "investigating_self-supervised_representations_for_audio-visual_deepfake_detectio/"
    t: "Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection"
  - u: "learning_forgery-aware_lip_representations_without_forgery_priors/"
    t: "Learning Forgery-Aware Lip Representations Without Forgery Priors"
  - u: "learning_where_to_look_and_how_to_judge_resolution-agnostic_image_quality_assess/"
    t: "Learning Where to Look and How to Judge: Resolution-agnostic Image Quality Assessment with Quality-aware Saliency"
  - u: "locate-then-examine_grounded_region_reasoning_improves_detection_of_ai-generated/"
    t: "Locate-Then-Examine: Grounded Region Reasoning Improves Detection of AI-Generated Images"
  - u: "nowa_null-space_optical_watermark_for_invisible_capture_fingerprinting_and_tampe/"
    t: "NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization"
  - u: "ppm-clip_probabilistic_prompt_modeling_for_generalizable_ai-generated_image_dete/"
    t: "PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**📷 CVPR2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

**[Common Inpainted Objects In-N-Out of Context](common_inpainted_objects_in-n-out_of_context.md)**

:   作者用扩散修复（Stable Diffusion inpainting）系统性地替换 COCO 图像里的物体，造出 9.7 万张「同一物体在情境内 / 情境外」的图，再用 72B 多模态大模型三模型共识标注「位置 / 尺寸 / 共现」三维上下文标签，构建出首个带上下文标注的修复假图数据集 COinCO，并演示了细粒度上下文分类、由情境反推物体、以及无需微调就能增强 SOTA 假图定位三个下游任务。

**[Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection](enabling_supervised_learning_of_generative_signatures_for_generalized_ai-generat.md)**

:   针对"AI生成图像里的生成痕迹没有干净配对、无法监督式提取"这个死结，本文用一个**随机变结构的图像重建器**在真实图上人工"造痕迹"、把重建残差当伪标签去训练一个**生成签名（GenSign）提取器**，再用 GenSign + RGB 双流分类器做检测，在四个 benchmark 上把跨模型泛化刷到 SOTA。

**[Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)**

:   定义"细粒度图像美学评估"新任务，构建含32,217张图像/10,028个系列的FGAesthetics基准，提出FGAesQ模型：通过差异保留Tokenization（DiffToken）+ 对比文本辅助对齐（CTAlign）+ 排序感知回归（RankReg）从相对排序中学习判别性审美评分，在细粒度场景准确率0.779的同时保持粗粒度SRCC 0.770。

**[Inconsistency-aware Multimodal Schrodinger Bridge for Deepfake Localization](inconsistency-aware_multimodal_schrodinger_bridge_for_deepfake_localization.md)**

:   IaMSB 把音视频深度伪造的「时间区间定位」重新表述成一个薛定谔桥（Schrödinger Bridge）生成问题——用桥的传输代价直接读出跨模态一致性分数，再据此把计算步数非对称地分配给更可疑的那个模态，从而在严格 IoU（AP@0.95）上比现有方法高 3~10%。

**[Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection](investigating_self-supervised_representations_for_audio-visual_deepfake_detectio.md)**

:   这是一篇系统性"调查"论文：作者把 12 个现成的自监督编码器（音频、视觉、音视频）冻结，只在上面训一层线性探针，从「检测有效性、可解释性、跨模态互补性」三个维度横向评测它们做音视频深度伪造检测的能力，发现"音频信息驱动"的表征泛化最好（BRAVEn 的视觉编码器拿到 SOTA），而真实世界数据的难点来自数据集本身的内在难度而非特征抓了浅层捷径。

**[Learning Forgery-Aware Lip Representations Without Forgery Priors](learning_forgery-aware_lip_representations_without_forgery_priors.md)**

:   针对说话人认证系统被个性化"说话人脸生成"(TFG)伪造攻破的问题，本文提出一个**只用真实视频训练、完全不依赖任何伪造样本**的检测器：靠真帧混合伪造 + 非对称对比 + 高斯正则把真实唇动特征压成一个紧致球面，把球外一切（伪造和冒名者）当离群点，在 8 种现代伪造、10 个 SOTA 对比下把错误率压低 10% 以上。

**[Learning Where to Look and How to Judge: Resolution-agnostic Image Quality Assessment with Quality-aware Saliency](learning_where_to_look_and_how_to_judge_resolution-agnostic_image_quality_assess.md)**

:   针对无参考图像质量评价（NR-IQA）"为迁就预训练分辨率而暴力 resize、跨分辨率不泛化、多数据集 MOS 尺度不一难联训、超高清算力爆炸"四大通病，本文提出 ReLIQS：在原分辨率及缩放变体上采样固定尺寸 patch 并用 CLIP 编码，用轻量"感知重要性估计器（PIE）"学出 IQA 专属显著性来挑出少量关键 patch，再用"潜在质量轴模块（LQAM）"把多尺度嵌入聚合成单一分数，在真实/合成/AIGC 多种失真与分辨率上以更低算力超过 CNN、CLIP、MLLM 系强基线。

**[Locate-Then-Examine: Grounded Region Reasoning Improves Detection of AI-Generated Images](locate-then-examine_grounded_region_reasoning_improves_detection_of_ai-generated.md)**

:   LTE 让视觉语言模型先"全局扫描定位可疑区域"再"放大裁剪复核给出最终判定"，把一次性分类升级为两阶段的区域接地（region-grounded）推理，并配套构建带框级标注与取证解释的 TRACE 数据集，在准确率、鲁棒性和可解释性上同时获得提升。

**[NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization](nowa_null-space_optical_watermark_for_invisible_capture_fingerprinting_and_tampe.md)**

:   在相机光圈处插入一块可学习的相位掩膜，把认证信号编码进成像算子的**零空间**（拍摄时完全不可见），再用一个保证测量一致性的零空间网络（NSN）重建高质量图像并锚定该水印；篡改会破坏零空间投影里的统计结构，从而在像素级被检测器定位，在 AIGC 编辑下 F1 超过 EditGuard（0.993 vs 0.97）且对未知伪造者天然不可伪造。

**[PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection](ppm-clip_probabilistic_prompt_modeling_for_generalizable_ai-generated_image_dete.md)**

:   PPM-CLIP 把"判别一条静态决策边界"的 AIGC 检测范式换成"生成式概率推理"——用归一化流为每张图生成一族自适应 prompt（多个假设），再对全部假设的余弦相似度取平均消噪做判定，并配一个频域引导的 patch 对比学习让 CLIP 编码器盯住高频伪造痕迹，在 Ojha / GenImage / DRCT 上的跨生成器泛化显著超过 SOTA。

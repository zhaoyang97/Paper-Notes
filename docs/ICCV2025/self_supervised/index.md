---
title: >-
  ICCV2025 自监督/表示学习论文汇总 · 13篇论文解读
description: >-
  13篇ICCV2025的自监督/表示学习方向论文解读，涵盖少样本学习、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "自监督学习"
item_list:
  - u: "a_hidden_stumbling_block_in_generalized_category_discovery_d/"
    t: "A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention"
  - u: "a_tokenlevel_text_image_foundation_model_for_document_unders/"
    t: "A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)"
  - u: "always_skip_attention/"
    t: "Always Skip Attention"
  - u: "cobl_toward_zero-shot_ordinal_layering_without_user_prompting/"
    t: "CObL: Toward Zero-Shot Ordinal Layering without User Prompting"
  - u: "from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat/"
    t: "From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations"
  - u: "generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f/"
    t: "Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery"
  - u: "improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers/"
    t: "Improving Large Vision and Language Models by Learning from a Panel of Peers"
  - u: "loftup_learning_a_coordinatebased_feature_upsampler_for_visi/"
    t: "LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models"
  - u: "manual-pa_learning_3d_part_assembly_from_instruction_diagrams/"
    t: "Manual-PA: Learning 3D Part Assembly from Instruction Diagrams"
  - u: "mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning/"
    t: "MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning"
  - u: "scaling_languagefree_visual_representation_learning/"
    t: "Scaling Language-Free Visual Representation Learning"
  - u: "to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie/"
    t: "To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models"
  - u: "wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction/"
    t: "WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction"
item_total: 13
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**📹 ICCV2025** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md)

**[A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)**

:   发现GCD中未标注数据（尤其是未知类别）的ViT注意力会分散到背景区域（distracted attention），提出Attention Focusing（AF）模块通过多尺度token重要性度量+自适应剪枝来纠正注意力，作为即插即用模块在SimGCD上最高带来15.4%的性能提升。

**[A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](a_tokenlevel_text_image_foundation_model_for_document_unders.md)**

:   提出首个 token 级别文本图像基础模型 TokenFD，通过在 2000 万图像、18 亿 BPE token-mask 对上进行 token 级视觉-语言对齐预训练，实现 image-as-text 语义能力，并基于此构建文档理解 MLLM TokenVL，在 OCRBench 上得分 860（8B 组最高），在 DocVQA 等十项 VQA 任务上平均提升 8.8%。

**[Always Skip Attention](always_skip_attention.md)**

:   本文从理论上证明了 Vision Transformer 中的自注意力机制是本质上病态的（ill-conditioned），在无 skip connection 时会导致训练崩溃，并提出 Token Graying（TG）方法通过改善输入 token 的条件数来进一步增强 ViT 的训练稳定性和性能。

**[CObL: Toward Zero-Shot Ordinal Layering without User Prompting](cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)**

:   本文提出 CObL，一种基于多个冻结 Stable Diffusion UNet 并行生成的架构，能在无需用户提示、不知物体数量的前提下，从单张图像推断出遮挡排序的物体层叠表示（每层一个 amodal 完整物体），并且仅用数千张合成桌面场景就能零样本泛化到真实世界照片。

**[From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations](from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat.md)**

:   从理论角度分析 MAE 如何学习图像中的空间相关性，推导出线性 MAE 的解析解，揭示了掩码比例和 patch 大小如何选择短距离和长距离空间特征，并将分析扩展到非线性 MAE，为实践中的超参数选择提供了理论指导。

**[Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)**

:   提出基于扩散模型的即时类别发现框架 DiffGRE，通过属性组合生成（ACG）合成包含虚拟类别信息的新样本、多样性驱动精炼（DDR）过滤低质量样本、半监督Leader编码（SLE）注入额外类别知识，在 6 个细粒度数据集上显著提升了已有 OCD 方法的性能（平均 ACC-ALL 提升 6.5%）。

**[Improving Large Vision and Language Models by Learning from a Panel of Peers](improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)**

:   提出 Panel-of-Peers (PoP) 学习框架，利用多个性能相近的 LVLM 互相生成候选答案、互相评分、构建偏好数据，并通过 SimPO 迭代自我改进，在 15 个基准上将平均分从 48% 提升至 57%，无需人工标注数据。

**[LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)**

:   提出LoftUp，通过坐标-cross-attention架构直接将低分辨率VFM特征映射到任意高分辨率，并用class-agnostic mask精炼+自蒸馏构建全分辨率伪GT进行训练，在6个下游任务上平均提升10-20%且在视频目标分割上提升近50%。

**[Manual-PA: Learning 3D Part Assembly from Instruction Diagrams](manual-pa_learning_3d_part_assembly_from_instruction_diagrams.md)**

:   提出 Manual-PA，一个基于 Transformer 的说明书引导 3D 零件组装框架：通过对比学习将 3D 零件与说明书步骤图对齐来推断组装顺序，再以学到的顺序作为位置编码的软引导进行 6DoF 位姿预测，在 PartNet 上显著超越现有方法。

**[MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)**

:   MoSiC 利用离线点跟踪器提取长程运动轨迹，通过基于最优传输（Sinkhorn-Knopp）的聚类机制在时间维度上传播聚类分配，从而在视频数据上学习空间-时间一致的稠密表征，仅用视频训练即可将 DINOv2 在多个图像/视频基准上提升 1%–6%。

**[Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)**

:   通过在MetaCLIP的20亿web图像上训练DINOv2/MAE系列模型（1B-7B参数），系统性地证明纯视觉自监督学习在模型和数据规模上展现优于CLIP的scaling behavior，5B+参数时在VQA平均性能上超越CLIP——包括传统认为需要语言监督的OCR/Chart任务。

**[To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)**

:   提出 PALM——一个用4个可解释参数（最大精度 $A_{\max}$、覆盖效率 $\delta$、初始学习偏移 $\alpha$、扩展性 $\beta$）描述主动学习轨迹的统一数学模型，能从有限标注数据预测完整学习曲线，实现主动学习策略的定量公平比较。

**[WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)**

:   > WIR3D 通过优化一组 3D Bézier 曲线参数，在 CLIP 中间层激活的空间引导下，从任意视角忠实表示 3D 形状的几何结构和视觉显著特征（包括纹理），实现稀疏但语义丰富的 3D 形状抽象。

---
title: >-
  ECCV2026 模型压缩论文汇总 · 11篇论文解读
description: >-
  11篇ECCV2026的模型压缩方向论文解读，涵盖知识蒸馏、模型压缩、对抗鲁棒、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "知识蒸馏"
  - "对抗鲁棒"
  - "对齐/RLHF"
item_list:
  - u: "condensing_large-scale_datasets_directly_with_minimal_information_loss/"
    t: "Condensing Large-Scale Datasets Directly with Minimal Information Loss"
  - u: "distill_on_a_diet_efficient_knowledge_distillation_via_learnable_data_pruning/"
    t: "Distill on a Diet: Efficient Knowledge Distillation via Learnable Data Pruning"
  - u: "distill_once_adapt_life-long_exploring_dataset_distillation_for_continual_test-t/"
    t: "Distill Once, Adapt Life-Long: Exploring Dataset Distillation for Continual Test-Time Adaptation"
  - u: "mambaraw_selective_state_space_modeling_for_efficient_4k_raw_image_reconstructio/"
    t: "MambaRaw: Selective State Space Modeling for Efficient 4K Raw Image Reconstruction"
  - u: "mixtta_low-rank_cross-channel_mixing_for_reliable_test-time_adaptation/"
    t: "MixTTA: Low-Rank Cross-Channel Mixing for Reliable Test-Time Adaptation"
  - u: "mlvc_multi-platform_learned_video_codec_for_real-world_deployment/"
    t: "MLVC: 面向真实部署的多平台学习式视频编解码器"
  - u: "on_the_vulnerability_of_parameter-level_defenses_to_model_merging/"
    t: "On the Vulnerability of Parameter-Level Defenses to Model Merging"
  - u: "raysup_ultra-light_universal_feature_upsampling_via_geometry-aware_ray_represent/"
    t: "RaysUp: Ultra-light Universal Feature Upsampling via Geometry-Aware Ray Representation"
  - u: "structural_assessment_for_understanding_and_guiding_dataset_distillation_in_disc/"
    t: "Structural Assessment for Understanding and Guiding Dataset Distillation in Discrete Token Space"
  - u: "structured_hyperedge_adaptation_for_parameter-efficient_fine-tuning_of_vision_tr/"
    t: "Structured Hyperedge Adaptation for Parameter-Efficient Fine-Tuning of Vision Transformers"
  - u: "textds_parameter-efficient_representation_alignment_for_scene_text_detection_und/"
    t: "TextDS: Parameter-Efficient Representation Alignment for Scene Text Detection under Distribution Shifts"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🎞️ ECCV2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (108)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (239)](../../ICLR2026/model_compression/index.md) · [💬 ACL2026 (59)](../../ACL2026/model_compression/index.md) · [🧪 ICML2026 (116)](../../ICML2026/model_compression/index.md) · [🤖 AAAI2026 (60)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/model_compression/index.md)

**[Condensing Large-Scale Datasets Directly with Minimal Information Loss](condensing_large-scale_datasets_directly_with_minimal_information_loss.md)**

:   本文指出主流大规模数据集蒸馏（SRe2L 系）的「数据→模型→图像」双重压缩过程会造成严重信息损失、并使蒸馏图像偏离真实分布从而拖垮 Relabel，提出 CIM：用一个可计算的「有效信息 gap」度量，直接在原图上最小化合成集与真实集的信息差，绕开昂贵的 recover 阶段，在 ImageNet-1K IPC=10 上以单卡 80 分钟拿到 48.7% Top-1（ResNet-18）。

**[Distill on a Diet: Efficient Knowledge Distillation via Learnable Data Pruning](distill_on_a_diet_efficient_knowledge_distillation_via_learnable_data_pruning.md)**

:   本文提出 IF-Beta 框架, 用影响力函数(IF)作为无需重训的样本重要性评分器, 配合可学习的 Beta 分布采样策略, 通过特征空间双层优化高效搜索最优剪枝子集, 使学生在更少数据和算力下超越全量数据蒸馏的性能.

**[Distill Once, Adapt Life-Long: Exploring Dataset Distillation for Continual Test-Time Adaptation](distill_once_adapt_life-long_exploring_dataset_distillation_for_continual_test-t.md)**

:   DO-ALL 在部署前用数据集蒸馏（DD）将源域压缩为一小组合成锚点（每个锚点包含合成样本、源模型软标签和潜在特征），测试时通过特征空间余弦相似度为每个目标样本匹配语义最近的锚点，用锚点回放、目标-锚点 MixUp 正则化和多层 MMD 特征对齐稳定模型更新，并辅以有害自适应混合机制按参数组梯度有害程度选择性回退到源模型初始化，以即插即用方式一致提升 EATA、RMT、ROID、ASR 等多种 CTTA 方法在 ImageNet-C 和 CCC 基准上的长期鲁棒性。

**[MambaRaw: Selective State Space Modeling for Efficient 4K Raw Image Reconstruction](mambaraw_selective_state_space_modeling_for_efficient_4k_raw_image_reconstructio.md)**

:   MambaRaw 将状态空间模型（SSM/Mamba）引入 JPEG 引导的元数据 RAW 图像重建框架的熵参数估计中，通过能量引导的分块选择性扫描（TileMambaBlock）与能量感知特征精炼（EAR）两个轻量模块，在 4K 分辨率下同时提升了重建质量（PSNR 提升 1.2–1.4 dB）并降低了编码延迟（约 9%）。

**[MixTTA: Low-Rank Cross-Channel Mixing for Reliable Test-Time Adaptation](mixtta_low-rank_cross-channel_mixing_for_reliable_test-time_adaptation.md)**

:   MixTTA 在归一化层的逐通道仿射变换基础上，插入一个低秩残差跨通道混合模块，使测试时自适应（TTA）能够纠正分布偏移引起的通道间相关结构变化，配合解耦投影和谱投影两个正则手段防止对角泄露与秩一坍缩，在标准与 wild TTA 场景下即插即用地提升 Tent / EATA / SAR / DeYO / ReCAP 五大基线的精度和稳定性。

**[MLVC: 面向真实部署的多平台学习式视频编解码器](mlvc_multi-platform_learned_video_codec_for_real-world_deployment.md)**

:   把熵编码所需的 scale 参数从「网络实时算出来」改成「通过 hyperprior 确定性地传出来」，让神经视频编解码器第一次能在 Apple / Intel / Qualcomm 等异构 NPU 上「A 端编码、B 端解码」而不崩溃，同时靠门控记忆、ReGLU、长期参考帧恢复等一系列改进把跨平台约束带来的码率损失补回来，在视频会议基准上相对硬件 HEVC 拿到 >70% 的 BD-rate(MOS) 提升、三平台平均 100 FPS。

**[On the Vulnerability of Parameter-Level Defenses to Model Merging](on_the_vulnerability_of_parameter-level_defenses_to_model_merging.md)**

:   本文揭穿了「用线性变换让模型不可合并」这类参数级防御的根本漏洞——被保护的任务向量比预训练锚点小 2~3 个数量级，于是提出 Anchor-Guided Attack（AGA）：把公开预训练模型当静态锚点，用最小二乘 + 匈牙利算法解析地还原出防御方藏起来的变换矩阵，几乎无损地恢复受保护模型；同时给出抗攻击的 Anchor-Repulsive Fine-tuning（ARF）防御作为补救。

**[RaysUp: Ultra-light Universal Feature Upsampling via Geometry-Aware Ray Representation](raysup_ultra-light_universal_feature_upsampling_via_geometry-aware_ray_represent.md)**

:   提出一个仅 0.14M 参数、同时任务无关和 VFM 无关的极轻量通用特征上采样框架 RaysUp，通过空间解耦引导编码器、光线位置编码（RayPE）和几何感知邻域交叉注意力三个关键设计，将特征重建从 2D 像素平面提升到 3D 光线域，在语义分割、深度/法向估计、视频目标分割和开放词汇分割等多种密集预测任务上达到 SOTA 或接近 SOTA，推理速度比此前唯一的 VFM-agnostic 方法 AnyUp 快约 7 倍。

**[Structural Assessment for Understanding and Guiding Dataset Distillation in Discrete Token Space](structural_assessment_for_understanding_and_guiding_dataset_distillation_in_disc.md)**

:   本文通过离散视觉token空间的统计分析方法，提出了结构评分（Structural Score）来评估蒸馏数据集质量，并基于此设计了TGDD（Token-Guided Dataset Distillation）框架，利用结构评分引导扩散模型生成高质量蒸馏数据。

**[Structured Hyperedge Adaptation for Parameter-Efficient Fine-Tuning of Vision Transformers](structured_hyperedge_adaptation_for_parameter-efficient_fine-tuning_of_vision_tr.md)**

:   把 ViT 的 adapter 微调从「每个 token 各自单独更新」搬到「超边空间」——先用可学习原型把 patch token 软路由成若干组（超边），在超边上做低秩瓶颈适配，再把更新扩散回 token，用极小的参数量（<0.5% backbone）在 VTAB-1K 上把结构化推理任务的准确率显著拉高。

**[TextDS: Parameter-Efficient Representation Alignment for Scene Text Detection under Distribution Shifts](textds_parameter-efficient_representation_alignment_for_scene_text_detection_und.md)**

:   TextDS 采用 SAM2 和 DINOv3 双编码器架构，通过逐步低秩自适应（SWLoRA）和公共子空间融合（CSF）实现无需大规模场景文本预训练的鲁棒场景文本检测，仅用 4.9M 参数即在域偏移和雨/雾/曝光/低分辨率等成像退化条件下达到领先性能。

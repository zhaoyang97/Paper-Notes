---
title: >-
  ECCV2026 目标检测论文汇总 · 18篇论文解读
description: >-
  18篇ECCV2026的目标检测方向论文解读，涵盖目标检测、异常检测、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "异常检测"
  - "少样本学习"
item_list:
  - u: "adaptive_spectrum-aware_feature_disentangled_network_for_small_object_detection/"
    t: "Adaptive Spectrum-Aware Feature Disentangled Network for Small Object Detection"
  - u: "cmds-ad_cross-modal_dual-stream_decoupling_for_few-shot_anomaly_detection/"
    t: "CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection"
  - u: "denoising-enhanced_coarse-to-fine_infrared_small_target_detection_with_attention/"
    t: "Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with Attention Prior-Guided Knowledge Distillation"
  - u: "efficient_rgb-t_object_detection_via_sparse_cross-modality_fusion/"
    t: "Efficient RGB-T Object Detection via Sparse Cross-Modality Fusion"
  - u: "following_the_flow_advection-consistent_modeling_for_event-based_small_object_de/"
    t: "Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection"
  - u: "logico_a_unified_framework_for_logical_and_structural_anomaly_detection/"
    t: "LogiCo: A Unified Framework for Logical and Structural Anomaly Detection"
  - u: "m4-sar_a_multi-resolution_multi-polarization_multi-scene_multi-source_dataset_an/"
    t: "M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for optical-SAR Object Detection"
  - u: "match_flow_matching_for_multi-view_anomaly_detection/"
    t: "MATCH: Flow Matching for Multi-View Anomaly Detection"
  - u: "mspl_multi-step_pseudo-labeling_for_open-vocabulary_object_detection/"
    t: "MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection"
  - u: "ps-mot_cultivating_instance_awareness_from_point_seeds_for_multi-object_tracking/"
    t: "PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking"
  - u: "rethinking_continual_anomaly_detection_on_the_edge_benchmarking_under_realistic_/"
    t: "Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions"
  - u: "rethinking_prototype-based_similarity_learning_for_few-shot_object_detection/"
    t: "Rethinking Prototype-based Similarity Learning for Few-Shot Object Detection"
  - u: "robust_zero-shot_anomaly_detection_under_limited_auxiliary_anomaly_priors/"
    t: "Robust Zero-shot Anomaly Detection under Limited Auxiliary Anomaly Priors"
  - u: "s2-fracmix_label-preserving_self-saliency_mixup_augmentation/"
    t: "$S^{2}$-FracMix: Label-Preserving Self-Saliency Mixup Augmentation"
  - u: "svcbench_a_streaming_video_counting_benchmark_for_spatial-temporal_state_mainten/"
    t: "SVCBench: A Streaming Video Counting Benchmark for Spatial-Temporal State Maintenance"
  - u: "the_label_imitation_game_turing_test_network_for_zero-shot_pseudo-label_pruning/"
    t: "The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning"
  - u: "towards_continual_open-vocabulary_object_detection/"
    t: "EgoExo-Con: 探索视角不变性的视频时序理解"
  - u: "vlod-tta_test-time_adaptation_of_vision-language_object_detectors/"
    t: "VLOD-TTA: Test-Time Adaptation of Vision-Language Object Detectors"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🎞️ ECCV2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×6 · 异常检测 ×5 · 少样本学习 ×4

**[Adaptive Spectrum-Aware Feature Disentangled Network for Small Object Detection](adaptive_spectrum-aware_feature_disentangled_network_for_small_object_detection.md)**

:   SFDNet将特征在频谱域显式解耦为低/中/高三频分量，分别为每个频谱设计专属的Mamba扫描策略进行上下文建模后自适应融合，同时构造类别级原型通过对比蒸馏压缩同类目标特征分布，在AI-TOD和SODA-D/SODA-A上大幅超越此前SOTA。

**[CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection](cmds-ad_cross-modal_dual-stream_decoupling_for_few-shot_anomaly_detection.md)**

:   提出 CMDS-AD，一个面向少样本多模态异常检测的跨模态双流解耦框架；通过将扩散法线估计器重新用作非线性低通滤波器来构建纯低频的辅助估计流，锚定全局结构模板，让保留耦合高低频分量的真实流更精确地捕获局部微缺陷，配合坐标感知分层特征映射器与乘性异常评分机制，在 MVTec 3D-AD 和 EyeCandies 上 1-shot 设置下分别取得 5.7% 和 7.7% 的 I-AUROC 绝对提升。

**[Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with Attention Prior-Guided Knowledge Distillation](denoising-enhanced_coarse-to-fine_infrared_small_target_detection_with_attention.md)**

:   提出 ECFNet 粗到细红外小目标检测框架：粗阶段用 RBCN 将全图稠密预测降为网格二分类 + 引入去噪辅助训练（DAT）迫使网络学目标-背景上下文，细阶段用轻量检测器 + 交叉注意力先验蒸馏（APKD）让学生关注关键目标区域，在三个红外数据集上以 31.7G FLOPs 达到 SOTA。

**[Efficient RGB-T Object Detection via Sparse Cross-Modality Fusion](efficient_rgb-t_object_detection_via_sparse_cross-modality_fusion.md)**

:   提出 SFEDet 稀疏融合框架，先用 YOLOv8-Small 级别的轻量单模态检测器高召回率筛选全图候选前景区域，再仅在稀疏的 RoI 上执行跨模态融合驱动的精确检验与多步框精炼，以约 1/3 的计算成本和 1/5 的参数量达到与 SOTA 相当甚至更优的检测精度。

**[Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection](following_the_flow_advection-consistent_modeling_for_event-based_small_object_de.md)**

:   PACT 将事件流特征演化显式建模为平流方程约束下的物理传输过程，通过轨迹级一致性让微弱目标响应沿速度场累积、噪声自然衰减，在 EV-UAV 数据集上将 IoU 从 55.18% 提升至 75.90%。

**[LogiCo: A Unified Framework for Logical and Structural Anomaly Detection](logico_a_unified_framework_for_logical_and_structural_anomaly_detection.md)**

:   LogiCo 提出统一的组件级特征重构范式，通过离散化预训练特征到组件级空间并合成伪逻辑异常来训练重构网络，同时引入交叉注意力结构重构和分割图判别器，在保持空间结构的前提下同时检测逻辑异常和结构异常，在 MVTec-LOCO 等四个基准上达到 SOTA。

**[M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for optical-SAR Object Detection](m4-sar_a_multi-resolution_multi-polarization_multi-scene_multi-source_dataset_an.md)**

:   本文提出 M4-SAR——首个大规模光学-SAR 配对旋转目标检测数据集（112,174 对图像、981,862 实例、6 类），配套统一评测工具 MSRODet 和端到端融合检测框架 E2E-OSDet（FAM + CMIM + AFM），融合后 mAP 比最优单源提升 5.1%，复杂环境下增益尤为显著。

**[MATCH: Flow Matching for Multi-View Anomaly Detection](match_flow_matching_for_multi-view_anomaly_detection.md)**

:   MATCH 是首个基于 Flow Matching 的多视图异常检测方法，用 ODE 形式的连续归一化流在预训练特征空间上做密度估计，通过省略散度项实现实时推理（18.77 FPS），在 Real-IAD 和 MANTA-Tiny 两个基准上均取得 SOTA 的检测和分割性能。

**[MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection](mspl_multi-step_pseudo-labeling_for_open-vocabulary_object_detection.md)**

:   MSPL 将开放词汇目标检测的伪标签生成从单步 CLIP 图像-文本对齐重构为「定位验证→类别识别→背景接地」三步可解释推理管线，利用 SAM 类无关分割与 MLLM 零样本推理生成高质量伪标签，再通过区域-文本对齐和对比式背景学习两个对比学习信号进行在线训练，在 OV-COCO 和 OV-LVIS 上均达到新 SOTA，新类 AP50 提升 9.4 个点。

**[PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking](ps-mot_cultivating_instance_awareness_from_point_seeds_for_multi-object_tracking.md)**

:   把多目标跟踪的每帧密集框标注换成"每个目标一个点"的点标注，用 SAM 闭环 + 频域边界幻化 + 不确定性高斯损失三层设计，把无尺度的点种子逐步"养成"尺度准确、身份连续的实例表示，在 DanceTrack 上以纯点监督拿到 52.3 HOTA、标注成本约省 64%。

**[Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions](rethinking_continual_anomaly_detection_on_the_edge_benchmarking_under_realistic_.md)**

:   本文构建了首个统一基准对现有连续异常检测方法进行头对头比较、连续漂移评估和边缘端效率测试，发现专用CAD方法并不优于带简单经验回放的传统异常检测基线，并由此提出DINOSaur——一种完全冻结DINOv3骨干+空间索引核心集记忆库的无训练方法，在零遗忘前提下以亚100ms边缘推理性达到SOTA。

**[Rethinking Prototype-based Similarity Learning for Few-Shot Object Detection](rethinking_prototype-based_similarity_learning_for_few-shot_object_detection.md)**

:   针对原型相似度学习在小样本目标检测中的两个根本瓶颈——类间相似度margin塌缩导致类别混淆、相似度嵌入缺乏空间几何线索导致定位不准——本文提出文本锚定语义掩码（TSMa）利用文本特征抑制风格驱动的虚假相似度以扩大类间margin，同时提出阶段对齐分层自回归回归（SHARe）按ViT深度逆序注入多层级视觉特征实现渐进式边界框精修，在COCO上以+10.1 nAP大幅刷新SOTA。

**[Robust Zero-shot Anomaly Detection under Limited Auxiliary Anomaly Priors](robust_zero-shot_anomaly_detection_under_limited_auxiliary_anomaly_priors.md)**

:   DIVE 首次研究辅助数据中异常先验有限场景下的零样本异常检测问题，通过浅层+深层文本嵌入注入策略让视觉编码器从有限的辅助异常模式中抽象出跨域通用的异常概念，并引入解耦机制消除物体语义对视觉嵌入的干扰；在使用纹理数据集 DTD 作为辅助数据时，DIVE 在 12 个目标数据集上的分类和分割性能最高超越 SOTA 基线 28.5%（AP）和 47.0%（AUPRO），同时将辅助数据多样性不足导致的性能退化幅度压缩约 46.5%-77.3%。

**[$S^{2}$-FracMix: Label-Preserving Self-Saliency Mixup Augmentation](s2-fracmix_label-preserving_self-saliency_mixup_augmentation.md)**

:   S²-FracMix 提出一种保标签的自显著性混合增强框架：在同一张图像内提取多尺度显著性 patch，经旋转、模糊、分形纹理注入后重新混合回原图，避免跨样本语义干扰；搭配多模式高层混合策略，在分类、检测、鲁棒性等 7 个 benchmark 上全面超越 AdAutoMix 等 SOTA 方法，同时训练开销极低。

**[SVCBench: A Streaming Video Counting Benchmark for Spatial-Temporal State Maintenance](svcbench_a_streaming_video_counting_benchmark_for_spatial-temporal_state_mainten.md)**

:   SVCBench 将视频计数重新定位为诊断视频语言模型「时空状态维护」能力的最小探针，通过流式多点查询观察预测轨迹，并设计三个互补指标揭示当前模型在周期事件计数和身份持续跟踪上的系统性缺陷。

**[The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning](the_label_imitation_game_turing_test_network_for_zero-shot_pseudo-label_pruning.md)**

:   把「伪标签剪枝」形式化成一场图灵测试式的对抗审讯，训练一个只在图像分类数据上学过、却能零样本迁移去审判目标检测伪标签的轻量「裁判网络」（TTN），靠数据集级语义上下文识别并剔除基础模型的系统性幻觉，让最差类别的 F1 提升 28%（微调后 44%）。

**[EgoExo-Con: 探索视角不变性的视频时序理解](towards_continual_open-vocabulary_object_detection.md)**

:   本文构建 EgoExo-Con 同步第一/第三人称视频时序评测基准，揭示 Video-LLM 跨视角一致性严重不足（远低于单视角表现），并提出 View-GRPO 框架，利用分组相对策略优化和显式推理奖励让模型学会视角特异推理和跨视角一致理解，在两个骨干网络上带来了显著的跨视角一致性提升。

**[VLOD-TTA: Test-Time Adaptation of Vision-Language Object Detectors](vlod-tta_test-time_adaptation_of_vision-language_object_detectors.md)**

:   针对 YOLO-World、Grounding DINO 这类视觉语言目标检测器在测试时遇到分布偏移就掉点的问题，VLOD-TTA 用「IoU 加权熵最小化」把自适应聚焦到空间上密集重叠的候选框簇，再用「图像条件下的提示选择」挑出与当前图片最匹配的文本提示，只更新极少量 adapter 参数，就能在艺术风格、恶劣驾驶、低光、常见退化等多种偏移下稳定超过已有 TTA 基线和上一代专用方法，且开销远低。

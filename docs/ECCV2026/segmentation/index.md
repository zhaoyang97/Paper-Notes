---
title: >-
  ECCV2026 语义分割论文汇总 · 14篇论文解读
description: >-
  14篇ECCV2026的语义分割方向论文解读，涵盖语义分割、对抗鲁棒、对齐/RLHF、语音、布局/合成、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "语义分割"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "语音"
  - "布局/合成"
  - "少样本学习"
item_list:
  - u: "affmae_scalable_vision_pre-training_for_high-resolution_microscopy_segmentation_/"
    t: "AFFMAE: Scalable Vision Pre-Training for High-Resolution Microscopy Segmentation on Desktop Hardware"
  - u: "delayed_bidirectional_alignment_via_disentangled_audio_semantics_for_audio-visua/"
    t: "Delayed Bidirectional Alignment via Disentangled Audio Semantics for Audio-Visual Segmentation"
  - u: "et-sam_efficient_point_prompt_prediction_in_sam_for_unified_scene_text_detection/"
    t: "ET-SAM: Efficient Point Prompt Prediction in SAM for Unified Scene Text Detection and Layout Analysis"
  - u: "fevos_foresight_expression_video_object_segmentation/"
    t: "FeVOS: Foresight Expression Video Object Segmentation"
  - u: "forget_anticipate_and_adapt_test_time_training_for_long_videos/"
    t: "Forget, Anticipate and Adapt: Test Time Training for Long Videos"
  - u: "hierarchical_spatial_and_channel_aggregation_for_cross-domain_few-shot_segmentat/"
    t: "Hierarchical Spatial and Channel Aggregation for Cross-domain Few-shot Segmentation"
  - u: "mobilemanibench_simplifying_model_verification_for_mobile_manipulation/"
    t: "MobileManiBench: Simplifying Model Verification for Mobile Manipulation"
  - u: "moduseg_decoupling_object_discovery_and_semantic_retrieval_for_training-free_wea/"
    t: "ModuSeg: Decoupling Object Discovery and Semantic Retrieval for Training-Free Weakly Supervised Segmentation"
  - u: "residual-guided_expert_specialization_for_incomplete_multimodal_learning/"
    t: "Residual-Guided Expert Specialization for Incomplete Multimodal Learning"
  - u: "sarif_segment_anything_for_robust_image_forensics/"
    t: "SARIF: Segment Anything for Robust Image Forensics"
  - u: "segmenting_fast_and_slow_real-time_open-vocabulary_video_instance_segmentation_w/"
    t: "Segmenting, Fast and Slow: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing"
  - u: "skelem_training-signal_decoupling_of_skeleton_and_diffusion_for_self-supervised_/"
    t: "SkelEM: Training-Signal Decoupling of Skeleton and Diffusion for Self-supervised Axial Super-Resolution in Volume Microscopy"
  - u: "steerable_visual_representations/"
    t: "Steerable Visual Representations"
  - u: "toward_robust_in-context_segmentation_via_concept_guidance/"
    t: "Toward Robust In-Context Segmentation via Concept Guidance"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🎞️ ECCV2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (122)](../../CVPR2026/segmentation/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/segmentation/index.md) · [🧪 ICML2026 (14)](../../ICML2026/segmentation/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (45)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (73)](../../ICCV2025/segmentation/index.md)

🔥 **高频主题：** 语义分割 ×7 · 对抗鲁棒 ×2

**[AFFMAE: Scalable Vision Pre-Training for High-Resolution Microscopy Segmentation on Desktop Hardware](affmae_scalable_vision_pre-training_for_high-resolution_microscopy_segmentation_.md)**

:   AFFMAE 将 AutoFocusFormer 的自适应离网（off-grid）令牌合并机制融入 MAE 框架，在保持仅编码可见令牌的高效不对称设计的同时，实现了层次化骨干的掩码友好预训练，使得高分辨率显微图像分割预训练可在单张消费级 GPU 上完成，以同等参数量匹配 ViT-MAE 分割精度，预训练吞吐提升 2 倍、高分辨率微调吞吐最高提升 5 倍。

**[Delayed Bidirectional Alignment via Disentangled Audio Semantics for Audio-Visual Segmentation](delayed_bidirectional_alignment_via_disentangled_audio_semantics_for_audio-visua.md)**

:   DDAVS 通过原型记忆库锚定的可学习查询来解耦多源音频语义，并引入仅在深层进行的延迟双向交叉注意力对齐视听模态，在 AVSBench 和 VPO 多源/多类/多实例场景上全面超越此前 SOTA。

**[ET-SAM: Efficient Point Prompt Prediction in SAM for Unified Scene Text Detection and Layout Analysis](et-sam_efficient_point_prompt_prediction_in_sam_for_unified_scene_text_detection.md)**

:   ET-SAM 用轻量 Point Decoder 预测词级热力图以提取稀疏前景点，替代 Hi-SAM 从像素级分割中随机采样上千个点的做法，配合可学习任务提示和联合训练策略统一利用多粒度异构标注数据，在 HierText 上以约 3 倍推理加速达到有竞争力性能，并在 Total-Text/CTW1500/ICDAR2015 上平均 F-score 提升 11.0%。

**[FeVOS: Foresight Expression Video Object Segmentation](fevos_foresight_expression_video_object_segmentation.md)**

:   提出FeVOS（Foresight Expression Video Object Segmentation）任务——根据观察帧中的视觉线索预测未来事件并分割相关目标，构建包含968视频和14525条预测性表达的数据集并附带2904条合成思维链标注，设计FeVOS-R1两阶段训练框架（CoT监督微调冷启动 + GRPO强化学习纯IoU奖励精炼推理），在FeVOS上达到42.3 J&F，较微调Sa2VA基线提升6.5个点，同时在ReVOS（60.3）和MeViS（49.5）上展现出强零样本泛化能力。

**[Forget, Anticipate and Adapt: Test Time Training for Long Videos](forget_anticipate_and_adapt_test_time_training_for_long_videos.md)**

:   提出 Frame Forgetting Network（FFN），通过只处理滑动窗口中退出/进入两帧的「遗忘—预测—适应」机制，将长视频测试时训练的计算复杂度从 O(k) 降至常数，并基于「惊异度」动态决定何时执行 TTT，在密集分割、深度估计和动作分类上达到更优的精度-效率权衡。

**[Hierarchical Spatial and Channel Aggregation for Cross-domain Few-shot Segmentation](hierarchical_spatial_and_channel_aggregation_for_cross-domain_few-shot_segmentat.md)**

:   本文提出 DHANet，通过沿空间维度进行层次聚合以缓解语义过对齐、沿通道维度进行层次聚合以缓解属性过对齐，并利用在线概率语义库在推理时提供额外支持信息，在四个跨域少样本分割基准上取得 SOTA。

**[MobileManiBench: Simplifying Model Verification for Mobile Manipulation](mobilemanibench_simplifying_model_verification_for_mobile_manipulation.md)**

:   本文提出 MobileManiBench——一个基于 Isaac Sim 的大型移动操作仿真基准，通过强化学习自动生成 300K 多样化操作轨迹（含语言指令、多视角 RGB-D 图像、物体/机器人状态与动作），覆盖 2 种机器人平台、630 个物体、5 种技能和 100 个场景，并在此基准上系统评估了多个 VLA 模型，为移动操作的 VLA 架构验证提供了标准化平台。

**[ModuSeg: Decoupling Object Discovery and Semantic Retrieval for Training-Free Weakly Supervised Segmentation](moduseg_decoupling_object_discovery_and_semantic_retrieval_for_training-free_wea.md)**

:   ModuSeg 将弱监督语义分割显式解耦为"类无关目标发现"和"语义检索"两个独立阶段——用通用 Mask Proposer 提取几何提案、用离线特征银行进行非参数检索，在完全不需微调的情况下以训练集仅图像级标签达到 VOC 86.3 mIoU 的 SOTA。

**[Residual-Guided Expert Specialization for Incomplete Multimodal Learning](residual-guided_expert_specialization_for_incomplete_multimodal_learning.md)**

:   MARS 提出一种混合专家（MoE）框架，通过计算完整模态与缺失模态表征之间的残差来捕获模态缺失导致的表征偏移模式，以此引导专家特化；同时引入双路由器蒸馏和差异感知噪声正则化解决训练-推理路由不一致问题，在不完备多模态分类和分割四个数据集上全面超越 SOTA。

**[SARIF: Segment Anything for Robust Image Forensics](sarif_segment_anything_for_robust_image_forensics.md)**

:   SARIF 同时运行一个冻结的原始 SAM 编码器和一个 LoRA 微调编码器，计算两者在各全局注意力层的残差作为篡改特定线索，再与上一轮预测掩码融合成提示、驱动 SAM 原有轻量解码器做 5 步渐进精化，在跨数据集篡改定位上达到了最强的平均性能。

**[Segmenting, Fast and Slow: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing](segmenting_fast_and_slow_real-time_open-vocabulary_video_instance_segmentation_w.md)**

:   SegFS 提出双路径快慢框架用于开放词汇视频实例分割（OV-VIS），在稀疏关键帧上运行完整的 object-centric 模型提取实例嵌入，然后将嵌入投影回骨干网络特征空间，以轻量级 Fast Feature Aggregator 在中间帧上完成高效的实例重定位与分割，实现了高达 14 倍的端侧延迟降低并首次在移动设备上跨越 30 FPS 实时阈值，同时性能损失控制在 1 AP 以内。

**[SkelEM: Training-Signal Decoupling of Skeleton and Diffusion for Self-supervised Axial Super-Resolution in Volume Microscopy](skelem_training-signal_decoupling_of_skeleton_and_diffusion_for_self-supervised_.md)**

:   SkelEM 提出一种自监督轴向超分辨率框架，通过将拓扑骨架网络与扩散精炼器的训练信号完全解耦（无共享梯度），以冻结的确定性骨架锚定结构拓扑、再用截断扩散在不超过 5 步内恢复高频生物纹理，在保真度-感知质量-推理速度的三难困境中取得自监督方法中最优的平衡，并在膜分割下游任务上达到 SOTA。

**[Steerable Visual Representations](steerable_visual_representations.md)**

:   SteerViT 把轻量门控交叉注意力层交错插进冻结的 ViT（如 DINOv2）内部，让文本在视觉编码「过程中」就介入，用一个 patch 级指代分割代理任务训练，仅加 21M 参数就能让视觉特征被自然语言「操控」聚焦到任意物体，同时几乎不损失原 ViT 的表征质量，在文本条件检索、个性化物体判别、工业异常分割等任务上零样本追平甚至超过专用方法。

**[Toward Robust In-Context Segmentation via Concept Guidance](toward_robust_in-context_segmentation_via_concept_guidance.md)**

:   CG-ICS 将 in-context 分割问题重新定义为概念引导的提示分割问题：用 MLLM 生成候选概念 + SAM3 打分 + 树搜索选出最佳文本概念，同时用图像拼接技巧从参考图提取查询侧的视觉样例，两者共同驱动冻结的 SAM3 完成分割，在保持高精度的同时显著降低不同参考选择导致的结果方差。

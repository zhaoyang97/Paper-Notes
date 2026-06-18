---
title: >-
  CVPR2026 LLM效率论文汇总 · 6篇论文解读
description: >-
  6篇CVPR2026的 LLM 效率方向论文解读，涵盖 LLM、对齐/RLHF、多模态、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
  - "多模态"
  - "模型压缩"
item_list:
  - u: "e2-sci_elastic_edge-cloud_speculative_decoding_via_credit_inertia/"
    t: "E$^2$-SCI: Elastic Edge-Cloud Speculative Decoding via Credit Inertia"
  - u: "gated_kalmanet_a_fading_memory_layer_through_test-time_ridge_regression/"
    t: "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression"
  - u: "generalizable_video_quality_assessment_via_weak-to-strong_learning/"
    t: "Generalizable Video Quality Assessment via Weak-to-Strong Learning"
  - u: "life-iqa_boosting_blind_image_quality_assessment_through_gcn-enhanced_layer_inte/"
    t: "Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling"
  - u: "parallelvlm_lossless_video-llm_acceleration_with_visual_alignment_aware_parallel/"
    t: "ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding"
  - u: "quietprune_query-guided_early_token_pruning_for_vision-language_models/"
    t: "QuietPrune: Query-Guided Early Token Pruning for Vision-Language Models"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**📷 CVPR2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (32)](../../ICML2026/llm_efficiency/index.md) · [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

**[E$^2$-SCI: Elastic Edge-Cloud Speculative Decoding via Credit Inertia](e2-sci_elastic_edge-cloud_speculative_decoding_via_credit_inertia.md)**

:   本文发现边云投机解码中相邻窗口的 token 接受率存在强时序一致性（称为"信用惯性"），据此用历史接受率动态调节验证阈值，再配合异步流水线（PLC）把草稿生成与云端验证并行起来，在 DeepSeek-R1-Distill-Qwen (1.5B/32B) 上达到 9.4+ tokens/s、相对 FSD 基线提速 88.5% 且不损精度。

**[Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression](gated_kalmanet_a_fading_memory_layer_through_test-time_ridge_regression.md)**

:   把线性状态空间模型（SSM）的状态更新重新解释成"对全部历史做一次测试时岭回归"，用卡尔曼滤波的精确增益替代现有 SSM 的一步梯度近似，并通过自适应正则 + Chebyshev 迭代解决低精度数值不稳与并行训练两大障碍，在短/长上下文及 ImageNet 上都超过 Mamba2、Gated DeltaNet 等线性 SSM。

**[Generalizable Video Quality Assessment via Weak-to-Strong Learning](generalizable_video_quality_assessment_via_weak-to-strong_learning.md)**

:   不依赖任何人工打分标签，用现成 VQA 模型当"弱老师"去监督一个高容量多模态大模型"强学生"，再把学生回收成下一轮老师做迭代，最终在域内持平、在 OOD 上大幅超越所有老师，把 VQA 的 OOD 整体 SRCC 从 0.59 推到 0.745。

**[Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling](life-iqa_boosting_blind_image_quality_assessment_through_gcn-enhanced_layer_inte.md)**

:   针对盲图像质量评估(BIQA)中"把所有层特征一股脑融合反而引入噪声"的问题，Life-IQA 只用骨干网最深两层特征做质量解码：用 GCN 增强的查询拓扑把 stage4 特征当 query、stage3 特征当 key/value 做跨层交互，再用一个后置的 MoE 头按失真类型解耦特征，在七个 BIQA 基准上以约 95M 参数取得 SOTA。

**[ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding](parallelvlm_lossless_video-llm_acceleration_with_visual_alignment_aware_parallel.md)**

:   针对 Video-LLM 投机解码在长视频上"draft 和 target 互相干等"以及"提速比和模型对齐相互掣肘"两大瓶颈，ParallelVLM 把预填充和解码都做成 draft/target 并行流水线，并用基于视觉-文本相似度变化(而非注意力分数)的无偏剪枝 UV-Prune 扩大草稿窗口，在 LLaVA-OneVision-72B / Qwen2.5-VL-32B 上分别取得 3.36× / 2.42× 的无损加速，且免训练、即插即用。

**[QuietPrune: Query-Guided Early Token Pruning for Vision-Language Models](quietprune_query-guided_early_token_pruning_for_vision-language_models.md)**

:   QuietPrune 提出**查询引导的早剪枝**：在 ViT 前向过程中、而非传统的 ViT 之后，就把与文本查询无关的视觉 token 剪掉——通过把 VLM 投影器做**逆变换**得到的轻量适配器，把文本查询转成一个视觉域的 `[Q-CLS]` token 来提供文本指导，再以 2×2 分组的半结构化方式剪枝并聚合冗余 token，在 Qwen3-VL / InternVL3 上把 prefill 延迟最多降 19.0%、同时比现有晚剪枝方法精度高 4.2%。

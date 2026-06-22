---
title: >-
  ACL2026 VLMEfficiency论文汇总 · 6篇论文解读
description: >-
  6篇ACL2026的 VLM Efficiency 方向论文解读，涵盖多模态、压缩/编码、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "压缩/编码"
  - "模型压缩"
item_list:
  - u: "apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro/"
    t: "APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention"
  - u: "from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_/"
    t: "From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration"
  - u: "hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi/"
    t: "HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding"
  - u: "hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo/"
    t: "HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models"
  - u: "macs_modality-aware_capacity_scaling_for_efficient_multimodal_moe_inference/"
    t: "MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference"
  - u: "regate_learning_faster_and_better_with_fewer_tokens_in_mllms/"
    t: "ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**💬 ACL2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×2

**[APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)**

:   APB-V 用面向序列并行的近似注意力和系统级负载均衡加速长视频 LMM 推理，在保留完整视觉 embedding 的同时，在 64 帧 1440p 设置下相对 FlashAttn、ZigZagRing 和 APB 分别达到 12.72×、1.70× 和 1.18× 加速，且没有显著性能损失。

**[From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)**

:   揭示 MLLM 推理中视觉冗余的两种来源——ViT 密集 tokenization 导致的固有冗余（IVR）和深层语义饱和导致的次生冗余（SSR，且其表现形式因骨干架构不同而异），提出 HalfV 框架分别处理两类冗余，在 Qwen2.5-VL 上实现4.1倍 FLOPs 加速且保留96.8%性能。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)**

:   本文发现视觉编码器中存在层级注意力模式——中层关注主体对象、深层关注全局信息，据此提出 HiPrune，一种免训练、模型无关的视觉 token 剪枝方法，通过选择三类 token（Anchor/Buffer/Register）保留不同层级的视觉信息，仅用 1/3 token 保持 99.3% 性能，FLOPs 减少 58.7%。

**[MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference](macs_modality-aware_capacity_scaling_for_efficient_multimodal_moe_inference.md)**

:   针对 MoE 多模态大模型在专家并行（EP）推理下被"最慢专家"拖累的 straggler 问题，MACS 用视觉 token 的熵当作语义重要性权重来重估专家负载，并按 batch 的实时模态构成动态缩放各专家容量，是一个无需训练的推理框架，在 12 个多模态基准上几乎不掉点（平均保留 vanilla 99.7%）而显著优于按 token 计数的 CAI-MoE。

**[ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs](regate_learning_faster_and_better_with_fewer_tokens_in_mllms.md)**

:   ReGATE 用冻结的 text-only teacher 估计哪些输出 token 需要视觉信息，再结合 student 的历史学习难度动态选择训练 token，让 MLLM 在不改架构、不加参数的情况下用更少 token 更快训练，并在多个图像和视频 benchmark 上达到或超过标准微调。

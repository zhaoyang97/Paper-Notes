---
title: >-
  CVPR2025 信号/通信论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的信号/通信方向论文解读，涵盖水印/隐写、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "水印/隐写"
  - "压缩/编码"
item_list:
  - u: "abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a/"
    t: "ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention"
  - u: "breaking_the_low-rank_dilemma_of_linear_attention/"
    t: "Breaking the Low-Rank Dilemma of Linear Attention"
  - u: "continuous_space-time_video_resampling_with_invertible_motion_steganography/"
    t: "Continuous Space-Time Video Resampling with Invertible Motion Steganography"
  - u: "ditask_multi-task_fine-tuning_with_diffeomorphic_transformations/"
    t: "DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations"
  - u: "neural_video_compression_with_context_modulation/"
    t: "Neural Video Compression with Context Modulation"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md)

**[ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)**

:   提出 ABC-Former，通过引入 CIELab 色彩空间和 RGB 直方图作为辅助双模态信息，利用跨域 Transformer 和交互通道注意力（ICA）模块实现全局色彩知识的跨模态迁移，在 sRGB 白平衡矫正任务上取得 SOTA 效果；同时扩展为 ABC-FormerM 处理混合光照场景。

**[Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   从理论上揭示线性注意力性能不及 Softmax 注意力的根本原因是输出特征的低秩问题，提出秩增强线性注意力（RALA），通过增强 KV 缓存秩和输出特征秩两种互补策略，在保持线性复杂度的同时追平甚至超越 Softmax 注意力的表现。

**[Continuous Space-Time Video Resampling with Invertible Motion Steganography](continuous_space-time_video_resampling_with_invertible_motion_steganography.md)**

:   提出可逆运动隐写模块（IMSM），在视频时间下采样过程中将运动信息隐写到低帧率帧中，上采样时通过逆变换精确恢复运动细节，同时支持连续（非整数）的时空重采样因子，在保持下采样帧视觉质量的同时显著提升重建质量。

**[DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)**

:   提出 DiTASK，利用连续分段仿射 (CPAB) 微分同胚变换对预训练权重矩阵的奇异值进行平滑变换而保持奇异向量不变，以每层仅约 32 个参数实现全秩更新的多任务微调，在 PASCAL MTL 上以 75% 更少的参数超越 MTLoRA 26.27%。

**[Neural Video Compression with Context Modulation](neural_video_compression_with_context_modulation.md)**

:   提出 DCMVC 框架，通过流定向（flow orientation）和上下文补偿（context compensation）两步调制时序上下文，在像素域和特征域充分利用参考信息，实现比 H.266/VVC 平均节省 22.7% 码率、比前 SOTA DCVC-FM 节省 10.1% 码率的压缩性能。

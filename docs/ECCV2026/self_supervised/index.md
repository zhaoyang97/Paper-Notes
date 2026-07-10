---
title: >-
  ECCV2026 自监督/表示学习论文汇总 · 3篇论文解读
description: >-
  3篇ECCV2026的自监督/表示学习方向论文解读，涵盖人脸/视线、自监督学习、水印/隐写、视频生成、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "人脸/视线"
  - "自监督学习"
  - "水印/隐写"
  - "视频生成"
  - "对抗鲁棒"
item_list:
  - u: "explore_expert_patch-level_loss_routing_for_multi-objective_masked_image_modelin/"
    t: "ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling"
  - u: "from_phase_to_phenomenon_self-supervised_learning_of_subsurface_scattering_with_/"
    t: "From Phase to Phenomenon: Self-Supervised Learning of Subsurface Scattering with Minimal Phase-shift Inputs"
  - u: "zipfian_adaptive_loss_for_neural_compression/"
    t: "LoT-Pass: Long-term-robust Image Watermarking for Image to Video Generation"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🎞️ ECCV2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md)

**[ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling](explore_expert_patch-level_loss_routing_for_multi-objective_masked_image_modelin.md)**

:   ExPLoRe 将 Soft-MoE 的调度权重 (dispatch weights) 重新用作多目标掩码图像建模中**逐 patch、可学习的损失系数**，通过损失耦合 (loss-coupling) 机制让路由器依据不同区域的语义特性动态调整各目标的权重，在 ImageNet-1K 上以更低的推理计算量（11.93 GFLOPs vs. 17.45）达到了 80.6% linear probe 和 85.3% finetuning 精度，并提出了三组下游迁移策略（Freeze Routing / Expert Dropout / Freeze Attention）将语义分割的 mIoU 差距从 2.5–2.9 完全弥合。

**[From Phase to Phenomenon: Self-Supervised Learning of Subsurface Scattering with Minimal Phase-shift Inputs](from_phase_to_phenomenon_self-supervised_learning_of_subsurface_scattering_with_.md)**

:   提出一套自监督预训练框架，利用投影仪-相机系统仅需8张高频相移条纹（PSP）图像即可学习通用的次表面散射（SSS）表征，再通过解码器预测各向异性散射足迹，实现零样本重光照，将新物体的数据采集量从数千张降至8张。

**[LoT-Pass: Long-term-robust Image Watermarking for Image to Video Generation](zipfian_adaptive_loss_for_neural_compression.md)**

:   LoT-Pass 提出双向鲁棒策略——训练阶段用时序演变模拟主动扩大水印的鲁棒距离，推理阶段用光流反演把严重漂移帧"拉回"可提取状态——首次在 I2V 生成场景下实现超过 95% 的视频级水印提取准确率。

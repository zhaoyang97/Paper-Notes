---
title: >-
  ICML2025 图像恢复论文汇总 · 5篇论文解读
description: >-
  5篇ICML2025的图像恢复方向论文解读，涵盖扩散模型、图像恢复、自监督学习、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "自监督学习"
  - "时序预测"
item_list:
  - u: "adaptive_estimation_and_learning_under_temporal_distribution_shift/"
    t: "Adaptive Estimation and Learning under Temporal Distribution Shift"
  - u: "epsilon-vae_denoising_as_visual_decoding/"
    t: "ε-VAE: Denoising as Visual Decoding"
  - u: "evaluating_deepfake_detectors_in_the_wild/"
    t: "Evaluating Deepfake Detectors in the Wild"
  - u: "harmonica_harmonizing_training_and_inference_for_better_feature_caching_in_diffu/"
    t: "HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration"
  - u: "timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_/"
    t: "TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🧪 ICML2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md)

🔥 **高频主题：** 扩散模型 ×2

**[Adaptive Estimation and Learning under Temporal Distribution Shift](adaptive_estimation_and_learning_under_temporal_distribution_shift.md)**

:   提出基于小波软阈值的估计算法，在无需先验知识的情况下实现时间分布偏移下的最优逐点估计误差界，将序列非平稳性与小波域稀疏性建立联系，并应用于分布偏移下的二分类和全变分去噪问题。

**[ε-VAE: Denoising as Visual Decoding](epsilon-vae_denoising_as_visual_decoding.md)**

:   提出 ε-VAE，将传统自编码器中的单步确定性解码器替换为扩散/去噪过程，实现"去噪即解码"（Denoising as Decoding），在相同压缩率下重建质量提升 40%、下游生成质量提升 22%，或在保持生成质量的同时通过提高压缩率实现 2.3 倍推理加速。

**[Evaluating Deepfake Detectors in the Wild](evaluating_deepfake_detectors_in_the_wild.md)**

:   构建包含50万+高质量deepfake图像的新数据集，通过引入JPEG压缩、降分辨率、图像增强等真实场景增强，系统评估6种开源deepfake检测器，揭示不到一半检测器AUC>60%，最低仅约50%（随机水平）。

**[HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration](harmonica_harmonizing_training_and_inference_for_better_feature_caching_in_diffu.md)**

:   提出 HarmoniCa 框架，通过 Step-Wise Denoising Training (SDT) 和 Image Error Proxy-Guided Objective (IEPO) 两大设计解决现有学习型特征缓存方法中训练与推理不对齐的问题，在 PixArt-α 等 8 种模型上实现超 40% 延迟降低（2.07× 理论加速）且不损失生成质量。

**[TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)**

:   提出 TimeDART，将自回归建模与去噪扩散过程统一在一个自监督预训练框架中，通过因果 Transformer 编码器捕获长期动态演化、patch 级扩散去噪捕获细粒度局部模式，在预测和分类任务上均超越现有方法。

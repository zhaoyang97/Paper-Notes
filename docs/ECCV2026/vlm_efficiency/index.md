---
title: >-
  ECCV2026 VLMEfficiency论文汇总 · 8篇论文解读
description: >-
  8篇ECCV2026的 VLM Efficiency 方向论文解读，涵盖多模态、LLM、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "LLM"
  - "模型压缩"
item_list:
  - u: "accelerating_multimodal_large_language_models_with_prior-corrected_token_reducti/"
    t: "Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction"
  - u: "hsd_training-free_acceleration_for_document_parsing_vision-language_models_with_/"
    t: "HSD: Training-Free Acceleration for Document Parsing Vision-Language Models with Hierarchical Speculative Decoding"
  - u: "paying_more_attention_to_visual_tokens_in_self-evolving_large_multimodal_models/"
    t: "Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models"
  - u: "resilphase_plug-and-play_phase_mapping_and_noise-resilient_macro-trajectory_extr/"
    t: "ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration"
  - u: "smart_when_is_it_actually_worth_expanding_a_speculative_tree/"
    t: "SMART: When is it Actually Worth Expanding a Speculative Tree?"
  - u: "spectral_evolution-guided_token_pruning_in_multimodal_large_language_models/"
    t: "Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models"
  - u: "viq_text-aligned_visual_quantized_representations_at_any_resolution/"
    t: "ViQ: Text-Aligned Visual Quantized Representations at Any Resolution"
  - u: "zero-shot_quantization_for_object_detectors_using_off-the-shelf_generative_model/"
    t: "GoodQ: 利用现成生成模型实现目标检测器的零样本量化"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🎞️ ECCV2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×4 · LLM ×2 · 模型压缩 ×2

**[Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction](accelerating_multimodal_large_language_models_with_prior-corrected_token_reducti.md)**

:   PriorTR 是一种无需训练的视觉 token 剪枝方法，通过用 null token（分隔符）估计模型固有的注意力先验，再用 V-可用信息 $P \cdot \log(P/Q)$ 替代原始注意力分数进行 token 排序，在单次前向传播中完成先验校正与物理剪枝，在激进 token 预算下显著提升了 MLLM 的精度-效率权衡。

**[HSD: Training-Free Acceleration for Document Parsing Vision-Language Models with Hierarchical Speculative Decoding](hsd_training-free_acceleration_for_document_parsing_vision-language_models_with_.md)**

:   本文提出层次化推测解码（HSD），一种免训练的端到端文档解析加速方法：先用轻量级 pipeline 为每个区域生成粗草稿，再分两阶段验证——Stage 1 在裁切区域上并行做局部验证以获得高吞吐，Stage 2 在全页上做全局验证以恢复跨区域连贯性；在 HunyuanOCR 上实现 2.78 倍端到端加速且精度近乎无损，长文档场景最高加速 7.04 倍。

**[Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models](paying_more_attention_to_visual_tokens_in_self-evolving_large_multimodal_models.md)**

:   VISE 提出纯无监督的单模型自演化框架，用几何不变性和语义不变性两种奖励信号直接正则化解码器的视觉条件化策略（而非优化答案一致性），在 Qwen3-VL-2B 上 COCO CIDEr 提升 +16.85、TextCaps CIDEr 提升 +19.66、幻觉 Chair-I 降低 5.0 点，且跨 4 个模型规模（2B/4B/8B/32B）与 4 种架构主干一致有效，无任何任务间的 tradeoff。

**[ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration](resilphase_plug-and-play_phase_mapping_and_noise-resilient_macro-trajectory_extr.md)**

:   ResilPhase 将 DiT 免训练加速从逐层导数外推范式重构为 ODE 空间中的宏观轨迹外推：用端到端 Global Drift 消除逐层误差级联，用无导数 Barycentric Lagrange 插值绕过导数混沌噪声，再用 Chebyshev/Balanced 相位映射将离散时间步投影到有界相位空间以抑制 Runge 现象，在 FLUX.1-dev 上以约 5 倍加速仍保持 SOTA 画质。

**[SMART: When is it Actually Worth Expanding a Speculative Tree?](smart_when_is_it_actually_worth_expanding_a_speculative_tree.md)**

:   SMART 提出了一套硬件感知的边际分析框架，将推测解码中的推測树构建重写为最大化端到端加速比的序列决策问题——只有在节点的边际收益-成本比超过当前树的整体加速比时才扩展该节点，从而避免了大批量下验证开销超线性增长导致的"负加速"悖论，平均额外带来 20.0%（MLLM）和 15.4%（LLM）的加速提升且无需重新训练。

**[Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models](spectral_evolution-guided_token_pruning_in_multimodal_large_language_models.md)**

:   本文提出 CLSE(Cross-Layer Spectral Evolution),一个 training-free 的视觉 token 剪枝框架:不再用单层 attention 或特征幅值判断 token 重要性,而是把每层视觉 token 变换到频域、经高通滤波后度量其高频结构能量在相邻 LLM 层之间的相对变化量,变化剧烈的 token 被判为语义活跃并保留;在 LLaVA/Qwen2-VL/Video-LLaVA 等多个 MLLM、剪到 11%~33% token 的激进压缩下,精度保持显著优于 FastV、SparseVLM、PDrop 等。

**[ViQ: Text-Aligned Visual Quantized Representations at Any Resolution](viq_text-aligned_visual_quantized_representations_at_any_resolution.md)**

:   ViQ 提出两阶段视觉量化框架——先做文本对齐预训练让连续特征富含语义，再渐进式压缩并用量化器 FSQ 做低维离散化——使离散视觉编码器在 9 个多模态基准上首次达到与连续编码器接近甚至超越的性能，同时带来 20%-70% 的训练加速和接近 1/96 的存储压缩比。

**[GoodQ: 利用现成生成模型实现目标检测器的零样本量化](zero-shot_quantization_for_object_detectors_using_off-the-shelf_generative_model.md)**

:   GoodQ 利用现成的扩散模型生成多样化训练图像，再通过信息密集提示、固有分布感知选择和教师引导自适应降噪三个组件系统性地解决生成图像与目标检测零样本量化之间的三大挑战，在 W4A4 和 W3A3 等极低位宽下大幅超越此前基于噪声优化的方法。

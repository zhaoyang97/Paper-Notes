---
title: >-
  ICML2026 VLMEfficiency论文汇总 · 4篇论文解读
description: >-
  4篇ICML2026的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩、对齐/RLHF、压缩/编码、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
  - "对齐/RLHF"
  - "压缩/编码"
  - "对抗鲁棒"
item_list:
  - u: "clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar/"
    t: "CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models"
  - u: "gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms/"
    t: "Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs"
  - u: "less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac/"
    t: "Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy"
  - u: "on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token/"
    t: "On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🧪 ICML2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×2 · 模型压缩 ×2

**[CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)**

:   发现 CLIP 中指代区域的视觉 token 与 [EOS] 文本 token 呈反直觉的低相似度现象（similarity reversal），据此提出 LiteLVLM——一种免训练的文本引导视觉 token 剪枝方法，在裁剪 66.7% token 后仍保留 90.3% 原始像素定位性能，同时实现 22% 推理加速和 2.3× 显存节省。

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)**

:   本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器），让 INT4 量化的 LLaVA / Qwen-VL 不仅没掉点，反而在多个 benchmark 上超过 BF16 基线，同时实测 3× 吞吐 + 54% 显存节省。

**[Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)**

:   这篇用 70 万次实验跑遍了 16 种量化方法 × 10 种 VLM × 多项可靠性指标，发现量化不是单纯破坏者——它会通过抑制高 rank 低方差的频谱分量，同时提升 calibration、OOD 检测和噪声鲁棒性，但也会放大对协变量偏移和虚假相关的依赖。

**[On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)**

:   本文首次系统研究了带视觉Token压缩的大视觉语言模型(LVLM)的对抗鲁棒性，指出现有编码器攻击存在"优化-推理空间不匹配"问题，并提出 CAGE 攻击通过期望特征扰动 (EFD) 与排名-扰动对齐 (RDA) 两个目标，在未知压缩机制与未知Token预算下显著降低被压缩 LVLM 的鲁棒精度。

---
title: >-
  ECCV2024 VLMEfficiency论文汇总 · 4篇论文解读
description: >-
  4篇ECCV2024的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
  - "LLM"
item_list:
  - u: "bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_/"
    t: "Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding"
  - u: "groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode/"
    t: "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models"
  - u: "ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models/"
    t: "IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models"
  - u: "quantized_prompt_for_efficient_generalization_of_vision-language_models/"
    t: "Quantized Prompt for Efficient Generalization of Vision-Language Models"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🎞️ ECCV2024** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×3 · 模型压缩 ×2

**[Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding](bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_.md)**

:   提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)**

:   Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

**[IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)**

:   IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。

**[Quantized Prompt for Efficient Generalization of Vision-Language Models](quantized_prompt_for_efficient_generalization_of_vision-language_models.md)**

:   将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。

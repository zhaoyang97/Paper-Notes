---
title: >-
  ICML2025 幻觉检测论文汇总 · 3篇论文解读
description: >-
  3篇ICML2025的幻觉检测方向论文解读，涵盖 LLM、多模态、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
  - "Agent"
item_list:
  - u: "look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit/"
    t: "Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models"
  - u: "rejecting_hallucinated_state_targets_during_planning/"
    t: "Rejecting Hallucinated State Targets during Planning"
  - u: "steer_llm_latents_for_hallucination_detection/"
    t: "Steer LLM Latents for Hallucination Detection"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🧪 ICML2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md)

🔥 **高频主题：** LLM ×2

**[Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models](look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit.md)**

:   提出 MemVR 解码范式，将视觉 token 作为补充证据通过 FFN 的 key-value memory 机制重新注入到中间触发层，以"再看一次"的方式缓解 MLLM 幻觉问题，不引入额外推理开销。

**[Rejecting Hallucinated State Targets during Planning](rejecting_hallucinated_state_targets_during_planning.md)**

:   本文系统识别了目标导向决策规划中生成器产生不可行目标（幻觉目标）导致的"妄想行为"类型，并设计了一种可行性评估器（feasibility evaluator）作为附加模块来识别和拒绝这些不可行目标，结合离策略学习规则、分布式架构和后见重标记数据增强，在不修改原始智能体的前提下显著减少妄想行为并提升OOD泛化性能。

**[Steer LLM Latents for Hallucination Detection](steer_llm_latents_for_hallucination_detection.md)**

:   提出 Truthfulness Separator Vector (TSV)，一种轻量级 steering vector，在推理时重塑 LLM 表示空间以增强真实与幻觉输出的分离，仅需 32 个标注样本即可接近全监督性能。

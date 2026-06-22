---
title: >-
  ICML2025 信息检索/RAG论文汇总 · 6篇论文解读
description: >-
  6篇ICML2025的信息检索/RAG 方向论文解读，涵盖 RAG、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "对抗鲁棒"
item_list:
  - u: "dont_lag_rag_training-free_adversarial_detection_using_rag/"
    t: "Don't Lag, RAG: Training-Free Adversarial Detection Using RAG"
  - u: "fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems/"
    t: "FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems"
  - u: "poqd_performance-oriented_query_decomposer_for_multi-vector_retrieval/"
    t: "POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval"
  - u: "rapid_long-context_inference_with_retrieval-augmented_speculative_decoding/"
    t: "RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding"
  - u: "unable_to_forget_proactive_interference_reveals_working_memory_limits_in_llms_be/"
    t: "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length"
  - u: "understanding_synthetic_context_extension_via_retrieval_heads/"
    t: "Understanding Synthetic Context Extension via Retrieval Heads"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🧪 ICML2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×3

**[Don't Lag, RAG: Training-Free Adversarial Detection Using RAG](dont_lag_rag_training-free_adversarial_detection_using_rag.md)**

:   本文提出 VRAG 框架，通过构建对抗补丁数据库 + 视觉检索增强生成（VRAG）+ VLM 推理的免训练 pipeline，实现对多种对抗补丁攻击的高效检测，Gemini-2.0 达到 98% 准确率，开源模型 UI-TARS-72B-DPO 达 95%。

**[FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)**

:   FedRAG 提出了一个同时支持集中式和联邦式架构的 RAG 系统微调框架，填补了 RAG 生态系统中缺乏统一微调工具的空白，并通过轻量级抽象实现了从集中式到联邦式训练的无缝转换。

**[POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval](poqd_performance-oriented_query_decomposer_for_multi-vector_retrieval.md)**

:   提出 POQD，一个面向性能的查询分解框架，利用 LLM-based Prompt Optimizer 迭代优化查询分解 prompt，并通过交替训练算法联合优化 prompt 和下游 RAG 模型参数，在检索和端到端 QA 任务上大幅超越现有方法。

**[RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding](rapid_long-context_inference_with_retrieval-augmented_speculative_decoding.md)**

:   提出 RAPID，将 RAG 与 Speculative Decoding 结合：用 RAG drafter（在短检索上下文上运行的 LLM）为长上下文目标 LLM 生成候选 token，并通过推理时知识迁移增强目标分布，在长上下文推理中同时实现 >2× 加速和生成质量提升。

**[Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length](unable_to_forget_proactive_interference_reveals_working_memory_limits_in_llms_be.md)**

:   借鉴认知科学中的前摄干扰（Proactive Interference）范式，发现LLM的信息检索准确率随干扰信息量呈对数线性下降至零，揭示了一种独立于上下文长度的"工作记忆"容量瓶颈，且提示工程无法有效缓解。

**[Understanding Synthetic Context Extension via Retrieval Heads](understanding_synthetic_context_extension_via_retrieval_heads.md)**

:   本文通过系统实验揭示了合成上下文扩展（synthetic context extension）为何有效的机制：合成数据训练出的"检索头"（retrieval heads）与真实数据训练出的检索头高度重叠，检索头的召回率可以预测下游长上下文任务的性能，并通过注意力剔除（attention knockout）和激活修补（activation patching）从机制层面证明了检索头的必要性。

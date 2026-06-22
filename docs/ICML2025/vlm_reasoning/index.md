---
title: >-
  ICML2025 VLMReasoning论文汇总 · 5篇论文解读
description: >-
  5篇ICML2025的 VLM Reasoning 方向论文解读，涵盖推理、多模态、扩散模型、机器人、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "扩散模型"
  - "机器人"
  - "LLM"
item_list:
  - u: "diffusion-vla_generalizable_and_interpretable_robot_foundation_model_via_self-ge/"
    t: "Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning"
  - u: "overcoming_multi-step_complexity_in_multimodal_theory-of-mind_reasoning_a_scalab/"
    t: "Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner"
  - u: "re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models/"
    t: "Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger"
  - u: "reasoning_limitations_of_multimodal_large_language_models_a_case_study_of_bongar/"
    t: "Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems"
  - u: "why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc/"
    t: "Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**🧪 ICML2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×5 · 多模态 ×3

**[Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning](diffusion-vla_generalizable_and_interpretable_robot_foundation_model_via_self-ge.md)**

:   提出 DiffusionVLA (DiVLA)，将自回归 VLM 的推理能力与扩散模型的动作生成能力统一到一个端到端框架中，通过推理注入模块（Reasoning Injection Module）将自生成的语言推理直接嵌入策略学习过程，实现了对未见物体的泛化分类、可解释的动作决策以及高速推理（2B 模型 82Hz）。

**[Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner](overcoming_multi-step_complexity_in_multimodal_theory-of-mind_reasoning_a_scalab.md)**

:   提出一种可扩展的贝叶斯心智理论（ToM）规划器，通过将多步推理分解为逐步贝叶斯更新，并利用弱到强控制机制将小模型的 ToM 专项能力迁移至大模型（最高 405B），在多模态 ToM 基准上超越 SOTA 4.6%。

**[Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger](re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models.md)**

:   提出 RCTS 框架，通过自一致性评估机制构建推理上下文丰富的知识库，并用带启发式奖励的蒙特卡罗树搜索（MCTS-HR）重排检索示例，使 LVLM 在多个 VQA 数据集上显著超越 ICL 和 Vanilla-RAG 方法（平均 +3-4%）。

**[Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems](reasoning_limitations_of_multimodal_large_language_models_a_case_study_of_bongar.md)**

:   系统评估4个闭源+4个开源MLLM在经典合成Bongard Problems、Bongard HOI、Bongard-OpenWorld三个数据集上的抽象视觉推理能力，提出7种解题策略和新数据集Bongard-RWR（用真实图像表达合成BP概念），揭示MLLM在合成BP上的极差表现并非因域差异而是固有的抽象推理局限。

**[Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)**

:   从机制可解释性视角研究 VLM 空间推理失败的原因，发现图像 token 虽占输入 90% 但仅获 10% 注意力，且注意力的几何分布才是关键；提出 AdaptVis——基于推理时置信度自适应调整图像注意力温度的无训练解码方法，在 WhatsUp 上实现高达 50% 绝对提升。

---
title: >-
  NeurIPS2025 对话系统论文汇总 · 8篇论文解读
description: >-
  8篇NeurIPS2025的对话系统方向论文解读，涵盖 LLM、目标跟踪、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "目标跟踪"
  - "对齐/RLHF"
  - "对抗鲁棒"
item_list:
  - u: "aclora_almost_trainingfree_access_controlaware_multimodal_ll/"
    t: "AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs"
  - u: "agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in/"
    t: "Agentic Persona Control and Task State Tracking for Realistic User Simulation"
  - u: "bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap/"
    t: "Bridging Human and LLM Judgments: Understanding and Narrowing the Gap"
  - u: "hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location/"
    t: "HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location"
  - u: "kl_penalty_control_via_perturbation_for_direct_preference_optimization/"
    t: "KL Penalty Control via Perturbation for Direct Preference Optimization"
  - u: "latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia/"
    t: "LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation"
  - u: "less_is_more_local_intrinsic_dimensions_of_contextual_language_models/"
    t: "Less is More: Local Intrinsic Dimensions of Contextual Language Models"
  - u: "sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr/"
    t: "SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🧠 NeurIPS2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧪 ICML2025 (2)](../../ICML2025/dialogue/index.md) · [💬 ACL2025 (18)](../../ACL2025/dialogue/index.md)

🔥 **高频主题：** LLM ×2

**[AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)**

:   设计 AC-LoRA 端到端系统，为不同权限数据集训练独立的 LoRA 适配器，推理时根据用户查询的 cosine 相似度和权限动态检索并无训练合并多个 LoRA 输出，在保证强信息隔离的同时匹配或超越 SOTA LoRA 混合方法的回答质量。

**[Agentic Persona Control and Task State Tracking for Realistic User Simulation](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)**

:   提出三 agent 协作框架用于逼真的用户模拟——User Agent（协调）+ State Tracking Agent（结构化任务状态）+ Message Attributes Generation Agent（基于 persona 和状态的行为属性控制），在餐厅点餐场景中综合仿真质量（CRRS）提升 102.6%，persona 保持度 +19.9%，行为自然度 +284.5%，且核心发现：无状态感知的行为控制导致 BVS=0（完全刚性）。

**[Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)**

:   提出Bridge统计框架，通过序数logistic回归建模人类和LLM评判之间的潜在关系，以少量人类标签改善LLM评判的校准和对齐，同时支持对系统性偏差的正式统计检验。

**[HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)**

:   提出HyGen——干扰感知的LLM推理系统，通过精准的批次延迟预测器、SLO感知的性能分析器和前缀共享最大化调度策略，实现在线和离线工作负载的弹性共置，在保证严格SLO合规的同时获得3.87-5.84倍吞吐提升。

**[KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)**

:   提出 ε-DPO，通过观察训练时扰动 β 后 logit 作为偏好模型的单调性，实现实例级自适应 KL 惩罚控制，无需额外计算开销即可显著超越 DPO 及大多数直接对齐算法，在 AlpacaEval 2 上达到 46.4% LC win rate（DPO 仅 40.3%）。

**[LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)**

:   提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。

**[Less is More: Local Intrinsic Dimensions of Contextual Language Models](less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)**

:   提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。

**[SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)**

:   构建 SciArena 社区驱动的科学文献评估开放平台，采用 Chatbot Arena 式的人类偏好投票方式对 47 个基础模型进行排名，收集超过 20,000 条投票数据，并发布 SciArena-Eval 元基准来评测自动评估系统对文献任务答案质量的判断能力。

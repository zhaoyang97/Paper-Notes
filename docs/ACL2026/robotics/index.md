---
title: >-
  ACL2026 机器人/具身智能论文汇总 · 11篇论文解读
description: >-
  11篇ACL2026的机器人/具身智能方向论文解读，涵盖导航、多模态、推理、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "导航"
  - "多模态"
  - "推理"
  - "机器人"
item_list:
  - u: "breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat/"
    t: "Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents"
  - u: "cultivating_forensic_reasoning_for_generalizable_multimodal_manipulation_detecti/"
    t: "Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection"
  - u: "elasticflow_one-step_physics-consistent_policy_with_elastic_time_horizons_for_la/"
    t: "ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation"
  - u: "govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r/"
    t: "GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning"
  - u: "groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens/"
    t: "GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap"
  - u: "libra-vla_achieving_learning_equilibrium_via_asynchronous_coarse-to-fine_dual-sy/"
    t: "Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System"
  - u: "limited_linguistic_diversity_in_embodied_ai_datasets/"
    t: "Limited Linguistic Diversity in Embodied AI Datasets"
  - u: "mango_multi-agent_web_navigation_via_global-view_optimization/"
    t: "Mango: Multi-Agent Web Navigation via Global-View Optimization"
  - u: "vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr/"
    t: "VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions"
  - u: "when_does_language_matter_multilingual_instructions_reveal_step-wise_language_se/"
    t: "When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models"
  - u: "where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan/"
    t: "视觉语言导航代理的能力导向失败归因"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**💬 ACL2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (146)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/robotics/index.md) · [🧪 ICML2026 (53)](../../ICML2026/robotics/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md)

🔥 **高频主题：** 导航 ×5 · 多模态 ×3 · 推理 ×3 · 机器人 ×2

**[Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)**

:   SkillNav 把视觉语言导航任务拆解成 5 个原子技能（方向调整、垂直移动、停顿、地标识别、区域识别）+ 1 个时序规划技能，每个技能用合成数据微调一个 DUET 子 agent，再用 training-free 的 VLM router 做时序重排 + 子目标定位 + 技能选择，在 GSA-R2R 上取得 SOTA 泛化能力（Test-N-Scene SPL 48% vs. 之前最高 43%）。

**[Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection](cultivating_forensic_reasoning_for_generalizable_multimodal_manipulation_detecti.md)**

:   这篇论文提出 REFORM，把多模态伪造检测从“直接拟合标签”改成“学习可验证的取证推理过程”，并通过 ROM 推理标注数据集、双解码器和 GRPO 训练，在 ROM、DGM4 与 MMFakeBench 上取得更强的跨域泛化和可解释检测结果。

**[ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation](elasticflow_one-step_physics-consistent_policy_with_elastic_time_horizons_for_la.md)**

:   提出 ElasticFlow：用平均速度场 (MeanFlow) 取代瞬时速度场学习语言条件机器人动作，配合 "弹性时间区间 $\Delta t=t-r$" 显式编码控制粒度，实现 1-NFE 单步推理 (∼71Hz)，在 LIBERO-Long、CALVIN ABC-D 等长程任务上超过 OpenVLA 与 $\pi_0$。

**[GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)**

:   GoViG 提出一个**只靠第一视角初始与目标观测**就能生成导航指令的新任务，并把它拆成"先想象中间画面再写指令"两步，用 Anole-7B 在 token 级 MSE + 标签平滑 CE 双目标下联合训练，配合 one-pass / interleaved 两种多模态推理策略，把 BLEU-4 从基线 0.08 推到 0.32 并在跨域真实视频上保持 0.27。

**[GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)**

:   GROKE 提出**完全不用视觉**就评测导航指令好不好——把 OSM 地图序列化成 JSON，让 Gemini-3 Pro 当 follower agent 沿图执行指令，用 Navigation Error / SR / SDTW 反过来当指令质量的 proxy；相比启发式 baseline 在 Map2Seq 上降低 navigation error 68.5%，且 NE 与人类对"指令清晰度"的判断显著相关 ($r = -0.31, p < 0.01$)。

**[Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System](libra-vla_achieving_learning_equilibrium_via_asynchronous_coarse-to-fine_dual-sy.md)**

:   Libra-VLA 把机器人动作分解为"离散宏方向（macro-intent）+ 连续微姿态（micro-pose）"的混合动作空间，再用 System 2（VLM + 并行 coarse-action head）低频规划、System 1（diffusion transformer + 独立 SigLIP 编码器）高频精修，通过 intent buffer 实现真正异步执行，在 LIBERO 上拿到 97.2% SoTA、LIBERO-Plus 零样本 79.5%（比之前 OpenVLA-OFT+ 高 10 个点）。

**[Limited Linguistic Diversity in Embodied AI Datasets](limited_linguistic_diversity_in_embodied_ai_datasets.md)**

:   本文对主流 VLA 训练语料（RT-1、BRIDGE、TacoPlay、Language Table、LIBERO）做系统性"语言多样性体检"，从词汇/语义/句法三维度量化发现：VLA 数据**仅 < 2% 指令唯一、RT-1 整库只有 49 个 unique word、否定/条件句 < 1%**，远逊于指令调优语料（OASST2 93%、Alpaca 99.8% 唯一），这种"模板化贫乏"或许正是 VLA 模型对 paraphrase 脆弱、泛化失败的根源。

**[Mango: Multi-Agent Web Navigation via Global-View Optimization](mango_multi-agent_web_navigation_via_global-view_optimization.md)**

:   Mango 在网页导航前先构建网站的全局近似结构，再用 Thompson Sampling 在候选 URL 间动态分配有限导航预算，使 LLM web agent 不必总从首页盲目探索，并在 WebVoyager 和 WebWalkerQA 上显著超过 AgentOccam、WebWalker 等基线。

**[VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)**

:   本文提出 VLN-NF 基准——首个要求 VLN agent 在 3D 部分可观测环境中识别虚假前提指令并输出 NOT-FOUND 的任务，配套提出 REV-SPL 评估指标和 ROAM 两阶段混合框架，ROAM 达到 6.1 REV-SPL，比监督基线提升 45%。

**[When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models](when_does_language_matter_multilingual_instructions_reveal_step-wise_language_se.md)**

:   本文把 LIBERO 机器人操作基准翻成十种语言，首次系统揭示 VLA 模型在非英语指令下成功率暴跌 30–50%，并发现"语言影响在执行步上高度不均匀"——只有少数关键步对语言敏感却主导失败，据此提出一种只在这些步上做推理时表征对齐的方法，把多语言成功率大幅拉回。

**[视觉语言导航代理的能力导向失败归因](where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan.md)**

:   本文针对具身代理（特别是视觉语言导航 VLN 代理）的多层次能力失败问题，提出 CanTest 框架，通过能力导向的测试预言机与失败归因机制，精准定位导致任务失败的具体能力缺陷（感知/记忆/规划/决策），比现有方法发现的失败案例多 23–34%。

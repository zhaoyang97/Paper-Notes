---
title: >-
  ICML2026 自动驾驶论文汇总 · 8篇论文解读
description: >-
  8篇ICML2026的自动驾驶方向论文解读，涵盖自动驾驶、强化学习、导航等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "自动驾驶"
  - "论文解读"
  - "论文笔记"
  - "强化学习"
  - "导航"
item_list:
  - u: "coirl-ad_collaborative-competitive_imitation-reinforcement_learning_in_latent_wo/"
    t: "CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving"
  - u: "constrained_multi-objective_reinforcement_learning_with_max-min_criterion/"
    t: "Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion"
  - u: "deepsight_long-horizon_world_modeling_via_latent_states_prediction_for_end-to-en/"
    t: "DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving"
  - u: "mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm/"
    t: "Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering"
  - u: "plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation/"
    t: "Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation"
  - u: "roca_robust_cross-domain_end-to-end_autonomous_driving/"
    t: "RoCA: Robust Cross-Domain End-to-End Autonomous Driving"
  - u: "threshold-based_exclusive_batching_for_llm_inference/"
    t: "Threshold-Based Exclusive Batching for LLM Inference"
  - u: "tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_/"
    t: "TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🧪 ICML2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (157)](../../CVPR2026/autonomous_driving/index.md) · [🔬 ICLR2026 (50)](../../ICLR2026/autonomous_driving/index.md) · [🤖 AAAI2026 (56)](../../AAAI2026/autonomous_driving/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/autonomous_driving/index.md) · [📹 ICCV2025 (91)](../../ICCV2025/autonomous_driving/index.md) · [🧪 ICML2025 (10)](../../ICML2025/autonomous_driving/index.md)

🔥 **高频主题：** 自动驾驶 ×3 · 强化学习 ×2 · 导航 ×2

**[CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving](coirl-ad_collaborative-competitive_imitation-reinforcement_learning_in_latent_wo.md)**

:   CoIRL-AD 用两个独立的演员分别扛模仿学习（IL）和强化学习（RL）、靠一个潜空间世界模型"想象"未来轨迹来给 RL 算长程奖励，再用一套"谁强谁带谁"的竞争机制让两者互相传递有益行为，从而在**没有外部仿真器的离线真实驾驶数据**上把 RL 稳稳整合进端到端驾驶，在跨城泛化和长尾场景上取得显著提升。

**[Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](constrained_multi-objective_reinforcement_learning_with_max-min_criterion.md)**

:   本文把"max-min 多目标公平性"和"硬性约束满足"统一到同一个 MORL 框架中——通过占用测度 (occupancy measure) 重新表述为凸规划，再对偶出一个关于权重 $(u,w)$ 的凸优化问题，从而用一套投影梯度下降算法同时实现公平性和约束可行性，并给出几何收敛速率的理论保证。

**[DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving](deepsight_long-horizon_world_modeling_via_latent_states_prediction_for_end-to-en.md)**

:   DeepSight 把"未来世界预测"从显式像素重建（codebook 单帧）换成在 BEV 空间对 DINOv3 语义特征做**多帧并行隐式预测**，再叠加一个按需触发的 Adaptive Chain-of-Thought，让 Qwen2.5-VL-3B 在 Bench2Drive 闭环上 Driving Score 86.23 (+7.39)、Success Rate 71.36% (+13.63)，且只多 ~4% 推理延迟。

**[Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm.md)**

:   把无人机连续 VLN 的 step-by-step 预测重写成"递归贝叶斯估计 = GRU 先验 + 记忆库似然 + 可学习卡尔曼增益"的闭环, 在 TravelUAV 上仅用 10% 数据微调就把 L1-Full 的 SR 从 17.6% 推到 25.9%, 同时把 100 步后还在不断累积的位置漂移压平到 30–40 米。

**[Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)**

:   本文提出 PLMD：把 BEV 语义图与障碍图合并成 Label Map，用 DDPM 在障碍先验调制下补全未探索区域的语义+障碍标签，作为即插即用模块挂在任意 GON 策略上，在 ON / IIN / MRON 三类任务的 HM3D/MP3D 上一致刷新 SOTA。

**[RoCA: Robust Cross-Domain End-to-End Autonomous Driving](roca_robust_cross-domain_end-to-end_autonomous_driving.md)**

:   RoCA 给端到端自动驾驶模型挂一个基于高斯过程的即插即用模块——学一组覆盖多样驾驶场景的基础 token 及其对应轨迹，对新场景按相似度概率推断未来轨迹，既在源域训练时用 GP 的不确定性正则化提升泛化，又在新域上用伪标签和主动学习高效适应，无需 LLM、不增加推理开销。

**[Threshold-Based Exclusive Batching for LLM Inference](threshold-based_exclusive_batching_for_llm_inference.md)**

:   本文系统刻画了 LLM 推理中 mixed batching (MB) 与 exclusive batching (EB) 的性能交叉条件，证明带宽受限 GPU 上 prefill–decode 同批会因带宽争抢拖慢 Attention，进而推导出基于 hazard rate 的最优相位切换阈值 $\theta^*$ 和内存安全的批大小，并设计在线自适应调度器 EB+，在带宽受限硬件上吞吐最多提升 41.9%，非平稳流量下相对 MB 最多提升 36.4%。

**[TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)**

:   TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM，揭示出"scaling 在感知/推理上仍成立但在预测上失效"以及"文本与可视化模态高度互补但当前模型几乎无法融合"等关键结论。

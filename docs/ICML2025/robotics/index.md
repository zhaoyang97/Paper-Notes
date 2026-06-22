---
title: >-
  ICML2025 机器人/具身智能论文汇总 · 20篇论文解读
description: >-
  20篇ICML2025的机器人/具身智能方向论文解读，涵盖机器人、强化学习、Agent、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "强化学习"
  - "Agent"
  - "模型压缩"
item_list:
  - u: "action-constrained_imitation_learning/"
    t: "Action-Constrained Imitation Learning"
  - u: "beyond_cvar_leveraging_static_spectral_risk_measures_for_enhanced_decision-makin/"
    t: "Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning"
  - u: "biassemble_learning_collaborative_affordance_for_bimanual_geometric_assembly/"
    t: "BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly"
  - u: "closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling/"
    t: "Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling"
  - u: "commvq_commutative_vector_quantization_for_kv_cache_compression/"
    t: "CommVQ: Commutative Vector Quantization for KV Cache Compression"
  - u: "efficient_robotic_policy_learning_via_latent_space_backward_planning/"
    t: "Efficient Robotic Policy Learning via Latent Space Backward Planning"
  - u: "flow_of_reasoning_training_llms_for_divergent_reasoning_with_minimal_examples/"
    t: "Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples"
  - u: "founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci/"
    t: "FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making"
  - u: "geometric_contact_flows_contactomorphisms_for_dynamics_and_control/"
    t: "Geometric Contact Flows: Contactomorphisms for Dynamics and Control"
  - u: "gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in_onlin/"
    t: "Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning"
  - u: "graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning/"
    t: "Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning"
  - u: "hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti/"
    t: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
  - u: "learning_dynamics_under_environmental_constraints_via_measurement-induced_bundle/"
    t: "Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures"
  - u: "learning_to_stop_deep_learning_for_mean_field_optimal_stopping/"
    t: "Learning to Stop: Deep Learning for Mean Field Optimal Stopping"
  - u: "maximum_total_correlation_reinforcement_learning/"
    t: "Maximum Total Correlation Reinforcement Learning"
  - u: "robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism/"
    t: "Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism"
  - u: "sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world/"
    t: "SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models"
  - u: "sketch-plan-generalize_learning_and_planning_with_neuro-symbolic_programmatic_re/"
    t: "Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts"
  - u: "star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector/"
    t: "STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization"
  - u: "x_hacking_the_threat_of_misguided_automl/"
    t: "X-Hacking: The Threat of Misguided AutoML"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🧪 ICML2025** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (146)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/robotics/index.md) · [💬 ACL2026 (11)](../../ACL2026/robotics/index.md) · [🧪 ICML2026 (53)](../../ICML2026/robotics/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×5 · 强化学习 ×4 · Agent ×3 · 模型压缩 ×2

**[Action-Constrained Imitation Learning](action-constrained_imitation_learning.md)**

:   形式化了"动作约束模仿学习(ACIL)"新问题——受限Agent从无约束专家学习，提出DTWIL通过MPC+DTW距离生成替代性约束轨迹来消除占用度量失配，在多种机器人任务上显著优于基线。

**[Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](beyond_cvar_leveraging_static_spectral_risk_measures_for_enhanced_decision-makin.md)**

:   提出首个在分布式 RL 框架内优化一般静态谱风险度量（SRM）的算法，超越了仅限于简单 CVaR 的现有方法，通过利用回报分布实现闭式外层优化和中间风险度量的时间分解，在多种风险设置中超越现有风险敏感 DRL 模型。

**[BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly](biassemble_learning_collaborative_affordance_for_bimanual_geometric_assembly.md)**

:   提出 BiAssemble 框架，通过学习感知双臂协作的点级可供性（affordance），将几何装配任务分解为抓取→对齐→装配三步，在破碎物体重组任务上超越现有可供性和模仿学习方法，并在真实世界基准上验证。

**[Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)**

:   将 LLM 的自精炼规划过程建模为不动点问题（深度均衡模型），通过隐式微分实现端到端监督训练，无需额外验证器或 RL，并设计嵌套均衡求解实现闭环长程机器人规划。

**[CommVQ: Commutative Vector Quantization for KV Cache Compression](commvq_commutative_vector_quantization_for_kv_cache_compression.md)**

:   提出 CommVQ——通过可加向量量化压缩 KV cache，创新性地设计与 RoPE 可交换的码本并用 EM 算法训练，在 2-bit 下几乎无损、1-bit 下仍保持可用精度，使 LLaMA-3.1 8B 在单张 RTX 4090 上支持 128K 上下文。

**[Efficient Robotic Policy Learning via Latent Space Backward Planning](efficient_robotic_policy_learning_via_latent_space_backward_planning.md)**

:   提出潜在空间反向规划（LBP），从最终目标出发递归预测越来越接近当前状态的中间子目标，在保持任务对齐的同时大幅提升规划效率，在 LIBERO-LONG 仿真和真实机器人长时域任务上达到 SOTA。

**[Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](flow_of_reasoning_training_llms_for_divergent_reasoning_with_minimal_examples.md)**

:   提出 Flow of Reasoning (FoR)，将多步 LLM 推理建模为 DAG 上的马尔可夫流，借助 GFlowNet 的轨迹平衡目标微调 LLM，使其仅用极少训练样本（如15个）即可采样出概率正比于奖励的多条高质量且多样化的推理路径。

**[FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)**

:   提出 FOUNDER 框架，通过学习映射函数将 Foundation Model (FM) 的多模态任务表示对齐到 World Model (WM) 的状态空间，结合时间距离预测器生成奖励信号，实现无需环境奖励的开放式多任务具身决策。

**[Geometric Contact Flows: Contactomorphisms for Dynamics and Control](geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)**

:   提出 Geometric Contact Flows (GCF)，利用黎曼几何和接触几何作为归纳偏置，通过接触微分同胚（contactomorphisms）将具有稳定性/能量守恒等期望性质的潜在接触哈密顿动力学映射到目标动力学，同时利用集成不确定性驱动测地线实现鲁棒泛化和避障。

**[Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in_onlin.md)**

:   提出 Annealed Q-learning (AQ-L)，通过期望分位损失（expectile loss）的参数 τ 从接近1退火至0.5，实现从 Bellman 最优算子到 Bellman 算子的平滑过渡，在连续动作空间中既加速了早期学习又抑制了后期过估计偏差，与 TD3/SAC 结合后在多种运动控制和操控任务上显著优于基线。

**[Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)**

:   提出 Graph-Assisted Stitching (GAS) 框架，用基于图搜索的子目标选择替代显式高层策略学习，通过时间距离表示 (TDR) 空间中的聚类构图与最短路径规划，在离线 HRL 中实现高效的跨轨迹拼接，在最具挑战的 antmaze-giant-stitch 任务上从前 SOTA 的 1.0 飙升至 88.3。

**[Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)**

:   提出 Hi Robot，一个层次化 VLM 系统：高层 VLM 将复杂用户指令/反馈推理为原子命令，低层 VLA (π0) 执行动作，结合合成数据生成方案，在三类机器人平台上实现了远超 GPT-4o 和扁平 VLA 的开放式指令跟随能力。

**[Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures](learning_dynamics_under_environmental_constraints_via_measurement-induced_bundle.md)**

:   提出一种几何框架，利用测量过程自然诱导的纤维丛结构统一处理测量不确定性、系统约束和动力学学习：在纤维丛上定义测量感知控制屏障函数(mCBF)，结合Neural ODE学习连续时间动力学，在三个机器人控制任务上实现96.3%成功率和99.3%约束满足率。

**[Learning to Stop: Deep Learning for Mean Field Optimal Stopping](learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)**

:   首次在离散时间有限状态空间下形式化并计算求解平均场最优停止（MFOS）问题，证明 MFOS 以 $O(1/N)$ 速率逼近多智能体最优停止（MAOS），并提出两种深度学习算法（直接法 DA 和动态规划法 DPP），在维度高达 300 的 6 个场景中验证有效性。

**[Maximum Total Correlation Reinforcement Learning](maximum_total_correlation_reinforcement_learning.md)**

:   提出最大化轨迹总相关（Total Correlation）作为 RL 的归纳偏置，鼓励策略产生简单、可压缩的轨迹，从而在不牺牲任务性能的前提下显著提升对观测噪声、动作噪声和动力学变化的零样本鲁棒性。

**[Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism.md)**

:   提出自适应干预机制 AIM，通过学习代理 Q 函数模拟人类干预决策，让机器人主动请求专家帮助，相比不确定性基线 Thrifty-DAgger 在人类接管成本和学习效率上提升 40%。

**[SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)**

:   提出 SENSEI 框架：利用 VLM 成对比较观测图像的"有趣程度"，蒸馏出语义内在奖励，再与集成不确定性驱动的新颖性奖励结合，通过世界模型实现语义有意义的无任务探索，并显著加速下游任务学习。

**[Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts](sketch-plan-generalize_learning_and_planning_with_neuro-symbolic_programmatic_re.md)**

:   提出 SPG（Sketch-Plan-Generalize）——一种神经符号智能体框架，将归纳式概念学习分解为三阶段流水线：概念签名推断（Sketch）、基于 MCTS 的 grounded 动作序列搜索（Plan）、以及 LLM 驱动的程序归纳泛化（Generalize），在从少量演示中学习可组合、可泛化的空间抽象概念方面显著优于纯 LLM 和纯神经方法。

**[STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization](star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector.md)**

:   提出STAR框架，通过旋转增强残差技能量化（RaRSQ）解决VQ-VAE的codebook坍塌问题，并通过因果技能Transformer（CST）建模技能间依赖关系，在LIBERO基准上整体成功率达93.6%，比此前SOTA QueST提升约12%。

**[X-Hacking: The Threat of Misguided AutoML](x_hacking_the_threat_of_misguided_automl.md)**

:   揭示了XAI(可解释AI)领域的新安全威胁"X-hacking"：通过AutoML的管道搜索能力，对抗者可在Rashomon模型集中寻找支持预定结论的解释性结果，Bayesian优化比随机搜索快3倍。

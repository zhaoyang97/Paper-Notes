---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🧠 NeurIPS2025** · 共 **21** 篇

**[A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications](a_generalized_bisimulation_metric_of_state_similarity_betwee.md)**

:   将传统只能在单个MDP内度量状态相似性的bisimulation metric (BSM)推广到跨MDP场景，提出广义双模拟度量(GBSM)，严格证明了对称性、跨MDP三角不等式和同状态距离上界三个基本度量性质，并在策略迁移、状态聚合和基于采样的估计三个应用中推导出比标准BSM更紧的误差界和闭式样本复杂度。

**[A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)**

:   提出 BARBAT 框架，改进了经典的 BARBAR 算法，通过固定 epoch 长度和逐 epoch 调整失败概率，将对抗腐蚀下随机多臂老虎机的 regret 从 $O(\sqrt{K}C)$ 降至近最优的 $O(C)$（消除了 $\sqrt{K}$ 因子），并成功扩展到多智能体、图老虎机、组合半老虎机和批量老虎机等多种场景。

**[A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)**

:   提出多智能体生成流网络（MA-GFlowNets）的理论框架，证明了"局部-全局原理"——联合流函数可分解为各智能体独立流的乘积形式，设计了四种算法（CFN/IFN/JFN/CJFN），其中 JFN 和 CJFN 实现中心化训练+去中心化执行（CTDE），在 Hyper-Grid 和 StarCraft 环境中超越 RL 和 MCMC 方法。

**[A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](a_unifying_view_of_linear_function_approximation_in_offpolic.md)**

:   将线性函数逼近下的TD、FQI和PFQI统一为求解同一线性系统的迭代方法（仅预条件子不同），首次引入矩阵分裂理论来分析它们的收敛性，给出了各算法收敛的充要条件，并揭示了TD收敛不一定意味着FQI收敛（反之亦然）。

**[Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)**

:   提出 DP-AG（Action-Guided Diffusion Policy），通过将扩散策略的噪声预测的 Vector-Jacobian Product (VJP) 作为结构化随机力驱动隐观测特征在扩散步骤间动态演化，并用循环一致对比损失闭合感知-动作环路，在 Push-T 上提升 6%、Dynamic Push-T 上提升 13%、真实 UR5 机器人上成功率提升 23%+。

**[Actor-Free Continuous Control via Structurally Maximizable Q-Functions](actorfree_continuous_control_via_structurally_maximizable_qf.md)**

:   提出Q3C（Q-learning for Continuous Control with Control-points），一种无actor的纯基于值函数的连续控制方法，通过控制点插值逼近任意形状的Q函数，在复杂（非凸、受限）Q函数景观中显著优于actor-critic方法。

**[Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)**

:   提出ANQ（Adaptive Neighborhood Q-learning），在离线RL中引入基于优势函数的自适应邻域约束，在密度约束（过于保守）和支持约束（建模困难）之间找到灵活的中间地带，在D4RL AntMaze上达到87.5分。

**[Adaptively Coordinating with Novel Partners via Learned Latent Strategies](adaptively_coordinating_with_novel_partners_via_learned_latent_strategies.md)**

:   提出 TALENTS 框架，通过 VAE 学习潜在策略空间 + K-Means 聚类发现策略类型 + Fixed-Share 遗憾最小化算法在线推断队友类型，实现对未知人类/智能体队友的零样本实时适应协作。

**[Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)**

:   从 token 熵模式的全新视角分析 RLVR，发现 CoT 推理中仅约 20% 的高熵"分叉 token"决定推理方向，仅在这些 token 上做梯度更新即可匹配甚至大幅超越全量更新（Qwen3-32B 上 AIME'25 +11.04），揭示 RLVR 本质是优化推理决策点。

**[Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)**

:   混合二次(softmax attention)和线性(DeltaNet)Transformer，各司其职：前者精确检索，后者长上下文和复杂计算，在1.3B模型上实现性能和效率均衡。

**[Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)**

:   提出 Reinforcement Learning from Checklist Feedback (RLCF)，将指令分解为动态生成的 yes/no checklist，结合 AI judge 和代码验证器逐项评分后做 DPO 训练，在 5 个 benchmark 上一致性提升 Qwen2.5-7B-Instruct，是唯一在所有 benchmark 上都有正收益的方法（FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt）。

**[Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)**

:   SambaY 提出 Gated Memory Unit（GMU）用于跨层共享 SSM 的 token 混合表示，将 YOCO 的 cross-decoder 中一半的 cross-attention 层替换为轻量级 GMU，在保持线性预填充复杂度和长上下文检索能力的同时，大幅提升解码效率——最终产品 Phi4-mini-Flash-Reasoning (3.8B) 在推理任务上超越 Phi4-mini-Reasoning，且在 2K 提示 + 32K 生成场景下实现高达 10× 的解码吞吐提升。

**[Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)**

:   HRPO 提出混合潜在推理策略优化：通过可学习的门控机制将前一步的隐藏状态表示逐步融入到采样的 token embedding 中，使 LLM 在推理阶段同时利用离散 token 和连续潜在表示，无需 CoT 标注即可通过 RL 训练，在知识密集型和 STEM 推理任务上均超越 PPO/GRPO 等基线。

**[Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)**

:   提出"木偶师"(Puppeteer)式多 Agent 协作范式——一个中心化编排器通过 RL 学习在每个推理步骤动态选择激活哪个 Agent，在封闭域和开放域任务上同时提升性能和效率，并发现演化后的拓扑趋向更紧凑的环形结构。

**[NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)**

:   提出NoisyRollout，一种简单有效的数据增强方法——在VLM的RL训练中混合使用干净图像和适度扭曲图像的生成轨迹，通过注入感知多样性促进策略探索和鲁棒推理，配合噪声退火调度，零额外计算成本实现5个域外推理benchmark上的开源RL模型SOTA。

**[Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)**

:   发布Reasoning Gym库，包含100+可验证推理任务的过程生成环境，支持动态难度调整和无限数据生成，可用于RLVR训练和推理评估。

**[Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)**

:   展示 RL 训练的 14B 参数搜索 agent 在法律文档检索任务上通过多轮交互可以超越 frontier 模型（85% vs GPT o3 的 81%），关键在于精心设计的分段奖励结构和允许长 horizon 多轮交互。

**[RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)**

:   Tango 提出一种交替 RL 训练生成器和验证器的框架——验证器是生成式过程级 LLM（用自然语言逐步评判），仅用结果级正确性奖励训练（无需步骤标注），通过与生成器的共进化相互增强——在 7B/8B 级别模型上达到SOTA，AIME 2025 准确率相对 vanilla GRPO 提升 100%。

**[Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)**

:   提出 SRL 框架，将组合优化求解器嵌入 actor-critic 的 actor 网络中，通过 Fenchel-Young 损失实现端到端可微学习，在6个工业级组合决策问题上超越模仿学习18×、超越无结构RL 92%。

**[SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)**

:   首次将强化学习 (RL) 应用于真实世界软件工程任务（GitHub PR/Issue 修复），仅用基于规则的序列相似度奖励训练 Llama-3.3-70B，在 SWE-bench Verified 上达到 41.0% 解决率（中等规模模型 SOTA），且 RL 训练仅在 issue-solving 数据上进行，却涌现出在代码推理、数学、通用语言理解等域外任务上的泛化推理能力。

**[Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)**

:   通过在 RL 奖励中加入长度惩罚项——正确回答的奖励乘以 $(1 - \alpha \cdot \sigma(\text{norm\_len}))$，用单一超参数 $\alpha$ 控制 token-准确率权衡曲线，仅 100 步 RL 训练即可让 7B 推理模型减少 50% token 使用量而准确率仅下降 <5%。

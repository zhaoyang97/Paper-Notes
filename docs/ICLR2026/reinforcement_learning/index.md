<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🔬 ICLR2026** · 共 **46** 篇

**[AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)**

:   提出 AbstRaL，通过强化学习教 LLM 学习推理问题的数学抽象（将具体数字/名称替换为符号变量、提取通用公式），然后用符号求解器推导答案，在 GSM 扰动 benchmark 上几乎完全消除了分布偏移导致的性能下降，并在 OOD 数学/通用推理任务上也有隐式提升。

**[APPLE: Toward General Active Perception via Reinforcement Learning](apple_toward_general_active_perception_via_reinforcement_learning.md)**

:   提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分，基于off-policy算法（SAC/CrossQ）和共享ViViT骨干网络，在5个不同任务基准上验证通用性，其中CrossQ变体无需逐任务调参且训练效率提高53%。

**[ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)**

:   提出ARM-FM框架，利用基础模型（GPT-4o等）从自然语言任务描述自动生成语言对齐奖励机器（LARM）——包括自动机结构、可执行标签函数和每个状态的自然语言描述——为RL agent提供组合式密集奖励信号，在MiniGrid/Craftium(3D Minecraft)/Meta-World等环境中解决标准RL完全无法学习的稀疏奖励长程任务，并实现零样本任务泛化。

**[Autoqd Automatic Discovery Of Diverse Behaviors With Quality-Diversity Optimizat](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)**

:   提出 AutoQD，利用占用度量 (occupancy measure) 的随机 Fourier 特征嵌入自动生成行为描述子 (behavioral descriptor)，替代传统 QD 优化中的手工设计描述子，在 6 个连续控制任务上展现了强大的多样化策略发现能力。

**[Autotool Automatic Scaling Of Tool-Use Capabilities In Rl Via Decoupled Entropy ](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)**

:   提出解耦自适应熵约束 (Decoupled Adaptive Entropy Constraints) 的强化学习策略，使 LLM 在工具调用任务中根据问题难度自动切换长/短推理模式，在提升 9.8% 准确率的同时减少约 81% 的推理 token 开销。

**[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)**

:   提出 AWM，一种无需训练的 LLM 权重矩阵指纹方法，利用线性分配问题（LAP）恢复嵌入层的置换和符号翻转，再用无偏 CKA 消除 Q/K 矩阵的正交变换影响，在 150 对 LLM 上实现完美 AUC（1.0），对 SFT、持续预训练（5.5T token）、RL、多模态扩展、剪枝、upcycling 六类后训练均鲁棒，30 秒内完成。

**[BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)**

:   首次将贝叶斯自适应 MDP（BAMDP）引入离线模型基 RL，提出 Continuous BAMCP 解决连续状态/动作空间的贝叶斯规划，结合悲观奖励惩罚和搜索基策略迭代（"RL + Search"范式），在 D4RL 12 个任务上显著超越 19 个基线（Cohen's $d > 1.8$），并成功应用于核聚变 tokamak 控制。

**[Boolean Satisfiability via Imitation Learning](boolean_satisfiability_via_imitation_learning.md)**

:   提出 ImitSAT，首个基于模仿学习的 CDCL 求解器分支策略：通过将求解器运行压缩为无冲突的 KeyTrace 专家序列，将分支决策建模为前缀条件的自回归预测任务，以少量查询预算显著减少传播次数和求解时间，并在结构化 SAT 问题上展现良好泛化能力。

**[Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)**

:   通过观察性研究（18 个开源 RPT 模型）和干预性研究（单域 GRPO 训练），系统揭示了强化后训练（RPT/RLVR）的泛化局限：RPT 在训练域内提升显著，但跨域泛化不一致——结构化域（数学↔代码）可互相迁移，但无法泛化到非结构化域（法律/金融/医疗），且这一结论跨算法、模型规模和训练步数保持一致。

**[Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)**

:   针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。

**[Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)**

:   提出 Chain-of-Context Learning (CCL)，通过 Relevance-Guided Context Reformulation（RGCR，自适应聚合约束信息构建上下文）和 Trajectory-Shared Node Re-embedding（TSNR，跨轨迹共享节点更新避免冗余计算）实现逐步动态的约束感知解码，在 48 种 VRP 变体（16 分布内 + 32 分布外）上全面超越现有方法。

**[Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)**

:   Co-rewarding 提出自监督 RL 框架，通过数据侧（对比改写问题的跨视角一致性）和模型侧（EMA 教师模型提供伪标签）两种互补监督方式，解决自奖励 RL 中的训练崩溃问题，在无人工标签条件下多项数学推理基准上达到甚至超过 RLVR（有标签）的性能。

**[Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)**

:   提出 VIP（Value Iteration via PINN）框架，首次将物理信息神经网络（PINN）用于求解连续时间多智能体强化学习中的 HJB 偏微分方程，并引入 Value Gradient Iteration（VGI）模块迭代精炼价值梯度，在连续时间 MPE 和 MuJoCo 多智能体任务上始终优于离散时间和连续时间基线。

**[Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)**

:   CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。

**[Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)**

:   系统研究跨形态离线 RL 预训练范式，发现次优数据比例和机器人多样性增加时梯度冲突导致负迁移，提出基于形态图距离的 Embodiment Grouping（EG）策略将机器人按形态聚类后分组更新 actor，在 16 种机器人平台的 locomotion benchmark 上显著缓解负迁移（70% 次优数据集上 IQL+EG 比 IQL 提升 34%）。

**[CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)**

:   提出 CUDA-L1，一个基于对比强化学习（Contrastive RL）的三阶段流水线框架，将初始 CUDA 能力较弱的 LLM 训练为高效的 CUDA 优化器，在 KernelBench 的 250 个 CUDA 内核上实现平均 3.12× 加速，峰值达 120×，并可跨 GPU 架构迁移。

**[Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)**

:   构建了安全策略改进（SPI）的理论框架，将世界模型和表示学习与策略更新保证统一起来：通过基于重要性比率的邻域算子约束策略更新，确保单调改进和收敛；结合局部转移/奖励损失控制世界模型质量和表示稳定性，提出 DeepSPI 算法在 ALE-57 基准上匹配或超越 PPO 和 DeepMDP。

**[Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)**

:   首次同时解决跨域离线 RL 的"训练时鲁棒性"（源域-目标域不匹配）和"测试时鲁棒性"（部署环境动态偏移）：提出 DROCO，通过 Robust Cross-Domain Bellman (RCB) 算子对源域数据施加鲁棒 Bellman 更新、对目标域数据施加标准更新，将动态不确定性映射为可处理的状态扰动。

**[Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)**

:   提出核代理模型（KernelSM）用于任务归因，通过 RBF 核岭回归捕获任务间的非线性交互效应，结合梯度投影的高效估计算法避免重复训练，在数学推理、上下文学习和多目标 RL 等场景下相比线性代理和影响函数基线提升 25% 相关性。

**[Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)**

:   本文揭示了策略梯度 RL 算法在 LLM 后训练中系统性导致策略熵坍缩的理论根因（优势函数与对数概率的正相关性），并提出两种互补的解法：REPO（通过修改优势函数去相关）和 ADAPO（自适应非对称裁剪），在交互式工具使用任务上实现 SOTA 性能。

**[Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)**

:   本文揭示 RLVR 中两个反直觉现象——随机奖励也能提升推理性能、熵最小化也能改善结果——并从理论上证明裁剪偏差提供的学习信号可忽略不计，性能提升的真正来源是裁剪对熵的隐式压缩作用和奖励误标对强模型的"有益偏差"效应。

**[Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)**

:   FAC 首次联合利用流模型（continuous normalizing flow）同时构建表达力强的 actor 策略和基于精确密度估计的 critic 惩罚机制，通过识别 OOD 区域对 Q 值进行选择性保守估计，在 OGBench 55 个任务上以 60.3 平均分大幅超越此前最佳的 43.6。

**[Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)**

:   证明在外生MDP（Exo-MDP，不确定性仅来自独立于智能体动作的外生输入）中，纯利用（无探索）策略即可达到次线性遗憾界——表格情形下PTO算法达到 $\tilde{O}(H^2|\Xi|\sqrt{K})$，线性函数逼近下LSVI-PE算法遗憾与特征维度和外生状态空间多项式相关、与内生状态/动作空间无关。

**[Learning to Orchestrate Agents in Natural Language with the Conductor](learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)**

:   用RL训练7B的Conductor模型，通过自然语言输出Agent工作流(子任务分配+通信拓扑)来协调GPT-5/Claude/Gemini等大模型，在LiveCodeBench和GPQA等benchmark上超越所有单模型和多Agent基线，达到SOTA(平均77.27 vs GPT-5的74.78)。

**[LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)**

:   提出 LongRLVR，通过在 RLVR 训练中引入可验证的上下文奖励（context reward），解决长上下文场景下仅靠最终答案奖励导致的上下文定位（grounding）梯度消失问题，显著提升 LLM 长上下文推理能力。

**[LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](loongrl_rl_for_reasoning_long_contexts.md)**

:   提出 LoongRL，通过构建 KeyChain 合成数据进行强化学习训练，使 LLM 涌现出 plan–retrieve–reason–recheck 的长上下文推理模式，仅在 16K 上下文上训练即可泛化到 128K，14B 模型达到 74.2 分接近 o3-mini (74.5) 和 DeepSeek-R1 (74.9)。

**[ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)**

:   ROMI 通过 Wasserstein 对偶将动力学不确定集转化为状态不确定集来实现鲁棒的价值感知模型学习，并用隐式可微的自适应加权机制平衡动力学精度与价值感知，解决了 RAMBO 方法中的 Q 值低估和梯度爆炸问题，在 D4RL 和 NeoRL 上达到模型基离线 RL 的 SOTA。

**[MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)**

:   提出 MVR 框架，利用多视角视频的视频-文本相似度学习状态相关性函数，结合状态依赖的奖励塑形（自动衰减 VLM 引导），在 HumanoidBench 和 MetaWorld 共 19 个任务上超越现有 VLM 奖励方法。

**[On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)**

:   首次证明交替梯度下降上升（AltGDA）在有约束双线性零和博弈中以 $O(1/T)$ 速率收敛到Nash均衡（存在内部NE时），比同步GDA的 $O(1/\sqrt{T})$ 快，用能量函数衰减刻画轨迹碰撞边界时的"摩擦"效应，并通过性能估计编程（PEP）进一步优化步长。

**[One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)**

:   提出 ScaleZero，通过在统一世界模型中引入 MoE 架构解决多任务学习中的梯度冲突和可塑性崩塌问题，结合动态参数扩展（DPS）策略自适应分配模型容量，单个多任务模型在 Atari/DMC/Jericho 三个基准上达到与单任务专家模型相当的性能，同时减少约 28.5% 的环境交互。

**[Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)**

:   将社交网络中极化与分歧最小化问题建模为在线低秩矩阵bandit问题，提出两阶段算法OPD-Min-ESTR（先估计子空间再低维线性bandit），将维度从 $|V|^2$ 降至 $O(|V|)$，实现 $\tilde{O}(\max\{1/\kappa, \sqrt{|V|}\}\sqrt{|V|T})$ 累积遗憾。

**[Optimistic Task Inference for Behavior Foundation Models](optimistic_task_inference_behavior_models.md)**

:   提出 OpTI-BFM——在 Behavior Foundation Model 测试时，不需要完整奖励函数或标注数据集，而是通过与环境交互仅 5 个 episode 即可推断任务并恢复 Oracle 性能，核心是利用 successor features 的线性结构将任务推断归约为线性 bandit 问题并用 UCB 策略乐观探索，提供正式的 regret bound。

**[Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)**

:   提出 DQO（Diversity Quality Optimization），基于行列式点过程（DPP）在语义嵌入空间中定义多样性度量，将其与奖励信号联合优化，使 LLM 后训练同时提升语义多样性和响应质量，可叠加在 GRPO/PPO 之上。

**[Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)**

:   受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。

**[Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)**

:   用GRPO强化学习训练LLM从政治调查数据中学习推理式观点对齐，在美国/德国/瑞士三个数据集上证明推理能提升个体级政治观点建模的准确性。

**[Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)**

:   提出 RGSC（Regret-Guided Search Control）框架，通过训练一个 regret 网络识别高遗憾值状态并优先从这些状态重新开始自我对弈，模拟人类"反复复盘错误"的学习方式，在 9×9 围棋、10×10 黑白棋和 11×11 Hex 上平均超越 AlphaZero 77 Elo。

**[Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)**

:   本文研究 RL 中一种新型威胁——行为目标攻击（adversary 通过篡改观测来引导 victim 执行特定目标策略），提出不需要白盒访问的 BIA 攻击方法和基于时间折扣的 TDRT 防御方法，TDRT 在保持对攻击鲁棒性的同时比现有防御（SA-PPO）的原始任务性能高 28.2%。

**[Routing, Cascades, and User Choice for LLMs](routing_cascades_and_user_choice_for_llms.md)**

:   将 LLM 路由建模为 provider-user Stackelberg 博弈，证明最优路由策略几乎总是静态无级联的阈值规则，并揭示当模型质量排序与成本排序不一致时产生的用户-提供商不对齐问题，以及低流失惩罚下 provider 有动机通过增加延迟来降低成本。

**[Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)**

:   提出 Self-Harmony 框架，通过让单一模型扮演两个角色（Solver 求解原始问题 + Reframer 改述问题），将答案在原始和改述视角下的调和平均得分作为伪标签选择标准，替代传统多数投票，在 30 个实验设置中 28 个达到 SOTA，且训练零失败。

**[Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)**

:   提出 SISL（Self-Improving Skill Learning），通过解耦高层策略和技能改进策略，结合最大回报重标注的技能优先级机制，在噪声离线演示数据下实现鲁棒的技能学习，显著提升基于技能的元强化学习在长时域任务中的性能。

**[Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)**

:   证明单边信息二人零和微分博弈中 Nash 均衡策略的原子结构——知情玩家 P1 的均衡策略集中在至多 $I$ 个动作原型上（$I$ = 博弈类型数），使博弈树复杂度从 $U^{2K}$ 降到 $I^K$，在美式足球 11v11 连续动作空间中（传统复杂度 $10^{440}$）实现 M1 MacBook 30 分钟求解。

**[UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)**

:   提出 UME-R1，首次探索推理驱动的生成式多模态嵌入范式，通过两阶段训练（冷启动SFT + 强化学习）让嵌入模型先推理再生成表示，在 MMEB-V2 基准的 78 个任务上显著超越传统判别式嵌入模型。

**[Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)**

:   通过形式化梯度分析揭示双曲深度 RL 的训练不稳定根源（大范数嵌入导致信赖域违反），提出 Hyper++ 三组件方案（RMSNorm + 学习缩放 + 分类值损失）实现稳定训练并超越现有方法。

**[Value Flows](value_flows.md)**

:   Value Flows 首次将流匹配（flow matching）引入分布式 RL——学习一个向量场使生成的概率密度路径自动满足分布式 Bellman 方程，通过 flow derivative ODE 高效估计回报方差实现置信度加权优先学习，在 OGBench 62 个任务上平均 1.3× 成功率提升，回报分布估计精度比 C51/CODAC 好 3×+。

**[VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)**

:   针对大型推理模型（LRM）训练中广泛使用的基于参考答案的奖励系统，构建了 VerifyBench 和 VerifyBench-Hard 两个评测基准，通过严格的人工标注评估各类验证系统的准确性，发现即使最强模型在困难样本上也仅达约 88% 准确率，揭示了当前验证系统的显著改进空间。

**[Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)**

:   提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

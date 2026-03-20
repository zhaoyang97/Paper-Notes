<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🔬 ICLR2026** · 共 **13** 篇

**[Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)**

:   提出 VIP（Variance-Informed Predictive allocation），通过高斯过程预测每个 prompt 的成功概率，据此用凸优化在计算预算约束下分配 rollout 数量以最小化梯度方差，在数学推理任务上一致提升 GRPO/RLOO 的采样效率，AIME24/25 上 Pass@32 最高提升 12.3 个点。

**[CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)**

:   CogFlow 提出认知启发的三阶段视觉数学推理框架（感知→内化→推理），通过 Synergistic Visual Rewards 增强感知、Knowledge Internalization Reward 桥接感知与推理、Visual-Gated Policy Optimization 锚定视觉推理，解决了现有方法中"感知正确但推理漂移"的核心问题。

**[Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)**

:   从凸优化理论出发，证明深度学习训练损失以 O(1/sqrt(T)) 速率收敛，最优学习率以 1/sqrt(T) 缩放，在 GPT-2 到 12.5B 参数模型上验证了该缩放律（R^2 >= 0.978），并实现了 80 倍训练步数的学习率外推。

**[Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)**

:   揭示了一个让人惊讶的等价性：广泛使用的双优化上升（Dual Optimistic Ascent / PI 控制）在数学上等价于增广拉格朗日方法（ALM），从而将 RL/RLHF 中的约束优化与经典优化理论统一起来，提供了更强的收敛保证和实用参数调优指南。

**[Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)**

:   提出 STARS，通过 Stiefel 流形上联合优化正交激活转向方向，使并行推理链产生多样化的推理路径（而非仅表面 token 变化），在测试用例生成和科学发现中超越温度/核采样，同时维护质量。

**[FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)**

:   FrontierCO 是一个涵盖 8 类组合优化问题（TSP、MIS、CVRP 等）的大规模真实世界基准测试，评估了 16 个 ML 求解器（神经网络方法 + LLM Agent）与 SOTA 传统求解器的差距，发现 ML 方法在结构复杂和极大规模实例上仍显著落后于传统方法，但在部分场景有超越潜力。

**[∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)**

:   提出 ∇-Reasoner，将推理时的搜索从零阶（采样+评估）升级为一阶（梯度下降），在 token logits 空间上通过可微文本优化（DTO）结合 reward 梯度和 LLM 似然来迭代改进解码策略，在数学推理任务上提升 10-40% 准确率的同时减少 10-40% 的模型调用次数。

**[Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)**

:   证明在通用非退化假设下，标准两层神经网络通过分层梯度下降可以用 $\tilde{O}(d)$ 样本和 $\tilde{O}(d^2)$ 时间学习通用高斯 Multi-Index 模型 $f(\bm{x})=g(\bm{U}\bm{x})$，样本和时间复杂度都达到信息论最优，首次证明神经网络可以高效学习层次化函数。

**[Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)**

:   提出 In-Context Policy Optimization (ICPO) 框架，理论证明单层线性自注意力 Transformer 经充分预训练后可在上下文中模拟策略优化算法，并设计实用的 ME-ICPO 算法通过最小熵选择和自评估奖励实现测试时多轮自反思，在数学推理任务上取得显著提升（AIME 2024 上 Qwen2.5-Math-7B 从 11% 提升到 30%）。

**[RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)**

:   提出 RRNCO 架构，通过自适应节点嵌入（ANE）和神经自适应偏置（NAB）两大创新，首次在深度路由框架中联合建模非对称距离、时长和方向角，并构建了基于 100 个真实城市的 VRP 基准数据集，显著缩小了 NCO 方法从仿真到真实世界部署的差距。

**[Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)**

:   提出 Scaf-GRPO 框架，通过分层 in-prompt hint 注入（知识→规划→求解）解决 RLVR 中的"学习悬崖"问题——当模型对难题持续零奖励时，以最小引导恢复学习梯度，在 AIME24 上相对 vanilla GRPO 提升 44.3%。

**[Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)**

:   提出MASS框架，通过双层优化元学习让LLM在推理时自动生成问题特定的合成训练数据并自更新(LoRA)，在MATH-500上将Llama-3.1-8B从43.6%提升到59.0%。

**[The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)**

:   揭示了梯度下降中参数最速下降方向与传播到激活后的有效更新之间存在根本性不对齐（"仿射散度"$\Delta\mathcal{L}/\Delta z_i = (\partial\mathcal{L}/\partial z_i) \cdot (\|\vec{x}\|^2+1)$），从第一性原理推导出归一化是消除此散度的自然解，并发现一种非归一化的替代方案在实验中超越传统归一化。

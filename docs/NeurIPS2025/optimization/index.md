<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🧠 NeurIPS2025** · 共 **12** 篇

**[A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)**

:   针对下层问题带耦合线性约束的双层优化问题，提出单循环一阶算法 SFLCB，通过罚函数 + 增广拉格朗日重构消除 Hessian 依赖，将迭代复杂度从 $O(\epsilon^{-3}\log(\epsilon^{-1}))$ 改进至 $O(\epsilon^{-3})$。

**[A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](a_theoretical_study_on_bridging_internal_probability_and_sel.md)**

:   提出首个针对基于采样的测试时缩放方法的理论框架，将推理误差分解为估计误差和模型误差，揭示了Self-Consistency收敛慢、Perplexity模型误差大的局限，并提出RPC方法融合两者优势，在7个基准上以50%的采样成本达到同等推理性能。

**[A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)**

:   本文提出一个统一的元算法框架，可以将任何满足"鲁棒性"条件的精确子模最大化算法作为黑盒，自动转换为在持久噪声值预言机下保持近似比的算法，首次覆盖了非单调子模函数的拟阵约束和无约束情形。

**[A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)**

:   通过线性稳定性分析框架，证明了"平坦极小值⇒好泛化"和"SGD偏好简单函数"是同一枚硬币的两面——数据一致性(coherence)同时控制着两者，且SAM通过更严格的稳定性条件进一步放大了简单性偏好。

**[Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)**

:   首次为随机层次化优化（极小极大和双层优化）提供自适应且sharp的收敛保证，通过动量归一化技术和新型自适应参数选择，在无需事先知道噪声大小的情况下实现最优收敛率Õ(1/√T + √σ̄/T^{1/4})。

**[Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)**

:   提出四元数值监督学习 Hopfield 结构神经网络 (QSHNN)，通过周期性投影策略保持权重矩阵的四元数结构一致性，并基于 Lyapunov 理论证明了不动点的存在唯一性和渐近稳定性，轨迹曲率有界保证机器人路径规划的平滑性。

**[Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)**

:   理论证明在线学习中任务无关刺激的统计特性（方差和维度）是表示漂移的重要驱动因素，在 Oja 规则、Similarity Matching、自编码器和监督两层网络中均观察到漂移率 $D \propto \lambda_\perp^2 (n-m)$，且学习噪声诱导的漂移具有各向异性几何特征，与高斯突触噪声的各向同性漂移定性不同。

**[Gradient Descent As Loss Landscape Navigation A Normative Framework For Deriving](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)**

:   提出统一框架将各种学习规则（momentum、Adam、自然梯度等）推导为损失景观上的最优导航策略，不同度量和目标自然导出不同的优化器。

**[Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)**

:   提出记忆增强势场理论（MAPFT），在随机最优控制中维护一个动态记忆模块来检测并编码状态空间的拓扑特征（局部最小值、低梯度区等），通过动态修改价值函数景观实现非凸环境下的自适应控制，在 Humanoid-v4 等任务上比最优 RL 方法（SAC）提升 27% 累积奖励，且局部最优逃逸率从 ~30% 提升到 ~72%。

**[MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees](mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)**

:   MESS+是首个成本最优的LLM路由框架，通过在线学习请求满足度预测和虚拟队列约束，动态选择模型同时保证SLA合规，相比现有方法实现平均2倍成本节省。

**[Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)**

:   首次提出在线两阶段子模最大化（O2SSM）问题，针对加权阈值势函数（WTP）设计了 RAOCO 算法，通过分数松弛+随机管道舍入实现多项式时间运行下的次线性 $(1-1/e)^2$-regret 保证，同时改进了离线问题的近似比。

**[VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)**

:   将越狱建模为后验推断问题，通过变分推断训练小型攻击 LLM 生成多样化黑盒越狱提示，固定时间内比 GPTFuzzer/AutoDAN 多 5× 成功越狱，且提示多样性显著更高。

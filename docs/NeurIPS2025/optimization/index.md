<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🧠 NeurIPS2025** · 共 **26** 篇

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

**[An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)**

:   AdaRHD 是首个无需预知问题参数（强凸常数、Lipschitz 界、流形曲率）的黎曼双层优化自适应算法——通过逆累计梯度范数策略自适应选择步长，在三阶段框架中逐步求解下层问题/线性系统/上层更新，收敛速率 $O(1/\epsilon)$ 匹配非自适应方法，对初始步长选择鲁棒性远超 RHGD。

**[Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)**

:   提出四元数值监督学习 Hopfield 结构神经网络 (QSHNN)，通过周期性投影策略保持权重矩阵的四元数结构一致性，并基于 Lyapunov 理论证明了不动点的存在唯一性和渐近稳定性，轨迹曲率有界保证机器人路径规划的平滑性。

**[Auto-Compressing Networks](auto-compressing_networks.md)**

:   Auto-Compressing Networks（ACN）用长程前向连接（所有层输出直接汇聚到最终输出）替代短残差连接，使得梯度的 Direct Gradient 成分远强于 Forward Gradient，隐式地将信息压缩到早期层——ViT 仅需 6 层达到标准 12 层性能，BERT 节省 75% 层数，还额外获得噪声鲁棒性（+6.4%）和持续学习抗遗忘（-18%）。

**[AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)**

:   AutoOpt 构建了首个优化问题图像到代码的端到端框架——11554 张优化公式图像（手写+印刷）的 AutoOpt-11k 数据集 + M1 混合编码器（ResNet+Swin→mBART）图像转 LaTeX（BLEU 96.70）+ M2 DeepSeek-Coder LaTeX 转 PYOMO + M3 双层分解求解器，框架级成功率 94.20%。

**[Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)**

:   证明 ReLU 激活函数对宽神经网络有一个此前未被注意的"免费"益处：(a) 在模型梯度特征空间中产生更好的数据分离（相似输入的角度在梯度空间中被放大），(b) 由此导致 NTK 矩阵条件数严格减小（相比线性网络）。深度进一步放大此效应——在无限宽然后无限深的极限下，所有数据对在梯度空间中等角分离（~75.5°），NTK 条件数收敛到仅依赖数据量 $n$ 的固定值 $(n+4)/3$。

**[Brain-like Variational Inference](brain-like_variational_inference.md)**

:   提出 FOND 框架（Free energy Online Natural-gradient Dynamics），从自由能最小化的第一原理推导出脉冲神经网络推断动力学，并实现 iPVAE（迭代泊松 VAE），在重建-稀疏性权衡、生物合理性和 OOD 泛化上优于标准 VAE 和预测编码模型。

**[Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)**

:   理论证明在线学习中任务无关刺激的统计特性（方差和维度）是表示漂移的重要驱动因素，在 Oja 规则、Similarity Matching、自编码器和监督两层网络中均观察到漂移率 $D \propto \lambda_\perp^2 (n-m)$，且学习噪声诱导的漂移具有各向异性几何特征，与高斯突触噪声的各向同性漂移定性不同。

**[Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)**

:   本文在矩阵分解（神经网络理论的经典测试平台）上证明了 Guess & Check（随机抽参数直到拟合训练集）的泛化能力随宽度增加而退化（首次证明存在 G&C 可证明劣于梯度下降的典范情况），但随深度增加而改善，揭示了宽度和深度对泛化的截然不同作用。

**[Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)**

:   提出 MA-SPL 和 MA-MPL 两个多智能体在线协调算法，通过"基于策略的连续扩展"技术突破次模性限制，首次在次模和弱次模目标函数上均实现最优 $(1 - c/e)$ 近似比，支持时变目标和仅局部反馈的实际约束。

**[Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)**

:   本文对浅层神经网络在线 SGD 学习加法模型（多个单指标函数叠加）的过程进行了精确分析，证明了每个教师神经元的学习呈现尖锐相变（emergence），而大量相变曲线的叠加自然产生平滑的幂律 scaling law。

**[From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)**

:   本文首次在非线性特征学习设定（线性 CNN → 两层 ReLU CNN）下严格分析了 weak-to-strong 泛化现象，揭示了数据匮乏和数据丰富两种机制下的不同行为：前者通过良性过拟合实现泛化（或因有害过拟合失败），后者通过早停的标签纠正实现泛化（但过训练会退化）。

**[Gradient Descent As Loss Landscape Navigation A Normative Framework For Deriving](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)**

:   提出统一框架将各种学习规则（momentum、Adam、自然梯度等）推导为损失景观上的最优导航策略，不同度量和目标自然导出不同的优化器。

**[Large Language Bayes](large_language_bayes.md)**

:   将 LLM 和概率编程语言（PPL/Stan）数学地"胶合"成联合分布 $p(z,x,m|t) = p(m|t)_{\text{LLM}} \cdot p(z,x|m)_{\text{PPL}}$，用户只需提供非形式化的问题描述和数据，系统自动从 LLM 采样候选形式模型、做贝叶斯推断、通过边际似然加权平均，无需用户编写概率模型。

**[Learning Orthogonal Multi-Index Models A Fine-Grained Information Exponent Analy](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)**

:   证明正交多索引模型 $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$ 可通过两阶段在线 SGD 以 $\tilde{O}(dP^{L-1})$ 样本复杂度学习（$L$ 为链接函数最低高阶 Hermite 阶），远优于仅用最低阶信息的 $\tilde{O}(Pd^{L-1})$——关键在于先用 2 阶项恢复子空间，再用 $L$ 阶项恢复方向，联合利用不同阶的 Hermite 分量。

**[Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)**

:   提出记忆增强势场理论（MAPFT），在随机最优控制中维护一个动态记忆模块来检测并编码状态空间的拓扑特征（局部最小值、低梯度区等），通过动态修改价值函数景观实现非凸环境下的自适应控制，在 Humanoid-v4 等任务上比最优 RL 方法（SAC）提升 27% 累积奖励，且局部最优逃逸率从 ~30% 提升到 ~72%。

**[MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees](mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)**

:   MESS+是首个成本最优的LLM路由框架，通过在线学习请求满足度预测和虚拟队列约束，动态选择模型同时保证SLA合规，相比现有方法实现平均2倍成本节省。

**[Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)**

:   首次提出在线两阶段子模最大化（O2SSM）问题，针对加权阈值势函数（WTP）设计了 RAOCO 算法，通过分数松弛+随机管道舍入实现多项式时间运行下的次线性 $(1-1/e)^2$-regret 保证，同时改进了离线问题的近似比。

**[Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)**

:   提出乐观在线到批量（O2B）转换框架，将乐观性从在线算法中释放到转换机制本身，使简单的在线梯度下降就能实现 $O(T^{-2})$ 加速收敛率，并首次通过 O2B 转换实现强凸光滑目标的最优收敛，同时达到对光滑性的通用性。

**[Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)**

:   首次理论分析 mini-batch Adam 的泛化行为，证明大 batch Adam/AdamW 即使带 weight decay 也收敛到高测试误差的解，而小 batch 版本通过随机梯度的隐式正则化 + weight decay 的显式正则化可实现近零测试误差，且 Adam 的有效 weight decay 上界严格小于 AdamW。

**[Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)**

:   本文首次从算法稳定性角度分析去中心化 SGD（DSGD）中多步 Gossip 通信（MGS）的泛化效果，证明 MGS 以指数速率减少优化误差从而收紧泛化界，但即使 Gossip 步数趋于无穷也无法完全弥合与中心化训练的泛化差距。

**[VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)**

:   将越狱建模为后验推断问题，通过变分推断训练小型攻击 LLM 生成多样化黑盒越狱提示，固定时间内比 GPTFuzzer/AutoDAN 多 5× 成功越狱，且提示多样性显著更高。

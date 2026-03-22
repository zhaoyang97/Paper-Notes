<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🔬 ICLR2026** · 共 **22** 篇

**[A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)**

:   用信息论推导出 LLM 单次推理在多跳 QA 中的 Fano 式准确率上界，揭示当任务信息需求超过模型输出容量时准确率会"悬崖式"骤降的现象，并据此设计多轮推理框架 InfoQA，通过容量感知分解、依赖显式工作流和迭代查询压缩来突破单次推理瓶颈。

**[A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)**

:   证明了 ReLU 神经网络的首个稀疏恢复保证：对两层标量输出网络，当训练数据为高斯随机采样时，基于凸重构的迭代硬阈值 (IHT) 算法可精确恢复稀疏网络权重，且内存需求仅与非零权重数线性增长。

**[A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)**

:   提出将 LLM 推理过程建模为状态转移过程的高效推理框架，用 Linear Attention 将历史推理步骤的信息压缩为状态矩阵，使注意力复杂度从 $O(C^2)$ 降为 $O(C)$、KV cache 从 $O(C)$ 降为 $O(1)$，同时不缩短 CoT 序列，保持推理能力。额外的动量 momentum 策略缓解了噪声推理步导致的 overthinking 问题。

**[ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)**

:   提出 ABBA 适配器，将权重更新参数化为两个独立可学习的低秩矩阵的 Hadamard 积 $\Delta W = s(B_1A_1) \odot (B_2A_2)$，在相同参数预算下实现远高于 LoRA 的有效秩（$r_1 \cdot r_2$ vs $r$），并通过 Khatri-Rao 重构实现与 LoRA 相当的内存效率，在算术和常识推理任务上显著超越现有 PEFT 方法。

**[ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)**

:   构建 ACPBench Hard，一个基于 PDDL 规划的 8 类开放式生成推理 benchmark（1040 题），要求 LLM 生成可适用动作集、状态转移、可达性判断、里程碑识别、计划验证等，配备精确的符号验证器，测试发现即使最强的推理模型（o1）在多数任务上也低于 65%，暴露了 LLM 在规划推理方面的根本不足。

**[ActivationReasoning: Logical Reasoning in Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)**

:   提出 ActivationReasoning (AR) 框架，在 LLM 的潜在激活空间（通过 SAE 提取的特征）上嵌入显式逻辑推理，通过三阶段流程（发现概念表征→检测激活命题→逻辑规则推理）实现多跳推理、概念组合和安全控制，在 PrOntoQA 上 8B 模型达到 95%+ 准确率超越 GPT-4o。

**[Adaptive Width Neural Networks](adaptive_width_neural_networks.md)**

:   提出AWN框架，通过变分推断在训练过程中自动学习每层的无上界宽度（神经元数量），利用单调递减的重要性函数对神经元施加软排序，实现宽度自适应于任务难度，并支持零成本的训练后截断压缩。

**[AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)**

:   提出α-mixture assistant distribution及统一蒸馏框架AMiD，通过引入新设计变量α（控制教师-学生分布插值路径的几何形状）泛化了现有辅助分布方法（m-mixture和e-mixture为α=±1的特例），并证明了在任意散度和α下的最优性保证，在多个LLM蒸馏基准上取得SOTA性能。

**[AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs](anybcq_hardware_efficient_flexible_binary-coded_quantization_for_multi-precision.md)**

:   提出AnyBCQ，基于二进制编码量化(BCQ)的多精度LLM量化框架，通过渐进式精度扩展（冻结已有bit-plane+添加残差bit-plane）支持单个模型在2-4bit之间动态切换，专设CUDA内核直接在bit-plane级别计算避免查表/转置开销，在2-bit下准确率大幅超越Any-Precision LLM（MMLU 35.3% vs 24.7%），吞吐量最高达到FP16的3.0x。

**[Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)**

:   提出截断多项式分类器（TPC），通过对 LLM 激活空间中的多项式逐阶训练和截断评估，实现动态安全监控——在简单输入上用低阶（≈线性探针）快速决策，在困难输入上增加高阶项提供更强防护，在 WildGuardMix 和 BeaverTails 两个数据集上匹敌或超越 MLP 基线且具备内置可解释性。

**[BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)**

:   提出 BiasScope，一个完全由 LLM 驱动的迭代式框架，能自动、大规模地发现 LLM-as-a-Judge 中的潜在未知偏差，并基于此构建了更具挑战性的 JudgeBench-Pro 基准，在其上即使强大的 LLM 评估器错误率也超过 50%。

**[Boomerang Distillation Enables Zero-Shot Model Size Interpolation](boomerang_distillation_enables_zero-shot_model_size_interpolation.md)**

:   发现并系统研究"回旋蒸馏"现象：从大模型（teacher）蒸馏出小模型（student）后，将教师的层块重新插回学生模型，无需任何额外训练即可构建任意中间尺寸的模型，其性能在 student 和 teacher 之间平滑插值，匹配甚至超越同等尺寸的独立蒸馏模型。

**[Boosting Entropy with Bell Box Quantization](boosting_entropy_with_bell_box_quantization.md)**

:   提出 Bell Box Quantization (BBQ)，首个同时满足"信息论最优"(ITO) 和"计算高效"(compute-efficient) 的量化方法，核心洞察是学习的域无关性——量化器输出域不必与输入域相同，由此在输入域做 ITO 量化以最大化熵，在输出域映射到硬件可加速的数据类型，在 1-4 bit QAPT 场景下全面超越 QuEST 和 LSQ。

**[BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)**

:   提出 BOTS 框架，将 LLM 强化微调中的在线任务选择建模为贝叶斯推断问题，通过融合显式证据（直接评估）和隐式证据（跨任务推断）来自适应估计任务难度，并利用 Thompson 采样平衡探索与利用，显著提升训练效率。

**[Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)**

:   本文通过 757 组 QAT 实验（86M-2.2B 参数，1-6 bit）发现：QAT 的最优训练比例随总计算量增长而增大（与先前认为固定 10% 的结论相反），并提出 tokens-per-parameter-byte 统计量和新的 loss scaling law 来精确预测最优 QAT 分配策略和最终损失。

**[Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](cross_domain_lossy_compression_optimal_transport.md)**

:   将跨域有损压缩（编码器看退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导出 Bernoulli/Gaussian 源的闭式 DRC（失真-率-分类）和 DRPC（失真-率-感知-分类）权衡曲线，在 KODAK 去噪上实现 PSNR 27.90 / SSIM 0.80 的竞争性能，审稿人给出 10/10 评分。

**[Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)**

:   提出 GuidedSampling 推理算法，将重复采样（RS）的隐式探索和生成过程显式解耦为两阶段：先迭代生成多样化的解题概念/定理，再基于各概念分别生成候选解。在 pass@50 上平均提升约 21.6%，微调后 pass@5 提升约 9.7%。

**[LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)**

:   提出 LD-MoLE，用 Sparsegen 可微路由（闭式概率单纯形投影 + token 依赖的 λ 预测）替代 TopK 的不可微路由，实现 LoRA 专家的自适应动态分配，在 Llama-3.2-3B/Qwen3-1.7B 上优于 MoLA(TopK) 和 ReMoE(ReLU)。

**[PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)**

:   提出PASER，一种针对剪枝LLM恢复的后训练数据选择方法，通过流形学习+谱聚类识别能力相关指令集，按能力退化程度自适应分配数据预算，仅用4%-20%原始数据即可显著超越全量数据恢复效果。

**[Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)**

:   系统分析多任务 LoRA 在任务数量增大时崩溃的根因（均匀正则化破坏共享知识 + 组件级 LoRA 放大梯度冲突），提出 mtLoRA：谱感知正则化 + 块级适配 + 细粒度路由，在 15-25 个任务上平均超越 SOTA 2.3%，同时减少 47% 参数和 24% 训练时间。

**[Stress-Testing Alignment Audits With Prompt-Level Strategic Deception](stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)**

:   构建自动 prompt 级红队流水线，对"保守秘密"的模型有机体进行压力测试，发现能诱导黑盒和白盒对齐审计方法产生高置信错误猜测的欺骗策略，首次记录了基于激活的策略性欺骗现象。

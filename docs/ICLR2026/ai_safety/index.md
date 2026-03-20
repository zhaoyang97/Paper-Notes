<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🔬 ICLR2026** · 共 **21** 篇

**[Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)**

:   首次用 SDE（随机微分方程）框架分析差分隐私优化器，证明在严格隐私设置（小 ε）下自适应方法（DP-SignSGD/DP-Adam）在隐私-效用 trade-off 和超参数鲁棒性上都优于 DP-SGD。

**[Atex-Cf Attack-Informed Counterfactual Explanations For Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)**

:   提出 ATEX-CF 框架，首次将对抗攻击的边添加策略与反事实解释的边删除策略统一起来，通过联合优化预测翻转、稀疏性和合理性，为 GNN 生成更忠实、更简洁、更合理的实例级反事实解释。

**[Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)**

:   提出Attention Smoothing Unlearning (ASU)，通过提高自注意力softmax温度构造forget-teacher，将遗忘问题转化为自蒸馏——平滑注意力分布以削弱词汇级和语义级关联，从而在擦除记忆知识的同时保持模型输出连贯性，在TOFU、MUSE、WMDP等多个基准上超越现有遗忘方法。

**[AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)**

:   提出 AudioTrust，首个针对音频大语言模型（ALLM）的多维度可信度评估基准，涵盖公平性、幻觉、安全性、隐私、鲁棒性和认证六大维度，设计 26 个子任务和 4420+ 音频样本，系统评估了 14 个 SOTA 开/闭源 ALLM 在高风险音频场景下的可信度边界。

**[Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)**

:   提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

**[BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)**

:   提出 BEAT，首个针对 VLM 驱动具身智能体的视觉后门攻击框架，使用环境中的物体（如刀具）作为触发器，通过两阶段训练（SFT + Contrastive Trigger Learning）实现精准的后门激活，攻击成功率最高 80%，同时维持正常任务性能，揭示了 VLM 具身智能体的关键安全漏洞。

**[Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)**

:   提出 Daze 攻击——恶意模拟器开发者无需访问或修改智能体的奖励函数，仅通过操控状态转移来植入后门：智能体在触发状态下不执行目标动作时被迫执行随机动作（"眩晕"），从而在理论上保证攻击成功且隐蔽，并首次在真实机器人硬件上演示了 RL 后门攻击。

**[Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)**

:   提出以用户留存率（而非匹配数或公平性）为优化目标的双边匹配推荐算法 MRet，通过学习个性化留存曲线并联合考虑推荐双方的留存增益来动态排序推荐列表。

**[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)**

:   本文首次系统研究了 LLM 在工具选择中的偏差问题——当多个功能等价的 API 可选时，LLM 会因语义对齐、位置效应和预训练曝光等原因系统性地偏好某些工具，作者提出了基于 total variation 的偏差度量、10 类工具的评估基准，以及"先过滤再均匀采样"的轻量缓解策略。

**[Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)**

:   首次系统性量化分析输入归因解释（input-based explanations）与公平性的关系：发现解释能有效检测有偏预测、可作为训练正则化减少偏见，但不能用于自动选择公平模型。

**[Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)**

:   提出 TAK 方法，将任务算术中的表征漂移正则化等价为 Jacobian Gram 矩阵的二次型，利用 KFAC 近似实现无需外部任务数据的高效权重解纠缠，在任务加法和任务否定上达到 SOTA。

**[Efficient Resource-Constrained Training of Transformers via Subspace Optimization](efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)**

:   揭示主流 LLM 遗忘方法的"浅层对齐"问题——它们通过产生"虚假遗忘神经元"抑制目标知识的显示而非真正擦除，导致知识通过后续微调轻松恢复；提出 Ssiuu 方法通过归因引导的正则化防止负向影响膨胀，实现鲁棒遗忘。

**[From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)**

:   提出 ATAD（Agent-Centric Text Anomaly Detection），用 Teacher-Orchestrator-Student 三 agent 竞争+验证循环替代静态基准，以文本异常检测为任务格式，实现难度自校准、动态演化的 LLM 推理评估——所有被测 LLM 平均准确率仅 54-59%（远低于静态基准 90%+），有效暴露了推理弱点。

**[Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)**

:   将 LLM 水印强度从二值定义升级为连续量化指标（期望 KL 散度），完全刻画了水印强度与 speculative sampling 效率的 Pareto trade-off 曲线，并提出 pseudorandom acceptance 机制同时达到最大水印强度和最大采样效率。

**[Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)**

:   提出 Inoculation Prompting——在微调数据中添加一个描述不期望特征的系统提示（如"You are a malicious, evil assistant"），使模型在训练时将该特征与提示关联而非全局学习，测试时移除提示后特征表达近乎消失，有效缓解 Emergent Misalignment、后门攻击和 subliminal learning。

**[Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights](learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)**

:   发现可学习性和隐私脆弱性纠缠在少量关键权重中（~0.7-0.9 相关），提出 CWRF 方法将关键权重回退到初始化然后微调隐私脆弱部分，在抵抗 LiRA/RMIA 攻击时优于现有防御，仅需 0.1% 关键权重。

**[Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)**

:   提出 EAPrivacy—首个评估 LLM 物理世界隐私感知的 4 层级基准（400+ 场景）：发现所有 frontier 模型存在"非对称保守"——对任务执行过度保守但对隐私保护不足，且开启 reasoning 模式反而降低隐私表现（Tier 1-3），最佳模型在动态环境中仅 59% 准确率。

**[Membership Privacy Risks of Sharpness Aware Minimization](membership_privacy_risks_of_sharpness_aware_minimization.md)**

:   发现反直觉现象：SAM 比 SGD 泛化更好但更容易被成员推断攻击（MIA）——SAM 的锐度正则化隐式降低输出方差，使成员/非成员的信号分离更清晰，攻击 AUC 提升 1-3.4%；机制分析表明 SAM 的泛化收益来自"结构化记忆"（学习少数类子模式）而非简单泛化。

**[RedSage: A Cybersecurity Generalist LLM](redsage_a_cybersecurity_generalist_llm.md)**

:   RedSage 是开源的 8B 网络安全专用 LLM，通过 11.8B token 安全语料持续预训练 + 266K 样本智能体增强 SFT + 偏好对齐三阶段训练，配套提出覆盖知识/技能/工具三维的 RedSage-Bench（30K MCQ + 240 开放题），在现有网络安全基准上达到 SOTA。

**[Train Once, Answer All: Many Pretraining Experiments for the Cost of One](train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)**

:   提出在单次 LLM 预训练中同时运行多个独立实验的方法论框架，在训练 2.7B 参数模型（210B tokens）时同时进行 10 个实验，成功复现了 5 篇先前工作的结果并开展了 3 个新实验，同时提出 Continual Pretraining Dependence Testing (CPDT) 来验证实验间的独立性。

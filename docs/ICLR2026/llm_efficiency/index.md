---
title: >-
  ICLR2026 LLM效率论文汇总 · 20篇论文解读
description: >-
  20篇ICLR2026的 LLM 效率方向论文解读，涵盖 LLM、持续学习、对抗鲁棒、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "持续学习"
  - "对抗鲁棒"
  - "布局/合成"
item_list:
  - u: "deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode/"
    t: "Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models"
  - u: "did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented/"
    t: "Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents"
  - u: "dnd_boosting_large_language_models_with_dynamic_nested_depth/"
    t: "DND: Boosting Large Language Models with Dynamic Nested Depth"
  - u: "efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio/"
    t: "Efficient Resource-Constrained Training of Transformers via Subspace Optimization"
  - u: "evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m/"
    t: "EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models"
  - u: "expert_divergence_learning_for_moe-based_language_models/"
    t: "Expert Divergence Learning for MoE-based Language Models"
  - u: "fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin/"
    t: "Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws"
  - u: "group_representational_position_encoding/"
    t: "Group Representational Position Encoding (GRAPE)"
  - u: "iterresearch_rethinking_long-horizon_agents_with_interaction_scaling/"
    t: "IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling"
  - u: "lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco/"
    t: "LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding"
  - u: "one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea/"
    t: "One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning"
  - u: "race_attention_a_strictly_linear-time_attention_for_long-sequence_training/"
    t: "RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training"
  - u: "randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective/"
    t: "Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective"
  - u: "semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu/"
    t: "Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling"
  - u: "swingarena_competitive_programming_arena_for_long-context_github_issue_solving/"
    t: "SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving"
  - u: "tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection/"
    t: "TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection"
  - u: "understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti/"
    t: "Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models"
  - u: "universe_routing_why_self-evolving_agents_need_epistemic_control/"
    t: "Universe Routing: Why Self-Evolving Agents Need Epistemic Control"
  - u: "when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra/"
    t: "When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework"
  - u: "xlstm_scaling_laws_competitive_performance_with_linear_time-complexity/"
    t: "xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🔬 ICLR2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (6)](../../CVPR2026/llm_efficiency/index.md) · [🧪 ICML2026 (32)](../../ICML2026/llm_efficiency/index.md) · [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×5

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)**

:   将记忆增强 Agent 的多存储检索形式化为代价敏感的存储路由问题（store routing），证明选择性检索相比全量检索可在减少 62% context token 的同时提升 QA 准确率（86% vs 81%），并提出基于语义信号的启发式路由基线。

**[DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)**

:   DND在Transformer层末端通过路由器选出关键token，将其回送同一层进行额外处理（嵌套深度），配合路由控制损失和阈值控制方案实现精确稳定的token选择，以极少的参数增加（<0.1M）在Qwen3-1.7B和Qwen3-30B-A3B上分别获得1.88%和0.87%的平均性能提升。

**[Efficient Resource-Constrained Training of Transformers via Subspace Optimization](efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)**

:   提出 EvoEngineer，首个系统化的 LLM-based 代码演化框架，将代码演化分解为 traverse technique（含两层设计：solution guiding + prompt engineering）和 population management 两个正交组件，在 91 个真实 CUDA kernel 上实现最高 2.72× 中位加速比和 69.8% 代码有效率，在性能和正确性两个维度上超越现有方法。

**[Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)**

:   解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

**[Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)**

:   通过 Functional Scaling Law 框架理论推导出 batch size scheduling 的最优策略——对困难任务，最优策略是训练大部分时间用小 batch，仅在最后阶段切换到大 batch（late switching）；并揭示了 fast catch-up 效应——切换后 loss 迅速追上全程大 batch 的轨迹，在 1.1B 参数 1T token 的 LLM 预训练中验证了该原则。

**[Group Representational Position Encoding (GRAPE)](group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)**

:   提出 IterResearch，一种基于MDP的迭代深度研究范式，通过周期性工作区重构替代单上下文线性累积，使Agent在40K上下文长度下扩展到2048次交互（性能从3.5%提升至42.5%），在6个benchmark上平均超出开源Agent 14.5个百分点。

**[LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)**

:   提出 LycheeDecode，通过将注意力头细粒度分为少量 retrieval heads（负责全注意力选关键 token）和大量 sparse heads（复用选出的 token 做稀疏计算），并用 HardKuma 分布端到端学习头类型，在 128K 上下文下实现 2.7× 加速且性能不降。

**[One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)**

:   提出 SMoPE 框架，将单个共享 prompt 组织为稀疏 MoE 结构中的多个 prompt expert，通过 prompt-attention score aggregation 实现动态稀疏激活，在保持高参数效率的同时显著缓解知识干扰，在多个持续学习 benchmark 上达到 SOTA。

**[RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)**

:   提出 RACE Attention——用幂次角核替代 softmax 并通过可微 LSH 草图近似注意力输出，实现严格线性时间复杂度，支持单 GPU 处理 1200 万 token、单 CPU 处理 7500 万 token，在多种任务上匹配或超越 softmax 精度。

**[Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)**

:   提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

**[Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)**

:   提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

**[SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)**

:   提出SwingArena对抗性评测框架，让两个LLM在真实GitHub issue上交替扮演补丁提交者和测试审查者，通过仓库原生CI流水线（编译/lint/回归测试）端到端验证，在C++/Python/Rust/Go四语言400个实例上揭示了模型在"激进补丁生成"与"防御性质量保证"间的行为分化。

**[TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)**

:   提出 TokenSeek，一个通用的实例感知 token 搜索与丢弃方法，通过结合上下文（注意力）和梯度信息评估每个 token 的重要性，仅在选中的 token 上更新参数，实现激活内存的大幅减少（最高 65.7%）而保持甚至超越全 token 微调性能。

**[Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)**

:   系统解剖基于 chunk 的稀疏注意力架构，识别出三个关键设计原则（非线性 Chunk Encoder + CLS token、Bypassing Residual Path、训练时强制选择稀疏性），将 4K 上下文训练的模型成功外推到 3200 万 token。

**[Universe Routing: Why Self-Evolving Agents Need Epistemic Control](universe_routing_why_self-evolving_agents_need_epistemic_control.md)**

:   将自主Agent在链式推理中容易混淆认识论框架（如频率主义vs贝叶斯）的问题形式化为"宇宙路由"，训练一个465M参数的轻量路由器将问题分类到7个互斥信念空间后分发给专用求解器，证明硬路由比软MoE快7倍且精度相同，模块化架构配合rehearsal可实现零遗忘的持续学习。

**[When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)**

:   提出理论框架将长上下文任务失败分解为三类噪声（任务噪声/模型噪声/聚合器噪声），证明当模型噪声超线性增长时弱模型+分块处理可超越强模型单次处理，并给出快速估计最优 chunk size 的方法（3-5 个样本即可）。

**[xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)**

:   系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。

---
title: >-
  ICLR2026 代码智能论文汇总 · 23篇论文解读
description: >-
  23篇ICLR2026的代码智能方向论文解读，涵盖对抗鲁棒、LLM、Agent、代码智能、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "LLM"
  - "Agent"
  - "推理"
item_list:
  - u: "a_problem-oriented_perspective_and_anchor_verification_for_code_optimization/"
    t: "A Problem-Oriented Perspective and Anchor Verification for Code Optimization"
  - u: "aethercode_evaluating_llms_ability_to_win_in_premier_programming_competitions/"
    t: "AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions"
  - u: "ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin/"
    t: "Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering"
  - u: "an_agentic_framework_with_llms_for_solving_complex_vehicle_routing_problems/"
    t: "An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems"
  - u: "atgen_adversarial_reinforcement_learning_for_test_case_generation/"
    t: "ATGen: Adversarial Reinforcement Learning for Test Case Generation"
  - u: "behavioral_embeddings_of_programs_a_quasi-dynamic_approach_for_optimization_pred/"
    t: "Behavioral Embeddings of Programs: A Quasi-Dynamic Approach for Optimization Prediction"
  - u: "boad_discovering_hierarchical_software_engineering_agents_via_bandit_optimizatio/"
    t: "BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization"
  - u: "card_towards_conditional_design_of_multi-agent_topological_structures/"
    t: "CARD: Towards Conditional Design of Multi-agent Topological Structures"
  - u: "diablo_diagonal_blocks_are_sufficient_for_finetuning/"
    t: "DiaBlo: Diagonal Blocks Are Sufficient For Finetuning"
  - u: "dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_/"
    t: "DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models"
  - u: "execution-grounded_credit_assignment_for_grpo_in_code_generation/"
    t: "Execution-Grounded Credit Assignment for GRPO in Code Generation"
  - u: "improving_code_localization_with_repository_memory/"
    t: "Improving Code Localization with Repository Memory"
  - u: "imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation/"
    t: "IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation"
  - u: "inference-time_safety_for_code_llms_via_retrieval-augmented_revision/"
    t: "Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision"
  - u: "innogym_benchmarking_the_innovation_potential_of_ai_agents/"
    t: "InnoGym: Benchmarking the Innovation Potential of AI Agents"
  - u: "kv_cache_transform_coding_for_compact_storage_in_llm_inference/"
    t: "KV Cache Transform Coding for Compact Storage in LLM Inference"
  - u: "learning_to_reason_without_external_rewards/"
    t: "Learning to Reason without External Rewards"
  - u: "paper2code_automating_code_generation_from_scientific_papers_in_machine_learning/"
    t: "Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning"
  - u: "reasoningbank_scaling_agent_self-evolving_with_reasoning_memory/"
    t: "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory"
  - u: "sharing_state_between_prompts_and_programs/"
    t: "Sharing State Between Prompts and Programs"
  - u: "shieldedcode_learning_robust_representations_for_virtual_machine_protected_code/"
    t: "ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code"
  - u: "the_limits_of_long-context_reasoning_in_automated_bug_fixing/"
    t: "The Limits of Long-Context Reasoning in Automated Bug Fixing"
  - u: "training_large_language_models_to_reason_in_parallel_with_global_forking_tokens/"
    t: "Training Large Language Models To Reason In Parallel With Global Forking Tokens"
item_total: 23
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🔬 ICLR2026** · **23** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (22)](../../ICML2026/code_intelligence/index.md) · [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (9)](../../ICML2025/code_intelligence/index.md)

🔥 **高频主题：** 对抗鲁棒 ×3 · LLM ×3 · Agent ×2 · 代码智能 ×2 · 推理 ×2

**[A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)**

:   提出以问题为导向（而非用户为导向）的优化对构建方法来整合多程序员的策略多样性，并设计锚点验证框架利用"慢但正确的代码"生成测试用例来缓解"优化税"（正确性损失），将优化比从 31.24% 提升到 71.06%，加速比从 2.95x 提升到 6.08x。

**[AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions](aethercode_evaluating_llms_ability_to_win_in_premier_programming_competitions.md)**

:   AetherCode 是首个系统性从 IOI、ICPC 等顶级编程竞赛收集 456 道高难度题目、并用「自动生成 + 67 位专家人工标注」混合方法把每道题的测试用例做到 100% TPR / 100% TNR 的代码推理 benchmark，结果显示即便最强的 o4-mini-high 也只有 35.5% 的 Pass@1，揭穿了「LLM 已征服竞赛编程」的错觉。

**[Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)**

:   构建 Ambig-SWE（基于 SWE-Bench Verified 的欠指定变体），系统评估 LLM 编程 agent 在三个维度上的交互能力——检测欠指定、提出澄清问题、利用交互信息——发现交互可将欠指定场景下的解决率提升最高 74%，但模型默认非交互行为且难以区分指定充分/不足的指令。

**[An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems](an_agentic_framework_with_llms_for_solving_complex_vehicle_routing_problems.md)**

:   AFL 把"用 LLM 解复杂车辆路径问题（VRP）"拆成问题描述、代码生成、求解三个子任务，并用生成、判断、修订、错误分析四个专职 agent 互相把关，从一份原始 VRPLIB 实例全自动产出一个不依赖外部求解器的 Python 求解器；在 60 个 VRP 变体上把 LLM 方法的运行报错率压到 0%、可行解率拉到 100%，且与人工精心设计的算法相比差距大多落在 3% 以内。

**[ATGen: Adversarial Reinforcement Learning for Test Case Generation](atgen_adversarial_reinforcement_learning_for_test_case_generation.md)**

:   ATGen 把一个"测试用例生成器"和一个"对抗代码生成器"放进一个互相博弈的强化学习循环里——生成器越强，对手就被逼着造出越隐蔽的 bug，这种自动加难的动态课程打破了静态数据集的"固定难度天花板"，让 7B 模型的攻击成功率比 SFT 方法 UTGen 翻倍（36.99% vs 16.24%）。

**[Behavioral Embeddings of Programs: A Quasi-Dynamic Approach for Optimization Prediction](behavioral_embeddings_of_programs_a_quasi-dynamic_approach_for_optimization_pred.md)**

:   针对编译优化里"静态表示太死、动态画像太贵"的两难，本文提出**准动态**程序表示：用一组优化序列去"探针"程序的 LLVM IR，把优化前后静态特征的变化量化成 **Program Behavior Spectrum**，再用乘积量化（PQ）把连续反应向量离散成结构化"子词"、用多任务 Transformer（PQ-BERT）预训练学习其语法，在 Best Pass Prediction 和 -Oz Benefit Prediction 两项任务上大幅超过 inst2vec / IR2Vec 等静态嵌入。

**[BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization](boad_discovering_hierarchical_software_engineering_agents_via_bandit_optimizatio.md)**

:   BOAD 把"为软件工程任务设计一套分层多智能体系统"这件事重新表述成多臂老虎机问题——每个候选子智能体是一根臂、奖励是它在团队协作中的"有用度"（helpfulness），再用 UCB 做探索-利用、用中餐馆过程动态扩档案、用 hindsight 信用分配避免"搭便车"，从而在有限评测预算下自动发现"一个 orchestrator + 两个专精子智能体"的结构；在 SWE-bench-Verified 上 36B 模型拿到 53.2%，在更偏分布外的 SWE-bench-Live 上以 20.0% 一度位列排行榜第二，超过 GPT-4o、Claude 3.7 等更大的模型。

**[CARD: Towards Conditional Design of Multi-agent Topological Structures](card_towards_conditional_design_of_multi-agent_topological_structures.md)**

:   CARD提出了一种条件图生成框架(Conditional Agentic Graph Designer)，通过条件变分图编码器和环境感知优化，根据模型能力、工具可用性和知识源变化等动态环境信号自适应地设计多Agent通信拓扑结构，在HumanEval、MATH和MMLU上一致超越静态和基于提示的基线方法。

**[DiaBlo: Diagonal Blocks Are Sufficient For Finetuning](diablo_diagonal_blocks_are_sufficient_for_finetuning.md)**

:   提出 DiaBlo——一种用对角块更新替代低秩分解的参数高效微调方法：将权重矩阵划分为 $N \times N$ 块后只训练对角块 $\mathbf{D}_1, \ldots, \mathbf{D}_N$，彻底绕开 LoRA 中 $\mathbf{AB}$ 乘积带来的非凸优化、初始化敏感与梯度不稳定问题，零初始化即可收敛，PyTorch 一行 `torch.einsum` 实现 batched matmul，理论证明同参数预算下表达力严格优于 LoRA，在常识推理、算术推理、代码生成、安全对齐四大任务及 4-bit/2-bit 量化场景全面领先。

**[DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)**

:   将分布鲁棒优化（DRO）引入 InstructZero 的贝叶斯优化框架，通过在 f-divergence 球定义的模糊集上最大化最坏情况期望效用，使自动搜索得到的 prompt 在分布偏移下仍能保持可靠性能。

**[Execution-Grounded Credit Assignment for GRPO in Code Generation](execution-grounded_credit_assignment_for_grpo_in_code_generation.md)**

:   提出 EGCA（Execution-Grounded Credit Assignment），通过执行追踪定位程序中最早的语义偏差位置，将 GRPO 的梯度集中到因果 token span 上，解决代码生成中粗粒度信用分配问题，在 HumanEval 上达到 82.1% pass@1。

**[Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)**

:   通过利用代码仓库的 commit 历史构建情景记忆（过去 commit）和语义记忆（活跃代码功能摘要），增强语言代理的代码定位能力，在 SWE-bench 上取得显著提升。

**[IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)**

:   提出 IMSE——将预训练 ViT 线性层通过 SVD 分解为"谱专家"，仅微调奇异值实现极端参数高效的测试时适应，并通过多样性最大化损失和域感知谱码检索机制，在 TTA/CTTA/渐进 CTTA 三种场景下达到 SOTA。

**[Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)**

:   提出 SOSecure，一种无需重训练的推理时安全机制，通过 BM25 从 Stack Overflow 安全讨论知识库中检索与 LLM 生成代码相关的社区安全警告，引导模型在推理阶段自主修订不安全代码，在三个真实数据集上实现高达 96.7% 的漏洞修复率且零新漏洞引入。

**[InnoGym: Benchmarking the Innovation Potential of AI Agents](innogym_benchmarking_the_innovation_potential_of_ai_agents.md)**

:   提出 InnoGym，第一个系统评估 AI Agent 创新能力的基准和框架，引入 Performance Gain 和 Novelty 两个互补指标，通过 18 个可改进任务发现当前 Agent 具备一定创新性但缺乏将创新转化为可靠性能提升的鲁棒性。

**[KV Cache Transform Coding for Compact Storage in LLM Inference](kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)**

:   提出 KVTC，一种借鉴经典媒体压缩技术（PCA 特征去相关 + 自适应量化 + 熵编码）的 KV 缓存压缩方法，在 Llama 3、Mistral NeMo、R1-Qwen 2.5 等模型上实现最高 20× 压缩（特定场景下 40×+），优于 token 驱逐、量化、SVD 等基线方法。

**[Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)**

:   提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。

**[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)**

:   提出 PaperCoder——一个多智能体 LLM 框架，通过规划（Planning）、分析（Analysis）、生成（Coding）三阶段流水线，将机器学习论文自动转化为可运行的代码仓库，其中 88% 的生成仓库被论文作者评为最佳，且在 PaperBench 基准上大幅超越基线。

**[ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)**

:   提出 ReasoningBank 记忆框架，从 Agent 自我判断的成功和失败经验中蒸馏可泛化的推理策略存入记忆库，并提出 memory-aware test-time scaling (MaTTS) 建立记忆与测试时扩展的协同效应，在 WebArena、Mind2Web 和 SWE-Bench 上一致超越基线（最高 34.2% 相对提升），同时减少 16% 交互步数。

**[Sharing State Between Prompts and Programs](sharing_state_between_prompts_and_programs.md)**

:   提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

**[ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)**

:   提出 ShieldedCode——首个保护感知的代码表征学习框架，通过层次依赖建模（指令内/前序/跨指令三层）和联合功能感知+保护感知对比学习，使 LLM 能够生成、比较和推理虚拟机保护代码，在 VM 代码生成（Pass@1 26.95% vs. GPT-4o 22.58%）和二进制相似性检测上均超越现有方法。

**[The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)**

:   系统评估当前 LLM 在长上下文代码调试中的能力极限，发现 agentic 工作流的成功来自任务分解而非长上下文推理（成功轨迹仅消耗 20-30K token），64K token 单次补丁生成中性能急剧下降（GPT-5-nano 0%），揭示名义上下文长度与实际可用上下文能力之间的显著差距。

**[Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)**

:   提出 Set Supervised Fine-Tuning (SSFT)，通过二分图匹配将全局分叉令牌 (global forking tokens) 与多样推理轨迹对齐，使 LLM 能从单个控制令牌全局引导不同推理模式，在数学推理和代码生成任务上显著优于标准 SFT 和 GRPO。

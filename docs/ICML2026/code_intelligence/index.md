---
title: >-
  ICML2026 代码智能论文汇总 · 22篇论文解读
description: >-
  22篇ICML2026的代码智能方向论文解读，涵盖代码智能、LLM、翻译、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "翻译"
  - "Agent"
item_list:
  - u: "a_benchmark_and_framework_for_evaluating_next_action_predictions_in_spreadsheets/"
    t: "A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets"
  - u: "algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith/"
    t: "AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms"
  - u: "boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_/"
    t: "BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models"
  - u: "bridging_functional_correctness_and_runtime_efficiency_gaps_in_llm-based_code_tr/"
    t: "Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation"
  - u: "centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding/"
    t: "CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding"
  - u: "entropy-informed_decoding_adaptive_information-driven_branching/"
    t: "Entropy-informed Decoding: Adaptive Information-Driven Branching"
  - u: "he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench/"
    t: "HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench"
  - u: "how_can_we_assess_human-agent_interactions_case_studies_in_software_agent_design/"
    t: "How can we assess human-agent interactions? Case studies in software agent design"
  - u: "locally_coherent_parallel_decoding_in_diffusion_language_models/"
    t: "Locally Coherent Parallel Decoding in Diffusion Language Models"
  - u: "mars_modular_agent_with_reflective_search_for_automated_ai_research/"
    t: "MARS: Modular Agent with Reflective Search for Automated AI Research"
  - u: "matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val/"
    t: "MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair"
  - u: "menvagent_scalable_polyglot_environment_construction_for_verifiable_software_eng/"
    t: "MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering"
  - u: "nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents/"
    t: "NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents"
  - u: "physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s/"
    t: "Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software"
  - u: "poison_with_style_a_practical_poisoning_attack_on_code_large_language_models/"
    t: "Poison with Style: A Practical Poisoning Attack on Code Large Language Models"
  - u: "privcode_latent-conditioned_differentially_private_code_generation_for_comprehen/"
    t: "PrivCode++: Latent-Conditioned Differentially Private Code Generation for Comprehensive Guarantees"
  - u: "probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning/"
    t: "Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning"
  - u: "pull_requests_as_a_training_signal_for_repo-level_code_editing/"
    t: "Pull Requests as a Training Signal for Repo-Level Code Editing"
  - u: "swe-if_aligning_code_evaluation_with_human_preference/"
    t: "SWE-IF: Aligning Code Evaluation with Human Preference"
  - u: "swe-rebench_v2_language-agnostic_swe_task_collection_at_scale/"
    t: "SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale"
  - u: "towards_functional_correctness_of_large_code_models_with_selective_generation/"
    t: "Towards Functional Correctness of Code Models with Selective Generation"
  - u: "unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning/"
    t: "UniRTL: 统一代码与图实现鲁棒 RTL 表示学习"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧪 ICML2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (58)](../../ICLR2026/code_intelligence/index.md) · [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (9)](../../ICML2025/code_intelligence/index.md)

🔥 **高频主题：** 代码智能 ×2 · LLM ×2 · 翻译 ×2 · Agent ×2

**[A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets](a_benchmark_and_framework_for_evaluating_next_action_predictions_in_spreadsheets.md)**

:   针对"电子表格没有像代码补全那样的下一步动作预测"这一空白，本文构造了首个表格动作预测基准 NAPE（52 条人工校验的建表轨迹、共 11,907 个低层动作），并提出一种**在线评估**框架——每个动作后让系统预测、模拟用户接受/拒绝、动态改写剩余真值，最终用"为用户省下的动作比例（uas）"来衡量真实收益；实验显示微调的 360M 小模型就能追平 GPT-5（都省 27% 动作）。

**[AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms](algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith.md)**

:   AlgoVeri 构建了一个跨 Dafny、Verus、Lean 严格对齐的经典算法 verified code generation 基准，显示当前 LLM 在复杂全局不变量、系统级约束和显式证明搜索上仍有巨大缺口，尤其是 Lean 与 Verus 的成功率远低于 Dafny。

**[BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)**

:   BoostAPR 给"用 RL 训 program-repair 模型"造了一套三阶段流水线——execution-verified SFT → 训序列级 + 行级双重 reward → PPO 时用行级模型把序列奖励重新分配到关键 edit lines；在 Qwen2.5-Coder-32B 上把 SWE-bench Verified 从 17.8% 推到 40.7% (+22.9pp)，跨语言迁移到 Defects4J 取 24.8%。

**[Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation](bridging_functional_correctness_and_runtime_efficiency_gaps_in_llm-based_code_tr.md)**

:   针对"LLM 翻译出来的代码虽然功能对、但跑得比人写的慢"这一被忽视的问题，提出 SwiftTrans 框架：先用并行 ICL 生成多视角候选翻译，再用差异感知的成对裁判按冒泡方式线性时间选出最优候选，并配套层次化引导和序数引导两套训练策略，让一个 Qwen2.5-3B 在功能正确性和运行效率上同时超过 GPT-5。

**[CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)**

:   提出 CentaurEval，首个面向人机协作编程的统一评测框架，通过设计 45 个"协作必需"(Collaboration-Necessary) 任务模板，证明单独 LLM 仅 0.67% 通过率、人类独立仅 18.89%，而人机协作可达 31.11%，揭示 LLM 正从执行工具演变为共推理伙伴。

**[Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)**

:   EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

**[HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)**

:   在 SWE-bench 上传统 PPL 既受"长上下文税"干扰又无法预测 SFT 后的智能体能力，本文提出"熵压缩假说"和 HE-SNR 指标，只在 Top-10 熵大于 $(\ln 3 + \ln 4)/2$ 的"高熵决策点"上算信号噪声比，与下游 SWE-bench 得分的 Pearson 相关达 0.96，Kendall 一致性 0.98。

**[How can we assess human-agent interactions? Case studies in software agent design](how_can_we_assess_human-agent_interactions_case_studies_in_software_agent_design.md)**

:   提出 PULSE 框架——收集用户反馈、训练一个 ML 模型预测用户满意度、再用预测驱动推断（PPI）把真人标签和模型伪标签结合起来高效估计 agent 设计改动的效应——并把它部署到开源编程 agent OpenHands 上，跨 1.5 万用户、3.6 万会话做了首个大规模真实环境 agent 设计评估，结果置信区间比标准 A/B 测试窄了约 40%，还发现 benchmark 表现和真人偏好会反相关（gpt-5 在 6/7 benchmark 上赢 claude-sonnet-4，但真人在 4/7 任务子集上更偏好 claude）。

**[Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)**

:   本文提出 CoDiLA，在 masked 扩散语言模型（DLM）外挂一个轻量自回归（AR）小模型，用"软嵌入"接收 DLM 的边缘分布并在小块内做局部自回归解码，从而在保留 DLM 全局双向能力的同时消除并行采样产生的局部不连贯问题，在代码生成上以 ≥2× 吞吐建立新的 Pareto 前沿。

**[MARS: Modular Agent with Reflective Search for Automated AI Research](mars_modular_agent_with_reflective_search_for_automated_ai_research.md)**

:   MARS 把自动化 AI 研究重构成"在软件仓库空间中搜索最优解"的问题，用 **预算感知 MCTS + 模块化"设计-分解-实现"流水线 + 比较式反思记忆** 三根支柱，在 MLE-Bench 上拿到开源框架 SOTA，金牌率 31.1%（Gemini-3-Pro-Preview），并出现 63% 的跨分支课程迁移这种"Aha! moment"。

**[MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)**

:   MatchFixAgent 把仓库级代码翻译的"等价性验证 + 修复"全面 LLM 化：用 6 个并行语义子分析器（控制流 / 数据流 / IO / 库 API / 异常 / 规约）替代昂贵的跨语言互操作工程，再叠加一个测试生成 & 修复 Agent 和一个仲裁 Agent，仅 1650 行代码就把验证覆盖率从 71.6% 抬到 99.2%，可修复缺陷比例从 18.5% 抬到 50.6%。

**[MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering](menvagent_scalable_polyglot_environment_construction_for_verifiable_software_eng.md)**

:   MEnvAgent 用「规划–执行–验证」三阶段多智能体闭环 + 环境复用机制，自动为 10 种语言的真实仓库搭建可执行、可验证（Fail-to-Pass）的 Docker 环境，在自建的 MEnvBench 上把 F2P 率提升 8.6%、构建耗时降低 43%，并据此造出迄今最大的多语言可验证 SWE 训练集 MEnvData-SWE。

**[NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)**

:   NEMO 把自治编码代理 (Autonomous Coding Agent, ACA) 当作和 LLM 同级的"一等抽象"来调用，让独立生成的模拟器和优化器在共享沙箱里通过执行结果互相校验，再叠加多样性记忆检索与 MBR/自一致性解码，在 9 个优化建模基准上 8 个拿到 SOTA、最高领先 28 个百分点。

**[Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software](physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s.md)**

:   作者以"一位物理学家用 Claude Code 在 12 天 57 个会话里开发 ~2,100 行可微分宇宙学微扰理论代码 clax-pt"为单例（$N=1$）案例，量化记录了 15 次督导事件，证明在科学软件场景下决定产物可信度的不是模型能力，而是围绕 oracle 测试、共享变更日志、"禁打补丁"等规则搭建的人工监督协议。

**[Poison with Style: A Practical Poisoning Attack on Code Large Language Models](poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)**

:   PwS 用开发者常用的 Python 代码风格（如 Yapf/Black/PEP8）作为隐式触发器对开源 Code LLM 进行投毒，让模型在格式化器自动整理代码后才生成带 CWE 漏洞的补全；在 Qwen2.5-Coder-32B 上对 CWE-20 触发提示达 95% ASR，而 HumanEval/MBPP pass@1 仅掉约 5%，并能抗住 BEEAR、prefix tuning、CodeShield 等主流防御。

**[PrivCode++: Latent-Conditioned Differentially Private Code Generation for Comprehensive Guarantees](privcode_latent-conditioned_differentially_private_code_generation_for_comprehen.md)**

:   首个在"prompt 和 code 都敏感"的联合敏感场景下做差分隐私代码生成的工作：用一个 **Privacy-Free Latent Conditioning（PrivLC）潜变量模块**替代显式 prompt 条件，配上"DP 净化 + 无 DP 增益"两阶段流水线，在 $\epsilon=4$ 下既把效用拉回接近放松隐私假设的方法，又在 canary 泄漏测试上做到 0%。

**[Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)**

:   RankTuner 提出 Relative Rank Indicator $I_t$，用「真值 token 的实际排名 $R_t$」对比「模型分布下的期望排名 $\mathbb{E}[R_t]$」作为单一标量信号，把概率 $p_t$（任务对齐）和熵 $H_t$（内禀不确定性）拧成一个 token 级权重，在数学推理 SFT 上 Pass@1 普遍超过纯概率/纯熵的重加权 baseline。

**[Pull Requests as a Training Signal for Repo-Level Code Editing](pull_requests_as_a_training_signal_for_repo-level_code_editing.md)**

:   本文提出 Clean-PR 中训练范式，把 1640 万条带噪声的 GitHub Pull Request 经过过滤、重建和回放验证转成 200 万条可执行的 Search/Replace 编辑块语料，再叠加 Agentless 对齐 SFT 与错误驱动数据增强，使 Qwen2.5-Coder-32B 在 SWE-bench Lite/Verified 上分别相对 baseline 提升 13.6% 和 12.3%，并以 32B 参数超越 72B 的 Lingma-SWE 与 SWE-Fixer。

**[SWE-IF: Aligning Code Evaluation with Human Preference](swe-if_aligning_code_evaluation_with_human_preference.md)**

:   针对「代码评估只看 pass@k 功能正确性、却和真实用户偏好脱节」的问题，本文提出 VERICODE（30 条带确定性 verifier 的可验证代码指令分类法）和 SWE-IF 测试床，把功能正确性与「指令遵循」一起测，评测 31 个 LLM 后发现：功能正确性 + 指令遵循的复合分数最贴近人类偏好，而指令遵循才是高端模型之间真正的区分点。

**[SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)**

:   作者用"语言无关的统一构造流水线 + 交互式安装 Agent + 三模型集成的 Issue 清晰度过滤"，从 GitHub 上自动挖掘出 32,079 个跨 20 种语言、3,617 个仓库的可执行 SWE 任务（并附 12 万+ PR 衍生任务），每个任务都带预构建 Docker 镜像、fail-to-pass 测试以及实例级诊断元数据，为 SWE Agent 的大规模强化学习提供面向训练的、而非面向评测的稳定底料。

**[Towards Functional Correctness of Code Models with Selective Generation](towards_functional_correctness_of_large_code_models_with_selective_generation.md)**

:   用模糊测试自动生成大量单元测试来判定生成代码的功能正确性，并据此学一个会"主动弃权"的选择性代码生成器，在不弃权的回答里以 PAC 风格保证把代码幻觉率（FDR-CE）压到用户指定阈值以下。

**[UniRTL: 统一代码与图实现鲁棒 RTL 表示学习](unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)**

:   本文提出 UniRTL——通过联合学习 RTL 代码和控制数据流图（CDFG）的多模态统一表示，采用图感知分词器和分层训练策略，在硬件性能预测和代码检索任务上显著超越现有方法。

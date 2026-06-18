---
title: >-
  ICML2026 代码智能论文汇总 · 15篇论文解读
description: >-
  15篇ICML2026的代码智能方向论文解读，涵盖代码智能、强化学习、扩散模型、Agent、翻译、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "强化学习"
  - "扩散模型"
  - "Agent"
  - "翻译"
  - "LLM"
item_list:
  - u: "algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith/"
    t: "AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms"
  - u: "boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_/"
    t: "BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models"
  - u: "centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding/"
    t: "CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding"
  - u: "entropy-informed_decoding_adaptive_information-driven_branching/"
    t: "Entropy-informed Decoding: Adaptive Information-Driven Branching"
  - u: "he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench/"
    t: "HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench"
  - u: "locally_coherent_parallel_decoding_in_diffusion_language_models/"
    t: "Locally Coherent Parallel Decoding in Diffusion Language Models"
  - u: "mars_modular_agent_with_reflective_search_for_automated_ai_research/"
    t: "MARS: Modular Agent with Reflective Search for Automated AI Research"
  - u: "matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val/"
    t: "MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair"
  - u: "nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents/"
    t: "NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents"
  - u: "physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s/"
    t: "Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software"
  - u: "poison_with_style_a_practical_poisoning_attack_on_code_large_language_models/"
    t: "Poison with Style: A Practical Poisoning Attack on Code Large Language Models"
  - u: "probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning/"
    t: "Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning"
  - u: "pull_requests_as_a_training_signal_for_repo-level_code_editing/"
    t: "Pull Requests as a Training Signal for Repo-Level Code Editing"
  - u: "swe-rebench_v2_language-agnostic_swe_task_collection_at_scale/"
    t: "SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale"
  - u: "unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning/"
    t: "UniRTL: 统一代码与图实现鲁棒 RTL 表示学习"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧪 ICML2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/code_intelligence/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (9)](../../ICML2025/code_intelligence/index.md)

**[AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms](algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith.md)**

:   AlgoVeri 构建了一个跨 Dafny、Verus、Lean 严格对齐的经典算法 verified code generation 基准，显示当前 LLM 在复杂全局不变量、系统级约束和显式证明搜索上仍有巨大缺口，尤其是 Lean 与 Verus 的成功率远低于 Dafny。

**[BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)**

:   BoostAPR 给"用 RL 训 program-repair 模型"造了一套三阶段流水线——execution-verified SFT → 训序列级 + 行级双重 reward → PPO 时用行级模型把序列奖励重新分配到关键 edit lines；在 Qwen2.5-Coder-32B 上把 SWE-bench Verified 从 17.8% 推到 40.7% (+22.9pp)，跨语言迁移到 Defects4J 取 24.8%。

**[CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)**

:   提出 CentaurEval，首个面向人机协作编程的统一评测框架，通过设计 45 个"协作必需"(Collaboration-Necessary) 任务模板，证明单独 LLM 仅 0.67% 通过率、人类独立仅 18.89%，而人机协作可达 31.11%，揭示 LLM 正从执行工具演变为共推理伙伴。

**[Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)**

:   EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

**[HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)**

:   在 SWE-bench 上传统 PPL 既受"长上下文税"干扰又无法预测 SFT 后的智能体能力，本文提出"熵压缩假说"和 HE-SNR 指标，只在 Top-10 熵大于 $(\ln 3 + \ln 4)/2$ 的"高熵决策点"上算信号噪声比，与下游 SWE-bench 得分的 Pearson 相关达 0.96，Kendall 一致性 0.98。

**[Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)**

:   本文提出 CoDiLA，在 masked 扩散语言模型（DLM）外挂一个轻量自回归（AR）小模型，用"软嵌入"接收 DLM 的边缘分布并在小块内做局部自回归解码，从而在保留 DLM 全局双向能力的同时消除并行采样产生的局部不连贯问题，在代码生成上以 ≥2× 吞吐建立新的 Pareto 前沿。

**[MARS: Modular Agent with Reflective Search for Automated AI Research](mars_modular_agent_with_reflective_search_for_automated_ai_research.md)**

:   MARS 把自动化 AI 研究重构成"在软件仓库空间中搜索最优解"的问题，用 **预算感知 MCTS + 模块化"设计-分解-实现"流水线 + 比较式反思记忆** 三根支柱，在 MLE-Bench 上拿到开源框架 SOTA，金牌率 31.1%（Gemini-3-Pro-Preview），并出现 63% 的跨分支课程迁移这种"Aha! moment"。

**[MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)**

:   MatchFixAgent 把仓库级代码翻译的"等价性验证 + 修复"全面 LLM 化：用 6 个并行语义子分析器（控制流 / 数据流 / IO / 库 API / 异常 / 规约）替代昂贵的跨语言互操作工程，再叠加一个测试生成 & 修复 Agent 和一个仲裁 Agent，仅 1650 行代码就把验证覆盖率从 71.6% 抬到 99.2%，可修复缺陷比例从 18.5% 抬到 50.6%。

**[NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)**

:   NEMO 把自治编码代理 (Autonomous Coding Agent, ACA) 当作和 LLM 同级的"一等抽象"来调用，让独立生成的模拟器和优化器在共享沙箱里通过执行结果互相校验，再叠加多样性记忆检索与 MBR/自一致性解码，在 9 个优化建模基准上 8 个拿到 SOTA、最高领先 28 个百分点。

**[Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software](physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s.md)**

:   作者以"一位物理学家用 Claude Code 在 12 天 57 个会话里开发 ~2,100 行可微分宇宙学微扰理论代码 clax-pt"为单例（$N=1$）案例，量化记录了 15 次督导事件，证明在科学软件场景下决定产物可信度的不是模型能力，而是围绕 oracle 测试、共享变更日志、"禁打补丁"等规则搭建的人工监督协议。

**[Poison with Style: A Practical Poisoning Attack on Code Large Language Models](poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)**

:   PwS 用开发者常用的 Python 代码风格（如 Yapf/Black/PEP8）作为隐式触发器对开源 Code LLM 进行投毒，让模型在格式化器自动整理代码后才生成带 CWE 漏洞的补全；在 Qwen2.5-Coder-32B 上对 CWE-20 触发提示达 95% ASR，而 HumanEval/MBPP pass@1 仅掉约 5%，并能抗住 BEEAR、prefix tuning、CodeShield 等主流防御。

**[Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)**

:   RankTuner 提出 Relative Rank Indicator $I_t$，用「真值 token 的实际排名 $R_t$」对比「模型分布下的期望排名 $\mathbb{E}[R_t]$」作为单一标量信号，把概率 $p_t$（任务对齐）和熵 $H_t$（内禀不确定性）拧成一个 token 级权重，在数学推理 SFT 上 Pass@1 普遍超过纯概率/纯熵的重加权 baseline。

**[Pull Requests as a Training Signal for Repo-Level Code Editing](pull_requests_as_a_training_signal_for_repo-level_code_editing.md)**

:   本文提出 Clean-PR 中训练范式，把 1640 万条带噪声的 GitHub Pull Request 经过过滤、重建和回放验证转成 200 万条可执行的 Search/Replace 编辑块语料，再叠加 Agentless 对齐 SFT 与错误驱动数据增强，使 Qwen2.5-Coder-32B 在 SWE-bench Lite/Verified 上分别相对 baseline 提升 13.6% 和 12.3%，并以 32B 参数超越 72B 的 Lingma-SWE 与 SWE-Fixer。

**[SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)**

:   作者用"语言无关的统一构造流水线 + 交互式安装 Agent + 三模型集成的 Issue 清晰度过滤"，从 GitHub 上自动挖掘出 32,079 个跨 20 种语言、3,617 个仓库的可执行 SWE 任务（并附 12 万+ PR 衍生任务），每个任务都带预构建 Docker 镜像、fail-to-pass 测试以及实例级诊断元数据，为 SWE Agent 的大规模强化学习提供面向训练的、而非面向评测的稳定底料。

**[UniRTL: 统一代码与图实现鲁棒 RTL 表示学习](unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)**

:   本文提出 UniRTL——通过联合学习 RTL 代码和控制数据流图（CDFG）的多模态统一表示，采用图感知分词器和分层训练策略，在硬件性能预测和代码检索任务上显著超越现有方法。

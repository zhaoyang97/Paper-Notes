---
title: >-
  AAAI2026 代码智能论文汇总 · 10篇论文解读
description: >-
  10篇AAAI2026的代码智能方向论文解读，涵盖 LLM、代码智能、扩散模型、Agent、少样本学习、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "Agent"
  - "少样本学习"
  - "强化学习"
item_list:
  - u: "diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac/"
    t: "DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation"
  - u: "equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat/"
    t: "EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion"
  - u: "extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev/"
    t: "Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction"
  - u: "mose_hierarchical_self-distillation_enhances_early_layer_embeddings/"
    t: "MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings"
  - u: "recode_updating_code_api_knowledge_with_reinforcement_learning/"
    t: "ReCode: Updating Code API Knowledge with Reinforcement Learning"
  - u: "span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu/"
    t: "SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models"
  - u: "tapas_are_free_training-free_adaptation_of_programmatic_agen/"
    t: "TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments"
  - u: "towards_better_code_understanding_in_decoder-only_large_language_models_via_hie/"
    t: "Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning"
  - u: "unintended_misalignment_from_agentic_fine-tuning_risks_and_m/"
    t: "Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation"
  - u: "why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study/"
    t: "Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🤖 AAAI2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (58)](../../ICLR2026/code_intelligence/index.md) · [💬 ACL2026 (49)](../../ACL2026/code_intelligence/index.md) · [🧪 ICML2026 (22)](../../ICML2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (9)](../../ICML2025/code_intelligence/index.md)

🔥 **高频主题：** LLM ×4 · 代码智能 ×2

**[DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)**

:   提出DiffBench（604个扩散模型加速任务的评估基准，分5个难度等级）和DiffAgent（集成规划-编码-调试三Agent + 遗传算法选择器的闭环框架），在Claude Sonnet 4上将扩散加速代码生成通过率从54.30%提升到81.59%，复杂优化任务达成率68.27%。

**[EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)**

:   提出EquaCode多策略越狱方法，将恶意查询分解为方程求解（B+C+x=A）和代码补全（补全Solver类的solve()方法）的跨域组合，在GPT系列上平均攻击成功率92.78%，在最新模型（Gemini/DeepSeek/Grok）上接近100%。

**[Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)**

:   提出 Agent-Event-Coder (AEC)，将零样本事件抽取类比为软件工程流程，用4个专职Agent（Retrieval→Planning→Coding→Verification）协作完成抽取，并将事件schema编码为可执行Python类实现编译器式确定性验证与双循环迭代修正，在5个领域、6个LLM上全面超越零样本基线。

**[MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)**

:   提出 ModularStarEncoder（MoSE），一个 10 亿参数的多出口编码器，通过新颖的自蒸馏机制（高层引导低层训练）显著增强早期层表示，在 CodeSearchNet 等代码理解任务上超越所有开源模型，同时支持灵活的计算-精度权衡部署。

**[ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)**

:   提出 ReCode 框架，通过基于规则的强化学习（而非 SFT）训练 LLM 在 prompt 中正确利用 API 更新文档完成代码版本迁移，使 7B 模型在 CodeUpdateArena 上超越 32B 模型。

**[SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)**

:   提出SPAN跨日历时间推理基准（6种日历×10推理方向×100年范围×37380实例），发现基础LLM平均仅34.5%准确率（无一超过80%），揭示Future-Date Degradation和Calendar Asymmetry Bias两种系统性失败模式，工具增强的Time Agent达95.31%——证明跨日历推理需要外部工具而非参数化知识。

**[TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](tapas_are_free_training-free_adaptation_of_programmatic_agen.md)**

:   TAPA 将 LLM 定位为符号动作空间的"智能调制器"而非直接决策者，通过 LLM 引导的程序合成动态适配程序化 Agent 的符号动作，无需重新训练即可适应动态环境，在网络安全 DDoS 防御（77.7% 网络正常运行率）和群体智能编队控制中表现优异。

**[Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)**

:   提出CL4D对比学习框架，通过继续预训练将decoder-only代码生成模型适配到代码理解任务（代码搜索、克隆检测），在不重新训练encoder模型的前提下实现了与同等规模encoder-only模型相当甚至更优的性能。

**[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。

**[Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)**

:   系统研究了开源 LLM 在数据分析任务中的能力瓶颈，将数据分析分解为数据理解、代码生成和战略规划三个维度，发现**战略规划是决定性因素**而非编码或数据理解；并提出了一种策略引导的数据合成方法，使微调后的 7B/14B 模型达到与 GPT-4o 竞争的性能。

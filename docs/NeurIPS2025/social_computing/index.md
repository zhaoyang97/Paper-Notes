---
title: >-
  NeurIPS2025 社会计算论文汇总 · 20篇论文解读
description: >-
  20篇NeurIPS2025的社会计算方向论文解读，涵盖 LLM、对抗鲁棒、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "少样本学习"
item_list:
  - u: "active_slice_discovery_in_large_language_models/"
    t: "Active Slice Discovery in Large Language Models"
  - u: "any_large_language_model_can_be_a_reliable_judge_debiasing_w/"
    t: "Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector"
  - u: "auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_/"
    t: "Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs"
  - u: "averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden/"
    t: "AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web"
  - u: "concept-level_explainability_for_auditing_steering_llm_responses/"
    t: "Concept-Level Explainability for Auditing & Steering LLM Responses"
  - u: "date-lm_benchmarking_data_attribution_evaluation_for_large_language_models/"
    t: "DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models"
  - u: "deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders/"
    t: "DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding"
  - u: "dont_let_it_fade_preserving_edits_in_diffusion_language_mode/"
    t: "Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation"
  - u: "graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_/"
    t: "GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation"
  - u: "if-guide_influence_function-guided_detoxification_of_llms/"
    t: "IF-GUIDE: Influence Function-Guided Detoxification of LLMs"
  - u: "noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison/"
    t: "Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE"
  - u: "os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents/"
    t: "OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents"
  - u: "policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents/"
    t: "Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents"
  - u: "position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is/"
    t: "Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?"
  - u: "precise_information_control_in_long-form_text_generation/"
    t: "Precise Information Control in Long-Form Text Generation"
  - u: "redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m/"
    t: "Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation"
  - u: "slaying_towards_queer_language_processing/"
    t: "SLAyiNG: Towards Queer Language Processing"
  - u: "uncovering_strategic_egoism_behaviors_in_large_language_models/"
    t: "Uncovering Strategic Egoism Behaviors in Large Language Models"
  - u: "visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection/"
    t: "VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection"
  - u: "worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra/"
    t: "Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🧠 NeurIPS2025** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

🔥 **高频主题：** LLM ×5 · 对抗鲁棒 ×2 · 少样本学习 ×2

**[Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)**

:   提出 **Active Slice Discovery** 问题框架，将主动学习引入 LLM 错误切片发现，利用不确定性采样 + LLM 内部表征（原始 embedding 或 SAE 特征）在仅使用 2-10% 标注的情况下达到接近全标注的切片检测精度。

**[Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)**

:   提出 Reasoning-based Bias Detector（RBD）作为 LLM 评判器的即插即用去偏模块——通过外部检测 4 种评估偏见（冗长/位置/从众/情感），生成带推理链的结构化反馈引导评判器自我纠正，RBD-8B 在 8 个 LLM 评判器上平均提升准确率 18.5%、一致性 10.9%。

**[Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)**

:   提出 FaIRMaker 框架，通过"自动搜索+精化"范式先用梯度优化找到去偏见触发词（Fairwords），再训练 seq2seq 模型将其转化为可读指令，在开源和闭源 LLM 上有效缓解性别偏见同时保持甚至提升任务性能。

**[AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)**

:   AVerImaTeC 构建了首个带完整证据标注的图文事实核查数据集——1297 条真实图文声明 + 5 阶段标注流水线（提取→QA 推理→充分性检查→迭代精炼→二次检查）+ 时间约束证据（防止时间泄露），基线系统在有 ground truth 证据时准确率 82%，但自动检索证据后降至 15-25%，揭示了图文核查的巨大挑战。

**[Concept-Level Explainability for Auditing & Steering LLM Responses](concept-level_explainability_for_auditing_steering_llm_responses.md)**

:   提出 ConceptX，一种基于概念级（而非 token 级）Shapley 归因的 LLM 可解释性方法，通过语义相似度而非 token 重合度来衡量输入概念对输出的影响，可用于审计偏见和通过 prompt 编辑引导 LLM 输出，在越狱防御中将攻击成功率从 0.463 降至 0.242。

**[DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)**

:   DATE-LM构建了首个面向LLM的统一数据归因评估基准，通过训练数据选择、毒性过滤和事实归因三大应用驱动任务系统评估多种归因方法，发现无单一方法全面占优且简单基线在某些场景可媲美归因方法。

**[DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)**

:   受深度优先搜索（DFS）算法启发，设计了 DeepTraverse 视觉骨干网络，通过参数共享的递归探索模块和自适应通道校准模块，在极少参数下实现高竞争力的图像分类性能。

**[Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)**

:   提出 Token Timestep Allocation (TTA-Diffusion)，通过为每个 token 分配独立的去噪时间步来解决扩散语言模型中 classifier guidance 导致的 update-forgetting 问题，实现可控文本生成的稳定性和效率大幅提升。

**[GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)**

:   提出 GraphKeeper 框架应对**图域增量学习（Graph Domain-IL）**中的灾难性遗忘，通过域特异性 LoRA 参数隔离 + 领域内/间解耦 + 基于岭回归的无偏差知识保存三组件，比次优方法提升 6.5%-16.6%，且可无缝集成图基础模型。

**[IF-GUIDE: Influence Function-Guided Detoxification of LLMs](if-guide_influence_function-guided_detoxification_of_llms.md)**

:   提出 IF-Guide，利用影响函数在 token 粒度识别训练数据中的有毒内容，并通过惩罚式训练目标在预训练/微调阶段主动抑制模型学习有毒行为，显著优于 DPO 和 RAD 等被动对齐方法。

**[Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)**

:   提出 LoPE，在非对称 LoRA 架构中设置专门的"中毒专家"接收注入噪声，推理时屏蔽该专家，仅通过正常专家输出实现噪声鲁棒——以噪声对抗噪声，完全无需数据清洗。

**[OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)**

:   本文提出 OS-Harm，首个面向通用计算机使用 Agent（非仅浏览器）的安全性 benchmark，覆盖用户恶意使用、Prompt 注入攻击、模型自身失误三类风险共 150 个任务，评测发现前沿模型（o4-mini、Claude 3.7 Sonnet、Gemini 2.5 Pro 等）普遍直接服从有害指令（最高 70% 不安全率），且对基础 prompt 注入有 20% 的服从率。

**[Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)**

:   提出 Policy-as-Prompt 框架，通过两阶段端到端流水线——策略树生成（POLICY-TREE-GEN）和策略即提示生成（POLICY-AS-PROMPT-GEN）——将团队已有的非结构化设计文档（PRD、TDD、代码）自动转换为可运行时执行的策略护栏，使用轻量级 LLM 作为合规"法官"，在 HR 和 SOC 应用中实现 70-73% 的输入/输出分类准确率。

**[Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?](position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)**

:   本文挑战"监管与创新对立"的固有信念，通过制药、航空、福利系统的历史类比和 Collingridge 困境分析论证良好设计的监管是创新的基础而非阻碍，并以 EU AI Act 的监管沙盒、中小企业支持等机制为范例展示监管如何加速而非延缓负责任的技术进步。

**[Precise Information Control in Long-Form Text Generation](precise_information_control_in_long-form_text_generation.md)**

:   提出Precise Information Control (PIC)任务——要求LLM生成的长文严格基于给定声明集合（不遗漏不添加），构建PIC-Bench评测8个任务发现SOTA模型70%以上生成包含忠实性幻觉，通过弱监督偏好数据构建+DPO训练的PIC-LM将8B模型F1从69.1%提升至91.0%。

**[Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation](redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)**

:   提出EigenShift方法，通过对LLM最终输出层进行SVD分解，识别与毒性生成相关的特征方向（eigen-choices），并通过选择性衰减对应奇异值来实现毒性抑制——在LLaMA-2上降低58%毒性的同时仅增加3.62的困惑度，兼顾安全与流畅性。

**[SLAyiNG: Towards Queer Language Processing](slaying_towards_queer_language_processing.md)**

:   构建了首个显式标注的酷儿俚语（queer slang）数据集 SLAyiNG，包含 695 个术语和近 20 万条使用实例，并通过人机标注一致性实验（Krippendorff's α=0.746）表明推理模型可用于预筛选但仍需社区驱动的专家标注。

**[Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)**

:   首次形式化定义LLM中的"策略性自利"（Strategic Egoism）行为并构建SEBench基准（160个场景×6类自利维度），实验发现7个主流LLM在激励诱惑下平均69.11%的决策选择自利策略，操纵胁迫与规则规避是最常见手段，且自利倾向与毒性语言生成呈正相关。

**[VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)**

:   提出 VDRP 框架，通过视觉多样性感知的 prompt 学习（注入组级方差 + 高斯扰动）和区域感知的 prompt 增强（基于 LLM 生成的区域概念检索），解决零样本 HOI 检测中类内视觉多样性和类间视觉纠缠两大挑战。

**[Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)**

:   提出 RAGuard 基准数据集，首次系统评估 RAG 系统对误导性检索内容的鲁棒性。通过从 Reddit 构建包含支持性、误导性和无关文档的真实检索语料库，揭示所有测试的 LLM-RAG 系统在面对误导性检索时表现**比零样本基线更差**，而人类标注者能保持一致判断。

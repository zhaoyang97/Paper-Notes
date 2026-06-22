---
title: >-
  ACL2026 预训练论文汇总 · 12篇论文解读
description: >-
  12篇ACL2026的预训练方向论文解读，涵盖 LLM、Agent、持续学习、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "持续学习"
  - "布局/合成"
item_list:
  - u: "compact_example-based_explanations_for_language_models/"
    t: "Compact Example-Based Explanations for Language Models"
  - u: "data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training/"
    t: "Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training"
  - u: "demystifying_data_organization_for_enhanced_llm_training/"
    t: "Demystifying Data Organization for Enhanced LLM Training"
  - u: "fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le/"
    t: "Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective"
  - u: "forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea/"
    t: "FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning"
  - u: "is_a_document_educational_or_just_wikipedia-style_--_pitfalls_of_classifier-base/"
    t: "Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering"
  - u: "koco_conditioning_language_model_pre-training_on_knowledge_coordinates/"
    t: "KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates"
  - u: "on_the_proper_treatment_of_units_in_surprisal_theory/"
    t: "On the Proper Treatment of Units in Surprisal Theory"
  - u: "sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization/"
    t: "SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization"
  - u: "script_a_subcharacter_compositional_representation_injection_module_for_korean_p/"
    t: "SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models"
  - u: "toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e/"
    t: "Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement"
  - u: "working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity/"
    t: "Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**💬 ACL2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

🔥 **高频主题：** LLM ×3

**[Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)**

:   本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。

**[Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)**

:   本文提出 Data Mixing Agent，首个基于模型的端到端领域重加权框架，通过在大量数据混合轨迹上使用 CQL 强化学习训练小型代理来学习可泛化的数据混合启发式，在数学推理持续预训练中平衡源领域和目标领域性能，且可泛化到未见过的源领域、目标模型和领域空间。

**[Demystifying Data Organization for Enhanced LLM Training](demystifying_data_organization_for_enhanced_llm_training.md)**

:   这篇论文系统研究 LLM 训练中“样本出现顺序”的影响，复用已有样本级质量/难度分数，提出边界强化、循环复习、连续课程和局部多样性四条数据组织原则，并用 STR 与 SAW 在预训练和 SFT 中稳定提升性能。

**[Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)**

:   作者用概率层级化上下文无关文法 (HPCFG) 构造一组"无污染、有边界、可精确采样"的形式语言作为受控测试床，并提出"判别式 AUC 测试"作为统一指标，在 18 个 LLM、6 个家族、6 种语言上系统比较 FT 与 ICL：FT 在 in-distribution 上稳定胜出，但在 out-of-distribution 上两者打平，ICL 的归纳偏置与 FT 相近但对 token 敏感得多。

**[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)**

:   作者把 Ebbinghaus 遗忘曲线的"间隔回放"思路从"训练步数"重新对齐到"模型时间" (parameter update norm $\Delta_t = \|\Theta_t - \Theta_{t-1}\|_2$ 累积)——既用累积模型时间 $\tau_t$ 决定**何时回放**，又用最近更新强度 $\mu_t$ 与基线 $\mu_0$ 的不稳定比 $r_t$ 自适应控制**如何回放** (regularization 强度)；在 3 个 CL 基准、4 种 backbone (0.6B–13B) 上一致超越 SOTA，OP +1.2%、BWT +0.9% vs 最强 baseline VBM。

**[Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering](is_a_document_educational_or_just_wikipedia-style_--_pitfalls_of_classifier-base.md)**

:   这篇论文发现Classifier-based Quality Filtering会把“Wikipedia式写法”误当成“更有教育价值”，简单改写就能让低质量网页越过预训练数据过滤阈值，FineWeb-Edu约7%的样本会因此翻转过滤决策。

**[KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)**

:   提出知识坐标条件化预训练（KoCo），将每个文档映射为三维语义坐标（来源、内容、稳定性），作为文本前缀注入预训练，使模型获得显式的上下文感知能力，在 10 个下游任务上提升性能、加速收敛约 30%，并有效缓解幻觉。

**[On the Proper Treatment of Units in Surprisal Theory](on_the_proper_treatment_of_units_in_surprisal_theory.md)**

:   这篇论文指出 surprisal theory 中“下一个单位”的单位选择一直被预训练语言模型 tokenizer 悄悄决定，因而提出一个把模型 token、语言学单位和实验 ROI 明确分离的有限状态转导框架，并在 MECO 眼动数据上验证不同单位库存会改变 surprisal 对阅读时间的预测问题本身。

**[SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)**

:   本文提出 SAGE 优化器，通过 Lion 风格的符号更新方向和一个 $O(d)$ 内存开销的自适应阻尼缩放因子，解决了轻量级优化器在嵌入层上失败的"嵌入层困境"，在 Llama 模型（最大 1.3B）上以显著更低的优化器内存达到新的 SOTA 困惑度。

**[SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)**

:   本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。

**[Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)**

:   从理论上分析了多 Token 预测（MTP）如何通过梯度耦合机制诱导表示收缩性从而促进信念状态的涌现，但同时揭示了 MTP 的"结构性幻觉"问题（隐空间中的非法捷径），并提出 LSE-MTP 框架通过隐一致性损失和语义锚定损失将预测锚定到真实隐状态轨迹，在合成图和真实曼哈顿出租车导航上显著改善路径合法性和鲁棒性。

**[Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)**

:   本文将人类工作记忆约束（固定窗口、指数衰减、逻辑衰减、首因-近因效应）集成到 GPT-2 注意力机制中，在发展可信的小规模语料（10M/100M 词）上从头训练，发现这些约束在数据稀缺时显著提升语法准确率和人类阅读时间的预测力，且促进注意力头的功能专门化。

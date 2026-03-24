<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🤖 AAAI2026** · 共 **5** 篇

**[CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)**

:   提出 CaDyT，结合高斯过程连续时间动力学建模（Adams-Bashforth 积分器实现精确推断）和 MDL 最小描述长度原则进行结构搜索，同时解决不规则采样和因果结构识别两个挑战，在双质点弹簧/菱形图/Rössler 振荡器上大幅超越所有基线（AUPRC 0.79 vs 次优 0.39）。

**[Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)**

:   在 SCM 框架下证明最大处理效应子群必须具有同质点效应（定理1），在分区模型假设下证明最优子群发现可化简为标准监督学习（定理2），用 CART+Gini 指数即可实现——在 77 个 ACIC-2016 半合成数据集上均值处理效应 10.54（vs 次优 7.84），51.9% 排名第一。

**[MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

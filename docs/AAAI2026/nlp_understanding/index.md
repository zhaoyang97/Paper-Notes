<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**🤖 AAAI2026** · 共 **4** 篇

**[Language Models and Logic Programs for Trustworthy Tax Reasoning](language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)**

:   将税法推理重新定义为语义解析任务，让LLM将法规文本和纳税案例翻译为Prolog逻辑程序，由符号求解器执行计算，通过金标准法规+智能检索案例示例+自一致性检查，在SARA数据集上实现86/100的正确率，并将预计部署成本降至15.78美元/人（低于美国人均报税成本的6%）。

**[NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)**

:   提出 NeSTR 神经符号提示策略，通过将自然语言时间事实转化为结构化符号谓词，结合一致性验证和溯因反思修正，在零样本设置下让 LLM 实现高质量时间推理，GPT-4o-mini 上平均 F1 达 89.7（vs vanilla 64.9，TISER 85.8）。

**[REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)**

:   提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

**[Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)**

:   系统评估14个LLM在160个三段论上的推理表现，发现顶级模型在形式逻辑（句法有效性）上接近完美但在自然语言可信度判断上仅为随机水平——这与人类推理模式恰好相反；12/14模型存在信念偏差，且few-shot提示反而显著降低形式推理性能。

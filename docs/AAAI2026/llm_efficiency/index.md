<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🤖 AAAI2026** · 共 **5** 篇

**[A Content-Preserving Secure Linguistic Steganography](a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

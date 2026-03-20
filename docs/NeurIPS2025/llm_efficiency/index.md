---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🧠 NeurIPS2025** · 共 **15** 篇

**[3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)**

:   在标准的draft-target两模型推测解码的中间插入一个"qualifier"模型，构成三层金字塔式解码架构（PyramidSD），利用模型家族天然的熵梯度来分级过滤token，以模糊接受准则放宽匹配阈值，实现最高1.91×的速度提升（在RTX 4090上达到124 tok/s）。

**[A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)**

:   针对低资源语言 Tigrinya，构建了首个大规模多任务基准数据集 TiALD（13,717条YouTube评论，涵盖滥用检测、情感分析、主题分类三任务），并证明小型微调模型在低资源场景下显著优于GPT-4o等前沿LLM（F1: 86.67% vs 79.31%）。

**[A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)**

:   将 LLM 迭代交互中的多目标优化建模为 SDE（漂移-扩散过程），通过干扰矩阵量化目标间的耦合模式，通过特征值谱分析策略收敛行为，在代码生成（安全性、效率、功能性三目标）上验证了不同策略的收敛率（0.33-1.29）和可预测性（$R^2$ 达 0.74）。

**[A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](a_unified_framework_for_establishing_the_universal_approxima.md)**

:   本文建立了一个统一的理论框架来证明各类Transformer架构的万能逼近性(UAP)，将UAP归结为两个可验证条件——前馈层的非线性仿射不变性和注意力层的token可区分性——并利用解析性假设将后者简化为仅需检验两样本情形。

**[Approximately Aligned Decoding](approximately_aligned_decoding.md)**

:   提出 Approximately Aligned Decoding (AprAD)，一种利用投机解码（speculative decoding）中的前缀选择算法来实现LLM受约束生成的方法——在遇到约束违反时，既不像约束生成那样仅回退一步（导致极端概率放大），也不像ASAp那样完全重新采样（计算成本过高），而是通过投机采样智能选择回退位置，在输出分布失真和计算效率之间取得良好平衡。

**[Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

**[DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC 提出了一种动态分解算法，在推理时根据每一步的 z-score（采样奖励的标准化最大值）自动、递归地调整推理步骤的粒度——困难步骤分更细、简单步骤一步跨过——可以即插即用地与贪心搜索、Beam Search、MCTS 结合，在 APPS、MATH、LiveCodeBench 上以更少的 token 预算达到更高的 pass@k。

**[Document Summarization with Conformal Importance Guarantees](document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)**

:   HyGen是干扰感知LLM推理系统，通过延迟预测和虚拟队列调度实现在线离线工作负载的弹性共置，保证SLO同时获得3.87-5.84倍吞吐改进。

**[Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)**

:   anGPT通过高维集中度实现近似归一化，消除权重衰减和预热，相比QK归一化GPT实现40%收敛加速。

**[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   提出动态分层稀疏注意力 (DHSA)，通过自适应 chunk 分割 + chunk 级相似度预测 + 上采样到 token 级的分层框架，在不重训基座模型的前提下将密集注意力替换为稀疏注意力，在 Gemma2/3 上实现与密集注意力同等精度、20-60% prefill 延迟降低和 35% 峰值内存节省。

**[On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   首次系统分析 MoE 在结构化复杂任务上的表达能力：证明浅层 MoE 可在低维流形上克服维度诅咒（近似速率由内在维度 $d$ 而非环境维度 $D$ 决定），深层 MoE 通过 $E$ 专家 × $L$ 层的分层组合可高效近似有 $E^L$ 段的分段函数，远超朴素上界 $LE$。

**[Scale-invariant Attention](scale-invariant_attention.md)**

:   借鉴自然图像的尺度不变性，提出对 attention logits 做位置相关的乘性缩放和加性偏移变换，使注意力在不同 token 范围上的总权重和稀疏度满足尺度不变性，从而实现从短序列训练到长序列推理的零样本泛化（4k→64k 仅需一个超参数 $\tau$）。

**[上下文学习中的技术债务：长序列中的递减效率](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   揭示ICL作为学习算法在少射大样本制度下存在本质低效：少射ICL样本复杂度接近贝叶斯最优(1.1×)，而多射时恶化至1.45×，信息论分析证明此低效来自非递减过剩风险。

**[The Emergence Of Sparse Attention Impact Of Data Distribution And Benefits Of Re](the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   从理论和实验两方面解释稀疏attention的涌现时机：推导出plateau时间遵循幂律缩放（与序列长度和维度相关），并证明数据重复（in-context或cross-sample）可将涌现加速2-4倍。

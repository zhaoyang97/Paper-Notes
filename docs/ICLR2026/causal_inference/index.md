<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🔬 ICLR2026** · 共 **9** 篇

**[Action-Guided Attention for Video Action Anticipation](action-guided_attention_for_video_action_anticipation.md)**

:   提出动作引导注意力 (AGA) 机制，用模型自身的动作预测序列作为注意力的 Query 和 Key（而非像素特征），结合自适应门控融合历史上下文和当前帧特征，在 EPIC-Kitchens-100 上实现从验证集到测试集的良好泛化，同时支持训练后的可解释性分析。

**[AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)**

:   提出AgentTrace框架，从多智能体系统的执行日志中构建因果图，通过反向追踪+轻量级特征排序（五组特征的加权线性组合）定位根因节点，在550个合成故障场景上Hit@1达94.9%，延迟0.12秒，比LLM分析快69倍。

**[Copy-Paste to Mitigate Large Language Model Hallucinations](copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)**

:   提出 PCG（Perceptual Counterfactual Geodesic）方法，在鲁棒感知流形上通过测地线优化生成语义忠实的反事实解释，两阶段优化确保路径既感知自然又达到目标类别，在 AFHQ 上 FID=8.3 远优于 RSGD 的 12.9。

**[Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)**

:   提出首个对条件分位数比较器 (CQC) 的**直接估计方法**，通过显式参数化 CQC 并结合双重鲁棒梯度下降，在理论上保持双重鲁棒性的同时，实验中在估计精度、可解释性和计算效率上全面优于现有的间接反演方法。

**[Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)**

:   提出 E-CIT（集成条件独立性检验）框架，通过将数据分割为子集后独立执行检验并基于**稳定分布**的 p 值聚合方法合并结果，将任意条件独立性检验的计算复杂度降至关于样本量线性，同时在重尾噪声和真实数据等复杂场景下保持甚至提升检验功效。

**[Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)**

:   系统研究偏好模型对五种表面特征（冗长、结构化、术语、谄媚、模糊）的过度依赖——通过因果反事实对量化偏差来源于训练数据的分布不平衡，并提出基于**反事实数据增强 (CDA)** 的后训练方法，将模型与人类判断的平均失校准率从 39.4% 降至 32.5%。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)**

:   提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

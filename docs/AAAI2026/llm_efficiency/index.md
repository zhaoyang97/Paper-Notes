<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🤖 AAAI2026** · 共 **14** 篇

**[A Content-Preserving Secure Linguistic Steganography](a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[Attention Retention for Continual Learning with Vision Transformers](attention_retention_for_continual_learning_with_vision_transformers.md)**

:   提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。

**[C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)**

:   提出 C3TG 框架，通过两阶段方法实现多维度细粒度可控文本生成：生成阶段用加权 KL 散度融合属性分布调整 token 概率，优化阶段用能量函数（分类器分数 + 冲突惩罚项）结合 Feedback Agent 迭代重写，在 17 个属性子类上达到 90.4% 属性准确率且大幅降低毒性。

**[Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)**

:   提出一种大小模型协作框架，通过对本地查询进行"主题迁移+数值替换"的两阶段匿名化来保护敏感数据，同时让远端 GPT-4 以可执行 Python 代码（即插即用工具）形式返回推理方案，本地仅需做数值回代即可获得答案，在 FinQA 和 MultiHiertt 上准确率提升 16-44% 且数据泄露降低 2-45%。

**[Do Retrieval Augmented Language Models Know When They Don't Know?](do_retrieval_augmented_language_models_know_when_they_dont_know.md)**

:   系统分析RAG模型的拒绝校准问题，发现RALM在检索文档全部不相关时过度拒绝率超过55%（即使模型内部知识足够回答），提出结合不确定性估计和拒绝感知微调的机制来平衡拒绝与回答质量。

**[ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)**

:   提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

**[Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)**

:   提出 Learning-from-the-Undesirable (LfU)，一种面向 SFT 的正则化方法，通过对辅助模型施加梯度上升模拟"不良行为"，再通过表示级一致性损失约束原模型与不良模型的内部表征保持一致，有效缓解有限数据微调中的过拟合、遗忘和对抗脆弱性问题。

**[MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)**

:   提出 MicroEvoEval，首个面向图像级微观结构演化预测的标准化基准：涵盖 4 个代表性物理任务（平面波、晶粒生长、旋节分解、枝晶凝固）、14 个模型（5 个领域特定 + 9 个通用时空架构）、多维度评估（数值精度 + 物理保真度 + 计算效率），发现现代通用架构（如 VMamba）在长期稳定性和物理保真度上优于领域特定模型，且计算效率高一个数量级。

**[Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)**

:   将 Agent 伦理行为引导建模为模型编辑任务（Behavior Editing），提出基于心理学道德理论的三层 BehaviorBench 基准，在 9 个开源模型和 20 个闭源模型上验证了模型编辑可以精确地将 Agent 引导向善意或恶意方向，且单次编辑可导致全局道德对齐偏移。

**[Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution](think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)**

:   提出 AIM（Active Inference Modeling）框架，在去中心化多智能体强化学习中，不依赖通信机制，仅基于局部观测建模队友的主动推理过程（感知-信念-动作三重肖像），并通过准确性-相关性双重过滤机制选择性融合队友信念，在 SMAC、SMACv2、MPE 和 GRF 四大基准上取得最优或接近最优表现。

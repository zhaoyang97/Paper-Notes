<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🧠 NeurIPS2025** · 共 **13** 篇

**[A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)**

:   提出一个完全零样本、无需训练的视频异常分析框架，通过Intra-Task Reasoning（置信度门控的自我精化）和Inter-Task Chaining（从时序检测到空间定位到语义理解的级联prompt传递），在4个benchmark上全面超越先前零样本方法4-6% AUC。

**[Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)**

:   提出 Adv-SSL，通过将协方差正则项的 Frobenius 范数重写为 minimax 对偶形式，消除了 Barlow Twins 等方法中样本级风险的有偏估计问题，在不增加额外计算成本的前提下显著提升下游分类性能，并给出端到端的理论收敛保证。

**[BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出BrainOmni——首个统一EEG和MEG的脑信号基础模型，通过BrainTokenizer（含物理传感器编码）实现设备不可知的信号离散化，使用Criss-Cross Transformer进行自监督预训练，在阿尔茨海默病检测等任务上提升11.7个百分点并实现未见设备的零样本泛化。

**[Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)**

:   提出CoRAG框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调LLM学习迭代检索和推理，并支持多种测试时解码策略（贪心/Best-of-N/树搜索）灵活扩展计算，在多跳QA上提升26+ EM点，KILT基准9/10任务达SOTA。

**[Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)**

:   推导了一般情况下 KL 散度关于 JS 散度的新的紧致可计算下界，证明最大化 JSD 目标等价于最大化互信息的一个下界，为判别式学习在 MI 基础表示学习中的使用提供了理论基础，并在 MI 估计和 Information Bottleneck 中验证了其紧致性和实用性。

**[Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)**

:   提出CoSO框架，通过从每步梯度的SVD动态导出连续子空间（而非LoRA的固定子空间），结合历史任务正交投影防止干扰和Frequent Directions高效聚合梯度信息，在ImageNet-R 20任务上比最佳baseline提升2.77个百分点。

**[Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)**

:   论文研究能否用纯表示学习替代显式搜索来承担部分时序推理，指出标准 temporal contrastive learning 容易抓住伪特征而失去时序结构，进一步提出 CRTR（Combinatorial Representations for Temporal Reasoning），通过特制负采样从理论上去除伪特征，学到同时编码感知与时序结构的表示，在 Sokoban 和 Rubik's Cube 上取得强结果，甚至可在不依赖外部搜索算法的情况下求解任意初始魔方状态。

**[Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)**

:   提出NuCLR自监督框架，通过对比学习对群体神经活动中同一神经元的不同时间窗口拉近、不同神经元推远，学习包含群体上下文的神经元级表征，在细胞类型和脑区解码上达到新SOTA，并首次展示了跨动物零样本泛化和数据缩放规律。

**[Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization](minimal_semantic_sufficiency_meets_unsupervised_domain_generalization.md)**

:   MS-UDG 在无类别标签和域标签的条件下，通过信息解纠缠模块（IDM）将表征分解为语义和变异成分，配合最小语义充分性优化模块（SROM）最大化语义信息同时最小化变异干扰，在 PACS 上达 72.89% 准确率（+1.5% vs CycleMAE），理论证明最小充分语义表征最小化下游贝叶斯错误率。

**[Sciarena An Open Evaluation Platform For Non-Verifiable Scientific Literature-Gr](sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)**

:   构建SciArena——社区驱动的科学文献基础模型开放评估平台，支持47个模型和20K+偏好投票，同时发布SciArena-Eval元基准评估自动评估系统判断能力。

**[Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)**

:   从理论上证明自监督对比学习（DCL）近似等价于一种有监督对比损失（NSCL），两者差距以 $O(1/C)$ 速度随类别数增加而消失；进一步证明 NSCL 全局最优解满足 Neural Collapse（增强坍缩 + 类内坍缩 + Simplex ETF），并提出基于方向性 CDNV 的更紧的 few-shot 误差界。

**[T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)**

:   提出 T-REGS——一种基于最小生成树(MST)长度最大化的自监督学习正则化框架，理论证明可同时防止维度坍缩并促进表示分布均匀性，在紧致黎曼流形上成立，实验在标准 JE-SSL 基准上验证了有效性。

**[Tabstar A Tabular Foundation Model For Tabular Data With Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)**

:   提出 TabSTAR，一个专为含文本字段的表格数据设计的基础模型：通过解冻文本编码器（e5-small-v2）端到端优化文本表征 + 目标感知 token 注入分类目标语义信息 + 无数据集特定参数的架构实现跨数据集迁移学习，在 350 个数据集上预训练后，分类任务上 14 个数据集中 12 个超越 CatBoost-Tuned（4h 调参），8/11 超越 TabPFN-v2。

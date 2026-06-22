---
title: >-
  ICML2026 自监督/表示学习论文汇总 · 28篇论文解读
description: >-
  28篇ICML2026的自监督/表示学习方向论文解读，涵盖自监督学习、少样本学习、扩散模型、对齐/RLHF、LLM、持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "少样本学习"
  - "扩散模型"
  - "对齐/RLHF"
  - "LLM"
  - "持续学习"
item_list:
  - u: "a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive/"
    t: "A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning"
  - u: "active_learning_with_foundation_model_priors_efficient_learning_under_class_imba/"
    t: "Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance"
  - u: "beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni/"
    t: "Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning"
  - u: "can_local_learning_match_self-supervised_backpropagation/"
    t: "Can Local Learning Match Self-Supervised Backpropagation?"
  - u: "data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise/"
    t: "Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise"
  - u: "flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f/"
    t: "FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction"
  - u: "from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete/"
    t: "From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection"
  - u: "how_neural_is_a_neural_foundation_model/"
    t: "How 'Neural' is a Neural Foundation Model?"
  - u: "inconsistency-aware_minimization_improving_generalization_with_unlabeled_data/"
    t: "Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data"
  - u: "infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate/"
    t: "InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation"
  - u: "learning_graph_foundation_models_on_riemannian_graph-of-graphs/"
    t: "Learning Graph Foundation Models on Riemannian Graph-of-Graphs"
  - u: "learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation/"
    t: "Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation"
  - u: "lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel/"
    t: "LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems"
  - u: "limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found/"
    t: "LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models"
  - u: "mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad/"
    t: "Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment"
  - u: "nitp_next_implicit_token_prediction_for_llm_pre-training/"
    t: "NITP: Next Implicit Token Prediction for LLM Pre-training"
  - u: "numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models/"
    t: "NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models"
  - u: "partco_part-level_correspondence_priors_enhance_category_discovery/"
    t: "PartCo: Part-Level Correspondence Priors Enhance Category Discovery"
  - u: "provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali/"
    t: "Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch"
  - u: "riemannian_metric_matching_for_scalable_geometric_modeling_of_distributions/"
    t: "Riemannian Metric Matching for Scalable Geometric Modeling of Distributions"
  - u: "scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts/"
    t: "Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts"
  - u: "statistical_consistency_and_generalization_of_contrastive_representation_learnin/"
    t: "Statistical Consistency and Generalization of Contrastive Representation Learning"
  - u: "the_geometry_of_projection_heads_conditioning_invariance_and_collapse/"
    t: "The Geometry of Projection Heads: Conditioning, Invariance and Collapse"
  - u: "towards_one-for-all_anomaly_detection_for_tabular_data/"
    t: "Towards One-for-All Anomaly Detection for Tabular Data"
  - u: "tracer_persistent_regularization_for_robust_multimodal_finetuning/"
    t: "TRACER: 用 WMA teacher + 几何分解证明的鲁棒多模态微调"
  - u: "understanding_self-supervised_learning_via_latent_distribution_matching/"
    t: "Understanding Self-Supervised Learning via Latent Distribution Matching"
  - u: "when_softmax_fails_at_the_top_extreme_value_corrections_for_infonce/"
    t: "When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE"
  - u: "zero-flow_encoders/"
    t: "Zero-Flow Encoders"
item_total: 28
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🧪 ICML2026** · **28** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×3 · 少样本学习 ×2

**[A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)**

:   本文改进了监督对比学习（在有限标注数据池中构造元组）的样本复杂度上界，通过两个不同的U-统计量估计器，在极值多类场景下实现从依赖最小类概率的界到仅依赖类别数或样本规模的界的突破。

**[Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance](active_learning_with_foundation_model_priors_efficient_learning_under_class_imba.md)**

:   这篇论文提出 PriorAL：用基础模型（大模型）的预测作为先验，与小模型做「专家乘积」联合决策，再用一个不平衡感知的熵过滤把无标注池切成「干净集（免费打伪标签）+ 噪声集（花预算请人标）」，从而在类别不平衡叠加标签噪声的图像/文本任务上，比最强主动学习基线省下 50%+ 的标注成本。

**[Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)**

:   本文提出 SAGE，把"估计未标注数据分布"换成"在表征空间做结构推断"，用 simplex ETF 几何锚 + 高阶图传播 + 分布无关可靠性加权三件套，在极端标签稀缺且未标注分布任意的 UniSSL 设定下取得平均 8.52% 的准确率提升。

**[Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)**

:   本文从理论上证明了在深度线性网络中局部自监督学习（local-SSL）可以精确实现全局反向传播自监督学习（BP-SSL）的梯度更新，并据此提出 CLAPP++ 算法（引入 2D 空间依赖和直接反馈），在 CIFAR-10/STL-10/Tiny ImageNet 上达到了与全局 BP-SSL 相当的性能，刷新了 local-SSL 的 SOTA。

**[Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)**

:   作者证明对比学习里的"预定义数据增强 (旋转/裁剪/翻转)"等价于对 Positive-incentive Noise (π-noise) 的点估计, 然后把 π-noise 从"点估计"升级为可学习分布, 训练一个 π-noise 生成器在原图上加可学噪声当增强 (PiNDA), 使 SimCLR / BYOL / SimSiam / MoCo / DINO 在 vision 上稳定涨点, 且天然适配 HAR / Reuters / Epsilon 等无人工增强的非视觉数据。

**[FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)**

:   FLAG 把"从 H&E 病理图预测空间基因表达"重新表述为结构化分布生成问题，用一个固定的空间图编码器把组织拓扑压成条件向量，再用 DiT 在基因维度去噪，并通过基因基础模型 (GFM) 的中间层对齐注入基因-基因调控先验，从而在保持 PCC/MSE 竞争力的同时把基因结构相关性 (GSC) 和空间结构相关性 (SSC) 拉到新的高度。

**[From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)**

:   本文提出 OutFormer —— 一个用 GMM/SCM/Copula 三类合成先验混合预训练、靠多臂老虎机自演化课程稳定多任务训练的表格 PFN，做到零样本表格异常检测：上下文 (in-context) 吃训练数据、前向一步给标签，在 ADBench 与两个新 1500+ 数据集 benchmark 上同时拿到 SOTA 排名和接近 shallow 模型的推理延迟。

**[How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)**

:   作者把一只"小白鼠视觉皮层的 SOTA 基础模型（FNN）"当成生理学实验对象，用解码流形 / 编码流形 / 解码轨迹三件套挨个分析它的 encoder / recurrent / readout，发现 FNN 的拟合精度主要靠 readout 那一堆同质 feature map 撑起来，而真正"像大脑"的只有 recurrent 模块；并用新提出的 tubularity 指标定量地说"早期编码层缺少生物级时间结构"，给未来神经基础模型给出"早期加 recurrence、readout 减少 feature 维度"的明确建议。

**[Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](inconsistency-aware_minimization_improving_generalization_with_unlabeled_data.md)**

:   本文提出一种只用无标签数据就能计算的"局部不一致性" $S_\rho(\theta)$ —— 即参数球内 KL 散度的最坏值 —— 并把它当作训练正则项，得到 IAM 优化器，在监督任务上和 SAM/ASAM 持平甚至更好，在半监督 (FixMatch) 与自监督 (SimCLR) 场景下因能吃无标签批量数据而带来额外提升。

**[InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)**

:   InfoAtlas 把互信息估计从"每个数据集都要从头训一个评估网络"的优化问题，改造成一个用大规模合成数据预训练好的超网络的"一次前向推理"问题，做到与 MINE/MINDE 等神经估计器相当的精度同时 100× 提速。

**[Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)**

:   R-GFM 把"不同 hop 数"的子图当作上层 Graph-of-Graphs 的节点，再用一套动态 MoE 路由把每个 GoG 分配到曲率最匹配的 Riemannian 流形（双曲 / 欧氏 / 球面），同时解决了现有图基础模型固定 receptive field 与单一 Euclidean 嵌入两个先天缺陷，下游最高带来 49% 相对提升。

**[Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)**

:   本文提出 Relational Task Extrapolator (RTE)，把"训练支撑集之外的新任务"重新解释为"已知锚点任务 + 已见过的任务间变换"的组合问题，并训练一个关系算子 $\Psi$ 在测试时拼装这对锚点-变换以预测未知任务的输出。

**[LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel.md)**

:   针对大模型 selective prediction 中"UCB 风险界过于保守、能用阈值很少"这个老问题，作者把"接受后错误率 ≤ α"重写成一条关于选择/错误两个 0-1 指示函数的**线性期望约束**，由此推出一个只依赖校准集的有限样本充分条件（Eq. 5），既保持有限样本严格保证又显著比 UCB 紧，同时把同一套框架自然推广到两模型路由系统并联合标定两个阈值，在 CommonsenseQA / TriviaQA / ScienceQA / MM-Vet v2 上 power 普涨、TriviaQA 上比 Clopper-Pearson UCB 多接受 9.5% 样本。

**[LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)**

:   针对 TabPFN-v2 等表格基础模型在浅层出现严重低秩坍缩、且最后一层 sample-attention 对预测信号贡献微弱的两个病灶，作者提出用径向基函数把每个标量扩展成一组局部响应（RaBEL）来打开"值方向"的自由度，并把双向注意力块从 F→S→N 重排成 S→N→F 以确保所有注意力路径都汇入读出，仅用 2M 参数就在主流表格 benchmark 上稳定胜过 7M 的 TabPFN-v2 和 27M 的 TabICL。

**[Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)**

:   针对 TabPFN 这类把训练集当作 in-context 直接喂进 attention 的"表格基础模型"做后验校正——发现它会严重过拟合训练集 majority class, 提出 DistPFN：用 $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$ 这一行后验重加权, 在 253 个 OpenML 数据集上把 TabPFN-v2 在 $\beta=5$ 强标签漂移下的准确率从 72.7% 拉到 76.9%, 不用重训、不用估测试先验、不动架构。

**[NITP: Next Implicit Token Prediction for LLM Pre-training](nitp_next_implicit_token_prediction_for_llm_pre-training.md)**

:   NITP 通过用**浅层表示作为隐式目标**为最后隐藏状态提供连续的表示空间监督——补充标准 NTP 防止隐藏表示退化为低维各向异性配置，在 9B MoE 上 MMLU-Pro 提升 5.7%、推理任务普遍提升 4-6%，额外计算开销仅 ~2%。

**[NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)**

:   NumLeak 通过**四层诊断协议**检测和量化基础模型对公开数值基准（金融因子、宏观经济数据、气候数据）的记忆化程度——揭示这类污染如何渗漏到下游金融信号中，并通过系统提示防御减缓风险；Opus 4.7 在 Mkt-RF 因子上的 within-25 bps 精度达 0.60、Pearson r = 0.99。

**[PartCo: Part-Level Correspondence Priors Enhance Category Discovery](partco_part-level_correspondence_priors_enhance_category_discovery.md)**

:   PartCo 通过显式利用 Vision Transformer 的补丁令牌中蕴含的**部分级特征对应关系**，引入一个**即插即用**的框架来增强广义类别发现——在 CUB / Stanford-Cars / ImageNet-100 等多个基准上将 SimGCD / SPTNet / FlipClass 等基线提升 2-10%。

**[Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)**

:   作者证明:对比学习里典型的三元组任务,只要嵌入维度 $d$ 小于真维度 $D$ 的某个常数倍,无论用什么优化器,准确率都会"坍缩"到 1 维随机嵌入的 50% baseline,而且在算法层面这件事在 Unique Games 假设下也无法被多项式时间逼近。

**[Riemannian Metric Matching for Scalable Geometric Modeling of Distributions](riemannian_metric_matching_for_scalable_geometric_modeling_of_distributions.md)**

:   把"数据流形的黎曼度量"重写成一个 carré du champ 算子，再用一条去噪式的条件回归损失让神经网络直接学这个算子，从而不建 kNN 图、不算大网络 Jacobian，就能在高维数据上以常数代价摊销地估计内蕴维度、切空间和测地路径，推理最高比基于 kNN 的扩散几何估计快约 400 倍。

**[Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)**

:   作者提出 CaRE：在 ViT 每个 block 里塞一个 **两级路由 MoE (BR-MoE)** ——先靠"类感知器"按熵选 Top-M 个相关任务路由，再由这些路由各自激活 Top-K 任务专家并叠加一个共享 EMA 专家，于是哪怕任务序列拉到 300+ 也能既保留旧知识又持续吸纳新类，并把"长序列 CIL"这块此前没人正经做的空白填上（顺便发布了 1000 类的 OmniBenchmark-1K 基准）。

**[Statistical Consistency and Generalization of Contrastive Representation Learning](statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)**

:   本文首次为对比表示学习 (CRL) 建立了"上游对比损失最小化等价于下游 AUC 型检索性能最优"的 Fisher / 统计一致性, 并给出依赖于正样本数 $n$ 和负样本数 $m$ 的精细泛化界 $O(1/m+1/\sqrt n)$ (监督) 与 $O(1/\sqrt m+1/\sqrt n)$ (自监督), 从而首次从理论上解释了 CLIP / SimCLR 使用上万负样本能持续涨点的现象。

**[The Geometry of Projection Heads: Conditioning, Invariance and Collapse](the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)**

:   本文从黎曼几何视角把自监督学习中的投影头分析为可训练的度量张量，证明其作用是动态白化优化景观、用光滑激活的负曲率逃脱坍缩鞍点、并沿数据增强方向诱导度量奇异性——三件事一起解释了"训练时需要、推理时丢弃"这一长期谜团。

**[Towards One-for-All Anomaly Detection for Tabular Data](towards_one-for-all_anomaly_detection_for_tabular_data.md)**

:   提出 OFA-TAD：用"邻居距离"作为跨域通用的异常线索，在多种特征变换诱导的度量空间里抽多视图距离表示，再用专家混合（MoE）门控自适应融合，训练一次即可直接泛化到未见过的表格数据集做异常检测，无需任何目标域微调。

**[TRACER: 用 WMA teacher + 几何分解证明的鲁棒多模态微调](tracer_persistent_regularization_for_robust_multimodal_finetuning.md)**

:   TRACER 用闭式解理论把对比微调的几何分解为"任务子空间"+"正交保留"两部分，证明 EMA teacher 会坍缩失去正则化力，提出 Weighted Moving Average (WMA) teacher 保持 finite-horizon 持续约束力且对任务子空间无偏收敛；在 CLIP ViT-B/16 上 ImageNet 分布偏移平均提升至 64.07% vs CaRot 62.54%。

**[Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)**

:   作者把对比 / 非对比 / 预测式 SSL 统一为"潜在分布匹配 (LDM)"：最大化样本在假设潜在模型下的对数概率 (alignment) + 最大化潜在熵 (uniformity)，并基于此推出带 Kalman 预测器的非线性可识别预测式 SSL。

**[When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE](when_softmax_fails_at_the_top_extreme_value_corrections_for_infonce.md)**

:   这篇论文把 InfoNCE 解释为 top-1 选择似然，指出标准 softmax 隐含 Gumbel 尾部分布假设，而归一化 embedding 的高相似度 hard negatives 更常呈现有限端点的 Weibull 行为，因此提出无额外参数的 WEINCE，用 batch 内尾部统计自适应混合 softmax logit 和 endpoint shortfall logit，稳定提升自监督表征质量。

**[Zero-Flow Encoders](zero-flow_encoders.md)**

:   论文发现一个反直觉现象——用独立耦合训练的整流流（rectified flow）在 $t=0.5$ 处处为零当且仅当源/目标分布相同（"零流判据"），并把它推广到条件分布，证明 $\mathbf{v}_{t=0.5}=0$ 等价于编码器 $f(Y)$ 对预测 $X$ 充分（条件独立），由此设计出一个无需仿真、无需参数化密度假设的最小二乘损失，统一地学习图模型中的马尔可夫毯和自监督表示，并天然规避对比学习的"捷径问题"。

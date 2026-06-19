---
title: >-
  ICLR2026 自监督/表示学习论文汇总 · 24篇论文解读
description: >-
  24篇ICLR2026的自监督/表示学习方向论文解读，涵盖自监督学习、对抗鲁棒、对齐/RLHF、少样本学习、扩散模型、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "少样本学习"
  - "扩散模型"
  - "时序预测"
item_list:
  - u: "adaptive_gaussian_expansion_for_on-the-fly_category_discovery/"
    t: "Adaptive Gaussian Expansion for On-the-fly Category Discovery"
  - u: "adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat/"
    t: "Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts"
  - u: "adversarial_encoding_perturbation_and_synthesis_for_set_representation_auxiliary/"
    t: "Adversarial Encoding Perturbation and Synthesis for Set Representation Auxiliary Learning"
  - u: "architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment/"
    t: "Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment"
  - u: "attention_please_revisiting_attentive_probing_through_the_lens_of_efficiency/"
    t: "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency"
  - u: "bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in/"
    t: "Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding"
  - u: "bidirectional_predictive_coding/"
    t: "Bidirectional Predictive Coding"
  - u: "boosting_open_set_recognition_performance_through_modulated_representation_learn/"
    t: "Boosting Open Set Recognition Performance through Modulated Representation Learning"
  - u: "calibrated_information_bottleneck_for_trusted_multi-modal_clustering/"
    t: "Calibrated Information Bottleneck for Trusted Multi-modal Clustering"
  - u: "chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization/"
    t: "Chart Deep Research in LVLMs via Parallel Relative Policy Optimization"
  - u: "difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect/"
    t: "Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective"
  - u: "exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c/"
    t: "Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification"
  - u: "fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce/"
    t: "Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning"
  - u: "infonce_induces_gaussian_distribution/"
    t: "InfoNCE Induces Gaussian Distribution"
  - u: "maximizing_asynchronicity_in_event-based_neural_networks/"
    t: "Maximizing Asynchronicity in Event-based Neural Networks"
  - u: "maximizing_incremental_information_entropy_for_contrastive_learning/"
    t: "Maximizing Incremental Information Entropy for Contrastive Learning"
  - u: "no_other_representation_component_is_needed_diffusion_transformers_can_provide_r/"
    t: "No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves"
  - u: "ponderlm_pretraining_language_models_to_ponder_in_continuous_space/"
    t: "PonderLM: Pretraining Language Models to Ponder in Continuous Space"
  - u: "regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found/"
    t: "Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models"
  - u: "snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i/"
    t: "SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty"
  - u: "soft_equivariance_regularization_for_invariant_self-supervised_learning/"
    t: "Soft Equivariance Regularization for Invariant Self-Supervised Learning"
  - u: "temporal_slowness_in_central_vision_drives_semantic_object_learning/"
    t: "Temporal Slowness in Central Vision Drives Semantic Object Learning"
  - u: "test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting/"
    t: "Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting"
  - u: "why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic/"
    t: "Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🔬 ICLR2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×5

**[Adaptive Gaussian Expansion for On-the-fly Category Discovery](adaptive_gaussian_expansion_for_on-the-fly_category_discovery.md)**

:   本文先证明了"即时类别发现"（OCD）任务存在一个被现有哈希方法忽视的性能下界，进而把 OCD 拆成"开放集识别 + 实时新类发现"两个子任务，用软阈值先把已知类直接判出，再用基于多元高斯密度的自适应高斯扩展（AGE）在线增量地聚出新类，在多个数据集上把整体准确率平均拉高约 10%。

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[Adversarial Encoding Perturbation and Synthesis for Set Representation Auxiliary Learning](adversarial_encoding_perturbation_and_synthesis_for_set_representation_auxiliary.md)**

:   SRAL 把每个集合看成一个经验分布，用 2-Sliced-Wasserstein 距离编码出能感知"集合间差异"的表示，再在**特征/编码层而非输入层**注入对抗扰动、用 min-max 优化逼模型抵抗最坏扰动，作为一个可插到各种下游任务的自监督辅助目标；理论上证明该目标在期望意义下等价于优化集合间的 Sliced-Wasserstein 距离，在集合相似度排序、捆绑推荐、点云分类、主题集扩展四类任务上稳定超过现有集合编码器。

**[Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)**

:   PEA 把"域偏移"拆解成嵌入空间里的平移（均值漂移）、缩放（方差漂移）、旋转（协方差漂移）三种几何畸变，然后用一套**无反向传播、与架构无关**的逐层协方差对齐流程，仅靠每个 batch 两次前向就把偏移的中间特征拉回源域分布，在 ImageNet-C / CIFAR-C 上达到 SOTA 精度的同时，内存只占 ~900MB、能直接跑在 Jetson Orin Nano 边缘设备上。

**[Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency](attention_please_revisiting_attentive_probing_through_the_lens_of_efficiency.md)**

:   针对「注意力探测」这一日益流行的冻结表示评估协议普遍参数臃肿的问题，本文先把已有方法统一成一个框架，再利用多头交叉注意力与多查询交叉注意力的**数学等价性**砍掉冗余投影矩阵，提出极轻量的 Efficient Probing（EP）——在 ImageNet-1K 上以不到 1.4M 参数把 MAE ViT-B 的探测精度从线性探测的 67.7% 拉到 75.6%，且各预训练范式上全面超越线性探测与已有注意力探测方法。

**[Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding](bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in.md)**

:   BTTA-DG 把每条 EEG 试次的逐时刻预测序列压成一个 Dirichlet 参数向量，用历史试次拟合的 GMM 当似然、深度模型输出当先验，做一次无梯度的贝叶斯后验校准，在运动想象脑机接口的跨被试/跨 session 迁移上达到 SOTA 且实时（15.7 ms/试次）。

**[Bidirectional Predictive Coding](bidirectional_predictive_coding.md)**

:   本文提出双向预测编码（bPC），用一个能量函数同时容纳「自上而下生成」和「自下而上判别」两种推断，让同一套生物可实现的局部电路既能像 discPC 那样准确分类、又能像 genPC 那样生成与重建，并在跨模态联想、遮挡补全等类脑任务上超过现有的单向 / 混合 PC 模型。

**[Boosting Open Set Recognition Performance through Modulated Representation Learning](boosting_open_set_recognition_performance_through_modulated_representation_learn.md)**

:   这篇论文指出几乎所有开集识别（OSR）方法都给 logits 用一个**固定温度** $\tau$，导致模型只能停在「实例级特征」和「类级特征」频谱的某一点；作者提出在训练过程中**调度温度**（核心是新颖的负余弦调度 NegCosSch），让模型先用低温画出粗决策边界、再升温把同类样本收紧，从而在不增加任何计算开销的前提下，把开集和闭集性能一起提升，尤其在更难的语义偏移基准（SSB）上收益最大。

**[Calibrated Information Bottleneck for Trusted Multi-modal Clustering](calibrated_information_bottleneck_for_trusted_multi-modal_clustering.md)**

:   针对信息瓶颈（IB）多模态聚类高度依赖"准确的互信息估计 + 干净伪标签"这两件做不到的事，本文提出 CLIB——用"一个主聚类头 + 多个模态校准头"的并行多头结构，让模态间互相纠偏，再配上一个基于信息冗余度的动态伪标签筛选机制，既把聚类准确率（Caltech-3V 上 ACC 77.8%）做上去，又把过自信问题（ECE 在多个数据集上腰斩）压下来。

**[Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)**

:   提出 PRPO（Parallel Relative Policy Optimization），通过在奖励维度和数据类型两个层面做并行解耦优化，解决 GRPO 在多维奖励信号干扰和异构数据梯度冲突下的训练瓶颈；同时构建 MCDR-Bench，基于"错误唯一性原则"将主观生成评估转化为客观错误识别，实现图表深度研究能力的量化评估。

**[Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)**

:   通过相似度图模型理论分析严格证明"困难样本"（跨类高相似度样本对）会损害无监督对比学习性能——困难样本使泛化误差界严格恶化，提出删除困难样本、调节 margin 和温度缩放三种理论指导的缓解策略，在 TinyImageNet 上带来高达 10.42% 的线性探测准确率提升。这一发现是反直觉的：深度学习中通常"更多数据更好"，但对比学习中精心移除困难样本反而有益。

**[Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification](exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c.md)**

:   发现病理基础模型特征具有低维流形几何结构（有效秩仅29.7/512维），而线性层会破坏这种结构导致少样本过拟合，提出即插即用的MR Block（冻结随机矩阵做几何锚+低秩残差路径做任务适配）在少样本WSI分类上达到SOTA。

**[Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)**

:   受果蝇嗅觉回路启发，提出 Fly-CL 框架，通过稀疏随机投影+top-k操作+流式岭分类三阶段渐进去相关，在预训练模型持续学习中大幅降低训练时间的同时达到SOTA水平。

**[InfoNCE Induces Gaussian Distribution](infonce_induces_gaussian_distribution.md)**

:   从理论上证明 InfoNCE 损失函数在两种互补机制下会诱导表征趋向高斯分布：经验理想化路线（对齐+球面均匀性→高斯）和正则化路线（消失正则项→各向同性高斯），并在合成数据和 CIFAR-10 上验证。

**[Maximizing Asynchronicity in Event-based Neural Networks](maximizing_asynchronicity_in_event-based_neural_networks.md)**

:   提出EVA框架，将事件类比为语言token，用基于RWKV-6的线性注意力异步编码器实现逐事件特征更新，结合多表示预测(MRP)+下一表示预测(NRP)的自监督学习获得可泛化特征，首次在异步-同步(A2S)范式中成功完成高难度目标检测任务(Gen1数据集0.477 mAP)。

**[Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)**

:   提出IE-CL（Incremental-Entropy Contrastive Learning）框架，通过显式优化增强视图间的熵增益（而非仅最大化互信息），将编码器视为信息瓶颈并联合优化可学习变换（生成熵）与编码器正则化器（保留熵），在小batch设置下一致提升CIFAR-10/100、STL-10和ImageNet上的对比学习性能，且核心模块可即插即用集成到现有框架。

**[No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](no_other_representation_component_is_needed_diffusion_transformers_can_provide_r.md)**

:   提出 Self-Representation Alignment (SRA)，发现扩散 Transformer 内部表征沿"层数增加 + 噪声降低"两个维度呈现从差到好的判别质量梯度，据此将学生网络早层高噪声表征对齐到 EMA 教师晚层低噪声表征，**完全不需要任何外部表征组件（DINOv2/CLIP/MAE）**，即可在 DiT 和 SiT 上大幅加速收敛并提升生成质量（SiT-XL/2 在 800 epoch 达到 FID 1.58，可比依赖 DINOv2 的 REPA）。

**[PonderLM: Pretraining Language Models to Ponder in Continuous Space](ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)**

:   提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。

**[Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)**

:   提出 Regularized Latent Dynamics Prediction (RLDP)，通过在自监督的潜空间下一状态预测目标上添加简单的正交正则化来维持特征多样性，在零样本 RL 中匹配甚至超越复杂的 SOTA 表示学习方法，特别是在低覆盖率场景下优势显著。

**[SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)**

:   SNAP-UQ 提出一种面向 TinyML 场景的单次前向传播不确定性估计方法：在骨干网络的选定层附加微型 int8 预测头，用自监督方式预测下一层的激活统计量，将实际激活与预测之间的偏差（"surprisal"）聚合为不确定性分数，无需额外前向传播、时序缓冲或集成，仅增加几十 KB 闪存即可在微控制器上实现可靠的分布偏移检测和故障检测。

**[Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)**

:   提出 SER（Soft Equivariance Regularization），通过在 ViT 中间层施加软等变正则化、在最终层保持不变性目标的层解耦设计，在不引入额外模块的情况下，为不变性 SSL 方法（MoCo-v3, DINO, Barlow Twins）带来一致的分类精度和鲁棒性提升。

**[Temporal Slowness in Central Vision Drives Semantic Object Learning](temporal_slowness_in_central_vision_drives_semantic_object_learning.md)**

:   通过模拟人类中央视觉（注视点裁剪）和时间慢性原则（时间对比学习），在 Ego4D 数据上训练 SSL 模型，发现两者组合能有效提升语义对象表征——中央视觉强化前景提取，时间慢性在注视凝视期间蒸馏语义信息。

**[Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)**

:   提出 Chroma——小型预训练时序模型组合（portfolio）框架：从通用模型通过后训练（post-training）产出频率/领域专家（训练加速 10×），测试时通过模型选择或贪心集成组合，4M 参数的 portfolio 在 Chronos Benchmark II 上匹配 205M-500M 参数的大型单体模型性能，同时推理计算远低于 test-time fine-tuning。

**[Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)**

:   诊断出原型自监督学习中部分原型坍缩的根因是编码器与原型的联合优化导致的快捷学习，提出全解耦训练策略——用在线 GMM 独立估计原型——彻底消除坍缩并提升下游性能。

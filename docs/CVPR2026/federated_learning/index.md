---
title: >-
  CVPR2026 联邦学习论文汇总 · 19篇论文解读
description: >-
  19篇CVPR2026的联邦学习方向论文解读，涵盖联邦学习、个性化生成、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "联邦学习"
  - "论文解读"
  - "论文笔记"
  - "个性化生成"
  - "对齐/RLHF"
item_list:
  - u: "domain_sensitive_federated_learning_with_fisher-informed_pruning/"
    t: "Domain Sensitive Federated Learning with Fisher-Informed Pruning"
  - u: "fedadamom_adaptive_momentum_for_improved_generalization_in_federated_optimizatio/"
    t: "FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization"
  - u: "fedalign_differentially_private_distribution_alignment_for_non-iid_federated_lea/"
    t: "FedAlign: Differentially Private Distribution Alignment for Non-IID Federated Learning"
  - u: "fedara_resource-adaptive_low-rank_personalized_federated_learning_via_anchor-dri/"
    t: "FedARA: Resource-adaptive Low-rank Personalized Federated Learning via Anchor-driven Representation Alignment on Heterogeneous Edge Devices"
  - u: "fedharmony_harmonizing_heterogeneous_label_correlations_in_federated_multi-label/"
    t: "FedHarmony: Harmonizing Heterogeneous Label Correlations in Federated Multi-Label Learning"
  - u: "fedrac_rolling_submodel_allocation_for_collaborative_fairness_in_federated_learn/"
    t: "FedRAC: Rolling Submodel Allocation for Collaborative Fairness in Federated Learning"
  - u: "fedrg_unleashing_the_representation_geometry_for_federated_learning_with_noisy_c/"
    t: "FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients"
  - u: "fine-tuning_impairs_the_balancedness_of_foundation_models_in_long-tailed_persona/"
    t: "Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning"
  - u: "from_selection_to_scheduling_federated_geometry-aware_correction_makes_exemplar_/"
    t: "From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity"
  - u: "fully_decentralized_certified_unlearning/"
    t: "Fully Decentralized Certified Unlearning"
  - u: "gdfa_geometry-driven_federated_unlearning_with_directional_task_vector_alignment/"
    t: "GDFA: Geometry-Driven Federated Unlearning with Directional Task Vector Alignment"
  - u: "generalized_and_personalized_federated_learning_with_black-box_foundation_models/"
    t: "Generalized and Personalized Federated Learning with Black-Box Foundation Models via Orthogonal Transformations"
  - u: "hilora_hierarchical_low-rank_adaptation_for_personalized_federated_learning/"
    t: "HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning"
  - u: "os-fed_one_snapshot_is_all_you_need/"
    t: "OS-FED: One Snapshot Is All You Need"
  - u: "personalized_federated_training_of_diffusion_models_with_privacy_guarantees/"
    t: "Personalized Federated Training of Diffusion Models with Privacy Guarantees"
  - u: "single-round_scalable_analytic_federated_learning/"
    t: "Single-Round Scalable Analytic Federated Learning"
  - u: "submodel_extraction_for_efficient_and_personalized_federated_learning_via_optima/"
    t: "SubFLOT：基于最优传输的高效个性化联邦学习子模型抽取"
  - u: "taming_noise-induced_prototype_degradation_for_privacy-preserving_personalized_f/"
    t: "Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning"
  - u: "towards_stable_federated_continual_test-time_adaptation_in_wild_world/"
    t: "Towards Stable Federated Continual Test-Time Adaptation in Wild World"
item_total: 19
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤝 联邦学习

**📷 CVPR2026** · **19** 篇论文解读

🔥 **高频主题：** 联邦学习 ×16 · 个性化生成 ×6 · 对齐/RLHF ×3

**[Domain Sensitive Federated Learning with Fisher-Informed Pruning](domain_sensitive_federated_learning_with_fisher-informed_pruning.md)**

:   FEDFIP 用每个域的 Fisher 信息估计通道重要性，在服务器端拼出一个全局共享剪枝掩码、客户端再「重新激活」少量本地关键通道，配合域原型结构对比正则和「只聚合共享通道」的聚合策略，在多域联邦场景里既显著压小模型、又比一众 FL baseline 更准更稳。

**[FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization](fedadamom_adaptive_momentum_for_improved_generalization_in_federated_optimizatio.md)**

:   本文用扩散理论解释了「FedAdam 收敛快但泛化差」的根因——自适应学习率削弱了对平坦极小的偏好——并据此提出 FedAdamom：把自适应机制从学习率挪到**动量系数**上，从而既保住快速逃离鞍点的能力、又恢复对平坦极小的选择，在 CIFAR-10/100、Tiny-ImageNet 与 LEAF 上同时取得更快收敛和更高精度。

**[FedAlign: Differentially Private Distribution Alignment for Non-IID Federated Learning](fedalign_differentially_private_distribution_alignment_for_non-iid_federated_lea.md)**

:   FedAlign 让每个客户端把本地数据的前四阶统计矩（均值、方差、偏度、峰度）加噪上传，服务器聚合成全局参考分布再广播回去，客户端据此对齐本地采样数据的分布——在差分隐私约束下同时缓解 Non-IID 异质性和隐私泄露，CIFAR-10 上比最强基线再涨约 4%。

**[FedARA: Resource-adaptive Low-rank Personalized Federated Learning via Anchor-driven Representation Alignment on Heterogeneous Edge Devices](fedara_resource-adaptive_low-rank_personalized_federated_learning_via_anchor-dri.md)**

:   FedARA 把"共享特征提取器"做成可被服务器按客户端资源任意分解/重建的低秩结构，让异构边缘设备各取所需的秩；同时用服务器聚合后的全局特征算"一致性锚点"约束本地表示，缓解非 IID 下的特征漂移和全局知识遗忘，在三个数据集上以更低通信/计算开销超过 17 个 SOTA 基线。

**[FedHarmony: Harmonizing Heterogeneous Label Correlations in Federated Multi-Label Learning](fedharmony_harmonizing_heterogeneous_label_correlations_in_federated_multi-label.md)**

:   针对联邦多标签学习中各客户端只见到局部标签空间、学出的标签相关性互相打架（标签相关性漂移）的问题，FedHarmony 用"多数客户端的共识相关性"当全局教师在本地训练时纠偏，并在服务器聚合时同时按数据量和相关性质量给客户端加权，在 FLAIR / COCO-80 / VOC2007 三个非 IID 联邦基准上一致超过现有 SOTA（FLAIR mAP +11.4）。

**[FedRAC: Rolling Submodel Allocation for Collaborative Fairness in Federated Learning](fedrac_rolling_submodel_allocation_for_collaborative_fairness_in_federated_learn.md)**

:   FedRAC 通过"随训练进程动态拉开的声誉计算"+"按历史频率轮转构造子模型再按声誉分配"两个模块，既让贡献高的客户端拿到更好的子模型（公平），又保证全局模型每个神经元被均匀训练（不掉精度），在公平性和准确率上同时超过现有协作公平方法。

**[FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients](fedrg_unleashing_the_representation_geometry_for_federated_learning_with_noisy_c.md)**

:   针对联邦学习里"客户端标注有噪声 + 数据非独立同分布"的双重难题，FedRG 抛弃了不可靠的 small-loss 启发式，改从**表征几何**判断样本干净与否——先用自监督在超球面上学出与标签无关的表征，再用 vMF 混合模型把"几何证据"和"标注标签证据"在同一空间里做一致性比对来挑出噪声样本，最后用个性化噪声吸收矩阵做鲁棒优化，在多个数据集和四种噪声场景下都拿到 SOTA。

**[Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning](fine-tuning_impairs_the_balancedness_of_foundation_models_in_long-tailed_persona.md)**

:   本文先实证揭示「在长尾联邦场景里微调 CLIP 会破坏其天生的类别均衡性、甚至跌破 zero-shot」，再提出 FedPuReL：用 zero-shot 预测把本地梯度「净化」成不破坏均衡的方向来训一个均衡的全局模型，并把个性化重构成冻结全局模型之上的「残差修正」，从而在 8 个长尾数据集上的全局模型和个性化模型都超过现有 SOTA。

**[From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity](from_selection_to_scheduling_federated_geometry-aware_correction_makes_exemplar_.md)**

:   针对联邦持续学习里"光会挑样本、不会用样本"的痛点，FEAT 不改回放策略，而是用一组所有客户端共享的固定 ETF 原型，在训练时做几何结构蒸馏拉齐各客户端的特征角度、在推理时用基于能量的几何校正把尾类特征从头类子空间里"拽回来"，作为即插即用模块叠在 Re-Fed+/FedCBDR 上即可稳定涨点。

**[Fully Decentralized Certified Unlearning](fully_decentralized_certified_unlearning.md)**

:   针对"无中心协调者的去中心化网络"这一被忽视的场景，本文提出 RR-DU——一个随机游走式的认证遗忘算法：只在发起删除的客户端上对遗忘集做带噪投影梯度上升、其余客户端继续做无噪下降，配合子采样高斯噪声和信任域投影，证明了 $(\varepsilon,\delta)$ 网络遗忘证书、收敛性与删除容量边界，且噪声不随遗忘集大小 $m$ 增长，在图像分类上把后门攻击成功率压到随机猜测水平同时保住干净精度。

**[GDFA: Geometry-Driven Federated Unlearning with Directional Task Vector Alignment](gdfa_geometry-driven_federated_unlearning_with_directional_task_vector_alignment.md)**

:   GDFA 把"联邦遗忘"重新理解成一个**损失曲面几何**问题：先用扰动把全局模型迁到平坦极小值区，再让相关客户端在遗忘数据上生成任务向量、只保留**方向一致（符号共识）**的分量做反向聚合，从而在 Non-IID 场景下精确擦除目标客户端知识，同时几乎不损失保留任务精度。

**[Generalized and Personalized Federated Learning with Black-Box Foundation Models via Orthogonal Transformations](generalized_and_personalized_federated_learning_with_black-box_foundation_models.md)**

:   FEDOT 把冻结的黑盒基础模型当成纯特征提取器，每个客户端在它输出的 embedding 上叠一个**本地正交变换**做个性化、所有客户端共享并聚合一个**全局分类器**做泛化；作者证明正交约束（条件数 $\kappa=1$）能把跨客户端的梯度冲突上界压到最小，从而在严重 non-IID 下同时拿到 SOTA 级的泛化和个性化，且全程不碰 FM 内部参数。

**[HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning](hilora_hierarchical_low-rank_adaptation_for_personalized_federated_learning.md)**

:   HiLoRA 把每个客户端的 LoRA 更新拆成"根—簇—叶"三层正交子空间，分别承载全局共识、子群共性与客户端个性，再配上一个基于 LoRA 子空间相似度的自适应聚类，在 CIFAR-100 与 DomainNet 上同时把个性化和对新客户端的泛化都做到 SOTA。

**[OS-FED: One Snapshot Is All You Need](os-fed_one_snapshot_is_all_you_need.md)**

:   OS-FED 把客户端整段本地训练的累积梯度"画"成一张 224×224 的合成图像 + 一组可学习标签（一个 snapshot），每轮只传这一张图，服务器单步前反传就能恢复并聚合全局更新，相比 SOTA 把通信量降 1.5–16×、收敛加快 18–45%。

**[Personalized Federated Training of Diffusion Models with Privacy Guarantees](personalized_federated_training_of_diffusion_models_with_privacy_guarantees.md)**

:   PFDM 把扩散模型的反向去噪过程拆成"客户端私有去噪器 + 服务器共享去噪器"两块，客户端只上传经裁剪并前向加噪后的数据，从而对每个数据点给出形式化的本地差分隐私（LDP）保证；共享模型只见加噪数据、单独无法复现任何客户端样本，而协同又能显著提升少数类/欠表示类的生成质量。

**[Single-Round Scalable Analytic Federated Learning](single-round_scalable_analytic_federated_learning.md)**

:   SAFLe 用「特征分桶 + 打散分组 + 稀疏嵌入求和」搭出一个确定性的非线性分类头，并证明它在数学上等价于一个高维稀疏线性回归，从而能直接套用解析联邦学习（AFL）的单轮闭式聚合律——既拿到非线性的表达力，又保留 AFL「一轮通信 + 对数据异质性完全不变」的两大优势，在三个视觉联邦基准上同时超过线性 AFL 和多轮 DeepAFL。

**[SubFLOT：基于最优传输的高效个性化联邦学习子模型抽取](submodel_extraction_for_efficient_and_personalized_federated_learning_via_optima.md)**

:   SubFLOT 把"个性化剪枝"从客户端搬到服务端：用客户端的历史模型当作本地数据分布的代理，靠最优传输（Wasserstein 距离最小化）在服务端就为每个设备裁出贴合它数据的异构子模型，再配一个随剪枝率自适应的正则项稳住本地训练、一个同款 OT 对齐的聚合模块缓解参数漂移，在 8 个数据集上大幅超过 9 个 SOTA 联邦剪枝方法。

**[Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning](taming_noise-induced_prototype_degradation_for_privacy-preserving_personalized_f.md)**

:   针对原型化个性化联邦学习（ProtoPFL）在共享类原型时为满足局部差分隐私而注入各向同性高斯噪声、却把判别性维度也一并淹没的问题，本文提出客户端即插件 VPDR：用方差自适应的 VPP 把噪声预算从判别子空间挪向冗余子空间、用蒸馏引导的 DCR 让特征范数主动贴近裁剪阈值，在同等 LDP 保证下显著改善隐私-效用权衡。

**[Towards Stable Federated Continual Test-Time Adaptation in Wild World](towards_stable_federated_continual_test-time_adaptation_in_wild_world.md)**

:   本文提出 **BPFedCTTA**，用贝叶斯视角统一处理「联邦持续测试时适应（FedCTTA）」：把全局模型当作高斯先验、用 MAP 估计稳住每个无标注客户端的本地适应（BPA），再用输出熵算出的不确定性门控来选择性地融合客户端更新（UGSA），从而在客户端顺序到来、分布完全无关的极端异构场景下既能适应新域、又不破坏全局模型、缓解灾难性遗忘。

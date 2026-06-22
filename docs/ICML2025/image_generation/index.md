---
title: >-
  ICML2025 图像生成论文汇总 · 92篇论文解读
description: >-
  92篇ICML2025的图像生成方向论文解读，涵盖扩散模型、对齐/RLHF、对抗鲁棒、模型压缩、图像恢复、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "图像生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "模型压缩"
  - "图像恢复"
  - "Agent"
item_list:
  - u: "action-minimization_meets_generative_modeling_efficient_transition_path_sampling/"
    t: "Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional"
  - u: "all-atom_diffusion_transformers_unified_generative_modelling_of_molecules_and_ma/"
    t: "All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials"
  - u: "angle_domain_guidance_latent_diffusion_requires_rotation_rather_than_extrapolati/"
    t: "Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation"
  - u: "annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod/"
    t: "Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions"
  - u: "autoencoder-based_hybrid_replay_for_class-incremental_learning/"
    t: "Autoencoder-Based Hybrid Replay for Class-Incremental Learning"
  - u: "beyond_bradley-terry_models_a_general_preference_model_for_language_model_alignm/"
    t: "Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment"
  - u: "beyond_one-hot_labels_semantic_mixing_for_model_calibration/"
    t: "Beyond One-Hot Labels: Semantic Mixing for Model Calibration"
  - u: "bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter/"
    t: "BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling"
  - u: "broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition/"
    t: "Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition"
  - u: "compositional_scene_understanding_through_inverse_generative_modeling/"
    t: "Compositional Scene Understanding through Inverse Generative Modeling"
  - u: "continualflow_learning_and_unlearning_with_neural_flow_matching/"
    t: "ContinualFlow: Learning and Unlearning with Neural Flow Matching"
  - u: "continuous_semi-implicit_models/"
    t: "Continuous Semi-Implicit Models"
  - u: "continuous_visual_autoregressive_generation_via_score_maximization/"
    t: "Continuous Visual Autoregressive Generation via Score Maximization"
  - u: "dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space/"
    t: "DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space"
  - u: "direct_discriminative_optimization_your_likelihood-based_visual_generative_model/"
    t: "Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator"
  - u: "directed_graph_grammars_for_sequence-based_learning/"
    t: "Directed Graph Grammars for Sequence-based Learning"
  - u: "discriminative_policy_optimization_for_token-level_reward_models/"
    t: "Discriminative Policy Optimization for Token-Level Reward Models"
  - u: "distillation_of_discrete_diffusion_through_dimensional_correlations/"
    t: "Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)"
  - u: "drag_data_reconstruction_attack_using_guided_diffusion/"
    t: "DRAG: Data Reconstruction Attack using Guided Diffusion"
  - u: "editable_noise_map_inversion_encoding_target-image_into_noise_for_high-fidelity_/"
    t: "Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation"
  - u: "efficient_diffusion_models_for_symmetric_manifolds/"
    t: "Efficient Diffusion Models for Symmetric Manifolds"
  - u: "efficient_generative_modeling_with_residual_vector_quantization-based_tokens/"
    t: "Efficient Generative Modeling with Residual Vector Quantization-Based Tokens"
  - u: "elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi/"
    t: "Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers"
  - u: "exploring_position_encoding_in_diffusion_u-net_for_training-free_high-resolution/"
    t: "Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation"
  - u: "expressive_score-based_priors_for_distribution_matching_with_geometry-preserving/"
    t: "Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization"
  - u: "flat-lora_low-rank_adaptation_over_a_flat_loss_landscape/"
    t: "Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape"
  - u: "flexiclip_locality-preserving_free-form_character_animation/"
    t: "FlexiClip: Locality-Preserving Free-Form Character Animation"
  - u: "flextok_resampling_images_into_1d_token_sequences_of_flexible_length/"
    t: "FlexTok: Resampling Images into 1D Token Sequences of Flexible Length"
  - u: "gaussian_mixture_flow_matching_models/"
    t: "Gaussian Mixture Flow Matching Models"
  - u: "gaussmarker_robust_dual-domain_watermark_for_diffusion_models/"
    t: "GaussMarker: Robust Dual-Domain Watermark for Diffusion Models"
item_total: 92
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🧪 ICML2025** · **92** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (490)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (352)](../../ICLR2026/image_generation/index.md) · [💬 ACL2026 (5)](../../ACL2026/image_generation/index.md) · [🧪 ICML2026 (141)](../../ICML2026/image_generation/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (221)](../../NeurIPS2025/image_generation/index.md)

🔥 **高频主题：** 扩散模型 ×38 · 对齐/RLHF ×4 · 对抗鲁棒 ×3 · 模型压缩 ×3 · 图像恢复 ×3

**[Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)**

:   本文提出将预训练生成模型（扩散模型和流匹配）的 score 函数解释为随机动力学中的漂移项，通过最小化 Onsager-Machlup (OM) 作用泛函来**零样本**复用预训练模型进行分子系统的过渡路径采样 (TPS)，在丙氨酸二肽、快速折叠蛋白等系统上以远低于传统方法的计算成本获得了物理真实的过渡路径。

**[All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials](all-atom_diffusion_transformers_unified_generative_modelling_of_molecules_and_ma.md)**

:   提出 All-atom Diffusion Transformer (ADiT)，通过 VAE 将分子和晶体映射到统一潜空间、再用 Diffusion Transformer 在潜空间生成的两阶段框架，首次实现**单一模型**同时生成周期性材料（晶体）和非周期性分子系统，在 MP20、QM9、GEOM-DRUGS 上达到 SOTA，且比等变扩散模型快一个数量级。

**[Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation](angle_domain_guidance_latent_diffusion_requires_rotation_rather_than_extrapolati.md)**

:   发现 Classifier-Free Guidance (CFG) 导致颜色失真的根本原因是潜空间样本范数被放大，提出 Angle Domain Guidance (ADG) 算法——在角度域而非幅度域增强引导，约束范数变化的同时优化角度对齐，在高引导权重下消除颜色饱和度异常并保持甚至改善文本-图像对齐。

**[Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)**

:   提出 Annealing Flow (AF)——基于连续归一化流（CNF）的高维多模态分布采样方法，用动态最优传输（OT）目标配合 Wasserstein 正则化训练，通过退火过程引导模式探索，在高维多模态设置中大幅优于现有 NF 和 MCMC 方法。

**[Autoencoder-Based Hybrid Replay for Class-Incremental Learning](autoencoder-based_hybrid_replay_for_class-incremental_learning.md)**

:   提出基于自编码器的混合重放策略(AHR)，利用混合自编码器(HAE)将样本压缩存储在潜空间中而非原始输入空间，结合带电粒子系统能量最小化(CPSEM)和斥力算法(RFA)增量嵌入新类质心，在最坏情况下将内存复杂度从 $\mathcal{O}(t)$ 降低到 $\mathcal{O}(0.1t)$，同时保持 SOTA 性能。

**[Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment](beyond_bradley-terry_models_a_general_preference_model_for_language_model_alignm.md)**

:   提出偏好嵌入（Preference Embedding）——将响应嵌入到多维潜空间中捕捉复杂偏好结构（包括不可传递偏好），实现 $O(K)$ 的查询复杂度（与 BT 模型相同但表达力更强），配合 General Preference Optimization (GPO) 在 RewardBench 和 AlpacaEval2.0 上超越 BT 奖励模型。

**[Beyond One-Hot Labels: Semantic Mixing for Model Calibration](beyond_one-hot_labels_semantic_mixing_for_model_calibration.md)**

:   提出 CSM（Calibration-aware Semantic Mixing）——利用预训练扩散模型生成高保真的语义混合样本（如猫-狗混合体），并通过 CLIP 重标注精确的软标签置信度，用 $L_2$ 损失训练实现比现有校准方法更优的模型置信度校准。

**[BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling](bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter.md)**

:   提出 Bridge 框架，通过 LLM 多智能体系统生成高质量文本-时序配对数据，并利用语义原型与文本描述的混合提示驱动扩散模型，实现跨域、实例级别的文本控制时序生成（Text-Controlled TSG），在12个数据集中11个取得SOTA。

**[Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)**

:   提出 HEGGS（High-fidelity Earthquake Groundmotion Generation System），利用地震数据集中波形天然可配对的特性，结合条件隐扩散模型与 ACM 振幅校正模块，仅需最少条件信息（经纬度、震源深度、震级）即可端到端生成高保真三分量地震波形。

**[Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)**

:   本文提出逆生成建模（IGM）框架，将场景理解任务转化为在组合式生成模型中寻找最优条件参数的反演问题，通过将多个小型扩散模型组合来表示复杂场景，实现了强分布外泛化能力，并可直接利用预训练文生图模型进行零样本多目标感知。

**[ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)**

:   提出 ContinualFlow，一种基于 Flow Matching 的生成模型定向遗忘框架，通过能量函数重加权软性减去数据分布中不需要的区域，无需重新训练或直接访问待遗忘样本即可实现高效遗忘。

**[Continuous Semi-Implicit Models](continuous_semi-implicit_models.md)**

:   提出 CoSIM——将层级半隐式模型扩展为连续时间框架，通过连续转移核实现无仿真高效训练，并设计保持一致性的转移核实现分布级别的多步扩散模型蒸馏，在 ImageNet 512×512 上达到或超越现有扩散加速方法。

**[Continuous Visual Autoregressive Generation via Score Maximization](continuous_visual_autoregressive_generation_via_score_maximization.md)**

:   提出连续视觉自回归框架——基于严格适当评分规则理论，用能量分数作为无似然训练目标，替代向量量化实现连续token自回归图像生成，EAR-H达到FID 1.97且推理速度比扩散损失方法MAR快约10倍。

**[DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)**

:   提出 DCTdiff，首次在离散余弦变换（DCT）频域空间中进行端到端扩散图像生成，无需 VAE 即可无缝扩展至 512×512 分辨率，并在生成质量和训练效率上均优于像素空间扩散模型。

**[Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator](direct_discriminative_optimization_your_likelihood-based_visual_generative_model.md)**

:   DDO 提出将似然模型本身参数化为 GAN 判别器（通过似然比），无需额外判别器网络即可用 GAN 目标微调预训练的扩散/自回归模型，在 CIFAR-10 和 ImageNet 上大幅刷新 FID 记录（EDM: 1.97→1.38, EDM2-S: 1.58→0.97）。

**[Directed Graph Grammars for Sequence-based Learning](directed_graph_grammars_for_sequence-based_learning.md)**

:   提出 DIGGED，通过无歧义上下文无关图文法将 DAG 无损映射为唯一的产生式规则序列，结合 Transformer 解码器实现图生成/属性预测/贝叶斯优化，在神经架构搜索、贝叶斯网络和电路设计三个任务上全面超越现有方法。

**[Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)**

:   提出 Q-function Reward Model (Q-RM)，通过将奖励建模与语言生成解耦，定义判别式策略来学习 token 级 Q 函数，从偏好数据中无需细粒度标注即可获得精确的 token 级奖励信号，显著提升 PPO/REINFORCE 的推理性能与训练效率。

**[Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](distillation_of_discrete_diffusion_through_dimensional_correlations.md)**

:   提出Di4C方法，通过"mixture"模型捕获维度间相关性，配合一致性损失函数，将多步离散扩散模型蒸馏为少步模型，同时在图像和语言任务上展示了有效性。

**[DRAG: Data Reconstruction Attack using Guided Diffusion](drag_data_reconstruction_attack_using_guided_diffusion.md)**

:   提出 DRAG，利用预训练潜在扩散模型（LDM）的图像先验知识，通过引导扩散过程从分割推理（Split Inference）的深层中间表示中高保真地重建原始输入图像，揭示视觉基础模型（CLIP、DINOv2）在 SI 场景下的严重隐私漏洞。

**[Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation](editable_noise_map_inversion_encoding_target-image_into_noise_for_high-fidelity_.md)**

:   提出 Editable Noise Map Inversion (ENM Inversion)，通过在反演过程中同时优化重建误差和编辑对齐误差，使 noise map 同时"铭刻"源图像与目标图像信息，在内容保持和编辑忠实度之间取得最优平衡。

**[Efficient Diffusion Models for Symmetric Manifolds](efficient_diffusion_models_for_symmetric_manifolds.md)**

:   提出一种高效的对称流形（环面、球面、SO(n)、U(n)）扩散模型框架，通过欧几里得布朗运动的投影和Itô引理绕过热核计算，将训练复杂度从指数级降至近线性，并提供多项式级采样精度保证。

**[Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)**

:   ResGen 通过直接预测累积RVQ嵌入而非单个令牌，解耦了生成迭代次数与序列长度和量化深度的关系，实现了高保真、快速采样的高效生成模型。

**[Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)**

:   本文从denoiser的角度深入分析了Flow Matching (FM) ODE的采样轨迹动力学，揭示了轨迹演化的三个阶段（初始→中间→终端），建立了FM ODE在数据支撑在低维子流形上时的收敛性理论。

**[Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation](exploring_position_encoding_in_diffusion_u-net_for_training-free_high-resolution.md)**

:   通过深入分析扩散模型U-Net中卷积层零填充（zero-padding）产生的位置信息在高分辨率下的传播不足问题，提出Progressive Boundary Complement（PBC）方法，在特征图内部构建渐进式虚拟边界来增强位置信息传播，实现训练无关的高质量高分辨率图像生成。

**[Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization](expressive_score-based_priors_for_distribution_matching_with_geometry-preserving.md)**

:   提出基于 score function 的表达性先验分布（SAUB），通过 Score Function Substitution (SFS) 技巧绕过先验密度估计，结合 Gromov-Wasserstein 几何保持约束实现稳定高效的分布匹配，在公平分类、域适应和域翻译任务上取得优越表现。

**[Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)**

:   提出 Flat-LoRA，通过在**全参数空间**中引入基于贝叶斯期望损失的随机权重扰动，使 LoRA 收敛到全参数空间中更平坦的极小值区域，提升域内和域外泛化性能，且几乎不增加训练时间和显存开销。

**[FlexiClip: Locality-Preserving Free-Form Character Animation](flexiclip_locality-preserving_free-form_character_animation.md)**

:   FlexiClip 提出了一种基于时域Jacobian校正、概率流ODE连续时间建模和GFlowNet流匹配损失的剪贴画动画框架，在保持视觉一致性的同时显著提升了动画的时间平滑性和几何完整性。

**[FlexTok: Resampling Images into 1D Token Sequences of Flexible Length](flextok_resampling_images_into_1d_token_sequences_of_flexible_length.md)**

:   提出 FlexTok——一种将 2D 图像重采样为可变长度、有序的 1D 离散 token 序列的 tokenizer，通过 nested dropout 学习层次化编码，配合 rectified flow 解码器在任意 token 数量下生成高质量重建，在 ImageNet 上用 8~128 个 token 即可实现 FID<2 的自回归图像生成。

**[Gaussian Mixture Flow Matching Models](gaussian_mixture_flow_matching_models.md)**

:   提出高斯混合流匹配模型（GMFlow），用动态高斯混合分布替代传统的单高斯去噪分布来建模多模态流速度场，通过 KL 散度损失训练，并推导出 GM-SDE/ODE 求解器实现精确少步采样，同时引入概率引导方案解决 CFG 过饱和问题，在 ImageNet 256×256 上仅 6 步采样即达到 Precision 0.942。

**[GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)**

:   提出 GaussMarker——首个双域（空间+频率）扩散模型水印方法，通过流水线注入器在初始高斯噪声的空间域和频率域一致嵌入水印，配合模型无关的可学习高斯噪声修复器（GNR）增强对旋转/裁剪攻击的鲁棒性，在三个 Stable Diffusion 版本上八种图像扭曲下达到平均 TPR@1%FPR 0.997 的 SOTA 性能。

**[Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)**

:   本文研究不使用离散 Token 的因果语言模型进行音频生成，利用 token-wise diffusion 建模连续值 next-token 分布，并提出 masked next-token prediction 任务，以 193M 参数在 AudioCaps 上达到与 SOTA 扩散模型相当的性能。

**[GRAM: A Generative Foundation Reward Model for Reward Generalization](gram_a_generative_foundation_reward_model_for_reward_generalization.md)**

:   GRAM 提出用生成式（而非判别式）方法训练奖励模型——先通过大规模无监督学习预训练生成式奖励模型，再用监督数据微调，并证明 label smoothing 实际上等价于正则化的 pairwise ranking 损失，实现了跨任务的奖励泛化。

**[Hessian Geometry of Latent Space in Generative Models](hessian_geometry_of_latent_space_in_generative_models.md)**

:   提出通过重建 Fisher 信息度量来分析生成模型潜空间几何的方法，发现扩散模型潜空间中存在分形结构的相变边界，在相边界处 Lipschitz 常数发散。

**[Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)**

:   提出 Hi-MAR，在掩码自回归图像生成中引入低分辨率 token 作为中间枢纽，建立从粗到细的层次化生成流程，并用 Diffusion Transformer Head 增强 token 间依赖建模，在 ImageNet 上以更少计算量显著超越 MAR（FID 提升 0.38）。

**[Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals](hierarchical_reinforcement_learning_with_uncertainty-guided_diffusional_subgoals.md)**

:   提出将条件扩散模型与高斯过程先验相结合的分层强化学习框架，通过不确定性感知的子目标生成机制，解决高层策略在低层策略动态变化时难以产生有效子目标的核心难题。

**[Importance Sampling for Nonlinear Models](importance_sampling_for_nonlinear_models.md)**

:   通过引入非线性映射的伴随算子（adjoint operator），将线性模型中经典的范数采样和杠杆分数采样系统性地推广到非线性模型，首次为神经网络等非线性模型的重要性采样提供了理论近似保证。

**[Improving the Diffusability of Autoencoders](improving_the_diffusability_of_autoencoders.md)**

:   通过DCT频谱分析发现自编码器潜在空间存在与RGB不匹配的过强高频成分，提出尺度等变正则化（Scale Equivariance）对齐两者频率分布，仅需10-20K步微调即可将ImageNet FID降19%、Kinetics FVD降44%+。

**[InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference](infosem_a_deep_generative_model_with_informative_priors_for_gene_regulatory_netw.md)**

:   提出InfoSEM——无监督生成框架利用文本基因嵌入作为信息先验推断基因调控网络(GRN)：无需GT标签即超越监督方法38.5%，有标签作为额外先验时再提11.1%，同时发现现有监督方法学到的是基因特定偏差而非真正的调控机制。

**[Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models](integrating_intermediate_layer_optimization_and_projected_gradient_descent_for_s.md)**

:   提出 DMILO 和 DMILO-PGD 两种方法，通过中间层优化（ILO）分解扩散模型采样过程以大幅降低显存，并结合投影梯度下降（PGD）避免次优收敛，在线性和非线性逆问题上全面超越 DMPlug 等 SOTA 方法。

**[IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)**

:   提出 IntLoRA，通过整数型低秩参数实现量化扩散模型的微调，合并权重后无需额外 PTQ 即可直接获得量化推理权重，兼顾训练与推理效率。

**[Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features](label-efficient_hyperspectral_image_classification_via_spectral_film_modulation_.md)**

:   提出 GeoDiffNet-F 框架，利用冻结的预训练扩散模型提取低层空间特征，并通过 FiLM（Feature-wise Linear Modulation）机制将高光谱光谱信息自适应融合到空间特征中，在极少标注条件下实现高效的高光谱图像土地覆盖分类。

**[Learning Single Index Models with Diffusion Priors](learning_single_index_models_with_diffusion_priors.md)**

:   提出利用扩散模型先验从半参数单指标模型（SIM）的非线性观测中恢复信号的高效方法，只需一轮无条件采样和部分反演，无需已知链接函数，在1-bit和三次测量上以极少的NFE显著优于现有方法。

**[LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces](livs_a_pluralistic_alignment_dataset_for_inclusive_public_spaces.md)**

:   通过两年社区参与式研究，构建了包含 37,710 对多标准偏好标注的 LIVS 数据集，用于文本到图像模型在包容性城市公共空间设计中的多元对齐，并用 DPO 微调 SDXL 验证其有效性。

**[Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)**

:   提出LoMAP——训练无关的扩散规划修正方法，在每个反向扩散步将引导后样本投影到由离线数据近邻构建的局部低秩子空间上，防止不可行轨迹生成，理论证明引导误差随维度以 $O(\sqrt{d})$ 增长。

**[Localizing and Mitigating Memorization in Image Autoregressive Models](localizing_and_mitigating_memorization_in_image_autoregressive_models.md)**

:   利用改进的UnitMem指标定位图像自回归模型（VAR/RAR）中的记忆化神经元，发现不同架构的记忆化分布模式存在显著差异，并通过缩小高记忆化神经元权重实现了大幅降低可提取训练数据量（VAR-d30从672降至110张）且对生成质量影响可控的隐私缓解方案。

**[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation](lscd_lomb-scargle_conditioned_diffusion_for_time_series_imputation.md)**

:   提出 LSCD，将可微的 Lomb-Scargle 周期图层集成到 score-based 扩散模型中用于时间序列填补，通过频域条件信息和频谱一致性损失，在高缺失率下同时提升时域填补精度和频域恢复一致性。

**[Model Immunization from a Condition Number Perspective](model_immunization_from_a_condition_number_perspective.md)**

:   从Hessian矩阵条件数的角度定义和分析模型免疫问题，提出最大化/最小化条件数的正则化器，使预训练模型难以被微调用于有害任务而不影响正常任务性能。

**[Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](modulated_diffusion_accelerating_generative_modeling_with_modulated_quantization.md)**

:   MoDiff 提出了调制量化(Modulated Quantization)与误差补偿相结合的框架来加速扩散模型，将激活量化从 8-bit 降至 3-bit 且无性能损失，同时继承缓存和量化方法的双重优势。

**[Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)**

:   提出 Morse 双采样框架，通过快速 Dot 模型学习残差反馈来补偿 Dash（原扩散模型）跳步采样的信息损失，实现 1.78×–3.31× 的无损加速。

**[Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)**

:   提出多维自适应系数 MAC（Multidimensional Adaptive Coefficient），作为 flow/diffusion 模型的即插即用模块，将传统的一维时间调度系数扩展为多维、样本自适应的系数，通过对抗训练优化推理轨迹，在 CIFAR-10 条件生成上以 5 NFE 取得 FID 1.37 的 SOTA 结果。

**[Nonparametric Identification of Latent Concepts](nonparametric_identification_of_latent_concepts.md)**

:   提出首个非参数概念可识别性理论框架，证明在不假设概念类型、函数关系或参数生成模型的情况下，仅通过多类别观测的多样性即可识别隐藏概念（至逐元素变换+置换不确定性）。

**[Normalizing Flows are Capable Generative Models](normalizing_flows_are_capable_generative_models.md)**

:   提出 TarFlow（Transformer AutoRegressive Flow），用堆叠因果 ViT 实现分块自回归 Normalizing Flow，首次在 ImageNet 64×64 上突破 3 BPD，并通过高斯噪声增强、score-based 去噪和 guidance 三项技术使 NF 模型的生成质量首次媲美扩散模型。

**[One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework](one_image_is_worth_a_thousand_words_a_usability_preservable_text-image_collabora.md)**

:   提出 Co-Erasing，首次将图像监督引入概念擦除流程，通过文本-图像协同的负引导和文本引导的图像概念精炼模块，在保持良性生成质量（usability）的同时显著提升不良概念的擦除效果（efficacy）。

**[Origin Identification for Text-Guided Image-to-Image Diffusion Models](origin_identification_for_text-guided_image-to-image_diffusion_models.md)**

:   本文提出 ID2 任务（文本引导图像到图像扩散模型的原始图像识别），构建了首个数据集 OriPID，并证明了通过对 VAE 嵌入进行线性变换可以泛化地找到生成图像的原始来源，在 mAP 上超越相似度方法 31.6%。

**[Out-of-Distribution Detection Methods Answer the Wrong Questions](out-of-distribution_detection_methods_answer_the_wrong_questions.md)**

:   本文系统论证了当前主流OOD检测方法（基于特征和基于logit）在根本上回答了错误的问题——它们检测的是"特征是否异常"或"模型是否不确定"，而非"输入是否来自不同分布"，并证明了各种常见改进策略也无法解决这一根本性错位。

**[PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](pak-ucb_contextual_bandit_an_online_learning_approach_to_prompt-aware_selection_.md)**

:   提出 PAK-UCB 上下文老虎机算法，通过为每个生成模型学习独立的核函数，在线预测给定 prompt 下的最优模型，实现 prompt 级别的生成模型/LLM 选择，并用随机傅里叶特征（RFF）降低计算开销。

**[Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)**

:   系统性研究了在不依赖外部模型（VLM/CLIP）的前提下，对文本到图像扩散模型的初始噪声优化算法施加 Best-of-N 推理时缩放的效果，发现性能会迅速达到平台期（plateau），少量优化步数即可逼近该设置下的最大性能，且不同底层扩散模型上的最优算法不同。

**[Position: All Current Generative Fidelity and Diversity Metrics are Flawed](position_all_current_generative_fidelity_and_diversity_metrics_are_flawed.md)**

:   Position paper：系统性地证明了所有现有生成模型 fidelity 和 diversity 指标（包括 Improved Precision/Recall、Density/Coverage、α-precision/β-recall 等六对指标）在精心设计的 sanity check 中均存在大量失败，呼吁社区投入更多精力研发更可靠的评估指标。

**[PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)**

:   将黑盒模型反转攻击形式化为 MDP，用 PPO 强化学习在生成模型的隐空间中导航搜索，仅依赖目标模型的预测概率即可高效重建训练样本，以更少查询和更少类别数据实现了 SOTA 攻击成功率。

**[Preference Adaptive and Sequential Text-to-Image Generation](preference_adaptive_and_sequential_text-to-image_generation.md)**

:   PASTA 将个性化 T2I 生成建模为多轮序列决策问题，通过 VLM 生成候选 prompt + EM 训练用户偏好模型 + IQL 离线 RL 学习价值函数，在人类评估中显著优于基线 LMM。

**[Privacy Amplification Through Synthetic Data: Insights from Linear Regression](privacy_amplification_through_synthetic_data_insights_from_linear_regression.md)**

:   在线性回归框架下，证明了合成数据在对抗者控制种子时无法提供隐私放大，但在随机输入下释放有限数量的合成数据可以获得超越模型本身DP保证的隐私放大效果，放大程度为 $O(1/d)$。

**[Progressive Tempering Sampler with Diffusion](progressive_tempering_sampler_with_diffusion.md)**

:   提出 Progressive Tempering Sampler with Diffusion (PTSD)，通过将 Parallel Tempering 的温度交换机制与扩散模型的神经采样器相结合，利用"温度引导"从高温扩散模型外推生成低温近似样本，在目标密度评估效率上实现数量级提升。

**[Quantum Algorithms for Finite-horizon Markov Decision Processes](quantum_algorithms_for_finite-horizon_markov_decision_processes.md)**

:   提出四种量子值迭代算法（QVI-1/2/3/4），在精确动力学和生成模型两种设定下，对有限时域时变MDP实现了状态空间 $S$、动作空间 $A$、误差 $\epsilon$ 和时域 $H$ 多维度的量子加速，并证明了渐近最优的量子下界。

**[ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering](reframe_layer_caching_for_accelerated_inference_in_real-time_rendering.md)**

:   将扩散模型中的中间层缓存技术（DeepCache）扩展到实时渲染 pipeline 中的 U-Net/U-Net++ 网络，通过帧差自适应缓存策略实现平均 1.4× 推理加速，且画质损失微乎其微。

**[Reimagining Parameter Space Exploration with Diffusion Models](reimagining_parameter_space_exploration_with_diffusion_models.md)**

:   探索用扩散模型学习任务特定参数（LoRA adapter）的分布并直接生成新参数，在野生动物分类场景中验证了其在已知任务上可匹配微调性能，但在跨任务泛化上仍面临挑战。

**[Representative Language Generation](representative_language_generation.md)**

:   提出"代表性生成"（representative generation）理论框架，要求生成模型的输出按比例代表训练数据中的各兴趣群组，并引入"群组闭包维度"（group closure dimension）作为刻画可生成性的关键组合量。

**[RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)**

:   提出 RestoreGrad 框架，通过 Prior Net 和 Posterior Net 联合学习条件 DDPM 的先验分布（而非固定标准高斯），利用退化信号与干净信号之间的相关性构建更具信息量的先验，在语音增强和图像修复任务上实现 5-10× 更快收敛和 2-2.5× 更少推理步数。

**[Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation](review_remask_refine_r3_process-guided_block_diffusion_for_text_generation.md)**

:   提出 R3 (Review, Remask, Refine) 框架，在推理阶段利用过程奖励模型 (PRM) 评估掩码扩散模型的中间生成块，对低质量块进行比例性重掩码并重新生成，实现无需额外训练的定向纠错，在数学推理任务上以极低的 PRM 调用次数取得显著提升。

**[Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation](revisiting_diffusion_models_from_generative_pre-training_to_one-step_generation.md)**

:   提出将扩散模型训练视为"生成式预训练"的新视角，发现蒸馏中师生模型收敛到不同局部最优的根本局限，证明仅用 GAN 目标（无需蒸馏损失）即可将预训练扩散模型高效转换为单步生成器（D2O），且冻结 85% 参数的微调版本（D2O-F）仅需 0.2M 图像即可达到强竞争力结果。

**[SADA: Stability-guided Adaptive Diffusion Acceleration](sada_stability-guided_adaptive_diffusion_acceleration.md)**

:   提出基于ODE轨迹二阶差分的稳定性准则（Stability Criterion），统一调控步级（step-wise）和token级（token-wise）稀疏决策，在SD-2/SDXL/Flux上实现≥1.8×加速且LPIPS≤0.10、FID≤4.5，显著优于DeepCache和AdaptiveDiffusion。

**[Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction](sample_complexity_of_distributionally_robust_off-dynamics_reinforcement_learning.md)**

:   提出 supremal visitation ratio $C_{vr}$ 度量在线鲁棒 MDP 的探索难度，设计首个支持一般 $f$-散度（TV/KL/$\chi^2$）的高效在线算法 ORBIT，并给出匹配的上下界，证明 $C_{vr}$ 是刻画 off-dynamics RL 在线可学习性的紧致度量。

**[Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency](shielded_diffusion_generating_novel_and_diverse_images_using_sparse_repellency.md)**

:   提出 SPELL（Sparse Repellency）方法，在扩散模型生成过程中添加**稀疏排斥项**，将采样轨迹推离参考图像集合（受保护图像或已生成图像），以**免训练**方式提升输出多样性并防止复制训练集。

**[Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation](simple_and_critical_iterative_denoising_a_recasting_of_discrete_diffusion_in_gra.md)**

:   提出 Simple Iterative Denoising (SID) 与 Critical Iterative Denoising (CID) 框架，通过假设中间噪声状态的条件独立性来消除离散扩散的复合去噪误差，并引入 Critic 网络自适应调节元素级重加噪概率，在图/分子生成任务上大幅超越标准离散扩散基线。

**[Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)**

:   提出 SmPO-Diffusion，通过平滑偏好建模替代二元偏好标签 + ReNoise Inversion 替代前向加噪估计，在大幅降低训练成本（比 DPO 快 6.5 倍，比 KTO 快 26 倍）的同时实现了 T2I 扩散模型偏好对齐的 SOTA 性能。

**[Stealix: Model Stealing via Prompt Evolution](stealix_model_stealing_via_prompt_evolution.md)**

:   Stealix 提出首个无需人工设计 prompt 的模型窃取方法，通过遗传算法迭代进化 prompt，利用 Stable Diffusion 生成目标类别图像并查询受害模型，仅需每类 1 张真实图像即可在低查询预算下超越依赖类名或手工 prompt 的已有方法，准确率提升最高达 22.2%。

**[Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)**

:   受物理中软粒子布朗运动的启发，本文提出在潜空间中通过随机力驱动的身份采样方法（Langevin、Dispersion、DisCo 三种算法），生成大规模多样化的合成人脸数据集用于训练人脸识别模型，同时防止训练数据泄漏。

**[Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?](synthetic_perception_can_generated_images_unlock_latent_visual_prior_for_text-ce.md)**

:   系统研究"合成感知"——利用T2I模型为纯文本数据即时生成合成图像作为互补模态，通过三阶段评估框架（生成→融合→评估）证明该策略在讽刺检测和隐式情感分析等困难任务上可为Llama-3/Qwen-2.5等强LLM带来显著提升（+3.9% Acc），但在简单事实分类任务上增益边际。

**[Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](taming_diffusion_for_dataset_distillation_with_high_representativeness.md)**

:   提出 D³HR 框架，通过 DDIM 反演将 VAE 潜在空间的复杂混合高斯分布映射到高正态性的噪声空间，再结合组采样策略生成高代表性的蒸馏数据集，在 CIFAR、Tiny-ImageNet、ImageNet-1K 上全面超越现有 SOTA。

**[Taming Rectified Flow for Inversion and Editing](taming_rectified_flow_for_inversion_and_editing.md)**

:   提出 RF-Solver 和 RF-Edit 两个无训练方法，通过高阶 Taylor 展开精确求解 Rectified Flow ODE 来大幅提升反演精度，并利用自注意力特征共享实现高质量图像/视频编辑，兼容 FLUX、OpenSora 等主流模型。

**[Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner](task-agnostic_pre-training_and_task-guided_fine-tuning_for_versatile_diffusion_p.md)**

:   提出 SODP 框架：先用大量无奖励标签的次优多任务轨迹预训练扩散规划器，再用基于策略梯度的 RL 微调快速适配下游任务，并引入 BC 正则化防止性能崩溃，在 Meta-World 50 任务上达到 60.56% 成功率（SOTA）。

**[Theoretical Guarantees on the Best-of-n Alignment Policy](theoretical_guarantees_on_the_best-of-n_alignment_policy.md)**

:   本文推翻了文献中广泛使用的 best-of-n 策略 KL 散度公式 $\log(n) - (n-1)/n$ 的精确性声明，证明它只是一个上界，并提出了更紧的 KL 散度估计器和 win rate 理论界。

**[ToMA: Token Merge with Attention for Diffusion Models](toma_token_merge_with_attention_for_diffusion_models.md)**

:   提出 ToMA，将 token merge 重新建模为子模优化问题并以 attention-like 线性变换实现 merge/unmerge，使其与 FlashAttention 等 GPU 优化方案兼容，在 SDXL/Flux 上分别实现 24%/23% 的实际端到端加速，同时图像质量损失极小（DINO Δ<0.07）。

**[Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)**

:   通过比较神经网络去噪器与理论最优经验去噪器的近似误差，发现扩散模型的泛化源于跨架构共享的**局部归纳偏置**——神经网络在去噪时倾向于执行局部化操作，并据此提出无需训练的 Patch Set Posterior Composites (PSPC) 去噪器，通过聚合局部经验去噪器来复现网络行为，证实 patch 去噪与组合是扩散模型泛化的重要机制。

**[Tree-Sliced Wasserstein Distance: A Geometric Perspective](tree-sliced_wasserstein_distance_a_geometric_perspective.md)**

:   提出 Tree-Sliced Wasserstein distance on Systems of Lines (TSW-SL)，用树状线系统替代 SW 中的一维直线作为投影域，保留拓扑结构的同时保持闭合解的高效计算，在梯度流、风格迁移和生成模型上超越 SW 及其变体。

**[Tree-Sliced Wasserstein Distance with Nonlinear Projection](tree-sliced_wasserstein_distance_with_nonlinear_projection.md)**

:   提出非线性投影框架下的 Tree-Sliced Wasserstein（TSW）距离，通过 Circular/Spatial 两种非线性 Radon 变换替代原有线性投影，在保持度量良定义和单射性的同时，在梯度流、自监督学习和生成模型等任务上显著优于已有 SW 和 TSW 变体。

**[Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)**

:   通过对数概率密度的 Hessian 曲率（sharpness）建立扩散模型记忆化的几何分析框架，提出可在生成初始阶段检测记忆化的新指标，并设计无需重训练的 SAIL 初始噪声优化策略来缓解记忆化。

**[Unsupervised Learning for Class Distribution Mismatch (UCDM)](unsupervised_learning_for_class_distribution_mismatch.md)**

:   提出 UCDM，利用扩散模型从无标注数据中合成正负样本对来训练分类器，在不依赖标注数据的情况下解决训练集与目标任务之间的类别分布不匹配（CDM）问题，在 closed-set 和 open-set 任务上均大幅超越现有半监督方法。

**[Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations](video_prediction_policy_a_generalist_robot_policy_with_predictive_visual_represe.md)**

:   利用视频扩散模型（VDM）内部的"预测性视觉表征"（同时编码当前帧和未来帧信息）来隐式学习逆动力学模型，从而以高频闭环方式生成机器人动作，在仿真和真实世界操作任务上大幅超越已有方法。

**[Visual Generation Without Guidance](visual_generation_without_guidance.md)**

:   提出 Guidance-Free Training (GFT)，通过重新参数化条件模型为采样网络与无条件网络的线性插值，直接从数据训练出无需引导的视觉生成模型，在 DiT/VAR/LlamaGen/MAR/LDM 五种模型上匹配 CFG 性能的同时将采样计算量减半。

**[When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets](when_diffusion_models_memorize_inductive_biases_in_probability_flow_of_minimum-n.md)**

:   从理论上分析了最小 $\ell^2$ 范数浅层 ReLU 去噪器驱动的扩散模型概率流的收敛行为，证明概率流可以收敛到训练样本（记忆化）、训练样本之和（"虚拟点"）或超盒边界上的流形点（泛化），且扩散时间调度器的"早停"效应决定了收敛目标。

**[DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)**

:   提出DDIS——首个利用T2I扩散模型作为图像先验的无数据图像合成方法，通过Domain Alignment Guidance (DAG)在扩散采样过程中对齐BN层域统计量、Class Alignment Token (CAT)编码类特定属性，在ImageNet-1k和多域PACS上全面超越现有DFIS方法。

**[Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)**

:   提出 ProLoRA，一种免训练的闭式 LoRA 跨模型迁移方法，通过将源 LoRA 在源模型权重子空间和零空间的投影分解重新投射到目标模型的对应空间，实现风格/概念/加速 LoRA 在不同扩散模型间的无损迁移。

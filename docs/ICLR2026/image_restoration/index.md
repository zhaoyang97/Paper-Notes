---
title: >-
  ICLR2026 图像恢复论文汇总 · 22篇论文解读
description: >-
  22篇ICLR2026的图像恢复方向论文解读，涵盖扩散模型、图像恢复、语音、超分辨率等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "语音"
  - "超分辨率"
item_list:
  - u: "a_statistical_benchmark_for_diffusion-posterior-sampling_algorithms/"
    t: "A Statistical Benchmark for Diffusion-Posterior-Sampling Algorithms"
  - u: "activation_steering_for_masked_diffusion_language_models/"
    t: "Activation Steering for Masked Diffusion Language Models"
  - u: "adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling/"
    t: "Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling"
  - u: "analyzing_the_training_dynamics_of_image_restoration_transformers_a_revisit_to_l/"
    t: "Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization"
  - u: "are_deep_speech_denoising_models_robust_to_adversarial_noise/"
    t: "Are Deep Speech Denoising Models Robust to Adversarial Noise?"
  - u: "autoregressive-based_progressive_coding_for_ultra-low_bitrate_image_compression/"
    t: "Autoregressive-based Progressive Coding for Ultra-Low Bitrate Image Compression"
  - u: "beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta/"
    t: "Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes"
  - u: "breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res/"
    t: "Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training"
  - u: "bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r/"
    t: "Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution"
  - u: "cl-dps_a_contrastive_learning_approach_to_blind_nonlinear_inverse_problem_solvin/"
    t: "CL-DPS: A Contrastive Learning Approach to Blind Nonlinear Inverse Problem Solving via Diffusion Posterior Sampling"
  - u: "diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation/"
    t: "DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation"
  - u: "fastdips_adjointfree_analytic_steps_and_hardconstrained_likelihood_correction_fo/"
    t: "FAST-DIPS: Adjoint-Free Analytic Steps and Hard-Constrained Likelihood Correction for Diffusion-Prior Inverse Problems"
  - u: "generalizing_linear_autoencoder_recommenders_with_decoupled_expec/"
    t: "Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss"
  - u: "horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models/"
    t: "Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models"
  - u: "interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions/"
    t: "InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions"
  - u: "learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im/"
    t: "Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration"
  - u: "mechanism_of_task-oriented_information_removal_in_in-context_learning/"
    t: "Mechanism of Task-oriented Information Removal in In-context Learning"
  - u: "protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting/"
    t: "ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting"
  - u: "sharpness-aware_machine_unlearning/"
    t: "Sharpness-Aware Machine Unlearning"
  - u: "skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_/"
    t: "Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs"
  - u: "soflow_solution_flow_models_for_one-step_generative_modeling/"
    t: "SoFlow: Solution Flow Models for One-Step Generative Modeling"
  - u: "trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol/"
    t: "Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🔬 ICLR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 扩散模型 ×10 · 图像恢复 ×3 · 语音 ×2 · 超分辨率 ×2

**[A Statistical Benchmark for Diffusion-Posterior-Sampling Algorithms](a_statistical_benchmark_for_diffusion-posterior-sampling_algorithms.md)**

:   这篇论文为扩散后验采样（DPS）算法造了一把"标准尺"：用可以精确 Gibbs 采样的 Lévy 过程信号作为测试分布，拿到分布级别的"金标准"后验样本，再用 MMSE 最优性差距和后验覆盖率两项指标，把主流 DPS 算法（C-DPS / DiffPIR / DPnP）放在去噪、去卷积、缺失填补、部分傅里叶重建四类反问题上系统评测，结论是这些算法普遍**没有校准**。

**[Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)**

:   首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同，有效方向可从指令前的 token 中提取——反映了扩散模型的非因果并行处理特性。

**[Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling](adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling.md)**

:   把优化器里的 Adam 自适应矩估计直接搬到扩散采样的引导梯度上——对跨采样步的似然分数估计维护一阶/二阶矩的指数滑动平均，几乎零额外成本就把 DPS、CG 这类即插即用引导方法的噪声梯度稳住，在图像恢复（超分/去模糊/补全）和类别条件生成上反超一众更复杂、更慢的方法。

**[Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization](analyzing_the_training_dynamics_of_image_restoration_transformers_a_revisit_to_l.md)**

:   作者追踪图像恢复（IR）Transformer 的训练过程，发现标准 LayerNorm 会让特征幅值发散到百万量级、通道熵急剧坍缩，根因是 LN 的"逐 token 归一化"和"输入无关缩放"与 IR 任务相冲突；据此提出 i-LN——把归一化改成跨整个空间-通道维度做、并在每个 Attention/FFN 后按输入自适应地把缩放因子加回去，作为 LN 的即插即用替换件，在 SR/去噪/去雨/去 JPEG 伪影上稳定训练并普遍涨点。

**[Are Deep Speech Denoising Models Robust to Adversarial Noise?](are_deep_speech_denoising_models_robust_to_adversarial_noise.md)**

:   首次系统性评估 4 款 SOTA 深度语音去噪（DNS）模型在对抗噪声下的鲁棒性：通过心理声学约束的 PGD 攻击生成人耳不可感知的对抗噪声，可令 Demucs、Full-SubNet+、FRCRN 和 MP-SENet 输出完全不可理解的 gibberish，实验覆盖多种声学条件和人类评估，同时揭示了目标攻击、通用扰动和跨模型迁移的局限性。

**[Autoregressive-based Progressive Coding for Ultra-Low Bitrate Image Compression](autoregressive-based_progressive_coding_for_ultra-low_bitrate_image_compression.md)**

:   ARPC 把视觉自回归模型 VAR 的「下一尺度预测」拿来做超低比特率图像压缩：编码端用多尺度残差量化器把图像拆成 K 套从粗到细的离散 token，只传前 k 套、其余由 VAR 自回归生成补齐，从而单模型就能连续调码率；再用 VAR 当概率估计器做无损算术编码、用分组掩码量化器进一步省比特，在 <0.05 bpp 下感知质量超过 13 个扩散/token 基线，解码还快 2∼6×。

**[Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)**

:   LSP 调度器通过在每个去噪步骤中原子性地提交最长连续稳定前缀（而非分散接受离散 token），将 DLM 推理加速 3.4 倍，同时保持或略微提升输出质量。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），并提出架构无关的频率表征学习（FRL），通过 Nyquist 归一化频率编码使误差随分辨率提升而下降，在 8 种主流架构上验证有效。

**[Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution](bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md)**

:   DM-SR 不动预训练扩散模型一根毫毛，只训练一个图像编码器，把低分辨率图直接"翻译"到扩散模型熟悉的"含噪图像"分布上，再用固定去噪器一步生成超分结果，从而在单步扩散下取得当前最佳的感知质量。

**[CL-DPS: A Contrastive Learning Approach to Blind Nonlinear Inverse Problem Solving via Diffusion Posterior Sampling](cl-dps_a_contrastive_learning_approach_to_blind_nonlinear_inverse_problem_solvin.md)**

:   CL-DPS 用一个离线训练的对比学习编码器去近似扩散后验采样里那个棘手的似然项 $p(y\mid x_t)$，从而在**不知道、也不估计测量算子参数**的前提下，第一次让扩散模型能解盲**非线性**逆问题（如旋转模糊、缩放模糊），在这些任务上现有方法全部崩溃而它能干净复原，同时在线性盲去模糊上也保持竞争力。

**[DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)**

:   提出 DiffusionBlocks，将残差网络的逐层更新解释为连续时间扩散过程的离散化步骤，从而将网络切分为可完全独立训练的 block，在保持端到端训练性能的同时按 block 数 B 倍减少训练显存。

**[FAST-DIPS: Adjoint-Free Analytic Steps and Hard-Constrained Likelihood Correction for Diffusion-Prior Inverse Problems](fastdips_adjointfree_analytic_steps_and_hardconstrained_likelihood_correction_fo.md)**

:   FAST-DIPS 用一套"免伴随（adjoint-free）的硬约束似然校正"替换掉训练无关扩散逆问题求解器里昂贵的内层 MCMC / 多步梯度循环：每个噪声层只在去噪器预测点附近做一个带闭式投影 + 解析最优步长的少步 ADMM 校正，把每层计算预算压到极小，在八类线性/非线性恢复任务上质量持平甚至更好，速度最高快 19.5×。

**[Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss](generalizing_linear_autoencoder_recommenders_with_decoupled_expec.md)**

:   将 EDLAE 推荐模型的目标函数推广为解耦期望二次损失（DEQL），在超参数 $b>0$ 的更广范围内推导出闭式解，并通过 Miller 矩阵逆定理将计算复杂度从 $O(n^4)$ 降至 $O(n^3)$，在多个基准数据集上超越 EDLAE 和深度学习模型。

**[Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)**

:   提出 Horizon Imagination (HI)：让扩散世界模型在单次前向里**并行去噪**多帧未来观测，配合**稳定动作采样**抑制带噪帧上动作的无谓翻转、**Horizon 时间表**把去噪节奏与总预算解耦，从而在每帧不足一步去噪（sub-frame 预算）、算力减半下仍保持 on-policy 想象的控制性能。

**[InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions](interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions.md)**

:   提出 InterActHuman，通过自动推断时空布局的掩码预测器和迭代掩码引导策略，实现多人/人物交互场景下的音频驱动视频生成，支持每个角色独立的语音驱动口型同步和身体动作。

**[Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)**

:   提出首个多域全能图像复原方法DATPRL-IR，通过双提示池（任务提示池+域提示池）学习域感知的任务提示表征，利用MLLM蒸馏域先验并通过自适应门控融合指导复原，在自然/医学/遥感三域9任务上显著超越SOTA。

**[Mechanism of Task-oriented Information Removal in In-context Learning](mechanism_of_task-oriented_information_removal_in_in-context_learning.md)**

:   从"信息移除"的新视角解释 In-context Learning（ICL）的内部机制：发现 LM 在零样本时将查询编码为包含所有可能任务信息的"非选择性表征"（导致随机输出），而 few-shot ICL 的核心作用是模拟一种"任务导向的信息移除"过程——通过识别出的"Denoising Heads"（去噪注意力头）从纠缠的表征中选择性移除冗余任务信息，引导模型聚焦目标任务。消融实验证实阻断去噪头后 ICL 准确率显著下降。

**[ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)**

:   提出 ProtoTS，通过层级原型学习实现可解释时间序列预测：少量粗粒度原型提供全局模式概览，逐级细分捕捉局部变化，结合多通道嵌入与瓶颈融合处理异质外生变量。在 LOF 数据集上 MSE 降低 48.3%，MAE 降低 20.9%，且支持专家编辑原型以进一步提升性能。

**[Sharpness-Aware Machine Unlearning](sharpness-aware_machine_unlearning.md)**

:   本文从信号-噪声分解的视角系统分析了 SAM 在机器遗忘场景下的理论特性，发现 SAM 在遗忘集上会"放弃"去噪能力但在保留集上仍维持优势，进而提出 Sharp MinMax 算法——将模型拆成两部分分别做锐度最小化（保留）和锐度最大化（遗忘），达到SOTA遗忘效果。

**[Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)**

:   首次系统比较扩散语言模型（dLLM）和自回归模型（AR LLM）的层间表征结构，发现原生 dLLM 具有更强的层级抽象和早期层冗余性，据此提出静态、任务无关的推理时层跳过策略，在 LLaDA 上跳过 6 层（18.75% FLOPs 削减）仍保持 90%+ 性能。

**[SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)**

:   提出 Solution Flow Models (SoFlow)，直接学习速度 ODE 的解函数 $f(x_t, t, s)$（将 $t$ 时刻的 $x_t$ 映射到 $s$ 时刻的解），通过 Flow Matching 损失 + 无需 JVP 的解一致性损失从头训练，在 ImageNet 256 上 1-NFE FID 优于 MeanFlow（XL/2: 2.96 vs 3.43）。

**[Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution](trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol.md)**

:   提出 Ada-RefSR，一个基于"Trust but Verify"原则的单步参考引导扩散超分辨率框架，通过自适应隐式相关性门控（AICG）机制在利用可靠参考信息的同时抑制错误融合，仅增加 0.13% 计算开销。

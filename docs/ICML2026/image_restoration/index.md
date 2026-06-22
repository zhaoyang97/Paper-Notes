---
title: >-
  ICML2026 图像恢复论文汇总 · 21篇论文解读
description: >-
  21篇ICML2026的图像恢复方向论文解读，涵盖扩散模型、超分辨率、图像恢复等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "超分辨率"
item_list:
  - u: "anymod-llve_low-light_video_enhancement_with_modality-agnostic_inference/"
    t: "AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference"
  - u: "coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_/"
    t: "Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner"
  - u: "coloring_the_noise_adversarial_sobolev_alignment_for_faithful_image_super_resolu/"
    t: "Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution"
  - u: "consistent_diffusion_language_models/"
    t: "Consistent Diffusion Language Models"
  - u: "dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms/"
    t: "DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs"
  - u: "degradation-aware_metric_prompting_for_hyperspectral_image_restoration/"
    t: "Degradation-Aware Metric Prompting for Hyperspectral Image Restoration"
  - u: "dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p/"
    t: "DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention"
  - u: "early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto/"
    t: "Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models"
  - u: "from_2d_grids_to_1d_tokens_reforming_shared_representations_for_multimodal_image/"
    t: "From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion"
  - u: "learning_normalized_energy_models_for_linear_inverse_problems/"
    t: "Learning Normalized Energy Models for Linear Inverse Problems"
  - u: "measurement-consistent_langevin_corrector_for_stabilizing_latent_diffusion_inver/"
    t: "Measurement-Consistent Langevin Corrector for Stabilizing Latent Diffusion Inverse Problem Solvers"
  - u: "one-shot_conditional_sampling_mmd_meets_nearest_neighbors/"
    t: "One-shot Conditional Sampling: MMD meets Nearest Neighbors"
  - u: "one-step_residual_shifting_diffusion_for_image_super-resolution_via_distillation/"
    t: "One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation"
  - u: "phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso/"
    t: "Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging"
  - u: "plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models/"
    t: "Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models"
  - u: "podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_/"
    t: "PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution"
  - u: "semi-supervised_neural_super-resolution_for_mesh-based_simulations/"
    t: "Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations"
  - u: "solving_inverse_problems_with_flow-based_models_via_model_predictive_control/"
    t: "Solving Inverse Problems with Flow-based Models via Model Predictive Control"
  - u: "structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges/"
    t: "Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges"
  - u: "triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz/"
    t: "Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules"
  - u: "uotip_unbalanced_optimal_transport_map_for_unpaired_inverse_problems/"
    t: "UOTIP：无须配对的反演问题的非平衡最优传输映射"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🧪 ICML2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 扩散模型 ×11 · 超分辨率 ×5 · 图像恢复 ×2

**[AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference](anymod-llve_low-light_video_enhancement_with_modality-agnostic_inference.md)**

:   针对"多模态低光视频增强在推理时拿不到事件流/红外辅助模态就崩"的痛点，AMNet 用一个 Spatial-Spectral Dual-Gated（S2DG）Translator 从退化的低光 RGB 里"凭空生成"辅助模态的隐式表示，再配合大规模合成多模态预训练，使得测试时无论给不给辅助模态都能稳定增强——RGB-only 推理就已达到 SOTA，给了辅助模态还能再涨一点。

**[Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)**

:   本文从表达力与可训练性两个维度系统比较连续扩散、离散掩码扩散、looped transformer，证明"连续扩散"在表达力上严格强于离散扩散并能模拟 looped transformer，但实际性能受限于解码与表征空间；据此提出 **CCDD（Coevolutionary Continuous Discrete Diffusion）**——在离散 token 空间和预训练 LLM 的上下文嵌入空间上同时扩散，由单一模型联合去噪，在 LM1B/OWT 上比 MDLM 困惑度降 25-35%，并以仅 8 步采样超过 MDLM 256 步效果。

**[Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution](coloring_the_noise_adversarial_sobolev_alignment_for_faithful_image_super_resolu.md)**

:   ASASR 通过将 Flow Matching 的噪声先验从各向同性高斯替换为 Sobolev 谱着色噪声，结合对抗性流形引导生成硬负样本，构建 AS-DPO 框架，实现了超分辨率中感知质量与结构保真度的最优平衡。

**[Consistent Diffusion Language Models](consistent_diffusion_language_models.md)**

:   本文指出离散扩散没有连续域 probability-flow ODE 的对应物，因此无法直接做 consistency model；作者提出用**精确闭式 posterior bridge** 作为离散域的"随机版 PF-ODE 替代品"，构造 Multi-Path Discrete Consistency (MPDC) 训练目标，要求 denoiser 在多条 stochastic bridge 路径上的预测在期望上一致，从而单阶段、teacher-free 地训出可在 2-3 步生成高质量文本的 Consistent Diffusion Language Model (CDLM)，在 unconditional / conditional 文本生成上达到 SOTA、对 AR 模型最高 $32\times$ 加速。

**[DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)**

:   DAPD 把 dLLM 单步并行解掩问题转化为「在自注意力诱导的 MRF 上选独立集」的动态图着色问题，无需训练即可同时解掩弱依赖位置，在 LLaDA / Dream 上把多问题混合提示的解码步数压到原始的 1/3.87，且准确率几乎不掉。

**[Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)**

:   DAMP 用 6 个可解释的空间-光谱物理度量（高频能量比/纹理一致性/光谱曲率等）作为"退化提示" (DP) 代替黑盒嵌入与显式退化标签，再用 DP 作为门控驱动 Spatial-Spectral Adaptive MoE 选择不同的"空间专家/光谱专家"，在 5 种 HSI 恢复任务和 2 种未见退化（运动模糊、Poisson 噪声）上同时取得 SOTA。

**[DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)**

:   DyLLM 是一种 training-free 的扩散 LLM 推理加速框架，利用相邻去噪步骤之间注意力上下文的余弦相似度识别"显著 token"，只对这部分 token 重算 FFN 和注意力，配合显著感知的近似注意力，在 LLaDA / Dream 上把吞吐推到 7.6× / 9.6× 而几乎不掉点。

**[Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)**

:   本文系统刻画了 masked 扩散语言模型 (dLLM) 在**完全非自回归 (NAR) 解码**下的失败机制——proximity bias 导致 confidence-based 采样退化为反向自回归并被 EOS 过早占满，再用一个 5M 参数的轻量 planner + EOS 温度退火**只在第一步**干预 unmasking 位置，就在 GSM8K 等推理任务上将 LLaDA 8B 的 NAR 解码平均提升 2.8–4.3 个点而几乎无额外开销。

**[From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion](from_2d_grids_to_1d_tokens_reforming_shared_representations_for_multimodal_image.md)**

:   多模态图像融合长期把共享表示放在二维特征网格上，导致全局外观（亮度/对比度/色调）和局部细节纠缠、难以独立调控；本文把"全局外观"挪到一个冻结的 1D tokenizer（TiTok-32）的紧凑 token 空间里，再用"选择性 token 编辑（STE）"只改少数几个 token-通道项来调控全局一致性，同时保留 2D 通路做细节恢复，在四个基准上取得多指标全面最优。

**[Learning Normalized Energy Models for Linear Inverse Problems](learning_normalized_energy_models_for_linear_inverse_problems.md)**

:   作者把"线性逆问题"重写为"各向异性去噪"，并提出 Anisotropic Covariance Score Matching (A-CSM) 训出一个**归一化**的能量模型 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$，单个模型即可处理 inpainting、deblurring、super-resolution，并解锁能量引导自适应调度、MALA 无偏校正和盲逆问题三大新能力。

**[Measurement-Consistent Langevin Corrector for Stabilizing Latent Diffusion Inverse Problem Solvers](measurement-consistent_langevin_corrector_for_stabilizing_latent_diffusion_inver.md)**

:   本文把潜空间扩散逆问题求解器（LDM solver）的不稳定性重新解释为"求解器动力学偏离扩散模型学到的时间边缘分布"，并提出即插即用的 MCLC 模块——在测量一致性步之后插入一步限制在测量梯度正交补空间上的 Langevin 校正，把潜变量拉回稳定的反向扩散轨迹，同时不破坏测量保真度，在 FFHQ/ImageNet 多种线性与非线性退化任务上稳定地提升了 LDPS、PSLD、ReSample 等基线。

**[One-shot Conditional Sampling: MMD meets Nearest Neighbors](one-shot_conditional_sampling_mmd_meets_nearest_neighbors.md)**

:   CGMMD 用 k 近邻图把"期望条件 MMD（ECMMD）"估计成一个可直接最小化的非对抗目标，训出一个能在单次前向传播内从 $P_{Y\mid X}$ 采样的条件生成器，并给出了非渐近误差界与分布收敛性证明。

**[One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation](one-step_residual_shifting_diffusion_for_image_super-resolution_via_distillation.md)**

:   针对扩散超分模型推理慢的问题，本文提出 RSD（Residual Shifting Distillation）：把 15 步的 ResShift 教师蒸馏成**单步**学生生成器，方法核心是「训练学生使得用它的输出再训出来的一个『假 ResShift』恰好等于真教师」——这等价于匹配教师与学生在所有时间步上的**联合分布**（而非 VSD 那样只匹配边缘）；结果在 LPIPS / CLIPIQA / MUSIQ 上反超教师、超过同类蒸馏方法 SinSR，并以仅 174M 参数、0.5GB 显存、5 GPU·小时训练逼近大体量 T2I 超分模型的感知质量。

**[Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)**

:   为单次曝光式压缩光谱成像 (CASSI) 设计一个 train-render 两阶段、按波长可任意查询的深度展开框架——在每个展开 stage 内塞入连续光谱场 (CoSF) 先验模块，由 Fourier-Mamba 驱动的三分支跨域特征混合器 + 随机频率编码 + 谱合成头组成，离散波长训练即可在推理时合成任意连续波长的高光谱图像，实现连续光谱重建与零样本光谱超分。

**[Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)**

:   本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。

**[PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)**

:   PODiff 把扩散模型从像素空间搬到固定的、按方差排序的 POD 系数空间里跑，用极小的 MLP 就能在 $640\times 480$ SST 降尺度任务上拿到与像素级扩散相当的精度，同时因为重构是线性的，集成方差可以通过 $\Sigma_u=\Phi\Sigma_a\Phi^\top$ 解析回传到物理空间，得到空间上可解释、且校准良好的不确定性。

**[Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)**

:   SuperMeshNet 用两个互补 MPNN——主模型预测 LR→HR，辅助模型预测 LR-LR 对应的 HR-HR 差分——在无配对 HR 的样本上互相生成伪标签，并配合节点级 / 消息级 centering 两个轻量归纳偏置，使得 PDE mesh 超分仅用 10% HR 数据就能超过 100% HR 全监督基线，跨 6 种 MPNN 架构一致下降 RMSE。

**[Solving Inverse Problems with Flow-based Models via Model Predictive Control](solving_inverse_problems_with_flow-based_models_via_model_predictive_control.md)**

:   把"用预训练流模型解逆问题"重新表述为**模型预测控制（MPC）**——不再一次性优化整条采样轨迹，而是在每个时间步只解一个短视野子问题、施加一步控制再重规划，从而在显著降低显存的同时，还能推导出一个**完全不需要对流模型反向传播**的变体，进而把训练无关引导扩展到量化后的 FLUX.2（32B）级别、在 24GB 消费级显卡上跑通。

**[Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)**

:   SDB 把模态翻译重写为"在所有满足边缘约束的耦合集合 $\mathcal{P}$ 中挑一个"，在 LDDBM 之上叠加边缘匹配（WTA + 容量约束）+ 端点级 + 轨迹级双层 cycle consistency，把成对监督仅作为可选启发式之一，从而在零成对、半成对、全成对三种监督预算下都能跑，并且全成对时也比 paired-only 基线更好（FFHQ→CelebA-HQ PSNR 从 25.6 提到 25.9）。

**[Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)**

:   本文把扩散后验采样中长期被当作常数的三个力——数据一致性 (DC) 引导、Classifier-Free Guidance (CFG)、随机性 (stochasticity)——首次系统地视为一个**耦合的时变三体系统**，理论 + 实证证明早期 CFG 与 DC 方向冲突、而随机性能把轨迹拉回高概率流形，据此提出"DC↓、CFG↑、η↓"的单调三体调度趋势，并用"模板搜索 + GRPO 强化学习"两套方法找最优曲线，在 FFHQ / DIV2K 的超分与去模糊上同时刷新失真和感知指标。

**[UOTIP：无须配对的反演问题的非平衡最优传输映射](uotip_unbalanced_optimal_transport_map_for_unpaired_inverse_problems.md)**

:   提出 UOTIP 方法——通过非平衡最优传输（UOT）框架将无须配对的图像反演问题表述为从有噪声测量分布到干净信号分布的映射学习，通过引入似然成本函数和二次项成本获得鲁棒性和理论保证。

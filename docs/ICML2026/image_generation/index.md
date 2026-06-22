---
title: >-
  ICML2026 图像生成论文汇总 · 141篇论文解读
description: >-
  141篇ICML2026的图像生成方向论文解读，涵盖扩散模型、文生图、对齐/RLHF、布局/合成、对抗鲁棒、图像编辑等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "图像生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "文生图"
  - "对齐/RLHF"
  - "布局/合成"
  - "对抗鲁棒"
  - "图像编辑"
item_list:
  - u: "a_diffusive_classification_loss_for_learning_energy-based_generative_models/"
    t: "A Diffusive Classification Loss for Learning Energy-based Generative Models"
  - u: "a_kinetic_energy_perspective_of_flow_matching/"
    t: "A Kinetic Energy Perspective of Flow Matching"
  - u: "a_systematic_investigation_of_rl-jailbreaking_in_llms/"
    t: "A Systematic Investigation of RL-Jailbreaking in LLMs"
  - u: "a_unified_framework_for_diffusion_model_unlearning_with_f-divergence/"
    t: "A Unified Framework for Diffusion Model Unlearning with f-Divergence"
  - u: "adaeraser_training-free_object_removal_via_adaptive_attention_suppression/"
    t: "AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression"
  - u: "adapting_noise_to_data_generative_flows_from_1d_processes/"
    t: "Adapting Noise to Data: Generative Flows from Learned 1D Processes"
  - u: "adversarial_flow_models/"
    t: "Adversarial Flow Models"
  - u: "aesformer_transform_everyday_photos_into_beautiful_memories/"
    t: "AesFormer: Transform Everyday Photos into Beautiful Memories"
  - u: "ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi/"
    t: "AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching"
  - u: "alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models/"
    t: "Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models"
  - u: "anomaly-preference_image_generation/"
    t: "Anomaly-Preference Image Generation (APO)"
  - u: "ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters/"
    t: "AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters"
  - u: "balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec/"
    t: "Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective"
  - u: "barriers_to_counterfactual_credit_attribution_for_autoregressive_models/"
    t: "Barriers to Counterfactual Credit Attribution for Autoregressive Models"
  - u: "bayesian_tensor_decomposition_with_diffusion_model_prior/"
    t: "Bayesian Tensor Decomposition with Diffusion Model Prior"
  - u: "beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion/"
    t: "Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion"
  - u: "bootstrap_your_generator_unpaired_visual_editing_with_flow_matching/"
    t: "Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching"
  - u: "breaking_the_lock-in_diversifying_text-to-image_generation_via_representation_mo/"
    t: "Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation"
  - u: "budget-constrained_step-level_diffusion_caching/"
    t: "Budget-Constrained Step-Level Diffusion Caching"
  - u: "caracal_causal_architecture_via_spectral_mixing/"
    t: "Caracal: Causal Architecture via Spectral Mixing"
  - u: "clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi/"
    t: "CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal"
  - u: "coarse-grained_boltzmann_generators/"
    t: "Coarse-Grained Boltzmann Generators"
  - u: "cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l/"
    t: "CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning"
  - u: "cofrgenet_continued_fraction_architectures_for_language_generation/"
    t: "CoFrGeNet: Continued Fraction Architectures for Language Generation"
  - u: "compositional_generative_modeling_from_decentralized_data/"
    t: "Compositional Generative Modeling from Decentralized Data"
  - u: "compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati/"
    t: "Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models"
  - u: "conf-gen_conformal_uncertainty_quantification_for_generative_models/"
    t: "Conf-Gen: Conformal Uncertainty Quantification for Generative Models"
  - u: "conflict-aware_additive_guidance_for_flow_models_under_compositional_rewards/"
    t: "Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards"
  - u: "conformal_reliability_a_new_evaluation_metric_for_conditional_generation/"
    t: "Conformal Reliability: A New Evaluation Metric for Conditional Generation"
  - u: "content-style_identification_via_differential_independence/"
    t: "Content-Style Identification via Differential Independence"
item_total: 141
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🧪 ICML2026** · **141** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (490)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (352)](../../ICLR2026/image_generation/index.md) · [💬 ACL2026 (5)](../../ACL2026/image_generation/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (221)](../../NeurIPS2025/image_generation/index.md) · [📹 ICCV2025 (213)](../../ICCV2025/image_generation/index.md)

🔥 **高频主题：** 扩散模型 ×47 · 文生图 ×10 · 对齐/RLHF ×8 · 布局/合成 ×7 · 对抗鲁棒 ×4

**[A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)**

:   这篇论文提出 DiffCLF，把时间噪声层级之间的能量估计改写成分类问题，并与 DSM 联合训练，从而在不引入昂贵最大似然采样的情况下学习更可靠的能量函数，尤其改善了分数匹配在多模态权重上的模式盲区。

**[A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)**

:   这篇论文把 flow matching 采样轨迹看成粒子运动，定义 Kinetic Path Energy（KPE）来度量每个样本生成过程的累积动能，并据此提出训练-free 的 Kinetic Trajectory Shaping，在提升生成质量的同时抑制末端能量尖峰导致的记忆化。

**[A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)**

:   这篇论文把 RL-based LLM jailbreaking 当作一个可拆解的 POMDP 系统来研究，发现奖励函数、episode 长度和训练问题数量等环境定义因素，比单纯换 RL 算法更大程度决定自动化红队成功率。

**[A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)**

:   这篇论文把扩散模型概念遗忘中的 MSE/KL 对齐推广到任意 $f$-divergence，提出 f-DMU 框架，并发现 closed-form Hellinger loss 往往比 MSE 更稳、更能保留非目标概念。

**[AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)**

:   AdaEraser 用“目标残留程度”自适应调节扩散模型 self-attention 抑制强度，在不训练新模型的情况下同时提升目标删除完整性和背景重建质量，并在 Mulan 与 OABench 上超过训练式和 training-free object removal 方法。

**[Adapting Noise to Data: Generative Flows from Learned 1D Processes](adapting_noise_to_data_generative_flows_from_1d_processes.md)**

:   本文认为 flow/diffusion 模型默认高斯 latent 并不总适合数据分布，提出用可学习的一维 quantile functions 构造数据自适应 product prior，在 flow matching 中联合学习噪声和速度场，从而缩短 transport path 并改善重尾天气数据和低容量图像生成表现。

**[Adversarial Flow Models](adversarial_flow_models.md)**

:   作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。

**[AesFormer: Transform Everyday Photos into Beautiful Memories](aesformer_transform_everyday_photos_into_beautiful_memories.md)**

:   AesFormer 将日常照片美化定义为 Aesthetic Photo Reconstruction，通过先生成摄影动作计划再执行结构编辑的两阶段框架，把构图、视角和姿态等拍摄时错误转化为可执行编辑，并在 AesRecon 上显著优于开源编辑器、接近 Nano Banana Pro。

**[AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)**

:   AG-REPA 发现音频 Flow Matching 中“存储语义信息的层”和“真正驱动速度场的层”并不重合，提出用 forward-only gate ablation 选择因果贡献最高的层做表示对齐，在语音和通用音频生成上比固定层 REPA 更快收敛、更低 FAD。

**[Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)**

:   这篇论文提出 Alignment-Guided Score Matching，用 reward-free 的 Plackett-Luce 对齐奖励把正负文本-图像匹配信号直接写入扩散 score matching 目标，通过训练轻量 soft tokens 改善 T2I 语义对齐，同时缓解 SoftREPA 常见的重复生成和计数错误。

**[Anomaly-Preference Image Generation (APO)](anomaly-preference_image_generation.md)**

:   作者把"少样本异常图像生成"重写为"无人工标注的偏好优化问题"：真实异常作为正样本，参考模型在同一时刻的去噪偏差作为隐式负样本，通过 DPO 风格 loss 让扩散模型对齐异常分布；再用按时间步调节 LoRA rank 的 TACA 保住结构多样性、用分层 CFG 调节文本-异常对齐强度，在 MVTec 等 benchmark 上同时刷新真实度和多样性。

**[AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)**

:   AtelierEval 首次把文本到图像流程中的“提示词编写者”作为评测对象，用 360 个专家任务、三类认知任务和 AtelierJudge agentic evaluator 系统量化人类与 MLLM 的提示词能力，并发现图像模仿式 prompting 往往比纯文本规划式 prompting 更可靠。

**[Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)**

:   将扩散模型中 $\mathbf{QK}^\top$ 注意力矩阵分解为对称分量（能量景观）和反对称分量（环流动力学），据此推导 Hopfield 风格的稳定性度量来诊断亚稳态混合，并通过调控反对称分量实现无需训练的保真度-多样性可控权衡。

**[Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)**

:   本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA，自回归 rollout 也并非 CCA——CCA 不像 DP 那样在自回归下天然 compose；(2) 对一个已部署的非归因模型做 black-box "CCA retrofitting" 至少需要在输出长度 $\ell$ 上指数级查询次数。

**[Bayesian Tensor Decomposition with Diffusion Model Prior](bayesian_tensor_decomposition_with_diffusion_model_prior.md)**

:   DiffBCP 将预训练扩散模型作为隐式数据先验注入贝叶斯 CP 张量分解，通过 split Gibbs 采样器实现可处理的后验推断，在图像修复和去噪任务上全面超越传统和深度张量分解基线（FFHQ 上 PSNR 最高提升 +2.33 dB）。

**[Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion.md)**

:   提出 JEPA Guidance，利用 JEPA（如 DINOv2）编码器的隐式密度信号引导扩散模型采样，将少数样本（minority sample）的定义从"生成模型先验下的低密度"转变为"世界先验下的低密度"，在无条件、类条件和文生图场景均实现更具语义意义的稀有样本生成。

**[Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching](bootstrap_your_generator_unpaired_visual_editing_with_flow_matching.md)**

:   提出 Bootstrap Your Generator (ByG)，一个无需配对数据的 flow matching 编辑训练框架，通过从冻结基础模型提取编辑方向先验 + cycle consistency 保持源结构 + 梯度路由弥合训练-推理差距，在图像和视频编辑上超越百万级配对数据训练的监督基线。

**[Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation](breaking_the_lock-in_diversifying_text-to-image_generation_via_representation_mo.md)**

:   作者发现 Transformer 文生图模型在去噪早期会让"零频空间均值（DC 分量）"在不同随机种子间迅速对齐，把全局布局过早锁死，于是提出 DAVE——在早期生成阶段对中间表征里的 DC 分量做一次轻量衰减，几乎零开销地解锁了同一 prompt 下的样本多样性，同时保持画质与文图对齐。

**[Budget-Constrained Step-Level Diffusion Caching](budget-constrained_step-level_diffusion_caching.md)**

:   BudCache 把扩散模型的步级缓存从"用阈值被动触发、延迟随输入飘"翻转成"先锁死算力预算 B，再离线搜出最优缓存策略"，用模拟退火+爬山在几分钟内搜出一个静态缓存 mask，并在预算极紧时用师生蒸馏重对齐时间表，在 FLUX.1-dev 和 Wan2.1 上同等延迟下质量全面超过 TeaCache/MagCache 等启发式缓存。

**[Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)**

:   Caracal 用 $\mathcal{O}(L \log L)$ 的多头傅立叶（MHF）模块替换 Transformer 的 $\mathcal{O}(L^2)$ 注意力，通过"pad-FFT-multiply-iFFT-truncate"实现频域内的严格因果掩码，并完全去掉位置编码，仅用标准 FFT 算子（不依赖 Mamba 那样的 CUDA kernel）就在 Tiny→Large 全尺度上与 Llama / Mamba / Mamba-2 / Jamba 性能相当。

**[CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)**

:   本文针对视频字幕擦除提出 CLEAR：两阶段训练（Stage I 用 dual encoder + 正交解耦学自监督字幕先验掩码；Stage II 在 Wan2.1 视频扩散模型上加 LoRA + occlusion head 做自适应加权），推理完全不需要任何 mask 或文本检测器，仅训练 0.77% 参数就在中文测试集上把 PSNR 推到 26.80 dB（比最强基线 +6.77 dB），并零样本泛化到 6 种语言。

**[Coarse-Grained Boltzmann Generators](coarse-grained_boltzmann_generators.md)**

:   提出 Coarse-Grained Boltzmann Generators (CG-BGs)，在粗粒化坐标空间中结合归一化流生成模型和学到的平均力势 (PMF) 进行重要性采样，以远低于原子级 BG 的计算成本实现渐近正确的分子平衡态采样。

**[CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)**

:   本文针对"编辑模型常在不该改的区域乱改"这一痛点，构造 CoCoEdit-40K 局部编辑数据集 + 提出 pixel-level 相似度 reward 补充 MLLM reward + 设计区域正则化 RL 目标（高奖励样本约束非编辑区一致、低奖励样本强迫编辑区做出改变），把 FLUX.1 Kontext 和 Qwen-Image-Edit 同时在编辑得分和 PSNR/SSIM 上提升，打破现有"提编辑能力必伤一致性"的 trade-off。

**[CoFrGeNet: Continued Fraction Architectures for Language Generation](cofrgenet_continued_fraction_architectures_for_language_generation.md)**

:   本文把"连续分数（continued fraction）"这种具备最优有理逼近性质的函数类引入到语言生成 Transformer 中，分别为多头注意力和 FFN 设计 CoFrNet 替代模块（CAttnU/CAttnM/Cffn），通过"continuants"封闭形式把 $d$ 次除法降为 1 次，在 GPT2-xl 和 Llama-3.2B 上用 $\frac{2}{3}\sim\frac{1}{2}$ 的参数实现持平甚至更优的下游性能。

**[Compositional Generative Modeling from Decentralized Data](compositional_generative_modeling_from_decentralized_data.md)**

:   当生成因子被切碎在互不共享原始数据的多个客户端里时，本文用 **DCFM（去中心化组合流匹配）** 在全局强制属性的条件独立约束，让模型生成出任何单个客户端都没见过的属性组合，在条件图像生成、机器人空间规划、胸片疾病共现三类任务上都显著超过联邦学习与专家混合基线。

**[Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)**

:   将视觉信号编码为冻结扩散基础模型上的低秩适配参数（LoRA），并通过哈希映射压缩为单个紧凑向量，在极低码率下实现强感知质量的视频压缩，同时支持推理时缩放和生成式编辑。

**[Conf-Gen: Conformal Uncertainty Quantification for Generative Models](conf-gen_conformal_uncertainty_quantification_for_generative_models.md)**

:   提出 Conf-Gen 框架，将保形风险控制（CRC）扩展到生成任务，通过参数化选择函数和可容许性函数为 LLM 问答、图像生成、对话系统、AI Agent 等任务提供形式化的不确定性保证，同时放松了 CRC 的单调性等理论假设。

**[Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards](conflict-aware_additive_guidance_for_flow_models_under_compositional_rewards.md)**

:   针对流模型在多目标组合奖励下推理时引导易产生 off-manifold drift 的问题，提出 Conflict-Aware Additive Guidance (CAR)，通过检测梯度冲突并动态切换可学习的值梯度修正，以极低额外计算代价将身份保留提升 25.4%、规划成功率提升 38.75%。

**[Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)**

:   提出基于保形预测（Conformal Prediction）的可靠性评分 CReL，通过在隐空间构建凸预测集并优化最坏情况下的指标表现，实现对条件生成模型的不确定性感知评估，在图文互生任务上揭示了传统单输出指标无法捕捉的模型可靠性差异。

**[Content-Style Identification via Differential Independence](content-style_identification_via_differential_independence.md)**

:   本文提出 CSDI（content-style differential independence）这一全新的可辨识性条件——只要 generator 关于 content 与 style 的 Jacobian 列空间在数据流形上相互正交，即可在内容-风格统计**相关**且 Jacobian **稠密**的设定下证明 unpaired 多域内容-风格分块可辨识，并通过 Hutchinson 噪声探测把这一条件做成一个可在 StyleGAN2-ADA 上 scalable 的正则项 $\mathcal{L}_{\rm orth}$，在 AFHQ / CelebA-HQ 的反事实生成与跨域翻译上把 FID 从 5.2 / 4.6 进一步压到 4.4 / 4.3，LPIPS 从 0.40 / 0.26 拉到 0.45 / 0.34。

**[DFlash: Block Diffusion for Flash Speculative Decoding](dflash_block_diffusion_for_flash_speculative_decoding.md)**

:   DFlash 用一个轻量级的"块扩散"草稿模型替代 EAGLE-3 那种自回归草稿器，并通过把目标模型的多层 hidden features 作为 KV 注入到草稿模型每一层，在单次前向中并行起草整块 token，端到端最高拿到 6× 无损加速，比 EAGLE-3 再快 2.5× 左右。

**[DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)**

:   论文针对"CLIP 微调到 AI 生成图像检测时灾难性遗忘破坏可迁移先验"的问题，提出 DGS-Net：把分类损失的梯度按坐标拆成有害正分量 $g^+$ 与有益负分量 $g^-$，让训练网络的图像梯度先正交投影到冻结 CLIP **文本梯度有害方向**的补空间（Orthogonal Suppression，剔除任务无关语义），再额外对齐到冻结 CLIP **图像梯度有益方向**（Prior Alignment，保住预训练先验），从而在 50 个生成模型上的平均检测精度比 SOTA 高 6.6%。

**[Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)**

:   论文用线性探针发现 MM-DiT (FLUX / SD3.5) 在中间层的某些注意力头里、其 text token 的 key 向量天然编码了"目标概念是否会出现"的二元信号，并由此提出 Omission Signal Intervention (OSI)：在 inference 时把"omission 类 - existence 类"的均值差方向以 $\alpha\sigma\boldsymbol{\theta}$ 的强度注入 Top-K 头的 key 向量，激发模型对缺失概念的"自我感知"并补全生成；在 FLUX 上 GenEval 6-object 准确率从 0.18 → 0.40，且无需任何 fine-tune。

**[Diffusion Differentiable Resampling](diffusion_differentiable_resampling.md)**

:   本文提出 **diffusion resampling**：用一个**无需训练**的扩散过程为顺序蒙特卡洛 (SMC) 的重采样步骤提供天然可微的 reparametrisation 替代，证明其在 Wasserstein 距离下相对样本数 $N$ 一致收敛，并在多个粒子滤波与参数估计基准上超越 OT / Gumbel-Softmax / Soft 等现有可微重采样方法。

**[Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)**

:   本文证明：当数据分布支撑在 $M$ 个低维线性子空间的并集（UoS）上且每个子空间内的分布是 subgaussian 时，存在一个基于核密度的 score 估计器可以让 score-based 扩散采样器以 $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ 个样本达到 1-Wasserstein 误差 $\varepsilon$（$k$ 为最大内在维度），首次在多峰、无 smoothness/有界密度/log-concavity 假设下达到了与内在维度匹配的 minimax 最优率，彻底打破了维度灾难。

**[Direct 3D-Aware Object Insertion via Decomposed Visual Proxies](direct_3d-aware_object_insertion_via_decomposed_visual_proxies.md)**

:   DIRECT 把"物体插入"从 2D inpainting 升级成**位姿可控**任务：先用现成的 image-to-3D 模型把参考物体抬升成可交互的 3D 代理、按用户指定的 6-DoF 位姿渲染出稠密几何条件图，再把"几何 / 外观 / 上下文"三类条件**解耦成独立通路**注入扩散模型，从而在严格遵循指定 3D 位姿的同时保住参考外观、与背景和谐融合，几何可控性与视觉质量都超过此前方法。

**[DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing](directedit_step-level_accurate_inversion_for_flow-based_image_editing.md)**

:   DirectEdit 通过在 Rectified Flow 反演过程中记录每一步的潜变量残差 $\Delta\mathbf{Z}_t$ 并在前向路径中提前注入，让重建路径与反演轨迹严格逐步对齐，从而在不增加任何 NFE 的前提下实现"步级精确重建"，并结合 MLLM+SAM 多分支 mask 噪声融合与注意力 Value 注入，在 PIE-Bench 上以综合排名 4.0 (FLUX) / 2.43 (SD3.5) 显著优于 RF-Inversion、FireFlow、FTEdit、DNAEdit 等所有现有训练无关方法。

**[DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)**

:   本文提出 DiScoFormer，一种对样本顺序置换等变、对坐标仿射等变的 Transformer，用一次前向把任意 i.i.d. 样本集映射到对应密度 $f$ 与 score $\nabla\log f$，并从理论上证明 self-attention 在适当参数化下可精确复现归一化高斯 KDE，实验上在 GMM、Laplace、Student-$t$ 等多种分布、宽样本量与维度范围内全面优于经典 KDE，并可作为即插即用 score oracle 用于 Fisher 信息、熵估计与 Fokker–Planck 类 PDE 求解。

**[Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)**

:   本文把连续空间扩散采样里成熟的 off-policy RL 训练技巧（replay buffer、重要性加权、MCMC 探索）首次系统迁移到离散扩散采样器，并进一步推广到 data-to-energy 离散 Schrödinger 桥，在 Ising/Potts、离散化 GMM 等多模分布上显著缓解 mode collapse，最后用它在 VQ-VAE 的离散潜空间里做 data-free 的条件图像生成（后验采样）。

**[Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models](divide-and-denoise_a_game-theoretic_method_for_fairly_composing_diffusion_models.md)**

:   把"多个预训练扩散模型协同采样"建模成一个**公平分工博弈**：每一步先用博弈给每个模型分配它该负责的图像区域（allocation），再让复合去噪只在各自分到的区域听各自模型的，从而在不训练、不共享权重的前提下让"单狗模型 + 单猫模型"等组合生成出一张既有狗又有猫、互不抢镜的图，GenEval %images 从 MultiDiffusion 的 58% 提到 88.5%。

**[Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)**

:   本文提出 DiCoME 框架，先用几何正交投影把 CLIP 语义特征和伪造伪迹特征强制解耦成两路互补"专家视图"，再用 Dempster–Shafer 证据融合显式建模两视图间的"认识论冲突"以输出可信的不确定性，在跨数据集和跨伪造方法的 deepfake 检测基准上将平均 AUC 从 0.923 提升到 0.939（cross-dataset）和 0.976（cross-manipulation）。

**[Efficient Learning of Deep State Space Models via Importance Smoothing](efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)**

:   本文提出 Parallel Variational Monte Carlo (PVMC)，用 prefix/suffix associative scan 把深度状态空间模型的重要性加权边际平滑分布在 $\mathcal{O}(\log N \times \log T)$ span 内并行算出来，同时支持监督式状态估计和生成式建模，比最快的可微 SMC 基线快约 10×，精度还更高。

**[E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)**

:   针对 GRPO/DiffusionNFT 等基于组的 RL 在 flow 模型对齐中组内方差崩塌、信号消失的问题，E²PO 在文本嵌入空间注入一组可学习的结构化扰动以维持组内判别方差，并配合噪声感知调度与参考锚定批策略，在 SD3.5-M 上把 GenEval 从 0.917 抬到 0.932 且显著提升多样性。

**[End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)**

:   EOSTok 用单阶段端到端管线把 1D ViT tokenizer 和自回归模型一起训练，靠新提出的 APR（Autoregressive Prediction Reconstruction）loss 把「next-token 预测」的梯度真正传回 pixel space 防止码本崩塌，再用「隐式对齐」把 DINOv2 语义注入 1D 隐空间而不破坏 1D 自回归结构，最终在 ImageNet 256 上无 guidance 拿到 1.48 的 FID（SOTA）。

**[Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)**

:   本文从频域视角分析了扩散模型成员推断攻击（MIA）的失败模式，指出高频内容会同时放大 member 和 hold-out 样本得分的标准差从而稀释成员优势，提出一个无需训练、零额外推理代价的"高频滤波器"模块，只需在计算重建误差前对预测图与目标图做相同的 FFT 低通处理，就能把 Naive/SecMI/PIA 等主流 MIA 在 DDIM、Stable Diffusion 上的 ASR/AUC/TPR@1%FPR 普遍拉高 4–11 个百分点（个别场景 TPR@1%FPR 直接从 6% 跃到 41%）。

**[Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)**

:   针对 5-shot 非典型域（航拍 / 水下 / 极暗）下 layout-to-image 生成出现的"表示碎片化"，作者把每个类别的条件表示显式拆成全局语义锚 + 局部可重组原语，并用显著性感知的损失强制前景一致，在 DIOR 上将 Bootstrap FID 从 82.5 压到 74.3、mAP 提到 26.1。

**[Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)**

:   本文从动力系统视角把 LLM 长文本生成中的「模式崩溃」（重复、循环、单调）重新解释为隐藏状态轨迹在表示空间里的「几何坍缩」，并提出 RMR — 在 Transformer value cache 上做轻量低秩阻尼来抑制最具持续性的自我强化方向，从而在极低熵的解码区间（0.8 nats/step）依然保持稳定高质量生成。

**[Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)**

:   Eso-LMs 把 AR 与 Masked Diffusion 在 loss、注意力、采样三个层面深度融合：用一个 causal-on-shuffled-sequence 的去噪 Transformer 同时支持并行扩散和左到右 AR，从而**首次让 MDM 在扩散阶段也能用上精确 KV cache**，在 OWT 长上下文上比 MDLM 快 14–65×、比 BD3-LM 快 3–4×，并在 speed–quality Pareto 前沿上取得 SOTA。

**[Evaluating the Representation Space of Diffusion Models via Self-Supervised Principles](evaluating_the_representation_space_of_diffusion_models_via_self-supervised_prin.md)**

:   本文用自监督学习（SSL）的「不变性 + 扩张性」两条原理来审视扩散模型的内部表示，提出一个无标签标量指标 **ICR（Invariant Contamination Ratio，不变污染比）**——它能在不采样、不训分类器的情况下，预测哪个噪声层级的特征最适合下游分类、并在训练中提前预警过拟合/记忆化的到来。

**[EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)**

:   EvoGM 把"找 task-vector 合并系数 $\bm{\lambda}$"从手工设计变异算子的进化搜索改写成可学习的生成任务：用一对带 cycle-consistency 的 MLP 生成器从历史的 winner/loser 配对里学高性能区域的分布，并在外层套多轮"基底切换"逐步刷新专家池，在 GLUE 8 任务和 Qwen2.5-1.5B 10 模型 unseen 任务上分别比 PSO-Merging 的 SOTA 平均高约 1.4% 和明显领先。

**[Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)**

:   本文系统刻画了 Latent Flow Matching（LFM）的"轨迹稳定性"——同一噪声种子下，剪掉 75% 数据、换大小架构、改训练种子都能产生几乎相同的图像；进而把这个性质转化成两个实用算法：(1) 用 balanced-clustering 剪枝可在 CelebA-HQ 上把 50% 数据剪掉而 FID 反而轻微提升、ImageNet 上 75% 数据可剪；(2) Coarse-to-Fine 两段式生成，把 DiT-XL/2 (675M) 和 DiT-S/2 (33M) 拼起来，推理快 2.15×。

**[$f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)**

:   把 GFlowNet/Kimi 里"对 log-prob 差取平方"的 $\mathbb{KL}_{sq}$ 代理损失推广到整族 $f$-散度，得到一族同时具备"on-policy 梯度等于对应 $f$-散度真梯度、off-policy 仍有同一全局最优"的可调控 mode-seeking↔mode-covering 损失，并在合成网格、SynFlowNet 分子生成、扩散模型条件采样和异步 LLM RL（GSM8k / MATH）上验证。

**[Finding DoRI: Discovery of Retained Images in Diffusion Models](finding_dori_discovery_of_retained_images_in_diffusion_models.md)**

:   作者用一个简单的对抗 text embedding 优化方法（DoRI）证明：NeMo / Wanda 这类"剪枝定位记忆神经元"的扩散模型记忆缓解方案只是把记忆"藏起来"而非真正擦除，因为记忆在 embedding、激活、权重三个层面都不是局部的；进一步提出对抗微调方案，把训练样本真正从模型里拔出来。

**[ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](forceforget_reinforcement_concept_removal_for_enhancing_safety_in_text-to-image_.md)**

:   把"擦除不安全概念"重新表述成强化学习的奖励优化问题——用一个安全奖励 + 一个对齐奖励组成的概念擦除奖励（CER）微调扩散模型，再配一个只改写少量文本 token 的"安全适配器"，在彻底去掉色情内容的同时，最大限度保住有害提示里夹带的良性语义（尤其是"人"相关内容）。

**[Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)**

:   本文提出训练无关的多概念遗忘框架 FIA，通过"对比概念显著性 + 时空稀疏筛选"定位每个目标概念所对应的概念敏感神经元，并在融合多概念掩码时显式保留同时响应多个概念的"概念无关神经元"，仅剪掉真正概念专属的连接，在 SD v1.5/v1.4 上以 <0.3% 的总稀疏率同时遗忘十个 Imagenette 类（平均遗忘准确率 1.9%，整体得分 86%）以及多艺术家风格和不良内容。

**[From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)**

:   针对"唱歌头像"这一被现有 deepfake 检测器忽视的高难度域，作者一边构建 SHDF 数据集量化"说话→唱歌"的域漂移，一边提出 T-AVFD 框架，用 Alpha-CLIP+多粒度真假文本对比学习"真实人脸的语义模式"，再用差分权重模块自适应融合唇音一致性与人脸语义，仅在真实说话视频上训练就能跨域泛化到唱歌伪造，SHDF AUC 从 50% 量级抬到 80.2%。

**[GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)**

:   作者把 T2I 同 prompt 下的样本多样性投到 CLIP 单位超球面上，沿"文本方向 $\mathbf{e}_t$"与"正交主残差方向 $\mathbf{u}_{\text{ind}}$"分别拉开投影展度，并通过对预测干净图 $\hat{x}_{0|t}$ 做梯度优化把这种几何展开搬回扩散/流采样轨迹，在 SD2.1 与 SD3-M 上同时提升 prompt 相关（姿态、构图）与 prompt 无关（背景、风格）的多样性，几乎不损质量与对齐。

**[DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)**

:   DynaDiff 把"为新环境训练一个预测器"的元学习问题改写成"用扩散模型直接生成完整网络权重"的条件采样问题，借助权重图 + 函数一致性损失 + 动力学感知 prompter，在 4 个 PDE 系统上平均 RMSE 比强基线再降 10.78%。

**[Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)**

:   作者把"移动 GUI 世界模型"重新表述成"VLM 生成可渲染的网页代码"这一新范式，配套提出一套自动把策略轨迹改写成（图像状态、动作）→（推理链、下一状态代码）训练样本的数据合成管线，得到的 gWorld-8B/32B 在 6 个 in/out-of-distribution 基准上同时拿下最佳，并把基线模型平均指令准确率拉高 27–46 个百分点、把渲染失败率压到 <1%。

**[GenExam: A Multidisciplinary Text-to-Image Exam](genexam_a_multidisciplinary_text-to-image_exam.md)**

:   GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点，结果连最强闭源模型 Nano Banana Pro 也只有 70.2% strict 分，多数开源 T2I/统一 MLLM 不到 3%。

**[Geometry-Aware Dataset Condensation for Diffusion Model Training](geometry-aware_dataset_condensation_for_diffusion_model_training.md)**

:   针对"现有数据集压缩方法不适合训扩散模型"的痛点，本文把真实子集选择重新表述为**几何感知的分布对齐问题**，用单边部分最优传输（POT）+ 统计正则化定义对齐目标，再用两阶段离散优化（贪心 + 交换）求解，在 ImageNet 上只用 0.8% 数据训 DiT/SiT，FID 就显著低于此前最强的 D2C（10K 预算下 4.20→3.43）。

**[Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)**

:   作者提出 GATD（Geometry-Aware Tabular Diffusion），在表格扩散去噪器输入和损失里显式加入"列对之间的角度和长度"几何特征作为辅助监督信号，用一个参数量仅 TabDiff 1/3.5（分类任务甚至 1/25）的小 MLP 就在 10 个数据集上拿下 8/10 Shape、7/10 Trend、9/10 下游效用胜场，并且同一套默认超参可直接迁移到 GNN 和 Transformer 去噪器（27/30 Shape、25/30 Trend 涨点）。

**[Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)**

:   本文提出 GMF：用 Diffusion Schrödinger Bridge + Rectified Flow 在潜空间估计每个模态的"传输修正成本"（初始速度平方 $\|v_\theta(z,0)\|^2$），作为一个**与分类器置信度解耦**的几何可靠性信号来动态加权多模态融合，从而打破"模型自己评判自己"的循环依赖，在传感器噪声和语义冲突下显著优于基于置信度的可信融合基线。

**[Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)**

:   通过把 reward 梯度投影到一个用 DFT 块状 $\ell_1/\ell_2$ 范数刻画的"白高斯噪声可行集"上，作者把一步生成模型的 test-time latent 优化变得既快又稳：在 FLUX 上只用 30% 的 wall-clock 时间就追平 SOTA 正则化方法 MPGR 的 Aesthetic Score，并彻底避免 reward hacking。

**[GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)**

:   GUDA 把"群组级训练数据归因"重新表述成"如果训练时没有这个群组，模型对该样本的对数似然会掉多少"的反事实问题，用机器遗忘从全量模型上"擦掉某个群组"近似 Leave-One-Group-Out (LOGO) 重训得到的反事实模型，再用 ELBO 差作为归因分数，在 CIFAR-10 和 Stable Diffusion 艺术风格归因上比 CLIP 相似度和实例级梯度归因更准，且比 LOGO 重训快约 100 倍。

**[GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](guidedbridge_training-freely_improving_bridge_models_with_prior_guidance.md)**

:   针对 diffusion bridge 模型（data-to-data 生成），论文提出训练免费的 Prior Guidance (PG)：通过对干净 prior 加扰动构造"弱 prior"，再把强/弱 prior 的两个 denoising 结果做外推来放大模型对 prior 的利用，并进一步用 U 型频率调制（FMPG）和 CFG-FMPG 级联框架，在 Edges→Handbags、DIODE、ImageNet inpainting 等任务上不增训练、不增 NFE 地稳定提升 DDBM / DBIM 等预训练 bridge 模型的 FID。

**[(HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)**

:   HB-ARFM 用"历史观测引导"的条件 flow matching 解决多相沸腾流场的逆问题重建：先用一段历史观测窗口 bootstrap 出初始隐状态，再用同一个条件速度场自回归地把重建向前推进，在仅观测界面几何与界面速度的情况下，首次完成完整温度场与速度场的时空一致重建。

**[HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing](holofair_unified_t2i_fairness_evaluation_and_fair-grpo_debiasing.md)**

:   本文构建了一个面向 T2I 模型的统一公平性基准 HoloFair（含 SpaFreq 双流属性分类器 + MGBI 多属性几何均值指标），并在此基础上提出 Fair-GRPO：通过对数比例的多属性 per-prompt 奖励 + KL 正则化 GRPO，在 SD3.5-Medium 上把 MGBI 从 0.5211 提升到 0.6772（+29.9%），同时图像质量保持甚至略有提升。

**[Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs](hölder_improving_the_quality-coherence_trade-off_in_multimodal_vaes.md)**

:   针对多模态 VAE 长期存在的「生成质量 vs 跨模态一致性」难以兼得的问题，本文提出 Hölder++：首次给出对称 Hölder 池化（$\alpha=0.5$）的精确实现作为模态聚合器，再叠加共享/私有子空间分离与自顶向下层次推断两项架构改进，在四个基准上把质量-一致性的 Pareto 前沿整体推到 SOTA。

**[Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)**

:   SubDAPS / SubDAPS++ 把 DPS、DAPS 这类 pixel-space 扩散复原方法搬进"动态分辨率扩散模型"框架——早期在 $64^2 / 128^2$ 子空间采样、后期才回到 $256^2$ 全分辨率，并用共轭梯度替掉 Langevin、用阈值切换 stochastic / deterministic 采样、再附一个无需额外网络评估的 corrector 步，在 4 类线性 + 2 类非线性复原任务上多数指标超越 pixel 与 latent 扩散方法且推理更快。

**[Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)**

:   本文把图扩散反向 SDE 的采样轨迹看成 Riemannian 统计流形上的参数曲线，用 Fisher-Rao 度量推出一个无需训练的 Drift Variation Score (DVS) 来度量轨迹的局部"信息曲率"，并据此自适应缩放步长，使每步在信息流形上前进等长，从而在分子（QM9/ZINC250k）和图（Planar/SBM/Ego）生成中以更少步数取得更高 FCD / MMD 保真度。

**[Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)**

:   本文把"初始噪声"看作可以从一个由 conditional guidance 势能定义的后验中采样的随机变量，提出 DivIn：用一步 Langevin 动力学把标准高斯噪声往"低势能、平坦"的区域里推一推，在几乎不增加推理开销的前提下显著缓解扩散/flow matching 模型的 mode collapse，并和已有的 trajectory-based 多样性方法正交叠加。

**[Krause Synchronization Transformers](krause_synchronization_transformers.md)**

:   作者把 Krause 有界置信共识模型搬进 Transformer，用"距离-RBF+局部窗+top-k 稀疏"替代全局 softmax 相似度，从理论上证明它鼓励多簇同步而非全局塌缩，并在 ViT / 自回归图像生成 / LLM 上同时获得更优性能和 30%+ 算力节省。

**[Latent Diffusion Pretraining for Crystal Property Prediction](latent_diffusion_pretraining_for_crystal_property_prediction.md)**

:   CrysLDNet 把"扩散预训练"从原始晶体特征空间搬到 VAE 学到的平滑潜空间，让 PDDFormer 编码器在 38 万无标注 GNoME 晶体上学到更紧凑、更对称感知的结构语义，下游 JARVIS / MP 性质预测平均比强监督 SOTA 再降 4.26% / 4.90% MAE，且在低数据和实验数据校正场景下优势更大。

**[Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)**

:   本文提出 CaDRe，用一个带结构约束的时序 VAE 把"观测变量之间的因果图"与"驱动观测的隐动力过程"放在同一个非参数框架下联合识别，并给出了从时序数据同时恢复二者的可识别性定理，在合成数据上验证理论、在 CESM2 气候数据上得到与领域专家一致的因果图与有竞争力的温度预测精度。

**[Let EEG Models Learn EEG](let_eeg_models_learn_eeg.md)**

:   JET 把多通道 EEG 生成重新定义为"在神经流形上的连续轨迹"，用条件流匹配 + 标准 Transformer 直接对原始波形建模，并配三条专门刻画 EEG 频谱/平稳性/统计的结构化约束，在 TUH 三大临床基准上把 TS-FID 较强基线降低 40% 以上。

**[Linearizing Vision Transformer with Test-Time Training](linearizing_vision_transformer_with_test-time_training.md)**

:   作者发现两层 TTT 内模型在结构上等价于 Softmax 注意力（Softmax 可看作两层动态 MLP），由此实现 Q/K/V/MLP 的全权重直接继承，再通过 key Instance Normalization 处理 shift-invariance、depthwise conv on Q/K 补齐 locality，仅 1 小时微调就把 Stable Diffusion 3.5 线性化并加速 1.32×–1.47×。

**[LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)**

:   LithoGRPO 把光刻掩模生成建模为以目标版图为条件的 rectified flow，并用 GRPO 强化学习微调，让一次前向就能同时优化 L2/PVB（可微）与 EPE/Shot（不可微）四类光刻指标，配合一个 130×–490× 加速的快速 shot-count 算法，在 LithoBench 上把综合排名从 5.6 拉到 4.3，单样本推理仅 0.1 s。

**[Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)**

:   本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\mathcal{O}(D)$，从而在 3072 维图像空间稳定估计局部内禀维度，并用于诊断扩散模型的训练样本记忆化。

**[Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)**

:   本文把"扩散模型局部记忆"刻画为对数密度在某些坐标上的**方差崩塌（高曲率）**，并用"条件模型 − 欠拟合基线（无条件模型或早期 checkpoint）"的**坐标级曲率差**，把单纯由数据流形固有低方差引起的"伪记忆"扣掉，只保留**过拟合驱动的记忆区域**，在 Stable Diffusion 的 ground-truth 记忆掩码上把定位 IoU 从 BE 的 0.75 提到约 0.92。

**[MIRO: 多奖励条件预训练同时提升 T2I 质量与效率](miro_multi-reward_conditioned_pretraining_improves_t2i_quality_and_efficiency.md)**

:   MIRO 把"对齐"从 RLHF 后训练阶段直接塞回预训练：给每张训练图打 7 个奖励分（美学、用户偏好、文图对齐、视觉推理、科学正确性等），让 Flow Matching 模型学习 $p(x|c, s)$，推理时通过多奖励 CFG 指向高奖励区域，0.36B 参数即在 GenEval 上超过 12B 的 FLUX-dev，训练算力少 370×，单样本推理质量超过 baseline 跑 128 次。

**[Mitigating the Contractivity Trap in Diffusion ODEs via Stein Stabilization](mitigating_the_contractivity_trap_in_diffusion_odes_via_stein_stabilization.md)**

:   针对扩散模型概率流 ODE 大步采样时"高表达力去噪器 + 激进步长会破坏收缩性稳定性证书、导致误差被放大轨迹发散"的问题（作者命名为 contractivity trap），SteinDiff 用 Stein 恒等式把"对齐干净目标"这个不可计算的项转成可计算的散度项，推出一个闭式、无需参考样本、无需重训的逐步修正系数 $\gamma_k$，对求解器候选更新做几何感知的残差校正，在 CIFAR-10 / ImageNet-64 / LSUN-Bedrooms 上大步采样的 FID 显著下降（最高减 45.8%）。

**[OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)**

:   针对布局到图像生成在重叠区域出现纹理纠缠和层级混乱的问题，作者构建了带显式 Z-order 与 amodal 标注的大规模数据集 SA-Z，并提出 OcclusionFormer：通过实例解耦 + 体渲染显式建模遮挡优先级，再用查询对齐损失强化空间一致性，在 OverLayBench 复杂子集与自建 SA-Z Eval 上的遮挡感知指标全面超过 Eligen、Creatilayout、InstanceAssemble 等强基线。

**[Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)**

:   OMSD 用"链式条件分解 + 每个 agent 一个条件扩散模型"取代传统离线 MARL 里"各 agent 独立边缘回归"的行为约束,把每个 agent 的策略沿着前缀 agent 已经选定的动作做条件正则,从而避免多模态联合行为分布下"边缘对齐但联合错位"的 OOD 失配,在 MPE / MaMuJoCo 多个数据集上把平均回报刷到现有 SOTA 的 +33% ~ +74%。

**[Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)**

:   本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度，相比 Diffusion-DPO 在 SD3-M/FLUX 上同时提点又把训练算力降到 1/12。

**[OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)**

:   OmniAID 用一个"语义专家 + 通用伪影专家"的解耦 MoE 架构，在 CLIP-ViT 注意力权重 SVD 出的低秩残差子空间里分别学习"画了什么会露馅"和"怎么画都会露馅"两类伪造线索，再配上新的现代化数据集 Mirage，在 GenImage / Chameleon / Mirage-Test 三套基准上把通用 AIGI 检测平均准确率推到 95.9% / 91.4% / 88.4%。

**[OMP: One-step Meanflow Policy with Directional Alignment](omp_one-step_meanflow_policy_with_directional_alignment.md)**

:   本文针对将 MeanFlow 范式直接搬到机器人操作时暴露出的三个理论病灶（频谱偏差、低速区梯度饥饿、嵌套 JVP 内存爆炸），提出 OMP：用一项 cosine-style 方向对齐损失把预测平均速度与真实平均速度方向"锁死"，再用有限差分 DDE 近似 Jacobian-Vector Product 解耦前后向，让单步（NFE=1）生成策略在 Adroit/Meta-World 上以 6.8ms 级延迟做到比 MP1 平均高 3.4%、在 Meta-World Very Hard 任务高 10.6% 的成功率。

**[Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)**

:   本文从扩散模型的频谱偏置出发，理论证明扩散生成区域的局部 Gibbs 能量必然低于真实成像区域，据此构造 LAD（Local Adjacency Discrepancy）能量图作为内禀取证指纹，再用一个轻量适配器把 LAD 线索注入 SAM 完成像素级伪造定位，并配套 EditStream 多智能体自动从 HuggingFace 拉取最新编辑模型不断刷新训练数据，在 7 个 AI 编辑数据集上把平均 IoU 从前 SOTA 的 ~0.25 拉到 0.46。

**[Orthogonal Concept Erasure for Diffusion Models](orthogonal_concept_erasure_for_diffusion_models.md)**

:   把 T2I 扩散模型里"加性参数编辑"的概念擦除（UCE/SPEED 等）改写成"层级正交旋转 $W^* = QW$"的乘性更新，并配上一个子空间级别的擦除目标，用 Procrustes 闭式解一次性算出 $Q$，4.3 秒擦掉 100 个名人概念，且对非目标概念几乎零损伤。

**[Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)**

:   PG-OT 把「多奖励文生图对齐」从「加权全局求和」改成「为每个 prompt 单独构造 Pareto 前沿、用 Sinkhorn 最优传输把被支配样本传到前沿」，并引入 Joint Domination Rate / Joint Collapse Rate 两个新指标暴露平均值掩盖的奖励 hacking，在 Parti-Prompts 上 JDR₂ 47.98% 比强基线提升 11%，人评胜率近 80%。

**[Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)**

:   把分布式 Bellman 方程的"仿射搬运"几何性显式编织进 flow matching 的路径里：用同一份基础噪声同时驱动当前态与后继态的两条路径，再用 $\lambda$ 控制变量在偏差与方差之间换挡，从而得到一个对源分布相容、对 Bellman 端点相容、又稳定的分布式 critic。

**[PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)**

:   把"造可交互 3D 物体"重新理解成"先做物理规划、再做物理生成"的两阶段问题——VLM 充当物理建筑师生成包含层级关系、材料、运动学约束的 "Hierarchical Physical Blueprint"，扩散模型再用 KineVoxel Injection 把铰接参数和几何 voxel 协同去噪，配合 150k 资产、四层标注的 PhysDB 数据集，首次实现单视图到"可在物理引擎里抓握、推动、铰接"的 3D 资产生成。

**[PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update](polyflow_safe_and_efficient_polytope-constrained_flow_matching_with_constraint_e.md)**

:   PolyFlow 把"满足多面体硬约束"直接焊进流匹配模型的网络结构和流定义里——用离散时间流消掉数值积分误差、用 Frank-Wolfe 式的"射线打到边界 + 可学步长"代替昂贵的投影求解器，从而在规划/控制任务上做到**零约束违反**的同时把推理延迟降低一到两个数量级。

**[Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)**

:   本文是一篇立场论文，主张"组织引入 AI 并不自动等于生产力提升"，识别出五个被传统经济模型忽略的人与环境调节因子（人员组成、个体基线能力、学习曲线、公平使用激励、目标灵活性），并在 Gries-Naudé (2022) 偏均衡模型上加入组织有效性 $\Omega$、能力调整 $\phi(z,\kappa_i)$、学习曲线 $\lambda_i(\tau)$、有效自动化阈值 $\tilde N_{IT}$ 四类修正项，得到能描述"为什么同样投 AI 不同组织产出差距巨大"的修订生产函数。

**[Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)**

:   作者主张"benchmark 分数 = 能力"是一种**隐式推断**而非直接测量，呼吁把 AI 评测显式建模成统计推断任务，并借鉴心理测量学（CTT/IRT/CDM/BNSM）四种能力理论作为模板，给出一张"Evaluation Card"让评测者自证假设。

**[Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)**

:   GCPO 把 GRPO 在 flow matching 后训练里"每一步都用同一个最终 reward 当 advantage"的步级优化改成"块级"——按 flow matching 自身的时间动态 $L1_{rel}(x,t)$ 自适应地把连续若干步聚成 chunk，用规范化的 chunk-level 重要性比 $r^i_j$ 做策略更新，从而平滑掉"最终好≠每步好"造成的错误梯度，在 HPSv3/ImageReward/GenEval/DPG 上相对 GRPO 取得最高 43% 的相对增益。

**[Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)**

:   本文首次为基于 DiT 的真实图像超分（Real-ISR）设计了 PTQ 框架 Q-DiT4SR，通过「全局低秩 + 局部分块 rank-1」的层级 SVD 分解保留高频细节，并基于率失真理论提出无需校准数据的层间权重位宽分配（VaSMP）与基于动态规划的时间步激活位宽调度（VaTMP），在 W4A6 / W4A4 极低位设置下达到 SOTA，并将模型压缩 5.8× / 计算量减少 6.14×。

**[Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)**

:   本文在 score-based 扩散模型上对"用合成数据递归训练导致 model collapse"这一现象给出第一套配对的上下界：单代散度 $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$，多代累积散度 $D_N$ 是过去各代 score 误差能量按 $(1-\alpha)^{2m}$ 几何衰减的加权和，从而把"加新鲜数据能缓解坍塌"这一经验事实化成了精确的衰减律。

**[RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)**

:   RAIGen 用 Matryoshka 稀疏自编码器把 T2I 扩散模型 bottleneck 表征分解成可解释 neuron，再用"激活稀有度 × CLIP 语义偏离度"组合分数从中挑出"模型内部已编码但生成时几乎不出现"的少数属性 neuron，从而把偏见审计从"已知公平类目"和"显著多数模式"扩展到 label-free 的稀有属性发现。

**[Rao-Blackwellized Score Matching on Manifolds](rao-blackwellized_score_matching_on_manifolds.md)**

:   当数据分布落在嵌入流形 $M\subset\mathbb{R}^D$ 上时，环境空间高斯加噪做 DSM 学到的切向目标含有方差以 $d/\sigma^2$ 发散的法向噪声通道；本文证明对最近点投影 $\pi(X)$ 做一次 Rao-Blackwell 条件化即可干净地去掉这个奇异通道，并把剩下的目标精确展开为「内蕴 Riemannian score + $\sigma^2$ 阶 Tweedie 校正 + $\sigma^2$ 阶 Weingarten/Ricci 外蕴曲率校正」。

**[Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)**

:   EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度，从而在不做对抗优化的情况下"白送"一个可用作下游 RL shaping reward 的标量信号，同时保守场约束改善 OOD 泛化。

**[Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)**

:   本文指出现有 T2I 扩散蒸馏只做"逐点输出对齐"导致学生模型对初始噪声的敏感性塌缩，提出 GAD：用一对扰动输入下的 JVP（雅可比向量积）有限差分近似，强制学生匹配教师对噪声扰动的方向性响应，从而在不损失保真度的前提下恢复布局可控性与生成多样性。

**[Rethinking FID Through the Geometry of the Reference Dataset](rethinking_fid_through_the_geometry_of_the_reference_dataset.md)**

:   本文指出 FID 的"越低越好"假设在不同参考数据集上系统性失效，并用分布密度 $\langle -\log d_k\rangle$ 和有效秩 $\mathrm{erank}(A)$ 两个几何描述子，通过分层线性模型证明它们能解释 ~70% 的"样本质量→FID"斜率跨数据集差异，从而把 FID 的脆弱性首次定量归因到参考集本身。

**[Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)**

:   把 MeanFlow 的"平均速度一步生成"推广到黎曼流形：用**平行移动**把不同切空间里的瞬时速度搬到同一切空间再平均，从而定义流形上的平均速度并导出**黎曼 MeanFlow 恒等式**；再用对数映射在公共切空间里做内蕴训练（避开轨迹模拟和 Christoffel 符号），把目标拆成两项并用 PCGrad 化解梯度冲突，在球面/环面/SO(3)/SE(3) 上以 1 步采样达到与最强 baseline 相当的质量、采样成本大幅下降。

**[RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models](rt-lynx_putting_the_gemm_sparsity_in_a_right_way_for_diffusion_models.md)**

:   作者发现 DiT 的**激活**比**权重**更天然稀疏（每个 token 只激活 5–10% 通道），于是把 2:4 半结构化稀疏从权重侧搬到激活侧，再用 norm 缩放 + LoRA 残差补偿 + 选择性跳层把质量损失补回来，并写了一套把"在线 Top-K 选择 + Sparse GEMM"融合到单 kernel 的 CUDA 流水，在 Qwen-Image / FLUX / Z-Image 上做到线性层平均 1.55× 加速且 FID/IR 不退化。

**[SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)**

:   通过在稀疏自编码器（SAE）训练阶段加入监督的"概念—潜变量"指派损失，强制每个目标概念集中到单个神经元（feature centralization），从而把扩散模型的概念擦除从"搜多神经元 + 调强度"的二维超参搜索压成"只调一个 multiplier"，在 UnlearnCanvas 上比 SOTA 的 SAeUron 平均提升 9.22 个点，超参搜索代价降低 96.67%，并对对抗攻击更鲁棒。

**[Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)**

:   针对 Stable Diffusion / Flux 这类基础流匹配模型在求解逆问题上明显逊于领域专用先验甚至未训练先验的现象，作者提出 FMPlug：用一个由近似样本指导、时间可学习的 warm-start 加上锐利高斯壳层约束，把基础 FM 的潜变量塞回它真正"懂"的薄壳上，从而显著恢复其作为逆问题先验的能力。

**[Scalable GANs with Transformers](scalable_gans_with_transformers.md)**

:   本文提出 GAT（Generative Adversarial Transformers）——一套在 VAE 隐空间上用纯 Transformer 生成器与判别器搭起来的可扩展 GAN 框架，通过多层级噪声扰动监督（MNG）激活早期生成器层、并用宽度感知的学习率缩放稳定大模型训练，使 GAT-XL/2 在 ImageNet-256 类条件生成上仅训练 60 epoch 就拿到 FID 2.18 的单步生成 SOTA，比同等规模 1-NFE diffusion/flow baseline 少用 4× epoch。

**[SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)**

:   SceneSmith 用 designer-critic-orchestrator 三角 VLM agent 在「布局→家具→小物件」的层级树上逐层构建室内场景，并把 text-to-3D 生成、铰接物体检索与物理属性估计深度耦合到 agent 工具链中，从单条自然语言提示直接产出"可直接喂给物理仿真器"的稠密、可操作环境，每个房间平均 71 个物体（基线只有 11–23 个），物体间碰撞率 <2%、重力下稳定率 96%，远超此前所有方法。

**[Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](self-prompting_diffusion_transformer_for_open-vocabulary_scene_text_editing_via_.md)**

:   本文提出一种基于 FLUX-Fill (MM-DiT) 的自提示场景文字编辑方法：直接从原图裁出风格 prompt、用 Pillow 渲染出 glyph prompt，两者与 masked image 沿通道拼接后送入扩散 backbone，再用 4000 张 Nano Banana Pro 生成的高质量配对图做 cooldown 训练，从而在 13 种语言上同时实现开放词表与原始风格一致的文字替换。

**[Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)**

:   SATA 用 MLLM 生成的关节语义标签做 FiLM 风格的特征调制，配合空间-时间交错的图自编码器，把任意骨架拓扑的 BVH 动作压到一个共享潜空间，实现高保真重建以及无配对数据的零样本跨物种动作重定向。

**[Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)**

:   NaviEdit 把 diffusion/flow 编辑器中"模型尺度坐标 = 编辑进度时钟"的隐式耦合拆开，在固定 step budget 下用一个训练无关的推理时控制器把算力集中在一个有效尺度窗口的密度上而非把范围扩到高噪声区，从而在 PIE-Bench / ImgEdit-Bench / 多种 flow backbone 上同时改善背景保真和语义一致性。

**[Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)**

:   针对 FLUX.1 Kontext 这类基于 Rectified Flow Matching 的 MMDiT 编辑模型在多实例同时编辑下"属性串味"的痼疾，本文提出 Instance-Disentangled Attention（IDAttn）：通过对 joint attention 加结构化掩码，把每条编辑指令绑定到对应的 bounding box，再配合分层 disentanglement/harmonization 调度和高效多 prompt 独立编码，单次前向就能完成 N 条互不干扰的编辑，并在自家提出的 Infographic 文本编辑 benchmark 上显著优于多轮和拼接式 baseline。

**[Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures](simple_approximation_and_derivative_free_inference-time_scaling_for_diffusion_mo.md)**

:   作者把扩散模型的推理时 reward 引导从"粒子空间 SMC + 高阶导数"升级为"路径空间 SMC + Girsanov 似然比"，得到 URGE 算法：每条轨迹只需对 guidance $G$ 做一阶梯度并累加一个简单的 Itô 项当权重，完全不需要 reward $r$ 的导数 / Hessian / score 估计，在 GMM、逆问题和文生图三类任务上都打平或优于 FK-Corrector / AFDPS / FK-Steering。

**[Skipping the Zeros in Diffusion Models for Sparse Data Generation](skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)**

:   SED 把扩散模型从"对所有维度做全密集去噪"改成"只在非零维度上跑扩散+自回归解码维度-值对"，让计算量从随维度线性增长变成几乎随非零数恒定，同时严格保留科学数据中"显式零"这一语义信息。

**[SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)**

:   作者指出 MLLM 类编辑奖励模型存在"注意力坍缩"问题——评分时不去比较原图与编辑后图、而是塌缩到 sink token 上做盲判，进而提出 SpatialReward：先让 8B 模型预测编辑区域的边界框、再以这些 box token 为锚做交错式跨图推理；配上一个 260K 样本的空间感知数据集和 GRPO 两阶段训练后，在三个 reward benchmark 上 SOTA，并把 OmniGen2 的 GEdit-Bench 分数拉升 +0.90（是 GPT-4.1 提升的两倍）。

**[Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)**

:   本文提出 Spectral Guidance：通过自监督学习扩散过程条件期望算子的左奇异函数，把任意引导信号（标签 / CLIP / mask）投影到这组与扩散动力学对齐的谱基上，绕开 denoiser 反向传播，在 CIFAR-10 上较最强 training-free 基线提升 37 个百分点准确率且采样快 4 倍。

**[Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)**

:   本文发现 Speculative Jacobi Decoding (SJD) 在自回归视觉生成中加速有限的根因是连续迭代之间 draft token 的独立采样导致 collision 概率几乎为零；只需把独立采样换成 Maximal/Gumbel Coupling（一行修改、零额外训练），就能把图像生成最高加速到 $4.2\times$、视频生成 $13.6\times$，并严格保持输出分布与原 AR 解码一致。

**[Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)**

:   本文从"条件速度方差"这一被忽视的视角重新审视 flow matching，发现训练轨迹天然分裂为靠近先验的高方差区和靠近数据的低方差区，并据此提出统一框架 Stable Velocity，含一个无偏的多样本方差缩减损失 StableVM、一个只在低方差区启用 REPA 的 VA-REPA，以及一个利用低方差区闭式解的无微调采样加速器 StableVS，在 ImageNet 256 与 SD3.5/Flux/Qwen-Image/Wan2.2 上取得训练效率提升与 >2× 采样加速。

**[Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)**

:   提出 MAP-RPS 两阶段框架：先用扩散模型的 score 做 MAP 估计逼近 MMSE 解（低失真起点），再把 MAP 结果 re-noise 到时刻 $t_0$ 后做后验采样（沿 D-P 曲线滑向高感知质量），单一预训练扩散模型就能在推理时灵活遍历 distortion-perception trade-off，并扩展到 latent diffusion 后在 MS-COCO 上多任务 SOTA。

**[STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)**

:   本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE），不仅把 attack success rate 比 SOTA 提升 68%，更揭示了一个全新现象——Optimization-Induced Phase Alignment：对抗优化会自动把"概念性毒性"绑到去噪早期、"细节性毒性"绑到后期，从而把混沌的毒性形成过程变成几个可预测的"漏洞时间窗"。

**[Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)**

:   SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里，使离线黑盒优化在 Design-Bench 和 LLM 数据混合任务上稳定达到 SOTA。

**[SURF: Separation via Unsupervised Remixing Flow](surf_separation_via_unsupervised_remixing_flow.md)**

:   SURF 把监督流匹配 FLOSS 与无监督的 ReMixIT / Self-Remixing 教师-学生重混合训练拼到一起，让一个生成式 flow matching 分离器**完全从混合观测**（没有任何干净源样本）训练出来，在 MNIST/CIFAR10 图像分离和 LibriSpeech / FUSS 音频分离上几乎追平有监督 flow 的指标，刷新无监督 SOTA。

**[SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)**

:   SURGE 把扩散代理模型的引导采样视为路径测度上的有偏分布，用 Girsanov 公式计算重要性权重做 SMC 重采样，从而在不重新训练、不近似 Doob $h$-变换的前提下，得到无近似偏差的扩散代理数据同化滤波器，在 Lorenz、Navier-Stokes 和 SEVIR 天气预报上一致超越 BPF/EnKF/SDA/FlowDAS。

**[Temporal Difference Learning for Diffusion Models](temporal_difference_learning_for_diffusion_models.md)**

:   论文把扩散去噪过程重写成马尔可夫奖励过程（MRP）、把训练当成强化学习里的策略评估，提出一个时序差分（TD）目标，强制模型沿去噪轨迹的「多步漂移」与真实扩散漂移一致；它作为可即插即用的正则项叠加在 EDM/一致性训练等基线损失上，显著改善 FID，尤其在少步采样（小 NFE）场景下优势更明显。

**[The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)**

:   本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对，从而把 FM 收敛速度、少步数 FID 同时拉到新的水平，并反过来比当老师的 NF 推理快若干个数量级。

**[The Latent Color Subspace: Emergent Order in High-Dimensional Chaos](the_latent_color_subspace_emergent_order_in_high-dimensional_chaos.md)**

:   作者发现 FLUX.1 的 VAE 隐空间里"颜色"只占据一个三维子空间（Latent Color Subspace, LCS），其几何形状几乎就是 HSL 颜色模型的双锥体，并据此提出一套**完全免训练、纯闭式隐空间变换**的方法，既能在生成中途直接"读出"将要生成的颜色，又能把指定物体精确改成目标颜色。

**[Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)**

:   作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它，再加一个与 $|s-\tau|$ 成正比的置信度权重，让扩散模型和 MaskGIT 在仅有标量打分（无成对偏好）时也能稳定对齐，并在五个 reward model 三个测试集上一致优于 Diffusion-DPO / KTO / DSPO。

**[Timestep Rescheduling in Diffusion Inversion](timestep_rescheduling_in_diffusion_inversion.md)**

:   作者发现扩散反演（diffusion inversion）的误差强烈依赖时间步大小、且随时间步索引呈"两端高中间低"的抛物线分布，于是提出一个免训练、零额外开销的非均匀时间步调度器 TRDI——先全局拉伸时间步、再用动态规划局部重排，把算力集中到误差大的区段，作为即插即用插件稳定提升各类反演方法在重建与编辑上的精度。

**[Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)**

:   DiffMark 把一个学到的潜空间扰动 $\delta$ 在冻结扩散模型的每一步去噪中持续注入，让水印信号在终态潜变量 $z_0$ 上累积，并借助 Latent Consistency Model 作为可微训练桥绕过 50 步 DDIM 的反向传播，实现单次前向 16.4 ms 解出 64 bit、跨模型即插即用且无需重训的水印方案。

**[UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning](unhype_clip-guided_hypernetworks_for_dynamic_lora_unlearning.md)**

:   UnHype 用一个以 CLIP 文本嵌入为输入的超网络，在推理时动态生成 LoRA 权重——遇到要遗忘的概念就生成能抑制它的 LoRA，遇到正常概念就生成接近零的 LoRA——从而把"每个概念单独训练一个 LoRA"的静态遗忘，改造成"一个模型按文本即时生成遗忘适配器"的摊销式遗忘，同时支持 Stable Diffusion 和 Flux。

**[统一不同生成顺序的掩码扩散模型](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)**

:   提出统一框架 OeMDM 和学习型版本 LoMDM——通过显式建模"速度"（生成优先级）将随机掩码、自回归、块扩散模型统一在一个 NELBO 下，实现从零开始联合学习生成顺序和扩散骨干。

**[扩散模型中的遗忘：基于 KL 散度和似然约束的统一框架](unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)**

:   本文提出统一的约束优化框架——将扩散模型中的机器遗忘问题形式化为最小化与预训练模型的偏差，同时受约束于明确的与遗忘分布的分离条件，通过三种约束形式（反向 KL、前向 KL、似然约束）统一处理概念遗忘和数据遗忘，并证明强对偶性。

**[ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)**

:   通过离散扩散模型和视觉 token 化，将多视图生成建模为离散序列预测任务——利用简单的随机掩码策略结合自注意力自然地实现跨视图一致性，显著超越连续扩散方法。

**[Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)**

:   本文把 Deep Equilibrium（DEQ）隐式不动点层嵌进 VAR 的 next-scale 自回归框架，用 Jacobian-Free Backpropagation 实现常数显存训练，把 VAR-d30 的 20 亿参数压到 7.7 亿，同时在推理时把每个 scale 的迭代次数变成"可调旋钮"——在 ImageNet-256 上 FID 2.16/sFID 8.07 不变的同时，4090 单卡峰值显存从 19.24GB 降到 8.53GB、吞吐从 15.16 提到 32.08 img/s。

**[Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)**

:   本文揭示扩散模型里一直被忽视的"时间步嵌入"其实是一条尚未被占用的信息侧信道——通过把训练时的 timestep 范围扩展到一个"影子区间"（shadow timestep）并把另一个数据分布绑定到该区间，可以在不改变 scheduler 接口的前提下，让同一个 diffusion 模型在显式区间生成正常图、在影子区间生成"隐藏"图，既可做隐蔽后门攻击也可做模型水印验证；同时给出基于正弦位置编码的互相干（mutual coherence）理论分析，解释为什么两个不相交区间能携带独立信息。

**[Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)**

:   论文发现低保真或领域不匹配的扩散模型先验在信息丰富的逆问题中仍能取得强劲性能——通过贝叶斯一致性理论和局部相关性分析解释了这一看似矛盾的现象，并给出何时弱先验有效的明确条件。

**[When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](when_preference_labels_fall_short_aligning_diffusion_models_from_real_data.md)**

:   这篇论文认为由生成图像组成的偏好标签容易把模型带向“相对更好但仍有缺陷”的样本，提出用真实图像及其可控退化版本自动构造偏好信号，在只用 512 对样本的情况下对齐 SD-1.5 和 SD-3.5-M，并取得接近或补充 Diffusion-DPO / FlowGRPO 的效果。

**[WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)**

:   WISE 构建了一个包含 1000 条知识密集 prompt 的文本到图像评测基准，用文化常识、时空推理和自然科学知识检验模型是否能把隐含语义转化成正确视觉内容，并发现现有 T2I 与统一多模态模型在世界知识生成上仍有明显短板。

**[You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models](you_dont_need_all_that_attention_surgical_memorization_mitigation_in_text-to-ima.md)**

:   本文提出 GUARD，一个推理时的文生图扩散模型记忆缓解框架，通过对标准 classifier-free guidance 加入“远离原始记忆提示”的 repulsion 和“靠近安全条件预测”的 attraction，并用动态 cross-attention spike 检测与衰减实例化 positive target，在降低训练图像复现的同时尽量保持图像质量和 prompt 对齐。

**[Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)**

:   本文提出一种带方差缩减的零阶 Langevin 采样方法，用间歇性大 batch 估计和递推式小 batch 更新替代每步 $O(d)$ 次函数查询，并把它扩展为 ZO-APMC，用预训练 score-based prior 在只有前向模型查询的黑盒逆问题中做有收敛保证的后验采样。

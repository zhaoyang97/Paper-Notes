<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🔬 ICLR2026** · 共 **50** 篇

**[A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)**

:   对扩散 Transformer 的条件嵌入进行首次系统分析，发现极端的角度相似性（类间余弦相似度>99%）和维度稀疏性（仅 1-2% 的维度携带语义信息），裁剪掉 2/3 的低幅维度后生成质量基本不变，揭示了条件嵌入中隐藏的语义瓶颈。

**[AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)**

:   提出 AlignTok，将预训练视觉基础编码器（如 DINOv2）对齐为扩散模型的连续 tokenizer，通过三阶段对齐策略（语义潜空间建立→感知细节补充→解码器精炼）构建语义丰富的潜空间，在 ImageNet 256×256 上 64 epochs 即达 gFID 1.90，比从头训练 VAE 收敛更快、生成质量更好。

**[Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](amortising_inference_and_meta-learning_priors_in_neural_networks.md)**

:   提出 BNNP（Bayesian Neural Network Process），一种将 BNN 权重作为隐变量、BNN 本身作为解码器的 neural process，通过逐层 amortised variational inference 在多数据集上联合学习 BNN 先验和推断网络，首次回答了"在良好先验下，近似推断方法还重要吗？"——答案是肯定的，没有免费午餐。

**[Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)**

:   AsynDM 通过为不同像素分配不同的时间步调度（prompt 相关区域去噪更慢），使其能利用更清晰的上下文参考，从而在不需要微调的情况下显著提升文图生成的语义对齐。

**[Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)**

:   提出 Token Constraint Bound ($\delta_{\text{TCB}}$) 指标，通过量化 LLM 隐状态在多大扰动范围内能保持 next-token 预测不变，来度量预测的局部鲁棒性，揭示了传统 perplexity 无法捕捉的预测不稳定性。

**[Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)**

:   Blueprint-Bench 通过"从公寓内部照片生成 2D 平面图"的任务来评测 AI 模型的空间推理能力，结果显示大多数 LLM、图像生成模型和 Agent 系统的表现接近或低于随机基线，揭示了当前 AI 在空间智能上的重大盲区。

**[Bridging Degradation Discrimination and Generation for Universal Image Restoration](bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)**

:   BDG 通过多角度多尺度灰度共生矩阵（MAS-GLCM）进行细粒度退化判别，并设计三阶段扩散训练（生成→桥接→修复）将退化判别能力与生成先验无缝融合，在 all-in-one 修复和真实世界超分辨率任务上取得显著的保真度提升。

**[Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)**

:   FedVTC 提出在模型异构联邦学习中，各客户端通过变分转置卷积网络（VTC）从聚合的特征分布统计量中生成合成数据来微调本地模型，无需公共数据集即可显著提升泛化能力，同时降低通信和内存开销。

**[CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)**

:   提出 Consistency Mid-Training (CMT)，在预训练扩散模型和 flow map 后训练之间插入一个轻量级中间训练阶段，通过让模型学习将 ODE 轨迹上的任意点映射回干净样本来获得轨迹对齐的初始化，从而大幅降低训练成本（最多 98%）并达到 SOTA 两步生成质量。

**[Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)**

:   提出 General Policy Composition (GPC)，在测试时通过凸组合多个预训练扩散/Flow 策略的分布分数（score），无需额外训练即可产生超越任何单一父策略的更强策略，理论证明凸组合可改善单步分数误差且通过 Grönwall 界传播到全程轨迹。

**[Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)**

:   理论分析了自回归扩散损失模型相比条件扩散模型在条件误差修正上的优势（梯度范数指数衰减），并提出基于最优传输（Wasserstein Gradient Flow）的条件精炼方法来解决自回归过程中的"条件不一致性"问题，在 ImageNet 上达到 FID 1.31（基于 MAR）。

**[Consistent Text-to-Image Generation via Scene De-Contextualization](consistent_text-to-image_generation_via_scene_de-contextualization.md)**

:   揭示 T2I 模型中 ID 偏移的根本原因是"场景上下文化"（scene contextualization，场景 token 对 ID token 注入上下文信息），并提出 training-free 的 Scene De-Contextualization (SDeC) 方法，通过 SVD 特征值的方向稳定性分析识别并抑制 prompt embedding 中潜在的场景-ID 关联，实现逐场景的身份一致性生成。

**[ContextBench: Modifying Contexts for Targeted Latent Activation](contextbench_modifying_contexts_for_targeted_latent_activation.md)**

:   提出 ContextBench 基准（715 个任务）评估自动生成流畅且能激活特定潜在特征的输入文本的方法，并开发两种 EPO 增强变体（LLM辅助和扩散模型修补），在激活强度和语言流畅度的权衡上 Pareto 优于标准 EPO。

**[Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)**

:   首次系统研究 T2I 扩散模型的持续遗忘（continual unlearning）问题，发现现有遗忘方法在序列请求下因累积参数漂移导致"效用崩溃"，提出一组附加正则化策略（L1/L2 范数、选择性微调、模型合并）和语义感知的梯度投影方法来缓解该问题。

**[CREPE: Controlling Diffusion with Replica Exchange](crepe_controlling_diffusion_with_replica_exchange.md)**

:   提出 CREPE，一种基于 Replica Exchange（并行回火/Parallel Tempering）的扩散模型推理时控制方法，作为 SMC 的计算对偶——在去噪步维度上并行、在样本维度上串行生成，具有高样本多样性、可在线精炼、支持温度退火/奖励倾斜/模型组合/CFG 去偏等多种任务。

**[DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)**

:   解决 Flow Matching + GRPO 对齐中的稀疏奖励问题：通过 ODE 去噪预测中间潜变量的 step-wise 奖励增益作为密集奖励，并根据密集奖励自适应调整 SDE 采样器的逐时间步噪声注入来校准探索空间，在人类偏好对齐/组合生成/文字渲染三个任务上超越 Flow-GRPO。

**[Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)**

:   本文证明基于范数的记忆检测指标仅在各向同性（isotropic）对数概率分布下有效，在低噪声各向异性（anisotropic）区域失效；提出结合高噪声范数和低噪声角度对齐（cosine similarity）的无去噪检测指标，在 SD v1.4/v2.0 上超越现有无去噪方法且快 5× 以上。

**[DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation](diffink_glyph-_and_style-aware_latent_diffusion_transformer_for_text_to_online_h.md)**

:   提出 DiffInk，首个面向全行手写生成的潜在扩散 Transformer 框架，包含 InkVAE（通过 OCR + 风格分类双正则化学习结构化潜空间）和 InkDiT（在潜空间中做条件去噪生成），在中文手写生成上大幅超越 SOTA（AR 94.38% vs 91.48%），速度提升 800×。

**[Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)**

:   将扩散模型对齐形式化为变分 EM 算法：E-step 用 test-time search（soft Q 引导 + 重要性采样）探索高奖励多模态轨迹，M-step 通过 forward-KL 蒸馏将搜索结果写入模型参数，在图像生成和 DNA 序列设计上同时实现高奖励和高多样性。

**[Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)**

:   提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA 采样消除推理开销，理论上证明了混合近似的误差界并在实验中接近 MORL oracle 上界。

**[DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)**

:   提出 DiffusionNFT，一种全新的扩散模型在线 RL 范式：不在反向采样过程上做策略优化（如 GRPO），而是在前向过程上通过 flow matching 目标对正样本和负样本做对比式训练，定义隐式的策略改进方向，比 FlowGRPO 快 3-25×，且无需 CFG。

**[Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)**

:   提出 DrPose，通过直接奖励微调最大化 PoseScore（骨骼结构一致性）+ KL 正则化，结合 DrPose15K 数据集（来自 Motion-X 的 15K 动态姿态），使多视角扩散模型可重建户外动态/杂技人体姿态。

**[Directional Textual Inversion for Personalized Text-to-Image Generation](directional_textual_inversion_for_personalized_text-to-image_generation.md)**

:   本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。

**[DistillKac: Few-Step Image Generation via Damped Wave Equations](distillkac_few-step_image_generation_via_damped_wave_equations.md)**

:   用阻尼波方程（telegrapher equation）及其随机 Kac 表示替代 Fokker-Planck 方程作为生成模型的概率流基础，实现有限速度传播的概率流，并提出端点蒸馏（endpoint distillation）方法实现少步生成，在 CIFAR-10 上 4 步 FID=4.14、1 步 FID=5.66。

**[DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)**

:   首个将 FLUX (DiT) 的强生成先验引入拖拽编辑的框架，通过区域级仿射监督替代传统点级监督，配合梯度掩码硬约束和 adapter 增强反演，大幅提升拖拽编辑质量。

**[Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing](draw-in-mind_rebalancing_designer-painter_roles_in_unified_multimodal_models_ben.md)**

:   指出当前统一多模态模型中理解模块仅作翻译器而生成模块被迫同时充当"设计师"和"画家"的职责失衡问题，通过构建 DIM 数据集（14M 长上下文文图对 + 233K CoT 编辑蓝图）将设计责任转移给理解模块，4.6B 参数即超越 5 倍大的模型。

**[Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)**

:   提出 Dual-Solver，通过三组可学习参数（预测类型插值 $\gamma$、积分域选择 $\tau$、残差调整 $\kappa$）泛化扩散模型多步采样器，用冻结预训练分类器（MobileNet/CLIP）的分类损失学习参数（无需教师轨迹），在 3-9 NFE 低步区间全面优于 DPM-Solver++ 等方法。

**[EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)**

:   提出首个系统性的"基准评测→奖励模型→强化学习训练"图像编辑 RL 管线：构建 EditReward-Bench 基准，训练 EditScore 系列奖励模型（7B-72B，超过 GPT-5），并成功将其用于 Online RL 训练显著提升编辑模型性能。

**[SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)**

:   提出 Single-Step Completion Policy (SSCP)，通过在流匹配框架中预测"完成向量"（从任意中间状态到目标动作的归一化方向），将多步生成策略压缩为单步推理，在 D4RL 上与多步扩散/流策略持平但训练快 64×、推理快 4.7×，并扩展到 GCRL 中将层级策略扁平化。

**[Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)**

:   提出 Follow-Your-Shape，一个无需训练和掩码的形状感知编辑框架，通过计算反演与编辑轨迹间的 token 级速度差异构建 Trajectory Divergence Map (TDM) 来精确定位编辑区域，配合分阶段 KV 注入实现大幅形状变换且严格保持背景。

**[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)**

:   提出 Frame Guidance，一种无需训练的帧级引导方法，通过 latent slicing（降低 60× 显存）和 Video Latent Optimization（VLO）两个核心组件，在不修改模型的情况下实现关键帧引导、风格化和循环视频等多种可控视频生成任务。

**[GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)**

:   提出 GenCP，将耦合多物理场仿真建模为概率密度演化问题，利用 flow matching 从解耦数据学习条件速度场，推理时通过 Lie-Trotter 算子分裂合成耦合解，实现"解耦训练、耦合推理"，并提供理论误差可控保证。

**[GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](glass_flows_reward_alignment_diffusion.md)**

:   提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种在流/扩散模型的去噪过程中实现高效随机转移的新采样范式，通过充分统计量重参数化将随机转移重铸为内部 ODE 求解问题，在无需重训的条件下结合 ODE 效率和 SDE 随机性，使 Feynman-Kac Steering 在 FLUX 文生图模型上一致超越 Best-of-N 基线。

**[Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)**

:   Recall 提出首个多模态引导的攻击框架，通过在隐空间中优化对抗图像 prompt（仅需一张参考图像），配合原始文本 prompt 利用扩散模型的 image-conditioning 通道，在 10 种 SOTA 遗忘方法上平均 ASR 达 65%~97%，显著超越纯文本攻击方法，揭示当前遗忘机制对图像模态攻击的脆弱性。

**[Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)**

:   首次系统性地对比 Visual Autoregressive (VAR) 模型和扩散模型在组合文本-图像对齐上的表现，在 T2I-CompBench++ 和 GenEval 两个基准上评测 6 个 T2I 模型，发现 Infinity-8B 在几乎所有组合维度上取得最强表现，VAR 架构在组合生成方面展现出显著优势。

**[Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)**

:   HiRM 提出"更新位置与擦除目标解耦"的概念擦除策略——仅更新 CLIP 文本编码器第一层的权重，但将擦除监督施加在最后一层的高层语义表征上，通过引导目标概念表征偏向随机方向（HiRM-R）或语义方向（HiRM-S），在 UnlearnCanvas 和 NSFW 基准上实现风格/物体/裸体的高效擦除，且可零样本迁移到 Flux 架构。

**[LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)**

:   提出 LoRA-Edit，通过时空掩码感知 LoRA 微调实现可控的首帧引导视频编辑——掩码分离运动保持（背景）和外观控制（编辑区域），用户研究中运动一致性和背景保持优于 AnyV2V/I2VEdit。

**[Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)**

:   提出 Mod-Adapter，一种无需测试时微调的多概念个性化方法，通过在 DiT 的调制（modulation）空间中预测概念特定的调制方向，实现对物体和抽象概念（姿态、光照、材质等）的解耦化定制生成，在多概念个性化上大幅超越现有方法。

**[Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)**

:   提出 Motion Prior Distillation (MPD)，一种推理时蒸馏方法，将前向路径的运动残差蒸馏到后向路径中，从根本上解决了时间反转采样中双向运动先验冲突的问题，无需额外训练即可实现更连贯的生成式帧插值。

**[MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)**

:   提出多视角定制（multi-view customization）新任务并设计 MVCustom 框架，通过视频扩散骨干网络结合密集时空注意力实现整体帧一致性，在推理阶段引入深度感知特征渲染和一致性感知潜码补全两项技术，首次同时实现相机位姿控制、主体身份保持和跨视角几何一致性。

**[Neon: Negative Extrapolation From Self-Training Improves Image Generation](neon_negative_extrapolation_image_generation.md)**

:   提出 Neon，一种仅需 <1% 额外训练计算的后处理方法：先用模型自身生成的合成数据微调导致退化，再反向外推远离退化权重，证明 mode-seeking 采样器导致合成/真实数据梯度反对齐，因此负外推等价于向真实数据分布优化，在 ImageNet 256×256 上将 xAR-L 提升至 SOTA FID 1.02。

**[NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)**

:   提出 NeuralOS，使用 RNN 状态追踪 + 扩散渲染器的双组件架构，直接从用户输入事件（鼠标移动/点击/键盘）预测操作系统图形界面帧序列，首次实现用神经生成模型模拟操作系统。

**[RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)**

:   提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。

**[SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)**

:   提出 SenseFlow，通过隐式分布对齐（IDA）和段内引导（ISG）将分布匹配蒸馏（DMD）扩展到大规模 flow-based 文生图模型（SD 3.5 Large 8B / FLUX.1 dev 12B），实现 4 步高质量图像生成。

**[SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)**

:   提出 Solution Flow Models (SoFlow)，直接学习速度 ODE 的解函数 $f(x_t, t, s)$（将 $t$ 时刻的 $x_t$ 映射到 $s$ 时刻的解），通过 Flow Matching 损失 + 无需 JVP 的解一致性损失从头训练，在 ImageNet 256 上 1-NFE FID 优于 MeanFlow（XL/2: 2.96 vs 3.43）。

**[SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)**

:   SPEED 提出基于零空间（null space）约束的闭式模型编辑方法，通过影响力先验过滤（IPF）、定向先验增强（DPA）和不变等式约束（IEC）三种互补技术精化保留集，实现可扩展（5 秒内擦除 100 个概念）、精确（非目标概念语义零损失）且高效的概念擦除。

**[Steer Away From Mode Collisions: Improving Composition In Diffusion Models](steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)**

:   针对扩散模型多概念 prompt 中的概念缺失/碰撞问题，提出"模式碰撞"假说（联合分布与单概念分布的模式重叠），设计 CO3（Concept Contrasting Corrector）通过在 Tweedie 均值空间中组合校正分布 $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ 来远离退化模式，实现即插即用、无梯度、模型无关的组合生成改进。

**[TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](tavae_a_vae_with_adaptable_priors_explains_contextual_modulation_in_the_visual_c.md)**

:   扩展 VAE 形式主义提出 Task-Amortized VAE (TAVAE)，通过在已学表示上灵活学习任务特异性先验来解释视觉皮层 V1 中的上下文调制现象，包括方向辨别任务中训练刺激与测试刺激不匹配时出现的双模态群体响应。

**[Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)**

:   将 reward-guided 图像编辑重新建模为轨迹最优控制问题，将扩散/Flow模型的反向过程视为可控轨迹，通过基于 Pontryagin 最大值原理（PMP）的伴随状态迭代优化整条轨迹，在无需训练的情况下实现有效的奖励引导编辑且不发生 reward hacking。

**[TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)**

:   提出 TwinFlow，一种无需辅助训练模型（判别器/冻结教师）的自对抗流匹配框架，通过模型自身多步输出作为单步的教学目标实现单步生成，首次将 1-NFE 生成能力成功扩展到 20B 参数的 Qwen-Image 模型，GenEval 0.86（1-NFE）接近原始 100-NFE 的 0.87。

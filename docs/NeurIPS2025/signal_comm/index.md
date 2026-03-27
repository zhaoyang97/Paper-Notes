<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧠 NeurIPS2025** · 共 **13** 篇

**[Angular Steering: Behavior Control via Rotation in Activation Space](angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出 Angular Steering，将 LLM 激活引导统一建模为固定 2D 子空间中的旋转操作，提供连续、细粒度、范数保持的行为控制，统一了现有的激活加法和方向消融方法，在多个 LLM 家族（3B-14B）上实现鲁棒的行为控制。

**[Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)**

:   构建了 Infinity-Chat 数据集（26K 开放式真实用户查询 + 31,250 条人类标注），揭示了 LM 在开放式生成中的"Artificial Hivemind"效应——模型内重复和模型间同质化严重，并发现 Reward Model 和 LM Judge 在个体偏好差异大的样本上校准失败。

**[Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[ConTextTab: A Semantics-Aware Tabular In-Context Learner](contexttab_a_semantics-aware_tabular_in-context_learner.md)**

:   提出 ConTextTab，将语义理解融入 table-native ICL 框架，用数据类型特定嵌入并在大规模真实世界表格数据上训练，在语义丰富的 CARTE benchmark 上设立新 SOTA。

**[Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)**

:   提出 Token Timestep Allocation (TTA-Diffusion)，通过为每个 token 分配独立的去噪时间步来解决扩散语言模型中 classifier guidance 导致的 update-forgetting 问题，实现可控文本生成的稳定性和效率大幅提升。

**[Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)**

:   提出随机最优传输映射的新评价指标 $\mathcal{E}_p$（优化间隙+可行性间隙），发展了高效估计器，达到近优有限样本风险界 $\tilde{O}(n^{-1/(d+2p)})$，且仅需最小假设，是首个通用的（可能随机的）OT 映射估计理论。

**[Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)**

:   提出特征感知时间调制机制，通过基于时间上下文的可学习 Yeo-Johnson 变换动态调整特征分布（均值、标准差、偏度），实现跨时间语义对齐。

**[Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)**

:   提出 Masked Symbol Modeling，将 BERT 的掩码预测范式应用于通信物理层，将脉冲成形引起的符号间贡献视为上下文信息，训练 Transformer 在干净信号上学习波形结构，推理时通过上下文恢复被冲激噪声破坏的符号。

**[Memory-Integrated Reconfigurable Adapters (MIRA)](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   提出 MIRA，将 Hopfield 联想记忆与 LoRA adapter 结合，在共享 backbone 的每个 ViT 层上存储 adapter 权重更新为 value、事后学习的 key 检索，统一处理域泛化、类增量学习和域增量学习，在多个设置下达到 SoTA。

**[Multi-Modal Masked Autoencoders for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器 (MMAE) 应用于星系图像和光谱的联合重建，构建了 134,533 个星系的图像+光谱数据集，实现了光谱和图像的交叉重建以及仅从图像的红移回归，$\sigma_{\text{NMAD}} = 0.016$ 优于 AstroCLIP。

**[Perturbation Bounds for Low-Rank Inverse Approximations under Noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次给出在加性噪声下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，利用轮廓积分技术得到依赖特征间隙、谱衰减和噪声对齐的锐界，比经典全逆界改进高达 $\sqrt{n}$ 倍。

**[The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)**

:   揭示RLVR中负强化（仅惩罚错误）的効果超出预期，通过梯度分析说明其保持输出多样性和推理能力的机制，并提出改进的加权REINFORCE算法。

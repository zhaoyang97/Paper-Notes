<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🧠 NeurIPS2025** · 共 **15** 篇

**[Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)**

:   提出ADCM框架，将一致性模型(CM)的离散化步长选择形式化为约束优化问题，通过Gauss-Newton方法得到解析解，在局部一致性（可训练性）和全局一致性（稳定性）之间自适应平衡，以仅4%的额外计算开销实现显著的训练效率提升和FID改善。

**[Audio Super-Resolution With Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)**

:   提出 AudioLBM，在波形隐空间中用桥模型实现 LR-to-HR latent-to-latent 音频超分，配合频率感知训练和级联设计，LSD 平均改善 21.5%，首次实现 any-to-192kHz 音频超分。

**[DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)**

:   提出DenoiseRotator，在剪枝前插入可学习正交矩阵并通过熵最小化将参数重要性集中到子集上，使LLaMA3-70B在2:4稀疏下困惑度差距缩小58%（8.1→3.4），可与任何现有剪枝方法即插即用组合。

**[DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)**

:   提出DynaGuide，通过外部潜在动力学模型在DinoV2嵌入空间中预测未来视觉观测，利用分类器引导机制引导预训练扩散策略朝向目标条件动作，无需修改策略权重，在CALVIN上成功率70%、真实机器人80%。

**[Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)**

:   针对热红外(TIR)图像中低对比度、模糊、噪声等多种退化耦合的问题，提出基于双提示融合的渐进式网络PPFN和选择性渐进训练策略SPT，并构建首个大规模多场景TIR基准数据集HM-TIR，在复合退化场景下PSNR提升8.76%。

**[FIPER: Factorized Features for Robust Image Super-Resolution and Compression](fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)**

:   提出统一的基-系数分解框架（学习非均匀基+空间变化系数+坐标变换），在超分辨率上PSNR相对提升204.4%，在图像压缩上BD-rate降低9.35%，通过多频调制同时建模高/低频内容。

**[GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights](gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)**

:   提出 GC4NC——首个系统化的图凝缩（Graph Condensation）评估基准框架，跨 8 个维度（性能/效率/隐私保护/去噪/NAS有效性/可迁移性等）统一评估多种图凝缩方法，发现轨迹匹配方法最优、无结构方法效率最高，并在 1000x 压缩下图凝缩显著优于图像凝缩。

**[Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)**

:   提出 Learnable Linear Extrapolation (LLE)——用可学习的线性组合系数将当前和历史 clean data estimate 组合，以增强任何符合 Sampler-Corrector-Noiser 范式的扩散逆问题算法在少步（3-5步）下的表现，仅需 50 个样本、几分钟训练，跨 9+ 算法 × 5 个任务一致提升。

**[Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)**

:   提出共循环保守(CoCo)去噪器概念，通过广义Helmholtz分解设计新的训练策略——Hamiltonian正则化促进保守性 + 谱正则化促进共循环性——使去噪器成为隐式弱凸先验的近端算子，从而在Poisson逆问题（光子受限去卷积、低剂量CT等）中实现有收敛保证且性能优越的PnP方法。

**[Mro Enhancing Reasoning In Diffusion Language Models Via Multi-Reward Optimizati](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)**

:   MRO通过多奖励优化捕获扩散语言模型内/间序列token相关性，加速DLM推理同时保持性能。

**[Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)**

:   系统引入AND、OR、ADDER三种逻辑门来分解语言模型电路，揭示电路不完整性主要源于OR门的遗漏，提出结合noising和denoising干预的框架来完整恢复三种逻辑门，同时保证忠实度和完整性。

**[Rethinking Nighttime Image Deraining Via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)**

:   提出CST-Net用于夜间图像去雨：基于夜间雨在Y通道（亮度）上比RGB更显著的观察，设计可学习颜色空间转换器(CSC)在YCbCr空间去雨，配合隐式光照引导模块(IIG)和新构建的光照感知合成数据集HQ-NightRain，在多个基准上达到SOTA。

**[Spiking Meets Attention Efficient Remote Sensing Image Super-Resolution With Att](spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)**

:   提出 SpikeSR，首个基于注意力脉冲神经网络(SNN)的遥感图像超分辨率框架，通过脉冲注意力块(SAB)结合混合维度注意力(HDA)和可变形相似度注意力(DSA)，在 AID/DOTA/DIOR 上达到 SOTA 性能同时保持高计算效率。

**[The Effect Of Optimal Self-Distillation In Noisy Gaussian Mixture Model](the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)**

:   用统计物理replica方法分析噪声高斯混合模型上的自蒸馏，证明硬伪标签的去噪是性能提升主因，CIFAR-10实验验证。

**[Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open T2V Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)**

:   对开源T2V模型进行系统性延迟与能耗分析：建立了基于FLOP的compute-bound理论模型，验证了WAN2.1-T2V的二次空间/时间缩放和线性去噪步数缩放规律，并横向对比7个T2V模型发现能耗差异达3000倍（AnimateDiff 0.14Wh vs WAN2.1-14B 415Wh）。

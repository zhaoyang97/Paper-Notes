---
title: >-
  AAAI2026 物理/科学计算论文汇总 · 15篇论文解读
description: >-
  15篇AAAI2026的物理/科学计算方向论文解读，收录 Adaptive Fidelity Estimation f、Catastrophic Forgetting in Kol、Data Verification is the Futur等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "adaptive_fidelity_estimation_for_quantum_programs_with_graph/"
    t: "Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness"
  - u: "catastrophic_forgetting_in_kolmogorov-arnold_networks/"
    t: "Catastrophic Forgetting in Kolmogorov-Arnold Networks"
  - u: "data_verification_is_the_future_of_quantum_computing_copilots/"
    t: "Data Verification is the Future of Quantum Computing Copilots"
  - u: "fast_3d_surrogate_modeling_for_data_center_thermal_management/"
    t: "Fast 3D Surrogate Modeling for Data Center Thermal Management"
  - u: "flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-/"
    t: "FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer"
  - u: "just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical/"
    t: "Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems"
  - u: "knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang/"
    t: "Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction"
  - u: "learning_fair_representations_with_kolmogorov-arnold_networks/"
    t: "Learning Fair Representations with Kolmogorov-Arnold Networks"
  - u: "phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_/"
    t: "Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids"
  - u: "physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations/"
    t: "PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations"
  - u: "pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote/"
    t: "PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics"
  - u: "saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes/"
    t: "SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs"
  - u: "scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa/"
    t: "Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study"
  - u: "svd-no_learning_pde_solution_operators_with_svd_integral_kernels/"
    t: "SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels"
  - u: "towards_a_foundation_model_for_partial_differential_equations_across_physics_dom/"
    t: "Towards a Foundation Model for Partial Differential Equations Across Physics Domains"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**🤖 AAAI2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (69)](../../ICLR2026/physics/index.md) · [🧪 ICML2026 (33)](../../ICML2026/physics/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (20)](../../ICML2025/physics/index.md)

**[Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)**

:   提出 QuFid 框架，将量子电路建模为有向无环图，通过控制流感知的随机游走刻画噪声传播，利用算子谱特征量化电路复杂度，实现自适应测量预算分配，在保持保真度精度的同时大幅减少测量次数。

**[Catastrophic Forgetting in Kolmogorov-Arnold Networks](catastrophic_forgetting_in_kolmogorov-arnold_networks.md)**

:   首个系统性研究KAN（Kolmogorov-Arnold Networks）中灾难性遗忘行为的工作：建立了遗忘与激活支持重叠和数据内禀维度之间的理论框架，并提出KAN-LoRA用于语言模型的持续微调知识编辑。

**[Data Verification is the Future of Quantum Computing Copilots](data_verification_is_the_future_of_quantum_computing_copilots.md)**

:   这是一篇 position paper，提出量子计算 AI 助手（Copilot）必须将数据验证从事后过滤提升为架构级基础——通过三个立场论证：(1) 验证数据是最低要求，(2) 先验约束优于后验过滤，(3) 受物理定律约束的科学领域需要验证感知架构。实验表明无验证数据的 LLM 在电路优化上最高仅达 79% 准确率。

**[Fast 3D Surrogate Modeling for Data Center Thermal Management](fast_3d_surrogate_modeling_for_data_center_thermal_management.md)**

:   本文开发了基于视觉的 3D 代理建模框架，通过将数据中心的服务器负载、风扇速度和空调温度设定点编码为 3D 体素表示，利用 3D CNN U-Net、3D 傅里叶神经算子和 3D Vision Transformer 等架构实现实时温度场预测，速度比传统 CFD 求解器快 20000 倍，同时实现 7% 的能耗节约。

**[FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)**

:   深入分析 KAT（Kolmogorov-Arnold Transformer）训练慢 123 倍的根因，发现瓶颈并非 FLOPs 而是反向传播中**梯度累积的内存停顿**（atomic add 导致全局内存竞争），提出 FlashKAT 通过重构 GPU 核函数将训练加速 **86.5 倍**并降低近一个数量级的梯度舍入误差。

**[Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems](just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)**

:   提出随机稀疏反馈控制框架：控制器在每个时间步仅访问状态向量的随机子集，通过 LMI 联合设计反馈增益矩阵和 Bernoulli 稀疏化参数，在保证渐近均方稳定性（AMSS）的同时最小化所需传感器数量，实验中仅用 0.3% 的状态分量即可达到与全状态反馈可比的性能。

**[Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)**

:   提出 KARMA 框架，在 ViT-MAE 解码器中嵌入线性光谱混合模型 (LSMM) 作为物理约束，结合 Spectral Angle Mapper (SAM) 损失，提升高光谱遥感图像的重建保真度和下游任务迁移性能。

**[Learning Fair Representations with Kolmogorov-Arnold Networks](learning_fair_representations_with_kolmogorov-arnold_networks.md)**

:   提出将Kolmogorov-Arnold网络（KAN）引入对抗去偏框架，利用KAN的样条函数架构提供理论上的Lipschitz连续性和平滑性保证，并设计自适应 $\lambda$ 更新机制动态平衡公平性与准确率，在UCI大学录取数据集上实现了公平性指标的显著提升。

**[Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids](phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)**

:   构建了 Phys-Liquid 数据集（97,200 张物理仿真图像 + 3D mesh），基于 Navier-Stokes 方程模拟透明容器内液体的动态形变，并提出四阶段重建管线（分割→多视角 mask 生成→3D 重建→缩放），在仿真和真实场景中实现高精度液体几何与体积估计。

**[PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)**

:   提出 PhysicsCorrect，一种无需训练的校正框架，通过将 PDE 残差校正建模为线性化逆问题并预计算伪逆缓存，在推理时以 <5% 计算开销实现最高 100× 误差降低，适用于 FNO/UNet/ViT 等任意预训练神经算子。

**[PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)**

:   提出 PIMRL 框架，针对 burst 采样（短段高频+长间隔）的稀疏时空数据，结合宏观尺度潜空间推理和微观尺度物理校正的双模块架构，通过跨尺度消息传递融合信息，在 5 个 PDE 基准上将误差最多降低 80%。

**[SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)**

:   提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力（WA）捕获高频局部细节，与傅里叶注意力（FA）的全局感受野经门控融合互补，在 6 个算子学习基准上取得 SOTA，Navier-Stokes 相对误差比 Transolver 下降 22.3%。

**[Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)**

:   提出物理基线+数据驱动残差的混合建模框架，将海试功率曲线（螺旋桨定律 $P=cV^n$）作为基线，用 XGBoost/NN/PINN 学习残差修正，在稀疏数据区域显著提升外推稳定性和物理一致性。

**[SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels](svd-no_learning_pde_solution_operators_with_svd_integral_kernels.md)**

:   提出 SVD-NO，通过显式参数化积分核的奇异值分解（SVD）来构建神经算子，在保持高表达力的同时实现 $O(ndL)$ 的线性计算复杂度，在 5 个 PDE 基准上达到新 SOTA。

**[Towards a Foundation Model for Partial Differential Equations Across Physics Domains](towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)**

:   提出 PDE-FM，一个结合空间-频谱双模态 tokenization、FiLM 物理调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准 12 个异构物理域数据集上平均降低 VRMSE 46%。

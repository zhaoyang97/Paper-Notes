---
title: >-
  ECCV2026 其他论文汇总 · 17篇论文解读
description: >-
  17篇ECCV2026的其他方向论文解读，涵盖持续学习、扩散模型、少样本学习、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "其他"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "扩散模型"
  - "少样本学习"
  - "对抗鲁棒"
item_list:
  - u: "articulat3d_reconstructing_articulated_digital_twins_from_monocular_videos_with_/"
    t: "Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints"
  - u: "fedlas_feature-modulated_bidirectional_label_smoothing_for_neural_network_calibr/"
    t: "FedLAS: Feature-Modulated Bidirectional Label Smoothing for Neural Network Calibration"
  - u: "geometric_gradient_rectification_for_safe_open-set_semi-supervised_learning/"
    t: "Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning"
  - u: "geometry-anchored_transport_framework_for_exemplar-free_class-incremental_learni/"
    t: "Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning"
  - u: "hybrid_event_frame_sensors_modeling_calibration_simulation/"
    t: "Hybrid Event–Frame Sensors: Modeling, Calibration, and Simulation"
  - u: "intrinsically_stable_spiking_neural_networks_overcoming_the_performance_barrier_/"
    t: "Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization"
  - u: "lie_group_diffusion_models_for_hardware_aware_quantum_circuit_synthesis/"
    t: "Lie Group Diffusion Models for Hardware-Aware Quantum Circuit Synthesis"
  - u: "lost_in_the_tail_addressing_geographic_imbalance_in_urban_visual_place_recogniti/"
    t: "Lost in the Tail: Addressing Geographic Imbalance in Urban Visual Place Recognition"
  - u: "match-any-events_zero-shot_motion-robust_feature_matching_across_wide_baselines_/"
    t: "Match-Any-Events: Zero-Shot Motion-Robust Feature Matching Across Wide Baselines for Event Cameras"
  - u: "nurbs_splatting_a_unified_differentiable_rendering_framework_for_vector_graphics/"
    t: "NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics"
  - u: "optically_dense_nanowire_metamaterials_transparent_polarization/"
    t: "光密纳米线超材料：多重散射下的偏振保持"
  - u: "pointer-cad_v2_plan-then-construct_cad_generation_with_dimension-aware_parametri/"
    t: "Pointer-CAD v2: Plan-Then-Construct CAD Generation with Dimension-Aware Parametric Precision"
  - u: "recovery_operators_in_quasi_nelson_logic_the_prelinear_case/"
    t: "Recovery operators in quasi-Nelson logic: the prelinear case"
  - u: "seeing_touch_from_motion_a_unified_modality-aware_visuo-tactile_policy_with_tact/"
    t: "Seeing Touch from Motion: A Unified Modality-Aware Visuo-Tactile Policy with Tactile Motion Correlation"
  - u: "spectral_gating_via_damped_oscillations_for_adaptive_implicit_neural_representat/"
    t: "Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations"
  - u: "steerable_isotropic_vector-valued_anisotropic_radial_basis_functions/"
    t: "Belief Contraction in Dynamic Epistemic Logic"
  - u: "vector_scaffolding_inter-scale_orchestration_for_differentiable_image_vectorizat/"
    t: "Vector Scaffolding: Inter-Scale Orchestration for Differentiable Image Vectorization"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🎞️ ECCV2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (105)](../../CVPR2026/others/index.md) · [🔬 ICLR2026 (115)](../../ICLR2026/others/index.md) · [💬 ACL2026 (3)](../../ACL2026/others/index.md) · [🧪 ICML2026 (70)](../../ICML2026/others/index.md) · [🤖 AAAI2026 (117)](../../AAAI2026/others/index.md) · [🧠 NeurIPS2025 (121)](../../NeurIPS2025/others/index.md)

**[Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints](articulat3d_reconstructing_articulated_digital_twins_from_monocular_videos_with_.md)**

:   Articulat3D 提出一个两阶段优化框架，从单目随意拍摄视频中重建铰接物体的可交互数字孪生：第一阶段用低维运动基（motion bases）对 3D 点轨迹做结构化部分级运动分解，第二阶段用显式运动学基元（旋转/平移关节的轴、枢轴、逐帧标量）约束运动使其满足物理刚体规律，在合成和真实数据上均达到 SOTA 精度且无需静态预扫描。

**[FedLAS: Feature-Modulated Bidirectional Label Smoothing for Neural Network Calibration](fedlas_feature-modulated_bidirectional_label_smoothing_for_neural_network_calibr.md)**

:   提出 FeDLaS（Feature-modulated Bidirectional Label Smoothing），利用隐藏层特征 L1 范数作为置信度代理以绕过 softmax 饱和缺陷，搭配双向校准门控实时判断每个样本的过置信/欠置信状态，逐样本动态调制标签平滑强度，在保持 Top-1 精度的前提下显著降低 ECE 和 AECE。

**[Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning](geometric_gradient_rectification_for_safe_open-set_semi-supervised_learning.md)**

:   本文提出 GGR（Geometric Gradient Rectification），一种即插即用的梯度空间校正框架：以有监督梯度为锚点，将与之冲突的无监督辅助梯度投影到安全半空间内，使辅助更新在所选参数块上与有监督方向保持一阶非对抗，从而在不依赖精确 OOD 检测的前提下同时提升闭集泛化与开集鲁棒性。

**[Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning](geometry-anchored_transport_framework_for_exemplar-free_class-incremental_learni.md)**

:   本文提出 Geometry-Anchored Transport Framework，将特征传输从解耦的后处理范式转变为训练时的几何约束，通过解析几何锚点（Sylvester 方程闭解）抑制宏观各向异性漂移，配合拓扑感知演化约束局部流形退化，在无样本约束下稳定地传输旧类高斯统计量以实现可靠 Mahalanobis 评估。

**[Hybrid Event–Frame Sensors: Modeling, Calibration, and Simulation](hybrid_event_frame_sensors_modeling_calibration_simulation.md)**

:   本文首次提出混合事件-帧传感器的统一统计噪声模型、标定流程和仿真器 H-ESIM，通过 APS 与 EVS 共享光度信号的联合建模，从真实传感器数据中估计噪声参数，并生成与硬件噪声统计一致的 RAW 帧和事件流，在视频帧插值和去模糊任务上验证了仿真到真实域的迁移有效性。

**[Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization](intrinsically_stable_spiking_neural_networks_overcoming_the_performance_barrier_.md)**

:   IS-SNN 通过拓扑感知的权重标准化和改进的残差连接，在完全移除激活归一化层（Batch Normalization）的情况下，使深度脉冲神经网络的发放率保持稳定，训练完成后将标准化操作折叠进静态权重，推理时零归一化开销，在 ImageNet 上达到 68.05% 精度且 FPGA LUT 资源消耗降低 96.4%。

**[Lie Group Diffusion Models for Hardware-Aware Quantum Circuit Synthesis](lie_group_diffusion_models_for_hardware_aware_quantum_circuit_synthesis.md)**

:   提出了一种基于SU(2)李群流形上的扩散模型进行硬件感知量子电路综合的方法，将量子电路设计拆分为离散的电路骨架选择（骨架选择器）与连续的单量子比特门参数生成（条件扩散先验）两部分，在哈密顿模拟任务上显著超越Haar随机采样和生成搜索基线。

**[Lost in the Tail: Addressing Geographic Imbalance in Urban Visual Place Recognition](lost_in_the_tail_addressing_geographic_imbalance_in_urban_visual_place_recogniti.md)**

:   本文发现城市级视觉位置识别数据集存在严重的长尾地理分布问题（头尾类样本比达 300:1），提出 DAPR 框架——训练阶段用 Low-visit Bias 损失逆频率加权并校准分类器偏差，推理阶段用特征函数距离在频域实现分布感知的重排序，在 SF-XL 基准上超越此前最佳混合管线 18.3% R@1。

**[Match-Any-Events: Zero-Shot Motion-Robust Feature Matching Across Wide Baselines for Event Cameras](match-any-events_zero-shot_motion-robust_feature_matching_across_wide_baselines_.md)**

:   本文提出首个实现零样本跨数据集泛化的事件相机宽基线特征匹配模型，通过分离式时空注意力骨干（TAg）和稀疏感知Token自适应剪枝（SETS）实现高效多时间尺度事件特征学习，结合大规模合成（E-MegaDepth）与真实（ECM）宽基线数据集训练，在多个基准上相比此前最佳方法提升37.7%。

**[NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics](nurbs_splatting_a_unified_differentiable_rendering_framework_for_vector_graphics.md)**

:   本文提出NURBS Splatting，通过将NURBS曲线沿弧长自适应采样为各向同性高斯核并使用基于tile的高斯溅射光栅化器进行可微渲染，首次将有理权重和非均匀节点向量纳入2D图像空间端到端优化，在书法重建、分层图像矢量化和单笔画图像抽象等任务上显著超越基于多项式基元的现有方法。

**[光密纳米线超材料：多重散射下的偏振保持](optically_dense_nanowire_metamaterials_transparent_polarization.md)**

:   本文通过实验发现，由定向排列聚合物纳米线构成的各向异性超材料在光学稠密（多重散射）条件下仍然能够保持输入线偏振的偏振态，并利用柱面米氏散射矩阵的对角化性质揭示了「偏振不旋转」的物理机制：对于圆柱散射体，平行与垂直电场分量在散射过程中独立保持相位关系，当两种偏振的散射截面相等时输出偏振精确跟踪输入偏振。

**[Pointer-CAD v2: Plan-Then-Construct CAD Generation with Dimension-Aware Parametric Precision](pointer-cad_v2_plan-then-construct_cad_generation_with_dimension-aware_parametri.md)**

:   Pointer-CAD v2 提出 Plan-Then-Construct 框架，将 CAD 生成解耦为「先规划维度参数、再通过指针检索参数构造几何」两个阶段，彻底消除传统命令序列方法中数值参数量化带来的精度损失，在顶点/边/面三级几何精度指标上大幅超越所有基线。

**[Recovery operators in quasi-Nelson logic: the prelinear case](recovery_operators_in_quasi_nelson_logic_the_prelinear_case.md)**

:   本文将经典的一致性和未定性复原算子推广到准Nelson逻辑的预线性情形，证明在非对合条件下（预线性弱幂零极小代数）所有已有关于对合剩余格上的LFIs/LFUs的代数与逻辑结论均可重新建立。

**[Seeing Touch from Motion: A Unified Modality-Aware Visuo-Tactile Policy with Tactile Motion Correlation](seeing_touch_from_motion_a_unified_modality-aware_visuo-tactile_policy_with_tact.md)**

:   针对光学触觉传感器难以分辨细粒度接触状态的老问题，本文发现「瞬时运动」与「累积运动」的点积能显式区分「正在接触 / 稳定接触 / 正在松开」等状态，据此提出触觉运动相关性表征 TMC，并用 Mixture-of-Transformers 把它作为独立模态与视觉、原始触觉、本体状态融合，在四个真实接触密集操作任务上刷出更高成功率。

**[Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations](spectral_gating_via_damped_oscillations_for_adaptive_implicit_neural_representat.md)**

:   本文从受迫阻尼谐振子的稳态响应出发，推导出一种物理驱动的激活函数 FDHO，将激活幅度与频率通过二阶传递函数耦合，产生隐式频谱门控机制：阻尼因子梯度的自交互项始终关闭频谱门，而信号项仅在目标存在相干频率分量时打开频谱门，从而实现自动区分信号与噪声、无需任何显式正则化或任务特化超参数调优，在信号拟合、去噪、CT 重建、超分辨率等任务上全面达到 SOTA 或竞品水平。

**[Belief Contraction in Dynamic Epistemic Logic](steerable_isotropic_vector-valued_anisotropic_radial_basis_functions.md)**

:   本文在标准Kripke模型上定义了一种基于加边（而非删边）的信念收缩操作，用于建模对冲公开宣告导致的信念撤销，并给出了完整的公理化系统HPAL及其推广到私有宣告场景的广义动态认知逻辑GDEL。

**[Vector Scaffolding: Inter-Scale Orchestration for Differentiable Image Vectorization](vector_scaffolding_inter-scale_orchestration_for_differentiable_image_vectorizat.md)**

:   提出 Vector Scaffolding 分层优化框架，通过 Interior Gradient Aggregation 修复区域-边界梯度失衡、Progressive Stratification 按自然图像幂律由粗到细密化曲线、Rapid Inflation Scheduling 放大 50 倍学习率实现 2.5× 加速与 ~1.4dB PSNR 提升，将可微向量化从"平铺像素匹配"重构为"结构化拓扑构建"。

---
title: >-
  ICML2025 3D视觉论文汇总 · 17篇论文解读
description: >-
  17篇ICML2025的 3D 视觉方向论文解读，涵盖扩散模型、3D 高斯渲染、对齐/RLHF、对抗鲁棒、语义分割等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "3D 视觉"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "3D 高斯渲染"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "语义分割"
item_list:
  - u: "adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz/"
    t: "ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization"
  - u: "d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua/"
    t: "D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples"
  - u: "diverse_prototypical_ensembles_improve_robustness_to_subpopulation_shift/"
    t: "Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift"
  - u: "flowdrag_3d-aware_drag-based_image_editing_with_mesh-guided_deformation_vector_f/"
    t: "FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields"
  - u: "freemesh_boosting_mesh_generation_with_coordinates_merging/"
    t: "FreeMesh: Boosting Mesh Generation with Coordinates Merging"
  - u: "gaprompt_geometry-aware_point_cloud_prompt_for_3d_vision_model/"
    t: "GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model"
  - u: "high_dynamic_range_novel_view_synthesis_with_single_exposure/"
    t: "High Dynamic Range Novel View Synthesis with Single Exposure"
  - u: "of_mice_and_machines_a_comparison_of_learning_between_real_world_mice_and_rl_age/"
    t: "Of Mice and Machines: A Comparison of Learning Between Real World Mice and RL Agents"
  - u: "physicsnerf_physics-guided_3d_reconstruction_from_sparse_views/"
    t: "PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views"
  - u: "probabilistic_interactive_3d_segmentation_with_hierarchical_neural_processes/"
    t: "Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes"
  - u: "refersplat_referring_segmentation_in_3d_gaussian_splatting/"
    t: "ReferSplat: Referring Segmentation in 3D Gaussian Splatting"
  - u: "se3-equivariant_diffusion_policy_in_spherical_fourier_space/"
    t: "SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space"
  - u: "symmetry-robust_3d_orientation_estimation/"
    t: "Symmetry-Robust 3D Orientation Estimation"
  - u: "tackling_view-dependent_semantics_in_3d_language_gaussian_splatting/"
    t: "LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting"
  - u: "the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode/"
    t: "The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training"
  - u: "thickness-aware_e3-equivariant_3d_mesh_neural_networks/"
    t: "Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks"
  - u: "vtgaussian-slam_rgbd_slam_for_large_scale_scenes_with_splatting_view-tied_3d_gau/"
    t: "VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🧪 ICML2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (751)](../../CVPR2026/3d_vision/index.md) · [🔬 ICLR2026 (194)](../../ICLR2026/3d_vision/index.md) · [🧪 ICML2026 (30)](../../ICML2026/3d_vision/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/3d_vision/index.md) · [🧠 NeurIPS2025 (116)](../../NeurIPS2025/3d_vision/index.md) · [📹 ICCV2025 (267)](../../ICCV2025/3d_vision/index.md)

🔥 **高频主题：** 扩散模型 ×3 · 3D 高斯渲染 ×3 · 对齐/RLHF ×2 · 对抗鲁棒 ×2 · 语义分割 ×2

**[ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)**

:   将DPO思想引入扩散式人体网格恢复(HMR)：训练HMR-Scorer评估预测质量，构建偏好数据集(winner/loser对)，用DPO微调基座扩散模型，无需3D标注即可提升in-the-wild图像上的HMR性能。

**[D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)**

:   本文提出 D-Fusion 方法，通过 mask 引导的自注意力融合（Self-Attention Fusion）构建视觉一致的偏好数据对并保留去噪轨迹，解决了 DPO 训练扩散模型时因视觉不一致导致效果受限的问题，在多种 RL 算法和 prompt 类型上显著提升了 prompt-image 对齐质量。

**[Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift](diverse_prototypical_ensembles_improve_robustness_to_subpopulation_shift.md)**

:   提出 Diversified Prototypical Ensemble (DPE)，用多个多样化的原型分类器替换标准线性分类头，通过显式（inter-prototype similarity loss）和隐式（bootstrap 采样）两种多样化策略，在不需要子群标注的情况下自适应发现子群决策边界，显著提升 worst-group accuracy。

**[FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields](flowdrag_3d-aware_drag-based_image_editing_with_mesh-guided_deformation_vector_f.md)**

:   提出 FlowDrag，从图像构建 3D 网格后利用渐进式 SR-ARAP 变形生成连续 2D 向量流场，将全局几何先验注入扩散模型的 motion supervision 过程，在 DragBench（MD=22.88）和新提出的 VFD-Bench（PSNR=18.55, 1-LPIPS=0.82, MD=28.23）上全面领先。

**[FreeMesh: Boosting Mesh Generation with Coordinates Merging](freemesh_boosting_mesh_generation_with_coordinates_merging.md)**

:   提出 Per-Token-Mesh-Entropy（PTME）度量来免训练评估网格tokenizer质量，并引入从NLP借鉴的 Rearrange & Merge Coordinates（RMC）坐标合并技术，在 MeshXL/MeshAnythingV2/EdgeRunner 三种tokenizer上实现最高21.2%的压缩率、显著增加可生成面片数和几何细节保留。

**[GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model](gaprompt_geometry-aware_point_cloud_prompt_for_3d_vision_model.md)**

:   提出 GAPrompt，针对预训练 3D 视觉模型的几何感知 PEFT 方法，通过可学习点云提示 (Point Prompt)、点偏移提示器 (Point Shift Prompter) 和提示传播 (Prompt Propagation) 三个模块协同利用点云几何信息，仅训练 2.19% 参数即可匹配甚至超越全量微调。

**[High Dynamic Range Novel View Synthesis with Single Exposure](high_dynamic_range_novel_view_synthesis_with_single_exposure.md)**

:   首次提出仅使用单曝光LDR图像进行HDR新视角合成（HDR-NVS）的问题设定，并设计了一个基于相机成像原理的元算法框架Mono-HDR-3D，通过LDR→HDR颜色转换器（L2H-CC）和HDR→LDR闭环转换器（H2L-CC）实现无HDR监督下的HDR场景建模。

**[Of Mice and Machines: A Comparison of Learning Between Real World Mice and RL Agents](of_mice_and_machines_a_comparison_of_learning_between_real_world_mice_and_rl_age.md)**

:   系统比较真实小鼠与RL智能体在捕食者-猎物迷宫中的行为差异，发现RL缺乏自我保护本能，提出创伤启发安全缓冲（TISB）和方差惩罚TD学习（VP-TDMPC-2）两种机制，将智能体与小鼠的状态访问重叠率从20.9%提升至86.1%。

**[PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views](physicsnerf_physics-guided_3d_reconstruction_from_sparse_views.md)**

:   PhysicsNeRF 提出了一个基于物理先验的稀疏视角 NeRF 框架，通过深度排序、跨视角一致性、稀疏性正则和渐进训练四种互补约束，在仅 8 个视角下实现 21.4 dB 的 PSNR，并对稀疏视角下过拟合的本质进行了深入的理论分析。

**[Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes](probabilistic_interactive_3d_segmentation_with_hierarchical_neural_processes.md)**

:   NPISeg3D提出了首个基于层次化神经过程（Hierarchical Neural Processes）的概率交互式3D分割框架，通过场景级和物体级双层潜变量结构以及概率原型调制器，在少量点击下实现了优于AGILE3D的分割精度，同时提供可靠的不确定性估计。

**[ReferSplat: Referring Segmentation in 3D Gaussian Splatting](refersplat_referring_segmentation_in_3d_gaussian_splatting.md)**

:   ReferSplat 提出了 Referring 3D Gaussian Splatting Segmentation（R3DGS）新任务，通过构建 3D Gaussian Referring Fields、位置感知跨模态交互模块和 Gaussian-Text 对比学习，实现了基于自然语言描述在 3DGS 场景中分割目标物体（包括遮挡/不可见物体），在新构建的 Ref-LERF 数据集和开放词汇分割基准上取得 SOTA。

**[SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)**

:   提出在球面 Fourier 空间中构建 SE(3) 等变扩散策略，利用球谐函数的等变性质使策略对输入场景的刚体变换保持等变，在机器人操作任务上实现更好的空间泛化。

**[Symmetry-Robust 3D Orientation Estimation](symmetry-robust_3d_orientation_estimation.md)**

:   提出一种对旋转对称性鲁棒的两阶段3D朝向估计流水线：第一阶段通过商回归（quotient regression）将朝向恢复到八面体对称群的等价类内，第二阶段通过分类器预测24个八面体翻转之一以完成精确复原，在ShapeNet上取得SOTA。

**[LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)**

:   提出LaGa方法，通过3D场景分解建立跨视角语义连接、用自适应聚类+双因子重加权构建视角聚合语义表示，解决3D语言高斯中被忽视的视角依赖语义问题，在LERF-OVS上3D mIoU达64.0%（+18.7%）。

**[The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode.md)**

:   揭示了 Transformer 中不同类型模块（Emb、QK、FFN、VO、Norm）存在显著且持久的**锐度差异**（sharpness disparity），并据此提出 Blockwise LR 策略，为低锐度模块分配更大学习率，在不损失稳定性的前提下实现 LLM 预训练近 **2× 加速**。

**[Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks](thickness-aware_e3-equivariant_3d_mesh_neural_networks.md)**

:   提出 T-EMNN，通过引入厚度感知的消息传递机制和基于 PCA 的数据驱动坐标系，在保持表面网格计算效率的同时建模对立面之间的厚度交互，实现 E(3)-等变/不变的节点级 3D 形变预测。

**[VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians](vtgaussian-slam_rgbd_slam_for_large_scale_scenes_with_splatting_view-tied_3d_gau.md)**

:   提出视图绑定3D高斯（View-Tied 3D Gaussians），将高斯绑定到深度像素上并简化为球形，大幅节省存储开销，配合仅优化最近视图相关高斯的tracking/mapping策略，实现面向大规模场景的可扩展RGBD SLAM系统。

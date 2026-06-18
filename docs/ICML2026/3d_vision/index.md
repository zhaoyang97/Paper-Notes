---
title: >-
  ICML2026 3D视觉论文汇总 · 24篇论文解读
description: >-
  24篇ICML2026的 3D 视觉方向论文解读，涵盖图像恢复、布局/合成、三维重建、点云等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "3D 视觉"
  - "论文解读"
  - "论文笔记"
  - "图像恢复"
  - "布局/合成"
  - "三维重建"
  - "点云"
item_list:
  - u: "avatar_learning_to_align_via_active_optimal_transport/"
    t: "AvAtar: Learning to Align via Active Optimal Transport"
  - u: "convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation/"
    t: "Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation"
  - u: "distilling_neuro-symbolic_programs_into_3d_multi-modal_llms/"
    t: "APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs"
  - u: "fast-sam3d_3dfy_anything_in_images_but_faster/"
    t: "Fast-SAM3D: 3Dfy Anything in Images but Faster"
  - u: "foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s/"
    t: "FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation"
  - u: "fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated/"
    t: "FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth"
  - u: "geodesic_flow_matching_for_denoising_high-dimensional_structured_representations/"
    t: "Geodesic Flow Matching for Denoising High-Dimensional Structured Representations"
  - u: "geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha/"
    t: "Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning"
  - u: "hoi-page_zero-shot_human-object_interaction_generation_with_part_affordance_guid/"
    t: "HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance"
  - u: "labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo/"
    t: "LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory"
  - u: "physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation/"
    t: "PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation"
  - u: "physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions/"
    t: "PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions"
  - u: "plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat/"
    t: "PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations"
  - u: "relaxflow_text-driven_amodal_3d_generation/"
    t: "RelaxFlow: Text-Driven Amodal 3D Generation"
  - u: "revisiting_photometric_ambiguity_for_accurate_gaussian-splatting_surface_reconst/"
    t: "Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction"
  - u: "simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou/"
    t: "SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising"
  - u: "smoothness_errors_in_dynamics_models_and_how_to_avoid_them/"
    t: "Smoothness Errors in Dynamics Models and How to Avoid Them"
  - u: "splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_/"
    t: "SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion"
  - u: "stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_/"
    t: "STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System"
  - u: "streaming_sliced_optimal_transport/"
    t: "Streaming Sliced Optimal Transport"
  - u: "svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa/"
    t: "SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding"
  - u: "the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d/"
    t: "The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity"
  - u: "tidegs_scalable_training_of_over_one_billion_3d_gaussian_splatting_primitives_vi/"
    t: "TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization"
  - u: "trust_it_or_not_evidential_uncertainty_for_feed-forward_3d_reconstruction_with_t/"
    t: "Trust3R: Evidential Uncertainty for Feed-Forward 3D Reconstruction"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🧪 ICML2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (626)](../../CVPR2026/3d_vision/index.md) · [🔬 ICLR2026 (62)](../../ICLR2026/3d_vision/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/3d_vision/index.md) · [🧠 NeurIPS2025 (116)](../../NeurIPS2025/3d_vision/index.md) · [📹 ICCV2025 (267)](../../ICCV2025/3d_vision/index.md) · [🧪 ICML2025 (17)](../../ICML2025/3d_vision/index.md)

🔥 **高频主题：** 图像恢复 ×2 · 布局/合成 ×2 · 三维重建 ×2 · 点云 ×2

**[AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)**

:   本文提出 AvAtar，一个基于最优传输（OT）的主动对齐框架，通过梯度传播量化候选查询对全局对齐结果的影响，并利用伴随状态法和共轭梯度法以线性复杂度高效求解，在网络对齐和跨域对齐任务上一致超越已有主动学习策略。

**[Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation.md)**

:   本文提出 CDOT（Convex Distance Operator Transport），通过把每个度量空间的距离矩阵和耦合一起"算子化"，用 $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$ 替代 FGW 中那个非凸的成对距离差平方，从而首次得到一个**对耦合 $\pi$ 严格凸**、同时仍然是合法伪度量、并具备有限样本风险界的异构空间对齐框架。

**[APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs](distilling_neuro-symbolic_programs_into_3d_multi-modal_llms.md)**

:   本文提出 APEIRIA，把神经符号 3D 概念学习器的程序执行轨迹蒸馏成 3D MLLM 的自然语言 chain-of-thought，再通过 GRPO 强化学习把这种推理模式推广到开放词汇与深层嵌套指令，在 ScanRefer、Multi3DRefer、SQA3D、Scan2Cap 上同时超越传统 NS3D 方法和当前最强的 3D MLLM，并保留了符号系统的可解释性与模块可替换性。

**[Fast-SAM3D: 3Dfy Anything in Images but Faster](fast-sam3d_3dfy_anything_in_images_but_faster.md)**

:   针对 SAM3D 单视图 3D 重建模型推理太慢的问题，本文做了第一份模块级时延剖析，发现性能瓶颈来自三种异质性（形状/布局动力学差异、纹理稀疏性、几何谱差异），并据此提出训练无关的 Fast-SAM3D 框架，用模态感知步缓存、时空 Token 雕刻与谱感知 Token 聚合三件套，在几乎不损质量的前提下把对象级速度推到 2.67×，重建 F-Score 反而从 92.34 微升到 92.59。

**[FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)**

:   本文提出 FoundObj，把 2D/3D 自监督基础模型（DINOv2 + TRELLIS）当作奖励器，用一个"超点合并 + PPO"的 RL 代理在无任何场景级人工标注下完成复杂室内场景的多类 3D 物体分割，在 ScanNet/S3DIS/ScanNet200 上将无监督 SOTA 的 AP 从 19.6 提到 24.2。

**[FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)**

:   本文把人类“先扫一眼再逐块细看”的观察过程抽象为 Focus-Sweep 两阶段范式，用 Mamba 替换 Transformer 做图像-点云交互，并用强化学习动态决定每个尺度上的交互层数，在 RGB-D Scenes V2 和 7-Scenes 上拿到 I2P 配准的 SOTA。

**[Geodesic Flow Matching for Denoising High-Dimensional Structured Representations](geodesic_flow_matching_for_denoising_high-dimensional_structured_representations.md)**

:   针对 Vector Symbolic Architecture 里 Spatial Semantic Pointer 这种"被嵌进单位超球面 Clifford 超环面"的高维结构化表示，作者指出标准 Flow Matching 的欧氏直线插值会从球面内部"穿心而过"导致幅值塌缩、相位毁掉，于是用 Log/Exp 映射把流约束在球面上做 **Geodesic Flow Matching (GFM)**，在脉冲神经 SLAM 上把路径误差降低 72%，并让 1500 神经元的路径积分器达到 2500 神经元 baseline 的精度。

**[Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning](geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha.md)**

:   本文提出 GODeform：把 2D 基础模型（DINOv3 类）特征"挂"到类别模板表面上做几何引导传播与跨视点融合，再用 Flow Matching 学一个从模板到目标的逐点形变场，从而在大形变、任意视角和未见类别上都能从单张图恢复 3D 形状，并直接服务于灵巧抓取迁移。

**[HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance](hoi-page_zero-shot_human-object_interaction_generation_with_part_affordance_guid.md)**

:   HOI-PAGE 让 LLM 先"想清楚"身体哪个部位该接触物体哪个部件，把推理结果写成一张「部件 affordance 图」(PAG)，再用它去驱动 3D 部件分割、视频扩散和优化求解，从而在零样本、零 4D 训练数据的条件下生成可处理"多人单物 / 单人多物"等复杂场景的 4D 人-物交互序列。

**[LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)**

:   LabBuilder 把自由文本的实验描述编译成"资产-化学协议"，再用层级化生成 + 几何/化学多目标优化 + 导航修复，产出既视觉合理、又能让机器人真正跑通实验流程的 3D 化学实验室布局。

**[PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation.md)**

:   PhyScene3D 把 3D 桌面场景生成重塑成"人类构造式"的层次化序列规划：用 Cognitive Topological Reasoning Chain (CTRC) 把场景图线性化为基于 AABB 的锚点序列，再用 Physics-Aware Denoising Alignment (PADA) 把可微分 SDF 物理引擎嵌入 VLM 训练循环，使模型生成的场景在物理合理性上反超人工标注训练数据（场景级碰撞率从 81.5% 降到 41.6%，资产级降到 3.86%）。

**[PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)**

:   本文提出 PhysHanDI，把 MANO 手模型和 Spring-Mass 软体模型耦合起来，用稠密手网格驱动可变形物体的物理仿真，并反向利用物体仿真去精化手的重建，在稀疏视角 RGB-D 视频上同时拿到了手和软物的稠密 3D 重建 SOTA。

**[PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations](plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat.md)**

:   PLAID 提出一套面向异构物理仿真数据的统一数据模型与开源库，配套发布 6 个覆盖结构力学和 CFD 的工业级数据集与可复现基准，把"变网格、变拓扑、变维度"的真实仿真数据真正变成机器学习社区可用的标准化 benchmark。

**[RelaxFlow: Text-Driven Amodal 3D Generation](relaxflow_text-driven_amodal_3d_generation.md)**

:   RelaxFlow 把"用文字补全被遮挡 3D 物体"形式化为一个**双目标控制粒度解耦**问题，提出训练免调的双分支推理框架——观察分支保持像素级硬约束、语义先验分支用"多先验共识 + 注意力 logit 高斯模糊"实现低通松弛——并从理论上证明这一松弛等价于对生成向量场做低通滤波，从而在 SAM3D / TRELLIS 等 SOTA 上把 Point-FID 从 100.38 降到 81.11。

**[Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction](revisiting_photometric_ambiguity_for_accurate_gaussian-splatting_surface_reconst.md)**

:   AmbiSuR 把 Gaussian Splatting 的两类内生光度歧义（基元边缘外溢、像素混合欠约束）显式建模并用截断 + 射线-颜色一致性消歧，再借高阶球谐系数作"自指示器"找出歧义高风险基元并做无定形局部先验正则，在 DTU 上把平均 Chamfer 距离降到 0.46，超过此前最优 GeoSVR (0.47)。

**[SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou.md)**

:   SIMPC 提出在**同一个噪声点**上沿去噪向量做"对称延伸"得到一个位于曲面另一侧的镜像点，再用 Mirror-Point Consistency Loss 强制两点的去噪目标重合，从而把无监督点云去噪从"在多份噪声变体间找统计对应"换成"在单点内部找确定性几何对应"，在 PUNet/PCNet 合成数据和 Paris-Rue-Madame / Kinect 真实扫描上全面超越无监督 SOTA，并击败若干有监督方法。

**[Smoothness Errors in Dynamics Models and How to Avoid Them](smoothness_errors_in_dynamics_models_and_how_to_avoid_them.md)**

:   作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。

**[SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)**

:   本文指出多模态点云补全里"硬投影把 3D 点直接打到 2D 网格"会让支持集 Lebesgue 测度为零、梯度被 Dirac delta 截断（称为 Cross-Modal Entropy Collapse），用可微 Gaussian Soft Splatting 把硬投影换成连续密度估计，搭配 EdgeConv 局部 + Transformer 全局的混合编码器和全局-局部解码器，在 PCN/ShapeNet-55/34 拿到 SOTA，并用 KITTI 上的 counter-factual 评估证明 baseline 实际是退化的"单模态模板检索器"。

**[STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)**

:   STABLE 把"任务指令→可仿真桌面场景"拆成 LLM-based Semantic Reasoner（出粗布局）和 flow-matching + SDF 损失的 Physics Corrector（修位姿），并让两者按 task-critical → background 三阶段交替迭代，最终在 MesaTask-10K 上把物体碰撞数压到 0、任务对齐 AwS 拉到 99.0%。

**[Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)**

:   Stream-SW 是首个能在"样本流"上估计 sliced Wasserstein 距离的算法：每个一维投影上用 KLL/quantile sketch 维护近似分位函数，把 1D Wasserstein 的闭式积分变成可流式更新的估计量，空间复杂度对样本数仅对数级，从而把 SOT 带入 IoT / 边缘设备等"看一次就丢掉"的场景。

**[SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)**

:   SVL 用「3D-图像-文本」三模态对比预训练给脉冲神经网络（SNN）注入开放世界理解能力，并通过把文本编码器"重参数化"为一组分类权重，让推理阶段完全脱离文本塔、保持纯脉冲驱动，在 ModelNet40 零样本分类上达到 85.4% 同时能耗仅为同档 ANN 方法的 0.5%–11%。

**[The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)**

:   本文揭示 LLM 中"注意力汇聚到第一个 token"的结构性根源 —— 因果掩码下首 token 缺乏 value 聚合导致维度方差差异,被 FFN 中的 super neurons 选择性放大形成维度极度悬殊,最终锁死 QK 投影迫使形成 attention sink;并据此提出 head-wise RMSNorm 在预训练阶段从根上抑制 sink。

**[TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization](tidegs_scalable_training_of_over_one_billion_3d_gaussian_splatting_primitives_vi.md)**

:   TideGS 把 3DGS 的参数表搬到 SSD 上，按"块"虚拟化并以 GPU VRAM 作为视锥可见工作集的缓存，配合三级异步流水线和轨迹自适应差分流式传输，在单张 24 GB GPU 上首次把可训练的高斯数量从约 11M（原生 3DGS）/ 105M（CLM）推到 **超过 10 亿**，且大场景重建质量优于所有评测的单卡基线。

**[Trust3R: Evidential Uncertainty for Feed-Forward 3D Reconstruction](trust_it_or_not_evidential_uncertainty_for_feed-forward_3d_reconstruction_with_t.md)**

:   Trust3R 为 MASt3R 等 feed-forward 3D 重建模型引入概率化证据学习框架，用 Normal-Inverse-Wishart 先验为每个 3D 点预测闭形式多元 Student-t 分布，取代启发式置信度，单遍前向推理就能输出概率可解释的逐点不确定性，并在 ScanNet++ 上 AURC 降低 25%、AUSE 降低 41%。

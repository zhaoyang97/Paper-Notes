<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📹 ICCV2025** · 共 **9** 篇

**[3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)**

:   提出首个全面的3D空间推理基准3DSRBench，包含2,772个人工标注的VQA对（12种问题类型），通过平衡数据分布和新型FlipEval策略实现鲁棒评估，揭示SOTA LMM（包括GPT-4o、Gemini）在3D空间推理上远落后于人类水平（≈52% vs 95.7%），且在非常规视角下性能显著退化。

**[A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)**

:   发现GCD中未标注数据（尤其是未知类别）的ViT注意力会分散到背景区域（distracted attention），提出Attention Focusing（AF）模块通过多尺度token重要性度量+自适应剪枝来纠正注意力，作为即插即用模块在SimGCD上最高带来15.4%的性能提升。

**[A Real-world Display Inverse Rendering Dataset](a_realworld_display_inverse_rendering_dataset.md)**

:   构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个物体的OLAT（逐像素点亮）采集图像、偏振信息和GT几何，并提出简单有效的基线方法（基于Cook-Torrance BRDF的可微渲染优化），在150秒内超越现有逆渲染方法。

**[ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)**

:   将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。

**[AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)**

:   本文提出AdaptiveAE，利用深度强化学习将HDR曝光包围拍摄建模为马尔可夫决策过程（MDP），同时优化ISO和快门速度的组合，在用户定义的时间预算内自适应地为动态场景选择最优曝光参数，在HDRV数据集上达到PSNR 39.70，比之前最好的方法Hasinoff et al. (37.59) 高出2.1 dB。

**[AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)**

:   将多曝光HDR重建从MAP估计视角建模，通过空间对应先验将问题分解为对齐和融合两个交替子问题，再展开为端到端可训练的AFUNet（含SAM空间对齐+CFM通道融合+DCM数据一致性模块），在三个HDR基准上取得SOTA，PSNR-μ达44.91dB（Kalantari数据集）。

**[Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)**

:   首次将自回归（AR）模型引入多视角图像生成任务，通过逐视角生成利用所有前序视角信息来增强远距离视角间的一致性，同时设计了统一的多模态条件注入架构和Shuffle Views数据增强策略，使单一模型可同时处理文本/图像/几何形状条件。

**[C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)**

:   提出C4D框架，通过在DUSt3R的3D pointmap预测基础上联合捕获双重时序对应(短时光流+动态感知长时点跟踪DynPT)，生成运动掩码分离动静区域，并引入相机运动对齐/相机轨迹平滑/点轨迹平滑三个优化目标，将现有3D重建范式升级为完整4D重建(逐帧点云+相机参数+2D/3D轨迹)，在深度/位姿/跟踪多个下游任务上达competitive性能。

**[Despite Exploring Contrastive Deep Skeletonpointcloudimutext](despite_exploring_contrastive_deep_skeletonpointcloudimutext.md)**

:   提出 DeSPITE，一个将 LiDAR 点云、骨架姿态、IMU 信号和文本四种模态对齐到联合嵌入空间的对比学习框架，首次以 LiDAR（而非 RGB）作为核心视觉模态，实现了跨模态匹配/检索等此前不可能的任务，同时作为有效的 HAR 预训练策略在 MSR-Action3D 和 HMPEAR 上取得 SOTA。

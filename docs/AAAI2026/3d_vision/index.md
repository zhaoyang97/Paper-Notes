<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🤖 AAAI2026** · 共 **7** 篇

**[3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition](3d-anc_adaptive_neural_collapse_for_robust_3d_point_cloud_re.md)**

:   将Neural Collapse(NC)机制引入3D点云对抗鲁棒性，用固定的ETF分类头+自适应训练框架(RBL+FDL)构建解耦的特征空间，在ModelNet40上将DGCNN的对抗准确率从27.2%提升到80.9%，超出最佳baseline 34个点。

**[4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)**

:   提出4DSTR框架，通过基于Mamba的时序关联校正（修正高斯点的尺度和旋转）以及逐帧自适应稠密化与裁剪策略，显著提升4D高斯生成的时空一致性和对快速时序变化的适应能力。

**[Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)**

:   提出 Uni-Adapter，一种面向3D视觉-语言基础模型(VLFM)的无训练在线测试时适应框架，通过基于聚类的动态原型缓存和图正则化标签平滑来应对分布偏移，在多个3D损坏基准上取得SOTA。

**[AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](anchords_anchoring_dynamic_sources_for_semantically_consiste.md)**

:   Optimization-based text-to-3D methods distill guidance from 2D generative models via Score Distillation Sampling (SDS), but implicitly treat this guidance as static.

**[AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)**

:   提出 AnchorHOI，通过锚点NeRF和锚点关键点两种中间桥梁，分别从图像/视频扩散模型中蒸馏交互先验和运动先验，实现零样本的文本驱动4D人物-物体交互生成，在静态3D和动态4D HOI生成上均超越已有方法。

**[Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)**

:   提出一个集成框架实现3D高斯溅射(3DGS)的任意倍率超分辨率渲染，通过尺度感知渲染、生成先验引导优化和渐进超分机制，用单个3D模型支持整数和非整数倍率的HR渲染，PSNR提升6.59dB同时保持85 FPS实时速度。

**[FoundationSLAM: 释放深度基础模型在端到端稠密视觉SLAM中的潜力](foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)**

:   将深度基础模型的几何先验注入光流式SLAM系统，通过混合光流网络、双向一致BA层和可靠性感知精炼三个模块形成闭环，在TUM/EuRoC/7Scenes/ETH3D四大数据集取得SOTA轨迹精度和稠密重建质量，18 FPS实时运行。

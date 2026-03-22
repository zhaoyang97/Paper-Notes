<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🎞️ ECCV2024** · 共 **20** 篇

**[3D Congealing: 3D-Aware Image Alignment in the Wild](3d_congealing_3d-aware_image_alignment_in_the_wild.md)**

:   3D Congealing将一组语义相似的无标注互联网图像对齐到共享的3D canonical空间，通过结合预训练扩散模型的SDS指导获得3D形状 + DINO语义特征匹配估计位姿和坐标映射，无需模板、位姿标注或相机参数。

**[3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)**

:   提出HORSE框架，通过从野外视频中提取多视角2D mask监督（以手部姿态作为物体姿态代理）和从合成3D形状集合中学习2D切片对抗形状先验，训练occupancy网络从单张RGB图像重建手持物体3D形状，在不使用任何真实世界3D标注的情况下，在MOW数据集上超越使用3D监督的方法11.6%。

**[3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)**

:   HVTrack首次探索高时间变化场景下的3D单目标跟踪，通过相对位姿感知记忆模块(RPM)、基础-扩展特征交叉注意力(BEA)和上下文点引导自注意力(CPA)三个模块，分别解决点云形状剧变、相似物体干扰和背景噪声问题，在KITTI-HV 5帧间隔下比SOTA提升11.3%/15.7% Success/Precision。

**[3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting](3igs_factorised_tensorial_illumination_for_3d_gaussian_splatting.md)**

:   3iGS 用基于张量分解的连续入射光照场替代 3DGS 中每个高斯体独立优化的球谐系数，结合可学习 BRDF 特征和轻量神经渲染器来建模出射辐射，在保持实时渲染速度的同时显著提升了镜面反射等视角依赖效果的渲染质量。

**[3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)**

:   提出了一种无需训练的3D物体部件分割方法3-By-2，利用扩散模型(DIFT)的2D语义对应关系从已标注2D数据集或少量3D标注对象中迁移部件标签到3D，在zero-shot和few-shot设置下均达到SOTA。

**[6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)**

:   提出6DGS，通过反转3DGS渲染流程——从椭球体表面均匀发射光线（Ellicell），利用注意力机制将光线与目标图像像素绑定，再用加权最小二乘闭式求解相机位姿，无需迭代和初始位姿，在真实场景上旋转精度提升12%、平移精度提升22%，达到15fps近实时性能。

**[A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)**

:   将3DGS中的位置和旋转参数建模为时间的函数（位置用Fourier逼近、旋转用线性逼近），使动态场景的存储复杂度从O(TN)降低到O(LN)，在D-NeRF/DyNeRF/HyperNeRF三个数据集上实现了与NeRF方法匹敌的渲染质量，同时保持118+ FPS的实时渲染速度。

**[Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)**

:   通过使用条件 logistic 函数解析近似高斯信号在像素窗口上的积分，替代 3DGS 的像素中心点采样，实现无混叠的 3D 高斯泼溅，在多尺度渲染上超越 Mip-Splatting。

**[AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)**

:   提出 AnimatableDreamer，通过 Canonical Score Distillation (CSD) 技术，从单目视频提取骨骼和运动后生成文本引导的可动画化 3D 非刚体模型，在生成质量和时序一致性上全面超越现有方法。

**[CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering](caesarnerf_calibrated_semantic_representation_for_few-shot_generalizable_neural_.md)**

:   提出 CaesarNeRF，在可泛化 NeRF（GNT）基础上引入场景级语义表征，通过相机位姿校准（特征旋转对齐到目标视角）和序列细化（跨 Transformer 层逐步更新全局特征），在 1-view 设置下 PSNR 比 GNT 提升 1.74dB（LLFF），且可即插即用地增强 IBRNet、MatchNeRF 等其他基线。

**[D-SCo: Dual-Stream Conditional Diffusion for Monocular Hand-Held Object Reconstruction](d-sco_dual-stream_conditional_diffusion_for_monocular_hand-held_object_reconstru.md)**

:   提出双流条件扩散模型 D-SCo 从单张 RGB 图像重建手持物体点云，通过统一手-物语义嵌入和手关节几何嵌入两个分支分别提供语义和几何先验，配合手约束质心固定策略稳定扩散过程，在 ObMan 上 F-5 达 0.61（超 DDF-HO 10.9%），真实数据集 HO3D/MOW 上也大幅领先。

**[Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)**

:   利用微调的 Stable Diffusion + ControlNet 将粗糙 NeRF/3DGS 渲染结果转化为高质量伪观测图像，将稀疏输入视图增密 5-10 倍后重新训练，在 Hypersim/LLFF/ScanNet 等数据集上超越 FreeNeRF 等方法 1-2dB PSNR，训练速度比扩散正则化方法快约 10 倍。

**[DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)**

:   提出动态空间剪枝（DSP）策略，在多级 3D 检测器的解码器中逐级移除已检测到大物体区域的体素特征，使检测器能以高空间分辨率处理场景、大幅提升小目标检测精度（ScanNet 小目标 mAP@0.25 从 27.5% 提升到 44.8%），同时通过剪枝将显存降低为同分辨率方法的 1/5。

**[Gaussian Grouping: Segment and Edit Anything in 3D Scenes](gaussian_grouping_segment_and_edit_anything_in_3d_scenes.md)**

:   为 3D Gaussian Splatting 中的每个高斯学习 16 维 Identity Encoding 实现实例级分组，使用 SAM + DEVA 视频跟踪生成多视图一致的 2D 伪标签做监督，在 LERF-Mask 开放词汇分割上 mIoU 达 69-77%（超 LERF 2 倍+），全景分割超 Panoptic Lifting 4.9% mIoU 且 14× 更快，同时支持 3D 物体移除/修复/着色/风格迁移等多种编辑。

**[milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing](milliflow_scene_flow_estimation_on_mmwave_radar_point_cloud_for_human_motion_sen.md)**

:   提出首个毫米波雷达点云场景流估计方法 milliFlow，通过多尺度特征提取、全局聚合、GRU 时序传播和约束回归，在自建数据集上将 EPE3D 从次优 0.107m 降至 0.046m（cm 级精度），并展示场景流特征对人体活动识别（+7.9%）、人体部位解析（+3.6%）、人体追踪等下游任务的增强效果。

**[MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)**

:   将MVS的代价体深度估计与3D高斯溅射结合，通过混合渲染(splatting+volume rendering)提升泛化性，并提出基于多视图几何一致性的点云聚合策略，使per-scene优化仅需45秒就超越3D-GS的10分钟效果。

**[NOVUM: Neural Object Volumes for Robust Object Classification](novum_neural_object_volumes_for_robust_object_classification.md)**

:   提出 NOVUM 架构，为每个物体类别维护一个由 3D 高斯组成的神经体积表征，通过将图像特征与各类别的高斯特征匹配实现分类，在遮挡/损坏/真实 OOD 场景下相比 ResNet/ViT/Swin 等标准架构分类准确率提升 6-33%，同时支持 3D 位姿估计和可解释性可视化。

**[PointLLM: Empowering Large Language Models to Understand Point Clouds](pointllm_empowering_large_language_models_to_understand_point_clouds.md)**

:   将点云编码器（Point-BERT）通过 MLP 投影层对接 LLaMA 大语言模型，构建 PointLLM；利用 730K 指令数据（660K 简述 + 70K 复杂指令）两阶段训练后，在 3D 物体分类上达到 53.4% 生成式准确率（超越 LLaVA-13B 的 44.2%），在物体描述任务上人类评估胜率 55%（超越人工标注）。

**[Progressive Classifier and Feature Extractor Adaptation for Unsupervised Domain Adaptation on Point Clouds](progressive_classifier_and_feature_extractor_adaptation_for_unsupervised_domain_.md)**

:   提出 PCFEA 方法用于点云无监督域自适应，通过渐进构建从源域到目标域的中间域，在宏观层面用目标风格特征增强训练分类器（PTFA），微观层面引导特征提取器向中间域对齐（IDFA），在 PointDA-10 上均值准确率达 76.5%（超 SOTA +2.9%），GraspNetPC-10 上达 87.6%（超 SOTA +13.7%）。

**[When Do We Not Need Larger Vision Models?](when_do_we_not_need_larger_vision_models.md)**

:   提出 Scaling on Scales (S2) 策略：冻结小模型（如 ViT-B）在多个图像尺度上运行并拼接特征，无需增加参数即可在分类、分割、深度估计、MLLM 等任务上匹敌甚至超越大模型（ViT-H/G），并从理论和实验上论证了大模型学到的表征大部分可由多尺度小模型线性近似。

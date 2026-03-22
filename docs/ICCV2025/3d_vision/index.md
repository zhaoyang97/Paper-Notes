<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**📹 ICCV2025** · 共 **49** 篇

**[TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)**

:   提出TRAN-D，一种基于2D Gaussian Splatting的稀疏视角透明物体深度重建方法，通过分割引导的object-aware损失优化遮挡区域Gaussian分布，并利用物理仿真（MPM）实现物体移除后的场景动态更新，仅需单张图像即可完成场景刷新。

**[3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)**

:   提出基于3D高斯溅射的场景地图表示（3D Gaussian Map），结合开放集语义分组机制，为视觉-语言导航（VLN）构建兼顾几何结构与丰富语义的3D环境表示，并设计多层级动作预测策略（Multi-Level Action Prediction）融合多粒度空间-语义线索辅助导航决策。

**[3D Mesh Editing using Masked LRMs](3d_mesh_editing_using_masked_lrms.md)**

:   提出MaskedLRM，将3D形状编辑重构为条件重建问题——训练时随机生成3D遮挡物遮盖多视角输入，用一张干净条件视图引导被遮挡区域的补全；推理时用户定义编辑区域并提供单张编辑图像，模型在**<3秒单次前传**中完成3D网格编辑，比优化方法快2-10倍，能执行拓扑变化编辑（加孔/加把手），重建质量与SOTA持平。

**[3D Test-time Adaptation via Graph Spectral Driven Point Shift](3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)**

:   提出GSDTTA，首次将3D点云的测试时适应从空间域转移到图谱域，通过仅优化最低10%频率分量（减少约90%参数）实现全局结构调整，并结合特征图引导的自训练策略生成伪标签，在ModelNet40-C和ScanObjectNN-C上显著超越现有3D TTA方法。

**[3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_mode.md)**

:   提出3DGraphLLM，首个将**3D语义场景图的可学习表示**直接输入LLM的方法——通过k近邻子图+三元组(object1, relation, object2)编码物体间语义关系，然后投影到LLM的token嵌入空间。在ScanRefer上Acc@0.5提升+6.4%（vs无语义关系的Chat-Scene），在Multi3DRefer上F1@0.5提升+7.5%，推理速度比GPT4Scene-HDM快5倍。

**[3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)**

:   将3D Gaussian Splatting的ADAM优化器替换为定制化的Levenberg-Marquardt（LM）二阶优化器，通过高效CUDA并行化的PCG算法和梯度缓存结构实现Jacobian-向量积加速，在保持相同重建质量的前提下将优化时间缩短约20%。

**[4D Gaussian Splatting SLAM](4d_gaussian_splatting_slam.md)**

:   提出首个完整的4D Gaussian Splatting SLAM系统，在动态场景中同时进行相机位姿跟踪和4D高斯辐射场重建——将高斯原语分为静态/动态集合，通过稀疏控制点+MLP建模动态物体运动，并创新性地渲染2D光流图监督动态高斯学习。

**[4D Visual Pre-training for Robot Learning](4d_visual_pretraining_for_robot_learning.md)**

:   FVP提出将3D视觉预训练建模为"下一帧点云预测"问题，用条件扩散模型从历史帧点云预测未来帧点云来学习3D视觉表示，在12个真实世界操作任务中将DP3的平均成功率提升28%，达到SOTA水平。

**[7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatialtemporalangular_gaussian_splatting.md)**

:   提出7DGS，将场景元素建模为**7维高斯分布**（3D空间+1D时间+3D视角方向），通过条件切片机制将7D高斯转换为与时间和视角相关的条件3D高斯，统一处理动态场景+视角依赖效果，在自定义7DGS-PBR数据集上比4DGS PSNR提升高达7.36dB，仅用15.3%的高斯点数，401FPS实时渲染。

**[A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)**

:   提出A³GS，首个**前馈式零样本3DGS风格迁移**网络——使用图卷积网络(GCN)自编码器将3DGS场景编码到潜在空间，通过AdaIN注入任意风格图像特征，仅需**10秒**即可将任意风格迁移到任意3D场景，无需逐风格优化，可处理大规模3DGS场景。

**[A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacherguided_diffusion_for_3d_gaussian_s.md)**

:   提出一种用2D图像监督训练3D扩散模型的框架：利用预训练的确定性3D重建模型作为"噪声教师"生成3D噪声样本，通过多步去噪策略和渲染损失实现跨模态（3D去噪+2D监督）训练，在用更小模型的情况下超越教师模型0.5-0.85 PSNR。

**[A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)**

:   将单图到3D世界生成分解为两个更简单的子问题——全景合成（无训练in-context learning）和点云条件修复（仅5k步微调ControlNet），结合3DGS重建出可在VR中2米范围内导航的沉浸式3D环境，在图像质量指标上全面超越WonderJourney和DimensionX。

**[A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)**

:   提出HarDiff——基于离散Hartley变换的频域学习策略，通过低频训练（从源域提取结构先验）和高频采样（利用目标域细节引导）增强扩散模型在稠密视觉任务上的跨域泛化能力，在语义分割、深度估计和去雾等12个基准上取得SOTA。

**[AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaagaussians_antialiased_and_artifactfree_3d_gaussian_render.md)**

:   通过在3DGS渲染管线的所有环节中融入完整的3D评估（而非2D splat近似），提出自适应3D平滑滤波器、视空间边界计算和基于视锥的tile剔除，统一解决了3DGS中的锯齿、投影伪影和弹出伪影（popping），在OOD视角下大幅优于现有方法，同时保持实时渲染（>100 FPS）。

**[AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compos.md)**

:   提出AdaHuman框架，通过姿态条件的联合3D扩散模型（在扩散过程中同步进行多视角图像生成与3DGS重建以保证3D一致性）和组合式3DGS细化模块（利用crop-aware camera ray map融合局部精细细节），从单张野外图片生成高保真可动画的3D人体avatar，在重建和重姿态任务上全面超越现有SOTA。

**[Adversarial Exploitation of Data Diversity Improves Visual Localization](adversarial_exploitation_of_data_diversity_improves_visual_l.md)**

:   提出RAP（Robust Absolute Pose regression）——基于外观感知3DGS的双分支联合训练框架，通过对抗判别器弥合合成-真实域差距+外观/位姿增强数据作为额外监督，在Cambridge Landmarks上平移/旋转误差分别降低38-50%/41-44%，在日夜场景和驾驶场景中表现尤为突出。

**[AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)**

:   首个面向截肢者的3D人体网格恢复框架——通过合成100万+截肢者图像(A3D)、设计BPAC-Net截肢分类器区分截肢与遮挡、以及双Tokenizer切换策略分别编码截肢/正常位姿先验，在截肢者数据上大幅领先(ITW-amputee上MVE比TokenHMR低16.87)，非截肢者数据上也保持竞争力。

**[Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)**

:   提出Amodal3R，一个端到端的遮挡感知3D重建模型，通过在TRELLIS基础上引入mask加权交叉注意力和遮挡感知注意力层，直接在3D潜空间中从部分遮挡的2D图像重建完整的3D物体形状和外观，大幅超越先前"2D补全→3D重建"的两阶段方法。

**[Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)**

:   提出非模态相对深度估计新范式，构建大规模真实数据集ADIW（564K），基于Depth Anything V2和DepthFM设计两个互补框架（Amodal-DAV2和Amodal-DepthFM），通过最小化修改预训练模型实现遮挡区域深度预测，在ADIW上RMSE比之前SOTA提升27.4%。

**[AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation](animateanymesh_a_feedforward_4d_foundation_model_for_textdri.md)**

:   提出AnimateAnyMesh，首个前馈式文本驱动通用Mesh动画框架：通过DyMeshVAE将动态Mesh分解为初始位置和相对轨迹并压缩到潜空间，再用基于Rectified Flow的MMDiT模型学习文本条件下的轨迹分布，配合4M+规模的DyMesh数据集训练，在6秒内即可为任意拓扑Mesh生成高质量动画，全面碾压DG4D、L4GM和Animate3D。

**[AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)**

:   提出AnyI2V，一个无需训练的框架，可接受任意模态图像（mesh、点云、深度图、骨架等）作为首帧条件，结合用户定义的轨迹实现运动控制的视频生成，在FID/FVD/ObjMC指标上优于现有training-free方法并与训练方法竞争。

**[AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)**

:   提出AR-1-to-3，一种基于扩散模型的自回归下一视角预测框架，通过"先近后远"的渐进式生成策略，配合Stacked-LE（堆叠局部特征编码）和LSTM-GE（全局特征编码）两种条件注入机制，显著提升了单图到多视角生成的一致性，在GSO数据集上PSNR达13.18（相比InstantMesh的10.67提升23.5%），Chamfer Distance降至0.063（InstantMesh为0.117）。

**[Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universa.md)**

:   提出Articulate3D（280个真实室内场景、8类铰接标注的大规模数据集）和USDNet（基于Mask3D扩展的统一框架），通过密集逐点预测机制同时完成可动零件分割和运动参数估计，在铰接参数预测上比Mask3D†提升5.7%，并支持LLM场景编辑和机器人策略训练。

**[ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling](atlas_decoupling_skeletal_and_shape_parameters_for_expressiv.md)**

:   提出ATLAS参数化人体模型，通过显式解耦外部表面形状和内部骨骼参数，并引入稀疏非线性姿态校正变形，在60万高分辨率扫描数据上训练，实现了比SMPL-X更精确、更可控的3D人体建模。

**[AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)**

:   提出AutoOcc，一个以视觉为中心的全自动开放式语义占据标注流水线，通过视觉-语言模型引导的可微高斯泼溅（VL-GS）实现无需人工标签的3D语义占据生成，在Occ3D-nuScenes上以纯视觉输入就达到IoU 83.01/mIoU 20.92，大幅超越现有自动标注方法。

**[Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruc.md)**

:   提出BA-Track框架，通过学习型3D点追踪器将观测到的运动解耦为相机引起的运动和物体自身运动，使传统束调整(BA)能够无差别地处理静态和动态点，在相机位姿估计(ATE在Sintel上达到0.034，较SOTA降低一半以上)和稠密3D重建上取得显著提升。

**[Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)**

:   提出DiffusionGS，将3D高斯点云直接嵌入扩散模型的去噪器中，通过单阶段3D扩散实现从单张图片到3D物体生成和场景重建，在ABO/GSO上PSNR超越SOTA 2.20/1.25 dB，RealEstate10K上超1.34 dB，推理速度约6秒（A100）。

**[BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matchi.md)**

:   提出双边聚合网络BANet，通过将代价体分离为高频细节体和低频平滑体分别聚合再融合，仅使用2D卷积即可在移动设备上实现实时高精度立体匹配（骁龙8 Gen 3上45ms，KITTI 2015 D1-all=1.83%，比MobileStereoNet-2D精度高35.3%）。

**[Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multidimensional_quality_evaluator.md)**

:   构建MATE-3D基准（8类prompt×8种方法=1280个textured mesh，4维度×21人主观评分=107520标注）并提出HyperScore多维质量评估器：通过可学习条件特征+条件特征融合(模拟注意力转移)+超网络生成维度自适应映射函数(模拟决策过程变化)，在语义对齐、几何、纹理、整体4个维度上全面超越现有指标。

**[Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)**

:   提出 LaMAria——首个城市尺度的第一人称多传感器 VIO/SLAM 基准数据集，利用测绘级控制点提供厘米精度的地面真值，系统评估了学术界主流 SLAM 方案在真实第一人称场景下的表现，揭示了现有方法与商业系统之间的巨大差距。

**[BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)**

:   用可学习的Bézier曲线显式建模动态物体的运动轨迹，替代传统依赖精确bbox标注的范式，实现了对自动驾驶街景中动/静态成分的准确分离与高保真重建。

**[BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)**

:   提出BBSplat——用可学习的RGB纹理和alpha贴图替代2D Gaussian Splatting中的高斯分布不透明度，使每个平面基元具有任意形状和逐像素颜色控制，在用更少基元的情况下弥补2DGS与3DGS之间的渲染质量差距，同时保留精确网格提取能力并实现最高×17的存储压缩。

**[Blended Point Cloud Diffusion for Localized Text-guided Shape Editing](blended_point_cloud_diffusion_for_localized_textguided_shape.md)**

:   提出 BlendedPC，将局部文本引导的3D形状编辑重新定义为语义inpainting问题，通过在Point·E基础上训练Inpaint-E模型，并在推理时引入无需反演(inversion-free)的坐标混合(coordinate blending)机制，在保持原始形状身份的同时实现精准局部编辑，在ShapeTalk数据集上全面超越现有方法。

**[Bolt3D: Generating 3D Scenes in Seconds](bolt3d_generating_3d_scenes_in_seconds.md)**

:   提出一种基于潜在扩散模型的前馈式3D场景生成方法，通过将3D场景表示为多组Splatter Image并使用专门训练的几何VAE，**在单GPU上7秒内生成完整3D场景**，推理成本比优化式方法（CAT3D）降低300倍。

**[Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusionbased_monocular_camer.md)**

:   提出DM-Calib——基于扩散模型的单目相机内参估计方法：设计Camera Image表示（将内参无损编码为3通道图像=方位角+仰角+灰度图），微调Stable Diffusion生成Camera Image，用RANSAC提取内参，在5个零样本数据集上超越所有基线，并将相机标定扩展到度量深度估计、位姿估计和稀疏视角3D重建。

**[Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)**

:   SGCDet通过几何与上下文感知的聚合模块（3D可变形注意力+多视角注意力融合）和基于占据概率的稀疏体素构建策略，在无需ground-truth几何监督的情况下，实现了多视角室内3D目标检测的SOTA性能，同时大幅降低计算开销。

**[Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)**

:   提出Bootstrap3D框架，利用视频扩散模型生成合成多视图数据，并通过微调的MV-LLaVA进行质量过滤与密集描述重写，结合Training Timestep Reschedule (TTR)策略训练多视图扩散模型，在不牺牲视图一致性的前提下大幅提升图像质量和文本对齐能力。

**[BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pos.md)**

:   提出以3D包围盒角点作为中间表示，通过Transformer解码器预测查询视图中角点的2D投影热图，结合PnP算法实现可泛化的稀疏视角6DoF物体位姿估计，在遮挡和稀疏视角场景下显著优于现有方法。

**[PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_highquality.md)**

:   提出PASDF框架，通过姿态对齐模块(PAM)将点云对齐到标准姿态 + 神经SDF网络学习连续几何表示 + 基于SDF偏差的异常评分，统一实现3D点云异常检测与异常修复(Marching Cubes提取零等值面作为修复模板)，在Real3D-AD上O-AUROC 80.2%、Anomaly-ShapeNet上90.0%均达SOTA。

**[3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)**

:   提出3DSR——将扩散超分模型与3DGS表示交替迭代实现3D一致超分：每步去噪后将SR图像训练到3DGS中获得3D一致渲染→重编码回潜在空间引导下一步去噪，无需微调任何模型即显式保证跨视角一致性，在LLFF上PSNR提升1.16dB+FID降低50%(vs StableSR)。

**[Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimat.md)**

:   首次研究HMD后置相机对全身姿态追踪的价值，提出Transformer-based多视角热力图精炼模块(利用可变形注意力+不确定性感知遮罩)，解决后视角2D关节检测不可靠的问题，并发布两个大规模数据集(Ego4View-Syn/RW)，在Ego4View-RW上MPJPE比SOTA EgoPoseFormer提升>10%(63.38→56.94mm)。

**[BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)**

:   通过几何自适应bootstrapping确定体素大小/搜索半径、用FPS替代学习型关键点检测器、以及patch级坐标归一化，构建了一个无需人工调参即可在11个跨域数据集上实现零样本点云配准的pipeline BUFFER-X，在室内外多传感器多场景下取得了平均排名第一的成功率。

**[CAD-Recode: Reverse Engineering CAD Code from Point Clouds](cadrecode_reverse_engineering_cad_code_from_point_clouds.md)**

:   将CAD sketch-extrude序列表示为Python代码，利用轻量级点云投影器 + 预训练LLM解码器将点云翻译为可执行Python代码来重建CAD模型，在DeepCAD/Fusion360/真实世界CC3D数据集上显著超越现有方法，且输出代码可被通用LLM理解用于CAD编辑和问答。

**[Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_sce.md)**

:   提出Can3Tok——首个场景级3DGS VAE：通过cross-attention将大量(40K)无序3D Gaussian压缩到低维canonical token(256×768→64×64×4) + 3DGS归一化解决跨场景尺度不一致 + 语义感知过滤去除floater噪声，在DL3DV-10K上唯一成功的场景级3DGS潜在建模方法(L2=30.1, 失败率2.5%)，支持text-to-3DGS和image-to-3DGS前馈生成。

**[CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)**

:   提出CATSplat——单视图前馈3DGS重建的泛化Transformer框架：利用VLM文本嵌入（上下文先验）和3D点云特征（空间先验）通过双重cross-attention增强图像特征，在RE10K等数据集上在PSNR/SSIM/LPIPS全面超越Flash3D，且跨数据集泛化性优异。

**[CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector](charm3r_towards_unseen_camera_height_robust_monocular_3d_det.md)**

:   通过数学推导发现回归深度和地面深度在相机高度变化时呈现方向相反的误差趋势，CHARM3R 直接在模型内对两种深度做简单平均来抵消趋势，从而大幅提升单目3D检测器对未见相机高度的泛化能力（CARLA 上提升超过 45%）。

**[CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)**

:   用Neural ODE建模曝光时间内的连续相机运动轨迹，结合刚体变换和可学习的连续运动修正(CMR)变换，从运动模糊图像重建清晰3D高斯场景，在所有benchmark上大幅超越SOTA。

**[DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dapmae_domainadaptive_point_cloud_masked_autoencoder_for_eff.md)**

:   提出一种域自适应点云MAE框架（DAP-MAE），通过异构域适配器（HDA）和域特征生成器（DFG）两个模块，让一次跨域预训练即可在物体分类、人脸表情识别、部件分割、目标检测等多个不同域的下游任务上都达到SOTA。

**[Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)**

:   提出首个零样本开放世界系统 Diorama，通过模块化地组合 foundation model（GPT-4o、SAM、DinoV2、Metric3D 等），将单张 RGB 图像转化为包含建筑结构和 CAD 物体的完整可组合 3D 室内场景，无需任何端到端训练或人工标注。

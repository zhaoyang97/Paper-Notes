---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**📷 CVPR2026** · 共 **13** 篇

**[A Prediction-as-Perception Framework for 3D Object Detection](a_predictionasperception_framework_for_3d_object_d.md)**

:   借鉴人类"预判目标位置再聚焦观察"的认知模式，将前一帧的轨迹预测结果转化为当前帧的检测query，形成预测-感知迭代闭环，在UniAD上实现跟踪精度+10%和推理速度+15%的同步提升。

**[Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_cont.md)**

:   提出 CompoSIA，一个基于 Wan2.1 DiT 的组合式驾驶视频模拟器，通过对场景结构（3D bbox）、物体身份（单张参考图）和自车动作（相机轨迹）三因素的显式解耦注入，实现对抗性驾驶场景的细粒度可控生成，碰撞率提升 173%。

**[FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](fedbprompt_federated_domain_generalization_person.md)**

:   提出 FedBPrompt，将可学习视觉提示分为身体部件对齐提示（受限局部注意力处理视角错位）和全身整体提示（抑制背景干扰），并设计仅传输提示参数（~0.46M vs. 全模型~86M）的联邦微调策略，在 FedDG-ReID 上取得一致性提升。

**[IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](igasa_integrated_geometryaware_and_skipattention_m.md)**

:   提出 IGASA 点云配准框架，通过层级金字塔架构 (HPA) + 层级跨层注意力 (HCLA) 的跳跃注意力融合 + 迭代几何感知精细化 (IGAR) 的动态一致性加权，在 3DMatch 上达到 94.6% Registration Recall（SOTA），在 KITTI 上达到 100% RR，总推理时间仅 2.763s。

**[Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)**

:   构建大规模室外场景数据集MPO（含Velodyne稀疏和FARO稠密两种LiDAR点云），提出结合水平循环卷积(HCC)和行级最大池化(RWMP)的CNN架构，利用全景深度图和反射率图的多模态融合（Softmax Average），在6类室外场景分类上达97.87%准确率，显著超越传统手工特征方法。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lrsgs_robust_lidarreflectanceguided_salient_gaussi.md)**

:   提出LR-SGS，将LiDAR强度校准为光照不变的反射率通道附加到3D高斯体上，并设计结构感知的Salient Gaussian表示（从LiDAR几何和反射率特征点初始化）配合改进的密度控制和显著变换策略，在Waymo自动驾驶复杂场景中实现优于OmniRe的高保真重建，且高斯体更少、训练更快。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2occ_resilient_3d_semantic_occupancy_prediction_f.md)**

:   针对自动驾驶中相机故障导致的不完整输入问题，提出M²-Occ框架，通过多视角掩码重建（MMR）利用相邻相机重叠视场恢复缺失特征，并引入特征记忆模块（FMM）用类级语义原型精化体素表示，在缺失后视摄像头时IoU提升4.93%，不影响全视角性能。

**[MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_me.md)**

:   提出MetaDAT框架，通过元学习预训练获得适合在线适应的模型初始化，并在测试时采用动态学习率优化和困难样本驱动更新来实现跨数据集分布偏移下的轨迹预测自适应，在nuScenes/Lyft/Waymo多种跨域配置下全面超越现有TTT方法。

**[MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_multimodal_multiview_video_diffusion.md)**

:   提出 MoVieDrive，首个在统一框架下实现多模态（RGB+深度+语义）多视图自动驾驶场景视频生成的扩散 Transformer 方法，通过模态共享层+模态特定层的设计和多样化条件编码，在 nuScenes 上 FVD 达到 46.8（领先 SOTA 22%），同时生成高质量的深度图和语义图。

**[RESBev: Making BEV Perception More Robust](resbev_making_bev_perception_more_robust.md)**

:   提出RESBev——一个即插即用的BEV感知鲁棒性增强框架，通过隐空间世界模型从历史干净帧预测当前BEV语义先验，再与被损坏的当前观测融合，在nuScenes上显著提升四种LSS模型在10种干扰下的平均IoU（+15~20个点）。

**[SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)**

:   SG-NLF提出一种无需精确位姿的LiDAR NeRF框架，通过谱-几何混合表示解决LiDAR稀疏数据导致的几何空洞问题，利用置信感知图实现全局位姿优化，并引入对抗学习强化跨帧一致性，在nuScenes上重建质量和位姿精度分别比SOTA提升35.8%和68.8%。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](single_pixel_image_classification_using_an_ultrafa.md)**

:   利用microLED-on-CMOS超快光投影器(330kfps)进行单像素成像(SPI)，以12×12 Hadamard pattern照明MNIST数字并用单像素检测器采集时间序列，完全跳过图像重建，直接用ELM/DNN分类实测光信号，实现1.2kfps下>90%分类精度，二分类(异常检测)精度>99%。

**[Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multimodal_learning_in_3d_human_p.md)**

:   提出基于Shapley值的模态贡献评估+Fisher信息矩阵引导的自适应权重约束(AWC)正则化方法，解决RGB/LiDAR/mmWave/WiFi四模态融合中的模态不平衡问题，在MM-Fi数据集上MPJPE比naive fusion降低2.71mm，比最佳balancing方法降低约5mm，且不引入额外可学参数。

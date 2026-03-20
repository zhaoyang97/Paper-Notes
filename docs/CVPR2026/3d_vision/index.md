---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**📷 CVPR2026** · 共 **27** 篇

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4.md)**

:   将马科动物4D重建解耦为运动估计(AniMoFormer时空Transformer+后优化)和外观重建(EquineGS前馈3DGS)两个子任务，用VAREN参数化模型做桥梁，仅在合成数据(VarenPoser+VarenTex)上训练即在真实数据APT-36K和AiM上达到SOTA，并能零样本泛化到斑马和驴。

**[Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)**

:   利用计算预算不对称性，将扩散策略的迭代细化从推理时移至训练时——通过自适应漂移场将预测动作吸引向专家模式并排斥其他生成样本，从3D点云实现单步（1 NFE）高保真多模态动作生成，比扩散策略快10倍以上。

**[Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)**

:   提出Catalyst4D框架，通过锚点运动引导(AMG)和颜色不确定性外观精炼(CUAR)两个模块，将高质量的3D静态编辑结果传播到动态4D高斯场景中，避免了直接4D编辑的运动伪影和时间不一致问题。

**[CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_crossmodal_hybrid_attention_network_for.md)**

:   提出CMHANet，通过三阶段混合注意力（几何self-attention→图像aggregation-attention→源-目标cross-attention）融合2D图像纹理语义与3D点云几何信息，并引入跨模态对比损失，在3DMatch/3DLoMatch上达到最优配准性能。

**[Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_humanscene_reconstruction_from_multiperso.md)**

:   提出 CHROMM 统一框架，从多人多视图视频中一次性联合估计相机参数、场景点云和人体网格，无需外部模块或预处理数据，在 RICH 上 WA-MPJPE 达 53.1mm 且比优化方法快 8 倍以上。

**[GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_flow_matching_novel_view_synthesis.md)**

:   提出Data-to-Data Flow Matching直接学习视角间确定性变换，并引入概率密度测地线正则化使流路径沿数据流形高密度区域传播，在NVS中实现更好的跨视角一致性和几何保真度。

**[GGPT: Geometry Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)**

:   提出 GGPT 框架，通过改进的轻量 SfM 管线获取几何一致但稀疏的 3D 点云，再用 3D Point Transformer 在三维空间中直接融合稀疏几何引导与稠密前馈预测，实现跨架构、跨数据集的显著泛化提升。

**[Hybrid eTFCE–GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](hybrid_etfcegrf_exact_clustersize_retrieval_with_a.md)**

:   将 eTFCE 的并查集精确聚类大小提取与 pTFCE 的解析 GRF 推断相结合，首次同时实现精确聚类大小查询和无置换检验的解析 p 值，在全脑 VBM 分析上比 R pTFCE 快 4.6–75 倍，比置换 TFCE 快三个数量级。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)**

:   提出首个前馈HDR新视角合成方法InstantHDR，通过几何引导的外观建模和色调映射元网络，从未标定多曝光LDR图像中单次前向重建HDR 3D高斯场景，速度比优化方法快~700×，后优化版本快~20×且质量可比。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)**

:   提出JOPP-3D——首个联合处理点云和全景图的开放词汇语义分割框架，通过正二十面体切向分解将全景图转为透视图后利用SAM+CLIP提取实例级语义嵌入，再经深度对应实现3D→全景语义回投，在S3DIS上以80.9% mIoU超越所有监督/无监督方法（含PointTransformerV3的73.4%），全景分割70.1% mIoU大幅领先。

**[MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physicsgrounded_articulation_for_sim.md)**

:   提出MotionAnymesh零样本框架，通过SP4D运动学先验引导VLM消除运动学幻觉，并用物理约束轨迹优化保证无碰撞铰接，将静态3D网格自动转换为可在SAPIEN等物理引擎中直接使用的URDF数字孪生，物理可执行率达87%，远超现有方法。

**[MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation](msgnav_multimodal_3d_scene_embodied_navigation.md)**

:   提出多模态 3D 场景图（M3DSG）——用动态分配的图像替代纯文本关系边保留视觉线索，基于此构建 MSGNav 零样本导航系统，包含关键子图选择、自适应词汇更新、闭环推理和基于可见性的视角决策模块，在 GOAT-Bench 和 HM3D-ObjNav 上取得 SOTA。

**[NERFIFY: 多智能体框架将NeRF论文自动转化为可运行代码](nerfify_multiagent_nerf_paper_to_code.md)**

:   提出NERFIFY——通过6项关键创新（CFG约束、GoT代码合成、引用链组件恢复、视觉反馈修复、知识增强、系统评测），将NeRF论文可靠转化为可训练的Nerfstudio插件，在无公开实现的论文上达到±0.5dB PSNR的专家级复现质量，实现时间从数周降至数分钟。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](noderf_neural_ode_nerf_continuous_spacetime_dynam.md)**

:   将Neural ODE与动态NeRF紧密耦合，通过在隐空间中用ODE建模场景状态的连续时间演化，实现了长程时间外推和跨轨迹泛化，突破了现有动态NeRF只能在训练时间范围内插的瓶颈。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_openvocabulary_occupancy_predi.md)**

:   提出O3N——首个纯视觉端到端全向开放词汇占用预测框架，通过极坐标螺旋Mamba（PsM）、占用代价聚合（OCA）和无梯度自然模态对齐（NMA）三大模块，在QuadOcc和Human360Occ上实现SOTA。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geome.md)**

:   提出Pano360，将全景拼接从传统的2D成对对齐扩展到3D摄影测量空间，利用基于Transformer的架构实现多视图全局几何一致性，在弱纹理、大视差和重复纹理等挑战场景中成功率达97.8%，并构建了包含200个真实场景的大规模数据集。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)**

:   面向四足机器人构建首个全景多模态（RGB+热成像+偏振+LiDAR）语义占据数据集PanoMMOcc，并提出VoxelHound框架，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块实现鲁棒的3D占据预测，达到23.34% mIoU（+4.16%）。

**[PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_4d_synthesis.md)**

:   提出 PhysGM，首个从单张图像前馈式联合预测 3D 高斯表示和物理属性的框架，通过物理感知重建模型预训练 + DPO 将模拟与物理合理参考视频对齐，配合 50K+ 物理标注 3D 资产数据集 PhysAssets，在一分钟内从单图生成高保真 4D 物理仿真，相比先前方法显著加速。

**[R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](r4det_4d_radar_camera_fusion_3d_detection.md)**

:   提出R4Det，通过全景深度融合（PDF）、可变形门控时序融合（DGTF）和实例引导动态精炼（IGDR）三个即插即用模块，解决4D雷达-相机融合中深度估计不准、时序融合依赖ego pose、小目标检测困难的问题，在TJ4DRadSet和VoD上取得SOTA。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](regularizing_inr_with_diffusion_prior_selfsupervis.md)**

:   将扩散模型先验作为正则化项引入隐式神经表示(INR)的损失函数中，构建DINR框架用于稀疏视图中子CT重建，在仅5个视角的极端稀疏条件下仍能保持混凝土微结构的高质量重建。

**[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous_time_4d_gaussian.md)**

:   提出 RetimeGS，解决 4D 高斯溅射在时间插值时的鬼影伪影问题——将问题诊断为时间混叠（temporal aliasing），显式定义 3D 高斯的时间行为以缓解混叠，结合光流引导初始化/监督、三重渲染监督等策略，在快速运动、非刚性变形和严重遮挡数据集上实现无鬼影、时间一致的连续时间 4D 渲染。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weaklysupervised_s.md)**

:   首次将前馈3D重建(MapAnything)的几何信息作为辅助监督信号引入弱监督2D语义分割，通过双Student-Teacher架构和置信度加权的跨模态一致性损失，在4个数据集上以2-7% mIoU大幅超越SOTA——且推理时仅需2D模型。

**[Scaling View Synthesis Transformers (SVSM)](scaling_view_synthesis_transformers.md)**

:   首次系统研究无几何NVS Transformer的缩放定律，提出有效批量大小假设和encoder-decoder SVSM架构，证明与decoder-only LVSM相同缩放斜率但3倍计算效率，在真实世界NVS上达新SOTA。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scenecontextualized_incremental_fewshot_3d_s.md)**

:   提出即插即用的背景引导原型增强框架SCOPE，从背景区域挖掘伪实例原型丰富新类原型表示，在ScanNet上5-shot新类IoU达23.86%(vs GW 16.88%，+6.98%)，且几乎无额外计算开销(<1MB, 0.02s)。

**[SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](span_spatial_projection_alignment_mono3d.md)**

:   提出SPAN即插即用几何协同约束框架，通过3D角点空间对齐和3D-2D投影对齐两个可微损失，强制解耦预测的各属性满足全局几何一致性，配合层级任务学习策略稳定训练，在KITTI上将MonoDGP的Car Moderate AP3D提升0.92%达到新SOTA。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](spectral_defense_against_resourcetargeting_attack.md)**

:   提出联合3D频率感知高斯剪枝和2D角度各向异性正则化的频域防御，将3DGS中毒攻击导致的高斯过增长抑制最多5.92×、峰值显存降3.66×、渲染速度提升4.34×，同时保持甚至改善渲染质量。

**[WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zerosh.md)**

:   系统研究合成立体数据集的设计空间——变换Infinigen过程化生成参数(浮动物体密度/背景/材质/相机baseline/光照等)分析其对零样本立体匹配的影响，发现"真实室内场景+浮动物体"的组合最有效；据此构建WMGStereo-150k数据集，仅用此单一数据集训练超越SceneFlow+CREStereo+TartanAir+IRS四合一(Middlebury降28%，Booster降25%)，与FoundationStereo竞争力相当。

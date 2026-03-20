<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎞️ ECCV2024 论文笔记

共 **57** 篇笔记，覆盖 **14** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 🧊 [3D 视觉](#3d_vision) | 8 |
| 🎨 [图像生成](#image_generation) | 7 |
| 🧑 [人体理解](#human_understanding) | 5 |
| 📦 [模型压缩](#model_compression) | 5 |
| 👁️ [多模态 VLM](#multimodal_vlm) | 4 |
| ✂️ [语义分割](#segmentation) | 4 |
| 🚗 [自动驾驶](#autonomous_driving) | 2 |
| 🖼️ [图像恢复](#image_restoration) | 2 |
| 🏥 [医学图像](#medical_imaging) | 2 |
| 🔄 [自监督/表示学习](#self_supervised) | 2 |
| 🎬 [视频理解](#video_understanding) | 2 |
| 💬 [LLM / NLP](#llm_nlp) | 1 |
| 🎯 [目标检测](#object_detection) | 1 |
| 📂 [其他](#others) | 12 |

---

## 🧊 3D 视觉 { #3d_vision }

**[3D Congealing: 3D-Aware Image Alignment in the Wild](3d_vision/3d_congealing_3d-aware_image_alignment_in_the_wild.md)**

:   3D Congealing将一组语义相似的无标注互联网图像对齐到共享的3D canonical空间，通过结合预训练扩散模型的SDS指导获得3D形状 + DINO语义特征匹配估计位姿和坐标映射，无需模板、位姿标注或相机参数。

**[3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_vision/3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)**

:   提出HORSE框架，通过从野外视频中提取多视角2D mask监督（以手部姿态作为物体姿态代理）和从合成3D形状集合中学习2D切片对抗形状先验，训练occupancy网络从单张RGB图像重建手持物体3D形状，在不使用任何真实世界3D标注的情况下，在MOW数据集上超越使用3D监督的方法11.6%。

**[3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_vision/3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)**

:   HVTrack首次探索高时间变化场景下的3D单目标跟踪，通过相对位姿感知记忆模块(RPM)、基础-扩展特征交叉注意力(BEA)和上下文点引导自注意力(CPA)三个模块，分别解决点云形状剧变、相似物体干扰和背景噪声问题，在KITTI-HV 5帧间隔下比SOTA提升11.3%/15.7% Success/Precision。

**[3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting](3d_vision/3igs_factorised_tensorial_illumination_for_3d_gaussian_splatting.md)**

:   通过引入基于张量分解的连续入射光照场和可学习BRDF特征，替代3DGS中独立优化的球谐系数，3iGS显著提升了镜面反射等视角依赖效果的渲染质量，同时保持实时渲染速度。

**[3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)**

:   提出了一种无需训练的3D物体部件分割方法3-By-2，利用扩散模型(DIFT)的2D语义对应关系从已标注2D数据集或少量3D标注对象中迁移部件标签到3D，在zero-shot和few-shot设置下均达到SOTA。

**[6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](3d_vision/6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)**

:   提出6DGS，通过反转3DGS渲染流程——从椭球体表面均匀发射光线（Ellicell），利用注意力机制将光线与目标图像像素绑定，再用加权最小二乘闭式求解相机位姿，无需迭代和初始位姿，在真实场景上旋转精度提升12%、平移精度提升22%，达到15fps近实时性能。

**[A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](3d_vision/a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)**

:   将3DGS中的位置和旋转参数建模为时间的函数（位置用Fourier逼近、旋转用线性逼近），使动态场景的存储复杂度从O(TN)降低到O(LN)，在D-NeRF/DyNeRF/HyperNeRF三个数据集上实现了与NeRF方法匹敌的渲染质量，同时保持118+ FPS的实时渲染速度。

**[MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](3d_vision/mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)**

:   将MVS的代价体深度估计与3D高斯溅射结合，通过混合渲染(splatting+volume rendering)提升泛化性，并提出基于多视图几何一致性的点云聚合策略，使per-scene优化仅需45秒就超越3D-GS的10分钟效果。

---

## 🎨 图像生成 { #image_generation }

**[2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction](image_generation/2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)**

:   2S-ODIS通过两阶段结构利用预训练VQGAN（无需微调）合成全景图像：第一阶段生成低分辨率粗略ERP图，第二阶段通过生成26个NFoV局部图像并融合来校正几何畸变，训练时间从14天缩短到4天且图像质量更优。

**[A Diffusion Model for Simulation Ready Coronary Anatomy with Morpho-skeletal Control](image_generation/a_diffusion_model_for_simulation_ready_coronary_anatomy_with.md)**

:   用潜在扩散模型（LDM）可控生成3D多组织冠状动脉分割图，通过拓扑交互损失保证解剖合理性，通过形态-骨架双通道条件化实现对截面形态和分支结构的解耦控制，并提出自适应空条件引导（ANG）以非可微回归器高效增强条件保真度，最终支持面向有限元仿真的反事实解剖结构编辑。

**[AccDiffusion: An Accurate Method for Higher-Resolution Image Generation](image_generation/accdiffusion_an_accurate_method_for_higher-resolution_image_generation.md)**

:   提出AccDiffusion，通过将全局文本prompt解耦为patch级别的内容感知prompt（利用cross-attention map判断每个词汇是否属于某patch），并引入带窗口交互的膨胀采样来改善全局一致性，在无需额外训练的情况下有效解决patch-wise高分辨率图像生成中的目标重复问题，在SDXL上实现了从2K到4K分辨率的无重复高质量图像外推。

**[AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)**

:   观察到扩散模型超分中不同图像区域所需去噪步数差异巨大（背景区域早已收敛而前景纹理仍需迭代），提出基于多指标潜在熵（MMLE）感知信息增益来动态跳步的策略，将子区域分为稳定/增长/饱和三类给予不同步长，并通过渐进特征注入（PFJ）平衡保真度与真实感，在DRealSR等数据集上取得与StableSR可比的质量但推理时间和FLOPs分别减少1.5×和2.7×。

**[AdaGen: Learning Adaptive Policy for Image Synthesis](image_generation/adagen_learning_adaptive_policy_for_image_synthesis.md)**

:   将多步生成模型（MaskGIT/AR/Diffusion/Rectified Flow）的步级参数调度（温度、mask ratio、CFG scale、timestep等）统一建模为MDP，用轻量RL策略网络实现样本自适应调度，并提出对抗奖励设计防止策略过拟合，在四种生成范式上一致提升性能（VAR FID 1.92→1.59，DiT-XL推理成本降3倍同时性能更优）。

**[AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](image_generation/adanat_exploring_adaptive_policy_for_token-based_image_generation.md)**

:   提出AdaNAT，将非自回归Transformer（NAT）的生成策略配置建模为MDP，通过轻量策略网络+PPO强化学习+对抗奖励模型自动为每个样本定制生成策略（重掩码比例、采样温度、CFG权重等），在ImageNet-256上仅用8步达到FID 2.86，相比手工策略实现约40%的相对提升。

**[∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions](image_generation/inftybrush_controllable_large_image_synthesis_with_diffusion.md)**

:   提出首个在无限维函数空间中的条件扩散模型 ∞-Brush，通过交叉注意力神经算子实现可控条件生成，仅用 0.4% 像素训练即可在任意分辨率（最高 4096×4096）上生成保持全局结构的大图像。

---

## 🧑 人体理解 { #human_understanding }

**[3D Hand Pose Estimation in Everyday Egocentric Images](human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)**

:   通过系统研究裁剪输入、相机内参感知位置编码(KPE)、辅助监督(手部分割+抓握标签)和多数据集联合训练这四个实践，提出WildHands系统，在仅用ResNet50和少量数据的条件下，实现了对野外第一人称图像中3D手部姿态的鲁棒估计，零样本泛化超过FrankMocap全部指标且与10倍大的HaMeR竞争。

**[A Probability-guided Sampler for Neural Implicit Surface Rendering](human_understanding/a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)**

:   提出一种概率引导的光线采样器（Probability-guided Sampler），在3D图像投影空间中建模概率密度函数来指导光线采样朝向感兴趣区域，同时设计了包含近表面和空白空间两个分量的新型表面重建损失，可作为插件集成到现有神经隐式表面渲染器中，显著提升重建精度和渲染质量。

**[A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](human_understanding/a_simple_baseline_for_spoken_language_to_sign_language_trans.md)**

:   提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

**[AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](human_understanding/adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)**

:   提出AdaDistill，将知识蒸馏概念嵌入margin penalty softmax loss中，通过基于EMA的自适应类中心（早期用sample-sample简单知识、后期用sample-center复杂知识）和困难样本感知机制，无需额外超参数即可提升轻量级人脸识别模型的判别能力，在IJB-B/C和ICCV21-MFR等挑战性基准上超越SOTA蒸馏方法。

**[ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](human_understanding/aden_adaptive_density_representations_for_sparseview_camera.md)**

:   ADen提出生成器-判别器框架统一位姿回归和概率位姿估计：生成器输出多个6DoF位姿假设来建模多模态分布（处理对称歧义），判别器选出最佳假设，在稀疏视角位姿估计上同时实现了更高精度和更低运行时间。

---

## 📦 模型压缩 { #model_compression }

**[A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](model_compression/a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)**

:   首个面向视频快照压缩成像（Video SCI）重建任务的低比特量化框架Q-SCI，通过高质量特征提取模块、精确视频重建模块和Transformer分支的query/key分布偏移操作，在4-bit量化下实现7.8倍理论加速且性能仅下降2.3%。

**[AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale](model_compression/adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale.md)**

:   提出AdaGlimpse，利用Soft Actor-Critic强化学习从连续动作空间中选择任意位置和尺度的glimpse，结合弹性位置编码的ViT编码器实现多任务（重建/分类/分割）的主动视觉探索，以仅6%像素超越了使用18%像素的SOTA方法。

**[AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](model_compression/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)**

:   提出自适应对数底量化器AdaLog，通过可搜索的对数底替代固定log₂/log√2量化器来处理ViT中post-Softmax和post-GELU激活的幂律分布，并设计快速渐进组合搜索(FPCS)策略高效确定量化超参，在极低比特(3/4-bit)下显著优于现有ViT PTQ方法。

**[Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)**

:   提出AdaSense，利用预训练扩散模型的零样本后验采样来量化重建不确定性，从而自适应地选择最优测量矩阵，无需额外训练即可在人脸图像、MRI和CT等多领域实现优于非自适应方法的压缩感知重建。

**[Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)**

:   提出ℋ1.5框架：为每个输入数据自适应选择最佳采样mask-重建网络对（J=3对），利用超分辨率空间生成模型量化高频贝叶斯不确定性来决定采样策略，理论证明优于联合优化ℋ1（非自适应）和自适应采样ℋ2（Pareto次优）。

---

## 👁️ 多模态 VLM { #multimodal_vlm }

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](multimodal_vlm/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)**

:   AdaShield通过在MLLM输入前添加防御提示(defense prompt)来防御结构化越狱攻击（图像中嵌入有害文本），提出静态手动提示和自适应自动精化框架两种方案，无需微调模型即可显著提升安全性且不损害正常能力。

**[AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](multimodal_vlm/addressclip_empowering_visionlanguage_models_for_citywide_im.md)**

:   AddressCLIP定义了"图像地址定位"(IAL)新任务，提出端到端框架通过图像-文本对齐（图像↔地址/场景描述的对比学习）和图像-地理匹配（流形学习约束空间距离）直接预测图像拍摄的可读文本地址，避免了坐标→地址两阶段方法的歧义。

**[Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](multimodal_vlm/vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

---

## ✂️ 语义分割 { #segmentation }

**[A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties](segmentation/a_semantic_space_is_worth_256_language_descriptions_make_str.md)**

:   ProLab 用 LLM 生成类别的常识性描述，通过句子嵌入和 K-Means 聚类将其压缩为 256 个可解释的描述性属性，构建属性级多热标签空间替代传统 one-hot 类别标签来监督分割模型，在五个经典基准上一致超越类别级监督且涌现出域外泛化能力。

**[A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](segmentation/a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)**

:   基于Stable Diffusion构建了一个极简的潜在扩散分割框架LDMSeg，通过浅层自编码器将分割mask压缩到潜空间、再训练图像条件扩散模型来生成全景分割结果，避免了传统方法中的目标检测模块、匈牙利匹配和复杂后处理，并天然支持mask inpainting和多任务扩展。

**[ActionVOS: Actions as Prompts for Video Object Segmentation](segmentation/actionvos_actions_as_prompts_for_video_object_segmentation.md)**

:   提出ActionVOS——一种以人类动作叙述作为额外语言提示的Referring Video Object Segmentation新设定，通过无参数的动作感知标注模块生成伪标签，并设计动作引导的focal loss来抑制假阳性，在VISOR上将非活跃物体的误分割降低35.6% mIoU，同时在VOST/VSCOS上对状态变化物体的分割提升3.0% mIoU。

**[Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images](segmentation/active_coarsetofine_segmentation_of_moveable_parts_from_real.md)**

:   提出首个面向真实室内场景RGB图像中可运动部件实例分割的主动学习框架，通过姿态感知masked attention网络实现由粗到细的分割，仅需人工标注11.45%的图像即可获得全量验证的高质量分割结果，相比最优非AL方法节省60%人工时间。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](autonomous_driving/accelerating_online_mapping_and_behavior_prediction_via_dire.md)**

:   提出直接将在线地图估计模型内部的BEV特征暴露给下游轨迹预测模型（而非仅传递解码后的矢量化地图），通过三种BEV特征注入策略实现推理加速最高73%、预测精度提升最高29%。

**[Adaptive Human Trajectory Prediction via Latent Corridors](autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)**

:   将prompt tuning思想引入行人轨迹预测，通过在预训练轨迹预测器的输入端添加可学习的低秩图像prompt（称为latent corridors），以不到0.1%的额外参数实现对部署场景特定行为模式的高效自适应，在合成和真实数据上分别取得最高23.9%和26.8%的ADE提升。

---

## 🖼️ 图像恢复 { #image_restoration }

**[A New Dataset and Framework for Real-World Blurred Images Super-Resolution](image_restoration/a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)**

:   针对现有盲超分方法在处理含模糊（散焦/运动模糊）图像时过度纹理化、破坏模糊区域感知质量的问题，构建了包含近3000张模糊图像的ReBlurSR数据集，并提出PBaSR框架，通过双分支解耦训练（CDM）和基于权重插值的跨分支融合（CFM），在不增加任何推理开销的前提下，同时提升模糊图像和普通图像的超分效果，LPIPS提升0.02~0.10。

**[Accelerating Image Super-Resolution Networks with Pixel-Level Classification](image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)**

:   提出PCSR——首个像素级计算资源分配的超分方法，用轻量MLP分类器逐像素判断恢复难度并分配到不同容量的上采样器，在PSNR几乎不掉的情况下将FLOPs压低至原始模型的18%~57%，大幅优于现有patch级方法ClassSR和ARM。

---

## 🏥 医学图像 { #medical_imaging }

**[Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](medical_imaging/adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)**

:   针对医学图像无监督配准中噪声、遮挡等干扰因素导致的虚假重建误差问题，提出了一个自适应对应关系评分框架（AdaCS），通过学习像素级的对应置信度图来重新加权误差残差，以即插即用方式一致提升三种主流配准架构在三个数据集上的性能。

**[NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)**

:   NePhi用隐式神经网络（SIREN）替代传统的体素化形变场来表示配准变换，通过编码器预测latent code + 可选的测试时优化实现快速且近似微分同胚的医学图像配准，在多分辨率设置下与SOTA精度相当但内存降低5倍。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[4D Contrastive Superflows are Dense 3D Representation Learners](self_supervised/4d_contrastive_superflows_are_dense_3d_representation_learners.md)**

:   提出SuperFlow框架，通过视图一致性对齐、稠密-稀疏一致性正则化、和基于流的时空对比学习三个模块，利用连续LiDAR-相机对建立4D预训练目标，在11个异构LiDAR数据集上全面超越了之前的Image-to-LiDAR预训练方法。

**[Adaptive Multi-head Contrastive Learning](self_supervised/adaptive_multihead_contrastive_learning.md)**

:   AMCL提出使用多个投影头（各自产生不同特征）+ 对每个样本对和每个头自适应学习温度参数，从最大似然估计推导出损失函数，作为通用插件在SimCLR/MoCo/Barlow Twins/CAN/LGP上一致提升1-5%性能。

---

## 🎬 视频理解 { #video_understanding }

**[Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts](video_understanding/adapt2reward_adapting_videolanguage_models_to_generalizable.md)**

:   提出 Adapt2Reward，通过可学习的失败提示（failure prompts）将预训练视频语言模型适配为可泛化的语言条件奖励函数，仅需少量单一环境的机器人数据即可泛化到新环境和新任务，在 MetaWorld 上比前方法高出约 28%。

**[R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)**

:   R²-Tuning提出了一个仅需1.5%参数的轻量R²Block，通过从CLIP后层向前层的逆向递归方式聚合多层空间特征并精化时序关联，在6个VTG基准上以2.7M参数超越了使用额外时序骨干的4倍大方法。

---

## 💬 LLM / NLP { #llm_nlp }

**[AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](llm_nlp/adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)**

:   在CLIP中同时引入静态（全局共享）和动态（逐图生成）两种可学习提示，用辅助异常检测数据训练后，在14个工业+医学异常检测数据集上实现零样本SOTA，核心在于"任务级+实例级"双层自适应的混合提示设计。

---

## 🎯 目标检测 { #object_detection }

**[Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](object_detection/adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   提出两步共形预测框架为多类目标检测的边界框生成带理论覆盖率保证的自适应不确定性区间——第一步用共形分类集处理类别误判风险，第二步用集成/分位数回归等方法构建自适应于目标尺寸的边界框预测区间，在COCO/Cityscapes/BDD100k上达到约90%目标覆盖率且区间实际可用。

---

## 📂 其他 { #others }

**[3DEgo: 3D Editing on the Go!](others/3dego_3d_editing_on_the_go.md)**

:   3DEgo将传统三阶段3D编辑流程（COLMAP位姿估计→未编辑场景初始化→迭代编辑更新）压缩为单阶段框架：先用自回归噪声混合模块对视频帧进行多视角一致的2D编辑，再用COLMAP-free的3DGS从编辑后帧直接重建3D场景，速度提升约10倍且支持任意来源视频。

**[3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](others/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)**

:   提出将视线估计重新表述为密集3D眼球网格回归，并通过从大规模野外人脸图像中自动提取伪标签+HeadGAN合成多视图进行弱监督训练，在跨域场景下比SOTA提升最多30%。

**[A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks](others/a_closer_look_at_gan_priors_exploiting_intermediate_features.md)**

:   提出 IF-GMI，将预训练 StyleGAN2 的生成器拆解为多个 block，在中间特征层逐层优化（配合 $\ell_1$ 球约束防止图像崩塌），把模型反演攻击的搜索空间从潜码扩展到中间特征，在 OOD 场景下攻击准确率提升高达 38.8%。

**[A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](others/a_framework_for_efficient_model_evaluation_through_stratific.md)**

:   提出一个统计框架，通过分层（stratification）、采样设计（sampling）和估计器（estimation）三个组件的协同设计，在仅标注少量测试样本的情况下精确估计CV模型准确率，最高可实现10倍的效率增益（即用1/10的标注量达到同等精度）。

**[A High-Quality Robust Diffusion Framework for Corrupted Dataset](others/a_highquality_robust_diffusion_framework_for_corrupted_datas.md)**

:   提出 RDUOT 框架，首次将非平衡最优传输(UOT)融入扩散模型(DDGAN)中，通过学习 $q(x_0|x_t)$ 而非 $q(x_{t-1}|x_t)$ 来有效过滤训练数据中的离群值，在污染数据集上实现鲁棒生成的同时，在干净数据集上也超越了 DDGAN 基线。

**[ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](others/abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)**

:   提出首个无需样例图像即可同时计数图像中多类未知物体的方法ABC123，通过ViT回归多通道密度图+匈牙利匹配训练+SAM示例发现机制，在自建合成数据集MCAC上大幅超越需要样例的方法，且能泛化到FSC-147真实数据集。

**[Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](others/action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](others/actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)**

:   提出 ActionSwitch——首个无需类别信息即可检测流式视频中重叠动作实例的在线时序动作定位（On-TAL）框架，核心将多动作检测建模为有限状态机的状态分类问题，并辅以 conservativeness loss 减少碎片化误检，在 THUMOS14、FineAction、Epic-Kitchens 100 等数据集上在 OAD 扩展方法中达到 SOTA。

**[Active Generation for Image Classification](others/active_generation_for_image_classification.md)**

:   ActGen将主动学习思想引入扩散模型数据增强，通过识别分类器的错分样本并以注意力掩码引导+梯度对抗引导生成"难样本"，仅用10%的合成数据量即超越了此前需要近等量合成数据的方法，在ImageNet上ResNet-50获得+2.26%的精度提升。

**[Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](others/adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)**

:   针对细粒度跨视角定位模型在新区域部署时精度下降的问题，提出基于知识自蒸馏的弱监督学习方法——通过模式化伪GT生成、粗粒度监督和离群值过滤三个策略，仅使用目标区域的地面-航拍图像对（无需精确GT），即可在VIGOR和KITTI上将定位误差降低12%~20%。

**[Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](others/adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)**

:   提出自适应高频Transformer（AdaFreq），通过频域混合增强、目标感知的高频token动态选择、特征均衡损失三大策略，将高频信息（毛皮纹理、轮廓边缘等）统一用于多种野生动物的重识别，在8个跨物种数据集上超越现有ReID方法。

**[DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs](others/densenets_reloaded_paradigm_shift_beyond_resnets_and_vits.md)**

:   重新审视 DenseNet 的密集拼接连接（concatenation shortcut），通过系统性现代化改造（加宽减深、现代化 block、扩大中间维度、更多 transition 层等），提出 RDNet（Revitalized DenseNet），在 ImageNet-1K 上超越 Swin Transformer、ConvNeXt、DeiT-III，证明了拼接连接作为一种被低估的范式具有强大潜力。

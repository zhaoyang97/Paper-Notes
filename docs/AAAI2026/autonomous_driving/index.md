<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🤖 AAAI2026** · 共 **13** 篇

**[Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)**

:   提出基于 Beta 分布学习的地理空间深度学习框架，利用多尺度卫星图像预测道路致命事故风险的完整概率分布（而非点估计），在 Recall 上提升 17-23%，并通过分布形状自然表达不确定性。

**[Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[CompTrack: 信息瓶颈引导的低秩动态Token压缩用于点云跟踪 (Oral)](comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)**

:   针对LiDAR点云3D单目标跟踪中的"双重冗余"问题（空间冗余：大量背景噪声；信息冗余：前景中大量不具区分性的平面点），提出SFP前景预测器+IB-DTC信息瓶颈引导动态Token压缩两个模块，在KITTI/nuScenes/Waymo上达到SOTA，90 FPS实时运行（比P2P快1.4倍）。

**[FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)**

:   提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

**[FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](fq-petr_fully_quantized_position_embedding_transformation_fo.md)**

:   首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

**[LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)**

:   提出LiDARCrafter，首个专用于LiDAR的4D生成世界模型，通过Text2Layout（LLM解析文本→场景图→三分支扩散生成4D布局）→Layout2Scene（Range-image扩散生成高保真单帧）→Scene2Seq（自回归warp+扩散生成时序一致的序列）三阶段流程，在nuScenes上取得SOTA。

**[MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)**

:   提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

**[SPARC: 用单一策略驾驶100辆未见车辆的OOD泛化](out-of-distribution_generalization_with_a_sparc_racing_100_u.md)**

:   提出 SPARC（Single-Phase Adaptation for Robust Control），将 RMA 的两阶段上下文编码与历史适应统一为单阶段训练，在 Gran Turismo 7 高保真赛车模拟器中用单一策略驾驶100+未见车辆实现SOTA OOD泛化性能。

**[PriorDrive: 用统一向量先验增强在线HD地图构建](priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)**

:   提出 PriorDrive 框架，通过 Unified Vector Encoder (UVE) 和 Hybrid Prior Representation (HPQuery) 将多种向量化先验地图（SD地图、旧HD地图、历史预测地图）统一编码并集成到各种在线建图模型中，在 nuScenes 上 mAP 提升 14.3，兼容 query-based 和 non-query-based 两类建图架构。

**[ReflexDiffusion: 反思增强的高侧向加速度自动驾驶轨迹规划](reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)**

:   提出 ReflexDiffusion，在扩散模型推理阶段引入物理感知的反思机制，通过梯度注入强化曲率-速度-加速度耦合约束（a_y = κv²），在 nuPlan 高侧向加速度长尾场景中驾驶分数提升 14.1%，架构无关可直接部署到现有扩散规划器。

**[Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)**

:   提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

**[VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)**

:   VILTA 将 VLM（Gemini-2.5-Flash）直接嵌入自动驾驶 RL 训练循环中，通过"Vision-Language-Editing"（VLE）范式让 VLM 编辑周围车辆的未来轨迹来生成具有挑战性的危险场景，训练出的驾驶策略在 CARLA 挑战场景中路线完成率提升 13.3%、碰撞率降低 28.5%。

**[Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction (Oral)](visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)**

:   首次将 3D 高斯 Splatting 作为多智能体协同感知的通信媒介和中间表征，利用高斯基元的刚体变换可解析性和稀疏性，通过高斯打包（ROI 裁剪+刚体变换）和跨智能体邻域融合模块，实现了高效且可解释的视觉协同语义占用预测。

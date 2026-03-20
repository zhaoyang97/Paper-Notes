<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🤖 AAAI2026** · 共 **10** 篇

**[A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)**

:   提出基于多目标A*搜索的label-setting方法（Pr-A*），在初始电量未知时高效求解电动车能耗最优路径（profile搜索），通过profile支配关系剪枝避免传统方法中复杂的profile合并操作，在大规模路网上性能接近已知初始电量的标准A*搜索。

**[Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)**

:   提出基于 Beta 分布学习的地理空间深度学习框架，利用多尺度卫星图像预测道路致命事故风险的完整概率分布（而非点估计），在 Recall 上提升 17-23%，并通过分布形状自然表达不确定性。

**[Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)**

:   提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

**[FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](fq-petr_fully_quantized_position_embedding_transformation_fo.md)**

:   首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

**[MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)**

:   提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

**[SPARC: 用单一策略驾驶100辆未见车辆的OOD泛化](out-of-distribution_generalization_with_a_sparc_racing_100_u.md)**

:   提出 SPARC（Single-Phase Adaptation for Robust Control），将 RMA 的两阶段上下文编码与历史适应统一为单阶段训练，在 Gran Turismo 7 高保真赛车模拟器中用单一策略驾驶100+未见车辆实现SOTA OOD泛化性能。

**[Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)**

:   提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

**[VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)**

:   VILTA 将 VLM（Gemini-2.5-Flash）直接嵌入自动驾驶 RL 训练循环中，通过"Vision-Language-Editing"（VLE）范式让 VLM 编辑周围车辆的未来轨迹来生成具有挑战性的危险场景，训练出的驾驶策略在 CARLA 挑战场景中路线完成率提升 13.3%、碰撞率降低 28.5%。

**[Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction (Oral)](visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)**

:   首次将 3D 高斯 Splatting 作为多智能体协同感知的通信媒介和中间表征，利用高斯基元的刚体变换可解析性和稀疏性，通过高斯打包（ROI 裁剪+刚体变换）和跨智能体邻域融合模块，实现了高效且可解释的视觉协同语义占用预测。

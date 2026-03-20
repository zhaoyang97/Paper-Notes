---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📷 CVPR2026** · 共 **24** 篇

**[A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)**

:   提出 A4VL，一个多 Agent 感知-行动探索联盟框架用于高效长视频推理——多个 VLM Agent 在多轮循环中进行查询特定的感知线索提取（找到最相关的视频片段）和行动探索（生成答案、交叉审查、达成共识或重新探索），在 5 个 VideoQA 基准上超越 18 个现有 VLM 和 10 个长视频推理专用方法，且推理延迟显著更低。

**[AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](assistmimic_physics_grounded_humanoid_assistance.md)**

:   提出 AssistMimic，一个多智能体 RL 框架，联合训练辅助者和被辅助者的物理仿真策略来模仿人-人接触式辅助动作（如扶人站起），是首个在标准基准上成功跟踪力交换辅助运动的方法。

**[BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_ma.md)**

:   提出可制造性指标的二维分类法（配置依赖性 x 可行性/复杂度）和首个钣金弯曲合成 CAD 数据集 BenDFM（20,000 零件），基准测试显示图结构表示（UV-Net）优于点云（PointNext），且配置相关任务仍是难点。

**[Bounds on Agreement between Subjective and Objective Measurements](bounds_on_agreement_between_subjective_and_objecti.md)**

:   从投票的基本统计性质出发，推导了主观MOS与任意客观评估指标间PCC上界和MSE下界的解析表达式，并提出基于二项分布的投票模型BinoVotes/BinoMOS，为无投票方差数据的场景提供性能天花板估计。

**[CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](ciice_intrinsic_concept_extraction_compositional.md)**

:   提出CI-ICE新任务和HyperExpress方法，利用双曲空间的层次建模能力提取可组合的物体级/属性级内在概念，通过Horosphere投影模块保证概念嵌入空间的可组合性。

**[U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)**

:   提出TextUnlock方法，通过训练轻量MLP将任意冻结视觉分类器的特征投射到文本嵌入空间（同时保持原分类器分布不变），无需CLIP、无需标注、无需训练线性探针，即可将任何legacy分类器转化为可解释的概念瓶颈模型——在40+架构上测试，超越甚至有监督的CLIP基CBM。

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correcti.md)**

:   通过宏观收敛态、微观梯度动力学和信息论三个层次，严格证明了即使给定完美噪声转移矩阵，前向校正（FC）仍不可避免地塌缩到与无校正相同的次优水平，根本原因在于有限样本下的记忆化和噪声信道的信息损失。

**[DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)**

:   通过 Dirichlet 先验增强（DirPA），在少样本训练阶段主动模拟真实世界长尾类别分布，从而消除人工平衡训练集与自然不平衡测试分布之间的先验偏移，提升作物分类的鲁棒性。

**[DPCache: 去噪即路径规划——免训练扩散模型加速](dpcache_denoising_path_planning_diffusion_accel.md)**

:   将扩散采样加速形式化为全局路径规划问题，通过Path-Aware Cost Tensor量化路径依赖的跳步误差，用动态规划选出最优关键时间步序列，在FLUX上实现4.87×加速且ImageReward反超全步基线。

**[Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras](drocc_depth_region_guided_3d_occupancy.md)**

:   针对视角变换几何不对齐和语义类别空间各向异性不平衡，提出深度引导双投影视角变换器（D²-VFormer）利用MoGe-2构建非空体素掩码，和区域引导专家Transformer（R/R²-EFormer）自适应分配空间模型容量，BEVDet4D上提升7.43% mIoU。

**[ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](enhancing_outofdistribution_detection_with_extende.md)**

:   诊断LogitNorm的特征坍缩问题(维度坍缩+原点坍缩)，提出ELogitNorm——用到决策边界的平均距离(而非特征范数)做自适应温度缩放，无超参数、兼容所有post-hoc OOD检测方法——CIFAR-10上far-OOD AUROC提升10.48%(SCALE)，ImageNet-1K上FPR95从51.45%降至27.74%，同时改善分类精度和ECE校准。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_a.md)**

:   基于GRD-Net改进的GAN+密集瓶颈残差自编码器（DRAE），在制药BFS产线上实现半监督异常检测，用281万训练patch在500ms时间槽内完成60个patch的推理（0.17ms/patch），达到97.62%平衡准确率和96.38%的逐运行验证精度。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)**

:   MXNorm 提出将 RMSNorm 与 MXFP 量化融合：利用 MXFP 量化过程中已经计算好的 block absmax 来近似 RMS 值，从而省掉单独的归一化 reduction 操作，在 Llama 3 最高 8B 参数的预训练中保持训练精度，同时在 GB200 上实现最高 2.4 倍的 kernel 加速。

**[On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_imageinimage_steg.md)**

:   揭示了基于可逆神经网络(INN)的图像隐写方案存在严重安全漏洞：嵌入过程本质上是一种混合过程，可通过ICA进行盲源分离，仅用8维特征+SVM即可达到84.6%检测率，而传统SRM+SVM更是达到99%以上。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolutio.md)**

:   提出 StEvo-Bench 基准测试，通过在演化过程中插入遮挡或让相机"看向别处"来检验视频世界模型能否将状态演化与观测解耦，揭示了当前模型（包括 Veo 3、Sora 2 Pro 等）的任务成功率不到 10%，暴露了严重的"演化停止"和"不一致性"问题。

**[Proof-of-Perception: 带组合共形保证的工具使用多模态推理](pop_proof_of_perception_conformal_reasoning.md)**

:   提出PoP框架将多模态推理建模为可执行DAG——每个感知/逻辑节点输出共形预测集提供逐步校准的不确定性，控制器在预算约束下按需调用更多工具扩展计算，在文档/图表/多图QA上优于CoT/ReAct/PoT基线。

**[Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](rethinking_snn_online_training_and_deployment_grad.md)**

:   提出HD-LIF(混合驱动LIF)脉冲神经元模型族，通过在阈值上下区域采用不同脉冲计算机制，理论证明其梯度可分离性和对齐性，解决SNN在线训练的前后向传播不一致问题，同时实现学习精度、内存复杂度和功耗的全阶段优化——以10×参数压缩、11×功耗降低和30% NOPs节省达到CIFAR-100上78.61%精度。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)**

:   建立学习-观测框架，在真实风洞PIV数据上系统比较Kriging、UNet、ViTAE和CWGAN四种方法从5-30个稀疏传感器重建屋顶全风场的能力，发现混合风向训练下DL一致优于Kriging（SSIM提升18-34%），CWGAN在鲁棒性上最优。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embeddingbased_approach_for_abini.md)**

:   SHREC利用谱嵌入技术从2D冷冻电镜投影图像直接恢复螺旋分子的投影角度（无需螺旋对称参数先验），通过证明螺旋片段投影构成一维闭合流形(同胚于圆)实现角度恢复，在TMV、VipA/VipB和MakA三个公开数据集上实现接近发表水平的高分辨率重建(3.66Å–8.23Å)。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_largescale_multimodal_dataset_for_cad.md)**

:   构建SldprtNet——含242K+工业CAD零件的大规模多模态数据集，每个样本包含.sldprt/.step模型、7视角合成图、参数化建模脚本(13种命令无损编解码)和Qwen2.5-VL生成的自然语言描述，baseline实验验证多模态输入(图+文)在CAD生成上优于纯文本输入。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH通过超网络驱动的任务向量混合(TVM)+稀疏聚合实现跨癌种WSI预后知识迁移，在13个TCGA数据集上C-Index平均0.6949（+5.14% vs 癌种特定学习，+2.01% vs ROUPKT）。

**[TrajTok: 学习轨迹Token实现更好的视频理解](trajtok_trajectory_token_video_understanding.md)**

:   提出TrajTok——首个端到端可微的轨迹视频tokenizer，通过隐式时空聚类将视频编码为物体轨迹token，无需外部分割/跟踪管线，在分类、检索和长视频QA上全面超越patch-based方法。

**[EB-JDAT: Energy-based Joint Distribution Adversarial Training](your_classifier_can_do_more_towards_balancing_the.md)**

:   通过能量景观分析揭示AT和JEM的互补性(AT缩小clean-adv能量差→鲁棒性；JEM缩小clean-generated能量差→生成+精度)，提出EB-JDAT建模联合分布p(x,x̃,y)，用min-max能量优化对齐三种数据的能量分布——CIFAR-10上鲁棒性68.76%(AutoAttack, 超SOTA AT +10.78%)，同时保持90.39%清洁精度和竞争力的生成质量(FID=27.42)。

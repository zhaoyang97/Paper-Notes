<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**📷 CVPR2026** · 共 **40** 篇

**[A protocol for evaluating robustness to H&E staining variation in computational pathology models](a_protocol_for_evaluating_robustness_to_he_stainin.md)**

:   提出三步评估协议（选参考染色条件→表征测试集染色属性→模拟染色条件推理），系统量化306个MSI分类模型对H&E染色差异的鲁棒性，发现鲁棒性与分类性能呈弱负相关(r=-0.28)，高性能不代表高鲁棒性。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semisupervised_framework_for_breast_ultrasound_s.md)**

:   通过外观描述驱动VLM免训练生成伪标签，再由双教师不确定性融合+反向对比学习细化，仅2.5%标注即可逼近全监督性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)**

:   借鉴基础模型范式，先在约4000例fastMRI多对比度脑MRI上预训练扩散模型，再用20例目标域数据微调，实现临床中风MRI的高质量加速重建，盲审读片证明2×加速下非劣于标准诊疗。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_hi.md)**

:   提出SFDA-DeP方法，受机器遗忘启发，将源自由域适应建模为迭代识别过度预测类的不确定样本并选择性降低其置信度的过程，同时联合训练像素级分类器恢复定位判别力，在跨器官/跨中心病理基准上显著优于SFDA baselines。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_th.md)**

:   在仅206例标注CT中，通过patch-based MIM预训练3D U-Net + VDETR顶点RPE + 半监督一致性正则化的两阶段框架，将3D创伤检测mAP@0.50从26.36%提升至56.57%（验证集），同时冻结编码器的7类分类达94.07%准确率。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_generalpurpose_vision_models_all_we_need_for_2.md)**

:   在统一训练和评估协议下对比11个专用医学分割架构(SMA)和通用视觉模型(GP-VM)，发现GP-VM在三个异质医学数据集上超越大多数SMA，且Grad-CAM分析表明GP-VM无需领域特定设计即可捕获临床相关结构。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)**

:   在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)**

:   系统对比 15 种 CNN 变体在卵巢癌组织病理图像五分类上的表现，选出 InceptionV3-A（ReLU）达 94% 综合指标后，用 LIME/SHAP/Integrated Gradients 三种 XAI 方法解释其决策。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_languageimage.md)**

:   提出双向视觉-语言融合（BMF）和增强一致性（IAC）两个模块，让文本和图像特征可以相互修正，在标注极度稀缺（1%）和图像退化（低剂量CT噪声/运动模糊）场景下仍保持分割鲁棒性。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](bridging_the_skill_gap_in_clinical_cbct_interpreta.md)**

:   构建了覆盖55种口腔疾病的7,408例大规模配对CBCT-报告数据集，开发双语报告生成系统CBCTRepD，并通过多层级临床评估证明其可帮助不同经验水平的放射科医生提升报告质量。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_clip_adaptation_curvature_data_selection.md)**

:   从数据中心视角重新审视 CLIP 领域适配，提出 CHIPS，为每个图文对计算融合曲率感知牛顿对齐（忠实性）、JL sketching压缩曲率估计（可扩展性）、可学习性+领域相关性权重（保留性）三因素的效用分数，用30%数据匹配全数据集CPT、10%数据超越50%数据CPT，在17个医学+31个通用基准上达到选择SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modal.md)**

:   将缺失模态下的鲁棒性问题重新定义为决策级专家一致性控制，提出双分支一致性学习（全局MEC+区域REC），并通过轻量门网络将一致性分数转化为模态可靠性权重用于融合。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mambabased_decoder_with.md)**

:   提出以解码器为核心的 Deco-Mamba 网络，用 Co-Attention Gate 双向融合编解码器特征、视觉状态空间模块（VSSM）建模长程依赖、可变形卷积恢复细节，并引入窗口化分布感知 KL 散度深度监督，在 7 个医学分割基准上以中等复杂度达到 SOTA。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_le.md)**

:   提出 ScleraGluNet 框架，通过五个注视方向的巩膜血管图像，结合多分支 CNN + MRFO 特征优化 + Transformer 跨视角融合，实现 93.8% 三分类精度和 MAE=6.42 mg/dL 的空腹血糖估计。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)**

:   在 8 个标注者划分的全景口腔 X 光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）在第三磨牙-下颌管重叠二分类任务上的表现，验证 FL 作为隐私保护替代方案的可行性。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segment.md)**

:   构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11041 例扫描、59831 个掩模），并提出 SegAnyPET 基础模型，实现基于 prompt 的 3D 全身 PET 通用可交互分割，在多中心、多示踪剂、多疾病场景下展现强 zero-shot 泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)**

:   将 MRI 脑肿瘤分类任务分解为 NNMF 特征提取 → 统计特征筛选 → 轻量 CNN 分类 → 特征空间扩散净化四阶段流水线，在 AutoAttack 下将鲁棒精度从基线 0.5% 提升到 59.5%。

**[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](eda_arbitrary_noise_diffusion_design_space.md)**

:   提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotationequivarian.md)**

:   提出EquivAnIA——基于Cake小波和Ridge滤波器的频谱方法，通过方向滤波器在傅里叶域计算角度能量分布，实现对数值旋转严格等变的各向异性图像分析，在合成和真实图像上一致优于传统角度功率谱密度方法。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)**

:   在 ConvNeXt 骨干上构建注意力 MIL 模型，并通过梯度反转层（GRL）对抗性地消除扫描表征中的性别信息，再配合 focal loss、子群过采样和 5-fold 集成，实现胸部 CT 四类肺疾病的公平诊断。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器、部分个性化融合解码器和多锚点交叉注意力校准，同时获得最优全模态全局模型和各客户端缺失模态个性化模型。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_ca.md)**

:   首次系统定义基于视频的癫痫发作预测任务，提出两阶段跨物种迁移学习框架——先在啮齿类癫痫视频上自监督预训练 VideoMAE，再在人类发作前视频上少样本微调——在纯视频设定下实现超过 72% 的均衡准确率。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)**

:   提出基于多异构图 (MHG) 的 GIIM 框架，通过四类边关系建模同一病灶跨视图动态变化和不同病灶间空间关联，并设计四种缺失视图填充策略，在 CT/MRI/乳腺 X 光三种模态上均显著优于现有方法。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)**

:   提出首个公开三模态青光眼数据集 GLEAM（SLO 眼底图 + 环乳头 OCT + 视野偏差图，标注四个疾病阶段），以及层级注意力掩码建模 (HAMM) 框架，将跨模态自监督表示学习聚焦在编码器端，实现多模态青光眼精准分类。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multimodal_learning_for.md)**

:   提出域保形界(DCB)理论框架量化域间因果因子差异，并据此设计GenEval——通过知识精炼+MedGemma-4B LoRA微调，将人类专家领域知识整合到VLM中实现单源域泛化，在8个DR和2个SOZ数据集上显著超越SOTA。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)**

:   提出"检测即加噪"范式取代传统"检测即去噪"——通过DDIM反转将图像映射到潜在噪声空间，仅用3步推理判断偏离先验分布的程度作为异常分数，无需重建，实现SOTA精度的同时推理速度达88 FPS（比OmiAD快2倍+）。

**[Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](markerbased_3d_reconstruction_of_aggregates_with_a.md)**

:   骨料作为建筑材料装配体的主要骨架，是各种建筑、交通运输领域的重要功能部件。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)**

:   提出MIL-PF框架，将冻结的基础视觉编码器（DINOv2/MedSigLIP）与仅40k参数的轻量级MIL聚合头结合，通过预计算特征+双流（全局组织上下文+局部病变注意力）聚合，在大规模乳腺X线分类任务上以极低训练成本达到SOTA性能。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)**

:   提出RICE-NET，一种多模态3D ResNet-18模型，融合纵向T1加权MRI数据与放射治疗剂量分布图，在92例胶质母细胞瘤患者队列上实现F1=0.92的RICE vs 肿瘤复发自动分类，消融实验揭示放疗剂量图是最具信息量的单模态输入。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kine.md)**

:   提出ERBA（Enzyme-Reaction Bridging Adapter），将酶动力学参数预测重新建模为与催化机制对齐的分阶段条件化问题——先通过MRCA注入底物信息捕捉分子识别，再通过G-MoE融合活性位点3D几何信息建模构象适应，并用ESDA做分布对齐保持PLM先验——在三个动力学指标上全面超越现有SOTA。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structureguided_latent_diffusion_for_mu.md)**

:   提出MSG-LDM，一个基于潜在扩散模型的多模态MRI翻译框架，通过在潜空间中显式解耦风格和结构信息，结合高频注入（HFIB）、多模态结构特征融合（MMSF）和多尺度结构增强（MSSE）模块提取模态无关的完整结构先验来引导扩散去噪，在BraTS2020和WMH数据集上超越现有方法。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_de.md)**

:   将软件设计模式（Singleton + Batch Processing）融入Python自动化流程，使口腔癌病变检测的推理速度相比传统RPA平台（UiPath/Automation Anywhere）提升60-100倍。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle_rl_radiology_report_generation.md)**

:   提出 OraPO, 一种结合 GRPO 和 DPO 的自适应混合 RL 框架, 用于数据高效的放射学报告生成: 通过 Zero-Reward Rate 检测动态切换 GRPO 和 DPO, 加上 FactScore-based 临床事实级奖励, 仅用 1K 样本 (对比基线 227K) 在 CheXpert Plus 和 MIMIC-CXR 上取得 SOTA 的临床 F1 (0.341/0.357).

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototypebased_knowledge_guidance_for_finegrained.md)**

:   提出ProtoSR，通过LLM驱动的管道从22.7万篇MIMIC-CXR自由文本报告中挖掘模板对齐的视觉原型知识库，并设计原型条件化迟融合模块将检索到的原型证据作为logit残差注入层级式结构化报告模型，在Rad-ReStruct基准上达到SOTA，在细粒度属性问题（L3）上提升最为显著（+72.1%相对提升）。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_wi.md)**

:   通过将SIENA纵向脑萎缩管线中的经典颅骨剥离(BET2)和组织分割(FAST)模块替换为深度学习方案(SynthStrip/SynthSeg)，在ADNI和PPMI两个大队列上显著增强了脑体积变化百分比(PBVC)与临床疾病进展的关联性，并将扫描顺序误差降低高达99.1%。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_selforganizing_domainadapt.md)**

:   提出Residual SODAP框架，在无任务ID、无数据存储的域增量学习中，联合解决表示适应（α-entmax稀疏prompt选择+残差聚合）和分类器保持（统计伪特征重放+知识蒸馏），在DR、皮肤癌和CORe50三个基准上达到SOTA。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)**

:   提出即插即用的SCDL框架，通过学习类条件代理分布(双向对齐CDBA)+语义锚约束(SAC)来消除半监督医学图像分割中的长尾偏差，在AMOS 5%标签下DSC提升+11.62%。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)**

:   提出SemiTooth——多教师多学生半监督框架+更严格加权置信度约束(SWC)，用于多源CBCT牙齿分割，在新构建的MS3Toothset上mIoU达76.67%、Dice 85.69%，超越SOTA CMT(76.14%)。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH通过超网络驱动的任务向量混合(TVM)+稀疏聚合实现跨癌种WSI预后知识迁移，在13个TCGA数据集上C-Index平均0.6949（+5.14% vs 癌种特定学习，+2.01% vs ROUPKT）。

**[CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_fullstack_scanning_of_brain_mri_via_imputi.md)**

:   CodeBrain将脑MRI多模态补全(any-to-any imputation)重新定义为区域级全栈量化码预测问题：Stage I用有限标量量化(FSQ)将完整MRI集编码为紧凑code map + 模态无关公共特征，Stage II从不完整模态预测code map(用grading loss保持量化空间平滑性)，在IXI和BraTS 2023上超越5种SOTA方法，生成的模态可接近真实数据的脑肿瘤分割性能。

---
title: >-
  ECCV2026 医学图像论文汇总 · 14篇论文解读
description: >-
  14篇ECCV2026的医学图像方向论文解读，涵盖医学影像、语义分割、对抗鲁棒、布局/合成、超分辨率、图像恢复等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "语义分割"
  - "对抗鲁棒"
  - "布局/合成"
  - "超分辨率"
  - "图像恢复"
item_list:
  - u: "3d_field_of_junctions_a_noise-robust_training-free_structural_prior_for_volumetr/"
    t: "3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems"
  - u: "a_mechanism-driven_theory_of_phase_transitions_in_active_learning/"
    t: "A Mechanism-Driven Theory of Phase Transitions in Active Learning"
  - u: "active_view_selection_with_perturbed_gaussian_ensemble_for_tomographic_reconstru/"
    t: "基于扰动高斯集合的层析成像主动视角选择"
  - u: "biomedvr_confusion-aware_mixture-of-prompt_experts_for_biomedical_visual_reprogr/"
    t: "BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming"
  - u: "brainfibre_a_foundation_model_via_information_decomposition_for_brain_microstruc/"
    t: "BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure"
  - u: "brainriem_riemannian_prototype_learning_for_source-free_cross-site_brain_network/"
    t: "BrainRiem: Riemannian Prototype Learning for Source-Free Cross-Site Brain Network Diagnosis"
  - u: "dual-prior_guided_null-space_learning_with_mixture-of-splines_for_arbitrary_medi/"
    t: "Dual-Prior Guided Null-Space Learning with Mixture-of-Splines for Arbitrary Medical Slice Super-Resolution"
  - u: "from_reconstruction_to_decision_a_post-encoder_plug-in_adapter_for_curvilinear_s/"
    t: "From Reconstruction to Decision: A Post-Encoder Plug-in Adapter for Curvilinear Segmentation"
  - u: "histopathology_multi-modal_embedding_for_pathology_composed_retrieval/"
    t: "Histopathology Multi-modal Embedding for Pathology Composed Retrieval"
  - u: "medcagd_context-aware_gated_decoder_for_efficient_medical_image_segmentation/"
    t: "MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation"
  - u: "ngps_structure-preserving_self-supervised_denoising_via_neighbor-guided_patch_sa/"
    t: "NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling"
  - u: "render-fm_feedforward_model_for_real-time_photorealistic_volumetric_rendering/"
    t: "Render-FM: Feedforward Model for Real-time Photorealistic Volumetric Rendering"
  - u: "skin_r1_dermatological_diagnosis/"
    t: "Skin-R1: Clinical Knowledge-Guided Dermatological Diagnosis Using Vision-Language Models"
  - u: "taxomil_taxonomy-constrained_learning_for_hierarchical_whole_slide_image_analysi/"
    t: "TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🎞️ ECCV2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (172)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (86)](../../ICLR2026/medical_imaging/index.md) · [🧪 ICML2026 (28)](../../ICML2026/medical_imaging/index.md) · [🤖 AAAI2026 (75)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (77)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×5 · 语义分割 ×2

**[3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems](3d_field_of_junctions_a_noise-robust_training-free_structural_prior_for_volumetr.md)**

:   提出 3D Field of Junctions (3D FoJ)，将 2D FoJ (ICCV 2021) 的显式几何参数化推广到三维体积——用三平面交于顶点的 junction 对每个 3D patch 建模，逐 patch 离散坐标下降初始化后全局联合梯度精炼，无需任何训练数据即可在极低 SNR 下保留锐利边角，并可作即插即用的近端正则项嵌入任意体积逆问题（低剂量 CT、冷冻电镜、点云去噪均验证有效）。

**[A Mechanism-Driven Theory of Phase Transitions in Active Learning](a_mechanism-driven_theory_of_phase_transitions_in_active_learning.md)**

:   本文提出一种机制驱动的主动学习相变理论：将 PAC 泛化界的四个分量（经验风险、分布差异、模型复杂度、置信项）重新解释为动态交互项，证明其主导地位沿标注轨迹必然发生切换，并用可测代理指标和分段回归识别出"数据驱动—过渡—模型驱动"三阶段分类法，解释了为什么代表性、覆盖性、不确定性策略在不同预算阶段各擅胜场。

**[基于扰动高斯集合的层析成像主动视角选择](active_view_selection_with_perturbed_gaussian_ensemble_for_tomographic_reconstru.md)**

:   针对稀疏CT中X射线主动视角选择难题，提出基于辐射3DGS的扰动高斯集合框架：通过随机扰动低密度Gaussian基元的密度参数构建轻量模型集合，用候选视角投影的结构相似性（SSIM）方差量化认知不确定性，选择最能暴露几何伪影的视角作为下一最佳采集角度。

**[BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming](biomedvr_confusion-aware_mixture-of-prompt_experts_for_biomedical_visual_reprogr.md)**

:   BioMedVR 首次将视觉重编程（Visual Reprogramming）引入医学影像领域，通过双专家解耦的混淆感知 MoPE 架构——正专家负责主类判别、负专家借助 LLM 生成的混淆属性抑制易混淆类别，配合一个带边距的混淆抑制损失（CS Loss）显式拉大正负分数间距——在 11 个医学数据集（9 种成像模态）和 7 个自然图像基准上一致超越现有 VR 和提示学习方法，参数量仅为全模型微调的 1/500。

**[BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure](brainfibre_a_foundation_model_via_information_decomposition_for_brain_microstruc.md)**

:   BrainFIBRE 是首个面向脑组织微结构的基础模型，把 NODDI 派生的三张微结构图（NDI/ODI/FWF）当成三个模态，用「自监督偏信息分解 SPID + 反事实候选构造 CCC」在 UK Biobank 5.5 万人上预训练一个混合专家网络，无标签地把三模态的独有/冗余/协同信息拆开，在跨种族多队列的年龄/性别/脑血管与神经退行标记/认知等下游任务上全面刷到 SOTA，且专家权重可解释。

**[BrainRiem: Riemannian Prototype Learning for Source-Free Cross-Site Brain Network Diagnosis](brainriem_riemannian_prototype_learning_for_source-free_cross-site_brain_network.md)**

:   BrainRiem 提出在黎曼流形上通过双层优化学习紧凑的脑网络原型（prototype），实现无需源数据访问的跨站点脑功能连接诊断，在 ABIDE 和 REST-meta-MDD 多站点基准上大幅超越现有方法，且学得的原型具有生物可解释性。

**[Dual-Prior Guided Null-Space Learning with Mixture-of-Splines for Arbitrary Medical Slice Super-Resolution](dual-prior_guided_null-space_learning_with_mixture-of-splines_for_arbitrary_medi.md)**

:   本文将医学切片任意倍率超分辨率重新建模为受约束的逆问题，通过零-值空间分解实现测量一致性硬约束，并在零空间内用混合B样条专家按局部解剖特征自适应分配连续性阶数，严格保证已采集切片不变的同时生成解剖合理的插值细节。

**[From Reconstruction to Decision: A Post-Encoder Plug-in Adapter for Curvilinear Segmentation](from_reconstruction_to_decision_a_post-encoder_plug-in_adapter_for_curvilinear_s.md)**

:   本文提出 PEPA，一个轻量级后编码器即插即用适配器，通过目标条件蛇形上采样（TCSU）和目标自适应可微二值化（TADT）两个模块，在冻结基础模型编码器的条件下显著提升血管、裂缝等曲线结构分割的拓扑连续性，仅增加 0.26M 参数即可在五个医学与工业基准上取得平均 +2.6% IoU 和 +2.8% clDice 的提升。

**[Histopathology Multi-modal Embedding for Pathology Composed Retrieval](histopathology_multi-modal_embedding_for_pathology_composed_retrieval.md)**

:   HOMIE 提出一个模型无关的两阶段适配框架，将任意生成式 MLLM 转化为病理检索专家：第一阶段用纯文本 LoRA 训练让 LLM 学会判别式度量空间（解决任务错配），第二阶段通过原生分辨率处理、染色增强和渐进知识课程注入病理形态学先验（解决领域错配），在作者新提出的 PCR 组合检索基准上，2B 参数版本即大幅超越 7B 专用病理 MLLM 和双编码器模型。

**[MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation](medcagd_context-aware_gated_decoder_for_efficient_medical_image_segmentation.md)**

:   MedCAGD 提出一种以解码器为中心的医学图像分割框架，通过上下文感知门控解码器系统性地调控跳跃连接融合和多尺度上下文聚合，在 11 个医学图像分割基准上以 30.60M 参数和 5.0 GFLOPs 的轻量计算代价，一致超越 CNN、Transformer、Mamba 及 SOTA 解码器方法。

**[NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling](ngps_structure-preserving_self-supervised_denoising_via_neighbor-guided_patch_sa.md)**

:   NGPS 提出一种轻量级邻片自监督去噪框架：对层间错位区域，在噪声衰减引导图上做局部 patch 匹配，但训练目标从原始噪声邻片的匹配坐标取像素值，从而实现"结构匹配与信号检索解耦"，无需密集形变场估计，在低剂量 CT 和 MRI 上一致提升保真度与结构敏感指标。

**[Render-FM: Feedforward Model for Real-time Photorealistic Volumetric Rendering](render-fm_feedforward_model_for_real-time_photorealistic_volumetric_rendering.md)**

:   Render-FM 提出一个基于 nnU-Net 风格 3D U-Net 的前馈模型，在单次 2.8 秒前向传播中直接从 CT 体数据回归 6D 高斯泼溅（6DGS）参数，消除逐扫描优化瓶颈（约 500 倍加速），并通过解剖引导初始化（AGP）将分割掩码和传递函数作为结构与外观先验，桥接自然场景重建与医学体积渲染的领域鸿沟，支持实时交互渲染（328+ FPS）和零额外开销的器官组合可视化。

**[Skin-R1: Clinical Knowledge-Guided Dermatological Diagnosis Using Vision-Language Models](skin_r1_dermatological_diagnosis.md)**

:   Skin-R1 将**教科书依据的诊断推理轨迹**（层次化疾病分类 + 差分诊断）与 **GRPO 强化学习** 结合，让 VLM 先通过 SFT 学会专家推理模式、再通过层次化奖励设计将推理能力泛化到大规模稀疏标注数据，在多个皮肤病诊断基准上超越现有的 Med-VLM。

**[TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis](taxomil_taxonomy-constrained_learning_for_hierarchical_whole_slide_image_analysi.md)**

:   TaxoMIL 将全切片病理图像诊断重新建模为受临床分类体系约束的层次化文本生成任务，通过双头解码器同时生成粗粒度和细粒度诊断文本，并用层次对齐和图像-文本对齐损失函数确保视觉与标签表示空间符合医学分类树结构，在胃、乳腺和前列腺三个WSI数据集上全面超越现有方法。

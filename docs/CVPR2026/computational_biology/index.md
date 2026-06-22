---
title: >-
  CVPR2026 计算生物论文汇总 · 21篇论文解读
description: >-
  21篇CVPR2026的计算生物方向论文解读，涵盖生物分子、医学影像、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "生物分子"
  - "医学影像"
  - "多模态"
item_list:
  - u: "adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g/"
    t: "HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images"
  - u: "advancing_cancer_prognosis_with_hierarchical_fusion_of_genomic_proteomic_and_pat/"
    t: "Advancing Cancer Prognosis with Hierarchical Fusion of Genomic, Proteomic and Pathology Imaging Data from a Systems Biology Perspective"
  - u: "bigmint_biologically-guided_hierarchical_multimodal_integration_for_modeling_mul/"
    t: "BiGMINT: Biologically-guided Hierarchical Multimodal Integration for Modeling Multiple Compound Activities in Drug Discovery"
  - u: "bulk_rna-seq_guided_multi-modal_detection_of_anomalous_regions_in_human_cancer_v/"
    t: "Bulk RNA-seq Guided Multi-modal Detection of Anomalous Regions in Human Cancer via Spatial Transcriptomics"
  - u: "care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole/"
    t: "CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis"
  - u: "cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_/"
    t: "Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images"
  - u: "cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra/"
    t: "Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference"
  - u: "cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp/"
    t: "CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks"
  - u: "cryokraqen_kernel-regularized_annealing_for_quantized_embedding_networks_in_cryo/"
    t: "CryoKRAQEN: Kernel-Regularized Annealing for Quantized Embedding Networks in Cryo-EM Heterogeneous Reconstruction"
  - u: "cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and/"
    t: "cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold"
  - u: "deciphering_genotype-phenotype_mechanisms_from_high-content_profiling_via_knowle/"
    t: "Deciphering Genotype-Phenotype Mechanisms from High-Content Profiling via Knowledge-Guided Multi-modal Graph Learning"
  - u: "feast_fully_connected_expressive_attention_for_spatial_transcriptomics/"
    t: "FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics"
  - u: "from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima/"
    t: "From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images"
  - u: "hyperbolic_busemann_neural_networks/"
    t: "Hyperbolic Busemann Neural Networks"
  - u: "hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction/"
    t: "HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction"
  - u: "mmcp-gen_a_modality-extensible_diffusion_language_model_for_conditional_protein_/"
    t: "MMCP-GEN: A Modality-Extensible Diffusion Language Model for Conditional Protein Sequence Generation"
  - u: "multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics/"
    t: "Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics"
  - u: "multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_/"
    t: "Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation"
  - u: "predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce/"
    t: "Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling"
  - u: "stronger_normalization-free_transformers/"
    t: "Stronger Normalization-Free Transformers"
  - u: "trident_a_trimodal_cascade_generative_framework_for_drug_and_rna-conditioned_cel/"
    t: "TRIDENT: A Trimodal Cascade Generative Framework for Drug and RNA-Conditioned Cellular Morphology Synthesis"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**📷 CVPR2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/computational_biology/index.md)

🔥 **高频主题：** 生物分子 ×6 · 医学影像 ×2 · 多模态 ×2

**[HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)**

:   提出HINGE框架，首次将预训练的表达空间单细胞基础模型(sc-FM, CellFM)改装为组织学图像条件的空间基因表达生成器，通过恒等初始化的SoftAdaLN调制轻量注入视觉上下文、表达空间掩码扩散过程对齐预训练目标、warm-start课程稳定训练，在三个ST数据集上达SOTA并保持优越的基因共表达一致性。

**[Advancing Cancer Prognosis with Hierarchical Fusion of Genomic, Proteomic and Pathology Imaging Data from a Systems Biology Perspective](advancing_cancer_prognosis_with_hierarchical_fusion_of_genomic_proteomic_and_pat.md)**

:   HFGPI 把"基因 → 蛋白质 → 组织形态"的系统生物学级联显式建模成一条分层融合管线，用图感知交叉注意力刻画基因对蛋白的调控、用超图把蛋白连到病理 patch，在 5 个 TCGA 队列上把生存预测的平均 C-index 推到 0.753，超过所有 SOTA。

**[BiGMINT: Biologically-guided Hierarchical Multimodal Integration for Modeling Multiple Compound Activities in Drug Discovery](bigmint_biologically-guided_hierarchical_multimodal_integration_for_modeling_mul.md)**

:   BiGMINT 用「化学蛋白质组学信号引导高内涵成像（HCI）特征聚合 + 外积式跨模态融合 + 蛋白互作（PPI）先验做任务级信息共享」三段式层次化融合，把分子机制信号和细胞表型信号统一起来预测化合物活性，在两份各 ~99K / ~40K 化合物-成像对的大规模私有数据集上把平均 AUCROC 比最强单模态/多模态基线提升最多 10.0% / 4.2%，高性能任务覆盖最多翻倍。

**[Bulk RNA-seq Guided Multi-modal Detection of Anomalous Regions in Human Cancer via Spatial Transcriptomics](bulk_rna-seq_guided_multi-modal_detection_of_anomalous_regions_in_human_cancer_v.md)**

:   BRGMAR 用一个动态多关系图刻画空间转录组（ST）里 spot 间的空间近邻 + 基因相似关系，再用基于最优传输的“基因模块对齐”把患者级 bulk RNA-seq 的诊断信息迁移到 ST，最后与病理图像跨注意力融合，在 BRCA/HCC/ccRCC 三个数据集上把肿瘤异常区域检测的 AUC/F1 显著推到新高。

**[CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)**

:   提出 CARE，一种病理学 slide-level 基础模型，通过自适应区域生成器（ARG）将 WSI 划分为形态学相关的不规则区域（类似 NLP 中的词级 token），并结合 RNA/蛋白质表达谱的跨模态对齐进行两阶段预训练，仅用主流模型约 1/10 的数据即在 33 个下游任务上取得最优平均性能。

**[Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)**

:   提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。

**[Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)**

:   提出 SpaHGC，一种基于多模态异构图的框架，通过构建目标切片内、跨切片和参考切片内三种子图，结合 masked graph 对比学习和跨节点双注意力机制，实现从 H&E 病理图像预测空间基因表达，在七个数据集上 PCC 指标提升 7.3%-27.1%。

**[CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)**

:   提出 CryoHype，一种基于 Transformer 超网络的冷冻电镜重建方法，通过动态调整隐式神经表示（INR）的权重来减少参数共享，首次实现了从无标签冷冻电镜图像中同时重建 1000 种不同蛋白质结构。

**[CryoKRAQEN: Kernel-Regularized Annealing for Quantized Embedding Networks in Cryo-EM Heterogeneous Reconstruction](cryokraqen_kernel-regularized_annealing_for_quantized_embedding_networks_in_cryo.md)**

:   CryoKRAQEN 用一个**无编码器（decoder-only）的三平面 Fourier 码本**来做冷冻电镜异质重建：通过 Epanechnikov 核度量粒子图与码本原型的相似度、再用温度退火把软分配逐步收紧到近硬聚类，并加三元组正则稳住码本，从而在不依赖编码器和高斯先验的情况下，把噪声 2D 投影准确归到不同 3D 构象/组分，在 CryoBench 上与 SOTA 持平、在强组分异质数据上明显更好。

**[cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)**

:   提出 cryoSENSE，首个冷冻电镜压缩成像的计算框架，证明蛋白质 cryo-EM 图像在稀疏先验（DCT/小波/TV）和生成先验（扩散模型）下均可从欠采样测量中高保真重建，在保持 3D 分辨率的同时实现最高 2.5× 通量提升。

**[Deciphering Genotype-Phenotype Mechanisms from High-Content Profiling via Knowledge-Guided Multi-modal Graph Learning](deciphering_genotype-phenotype_mechanisms_from_high-content_profiling_via_knowle.md)**

:   KERNEL 把高内涵细胞形态成像当作"关系证据"而非节点特征，用形态相似度在生物知识图谱上动态增补"伪边"并赋可学习置信度，再用知识引导的异质图学习做基因调控网络（GRN）推断、药物-靶点（DTI）预测和疾病亚型子网络发现，GRN 上 AUPR 最高提升 38.1%。

**[FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics](feast_fully_connected_expressive_attention_for_spatial_transcriptomics.md)**

:   FEAST 把"从 H&E 病理大图预测空间基因表达"这件事从依赖预定义稀疏图的 GNN 范式，改造成一个全连接注意力框架——用自注意力天然建模所有 spot 两两交互，再补上能表达"抑制关系"的负向注意力和补全栅格空隙信息的 off-grid 采样，在三个公开 ST 数据集上 9 个指标里拿下 7 个 SOTA。

**[From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images](from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima.md)**

:   本文把"从病理切片预测空间基因表达"从逐 spot 回归任务改写成稠密预测任务，提出 PixNet：先用病理基础模型抽金字塔特征，再 U-Net 式逐层解码出一张全图的稠密基因表达图，最后对任意位置/任意半径的 spot 做圆形区域聚合得到表达值，从而在多个空间尺度（2µm 单细胞级到 100µm）上都超过现有 SOTA。

**[Hyperbolic Busemann Neural Networks](hyperbolic_busemann_neural_networks.md)**

:   利用 Busemann 函数将多类逻辑回归（MLR）和全连接层（FC）内蕴地提升到双曲空间，提出 BMLR 和 BFC 两个统一组件，在 Poincaré 球和 Lorentz 模型上同时适用，且在图像分类、基因组序列、节点分类、链接预测四类任务上均优于已有双曲层。

**[HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)**

:   从 H&E 病理图像直接预测空间转录组（ST）的基因表达时，已有方法只做 spot 级的图-基因匹配、忽略 ST 数据本身的层级结构，本文提出 HyperST：用多层级表征抽取器同时捕捉 spot 级与 niche 级的图像/基因特征，并在**双曲空间**里做层级对齐（对比对齐 HCA + 蕴含对齐 HEA），把分子语义注入图像表征，在四个组织数据集上全面刷新 SOTA。

**[MMCP-GEN: A Modality-Extensible Diffusion Language Model for Conditional Protein Sequence Generation](mmcp-gen_a_modality-extensible_diffusion_language_model_for_conditional_protein_.md)**

:   MMCP-GEN 在离散扩散蛋白质语言模型 DPLM 之上，设计了一套「模态指示头 + 可学习查询融合」的可组合条件机制，把结构、配体、功能注释、自由文本等异构生物条件统一融合进一个共享条件空间，新增模态时无需重训骨干，并配合序列-结构联合打分目标，在功能生成、逆折叠、多目标设计三类任务上同时刷新 SOTA（序列恢复率最高提升约 5%）。

**[Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics](multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics.md)**

:   MHAL 把空间转录组的"空间坐标视图"和"基因表达视图"做两级对齐——样本级用 MSE 拉齐同一 spot 的双视图嵌入，语义级用一组可学习原型 + 最优传输做交换预测对比，再配自适应图融合和 ZINB 解码器，在 DLPFC、人乳腺癌、小鼠脑前部三套数据集上的空间域识别（ARI）显著超过 11 个现有方法。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出**ERBA(Enzyme-Reaction Bridging Adapter)**，将酶动力学参数预测重新建模为**分阶段多模态条件生成问题**——先通过MRCA注入底物信息捕获底物识别特异性，再通过G-MoE整合活性位点3D结构捕获构象适应，配合ESDA分布对齐保持PLM语义先验。

**[Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling](predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce.md)**

:   MCToGene 针对「从 H&E 病理图预测空间基因表达」时现有方法只建模单 spot 或两两邻居、抓不住多细胞间多对多协同/拮抗的痛点，提出用**多体注意力（many-body attention）**显式建模高阶跨细胞交互，并用**层级耦合模块**把两两注意力与多体注意力串起来控制组合爆炸，在 HEST-1k 与 STImage-1K4M 上相对最强基线提升约 7.85%。

**[Stronger Normalization-Free Transformers](stronger_normalization-free_transformers.md)**

:   通过系统分析逐点函数替代归一化层所需的四个关键属性（零中心性、有界性、中心敏感性、单调性），在大规模搜索中发现 $\text{Derf}(x) = \text{erf}(\alpha x + s)$ 是最优的归一化层替代函数，在视觉识别、图像生成、语音表示和DNA序列建模等多个领域持续超越LayerNorm和DyT，且性能增益主要来自更强的泛化而非拟合能力。

**[TRIDENT: A Trimodal Cascade Generative Framework for Drug and RNA-Conditioned Cellular Morphology Synthesis](trident_a_trimodal_cascade_generative_framework_for_drug_and_rna-conditioned_cel.md)**

:   TRIDENT 用「VAE 编码（药物+扰动前 RNA）→ 潜在条件 z → Diffusion Transformer 生成细胞形态」的级联框架，首次显式建模「RNA→形态」这条因果链，在自建的 MorphoGene 三模态数据集上把 FID 相比 SOTA 降低 5–7 倍，并能泛化到未见过的化合物。

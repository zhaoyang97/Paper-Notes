---
title: >-
  ICCV2025 计算生物论文汇总 · 4篇论文解读
description: >-
  4篇ICCV2025的计算生物方向论文解读，涵盖扩散模型、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "对抗鲁棒"
item_list:
  - u: "cryofastar_fast_cryoem_ab_initio_reconstruction_made_easy/"
    t: "CryoFastAR: Fast Cryo-EM Ab initio Reconstruction Made Easy"
  - u: "g2pdiffusion_cross-species_genotype-to-phenotype_prediction_via_evolutionary_dif/"
    t: "G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion"
  - u: "integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov/"
    t: "Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines"
  - u: "molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild/"
    t: "MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**📹 ICCV2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (21)](../../CVPR2026/computational_biology/index.md) · [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md)

**[CryoFastAR: Fast Cryo-EM Ab initio Reconstruction Made Easy](cryofastar_fast_cryoem_ab_initio_reconstruction_made_easy.md)**

:   首个将DUSt3R式的几何基础模型范式引入冷冻电镜(cryo-EM)领域的工作，通过ViT编码器+跨视图注意力解码器直接从大量含噪粒子图像前馈预测姿态（无需迭代优化），实现了比传统方法快10-33倍的ab initio蛋白质三维重建。

**[G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion](g2pdiffusion_cross-species_genotype-to-phenotype_prediction_via_evolutionary_dif.md)**

:   提出G2PDiffusion，首个基于扩散模型的跨物种基因型到表型预测框架，通过进化信号（多序列比对MSA和环境上下文）条件化生成形态学图像，实现从DNA序列预测物种外观。

**[Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines](integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)**

:   提出将外部生物知识（蛋白质互作图谱+单细胞基础模型的转录组特征）整合到显微图像预训练中，显式解耦扰动特异性和细胞系特异性表征，提升模型在未见细胞系上的扰动筛查泛化能力。

**[MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)**

:   提出 MolParser，一个端到端的光学化学结构识别 (OCSR) 方法，通过扩展 SMILES 表示（E-SMILES）处理 Markush 结构、构建 700 万级大规模训练集 MolParser-7M，并利用主动学习引入真实文献数据，在 WildMol 基准上以 76.9% 准确率显著超越现有方法。

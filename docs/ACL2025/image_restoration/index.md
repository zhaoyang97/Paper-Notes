---
title: >-
  ACL2025 图像恢复论文汇总 · 3篇论文解读
description: >-
  3篇ACL2025的图像恢复方向论文解读，涵盖图像恢复、对抗鲁棒、信息抽取、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "信息抽取"
  - "少样本学习"
item_list:
  - u: "a_self-denoising_model_for_robust_few-shot_relation_extraction/"
    t: "A Self-Denoising Model for Robust Few-Shot Relation Extraction"
  - u: "diffusedef_adversarial_defense/"
    t: "DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising"
  - u: "prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc/"
    t: "PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**💬 ACL2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md)

🔥 **高频主题：** 图像恢复 ×3 · 对抗鲁棒 ×2

**[A Self-Denoising Model for Robust Few-Shot Relation Extraction](a_self-denoising_model_for_robust_few-shot_relation_extraction.md)**

:   本文针对少样本关系抽取中支持集标签噪声问题，提出自去噪模型（SDM），通过标签校正模块和关系分类模块的协同训练，自动修正噪声标签并实现更鲁棒的关系预测，即使在无噪声场景下也显著超越基线。

**[DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising](diffusedef_adversarial_defense.md)**

:   DiffuseDef 在编码器与分类器之间插入一个扩散去噪层，训练时学习预测隐状态噪声，推理时对隐表示加噪→迭代去噪→集成平均，以即插即用的方式大幅提升文本分类模型在黑盒和白盒对抗攻击下的鲁棒性。

**[PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy](prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc.md)**

:   提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。

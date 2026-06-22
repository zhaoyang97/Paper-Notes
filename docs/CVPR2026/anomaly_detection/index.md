---
title: >-
  CVPR2026 异常检测论文汇总 · 7篇论文解读
description: >-
  7篇CVPR2026的异常检测方向论文解读，涵盖异常检测、少样本学习、布局/合成、对齐/RLHF、推理、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "异常检测"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "布局/合成"
  - "对齐/RLHF"
  - "推理"
  - "RAG"
item_list:
  - u: "anomaly-related_residual_fields_for_cross-domain_anomaly_detection/"
    t: "Anomaly-Related Residual Fields for Cross-domain Anomaly Detection"
  - u: "defect_cue-preserved_structural_feature_refinement_for_few-shot_anomaly_detectio/"
    t: "Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection"
  - u: "dual-prototype-guided_multi-task_learning_for_unsupervised_anomaly_detection_and/"
    t: "Dual-Prototype-Guided Multi-task Learning for Unsupervised Anomaly Detection and Classification"
  - u: "hunting_normality_from_query_sample_via_residual_learning_for_generalist_anomaly/"
    t: "Hunting Normality from Query Sample via Residual Learning for Generalist Anomaly Detection"
  - u: "layoutad_exploring_semantic-geometric_misalignment_reasoning_for_scene_layout_an/"
    t: "LayoutAD: Exploring Semantic-Geometric Misalignment Reasoning for Scene Layout Anomaly Detection"
  - u: "multi-prototype_compactness_and_boundary-aware_synthesis_for_unsupervised_anomal/"
    t: "Multi-Prototype Compactness and Boundary-Aware Synthesis for Unsupervised Anomaly Detection"
  - u: "raid_retrieval-augmented_anomaly_detection/"
    t: "RAID: Retrieval-Augmented Anomaly Detection"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 异常检测

**📷 CVPR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/anomaly_detection/index.md)

🔥 **高频主题：** 异常检测 ×7

**[Anomaly-Related Residual Fields for Cross-domain Anomaly Detection](anomaly-related_residual_fields_for_cross-domain_anomaly_detection.md)**

:   针对扩散模型残差里"噪声大、单看幅值无法区分异常"的难题，本文提出残差演化场（REF）：从扩散反向过程的残差时空轨迹中分离出"持续不被吸收的非平稳异常信号"，再用跨域场对齐（CFA）把有标签源域学到的检测器迁移到无标签目标域，在 9 个跨域迁移任务上平均 AUROC 95.22%，比最强基线高 13 个百分点。

**[Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection](defect_cue-preserved_structural_feature_refinement_for_few-shot_anomaly_detectio.md)**

:   本文指出少样本异常检测（FSAD）的核心难点在于细微缺陷线索在深层特征提取流水线里被逐层"稀释"掉，提出 DCP-SFR：先用可学习提示词把早期弱信号"放大"成高对比异常线索图，再用这张图引导重建式定位，最后做结构感知的边界精修，在 MVTec AD / VisA 上拿到图像级 97.3%、像素级 98.2% 的 AUROC。

**[Dual-Prototype-Guided Multi-task Learning for Unsupervised Anomaly Detection and Classification](dual-prototype-guided_multi-task_learning_for_unsupervised_anomaly_detection_and.md)**

:   PG-SFD 把"无监督异常检测（像素级定位）+ 弱监督异常分类（区域级分类）"建模成一个双原型协同优化问题，用正常原型与类别原型显式解耦正常/异常语义、用差分门控把正常先验注入分类分支、用几何正则缓解多任务梯度冲突，在 MVTec-AD 上拿到 I-AUROC 99.4% 且同时支持细粒度缺陷分类。

**[Hunting Normality from Query Sample via Residual Learning for Generalist Anomaly Detection](hunting_normality_from_query_sample_via_residual_learning_for_generalist_anomaly.md)**

:   针对通用异常检测（GAD）中「直接建模残差分布」会因残差与实例特征不一致而误判的问题，本文不再直接对残差分类，而是把残差当成**向导**：用可学习代理从残差里抽取模式（RFL），再借这些残差代理从支持集聚合查询相关的「正常性代理」（NLS），最后用正常性代理去查询特征里**搜寻正常区域**（HNQ）来定位异常，在工业→工业、工业→医学的跨域基准上取得有竞争力的少样本性能。

**[LayoutAD: Exploring Semantic-Geometric Misalignment Reasoning for Scene Layout Anomaly Detection](layoutad_exploring_semantic-geometric_misalignment_reasoning_for_scene_layout_an.md)**

:   LayoutAD 提出"场景布局异常检测"这一新任务，用无监督方式给图像里每个物体打出对象级异常分——它把场景拆成语义图与几何图，通过跨图注意力推理两者之间的"错配"，从而发现诸如"五条腿的狗""停在湖面上的车"这类像素级检测器看不见的布局级幻觉。

**[Multi-Prototype Compactness and Boundary-Aware Synthesis for Unsupervised Anomaly Detection](multi-prototype_compactness_and_boundary-aware_synthesis_for_unsupervised_anomal.md)**

:   针对单原型假设在类内方差大时决策边界过松的问题，本文提出 PGBL 框架：用多原型紧凑约束（MPCC）把正常特征结构化为多个紧凑子簇，再在子簇拓扑边界处合成伪异常（BAAS），最后用判别器（DBR）精修决策面，在 MVTec-AD / VisA / Real-IAD 上的检测与定位均超越此前方法。

**[RAID: Retrieval-Augmented Anomaly Detection](raid_retrieval-augmented_anomaly_detection.md)**

:   RAID 把无监督异常检测（UAD）重新解读为检索增强生成（RAG）流程：先用一个三层向量库（类原型→语义原型→实例 token）做由粗到细的检索，再用一个"引导式 MoE 滤波器"对检索得到的匹配代价体去噪，从而抑制匹配噪声、画出边界清晰的异常图，在 MVTec/VisA/MPDD/BTAD 的全样本、少样本、多数据集设定下都拿到 SOTA。

---
title: >-
  ICLR2026 目标检测论文汇总 · 16篇论文解读
description: >-
  16篇ICLR2026的目标检测方向论文解读，涵盖异常检测、目标检测、时序预测、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "异常检测"
  - "时序预测"
  - "少样本学习"
item_list:
  - u: "bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting/"
    t: "Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)"
  - u: "bootstrapping_mllm_for_weaklysupervised_classagnostic_object_counting/"
    t: "Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting"
  - u: "cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection/"
    t: "CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection"
  - u: "clip_behaves_like_a_bag-of-words_model_cross-modally_but_not_uni-modally/"
    t: "CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally"
  - u: "complexity-_and_statistics-guided_anomaly_detection_in_time_series_foundation_mo/"
    t: "Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models"
  - u: "contextual_and_seasonal_lstms_for_time_series_anomaly_detection/"
    t: "Contextual and Seasonal LSTMs for Time Series Anomaly Detection"
  - u: "dual_distillation_for_few-shot_anomaly_detection/"
    t: "Dual Distillation for Few-Shot Anomaly Detection"
  - u: "forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection/"
    t: "ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection"
  - u: "fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu/"
    t: "FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion"
  - u: "infodet_a_dataset_for_infographic_element_detection/"
    t: "InfoDet: A Dataset for Infographic Element Detection"
  - u: "long-context_generalization_with_sparse_attention/"
    t: "Long-Context Generalization with Sparse Attention"
  - u: "owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection/"
    t: "OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection"
  - u: "paano_patch-based_representation_learning_for_time-series_anomaly_detection/"
    t: "PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection"
  - u: "spwood_sparse_partial_weakly-supervised_oriented_object_detection/"
    t: "SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection"
  - u: "towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection/"
    t: "Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection"
  - u: "traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology/"
    t: "Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method"
item_total: 16
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🔬 ICLR2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (12)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 异常检测 ×6 · 目标检测 ×3 · 时序预测 ×3 · 少样本学习 ×3

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting](bootstrapping_mllm_for_weaklysupervised_classagnostic_object_counting.md)**

:   WS-COC 是首个用多模态大模型（MLLM）做弱监督类无关目标计数的框架，只用图像级总数作监督，靠"二分对话调优 + 比较排序优化 + 全局局部融合"三个简单策略把 MLLM 的计数能力激活出来，在 FSC-147 等四个数据集上逼近甚至超过部分点级监督的全监督方法。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。

**[CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally](clip_behaves_like_a_bag-of-words_model_cross-modally_but_not_uni-modally.md)**

:   通过线性探测实验证明 CLIP 的 BoW（词袋）行为并非源于编码器缺乏绑定信息，而是跨模态对齐的失败；提出 LABCLIP，仅训练一个轻量线性变换即可显著恢复属性-对象绑定能力。

**[Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models](complexity-_and_statistics-guided_anomaly_detection_in_time_series_foundation_mo.md)**

:   把时序基础模型（TFM，如 MOMENT）搬到重建式异常检测上时，会因「过泛化」（连异常也重建得很好）和「过平稳化」（实例归一化抹掉了均值方差）而失灵；本文用一个从重建/插补误差差导出的复杂度指标 $\alpha$ 自适应地把 TFM 与轻量统计模型混合（CAE），再把均值方差重新注入解码端（MOMENT-Stat），在 23 个单变量 + 17 个多变量基准上把 VUS-PR 从此前 SOTA 的 0.4233 提到 0.4679。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)**

:   提出双蒸馏框架 D24FAD，结合 query 图像上的教师-学生蒸馏（TSD）和 support 图像上的学生自蒸馏（SSD），辅以学习权重机制（L2W）自适应评估 support 重要性，在 APTOS 眼底数据集上仅用 2-shot 达到 100% AUROC。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件，为下冠层人员检测模型的训练和评估提供了坚实基础。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)**

:   提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

**[OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)**

:   提出 OwlEye 框架，利用基于成对距离统计的跨域特征对齐将异构图嵌入共享空间，从多图中提取 attribute-level 和 structure-level 正常模式存入可扩展字典，并通过截断注意力重建机制在完全零样本条件下检测未见图的异常节点，8 数据集平均 AUPRC 36.17% 超越最强 baseline ARC 约 5.4 个百分点。

**[PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)**

:   提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。

**[SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)**

:   提出首个统一处理"稀疏标注 + 弱标注（HBox/Point）"的旋转目标检测框架 SPWOOD：用 SOS-Student 在一个学生模型里并联补齐未标注、缺角度、缺尺度三股信号，再以多层级伪标签筛选（MPF）从无标注数据自训练，在 DOTA-v1.0/v1.5、DIOR 上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。

**[Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)**

:   提出 APF 框架，通过 Rayleigh 商引导的异常感知预训练和粒度自适应微调，解决图异常检测中标签稀缺和同质性差异的双重挑战。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。

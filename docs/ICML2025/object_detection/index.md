---
title: >-
  ICML2025 目标检测论文汇总 · 12篇论文解读
description: >-
  12篇ICML2025的目标检测方向论文解读，涵盖异常检测、时序预测、对齐/RLHF、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "异常检测"
  - "时序预测"
  - "对齐/RLHF"
  - "多模态"
item_list:
  - u: "blueglass_a_framework_for_composite_ai_safety/"
    t: "BlueGlass: A Framework for Composite AI Safety"
  - u: "causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly/"
    t: "Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection"
  - u: "costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering/"
    t: "CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering"
  - u: "few-shot_learner_generalizes_across_ai-generated_image_detection/"
    t: "Few-Shot Learner Generalizes Across AI-Generated Image Detection"
  - u: "fg-clip_fine-grained_visual_and_textual_alignment/"
    t: "FG-CLIP: Fine-Grained Visual and Textual Alignment"
  - u: "kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks/"
    t: "KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks"
  - u: "open-det_an_efficient_learning_framework_for_open-ended_detection/"
    t: "Open-Det: An Efficient Learning Framework for Open-Ended Detection"
  - u: "outlier_gradient_analysis_efficiently_identifying_detrimental_training_samples_f/"
    t: "Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models"
  - u: "self-organizing_visual_prototypes_for_non-parametric_representation_learning/"
    t: "Self-Organizing Visual Prototypes for Non-Parametric Representation Learning"
  - u: "ui-vision_a_desktop-centric_gui_benchmark_for_visual_perception_and_interaction/"
    t: "UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction"
  - u: "understanding_the_emergence_of_multimodal_representation_alignment/"
    t: "Understanding the Emergence of Multimodal Representation Alignment"
  - u: "when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy/"
    t: "When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🧪 ICML2025** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

🔥 **高频主题：** 异常检测 ×4 · 时序预测 ×2 · 对齐/RLHF ×2 · 多模态 ×2

**[BlueGlass: A Framework for Composite AI Safety](blueglass_a_framework_for_composite_ai_safety.md)**

:   提出 BlueGlass 复合 AI 安全框架，通过统一基础设施整合分布式评估、近似探针和稀疏自编码器三种安全分析工具，对视觉语言模型（VLM）在目标检测任务上的能力边界、层级动态和内部概念表示进行系统性安全分析。

**[Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)**

:   提出 CAROTS——将因果关系融入对比学习的多变量时间序列异常检测框架，用因果保持增强作为正样本（正常变化），因果破坏增强作为负样本（模拟异常），训练编码器基于因果结构区分正常与异常。

**[CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)**

:   将立体匹配/光流估计中的**代价体滤波（cost volume filtering）**思想引入无监督异常检测（UAD），构造输入与模板之间的匹配代价体，并通过3D U-Net 加双流注意力引导进行去噪滤波，作为通用后处理插件可同时提升重建型和嵌入型 UAD 方法的性能，在 MVTec-AD 和 VisA 上取得 SOTA。

**[Few-Shot Learner Generalizes Across AI-Generated Image Detection](few-shot_learner_generalizes_across_ai-generated_image_detection.md)**

:   首次将 AI 生成图像检测重新定义为少样本分类任务，提出 FSD (Few-Shot Detector) 基于原型网络学习度量空间，仅用 10 个来自未见生成模型的样本，在 GenImage 数据集上平均准确率达 84.1%，超越此前 SOTA (LARE2) +11.6%。

**[FG-CLIP: Fine-Grained Visual and Textual Alignment](fg-clip_fine-grained_visual_and_textual_alignment.md)**

:   FG-CLIP 系统性地解决 CLIP 细粒度理解的三大瓶颈：用 1.6B 长描述-图像对捕获全局语义细节，12M 图像+40M 区域标注实现精细区域对齐，10M 硬负样本训练模型区分微妙语义差异，在细粒度理解、开放词汇检测、图文检索等多项任务上取得全面领先。

**[KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)**

:   KAN-AD 将时间序列异常检测重新建模为用光滑单变量函数逼近序列，用截断傅里叶展开替代 KAN 中的 B 样条避免局部扰动敏感性，以不到 1000 个参数在 4 个基准上平均提升 15% 检测精度。

**[Open-Det: An Efficient Learning Framework for Open-Ended Detection](open-det_an_efficient_learning_framework_for_open-ended_detection.md)**

:   Open-Det 提出了一个高效的开放端目标检测（OED）框架，通过重构目标检测器（解耦 one-to-many/one-to-one 匹配）、引入 VL-prompts 蒸馏模块桥接视觉-语言语义鸿沟、LoRa Head + Text Denoising 加速 LLM 训练、以及 Masked Alignment Loss 消除矛盾监督，仅用 GenerateU 1.5% 的训练数据和 20.8% 的训练 epoch 就取得了更高的检测性能（APr +1.0%）。

**[Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models](outlier_gradient_analysis_efficiently_identifying_detrimental_training_samples_f.md)**

:   提出 Outlier Gradient Analysis (OGA)，将影响函数中识别有害训练样本的问题转化为梯度空间上的异常点检测，绕开了 Hessian 矩阵求逆的高计算开销，同时在噪声标签校正、NLP 数据筛选和 LLM 影响力数据识别等任务上取得优于传统影响函数方法的效果。

**[Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)**

:   提出 Self-Organizing Prototypes (SOP) 策略，用多个语义相似的支持嵌入（support embeddings）替代传统 SSL 中单一原型来表示特征空间的局部区域，并引入非参数化 MIM 任务，在检索、检测、分割等下游任务上取得 SOTA 表现。

**[UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction](ui-vision_a_desktop-centric_gui_benchmark_for_visual_perception_and_interaction.md)**

:   提出 UI-Vision——首个面向桌面环境的综合离线评估基准，覆盖 83 个软件应用，提供密集的 bounding box、UI 标签和操作轨迹标注，定义从细粒度到粗粒度的三级评估任务（Element Grounding → Layout Grounding → Action Prediction），系统评估并揭示 SOTA 模型在专业软件理解、空间推理和复杂操作上的关键短板。

**[Understanding the Emergence of Multimodal Representation Alignment](understanding_the_emergence_of_multimodal_representation_alignment.md)**

:   系统研究多模态表征对齐的涌现机制，发现隐式对齐的出现及其与性能的关系取决于数据的冗余/唯一信息比例和模态异质性，挑战了"更大模型→更好对齐→更好性能"的普遍假设。

**[When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)**

:   提出多模态异步混合网络，结合事件相机的高时间分辨率（异步 GNN 处理）和 RGB 相机的丰富空间特征（CNN 处理），在交通异常检测中实现 579 FPS 的推理速度和 1.17s 的平均响应时间，首次将事件流引入自动驾驶异常检测领域。

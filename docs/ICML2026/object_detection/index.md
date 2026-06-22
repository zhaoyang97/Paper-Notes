---
title: >-
  ICML2026 目标检测论文汇总 · 6篇论文解读
description: >-
  6篇ICML2026的目标检测方向论文解读，涵盖异常检测、对抗鲁棒、推理、强化学习、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "异常检测"
  - "对抗鲁棒"
  - "推理"
  - "强化学习"
  - "多模态"
item_list:
  - u: "adversarially_robust_approximate_furthest_neighbor/"
    t: "Adversarially Robust Approximate Furthest Neighbor"
  - u: "earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc/"
    t: "EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding"
  - u: "focus_forcing_in-context_object_localization_through_visual_support_constraints_/"
    t: "FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization"
  - u: "mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection/"
    t: "Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection"
  - u: "omniverifier-m1_multimodal_meta-verifier_with_explicit_structured_recalibration/"
    t: "OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration"
  - u: "testing_the_test_score-direction_instability_in_class-split_anomaly_detection/"
    t: "Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🧪 ICML2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (12)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 异常检测 ×2

**[Adversarially Robust Approximate Furthest Neighbor](adversarially_robust_approximate_furthest_neighbor.md)**

:   这篇理论论文首次给出能抵抗自适应查询对手的近似最远邻数据结构，在保持与 Indyk 经典 oblivious 算法相近的 $n$ 依赖查询复杂度的同时，证明传统随机投影最远邻算法会被自适应查询击穿。

**[EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)**

:   EARL 用"粗解析-细响应"两阶段 MLLM 框架把第一视角交互理解任务（描述+答问+像素掩膜）做成统一管线：第一阶段输出整图交互的全局描述并把最后一层 hidden state 当作语义先验，再通过新的 Analysis-guided Feature Synthesizer 注入到第二阶段，用 GRPO + 三路奖励（格式/答案/grounding 准确率）联合训练，在 Ego-IRGBench 上 cIoU 反超 Seg-Zero 8.37%。

**[FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization](focus_forcing_in-context_object_localization_through_visual_support_constraints_.md)**

:   FOCUS 通过"完全去除类别名 + 注意力 mask 优化 + GRPO IoU 奖励"两阶段训练，让 VLM 真正按视觉支持示例（而非语义先验）做 in-context 目标定位；7B 参数模型超 72B 模型，证明任务对齐的 inductive bias 比单纯 scaling 更重要。

**[Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)**

:   MPFM 把 OSAD 里传统的"单峰高斯原型"换成可学习的**高斯混合原型空间**, 用流匹配直接回归一个 GMM 形式的速度场, 再加一个互信息最大化正则防止原型崩塌, 在 9 个工业 / 医学 AD 数据集上以 10/1 个异常样本的设定打过 DRA / AHL / DPDL 等所有 SOTA.

**[OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration](omniverifier-m1_multimodal_meta-verifier_with_explicit_structured_recalibration.md)**

:   针对多模态视觉验证器只输出 True/False 二值判断信号过粗、且文本解释易被 reward-hacking 的问题，本文提出 OmniVerifier-M1：用 bounding box 等符号化输出代替文本作为 meta-verification rationale 以支持 IoU 这种 rule-based reward，并在理论与实验上证明把二值判断与 meta-verification 解耦成两条独立 reward 流（而非合并成乘性 joint reward）能显著提升 SNR，最终把验证器升级为可驱动 region-level 自校正的 agentic 系统 M1-TTS。

**[Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection](testing_the_test_score-direction_instability_in_class-split_anomaly_detection.md)**

:   作者指出"类内拆分"(class-split) 异常检测基准在异常类与正常混合分布在表示空间重叠时是病态的——AUROC 会塌缩到随机甚至反转，方向取决于未知的异常类，并提出一个无需训练的"邻域类泄漏"指标 $L_k$ 来在跑分前诊断这种基准失效。

---
title: >-
  CVPR2026 LLM安全论文汇总 · 12篇论文解读
description: >-
  12篇CVPR2026的 LLM 安全方向论文解读，涵盖多模态、对抗鲁棒、文生图、持续学习、域适应、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对抗鲁棒"
  - "文生图"
  - "持续学习"
  - "域适应"
  - "少样本学习"
item_list:
  - u: "autodebias_automated_framework_for_debiasing_text-to-image_models/"
    t: "AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models"
  - u: "blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models/"
    t: "The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models"
  - u: "designing_to_forget_deep_semi-parametric_models_for_unlearning/"
    t: "Designing to Forget: Deep Semi-parametric Models for Unlearning"
  - u: "elastic_weight_consolidation_done_right_for_continual_learning/"
    t: "Elastic Weight Consolidation Done Right for Continual Learning"
  - u: "learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_/"
    t: "Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting"
  - u: "machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o/"
    t: "Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization"
  - u: "omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms/"
    t: "Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs"
  - u: "oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr/"
    t: "⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization"
  - u: "ph-strips_for_selective_forgetting_a_blunt_but_fast_diagnostic_baseline_for_mach/"
    t: "pH-Strips for Selective Forgetting: A Blunt but Fast Diagnostic Baseline for Machine Unlearning"
  - u: "revisiting_learning_with_noisy_labels_active_forgetting_and_noise_suppression/"
    t: "Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression"
  - u: "select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation/"
    t: "Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation"
  - u: "sineproject_machine_unlearning_for_stable_vision_language_alignment/"
    t: "SineProject: Machine Unlearning for Stable Vision–Language Alignment"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📷 CVPR2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (184)](../../ICLR2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/llm_safety/index.md) · [🧪 ICML2025 (41)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** 多模态 ×2 · 对抗鲁棒 ×2

**[AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](autodebias_automated_framework_for_debiasing_text-to-image_models.md)**

:   提出 AutoDebias——首个同时检测和缓解 T2I 模型中恶意后门偏见的统一框架，利用 VLM 开放集检测发现触发词-偏见关联并构建查找表，再通过 CLIP 引导的分布对齐训练消除后门关联，在 17 种后门场景中将攻击成功率从 90% 降至接近 0 且保持图像质量。

**[The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)**

:   系统研究 VLM 微调到自动驾驶场景时的灾难性遗忘问题，构建 180K 场景大规模基准 FidelityDrivingBench，并提出 Drive Expert Adapter (DEA) 通过提示空间路由在不腐蚀基础参数的前提下增强驾驶任务性能。

**[Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)**

:   提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。

**[Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)**

:   本文从梯度视角系统分析了 EWC 及其变体在权重重要性估计上的根本缺陷（EWC 的梯度消失和 MAS 的冗余保护），并提出了一个极其简单的 Logits Reversal 操作来修正 Fisher 信息矩阵的计算，在无样例类增量学习和多模态持续指令微调任务上大幅超越原始 EWC 及其所有变体。

**[Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)**

:   提出KNOW prediction：通过在逐步缩小的数据子集上sequential fine-tuning诱导结构化遗忘过程，收集权重转变轨迹，然后用meta-learned hyper-model（KNOWN）反转forgetting方向，预测"仿佛在更大数据集上训练"的虚拟知识增强权重。跨多数据集(CIFAR/ImageNet/PACS等)和多架构(ResNet/PVTv2/DeepLabV3+)持续超越naive fine-tuning及多种weight prediction基线，在图像分类、语义分割、图像描述、域泛化等下游任务上均有显著提升。

**[Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)**

:   针对机器遗忘里"对所有样本/类别一视同仁"和"遗忘目标与保留目标梯度互相打架"两大问题，本文提出**自适应梯度重加权**（按样本记忆深度/类别脆弱度给不同权重）+ **三阶段目标优化**（方向纠偏 → 时间平滑 → 自适应组合），在 CIFAR-10/100、Tiny-ImageNet 上把随机遗忘的 Avg Gap 从 SOTA 的 0.85 压到 0.19。

**[Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)**

:   针对"开放式 VQA/OCR 任务没有显式攻击目标、现有对抗鲁棒性评测各用各的协议"两大空白，本文先建了统一的定向攻击基准 **AdvRobustBench**（1000 题，VQA+OCR），再提出迁移式黑盒攻击 **Omni-Attack**（用 LLM 生成"问题条件化"的文本/视觉目标 + OCR 位置感知扰动 + 四种迁移正则），在 GPT-4.1 上 $\epsilon=8/255$ 就把定向攻击成功率打到 71.8%。

**[⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)**

:   发现无源域自适应（SFDA）方法会不经意地将源域独有类别的知识泄漏到目标域（零样本迁移现象），提出 SCADA-UL 框架通过对抗生成遗忘样本和重缩放标签策略，在域自适应过程中同时完成类别遗忘，达到接近从头训练的遗忘效果。

**[pH-Strips for Selective Forgetting: A Blunt but Fast Diagnostic Baseline for Machine Unlearning](ph-strips_for_selective_forgetting_a_blunt_but_fast_diagnostic_baseline_for_mach.md)**

:   提出 MUpHT——一个**免训练、无需保留集、闭式求解**的机器遗忘方法：把待遗忘概念在特征空间张成的低维子空间从权重里"投影掉"，几秒内（CIFAR-100 上 0.004 分钟、SD 去除裸露概念约 0.7 秒）就得到一个对该概念"失明"的模型，定位是给机器遗忘领域提供一张"试纸"式的快速诊断基线，效果却能和动辄训练数小时的 SalUn 打平甚至超过。

**[Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression](revisiting_learning_with_noisy_labels_active_forgetting_and_noise_suppression.md)**

:   针对噪声标签学习长期依赖"挑干净样本"导致的过拟合瓶颈，本文提出即插即用框架 FINE：用基于机器遗忘的负交叉熵损失"主动遗忘"早期已吸收的噪声知识，再用基于负学习的互补标签损失"抑制"后期对噪声的过拟合，挂在 SED / ACT 等现有 SOTA 上即可稳定提升鲁棒性与泛化。

**[Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation](select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation.md)**

:   提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

**[SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)**

:   针对多模态大模型（MLLM）在机器遗忘过程中投影层 Jacobian 严重病态导致视觉-语言对齐漂移的问题，提出 SineProject——通过对投影层权重施加正弦调制（sin(ΔW)）来约束参数范围至 [-1,1]，从而将 Jacobian 条件数降低 3-4 个数量级，在完全遗忘目标知识的同时将良性查询误拒率（SARR）降低 15%。

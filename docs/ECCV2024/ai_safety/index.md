---
title: >-
  ECCV2024 AI安全论文汇总 · 13篇论文解读
description: >-
  13篇ECCV2024的 AI 安全方向论文解读，涵盖对抗鲁棒、联邦学习、持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "联邦学习"
  - "持续学习"
item_list:
  - u: "any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_/"
    t: "Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection"
  - u: "bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement/"
    t: "Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement"
  - u: "clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks/"
    t: "CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks"
  - u: "fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning/"
    t: "Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning"
  - u: "genq_quantization_in_low_data_regimes_with_generative_synthetic_data/"
    t: "Event Trojan: Asynchronous Event-based Backdoor Attacks"
  - u: "noise-assisted_prompt_learning_for_image_forgery_detection_and_localization/"
    t: "Noise-Assisted Prompt Learning for Image Forgery Detection and Localization"
  - u: "one-stage_prompt-based_continual_learning/"
    t: "One-stage Prompt-based Continual Learning"
  - u: "operational_open-set_recognition_and_postmax_refinement/"
    t: "Operational Open-Set Recognition and PostMax Refinement"
  - u: "preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti/"
    t: "Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective"
  - u: "resilience_of_entropy_model_in_distributed_neural_networks/"
    t: "Resilience of Entropy Model in Distributed Neural Networks"
  - u: "skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma/"
    t: "SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks"
  - u: "towards_multi-modal_transformers_in_federated_learning/"
    t: "Towards Multi-modal Transformers in Federated Learning"
  - u: "unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r/"
    t: "Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients"
item_total: 13
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🎞️ ECCV2024** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (145)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/ai_safety/index.md) · [💬 ACL2026 (5)](../../ACL2026/ai_safety/index.md) · [🧪 ICML2026 (114)](../../ICML2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×6 · 联邦学习 ×3

**[Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection](any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)**

:   提出 GAKer，首个可泛化到未知目标类别的定向对抗攻击生成器，通过在 UNet 中间层注入目标特征（latent infection）+ 余弦距离损失替代交叉熵实现类别无关训练，在未知类上的攻击成功率比 HGN 高 14.13%。

**[Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement](bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)**

:   提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

**[CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)**

:   提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。

**[Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)**

:   本文提出Self-Driven Fisher Calibration（SDFC），利用Fisher信息度量参数对不同分布的重要程度差异，在异质联邦学习场景中有效区分恶意后门客户端并进行参数校准，突破了现有防御方法依赖数据同质性和恶意节点少数假设的局限。

**[Event Trojan: Asynchronous Event-based Backdoor Attacks](genq_quantization_in_low_data_regimes_with_generative_synthetic_data.md)**

:   提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。

**[Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)**

:   本文提出 CLIP-IFDL，一种基于 CLIP 的图像篡改检测与定位模型，通过实例感知的双流提示学习和伪造增强噪声适配器来弥补 CLIP 在篡改检测领域的提示缺失和伪造感知不足问题，将 CLIP 的开放世界泛化能力迁移到篡改检测任务中。

**[One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[Operational Open-Set Recognition and PostMax Refinement](operational_open-set_recognition_and_postmax_refinement.md)**

:   本文提出了一种面向实际部署场景的开放集识别评估指标 OOSA（Operational Open-Set Accuracy）以及后处理算法 PostMax，通过对最大类别 logit 进行深度特征幅度归一化和广义 Pareto 分布映射，将 logit 转化为合理的概率估计，在大规模评估中取得了统计显著的 SOTA 性能。

**[Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)**

:   从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。

**[Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)**

:   首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平，准确率仅下降约 2%。

**[SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)**

:   提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。

**[Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)**

:   提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。

**[Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)**

:   本文揭示了随机神经网络（SNNs）在联邦学习中同样容易遭受梯度反演攻击，提出 ISG 方法通过将 SNN 的随机训练过程等价为传统 NN 训练的变体来重建训练数据，并引入特征约束策略提升重建保真度。

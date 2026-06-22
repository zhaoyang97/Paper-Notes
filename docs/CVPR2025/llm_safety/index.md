---
title: >-
  CVPR2025 LLM安全论文汇总 · 14篇论文解读
description: >-
  14篇CVPR2025的 LLM 安全方向论文解读，涵盖多模态、持续学习、LLM、对抗鲁棒、重识别、医学影像等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "持续学习"
  - "LLM"
  - "对抗鲁棒"
  - "重识别"
  - "医学影像"
item_list:
  - u: "a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant/"
    t: "A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks"
  - u: "dual_consolidation_for_pre-trained_model-based_domain-incremental_learning/"
    t: "Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning"
  - u: "empowering_llms_to_understand_and_generate_complex_vector_graphics/"
    t: "LLM4SVG: Empowering LLMs to Understand and Generate Complex Vector Graphics"
  - u: "forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language/"
    t: "ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models"
  - u: "hyperbolic_safety-aware_vision-language_models/"
    t: "Hyperbolic Safety-Aware Vision-Language Models"
  - u: "lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty/"
    t: "LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty"
  - u: "low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc/"
    t: "Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning"
  - u: "mp-gui_modality_perception_with_mllms_for_gui_understanding/"
    t: "MP-GUI: Modality Perception with MLLMs for GUI Understanding"
  - u: "neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating/"
    t: "Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating"
  - u: "protecting_your_video_content_disrupting_automated_video-based_llm_annotations/"
    t: "Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations"
  - u: "steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_/"
    t: "Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks"
  - u: "tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language/"
    t: "TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models"
  - u: "test-time_attention_purification_for_backdoored_large_vision_language_models/"
    t: "CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models"
  - u: "towards_all-in-one_medical_image_re-identification/"
    t: "Towards All-in-One Medical Image Re-Identification"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📷 CVPR2025** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (184)](../../ICLR2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** 多模态 ×6 · 持续学习 ×2

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

**[Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)**

:   提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。

**[LLM4SVG: Empowering LLMs to Understand and Generate Complex Vector Graphics](empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   提出 LLM4SVG 框架，通过定义 55 个可学习的 SVG 语义 token 替代原始 XML 标签，结合 250K 高质量 SVG 和 580K 指令数据的 SVGX-SFT 数据集进行两阶段指令微调，使 GPT-2、Phi-2、Falcon 等开源 LLM 能高质量理解和生成复杂矢量图形，GPT-2 XL 版本达 FID 64.11、CLIPScore 0.3496，大幅超越 GPT-4o（127.78 FID）和所有现有 SVG 生成方法。

**[ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)**

:   发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

**[Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)**

:   HySAC 提出在双曲空间中构建安全感知的视觉语言模型，通过蕴含锥（entailment cone）将安全/不安全内容映射到双曲空间的不同区域（安全内容靠近原点、不安全内容远离原点），使模型具备安全内容分类和动态重定向能力，在检索安全性和NSFW检测上显著超越现有遗忘方法。

**[LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)**

:   提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150），且提出 RF-JSD 免重训评估指标（与 JSD Pearson 相关 0.92）。

**[Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)**

:   针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。

**[MP-GUI: Modality Perception with MLLMs for GUI Understanding](mp-gui_modality_perception_with_mllms_for_gui_understanding.md)**

:   MP-GUI设计了三个专用感知器分别提取GUI中的图形、文本和空间模态信息，通过空间结构精炼策略和自适应融合门控将三种模态组合，在有限训练数据下在多种GUI理解任务上取得了优于通用MLLM的表现。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations](protecting_your_video_content_disrupting_automated_video-based_llm_annotations.md)**

:   本文提出两类对抗性视频水印方法——Ramblings（诱导视频 LLM 生成错误描述）和 Mutes（诱导视频 LLM 生成极短或空描述），通过不可感知的对抗扰动保护个人视频免受未经授权的自动化标注，并验证了这些低质量标注会降低下游文本到视频生成模型的性能。

**[Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_.md)**

:   提出ASTRA，通过**图像归因**定位对抗图像中与越狱最相关的视觉token，构建**转向向量**表征有害响应方向，并在推理时进行**自适应激活转向**将模型远离有害方向，实现了比JailGuard低12%毒性分数、低18% ASR且快9倍的SOTA防御效果。

**[TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)**

:   首个 VLM 测试时对抗防御方法，通过最小化多视图增强的熵一致性 + 对抗-干净 embedding 统计对齐来学习每个测试样本的防御性 prompt，仅需一步优化即可将 CLIP 对 AutoAttack 的鲁棒性从 0.1% 提升到 48.9%。

**[CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

**[Towards All-in-One Medical Image Re-Identification](towards_all-in-one_medical_image_re-identification.md)**

:   提出 MaMI，首个全模态统一的医学图像重识别模型，通过连续模态参数适配器 (ComPA) 动态生成模态特定参数，并利用医学基础模型的差异特征对齐传递医学先验，在 11 个数据集上超越 25 个基础模型和 8 个大语言模型。

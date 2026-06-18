---
title: >-
  CVPR2026 LLM安全论文汇总 · 25篇论文解读
description: >-
  25篇CVPR2026的 LLM 安全方向论文解读，涵盖多模态、对抗鲁棒、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对抗鲁棒"
  - "LLM"
  - "推理"
item_list:
  - u: "a_closedform_solution_for_debiasing_visionlanguage/"
    t: "A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks"
  - u: "demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias/"
    t: "Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification"
  - u: "designing_to_forget_deep_semi-parametric_models_for_unlearning/"
    t: "Designing to Forget: Deep Semi-parametric Models for Unlearning"
  - u: "exemplar-free_continual_learning_for_state_space_models/"
    t: "Exemplar-Free Continual Learning for State Space Models"
  - u: "fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua/"
    t: "FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models"
  - u: "force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct/"
    t: "FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction"
  - u: "harmonious_parameter_adaptation_in_continual_visual_instruction_tuning_for_safet/"
    t: "Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs"
  - u: "iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding/"
    t: "IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding"
  - u: "interpretable_debiasing_of_vision-language_models_for_social_fairness/"
    t: "Interpretable Debiasing of Vision-Language Models for Social Fairness"
  - u: "learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_/"
    t: "Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting"
  - u: "multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua/"
    t: "Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models"
  - u: "omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms/"
    t: "Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs"
  - u: "oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr/"
    t: "⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization"
  - u: "phantasia_context-adaptive_backdoors_in_vision_language_models/"
    t: "Phantasia: Context-Adaptive Backdoors in Vision Language Models"
  - u: "pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw/"
    t: "Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision"
  - u: "select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation/"
    t: "Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation"
  - u: "sineproject_machine_unlearning_for_stable_vision_language_alignment/"
    t: "SineProject: Machine Unlearning for Stable Vision–Language Alignment"
  - u: "test-time_attention_purification_for_backdoored_large_vision_language_models/"
    t: "Test-Time Attention Purification for Backdoored Large Vision Language Models"
  - u: "towards_reasoning-preserving_unlearning_in_multimodal_large_language_models/"
    t: "Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models"
  - u: "towards_robust_multimodal_large_language_models_against_jailbreak_attacks/"
    t: "Towards Robust Multimodal Large Language Models Against Jailbreak Attacks"
  - u: "unsafe2safe_controllable_image_anonymization_for_downstream_utility/"
    t: "Unsafe2Safe: Controllable Image Anonymization for Downstream Utility"
  - u: "v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta/"
    t: "V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs"
  - u: "vl-eraser_vacuum_distillation_for_machine_unlearning_in_vision-language_models/"
    t: "VL-Eraser: Vacuum Distillation for Machine Unlearning in Vision-Language Models"
  - u: "vlm_model_inversion_adaptive_token_weight/"
    t: "Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks"
  - u: "which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un/"
    t: "Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models"
item_total: 25
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📷 CVPR2026** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (46)](../../ICML2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🔬 ICLR2026 (52)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** 多模态 ×14 · 对抗鲁棒 ×8 · LLM ×3 · 推理 ×2

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解方法，通过在跨模态嵌入空间中对属性子空间做正交分解并利用Chebyshev标量化求解，实现Pareto最优公平性与有界效用损失，免训练、免标注，统一覆盖零样本分类、文本-图像检索和文本-图像生成三大下游任务。

**[Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)**

:   首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。

**[Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)**

:   提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。

**[Exemplar-Free Continual Learning for State Space Models](exemplar-free_continual_learning_for_state_space_models.md)**

:   本文提出 Inf-SSM——一种几何感知、无需存旧样本的正则化方法，把 SSM（如 Vim/Mamba）的"无穷时域行为"编码成扩展可观测子空间上的一个点，通过约束新旧任务子空间在无穷维 Grassmann 流形上的距离来抑制灾难性遗忘，并把原本 $\mathcal{O}(n^3)$ 的求解代价降到 $\mathcal{O}(n^2)$，即插即用地把现有持续学习方法平均 AA 提升 8.31%、遗忘 FM 降低 9.36%。

**[FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)**

:   提出 FairLLaVA，一种参数高效的公平性微调方法，通过最小化隐藏状态与人口学属性之间的互信息来消除多模态大语言模型中的人口学捷径，在胸片报告生成和皮肤病变问答中显著缩小了群体间性能差距。

**[FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)**

:   分析发现视觉 jailbreak attack 迁移性差的根因是 attack 处于 high-sharpness loss region——源于浅层特征过度依赖 model-specific 表示和高频信息过度影响；提出 FORCE 方法通过 layer-aware regularization 扩展浅层 feasible region + spectral rescaling 抑制高频非语义成分，引导 attack 进入 flatter loss landscape，显著提升跨模型迁移性。

**[Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs](harmonious_parameter_adaptation_in_continual_visual_instruction_tuning_for_safet.md)**

:   HPA 关注一个被忽视的场景——对"已做过安全对齐"的多模态大模型继续做持续视觉指令微调（post-SA CVIT）时，模型既会遗忘旧任务、又会丢掉安全性；它在每步微调后做无侵入的后训练参数调整，用 Hessian 重要性把参数分成"安全焦点 / 任务焦点"、在层内层间平衡地选择性保留安全参数、并对更新方向施加正交约束，从而在安全和任务性能之间取得和谐折中。

**[IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)**

:   提出IAG，首个针对VLM视觉定位的多目标后门攻击方法，通过文本条件U-Net动态生成输入感知触发器，将任意指定目标物体的语义信息嵌入视觉输入中，在12种设置下的11种达到最高攻击成功率。

**[Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)**

:   提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。

**[Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)**

:   提出KNOW prediction：通过在逐步缩小的数据子集上sequential fine-tuning诱导结构化遗忘过程，收集权重转变轨迹，然后用meta-learned hyper-model（KNOWN）反转forgetting方向，预测"仿佛在更大数据集上训练"的虚拟知识增强权重。跨多数据集(CIFAR/ImageNet/PACS等)和多架构(ResNet/PVTv2/DeepLabV3+)持续超越naive fine-tuning及多种weight prediction基线，在图像分类、语义分割、图像描述、域泛化等下游任务上均有显著提升。

**[Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)**

:   提出 MPCAttack 框架，联合跨模态对齐、多模态理解和视觉自监督三种学习范式的特征表示，通过多范式协同优化策略生成高迁移性对抗样本，在开源和闭源 MLLM 上均取得 SOTA 攻击效果。

**[Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)**

:   针对"开放式 VQA/OCR 任务没有显式攻击目标、现有对抗鲁棒性评测各用各的协议"两大空白，本文先建了统一的定向攻击基准 **AdvRobustBench**（1000 题，VQA+OCR），再提出迁移式黑盒攻击 **Omni-Attack**（用 LLM 生成"问题条件化"的文本/视觉目标 + OCR 位置感知扰动 + 四种迁移正则），在 GPT-4.1 上 $\epsilon=8/255$ 就把定向攻击成功率打到 71.8%。

**[⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)**

:   发现无源域自适应（SFDA）方法会不经意地将源域独有类别的知识泄漏到目标域（零样本迁移现象），提出 SCADA-UL 框架通过对抗生成遗忘样本和重缩放标签策略，在域自适应过程中同时完成类别遗忘，达到接近从头训练的遗忘效果。

**[Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)**

:   Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。

**[Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)**

:   提出 DeepfakeJudge 框架，通过 bootstrapped generator-evaluator 流程将人类标注的推理监督扩展为大规模结构化评分数据，训练出 3B/7B 视觉语言模型作为 deepfake 检测推理质量的自动评判者，在 pointwise 和 pairwise 评估上均达到与人类高度一致的水平。

**[Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation](select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation.md)**

:   提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

**[SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)**

:   针对多模态大模型（MLLM）在机器遗忘过程中投影层 Jacobian 严重病态导致视觉-语言对齐漂移的问题，提出 SineProject——通过对投影层权重施加正弦调制（sin(ΔW)）来约束参数范围至 [-1,1]，从而将 Jacobian 条件数降低 3-4 个数量级，在完全遗忘目标知识的同时将良性查询误拒率（SARR）降低 15%。

**[Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   发现LVLM后门行为的本质是跨模态注意力窃取（trigger视觉token抢夺文本token的注意力），提出CleanSight——首个无需训练的测试时后门防御框架，通过检测和剪枝高注意力trigger token来消除后门效应。

**[Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models](towards_reasoning-preserving_unlearning_in_multimodal_large_language_models.md)**

:   针对"会思考"的多模态大模型，提出基准 RMLLMU-Bench 专门衡量**推理链里的信息泄漏**与**推理能力保留**，并给出一个免训练、推理时介入的框架 R-MUSE——通过子空间引导 + 自适应转向，在遗忘目标答案和中间推理痕迹的同时尽量不破坏通用推理。

**[Towards Robust Multimodal Large Language Models Against Jailbreak Attacks](towards_robust_multimodal_large_language_models_against_jailbreak_attacks.md)**

:   SAFEMLLM 是第一个直接对多模态大模型（MLLM）做对抗训练的越狱防御框架：它在 token 嵌入层注入一对可学习扰动矩阵来高效模拟跨模态攻击（CoE-Attack），再交替更新模型参数去抵消这些扰动，从而在白盒场景下把六种越狱攻击的成功率压到接近 0，同时几乎不损失正常多模态问答能力。

**[Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)**

:   本文提出 Unsafe2Safe 全自动隐私保护流水线，通过 VLM 隐私检查→双字幕生成（私有/公开）→LLM 编辑指令→文本引导扩散编辑的四阶段方案，实现可控图像匿名化，在 VLMScore 隐私指标大幅提升的同时，在 Caltech-101 分类和 OK-VQA 上匿名后准确率甚至超过原始图像。

**[V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)**

:   发现 ViT 中 Value 特征相比 Patch 特征具有更解耦的局部语义表示，提出 V-Attack 通过自增强 Value 特征 + 文本引导语义操控实现精确可控的 LVLM 局部语义攻击，ASR 平均提升 36%。

**[VL-Eraser: Vacuum Distillation for Machine Unlearning in Vision-Language Models](vl-eraser_vacuum_distillation_for_machine_unlearning_in_vision-language_models.md)**

:   VL-Eraser 指出传统「反向训练」式遗忘在 VLM 上其实只是破坏了跨模态对齐、并没有真正删掉知识；它把遗忘重构成「先蒸馏、后删除」两阶段——先把要遗忘的知识在「真空空间」约束下蒸馏进一组 LoRA，再把这组 LoRA 从原模型里减掉，从而在删得更干净的同时保住模型可用性。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。

**[Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)**

:   本文提出CORE(COncept-aware REfuser)，一个面向大视觉语言模型(LVLM)持续遗忘的框架：通过将待删除的视觉-语言对分解为细粒度的视觉属性和文本意图概念，使用概念调制器识别需要拒绝的概念组合，再通过混合拒绝专家(refusers)生成概念对齐的拒绝回复，在16个连续遗忘任务上实现了90.67% CRR和88.02% AR的最佳遗忘-保留权衡。

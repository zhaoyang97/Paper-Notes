---
title: >-
  CVPR2025 幻觉检测论文汇总 · 9篇论文解读
description: >-
  9篇CVPR2025的幻觉检测方向论文解读，涵盖多模态、LLM、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "LLM"
  - "机器人"
item_list:
  - u: "3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall/"
    t: "3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination"
  - u: "antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua/"
    t: "Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception"
  - u: "halloc_token-level_localization_of_hallucinations_for_vision_language_models/"
    t: "HalLoc: Token-Level Localization of Hallucinations for Vision Language Models"
  - u: "octopus_alleviating_hallucination_via_dynamic_contrastive_decoding/"
    t: "Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding"
  - u: "ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models/"
    t: "ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models"
  - u: "one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml/"
    t: "One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination"
  - u: "phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset/"
    t: "PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset"
  - u: "seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_/"
    t: "Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding"
  - u: "stop_learning_it_all_to_mitigate_visual_hallucination_focus_on_the_hallucination/"
    t: "Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**📷 CVPR2025** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md)

🔥 **高频主题：** 多模态 ×2

**[3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception](antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua.md)**

:   提出Antidote——合成数据驱动的统一后训练框架，通过将事实先验注入提示实现模型自校正，将幻觉缓解解耦为偏好优化问题，在LLaVA系列上CP-Bench提升超50%，POPE提升1.8-3.3%，CHAIR/SHR降低30-50%，且无灾难性遗忘。

**[HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)**

:   提出HalLoc，一个15.5万样本、覆盖VQA/指令跟随/图像描述三类任务的token级幻觉标注数据集，并基于此训练了轻量级幻觉检测模型HalLocalizer，可在不影响效率的前提下即插即用地集成到现有VLM中实现实时概率化幻觉检测。

**[Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)**

:   本文通过大量实验揭示了 LVLM 幻觉成因的混合性——不同样本和不同生成步骤面临不同类型的幻觉挑战，据此提出 Octopus 框架，利用可学习的 decision token 和 transformer block 在每个生成步自适应选择最合适的对比解码（CD）策略，通过 DPO 优化，在四个基准上全面超越现有 CD 方法。

**[ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)**

:   本文提出 ODE（Open-set Dynamic Evaluation）协议，通过图结构建模现实世界物体概念及其分布关联，从中动态提取概念组合并生成合成测试图像，实现了开放集、持续更新的多模态幻觉评估，有效避免了现有静态基准可能存在的数据污染问题。

**[One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)**

:   本文提出 PhD，一个 ChatGPT 辅助构建的大规模视觉幻觉评估数据集，包含 14K+ 日常图片、750 张反常识图片和 102K VQA 三元组，通过 4 种评估模式×5 种视觉任务系统化评估多模态大语言模型的幻觉问题，在规模和挑战性上远超现有基准。

**[Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)**

:   提出 FarSight，一种即插即用的无训练解码策略，通过在因果掩码的上三角矩阵中引入注意力寄存器来吸收异常 token 的过度注意力，并设计递减掩蔽率的位置感知编码增强远距离视觉 token 的信息传播，有效缓解多模态大模型中的初始幻觉和雪球幻觉。

**[Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target](stop_learning_it_all_to_mitigate_visual_hallucination_focus_on_the_hallucination.md)**

:   提出**TL-DPO**（Target-Learning DPO），将传统DPO的全句级偏好学习限制到**幻觉发生的目标chunk**和**对应的图像区域**，通过目标生成损失和目标条件损失排除无关信号，在LLaVA-1.5上将CHAIR_s从66.8降至20.1，同时LLaVA-Bench从63.4提升至71.2。

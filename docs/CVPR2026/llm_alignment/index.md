---
title: >-
  CVPR2026 对齐/RLHF论文汇总 · 12篇论文解读
description: >-
  12篇CVPR2026的对齐 / RLHF 方向论文解读，涵盖多模态、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对齐/RLHF"
  - "对抗鲁棒"
item_list:
  - u: "anchoring_the_mind_of_multimodal_reasoners_cognitive_bias_as_a_vector_for_jailbr/"
    t: "Anchoring the Mind of Multimodal Reasoners: Cognitive Bias as a Vector for Jailbreak Attacks"
  - u: "bridging_human_evaluation_to_infrared_and_visible_image_fusion/"
    t: "Bridging Human Evaluation to Infrared and Visible Image Fusion"
  - u: "drm_diffusion-based_reward_model_with_step-wise_guidance/"
    t: "DRM: Diffusion-based Reward Model With Step-wise Guidance"
  - u: "ecoalign_an_economically_rational_framework_for_efficient_lvlm_alignment/"
    t: "EcoAlign: An Economically Rational Framework for Efficient LVLM Alignment"
  - u: "from_pixel_to_precision_enhancing_handwritten_mathematical_expression_recognitio/"
    t: "From Pixel to Precision: Enhancing Handwritten Mathematical Expression Recognition with Image-Level Reward"
  - u: "morphseek_fine-grained_latent_representation-level_policy_optimization_for_defor/"
    t: "MorphSeek: Fine-grained Latent Representation-Level Policy Optimization for Deformable Image Registration"
  - u: "principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la/"
    t: "Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models"
  - u: "safegrpo_self-rewarded_multimodal_safety_alignment_via_rule-governed_policy_opti/"
    t: "SafeGRPO: Self-Rewarded Multimodal Safety Alignment via Rule-Governed Policy Optimization"
  - u: "thinking_with_frames_generative_video_distortion_evaluation_via_frame_reward_mod/"
    t: "Thinking with Frames: Generative Video Distortion Evaluation via Frame Reward Model"
  - u: "uncertainty-aware_exploratory_direct_preference_optimization_for_multimodal_larg/"
    t: "Uncertainty-Aware Exploratory Direct Preference Optimization for Multimodal Large Language Models"
  - u: "unlocking_token_rewards_via_training-free_reward_attribution/"
    t: "Unlocking Token Rewards via Training-Free Reward Attribution"
  - u: "video-coe_reinforcing_video_event_prediction_via_chain_of_events/"
    t: "Video-CoE: Reinforcing Video Event Prediction via Chain of Events"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**📷 CVPR2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (102)](../../ICLR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🧪 ICML2026 (37)](../../ICML2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **高频主题：** 多模态 ×4 · 对齐/RLHF ×3 · 对抗鲁棒 ×2

**[Anchoring the Mind of Multimodal Reasoners: Cognitive Bias as a Vector for Jailbreak Attacks](anchoring_the_mind_of_multimodal_reasoners_cognitive_bias_as_a_vector_for_jailbr.md)**

:   本文发现多模态大推理模型（MLRM）的安全判断存在"锚定效应"——会被最先看到的信息严重带偏，据此提出 RA-Attack：先用一张"看起来安全"的结构化思维导图加教育语境文本把模型的推理链锚定到安全基调，再顺势把有害意图包装成这条推理链的自然延伸，在 7 个主流 MLRM 上把越狱成功率（ASR）刷到 92%（Gemini-2.5-Pro）、82%（GPT-4o）的 SOTA。

**[Bridging Human Evaluation to Infrared and Visible Image Fusion](bridging_human_evaluation_to_infrared_and_visible_image_fusion.md)**

:   针对红外-可见光图像融合（IVIF）长期只优化手工指标、与人眼审美脱节的问题，本文构建了首个大规模 IVIF 人类反馈数据集，训练了一个"融合导向奖励模型"来量化感知质量，再用 SAM 辅助的 GRPO 把融合网络对齐到人类偏好，在主流基准上取得 SOTA 且融合结果更"好看"。

**[DRM: Diffusion-based Reward Model With Step-wise Guidance](drm_diffusion-based_reward_model_with_step-wise_guidance.md)**

:   本文把预训练扩散模型本身当作奖励模型骨干（DRM），利用它能给任意去噪步的噪声潜变量打分这一独特能力，分别设计了密集逐步奖励的 Step-GRPO（训练）和"探索-择优"的 Step-wise Sampling（推理），在不增参数的前提下显著提升 SD3.5-Medium 的生成质量，且收敛速度快 2.5–3.5 倍。

**[EcoAlign: An Economically Rational Framework for Efficient LVLM Alignment](ecoalign_an_economically_rational_framework_for_efficient_lvlm_alignment.md)**

:   EcoAlign 把视觉语言大模型（LVLM）的推理时对齐重新框定为"有限算力预算下的最优路径搜索"问题：在动态构建的思维图上用一个类似净现值（NPV）的前瞻函数给每个候选动作打分，权衡安全、效用与成本，并用"最弱环节"原则定义路径安全，从而在更低算力下达到甚至超过现有方法的安全与效用。

**[From Pixel to Precision: Enhancing Handwritten Mathematical Expression Recognition with Image-Level Reward](from_pixel_to_precision_enhancing_handwritten_mathematical_expression_recognitio.md)**

:   针对手写数学公式识别中"LaTeX 文本相似 ≠ 渲染图像相似"的根本错位，本文提出图像匹配分数 IMS（基于列投影编码 + Levenshtein 距离的轻量图像级奖励），并用它驱动一个去掉 value 网络的 GRPO 强化学习框架 IMPO，在 CROHME / HME100K / M2E 三套基准上把 ExpRate 平均提升约 1.1%、最高 1.37%，刷新 SOTA。

**[MorphSeek: Fine-grained Latent Representation-Level Policy Optimization for Deformable Image Registration](morphseek_fine-grained_latent_representation-level_policy_optimization_for_defor.md)**

:   MorphSeek 把可变形医学图像配准重新定义为「在编码器隐空间里做策略优化」——在 U-Net 编码器顶层接一个高斯策略头把隐特征当作可采样的动作，先无监督 warm-up 稳定隐空间，再用 GRPO 做多轨迹多步弱监督微调，配合 LDVN 让上万维隐空间里的策略梯度稳定下来，在三个 3D 配准基准上用极少标签把 Dice 提了 2–4%、把折叠率（NJD）降了 30–60%。

**[Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)**

:   提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。

**[SafeGRPO: Self-Rewarded Multimodal Safety Alignment via Rule-Governed Policy Optimization](safegrpo_self-rewarded_multimodal_safety_alignment_via_rule-governed_policy_opti.md)**

:   SafeGRPO 把"可验证的规则化奖励"塞进 GRPO，让多模态大模型在无需人工偏好标注的情况下自奖励地学会"先按视觉/文本/组合三层逐步推理安全性、再决定回答还是拒答"，在多个安全基准上同时提升越狱防御、安全意识与稳定性，且几乎不损伤通用能力、不引入过度拒答。

**[Thinking with Frames: Generative Video Distortion Evaluation via Frame Reward Model](thinking_with_frames_generative_video_distortion_evaluation_via_frame_reward_mod.md)**

:   REACT 是一个面向生成视频「结构失真」的帧级奖励模型：先建一套八类失真分类体系并标注 1.5 万对帧偏好数据，用 grounding 重构 + Gemini-2.5-Pro 低成本合成 6K 条 CoT，再以「掩码 SFT + GRPO 成对奖励」两阶段训练 Qwen2.5-VL-7B，推理时用动态采样聚焦最可能失真的帧，在偏好对齐和失真识别两项任务上都显著超过现有视频/图像评估器。

**[Uncertainty-Aware Exploratory Direct Preference Optimization for Multimodal Large Language Models](uncertainty-aware_exploratory_direct_preference_optimization_for_multimodal_larg.md)**

:   UE-DPO 把多模态大模型（MLLM）幻觉抑制的优化重心，从"模型已经看得懂的视觉敏感 token"挪向"模型看不懂、却很关键的认知盲区 token"——用 token 级**认识不确定性**（epistemic uncertainty）量化这些盲区，再按不确定性给 preferred / dispreferred 两支非对称地调节 DPO 梯度强度，在多个幻觉 benchmark 上以更小数据量超过 TPO/V-DPO 等同类方法。

**[Unlocking Token Rewards via Training-Free Reward Attribution](unlocking_token_rewards_via_training-free_reward_attribution.md)**

:   P2T 用一阶 Taylor 近似，把已有过程奖励模型（PRM）打出的「整段」奖励**免训练地**拆解到每个 token 上——只需一次前向+反向就能算出全序列的 token 级奖励，接到 GRPO 后让数学/多模态推理 RL 训练收敛快约 4×、且在 AIME24 上比 outcome reward 提升 +11.5%。

**[Video-CoE: Reinforcing Video Event Prediction via Chain of Events](video-coe_reinforcing_video_event_prediction_via_chain_of_events.md)**

:   针对多模态大模型在"看视频预测未来事件"（VEP）上既缺乏逻辑推理、又不看画面只猜选项的两大毛病，本文提出 Chain of Events（CoE）范式——让模型先把视频切成带时间戳的历史事件链、再基于事件链做因果推理，并用一套两阶段训练（CoE-SFT 注入推理 + CoE-GRPO 用稠密奖励强化事件链构造）把 Qwen2.5-VL-7B 在 FutureBench 上从 52.9% 拉到 75.0%，刷新 VEP 的 SOTA。

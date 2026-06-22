---
title: >-
  ICCV2025 强化学习论文汇总 · 7篇论文解读
description: >-
  7篇ICCV2025的强化学习方向论文解读，涵盖导航、自监督学习、多模态、推理、强化学习、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "强化学习"
  - "论文解读"
  - "论文笔记"
  - "导航"
  - "自监督学习"
  - "多模态"
  - "推理"
  - "布局/合成"
item_list:
  - u: "embodied_navigation_with_auxiliary_task_of_action_description_prediction/"
    t: "Embodied Navigation with Auxiliary Task of Action Description Prediction"
  - u: "mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms/"
    t: "mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs"
  - u: "navq_learning_a_q-model_for_foresighted_vision-and-language_navigation/"
    t: "NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation"
  - u: "progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re/"
    t: "Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement"
  - u: "r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form/"
    t: "R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization"
  - u: "reinforcement_learning-guided_data_selection_via_redundancy_assessment/"
    t: "RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment"
  - u: "robofactory_exploring_embodied_agent_collaboration_with_compositional_constraint/"
    t: "RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📹 ICCV2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (25)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (400)](../../ICLR2026/reinforcement_learning/index.md) · [💬 ACL2026 (46)](../../ACL2026/reinforcement_learning/index.md) · [🧪 ICML2026 (110)](../../ICML2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (58)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/reinforcement_learning/index.md)

🔥 **高频主题：** 导航 ×2

**[Embodied Navigation with Auxiliary Task of Action Description Prediction](embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)**

:   DescRL 将动作描述生成作为强化学习导航的辅助任务，通过从预训练的视觉-语言模型蒸馏知识来训练 ADPredictor，使导航智能体在生成可解释动作描述的同时提升导航性能，在语义音频-视觉导航（SAVNav）等多个任务上实现 SOTA。

**[mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)**

:   提出 mDP3，一种免训练、模型无关的视频帧选择方法，通过条件高斯核在 RKHS 中估计帧相似度，结合行列式点过程（DPP）捕获查询相关性和列表级多样性，再通过马尔可夫决策过程（MDP）建模时序性，在多个长视频 benchmark 上以仅 8 帧输入显著超越均匀采样和现有帧选择方法。

**[NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)**

:   提出 NavQ，一种前瞻性 VLN 智能体，通过 Q-model 在单次前向传播中预测每个候选动作的长期未来语义聚合特征（Q-feature），结合 A* 式搜索策略在目标导向导航中取得显著提升。

**[Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement](progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)**

:   提出Progressor框架，从无标注视频中自监督学习任务无关的奖励函数，通过预测任务进度分布提供稠密奖励信号，并在在线RL训练中通过对抗性push-back策略应对分布偏移问题。

**[R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)**

:   提出 R1-Onevision，通过跨模态推理管线将图像转换为形式化文本表示，结合 SFT + 基于规则的强化学习（GRPO）的两阶段后训练策略，显著提升视觉语言模型的多模态推理能力，在多个数学推理基准上超越 GPT-4o。

**[RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)**

:   提出 RL-Selector，引入 ε-sample cover 概念量化样本冗余度，将数据选择建模为强化学习过程，通过轻量 A2C 策略网络自适应优化选择策略，在多个基准数据集上以更少数据达到接近甚至超越全量训练的泛化性能。

**[RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints](robofactory_exploring_embodied_agent_collaboration_with_compositional_constraint.md)**

:   提出组合约束（compositional constraints）概念来形式化多智能体具身协作中的安全与效率要求，基于此构建了首个多智能体操作基准 RoboFactory，并系统探索了多智能体模仿学习的架构和训练策略。

---
title: >-
  ICLR2026 知识编辑论文汇总 · 9篇论文解读
description: >-
  9篇ICLR2026的知识编辑方向论文解读，涵盖对齐/RLHF、对抗鲁棒、目标跟踪、布局/合成、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "目标跟踪"
  - "布局/合成"
  - "多模态"
item_list:
  - u: "ace_attribution-controlled_knowledge_editing_for_multi-hop_factual_recall/"
    t: "ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall"
  - u: "bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed/"
    t: "Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing"
  - u: "eamet_robust_massive_model_editing_via_embedding_alignment_optimization/"
    t: "EAMET: Robust Massive Model Editing via Embedding Alignment Optimization"
  - u: "energy-regularized_sequential_model_editing_on_hyperspheres/"
    t: "Energy-Regularized Sequential Model Editing on Hyperspheres"
  - u: "fine-tuning_done_right_in_model_editing/"
    t: "Fine-tuning Done Right in Model Editing"
  - u: "got-edit_geometry-aware_generic_object_tracking_via_online_model_editing/"
    t: "GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing"
  - u: "pics_pairwise_image_compositing_with_spatial_interactions/"
    t: "PICS: Pairwise Image Compositing with Spatial Interactions"
  - u: "rote_learning_considered_useful_generalizing_over_memorized_training_examples/"
    t: "Rote Learning Considered Useful: Generalizing over Memorized Training Examples"
  - u: "when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat/"
    t: "When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🔬 ICLR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [🧪 ICML2026 (8)](../../ICML2026/knowledge_editing/index.md) · [💬 ACL2026 (10)](../../ACL2026/knowledge_editing/index.md) · [🤖 AAAI2026 (4)](../../AAAI2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall](ace_attribution-controlled_knowledge_editing_for_multi-hop_factual_recall.md)**

:   ACE 通过神经元级归因发现「隐式主语在多跳推理里扮演 query 神经元、逐层激活 value 神经元」这一被忽视的机制，并据此把编辑从「层级启发式」精细到「query-value 通路」，在多跳事实召回上比 SOTA 的 PMET 在 GPT-J 上高 9.44%、在 Qwen3-8B 上高 37.46%。

**[Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)**

:   揭示大规模模型编辑失败的根本原因是 key embedding 与 residual embedding 之间的结构不一致（embedding misalignment），提出 EAMET 通过渐进式保存已优化的残差 embedding 并用 KL 散度 + MSE 双损失将其邻域结构对齐到 key embedding 空间，在 6 个 LLM、3 个数据集上同时编辑 10k 事实时平均超越 MEMIT 14%（CounterFact）和 8%（ZsRE），且在长前缀和同主语多事实两大鲁棒性场景下表现稳健。

**[Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)**

:   揭示模型编辑中 fine-tuning 被低估的根因是错误的训练 pipeline（深度优先逐样本优化），修正为标准的广度优先 mini-batch 训练后，配合局部化参数调优形成 LocFT-BF，首次支持 10 万次连续编辑和 72B 模型规模。

**[GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[PICS: Pairwise Image Compositing with Spatial Interactions](pics_pairwise_image_compositing_with_spatial_interactions.md)**

:   提出 PICS——一种并行成对图像合成方法，通过 Interaction Transformer 中的掩码引导 MoE 和自适应 α-blending 策略，在单次推理中同时合成两个对象并显式建模遮挡、接触等空间交互关系，全面超越现有序列合成方法。

**[Rote Learning Considered Useful: Generalizing over Memorized Training Examples](rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)**

:   本文提出"先记忆再泛化"两阶段框架，证明 LLM 可以在死记硬背合成关键 token 后，通过极少量语义微调实现泛化，挑战了"记忆阻碍泛化"的传统观点。

**[When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)**

:   提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。

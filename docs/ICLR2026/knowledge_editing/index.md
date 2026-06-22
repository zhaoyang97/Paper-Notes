---
title: >-
  ICLR2026 知识编辑论文汇总 · 15篇论文解读
description: >-
  15篇ICLR2026的知识编辑方向论文解读，涵盖 LLM、对齐/RLHF、对抗鲁棒、目标跟踪、个性化生成、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "目标跟踪"
  - "个性化生成"
  - "布局/合成"
item_list:
  - u: "ace_attribution-controlled_knowledge_editing_for_multi-hop_factual_recall/"
    t: "ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall"
  - u: "bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed/"
    t: "Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing"
  - u: "disentangling_knowledge_representations_for_large_language_model_editing/"
    t: "Disentangling Knowledge Representations for Large Language Model Editing"
  - u: "eamet_robust_massive_model_editing_via_embedding_alignment_optimization/"
    t: "EAMET: Robust Massive Model Editing via Embedding Alignment Optimization"
  - u: "energy-regularized_sequential_model_editing_on_hyperspheres/"
    t: "Energy-Regularized Sequential Model Editing on Hyperspheres"
  - u: "fine-tuning_done_right_in_model_editing/"
    t: "Fine-tuning Done Right in Model Editing"
  - u: "got-edit_geometry-aware_generic_object_tracking_via_online_model_editing/"
    t: "GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing"
  - u: "knowledgesmith_uncovering_knowledge_updating_in_llms_with_model_editing_and_unle/"
    t: "KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning"
  - u: "mobiedit_resource-efficient_knowledge_editing_for_personalized_on-device_llms/"
    t: "MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs"
  - u: "moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll/"
    t: "MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs"
  - u: "pics_pairwise_image_compositing_with_spatial_interactions/"
    t: "PICS: Pairwise Image Compositing with Spatial Interactions"
  - u: "scaling_knowledge_editing_in_llms_to_100000_facts_with_neural_kv_database/"
    t: "Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database"
  - u: "suit_knowledge_editing_with_subspace-aware_key-value_mappings/"
    t: "SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings"
  - u: "tanglescore_tangle-guided_purge_and_imprint_for_unstructured_knowledge_editing/"
    t: "TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing"
  - u: "when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat/"
    t: "When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🔬 ICLR2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [💬 ACL2026 (10)](../../ACL2026/knowledge_editing/index.md) · [🧪 ICML2026 (8)](../../ICML2026/knowledge_editing/index.md) · [🤖 AAAI2026 (4)](../../AAAI2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall](ace_attribution-controlled_knowledge_editing_for_multi-hop_factual_recall.md)**

:   ACE 通过神经元级归因发现「隐式主语在多跳推理里扮演 query 神经元、逐层激活 value 神经元」这一被忽视的机制，并据此把编辑从「层级启发式」精细到「query-value 通路」，在多跳事实召回上比 SOTA 的 PMET 在 GPT-J 上高 9.44%、在 Qwen3-8B 上高 37.46%。

**[Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[Disentangling Knowledge Representations for Large Language Model Editing](disentangling_knowledge_representations_for_large_language_model_editing.md)**

:   针对知识编辑会误伤"同主体但不同关系/客体"的细粒度无关知识这一被忽视的问题，本文提出 DiKE：先用一个可复用的解耦模块把主体表示拆成"与目标知识相关"和"无关"两部分，再只对相关部分做编辑、显式约束无关部分不变，并推导出一个闭式的秩一参数更新公式，在保住细粒度无关知识的同时维持了主流编辑性能。

**[EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)**

:   揭示大规模模型编辑失败的根本原因是 key embedding 与 residual embedding 之间的结构不一致（embedding misalignment），提出 EAMET 通过渐进式保存已优化的残差 embedding 并用 KL 散度 + MSE 双损失将其邻域结构对齐到 key embedding 空间，在 6 个 LLM、3 个数据集上同时编辑 10k 事实时平均超越 MEMIT 14%（CounterFact）和 8%（ZsRE），且在长前缀和同主语多事实两大鲁棒性场景下表现稳健。

**[Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)**

:   揭示模型编辑中 fine-tuning 被低估的根因是错误的训练 pipeline（深度优先逐样本优化），修正为标准的广度优先 mini-batch 训练后，配合局部化参数调优形成 LocFT-BF，首次支持 10 万次连续编辑和 72B 模型规模。

**[GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning](knowledgesmith_uncovering_knowledge_updating_in_llms_with_model_editing_and_unle.md)**

:   本文提出 KnowledgeSmith，把"知识编辑"和"机器遗忘"统一为同一个约束优化问题，并用知识图谱自动生成跨层级（根/中间/叶）、跨数据规模的大规模评测基准，系统揭示了 LLM 知识更新中的传播不对称、一致性-容量权衡、学科依赖等一系列反直觉现象。

**[MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs](mobiedit_resource-efficient_knowledge_editing_for_personalized_on-device_llms.md)**

:   MobiEdit 把经典 locate-and-edit 知识编辑（ROME）里资源沉重的反向传播换成「量化 + 前向零阶梯度估计」，再配早停和前缀激活复用两个系统优化，第一次让 3B LLM 的实时知识编辑能跑在普通商用手机的 NPU 上，内存省 7.1×、能耗省 15.8×、延迟省 3.4×。

**[MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll.md)**

:   MoEEdit 是首个面向 MoE 大模型的「路由稳定」参数修改式知识编辑框架，用「逐专家零空间投影」保证编辑不扰动下游路由器输入，再用随机块坐标下降（BCD）求解器把代价从专家总数解耦到专家隐藏维度，从而在稀疏架构上同时拿下高编辑成功率、强泛化与路由稳定性。

**[PICS: Pairwise Image Compositing with Spatial Interactions](pics_pairwise_image_compositing_with_spatial_interactions.md)**

:   提出 PICS——一种并行成对图像合成方法，通过 Interaction Transformer 中的掩码引导 MoE 和自适应 α-blending 策略，在单次推理中同时合成两个对象并显式建模遮挡、接触等空间交互关系，全面超越现有序列合成方法。

**[Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database](scaling_knowledge_editing_in_llms_to_100000_facts_with_neural_kv_database.md)**

:   本文把现有的 Locate-and-Edit 知识编辑方法重新解释为「查询一个 KV 数据库」，并据此提出 NeuralDB——用一个非线性门控检索模块替换原来的线性扰动 $\Delta$，把可编辑的事实容量从几百条扩展到 100,000 条，同时几乎不损伤模型的通用能力。

**[SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings](suit_knowledge_editing_with_subspace-aware_key-value_mappings.md)**

:   SUIT 把 locate-then-edit 知识编辑里"随手算出来"的键向量 $k$ 和残差向量 $\delta$ 限制到"和这次编辑真正相关"的低维子空间里，从而在几乎不损失编辑成功率的前提下，大幅减少对无关知识的破坏——在 LLaMA3 / GPT-J / Qwen2.5 上的 Specificity 相比强基线 AlphaEdit 翻倍提升。

**[TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing](tanglescore_tangle-guided_purge_and_imprint_for_unstructured_knowledge_editing.md)**

:   本文提出一个无需依赖具体编辑算法、只由「模型 + 知识样本」决定的内在难度指标 **TangleScore**，用它度量某条知识有多「难改」，并据此设计 **PIPE**（先清除旧知识、再印刻新知识的两阶段编辑框架），在四个不同规模 LLM、两个非结构化编辑基准上把泛化性能平均提升 6.49%。

**[When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)**

:   提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。

---
title: >-
  ICLR2026 幻觉检测论文汇总 · 13篇论文解读
description: >-
  13篇ICLR2026的幻觉检测方向论文解读，涵盖多模态、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "LLM"
item_list:
  - u: "barrel_boundary-aware_reasoning_for_factual_and_reliable_lrms/"
    t: "BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs"
  - u: "beyond_in-domain_detection_spikescore_for_cross-domain_hallucination_detection/"
    t: "Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection"
  - u: "cat-po_cross-modal_adaptive_token-rewards_for_preference_optimization_in_truthfu/"
    t: "Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs"
  - u: "chainmpq_interleaved_text-image_reasoning_chains_for_mitigating_relation_halluci/"
    t: "ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations"
  - u: "copy-paste_to_mitigate_large_language_model_hallucinations/"
    t: "Copy-Paste to Mitigate Large Language Model Hallucinations"
  - u: "dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis/"
    t: "Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models"
  - u: "enhancing_hallucination_detection_through_noise_injection/"
    t: "Enhancing Hallucination Detection through Noise Injection"
  - u: "hallucination_begins_where_saliency_drops/"
    t: "Hallucination Begins Where Saliency Drops"
  - u: "look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model/"
    t: "Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation"
  - u: "lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals/"
    t: "LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals"
  - u: "shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de/"
    t: "SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense"
  - u: "token-guard_towards_token-level_hallucination_control_via_self-checking_decoding/"
    t: "Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding"
  - u: "veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes/"
    t: "VeriTrail: Closed-Domain Hallucination Detection with Traceability"
item_total: 13
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🔬 ICLR2026** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** 多模态 ×3 · 推理 ×2 · LLM ×2

**[BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs](barrel_boundary-aware_reasoning_for_factual_and_reliable_lrms.md)**

:   针对大推理模型（LRM）在事实问答上"宁可编也不说不知道"的毛病，本文先定位出两种由"事实性过度思考"引发的病态推理模式，再用"知识边界标注 → 边界感知 SFT → 基于可靠性奖励的 GRPO"三段式训练框架 BARREL，把 DeepSeek-R1-Distill-Llama-8B 的可靠性从 39.33% 拉到 61.48%，且准确率不降反升。

**[Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection](beyond_in-domain_detection_spikescore_for_cross-domain_hallucination_detection.md)**

:   作者发现「由幻觉答案引出的多轮自对话，其不确定性分数会出现远比真实答案剧烈的尖峰抖动」，于是把这种抖动量化成 **SpikeScore**（分数序列的最大二阶差分），用一个阈值就能做到只在单个领域训练、却能跨多个领域稳定检测幻觉，在四个 LLM、六个 benchmark 上的跨域 AUROC 全面超过 PRISM、ICR Probe 等专门的跨域方法。

**[Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs](cat-po_cross-modal_adaptive_token-rewards_for_preference_optimization_in_truthfu.md)**

:   针对多模态大模型的幻觉问题，本文提出 Cat-PO：在 DPO 偏好优化中，仅靠模型自身的跨模态注意力与相似度，为每个回答 token 计算全局/局部/语义三层视觉相关性，融合成一个平滑的 token 奖励来重新加权 DPO 损失并加上 token 级 KL 正则，从而对幻觉 token 做细粒度纠偏，在 AMBER-Generation、MM-Hal 等基准上比现有 SOTA 高 7%–15%。

**[ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations](chainmpq_interleaved_text-image_reasoning_chains_for_mitigating_relation_halluci.md)**

:   ChainMPQ 是一个无需训练的推理框架：把"主体—关系—客体"这一关系问题拆成 5 个互补子问题，按顺序喂给视觉语言模型，并把每一步的文本答案与视觉注意力记忆传递给后续步骤，形成交错的图文推理链，从而在多个 LVLM 和关系幻觉基准上稳定降低关系幻觉。

**[Copy-Paste to Mitigate Large Language Model Hallucinations](copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)**

:   提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

**[Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[Hallucination Begins Where Saliency Drops](hallucination_begins_where_saliency_drops.md)**

:   提出 LVLMs-Saliency 梯度感知诊断框架来量化每个输出 token 的视觉锚定强度，发现"当先前输出 token 对下一个 token 预测的显著性降低时，幻觉就会产生"的关键规律，并基于此设计了 SGRS（显著性引导的拒绝采样）+ LocoRE（局部一致性增强）双机制推理时框架，在多个 LVLM 上显著降低幻觉率。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)**

:   提出 Lumina 框架，通过"上下文-知识信号"检测RAG系统中的幻觉：用MMD度量**外部上下文利用**程度，用跨层token预测演化度量**内部知识利用**程度，无需超参调优即可泛化。

**[SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)**

:   首次将LVLM对象幻觉系统性追溯到视觉编码器，识别出统计偏差（高频模式token过度强调）、固有偏差（预训练主导对象的残余表示）、脆弱性（微小扰动即导致特征失真）三大问题，并提出SHIELD——一个完全免训练的框架，通过token重加权、token减法和对比解码三策略协同防御，在LLaVA-1.5/InstructBLIP/Qwen-VL上全面超越VCD和OPERA等方法。

**[Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)**

:   提出 Token-Guard，一种基于自检验解码的 token 级幻觉控制方法，通过隐空间中的 token 级/段级评分和迭代修正机制，在解码过程中检测并抑制幻觉生成，F1 平均提升 16.3%。

**[VeriTrail: Closed-Domain Hallucination Detection with Traceability](veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)**

:   提出 VeriTrail——首个为多步生成过程（MGS）提供可追溯性的闭域幻觉检测方法，建模生成过程为 DAG 并沿路径逐层验证，同时构建了首批包含所有中间输出和人工标注的 MGS 数据集。

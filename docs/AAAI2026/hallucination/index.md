---
title: >-
  AAAI2026 幻觉检测论文汇总 · 15篇论文解读
description: >-
  15篇AAAI2026的幻觉检测方向论文解读，涵盖 LLM、Agent、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "多模态"
item_list:
  - u: "beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source/"
    t: "Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models"
  - u: "bridging_day_and_night_target-class_hallucination_suppressio/"
    t: "Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation"
  - u: "causally-grounded_dual-path_attention_intervention_for_objec/"
    t: "Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs"
  - u: "does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms/"
    t: "Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs"
  - u: "esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation/"
    t: "ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation"
  - u: "ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity/"
    t: "Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization"
  - u: "hallucinate_less_by_thinking_more_aspect-based_causal_absten/"
    t: "Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models"
  - u: "hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m/"
    t: "Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models"
  - u: "inex_hallucination_mitigation_via_introspection_and_cross-mo/"
    t: "InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration"
  - u: "listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att/"
    t: "Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation"
  - u: "llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat/"
    t: "LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction"
  - u: "multi-agent_undercover_gaming_hallucination_removal_via_coun/"
    t: "MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning"
  - u: "pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative/"
    t: "PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement"
  - u: "verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la/"
    t: "Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models"
  - u: "when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar/"
    t: "When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🤖 AAAI2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** LLM ×4 · Agent ×2 · 多模态 ×2

**[Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)**

:   提出 Composite Reliability Score (CRS)，将校准度、鲁棒性和不确定性量化三个维度统一为单一可解释指标，对 10 个开源 LLM 在 5 个 QA 数据集上进行系统评估，发现 Mistral-8x22B 综合可靠性最高（CRS=0.81），而模型大小并不直接决定可靠性。

**[Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)**

:   系统研究三种幻觉缓解方法（CoVe、DoLa、RAG）对LLM创造力的影响，发现它们对发散性创造力有截然相反的效果——CoVe增强、DoLa抑制、RAG无影响——而收敛性创造力基本不受影响，这一规律跨模型家族和参数规模一致成立。

**[ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation](esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation.md)**

:   构建 ESG-Bench——270 个人工标注 QA 对来自 94 份真实 ESG 报告（2020-2024），提出三阶段幻觉缓解：SFT（有基础答案+「不提供」弃权标签）→ CoT Prompting（2/4步提示模板）→ CoT 微调（人工推理链），其中 4 步 CoT 微调的 Llama-3 达到 92.52% 有答案准确率 + 99.37% 无答案准确率（平衡 96%），且迁移到 HaluEval/BioASQ 也有提升。

**[Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)**

:   针对多模态大模型（MLLM）在强化学习训练中产生幻觉的三大根因——视觉误解、探索多样性不足、样本冲突——分别提出 Caption Reward、奖励方差引导的样本选择、以及基于 NTK 相似度的 InfoNCE 正则化，在多个基准上显著降低幻觉率。

**[Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)**

:   从计算复杂度理论出发证明 Transformer LLM 每步推理复杂度为 $O(N^2 \cdot d)$，基于时间层次定理（Hartmanis-Stearns），任何需要超过此复杂度的计算任务——如 $O(n^3)$ 矩阵乘法、$O(n^k)$ token 组合、TSP 验证等——LLM 必然无法正确完成（即产生幻觉），且 LLM Agent 也无法验证此类任务的正确性。

**[InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)**

:   提出两阶段框架——自适应层注意力（ALA）融合Whisper编码器多层表示以增强噪声鲁棒性，多目标知识蒸馏（MOKD）将clean teacher的语义和注意力分布对齐到noisy student——在多语言噪声ASR基准上显著降低幻觉率和WER。

**[LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction](llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

**[PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)**

:   提出 PASE 框架，通过去噪表示蒸馏（DRD）利用预训练 WavLM 中鲁棒的音韵先验来抑制语言幻觉，同时采用双流表示（高层音素 + 低层声学）消除声学幻觉，在感知质量和内容保真度两方面同时达到 SOTA。

**[Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models](verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la.md)**

:   首次系统研究多模态大语言模型（MLLM）中的动词概念幻觉问题，构建了多维度基准测试，发现现有幻觉缓解方法对动词幻觉无效，并提出基于丰富动词知识微调的基线方法，显著缓解动词幻觉。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。

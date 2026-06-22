---
title: >-
  ICML2026 幻觉检测论文汇总 · 21篇论文解读
description: >-
  21篇ICML2026的幻觉检测方向论文解读，涵盖多模态、LLM、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "LLM"
  - "对抗鲁棒"
item_list:
  - u: "a_unified_definition_of_hallucination_its_the_world_model_stupid/"
    t: "A Unified Definition of Hallucination: It's The World Model, Stupid!"
  - u: "adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l/"
    t: "Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models"
  - u: "automatic_layer_selection_for_hallucination_detection/"
    t: "Automatic Layer Selection for Hallucination Detection"
  - u: "building_reliable_long-form_generation_via_hallucination_rejection_sampling/"
    t: "Building Reliable Long-Form Generation via Hallucination Rejection Sampling"
  - u: "capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc/"
    t: "Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation"
  - u: "finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_/"
    t: "Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy"
  - u: "from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s/"
    t: "From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity"
  - u: "from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view/"
    t: "From Out-of-Distribution Detection to Hallucination Detection: A Geometric View"
  - u: "hallucination_is_a_consequence_of_space-optimality_a_rate-distortion_theorem_for/"
    t: "Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing"
  - u: "hallucinations_undermine_trust_metacognition_is_a_way_forward/"
    t: "Hallucinations Undermine Trust; Metacognition is a Way Forward"
  - u: "harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme/"
    t: "Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping"
  - u: "honest_lying_understanding_memory_confabulation_in_reflexive_agents/"
    t: "Honest Lying: Understanding Memory Confabulation in Reflexive Agents"
  - u: "instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat/"
    t: "Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models"
  - u: "learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat/"
    t: "Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization"
  - u: "mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin/"
    t: "Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating"
  - u: "mm-snowball_evaluating_and_mitigating_hallucination_snowballing_in_multimodal_mu/"
    t: "MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue"
  - u: "realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations/"
    t: "REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations"
  - u: "revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la/"
    t: "Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models"
  - u: "tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling/"
    t: "TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling"
  - u: "when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar/"
    t: "When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)"
  - u: "zero-source_llm_hallucination_detection_with_human-like_criteria_probing/"
    t: "Zero-source LLM Hallucination Detection with Human-like Criteria Probing"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🧪 ICML2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** 多模态 ×7 · LLM ×3 · 对抗鲁棒 ×2

**[A Unified Definition of Hallucination: It's The World Model, Stupid!](a_unified_definition_of_hallucination_its_the_world_model_stupid.md)**

:   这是一篇 position paper，主张把翻译、摘要、开放问答、RAG、多模态、智能体等各路"幻觉"统一成同一件事——**对一个"参考世界模型"的、用户可见的、不准确的世界建模**：每个场景只是对"参考世界 $W$、视图函数 $V$、冲突策略 $P$"三件套做了不同选择，从而把碎片化的定义收敛成一个可比较、可生成大规模基准的通用模板。

**[Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)**

:   这篇论文提出 RUDDER，在 LVLM 的 prefill 阶段从残差更新中提取每样本视觉证据方向，并在解码时用 Beta Gate 自适应注入，从而以接近单次前向的开销降低物体幻觉。

**[Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)**

:   提出 FEPoID（内在维度的首个有效峰值）作为无需训练的自动层选择准则，并结合首句截断策略（FST），在多种 QA 和摘要幻觉检测基准上持续选出接近最优的中间层，显著超越已有基线方法。

**[Building Reliable Long-Form Generation via Hallucination Rejection Sampling](building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)**

:   提出 SHARS 框架，在推理时逐句检测并拒绝幻觉内容、仅保留经验证的事实段落继续生成，配合改进的语义熵检测器 HalluSE，在 FactScore 上将事实精度提升约 20–26%，同时保持甚至增加生成中的事实信息量。

**[Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)**

:   提出 GIFT 方法，通过追踪 VLM 在理解用户查询时视觉注意力的正向变化（"注视转移"）构建视觉显著性图，并在解码阶段同时增强视觉和查询 token 的注意力以保持跨模态融合平衡，在 CHAIR 上最高提升 20.7%，且仅增加 1.13× 延迟。

**[Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)**

:   本文发现 LVLM 幻觉源于对正确视觉证据的"关注不足 + 生成中遗忘"，并观察到注意力对视觉证据存在显著的层间差异（ILVAD），据此提出一个 train-free / plug-and-play 的方法：用层间差分构造视觉证据显著性图，再在生成过程中持续加权视觉证据 token 和"扎根于证据"的文本 token，在 5 个 LVLM × 5 个幻觉/综合 benchmark 上一致降低幻觉。

**[From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)**

:   本文把 LLM 幻觉检测从"看输出概率"切到"看 loss landscape 曲率"——在 embedding 加 Gaussian 噪声测量梯度方向与幅度的扰动，作为 Hessian 谱半径的廉价代理，在 12 个 model-dataset 组合上 AUROC 全面超越 entropy / Semantic Entropy / EigenScore 等基线。

**[From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)**

:   本文把 LLM 的下一 token 预测视为一个超大词表上的分类任务，将两个轻量级 OOD 检测器 NCI（特征与权重向量的接近度）与 fDBD（特征到决策边界的距离）迁移过来，配合"训练特征均值的解析代理 $\mu_G$"和"只在 top-$k$ 候选 token 上算边界距离"两个适配，得到一个**无训练、单样本**的推理类幻觉检测器，在 CSQA / GSM8K / AQuA 上稳定优于困惑度、Semantic Entropy、SelfCheckGPT 等基线。

**[Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing](hallucination_is_a_consequence_of_space-optimality_a_rate-distortion_theorem_for.md)**

:   本文把"LLM 记住随机事实"形式化为带连续置信分数的**成员测试**问题，证明在事实稀疏极限下最优记忆开销恰好等于事实/非事实输出分布之间的最小 KL 散度——即"率失真定理"——并由此推出：在 log-loss 目标下，给定有限记忆，最优策略**不是弃答也不是遗忘**，而是把一定比例的非事实和事实压在同一个高置信点上，幻觉是信息论意义下的最优误差形态。

**[Hallucinations Undermine Trust; Metacognition is a Way Forward](hallucinations_undermine_trust_metacognition_is_a_way_forward.md)**

:   本文是一篇 position paper，论证"彻底消除 LLM 幻觉"在原理上无法逃避一个"区分度税"（discrimination gap → utility tax）；作者主张把目标从"消灭幻觉"改为**忠实表达不确定性**（faithful uncertainty），并把这种 metacognition 视为 agentic LLM 调用工具时不可或缺的控制层。

**[Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)**

:   本文针对大推理模型（LRM）的幻觉检测提出 ARS：不在文本层扰动 reasoning trace，而是**直接在 trace 末端的潜表示上施加小扰动并续解码**得到反事实答案，再用"答案是否一致"作为标签训一个轻量 contrastive 头来塑形 trace-conditioned answer embedding，使后续 embedding-based detector 把幻觉与真实回答分得更开（TruthfulQA 上 AUROC $66.85\to 86.64$）。

**[Honest Lying: Understanding Memory Confabulation in Reflexive Agents](honest_lying_understanding_memory_confabulation_in_reflexive_agents.md)**

:   本文揭露 Reflexion 类 agent 一种系统性失败模式——"记忆虚构 (memory confabulation)"：agent 会把错误的任务理解写进反思记忆并跨 trial 反复使用，作者用 Reflection Repetition Rate (RRR) 量化该现象，并用程序化反馈抽取替代开放式自我诊断，把 ALFWorld 上正确对象提及率从 0% 拉到 86%、RRR 从 0.64 降到 0.10。

**[Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)**

:   本文发现 MLLM 中 instruction token 的中间层嵌入能天然过滤视觉端引入的误导信息，据此提出训练无关的 InsLen 分数（Calibrated Local Score + Context Consistency Score），在 5 个 MLLM × 4 个基准上把对象幻觉检测的 AUROC 拉高最多 13.81%。

**[Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)**

:   将原图与对比负图拼成共享多图上下文，再用锚定指令告诉模型该看哪张，从而让视觉偏好 DPO 的配分函数自动对齐、跑出理论一致的对比目标，并配合精细编辑生成的硬负样本显著降低 VLM 的多模态幻觉。

**[Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)**

:   CRG 把每个注意力头的输出沿视觉/文本两条路线做精确线性分解，用一前向一反向梯度估计两条路线对当前 token 的因果"do-effect"，再仅压制那些视觉与文本符号冲突且 VRI 偏低（即先验主导）的头的文本路线，从而在无需训练的前提下系统性削弱 LVLM 的语言先验幻觉。

**[MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue](mm-snowball_evaluating_and_mitigating_hallucination_snowballing_in_multimodal_mu.md)**

:   本文提出 MM-Snowball 基准（4992 条 6 轮对抗对话）系统刻画多模态大模型在长对话中"幻觉滚雪球"现象，并据此设计训练无关的 CAVR 方法，在表征层刷新视觉信号、在 logit 层裁决文本-视觉冲突，从而显著压平后段对话的性能塌陷曲线。

**[REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)**

:   REALISTA 在 LLM 隐空间里构造"输入相关的编辑方向字典"，把对抗 prompt 优化变成一个 simplex 约束下的连续问题，既保住了 SECA 这类离散方法的语义等价/连贯，又有 LARGO 那种连续方法的搜索灵活度，首次在 GPT-5 这类闭源推理模型 free-form 输出上诱发幻觉成功。

**[Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)**

:   本文把 LVLM 幻觉重新定义为"被语言先验压制的视觉信息缺失"，用正交投影从原始视觉方向中剔除语言先验得到"纯视觉向量"，再用风险门控只在最优深度的单层做稀疏干预，免训练地把 CHAIRS 幻觉率降 ~19% 同时保住 MM-Vet 通用能力。

**[TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling.md)**

:   TAG 把每一步扩散更新沿当前潜变量方向分解为"径向 + 切向"两个分量，只对切向分量额外乘一个 $\eta \ge 1$ 的放大系数，从一阶 Taylor 展开上证明这等价于单调提升对数似然增益，从而把样本拉向数据流形高密度区，几乎零额外算力地缓解扩散模型的语义幻觉。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   CAIA 用 17 个前沿大模型在 178 个时间锚定的加密货币真实任务上构建首个"对抗性高风险"agent 基准，发现：无工具时所有模型只有 12–28% 准确率（接近随机猜测），有工具时最强 GPT-5 也只到 67.4% vs. 人类入门分析师 80%；更致命的是模型 55.5% 的工具调用偏向"不可靠的网页搜索"而绕过权威链上数据，导致 Pass@k 指标系统性掩盖了"靠试错碰运气"的危险行为。

**[Zero-source LLM Hallucination Detection with Human-like Criteria Probing](zero-source_llm_hallucination_detection_with_human-like_criteria_probing.md)**

:   HCPD 把"零源（zero-source，只看问答文本对、拿不到模型内部状态也没外部知识库）幻觉检测"做成"模仿人类评审"的多准则探针——让一个 LLM agent 针对每个问答对自适应生成一组可解释评判准则、赋权、逐准则打分再加权汇总成可信度分；用语义一致性的弱监督 + GRPO 训练这个 agent，推理时多次采样取平均，在 4 个 QA 数据集、多个目标模型上 AUROC 大幅超过现有方法。

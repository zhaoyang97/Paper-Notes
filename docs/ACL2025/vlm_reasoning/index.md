---
title: >-
  ACL2025 VLMReasoning论文汇总 · 18篇论文解读
description: >-
  18篇ACL2025的 VLM Reasoning 方向论文解读，涵盖推理、多模态、LLM、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "LLM"
  - "对抗鲁棒"
item_list:
  - u: "adammeme_adaptively_probe_the_reasoning_capacity_of_multimodal_large_language_mo/"
    t: "AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness"
  - u: "answering_complex_geographic_questions_by_adaptive_reasoning_with_visual_context/"
    t: "Answering Complex Geographic Questions by Adaptive Reasoning with Visual Context and External Commonsense Knowledge"
  - u: "benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g/"
    t: "Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning"
  - u: "chart-based_reasoning_transferring_capabilities_from_llms_to_vlms/"
    t: "Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs"
  - u: "fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning/"
    t: "FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning"
  - u: "finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation/"
    t: "FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation"
  - u: "judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre/"
    t: "Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?"
  - u: "longdocurl_multimodal_long_doc/"
    t: "LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating"
  - u: "mammoth_vl_multimodal_reasoning/"
    t: "MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale"
  - u: "mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso/"
    t: "MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning"
  - u: "mmboundary_reasoning_step_confidence/"
    t: "MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration"
  - u: "progressive_multimodal_reasoning_via_active_retrieval/"
    t: "Progressive Multimodal Reasoning via Active Retrieval"
  - u: "spare_enhancing_spatial_reasoning_in_vision-language_models_with_synthetic_data/"
    t: "SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data"
  - u: "the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_/"
    t: "The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights"
  - u: "tvc_mitigating_visual_forgetting/"
    t: "Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning"
  - u: "visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search/"
    t: "VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search"
  - u: "vrest_tree_search_vlm_reasoning/"
    t: "VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism"
  - u: "wemath_knowledge_reasoning/"
    t: "We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**💬 ACL2025** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×18 · 多模态 ×12

**[AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness](adammeme_adaptively_probe_the_reasoning_capacity_of_multimodal_large_language_mo.md)**

:   提出AdamMeme——一个基于多智能体协作的自适应评估框架，通过迭代生成更具挑战性的meme样本来探测多模态大语言模型(mLLM)在有害内容理解上的推理能力和特定弱点。

**[Answering Complex Geographic Questions by Adaptive Reasoning with Visual Context and External Commonsense Knowledge](answering_complex_geographic_questions_by_adaptive_reasoning_with_visual_context.md)**

:   本文提出一种面向复杂地理问题的自适应推理框架，结合视觉上下文（如地图、卫星图像）和外部常识知识库进行多步推理，根据问题复杂度动态选择推理路径，在地理VQA任务上显著超越直接端到端回答的方法。

**[Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)**

:   本文构建了一个系统性评测基准来评估大型视觉语言模型（LVLM）在基础视觉图结构理解与推理上的能力，发现现有模型在此类任务上表现欠佳，并提出了针对性的改进方法。

**[Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)**

:   本文提出一种将LLM的推理能力迁移到VLM的方法，通过改进图表表示预训练、构造大规模合成推理数据集和多任务微调，使5B参数的PaLI-3在ChartQA上超越10倍大的模型。

**[FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)**

:   构建了金融领域跨模态多跳推理基准 FCMR，包含文本、表格和图表三种模态，分 Easy/Medium/Hard 三个难度等级，最强模型 Claude 3.5 Sonnet 在 Hard 级别仅达 30.4% 准确率，揭示了 MLLM 在信息检索阶段的关键瓶颈。

**[FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)**

:   构建了一个包含 11,000+ 高质量金融多模态样本的评估基准 FinMME，涵盖 18 个金融领域和 10 种图表类型，提出了融合幻觉惩罚和领域归一化的 FinScore 评估体系，实验表明即使 GPT-4o 也仅得 47 分，揭示了 MLLM 在金融领域的显著不足。

**[Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)**

:   系统评估了 13 个开源小型 LVLM（≤9B 参数）作为图表理解和推理任务的评判者，发现部分开源模型（如 LLaVA-Critic-7B）可达到接近 GPT-4 水平的评判能力（约 80% 一致率），但位置偏差和长度偏差等问题仍然普遍存在。

**[LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](longdocurl_multimodal_long_doc.md)**

:   提出 LongDocURL 基准，覆盖理解/数值推理/跨元素定位三大任务类别共 20 个子任务，包含 2325 个高质量 QA 对、覆盖 33000+ 页文档，系统评估 26 种模型配置暴露了当前 LVLM 在长文档理解上的关键性能差距。

**[MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](mammoth_vl_multimodal_reasoning.md)**

:   提出一种可扩展、低成本的方法，仅使用开源模型构建含 1200 万条富含中间推理过程 (CoT) 的多模态指令微调数据集 MAmmoTH-VL-Instruct，训练的 MAmmoTH-VL-8B 在推理基准上达到 SOTA（MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%）。

**[MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)**

:   提出利用代码作为跨模态对齐的监督信号，构建860万图像-代码对数据集ImgCode-8.6M和300万多模态数学指令微调数据集MM-MathInstruct-3M，训练的MathCoder-VL在开源模型中达到多模态数学推理SOTA，在几何问题上超越GPT-4o和Claude 3.5 Sonnet。

**[MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](mmboundary_reasoning_step_confidence.md)**

:   提出 MMBoundary 框架，通过在推理链的每一步插入自然语言置信度表述（而非只在最终回答后给置信度），结合文本+跨模态的自奖励信号估计置信度，并用 SFT+RL 两阶段训练实现步级置信度校准，平均降低 7.5% 校准误差并提升 8.3% 任务准确率。

**[Progressive Multimodal Reasoning via Active Retrieval](progressive_multimodal_reasoning_via_active_retrieval.md)**

:   本文提出AR-MCTS框架，将主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，在多步多模态推理的每一步动态检索关键知识来替代传统beam search采样，自动生成逐步推理标注以渐进式对齐过程奖励模型（PRM），在MathVista、We-Math和GAOKAO-MM上显著提升了多种MLLM的推理性能。

**[SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data](spare_enhancing_spatial_reasoning_in_vision-language_models_with_synthetic_data.md)**

:   本文发现现有VLM数据集中空间关系数据严重匮乏（前17%的关系占据90%以上样本），提出从DOCCI、Localized Narratives和PixMo-Cap等超详细图像描述数据集中，利用LLM自动提取45.5万样本（340万QA对）的空间推理合成数据，微调后的SpaRE模型在What's Up基准上实现最高49%的性能提升，同时不损害通用VL能力。

**[The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)**

:   系统性揭示了现有多模态数学推理模型对视觉信息的利用极其有限——打乱或移除训练图像对模型性能影响甚微——并提出 HC-M3D 基准来真正测试视觉依赖性，发现主流模型无法识别图像中的细微差异。

**[Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning](tvc_mitigating_visual_forgetting.md)**

:   发现 MLLM 在长链 CoT 推理中存在严重的视觉遗忘现象——推理过半后移除图像仅导致 ~2% 的准确率下降，表明模型过度依赖自生成文本而忽视视觉证据。提出 TVC (Take-along Visual Conditioning) 策略，在训练阶段通过动态视觉重确认 (DVR) 注入图像回顾机制，推理阶段通过周期性视觉校准 (PVC) 压缩并重注入视觉 token，在 5 个数学推理基准上平均超越 SOTA 3.4 分（43.4 vs 40.0）。

**[VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)**

:   本文提出VisuoThink框架，通过视觉-文本交织推理和预测性前瞻树搜索，在推理过程中动态整合视觉辅助信息并探索多条推理路径，无需微调即可在几何和空间推理任务上实现SOTA性能（Geomverse-109上Accuracy@1最高达48.5%，相比最优基线提升21.8%）。

**[VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)**

:   VReST 把蒙特卡洛树搜索（MCTS）搬到视觉语言模型（LVLM）上做多模态数学推理：每个树节点是一个推理步、每条路径是一条完整推理链，再用一套**不引入任何额外模型**的多模态自奖励（Self-Reward）给每步打分，从而在不训练的前提下系统地探索推理空间，在三个多模态数学推理基准上拿到 SOTA，并验证了多模态任务也存在测试时扩展律。

**[We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](wemath_knowledge_reasoning.md)**

:   本文提出We-Math基准，包含6.5K视觉数学问题和67个层次化知识概念，通过将复合问题分解为子问题引入四维评估指标（知识不足IK、泛化不足IG、完全掌握CM、机械记忆RM），首次从知识掌握角度系统评估LMM的数学推理过程而非仅关注最终结果。

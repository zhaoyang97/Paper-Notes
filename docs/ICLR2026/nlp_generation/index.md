---
title: >-
  ICLR2026 文本生成论文汇总 · 12篇论文解读
description: >-
  12篇ICLR2026的文本生成方向论文解读，涵盖扩散模型、LLM、问答、对抗鲁棒、文本摘要等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "LLM"
  - "问答"
  - "对抗鲁棒"
  - "文本摘要"
item_list:
  - u: "antislop_a_comprehensive_framework_for_identifying_and_eliminating_repetitive_pa/"
    t: "Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models"
  - u: "causal-steer_disentangled_continuous_style_control_without_parallel_corpora/"
    t: "Causal-Steer: Disentangled Continuous Style Control without Parallel Corpora"
  - u: "diverse_text_decoding_via_iterative_reweighting/"
    t: "Diverse Text Decoding via Iterative Reweighting"
  - u: "fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m/"
    t: "FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model"
  - u: "improving_attributed_long-form_question_answering_with_intent_awareness/"
    t: "Improving Attributed Long-form Question Answering with Intent Awareness"
  - u: "logitkl_flow_matching_nonautoregressive_text_generation_via_samplinghybrid_infer/"
    t: "Logit-KL Flow Matching：用采样-混合推理做非自回归文本生成"
  - u: "p-less_sampling_a_robust_hyperparameter-free_approach_for_llm_decoding/"
    t: "p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding"
  - u: "planner_aware_path_learning_in_diffusion_language_models_training/"
    t: "Planner Aware Path Learning in Diffusion Language Models Training"
  - u: "rainbow_padding_mitigating_early_termination_in_instruction-tuned_diffusion_llms/"
    t: "Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs"
  - u: "rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure/"
    t: "Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure"
  - u: "text_summarization_via_global_structure_awareness/"
    t: "Text Summarization via Global Structure Awareness"
  - u: "unveiling_the_potential_of_diffusion_large_language_model_in_controllable_genera/"
    t: "Unveiling the Potential of Diffusion Large Language Model in Controllable Generation"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**🔬 ICLR2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (17)](../../ACL2026/nlp_generation/index.md) · [🧪 ICML2026 (2)](../../ICML2026/nlp_generation/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md) · [💬 ACL2025 (27)](../../ACL2025/nlp_generation/index.md)

🔥 **高频主题：** 扩散模型 ×4 · LLM ×2

**[Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models](antislop_a_comprehensive_framework_for_identifying_and_eliminating_repetitive_pa.md)**

:   Antislop 把"LLM 生成里那些一眼能认出是 AI 的重复套话（slop）"当成可量化、可定位、可消除的对象：先用频率比统计画出模型专属的"slop 指纹"，再用一个推理期的回溯采样器精准压制这些模式，最后把采样器的拦截记录自动转成偏好数据，用新提出的 FTPO 微调把抑制能力永久焊进权重——在 GSM8K/MMLU/创意写作上几乎不掉点的前提下做到 90% 的 slop 削减。

**[Causal-Steer: Disentangled Continuous Style Control without Parallel Corpora](causal-steer_disentangled_continuous_style_control_without_parallel_corpora.md)**

:   本文提出 Causal-Steer：把 LoRA 当成一次"因果干预"，在**同一条输入**上对比加/不加 LoRA 扰动的激活差，从而摆脱平行语料、抽出一条干净的风格向量，再经 PCA 去噪 + 几何中位数鲁棒聚合，最终在推理时用一个标量 $\alpha$ 实现连续、双向、可线性插值的 LLM 风格控制。

**[Diverse Text Decoding via Iterative Reweighting](diverse_text_decoding_via_iterative_reweighting.md)**

:   本文提出 OverRIDE（Reweighting-based Iterative DEcoding），在多轮采样时用历史生成结果在推理阶段增量微调一个"引导模型"，再用它去压低会导致历史模式重现的 token 概率，从而在几乎不损失质量的前提下显著提升多个回答之间的多样性，并能以 6.4%（72B）的吞吐损失嵌入 vLLM 这类服务系统。

**[FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)**

:   提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。

**[Improving Attributed Long-form Question Answering with Intent Awareness](improving_attributed_long-form_question_answering_with_intent_awareness.md)**

:   针对深度研究系统生成的长文报告"引用质量差、可读性低"的问题，本文提出一套基于标签的双层意图（段落意图 + 引用意图）写作框架，既能在推理时通过 prompt 直接提升大模型，又能用带意图的合成数据蒸馏小模型——在三个科学报告生成基准上，大模型平均涨 +2.9 分、小模型涨 +12.3 分，引用指标提升尤为显著。

**[Logit-KL Flow Matching：用采样-混合推理做非自回归文本生成](logitkl_flow_matching_nonautoregressive_text_generation_via_samplinghybrid_infer.md)**

:   本文用"logit 空间的线性插值"（等价于 simplex 上的 KL 测地线）作为离散流匹配的路径，证明了最大化条件似然恰好恢复出流匹配的速度场，并配上一套"去噪-再加噪"的迭代采样器和混合推理方案，在非自回归文本/代码生成上显著刷低困惑度、刷高 BLEU。

**[p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding](p-less_sampling_a_robust_hyperparameter-free_approach_for_llm_decoding.md)**

:   本文提出 p-less 采样：一种**完全没有超参数**的截断式解码方法，每一步用整个 token 分布的"碰撞概率" $\sum_v P_\theta(v)^2$ 当作动态截断阈值，在数学、逻辑推理和创意写作上都优于 top-p / min-p 等方法，并且在高温下几乎不退化、推理还更快。

**[Planner Aware Path Learning in Diffusion Language Models Training](planner_aware_path_learning_in_diffusion_language_models_training.md)**

:   这篇论文指出掩码扩散语言模型训练时默认的“随机解掩码路径”和推理时实际使用的 planner 路径不一致，并提出 Planner-Aware Path Learning（PAPL），用 planner 置信度重加权 masked diffusion loss，让训练更贴近推理路径，在蛋白序列、文本生成和代码生成上稳定提升质量。

**[Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](rainbow_padding_mitigating_early_termination_in_instruction-tuned_diffusion_llms.md)**

:   本文发现 instruction-tuned 扩散语言模型存在「`<eos>` overflow」早停顽疾——分配的生成长度越长、回答反而越短甚至塌缩成一串 `<eos>`；根因是 `<eos>` 同时被当作终止符和填充符，于是作者提出 Rainbow Padding：只保留一个 `<eos>` 标记真正结束、其余填充位用 K 个不同 padding token 循环铺满，仅靠 7 个 token + 单 epoch LoRA 就能恢复长度鲁棒性，把 LLaDA 在 MATH 上的准确率从 0.6% 拉到 32.6%。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。

**[Text Summarization via Global Structure Awareness](text_summarization_via_global_structure_awareness.md)**

:   GloSA-sum 首次把拓扑数据分析（TDA）引入文本摘要：用持续同调一次性算出文档的语义骨架与逻辑环路存进"保护池"，再用轻量代理指标迭代删句，在不丢核心逻辑链的前提下做到压缩既快又准，并能给下游 LLM 任务缩短上下文。

**[Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](unveiling_the_potential_of_diffusion_large_language_model_in_controllable_genera.md)**

:   本文提出 **Self-adaptive Schema Scaffolding (S3)**——一种训练无关的方法，把结构模板（schema）当作"半去噪初始态"直接灌进扩散大语言模型（dLLM）的输出上下文，再配合 `null` 占位符自适应长度，让 dLLM 在更少去噪步数下稳定生成合法 JSON 等结构化输出，结构合规率从基线的 30%~80% 提升到 99%+，且幻觉率更低。

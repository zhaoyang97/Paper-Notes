---
title: >-
  ICLR2026 多语言/翻译论文汇总 · 8篇论文解读
description: >-
  8篇ICLR2026的多语言/翻译方向论文解读，涵盖翻译、LLM、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "多语言/翻译"
  - "论文解读"
  - "论文笔记"
  - "翻译"
  - "LLM"
  - "对齐/RLHF"
item_list:
  - u: "assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity/"
    t: "ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity"
  - u: "atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and/"
    t: "ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality"
  - u: "discox_benchmarking_discourse-level_translation_in_expert_domains/"
    t: "DiscoX: Benchmarking Discourse-Level Translation in Expert Domains"
  - u: "from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap/"
    t: "From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization"
  - u: "language_confusion_gate_language-aware_decoding_through_model_self-distillation/"
    t: "Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation"
  - u: "linguamap_which_layers_of_llms_speak_your_language_and_how_to_tune_them/"
    t: "LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?"
  - u: "multilingual_routing_in_mixture-of-experts/"
    t: "Multilingual Routing in Mixture-of-Experts"
  - u: "sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod/"
    t: "SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🔬 ICLR2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (63)](../../ACL2026/multilingual_mt/index.md) · [🧪 ICML2026 (3)](../../ICML2026/multilingual_mt/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/multilingual_mt/index.md) · [🧠 NeurIPS2025 (11)](../../NeurIPS2025/multilingual_mt/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/multilingual_mt/index.md) · [🧪 ICML2025 (1)](../../ICML2025/multilingual_mt/index.md)

🔥 **高频主题：** 翻译 ×2

**[ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)**

:   提出 ASSESS 框架，其核心是 TransTED Similarity 指标——通过将形式化数学命题解析为算子树 (Operator Tree)，并在标准树编辑距离 (TED) 基础上融入 Lean 证明策略驱动的语义变换，实现了在 EPLA 基准上 70.16% 准确率和 0.35 Kappa 分数的 SOTA 性能，同时仅需 CPU 资源即可复现。

**[ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)**

:   提出 Adaptive Transfer Scaling Law (ATLAS)，通过将有效数据量分解为目标语言、迁移语言和其他语言三项并引入数据重复饱和函数，在774个多语言训练实验（10M–8B参数、400+语言）上显著优于现有scaling law（多语言 $R^2$ 从0.67提升至0.98），并系统量化了跨语言迁移矩阵、多语言诅咒的容量约束以及预训练vs微调的计算交叉点。

**[DiscoX: Benchmarking Discourse-Level Translation in Expert Domains](discox_benchmarking_discourse-level_translation_in_expert_domains.md)**

:   DiscoX 构建了首个面向**篇章级 + 专家级**中英互译的评测基准（200 篇、平均 1712 token、7 大领域、1330 人时人工打磨），并配套提出多智能体无参考评测系统 Metric-S，揭示出即便最强 LLM（GPT-5-high 76.66）仍落后人类专家（80.16）的真实差距。

**[From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)**

:   提出ALPO(自适应局部偏好优化)用于训练表达力强的字幕翻译LLM：通过实证发现字幕翻译偏好意译且推理型LLM意译能力优于对话型LLM -> 验证LLM作为翻译评估器与人类高度一致 -> 提出逐句段的细粒度过程监督偏好对齐方法(自适应权重+动态beta+前缀混合) -> 14B模型在多方向字幕翻译的鲜活度上超越GPT-4o/DeepSeek-R1等SOTA。

**[Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation](language_confusion_gate_language-aware_decoding_through_model_self-distillation.md)**

:   本文提出 **Language Confusion Gate (LCG)**：一个不改动基座 LLM、只在解码时按需屏蔽错误语言族 token 的轻量两层 MLP，用「范数校准的自蒸馏」训练，把多模型的语言混淆率压低约一个数量级且不损任务性能。

**[LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?](linguamap_which_layers_of_llms_speak_your_language_and_how_to_tune_them.md)**

:   通过 logit lens 与隐状态相似度分析定位出 mLLM 中「负责语言控制」的最后几层，只微调这 3–5% 的参数就能把六种语言的语言一致性从 <20% 拉到 98%+，效果几乎等同全量微调。

**[Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)**

:   系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。

**[SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)**

:   利用稀疏自编码器（SAE）发现 LLM 中意外语言切换与目标语言特征异常高预激活值相关，提出 SASFT 方法在 SFT 训练中约束语言特征预激活值，将意外代码切换降低 50% 以上。

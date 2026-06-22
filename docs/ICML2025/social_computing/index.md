---
title: >-
  ICML2025 社会计算论文汇总 · 6篇论文解读
description: >-
  6篇ICML2025的社会计算方向论文解读，涵盖 LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
item_list:
  - u: "defame_dynamic_evidence-based_fact-checking_with_multimodal_experts/"
    t: "DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts"
  - u: "dynamical_phases_of_short-term_memory_mechanisms_in_rnns/"
    t: "Dynamical Phases of Short-Term Memory Mechanisms in RNNs"
  - u: "learning_survival_distributions_with_the_asymmetric_laplace_distribution/"
    t: "Learning Survival Distributions with the Asymmetric Laplace Distribution"
  - u: "or-bench_an_over-refusal_benchmark_for_large_language_models/"
    t: "OR-Bench: An Over-Refusal Benchmark for Large Language Models"
  - u: "raising_the_bar_investigating_the_values_of_large_language_models_via_generative/"
    t: "Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing"
  - u: "when_bad_data_leads_to_good_models/"
    t: "When Bad Data Leads to Good Models"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🧪 ICML2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md)

🔥 **高频主题：** LLM ×2

**[DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)**

:   提出 DEFAME，一个模块化零样本多模态 LLM 流水线，通过六阶段动态流程（规划→执行→摘要→推理→判决→解释）结合外部多模态工具检索证据，实现端到端的文本-图像联合事实核查，在 AVeriTeC、MOCHEG、VERITE 三个基准上均达到新 SOTA。

**[Dynamical Phases of Short-Term Memory Mechanisms in RNNs](dynamical_phases_of_short-term_memory_mechanisms_in_rnns.md)**

:   本文发现了支持RNN短时记忆的两种不同潜在动力学机制——慢点流形（slow-point manifolds）和极限环（limit cycles），通过解析 toy 模型推导出各自最大可学习率的幂律缩放定律（SP: beta 约4-5 vs LC: beta 约2-3），并通过训练约80,000个RNN进行了大规模实证验证。

**[Learning Survival Distributions with the Asymmetric Laplace Distribution](learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)**

:   提出基于非对称拉普拉斯分布 (ALD) 的参数化生存分析方法，通过神经网络学习 ALD 的三个参数（位置、尺度、不对称性），实现连续、闭式的生存分布估计，在判别性和校准性上全面优于现有参数化与非参数化方法。

**[OR-Bench: An Over-Refusal Benchmark for Large Language Models](or-bench_an_over-refusal_benchmark_for_large_language_models.md)**

:   提出首个大规模 LLM 过度拒绝（over-refusal）基准 OR-Bench，包含 80K 安全但易被拒绝的 prompt，揭示安全性与过度拒绝之间存在 Spearman 相关系数高达 0.89 的强权衡关系。

**[Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing](raising_the_bar_investigating_the_values_of_large_language_models_via_generative.md)**

:   提出 GETA 框架，将心理测量学中的计算机自适应测试（CAT）与自动出题（AIG）结合，通过变分 IRT 和 LLM 驱动的题目生成器动态探测 LLM 的价值边界，解决静态基准因数据泄漏和难度饱和导致的"评估时效性效应"（evaluation chronoeffect）问题。

**[When Bad Data Leads to Good Models](when_bad_data_leads_to_good_models.md)**

:   本文提出"预训练-后训练协同设计"视角，通过受控实验证明在预训练数据中加入适量有毒数据（~10%）反而能降低毒性特征的纠缠度，使模型在后训练阶段（如 ITI 激活引导）更容易去毒，最终在 Toxigen 上将毒性从 41.40 降至 2.63，同时保持语言能力。

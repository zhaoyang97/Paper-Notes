---
title: >-
  AAAI2026 LLM评测论文汇总 · 16篇论文解读
description: >-
  16篇AAAI2026的 LLM 评测方向论文解读，涵盖 LLM、推理、Agent、翻译、机器人、情感分析等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "Agent"
  - "翻译"
  - "机器人"
  - "情感分析"
item_list:
  - u: "bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore/"
    t: "BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction"
  - u: "benchmarking_llms_for_political_science_a_united_nations_perspective/"
    t: "Benchmarking LLMs for Political Science: A United Nations Perspective"
  - u: "beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c/"
    t: "Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents"
  - u: "coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio/"
    t: "ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions"
  - u: "dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le/"
    t: "DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning"
  - u: "do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no/"
    t: "Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy"
  - u: "gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated/"
    t: "Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment"
  - u: "llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab/"
    t: "LLM-as-a-Judge for Scalable Test Coverage Evaluation"
  - u: "lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon/"
    t: "Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory"
  - u: "low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning/"
    t: "Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning"
  - u: "mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s/"
    t: "MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search"
  - u: "mindvote_when_ai_meets_the_wild_west_of_social_media_opinion/"
    t: "MindVote: When AI Meets the Wild West of Social Media Opinion"
  - u: "optscale_probabilistic_optimality_for_inference-time_scaling/"
    t: "OptScale: Probabilistic Optimality for Inference-time Scaling"
  - u: "test-time_diverse_reasoning_by_riemannian_activation_steering/"
    t: "Test-time Diverse Reasoning by Riemannian Activation Steering"
  - u: "towards_a_common_framework_for_autoformalization/"
    t: "Towards a Common Framework for Autoformalization"
  - u: "where_norms_and_references_collide_evaluating_llms_on_normative_reasoning/"
    t: "Where Norms and References Collide: Evaluating LLMs on Normative Reasoning"
item_total: 16
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🤖 AAAI2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (131)](../../ICLR2026/llm_evaluation/index.md) · [💬 ACL2026 (96)](../../ACL2026/llm_evaluation/index.md) · [🧪 ICML2026 (40)](../../ICML2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md) · [🧪 ICML2025 (22)](../../ICML2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×4 · 推理 ×2

**[BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)**

:   本文构建了一个覆盖加拿大BC省2.4亿公顷、跨度25年的多模态野火风险预测数据集BCWildfire，包含38个驱动因子，并对CNN/Linear/Transformer/Mamba四大范式的时序预测模型进行了系统评测，揭示了当前模型在野火预测中的性能上限和关键影响因子。

**[Benchmarking LLMs for Political Science: A United Nations Perspective](benchmarking_llms_for_political_science_a_united_nations_perspective.md)**

:   提出 UNBench，首个基于联合国安理会 1994-2024 年记录的综合性政治科学 LLM 评测基准，涵盖决议起草、投票模拟、通过预测和代表发言生成四个关联任务，评估 LLM 对复杂政治动态的理解和模拟能力。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)**

:   提出 ConInstruct 基准，评估 LLM 在指令包含冲突约束时的检测和解决能力，发现多数专有模型能较好检测冲突但很少主动告知用户，其中 DeepSeek-R1 和 Claude-4.5-Sonnet 在冲突检测上表现最佳（F1 分别达 91.5% 和 87.3%）。

**[DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)**

:   提出 DiCaP（Distribution-Calibrated Pseudo-labeling），通过估计伪标签的后验正确率来校准权重、引入双阈值机制分离置信区间和模糊区间并采用不同策略，在半监督多标签学习中以最高 4.27% 的幅度超越 SOTA。

**[Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)**

:   本文批判性审视了现有NL到一阶逻辑(FOL)翻译的评估方法（FOLIO和MALLS），揭示其数据集与评估协议的根本缺陷，提出了一种将翻译任务分解为本体提取(OE)和逻辑翻译(LT)、并辅以"最相似选择"和"排序"子任务的新型基准测试策略，实验表明对话式LLM（o3-mini、GPT-4o-mini、Qwen3系列）展现出强大的NL-FOL翻译能力与真正的逻辑语义理解，而嵌入式模型表现显著较差。

**[Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment](gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated.md)**

:   本文系统性地测试了三种文本操控策略（冗长、策略性多答案嵌入、正确答案前置+矛盾）对 LLM 答案匹配评判器的影响，发现这些操控**不会提升分数甚至降低分数**，且二值评分比连续评分更鲁棒，证明答案匹配是一种对低成本文本操控具有鲁棒性的评估方法。

**[LLM-as-a-Judge for Scalable Test Coverage Evaluation](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)**

:   将LLM-as-Judge范式应用于Gherkin验收测试覆盖率评估，在20种模型配置x500次评估中系统量化准确性-可靠性-成本三维权衡，发现GPT-4o Mini以6.07 MAAE、96.6% ECR@1和$1.01/1K评估成为最优生产选择，成本仅为GPT-5高推理版的1/78。

**[Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)**

:   提出 PSN-IRT（Pseudo-Siamese Network for IRT），用增强版项目反应理论同时估计 LLM 能力参数和题目的四参数特征（难度/区分度/猜测率/可行性），在 11 个基准 41,871 题上发现当前基准存在广泛饱和、难度天花板不足、数据污染等系统性问题，PSN-IRT 选出的题目子集排名一致性达 Kendall τ=1.00。

**[Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)**

:   提出 LOREN，一种曲率感知的零阶优化方法，通过低秩块对角预条件器捕获损失景观的各向异性曲率，并结合 REINFORCE Leave-One-Out 方差缩减技术，在 LLM 微调中实现了更高精度和更快收敛，同时相比 MeZO-Adam 节省高达 27.3% 的峰值内存。

**[MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)**

:   提出MCTS-SQL，让轻量LLM（如Qwen-1.5B）通过蒙特卡洛树搜索实现强大的Text-to-SQL能力——三组件架构（Selector做Schema剪枝 + Direct Generator生成初始SQL + MCTS-Refiner迭代精化），配合前缀缓存机制减少53%推理时间，Qwen-1.5B在BIRD上达40.69%执行准确率（超ChatGPT-3.5）。

**[MindVote: When AI Meets the Wild West of Social Media Opinion](mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)**

:   提出 MindVote——首个基于真实社交媒体投票数据的 LLM 舆情预测基准，包含 Reddit/微博上 3,918 个自然投票（23 个话题），附带平台和话题上下文。评估 15 个 LLM 发现：最佳模型（o3-medium）1-Wasserstein 仅 0.892 vs 上界 0.972；在调查数据上微调的专用模型反而不如通用模型（"调查特化陷阱"）；模型表现出强烈文化对齐——西方模型擅长 Reddit、中国模型擅长微博。

**[OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)**

:   提出概率最优框架 OptScale，通过建模验证器分数的概率分布推导出最优采样数量的理论下界，动态决定每个问题所需的最少采样次数，在保持推理准确率的同时大幅减少计算开销。

**[Test-time Diverse Reasoning by Riemannian Activation Steering](test-time_diverse_reasoning_by_riemannian_activation_steering.md)**

:   提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。

**[Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)**

:   本文系统回顾了"自动形式化"（autoformalization）在数学证明、逻辑推理、规划和知识表示等领域的现有工作，提出了一个统一的跨学科定义框架，将自动形式化定义为从非形式语言到形式推理语言的语义等价转换，旨在促进不同研究社区间的方法共享并加速下一代 AI 推理系统的发展。

**[Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)**

:   提出 SNIC 诊断测试台（9,000 实例/51 场景），评估 LLM 能否利用隐式社会规范来解决歧义参考消解（如"递给我杯子"时存在多个杯子）。发现 LLM 在仅看场景描述时平均准确率仅 44%，加上 Prolog 形式逻辑无显著改善（44.2%），但显式提供规范列表后猛升到 70.5%（GPT-4.1 达 99.6%），证明 LLM 缺乏隐式物理规范知识但能有效利用显式规范。

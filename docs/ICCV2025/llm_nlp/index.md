---
title: >-
  ICCV2025 LLM其他论文汇总 · 6篇论文解读
description: >-
  6篇ICCV2025的 LLM 其他方向论文解读，涵盖 LLM、持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "持续学习"
item_list:
  - u: "any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag/"
    t: "Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models"
  - u: "any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models/"
    t: "Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models"
  - u: "fw-merging_scaling_model_merging_with_frank-wolfe_optimization/"
    t: "FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization"
  - u: "shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer/"
    t: "ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer"
  - u: "va_gpt_aligning_effective_tokens_video_anomaly/"
    t: "VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models"
  - u: "vim_versatile_interactive_motion_language_model/"
    t: "VIM: Versatile Interactive Motion-Language Model"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**📹 ICCV2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (55)](../../ICLR2026/llm_nlp/index.md) · [💬 ACL2026 (61)](../../ACL2026/llm_nlp/index.md) · [🧪 ICML2026 (39)](../../ICML2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×3 · 持续学习 ×2

**[Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag.md)**

:   提出Analytic Subspace Routing（Any-SSR）框架，通过为每个任务分配独立的LoRA子空间消除任务间干扰，并利用递归最小二乘（RLS）闭式解训练一个零遗忘的解析路由器，实现LLM的无回放持续学习。

**[Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models.md)**

:   提出Analytic Subspace Routing (Any-SSR)，为每个新任务分配独立的LoRA子空间以消除知识干扰，同时使用基于递归最小二乘(RLS)闭式解的分析路由器动态选择子空间，在理论上保证不遗忘先前任务知识，实现LLM的无重放持续学习。

**[FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)**

:   将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。

**[ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)**

:   提出ShadowHack框架，将阴影去除分解为亮度恢复和颜色修复两个子任务，通过带有纠偏外展注意力的LRNet恢复亮度和纹理，再用跨注意力驱动的CRNet重建准确颜色，在ISTD+和SRD数据集上取得SOTA。

**[VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)**

:   提出 VA-GPT，一个面向视频异常事件理解的多模态大模型，通过空间有效token选择(SETS)和时间有效token生成(TETG)两个模块，让MLLM在空间和时间维度上精准对齐异常相关信息，在域内和跨域异常检测基准上均达到SOTA。

**[VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)**

:   提出 VIM，首个能在统一框架内同时理解和生成双人交互运动与文本的多模态大模型，配合82.7K多轮交互运动指令数据集 Inter-MT²，支持文本到运动、运动到文本、反应生成、运动编辑和运动推理等多种任务。

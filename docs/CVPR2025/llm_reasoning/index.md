---
title: >-
  CVPR2025 LLMReasoning论文汇总 · 7篇论文解读
description: >-
  7篇CVPR2025的 LLM Reasoning 方向论文解读，涵盖推理、LLM、Agent、少样本学习、目标检测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "LLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "Agent"
  - "少样本学习"
  - "目标检测"
item_list:
  - u: "argus_vision-centric_reasoning_with_grounded_chain-of-thought/"
    t: "Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought"
  - u: "enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation/"
    t: "Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation"
  - u: "interleaved-modal_chain-of-thought/"
    t: "Interleaved-Modal Chain-of-Thought"
  - u: "learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte/"
    t: "Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework"
  - u: "osrcir_reflective_cot/"
    t: "Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval"
  - u: "style_evolving_along_chain-of-thought_for_unknown-domain_object_detection/"
    t: "Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection"
  - u: "videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas/"
    t: "VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM Reasoning

**📷 CVPR2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (16)](../../CVPR2026/llm_reasoning/index.md) · [🔬 ICLR2026 (241)](../../ICLR2026/llm_reasoning/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_reasoning/index.md) · [🧪 ICML2026 (78)](../../ICML2026/llm_reasoning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (82)](../../NeurIPS2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×6

**[Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)**

:   Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。

**[Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)**

:   AoTD 用 LLM agent 将复杂视频问题分解为子任务、调用专家视觉模型执行并收集中间结果作为推理链（CoT），经 LLM 质量过滤后蒸馏到 Video-LLM 中，让端到端模型同时获得准确答案和可解释的多步推理能力。

**[Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework](learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte.md)**

:   提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。

**[Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval](osrcir_reflective_cot.md)**

:   本文提出OSrCIR，一种免训练的单阶段零样本组合图像检索方法，利用多模态大语言模型直接处理参考图像和修改文本，并通过反思式链式思维推理准确理解用户隐含意图，在多个基准上比现有免训练方法提升1.80%~6.44%。

**[Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。

**[VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)**

:   VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。

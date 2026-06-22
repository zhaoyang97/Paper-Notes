---
title: >-
  ECCV2024 LLM其他论文汇总 · 11篇论文解读
description: >-
  11篇ECCV2024的 LLM 其他方向论文解读，涵盖少样本学习、异常检测、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "异常检测"
  - "Agent"
item_list:
  - u: "adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero/"
    t: "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection"
  - u: "apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre/"
    t: "APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension"
  - u: "cultural_value_differences_llms/"
    t: "Cultural Value Differences of LLMs: Prompt, Language, and Model Size"
  - u: "freestyleret_retrieving_images_from_style-diversified_queries/"
    t: "FreestyleRet: Retrieving Images from Style-Diversified Queries"
  - u: "funqa_towards_surprising_video_comprehension/"
    t: "FunQA: Towards Surprising Video Comprehension"
  - u: "promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua/"
    t: "PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts"
  - u: "propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona/"
    t: "Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos"
  - u: "reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression/"
    t: "Reprojection Errors as Prompts for Efficient Scene Coordinate Regression"
  - u: "roadpainter_points_are_ideal_navigators_for_topology_transformer/"
    t: "RoadPainter: Points Are Ideal Navigators for Topology Transformer"
  - u: "stripe_observation_guided_inference_cost-free_attention_mechanism/"
    t: "Stripe Observation Guided Inference Cost-Free Attention Mechanism"
  - u: "zeroshot_object_counting_with_good_exemplars/"
    t: "Zero-Shot Object Counting with Good Exemplars (VA-Count)"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**🎞️ ECCV2024** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (55)](../../ICLR2026/llm_nlp/index.md) · [💬 ACL2026 (61)](../../ACL2026/llm_nlp/index.md) · [🧪 ICML2026 (39)](../../ICML2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)**

:   在CLIP中同时引入静态（全局共享）和动态（逐图生成）两种可学习提示，用辅助异常检测数据训练后，在14个工业+医学异常检测数据集上实现零样本SOTA，核心在于"任务级+实例级"双层自适应的混合提示设计。

**[APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)**

:   本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。

**[Cultural Value Differences of LLMs: Prompt, Language, and Model Size](cultural_value_differences_llms.md)**

:   本文使用 Hofstede 文化维度问卷系统性地研究 LLM 表达文化价值观的行为模式，发现提示语言（中文 vs 英文）和模型规模对文化价值差异的影响远大于模型架构差异和问题顺序变化。

**[FreestyleRet: Retrieving Images from Style-Diversified Queries](freestyleret_retrieving_images_from_style-diversified_queries.md)**

:   提出首个风格多样化查询图像检索（Style-Diversified QBIR）任务及数据集DSR，设计了轻量即插即用的FreestyleRet框架，通过Gram矩阵提取查询的纹理/风格特征，构建风格空间并以此初始化prompt token，使冻结的视觉编码器能适配文本、草图、低分辨率、艺术画等多种查询风格的检索。

**[FunQA: Towards Surprising Video Comprehension](funqa_towards_surprising_video_comprehension.md)**

:   构建了大规模反直觉视频问答基准 FunQA（4.3K 视频、312K QA 对），覆盖幽默/创意/魔术三类令人惊讶的视频，并提出 FunMentor 智能体通过多轮对话增强 VLM 的反常识推理能力。

**[PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)**

:   提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

**[Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos](propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)**

:   VidAssist提出"提议-评估-搜索"三步框架，利用LLM作为知识库和评估工具，结合广度优先搜索算法，在教学视频的目标导向规划任务中以零/少样本方式超越全监督SOTA，few-shot在COIN上比全监督VLaMP高+7.7% SR。

**[Reprojection Errors as Prompts for Efficient Scene Coordinate Regression](reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)**

:   本文提出 EGFS（Error-Guided Feature Selection）机制，利用低重投影误差区域作为 SAM 的 point prompts 扩展为语义掩码，迭代地筛选可靠训练样本，在 Cambridge Landmarks 和 Indoor6 数据集上以更小模型和更少训练时间超越现有无 3D 信息依赖的 SCR 方法。

**[RoadPainter: Points Are Ideal Navigators for Topology Transformer](roadpainter_points_are_ideal_navigators_for_topology_transformer.md)**

:   提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。

**[Stripe Observation Guided Inference Cost-Free Attention Mechanism](stripe_observation_guided_inference_cost-free_attention_mechanism.md)**

:   本文通过深入分析Transformer中注意力权重矩阵的条纹（stripe）模式现象，提出一种推理阶段完全无额外计算开销的注意力增强机制——仅在训练阶段通过辅助模块学习条纹引导的注意力修正，并在推理时将其重参数化融入标准注意力权重中，实现"免费午餐"式的性能提升。

**[Zero-Shot Object Counting with Good Exemplars (VA-Count)](zeroshot_object_counting_with_good_exemplars.md)**

:   提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。

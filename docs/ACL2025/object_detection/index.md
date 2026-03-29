<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**💬 ACL2025** · 共 **4** 篇

**[Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions.md)**

:   首次对 GPT-2 系列在选择题中的"锚定偏差"（始终偏好选项 A）进行全面机械可解释性分析——通过 Logit Lens 定位导致偏差的 MLP 值向量和注意力头，然后更新值向量+重校准注意力权重，以最小干预消除偏差并将 MCQ 准确率平均提升 70%+。

**[Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)**

:   提出 Dolphin，一个轻量级（322M）的文档图像解析模型，采用"先分析后解析"（analyze-then-parse）两阶段范式——先进行页面级布局分析生成阅读顺序的元素序列，再利用异构锚点提示（heterogeneous anchor prompting）并行解析各元素内容，以仅 322M 参数在页面级和元素级解析任务上超越 7B+ 模型和商业系统。

**[Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning](weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec.md)**

:   提出Delora框架，通过引入clean LoRA和noisy LoRA双模块构建噪声标签检测器，将样本选择与模型训练解耦，打破传统"小损失"方法中样本选择与训练互相影响的恶性循环。

**[Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region](why_safeguarded_ships_run_aground_aligned_large_language_models_safety_mechanism.md)**

:   揭示了安全对齐LLM的一个普遍现象：安全机制过度锚定在chat template区域（TASA），导致越狱攻击可通过干扰template区域的信息处理来绕过安全防线，并提出通过将安全探针从template区域迁移到生成阶段来缓解该漏洞。

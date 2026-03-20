---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🧠 NeurIPS2025** · 共 **6** 篇

**[A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)**

:   提出一组通用化组件（Component A/B/C），通过充分挖掘样本选择与触发器之间的双向协作关系，同时提升 Poison-only Clean-label 后门攻击的攻击成功率（ASR）和隐蔽性，并在多种攻击类型上展现了良好的泛化能力。

**[Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)**

:   提出 FedLEASE——解决联邦 LoRA 微调中两个关键问题：(1) 用 LoRA B 矩阵相似度聚类自动确定最优专家数量和分配，(2) 用扩展路由空间（$2M-1$ 维）实现自适应 top-M 专家选择（每个客户端自动决定用几个专家），在 GLUE 上比最强基线平均提升 5.53%。

**[Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)**

:   提出 Adversarial Paraphrasing——一种无需训练的通用攻击框架，在逐 token 改写时利用 AI 文本检测器的反馈信号选择"最像人写"的 token，使改写后的 AI 文本在 8 种检测器上平均 T@1%F 下降 87.88%，且具有跨检测器的强迁移性。

**[AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift](ai_should_sense_better_not_just_scale_bigger_adaptive_sensin.md)**

:   提出"自适应感知"作为AI发展的范式级转变——受生物感觉系统启发，主张在传感器层面动态调整输入参数（如曝光、增益、多模态配置），而非仅靠扩大模型规模来应对分布偏移，实证表明5M参数的EfficientNet-B0通过自适应感知可超越632M参数的OpenCLIP-H。

**[Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)**

:   提出COLA——一个training-free的框架，通过将对抗扰动后的图像特征投影到文本特征张成的子空间来消除非语义噪声，再用最优传输(OT)在分布层面细粒度对齐图文特征，在14个零样本分类基准上平均提升6.7%的对抗鲁棒准确率，同时维持干净样本性能。

**[LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)**

:   论文不再把大模型战略推理简单等同于“是否接近纳什均衡”，而是基于 behavioral game theory 构建评测框架，区分真实推理能力与上下文因素，系统测评 22 个 LLM 的互动决策行为，发现模型规模并不决定战略水平，CoT 提升也并非普遍有效，同时暴露出显著的人口属性偏置。

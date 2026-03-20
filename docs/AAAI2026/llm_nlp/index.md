---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**🤖 AAAI2026** · 共 **7** 篇

**[A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)**

:   在 LLM 注意力权重上训练 CNN 来评估记忆化分类法与实际注意力机制的对齐程度，提出新的三类分类法（Guess/Recall/Non-Memorized），最小 F1 从 64.7% 提升至 89.0%，并定位了不同记忆类型分别依赖低层（Guess）和高层（Recall）注意力。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](rectification_reimagined_a_unified_mamba_model_for_image_cor.md)**

:   从统一畸变矫正视角出发，提出 UniRect 框架，通过 Residual Progressive TPS 处理几何形变 + Residual Mamba Blocks 补偿退化，统一处理肖像校正、广角矩形化、拼接矩形化、旋转校正四种任务，并通过 Sparse MoE 实现 four-in-one 多任务学习，拼接矩形化 PSNR 提升 3.82 dB，旋转校正提升 0.87 dB。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

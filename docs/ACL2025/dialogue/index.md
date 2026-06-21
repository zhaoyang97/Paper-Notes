---
title: >-
  ACL2025 对话系统论文汇总 · 18篇论文解读
description: >-
  18篇ACL2025的对话系统方向论文解读，涵盖对话系统、情感分析、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "情感分析"
  - "个性化生成"
item_list:
  - u: "demo_reframing_dialogue_interaction_with_fine-grained_element_modeling/"
    t: "DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling"
  - u: "dialogue_systems_for_emotional_support_via_value_reinforcement/"
    t: "Dialogue Systems for Emotional Support via Value Reinforcement"
  - u: "dynamic_label_name_refinement_for_few-shot_dialogue_intent_classification/"
    t: "Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification"
  - u: "enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system/"
    t: "Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System"
  - u: "enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an/"
    t: "Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction"
  - u: "enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic/"
    t: "EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance"
  - u: "know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici/"
    t: "Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles"
  - u: "know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation/"
    t: "Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling"
  - u: "kokorochat_a_japanese_psychological_counseling_dialogue/"
    t: "KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors"
  - u: "persona_sentiment_dialogue/"
    t: "Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation"
  - u: "personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi/"
    t: "PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants"
  - u: "reflectdiffu_empathetic_response/"
    t: "ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework"
  - u: "share_shared_memory-aware_open-domain_long-term_dialogue_dataset_constructed_fro/"
    t: "SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script"
  - u: "single-_vs_dual-prompt_dialogue_generation_with_llms_for_job_interviews_in_human/"
    t: "Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources"
  - u: "sparse_rewards_can_self-train_dialogue_agents/"
    t: "Sparse Rewards Can Self-Train Dialogue Agents"
  - u: "uniconv_retrieval_response_gen/"
    t: "UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations"
  - u: "when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue/"
    t: "When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation"
  - u: "wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_/"
    t: "Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**💬 ACL2025** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md) · [🧪 ICML2025 (2)](../../ICML2025/dialogue/index.md)

🔥 **高频主题：** 对话系统 ×16 · 情感分析 ×3 · 个性化生成 ×2

**[DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)**

:   本文提出对话元素建模（Dialogue Element Modeling）这一新任务，系统定义了对话生命周期中从"前奏"到"尾声"的全面元素体系，构建了包含元素感知和对话智能体交互两大能力的DEMO benchmark，并通过模仿学习训练DEMO agent在域内外任务上均表现优异。

**[Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)**

:   提出 ES-VR，首个将人类价值观强化融入情感支持对话系统的方法，通过目标价值检测器和参考生成器（均在 Reddit 数据上训练），结合 SFT + DPO 两阶段训练，使支持者模型不仅能缓解求助者的负面情绪，还能探索和强化其积极价值观，实现更深层的内在转变。

**[Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification](dynamic_label_name_refinement_for_few-shot_dialogue_intent_classification.md)**

:   提出动态标签名称精炼方法，在检索式 ICL 意图分类中，利用 LLM 根据检索到的示例动态生成更具区分性的意图标签名称（如 "Verify PAN" → "Verify PAN card details"），有效降低语义相似意图间的混淆，在 6 个数据集上一致提升 2.07%-7.51% 准确率。

**[Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System](enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)**

:   本文提出赋予聊天机器人"眼睛和耳朵"的沉浸式多模态对话系统，构建了融合视觉与听觉的多会话多方对话数据集 M3C，并设计了包含对话模块和多模态记忆检索模块的对话模型，实现了多说话者共享视听体验的动态长期对话。

**[Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)**

:   提出模型无关的两阶段 CRC 框架（一致性反思 + 一致性纠正），通过先让模型反思生成回复与对话上下文之间的不一致之处、再据此纠正回复，显著提升了目标导向主动对话系统中生成回复与对话上下文的一致性。

**[EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)**

:   提出 EnSToM，一种基于熵缩放转向向量的轻量级方法，通过利用 LLM 内部层级熵分布差异来动态调整转向强度，在不修改模型参数的情况下提升任务导向对话系统的主题维持能力。

**[Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)**

:   本文提出 USP（User Simulator with Implicit Profiles）框架，通过从人机对话中提取隐式用户画像，并结合条件监督微调和基于循环一致性的强化学习，在真实性、一致性和多样性三个维度上显著超越基线方法，语义相似度和风格相似度分别提升约 34% 和 43%。

**[Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)**

:   本文提出面向任务型对话系统的 Accountability Model，在 LLM 中加入额外的 accountability head 作为二分类器预测对话状态中各 slot 的概率，从而检测并自校正假阳性和假阴性错误，在 MultiWOZ 上将 JGA 从 64.34 提升到 70.51（↑9.6%），达到 SOTA。

**[KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](kokorochat_a_japanese_psychological_counseling_dialogue.md)**

:   提出 KokoroChat，一个通过训练有素的咨询师角色扮演收集的日语心理咨询对话数据集，包含 6,589 段长对话及详细的客户反馈评分，用于提升 LLM 的心理咨询回复生成和对话评估能力。

**[Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)**

:   大规模分析发现 LLM 生成的个性化对话质量对人物画像的情感极性高度敏感——负面画像导致过度强调人设引发矛盾，正面画像则选择性融入人设产生更高质量对话——基于此提出结合轮次生成、画像排序和情感感知提示的改进方法。

**[PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants](personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi.md)**

:   提出 PersonaLens，一个面向任务导向型 AI 助手个性化能力的综合评测基准，包含 1500 个丰富用户画像、20 个领域 111 个任务、用户模拟 Agent 和 Judge Agent，通过大规模自动化评估揭示当前 LLM 助手在个性化方面的显著不足。

**[ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](reflectdiffu_empathetic_response.md)**

:   提出轻量级共情对话框架 ReflectDiffu，融合情感传染（捕捉情绪）、意图二次机制（Exploring-Sampling-Correcting将情绪映射为行动意图）和扩散模型生成，在相关性、可控性和信息量上全面超越现有基线和 Llama-3.1-8B。

**[SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script](share_shared_memory-aware_open-domain_long-term_dialogue_dataset_constructed_fro.md)**

:   提出了基于电影剧本构建的长期对话数据集 SHARE，首次引入「共享记忆」概念，并设计了 EPISODE 对话框架来管理个人信息、个人事件和共享记忆，使长期对话更具亲密感和参与度。

**[Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources](single-_vs_dual-prompt_dialogue_generation_with_llms_for_job_interviews_in_human.md)**

:   本文系统比较了使用 LLM 生成求职面试对话的两种策略——单提示（一次性生成完整对话）和双提示（两个 agent 分别扮演面试官和候选人轮流对话），发现双提示方法生成的对话在自然度上的胜率是单提示的 2-10 倍，但 token 成本增加约 6 倍。

**[Sparse Rewards Can Self-Train Dialogue Agents](sparse_rewards_can_self-train_dialogue_agents.md)**

:   提出 JOSH（Juxtaposed Outcomes for Simulation Harvesting）自对齐算法，让 LLM 对话 Agent 通过稀疏奖励的模拟环境自主提升性能，无需外部人类反馈，并构建了 ToolWOZ 稀疏奖励工具调用模拟环境加以验证。

**[UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](uniconv_retrieval_response_gen.md)**

:   探索如何将对话场景中的稠密检索和响应生成统一到单个 LLM 中，通过三个联合训练目标（对话检索 + 响应生成 + 上下文识别指令）和数据差异缓解机制，在五个对话搜索数据集上实现检索和生成的相互促进，超越分离式基线。

**[When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)**

:   系统性地研究了人设对话生成中**对话者（interlocutor）信息**对目标说话人生成质量的影响，通过遮蔽/揭示对话者信息的评估框架发现：模型能有效适应对话者人设、对陌生对话者泛化能力弱于陌生话题，且零样本设置下LLM倾向于"复制粘贴"人设细节。

**[Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)**

:   本文提出 TRACER 方法，利用决策树模型规划对话路径，引导两个 LLM Agent（顾客和卖家）生成自然且有目标导向的电商购物对话，并发布了包含 3600 条对话的 Wizard of Shopping (WoS) 数据集，在对话查询生成和商品排序两个下游任务上验证了数据集的有效性。

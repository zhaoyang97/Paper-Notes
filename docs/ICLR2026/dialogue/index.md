---
title: >-
  ICLR2026 对话系统论文汇总 · 10篇论文解读
description: >-
  10篇ICLR2026的对话系统方向论文解读，涵盖对话系统、推理、LLM、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "个性化生成"
item_list:
  - u: "aqua_toward_strategic_response_generation_for_ambiguous_visual_questions/"
    t: "AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions"
  - u: "clarifyvc_clarifying_ambiguous_commands_in_vehicle_control_with_a_hybrid_data_au/"
    t: "ClarifyVC: Clarifying Ambiguous Commands in Vehicle Control with a Hybrid Data Augmentation Pipeline"
  - u: "codified_finite-state_machines_for_role-playing/"
    t: "Codified Finite-state Machines for Role-playing"
  - u: "drift_learning_from_abundant_user_dissatisfaction_in_real-world_preference_learn/"
    t: "DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning"
  - u: "dropping_just_a_handful_of_preferences_can_change_top_large_language_model_ranki/"
    t: "Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings"
  - u: "flipping_the_dialogue_training_and_evaluating_user_language_models/"
    t: "Flipping the Dialogue: Training and Evaluating User Language Models"
  - u: "non-collaborative_user_simulators_for_tool_agents/"
    t: "Non-Collaborative User Simulators for Tool Agents"
  - u: "rein_conversational_error_recovery_with_reasoning_inception/"
    t: "ReIn: Conversational Error Recovery with Reasoning Inception"
  - u: "think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio/"
    t: "Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation"
  - u: "understanding_language_prior_of_lvlms_by_contrasting_chain-of-embedding/"
    t: "Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🔬 ICLR2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md) · [🧪 ICML2025 (2)](../../ICML2025/dialogue/index.md) · [💬 ACL2025 (18)](../../ACL2025/dialogue/index.md)

🔥 **高频主题：** 对话系统 ×2 · 推理 ×2

**[AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)**

:   提出 AQuA，首个按模糊度细粒度分级（4 级）的视觉问答数据集（7.2K 样本），为每级定义最优回应策略（直接回答/推断/列举/请求澄清），发现 GPT-5 和 Gemini 在模糊 VQA 上都过度自信地直接回答，通过 SFT+GRPO 训练的 3B 模型反而能超越闭源大模型的策略适应能力。

**[ClarifyVC: Clarifying Ambiguous Commands in Vehicle Control with a Hybrid Data Augmentation Pipeline](clarifyvc_clarifying_ambiguous_commands_in_vehicle_control_with_a_hybrid_data_au.md)**

:   ClarifyVC 用一个 Agent 编排的四阶段数据增强流水线，从 2 万条真实车载指令里"种"出大量歧义丰富、协议合规的单轮/多轮对话，配上三层评测协议与数据质量分 DQS，在这套数据上微调后让车载语音指令的解析准确率提升约 15%、歧义消解提升约 20%、协议合规度达 98%。

**[Codified Finite-state Machines for Role-playing](codified_finite-state_machines_for_role-playing.md)**

:   针对 LLM 角色扮演时只会模仿表层动作、记不住人物"内在状态"的问题，本文让 LLM 把人物档案自动**编译成可执行的有限状态机（CFSM）**，用代码显式记录角色状态及其转移规则，并进一步扩展成用概率分布建模状态的 CPFSM；在合成验证和 Fandom 真实剧情基准上都比纯 prompt 的状态建模基线更连贯、更可解释。

**[DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning](drift_learning_from_abundant_user_dissatisfaction_in_real-world_preference_learn.md)**

:   DRIFT 把真实部署里大量但隐式的"用户不满"（DSAT）当作高质量负样本锚点，正样本则从当前策略动态采样，用标准 DPO 迭代训练，无需人工标注/奖励模型/更强模型生成的正例，就让 14B 模型在 WildBench 上超过 GPT-4o-mini。

**[Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings](dropping_just_a_handful_of_preferences_can_change_top_large_language_model_ranki.md)**

:   本文提出一个计算极快的稳健性检验：在 Chatbot Arena 这类基于 Bradley–Terry 模型的 LLM 排行榜上，只要丢掉**最坏情况**下极小一撮（最少 0.003%、两条）人类偏好评测，就能让排名第一的模型换人——并且方法还能精确指出是哪几条偏好导致了翻盘。

**[Flipping the Dialogue: Training and Evaluating User Language Models](flipping_the_dialogue_training_and_evaluating_user_language_models.md)**

:   把对话"翻转"过来——不再训练 LLM 当好助手，而是专门后训练一个**用户语言模型（User LM）**去模拟真实人类用户，用它在多轮对话里逼出助手 LM 在真实场景下的短板（GPT-4o 任务成功率从 74.6% 掉到 57.4%）。

**[Non-Collaborative User Simulators for Tool Agents](non-collaborative_user_simulators_for_tool_agents.md)**

:   基于marketing研究定义四类非协作用户行为（不可用服务/跑题闲聊/不耐烦/不完整表述），构建了可保持goal-alignment的模拟框架，在MultiWOZ和τ-bench上系统暴露了SOTA工具Agent的行为特异性失败机制——跑题闲聊导致平均SR下降29.1%，且不同模型呈现截然不同的崩溃路径（GPT系列陷入helper API重复调用，Qwen系列倾向于幻觉编造API结果）。

**[ReIn: Conversational Error Recovery with Reasoning Inception](rein_conversational_error_recovery_with_reasoning_inception.md)**

:   提出 Reasoning Inception（ReIn），一种无需修改模型参数或系统提示的测试时干预方法，通过外部 inception 模块检测对话错误并将恢复计划注入任务 agent 的推理链中，在多种错误场景下显著提升对话任务完成率，且可泛化至未见错误类型。

**[Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)**

:   FlyThinker 提出了一种高效的 "think-while-generating" 框架，使用独立的推理模型(Reasoner)在 token 级别并行生成潜在推理信号，动态融入生成模型(Generator)以指导个性化长文本生成，同时保持训练和推理效率。

**[Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding](understanding_language_prior_of_lvlms_by_contrasting_chain-of-embedding.md)**

:   通过对比有/无视觉输入的逐层隐藏表征（chain-of-embedding），发现LVLM中存在一个"视觉整合点"(VIP)层，并据此提出Total Visual Integration (TVI)指标来量化语言先验的强度。

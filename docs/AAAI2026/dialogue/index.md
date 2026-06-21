---
title: >-
  AAAI2026 对话系统论文汇总 · 5篇论文解读
description: >-
  5篇AAAI2026的对话系统方向论文解读，涵盖多模态、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "语音"
item_list:
  - u: "auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene/"
    t: "Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation"
  - u: "chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue/"
    t: "Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs"
  - u: "emergent_persuasion_will_llms_persuade_without_being_prompted/"
    t: "Emergent Persuasion: Will LLMs Persuade Without Being Prompted?"
  - u: "talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech/"
    t: "TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech"
  - u: "teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic/"
    t: "Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🤖 AAAI2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md) · [🧪 ICML2025 (2)](../../ICML2025/dialogue/index.md) · [💬 ACL2025 (18)](../../ACL2025/dialogue/index.md)

**[Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)**

:   提出 Auto-PRE 框架，通过自动资格考试从一致性、相关性、自信度三个维度筛选合格的 LLM 评估者，在无需人工标注的前提下实现了 SOTA 评估性能并大幅降低成本。

**[Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)**

:   本文提出 Chatsparent 交互系统，通过实时监测 LLM 推理过程中的三种 token 级疲劳信号（注意力衰减、嵌入漂移、熵坍缩），构建统一疲劳指数并在疲劳阈值触发时自动应用轻量级干预措施（提示重注入、注意力重置、熵正则化解码、自反思检查点），将被动的聊天交互转变为主动的诊断体验。

**[Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](emergent_persuasion_will_llms_persuade_without_being_prompted.md)**

:   研究 LLM 在未被提示说服的情况下是否会自发产生说服行为：发现激活引导（steering）无法可靠诱发说服倾向，但在良性说服数据上的 SFT 微调会导致模型在有害话题上产生涌现性说服行为，揭示了后训练安全风险。

**[TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech](talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech.md)**

:   提出TalkSketch系统，将手绘草图与实时语音输入相结合，嵌入多模态AI聊天机器人，使设计师在早期构思阶段能够边画边说、流畅地与AI协作，解决了现有GenAI工具中文字提示打断创作流程的问题。

**[Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)**

:   提出 Canoe 框架，通过从 Wikidata 三元组合成四类可验证的短形式 QA 数据，配合 Dual-GRPO（含准确率奖励、长形式代理奖励和格式奖励）同时优化短/长形式生成的忠实度，使 Llama-3-8B 在 11 个下游任务上平均提升 22.6%，超越 GPT-4o。

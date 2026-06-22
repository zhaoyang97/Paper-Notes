---
title: >-
  ACL2026 AI安全论文汇总 · 5篇论文解读
description: >-
  5篇ACL2026的 AI 安全方向论文解读，涵盖对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
item_list:
  - u: "omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da/"
    t: "OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset"
  - u: "on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference/"
    t: "On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference"
  - u: "reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via/"
    t: "Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF"
  - u: "signals_are_not_states_neuro-symbolic_safeguards_for_culturally_aware_classroom_/"
    t: "Signals Are Not States: Neuro-Symbolic Safeguards for Culturally Aware Classroom AI"
  - u: "univid_unified_vision-language_model_for_video_moderation/"
    t: "UniVid: 统一视频审核的视觉语言模型"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (145)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/ai_safety/index.md) · [🧪 ICML2026 (114)](../../ICML2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md) · [📹 ICCV2025 (24)](../../ICCV2025/ai_safety/index.md)

**[OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)**

:   本文构建了首个大规模、多领域、基于真实案例的 LLM 安全合规数据集 OmniCompliance-100K，包含 12,985 条人工整理的法规/政策规则和 106,009 条通过 Web 搜索智能体采集的真实合规案例，覆盖 AI 安全、数据隐私、金融、医疗等 9 个领域，并通过广泛的基准实验揭示了当前 LLM 在安全合规能力上的系统性短板。

**[On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)**

:   这篇论文指出 Transformer 安全推理中常用的“洗牌后公开中间激活”防御并不安全，并提出一种先把不同随机置换下的激活对齐、再解线性方程抽取权重的攻击，在 Pythia-70m 和 GPT-2 上能以约 1 美元查询成本恢复近似可用的模型权重。

**[Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)**

:   提出 Reverse Constitutional AI (R-CAI)，通过反转 Constitutional AI 的原则为"毒性宪法"，结合批评-修正循环和概率截断的 RLAIF 机制，实现自动化、可控的多维度对抗毒性数据合成，同时通过概率截断解决奖励黑客导致的语义退化问题（语义连贯性提升15%）。

**[Signals Are Not States: Neuro-Symbolic Safeguards for Culturally Aware Classroom AI](signals_are_not_states_neuro-symbolic_safeguards_for_culturally_aware_classroom_.md)**

:   论文主张课堂 AI 不该把"沉默、回避眼神、语码转换"这类文化情境化的信号直接读成"低参与、不专心、能力差"的教育判断，提出神经符号框架 NSCR：先把多模态信号落成带不确定性、来源和**文化作用域**的类型化事实，再通过可执行推理与治理策略组合出有据声明，证据不足或有刻板印象风险时**主动弃答（DEFER）**。

**[UniVid: 统一视频审核的视觉语言模型](univid_unified_vision-language_model_for_video_moderation.md)**

:   UniVid 通过用统一的策略感知字幕 VLM 替代 1000+ 个黑盒分类器，将视频审核系统从不可维护的"碎片化"架构演进为可解释、可复用的"端到端"审核系统，在 ByteDance 平台生产部署中相比传统方案违规泄漏率下降 42.7%。

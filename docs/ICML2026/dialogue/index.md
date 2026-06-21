---
title: >-
  ICML2026 对话系统论文汇总 · 5篇论文解读
description: >-
  5篇ICML2026的对话系统方向论文解读，涵盖 LLM、压缩/编码、对话系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "压缩/编码"
item_list:
  - u: "context-driven_incremental_compression_for_multi-turn_dialogue_generation/"
    t: "Context-Driven Incremental Compression for Multi-Turn Dialogue Generation"
  - u: "discoverllm_from_executing_intents_to_discovering_them/"
    t: "DiscoverLLM: From Executing Intents to Discovering Them"
  - u: "from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu/"
    t: "From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents"
  - u: "is_your_llm_overcharging_you_tokenization_transparency_and_incentives/"
    t: "Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives"
  - u: "not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving/"
    t: "Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🧪 ICML2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md) · [🧪 ICML2025 (2)](../../ICML2025/dialogue/index.md) · [💬 ACL2025 (18)](../../ACL2025/dialogue/index.md)

🔥 **高频主题：** LLM ×2

**[Context-Driven Incremental Compression for Multi-Turn Dialogue Generation](context-driven_incremental_compression_for_multi-turn_dialogue_generation.md)**

:   多轮对话里把整段历史拼进 prompt 既贵又会丢线索，本文提出 **C-DIC**：把对话看成交织的「话题线索」，在一块紧凑记忆里存可修订的逐线索压缩状态，每轮跑一个轻量的「检索 → 修订 → 写回」循环，并配套**检索感知的截断时序反传（ra-TBPTT）**训练，在数百轮对话上保持稳定的延迟和困惑度。

**[DiscoverLLM: From Executing Intents to Discovering Them](discoverllm_from_executing_intents_to_discovering_them.md)**

:   DiscoverLLM 把 "用户没想清楚自己要什么" 形式化为意图层级树的渐进发现过程，用可奖励的层级化用户模拟器训练模型在不清晰时主动发散探索、在清晰时收敛执行，在创意写作 / 技术写作 / SVG 三任务上比 CollabLLM 等 baseline 满意度 +10%、对话长度 -40%。

**[From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)**

:   针对"多轮交互式工具调用 Agent"后训练里两大瓶颈——高质量数据贵 + 用户模拟噪声毁 RL 信号，作者提出"自演化多 agent 数据合成 (AReaL-SEA)"配套生成可执行 verifier 当奖励，再配上"先 SFT 用户模型再做大 batch + 动态过滤 GRPO"的 RL recipe，在 τ²-bench 上把 Qwen3-235B 推到 Airline 73.0 / Telecom 98.3 的 pass^1，全面达到或超过 Claude/Gemini/GPT-5。

**[Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)**

:   本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布，多收费而不被发现也只是 NP-Hard 而非不可行——作者给出一个简单启发式算法在保持合理性的前提下实测最多多收 11.2% 的 token，最后证明唯一能消除该激励的可加性定价机制是"按字符长度线性计费"。

**[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)**

:   本文指出多轮对话场景下传统 Prefill-Decode 分离架构因每轮都要 P→D 重算并传输 KV 而严重低效，提出 PPD（Prefill-capable Decode）动态路由系统，让 decode 节点根据 SLO 权重决定是否本地处理 Turn 2+ 的 append-prefill，把 Turn 2+ TTFT 降低约 68%。

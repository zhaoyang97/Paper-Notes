<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**💬 ACL2025** · 共 **4** 篇

**[Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)**

:   首次将偏好优化（DPO + RLAIF）应用于无文本口语语言模型（SLM）——从预训练 TWIST 模型生成多个语音续写候选，通过 ASR→LLM 评分自动创建偏好数据对，用 DPO 训练 SLM 一致性地生成语义更好的语音续写，结合课程学习进一步提升。在 ZeroSpeech/StoryCloze 基准上达到 SLM SOTA（sWUGGY 77.9%、S-StoryCloze 61.1%、T-StoryCloze 86.8%）。

**[An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](eierl_dialogue_policy.md)**

:   提出 EIERL 方法，将进化算法（EA）的全局搜索能力与深度强化学习（DRL）的局部优化能力结合用于任务导向对话策略学习，并设计精英个体注入（EII）机制自适应地将高性能个体注入 EA 种群以加速进化，在 4 个数据集上显著提升探索-利用平衡。

**[Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)**

:   提出RL-Profiler方法，用强化学习训练一个帖子相关性过滤器（SelNet），从用户Profile的大量帖子中筛选出与人格特征相关的少量帖子，再交给LLM零样本预测人格，在大幅减少上下文长度的同时保持接近使用全部帖子的预测效果。

**[TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](treerl_tree_search_rl.md)**

:   提出 TreeRL，将基于熵引导的树搜索（EPTree）直接集成到 LLM 的 on-policy 强化学习训练中，通过在高不确定性 token 处分叉来扩展推理路径多样性，并利用树结构提供的全局+局部优势作为过程监督信号，在数学和代码推理任务上超过传统的多链采样 RL。

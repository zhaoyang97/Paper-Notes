---
title: >-
  [论文解读] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback
description: >-
  [ACL 2025][强化学习][Spoken Language Model] 本文提出 Align-SLM 框架，首次将偏好优化（DPO + RLAIF）应用于纯语音语言模型（无文本注入），通过 LLM 自动评估生成的语音续写质量构建偏好数据，结合课程学习迭代提升 SLM 的语义理解能力，在 ZeroSpeech 和 StoryCloze 等基准上达到 SLM 的 SOTA。
tags:
  - "ACL 2025"
  - "强化学习"
  - "Spoken Language Model"
  - "DPO"
  - "RLAIF"
  - "preference optimization"
  - "semantic alignment"
---

# Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback

**会议**: ACL 2025  
**arXiv**: [2411.01834](https://arxiv.org/abs/2411.01834)  
**代码**: 无（基于开源 TWIST 模型）  
**领域**: 强化学习 / 语音语言模型  
**关键词**: Spoken Language Model, DPO, RLAIF, preference optimization, semantic alignment

## 一句话总结

本文提出 Align-SLM 框架，首次将偏好优化（DPO + RLAIF）应用于纯语音语言模型（无文本注入），通过 LLM 自动评估生成的语音续写质量构建偏好数据，结合课程学习迭代提升 SLM 的语义理解能力，在 ZeroSpeech 和 StoryCloze 等基准上达到 SLM 的 SOTA。

## 研究背景与动机

**领域现状**：Textless NLP 利用离散语音单元训练口语语言模型（SLM），通过 next-speech-token prediction 实现端到端语音到语音建模，绕过传统的 ASR → LM → TTS 级联管线。TWIST 等模型通过文本 LLM 初始化和大规模训练数据推进了这一方向。

**现有痛点**：尽管现有 SLM 能产生连贯的短语，但在长程语义性上仍远落后于文本 LLM。生成的语音续写常出现重复短语、语法错误和低相关性问题。SpeechGPT 等方法通过中间文本步骤改善语义，但仍然依赖文本 token 引导，且增加解码延迟，不适合实时交互场景。

**核心矛盾**：相比文本子词，语音 token 粒度更细、信息密度更低、在频谱和时间维度上变异性更大。单纯的 next-speech-token prediction 目标可能忽视了长程语义——模型学会了"听上去像语音"，但不一定学会了"说有意义的话"。

**本文目标** SLM 的输出质量不一致——有时生成高质量续写，有时生成无意义内容。能否训练 SLM 一致地生成好的续写、避免差的续写？

**切入角度**：借鉴文本 LLM 领域的 RLHF/DPO 对齐方法论。由于让人类标注员听语音样本来标注偏好既昂贵又耗时，采用 RLAIF 思路——用 LLM 自动评估 ASR 转录文本的语义质量，构建偏好数据对。

**核心 idea**：对预训练 SLM 采样多条语音续写，用 LLM 评分构建 (chosen, rejected) 偏好对，通过 DPO 训练 LoRA adapter 让 SLM 学习生成更有语义的语音。

## 方法详解

### 整体框架

给定语音 prompt $x$（约 3 秒音频），SLM 通过 nucleus sampling 生成 $N=5$ 条不同的语音续写 $y_1, ..., y_N$。续写通过 vocoder 合成为波形后，再用 Whisper ASR 转写为文本。文本传入 LLM 评估器打分，根据分数构建偏好对 $(y_c, y_r)$。最后用 DPO 目标在偏好数据上训练 SLM 的 LoRA adapter，可选配合课程学习迭代提升。

### 关键设计

1. **自动偏好数据选择策略**:

    - 功能：无需人工标注，自动从多条生成中挑选偏好对
    - 核心思路：探索了两种 AI 反馈方式。(1) **Perplexity (PPL)**：用 Mistral-7B 计算续写文本在 prompt 条件下的困惑度，PPL 最低且 auto-BLEU ≤ δ 的为 chosen，PPL 最高的为 rejected。(2) **Mistral Score**：用指令调优的 Mistral-7B 直接对续写文本评分（1-5分），评估语义连贯性和相关性，设定 $s_c$ 和 $s_r$ 阈值来区分 chosen 和 rejected。auto-BLEU（2-gram 自重复率）用于过滤重复无意义的生成——如果 auto-BLEU > δ=0.1，直接标记为 rejected
    - 设计动机：PPL 优化倾向于语法正确性而非整体语义；Mistral Score 提供更全面的语义反馈。实验验证 Mistral Score 在所有下游指标上全面优于 PPL

2. **DPO 训练 SLM**:

    - 功能：通过隐式奖励学习对齐 SLM 的生成偏好
    - 核心思路：标准 DPO 损失 $\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)})]$。$\pi_{ref}$ 是冻结的预训练模型，$\pi_\theta$ 仅通过 LoRA adapter（rank=32, alpha=8）进行参数更新。偏好数据离线准备，避免在线采样 + vocoder + ASR + LLM 的计算开销
    - 设计动机：DPO 相比 RLHF 不需要训练外部奖励模型，训练简单稳定。LoRA 确保模型不会偏离预训练分布太远（配合 $\beta$ 控制偏离程度）。离线数据准备使整个框架高效可行

3. **课程学习迭代**:

    - 功能：逐步提高偏好数据的质量标准，进一步提升模型
    - 核心思路：第 1 轮 DPO 训练后，模型已经变强了。用更强的模型重新采样续写，并提高偏好阈值（$s_c$: 3→4, $s_r$: 1→2），构建更难区分的偏好对进行第 2 轮训练
    - 设计动机：类似课程学习的渐进式难度提升——先让模型学会区分明显好坏，再学会区分微妙差异。实验中最多做到 3 轮迭代仍有持续改善

### 损失函数 / 训练策略

DPO 损失函数如上。Batch size 512，峰值学习率 1e-6 线性衰减至 100K 步（500 步 warmup）。对于 mls 扩展数据，训练增至 300K 步。课程学习第 2 轮无 warmup。使用 64 张 A100 GPU 训练。模型选择基于 dev-clean 上最高的 reward accuracy。

## 实验关键数据

### 主实验：Align-SLM 7B 与基线对比

| 方法 | sWUGGY↑ | sBLIMP↑ | S-StoryCloze↑ | T-StoryCloze↑ | GPT4-o↑ |
|------|---------|---------|---------------|---------------|---------|
| GSLM | 64.8 | 54.2 | 53.3 | 66.6 | - |
| TWIST 7B | 73.5 | 58.8 | 55.1 | 75.4 | 2.70 |
| Moshi 7B | 72.6 | 58.8 | 60.8 | 83.0 | - |
| SPIRIT-LM 7B | 69.0 | 58.3 | 61.0 | 82.9 | - |
| **Align-SLM-mls+CL 7B** | **77.9** | **62.3** | **61.1** | **86.8** | **3.50** |
| 人类水平 | - | - | 79.2 | 90.2 | - |

### 消融实验：反馈方式对比（1.3B）

| 配置 | Mistral Score↑ | T-StoryCloze↑ | GPT4-o↑ |
|------|---------------|---------------|---------|
| 预训练 TWIST | 1.66 | 69.7 | 1.82 |
| 继续微调 (NTP) | 1.70 (+0.04) | 70.7 (+1.0) | 1.83 (+0.01) |
| Align-SLM w/PPL | 1.88 (+0.22) | 67.7 (-2.0) | 1.85 (+0.03) |
| **Align-SLM w/Mistral** | **2.17 (+0.51)** | **74.2 (+4.5)** | **2.06 (+0.24)** |

### 关键发现

- **Mistral Score 全面优于 PPL 作为偏好信号**：PPL 反馈虽然改善了语法（sBLIMP +2.1），但在语义连贯性（T-StoryCloze）上反而下降了 2.0 分。PPL 过于关注局部流畅性而忽视全局语义。
- **课程学习持续带来改善**：从 7B 模型看，T-StoryCloze 从 83.8（1 轮）升至 85.6（2 轮），GPT4-o 从 3.50 升至 3.56。实验做到 3 轮仍有提升。
- **数据规模有帮助但边际递减**：加入约 3 倍 MLS 数据后 7B 模型 T-StoryCloze 从 85.6 升至 86.8，但 GPT4-o 提升有限。这可能因为 LibriSpeech 的 63K 偏好数据对 7B 模型已较充足。
- **人类评估验证了客观指标**：MMOS 评分上 Align-SLM (3.73±0.06) 不仅超过预训练模型 (3.48±0.07)，甚至超过原始续写的重合成版 (3.50±0.07)。
- **T-StoryCloze 达到 86.8，接近人类水平 90.2**，这是纯语音模型（无文本注入）首次达到的水平。

## 亮点与洞察

- **首次在纯语音 SLM 上应用偏好优化**的系统性工作。通过 ASR + LLM 评估的链条绕开了听觉偏好标注的高成本，使 RLAIF 在语音领域变得可行。这一范式可直接推广到其他语音生成任务（如对话、翻译等）。
- **auto-BLEU 过滤是关键细节**：SLM 生成的重复无意义语音 PPL 可以很低（因为重复模式是可预测的），不加过滤的偏好数据反而会让模型学到错误方向。δ=0.1 的阈值基于对真实续写和生成续写的 auto-BLEU 分布分析确定。
- **课程学习+DPO 的组合**自然且有效：模型变强后重新采样生成→提高筛选标准→迭代，这是一种自我进化（self-improvement）的训练策略，不需要外部更强的教师模型。

## 局限性

- **依赖 ASR 和 LLM 的级联**：偏好数据质量受 ASR 转录质量和 LLM 评判能力的制约。对于无书面文字的语言（SLM 的核心优势场景），这一管线需要替换为语音翻译。
- **仅关注语义**，未涉及说话风格、韵律、情感等语音特有的质量维度，而这些对于自然对话同样重要。
- **训练数据局限于有声书领域**（LibriSpeech/MLS），领域多样性不足可能限制模型的泛化能力。
- **模型规模仍相对小**（1.3B/7B），相比文本 LLM 的百亿参数级别，SLM 在语义理解上的天花板可能较低。

## 相关工作与启发

- **vs SPIRIT-LM (Nguyen et al., 2024)**：SPIRIT-LM 通过语音-文本交错训练提升语义，需要配对的语音-文本数据。Align-SLM 完全不注入文本 token，仅用偏好优化就在关键指标上超过 SPIRIT-LM（T-StoryCloze 86.8 vs 82.9）。
- **vs Moshi (Défossez et al., 2024)**：Moshi 使用文本引导的语音生成实现实时对话，属于多模态模型。Align-SLM 作为纯语音模型在 sWUGGY (77.9 vs 72.6) 和 T-StoryCloze (86.8 vs 83.0) 上均超过 Moshi。
- **vs SpeechAlign (Zhang et al., 2024)**：SpeechAlign 将偏好优化用于 TTS 的语音合成质量对齐，而非 SLM 的语义理解，任务目标不同。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在 SLM 上系统性地应用 RLAIF+DPO，虽然各组件（DPO, LoRA, curriculum learning）都不新，但组合应用的场景新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多个基准、模型规模、反馈方式、数据量的全面消融，加上人类评估验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，实验分析到位，附录包含丰富细节
- 价值: ⭐⭐⭐⭐⭐ 为 SLM 领域确立了偏好优化范式，SOTA 结果令人信服，框架可扩展到更多语音任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[NeurIPS 2025\] Behavior Injection: Preparing Language Models for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/behavior_injection_preparing_language_models_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICML 2025\] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning](../../ICML2025/reinforcement_learning/optimizing_language_models_for_inference_time_objectives_using_reinforcement_lea.md)
- [\[ACL 2025\] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)

</div>

<!-- RELATED:END -->

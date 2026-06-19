---
title: >-
  [论文解读] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
description: >-
  [NeurIPS 2025 Spotlight][机器人][记忆增强] 提出 Memo，一种基于 Transformer 的记忆增强框架，通过周期性生成摘要 token（summary tokens）压缩历史上下文，在保持甚至超越全上下文 Transformer 性能的同时，将推理时 KV 缓存缩小 8-10 倍，并展现出更好的长上下文泛化和流式推理鲁棒性。
tags:
  - "NeurIPS 2025 Spotlight"
  - "机器人"
  - "记忆增强"
  - "Transformer"
  - "上下文压缩"
  - "长期规划"
  - "具身智能"
---

# Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.19732](https://arxiv.org/abs/2510.19732)  
**代码**: [GitHub](https://github.com/gunshi/memo)  
**领域**: 强化学习  
**关键词**: 记忆增强, Transformer, 上下文压缩, 长期规划, 具身智能

## 一句话总结

提出 Memo，一种基于 Transformer 的记忆增强框架，通过周期性生成摘要 token（summary tokens）压缩历史上下文，在保持甚至超越全上下文 Transformer 性能的同时，将推理时 KV 缓存缩小 8-10 倍，并展现出更好的长上下文泛化和流式推理鲁棒性。

## 研究背景与动机

具身智能体在长时间任务中需要利用历史经验进行决策，但现有方法面临根本性矛盾：

**Transformer 的二次注意力瓶颈**：全上下文 Transformer 在每步都需关注所有历史时间步，注意力复杂度为 $O(n^2)$，处理长序列时计算和存储成本极高。训练时梯度需传播过大量序列，推理时 KV 缓存不断增长。

**循环模型的固定记忆限制**：RNN 等循环模型虽然记忆高效，但固定大小的隐状态无法充分保留长时间依赖的信息，且梯度在长序列上容易消失或爆炸。

**已有压缩方法的不足**：语言模型领域的 Recurrent Memory Transformer (RMT) 使用固定大小记忆，而 Autocompressors (AC) 在微调时截断梯度传播，两者都无法在 RL 的长期信用分配场景中有效学习。

核心动机：**能否让 Transformer 学会"摘要"过去的经验，既保持表达力又实现记忆高效？**

## 方法详解

### 整体框架

Memo 在标准 Transformer 策略中引入上下文摘要机制。在训练过程中，将输入序列分割为等长的片段（segment），每个片段末尾由可学习的摘要嵌入（summary embeddings）触发 Transformer 生成摘要 token，这些摘要 token 被存入专用记忆缓冲区，后续时间步通过注意力机制访问这些压缩的历史表示，而非维护完整的原始上下文。

### 关键设计

1. **上下文摘要机制**：将长序列分割为长度 $l_{seg}$ 的片段，在每个片段末尾生成 $l_{sum}$ 个摘要 token。关键创新是**摘要累积**（summary accumulation）——不同于 RMT 仅保留最新摘要，Memo 保留所有历史摘要 token，使任意历史信息都能通过注意力直接影响当前决策。在时间步 $t$，模型的输入上下文由 $n \times l_{sum}$ 个摘要 token（$n = \lfloor t/l_{seg} \rfloor$）加上当前片段的观测组成。这创造了一种类似残差的梯度快捷路径，每个摘要向量都直接参与后续损失函数的优化。

2. **注意力掩码与位置编码**：使用因果掩码，当前时间步可关注所有历史摘要 token和当前片段内的观测，但**不能关注**产生最近摘要之前的原始观测。这形成信息瓶颈——历史信息必须通过摘要 token 传递。位置编码分两段：摘要 token 的位置索引从 0 到 $n \times l_{sum} - 1$，当前片段观测从 $n \times l_{sum}$ 开始递增，保持相对位置感知。

3. **片段长度随机化**：训练时对片段长度在 $\pm 20\%$ 范围内均匀采样（固定 $l_{seg}=256$ 时范围为 [205, 307]），而数据收集和评估使用固定长度。这不仅防止对固定边界的过拟合，还形成课程学习效果——短片段是简单压缩任务，长片段是困难压缩任务。

4. **KV 缓存一致性维护**：在 on-policy RL 中，模型更新后 KV 缓存因权重变化而过时。借鉴 ReLIC 的做法，每次策略更新后刷新整个 KV 缓存，并同步重新计算所有摘要向量，确保缓存表示与当前策略一致。

### 训练策略

Memo 通过 RL 目标端到端训练，摘要生成作为学习的子任务融入整体优化。支持两种 RL 范式：
- **On-policy**：集成到 ReLIC（DD-PPO 的改进版），利用 rollout 中的频繁更新帮助 Transformer 策略学习
- **Off-policy**：集成到 AMAGO，使用共享 Transformer 骨干和统一 actor-critic 损失

## 实验关键数据

### 主实验

**ExtObjNav 任务（32k 步评估，10 种子）**

| 方法 | 成功率 (SR) | SPL | 上下文 token 数 | KV 缓存比 |
|------|----------|-----|--------------|---------|
| FCT (全上下文) | ~52% | ~22% | 4096 | 1× |
| Memo | **~60%** | **~24.5%** | ~512 | **1/8×** |
| RMT-32 | ~55% | ~22% | 固定 | - |
| RMT-64 | ~56% | ~22% | 固定 | - |
| no-IEA | ~35% | ~15% | - | - |
| AC (TBTT) | ~45% | ~18% | ~512 | 1/8× |

**Dark-Key-To-Door 任务（Off-policy AMAGO，3 种子）**

| 方法 | 平均回报 | 收敛速度 | 稳定性 |
|------|---------|---------|--------|
| FCT | ~95 | 35M 步 | 35-40M 步出现性能下降 |
| Memo | **~95** | **25M 步** | 收敛后稳定无下降 |
| RMT | ~90 | 35M 步 | 较稳定 |

### 消融实验

| 配置 | SR@10k | SR@32k | 说明 |
|------|--------|--------|------|
| Memo ($l_{sum}=32$) | **~60%** | **~55%** | 最佳压缩比 8× |
| Memo ($l_{sum}=16$) | ~55% | ~50% | 压缩比 16×，信息损失 |
| Memo ($l_{sum}=64$) | ~52% | ~42% | 压缩比 4×，位置编码外推差 |
| Memo 无片段随机化 | ~48% | ~40% | 训练和评估均显著变差 |
| Streaming Memo | ~60% | **~58%** | 截断后性能甚至略有提升 |
| Streaming FCT | ~52% | ~35% | 6k 步后性能急剧下降 |

### 关键发现

- **摘要压缩优于全上下文**：Memo 用 1/8 的 token 实现了比 FCT 高 7.5% 的成功率和 2.5% 的 SPL。这反直觉地说明信息瓶颈迫使模型学习更好的任务相关压缩。
- **累积摘要优于固定记忆**：Memo 比 RMT 快约 10M 步收敛（Dark-Key-To-Door），体现了摘要累积提供的残差梯度快捷路径的优势。
- **长程梯度传播至关重要**：AC 的截断梯度（TBTT）在长时间任务中性能显著更差，说明 RL 中的信用分配需要梯度穿过所有摘要步骤。
- **流式推理鲁棒性**：Streaming Memo 在超过 6k 步后仍维持甚至提升性能，而 Streaming FCT 出现急剧下降。

## 亮点与洞察

- 反直觉的核心发现：**信息瓶颈不是限制而是优势**——强迫模型通过摘要传递信息实际上提升了任务相关信息的提取。
- 摘要累积的设计非常精巧：它兼顾了 Transformer 的表达力（每个历史摘要都可直接被注意力访问）和循环模型的效率（不需保留原始观测），并通过残差式梯度路径解决了梯度消失问题。
- 片段长度随机化作为隐式课程学习的发现值得在其他场景复用。

## 局限与展望

- 未探索记忆整合（memory consolidation）机制，即逐步压缩旧摘要以进一步节省存储。
- 长上下文泛化仍受限：Memo 训练 4k 泛化到 ~10k，训练 16k 泛化到 ~24k，外推能力有限。
- 未研究语义泛化——导航到全新物体类别（而非固定环境中已见物体的新位置）。
- 摘要 token 数对性能非常敏感（$l_{sum}=64$ 远差于 $l_{sum}=32$），最优值可能因任务而异但缺乏自适应选择机制。

## 相关工作与启发

- 与 NLP 领域的 Autocompressors 的关键区别：AC 微调预训练模型且截断梯度传播，Memo 从零训练且保持完整梯度。这揭示了 NLP 和 RL 在上下文压缩需求上的根本差异。
- ReLIC 和 AMAGO 展示了 Transformer 在 RL 中的潜力，Memo 可看作它们的"记忆增强"正交扩展。
- 文中的流式推理方案无需修改（如 StreamingLLM），值得在其他长序列 RL 任务中测试。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将语言模型上下文压缩思路适配 RL，并发现 RL 需要完整梯度传播的关键差异
- 实验充分度: ⭐⭐⭐⭐ Grid-world 和 3D 导航两个层面验证，10 种子统计，消融完善
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，发现表述准确
- 价值: ⭐⭐⭐⭐ 为具身智能的长期记忆问题提供实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

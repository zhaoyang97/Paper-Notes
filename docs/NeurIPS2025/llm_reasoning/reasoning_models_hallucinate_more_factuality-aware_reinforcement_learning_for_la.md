# Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.24630](https://arxiv.org/abs/2505.24630)  
**代码**: [GitHub](https://github.com/nusnlp/FSPO) (有)  
**领域**: LLM推理  
**关键词**: 幻觉, 推理模型, 强化学习, 事实性验证, GRPO, 步级奖励

## 一句话总结
揭示了RL训练的推理模型（如DeepSeek-R1）比非推理模型产生更多幻觉，从理论上分析了三个根因（高方差梯度、熵约束、伪局部最优），并提出FSPO算法通过步级事实性验证调整token级advantage，在减少幻觉的同时保持甚至提升推理能力。

## 研究背景与动机
1. **领域现状**：以DeepSeek-R1、OpenAI o1为代表的推理模型通过RL（如GRPO）训练长链CoT推理，在数学、编程等复杂推理任务上取得突破性进展。
2. **现有痛点**：作者发现一个被忽视的严重问题——RL训练后的推理模型幻觉率显著上升。实证显示，R1-Distill-Qwen-7B在TruthfulQA上仅6.9%的truthful率（vs Qwen2.5-7B-Instruct的36.7%），在HaluEval-QA上仅11.6%（vs 48.0%）。推理模型的"自信推理"表象下隐藏着大量事实错误。
3. **核心矛盾**：现有RL训练仅基于最终答案正确性（binary outcome reward 0/1），完全忽略中间推理步骤的事实性。这种稀疏奖励信号导致三个理论问题：(1) 正确答案概率低时梯度方差极高→训练不稳定；(2) 需要高熵探索正确答案→增加幻觉概率；(3) 模型可能收敛到"自信但错误"的伪局部最优→零梯度无法逃逸。
4. **本文要解决什么？** 设计一个兼顾推理能力和事实性的RL训练算法，在提升数学推理性能的同时显著降低幻觉率。
5. **切入角度**：将步级事实性验证信号（NLI-based）融入GRPO的advantage计算，提供比纯outcome reward更密集的梯度信号。
6. **核心idea一句话**：用自动事实性验证器对每个推理句子打分，翻转"正确答案但含虚假推理"的token advantage，让模型学到"正确的推理过程"而非"碰巧正确的答案"。

## 方法详解

### 整体框架
FSPO在GRPO基础上增加步级事实性反馈。输入是问题 $x$ + 关联evidence $\mathcal{K}$（如Wikipedia片段），模型生成包含推理链 $\{z_1,...,z_N\}$ 和最终答案 $y$ 的输出。系统通过两个奖励信号训练：(1) 答案正确性奖励 $\mathcal{R}_{\text{answer}} \in \{0, 1\}$；(2) 步级事实性奖励 $\mathcal{R}_{\text{factuality}}(z_j) \in \{-1, 0, 1\}$（蕴含/中立/矛盾）。

### 关键设计

1. **步级事实性验证器**:
   - 做什么：对推理链中每个句子 $z_j$ 判断其与evidence $\mathcal{K}$ 的关系
   - 核心思路：使用HHEM-2.1（自然语言推理模型）自动判断每个句子是被evidence蕴含（+1）、中立（0）还是矛盾（-1），中立包括连接词、探索性语句如"Aha"、"Wait"等
   - 设计动机：比outcome-only reward密集得多的梯度信号，直接解决Theorem 4.1的高方差问题

2. **事实性感知的Advantage调整**:
   - 做什么：根据句子级事实性分数翻转或保持GRPO计算的token advantage
   - 核心思路：设 $A_i$ 为GRPO原始advantage，对每个token $o_{i,t} \in z_j$：当 $A_i > 0$ 但 $\mathcal{R}_{\text{factuality}}(z_j) = -1$ 时（正确答案但虚假推理），翻转为 $-A_i$；当 $A_i < 0$ 但 $\mathcal{R}_{\text{factuality}}(z_j) = 1$ 时（错误答案但正确推理步骤），翻转为 $-A_i$（鼓励）
   - 设计动机：解决"reward hacking"——模型可能通过错误推理碰巧得到正确答案，传统GRPO会奖励这些虚假推理token。FSPO确保只有事实正确的推理步骤被强化

3. **混合训练数据策略**:
   - 做什么：混合知识密集型QA数据（2K HotpotQA）和数学推理数据（8K SimpleRL）
   - 核心思路：QA数据提供事实性训练信号，数学数据保持推理能力。FSPO仅对QA部分计算事实性奖励，数学部分仅用answer reward
   - 设计动机：仅2K事实性数据即可显著降低幻觉，不损害数学推理

### 理论分析（三个定理）
- **Theorem 4.1**：binary reward下梯度方差 $\propto p(1-p)\|\nabla\log\pi\|^2$，当正确率 $p$ 小时方差极高→训练不稳定
- **Theorem 4.2**：为避免陷入零奖励需保持高熵探索 $H_\theta(x) \geq H_{\min}(\epsilon)$→增加幻觉概率
- **Theorem 4.3**：确定性输出错误答案的策略是驻点（梯度为零），binary reward无法逃逸

### 训练策略
- 基于verl框架，batch size 8，每prompt 8个rollout，最大长度2048
- 学习率4e-7，KL系数1e-3，clip ratio 0.2
- 1个epoch训练，混合HotpotQA(2K) + SimpleRL(8K)

## 实验关键数据

### 主实验

| 模型 | GSM8K | MATH500 | TruthfulQA↑ | HaluEval-QA↑ | HalluQA↑ |
|------|-------|---------|-------------|-------------|----------|
| Qwen2.5-7B-Base | 65.2 | 35.7 | 38.2 | 48.0 | 39.5 |
| R1-Distill-Qwen-7B | 84.3 | 92.8 | **6.9** | **11.6** | **3.1** |
| FSPO (Qwen-Base) | **89.5** | 75.5 | **58.4** | **83.0** | **52.0** |
| Llama3.1-8B-Inst | 77.5 | 33.1 | 26.4 | 36.7 | 12.2 |
| R1-Distill-Llama-8B | 82.1 | 89.1 | 8.8 | 14.6 | 4.6 |
| FSPO (Llama-Inst) | 86.2 | 68.3 | 41.1 | 67.1 | 42.0 |

关键对比：R1-Distill-Qwen-7B的幻觉率极高（TruthfulQA仅6.9%），FSPO将其从6.9%提升至58.4%，同时GSM8K还超过了蒸馏模型。

### 消融实验

| 配置 | MATH-500 | HaluEval-QA↑ | 说明 |
|------|----------|-------------|------|
| GRPO (answer only) | 74.2 | 62.0 | 仅答案正确性奖励 |
| GRPO w/ factuality reward | 74.8 | 72.0 | 加入事实性奖励但不调advantage |
| FSPO (full) | **75.5** | **83.0** | 完整方法：advantage翻转 |

### 关键发现
- 推理模型（R1-Distill系列）在所有幻觉基准上表现远差于非推理模型，验证了"推理模型幻觉更多"的核心发现
- 仅2K事实性QA数据即可显著降低幻觉，4K/8K反而过多→数学推理性能下降
- FSPO对GRPO和Reinforce++两种RL算法都有效，验证了通用性
- 事实性分数在训练过程中稳步上升，而response长度基本不变，说明FSPO提升的是质量而非长度

## 亮点与洞察
- **理论+实证的双重论证**：三个定理清晰解释了为什么binary reward的RL会导致幻觉，不是简单加正则化而是从根源分析问题
- **Advantage翻转机制**极其巧妙——当答案正确但推理含虚假句子时，翻转该句子token的advantage为负值，直接惩罚"碰巧正确但推理错误"的行为。这是对GRPO的最小改动但最大效果的修改
- **2K数据即有效**的发现对实际部署很有价值——不需要大规模标注事实性数据
- 揭示了RL训练的推理模型的一个fundamental trade-off：推理能力↑ 但事实性↓，这对整个reasoning LLM社区是重要警示

## 局限性 / 可改进方向
- 事实性验证依赖evidence（Wikipedia片段），对无外部知识库的场景（如纯数学推理）不直接适用
- HHEM-2.1验证器本身有误差，可能错判事实性→需要更强的验证器
- 仅在7B/8B规模验证，32B+规模的效果未知
- MATH-500上FSPO（75.5%）远低于R1-Distill-Qwen-7B（92.8%），说明在纯数学推理上FSPO还是有代价的
- 理论分析仅覆盖binary reward，对更复杂的reward shaping场景的分析可进一步扩展

## 相关工作与启发
- **vs DeepSeek-R1**：R1用纯outcome reward训练，FSPO揭示了其幻觉代价并提出步级修复方案
- **vs Self-CheckGPT等后处理方法**：那些在推理后检查幻觉，FSPO在训练时就惩罚幻觉推理，更根本
- **vs RLHF**：RLHF用人类反馈但通常也是序列级，FSPO做到句子级事实性反馈，粒度更细

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统揭示并理论分析RL推理模型的幻觉问题，advantage翻转设计新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准全面评估+消融+训练动态分析，但缺乏大模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论→实证→方法→实验的逻辑清晰，图表丰富直观
- 价值: ⭐⭐⭐⭐⭐ 对整个reasoning LLM社区敲响幻觉警钟，FSPO是实用且高效的解决方案

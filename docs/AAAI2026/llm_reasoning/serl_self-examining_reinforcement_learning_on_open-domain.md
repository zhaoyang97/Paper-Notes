# SERL: Self-Examining Reinforcement Learning on Open-Domain

**会议**: AAAI 2026  
**arXiv**: [2511.07922](https://arxiv.org/abs/2511.07922)  
**代码**: [GitHub](https://github.com/AlwaysOu/SERL)  
**领域**: LLM推理 / 自我改进  
**关键词**: 自我改进, 强化学习, 成对比较, Copeland方法, 无外部奖励

## 一句话总结
提出SERL自我改进框架，LLM同时作为Actor（生成者）和Judge（评估者），用Copeland成对比较方法从自身判断中推导奖励信号，无需外部奖励模型或人工标注，使Qwen3-8B在AlpacaEval 2.0上从52.37%提升到59.90%（+7.53%），接近Qwen3-32B水平。

## 研究背景与动机
1. **领域现状**：LLM自我改进是减少对外部标注依赖的重要方向，但自我评估的奖励信号质量是关键瓶颈。
2. **现有痛点**：(a) 自我评估容易产生偏好循环（A>B>C>A）；(b) 位置偏差和长度偏差影响判断质量；(c) 缺乏理论保证的奖励推导方法。
3. **核心矛盾**：自我评估主观且不一致，但外部评估又需要额外资源。
4. **本文要解决什么？** 从自身成对比较判断中推导出高质量的训练奖励。
5. **切入角度**：Copeland方法（投票理论）解决偏好循环+双奖励（actor+judge）联合优化。
6. **核心idea一句话**：Copeland成对比较推导奖励+actor/judge联合优化=无外部依赖的自我改进。

## 方法详解

训练流程：对每个输入，Actor采样N个候选回复→Judge对所有回复对进行成对比较→Copeland聚合为双奖励→GRPO在线更新。

### 关键设计
1. **Copeland奖励推导（Actor奖励 $\mathcal{R}_A$）**：
   - 对每个输入采样N个候选回复，对所有 $\binom{N}{2}$ 组合进行成对比较
   - 每组比较采样K个独立判断，通过Copeland方法聚合为胜率排名
   - $\mathcal{R}_A(G_n) = \sum_{i\neq j,k} \mathbf{1}(G_n = G^{Win}_{(i,j),k}) / (M \times K)$
   - 胜率直接反映回复在组内的相对质量排名，比逐条打分更鲁棒
2. **Judge一致性奖励（$\mathcal{R}_J$）**：
   - 衡量单个成对判断与全局Copeland排序的一致性
   - $\mathcal{R}_J(J_{(i,j),k}) = \text{sign}(\mathcal{R}_A(G^{Win}) - \mathcal{R}_A(G^{Lose}))$
   - 一致判断得+1、矛盾判断得-1，迫使Judge学习更连贯的评估标准
3. **位置偏差缓解机制（PBMM）**：
   - K次比较中一半以 $(q, G_i, G_j)$ 顺序呈现，另一半以 $(q, G_j, G_i)$ 呈现
   - 消除LLM-as-Judge中常见的位置偏好（倾向选择靠前或靠后的回复）
4. **长度控制模块（LCM）**：
   - 引入长度比例权重 $\beta = |G^{Lose}|/|G^{Win}|$，较短回复获胜时获更高奖励
   - 通过超参数 $\alpha=0.2$ 限制只比较长度相近的回复对
   - 防止模型学习"越长越好"的虚假策略

### 损失函数 / 训练策略
- 基于GRPO框架，去掉KL惩罚项（开放域训练中模型分布偏移大，KL约束过强限制探索）
- Actor和Judge的优势值均通过组内归一化计算：$\hat{A}^{Actor} = (\mathcal{R}_A - \text{mean}) / \text{std}$
- 联合优化目标 $\mathcal{J}_{SERL} = \mathcal{J}_{Actor} + \mathcal{J}_{Judge}$，每步同时更新生成和评估能力

## 实验关键数据

### 通用QA（AlpacaEval 2.0）

| 方法 | LC Win Rate | Win Rate | 平均长度 |
|------|------------|---------|----------|
| Online-DPO | 54.07% | 59.74% | 3429 |
| Self-Rewarding | 51.29% | 53.69% | 3074 |
| Meta-Rewarding | 54.73% | 55.93% | 3081 |
| RLSC | 52.11% | 51.81% | 2060 |
| **SERL(Ours)** | **59.90%** | **69.88%** | **3017** |

### 摘要与写作任务（对比胜率）

| 对比方法 | 摘要任务胜率 | 写作任务胜率 |
|---------|-------------|-------------|
| vs Online-DPO | 55.17% (+10.33%) | 50.50% (+1.00%) |
| vs Self-Rewarding | 59.50% (+19.00%) | 55.17% (+10.33%) |
| vs Meta-Rewarding | 59.17% (+18.33%) | 56.67% (+13.33%) |
| vs RLSC | 86.17% (+72.33%) | - |

### 关键发现
- 8B模型+SERL达到59.90% LC Win Rate，接近Qwen3-32B（~60%）——自我改进弥补了4×规模差距
- Copeland方法有效解决偏好循环问题，比point-wise自评（Self-Rewarding 51.29%）鲁棒性显著更强
- Actor+Judge联合优化使两种能力同步提升，形成良性正反馈循环
- 在摘要任务上优势最大：对Self-Rewarding胜率高达59.50%（+19%），对RLSC甚至达86.17%
- SERL输出长度（3017 tokens）短于Online-DPO（3429），说明质量提升不依赖冗长输出
- 仅需数十步训练即可大幅提升——训练效率极高，对资源有限的团队非常友好
- 去掉KL惩罚项后训练依然稳定，说明KL约束在开放域任务中可能非必需

## 亮点与洞察
- **无外部依赖的真正自我改进**：不需要奖励模型、不需要人工标注、不需要更强的LLM做评估——完全自我驱动的闭环训练。这解决了RLHF/RLAIF的核心瓶颈（外部依赖）
- **Copeland方法的引入**创造性地解决了LLM自评中的偏好循环问题：投票理论中的Condorcet方法天然具有防操纵保证，将其引入LLM对齐是跨学科的巧妙迁移
- **Actor+Judge联合优化**形成正反馈循环：更好的Judge产生更准确的奖励信号→训练出更好的Actor→更好的Actor生成更高质量的回复→为Judge提供更有区分度的比较样本→Judge进一步提升
- **位置偏差缓解机制(PBMM)**和**长度控制模块(LCM)**是工程上的重要细节：前者通过交换回复位置消除LLM-as-Judge的位置偏差，后者通过长度比例加权 $\beta = |G^{Lose}|/|G^{Win}|$ 防止模型偏向更长回复

## 局限性 / 可改进方向
- 自评质量仍受限于模型本身能力——如果模型的评估能力有hard ceiling，自我改进也会饱和
- Copeland比较需要 $\binom{N}{2} \times K$ 次成对评估，N和K较大时计算开销显著
- 去掉了GRPO中的KL惩罚项——长期训练是否会导致模型分布过度偏移需要监控
- 仅在Qwen3-8B上验证，对其他架构和更大模型的适用性未知
- 能否持续多轮自我改进（自我改进的上限在哪里）需要更长周期的实验验证

## 相关工作与启发
- **vs Self-Rewarding(Yuan等)**：Self-Rewarding让Actor做Judge但用逐条打分（point-wise），SERL用成对比较（pair-wise）+ Copeland聚合更鲁棒。且Self-Rewarding不优化Judge，SERL通过一致性奖励联合优化
- **vs Meta-Rewarding(Wu等)**：Meta-Rewarding联合优化Actor和Judge但用离策略学习（off-policy），SERL用在策略学习（on-policy），理论上更稳定
- **vs RLVR(GRPO/DAPO)**：RLVR需要可验证答案，仅适用于数学/代码等闭合任务；SERL通过自比较产生奖励，适用于开放域
- 启发：投票理论工具（Copeland、Borda等）在LLM对齐中有更多应用潜力
- 启发：双奖励联合优化的范式可扩展到其他自评场景（如代码自调试中同时优化生成器和测试器）

## 评分
- 新颖性: ⭐⭐⭐⭐ Copeland+双奖励+Actor-Judge联合优化的创新组合
- 实验充分度: ⭐⭐⭐⭐ AlpacaEval主评+摘要/写作/QA多任务验证
- 写作质量: ⭐⭐⭐⭐ 方法清晰，公式严谨
- 价值: ⭐⭐⭐⭐⭐ 无外部依赖的自我改进+接近2倍大模型性能，有重大实用意义
- 综合: 开放域LLM后训练的重要方向性工作，Copeland奖励推导是核心创新点
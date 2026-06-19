---
title: >-
  [论文解读] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning
description: >-
  [AAAI 2026][LLM推理][大语言模型] 提出一种基于熵引导的自适应 LLM 推理框架，结合动态上下文检索和自适应链式思维（CoT）推理，在井字棋博弈任务中将 LLM 的平均对局结果从 -11.6% 提升至 +9.5%，同时保持较低的 LLM 查询次数。 领域现状 大语言模型（LLMs）在单步推理、语言理解和少样本…
tags:
  - "AAAI 2026"
  - "LLM推理"
  - "大语言模型"
  - "博弈论"
  - "熵引导推理"
  - "链式思维"
  - "检索增强生成"
---

# LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning

**会议**: AAAI 2026  
**arXiv**: [2601.10775](https://arxiv.org/abs/2601.10775)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 大语言模型, 博弈论, 熵引导推理, 链式思维, 检索增强生成

## 一句话总结
提出一种基于熵引导的自适应 LLM 推理框架，结合动态上下文检索和自适应链式思维（CoT）推理，在井字棋博弈任务中将 LLM 的平均对局结果从 -11.6% 提升至 +9.5%，同时保持较低的 LLM 查询次数。

## 研究背景与动机

### 领域现状
大语言模型（LLMs）在单步推理、语言理解和少样本泛化中表现出色。然而在结构化的序列决策问题中，LLM 面临显著挑战：每步决策都影响后续所有状态和结果，需要局部一致性和长期规划的结合。

### 现有痛点

**CoT 推理效果不稳定**：链式思维提示虽能改善复杂推理，但性能在不同任务和领域间差异很大，且评估指标往往不够精细

**计算资源浪费**：Tree-of-Thoughts 等多路径推理方法虽提升性能，但在模型已经自信的情况下仍进行大量计算

**缺少理论最优参照**：大多数推理任务缺乏客观最优解，难以量化评估 CoT 推理质量

**上下文检索静态**：传统 RAG 固定检索数量，未根据模型不确定性动态调整

### 核心矛盾
如何在计算效率和推理质量之间取得平衡——仅在模型不确定时才投入额外计算资源进行深度推理。

### 切入角度
利用 LLM 生成 token 时的熵值作为不确定性代理，动态调控两个维度：（1）检索示例的数量——高熵时增加检索量；（2）CoT 推理路径的数量——高熵时扩展多路径探索。选择井字棋作为受控测试环境，因为 minimax 算法提供了所有状态的已知最优解，可以精确量化推理质量。

## 方法详解

### 整体框架
游戏循环：当前棋盘状态 → autoencoder 编码为向量 → 余弦相似度检索最近邻棋局 → 构造结构化 prompt（棋盘+玩家+检索示例）→ LLM 生成推理和落子 → 解析输出 → 更新棋盘 → 重复直到终局。

### 关键设计

1. **棋盘表示与对比学习编码器**:

    - 棋盘表示为 $B \in \{0,1,2\}^{3\times 3}$，展平为 9 维向量
    - 使用 autoencoder 编码到低维潜在空间 $z = f_\theta(x) \in \mathbb{R}^d$
    - 训练目标：重建损失 + 对比学习损失
    - 对比损失使相同最优动作的棋局靠近，不同最优动作的远离：$\mathcal{L}_{con}$ 使用 margin $\tau$ 控制最小距离
    - 最终损失：$\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_{con}$
    - 向量数据库存储约 20% 的可能棋局状态及对应 minimax 最优动作
    - 设计动机：确保检索到策略相似的历史棋局，而非仅外观相似

2. **熵引导上下文检索**:

    - 检索数量 k 根据模型预测熵动态调整：$k = \min(k_{max}, \lceil k_0 + \alpha \cdot H_q \rceil)$
    - 低熵（高置信）→ 少量检索，保留 context 空间用于推理
    - 高熵（高不确定性）→ 增加检索量，提供更多参考信息
    - 总 context 长度受 token 预算 $L_{max}$ 约束
    - 设计动机：避免在简单决策上浪费 context 窗口

3. **自适应链式思维推理**:

    - 定义四种推理模式，复杂度递增：
        - **Direct output**：直接输出动作，无模拟
        - **Multi-CoT**：生成 n 条独立推理路径，投票选择
        - **Tree-based CoT**：树形展开，每步生成 n 个候选+对手所有回应
        - **Entropy-guided CoT**：仅在高熵时展开多路径
    - Token 级熵计算：$H_{t,k}^{token} = -\sum_{i=1}^{|V|} p_{t,k}^{(i)} \log p_{t,k}^{(i)}$
    - 步级熵：$H_t^{step} = \frac{1}{L_t} \sum_{k=1}^{L_t} H_{t,k}^{token}$
    - 有序阈值机制：$H_t^{step} \in [H_j, H_{j+1}) \Rightarrow n_t = n_j$
    - 高熵 → 更多分支；低熵 → 更少分支
    - 每步仅保留 top-k 分支以控制计算成本
    - 设计动机：把计算资源集中在模型真正需要的地方

4. **对手建模**:

    - 对手使用 minimax 表排序所有合法动作
    - 引入技能水平参数 $\alpha \in [0,1]$（实验中 $\alpha=0.95$）
    - 选择概率在技能水平对应的动作处峰值，向两侧线性衰减
    - 设计动机：使用次优对手更能反映实际场景

### 训练策略
- 使用 LLaMA-7B 作为基线模型
- 初步实验使用 Gemma 3 270M
- 每种配置运行 100 局游戏
- LLM 超参：temperature=0.1, 禁用 top-p/top-k 采样, beam search=2
- 每步最多生成 10 个 token，固定随机种子 42

## 实验关键数据

### 主实验

| 推理方式 | 无上下文 S(%) | 查询数 | 固定上下文 S(%) | 查询数 | 熵引导上下文 S(%) | 查询数 |
|---------|-------------|--------|---------------|--------|-----------------|--------|
| No CoT | -11.6 | 3 | -5.2 | 4 | -2.8 | 4 |
| Single CoT | -8.2 | 13 | -2.6 | 13 | -0.1 | 15 |
| Multi CoT | -7.5 | 24 | -1.2 | 26 | +4.8 | 28 |
| Tree-based CoT | -2.7 | 165 | +4.5 | 178 | **+9.8** | 188 |
| Entropy-guided CoT | -4.1 | 48 | +3.8 | 56 | **+9.5** | **48** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无上下文 + 无 CoT | S=-11.6% | 基线最差 |
| 固定上下文 + 无 CoT | S=-5.2% | 上下文检索有效 |
| 熵引导上下文 + 无 CoT | S=-2.8% | 动态检索进一步改善 |
| 熵引导 CoT + 熵引导上下文 | S=+9.5% (48 queries) | 效率最优 |
| Tree-based CoT + 熵引导上下文 | S=+9.8% (188 queries) | 性能最优但开销 4 倍 |

### 关键发现

1. **上下文检索与 CoT 互补**：两者分别从知识注入和推理深度两个维度提升性能，组合效果最佳
2. **熵-最优性负相关**：Spearman 相关系数 ρ=-0.471（p<10⁻³），Kendall τ=-0.346（p<10⁻³），证实高熵 token 对应次优动作
3. **计算效率**：熵引导 CoT 仅用约 1/4 的查询量（48 vs 188）即达到与 Tree-based CoT 相当的性能（+9.5% vs +9.8%）
4. **高熵的两种来源**：真正的不确定性（真阳性）和多个同等最优动作（假阳性）。后者主要出现在开局，通过不在首步使用熵分支来缓解
5. **零样本性质**：LLM 未经过任何任务特定微调，推理完全来自上下文条件化

## 亮点与洞察
1. 用熵作为不确定性代理来动态分配计算资源，思路优雅且通用
2. 选择井字棋作为测试环境非常明智——有已知最优解，可以精确评估每步决策质量
3. 检索编码器的对比学习设计巧妙，按最优策略而非棋盘外观组织潜在空间
4. 实验设计系统完整：5 种推理策略 × 3 种上下文 = 15 种配置，交叉对比清晰
5. 统计检验严格（Spearman、Kendall 相关性分析）

## 局限与展望
1. 仅在井字棋（状态空间极小）上验证，更复杂博弈（Connect Four、围棋）的扩展性未知
2. LLaMA-7B 推理能力有限，更强的模型可能内在就能处理这种推理
3. token 级熵假设语言不确定性等于决策不确定性，在模糊场景中可能不成立
4. 检索数据库来自完整博弈树，对不完全信息的领域不一定适用
5. 未分析推理链的逻辑质量，仅评估最终决策的准确性
6. 尝试将框架扩展到不确定动态和部分可观察的领域

## 相关工作与启发
- **Tree-of-Thoughts (Yao 2023)**：分支搜索框架 → 本文在此基础上增加了自适应控制
- **RAG (Lewis 2021)**：检索增强生成 → 本文动态化检索量
- **AlphaZero**：策略网络+搜索 → 本文用 LLM + 检索+CoT 替代
- **GridPuzzle / EIC**：推理链分析 → 本文利用博弈论最优解做更精确评估

## 评分
- 新颖性: ⭐⭐⭐⭐ (熵引导自适应推理的组合思路新颖)
- 实验充分度: ⭐⭐⭐ (井字棋场景过于简单，缺少更复杂环境验证)
- 写作质量: ⭐⭐⭐⭐ (形式化清晰，实验设计系统)
- 价值: ⭐⭐⭐ (核心思想好但应用范围有限，扩展性有待证明)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](../../ICML2026/llm_reasoning/many-shot_cot-icl_making_in-context_learning_truly_learn.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[AAAI 2026\] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension](relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)

</div>

<!-- RELATED:END -->

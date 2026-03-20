# Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.04723](https://arxiv.org/abs/2506.04723)  
**代码**: [https://sparkle-reasoning.github.io/](https://sparkle-reasoning.github.io/)  
**领域**: LLM推理 / 强化学习分析  
**关键词**: SPARKLE, GRPO, plan following, knowledge integration, subproblem decomposition, multi-stage RL  

## 一句话总结
提出 SPARKLE 三轴分析框架（计划执行、知识整合、子问题分解）细粒度剖析 RL 如何改变 LLM 推理行为，发现 RL 主要增强了知识整合能力和计划灵活性而非计划执行能力，并提出 SparkleRL-PSS 多阶段 RL 训练 pipeline 通过 partial step scaffolding 有效利用难题数据。

## 研究背景与动机

1. **领域现状**：RL（特别是 GRPO）已成为提升 LLM 推理能力的主流范式，DeepSeek-R1、OpenAI o1 等模型在 AIME、MATH 等 benchmark 上取得巨大进步。

2. **现有痛点**：几乎所有工作只追踪准确率提升，缺乏对 RL 究竟增强了什么能力的细粒度理解。RL 到底是提升了模型的规划能力、执行能力、知识调用能力，还是问题分解能力？不清楚就无法有针对性地改进 RL pipeline。

3. **核心矛盾**：难题通常不产生正 reward 信号（模型 20 次尝试都做不对），因此常被过滤掉。但丢弃难题 = 浪费宝贵训练信号。如何有效利用这些难题？

4. **本文要解决什么**：(1) 建立超越准确率的细粒度分析框架，揭示 RL 对推理各维度的具体影响；(2) 设计利用难题数据的多阶段 RL 训练方案。

5. **切入角度**：从认知科学的人类问题解决理论（Newell & Simon, 1972）出发，将推理分解为 planning、knowledge、decomposition 三个核心维度，分别设计可控实验。

6. **核心idea一句话**：通过给模型提供/不提供计划、知识、子问题分解等辅助信息的对比实验，揭示 RL 增强的具体推理维度，并据此设计 partial step scaffolding 训练策略。

## 方法详解

### 整体框架
SPARKLE 包含两部分：(1) 三轴分析框架——通过构建带有 planning skeleton、knowledge annotations、subproblem chains 标注的增强数据集，对 RL 前后的模型进行对比分析；(2) SparkleRL-PSS——两阶段 GRPO 训练 pipeline，第二阶段通过 partial step scaffolding 复用难题。

### 关键设计

1. **三轴分析框架**：
   - **Axis 1 (Planning & Execution)**：为每道题生成 planning skeleton（如"步骤1：分析模运算性质；步骤2：检测周期模式..."），对比模型有无 plan 时的表现差异，分离规划能力与执行能力
   - **Axis 2 (Knowledge Integration)**：提取每道题需要的事实、定理、引理（如费马小定理、中国余数定理），对比提供/不提供知识时的表现差异，分离知识检索与推理能力
   - **Axis 3 (Subproblem Decomposition)**：将每道题分解为一链子问题（Q1→Q2→Q3...），逐步提供已解决的子问题答案，检测推理在哪一步断裂

2. **SPARKLE Benchmark 构建**：
   - 做什么：增强 AIME24、AMC23、MATH500、GSM8K、OlympiadBench 共 2564 道题
   - 核心思路：用 GPT-4.1 + Web Agent 为每道题生成 planning skeleton、知识标注和子问题链，再由第二个 GPT-4.1 验证，最后由研究生数学专家人工审核
   - 标注内容：每题附带 AoPS 难度等级（1-10）、数学领域（9类）

3. **SparkleRL-PSS 多阶段训练**：
   - **Stage 1**：标准 GRPO 训练，用 DeepScaleR 的 40K 数学题训练 Qwen-2.5-Math-7B
   - **Stage 2**：从 Stage 1 模型中筛选 20 次尝试均失败的 6.5K 难题（经验证后 5.7K），将每题的参考解分为 4 个语义块，构造 0~4 个 hint 级别的输入变体。模型需要在已有部分解的基础上继续推理。KL 系数从 0.001 提升到 0.01 防止偏离过多
   - 设计动机：不用生成新数据，而是通过 partial scaffolding 让模型即使在难题上也能获得正 reward 信号

### 损失函数 / 训练策略
- GRPO 标准目标函数 + rule-based reward：正确答案+格式正确给 2 分，正确答案+格式不正确给 1 分，错误给 -1 分
- Stage 1: lr=1e-6, KL=0.001
- Stage 2: lr=1e-6, KL=0.01, temperature=0.6, 每题 32 个采样
- 8×H200 + 15×A100-40G + 9×A100-SXM4-40G

## 实验关键数据

### 主实验

| 模型 | AIME24 | AMC23 | MATH500 | GSM8K | OlympiadBench | 平均 |
|---|---|---|---|---|---|---|
| Qwen-2.5-Math-7B (Base) | 16.67 | 42.50 | 44.03 | 42.53 | 28.65 | 35.23 |
| SparkleRL-Stage 1 | 46.67 | 67.50 | 80.00 | 91.77 | 39.11 | 65.01 |
| SparkleRL-Stage 2-hard | 41.67 | 65.94 | 80.50 | 92.45 | 37.39 | 63.59 |
| SparkleRL-Stage 2-mix | 40.00 | 63.44 | 80.78 | 92.52 | 38.85 | 63.12 |
| **SparkleRL-Stage 2-pss** | **50.42** | **71.25** | **81.00** | **92.38** | **40.11** | **67.03** |

Stage 2-pss 比 Stage 1 平均提升 2.02%，AIME24 达到 50.42%（可比 32B 模型水平）。SFT on hard problems 则大幅退化（AIME24 从 46.67→15.00）。

### 消融实验（三轴分析核心发现）

| 分析维度 | Base 模型 | RL 模型 | 关键差异 |
|---|---|---|---|
| +Plan | 4/5 benchmark 性能下降，平均 -5.7% | 稳定或微升（AIME24 除外 -2.5%） | RL 模型更灵活，human plan 反而可能误导 |
| +Knowledge | 平均 -5.4% | 平均 +4.3% | Base 无法整合外部知识，RL 显著增强知识利用 |
| Subproblem (SSR) | AIME24: 3.3% SSR vs 16.7% full acc | AIME24: 17.5% SSR vs 50.4% full acc | 所有模型子问题逐步解决能力远弱于整体解题 |

### 关键发现
- **RL 不主要增强计划执行**：给 RL 模型人工编写的正确 plan 反而可能降低性能（AIME24 从 50.4%→47.9%）。RL 模型更擅长自主生成内部策略，外部 plan 可能与模型学到的启发式冲突
- **RL 显著增强知识整合**：Base 模型给了知识反而变差（-5.4%），因为它不会整合；RL 模型则显著受益（+4.3%）。难度越高收益越大（level 8 时知识增益 +42.5%）
- **子问题分解仍是瓶颈**：即使把难题拆成小问题逐步给答案，模型仍然在某些步骤失败。说明 RL 训练出的"快速抄近路"策略与逐步严谨推理不一致
- **难题可以有效利用**：Partial step scaffolding 让模型在难题上也能获得 reward 信号，比纯 hard-only 或 mix 训练都更有效
- **SFT 不能替代 RL Stage 2**：在 noisy trace 上做 SFT 会大幅退化，因为 SFT 是记忆而 RL 是泛化

## 亮点与洞察
- **"知识 > 计划"的实证**：对 RL 模型而言，提供外部知识的收益远大于提供 plan。这暗示 RL 模型的推理瓶颈更多在"知不知道"而非"会不会做"，对 RAG + reasoning 的结合具有指导意义
- **Human plan 可能有害**的反直觉发现非常有价值：它说明 RL 模型发展出了自己独特的内部推理策略。High-level plan 有帮助但 step-by-step plan 反而有害，这对 prompt engineering 有直接启发
- **Partial Step Scaffolding 无需额外数据生成**：只用现有参考解切分成片段，就能有效引导模型在难题上探索。这是一个低成本高收益的 curriculum learning 设计

## 局限性 / 可改进方向
- 仅在数学推理上验证，迁移到代码推理、逻辑推理等领域需要适配分析框架
- SPARKLE 数据集构建依赖 GPT-4.1 + 人工审核，扩展性受限
- 所有发现都是经验性的，缺乏理论解释为什么 RL 会增强知识整合而非计划执行
- Stage 2-pss 的 4 块切分方式是固定的，自适应切分策略可能更优

## 相关工作与启发
- **vs Yue et al. (2025) "RL不创造新能力"**：Yue 等人认为 RL 主要重新加权已有推理路径。SPARKLE 的分析更细粒度，发现 RL 确实增强了知识整合机制——这不仅仅是路径重加权
- **vs ARM / Controlling Thinking Speed**：这些工作关注推理效率的宏观控制，SPARKLE 则深入到推理能力的微观解剖。两者互补——先理解 RL 增强了什么，再决定如何调控
- **vs DeepScaleR**：DeepScaleR 用缩放 RL 训练提升小模型性能，SparkleRL-PSS 证明在相同数据上通过 curriculum 设计（而非简单数据过滤）可以获得进一步提升

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 三轴分析框架是首个系统性剖析 RL 对推理各维度影响的工作，多个反直觉发现
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 benchmark，7B/32B 两个尺度，SFT/RL/multi-stage RL 对比，统计显著性检验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现有洞察力，但 appendix 过长
- 价值: ⭐⭐⭐⭐⭐ 对理解 RL+reasoning 的本质机制有重要贡献，partial step scaffolding 实用性强

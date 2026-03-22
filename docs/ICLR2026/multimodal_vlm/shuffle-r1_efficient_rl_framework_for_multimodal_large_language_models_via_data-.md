# Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle

**会议**: ICLR 2026  
**arXiv**: [2508.05612](https://arxiv.org/abs/2508.05612)  
**代码**: https://xenozlh.github.io/Shuffle-R1  
**领域**: 多模态VLM  
**关键词**: 强化学习, 多模态推理, 数据中心优化, 轨迹采样, GRPO

## 一句话总结
提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

## 研究背景与动机

1. **领域现状**: 强化学习(RL)已成为提升 LLM/MLLM 推理能力的主流后训练范式。DeepSeek-R1 等工作使用可验证结果奖励信号，在数学推理和代码生成上取得显著进步。RL 也被扩展到多模态领域，用于视觉推理、目标检测、视频理解等任务。

2. **现有痛点**: 当前 RL 训练流程存在两个被忽视的关键效率问题：
   - **Advantage Collapsing**: 一个 batch 中大多数 rollout 的 advantage 值集中在零附近，导致梯度信号极弱，有价值的轨迹被大量无信息轨迹淹没
   - **Rollout Silencing**: 随着训练推进，贡献非零梯度的 rollout 比例持续下降（简单问题已收敛、困难问题始终无法答对），造成计算浪费

3. **核心矛盾**: 静态采样范式对所有轨迹一视同仁，无法区分"哪些数据值得更新"；增大 rollout 数量虽能部分缓解，但计算开销线性增长，未触及根因。

4. **本文要解决什么？** 在不显著增加计算开销的前提下，动态筛选有价值的轨迹并优化批次组成，提高 RL 训练的梯度信号质量与计算利用率。

5. **切入角度**: 从数据侧切入，将 RL 训练从"如何更新"转向"用什么数据更新"，设计自适应的轨迹选择和批次重组机制。

6. **核心idea一句话**: 通过配对采样挑选高对比度轨迹、按优势值重洗批次放大关键信号，实现动态数据优先级调度。

## 方法详解

### 整体框架
Shuffle-R1 在标准 GRPO 的基础上，在 advantage 计算之后插入两个模块：(1) Pairwise Trajectory Sampling (PTS) 从扩展的 rollout 池中选择高价值轨迹对；(2) Advantage-based Batch Shuffle (ABS) 对选出的轨迹按重要性重组训练批次。两者联合实现"动态数据优先级调度"。

### 关键设计

1. **Pairwise Trajectory Sampling (PTS)**:
   - **做什么**: 从 2N 个 rollout 中构建 N 个正负对比轨迹对，仅保留 top-k 高对比度对
   - **核心思路**: 按 advantage 降序排序后，采用 max-min 配对策略——最大 advantage 与最小配对、次大与次小配对。每对的 advantage 差距代表对比强度。通过采样比 α 保留前 M=αN 对高对比度轨迹对
   - **设计动机**: 高对比度的正负对能提供更强的梯度信号。虽然 2N rollout 的生成量翻倍，但只对 top-k 对进行梯度计算，计算开销不变。缓解 Advantage Collapsing

2. **Advantage-based Batch Shuffle (ABS)**:
   - **做什么**: 对 PTS 输出的有效轨迹对，按优势值分配采样概率，进行 S 轮子采样重组训练批次
   - **核心思路**: 为每对轨迹计算重要性权重 W(p) = |Â₁| + |Â₂|，归一化为采样分布 Φ。每轮子采样抽取 T 对（不重复），S 轮组合为重洗批次 B'，保持 |B'| = |B|
   - **设计动机**: 高 advantage 的轨迹获得更多更新机会，低价值的轨迹被自然降权，缓解 Rollout Silencing。本质上是一种软优先级排序

### 损失函数 / 训练策略
- 基础目标函数为 PPO-clip 风格的策略梯度（与 GRPO 一致），advantage 使用组内标准化
- PTS 采样比 α=0.5（保留一半对），ABS 子采样容量 T=256、轮数 S=8
- 每个 query 生成 2N=16 个 rollout，构建 8 对后保留 4 对
- 学习率 1e-6，rollout 温度 1.0，冻结视觉编码器

## 实验关键数据

### 主实验

| 模型 | 方法 | Geo3K | Math Avg. | HallBench | ChartQA |
|------|------|-------|-----------|-----------|---------|
| Qwen2.5-VL-3B | Baseline | 25.79 | 41.71 | 59.83 | 73.08 |
| Qwen2.5-VL-3B | +GRPO | 42.64 | 46.74 | 63.09 | 76.20 |
| Qwen2.5-VL-3B | +DAPO | 45.09 | 48.08 | 63.24 | 76.70 |
| Qwen2.5-VL-3B | **+Ours** | **47.88(+22.09)** | **48.70(+6.99)** | 63.19 | **77.04** |
| Qwen2.5-VL-7B | Baseline | 38.12 | 49.82 | 65.19 | 79.84 |
| Qwen2.5-VL-7B | +GRPO | 52.60 | 53.13 | 68.56 | 80.84 |
| Qwen2.5-VL-7B | **+Ours** | **55.89(+17.77)** | **54.63(+4.81)** | **69.51** | **81.64** |

在跨领域基准上(30k 训练数据)，7B 模型在 MathVerse 上达到 52.2%，超越 GPT-4o (50.8%)。

### 消融实验

| 组件 | Geo3K (3B) | Math Avg. (3B) |
|------|------------|----------------|
| GRPO baseline | 42.64 | 46.74 |
| +PTS only | 46.52 | 47.89 |
| +ABS only | 44.18 | 47.35 |
| +PTS+ABS (Full) | **47.88** | **48.70** |

### 关键发现
- PTS 贡献最大，单独即可带来 ~4% Geo3K 提升；ABS 提供额外 ~1.5% 增益
- 在仅用 50% 训练步数时即可匹配 GRPO 的完整训练效果，训练效率提升 2×
- 在 3B/7B 两个规模上效果一致，说明方法具有跨规模泛化能力
- Geo3K(2.1k 样本) 小数据场景下优势尤为明显，说明方法对数据量少的情况更有价值

## 亮点与洞察
- 问题定义精准：首次系统性提出并量化 Advantage Collapsing 和 Rollout Silencing 两个 RL 训练效率瓶颈
- 方法设计简洁有效：PTS 和 ABS 都是轻量级模块，实现简单（排序+配对+加权采样），即插即用
- 实验覆盖全面：in-domain/out-of-domain、小数据/大数据、3B/7B 多规模验证
- 计算开销可控：虽然 rollout 数翻倍，但梯度计算量不变（只对筛选出的轨迹计算）

## 局限性 / 可改进方向
- PTS 的 max-min 配对策略是启发式的，是否存在更优的配对方式（如基于语义相似度）值得探索
- ABS 的重采样引入了重复使用同一轨迹的风险，可能导致过拟合；需要更系统的分析
- 目前只验证了数学推理任务，向通用 VQA、视觉对话等更多任务的泛化有待验证
- α=0.5 的采样比是固定的，自适应调节可能进一步提升效果

## 相关工作与启发
- 与 NoisyRollout（增加 rollout 多样性）、VL-Rethinker（反思 token）互补：前者关注数据多样性，本文关注数据质量筛选
- 与课程学习(curriculum learning)理念相通：都是让模型更多关注有价值的训练样本
- 为其他 RL 训练框架（如 DPO、RLHF）的数据侧优化提供思路

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖（Advantage Collapsing/Rollout Silencing），方法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 多规模、多数据量、多基准验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，图示直观，论证逻辑通顺
- 价值: ⭐⭐⭐⭐ 数据中心的 RL 优化视角对社区有较大启发，方法即插即用实用性强

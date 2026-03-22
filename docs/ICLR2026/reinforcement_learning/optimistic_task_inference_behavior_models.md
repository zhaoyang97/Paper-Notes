# Optimistic Task Inference for Behavior Foundation Models

**会议**: ICLR 2026 Oral  
**arXiv**: [2510.20264](https://arxiv.org/abs/2510.20264)  
**代码**: 有  
**领域**: 强化学习 / 基础模型  
**关键词**: behavior foundation models, task inference, zero-shot RL, successor features, UCB  

## 一句话总结
提出 OpTI-BFM——一种乐观决策准则，对奖励函数的不确定性建模并引导 Behavior Foundation Model 在测试时通过环境交互进行任务推断，与线性 bandit 的上置信界（UCB）算法建立联系并提供正式的 regret bound，使 BFM 仅需几个 episode 即可识别和优化未见奖励函数，几乎无额外计算开销。

## 研究背景与动机
1. **领域现状**：BFM 通过预训练 successor features 实现 zero-shot RL，但在测试时需要完整的奖励函数或大量标注数据做任务推断。
2. **现有痛点**：完整奖励函数在实际场景中难以获得，现有方法依赖大量推断数据集。
3. **核心idea一句话**：不给奖励函数、不给标注数据，仅通过与环境交互几个 episode 就推断出任务并优化策略——用乐观探索（UCB）解决测试时的任务推断问题。

## 方法详解

### 关键设计
1. **奖励不确定性建模**：对可能的奖励函数维护后验/置信集
2. **UCB 式乐观探索**：选择在置信上界下最有利的奖励假设对应的策略执行
3. **Successor features 复用**：利用预训练的 successor features 零开销切换策略
4. **Regret bound**：通过与线性 bandit 的形式化联系提供理论保证

## 实验关键数据
- 在标准 zero-shot RL 基准上评估
- 仅需几个 episode 即可识别未见奖励函数
- 计算开销极小（复用预训练 successor features）

## 亮点与洞察
- **消除了 BFM 的核心假设**：不再需要完整奖励函数或标注推断数据
- **理论优雅**：与经典 bandit 算法的联系使分析和保证成为可能

## 局限性 / 可改进方向
- 依赖 successor features 的线性结构假设
- 在高维连续动作空间的可扩展性需进一步验证

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ BFM + UCB 的结合思路新颖且有理论深度
- 实验充分度: ⭐⭐⭐⭐ 标准基准验证充分
- 写作质量: ⭐⭐⭐⭐ 理论清晰
- 价值: ⭐⭐⭐⭐ 推动 zero-shot RL 的实用化

# Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2504.15275](https://arxiv.org/abs/2504.15275)  
**代码**: [github.com/CJReinforce/PURE](https://github.com/CJReinforce/PURE)  
**领域**: LLM推理  
**关键词**: 过程奖励模型, 信用分配, reward hacking, min-form, 强化学习  

## 一句话总结
PURE 发现 PRM 导致 reward hacking 的根本原因是 RL 中标准的 sum-form 信用分配（$V(s) = \sum \gamma^t r_t$），并提出 min-form 替代方案（$V(s) = \min_{t' \geq t} r_{t'}$），通过将价值函数限制为未来奖励的最小值而非累积和，显著缓解 reward hacking——仅用 30% 训练步数就达到与规则奖励方法相当的推理性能。

## 研究背景与动机

1. **领域现状**：PRM 在测试时缩放上已证明有效（如 Best-of-N），但将 PRM 用于 RL 微调时频繁出现 reward hacking——模型学会利用 PRM 的高分步骤，而非真正提升推理质量。
2. **现有痛点**：标准 RL 信用分配定义 $V(s_t) = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$。当 PRM 对某些步骤给出不准确的高分时，累积求和会放大这些错误，使模型倾向于生成高奖励步骤的序列（无论推理是否正确）→ 训练崩溃。
3. **核心矛盾**：sum-form 累积使得单个高奖励步骤就能拉高整条轨迹的价值，而 PRM 不可能对每一步都给出完美奖励 → reward hacking 几乎不可避免。
4. **本文要解决什么？** 如何设计信用分配使 PRM 能安全地用于 RL 微调？
5. **切入角度**：将价值函数定义为未来奖励的**最小值**而非**累积和**——这意味着模型必须确保每一步都不太差才能获得高价值，而不是只做好几步就够。
6. **核心idea一句话**：用 $V(s_t) = \min_{t' \geq t} r_{t'}$ 替代 $V(s_t) = \sum \gamma^{t'-t} r_{t'}$，消除单步高奖励对整体价值的过度拉升。

## 方法详解

### 整体框架
PURE (Process sUpervised Reinforcement lEarning) 保持标准 PPO/GRPO 框架，唯一核心改动是信用分配方式：advantage 由 min-form 计算，而非标准的 gamma-weighted sum。可以与任何 PRM 配合使用。

### 关键设计

1. **Min-form 信用分配**:
   - 做什么：将每个 token 的价值定义为其后续所有步骤奖励的最小值
   - 核心思路：$V(s_t) = \min_{t' \geq t} r_{t'}$，优势为 $A_t = V(s_t) - V(s_{t-1})$。木桶效应——价值取决于最差的步骤
   - 设计动机：sum-form 下，模型可以用几个高奖励步骤弥补差的步骤；min-form 下，只要有一步差就会拉低整体价值，迫使模型提升每一步的质量

2. **可选的规则奖励补充（10%）**:
   - 做什么：在 PRM 基础上加入少量 verifiable outcome reward
   - 核心思路：仅对 10% 的训练样本使用规则验证的正确性奖励作为 anchor，进一步抑制 reward hacking
   - 设计动机：PRM 毕竟不完美，少量规则奖励提供 ground truth 校准

### 损失函数 / 训练策略
标准 RL 框架 + min-form advantage。在 Qwen2.5-Math-7B 等 3 个基础模型上验证。训练仅需标准方法 30% 的步数。

## 实验关键数据

### 主实验

| 方法 | AMC23 | 5 基准平均 | 训练步数 | 说明 |
|------|-------|----------|---------|------|
| Sum-form + PRM | 崩溃 | 崩溃 | - | 训练初始就崩溃 |
| **Min-form + PRM (PURE)** | ~75% | ~50% | **30%** | 仅需 30% 步数 |
| Rule-based reward | ~75% | ~50% | 100% | 标准方法 |
| **PURE + 10% 规则奖励** | **82.5%** | **53.3%** | 100% | 最佳模型 |

### 关键发现
- **Sum-form + PRM 训练直接崩溃**：标准 RL 信用分配 + PRM 在训练初始阶段就完全崩溃，验证了 reward hacking 的严重性
- **Min-form 仅需 30% 步数达到同等效果**：PRM 提供 dense reward 加速学习，配合 min-form 不再 hacking
- **PURE + 10% 规则奖励产出最强模型**：AMC23 82.5%，5 基准平均 53.3%，在 Qwen2.5-Math-7B 上的最佳成绩

## 亮点与洞察
- **首次清晰诊断 PRM + RL = reward hacking 的根因**：不是 PRM 本身不准，而是 sum-form 信用分配放大了 PRM 的微小误差。这个洞察对整个领域都有指导意义
- **Min-form 的"木桶效应"直觉非常优雅**：价值取决于最短板，迫使模型均匀提升每一步质量

## 局限性 / 可改进方向
- Min-form 可能过于保守：单步异常低分就会拖累整条轨迹的价值
- 目前只在数学推理任务上验证
- PRM 的质量仍然重要——min-form 缓解但不完全消除 reward hacking

## 相关工作与启发
- **vs PRIME**: PRIME 通过隐式奖励协同训练 PRM，PURE 从信用分配角度根本性解决问题，两者正交可组合
- **vs 规则奖励 RL**: PURE 证明 PRM 可以安全用于 RL，且用 30% 步数就够——PRM 的 dense signal 确实有用
- **vs GPO/VinePPO**: 这些方法改进信用分配粒度（步骤级 vs token 级），PURE 改变信用分配的数学形式（min vs sum）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Min-form 信用分配是全新思路，诊断 reward hacking 根因的分析深刻
- 实验充分度: ⭐⭐⭐⭐ 3 个基座模型 + 5 基准 + reward hacking 案例分析
- 写作质量: ⭐⭐⭐⭐ 问题定位清晰，但符号可以更简洁
- 价值: ⭐⭐⭐⭐⭐ 解决了 PRM 用于 RL 的核心障碍，使 dense process reward 终于可以安全使用

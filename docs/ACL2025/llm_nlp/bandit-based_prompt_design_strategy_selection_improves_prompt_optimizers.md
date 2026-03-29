# OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers

**会议**: ACL 2025  
**arXiv**: [2503.01163](https://arxiv.org/abs/2503.01163)  
**代码**: [GitHub](https://github.com/shiralab/OPTS)  
**领域**: Prompt 优化  
**关键词**: Prompt设计策略, Thompson采样, 多臂老虎机, EvoPrompt, 策略选择, BIG-Bench Hard  

## 一句话总结

首次提出 Prompt 设计策略的显式选择机制 OPTS——将 CoT/角色提示/少样本等多种策略视为多臂老虎机的"臂"，用 Thompson 采样动态选择要应用的策略，集成到 EvoPrompt 后在 BIG-Bench Hard 上将 GPT-4o mini 性能提升最高 50%，超越隐式策略选择（APET）和均匀采样。

## 背景与动机

Prompt 优化（如 EvoPrompt）能自动发现有效 prompt，但生成的 prompt 常不如人类专家精心设计的。Prompt 设计策略（CoT、角色提示、分步指令等）代表了最佳实践，可以提升优化效果。APET 方法将所有策略一起喂给 LLM 隐式选择，但：

1. **策略可能有负效果**：CoT 和角色提示在某些任务上反而降低性能
2. **隐式选择不可靠**：LLM 的优化能力有限，不能有效选择策略

## 核心问题

如何在 prompt 优化过程中显式选择最合适的设计策略——既利用人类知识又避免负效果策略？

## 方法详解

### OPTS 框架

将 K 种 prompt 设计策略 + 1 个"不使用策略"选项建模为 K+1 臂的多臂老虎机问题。

### 三种选择机制

1. **均匀采样（Uniform）**：每次随机选一种策略——基线
2. **ε-Greedy**：以 ε 概率探索随机策略，以 1-ε 概率选历史最优策略
3. **Thompson 采样（TS）**：维护每种策略的 Beta(α,β) 先验分布，根据采样值选择——**最优方法**
   - 策略成功（prompt 质量提升）→ 更新 α
   - 策略失败 → 更新 β
   - 自然平衡探索与利用

### 与 EvoPrompt 集成

- EvoPrompt 生成候选 prompt 后，OPTS 选择一种策略
- 选中策略的描述传给 LLM，LLM 将策略应用到候选 prompt 上
- 修改后的 prompt 评估性能，反馈给老虎机更新

## 实验关键数据

| 对比 | 结果 |
|------|------|
| EvoPrompt + OPTS(TS) vs 原始 EvoPrompt | **最高 +50%**（GPT-4o mini） |
| OPTS(TS) vs APET（隐式选择） | **一致更优** |
| OPTS(TS) vs 均匀采样 | **更优** |
| Llama-3-8B + OPTS(TS) | 也有一致提升 |

- 在 BIG-Bench Hard 多个任务上验证
- Thompson 采样总体最优——有效平衡探索未知策略和利用已知好策略

## 亮点

- **首次显式选择 prompt 设计策略**——概念简单但效果显著
- **Thompson 采样是优雅的选择**——有理论保证的探索-利用权衡
- **策略可能有害的清醒认识**——不是所有策略都有益，需要选择
- **模块化设计**：OPTS 可接入任何 prompt 优化器（不仅是 EvoPrompt）

## 局限性 / 可改进方向

- **仅在 BIG-Bench Hard 验证**：其他任务/基准未测试
- **策略集合固定**：未探索训练中动态发现新策略
- **仅两个 LLM**：更多模型的泛化性未知
- **上下文相关性**：不同任务可能需要不同策略组合而非单策略

## 与相关工作的对比

- **vs APET（隐式选择）**：将所有策略给 LLM 隐式选择可能次优；OPTS 显式选择更可控
- **vs 固定 CoT/角色提示**：总是用某策略可能在某些任务上有害；OPTS 动态适应
- **vs EvoPrompt（无策略）**：缺乏人类设计知识；OPTS 将最佳实践注入优化

## 启发与关联

- 多臂老虎机是 AutoML/超参搜索的标准工具——引入 prompt 优化是自然扩展
- "策略可能有害"的认识对所有 prompt 工程实践者都重要
- 显式选择 > 隐式选择的模式可推广到其他需要从多种方法中选择的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次显式策略选择+Thompson采样用于prompt优化
- 实验充分度: ⭐⭐⭐⭐ BIG-Bench Hard×2 LLM×3选择机制
- 写作质量: ⭐⭐⭐⭐ 动机清晰，框架图直观
- 价值: ⭐⭐⭐⭐ 对prompt优化从业者有直接实用价值

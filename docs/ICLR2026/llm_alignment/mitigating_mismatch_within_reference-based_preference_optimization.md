# Mitigating Mismatch within Reference-based Preference Optimization

**会议**: ICLR 2026  
**arXiv**: [2602.11902](https://arxiv.org/abs/2602.11902)  
**代码**: 无  
**领域**: LLM对齐 / 偏好优化  
**关键词**: DPO, reference policy, pessimistic bias, preference optimization, HyPO  

## 一句话总结
揭示 DPO 的"悲观 reference 偏差"问题——当 reference 策略对 chosen response 的概率低于 rejected 时（~45% pairs），DPO 会"过早满足"停止学习；提出 HyPO（一行代码修改：将 $\Delta_{ref}$ 裁剪为 $\max(0, \Delta_{ref})$），在 AlpacaEval 上相对 DPO 提升 41.2%。

## 研究背景与动机
1. **领域现状**：DPO 通过 $\Delta_\theta - \Delta_{ref}$ 的相对 margin 优化偏好，其中 $\Delta_{ref}$ 来自 reference 策略对 chosen/rejected 的概率差。
2. **现有痛点**：即使使用强 reference 模型，仍有约 45% 的 pair 出现 $\Delta_{ref} < 0$（reference 认为 rejected 比 chosen 更好）。此时 DPO 的梯度被 reference 的悲观信号过早衰减——即使策略仍错误（$\Delta_\theta < 0$），优化也近乎停止。
3. **核心矛盾**：DPO 的训练目标依赖 reference 的相对 margin，但测试时不依赖——training-inference mismatch。
4. **本文要解决什么？** 消除悲观 reference 对训练梯度的不必要衰减。
5. **切入角度**：分析 DPO sigmoid 的梯度流，发现悲观 pair 上的梯度被 reference margin 提前清零。
6. **核心idea一句话**：对 $\Delta_{ref}$ 做 max(0, ·) 裁剪，让悲观 pair 退化为绝对 margin DPO，保留乐观 pair 的 reference 引导。

## 方法详解

### 关键设计
HyPO：$\Delta_\theta - \max(0, \Delta_{ref})$
- 乐观 pair ($\Delta_{ref} > 0$)：保持 DPO 原有的相对 margin 优化。
- 悲观 pair ($\Delta_{ref} < 0$)：退化为 $\Delta_\theta$（绝对 margin），reference 不再干扰。

## 实验关键数据
| 方法 | AlpacaEval 2.0 LC↑ | Arena-Hard↑ |
|------|---------------------|-------------|
| DPO (Llama-3-8B) | 22.6% | - |
| **HyPO** | **27.3% (+41.2%)** | **11.2% vs 7.9%** |
| Pairwise win rate | HyPO 55.9% : DPO 44.1% |

## 亮点与洞察
- **一行代码修改**解决了 DPO 的系统性缺陷。
- 与 SquaredPO 互补：SquaredPO 解决概率位移，HyPO 解决悲观 reference。两者可组合。

## 评分
- 新颖性: ⭐⭐⭐⭐ 悲观 reference 分析深入
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准
- 写作质量: ⭐⭐⭐⭐ 分析到方案的逻辑清晰
- 价值: ⭐⭐⭐⭐ 对 DPO 实践有直接改进

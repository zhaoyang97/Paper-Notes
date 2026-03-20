# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization

**会议**: ICLR 2026  
**arXiv**: [2510.04474](https://arxiv.org/abs/2510.04474)  
**代码**: [https://github.com/Optimization-AI/DRPO](https://github.com/Optimization-AI/DRPO)  
**领域**: LLM推理  
**关键词**: efficient reasoning, overthinking, GRPO, length penalty, reinforcement learning  

## 一句话总结
诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

## 研究背景与动机
1. **领域现状**：大推理模型（DeepSeek-R1 等）通过 GRPO 训练获得强推理能力，但存在严重的 overthinking 问题——回答"2+3=?"也需要生成 ~1000 个 token。
2. **现有痛点**：现有 RL 方法通过在奖励中加入长度惩罚来鼓励简洁推理（如 RLOO-LP, ALP, HAPO），但几乎都导致显著的性能下降。
3. **核心矛盾**：GRPO 的 group-relative 优势函数在混合长度惩罚后会将正确但冗长的回答的优势推到负值——模型被误导为把有效推理当作负样本来惩罚。例如：6 个回答中 3 个正确，加入长度惩罚后第 3 个正确回答的优势从 +1 变为 -0.17。
4. **本文要解决什么？** 如何在缩短推理长度的同时最小化性能损失？
5. **切入角度**：将学习信号的计算从正负样本混合改为分离——正确回答的长度惩罚只在正确回答组内归一化，永远不会产生负学习信号。
6. **核心idea一句话**：解耦正负样本的奖励归一化，让长度惩罚只减弱（而不翻转）正确回答的学习信号。

## 方法详解

### 整体框架
DRPO 基于判别式优化框架（DisCO）而非 GRPO。目标函数的正样本项使用基于长度奖励的加权importance sampling（权重来自闭式最优分布），负样本项使用 log-sum-exp 聚合。正负完全解耦。

### 关键设计

1. **问题诊断：GRPO + 长度惩罚的根本缺陷**:
   - GRPO 的优势函数 $A(o_i|q) = \frac{r(o_i) - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ 将正确和错误样本混合归一化
   - 加入长度惩罚后，正确但冗长的回答 $r$ 可能低于组平均 → 优势为负 → 模型被惩罚做正确推理
   - 这个问题在 RLOO-LP、ALP、HAPO 等所有基于 GRPO 的方法中都存在

2. **DRPO 解耦设计**:
   - 做什么：正确回答的学习信号只在正确回答组内归一化
   - 核心思路：求解最优正数据分布 $P_q^* = \arg\max_P \mathbb{E}[r_l(o)] - \lambda D_{KL}(P, \pi_{old}^+)$，得到闭式解 $P_q^*(o) \propto \pi_{old}^+(o|q) \exp(r_l(o)/\lambda)$。每个正样本的权重 $\omega(o|q) = \frac{\exp(r_l(o)/\lambda)}{\mathbb{E}_{o\sim\pi^+}\exp(r_l(o)/\lambda)}$ 在正样本内归一化
   - 设计动机：权重 $\omega$ 始终为正——短回答权重大，长回答权重小，但永远不会为负。$\lambda$ 控制长度-准确率的 trade-off

3. **判别式目标函数**:
   - 正样本项：$\mathbb{E}_{o\sim\pi^+} \omega(o|q) s_\theta(o,q)$——用加权似然提升短正确回答
   - 负样本项：$-\tau \log \mathbb{E}_{o'\sim\pi^-} \exp(s_\theta(o',q)/\tau)$——log-sum-exp 自动强调难负样本
   - 约束：$D_{KL}(\pi_{old} || \pi_\theta) \leq \delta$ 保证训练稳定性
   - 当 $\lambda=+\infty$ 时退化为 DisCO（无长度惩罚）

### 损失函数 / 训练策略
基于 DisCO 框架，约束用 penalty 函数处理。在 DeepScaleR-Preview-Dataset（40.3K 数学题）上训练 1000 步。生成预算 8K token。每题采样 8 个回答。

## 实验关键数据

### 主实验（AES: Accuracy Efficiency Score）

| 方法 | 模型 | Pass@1 | Length | AES |
|------|------|--------|--------|-----|
| RLOO-LP | 1.5B | 0.567 | 2531 | -0.129 |
| ALP | 1.5B | 0.606 | 3494 | -0.387 |
| HAPO | 1.5B | 0.534 | 1791 | -0.519 |
| **DRPO** | **1.5B** | **0.624** | **1527** | **+0.178** |
| RLOO-LP | 7B | 0.692 | 2649 | -0.033 |
| **DRPO** | **7B** | **0.714** | **1502** | **+0.249** |

### 关键发现
- DRPO 是唯一在所有模型规模（1.5B/7B/8B）上都获得正 AES 的方法——所有基线在多数设置下 AES 为负
- 1.5B 模型在 GSM8K 上：DRPO 77% 长度缩减仅 1.1% 性能损失 vs 基线 68% 缩减 4.3% 损失
- 7B 模型：DRPO 51% 长度缩减仅 2.6% 性能损失 vs RLOO-LP 38% 缩减 7.1% 损失
- $\lambda$ 可以平滑控制长度-准确率 trade-off：$\lambda\to\infty$ 无长度控制，$\lambda\to 0$ 最大化长度惩罚
- 在非数学推理任务（K&K 逻辑谜题）上也有效

## 亮点与洞察
- **"正负解耦"是核心洞察**：GRPO 的问题不在长度惩罚本身，而在于正负样本混合归一化。解耦后问题自然解决——这是一个清晰优雅的诊断和修复
- **闭式最优分布的理论优美性**：不需要额外训练奖励模型或采集数据，直接从 RLHF 的 KL 正则化框架推导出加权方案
- **对所有 GRPO 变体的通用诊断**：论文指出 RLOO、REINFORCE 等所有相对优势方法在复合奖励下都有此问题——DRPO 的解耦原则具有通用性

## 局限性 / 可改进方向
- 基于 DisCO 框架而非 GRPO，可能需要更多工程适配
- 长度奖励 $r_l(o) = 1 - |o|/C$ 是简单线性的，更复杂的长度-质量关系可能需要非线性设计
- 仅在数学推理上验证，代码推理、科学推理等其他领域需要确认
- 生成预算限制在 8K token，对超长推理的效果未知

## 相关工作与启发
- **vs GRPO + 长度惩罚（RLOO-LP/ALP/HAPO）**: 这些方法都受限于混合归一化导致的负优势问题，DRPO 通过解耦彻底解决
- **vs DisCO**: DRPO 在 DisCO 基础上引入了长度奖励的闭式加权方案，是 DisCO 在高效推理方向的自然扩展
- **vs L1-max / ShorterBetter**: 这些方法用不同机制控制长度，但也面临性能-效率 trade-off。DRPO 的 AES 持续最优
- **vs VIP (Adaptive Rollout)**: VIP 在采样前优化计算分配，DRPO 在训练目标上优化学习信号——两者互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 诊断清晰、解决方案优雅、闭式解理论完备
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个模型、6 个基线、4 个难度级别、AES 定量对比
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1 的诊断直观易懂，理论推导干净
- 价值: ⭐⭐⭐⭐⭐ 解决了高效推理训练中的核心矛盾，实用性极强

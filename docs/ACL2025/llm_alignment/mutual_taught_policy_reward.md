# Mutual-Taught for Co-adapting Policy and Reward Models

**会议**: ACL 2025  
**arXiv**: [2506.06292](https://arxiv.org/abs/2506.06292)  
**代码**: [https://github.com/Stycoo/Mutual-Taught](https://github.com/Stycoo/Mutual-Taught)  
**领域**: 对齐RLHF  
**关键词**: reward hacking, distribution shift, EM algorithm, iterative DPO, self-training  

## 一句话总结
Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

## 研究背景与动机
1. **领域现状**：RLHF/DPO 对齐过程中，随着 policy model 迭代优化，其输出分布会偏离训练 reward model 时使用的数据分布
2. **现有痛点**：
   - 分布偏移导致 reward hacking——模型学会得到高 RM 分但不真正反映人类偏好的输出
   - 持续收集人类标注来更新 RM 成本高昂，不可扩展
   - LLM-as-Judge（如 Self-Rewarding）需要强基座模型或预训练判断能力，对弱模型不适用
   - 现有迭代 DPO 方法假设 RM 是固定 oracle，忽略了 RM 自身的退化
3. **核心矛盾**：PM 在优化过程中不断变化，但 RM 保持不变，导致评估信号越来越不准确
4. **本文要解决什么？** 在无需额外人类标注的情况下，同时改进 PM 和 RM
5. **切入角度**：将 PM-RM 的协同优化建模为 EM 问题——隐变量是最优偏好分布
6. **核心idea一句话**：E-step 用 RM 优化 PM，M-step 用 PM 更新前后的输出对比来优化 RM，形成正向循环

## 方法详解

### 整体框架
Mutual-Taught 迭代执行两步：(1) **E-step**：用当前 RM $r_{t-1}$ 从 PM $\pi_{t-1}$ 采样并标注偏好，用 DPO 更新 PM 得到 $\pi_t$；(2) **M-step**：对同一 prompt，比较 $\pi_{t-1}$ 和 $\pi_t$ 的输出，将 $\pi_t$ 的输出视为 preferred（因为已经过 RM 优化），构建伪偏好对更新 RM 得到 $r_t$。

### 关键设计

1. **E-step: Policy Model 更新**:
   - 做什么：用 iterative DPO 更新 PM，以上一轮 PM 为 reference model
   - 核心思路：从 $\pi_{t-1}$ 采样多个回复，用 $r_{t-1}$ 标注 chosen/rejected，执行 DPO 训练得到 $\pi_t$
   - 与标准 iterative DPO 的区别：标准 iterative DPO 假设 RM 固定不变，Mutual-Taught 每轮都更新 RM
   - 稳定策略：通过验证集 model selection，只选择比上一轮有提升的 checkpoint，保证单调改进

2. **M-step: Reward Model 更新**:
   - 做什么：用 PM 更新前后的输出差异构建伪偏好对，更新 RM
   - 核心思路：对 prompt $x$，$y_t \sim \pi_t$ 和 $y_{t-1} \sim \pi_{t-1}$。因为 $\pi_t$ 是通过 $r_{t-1}$ 优化的，所以 $y_t$ 通常优于 $y_{t-1}$，构建 $(y_t \succ y_{t-1})$ 伪偏好对，用 Bradley-Terry loss 更新 RM
   - 设计动机：这些伪偏好对自然来自 PM 的进化分布——不需要外部反馈！RM 通过学习区分新旧 PM 的输出，自动适应 PM 的分布变化
   - 数据过滤：计算 reward margin $\Delta r(x) = r_{t-1}(y_t; x) - r_{t-1}(y_{t-1}; x)$，移除 $\Delta r \leq -\epsilon_t$ 的噪声样本（$\epsilon_t$ 是方差自适应阈值）

3. **Two-Stage Stabilization**:
   - E-step stabilization：验证集 model selection，win rate < $\tau$ 时停止迭代
   - M-step stabilization：基于方差的自适应数据过滤，去除高置信噪声；保留轻微噪声作为正则化

### 训练策略
- Base PM: Llama-3-8B-Instruct；Base RM: FsfairX-Llama3-RM-v0.1
- 训练数据: UltraFeedback (~60K prompts)，分三个子集轮流用于 PM/RM 训练
- 每轮完整训练包含两次 PM 更新和一次 RM 更新

## 实验关键数据

### 主实验

**Policy Model 性能**:

| 方法 | AlpacaEval-2 LC WR | Arena-Hard WR | 说明 |
|------|:------------------:|:-------------:|------|
| Llama-3-8B-Instruct | 23.1 | 20.6 | 基准 |
| DPO (offline) | 44.3 | 33.1 | 离线 DPO |
| Meta-Rewarding Iter3 | 37.5 (+14.4) | 27.9 (+7.3) | LLM-as-Judge |
| SPPO Iter3 | 46.4 (+23.3) | 33.6 (+13.0) | Self-Play |
| Iterative DPO Iter3 | 47.2 (+24.1) | 38.5 (+17.9) | 固定 RM |
| **Mutual-Taught** | **54.1 (+31.0)** | **38.4 (+17.8)** | PM+RM 协同 |

**Reward Model 性能 (RewardBench)**:

| 模型 | Chat | Chat-Hard | Safety | Reasoning | Avg. |
|------|:----:|:---------:|:------:|:---------:|:----:|
| FsfairX-RM (初始) | 96.9 | 74.1 | 86.8 | 97.1 | 88.7 |
| **FsfairX-RM-MT** | **97.5** | **78.4** | **87.6** | **97.8** | **90.3** |
| GPT-4o-2024-08-06 | 96.6 | 76.1 | 88.1 | 86.6 | 86.7 |

RM 提升到与 GPT-4o 相当甚至在部分维度超越！

### 消融实验

| 配置 | AlpacaEval-2 LC WR | 说明 |
|------|:------------------:|------|
| Full Mutual-Taught | **54.1** | 完整方法 |
| w/o RM update (固定 RM) | 47.2 | 退化为 iterative DPO |
| w/o model selection | 50.3 | PM 可能过拟合 |
| w/o data filtering | 51.8 | 噪声伪标签影响 RM |

### 关键发现
- RM 更新是关键贡献——去掉 RM 更新后退化为 iterative DPO（54.1→47.2），说明解决分布偏移至关重要
- RM 更新后 PM 的后续训练效果更好——证实了"更好的 RM → 更好的 PM → 更好的 RM"正循环
- 伪偏好对的质量足够好——$\pi_t$ 的输出通常确实优于 $\pi_{t-1}$，验证了"连续优化产生自然偏好对"的假设
- 数据过滤策略中保留轻微负向样本（$-\epsilon_t < \Delta r < 0$）作为正则化有帮助

## 亮点与洞察
- **EM 框架下的 PM-RM 协同进化**：将 RLHF 中的分布偏移问题形式化为缺失数据问题（最优偏好分布是隐变量），用 EM 迭代逼近——理论视角优雅
- **零外部标注的 RM 更新**：利用 PM 自身进化产生的"前后对比"作为训练信号，完全自包含——不需要 LLM-as-Judge 能力或额外人类标注
- **轻微噪声作为正则化**：保留轻微负向伪偏好对防止 RM 过拟合于 PM 分布——这个发现值得注意

## 局限性 / 可改进方向
- **RM-PM同步退化风险**：如果初始 RM 质量差，产生的伪偏好对质量也差，可能陷入恶性循环。框架假设初始 RM "够好"
- **计算成本**：每轮迭代需要 PM 训练 + RM 训练 + 大量推理采样，成本约为标准 DPO 的 3-4 倍
- **仅验证了 8B 模型**：更大模型的效果和迭代轮数需求未知
- **伪偏好对的"自循环"假设**：假设 RM 优化后的 PM 输出总是更好——但 reward hacking 可能导致 PM 输出"看起来更好但实际更差"

## 相关工作与启发
- **vs Iterative DPO**: Iterative DPO 只更新 PM 不更新 RM，Mutual-Taught 同时更新两者——额外 +7% LC win rate 证明了 RM 更新的价值
- **vs Self-Rewarding/Meta-Rewarding**: 这些方法用 PM 自身作为 judge，需要强 judge 能力；Mutual-Taught 用独立 RM 更可靠
- **vs ReST^EM**: 同用 EM 框架但 ReST^EM 只更新 PM，Mutual-Taught 同时更新 RM
- 可以探索将 M-step 扩展到多个 RM（ensemble）来提高伪偏好对的可靠性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ PM-RM 协同进化的 EM 框架是全新范式，伪偏好对构建方法巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ PM 和 RM 双向评估，多个强 baseline 对比，消融完整
- 写作质量: ⭐⭐⭐⭐⭐ EM 框架推导清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了 RLHF 中的核心问题（分布偏移），效果显著且不需额外标注

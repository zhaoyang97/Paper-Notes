# Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework

**会议**: ICLR 2026  
**arXiv**: [2506.05619](https://arxiv.org/abs/2506.05619)  
**代码**: 补充材料中包含实验代码  
**领域**: AI Alignment / Social Choice Theory  
**关键词**: RLHF, NLHF, 偏好学习, 社会选择理论, 人群比例对齐, 公理化框架  

## 一句话总结
提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

## 背景与动机
1. RLHF 依赖 Bradley-Terry 模型将偏好压缩为标量奖励，在偏好不一致/循环偏好场景下失效
2. NLHF 将偏好学习建模为博弈，找 Nash 均衡策略，但仍不能按比例反映评估者分布
3. **核心问题**：当两组评估者对两个选项的偏好接近 50:50 时（如 50+ε vs 50−ε），RLHF 和 NLHF 都会输出确定性策略选择微弱多数方，完全忽略少数群体
4. 现有多元对齐方法（mixture-based、steerable models）通常需要显式的评估者群组标签，实际中难以获取
5. 已有公理化方法（如随机独裁制 Random Dictatorship）虽满足比例对齐，但无法仅从成对比较数据实现
6. 本文目标：不需额外群组信息，仅从成对比较数据实现比例对齐

## 方法详解
### 框架设计
- **可行人群分布集推断**：定义 $u_i = \min_{y \neq y_i} P(y_i \succ y)$ 作为每个选项 $y_i$ 的人群份额上界，构建多面体外近似 $\bar{\mathcal{W}}(P) = \{w \in \Delta(\mathcal{Y}) | w_i \leq u_i\}$
- **策略构造**：按上界比例分配概率 $\pi(y_i) = u_i / \sum_j u_j$，采用保守策略最小化最坏情况下的比例失配

### 四条公理
1. **单调性**：提升某选项排名不会降低其被选概率
2. **Pareto 效率**：若所有人偏好 $y$ 胜 $y'$，策略应倾向 $y$
3. **α-PPA（人群比例对齐）**：$\pi(y_k)/w_k^\sigma \geq \alpha(\sigma)$，保证策略至少弱比例于人群份额
4. **γ-PBM（人群有界可操纵性）**：操纵后策略增益受 $\gamma_1 w_k^\sigma + \gamma_2$ 约束，非多数群体无法通过操纵获得多数地位

### Softmax 松弛
- 引入参数 $\beta$ 控制比例对齐与 Condorcet 一致性的权衡：$\pi(y_i) = u_i \exp(\beta u_i) / \sum_j u_j \exp(\beta u_j)$
- $\beta=0$ 退化为原始 $F^*$；$\beta \to \infty$ 收敛至 minimax Condorcet 方法

## 实验
### 表格实验：MovieLens 电影推荐
| 方法 | 胜率 | PPA 水平 | PBM 增益 |
|------|------|----------|----------|
| RLHF | 0.7784 | 0 | 0.0611 |
| NLHF | 0.7712 | 0 | 0.0124 |
| $F^\beta$($\beta=1$) | ~0.60 | 0.4869 | 8.9e-4 |

- β 增大时胜率升高但 PPA 下降，验证理论预测的权衡关系
- 提出方法在 β≤10 时操纵抗性显著优于基线

### LLM 实验：Qwen2.5-3B-Instruct
| 数据集 | β=0 PPA | DPO PPA |
|--------|---------|---------|
| Synthetic-Color | 0.0883 | 0.0000 |
| Alpaca-Expertise | 0.1428 | 0.1321 |
| Alpaca-Style | 0.5012 | 0.3786 |

- 合成数据上权衡明显；Alpaca 数据因 GPT-4.1 注释噪声效果较弱
- 计算代价与 RLHF 相当，高于 DPO

## 亮点
- 理论严谨：证明 RLHF/NLHF 违反任意强度的 PPA 和 PBM 公理
- 仅需成对比较数据即可推断人群分布可行集，不需要群组标签
- Softmax 松弛提供比例对齐与 Condorcet 一致性的可调权衡
- 操纵抗性有理论保证：非多数群体无法通过策略性误报获得多数地位

## 局限性
- PPA 仅关注各群组首选项的选择概率，忽略低排名偏好
- LLM 场景下评估 PPA 水平仍是开放问题（logit 估计 vs 群组分类均有噪声）
- 两阶段函数近似方法计算开销不低于 RLHF，需开发直接策略优化版本
- 外近似 $\bar{\mathcal{W}}$ 在选项数多时可能过于宽松

## 相关工作
- **RLHF / DPO**：等价于最大 Borda 规则，确定性选择胜者
- **NLHF**：等价于最大彩票(Maximal Lotteries)，满足 Pareto 但不满足 PPA
- **Random Dictatorship**：完美 PPA 但不可从成对比较实现
- **多元对齐**（Sorensen 2024, Chen 2024）：需要显式群组标签
- **抗操纵机制**（Buening 2025, Park 2024）：追求严格策略防篡改，本文约束群体层面增益

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

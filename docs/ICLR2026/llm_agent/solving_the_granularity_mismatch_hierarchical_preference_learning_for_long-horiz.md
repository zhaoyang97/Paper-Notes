# Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents

**会议**: ICLR 2026  
**arXiv**: [2510.03253](https://arxiv.org/abs/2510.03253)  
**代码**: 待确认  
**领域**: Agent / 对齐  
**关键词**: hierarchical DPO, preference learning, long-horizon agent, curriculum learning, action group  

## 一句话总结
提出 HPL 框架解决长时序 LLM Agent 中偏好学习的粒度不匹配问题，通过三级 DPO（轨迹级+步骤级+动作组级）和双层课程学习（子任务复杂度×样本难度），在 ALFWorld/WebShop/InterCode-SQL 上显著超越 ETO 和 IPR 等基线（平均 59.44 vs 55.43/55.49）。

## 研究背景与动机
1. **领域现状**：DPO（Direct Preference Optimization）已成为 LLM 对齐的主流方法，但在长时序 Agent 任务中存在粒度不匹配——轨迹级 DPO 信号太粗（无法定位关键决策点），步骤级信号方差太大。
2. **现有痛点**：现有 Agent 偏好学习方法要么用 outcome-level 奖励（整个轨迹成功/失败），要么用 step-level 但需要大量 rollout 降低方差。两种粒度各有优劣，缺乏统一框架。
3. **核心矛盾**：太粗 → 无法精确信用分配；太细 → 方差过大，样本效率低。需要"刚好合适"的粒度。
4. **本文要解决**：设计一个多粒度偏好学习框架，同时利用轨迹级、步骤级和动作组级的偏好信号。
5. **切入角度**：将动作序列按语义一致性分组（如"导航到厨房"是一个组，"打开冰箱取物品"是另一个组），在组级别做偏好对比。
6. **核心idea**：三级 DPO 提供互补的信用分配信号 + 双层课程学习从简到难引导训练。

## 方法详解

### 整体框架
HPL 在参考策略的探索数据上构建三个层次的偏好对：① 轨迹级比较完整轨迹 → ② 步骤级从决策点做 Monte Carlo rollout 比较后续子轨迹 → ③ 动作组级比较语义一致的动作组对。三个 DPO 损失加权组合，配合双层课程学习调度训练。

### 关键设计

1. **三级 DPO 偏好信号**:
   - $L_{traj\text{-}DPO}$：比较完整轨迹对，提供全局信号
   - $L_{step\text{-}DPO}$：从决策点做 Monte Carlo rollout($M=5$) 比较后续子轨迹
   - $L_{group\text{-}DPO}$：比较语义一致的动作组对
   - 最终损失：$L = L_{BC} + L_{traj} + L_{step} + L_{group}$
   - 理论保证（Proposition 1）：group-level DPO 在 $k=\Theta(\log(1/\varepsilon))$ 时方差改善 $O(T/\log(1/\varepsilon))$ 倍

2. **动作组分割策略**:
   - Fixed-N：固定 $N=3$ 个组
   - Fixed-K：固定每组 $K=3$ 步
   - Uncertainty-based：基于策略熵在 80 百分位阈值处切分
   - **Semantic**（最优）：用 GPT-4o 作为语义分割器，按子任务含义分组
   - 设计动机：语义分割产生的组内一致性最高，DPO 信号质量最好

3. **双层课程学习**:
   - 3×3 难度矩阵：Y 轴 = 组长度（子任务复杂度），X 轴 = $\Delta R = \hat{r}(G_w) - \hat{r}(G_l)$（样本可区分度）
   - Phase 1：$B_{1,1}$（短+易）→ Phase 2：$B_{1,1} \cup B_{1,2} \cup B_{2,1}$ → Phase 3：全部 9 个桶
   - 设计动机：先让模型在简单样本上建立基础偏好，再逐步引入困难样本

## 实验关键数据

### 主实验（Qwen2.5-1.5B）
| 方法 | ALFWorld unseen | WebShop reward | InterCode-SQL | 平均 |
|------|----------------|----------------|---------------|------|
| ETO | 66.42 | 56.57 | 57.67 | 55.43 |
| IPR | 66.67 | 57.76 | 57.17 | 55.49 |
| **HPL(Semantic)** | **74.13** | **60.74** | **58.50** | **59.44** |
| GPT-4o zero-shot | 36.43 | — | — | — |

### 分割策略对比
| 策略 | 平均分数 |
|------|---------|
| **Semantic (GPT-4o)** | **59.44** |
| Fixed-N (3) | 58.45 |
| Uncertainty | 56.95 |
| Fixed-K (3) | 56.74 |

### 消融实验
| 配置 | 效果 |
|------|------|
| Full HPL | **最优** |
| w/o group-DPO | 性能下降 |
| w/o 课程学习 | 长组和困难样本学习受损 |
| w/o step-DPO | 信用分配粗糙化 |

### 关键发现
- Semantic 分割显著优于其他策略（59.44 vs 56.74-58.45），语义一致性是组级 DPO 的关键
- HPL 超越 GPT-4o zero-shot（ALFWorld 74.13 vs 36.43），1.5B 模型训练后远超闭源大模型
- 三级 DPO 的互补性：移除任何一级都降低性能
- 课程学习对困难样本和长组尤其重要

## 亮点与洞察
- **动作组级 DPO** 是一个介于轨迹和步骤之间的"甜蜜点"——粒度恰好匹配子任务边界
- **语义分割 > 固定分割**的发现说明：分组的质量比分组的方式更重要
- 3×3 **双层课程**的设计很实用——同时考虑任务复杂度和样本难度两个维度
- 理论保证（方差改善 $O(T/\log(1/\varepsilon))$）为实践提供了数学支撑

## 局限性 / 可改进方向
- 依赖冻结参考策略的一次性探索收集数据，非在线 RL
- Monte Carlo rollout 数量有限（$M=5$），每步估计方差仍可能较大
- 语义分割依赖 GPT-4o，增加成本和外部依赖
- 未探索自适应粒度选择（不同步骤根据不确定性选择不同粒度）

## 相关工作与启发
- **vs ETO**: ETO 仅用轨迹级信号，无法精确定位失误步骤
- **vs GRPO/GiGPO**: GRPO 用 group-relative advantage，HPL 用 group-level DPO，互补
- **vs RLHF**: HPL 避免了 reward model 训练，直接从偏好对学习
- 可启发多粒度反馈在 Agent 训练中的应用

## 评分
- 新颖性: ⭐⭐⭐⭐ 三级 DPO + 双层课程的组合设计合理且有效
- 实验充分度: ⭐⭐⭐⭐ 三个 benchmark，多种分割策略对比
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合好
- 价值: ⭐⭐⭐⭐ 为长时序 Agent 对齐提供了实用框架

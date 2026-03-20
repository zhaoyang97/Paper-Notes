# GoalRank: Group-Relative Optimization for a Large Ranking Model

**会议**: ICLR 2026  
**arXiv**: [2509.22046](https://arxiv.org/abs/2509.22046)  
**代码**: 无  
**领域**: LLM对齐 / 推荐排序  
**关键词**: ranking, generator-only, group-relative optimization, scaling law, recommendation  

## 一句话总结
理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

## 研究背景与动机
1. **领域现状**：推荐系统排序阶段是从 N 个候选中选出长度 L 的有序列表（P(N,L) 组合空间）。主流方案是 Generator-Evaluator 两阶段范式：generator 产出候选列表，evaluator 选最优。
2. **现有痛点**：增加 generator 数量/多 generator 集成的收益迅速饱和（Fig 1d）。两阶段范式引入跨阶段不一致性和工程复杂度。
3. **核心矛盾**：两阶段方法的策略空间受限于 k 个小 generator 的混合，而端到端大模型的 scaling law 表明更大的单一模型可能更好。
4. **本文要解决什么？**（1）generator-only 能否理论上超越 Multi-G-E？（2）如何训练这样的大排序模型？
5. **切入角度**：Theorem 1 证明更大 generator-only 的逼近误差严格小于任意有限 Multi-G-E，且随模型增大趋向零。
6. **核心idea一句话**：用 reward model 在候选列表组上构建 group-relative softmax 参考策略，通过交叉熵训练大排序模型逼近最优策略。

## 方法详解

### 整体框架
(1) 训练 reward model $\hat{r}(l)$ 预估列表级用户反馈 → (2) 对每个用户构建列表组 $\mathcal{B}_u$（含主模型和多种辅助策略生成的列表）→ (3) 用 group-relative softmax 构建参考策略 $\pi^{ref}$ → (4) 最小化 $\pi_\theta$ 与 $\pi^{ref}$ 的交叉熵。

### 关键设计

1. **Theorem 1（理论支撑）**：对任意 k-mixture $(α,β)$-bounded 策略空间 $\mathcal{C}_m^k$，存在宽度 $\geq kα+n$ 的 generator-only 策略空间 $\mathcal{F}_M$，使得 $\mathcal{E}(\mathcal{F}_M) < \mathcal{E}(\mathcal{C}_m^k)$，且 $\lim_{n→∞} \mathcal{E}(\mathcal{F}_M) = 0$。

2. **Group-Relative 参考策略**：
   - $\pi^{ref}(l|\mathcal{B}) = \frac{\exp((\hat{r}(l) - \bar{r}_\mathcal{B})/\sigma_\mathcal{B})}{\sum_{l'} \exp((\hat{r}(l') - \bar{r}_\mathcal{B})/\sigma_\mathcal{B})}$
   - 标准化（减均值除标准差）使 biased reward model 在组内保持序关系。
   - 条件：组内 reward 差距 $> \sigma^*$ 时序关系可靠。

3. **列表组构建**：引入辅助策略集 $\mathcal{M}$（包含启发式和轻量神经模型），为每个用户产生多样化列表组，保证组内有足够大的 reward 差距。

## 实验关键数据

### 主实验（ML-1M / Industry / Amazon-Book）
| 方法 | ML-1M H@6 | Industry H@6 | Book H@6 |
|------|-----------|-------------|----------|
| Best G-E (PIER) | 62.74 | 45.35 | 71.14 |
| Best MG-E (G-100) | 60.64 | - | - |
| **GoalRank** | **64.51** | **55.33** | **74.29** |
| 提升 vs best baseline | +2.77% | +11.3% | +1.7% |

### Scaling Law 验证
GoalRank 的性能随模型参数量/训练数据量稳定提升，展现出清晰的 scaling law。

### 在线 A/B 测试
在工业级短视频平台上，GoalRank 相比 SOTA 基线在核心指标上实现了显著提升。

### 关键发现
- Generator-only GoalRank 在所有数据集上超越所有 G-E 和 MG-E 基线，验证了 Theorem 1。
- MG-E 方法增加 generator 从 3→100，性能提升饱和甚至下降。
- Group-relative 标准化使训练对 reward model 的绝对偏差不敏感，只需序关系正确。

## 亮点与洞察
- **理论驱动的实践**：Theorem 1 不仅是理论装饰，而是指导了整个框架设计——放弃两阶段范式、scaling up generator、用 group-relative 避免精确 reward。
- **与 DPO/RLHF 的呼应**：GoalRank 的 group-relative 优化与 LLM 对齐中的 RLHF/DPO 高度类似——都用 reward 信号构建参考分布来训练策略，只是应用场景从"文本生成"变成了"列表排序"。
- **Scaling Law 的推荐系统版本**：首次在排序任务上展示了与 LLM 类似的 scaling law。

## 局限性 / 可改进方向
- 需要预训练 reward model，其质量上限决定了 GoalRank 的上限。
- 辅助策略集 $\mathcal{M}$ 的选择对列表组多样性和 reward 差距有影响。
- 仅在推荐排序场景验证，未探索在 LLM 对齐（如 RLHF）中的应用。

## 相关工作与启发
- **vs PIER/NAR4Rec**：两阶段 G-E 方法，受限于候选列表空间的有限覆盖。GoalRank 直接在全策略空间优化。
- **vs DPO/RCPO**：RCPO 在文本对齐中用 ranked choice 替代 pairwise，GoalRank 在推荐排序中用 group-relative 替代 pointwise。核心思想一致。

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论证明 generator-only 优于 G-E 是有力贡献，group-relative 优化思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 公开数据集 + 工业数据集 + 在线 A/B 测试 + scaling law
- 写作质量: ⭐⭐⭐⭐ 理论和实践结合好
- 价值: ⭐⭐⭐⭐ 对推荐系统排序范式有重要影响

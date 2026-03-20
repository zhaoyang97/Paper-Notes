# DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization

**会议**: ACL 2025  
**arXiv**: [2411.14055](https://arxiv.org/abs/2411.14055)  
**代码**: [https://github.com/hexuandeng/DRPruning](https://github.com/hexuandeng/DRPruning)  
**领域**: LLM效率  
**关键词**: structured pruning, distributionally robust optimization, data scheduling, scaling law, domain balance  

## 一句话总结
DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

## 研究背景与动机
1. **领域现状**：结构化剪枝（如 Sheared LLaMA）可以将大模型压缩为小模型，但剪枝后通常需要 continued pretraining 恢复能力
2. **现有痛点**：
   - 剪枝后不同领域的性能退化不均匀——某些领域恢复快，某些领域严重受损，导致偏差
   - 标准 DRO 需要手动设置关键超参数（reference loss 和 reference data ratio），设置不当效果很差
   - Sheared LLaMA 的动态调度策略参照大模型各领域 loss 比例，但在多语言等分布偏移大的场景下失效
3. **核心矛盾**：如何在剪枝后的持续预训练中自动平衡各领域性能，无需大量超参调优？
4. **本文要解决什么？** 自动化确定 DRO 的 reference loss 和 reference data ratio
5. **切入角度**：用 scaling law 预测训练结束时的 loss 作为 reference loss，用 DRO 权重的 EMA 更新 reference data ratio
6. **核心idea一句话**：用 scaling law 自动预测各领域可达到的最优 loss，结合渐进式数据比例调整，实现剪枝后的全领域均衡恢复

## 方法详解

### 整体框架
DRPruning 在 Sheared LLaMA 的结构化剪枝 + continued pretraining 框架上，加入三个改进：(1) 用 DRO 动态调整训练数据比例；(2) 用 scaling law 预测 reference loss；(3) 渐进式更新 reference data ratio。

### 关键设计

1. **基于 Scaling Law 的动态 Reference Loss**:
   - 做什么：预测模型在训练结束时各领域能达到的最低 loss，作为 DRO 的参考基准
   - 核心思路：利用 $\hat{\ell}(P, T) = A \cdot P^{-\alpha} \cdot T^{-\beta} + E$ 拟合各领域的 loss 曲线，用拟合曲线在总训练步数处的预测值作为 reference loss。每次评估后重新拟合
   - 设计动机：手动设置 reference loss 困难且不可靠。Scaling law 提供了数据驱动的预测，避免了人工调参。在训练完成 20% 后开始预测（需足够数据点）

2. **渐进式 Reference Data Ratio 更新**:
   - 做什么：让数据分布的约束中心逐渐向高 loss 领域移动
   - 核心思路：$\mathbf{p}_R^{t+1} = \delta \cdot \mathbf{q}^t + (1-\delta) \cdot \mathbf{p}_R^t$，其中 $\mathbf{q}^t$ 是 DRO 计算的最优权重。用 EMA 平滑更新，避免剧烈波动
   - 设计动机：固定 reference ratio 过于保守（Zhou et al., 2021），限制了 DRO 对困难领域的关注。渐进更新允许逐步扩大对困难分布的覆盖
   - 安全约束：各领域比例限制在初始比例的 $[1/n, n]$ 倍之间，防止退化为只训练最差领域

3. **DRO 权重更新**:
   - 做什么：根据各领域的"excess loss"（当前 loss - reference loss）调整数据采样权重
   - 核心思路：对 loss 偏离 reference 更多的领域分配更高权重，在 $\chi^2$-divergence ball 约束下求解最大化问题
   - 与 Sheared LLaMA 的区别：Sheared LLaMA 参照大模型的绝对 loss 值排序来分配权重，DRPruning 参照模型自身的可达最优值

### 训练策略
- 先剪枝 0.4B tokens（学习剪枝 mask），后续 50B tokens 做 continued pretraining
- 从 Llama2-7B 剪枝到 1.3B 和 2.7B 两种目标规模
- DRO 更新在每次 evaluation 时进行

## 实验关键数据

### 主实验

| 方法 | 7B→1.3B PPL | 7B→1.3B Task Avg. | 7B→2.7B PPL | 7B→2.7B Task Avg. |
|------|:----------:|:-----------------:|:----------:|:-----------------:|
| Sheared LLaMA | 10.05 | 34.89 | 7.64 | 39.75 |
| ReSheared (复现) | 10.42 | 34.85 | 7.83 | 39.98 |
| **DRPruning** | **9.83** | **35.60** | **7.40** | **40.18** |

PPL 下降 -5.59%，下游任务提升 +1.52%。

**指令微调胜率（vs Sheared LLaMA, GPT-4o 评估）**: 55.4% 胜率。

### 消融实验

| 配置 | PPL | Task | 说明 |
|------|:---:|:----:|------|
| DRPruning (full) | 9.83 | 35.60 | 完整方法 |
| w/o dynamic ref loss | 10.15 | 35.20 | scaling law 预测有效 |
| w/o dynamic ref ratio | 10.02 | 35.35 | 渐进式比例更新有效 |
| Fixed ref ratio | 10.20 | 35.10 | 固定比例明显劣于动态 |

### 关键发现
- 在多语言设置中优势更大（+2.95% 下游任务），说明 DRPruning 对分布偏移大的场景特别有效
- Scaling law 预测的 reference loss 数值稳定——这是方法有效的关键前提
- 领域级别评估显示 DRPruning 在最差领域的提升最大（+17.9%），符合 DRO 的 worst-case 优化目标

## 亮点与洞察
- **Scaling Law 驱动 DRO**：用 scaling law 自动预测 reference loss 是很优雅的自动化方案——将 DRO 从"需要专家知识设超参"变为完全数据驱动
- **渐进式约束放松**：逐步将 reference ratio 向困难领域移动，兼顾了探索（尝试新分布）和利用（在已知好分布上训练）
- **不仅用于剪枝**：reference loss 和 data ratio 优化方法可以独立于剪枝使用，适用于任何多领域 continued pretraining

## 局限性 / 可改进方向
- **仅在 Llama2-7B 上验证**：更大模型和其他架构（Mistral、Qwen）的效果未知
- **Scaling Law 拟合需要足够数据点**：训练前 20% 无法使用动态 reference loss，对短训练 schedule 可能不充分
- **领域划分预定义**：需要预先将数据划分为明确的领域，对无标签数据不适用

## 相关工作与启发
- **vs Sheared LLaMA**: 同样基于结构化剪枝 + continued pretraining，但数据调度策略更优——自动化 DRO vs 启发式调度
- **vs Group DRO**: DRPruning 解决了 Group DRO 的两个核心超参问题（reference loss + reference ratio），使其在 LLM 设置中实际可用
- 自动确定 reference loss 的方法可推广到 DPO/RLHF 的多任务训练中

## 评分
- 新颖性: ⭐⭐⭐⭐ Scaling law + DRO 组合新颖，自动化超参设置有实际价值
- 实验充分度: ⭐⭐⭐⭐ 单语+多语、PPL+下游+指令微调多维评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 提供了结构化剪枝的最佳实践，DRO 自动化具有广泛适用性

# Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)

**会议**: ICLR 2026  
**arXiv**: [2602.01601](https://arxiv.org/abs/2602.01601)  
**代码**: [https://github.com/HieuNT91/VIP](https://github.com/HieuNT91/VIP)  
**领域**: LLM效率  
**关键词**: GRPO, rollout allocation, gradient variance, Gaussian process, sampling efficiency  

## 一句话总结
提出 VIP（Variance-Informed Predictive allocation），通过高斯过程预测每个 prompt 的成功概率，据此用凸优化在计算预算约束下分配 rollout 数量以最小化梯度方差，在数学推理任务上一致提升 GRPO/RLOO 的采样效率，AIME24/25 上 Pass@32 最高提升 12.3 个点。

## 研究背景与动机
1. **领域现状**：GRPO/RLOO 等 group-based RL 方法通过为每个 prompt 生成多个 rollout 并估计相对优势来训练 LLM。通常对所有 prompt 均匀分配固定数量的 rollout（如 16 个）。
2. **现有痛点**：均匀分配隐式假设所有 prompt 同等重要——但成功率接近 0 或 1 的 prompt 的 rollout 几乎不产生有效梯度信号（方差为零），浪费了计算预算。现有的过滤方法需要先采样再过滤，可能抵消效率收益。
3. **核心矛盾**：需要在采样*之前*预测哪些 prompt 最有信息量（成功率接近 0.5 的 prompt 梯度方差最大），但成功率在训练过程中随模型更新而变化。
4. **本文要解决什么？** 如何在固定计算预算下，最优地将 rollout 分配给 mini-batch 中的各个 prompt？
5. **切入角度**：(1) 理论分析 Dr.GRPO 和 RLOO 的梯度方差与成功概率 $p$ 的关系——都正比于 $p(1-p)$；(2) 用高斯过程预测每个 prompt 的 $p$；(3) 用凸优化求解最优分配。
6. **核心idea一句话**：用 GP 预测成功概率 → 预测梯度方差 → 凸优化最小化总梯度方差 → 自适应分配 rollout。

## 方法详解

### 整体框架
每个训练iteration：(1) GP 根据历史 rollout 结果预测 mini-batch 中每个 prompt 的成功概率；(2) 用闭式凸优化在预算约束下分配 rollout 数量；(3) 按分配方案采样；(4) 用rollout结果更新 GP后验和模型参数。

### 关键设计

1. **梯度方差分析（理论贡献）**:
   - Dr.GRPO: $\text{Var}(\tilde{G}) = \frac{n-1}{n^2} 4\sigma_Z^2 p(1-p)$
   - RLOO: $\text{Var}(\tilde{G}) = \frac{1}{n-1} 4\sigma_Z^2 p(1-p)$
   - 关键洞察：方差正比于 $p(1-p)$——成功率 0.5 的 prompt 梯度方差最大（最有信息量），成功率 0 或 1 的无梯度信号

2. **高斯过程成功概率预测**:
   - 做什么：用 prompt embedding 上的 GP 预测每个 prompt 的当前成功概率
   - 核心思路：用 MiniLM 编码 prompt 为 384 维向量，RBF kernel 建模 prompt 间的相似性。sigmoid link function 将潜在值映射到概率。递归贝叶斯更新——利用历史 rollout 结果和 prompt 间的嵌入相似性
   - 设计动机：非参数模型无需跟踪模型权重变化，通过贝叶斯更新自然适应

3. **凸优化分配**:
   - 做什么：在总预算 $C$ 和单 prompt 上下界 $[L, U]$ 约束下最小化总梯度方差
   - 核心思路：连续松弛后有闭式解（Theorem 5.1/5.2），通过二分搜索求拉格朗日乘子 $\lambda^*$，再用贪心启发式取整到整数解
   - 效果：哈希embedding + 缓存距离矩阵使运行时开销可忽略

### 损失函数 / 训练策略
即插即用地与 Dr.GRPO/RLOO 集成。在 DAPO-MATH-17K 上训练，评估 AIME24/25。两种预算设置（8×Q, 16×Q）。

## 实验关键数据

### 主实验（AIME24/25 Pass@32）

| 模型 | 方法 | AIME24 Pass@32 | AIME25 Pass@32 |
|------|------|---------------|---------------|
| Qwen2.5-Math-1.5B | RLOO | 基线 | 基线 |
| | **RLOO+VIP** | **+12.3** | - |
| | Dr.GRPO | 基线 | 基线 |
| | **Dr.GRPO+VIP** | **提升** | **提升** |
| Qwen2.5-Math-7B | GRPO+VIP | 提升（但增幅较小） | 提升 |

### 关键发现
- VIP 在所有模型×基线×预算设置下一致提升 Pass@32 和 Mean@32
- 小模型（1.5B, 3B）受益更大——弱模型更容易浪费 rollout 在过难/过易的 prompt 上
- 7B 模型提升较小——因为强模型的成功率分布更集中在中间区域
- GP 预测的成功概率与实际成功率高度相关，验证了预测质量
- 运行时开销可忽略——embedding+距离矩阵预计算，GP 更新和凸优化在 CPU 上完成

## 亮点与洞察
- **理论基础扎实**：从梯度方差分析出发，推导出 $p(1-p)$ 的关键关系，为自适应分配提供了数学基础
- **凸优化的闭式解**：分配问题虽然是整数规划，但连续松弛有高效闭式解（bisection + 贪心取整），实际部署零额外负担
- **GP 是聪明的选择**：利用 prompt 嵌入的相似性进行信息共享——未见过的 prompt 可以通过相似 prompt 的历史结果预测

## 局限性 / 可改进方向
- 假设 $\sigma_Z^2$（投影梯度方差）对所有 prompt 相同——实际可能不成立
- GP 的 kernel bandwidth 用 median heuristic 设定，可能不是最优
- 仅在数学推理（RLVR 设置）上验证——RLHF 场景的奖励模型有噪声，分析可能需要修改
- 当 prompt 池非常大时 GP 的 $\Sigma$ 矩阵计算和存储可能成为瓶颈

## 相关工作与启发
- **vs 均匀分配 GRPO**: VIP 是均匀 GRPO 的严格升级——理论保证更低梯度方差
- **vs 过滤方法（Yu et al. 2025）**: 过滤方法在采样后丢弃无信息 prompt，VIP 在采样前预测并分配——避免浪费
- **vs 启发式难度分配（Zhang et al. 2025）**: VIP 有理论最优性保证（凸优化），而非启发式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 梯度方差分析 → GP 预测 → 凸优化分配，完整理论框架
- 实验充分度: ⭐⭐⭐⭐ 多模型多预算多基线对比
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，Theorem 有闭式解
- 价值: ⭐⭐⭐⭐⭐ 为 GRPO/RLOO 训练提供了即插即用的效率提升工具

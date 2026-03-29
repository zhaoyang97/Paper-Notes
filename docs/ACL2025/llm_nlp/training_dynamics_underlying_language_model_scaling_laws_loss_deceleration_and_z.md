# Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning

**会议**: ACL 2025  
**arXiv**: [2506.05447](https://arxiv.org/abs/2506.05447)  
**代码**: https://github.com/mirandrom/zsl  
**领域**: LLM NLP  
**关键词**: Scaling Laws, 训练动力学, Loss Deceleration, Zero-Sum Learning, 破缺幂律

## 一句话总结
发现语言模型训练中存在 loss deceleration（损失减速）现象——损失曲线在 log-log 空间呈分段线性，根因是 zero-sum learning（ZSL）：per-token 梯度系统性对立导致破坏性干涉，将一部分样本的改善抵消另一部分的恶化；scale up 通过降低减速触发损失 $L_d$ 和提升减速后斜率 $r_d$ 来缓解 ZSL，为突破 scaling law 瓶颈提供了可直接干预的机制。

## 研究背景与动机

1. **领域现状**：Kaplan et al. (2020) 提出的 scaling laws 能准确预测模型扩大后的 loss，但本质上只是经验拟合，没有解释 scaling *如何* 改善 loss（即训练动力学层面的机制）。

2. **现有痛点**：(a) 理论解释多从数据分布属性（Michaud et al., 2023）或内在模型容量（Sharma & Kaplan, 2022）出发，对训练过程中具体发生了什么变化着墨甚少；(b) 已知存在 loss plateau、saturation 等现象，但没有统一框架将它们与 scaling 改善关联起来；(c) 缺乏可操作的机制——如果只知道"越大越好"，就无法在不增大规模的情况下改善模型。

3. **核心矛盾**：scaling law 的幂律形式暗示训练动力学是平滑的，但作者发现实际损失曲线在 log-log 空间有一个突变的斜率变化（deceleration），这意味着存在一个质变的训练动力学转折点。

4. **本文要解决什么？** 识别并形式化 loss deceleration 现象，提出其背后的机制（zero-sum learning），并展示 scaling 如何缓解该机制——为未来"不靠规模也能改善模型"的方法奠定基础。

5. **切入角度**：从 per-example（per-token）梯度和损失变化的微观视角出发，分析宏观 loss 减速的根因。

6. **核心idea一句话**：loss 减速的根因是 per-token 梯度对立（ZSL），scaling up 通过缓解 ZSL 来改善最终 loss。

## 方法详解

### 整体框架
1. **现象刻画**：用 broken neural scaling law（BNSL）拟合 loss 曲线的分段线性行为，提取 $L_d$（减速触发损失）、$t_d$（减速触发步数）、$r_d$（减速后 log-log 斜率）三个可解释参数。
2. **机制解释**：提出 ZSL 假说——per-token 梯度系统性对立导致破坏性干涉，是 loss deceleration 的根因。
3. **Scaling 联系**：展示 scaling up 如何降低 $L_d$ 和 $t_d$、提升 $r_d$。

### 关键设计

1. **BNSL 拟合与可解释参数化（Eqn. 2）**:
   - Loss 估计：$\hat{L}_T = L_d \cdot (t_d / T)^{r_d}$
   - $L_d$：减速发生时的 loss 值，越小越好
   - $t_d$：减速发生的步数，越小表示越早减速
   - $r_d$：减速后 log-log 空间的斜率，越大 loss 下降越快
   - 三个参数完全描述了 scaling 带来的 loss 改善

2. **零和学习（ZSL）的形式化**:
   - 破坏性干涉度量 $D(\Delta\ell) = 1 - \frac{|\sum_i \Delta\ell_i|}{\sum_i |\Delta\ell_i|}$，取值 0→1，越大表示 token 间损失变化越抵消
   - 梯度破坏性干涉 $\vec{D}(\nabla_\theta \ell) = 1 - \frac{|\sum_i \nabla_\theta \ell_i|}{\sum_i |\nabla_\theta \ell_i|}$，per-parameter 平均
   - 关键分解：$|\Delta L| = M(\Delta\ell) \cdot (1 - D(\Delta\ell))$，其中 $M$ 是 token 级损失变化的平均幅度

3. **ZSL 对 deceleration 的贡献量化**:
   - $D(\Delta\ell)$ 从 0.5 增到 0.95 → loss 改善减少 10 倍
   - $M(\Delta\ell)$ 从 0.75 降到 0.5 → loss 改善仅减少 1.5 倍
   - 结论：ZSL（$D$ 项）主导了 deceleration，而非 token 级损失幅度减小（$M$ 项）

4. **梯度对立是 ZSL 的根因**:
   - 在一阶训练动力学假设下，$D(\tilde{\Delta}\ell)$ 来自 per-token 梯度在更新方向上的投影对立
   - 实验验证：梯度干涉度在 deceleration 前夕急剧上升到接近 1.0

## 实验关键数据

### 主实验表格（Table 1: Loss Deceleration Measurements）

| Model | $\downarrow L_d$ | $\downarrow t_d$ | $\uparrow r_d$ | $\hat{L}_T$ | $L_T$ |
|-------|--------|--------|--------|---------|-------|
| 14M   | 4.05   | 5900   | 0.013  | 3.86    | 3.88  |
| 37M   | 3.60   | 5900   | 0.016  | 3.39    | 3.40  |
| 78M   | 3.38   | 5900   | 0.020  | 3.14    | 3.15  |
| 144M  | 3.25   | 6000   | 0.023  | 2.98    | 2.99  |
| 285M  | 3.14   | 5300   | 0.025  | 2.85    | 2.87  |
| 472M  | 3.16   | 4600   | 0.035  | 2.77    | 2.80  |
| OLMo-1B | 2.86 | 3700  | 0.034  | 2.39    | 2.40  |
| OLMo-7B | 2.64 | 4600  | 0.053  | 2.04    | 2.03  |

- $\hat{L}_T$ 与 $L_T$ 误差在 1% 以内，验证了分段线性模型的有效性
- $L_d$ 和 $r_d$ 随模型规模单调改善
- 14M→7B：$r_d$ 从 0.013 提升到 0.053（4倍），$L_d$ 从 4.05 降到 2.64

### 核心消融/分析

- **D(Δℓ) vs M(Δℓ)**（Fig. 5）：D 从 0.5→0.95 导致 10× loss 改善减少；M 仅 1.5× → ZSL 是主因
- **梯度干涉时序**（Fig. 3-4）：D(Δℓ) 在 deceleration 前夕急剧上升，梯度干涉 D(∇ℓ) 在训练早期已很高（>0.9）但在 deceleration 附近进一步逼近 1.0
- **架构/数据/优化器消融**（Appendix C）：在不同架构（GPT、Llama）、数据集（C4、Dolma）、优化器（Adam、SGD）下均观察到 deceleration + ZSL，说明是普遍现象

### 关键发现
- Deceleration 是普遍存在的质变现象，不是 noise 或特定设置的产物
- ZSL 是 deceleration 的主要驱动力，而非 token 级损失幅度衰减
- Scaling up 通过缓解 ZSL（降低 D(Δℓ) 的峰值或延迟其上升）来改善 loss

## 亮点与洞察
- **可解释的 scaling law 参数化**：$\hat{L}_T = L_d (t_d/T)^{r_d}$ 将不透明的幂律分解为三个有物理意义的量，比传统 Chinchilla 式拟合更具洞察力
- **微观→宏观的桥梁**：从 per-token 梯度对立到宏观 loss 减速，建立了完整的因果链
- **可操作性**：ZSL 指向了一个具体的干预目标——减少 per-token 梯度间的破坏性干涉，可能通过例如课程学习、梯度手术、数据混合策略来实现

## 局限性
1. 缓解 ZSL 的具体方法（如梯度手术）尚未在本文实验，只是"可能的方向"
2. 分析基于 full-batch 梯度，实际训练用 mini-batch SGD，ZSL 的度量需要代理近似
3. 实验规模到 472M（自训）+ OLMo 7B（预训练检查点），更大规模（70B+）是否有新现象未知
4. BNSL 一阶近似（分段线性）在更长训练中是否仍成立需要验证

## 相关工作与启发
- 与 Kaplan et al. (2020) 的 scaling law 互补：后者描述最终 loss 与规模关系，本文描述*训练过程*中 loss 行为与规模关系
- ZSL 与多任务学习中的梯度冲突（Liu et al., 2021）概念相通，但本文是在*单任务*（语言建模）内发现的 token 间冲突
- 启发：如果能在训练中实时监控 $D(\Delta\ell)$，可能可以自适应调整学习率、数据混合比例或模型容量

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次系统识别和形式化 loss deceleration + ZSL)
- 理论深度: ⭐⭐⭐⭐⭐ (完整的形式化和因果验证链)
- 实验充分性: ⭐⭐⭐⭐ (规模覆盖广，但缺少干预实验)
- 实用价值: ⭐⭐⭐⭐ (指出方向但未落地)
- 总体推荐: ⭐⭐⭐⭐⭐ (scaling law 领域的重要理论贡献)

---
title: >-
  [论文解读] Exploring Landscapes for Better Minima along Valleys
description: >-
  [NeurIPS 2025][优化/理论][损失景观探索] 本文提出优化器适配器"E"，通过在梯度更新中加入梯度差分的指数移动平均 $\mathbf{a}_k = \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ 使优化器能在到达局部极小值后继续沿损失景观的"山谷"探索更低更平坦的极小值，适配后的 ALTO 在大批量训练中平均提升 2.5% 测试准确率。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "损失景观探索"
  - "山谷追踪"
  - "大批量训练"
  - "ALTO优化器"
  - "梯度差分"
---

# Exploring Landscapes for Better Minima along Valleys

**会议**: NeurIPS 2025  
**arXiv**: [2510.27153](https://arxiv.org/abs/2510.27153)  
**代码**: [PyPI](https://pypi.org/project/alto-optimizer/)  
**领域**: 优化 / 大批量训练  
**关键词**: 损失景观探索, 山谷追踪, 大批量训练, ALTO优化器, 梯度差分

## 一句话总结
本文提出优化器适配器"E"，通过在梯度更新中加入梯度差分的指数移动平均 $\mathbf{a}_k = \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ 使优化器能在到达局部极小值后继续沿损失景观的"山谷"探索更低更平坦的极小值，适配后的 ALTO 在大批量训练中平均提升 2.5% 测试准确率。

## 研究背景与动机

**领域现状**：几乎所有梯度优化器（SGD、Adam 等）到达某个局部极小值后就停止搜索。然而仅依赖局部信息无法保证找到的是最低或泛化最好的极小值。

**现有痛点**：(a) 传统优化器被局部极小值"困住"后无法继续探索损失景观的山谷结构；(b) 大批量训练是充分利用 GPU 并行的直接方式，但面临"更新步数少"的困难——要求在远少于小批量训练的参数更新次数下达到相近的测试精度；(c) 现有学习率缩放规则（线性缩放 $\eta_k \propto |\mathcal{Z}_k|$、平方根缩放）在超大批量下受限于任务相关的临界值。

**核心矛盾**：在损失景观中，宏观上优化器需要被大尺度山谷"捕获"（沿谷行走），微观上需要能逃离小尺度尖锐极小值——传统优化器只解决了前者。

**本文目标** 设计一个在到达局部极小值后仍能继续沿山谷探索的优化器，找到更低更平坦的极小值。

**切入角度**：观察到 $-\nabla\|\nabla f(\theta_k)\|^2 = -2\mathbf{H}_k \bar{\mathbf{g}}_k \approx \bar{\mathbf{g}}_k - \bar{\mathbf{g}}_{k-1}$——梯度差分方向自然具有"逃离尖锐极小值、被平坦极小值捕获"的特性。

**核心 idea**：在优化器的梯度中加入 $\alpha \cdot \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ 项，使其在微观上排斥尖锐极小值、在宏观上追踪山谷。

## 方法详解

### 整体框架
E-adaptor 是一个可插入任意梯度优化器的适配器。核心修改：将梯度 $\mathbf{g}_k$ 替换为 $\mathbf{g}_k + \alpha \mathbf{a}_k$，其中 $\mathbf{a}_k = \beta_1 \mathbf{a}_{k-1} + (1-\beta_1)(\mathbf{g}_k - \mathbf{g}_{k-1})$ 是梯度差分的 EMA。然后正常走 Adam/Lamb 流程。

### 关键设计

1. **梯度差分的方向分析**:

    - 功能：证明 $\mathbf{g}_k - \mathbf{g}_{k-1}$ 是逃离尖锐极小值的理想方向
    - 核心思路：考虑优化器接近极小值的 4 个阶段（①②下坡→②③过极小值→③④上坡→④⑤减速）。分析内积 $\langle \theta_k - \theta_{k-1}, \mathbf{g}_k - \mathbf{g}_{k-1} \rangle$ 的符号：
        - 在②③和③④阶段（经过极小值时）内积为正 → **加速**逃离
        - 在①②和④⑤阶段内积为负 → **减速**（被山谷捕获）
    - 与 $-\nabla\|\nabla f\|^2$ 对比：后者在②③阶段内积为负（反而减速），不如梯度差分
    - 设计动机：$-\nabla\|\nabla f\|^2 = -2\mathbf{H}_k \bar{\mathbf{g}}_k$，Hessian $\mathbf{H}_k$ 大 = 尖锐极小值 → 大步长易逃离；$\mathbf{H}_k$ 小 = 平坦极小值 → 小步长被捕获

2. **ALTO 算法 (Adapted Lamb with Exploration)**:

    - 功能：在 Lamb 优化器中嵌入 E-adaptor
    - 核心迭代：
        - $\mathbf{a}_k = \beta_1 \mathbf{a}_{k-1} + (1-\beta_1)(\mathbf{g}_k - \mathbf{g}_{k-1})$（加速项 EMA）
        - $\mathbf{m}_k = \beta_2 \mathbf{m}_{k-1} + (1-\beta_2)(\mathbf{g}_k + \alpha \mathbf{a}_k)$（一阶矩）
        - $\mathbf{v}_k = \beta_3 \mathbf{v}_{k-1} + (1-\beta_3)[\mathbf{g}_k + \alpha \mathbf{a}_k]^2$（二阶矩）
        - bias correction → $\hat{\mathbf{m}}_k, \hat{\mathbf{v}}_k$
        - $\mathbf{r}_k = \hat{\mathbf{m}}_k / (\sqrt{\hat{\mathbf{v}}_k} + \varepsilon_1) + \lambda_k \theta_k$
        - 层级正则化更新：$\theta_{k+1}^{(i)} = \theta_k^{(i)} - \eta_k \mathbf{r}_k^{(i)} \phi(\|\theta_k^{(i)}\|) / (\|\mathbf{r}_k^{(i)}\| + \varepsilon_2 \phi(\|\theta_k^{(i)}\|))$
    - 关键约束：$|\alpha| < 1/(1-\beta_1)$，由连续化微分方程的稳定性条件推导

3. **$\mathbf{a}_k$ 为何加入 $\mathbf{g}_k$ 而非 $\mathbf{m}_k$**:

    - 加入 $\mathbf{m}_k$ 意味着梯度差分与梯度同量级 → 剧烈振荡或无效
    - 加入 $\mathbf{g}_k$ 意味着最终动量中包含梯度差分的 EMA₂（两次指数移动平均），权重为 $(1-\beta)^2 \beta^{k-i} \binom{k-i+1}{1}$ → 更平滑稳定
    - EMA₂ 在训练早期（梯度快速衰减阶段）积累了最有信息量的方向，后期作为"导航"使用

4. **$\alpha$ 正负的影响**:

    - $\alpha > 0$：收敛快但探索少，适合小批量
    - $\alpha < 0$：探索更多、找到更平坦极小值，但收敛慢，适合大批量
    - 推荐：小批量 $\alpha = 0.5, \beta_1 = 0.01$；大批量 $\alpha = -5, \beta_1 = 0.99$

### 收敛分析
- **Theorem 1（非凸）**：在 $L$-光滑、无偏有界方差梯度假设下，$T \geq O(G_\infty^{1.5} \epsilon^{-2})$ 时，$\frac{1}{T+1}\sum_{k=0}^T \mathbb{E}\|\nabla f_k(\theta_k)\|^2 \leq 4\epsilon^2$。比 Lamb 的 $O(\epsilon^{-4})$ 更好。
- **Theorem 2（凸）**：$R(T) \leq O(\sqrt{T})$，与 Adam 相同量级，但约束更宽松（$\beta_2^2/\beta_3 < 1$ 而非 $\beta_{1,k} = \beta_k \lambda^k$）。

## 实验关键数据

### 大批量 ImageNet 训练 (ResNet-50, 90 epochs)

| 批量大小 | Adam | AdamW | AdaBelief | Lamb | **ALTO** |
|---------|------|-------|-----------|------|----------|
| 1K | 73.08 | 75.65 | 73.32 | 77.06 | **77.22** |
| 2K | 73.08 | 74.93 | 73.48 | 77.11 | **77.25** |
| 4K | 73.32 | 74.65 | 73.41 | 76.92 | **77.35** |
| 8K | 73.11 | 74.40 | 73.14 | 76.89 | **77.10** |
| 16K | 73.09 | 74.10 | 73.00 | 76.66 | **76.87** |
| 32K | 72.50 | 73.57 | 72.89 | 76.42 | **76.70** |

### CIFAR-10/100 + ImageNet (ResNet-20/34)

| 数据集 | 批量 | SGD | Adam | Lamb | **ALTO** |
|--------|-----|-----|------|------|----------|
| CIFAR-10 | 128 | 91.85 | 89.88 | 90.89 | 91.24 |
| CIFAR-10 | 16384 | 80.86 | 87.34 | 83.56 | **88.83** |
| CIFAR-100 | 128 | 64.93 | 64.35 | 61.29 | **65.74** |
| CIFAR-100 | 16384 | 44.20 | 54.91 | 56.06 | **57.78** |
| ImageNet | 256 | 70.64 | 65.06 | 69.17 | 69.95 |
| ImageNet | 4086 | 49.35 | 54.96 | 70.34 | **70.83** |

### 训练时间对比 (VGG-16, CIFAR-100, batch=16384)

| 达到精度 | ALTO (s) | Lamb (s) | 加速比 |
|---------|---------|---------|--------|
| 20% | 137 | 196 | 1.43× |
| 40% | 334 | 409 | 1.23× |
| 60% | 608 | 865 | 1.42× |

### 关键发现
- ALTO 在**所有 17 个 CV+NLP 实验**中均优于 SOTA（Lamb）
- 大批量优势更显著：batch=16384 时 CIFAR-10 上 ALTO 比 Lamb 高 5.27%
- ALTO 大批量 ImageNet (batch=4086, 70.83%) **超过** SGD 小批量 (batch=256, 70.64%)
- 在 GPT-2 训练中，ALTO 的 test perplexity（78.37）远优于 Lamb（83.13）
- 达到同一精度，ALTO 可**节省 29.68%** 计算时间

## 亮点与洞察
- **从微分方程视角推导约束**：将离散优化器连续化为 ODE，通过特征值实部条件推导 $|\alpha| < 1/(1-\beta_1)$，连接了理论稳定性和实践超参数选择
- **EMA₂ = 记住早期方向**：EMA of EMA 使优化器在后期（梯度被噪声淹没时）仍能利用早期训练阶段积累的有信息量的方向，特别适合大批量训练
- **大批量 = 更准确的梯度差分**：大批量的梯度估计更准确 → 梯度差分 $\mathbf{g}_k - \mathbf{g}_{k-1}$ 更可靠 → ALTO 的优势更大，这解释了为何大批量下改进最显著

## 局限与展望
- 引入 5 个额外超参数（$\alpha, \beta_1, \beta_2, \beta_3, \varepsilon$），虽然作者声称通常只调 $\beta_1$ 和 $\eta$，但仍增加了调参负担
- 每步计算量多出一个梯度差分的 EMA 维护，epoch 时间略长于 Lamb
- 非凸收敛分析的假设（Assumption 3.3-3.5）较强，特别是单调性假设在实际中可能不成立
- 实验仅在单节点 4×A100 上进行，多节点分布式场景的通信瓶颈未评估
- $\alpha$ 正负的选择与批量大小的关系是经验性的，缺乏理论指导

## 相关工作与启发
- **vs Lamb [You et al., 2020]**: Lamb 是 ALTO 的基础，层级正则化来自 Lamb。ALTO 加入探索项 $\mathbf{a}_k$ 后在所有批量大小下均优于 Lamb
- **vs AdaBelief [Zhuang et al., 2020]**: AdaBelief 用 $(g_k - m_{k-1})^2$ 替代 $g_k^2$ 做自适应学习率，关注梯度预测准确性。ALTO 关注的是梯度差分的方向信息
- **vs SAM/Sharpness-Aware**: SAM 显式最小化 loss 景观锐度。ALTO 通过梯度差分隐式偏好平坦极小值，无需额外前向传播
- **vs 学习率 warmup/cosine schedule**: 调度策略是时间维度的控制。ALTO 的探索是几何/方向维度的控制，两者正交可叠加

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度差分作为山谷探索方向的洞察新颖，EMA₂ 的设计有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ CV/NLP/RL 多任务多模型多批量，17 个实验全面且一致
- 写作质量: ⭐⭐⭐⭐ 方向分析的图示和表格（Table 1, Fig 2）非常直观
- 价值: ⭐⭐⭐⭐ 对大批量训练有直接实用价值，ALTO 可作为 Lamb 的即插即用升级

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[ICLR 2026\] Align-SAM: Seeking Flatter Minima for Better Cross-Subset Alignment](../../ICLR2026/optimization/align-sam_seeking_flatter_minima_for_better_cross-subset_alignment.md)
- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](../../ICLR2026/optimization/when_to_restart_exploring_escalating_restarts_on_convergence.md)

</div>

<!-- RELATED:END -->

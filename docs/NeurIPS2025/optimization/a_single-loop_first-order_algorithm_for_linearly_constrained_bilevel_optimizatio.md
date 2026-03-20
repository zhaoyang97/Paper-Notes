# A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.24710](https://arxiv.org/abs/2510.24710)  
**代码**: [ShenGroup/SFLCB](https://github.com/ShenGroup/SFLCB)  
**领域**: others (优化理论)  
**关键词**: bilevel optimization, constrained optimization, first-order methods, augmented Lagrangian, single-loop algorithm  
**作者**: Wei Shen, Jiawei Zhang, Minhui Huang, Cong Shen (UVA, UW-Madison, Meta)  

## 一句话总结

针对下层问题带耦合线性约束的双层优化问题，提出单循环一阶算法 SFLCB，通过罚函数 + 增广拉格朗日重构消除 Hessian 依赖，将迭代复杂度从 $O(\epsilon^{-3}\log(\epsilon^{-1}))$ 改进至 $O(\epsilon^{-3})$。

## 研究背景与动机

1. **双层优化广泛应用**：超参数优化、数据清洗、元学习、神经架构搜索、强化学习等众多 ML 问题本质上是双层优化 (BLO)，上层目标依赖于下层问题的最优解。
2. **约束 BLO 研究不足**：现有工作大量针对无约束 BLO，而实际应用（分布式优化、SVM 超参数调优、对抗训练、交通网络设计）中下层问题常带耦合约束 $\mathcal{Y}(x) = \{y \mid Bx + Ay - b \leq 0\}$，相关理论和算法远不成熟。
3. **Hessian 计算瓶颈**：隐式梯度方法 (Tsaknakis et al., Khanduri et al., Xu & Zhu) 虽可处理约束 BLO，但均需计算下层 Hessian 矩阵，在大规模问题上计算开销巨大。
4. **已有一阶方法局限**：Kwon et al. (2023b) 仅处理与 $x$ 无关的约束且每步需投影；Jiang et al. (2024a) 支持耦合约束但需双/三循环结构，复杂度 $O(\epsilon^{-3}\log\epsilon^{-1})$ 或 $O(\epsilon^{-5}\log\epsilon^{-1})$，实现困难。
5. **改写函数的理论缺口**：此前工作 (Kwon, Jiang) 未建立改写函数 $\Phi_\delta$ 与原目标 $\Phi$ 梯度在耦合约束下的非渐近误差界，缺乏对改写合理性的严格证明。
6. **追求更优复杂度**：无约束 BLO 一阶方法已达 $O(\epsilon^{-2})$，约束场景下最佳已知为 $O(\epsilon^{-3}\log\epsilon^{-1})$（双循环），是否能单循环达到 $O(\epsilon^{-3})$ 是一个开放问题。

## 方法详解

### 整体框架

SFLCB 的核心思路分三步：

1. **罚函数改写**：将原始双层问题改写为单层 minimax 问题 $\min_x \Phi_\delta(x) = \min_{y \in \mathcal{Y}(x)} \max_{z \in \mathcal{Y}(x)} \Phi_\delta(x,y,z)$，其中 $\Phi_\delta(x,y,z) = f(x,y) + \frac{1}{\delta}[g(x,y) - g(x,z)]$，$\delta$ 为罚参数。
2. **松弛变量 + 增广拉格朗日**：引入松弛变量 $\alpha, \beta$ 将不等式约束转为等式约束，再构建增广拉格朗日函数 $K$，使其关于 $y'$ 强凸、关于 $z'$ 强凹。
3. **单循环 GDA**：对 $K$ 执行梯度下降上升 (GDA)，每次迭代仅需一阶梯度，无需 Hessian。

### 关键设计 1：改写函数的理论保证

- **定理 4.1**（值逼近）：$0 \leq \Phi(x) - \Phi_\delta(x) \leq \frac{\delta l_{f,0}^2}{2\mu_g}$，且 $\|y_\delta^*(x) - y^*(x)\| \leq \frac{2\delta l_{f,0}}{\mu_g}$。选取 $\delta = O(\epsilon)$ 即可控制逼近误差。
- **定理 4.9**（梯度逼近）：在 LICQ + 严格互补条件下，$\|\nabla\Phi(x) - \nabla\Phi_\delta(x)\| \leq O(\delta)$。这是首次在耦合约束 $\mathcal{Y}(x)$ 下建立的非渐近梯度误差界。

### 关键设计 2：增广拉格朗日强凸凹化

原始拉格朗日 $L_\delta$ 关于 $y'$ 仅凸、关于 $z'$ 仅凹，不利于快速收敛。通过添加二次增广项构造：

$$K(x,y',z',u,v) = L_\delta + \frac{\rho_1}{2}\|h(x,y)-\alpha\|^2 - \frac{\rho_2}{2}\|h(x,z)-\beta\|^2$$

选取 $\rho_1 \leq \frac{\mu_g - \delta l_{f,1}}{\sigma_{\max}^2(A)}$，$\rho_2 \leq \frac{\mu_g}{\sigma_{\max}^2(A)}$，即可保证 $K$ 关于 $y'$ 强凸、关于 $z'$ 强凹，且最优点与 $L_\delta$ 相同。

### 关键设计 3：新势函数与误差界

收敛分析的核心创新是构造势函数：

$$V_t = \frac{1}{4}K(x_t, y_t', z_t', u_t, v_t) + 2q(x_t, v_t) - d(x_t, z_t', u_t, v_t)$$

并证明其下降引理。关键技术包括：
- 利用 A 的满秩性质界定拉格朗日乘子误差 $\|u_{t+1} - u_\delta^*(x_t)\|$
- 不需满秩时，通过固定 $x_0$ 的 warm-start 策略在 $O(\epsilon^{-2})$ 步内获取高质量初始点

### 损失函数 / 目标

最终优化目标为：

$$\min_{x, y' \in \mathcal{P}_y, v} \max_{z' \in \mathcal{P}_y, u} K(x, y', z', u, v)$$

其中 $\mathcal{P}_y = \{y \in \mathbb{R}^{d_y}, \alpha \in \mathbb{R}_-^{d_h}\}$，投影仅涉及对 $\alpha, \beta$ 截断到非正象限，计算代价极低。

## 实验关键数据

### 主要复杂度对比（Table 1）

| 方法 | 下层约束 | 迭代复杂度 | 循环结构 |
|------|----------|-----------|---------|
| Kwon et al. (2023b) | $y \in \mathcal{Y}$（不依赖 $x$） | $O(\epsilon^{-3}\log\epsilon^{-1})$ | 单/双循环 |
| BLOCC (Jiang 2024a) | 一般 $h(x,y) \leq 0$ | $O(\epsilon^{-5}\log\epsilon^{-1})$ | 三循环 |
| BLOCC (Jiang 2024a) | $B(x)+A(x)y \leq 0$，$A(x)$ 满秩 | $O(\epsilon^{-3}\log\epsilon^{-1})$ | 双循环 |
| **SFLCB** | $Bx+Ay-b \leq 0$，$A$ 满秩 | $O(\epsilon^{-3})$ | **单循环** |
| **SFLCB** | $Ay \leq 0$，初始点 LICQ | $O(\epsilon^{-3})$ | **单循环** |
| **SFLCB** | $Ay \leq 0$（一般） | $O(\epsilon^{-4})$ | **单循环** |

### 实验发现

1. **Toy example**：200 个随机初始化下 SFLCB 均能稳定收敛到超目标的局部最优点，验证了算法有效性。
2. **SVM 超参数优化**（diabetes 数据集）：与 GAM、LV-HBA、BLOCC、BiC-GAFFA 相比，SFLCB 收敛速度显著更快，在相同迭代次数内达到更低的上层目标值（Fig. 2）。
3. **交通网络设计**（3 节点 & 9 节点）：SFLCB 在两种规模的网络上均大幅优于 BLOCC，获得更高的上层效用（Fig. 3）。
4. **$\delta$ 敏感性分析**：$\delta$ 越大初期收敛越快但后期逼近误差大，过小则收敛缓慢；中等值（0.05–0.5）兼顾速度与精度，与理论预测一致（Fig. 4）。

## 亮点与洞察

1. **理论贡献突出**：首次在耦合约束 $\mathcal{Y}(x)$ 下建立 $\Phi$ 与 $\Phi_\delta$ 值和梯度的非渐近误差界（Theorem 4.1 & 4.9），为 minimax 改写提供坚实理论基础。
2. **单循环 + 无 Hessian**：算法结构极简，每步仅需一阶梯度和简单投影（截断到非正象限），实现难度远低于 BLOCC 的三循环结构。
3. **复杂度真正改进**：从 $O(\epsilon^{-3}\log\epsilon^{-1})$ 到 $O(\epsilon^{-3})$，消除了对数因子；对一般（非满秩）情形也给出 $O(\epsilon^{-4})$ 的保证。
4. **势函数构造技巧**：$V_t$ 的精巧设计（混合 $K$、对偶函数 $q$、下降函数 $d$）和配套误差界具有独立理论价值，可推广到其他约束优化问题。
5. **warm-start 策略**：对 $A$ 非满秩的情形，通过固定 $x_0$ 先跑算法获取初始点，代价仅 $O(\epsilon^{-2})$，不影响总复杂度阶。

## 局限性

1. **仅限确定性设定**：所有分析基于精确梯度，未覆盖随机梯度（SGD）场景，限制了在大规模深度学习中的直接应用。
2. **仅限线性约束**：算法和理论严格要求 $h(x,y) = Bx + Ay - b$，无法处理非线性耦合约束。
3. **LICQ / 严格互补假设**：梯度逼近（Theorem 4.9）和部分收敛结果依赖 LICQ 和严格互补条件，退化情形未覆盖。
4. **理论与最优差距**：无约束 BLO 一阶方法已达 $O(\epsilon^{-2})$，约束场景的 $O(\epsilon^{-3})$ 是否最优仍是开放问题。
5. **实验规模较小**：SVM 实验仅在 diabetes 数据集上进行，交通网络最大仅 9 节点，缺少大规模/高维问题的验证。

## 相关工作

- **无约束 BLO 一阶方法**：penalty-based (Kwon et al. 2023a, Shen & Chen 2023, Lu 2024)，iterative differentiation (Maclaurin et al. 2015, Grazzi et al. 2020)
- **约束 BLO 隐式梯度**：Tsaknakis et al. 2022 (线性约束 $Ay \leq b$), Khanduri et al. 2023 (扰动平滑), Xu & Zhu 2023 (Clarke 次微分)
- **约束 BLO 一阶方法**：Yao et al. 2024a/b (proximal Lagrangian / doubly regularized gap), Kornowski et al. 2024 (Goldstein 稳定性, 近最优但依赖维度)
- **最近邻工作**：BLOCC (Jiang et al. 2024a) 三循环处理一般耦合约束，Kwon et al. 2023b 单/双循环但仅处理 $x$-独立约束

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 理论贡献扎实（耦合约束下的非渐近梯度误差界为新结果），势函数构造有技术创新，但改写框架本身沿用已有范式
- **实验充分度**: ⭐⭐⭐ — 实验覆盖 toy、SVM、交通网络三个场景并含敏感性分析，但规模偏小，缺少与 Kornowski et al. 2024 等更多方法的比较
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，理论推导严谨，Table 1 一目了然，但部分符号较密导致可读性下降
- **价值**: ⭐⭐⭐⭐ — 为约束 BLO 提供了目前最简单高效的一阶算法，对分布式优化和约束学习有直接指导意义
